#!/usr/bin/env python3
"""Independent O / fixed / GD / frozen-G / GDT runner, validation-F1 selection.

Default smoke runs on CPU; production requires fresh independent gate evidence.
This command never modifies the canonical queue, baseline, or completed runs.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,Subset
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import geometry_embedding
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.geometry_prior_launch import write_json
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank
from keypoint_comparison1.landmark_detection.training import seed_everything
from scripts.evaluate_messenger_geometry_prior import load_model
from scripts.messenger_natural_occlusion_common import authenticate,datasets,verify_gates
from src.models.messenger_prototype_observer import attach_learnable_prototypes,LearnablePrototypeBank,sparse_point_samples
from src.models.messenger_occlusion_training import CenterUpdate,OcclusionCounters,SupervisionAudit,F1Checkpoint,O_METRIC,T_METRIC
from src.models.messenger_occlusion_augmentation import BoundedOccluderRGB,NaturalOcclusionDiscriminator
from src.models.messenger_natural_task import MessengerTask,NaturalUpdate
from src.data.messenger_natural_occlusion import NaturalOcclusionDataset,collate_natural,CompositionAudit


def fields(targets, device):
    xy=torch.stack([t['keypoints'] for t in targets]).to(device)
    states=torch.stack([torch.where(t['original_labeled'],torch.where(t['original_visible'],2,1),0) for t in targets]).to(device)
    geo=torch.stack([t['geometry_prior']['geometry'] for t in targets]).to(device)
    support=torch.stack([t['geometry_prior']['support'] for t in targets]).to(device)
    return xy,states,geo,support


@torch.no_grad()
def validate_observer(model, dataset, *, device, batch_size):
    counters=OcclusionCounters();model.eval()
    for images,targets in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_landmark_batch):
        xy,states,geo,support=fields(targets,device)
        prior=model.observe(images.to(device),geometry=geo,support=support)
        counters.add(prior,xy,states,support,input_hw=images.shape[-2:])
    return counters.compute()


def positive_weight_and_audit(model, train):
    k,_,h,w=model.bank.centers.shape;audit=SupervisionAudit(k,h,w);positive=negative=0
    for images,targets in DataLoader(train,batch_size=64,collate_fn=collate_landmark_batch):
        xy,states,g,s=fields(targets,torch.device('cpu'))
        bank=model.bank.fixed
        embedding=geometry_embedding(g)
        selected=torch.cdist((embedding-bank.geometry_mean.cpu())/bank.geometry_std.cpu(),bank.geometry_centers.cpu()).argmin(1)
        audit.add(selected,xy,states,s,input_hw=images.shape[-2:])
        _,_,counts=sparse_point_samples(torch.zeros_like(s,dtype=torch.float32),xy,states,s,input_hw=images.shape[-2:])
        positive+=counts['positives'];negative+=counts['negatives']
    if not positive or not negative:raise ValueError('Training requires both occluded and visible points')
    return negative/positive,dict(audit.compute(),positive_points=positive,negative_points=negative)


def scheduler(optimizer, *, steps, epochs, warmup):
    def scale(step):
        warmup_steps=steps*warmup;total=steps*epochs
        if step<warmup_steps:return (step+1)/max(1,warmup_steps)
        progress=(step-warmup_steps)/max(1,total-warmup_steps)
        return .01+.99*.5*(1+math.cos(math.pi*min(1.,progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)


def run(args):
    cfg,spec,parent_identity,checkpoint,identity=authenticate(args.config)
    if args.arm not in cfg['arms']:raise ValueError('Unknown arm')
    if args.arm=='observer' and (args.observer_checkpoint or args.generator_checkpoint):raise ValueError('O initializes from k-means only')
    if (args.arm=='frozen_g') != bool(args.generator_checkpoint):raise ValueError('Only frozen_g requires a selected G/T bundle')
    optional={}
    for name,path in [('observer',args.observer_checkpoint),('generator',args.generator_checkpoint)]:
        if path:optional[name]=dict(path=str(path.resolve()),sha256=file_hash(path))
    experiment=dict(identity=identity,arm=args.arm,optional_initialization=optional)
    if args.mode=='production':
        if args.gates is None:raise ValueError('Fresh per-arm four gates required')
        microbatch=verify_gates(args.gates,experiment);device=torch.device('cuda:0')
    else:microbatch=4;device=torch.device('cpu')
    torch.set_num_threads(2 if args.mode=='smoke' else 1);seed_everything(cfg['seed'])
    output=args.output.resolve()
    if output == (ROOT/spec['output_root']).resolve() or (ROOT/spec['output_root']).resolve() in output.parents:
        raise ValueError('Separate output required')
    if args.resume:
        old=json.loads((output/'contract.json').read_text())
        if old!=dict(experiment=experiment,mode=args.mode):raise ValueError('Resume identity changed')
        if (output/'completion.json').exists():raise ValueError('Do not retry completed experiments')
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'contract.json',dict(experiment=experiment,mode=args.mode))
    parent,provenance=load_model(spec,parent_identity,checkpoint,device)
    train,val=datasets(spec);raw=spec['parent_config'];data=raw['data']
    epochs=2 if args.mode=='smoke' else cfg['train']['epochs']
    warmup=1 if args.mode=='smoke' else cfg['train']['warmup_epochs']
    gen=torch.Generator().manual_seed(cfg['seed']);effective=cfg['train']['effective_batch']
    natural=None;composition_audit=None
    if args.arm=='observer':
        model=attach_learnable_prototypes(parent,enabled=True,logit_temperature=cfg['observer']['logit_temperature']);del parent
        if args.mode=='smoke':
            ids=sorted(range(len(train)),key=lambda i:hashlib.sha256(f"{cfg['seed']}:{train.samples[i]['annotation']['id']}:O-F1".encode()).hexdigest())[:128]
            train=Subset(train,ids)
        pos_weight,audit=positive_weight_and_audit(model,train)
        write_json(output/'supervision_audit.json',audit)
        runtime=CenterUpdate(model,lr=cfg['observer']['learning_rate'],positive_weight=pos_weight,anchor_weight=cfg['observer']['anchor_weight'])
        optimizers={'o':runtime.optimizer};modules={'o':model.bank};metric=O_METRIC
        loader=DataLoader(train,batch_size=effective,shuffle=True,generator=gen,collate_fn=collate_landmark_batch)
        validate=lambda batch:validate_observer(model,val,device=device,batch_size=batch)
    else:
        if args.observer_checkpoint:
            saved=torch.load(args.observer_checkpoint,map_location='cpu',weights_only=False)
            if (saved.get('schema')!='occlusion-f1-checkpoint-v1' or saved['contract']['metric']!=O_METRIC
                    or saved['contract']['experiment']['run']['identity']!=identity
                    or saved['epoch']!=saved['best_epoch']
                    or (args.mode=='production' and saved['contract']['experiment']['mode']!='production')):
                raise ValueError('O checkpoint identity/metric differs')
            parent.bank=LearnablePrototypeBank(parent.bank,logit_temperature=cfg['observer']['logit_temperature']).to(device)
            parent.bank.load_state_dict(saved['modules']['o'],strict=True)
        parent.requires_grad_(True);parent.observer.requires_grad_(False);parent.bank.requires_grad_(False)
        parent.base.messenger_recovery.mixing_enabled=True
        task=MessengerTask(parent,LandmarkLoss(raw['loss']),mean=data['mean'],std=data['std'],
            reconstruction_weight=cfg['task']['reconstruction_weight']).to(device)
        generator=BoundedOccluderRGB(max_rgb_delta=cfg['generator']['max_rgb_delta'],field_size=cfg['generator']['field_size']).to(device)
        discriminator=NaturalOcclusionDiscriminator().to(device)
        if args.generator_checkpoint:
            saved=torch.load(args.generator_checkpoint,map_location='cpu',weights_only=False)
            if (saved.get('schema')!='occlusion-f1-checkpoint-v1' or saved['contract']['metric']!=T_METRIC
                    or saved['contract']['experiment']['run']['identity']!=identity
                    or saved['epoch']!=saved['best_epoch']
                    or (args.mode=='production' and saved['contract']['experiment']['mode']!='production')):raise ValueError('G checkpoint identity differs')
            generator.load_state_dict(saved['modules']['g'],strict=True)
        bank=RealOcclusionBank(ROOT/cfg['material_bank']['path'],cfg['material_bank']['sha256'],
            annotation_path=data['annotations']['train'],image_root=data['root'])
        c=cfg['composition'];natural=NaturalOcclusionDataset(train,bank,seed=cfg['seed'],probability=c['probability'],bins=c['bins'],attempts=c['attempts'])
        training=natural
        if args.mode=='smoke':
            # Include requested synthetic and unmodified examples, without inspecting losses.
            chosen=sorted(natural.requests)[:8]
            chosen += [i for i in range(len(natural)) if i not in natural.requests][:8]
            training=Subset(natural,chosen)
        backbone_ids={id(p) for p in parent.base.backbone.parameters()}
        optimizers={'t':torch.optim.AdamW([
            dict(params=[p for p in task.parameters() if p.requires_grad and id(p) in backbone_ids],lr=cfg['task']['backbone_learning_rate']),
            dict(params=[p for p in task.parameters() if p.requires_grad and id(p) not in backbone_ids],lr=cfg['task']['learning_rate'])],weight_decay=1e-4)}
        if args.arm in ('gd','gdt'):
            optimizers.update(g=torch.optim.Adam(generator.parameters(),lr=cfg['generator']['learning_rate']),
                              d=torch.optim.Adam(discriminator.parameters(),lr=cfg['discriminator']['learning_rate']))
        modules={'task':task,'g':generator,'d':discriminator};metric=T_METRIC
        runtime=NaturalUpdate(generator,discriminator,task,optimizers=optimizers,arm=args.arm,
            **{key:cfg['generator'][key] for key in ('realism_weight','preserve_weight','hardness_weight')})
        loader=DataLoader(training,batch_size=effective,shuffle=True,generator=gen,collate_fn=collate_natural)
        validate=lambda batch:validate_prior(parent,val,device=device,batch_size=batch)
    schedulers={k:scheduler(o,steps=len(loader),epochs=epochs,warmup=warmup) for k,o in optimizers.items()}
    contract=dict(run=experiment,declared_config=cfg,mode=args.mode,maximum_epochs=epochs,warmup_epochs=warmup,effective_batch=effective,
        training_instances=len(loader.dataset),validation_instances=len(val),raw_checkpoint_no_ema=True,
        positive_weight=pos_weight if args.arm=='observer' else None)
    policy=F1Checkpoint(contract=contract,metric=metric,modules=modules,optimizers=optimizers,schedulers=schedulers,
        loader_generator=gen,minimum_epoch=1 if args.mode=='smoke' else cfg['train']['minimum_epoch'],
        patience=cfg['train']['patience'],min_delta=cfg['train']['min_delta'])
    global_step=0
    if args.resume:
        saved=torch.load(output/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        policy.restore(saved);global_step=int(policy.last_metrics['optimizer_step'])
        if args.arm=='observer':runtime.steps=global_step
        else:runtime.step=global_step
    run=None
    if args.mode=='production':
        import wandb
        run_id=hashlib.sha256(json.dumps(experiment,sort_keys=True).encode()).hexdigest()[:16]
        run=wandb.init(project='face_detection',group='messenger_natural_occlusion_v1',name=args.arm,
            id=run_id,resume='must' if args.resume else 'never',config=contract,dir=str(output))
        write_json(output/'wandb.json',dict(id=run.id,url=run.url))
    validation_batch=microbatch
    def validation():
        nonlocal validation_batch
        while True:
            try:return validate(validation_batch)
            except torch.OutOfMemoryError:
                if validation_batch==1:raise
                validation_batch=max(1,validation_batch//2);torch.cuda.empty_cache()
    try:
        if not args.resume:write_json(output/'epoch0_validation.json',validation())
        while policy.epoch<epochs and not policy.should_stop:
            epoch=policy.epoch+1;totals=[];composition_audit=CompositionAudit()
            if natural is not None:
                # Smoke uses epoch0 request inventory to exercise the selected requested samples.
                natural.set_epoch(0 if args.mode=='smoke' else epoch)
                parent.base.backbone.requires_grad_(epoch>cfg['train']['backbone_freeze_epochs'])
            for batch in loader:
                if args.arm=='observer':
                    images,targets=batch;xy,states,geo,support=fields(targets,torch.device('cpu'))
                    row=runtime.update(images,geo,support,xy,states,microbatch=microbatch)
                    updated=row['status']=='updated'
                else:
                    composition_audit.add(batch['metadata'])
                    row=runtime.update(batch,microbatch=microbatch);updated=True
                microbatch=min(microbatch,row['microbatch']);global_step=row['step'];totals.append(row)
                if updated:
                    for key,sched in schedulers.items():
                        if (key!='d' or row.get('d_updated')) and (key!='g' or row.get('g_updated')):sched.step()
            if natural is not None:
                audit=composition_audit.compute(**{k:cfg['composition'][k] for k in ('maximum_skip','maximum_stratum_skip','minimum_real_pair_fraction')})
                write_json(output/'composition_audit.json',dict(epoch=epoch,**audit))
                if args.mode=='production' and not audit['passed']:raise RuntimeError('Fresh composition distribution audit failed')
            metrics=validation();metrics['optimizer_step']=global_step
            improved=policy.record(metrics)
            complete=policy.epoch==epochs or policy.should_stop
            policy.save(output/'checkpoints',improved=improved,complete=complete)
            row=dict(epoch=policy.epoch,step=global_step,best_epoch=policy.best_epoch,best_f1=policy.best,
                metric=metric,validation=metrics,microbatch=microbatch,validation_batch=validation_batch,
                training_updates=len([r for r in totals if r['status']=='updated']))
            row['train']={key:sum(r.get('losses',{}).get(key,0.) for r in totals)/max(1,len(totals))
                          for key in {k for r in totals for k in r.get('losses',{})}}
            if args.arm=='observer':row['train']['o']=sum(r.get('loss',0.) for r in totals)/max(1,len(totals))
            row['gradient_norms']={key:max(r.get('gradient_norms',{}).get(key,0.) for r in totals)
                                   for key in {k for r in totals for k in r.get('gradient_norms',{})}}
            with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            write_json(output/'progress.json',row)
            if run is not None:run.log({metric:metrics[metric],'epoch':policy.epoch,'best_f1':policy.best,
                **{'train/'+k:v for k,v in row['train'].items()},
                **{'gradient_norm/'+k:v for k,v in row['gradient_norms'].items()}},step=global_step)
            print(json.dumps(row),flush=True)
        result=dict(status='completed_smoke_not_gate' if args.mode=='smoke' else 'complete',
            production_training='not run' if args.mode=='smoke' else 'run',test_used=False,checkpoint_selection=metric,
            best_epoch=policy.best_epoch,best_f1=policy.best,epochs=policy.epoch,steps=global_step,
            contract=contract,provenance=provenance,checkpoint_sha256=file_hash(output/'checkpoints/best.pt'))
        write_json(output/'completion.json',result)
        if run is not None:run.finish()
        return result
    except BaseException:
        if run is not None:run.finish(exit_code=1)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training_plans/messenger_natural_occlusion_v1.json')
    p.add_argument('--arm',choices=['observer','fixed','gd','frozen_g','gdt'],required=True)
    p.add_argument('--mode',choices=['smoke','production'],default='smoke')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--gates',type=Path)
    p.add_argument('--observer-checkpoint',type=Path);p.add_argument('--generator-checkpoint',type=Path)
    run(p.parse_args())
