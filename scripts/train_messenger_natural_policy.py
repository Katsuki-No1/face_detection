#!/usr/bin/env python3
"""Bounded CPU policy smoke with real train frames and full validation F1.

Not a production launcher/gate. Existing model, queue, and V1 contracts remain
unchanged. Full module/optimizer/loader/global RNG state uses F1Checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import write_json
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank
from keypoint_comparison1.landmark_detection.training import seed_everything
from scripts.evaluate_messenger_geometry_prior import load_model
from scripts.messenger_natural_occlusion_common import authenticate,datasets,TOPOLOGY
from src.data.messenger_policy_occlusion import POLICY_BINS,PolicyOcclusionDataset,audit_actions
from src.models.messenger_occlusion_policy import CompositionPolicy,FullCropDiscriminator,PolicyMessengerTask,PolicyUpdate
from src.models.messenger_occlusion_training import F1Checkpoint,T_METRIC
from src.models.messenger_gate_metrics import GateCounters,fixed_gate_probe,gate_deltas


def validate_config(cfg):
    c=cfg['composition']
    if (cfg['schema']!='messenger-natural-policy-v2' or cfg['topology']!=TOPOLOGY
        or c['split']!='train' or c['bins']!=[list(b) for b in POLICY_BINS]
        or c['bin_fractions']!=[.25]*4 or c['alpha'] is not False
        or cfg['discriminator']['input']!='full_face_crop_rgb'
        or cfg['discriminator']['loss']!='non_saturating_logistic'
        or cfg['task']['covered_position_weight']!=1.
        or cfg['task']['selection']!=T_METRIC or cfg['production']['enabled']):
        raise ValueError('Unsupported policy/topology/selection/production contract')


def prepare_rows(dataset, cfg):
    """Feasible per-bin fixture; scan failures remain explicit, never full-train evidence."""
    rows=[];scanned=[]
    for bin_id in range(4):
        ids=[i for i,b in dataset.requests.items() if b==bin_id]
        ids.sort(key=lambda i:hashlib.sha256(f'{dataset.seed}:{i}:policy-smoke'.encode()).hexdigest())
        count=0
        for i in ids[:cfg['candidate_scan_per_bin']]:
            row=dataset[i];scanned.append(dict(row['metadata'],candidate_count=len(row['candidates'])))
            if row['candidates']:
                rows.append(row);count+=1
            if count==cfg['requested_per_bin']:break
        if count!=cfg['requested_per_bin']:raise RuntimeError(f'Insufficient smoke fixtures in bin {bin_id}: {scanned}')
    natural=[i for i in range(len(dataset)) if i not in dataset.requests]
    natural.sort(key=lambda i:hashlib.sha256(f'{dataset.seed}:{i}:natural-smoke'.encode()).hexdigest())
    rows.extend(dataset[i] for i in natural[:cfg['natural_crops']])
    return rows,scanned


def run(args):
    cfg=json.loads(args.config.read_text());validate_config(cfg)
    base,spec,parent_identity,checkpoint,identity=authenticate(ROOT/cfg['base_config'])
    identity=dict(identity,policy_config_sha256=file_hash(args.config))
    # authenticate() discovers these sources as well; explicitly retain the runner.
    identity['sources'][str(Path(__file__).resolve().relative_to(ROOT))]=file_hash(__file__)
    contract=dict(identity=identity,config=cfg,mode='implementation_smoke_not_gate')
    output=args.output.resolve()
    if args.resume:
        if json.loads((output/'contract.json').read_text())!=contract or (output/'completion.json').exists():
            raise ValueError('Changed contract or already completed run')
    else:
        output.mkdir(parents=True,exist_ok=False);write_json(output/'contract.json',contract)
        for name in identity['sources']:
            target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    torch.set_num_threads(2);seed_everything(cfg['seed']);device=torch.device('cpu')
    parent,_=load_model(spec,parent_identity,checkpoint,device)
    parent.requires_grad_(True);parent.observer.requires_grad_(False);parent.bank.requires_grad_(False)
    parent.base.backbone.requires_grad_(False);parent.base.messenger_recovery.mixing_enabled=True
    train,val=datasets(spec);data=spec['parent_config']['data'];c=cfg['composition'];rt=cfg['runtime']
    bank=RealOcclusionBank(ROOT/base['material_bank']['path'],base['material_bank']['sha256'],
        annotation_path=data['annotations']['train'],image_root=data['root'])
    dataset=PolicyOcclusionDataset(train,bank,split='train',seed=cfg['seed'],
        **{k:c[k] for k in ('probability','material_candidates','placements_per_material','attempts')})
    rows,scanned=prepare_rows(dataset,rt)
    write_json(output/'fixtures.json',dict(scope='stratified feasible smoke subset; not full-train audit',
        scanned=scanned,indices=[r['metadata']['index'] for r in rows],
        candidates=[[c['metadata'] for c in r['candidates']] for r in rows]))
    task=PolicyMessengerTask(parent,LandmarkLoss(spec['parent_config']['loss']),mean=data['mean'],std=data['std'],
        reconstruction_weight=base['task']['reconstruction_weight'])
    g=CompositionPolicy();d=FullCropDiscriminator()
    opts={'g':torch.optim.Adam(g.parameters(),lr=cfg['generator']['learning_rate']),
          'd':torch.optim.Adam(d.parameters(),lr=cfg['discriminator']['learning_rate']),
          't':torch.optim.AdamW([p for p in task.parameters() if p.requires_grad],lr=base['task']['learning_rate'],weight_decay=1e-4)}
    runtime=PolicyUpdate(g,d,task,opts,**{k:cfg['generator'][k] for k in ('natural_weight','position_weight')})
    loader_rng=torch.Generator().manual_seed(cfg['seed'])
    policy=F1Checkpoint(contract=contract,metric=T_METRIC,modules={'g':g,'d':d,'t':task},
        optimizers=opts,schedulers={},loader_generator=loader_rng,minimum_epoch=1,patience=5)
    def validation():
        gates=GateCounters()
        metrics=validate_prior(parent,val,device=device,batch_size=4,
            diagnostics=lambda result,targets,hw:gates.add(result,targets,input_hw=hw,channel_adoption=parent.channel_adoption))
        metrics['gates']=gates.compute()
        return metrics
    if args.resume:
        policy.restore(torch.load(output/'checkpoints/last.pt',map_location='cpu',weights_only=False))
        runtime.step=policy.last_metrics['optimizer_step']
    else:
        write_json(output/'epoch0_validation.json',validation())
        write_json(output/'gate_reference.json',fixed_gate_probe(task,rows))
    gate_initial=json.loads((output/'gate_reference.json').read_text())
    val_initial=json.loads((output/'epoch0_validation.json').read_text())['gates']
    frozen={n:v.clone() for n,v in parent.state_dict().items() if isinstance(v,torch.Tensor) and n.startswith(('bank.','observer.','base.backbone.'))}
    while policy.epoch<rt['epochs']:
        order=torch.randperm(len(rows),generator=loader_rng).tolist();updates=[]
        for start in range(0,len(rows),rt['effective_batch']):
            update=runtime.update([rows[i] for i in order[start:start+rt['effective_batch']]],microbatch=rt['microbatch'])
            updates.append(update)
        assert all(torch.equal(v,parent.state_dict()[n]) for n,v in frozen.items())
        metrics=validation();metrics['optimizer_step']=runtime.step
        metrics['gates_delta_from_initial']=gate_deltas(metrics['gates'],val_initial)
        probe=fixed_gate_probe(task,rows)
        gate_report=dict(validation=metrics['gates'],fixed_train_probe=probe,
            validation_delta_from_initial=metrics['gates_delta_from_initial'],
            fixed_train_delta_from_initial=gate_deltas(probe['synthetic'],gate_initial['synthetic']))
        improved=policy.record(metrics);policy.save(output/'checkpoints',improved=improved,complete=policy.epoch==rt['epochs'])
        record=dict(epoch=policy.epoch,validation=metrics,updates=updates,
            composition=audit_actions([r for u in updates for r in u['records']]),gates=gate_report)
        with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        write_json(output/'progress.json',record)
        write_json(output/'gate_progress.json',dict(epoch=policy.epoch,**gate_report))
        print(json.dumps(dict(epoch=policy.epoch,f1=metrics[T_METRIC],losses=updates[-1]['losses'],
            g_artificial=probe['synthetic']['groups']['artificial_occluded'],
            g_natural=metrics['gates']['groups']['natural_occluded'],g_visible=metrics['gates']['groups']['visible'])),flush=True)
    write_json(output/'completion.json',dict(status='completed_implementation_smoke_not_gate',
        production_training='not run',test_used=False,best_f1=policy.best,best_epoch=policy.best_epoch,
        steps=runtime.step,train_crops=len(rows),validation_crops=len(val),
        observer_O_backbone_unchanged=True,checkpoint_sha256=file_hash(output/'checkpoints/best.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training_plans/messenger_natural_policy_v2.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    run(p.parse_args())
