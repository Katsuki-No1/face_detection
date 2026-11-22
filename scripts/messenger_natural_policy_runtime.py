"""Authenticated execution helpers for the independent image-policy experiment."""
import hashlib
import json
import math
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch

from scripts.train_messenger_natural_policy import (
    ROOT, validate_config, prepare_rows, authenticate, datasets, load_model,
    RealOcclusionBank, PolicyOcclusionDataset, PolicyMessengerTask, LandmarkLoss,
    CompositionPolicy, FullCropDiscriminator, PolicyUpdate, F1Checkpoint, T_METRIC,
    GateCounters, fixed_gate_probe, gate_deltas, validate_prior, file_hash, write_json,
    seed_everything,
)
from src.data.messenger_policy_occlusion import balanced_bins, audit_actions
from src.models.messenger_occlusion_policy import select_batch
from src.models.messenger_occlusion_augmentation import frozen
from keypoint_comparison1.landmark_detection.training import _scheduler


def authenticate_execution(path):
    cfg=json.loads(Path(path).read_text()); policy_path=ROOT/cfg['policy_config']
    policy=json.loads(policy_path.read_text());validate_config(policy)
    base,spec,parent,checkpoint,identity=authenticate(ROOT/policy['base_config'])
    if cfg['schema']!='messenger-policy-execution-v1' or cfg['effective_batch']!=64:
        raise ValueError('Invalid execution contract')
    override=json.loads((ROOT/cfg['override']).read_text())
    if file_hash(policy_path)!=override['target_config_sha256'] or cfg['small_scale_run_id']!=override['target_gate_run_id']:
        raise ValueError('One-off override identity mismatch')
    prerequisite=ROOT/cfg['ordering_prerequisite']
    history=json.loads(prerequisite.read_text())
    if not history or any(r['status']!='complete' for r in history):
        raise ValueError('Real-face Messenger production prerequisite incomplete')
    identity.update(execution_sha256=file_hash(path),policy_sha256=file_hash(policy_path),
                    override_sha256=file_hash(ROOT/cfg['override']),ordering_sha256=file_hash(prerequisite))
    # Capture runtime, gate and tests in addition to the inherited model/data inventory.
    for p in [Path(__file__),ROOT/'scripts/run_messenger_natural_policy_gate.py',
              ROOT/'tests/test_messenger_natural_policy_gate.py',
              ROOT/'keypoint_comparison1/landmark_detection/geometry_prior_validation.py']:
        identity['sources'][str(p.relative_to(ROOT))]=file_hash(p)
    contract=dict(schema='messenger-policy-formal-gates-v1',identity=identity,execution=cfg,
                  policy=policy,topology=policy['topology'],test_used=False)
    return cfg,policy,base,spec,parent,checkpoint,contract


def build_data(policy,base,spec):
    torch.set_num_threads(1);cv2.setNumThreads(0)
    train,val=datasets(spec); data=spec['parent_config']['data'];c=policy['composition']
    bank=RealOcclusionBank(ROOT/base['material_bank']['path'],base['material_bank']['sha256'],
        annotation_path=data['annotations']['train'],image_root=data['root'])
    dataset=PolicyOcclusionDataset(train,bank,split='train',seed=42,
        **{k:c[k] for k in ('probability','material_candidates','placements_per_material','attempts')})
    return dataset,val


def subset_indices(size,count):
    return sorted(range(size),key=lambda i:hashlib.sha256(f'42:formal-subset:{i}'.encode()).hexdigest())[:count]


def epoch_batches(dataset,indices,epoch):
    """Every selected row once. Equal-bin requests spread across effective windows."""
    dataset.set_epoch(epoch)
    eligible=set(dataset.eligible)
    dataset.requests=balanced_bins([i for i in indices if i in eligible],seed=42,epoch=epoch,
                                  probability=dataset.probability)
    rng=random.Random(f'42:formal-order:{epoch}')
    groups=[[i for i in indices if dataset.requests.get(i)==b] for b in range(4)]
    for group in groups:rng.shuffle(group)
    quota=min(map(len,groups));n=math.ceil(len(indices)/64);batches=[[] for _ in range(n)]
    for j in range(quota):
        batches[j%n].extend(group[j] for group in groups)
    assigned={i for batch in batches for i in batch}
    remaining=[i for i in indices if i not in assigned];rng.shuffle(remaining)
    for batch in batches:
        while len(batch)<64 and remaining:batch.append(remaining.pop())
        rng.shuffle(batch)
    if remaining or sorted(i for b in batches for i in b)!=sorted(indices):
        raise AssertionError('Batch partition lost/repeated a sample')
    return batches


def read_rows(dataset,ids):
    with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(dataset.__getitem__,ids))


class SafetyMonitor:
    def __enter__(self):
        self.rows=[];self.stop=threading.Event()
        def poll():
            while not self.stop.is_set():
                try:
                    total,used,temp=map(float,subprocess.check_output(['nvidia-smi',
                        '--query-gpu=memory.total,memory.used,temperature.gpu',
                        '--format=csv,noheader,nounits','--id=0'],text=True).strip().split(','))
                    self.rows.append(dict(total=total,used=used,temp=temp))
                except Exception as e:self.rows.append(dict(error=str(e)))
                self.stop.wait(.5)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def check(self):
        if not self.rows or any('error' in r for r in self.rows):raise RuntimeError('Missing safety measurements')
        if any(r['used']>.9*r['total'] or r['temp']>=83 for r in self.rows):raise RuntimeError('GPU safety limit exceeded')
    def metrics(self):
        self.check()
        return dict(maximum_temperature=max(r['temp'] for r in self.rows),
                    peak_vram_fraction=max(r['used']/r['total'] for r in self.rows),
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    def __exit__(self,*args):self.stop.set();self.thread.join()


class Runtime:
    def __init__(self,context,*,stage,microbatch,output):
        cfg,policy,base,spec,parent,checkpoint,contract=context
        self.cfg=cfg;self.contract=contract;self.stage=stage;self.output=output;self.microbatch=microbatch
        self.validation_batch=microbatch;self.validation_retries=[]
        torch.set_num_threads(1);seed_everything(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.device=torch.device('cuda:0')
        self.model,_=load_model(spec,parent,checkpoint,self.device)
        self.model.requires_grad_(True);self.model.observer.requires_grad_(False);self.model.bank.requires_grad_(False)
        self.model.base.messenger_recovery.mixing_enabled=True
        data=spec['parent_config']['data']
        self.task=PolicyMessengerTask(self.model,LandmarkLoss(spec['parent_config']['loss']),
            mean=data['mean'],std=data['std'],reconstruction_weight=base['task']['reconstruction_weight']).to(self.device)
        self.g=CompositionPolicy().to(self.device);self.d=FullCropDiscriminator().to(self.device)
        backbone={id(p) for p in self.model.base.backbone.parameters()}
        self.opts=dict(g=torch.optim.Adam(self.g.parameters(),lr=policy['generator']['learning_rate']),
            d=torch.optim.Adam(self.d.parameters(),lr=policy['discriminator']['learning_rate']),
            t=torch.optim.AdamW([
                dict(params=[p for p in self.task.parameters() if p.requires_grad and id(p) not in backbone],lr=base['task']['learning_rate']),
                dict(params=[p for p in self.model.base.backbone.parameters() if p.requires_grad],lr=base['task']['backbone_learning_rate'])],weight_decay=1e-4))
        # Construct with the complete trainable ownership, then stage backbone freezing.
        self.update=PolicyUpdate(self.g,self.d,self.task,self.opts,
            **{k:policy['generator'][k] for k in ('natural_weight','position_weight')})
        self.dataset,self.val=build_data(policy,base,spec)
        count=cfg[stage]['samples'] if stage in ('small_scale','overfit') else len(self.dataset)
        self.indices=subset_indices(len(self.dataset),count)
        epochs=cfg[stage]['epochs'] if stage in ('production','overfit','small_scale') else 40
        windows=math.ceil(count/64)
        self.schedulers={k:_scheduler(v,total_steps=windows*epochs,warmup_steps=windows*cfg['warmup_epochs'],
            minimum_ratio=cfg['minimum_lr_ratio']) for k,v in self.opts.items()}
        self.policy=F1Checkpoint(contract=dict(experiment=contract,stage=stage,microbatch=microbatch),metric=T_METRIC,
            modules=self.update.modules,optimizers=self.opts,schedulers=self.schedulers,
            loader_generator=torch.Generator().manual_seed(42),minimum_epoch=10,
            patience=5 if stage=='production' else 100,min_delta=1e-4)

    def freeze_epoch(self,epoch,*,full=False):
        self.model.base.backbone.requires_grad_(full or epoch>self.cfg['freeze_backbone_epochs'])

    def validation(self):
        while True:
            counters=GateCounters();clean=GateCounters()
            def collect(out,targets,hw):
                counters.add(out,targets,input_hw=hw,channel_adoption=self.model.channel_adoption)
                for i,t in enumerate(targets):
                    if not bool(t['face_occluded']) and bool(t['original_visible'].all()):
                        sliced={'prior_diagnostics':{k:out['prior_diagnostics'][k][i:i+1] for k in ('gate','anomalous')},
                                'decoded_coordinates':out['decoded_coordinates'][i:i+1]}
                        clean.add(sliced,[t],input_hw=hw,channel_adoption=self.model.channel_adoption)
            try:
                metrics=validate_prior(self.model,self.val,device=self.device,batch_size=self.validation_batch,diagnostics=collect)
                metrics['gates']=counters.compute();metrics['unoccluded_faces']=clean.compute();return metrics
            except torch.OutOfMemoryError:
                if self.validation_batch==1:raise
                self.validation_retries.append(self.validation_batch);self.validation_batch=max(1,self.validation_batch//2)
                torch.cuda.empty_cache()

    def train_window(self,rows):
        result=self.update.update(rows,microbatch=self.microbatch);self.microbatch=result['microbatch']
        for name,scheduler in self.schedulers.items():
            if name=='t' or result[name+'_updated']:scheduler.step()
        return result

    def fixed_loss(self,rows):
        fixed=[dict(r,_sampled_action=0) for r in rows]
        with frozen(self.task),torch.no_grad():
            fake,batch,*_=select_batch(self.g,fixed,self.device)
            return float(self.task.supervised(fake,batch))


def validation_contract(context):
    cfg,policy,base,spec,parent,checkpoint,contract=context
    return dict(annotations=contract['identity']['native_annotations']['val'],
        crop_sha256=file_hash(ROOT/spec['val_crops']),adapted_sha256=file_hash(ROOT/spec['val_annotations']),
        metric=T_METRIC,presence=.7,nme=.1,normalizer='1.20 * max(GT Head width, GT Head height)',
        crops=768,source_sha256=file_hash(ROOT/'keypoint_comparison1/landmark_detection/geometry_prior_validation.py'),
        precision=cfg['precision']['validation'])


def start_wandb(context,output,stage,resume):
    import wandb
    cfg=context[0]
    binding=output/'wandb_binding.json'
    if not binding.exists():write_json(binding,dict(id=wandb.util.generate_id()))
    run=wandb.init(project=cfg['wandb']['project'],id=json.loads(binding.read_text())['id'],
        name=cfg['architecture']+'/'+stage,dir=str(output),mode='online',resume='must' if resume else 'never',
        config=dict(contract=context[-1],stage=stage),settings=wandb.Settings(init_timeout=90))
    if run.offline:raise ValueError('Online W&B required')
    write_json(output/'wandb.json',dict(id=run.id,url=run.url,project=run.project));return run


def scalar_report(metrics,probe):
    def group(c,key):return c['groups'][key]
    groups=dict(artificial=group(probe['synthetic'],'artificial_occluded'),
        natural=group(metrics['gates'],'natural_occluded'),visible=group(metrics['gates'],'visible'),
        unoccluded_face=group(metrics['unoccluded_faces'],'visible'))
    result={'validation/f1':metrics[T_METRIC]}
    for name,values in groups.items():
        for k in ('count','g_mean','effective_coefficient_mean','position_smooth_l1_mean'):
            if values[k] is not None:result[name+'/'+k]=values[k]
    return result
