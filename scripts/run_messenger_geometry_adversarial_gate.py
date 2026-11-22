#!/usr/bin/env python3
"""Four fresh authenticated gates for validation-selected k with adversarial on."""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,Dataset,Subset
from scripts.train_messenger_geometry_adversarial import datasets
from scripts.messenger_geometry_adversarial_contract import authenticate, AdversarialRuntime as PriorUpdateRuntime
from scripts.run_messenger_geometry_prior_production import ensure_descriptor_limit
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import write_json,STAGES
from keypoint_comparison1.landmark_detection.geometry_prior_training import build_prior_model,generator_optimizer,file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_epochs import PriorEpochDriver,METRIC
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.training import seed_everything,_scheduler


class QualityFailure(RuntimeError):pass


def assess(values, *, higher, minimum_delta):
    if len(values)<2 or not all(math.isfinite(v) for v in values):
        raise QualityFailure('Missing or non-finite metric trajectory')
    best=(max if higher else min)(values[1:])
    delta=(best-values[0]) if higher else (values[0]-best)
    return dict(passed=delta>minimum_delta,initial=values[0],best_after_initial=best,
        improvement=delta,minimum_delta_exclusive=minimum_delta,direction='higher' if higher else 'lower',values=values)


def certify_preflight(report, identity, source_sha):
    if report.get('status')!='preflight_completed_not_formal_gate' or report.get('identity')!=identity:
        raise ValueError('Stale or incomplete preflight')
    if report.get('preflight_source_sha256')!=source_sha or report.get('candidates')!=[4,8,16] or report.get('effective_batch')!=64:
        raise ValueError('Preflight implementation or effective batch differs')
    attempts=report['attempts']
    if [a['microbatch'] for a in attempts]!=[4,8,16]:raise ValueError('Candidate inventory differs')
    for a in attempts:
        if a['status']!='safe':continue
        if not a.get('discriminator_updates_exercised'):
            raise ValueError('Adversarial GPU gate requires committed discriminator updates')
        if (a.get('identity')!=identity or a.get('preflight_source_sha256')!=source_sha
                or a.get('effective_batch')!=64 or a.get('full_validation_instances')!=768
                or len(a['rows'])!=2 or [r['step'] for r in a['rows']]!=[2,4]
                or any(r['backbone_frozen'] or r['amp_overflows'] or r['microbatch']!=a['microbatch'] for r in a['rows'])
                or a['maximum_temperature']>=83 or a['maximum_device_used_mib']>24564*.90
                or a['peak_reserved_mib']>24564*.90 or a['projected_seconds']<=0):
            raise ValueError('Safe preflight evidence violates the declared protocol')
        for r in a['rows']:
            if r['validation']['eligible_face_crops']!=768 or min(r['train_seconds'],r['validation_seconds'])<=0:
                raise ValueError('Missing full conditional validation or timing')
            numbers=[*r['train'].values(),r['validation'][METRIC],r['train_seconds'],r['validation_seconds']]
            if not all(math.isfinite(v) for v in numbers):raise ValueError('Non-finite candidate metrics')
    safe=[a for a in attempts if a['status']=='safe']
    if not safe:raise QualityFailure('No safe microbatch')
    fastest=min(a['projected_seconds'] for a in safe)
    chosen=min((a for a in safe if a['projected_seconds']<=fastest*1.03),key=lambda a:a['microbatch'])
    if chosen['microbatch']!=report['selected_microbatch']:raise ValueError('Batch choice does not match measured time')
    return dict(report,status='passed')


def schema(spec,identity):
    diagnostic=ROOT/'outputs/landmark_detection/messenger_geometry_prior/adversarial_launcher_check_20261013'
    result=json.loads((diagnostic/'launcher_check.json').read_text())
    if (result.get('identity')!=identity or result.get('status')!='launcher_checked_not_gate'
            or not result.get('resumed') or result.get('epoch')!=2 or result.get('step')!=4):
        raise ValueError('Adversarial launcher must pass real W&B/checkpoint resume diagnostic')
    state=torch.load(diagnostic/'checkpoints/last.pt',map_location='cpu',weights_only=False,mmap=True)
    if (not state.get('discriminator_optimizer',{}).get('state') or not state.get('discriminator')
            or set(state.get('schedulers',{}))!={'g','d'}):
        raise ValueError('Missing discriminator updates or two-scheduler resume evidence')
    del state
    raw=spec['parent_config'];cfg=raw['train']
    if raw['evaluation']['pipeline_contract']!={'type':'cascade','stages':['full_image_face_ellipse','face_crop_keypoint']}:
        raise ValueError('Missing HEAD-free ordered topology')
    expected=dict(epochs=40,warmup_epochs=5,flat_epochs=5,freeze_backbone_epochs=5,seed=42,intended_effective_batch_size=64)
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Training schedule changed')
    if (cfg['early_stopping']['minimum_epoch']!=10 or cfg['early_stopping']['patience_validations']!=5
            or cfg['early_stopping']['min_delta']!=1e-4 or cfg['checkpoint_selection']['metric']!=METRIC
            or raw['model']['continuous_recovery']['mixing_warmup_epochs']!=2):
        raise ValueError('Selection/mixing policy changed')
    inventory={}
    for split,record in raw['data']['continuous_source_inventory'].items():
        p=ROOT/record['path'];d=json.loads(p.read_text())
        actual=dict(images=len(d['images']),heads=len(d['annotations']),groups=len({i['source_video_id'] for i in d['images']}))
        if any(actual[k]!=record[k] for k in actual) or file_hash(p)!=record['sha256']:raise ValueError(f'Canonical {split} changed')
        inventory[split]=dict(actual,sha256=record['sha256'])
    train,val=datasets(spec)
    if len(train)!=6185 or len(val)!=768:raise ValueError('Predicted crop cohort changed')
    command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_geometry_prior_launch.py','tests/test_geometry_prior_epochs.py','tests/test_geometry_prior_validation.py','tests/test_geometry_prior_training.py','tests/test_geometry_prior_inputs.py','keypoint_comparison1/tests/test_geometry_prior_recovery.py','tests/test_messenger_prior_gates.py','tests/test_messenger_adversarial_contract.py']
    subprocess.run(command,cwd=ROOT,check=True)
    return dict(topology=['full_image_face_ellipse','face_crop_dino','messenger_prior_fusion','query_refinement'],canonical_inventory=inventory,train_instances=len(train),conditional_val_instances=len(val),tests=command)


class FixedSamples(Dataset):
    def __init__(self,dataset):
        dataset.set_epoch(0)
        self.samples=[dataset[i] for i in range(4)]
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return self.samples[index]


def learning(spec,identity,directory,*,small_scale):
    raw=spec['parent_config'];cfg=raw['train'];seed_everything(42);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=cfg['tf32'];torch.backends.cudnn.allow_tf32=cfg['tf32']
    train,val=datasets(spec);sampled=Subset(train,range(256)) if small_scale else FixedSamples(train)
    plan=json.loads((directory.parent/'gpu_step/batch_plan.json').read_text())
    if plan['identity']!=identity or plan['status']!='passed':raise ValueError('Stale GPU plan')
    batch=plan['selected_microbatch'] if small_scale else 4
    gen=torch.Generator().manual_seed(42)
    loader=DataLoader(sampled,batch_size=64,shuffle=True,num_workers=cfg['num_workers'] if small_scale else 0,
        collate_fn=collate_landmark_batch,generator=gen,persistent_workers=False)
    prior=spec['prior'];model,initialization=build_prior_model(raw,prior_path=ROOT/prior['path'],prior_sha256=prior['sha256'],fusion_mode='messenger' if spec['arm']=='control' else 'prior')
    model=model.cuda();opt=generator_optimizer(model,cfg)
    runtime=PriorUpdateRuntime(model,LandmarkLoss(raw['loss']).cuda(),opt,contract=dict(identity=identity,gate='small_scale' if small_scale else 'overfit'),amp=cfg['amp'],max_grad_norm=cfg['max_grad_norm'],ema_decay=cfg['ema']['decay'])
    epochs=20 if small_scale else 12
    scheduler=_scheduler(opt,total_steps=len(loader)*epochs,warmup_steps=len(loader)*5,minimum_ratio=cfg['minimum_lr_ratio'])
    driver=PriorEpochDriver(runtime,schedulers={'g':scheduler,'d':_scheduler(runtime.d_optimizer,total_steps=len(loader)*epochs,warmup_steps=len(loader)*5,minimum_ratio=cfg['minimum_lr_ratio'])},loader_generator=gen,effective_batch=64,microbatch=batch,
        freeze_backbone_epochs=5,mixing_warmup_epochs=2,minimum_epoch=10,patience=100,maximum_epochs=epochs)
    validate=lambda m:validate_prior(m,val,device=torch.device('cuda:0'),batch_size=batch)
    rows=[];values=[];best=-math.inf if small_scale else math.inf
    if small_scale:values.append(validate(runtime.ema.module)[METRIC])
    for _ in range(epochs):
        row=driver.run_epoch(loader,validate);rows.append(row)
        metric=row['validation'][METRIC] if small_scale else row['train']['total'];values.append(metric)
        if (metric>best if small_scale else metric<best):
            best=metric;torch.save(driver.checkpoint(),directory/'best.pt')
        write_json(directory/'trajectory.json',dict(rows=rows,values=values))
        print(json.dumps(dict(epoch=row['epoch'],step=row['step'],metric=metric)),flush=True)
    assessment=assess(values,higher=small_scale,minimum_delta=1e-4 if small_scale else 0.)
    details=dict(assessment=assessment,initialization=initialization,checkpoint=str(directory/'best.pt'),training_instances=len(sampled),conditional_val_instances=len(val))
    write_json(directory/'assessment.json',details)
    if not assessment['passed']:raise QualityFailure('Predeclared learning improvement failed; see assessment.json')
    return details


def run(config,stage):
    ensure_descriptor_limit()
    spec,identity=authenticate(config);directory=ROOT/'outputs/training_gates'/spec['architecture']/stage
    directory.mkdir(parents=True,exist_ok=False)
    policy=dict(overfit=dict(fixed_train_samples=4,epochs=12,metric='train.total',direction='lower',minimum_delta_exclusive=0.),
        small_scale=dict(train_samples=256,epochs=20,metric=METRIC,direction='higher',minimum_delta_exclusive=1e-4,initial='epoch0 EMA full conditional validation'),acceptance='best subsequent metric improves initial; finite values required; no threshold retries')
    write_json(directory/'declared_policy.json',policy)
    try:
        if stage!='schema':
            previous=STAGES[STAGES.index(stage)-1];marker=json.loads((directory.parent/previous/'success.json').read_text())
            if marker.get('identity')!=identity or marker.get('status')!='passed':raise ValueError('Previous gate missing/stale')
        if stage=='schema':details=schema(spec,identity)
        elif stage=='gpu_step':
            p=ROOT/'outputs/landmark_detection/messenger_geometry_prior/adversarial_preflight_20261013'/spec['arm']/'preflight_complete.json'
            details=certify_preflight(json.loads(p.read_text()),identity,file_hash(ROOT/'scripts/preflight_messenger_geometry_adversarial.py'))
            details['evidence_path']=str(p);details['evidence_sha256']=file_hash(p);write_json(directory/'batch_plan.json',details)
        else:details=learning(spec,identity,directory,small_scale=stage=='small_scale')
        result=dict(status='passed',architecture=spec['architecture'],gate_stage=stage,identity=identity,gate_implementation_sha256=file_hash(__file__),details=details)
        write_json(directory/'report.json',result);write_json(directory/'success.json',result)
    except Exception as error:
        write_json(directory/'report.json',dict(status='failed',classification='quality_failure' if isinstance(error,QualityFailure) else 'execution_error',error=str(error),traceback=traceback.format_exc(),identity=identity,gate_stage=stage))
        raise


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=STAGES,required=True);a=p.parse_args();run(a.config,a.stage)
