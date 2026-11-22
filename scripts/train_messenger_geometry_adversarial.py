#!/usr/bin/env python3
"""Selected-k adversarial production, with separate G/D schedules and faithful resume."""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader, Subset
from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset, CropAugmentation, collate_landmark_batch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import verify_gates, bind_run, start_wandb, write_json
from scripts.messenger_geometry_adversarial_contract import authenticate, AdversarialRuntime as PriorUpdateRuntime
from scripts.run_messenger_geometry_prior_production import ensure_descriptor_limit
from keypoint_comparison1.landmark_detection.geometry_prior_training import build_prior_model, generator_optimizer
from keypoint_comparison1.landmark_detection.geometry_prior_epochs import PriorEpochDriver, save_epoch, finalize_checkpoints, wandb_epoch_logger
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.training import seed_everything, _scheduler


def datasets(spec):
    raw = spec['parent_config']; data = raw['data']
    shared = dict(image_root=data['root'], input_size=data['input_size'], category_ids=data['category_ids'],
        require_visible_keypoint=False, augmentation=CropAugmentation(**data['augmentation']),
        mean=data['mean'], std=data['std'], geometry_prior_grid_size=(16,16), seed=42)
    train = TopDownFaceLandmarkDataset(annotation_path=data['annotations']['train'], training=True,
        detector_crop_cache=ROOT/spec['train_crops'],
        structure_head_annotations=data['structure_head_annotations'], **shared)
    val = TopDownFaceLandmarkDataset(annotation_path=ROOT/spec['val_annotations'], training=False,
        detector_crop_cache=ROOT/spec['val_crops'], **shared)
    if len(val) != 768:
        raise ValueError('Conditional validation cohort changed')
    return train, val


def run(args):
    ensure_descriptor_limit()
    spec, identity = authenticate(args.config)
    diagnostic = args.mode == 'launcher-check'
    batch = 4 if diagnostic else verify_gates(spec, identity)
    if not diagnostic and args.output:
        raise ValueError('Production output is fixed by config')
    output = args.output if diagnostic else ROOT/spec['output_root']
    if output is None:
        raise ValueError('Diagnostic requires separate output')
    output = output.resolve()
    if diagnostic and (output == (ROOT/spec['output_root']).resolve()
            or (ROOT/spec['output_root']).resolve() in output.parents):
        raise ValueError('Diagnostic must be outside production output')
    if not diagnostic and args.stop_after_epoch:
        raise ValueError('Production cannot change its stopping boundary')
    contract = dict(identity=identity, architecture=spec['architecture'], mode=args.mode,
        output=str(output), seed=42, microbatch=batch, effective_batch=64,
        diagnostic_overrides=dict(train_samples=128, maximum_epochs=2, freeze_epochs=1,
            warmup_epochs=1, mixing_warmup_epochs=1, minimum_epoch=1) if diagnostic else None)
    binding = bind_run(output, contract, resume=args.resume)
    torch.set_num_threads(1);seed_everything(42)
    raw = spec['parent_config'];cfg=raw['train']
    torch.backends.cuda.matmul.allow_tf32=bool(cfg['tf32'])
    torch.backends.cudnn.allow_tf32=bool(cfg['tf32'])
    train, val = datasets(spec)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(Subset(train, range(128)) if diagnostic else train,
        batch_size=64, shuffle=True, drop_last=False, num_workers=0 if diagnostic else int(cfg['num_workers']),
        collate_fn=collate_landmark_batch, generator=generator, persistent_workers=False)
    prior=spec['prior']
    model, initialization = build_prior_model(raw, prior_path=ROOT/prior['path'], prior_sha256=prior['sha256'],
        fusion_mode='messenger' if spec['arm']=='control' else 'prior')
    model=model.cuda();optimizer=generator_optimizer(model,cfg)
    runtime=PriorUpdateRuntime(model,LandmarkLoss(raw['loss']).cuda(),optimizer,contract=contract,
        amp=cfg['amp'],max_grad_norm=cfg['max_grad_norm'],ema_decay=cfg['ema']['decay'])
    maximum=2 if diagnostic else int(cfg['epochs']);warmup=1 if diagnostic else int(cfg['warmup_epochs'])
    scheduler=_scheduler(optimizer,total_steps=len(loader)*maximum,warmup_steps=len(loader)*warmup,
        minimum_ratio=float(cfg['minimum_lr_ratio']))
    driver=PriorEpochDriver(runtime,schedulers={'g':scheduler,'d':_scheduler(runtime.d_optimizer,total_steps=len(loader)*maximum,warmup_steps=len(loader)*warmup,minimum_ratio=float(cfg['minimum_lr_ratio']))},loader_generator=generator,
        effective_batch=64,microbatch=batch,freeze_backbone_epochs=1 if diagnostic else cfg['freeze_backbone_epochs'],
        mixing_warmup_epochs=1 if diagnostic else raw['model']['continuous_recovery']['mixing_warmup_epochs'],
        minimum_epoch=1 if diagnostic else cfg['early_stopping']['minimum_epoch'],
        patience=cfg['early_stopping']['patience_validations'],min_delta=cfg['early_stopping']['min_delta'],maximum_epochs=maximum)
    checkpoint=output/'checkpoints/last.pt'
    if args.resume and checkpoint.exists():
        driver.restore(torch.load(checkpoint,map_location='cpu',weights_only=False))
    run=start_wandb(spec,binding,output,resume=args.resume,diagnostic=diagnostic)
    log=wandb_epoch_logger(run)
    write_json(output/'initialization.json',initialization)
    validation_batch=batch
    def validate(m):
        nonlocal validation_batch
        while True:
            try:return validate_prior(m,val,device=torch.device('cuda:0'),batch_size=validation_batch)
            except torch.OutOfMemoryError:
                if validation_batch==1:raise
                validation_batch=max(1,validation_batch//2);torch.cuda.empty_cache()
    try:
        if driver.epoch==0:
            write_json(output/'epoch0_validation.json',validate(runtime.ema.module))
        while driver.epoch<maximum and not driver.should_stop:
            row=driver.run_epoch(loader,validate)
            save_epoch(driver,output/'checkpoints')
            # Checkpoint is durable before remote logging; replay must restore it.
            log(row)
            write_json(output/'progress.json',row)
            print(json.dumps(row),flush=True)
            if args.stop_after_epoch and driver.epoch>=args.stop_after_epoch:break
        completed=driver.epoch==maximum or driver.should_stop
        if completed:finalize_checkpoints(driver,output/'checkpoints')
        result=dict(status=('launcher_checked_not_gate' if diagnostic else 'complete') if completed else 'checkpointed',
            production_training='not run' if diagnostic else 'run',identity=identity,
            epoch=driver.epoch,step=runtime.step,best_epoch=driver.best_epoch,best_value=driver.best,
            resumed=args.resume,wandb_url=run.url,validation_batch=validation_batch)
        write_json(output/(('launcher_check.json' if completed else 'launcher_pause.json') if diagnostic else 'training_success.json'),result)
        run.finish()
        return result
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=['production','launcher-check'],default='production')
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epoch',type=int)
    run(p.parse_args())
