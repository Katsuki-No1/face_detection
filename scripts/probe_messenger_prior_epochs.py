#!/usr/bin/env python3
"""Two diagnostic epochs plus full conditional val and epoch resume; NOT model gates."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from torch.utils.data import DataLoader,Subset
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset,CropAugmentation,collate_landmark_batch
from keypoint_comparison1.landmark_detection.geometry_prior_training import build_prior_model,generator_optimizer,PriorUpdateRuntime,file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_epochs import PriorEpochDriver,save_epoch,finalize_checkpoints
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import backbone_state_sha256
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.training import seed_everything,_scheduler


def run(output):
    torch.set_num_threads(1);seed_everything(42)
    plan=yaml.safe_load((ROOT/'configs/training_plans/messenger_geometry_prior_v1.yaml').read_text())
    raw=load_landmark_config(ROOT/plan['initialization']['parent_training_config']).raw
    prep=ROOT/plan['feature_preparation']['output']
    report=json.loads((prep/'priors/preparation_report.json').read_text())
    prior=next(r for r in report['candidates'] if r['k']==1)
    valroot=ROOT/'outputs/landmark_detection/messenger_geometry_prior/preparation/validation_20261012'
    val_report=json.loads((valroot/'preparation_complete.json').read_text())
    for name,sha in val_report['artifacts'].items():
        if file_hash(valroot/name)!=sha:raise ValueError('Validation preparation artifact changed')
    crop=prep/'train_predicted_crops.json'
    if file_hash(crop)!=report['metadata']['crop_cache_sha256']:raise ValueError('Train cache changed')
    output.mkdir(parents=True,exist_ok=False)
    shared=dict(image_root=raw['data']['root'],input_size=256,category_ids=(2,),
        require_visible_keypoint=False,augmentation=CropAugmentation(**raw['data']['augmentation']),
        mean=raw['data']['mean'],std=raw['data']['std'],geometry_prior_grid_size=(16,16),seed=42)
    train=TopDownFaceLandmarkDataset(annotation_path=raw['data']['annotations']['train'],training=True,
        detector_crop_cache=crop,**shared)
    val=TopDownFaceLandmarkDataset(annotation_path=valroot/'val_head_keypoints.json',training=False,
        detector_crop_cache=valroot/'val_predicted_crops.json',**shared)
    if len(val)!=val_report['audit']['matched_eligible_heads']:raise ValueError('Conditional validation coverage changed')
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(Subset(train,range(128)),batch_size=64,shuffle=True,num_workers=0,
        collate_fn=collate_landmark_batch,generator=generator)
    model,initialization=build_prior_model(raw,prior_path=prior['path'],prior_sha256=prior['sha256'])
    model=model.cuda();observer_hash=backbone_state_sha256(model.observer)
    optimizer=generator_optimizer(model,raw['train'])
    runtime=PriorUpdateRuntime(model,LandmarkLoss(raw['loss']).cuda(),optimizer,
        contract=dict(scope='epoch_integration_diagnostic_not_production',prior=prior['sha256'],validation=file_hash(valroot/'preparation_complete.json')),amp=True)
    scheduler=_scheduler(optimizer,total_steps=4,warmup_steps=2,minimum_ratio=.01)
    driver=PriorEpochDriver(runtime,schedulers={'g':scheduler},loader_generator=generator,
        effective_batch=64,microbatch=4,freeze_backbone_epochs=1,mixing_warmup_epochs=1,
        minimum_epoch=1,patience=5,maximum_epochs=2)
    validate=lambda m:validate_prior(m,val,device=torch.device('cuda:0'),batch_size=16)
    rows=[]
    def record(row):
        total,temp=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
        row['gpu_temperature']=temp
        row['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        if temp>=83 or row['peak_reserved_mib']>total*.90:raise RuntimeError('Unsafe GPU diagnostic')
        rows.append(row)
        print(json.dumps(row),flush=True)
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix='epoch_check_',dir=output) as temp:
        checkpoint_dir=Path(temp)
        driver.run_epoch(loader,validate,logger=record);save_epoch(driver,checkpoint_dir)
        first=checkpoint_dir/'resume_epoch1.pt'
        torch.save(driver.checkpoint(),first)
        driver.run_epoch(loader,validate,logger=record);save_epoch(driver,checkpoint_dir)
        expected_step=runtime.step
        driver.restore(torch.load(first,map_location='cpu',weights_only=False))
        driver.run_epoch(loader,validate,logger=record)
        if runtime.step!=expected_step:raise ValueError('Epoch resume step differs')
        save_epoch(driver,checkpoint_dir);finalize_checkpoints(driver,checkpoint_dir)
    if backbone_state_sha256(model.observer)!=observer_hash or backbone_state_sha256(runtime.ema.module.observer)!=observer_hash:
        raise ValueError('Frozen observer changed')
    result=dict(status='epoch_validation_integration_checked_not_gate',production_training='not run',
        formal_four_gates='not run',scope='128 train samples; full530-image cached FACE inference ->768 eligible predicted crops for conditional validation; diagnostic schedule only',
        diagnostic_schedule=dict(max_epochs=2,freeze_epochs=1,warmup_epochs=1,mixing_warmup_epochs=1,effective_batch=64,microbatch=4),
        production_schedule='Unchanged: max40, freeze/warmup5, min10/patience5; not run',
        rows=rows,initialization=initialization,validation_preparation=val_report['audit'],
        observer_and_ema='bitwise_unchanged',epoch_checkpoint_resume='passed',
        maximum_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        checkpoints='temporary diagnostic checkpoints removed; no production checkpoint',wandb='not run: integration diagnostic only')
    (output/'probe_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Epoch/validation integration check completed',flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
