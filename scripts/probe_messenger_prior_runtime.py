#!/usr/bin/env python3
"""Real-data runtime integration check; NOT a production preflight or model gate."""
from argparse import ArgumentParser
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import numpy as np
import yaml
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset,CropAugmentation
from keypoint_comparison1.landmark_detection.geometry_prior_training import build_prior_model,generator_optimizer,PriorUpdateRuntime,file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import ConditionalFeatureDiscriminator,backbone_state_sha256
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.training import seed_everything


def gpu():
    line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    total,temp=map(float,line.split(','))
    return dict(total_mib=total,temperature=temp)


def compare_state(left,right,path='checkpoint'):
    if torch.is_tensor(left):
        torch.testing.assert_close(left.cpu(),right.cpu(),rtol=0,atol=0,msg=lambda message:f'{path}: {message}')
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        if set(left)!=set(right):raise ValueError(f'{path}: State key mismatch')
        for key in left:compare_state(left[key],right[key],f'{path}.{key}')
    elif isinstance(left,(tuple,list)):
        if len(left)!=len(right):raise ValueError(f'{path}: State length mismatch')
        for index,(a,b) in enumerate(zip(left,right)):compare_state(a,b,f'{path}[{index}]')
    elif left!=right:raise ValueError(f'{path}: Metadata mismatch')


def update_difference(left,right):
    return max(float((value-right[key].detach().cpu()).abs().max())
        for key,value in left.items() if torch.is_tensor(value) and value.is_floating_point())


def main(args):
    torch.set_num_threads(1);seed_everything(42)
    # The deployed Query uses grid_sample backward, which is nondeterministic
    # on CUDA. Require exact restored state; next-update drift is diagnostic.
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    plan=yaml.safe_load((ROOT/'configs/training_plans/messenger_geometry_prior_v1.yaml').read_text())
    raw=load_landmark_config(ROOT/plan['initialization']['parent_training_config']).raw
    prep=ROOT/plan['feature_preparation']['output']
    report=json.loads((prep/'priors/preparation_report.json').read_text())
    candidate=next(row for row in report['candidates'] if row['k']==1)
    if candidate['status']!='prepared' or file_hash(candidate['path'])!=candidate['sha256']:
        raise ValueError('Prior preparation authentication failed')
    crop=prep/'train_predicted_crops.json'
    if file_hash(crop)!=report['metadata']['crop_cache_sha256']:raise ValueError('Crop cache changed')
    args.output.mkdir(parents=True,exist_ok=False)
    model,initialization=build_prior_model(raw,prior_path=candidate['path'],prior_sha256=candidate['sha256'])
    model=model.to('cuda:0')
    observer_before=backbone_state_sha256(model.observer)
    ds=TopDownFaceLandmarkDataset(annotation_path=raw['data']['annotations']['train'],image_root=raw['data']['root'],
        input_size=256,training=True,augmentation=CropAugmentation(**raw['data']['augmentation']),
        mean=raw['data']['mean'],std=raw['data']['std'],category_ids=(2,),
        detector_crop_cache=crop,geometry_prior_grid_size=(16,16),seed=42)
    clean_ids={r['record_id'] for r in json.loads((ROOT/plan['clean_inventory_integration_20261012']['artifact']).read_text())['records']}
    pairs=[]
    for i,sample in enumerate(ds.samples):
        if str(sample['annotation']['id']) not in clean_ids:continue
        pairs.append(ds[i])
        if len(pairs)==16:break
    if len(pairs)!=16:raise ValueError('Insufficient real-data smoke samples')
    images=torch.stack([x for x,_ in pairs]);targets=[t for _,t in pairs]
    clean=torch.stack([t['clean_image'] for t in targets])
    geometry=torch.stack([t['geometry_prior']['geometry'] for t in targets])
    support=torch.stack([t['geometry_prior']['support'] for t in targets])
    d=ConditionalFeatureDiscriminator(384).to('cuda:0')
    opt=generator_optimizer(model,raw['train']);dopt=torch.optim.AdamW(d.parameters(),lr=1e-4)
    runtime=PriorUpdateRuntime(model,LandmarkLoss(raw['loss']).to('cuda:0'),opt,
        contract=dict(scope='integration_probe_not_training',prior_sha256=candidate['sha256']),
        discriminator=d,discriminator_optimizer=dopt,adversarial_weight=.01,amp=True)
    forward=model.forward
    def injected_first_forward_oom(images,**kwargs):
        if len(images)>2:raise torch.OutOfMemoryError('Injected before first scaler.scale')
        return forward(images,**kwargs)
    model.forward=injected_first_forward_oom
    try:oom_replay=runtime.update_window(images,clean,geometry,support,targets,microbatch=4)
    finally:model.forward=forward
    if oom_replay['microbatch']!=2 or oom_replay['effective_batch']!=16:raise ValueError('AMP OOM replay failed')
    attempts=[]
    for microbatch in [4,8,16]:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        results=[runtime.update_window(images,clean,geometry,support,targets,microbatch=microbatch) for _ in range(4)]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        sample=gpu();peak=torch.cuda.max_memory_reserved()/1024**2
        if sample['temperature']>=83 or peak>sample['total_mib']*.9:raise RuntimeError('Unsafe integration probe')
        if any(r['status']!='updated' for r in results):raise RuntimeError('AMP overflow in integration probe')
        attempts.append(dict(requested_microbatch=microbatch,updates=results,elapsed_seconds=elapsed,peak_reserved_mib=peak,gpu=sample))
        print(json.dumps(attempts[-1]),flush=True)
    # Exercise actual disk serialization of both optimizer states, AMP and EMA,
    # then replay the next update from that exact state. No checkpoint retained.
    with tempfile.TemporaryDirectory(prefix='prior_runtime_',dir=args.output) as temporary:
        checkpoint=Path(temporary)/'resume.pt';torch.save(runtime.checkpoint(epoch=1),checkpoint)
        expected=runtime.update_window(images,clean,geometry,support,targets,microbatch=4)
        expected_model={k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()}
        expected_d={k:v.detach().cpu().clone() for k,v in d.state_dict().items()}
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        runtime.restore(saved)
        compare_state(saved,runtime.checkpoint(epoch=1))
        actual=runtime.update_window(images,clean,geometry,support,targets,microbatch=4)
        replay_difference=dict(generator_max_abs=update_difference(expected_model,model.state_dict()),discriminator_max_abs=update_difference(expected_d,d.state_dict()))
        if expected['status']!='updated' or actual['status']!='updated':raise ValueError('Post-resume update did not commit')
        if actual['step']!=expected['step']:raise ValueError('Resume step mismatch')
    if backbone_state_sha256(model.observer)!=observer_before:raise ValueError('Frozen observer changed')
    if backbone_state_sha256(runtime.ema.module.observer)!=observer_before:raise ValueError('EMA observer changed')
    with torch.inference_mode():
        runtime.ema.module.eval()
        out=runtime.ema.module(images[:4].cuda(),geometry=geometry[:4].cuda(),support=support[:4].cuda())
        loss=runtime.criterion(out,targets[:4])['loss_total']
        if not torch.isfinite(loss):raise ValueError('Non-finite inference/loss after resume')
    result=dict(status='integration_checked_not_gate',production_training='not run',formal_four_gates='not run',
        validation='Full conditional validation not run; train-sample inference/loss diagnostic only',
        prior_sha256=candidate['sha256'],crop_cache_sha256=file_hash(crop),initialization=initialization,
        sample_annotation_ids=[t['annotation_id'] for t in targets],attempts=attempts,checkpoint_state_roundtrip='bitwise_exact_including_G_D_EMA_optimizers_AMP_RNG',
        next_update='finite_and_same_step; CUDA grid_sample prevents bitwise deterministic continuation',next_update_difference=replay_difference,
        injected_first_forward_amp_oom=oom_replay,
        frozen_observer_and_ema='bitwise_unchanged',post_resume_loss=float(loss),deterministic_algorithms='warn_only_grid_sample_backward_unsupported',
        remaining=['Epoch trainer, conditional validation, W&B and retention integration','Independent authenticated schema/gpu_step/overfit/small_scale','Production preflight uses effective64 and full validation, unlike this effective16 integration diagnostic'])
    (args.output/'probe_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Runtime integration probe completed',flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
