#!/usr/bin/env python3
"""Isolated effective64 joint-path 4/8/16 probes; no formal gate marker."""
from argparse import ArgumentParser
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,Subset
from scripts.train_messenger_geometry_adversarial import datasets
from scripts.messenger_geometry_adversarial_contract import authenticate, AdversarialRuntime as PriorUpdateRuntime
from scripts.run_messenger_geometry_prior_production import ensure_descriptor_limit
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import write_json
from keypoint_comparison1.landmark_detection.geometry_prior_training import build_prior_model,generator_optimizer,file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_epochs import PriorEpochDriver
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.training import seed_everything,_scheduler


def candidate(config,output,batch):
    ensure_descriptor_limit()
    spec,identity=authenticate(config);raw=spec['parent_config'];cfg=raw['train']
    torch.set_num_threads(1);seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32=cfg['tf32'];torch.backends.cudnn.allow_tf32=cfg['tf32']
    observations=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                total,used,temp=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,temperature.gpu','--format=csv,noheader,nounits','--id=0'],text=True).strip().split(','))
                observations.append(dict(total_mib=total,used_mib=used,temperature=temp))
            except Exception as error:observations.append(dict(error=str(error)))
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        train,val=datasets(spec);gen=torch.Generator().manual_seed(42)
        loader=DataLoader(Subset(train,range(128)),batch_size=64,shuffle=True,num_workers=cfg['num_workers'],persistent_workers=False,collate_fn=collate_landmark_batch,generator=gen)
        prior=spec['prior'];model,_=build_prior_model(raw,prior_path=ROOT/prior['path'],prior_sha256=prior['sha256'],fusion_mode='messenger' if spec['arm']=='control' else 'prior')
        model=model.cuda();opt=generator_optimizer(model,cfg)
        runtime=PriorUpdateRuntime(model,LandmarkLoss(raw['loss']).cuda(),opt,contract=dict(identity=identity,scope='isolated_joint_preflight'),amp=cfg['amp'],max_grad_norm=cfg['max_grad_norm'],ema_decay=cfg['ema']['decay'])
        driver=PriorEpochDriver(runtime,schedulers={name:_scheduler(optimizer,total_steps=4,warmup_steps=2,minimum_ratio=cfg['minimum_lr_ratio']) for name,optimizer in [('g',opt),('d',runtime.d_optimizer)]},loader_generator=gen,effective_batch=64,microbatch=batch,freeze_backbone_epochs=0,mixing_warmup_epochs=0,minimum_epoch=1,maximum_epochs=2)
        torch.cuda.reset_peak_memory_stats();rows=[]
        for _ in range(2):
            rows.append(driver.run_epoch(loader,lambda m:validate_prior(m,val,device=torch.device('cuda:0'),batch_size=batch)))
        if runtime.step!=4 or driver.microbatch!=batch or any(r['amp_overflows'] for r in rows):raise RuntimeError('Probe needs four finite committed updates at requested microbatch without OOM fallback')
        if not runtime.d_optimizer.state:raise RuntimeError('Adversarial preflight must exercise discriminator updates')
        stop.set();thread.join()
        if not observations or any('error' in o for o in observations):raise RuntimeError('Missing GPU safety measurements')
        peak=torch.cuda.max_memory_reserved()/1024**2
        if any(o['temperature']>=83 or o['used_mib']>.90*o['total_mib'] for o in observations) or peak>.90*observations[0]['total_mib']:raise RuntimeError('GPU safety limit exceeded')
        measured=rows[-1]
        result=dict(status='safe',microbatch=batch,identity=identity,rows=rows,peak_reserved_mib=peak,maximum_temperature=max(o['temperature'] for o in observations),maximum_device_used_mib=max(o['used_mib'] for o in observations),effective_batch=64,full_train_instances=len(train),full_validation_instances=len(val),projected_seconds=(measured['train_seconds']/128*len(train)+measured['validation_seconds'])*cfg['epochs'],warmup_epochs_excluded=1,overrides=dict(freeze_backbone_epochs=0,mixing_warmup_epochs=0,epochs=2),preflight_source_sha256=file_hash(__file__))
        result['discriminator_updates_exercised']=bool(runtime.d_optimizer.state)
        write_json(output,result)
    finally:
        stop.set();thread.join()


def run(config,output):
    spec,identity=authenticate(config);output.mkdir(parents=True,exist_ok=False);attempts=[]
    for batch in [4,8,16]:
        path=output/f'batch_{batch}.json'
        with (output/f'batch_{batch}.log').open('w') as log:
            result=subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--config',str(config.resolve()),'--output',str(path.resolve()),'--candidate',str(batch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        attempt=json.loads(path.read_text()) if result.returncode==0 and path.exists() else dict(status='unsafe',microbatch=batch,returncode=result.returncode,log=str(output/f'batch_{batch}.log'))
        attempts.append(attempt);write_json(output/'attempts.json',dict(attempts=attempts))
    safe=[a for a in attempts if a['status']=='safe']
    if not safe:raise RuntimeError('No safe candidate; inspect logs before any retry')
    best=min(a['projected_seconds'] for a in safe)
    chosen=min((a for a in safe if a['projected_seconds']<=best*1.03),key=lambda a:a['microbatch'])
    write_json(output/'preflight_complete.json',dict(status='preflight_completed_not_formal_gate',identity=identity,candidates=[4,8,16],effective_batch=64,selected_microbatch=chosen['microbatch'],attempts=attempts,selection_rule='fastest_projected_time_within3percent_smaller',production_training='not run',preflight_source_sha256=file_hash(__file__)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=int,choices=[4,8,16]);a=p.parse_args()
    candidate(a.config,a.output,a.candidate) if a.candidate else run(a.config,a.output)
