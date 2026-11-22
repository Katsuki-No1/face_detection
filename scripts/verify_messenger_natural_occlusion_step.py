#!/usr/bin/env python3
"""One real-data CPU G/D/T update and invariant audit, never a production gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import write_json
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank
from scripts.messenger_natural_occlusion_common import authenticate,datasets
from scripts.evaluate_messenger_geometry_prior import load_model
from src.data.messenger_natural_occlusion import NaturalOcclusionDataset,collate_natural
from src.models.messenger_occlusion_augmentation import BoundedOccluderRGB,NaturalOcclusionDiscriminator
from src.models.messenger_natural_task import MessengerTask,NaturalUpdate


def run(args):
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg,spec,parent_identity,checkpoint,identity=authenticate(args.config)
    args.output.mkdir(parents=True,exist_ok=False)
    parent,_=load_model(spec,parent_identity,checkpoint,torch.device('cpu'))
    data=spec['parent_config']['data'];train,_=datasets(spec)
    bank=RealOcclusionBank(ROOT/cfg['material_bank']['path'],cfg['material_bank']['sha256'],
        annotation_path=data['annotations']['train'],image_root=data['root'])
    natural=NaturalOcclusionDataset(train,bank)
    rows=[natural[i] for i in sorted(natural.requests)[:8]];batch=collate_natural(rows)
    task=MessengerTask(parent,LandmarkLoss(spec['parent_config']['loss']),mean=data['mean'],std=data['std'])
    g=BoundedOccluderRGB();d=NaturalOcclusionDiscriminator()
    opts={k:torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
          for k,model in [('g',g),('d',d),('t',task)]}
    runtime=NaturalUpdate(g,d,task,optimizers=opts)
    parent.base.backbone.requires_grad_(False)
    frozen={name:t.detach().clone() for name,t in parent.state_dict().items()
            if isinstance(t,torch.Tensor) and name.startswith(('observer.','bank.'))}
    result=runtime.update(batch,microbatch=4)
    assert all(torch.equal(t,parent.state_dict()[name]) for name,t in frozen.items())
    assert all(result['gradient_norms'][k]>0 for k in ('g','d','t'))
    with torch.no_grad():
        output=g(batch['clean'],batch['donor'],batch['mask'])['image']
        hidden=batch['clean'].clone();hidden[batch['mask'].expand_as(hidden)]=.123
        alternate=g(hidden,batch['donor'],batch['mask'])['image']
        outside=~batch['mask'].expand_as(output)
        assert torch.equal(output,alternate)
        assert torch.equal(output[outside],batch['clean'][outside])
    write_json(args.output/'verification.json',dict(status='passed_implementation_check_not_gate',
        production_training='not run',test_used=False,identity=identity,update=result,
        examples=[r['metadata'] for r in rows],frozen_observer_and_O_unchanged=True,
        hidden_recipient_invariant=True,outside_mask_exactly_preserved=True,
        source_sha256=file_hash(__file__)))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training_plans/messenger_natural_occlusion_v1.json')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
