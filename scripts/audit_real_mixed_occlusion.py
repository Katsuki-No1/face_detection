#!/usr/bin/env python3
"""Audit actual Messenger loader pairs; CPU only, no model accuracy claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.training import make_dataset
from keypoint_comparison1.landmark_detection.real_occlusion import digest,patch_avoids_faces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'keypoint_comparison1/configs/continuous_recovery_real_mixed.yaml')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--epoch',type=int,default=1)
    p.add_argument('--geometry-replay',action='store_true',help='Recover deterministic placement statistics; render the first 8 gradients only. Not a full RGB audit.')
    args=p.parse_args();torch.set_num_threads(1);cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.geometry_replay:
        import keypoint_comparison1.landmark_detection.real_occlusion as synthesis
        original=synthesis.opaque_boundary_gradient
        rendered=0
        def replay_gradient(target,pasted,mask,**kwargs):
            nonlocal rendered
            rendered+=1
            return original(target,pasted,mask,**kwargs) if rendered<=8 else pasted.copy()
        synthesis.opaque_boundary_gradient=replay_gradient
    raw=load_landmark_config(args.config).raw;rows=[];panels=[];eligible_count=0
    for seed in args.seeds:
        raw['train']['seed']=seed;ds=make_dataset(raw,split='train',training=True);ds.set_epoch(args.epoch)
        assert ds.real_occlusion_bank is not None
        eligible_count=int(sum(ds.real_occlusion_bank.eligible_sample(s) for s in ds.samples))
        for i in range(len(ds)):
            if ds._paired_occlusion_rng(i) is None:continue
            crop_seed=int.from_bytes(__import__('hashlib').sha256(f'{seed}:{args.epoch}:{i}:mixed-audit'.encode()).digest()[:4],'big')
            random.seed(crop_seed);np.random.seed(crop_seed);torch.manual_seed(crop_seed)
            rgb,t=ds[i];m=t['paired_occlusion'];row=dict(seed=seed,epoch=args.epoch,index=i,crop_seed=crop_seed,**m)
            assert ds.real_occlusion_bank.eligible_sample(ds.samples[i])
            assert torch.isfinite(rgb).all() and torch.isfinite(t['clean_image']).all()
            mask=t['occlusion_mask'];assert torch.equal(rgb[:,~mask],t['clean_image'][:,~mask])
            assert not t['synthetic_truncated'].any()
            if m['applied']:
                assert m['donor_video']!=m['recipient_video']
                assert 0<m['face_overlap']<=.4
                if m['requested_family']=='non_face':
                    assert patch_avoids_faces(m['donor_source_box'],ds.real_occlusion_bank.images[m['donor_file']]['face_exclusion_boxes'])
                if len(panels)<8:
                    panel=Image.new('RGB',(512,282),'white')
                    for j,im in enumerate([t['clean_image'],rgb]):
                        im=im.numpy().transpose(1,2,0)*np.array(ds.std)+np.array(ds.mean)
                        panel.paste(Image.fromarray(np.rint(np.clip(im,0,1)*255).astype(np.uint8)),(j*256,24))
                    ImageDraw.Draw(panel).text((4,4),f"{m['requested_family']} {m['requested_route']} {m['face_overlap']:.1%} / clean | student",fill='black')
                    panels.append(panel)
            rows.append(row)
            with (args.output/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if len(rows)%100==0:print(f'pairs {len(rows)}',flush=True)
    applied=[r for r in rows if r['applied']];family=Counter(r['requested_family'] for r in applied)
    routes=Counter(r['requested_route'] for r in applied)
    checks=dict(nonempty=bool(applied),face_applied_fraction_40_to_60=.4<=family['face']/max(1,len(applied))<=.6,
        skip_at_most_5_percent=(len(rows)-len(applied))/max(1,len(rows))<=.05,
        light_at_most_30_percent=routes['light']/max(1,len(applied))<=.30,
        strong_at_least_25_percent=routes['strong']/max(1,len(applied))>=.25)
    groups={}
    for seed in args.seeds:
        rr=[r for r in rows if r['seed']==seed];aa=[r for r in rr if r['applied']]
        groups[str(seed)]=dict(requested=len(rr),applied=len(aa),families=dict(Counter(r['requested_family'] for r in aa)))
    report=dict(status='passed' if all(checks.values()) else 'failed',audit_only=True,production_gate_success=False,
        scope='geometry_replay_first_8_gradients_only' if args.geometry_replay else 'full_rgb_loader_audit',
        training='not run',eligible_source_records=eligible_count,requested=len(rows),applied=len(applied),
        requested_families=dict(Counter(r['requested_family'] for r in rows)),applied_families=dict(family),
        routes=dict(routes),skip_reasons=dict(Counter(r['reason'] for r in rows if not r['applied'])),
        by_seed=groups,checks=checks,epoch=args.epoch,seeds=args.seeds,
        config_sha256=digest(args.config),manifest_sha256=raw['data']['augmentation']['paired_occlusion']['material_manifest_sha256'],
        limitations=['Actual loader and jitter/photometric/flip; deterministic audit RNG differs from production worker ordering.',
                    'No trained-model recovery accuracy evaluation. No validation/test augmentation or training.',
                    'Non-face exclusion covers annotated faces; unannotated faces remain a data-quality limitation.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if panels:
        montage=Image.new('RGB',(1024,282*((len(panels)+1)//2)),'white')
        for i,pa in enumerate(panels):montage.paste(pa,((i%2)*512,(i//2)*282))
        montage.save(args.output/'preview.png')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
