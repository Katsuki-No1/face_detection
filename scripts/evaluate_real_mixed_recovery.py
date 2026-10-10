#!/usr/bin/env python3
"""Paired validation diagnostic for trained recovery-on/off mixed-occlusion runs.

No training, gate success or checkpoint selection. Val recipients, train donors;
GT-crop feature diagnostics are separate from deployed keypoint evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import make_dataset,load_landmark_checkpoint
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank,apply_real_mixed,is_clean_annotation,identity,digest
from keypoint_comparison1.landmark_detection.real_recovery_metrics import RealRecoveryMeter
from src.data.landmark_occlusion import transform_ellipse_affine


class ValidationPairs:
    def __init__(self,raw,bank):
        raw=copy.deepcopy(raw);ev=raw['data']['evaluation_sets']['v2_val']
        raw['data']['annotations']['val']=ev['annotation']
        raw['data'].setdefault('image_roots',{})['val']=ev['root']
        raw['data'].pop('validation_detector_crop_cache',None)
        self.source=make_dataset(raw,split='val',training=False)
        canonical=ROOT/'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/val.json'
        if digest(canonical)!='0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0':
            raise ValueError('Canonical validation identity changed')
        data=json.loads(canonical.read_text());images={x['id']:x for x in data['images']}
        allowed={identity(images[a['image_id']]['file_name'],a['ellipse']) for a in data['annotations']
                 if a.get('attribute_valid') is True and a.get('source_face_index') is not None}
        self.indices=[i for i,s in enumerate(self.source.samples) if is_clean_annotation(s['annotation'])
                      and identity(s['image']['file_name'],s['annotation']['ellipse']) in allowed]
        val_videos={x['source_video_id'] for x in images.values()}
        if val_videos & set(bank.videos):raise ValueError('Train/val donor video leakage')
        self.bank=bank

    def __len__(self):return 2*len(self.indices)

    def __getitem__(self,index):
        i=self.indices[index//2];family=['face','non_face'][index%2]
        sample=self.source.samples[i];clean,target=self.source[i]
        pixels=np.rint(np.clip(clean.numpy().transpose(1,2,0)*np.array(self.source.std)+np.array(self.source.mean),0,1)*255).astype(np.uint8)
        lab=cv2.cvtColor(pixels.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        recipient=dict(video=target['source_video_id'],descriptor=np.r_[lab.mean((0,1)),lab[:,:,0].std()].tolist())
        # The proxy only supplies validation recipient context. donor() remains
        # bound to the original train-only bank; no val images enter its pool.
        proxy=SimpleNamespace(images={sample['image']['file_name']:recipient},
            eligible_sample=lambda s:is_clean_annotation(s['annotation']),
            manifest_sha256=self.bank.manifest_sha256,donor=self.bank.donor)
        seed=int.from_bytes(hashlib.sha256(f"42:{target['annotation_id']}:{family}:real_feature_val".encode()).digest()[:8],'big')
        out,mask,_,meta=apply_real_mixed(pixels,sample=sample,
            face_ellipse=transform_ellipse_affine(target['gt_ellipse'].numpy(),target['affine'].numpy()),
            keypoints=target['keypoints'].numpy()*(self.source.input_size-1),
            keypoint_visibility=target['keypoint_visibility'].numpy(),bank=proxy,
            rng=random.Random(seed),force_family=family)
        normalized=(out.astype(np.float32)/255-np.array(self.source.mean))/np.array(self.source.std)
        mask_tensor=torch.from_numpy(mask)
        occluded=torch.from_numpy(normalized.transpose(2,0,1).copy()).float()
        # Preserve the original normalized recipient exactly outside the paste.
        occluded=torch.where(mask_tensor[None],occluded,clean)
        return clean,occluded,mask_tensor,dict(
            paired_occlusion=meta,annotation_id=target['annotation_id'])


@torch.inference_mode()
def evaluate(model,teacher,pairs,scale,device):
    fixed=RealRecoveryMeter();student=RealRecoveryMeter();signature=hashlib.sha256();counts={}
    for i in range(len(pairs)):
        clean,occ,mask,target=pairs[i]
        for x in [clean,occ,mask]:signature.update(x.numpy().tobytes())
        signature.update(str(target['annotation_id']).encode())
        meta=target['paired_occlusion'];key=f"{meta['requested_family']}/{'applied' if meta['applied'] else 'skipped'}"
        counts[key]=counts.get(key,0)+1
        features=model.backbone(occ[None].to(device))
        if model.messenger_recovery is None:
            tokens=features.flatten(2).transpose(1,2)
            diag=dict(original_tokens=tokens,candidate_tokens=tokens,recovered_tokens=tokens)
        else:_,diag=model.messenger_recovery(features)
        teacher_clean=teacher(clean[None].to(device));student_clean=model.backbone(clean[None].to(device))
        fixed.update(diag,teacher_clean,mask[None].to(device),[target],channel_scale=scale)
        student.update(diag,student_clean,mask[None].to(device),[target],channel_scale=scale)
    return dict(fixed_teacher=fixed.report(),student_clean_reference=student.report(),
                counts=counts,paired_inputs_sha256=signature.hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--on-checkpoint',type=Path,required=True)
    p.add_argument('--off-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    configs=[load_landmark_config(ROOT/'keypoint_comparison1/configs'/name) for name in
             ['continuous_recovery_real_mixed.yaml','continuous_recovery_real_mixed_control.yaml']]
    on=configs[0].raw;pc=on['data']['augmentation']['paired_occlusion']
    bank=RealOcclusionBank(pc['material_manifest'],pc['material_manifest_sha256'],
                         annotation_path=on['data']['annotations']['train'],image_root=on['data']['root'])
    pairs=ValidationPairs(on,bank);device=torch.device(args.device)
    teacher_raw=copy.deepcopy(on);teacher_raw['model']['continuous_recovery']['enabled']=False
    teacher_model,_=build_landmark_model(teacher_raw,variant='query_refinement',initialize_from_detector=False)
    init=on['train']['finetune_checkpoint']
    if digest(init['path'])!=init['sha256']:raise ValueError('Fixed teacher checkpoint hash mismatch')
    load_landmark_checkpoint(teacher_model,init['path'],prefer_ema=True,strict=True)
    teacher=teacher_model.backbone.to(device).eval();del teacher_model
    results={};scale=None
    for name,cfg,checkpoint in zip(['recovery_on','recovery_off'],configs,[args.on_checkpoint,args.off_checkpoint]):
        model,_=build_landmark_model(cfg.raw,variant='query_refinement',initialize_from_detector=False)
        load_landmark_checkpoint(model,checkpoint,prefer_ema=True,strict=True);model.to(device).eval()
        if name=='recovery_on':
            if not bool(model.messenger_recovery.scale_ready):raise ValueError('Trained teacher scale is missing')
            scale=model.messenger_recovery.channel_scale.detach().clone()
        results[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                           **evaluate(model,teacher,pairs,scale,device))
        del model
    if results['recovery_on']['paired_inputs_sha256']!=results['recovery_off']['paired_inputs_sha256']:
        raise ValueError('On/off synthetic inputs differ')
    report=dict(status='diagnostic_complete_not_adoption',split='val',donor_split='train',
        scope='GT-crop paired feature diagnostic, not deployment metrics',eligible_validation_faces=len(pairs.indices),
        references='Fixed initial teacher and each student clean backbone; compare final output, not candidate alone',
        teacher_sha256=init['sha256'],manifest_sha256=bank.manifest_sha256,results=results,
        accuracy_claim=False,script_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
