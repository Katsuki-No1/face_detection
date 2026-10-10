#!/usr/bin/env python3
"""Prepare a train-only, hash-locked donor inventory, never checkpoint/gate success."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from keypoint_comparison1.landmark_detection.real_occlusion import digest,identity,is_clean_annotation
from scripts.prototype_cross_video_face_occlusion import context_descriptor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cv2.setNumThreads(1)
    canonical=ROOT/'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/train.json'
    adapter=ROOT/'outputs/dataset_adapters/face_pose_drive_v2_balanced_811/face_keypoints_train.json'
    image_root=ROOT/'outputs/dataset_cache/immutable_f493be94f9c30e54/face_pose_drive_v2_balanced_811'
    c=json.loads(canonical.read_text());a=json.loads(adapter.read_text())
    assert digest(canonical)=='b8a81c114c26a9476907ecb4611ee450e49a5f21a5683021841c8ddcb4a32ecb'
    assert len(c['images'])==4237 and len({i['source_video_id'] for i in c['images']})==44
    ai={i['id']:i for i in a['images']};by_file=defaultdict(list);heads=defaultdict(list)
    lookup=defaultdict(list)
    for ann in a['annotations']:
        im=ai[ann['image_id']];assert im['metadata']['split']=='train'
        by_file[im['file_name']].append(ann);lookup[identity(im['file_name'],ann['ellipse'])].append(ann)
    ci={i['id']:i for i in c['images']}
    for ann in c['annotations']:heads[ci[ann['image_id']]['file_name']].append(ann)
    records=[];faces=[];eligible=[]
    for n,im in enumerate(c['images']):
        file=im['file_name'];path=image_root/file;rgb=np.array(Image.open(path).convert('RGB'));h,w=rgb.shape[:2]
        boxes=[]
        # All adapter Faces, including missing/occluded/unassociated ones, are excluded.
        for ann in by_file[file]:
            cx,cy,rx,ry,t=ann['ellipse']
            if not np.isfinite([cx,cy,rx,ry,t]).all() or min(rx,ry)<=0:
                raise ValueError(f'Invalid source face geometry: {file}')
            ex=np.hypot(rx*np.cos(t),ry*np.sin(t));ey=np.hypot(rx*np.sin(t),ry*np.cos(t))
            margin=.05*2*max(rx,ry)
            boxes.append([max(0,cx-ex-margin),max(0,cy-ey-margin),min(w,cx+ex+margin),min(h,cy+ey+margin)])
        lab=cv2.cvtColor(cv2.resize(rgb,(64,36)).astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        descriptor=np.r_[lab.mean((0,1)),lab[:,:,0].std()].astype(float).tolist()
        records.append(dict(file=file,video=im['source_video_id'],height=h,width=w,sha256=digest(path),
                            descriptor=descriptor,face_exclusion_boxes=boxes))
        for ann in heads[file]:
            if ann.get('attribute_valid') is not True or ann.get('source_face_index') is None:continue
            key=identity(file,ann['ellipse']);matches=lookup[key]
            if len(matches)!=1:continue
            src=matches[0]
            if is_clean_annotation(src):eligible.append(key)
            if src.get('attributes',{}).get('face_occluded') is not False:continue
            x,y,bw,bh=ann['bbox'];box=[max(0,round(x-.05*bw)),max(0,round(y-.05*bh)),min(w,round(x+1.05*bw)),min(h,round(y+1.05*bh))]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            desc=context_descriptor(rgb,ann['bbox'],[an['bbox'] for an in heads[file]]) or descriptor
            faces.append(dict(id=src['id'],file=file,box=box,ellipse=ann['ellipse'],descriptor=desc))
        if n%600==0:print(f'images {n}/4237',flush=True)
    out=dict(schema='real-occlusion-bank-v1',split='train',canonical_path=str(canonical),
        canonical_sha256=digest(canonical),adapter_path=str(adapter),adapter_sha256=digest(adapter),
        image_root=str(image_root),images=records,faces=faces,eligible_recipients=eligible,
        code_sha256=digest(__file__),face_exclusion_margin_major_diameter_fraction=.05,
        limitations=['Excludes every annotated Face; unannotated faces cannot be guaranteed absent.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(out,f,separators=(',',':'))
    print(json.dumps(dict(path=str(args.output),sha256=digest(args.output),images=len(records),faces=len(faces),eligible=len(eligible))),flush=True)


if __name__=='__main__':main()
