"""Opt-in train-only face/non-face paste pairs for Messenger recovery.

The family is sampled once at 50/50; retries never switch family or severity.
Opaque crop-side RGB correction uses the adopted width48 / strength2 policy.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from src.data.landmark_occlusion import ellipse_region_mask
from scripts.opaque_boundary_gradient import opaque_boundary_gradient


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(file_name, ellipse):
    return json.dumps([file_name,[round(float(x),4) for x in ellipse]],separators=(',',':'))


def is_clean_annotation(annotation):
    points=np.asarray(annotation.get('keypoints',[]),dtype=float)
    return (annotation.get('attributes',{}).get('face_occluded') is False
            and points.shape==(15,) and np.isfinite(points).all()
            and np.all(points.reshape(5,3)[:,2]==2))


def patch_avoids_faces(box, face_boxes):
    x0,y0,x1,y1=box
    return all(x1<=a or x0>=c or y1<=b or y0>=d for a,b,c,d in face_boxes)


class RealOcclusionBank:
    def __init__(self,path,expected_sha256,*,annotation_path,image_root):
        if digest(path)!=expected_sha256:
            raise ValueError('Real occlusion manifest hash mismatch')
        self.data=json.loads(Path(path).read_text())
        if self.data.get('schema')!='real-occlusion-bank-v1' or self.data.get('split')!='train':
            raise ValueError('Real occlusion requires a train-only v1 bank')
        if digest(annotation_path)!=self.data['adapter_sha256']:
            raise ValueError('Recipient annotation does not match train-only donor bank')
        if digest(self.data['canonical_path'])!=self.data['canonical_sha256']:
            raise ValueError('Canonical train annotations changed')
        self.root=Path(image_root).resolve()
        if self.root!=Path(self.data['image_root']).resolve():
            raise ValueError('Recipient and donor image roots differ')
        self.images={r['file']:r for r in self.data['images']}
        self.eligible=set(self.data['eligible_recipients'])
        self.faces=self.data['faces']
        self.videos={}
        for row in self.images.values():
            self.videos.setdefault(row['video'],[]).append(row)
        if len(self.videos)<2 or not self.faces:
            raise ValueError('Need at least two videos and face donors')
        for file in self.images:
            if not (self.root/file).resolve().is_relative_to(self.root):
                raise ValueError('Donor path escapes immutable image root')
        self.cache=OrderedDict()
        self.manifest_sha256=expected_sha256

    def eligible_sample(self,sample):
        return (is_clean_annotation(sample['annotation']) and identity(
            sample['image']['file_name'],sample['annotation']['ellipse']) in self.eligible)

    def read(self,row):
        name=row['file']
        if name not in self.cache:
            path=self.root/name
            if digest(path)!=row['sha256']:
                raise ValueError('Donor image hash mismatch')
            self.cache[name]=np.array(Image.open(path).convert('RGB'))
            if len(self.cache)>16:self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]

    def donor(self,family,recipient,rng):
        if family=='face':
            # One closest-context candidate per video, then random among top 5.
            candidates={};target=np.asarray(recipient['descriptor'])
            for face in self.faces:
                row=self.images[face['file']]
                if row['video']==recipient['video']:continue
                distance=float(np.linalg.norm(np.asarray(face['descriptor'])-target))
                if row['video'] not in candidates or distance<candidates[row['video']][0]:
                    candidates[row['video']]=(distance,face)
            nearest=sorted(candidates.values(),key=lambda x:x[0])[:5]
            _,face=rng.choice(nearest);row=self.images[face['file']]
            x0,y0,x1,y1=face['box'];rgb=self.read(row)[y0:y1,x0:x1].copy()
            ell=np.array(face['ellipse'],float);ell[:2]-=[x0,y0]
            foreground=ellipse_region_mask(*rgb.shape[:2],ell)
            return rgb,foreground,dict(donor_file=row['file'],donor_video=row['video'],
                donor_source_box=face['box'],donor_annotation_id=face['id'])
        # Broad non-face source: no background/body semantic class restriction.
        video=rng.choice(sorted(v for v in self.videos if v!=recipient['video']))
        row=rng.choice(self.videos[video]);h,w=row['height'],row['width']
        for _ in range(64):
            side=min(h,w)*rng.uniform(.08,.40);aspect=np.exp(rng.uniform(np.log(.4),np.log(2.5)))
            bw=min(w,max(16,round(side*np.sqrt(aspect))));bh=min(h,max(16,round(side/np.sqrt(aspect))))
            x=rng.randrange(w-bw+1);y=rng.randrange(h-bh+1);box=[x,y,x+bw,y+bh]
            if patch_avoids_faces(box,row['face_exclusion_boxes']):
                return self.read(row)[y:y+bh,x:x+bw].copy(),None,dict(
                    donor_file=row['file'],donor_video=video,donor_source_box=box,
                    annotated_face_overlap_pixels=0)
        return None


def place_patch(image,ellipse,points,states,patch,foreground,rng,route,attempts=256):
    h,w=image.shape[:2];face=ellipse_region_mask(h,w,ellipse)
    total=int(face.sum())
    if not total:return None
    integral=cv2.integral(face.astype(np.uint8))
    xy=np.rint(points).astype(int);valid=(states>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    lo,hi={'light':(0,.1),'medium':(.1,.2),'strong':(.2,.4)}[route]
    for _ in range(attempts):
        scale=2*max(ellipse[2:4])*rng.uniform(.4,1.4)/max(patch.shape[:2])
        ph,pw=[max(2,round(n*scale)) for n in patch.shape[:2]]
        x=round(rng.uniform(ellipse[0]-pw,ellipse[0]+.6*max(ellipse[2:4])))
        y=round(rng.uniform(ellipse[1]-ph,ellipse[1]+.6*max(ellipse[2:4])))
        if route=='light' and valid.any():
            point=xy[rng.choice(np.flatnonzero(valid).tolist())]
            # Put a patch edge next to a known point, without imposing a global lower bound.
            edge=rng.randrange(4)
            if edge==0:x=int(point[0]);y=round(point[1]-rng.random()*ph)
            if edge==1:x=int(point[0])-pw+1;y=round(point[1]-rng.random()*ph)
            if edge==2:y=int(point[1]);x=round(point[0]-rng.random()*pw)
            if edge==3:y=int(point[1])-ph+1;x=round(point[0]-rng.random()*pw)
        a,b,c,d=max(0,x),max(0,y),min(w,x+pw),min(h,y+ph)
        if c<=a or d<=b:continue
        overlap=float((integral[d,c]-integral[b,c]-integral[d,a]+integral[b,a])/total)
        covered=valid&(xy[:,0]>=a)&(xy[:,0]<c)&(xy[:,1]>=b)&(xy[:,1]<d)
        if not lo<overlap<=hi or (route=='light' and not covered.any()):continue
        donor_face_overlap=None
        if foreground is not None:
            f=cv2.resize(foreground.astype(np.uint8),(pw,ph),interpolation=cv2.INTER_NEAREST)
            donor_face_overlap=float((f[b-y:d-y,a-x:c-x].astype(bool)&face[b:d,a:c]).sum()/total)
            if donor_face_overlap<.01:continue
        mask=np.zeros((h,w),bool);mask[b:d,a:c]=True
        resized=cv2.resize(patch,(pw,ph),interpolation=cv2.INTER_LINEAR)
        raw=image.copy();raw[b:d,a:c]=resized[b-y:d-y,a-x:c-x]
        # Explicit values prevent a future preview default from changing this experiment.
        out=opaque_boundary_gradient(image,raw,mask,width=48,strength=2)
        updated=states.copy();updated[covered]=1
        return out,mask,updated,dict(face_overlap=overlap,donor_face_overlap=donor_face_overlap,
            bbox_xyxy=[x,y,x+pw,y+ph],occluded_keypoint_indices=np.flatnonzero(covered).tolist(),
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return None


def apply_real_mixed(image,*,sample,face_ellipse,keypoints,keypoint_visibility,bank,rng,force_family=None):
    states=np.array(keypoint_visibility,dtype=np.int64)
    if force_family not in {None,'face','non_face'}:raise ValueError('Invalid forced family')
    family=force_family or ('face' if rng.random()<.5 else 'non_face')
    route=rng.choices(['light','medium','strong'],[.2,.4,.4])[0]
    base=dict(policy_version='real_mixed_v1',requested_family=family,requested_route=route,
              pair_kind='object',kind='object',appearance='real_mixed',
              manifest_sha256=bank.manifest_sha256,gradient_width=48,gradient_strength=2,
              alpha=False,coordinate_frame='pre_flip_crop_pixels_xy')
    def skip(reason):
        return image.copy(),np.zeros(image.shape[:2],bool),states.copy(),dict(base,applied=False,reason=reason)
    if not bank.eligible_sample(sample):return skip('recipient_not_originally_unoccluded')
    if route=='light' and not (states>0).any():return skip('no_in_crop_keypoint')
    recipient=bank.images[sample['image']['file_name']]
    for attempt in range(8):
        selected=bank.donor(family,recipient,rng)
        if selected is None:continue
        patch,foreground,meta=selected
        result=place_patch(image,face_ellipse,np.asarray(keypoints),states,patch,foreground,rng,route)
        if result is not None:
            out,mask,updated,placement=result
            return out,mask,updated,dict(base,**meta,**placement,applied=True,donor_attempts=attempt+1,
                recipient_video=recipient['video'],recipient_originally_unoccluded=True)
    return skip('family_placement_budget')
