"""Deterministic full-data sampling; natural point strata and added-area strata remain separate."""
import copy, hashlib, json, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset,CropAugmentation
from src.data.landmark_occlusion import apply_landmark_occlusion_numpy,ellipse_region_mask,transform_ellipse_affine
from scripts.messenger_followup_shared import ROOT,read,sha
from src.data.messenger_full_e_cache import ANNOTATIONS,DATA,build_crop_transform,ellipse_aligned_rectangle
from scripts.evaluate_messenger_geometry_prior import warp_crop
from PIL import Image

def make_base(split,training,config):
    aug=copy.deepcopy(config['data']['augmentation']);aug['occlusion_probability']=0;aug['landmark_guided_occlusion']=None
    aug.pop('paired_occlusion',None)
    allowed=CropAugmentation.__dataclass_fields__
    ds=TopDownFaceLandmarkDataset(annotation_path=ANNOTATIONS[split],image_root=DATA,input_size=256,training=training,
        augmentation=CropAugmentation(**{k:v for k,v in aug.items() if k in allowed}),mean=config['data']['mean'],std=config['data']['std'],
        category_ids=[2],require_visible_keypoint=False,seed=42)
    return ds

def proxy(states):return min(4,max(0,int(np.sum(np.asarray(states)!=2))-1))
def groups_for(ds):
    groups=[[] for _ in range(5)];missing=[]
    for i,s in enumerate(ds.samples):
        state=np.asarray(s['annotation']['keypoints']).reshape(5,3)[:,2].astype(int)
        if (state>0).any():groups[proxy(state)].append(i)
        else:missing.append(i)
    return groups,missing

def draw_plan(ds,variant,epoch,limit=None):
    groups,missing=groups_for(ds);rng=np.random.default_rng(42+epoch);n=max(map(len,groups)) if limit is None else limit
    if limit is not None:groups=[g[:limit] for g in groups];missing=missing[:limit]
    draws=[]
    for source in (0,1):
        for b,group in enumerate(groups):
            if not group:raise ValueError('Empty natural position stratum')
            order=[]
            while len(order)<n:order.extend(rng.permutation(group).tolist())
            for k,idx in enumerate(order[:n]):
                # Synthetic bins balance independently of recipient natural severity.
                draws.append((idx,1 if source and variant!='natural_only' else 0,(k+b)%5 if source and variant!='natural_only' else -1,epoch*1000000+source*100000+b*n+k))
        for k,idx in enumerate(rng.permutation(missing)):draws.append((int(idx),0,-1,epoch*1000000+700000+source*len(missing)+k))
    return [draws[i] for i in rng.permutation(len(draws))]

class FollowupDataset(Dataset):
    def __init__(self,base,variant,draws,training):
        self.base,self.variant,self.draws,self.training=base,variant,draws,training
        self.mean=np.array(base.mean,dtype=np.float32) if hasattr(base,'mean') else np.array([.485,.456,.406],dtype=np.float32)
        self.std=np.array(base.std,dtype=np.float32) if hasattr(base,'std') else np.array([.229,.224,.225],dtype=np.float32)
        self.videos=[str(s['image']['metadata']['source_video_id']) for s in base.samples]
    def __len__(self):return len(self.draws)
    def donor(self,recipient,rng):
        for _ in range(500):
            j=rng.randrange(len(self.base.samples))
            if self.videos[j]!=self.videos[recipient]:break
        else:raise ValueError('No cross-video donor')
        s=self.base.samples[j];a=s['annotation'];im=s['image'];path=DATA/im['file_name']
        assert sha(path)==im['metadata']['sha256']
        photo=np.asarray(Image.open(path).convert('RGB'));box,angle=ellipse_aligned_rectangle(a['ellipse'])
        crop=build_crop_transform(box,output_size=256,scale_factor=1.0,rotation_deg=angle)
        rgb=warp_crop(photo,crop);ellipse=transform_ellipse_affine(a['ellipse'],crop.matrix);alpha=ellipse_region_mask(256,256,ellipse)
        ys,xs=np.where(alpha);assert len(xs)>0
        # No expansion: exact rasterized ellipse, outside pixels always transparent.
        return rgb,alpha,j,ellipse
    def paste(self,rgb,face,recipient,rng,b):
        donor,alpha,j,de=self.donor(recipient,rng);yy,xx=np.where(face);cx,cy=float(xx.mean()),float(yy.mean())
        anchor=rng.randrange(9);cx+=(anchor%3-1)*float(xx.std())*.65;cy+=(anchor//3-1)*float(yy.std())*.65
        cx+=rng.uniform(-8,8);cy+=rng.uniform(-8,8);angle=rng.uniform(-180,180);target=(b+rng.uniform(.08,.92))/5
        lo,hi=.01,12.;best=None
        for _ in range(28):
            scale=(lo+hi)/2;m=cv2.getRotationMatrix2D((float(de[0]),float(de[1])),angle,scale);m[:,2]+=np.array([cx-de[0],cy-de[1]])
            mask=cv2.warpAffine(alpha.astype(np.uint8),m,(256,256),flags=cv2.INTER_NEAREST)>0
            fraction=float((mask&face).sum()/face.sum());candidate=(abs(fraction-target),mask,m,fraction)
            if best is None or candidate[0]<best[0]:best=candidate
            if fraction<target:lo=scale
            else:hi=scale
        _,mask,m,fraction=best
        if not b/5<=fraction<(b+1)/5:raise ValueError(f'Donor coverage outside assigned bin: {fraction} {b}')
        # Premultiply alpha to keep interpolation from drawing ellipse-exterior background.
        alpha_f=cv2.warpAffine(alpha.astype(np.float32),m,(256,256),flags=cv2.INTER_LINEAR)
        colour=cv2.warpAffine(donor.astype(np.float32)*alpha[...,None],m,(256,256),flags=cv2.INTER_LINEAR)
        colour=np.clip(colour/np.maximum(alpha_f[...,None],1e-6),0,255).astype(np.uint8)
        out=rgb.copy();out[mask]=colour[mask]
        return out,mask,{'shape':'donor_ellipse','fraction':fraction,'anchor':anchor,'donor_index':j,'donor_video':self.videos[j],'donor_crop_scale':1.0}
    def __getitem__(self,index):
        idx,added,b,seed=self.draws[index];state=random.getstate();random.seed(42+seed)
        try:im,t=self.base[idx]
        finally:random.setstate(state)
        rng=random.Random(8401+seed);rgb=np.clip(np.rint((im.numpy().transpose(1,2,0)*self.std+self.mean)*255),0,255).astype(np.uint8)
        original=np.asarray(self.base.samples[idx]['annotation']['keypoints']).reshape(5,3)[:,2].astype(int)
        states=t['keypoint_visibility'].numpy().copy();xy=t['keypoints'].numpy().copy();ellipse=transform_ellipse_affine(t['gt_ellipse'].numpy(),t['affine'].numpy())
        if t['flipped']:ellipse[0]=255-ellipse[0];ellipse[4]=-ellipse[4]
        face=ellipse_region_mask(256,256,ellipse);assert face.any()
        meta={'shape':'natural','fraction':0.,'anchor':-1};mask=np.zeros((256,256),bool)
        if added:
            if self.variant=='natural_baseline_occlusion':
                cfg={'probability':1.,'regular_probability':.5,'landmark_anchor_probability':.7,'face_overlap':[max(.001,b/5+.002),min(.999,(b+1)/5-.002)],'center_offset_fraction':.12,'maximum_attempts':128}
                for _ in range(8):
                    out,mask,st,meta=apply_landmark_occlusion_numpy(rgb,face_ellipse=ellipse,keypoints=xy*255,keypoint_visibility=states,config=cfg,rng=rng,force=True)
                    if meta.get('applied'):break
                else:raise ValueError('Unable to render baseline shape in required area bin')
                rgb=out
            elif self.variant=='natural_cross_video_face':rgb,mask,meta=self.paste(rgb,face,idx,rng,b)
            else:raise ValueError(self.variant)
            fraction=float((mask&face).sum()/face.sum());assert b/5<=fraction<(b+1)/5
            pix=np.clip(np.rint(xy*255),0,255).astype(int);covered=mask[pix[:,1],pix[:,0]];states[(states>0)&covered]=1;meta['fraction']=fraction
        valid=states>0;xy=np.where(valid[:,None],xy,0).astype(np.float32)
        tensor=torch.from_numpy(((rgb.astype(np.float32)/255-self.mean)/self.std).transpose(2,0,1).copy())
        meta.update(index=int(idx),source=int(added),area_bin=int(b),natural_bin=proxy(original),result_nonvisible=int((states!=2).sum()),annotation_id=int(t['annotation_id']),video=str(t['source_video_id']),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),all_missing=bool(not (original>0).any()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        # Crop changes can hide points; the positional mask follows actual valid coordinates.
        target={'keypoints':torch.from_numpy(xy),'labeled':torch.from_numpy(valid),'visible':torch.from_numpy(states==2)}
        norm=float(t['crop_side'])/(2*max(float(t['gt_ellipse'][2]),float(t['gt_ellipse'][3])))
        return tensor,target,states,meta,rgb,norm

def collate(rows):
    return torch.stack([r[0] for r in rows]),[r[1] for r in rows],np.stack([r[2] for r in rows]),[r[3] for r in rows],np.stack([r[4] for r in rows]),np.array([r[5] for r in rows],dtype=np.float32)
