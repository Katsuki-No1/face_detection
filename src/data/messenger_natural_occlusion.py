"""Deterministic train-only face/non-face composition for the new RGB experiment.

Uses the existing authenticated donor bank but a separate placement policy.
No old real-face acceptance contract or augmentation config is changed.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank
from src.data.landmark_occlusion import ellipse_region_mask
from scripts.opaque_boundary_gradient import opaque_boundary_gradient


BINS = ((.05,.20),(.20,.40),(.40,.60),(.60,.80))


def select_material(bank, family, recipient, rng):
    """Select without any descriptor computed from hidden recipient pixels."""
    if family=='non_face':
        # This legacy branch uses only recipient video, never its descriptor.
        return bank.donor(family,dict(video=recipient['video']),rng)
    if family!='face':raise ValueError('Unknown material family')
    by_video=defaultdict(list)
    for face in bank.faces:
        video=bank.images[face['file']]['video']
        if video!=recipient['video']:by_video[video].append(face)
    if not by_video:return None
    video=rng.choice(sorted(by_video));face=rng.choice(by_video[video]);row=bank.images[face['file']]
    x0,y0,x1,y1=face['box'];patch=bank.read(row)[y0:y1,x0:x1].copy()
    ellipse=np.asarray(face['ellipse'],dtype=float).copy();ellipse[:2]-=[x0,y0]
    foreground=ellipse_region_mask(*patch.shape[:2],ellipse)
    return patch,foreground,dict(donor_file=row['file'],donor_video=video,donor_source_box=face['box'],
        donor_annotation_id=face['id'],donor_selection='uniform different train video then face; no recipient RGB descriptor')


def balanced_requests(indices, *, seed, epoch, probability=.5, bins=BINS):
    if not 0 <= probability <= 1 or not bins: raise ValueError('Invalid composition schedule')
    rng = random.Random(f'{seed}:{epoch}:requests')
    chosen = sorted(indices); rng.shuffle(chosen)
    chosen = chosen[:round(len(chosen)*probability)]
    strata = [(family,b) for b in range(len(bins)) for family in ('face','non_face')]
    rng.shuffle(strata)
    schedule = [strata[i % len(strata)] for i in range(len(chosen))]; rng.shuffle(schedule)
    return dict(zip(chosen,schedule))


def place_material(patch, foreground, face, *, interval, rng, attempts=256):
    """Resize/truncate donor crop; audit actual binary paste/recipient overlap."""
    if face.dtype != np.bool_ or not face.any(): return None
    low,high = interval
    if not 0 < low < high <= 1: raise ValueError('Invalid severity interval')
    h,w = face.shape; area = int(face.sum())
    ys,xs = np.where(face); cy,cx = ys.mean(),xs.mean()
    for _ in range(attempts):
        desired = rng.uniform(low,high)*area
        aspect = patch.shape[1]/patch.shape[0]
        ph = max(2,round(math.sqrt(desired/aspect)*rng.uniform(1.,2.)))
        pw = max(2,round(ph*aspect))
        if ph > 4*h or pw > 4*w: continue
        x = round(cx+rng.uniform(-.65,.65)*(xs.max()-xs.min()+1)-pw/2)
        y = round(cy+rng.uniform(-.65,.65)*(ys.max()-ys.min()+1)-ph/2)
        a,b,c,d = max(0,x),max(0,y),min(w,x+pw),min(h,y+ph)
        if c <= a or d <= b: continue
        mask = np.zeros((h,w),bool); mask[b:d,a:c] = True
        fraction = int((mask & face).sum())/area
        if not low <= fraction < high: continue
        if foreground is not None:
            fg = cv2.resize(foreground.astype(np.uint8),(pw,ph),interpolation=cv2.INTER_NEAREST)
            if not (fg[b-y:d-y,a-x:c-x].astype(bool) & face[b:d,a:c]).any(): continue
        resized = cv2.resize(patch,(pw,ph),interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h,w,3),np.uint8); canvas[b:d,a:c] = resized[b-y:d-y,a-x:c-x]
        return canvas,mask,dict(overlap=fraction,bbox=[x,y,x+pw,y+ph],
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return None


def source_geometry_bin(sample):
    e = sample['detector_crop']['ellipse']; image = sample['image']
    major,minor = max(e[2:4]),min(e[2:4])
    scale = 2*major/math.sqrt(image['width']*image['height'])
    ratio = major/minor
    # Prespecified coarse matching; never relaxed when a stratum has no real.
    return (int(np.digitize(scale,[.05,.1,.2])), int(np.digitize(ratio,[1.3,1.8])),
            int((e[4] % math.pi)/(math.pi/4)))


class NaturalOcclusionDataset(Dataset):
    def __init__(self, clean_dataset, bank, *, seed=42, probability=.5, bins=BINS, attempts=256):
        if clean_dataset.training or not isinstance(bank,RealOcclusionBank):
            raise ValueError('Require unaugmented predicted train crops and authenticated real bank')
        if bank.data['adapter_sha256'] != hashlib.sha256(clean_dataset.annotation_path.read_bytes()).hexdigest():
            raise ValueError('Recipient split differs from train donor bank')
        self.clean = clean_dataset; self.bank = bank; self.seed = seed
        self.probability = probability; self.bins = tuple(tuple(x) for x in bins); self.attempts = attempts
        self.eligible = [i for i,s in enumerate(self.clean.samples) if bank.eligible_sample(s)]
        self.real_index = defaultdict(list)
        for i,s in enumerate(self.clean.samples):
            if s['image']['file_name'] not in bank.images: raise ValueError('Unverified train image')
            for slot in np.flatnonzero(s['labeled'] & ~s['visible']):
                self.real_index[(source_geometry_bin(s),int(slot))].append(i)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.requests = balanced_requests(self.eligible,seed=self.seed,epoch=epoch,probability=self.probability,bins=self.bins)

    def __len__(self): return len(self.clean)

    def pixels(self, tensor):
        return (tensor*torch.tensor(self.clean.std)[:,None,None]+torch.tensor(self.clean.mean)[:,None,None]).clamp(0,1)

    def __getitem__(self, index):
        file=self.clean.samples[index]['image']['file_name']
        self.bank.read(self.bank.images[file])  # authenticate recipient bytes, not only donor bytes
        normalized,target = self.clean[index]; clean = self.pixels(normalized)
        sample = self.clean.samples[index]; gp = target['geometry_prior']
        meta = dict(index=index,requested=index in self.requests,applied=False,reason='unaugmented',
                    recipient_video=self.bank.images[sample['image']['file_name']]['video'])
        row = dict(clean=clean,fixed=clean.clone(),donor=torch.zeros_like(clean),mask=torch.zeros(1,*clean.shape[-2:],dtype=torch.bool),
            real=clean.clone(),fake_xy=torch.zeros(2),real_xy=torch.zeros(2),slots=torch.tensor(0),
            real_valid=torch.tensor(False),geometry=gp['geometry'],support=gp['support'],
            covered=torch.zeros(5,dtype=torch.bool),target=copy.deepcopy(target),metadata=meta)
        if index not in self.requests: return row
        family,bin_id = self.requests[index]; meta.update(family=family,bin=bin_id,reason='placement_budget')
        rng = random.Random(f'{self.seed}:{self.epoch}:{index}:material')
        recipient = self.bank.images[sample['image']['file_name']]
        face = ellipse_region_mask(*clean.shape[-2:],gp['predicted_ellipse_crop'].numpy()).astype(bool)
        selected = select_material(self.bank,family,recipient,rng)
        if selected is None:
            meta['reason']='donor_budget'; return row
        patch,foreground,provenance = selected
        if provenance['donor_video']==meta['recipient_video']: raise ValueError('Same-video donor')
        placed = place_material(patch,foreground,face,interval=self.bins[bin_id],rng=rng,attempts=self.attempts)
        if placed is None: return row
        donor,mask,placement = placed
        row['donor']=torch.from_numpy(donor.copy()).permute(2,0,1).float()/255
        row['mask']=torch.from_numpy(mask[None].copy())
        clean_rgb=np.rint(clean.permute(1,2,0).numpy()*255).astype(np.uint8)
        pasted=np.where(mask[...,None],donor,clean_rgb)
        corrected=opaque_boundary_gradient(clean_rgb,pasted,mask,width=48,strength=2)
        # Preserve exact float clean pixels outside mask in both comparison arms.
        fixed=torch.from_numpy(corrected.copy()).permute(2,0,1).float()/255
        row['fixed']=torch.where(row['mask'],fixed,clean)
        points = target['keypoints']*(self.clean.input_size-1); states=target['keypoint_visibility'].clone()
        pixels = points.round().long(); h,w=mask.shape
        inside = torch.isfinite(points).all(1) & (pixels[:,0]>=0) & (pixels[:,0]<w) & (pixels[:,1]>=0) & (pixels[:,1]<h)
        states[~inside]=0
        covered = torch.zeros(5,dtype=torch.bool)
        for slot in range(5):
            if states[slot]>0 and mask[pixels[slot,1],pixels[slot,0]]: covered[slot]=True
        states[covered]=1
        row['covered']=covered
        row['target'].update(keypoint_visibility=states,labeled=states>0,visible=states==2)
        meta.update(provenance,**placement,applied=True,reason=None,covered_slots=covered.nonzero().flatten().tolist())
        candidates = covered.nonzero().flatten().tolist(); rng.shuffle(candidates)
        for slot in candidates:
            choices = [j for j in self.real_index[(source_geometry_bin(sample),slot)]
                if self.bank.images[self.clean.samples[j]['image']['file_name']]['video']!=meta['recipient_video']]
            rng.shuffle(choices)
            for j in choices[:16]:
                real_file=self.clean.samples[j]['image']['file_name']
                self.bank.read(self.bank.images[real_file])
                real_image,rt = self.clean[j]; real_image=self.pixels(real_image)
                real_xy=rt['keypoints'][slot]*(self.clean.input_size-1)
                half=15.5
                if not (points[slot]>=half).all() or not (points[slot]<=self.clean.input_size-1-half).all(): continue
                if not (real_xy>=half).all() or not (real_xy<=self.clean.input_size-1-half).all(): continue
                # Compare observed raw composite to observed real, never hidden clean RGB.
                observed=torch.where(row['mask'],row['donor'],clean)
                if abs(float(observed.mean()-real_image.mean()))>.2: continue
                row.update(real=real_image,fake_xy=points[slot],real_xy=real_xy,
                    slots=torch.tensor(slot),real_valid=torch.tensor(True))
                meta.update(real_index=j,real_video=self.bank.images[self.clean.samples[j]['image']['file_name']]['video'],
                    real_annotation_id=int(rt['annotation_id']),match_bin=source_geometry_bin(sample))
                return row
        meta['real_pair_reason']='no_matched_natural_state1_region'
        return row


def collate_natural(rows):
    return {key: ([r[key] for r in rows] if key in ('target','metadata') else torch.stack([r[key] for r in rows]))
            for key in rows[0]}


class CompositionAudit:
    def __init__(self): self.rows=Counter(); self.skips=Counter(); self.matched=0; self.applied=0
    def add(self, metadata):
        for row in metadata:
            if not row['requested']: continue
            key=f"{row['family']}/bin{row['bin']}"; self.rows[key]+=1
            if not row['applied']: self.skips[key]+=1
            else: self.applied+=1
            if 'real_index' in row: self.matched+=1
    def compute(self, *, maximum_skip=.10, maximum_stratum_skip=.20, minimum_real_pair_fraction=.5):
        requested=sum(self.rows.values()); skipped=sum(self.skips.values())
        fractions={k:self.skips[k]/n for k,n in self.rows.items()}
        enough=requested>0 and len(self.rows)==8 and min(self.rows.values())>=10
        return dict(requested=requested,skipped=skipped,requested_per_stratum=dict(self.rows),
            skipped_per_stratum=dict(self.skips),skip_fraction=skipped/max(1,requested),
            real_pair_fraction=self.matched/max(1,self.applied),
            passed=bool(enough and skipped/requested<=maximum_skip and max(fractions.values())<=maximum_stratum_skip
                        and self.matched/max(1,self.applied)>=minimum_real_pair_fraction),
            contract='new RGB experiment only; not historical real-face <=7% acceptance')
