"""Constant-memory point-level adoption diagnostics, never a training objective.

g is adoption, O/m is anomaly detection. Natural state1 and visible state2
are sparse annotations, not dense masks. Several points may share one token.
"""
import math

import torch
from torch.nn import functional as F

from src.data.messenger_policy_occlusion import compose
from src.models.messenger_occlusion_augmentation import frozen


GROUPS = ('artificial_occluded', 'natural_occluded', 'visible')


class GateCounters:
    def __init__(self):
        self.groups = {key:self.empty() for key in GROUPS}
        self.crops = 0
        self.excluded = {'missing':0, 'outside_or_invalid':0}

    @staticmethod
    def empty():
        return dict(count=0, sum=0., sum_square=0., minimum=math.inf, maximum=-math.inf,
                    below_001=0, below_0001=0, O_positive=0, outside_support=0,
                    effective_sum=0., position_sum=0.)

    @torch.no_grad()
    def add(self, output, targets, *, input_hw, channel_adoption, covered=None, bins=None):
        diag = output['prior_diagnostics']
        gate = diag['gate'].detach().float().cpu()
        anomaly = diag['anomalous'].detach().bool().cpu()
        if gate.ndim != 4 or gate.shape[1] != 1 or anomaly.shape != gate[:,0].shape:
            raise ValueError('Expected Bx1xHxW g and BxHxW O')
        if not torch.isfinite(gate).all() or (gate<0).any() or (gate>1).any():
            raise ValueError('Invalid adoption coefficient')
        channel = float(channel_adoption.detach().sigmoid().mean())
        coords = output['decoded_coordinates'].detach().float().cpu()
        if not torch.isfinite(coords).all():raise ValueError('Nonfinite keypoint coordinates')
        h,w = input_hw; gh,gw = gate.shape[-2:]
        for i,t in enumerate(targets):
            xy = t['keypoints'].detach().float().cpu()
            states = t['keypoint_visibility'].detach().cpu()
            support = t['geometry_prior']['support'].detach().bool().cpu()
            if support.shape != (gh,gw):raise ValueError('Support/g grid mismatch')
            valid = torch.isfinite(xy).all(-1) & (xy>=0).all(-1) & (xy<=1).all(-1)
            self.excluded['missing'] += int((states==0).sum())
            self.excluded['outside_or_invalid'] += int(((states>0)&~valid).sum())
            for slot in ((states>0)&valid).nonzero().flatten().tolist():
                # Match token assignment in the existing O diagnostic: normalized
                # xy -> crop pixel -> floor(pixel*grid/input), no edge clamping.
                x,y = xy[slot].tolist()
                gx,gy = int(x*(w-1)*gw/w),int(y*(h-1)*gh/h)
                artificial = covered is not None and bool(covered[i,slot])
                group = 'artificial_occluded' if artificial else ('natural_occluded' if states[slot]==1 else 'visible')
                keys=[group]
                if artificial and bins is not None:
                    if bins[i] not in range(4):raise ValueError('Missing artificial severity bin')
                    keys.append(f'artificial_bin{bins[i]}')
                g=float(gate[i,0,gy,gx]);m=bool(anomaly[i,gy,gx]);inside=bool(support[gy,gx])
                err=float(F.smooth_l1_loss(coords[i,slot],xy[slot],reduction='mean'))
                for key in keys:
                    s=self.groups.setdefault(key,self.empty())
                    s['count']+=1;s['sum']+=g;s['sum_square']+=g*g
                    s['minimum']=min(s['minimum'],g);s['maximum']=max(s['maximum'],g)
                    s['below_001']+=g<.01;s['below_0001']+=g<.001
                    s['O_positive']+=m;s['outside_support']+=not inside
                    s['effective_sum']+=g*int(m)*channel;s['position_sum']+=err
            self.crops+=1

    def compute(self):
        groups={}
        for key,s in self.groups.items():
            n=s['count'];mean=s['sum']/n if n else None
            groups[key]=dict(count=n,g_mean=mean,
                g_std=math.sqrt(max(0.,s['sum_square']/n-mean*mean)) if n else None,
                g_min=s['minimum'] if n else None,g_max=s['maximum'] if n else None,
                fraction_g_lt_0_01=s['below_001']/n if n else None,
                fraction_g_lt_0_001=s['below_0001']/n if n else None,
                O_positive_fraction=s['O_positive']/n if n else None,
                outside_support_count=s['outside_support'],
                effective_coefficient_mean=s['effective_sum']/n if n else None,
                position_smooth_l1_mean=s['position_sum']/n if n else None)
        return dict(groups=groups,crops=self.crops,excluded=self.excluded,
            definition='g adoption, not occlusion probability; effective coefficient = m*g*mean(sigmoid(channel))',
            denominator='annotated in-crop points, independent of predicted presence; visible means visible POINT, not necessarily unoccluded entire crop',
            position_metric='mean Smooth L1 of normalized crop xy; not GT Head normalized NME',
            sampling='floor(crop_pixel_xy * feature_grid / input_hw); includes outside-ellipse support points')


def gate_deltas(current, initial):
    """Same-input comparisons only; never silently compare changed denominators."""
    if current['crops']!=initial['crops'] or current['excluded']!=initial['excluded']:
        raise ValueError('Gate comparison population differs')
    result={}
    for key,c in current['groups'].items():
        base=initial['groups'][key]
        if c['count']!=base['count']:raise ValueError('Gate comparison point counts differ')
        result[key]={}
        for field in ('g_mean','effective_coefficient_mean','position_smooth_l1_mean'):
            a,b=c[field],base[field]
            result[key][field+'_delta']=None if a is None or b is None else a-b
            if field=='g_mean':
                result[key]['g_mean_relative_change']=None if a is None or b in (None,0.) else a/b-1
    return result


@torch.inference_mode()
def fixed_gate_probe(task, rows, *, actions=None, batch_size=4):
    """Same TRAIN syntheses and paired clean inputs at every epoch; no G sampling.

The clean reference uses the same point groups as its synthetic counterpart:
its artificial group denotes points that WILL be covered, not actual occlusion.
No validation/test augmentation is performed.
"""
    device=next(task.parameters()).device
    synthetic=GateCounters();clean=GateCounters();manifest=[]
    with frozen(task):
        for start in range(0,len(rows),batch_size):
            sub=rows[start:start+batch_size];images=[];targets=[];covered=[];bins=[]
            for row in sub:
                index=row['metadata']['index']
                action=(actions or {}).get(index,0) if row['candidates'] else None
                if action is not None:
                    candidate=row['candidates'][action]
                    image,t,c=compose(row['clean'],candidate['donor'],candidate['mask'],row['target'])
                    manifest.append(dict(index=index,action=action,**candidate['metadata']))
                else:
                    image,t,c=compose(row['clean'],row['clean'],torch.zeros(1,*row['clean'].shape[-2:],dtype=torch.bool),row['target'])
                images.append(image);targets.append(t);covered.append(c);bins.append(row['metadata'].get('bin'))
            batch=dict(target=targets,covered=torch.stack(covered).to(device),
                geometry=torch.stack([r['target']['geometry_prior']['geometry'] for r in sub]).to(device),
                support=torch.stack([r['target']['geometry_prior']['support'] for r in sub]).to(device))
            for which,pixels in ((synthetic,torch.stack(images)),(clean,torch.stack([r['clean'] for r in sub]))):
                output=task.output(pixels.to(device),batch)
                which.add(output,targets,input_hw=pixels.shape[-2:],channel_adoption=task.pipeline.channel_adoption,
                          covered=batch['covered'].cpu(),bins=bins)
    return dict(synthetic=synthetic.compute(),paired_clean=clean.compute(),manifest=manifest,
                scope='fixed train-only diagnostic; clean artificial group is the corresponding pre-paste point set')
