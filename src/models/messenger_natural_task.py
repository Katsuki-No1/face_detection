"""Adapter and disjoint optimization for the RGB-augmentation experiment."""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from keypoint_comparison1.landmark_detection.geometry_prior_recovery import generator_losses
from src.models.messenger_occlusion_training import rng_state, restore_rng
from src.models.messenger_occlusion_augmentation import separated_losses


class MessengerTask(nn.Module):
    def __init__(self, pipeline, criterion, *, mean, std, reconstruction_weight=.1):
        super().__init__()
        self.pipeline=pipeline; self.criterion=criterion; self.reconstruction_weight=reconstruction_weight
        self.register_buffer('mean',torch.tensor(mean)[None,:,None,None])
        self.register_buffer('std',torch.tensor(std)[None,:,None,None])
        self.pipeline.observer.requires_grad_(False).eval()
        self.pipeline.bank.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode); self.pipeline.observer.eval(); self.pipeline.bank.eval()
        if not any(p.requires_grad for p in self.pipeline.base.backbone.parameters()):
            self.pipeline.base.backbone.eval()
        return self

    def loss(self, pixels, batch, *, hardness=False):
        x=(pixels-self.mean)/self.std
        valid=torch.stack([t['labeled'] for t in batch['target']]).to(x.device)
        output=self.pipeline(x,geometry=batch['geometry'],support=batch['support'],augmentation_valid_mask=valid)
        if hardness:
            covered=batch['covered'].detach() & valid
            truth=torch.stack([t['keypoints'] for t in batch['target']]).to(x.device).detach()
            # Coordinates remain supervised even when the model reduces presence.
            position=F.smooth_l1_loss(output['decoded_coordinates'],truth,reduction='none').mean(-1)
            if 'presence_logits' not in output: raise ValueError('Hardness requires the declared presence head')
            presence=F.softplus(-output['presence_logits'])
            return ((position+presence)*covered).sum()/covered.sum().clamp_min(1)
        with torch.no_grad(),torch.autocast(x.device.type,enabled=False):
            clean=self.pipeline.observer(((batch['clean'].detach()-self.mean)/self.std).float())
        keypoint=self.criterion(output,batch['target'])['loss_total']
        return generator_losses(output,clean,keypoint,reconstruction_weight=self.reconstruction_weight)['total']


def slice_batch(batch, start, stop, device):
    return {key: (value[start:stop] if isinstance(value,list) else value[start:stop].to(device))
            for key,value in batch.items()}


class NaturalUpdate:
    """G/D/T effective-window updates with detached boundaries and atomic precheck.

    All gradients are computed at the same preupdate weights. OOM before commit
    replays the same window; partial optimizer commits require checkpoint restore.
    FP32 is explicit for this initial runtime; no unverified AMP resume contract.
    """
    def __init__(self, generator, discriminator, task, *, optimizers, arm='gdt',
                 realism_weight=1., preserve_weight=10., hardness_weight=.1, max_grad_norm=1.):
        if arm not in ('gdt','gd','frozen_g','fixed'): raise ValueError('Unsupported experiment arm')
        self.generator=generator; self.discriminator=discriminator; self.task=task
        self.optimizers=optimizers; self.arm=arm; self.max_grad_norm=max_grad_norm
        self.weights=dict(realism_weight=realism_weight,preserve_weight=preserve_weight,
                          hardness_weight=hardness_weight if arm=='gdt' else 0.)
        required={'g','d','t'} if arm in ('gdt','gd') else {'t'}
        if set(optimizers)!=required: raise ValueError('Optimizer inventory differs from arm')
        owners={'g':generator,'d':discriminator,'t':task}; seen=set()
        for name,opt in optimizers.items():
            params=[p for group in opt.param_groups for p in group['params']]
            ids={id(p) for p in params}
            if len(ids)!=len(params) or seen & ids: raise ValueError('Duplicate/shared optimizer ownership')
            if ids!={id(p) for p in owners[name].parameters() if p.requires_grad}: raise ValueError('Optimizer ownership mismatch')
            seen |= ids
        if any(p.requires_grad for p in task.pipeline.bank.parameters()): raise ValueError('O must be frozen in G/D/T arms')
        self.step=0; self.needs_restore=False

    def update(self,batch,*,microbatch):
        if self.needs_restore: raise RuntimeError('Restore complete checkpoint before continuing')
        n=len(batch['clean']); device=next(self.task.parameters()).device
        if microbatch<1 or not n: raise ValueError('Invalid effective batch')
        modules=(self.generator,self.discriminator,self.task)
        saved_rng=rng_state()
        buffers=[{name:v.detach().clone() for name,v in m.named_buffers()} for m in modules]
        retries=[]; g_active=bool(batch['mask'].any())
        while True:
            for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
            self.task.train(); self.generator.train(); self.discriminator.train()
            committing=False; totals={}; d_active=False
            try:
                for start in range(0,n,microbatch):
                    b=slice_batch(batch,start,min(n,start+microbatch),device); weight=len(b['clean'])/n
                    if self.arm in ('gdt','gd'):
                        losses=separated_losses(self.generator,self.discriminator,self.task,b,**self.weights)
                        # Denominators for point hardness and matched D regions are
                        # global effective-window counts, not mean of microbatch means.
                        hard_total=int(batch['covered'].sum()); hard_count=int(b['covered'].sum())
                        d_total=int(batch['real_valid'].sum()); d_count=losses['matched_regions']
                        hw=hard_count/max(1,hard_total); dw=d_count/max(1,d_total)
                        g=self.weights['realism_weight']*losses['realism']*dw
                        g=g+self.weights['preserve_weight']*losses['preserve']*weight
                        g=g-self.weights['hardness_weight']*losses['hardness']*hw
                        for value in (g,losses['t']):
                            if not torch.isfinite(value):raise FloatingPointError('Nonfinite G/T objective')
                        if losses['d'] is not None:
                            if not torch.isfinite(losses['d']):raise FloatingPointError('Nonfinite D objective')
                            (losses['d']*dw).backward(); d_active=True
                            totals['d']=totals.get('d',0.)+float(losses['d'].detach())*dw
                        g.backward(); (losses['t']*weight).backward()
                        totals['g']=totals.get('g',0.)+float(g.detach())
                        totals['t']=totals.get('t',0.)+float(losses['t'].detach())*weight
                        for name,factor in [('hardness',hw),('realism',dw),('preserve',weight)]:
                            totals[name]=totals.get(name,0.)+float(losses[name].detach())*factor
                    else:
                        with torch.no_grad():
                            image=(b['fixed'] if self.arm=='fixed' else self.generator(b['clean'],b['donor'],b['mask'])['image'])
                        loss=self.task.loss(image,b,hardness=False)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite task objective')
                        (loss*weight).backward(); totals['t']=totals.get('t',0.)+float(loss.detach())*weight
                active=[name for name in self.optimizers if (name!='d' or d_active) and (name!='g' or g_active)]
                norms={}
                for name in active:
                    parameters=[p for group in self.optimizers[name].param_groups for p in group['params']]
                    if not any(p.grad is not None for p in parameters):raise RuntimeError('No gradient for '+name)
                    norms[name]=float(torch.nn.utils.clip_grad_norm_(parameters,self.max_grad_norm,error_if_nonfinite=True))
                if any(p.grad is not None for p in self.task.pipeline.bank.parameters()): raise AssertionError('Gradient escaped into O')
                committing=True
                for name in active:self.optimizers[name].step()
                self.step+=1
                for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
                return dict(status='updated',step=self.step,losses=totals,microbatch=microbatch,retries=retries,
                            d_updated=d_active,g_updated=g_active and 'g' in self.optimizers,gradient_norms=norms)
            except torch.OutOfMemoryError:
                if committing:self.needs_restore=True;raise
                for opt in self.optimizers.values():opt.zero_grad(set_to_none=True)
                losses=b=g=loss=image=None
                with torch.no_grad():
                    for m,saved in zip(modules,buffers):
                        for name,value in m.named_buffers():value.copy_(saved[name])
                restore_rng(saved_rng)
                if device.type=='cuda':torch.cuda.empty_cache()
                if microbatch==1:raise
                retries.append(microbatch);microbatch=max(1,microbatch//2)
            except BaseException:
                # A failed update must not continue with partially changed BN/state.
                self.needs_restore=True
                raise
