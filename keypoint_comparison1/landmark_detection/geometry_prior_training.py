"""Opt-in prior runtime: authenticated construction, atomic G/D update windows.

This module is not a production launcher. Epoch scheduling, full validation,
W&B, isolated preflight and four-gate registration remain the caller's duty.
An OOM before optimizer commit may replay the same window at a smaller batch.
An OOM during commit must recover from a genuine complete checkpoint instead.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import random

import numpy as np
import torch

from .geometry_prior_recovery import (
    GeometryPriorBank, attach_geometry_prior, generator_losses, discriminator_loss,
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def build_prior_model(raw, *, prior_path, prior_sha256, fusion_mode="prior"):
    """Same base/warm start for matched control and every prior arm."""
    from .config import configured_keypoint_target_policy, configured_state_head_mode
    from .model import build_landmark_model
    from .training import initialize_from_finetune_checkpoint
    if file_hash(prior_path) != prior_sha256:
        raise ValueError("Prior file hash mismatch")
    base, _ = build_landmark_model(raw,variant="query_refinement",initialize_from_detector=False)
    initialization = initialize_from_finetune_checkpoint(base,raw["train"]["finetune_checkpoint"],
        variant="query_refinement",expected_target_policy=configured_keypoint_target_policy(raw),
        expected_state_head_mode=configured_state_head_mode(raw))
    bank=GeometryPriorBank(torch.load(prior_path,map_location="cpu",weights_only=True))
    model=attach_geometry_prior(base,bank,enabled=True,
        observation_backbone_sha256=raw["train"]["finetune_checkpoint"]["sha256"],fusion_mode=fusion_mode)
    return model, initialization


def generator_optimizer(model, train_config):
    from .training import _optimizer
    optimizer=_optimizer(model.base,train_config)
    optimizer.add_param_group(dict(params=[model.channel_adoption],
        lr=float(train_config.get("lr_messenger",1e-4)),weight_decay=0.,name="prior_channel_adoption"))
    return optimizer


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:torch.cuda.set_rng_state_all(state["cuda"])


class PriorEMA:
    """EMA understands immutable prior metadata in state_dict extra-state."""
    def __init__(self,model,decay=.9998):
        if not 0 <= decay < 1:raise ValueError("Invalid EMA decay")
        self.module=copy.deepcopy(model).requires_grad_(False).eval()
        self.decay=decay;self.updates=0

    @torch.no_grad()
    def update(self,model):
        self.updates+=1
        decay=self.decay*(1-math.exp(-self.updates/2000.))
        source=model.state_dict()
        for name,value in self.module.state_dict().items():
            other=source[name]
            if not torch.is_tensor(value):
                if value!=other:raise ValueError("Prior metadata changed during EMA update")
            elif name.startswith(("observer.","bank.")):value.copy_(other)
            elif value.is_floating_point():value.mul_(decay).add_(other,alpha=1-decay)
            else:value.copy_(other)


class OptimizerCommitError(RuntimeError):
    """Do not replay after a potentially partial optimizer commit."""


class PriorUpdateRuntime:
    def __init__(self, model, criterion, optimizer, *, contract, discriminator=None,
                 discriminator_optimizer=None, adversarial_weight=0., reconstruction_weight=.1,
                 amp=True, max_grad_norm=1., ema_decay=.9998):
        if any(not math.isfinite(v) or v<0 for v in [adversarial_weight,reconstruction_weight]):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not math.isfinite(max_grad_norm) or max_grad_norm<=0:raise ValueError("Invalid gradient norm")
        if adversarial_weight and model.fusion_mode=="messenger":
            raise ValueError("Matched control has adversarial disabled")
        if bool(discriminator is not None)!=bool(discriminator_optimizer is not None):
            raise ValueError("D and D optimizer must be supplied together")
        if bool(discriminator is not None)!=bool(adversarial_weight):
            raise ValueError("D exists only for the explicitly enabled adversarial arm")
        self.model=model;self.criterion=criterion;self.optimizer=optimizer
        self.discriminator=discriminator;self.d_optimizer=discriminator_optimizer
        self.device=next(model.parameters()).device
        self.adversarial_weight=adversarial_weight;self.reconstruction_weight=reconstruction_weight
        self.max_grad_norm=max_grad_norm
        self.scaler=torch.amp.GradScaler("cuda",enabled=amp and self.device.type=="cuda",init_scale=1024.)
        self.ema=PriorEMA(model,ema_decay)
        self.contract=copy.deepcopy(dict(experiment=contract,adversarial_weight=adversarial_weight,
            reconstruction_weight=reconstruction_weight,max_grad_norm=max_grad_norm,
            amp=self.scaler.is_enabled(),ema_decay=ema_decay,
            update_order="accumulate_G_and_D_at_same_preupdate_D_then_joint_finite_commit"))
        self.step=0;self.in_window=False;self.requires_checkpoint_restore=False
        g=self._owned(optimizer)
        if g!={id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("G optimizer must own exactly trainable model parameters")
        if discriminator is not None:
            d=self._owned(discriminator_optimizer)
            if d & g or d!={id(p) for p in discriminator.parameters() if p.requires_grad}:
                raise ValueError("D optimizer ownership mismatch")

    @staticmethod
    def _owned(optimizer):
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate optimizer parameters")
        return set(ids)

    def _zero(self):
        self.optimizer.zero_grad(set_to_none=True)
        if self.d_optimizer is not None:self.d_optimizer.zero_grad(set_to_none=True)

    def update_window(self,images,clean_images,geometry,support,targets,*,microbatch):
        """One effective update; preserve all samples if an OOM reduces microbatch.

        Inputs are already augmented on CPU/device by the caller. No data-order
        replay occurs. Both optimizers commit only after both gradient sets are
        finite; overflow skips both. Empty anomaly windows never update D.
        """
        if self.requires_checkpoint_restore:raise OptimizerCommitError("Restore a complete checkpoint first")
        if self.in_window:raise RuntimeError("An update window is already active")
        count=len(images)
        if not count or images.shape!=clean_images.shape or len(targets)!=count or len(geometry)!=count or len(support)!=count:
            raise ValueError("Paired window inventory mismatch")
        if not isinstance(microbatch,int) or microbatch<1:raise ValueError("Invalid microbatch")
        microbatch=min(microbatch,count)
        # Initialize the lazy AMP scale before forward: a first-forward OOM
        # must also be able to reset scaler bookkeeping before replay.
        self.scaler.scale(torch.zeros((),device=self.device))
        before_rng=rng_state()
        buffers={name:value.detach().clone() for name,value in self.model.named_buffers()}
        self.in_window=True;attempts=[]
        try:
            while True:
                self._zero();self.model.train()
                if not any(p.requires_grad for p in self.model.base.backbone.parameters()):
                    self.model.base.backbone.eval()
                if self.discriminator is not None:self.discriminator.train()
                phase="backward";totals={};d_active=False
                try:
                    for start in range(0,count,microbatch):
                        end=min(count,start+microbatch);weight=(end-start)/count
                        x=images[start:end].to(self.device);clean_x=clean_images[start:end].to(self.device)
                        g=geometry[start:end].to(self.device);s=support[start:end].to(self.device)
                        valid=torch.stack([t['labeled'].bool() for t in targets[start:end]]).to(self.device)
                        with torch.no_grad(),torch.autocast(self.device.type,enabled=False):
                            clean=self.model.observer(clean_x.float())
                        # Fixed observer/prior coordinates stay FP32 even under student AMP.
                        with torch.autocast(self.device.type,enabled=self.scaler.is_enabled()):
                            output=self.model(x,geometry=g,support=s,augmentation_valid_mask=valid)
                            task=self.criterion(output,targets[start:end])['loss_total']
                            losses=generator_losses(output,clean,task,discriminator=self.discriminator,
                                geometry=g,reconstruction_weight=self.reconstruction_weight,
                                adversarial_weight=self.adversarial_weight)
                            dloss=None
                            if self.discriminator is not None and output['prior_diagnostics']['anomalous'].any():
                                dloss=discriminator_loss(self.discriminator,output,clean,g)
                        if not torch.isfinite(losses['total']):raise FloatingPointError("Non-finite G objective")
                        if dloss is not None:
                            if not torch.isfinite(dloss):raise FloatingPointError("Non-finite D objective")
                            self.scaler.scale(dloss*weight).backward();d_active=True
                            totals['discriminator']=totals.get('discriminator',0.)+float(dloss.detach())*weight
                        self.scaler.scale(losses['total']*weight).backward()
                        for name,value in losses.items():totals[name]=totals.get(name,0.)+float(value.detach())*weight
                        del output,losses,task,clean,x,clean_x,g,s,valid,dloss
                    active=[self.optimizer]+([self.d_optimizer] if d_active else [])
                    for opt in active:self.scaler.unscale_(opt)
                    finite=all(torch.isfinite(p.grad).all() for opt in active for group in opt.param_groups
                        for p in group['params'] if p.grad is not None)
                    if not finite:
                        if not self.scaler.is_enabled():raise FloatingPointError("Non-finite gradients")
                        self.scaler.update(new_scale=self.scaler.get_scale()/2)
                        self._zero()
                        return dict(status="overflow_skipped_both",step=self.step,microbatch=microbatch,attempts=attempts)
                    for opt in active:
                        torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']],self.max_grad_norm,error_if_nonfinite=True)
                    phase="commit"
                    for opt in active:self.scaler.step(opt)
                    self.scaler.update();self.ema.update(self.model);self.step+=1
                    self._zero()
                    return dict(status="updated",step=self.step,microbatch=microbatch,effective_batch=count,
                        discriminator_updated=d_active,losses=totals,attempts=attempts)
                except torch.OutOfMemoryError as error:
                    if phase=="commit":
                        self.requires_checkpoint_restore=True
                        raise OptimizerCommitError("OOM during optimizer commit; restore last complete checkpoint") from error
                    attempts.append(dict(microbatch=microbatch,error="oom_before_optimizer_commit"))
                    self._zero()
                    # Drop the failed graph before attempting a smaller batch.
                    output=losses=task=clean=x=clean_x=g=s=valid=dloss=None
                    with torch.no_grad():
                        for name,value in self.model.named_buffers():value.copy_(buffers[name])
                    restore_rng(before_rng)
                    if self.scaler.is_enabled():self.scaler.update(new_scale=self.scaler.get_scale())
                    if self.device.type=="cuda":torch.cuda.empty_cache()
                    if microbatch==1:raise
                    microbatch=max(1,microbatch//2)
                except Exception as error:
                    if phase=="commit":
                        self.requires_checkpoint_restore=True
                        raise OptimizerCommitError("Incomplete optimizer commit; restore last complete checkpoint") from error
                    raise
        finally:
            self.in_window=False;self._zero()

    def checkpoint(self,*,epoch,schedulers=None,loader_generator=None):
        if self.in_window or self.requires_checkpoint_restore:
            raise RuntimeError("Cannot checkpoint an incomplete optimizer window")
        return dict(schema="geometry-prior-runtime-v1",contract=copy.deepcopy(self.contract),
            model=self.model.state_dict(),ema=self.ema.module.state_dict(),ema_updates=self.ema.updates,
            optimizer=self.optimizer.state_dict(),discriminator=self.discriminator.state_dict() if self.discriminator is not None else None,
            discriminator_optimizer=self.d_optimizer.state_dict() if self.d_optimizer is not None else None,
            scaler=self.scaler.state_dict(),epoch=int(epoch),step=self.step,rng=rng_state(),
            schedulers={k:v.state_dict() for k,v in (schedulers or {}).items()},
            loader_generator=loader_generator.get_state() if loader_generator is not None else None)

    def restore(self,state,*,schedulers=None,loader_generator=None):
        if self.in_window:raise RuntimeError("Cannot restore during an optimizer window")
        if state.get('schema')!='geometry-prior-runtime-v1' or state.get('contract')!=self.contract:
            raise ValueError("Resume experiment/runtime contract differs")
        if set(state['schedulers'])!=set(schedulers or {}):raise ValueError("Resume scheduler inventory differs")
        if (state['loader_generator'] is None)!=(loader_generator is None):raise ValueError("Resume loader RNG contract differs")
        if (state['discriminator'] is None)!=(self.discriminator is None):raise ValueError("Resume D contract differs")
        self.model.load_state_dict(state['model'],strict=True)
        self.ema.module.load_state_dict(state['ema'],strict=True);self.ema.updates=int(state['ema_updates'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.discriminator is not None:
            self.discriminator.load_state_dict(state['discriminator'],strict=True)
            self.d_optimizer.load_state_dict(state['discriminator_optimizer'])
        self.scaler.load_state_dict(state['scaler']);self.step=int(state['step'])
        for name,scheduler in (schedulers or {}).items():scheduler.load_state_dict(state['schedulers'][name])
        if loader_generator is not None:loader_generator.set_state(state['loader_generator'])
        restore_rng(state['rng']);self.requires_checkpoint_restore=False;self._zero()
        return int(state['epoch'])
