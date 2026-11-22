"""Independent O training and F1 checkpoint policy; no legacy run is modified."""
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch

from src.models.messenger_prototype_observer import prototype_loss, sparse_point_samples

O_METRIC = 'occlusion_f1'
T_METRIC = 'conditional_semantic_macro_f1_nme_0.10'


def rng_state():
    # CPU diagnostics must not create a CUDA context just to save RNG state.
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class OcclusionCounters:
    """State1 positive F1 at fixed margin>0, including support misses in crop.

    Not keypoint location F1. Missing and crop-outside points are excluded;
    support-outside labeled points remain false negatives/true negatives.
    """
    def __init__(self):
        self.slots = torch.zeros(5, 4, dtype=torch.int64)
        self.excluded = dict(missing=0, outside_crop=0, outside_support=0)

    def add(self, prior, xy, states, support, *, input_hw):
        if states.shape[1] != 5:
            raise ValueError('Five semantic slots required')
        margin = torch.where(support, prior['margin'].detach(), -torch.inf)
        for slot in range(5):
            values, labels, counts = sparse_point_samples(margin, xy[:, slot:slot+1], states[:, slot:slot+1],
                torch.ones_like(support), input_hw=input_hw)
            _, _, supported = sparse_point_samples(margin, xy[:, slot:slot+1], states[:, slot:slot+1], support, input_hw=input_hw)
            positive = labels.bool(); predicted = values > 0
            self.slots[slot] += torch.tensor([int((positive & predicted).sum()), int((~positive & predicted).sum()),
                int((positive & ~predicted).sum()), int((~positive & ~predicted).sum())])
            self.excluded['missing'] += counts['missing']
            self.excluded['outside_crop'] += counts['outside_crop']
            self.excluded['outside_support'] += supported['outside_support']

    def compute(self):
        tp, fp, fn, tn = self.slots.sum(0).tolist()
        if tp+fn == 0 or fp+tn == 0:
            raise ValueError('Validation must contain both occluded and visible points')
        semantic = 2*self.slots[:, 0] / (2*self.slots[:, 0]+self.slots[:, 1]+self.slots[:, 2]).clamp_min(1)
        return dict(occlusion_f1=2*tp/max(1, 2*tp+fp+fn), precision=tp/max(1, tp+fp),
            recall=tp/(tp+fn), visible_false_positive_rate=fp/(fp+tn), tp=tp, fp=fp, fn=fn, tn=tn,
            occlusion_semantic_macro_f1=float(semantic.mean()), semantic_f1=semantic.tolist(),
            semantic_confusion=self.slots.tolist(), exclusions=self.excluded.copy(),
            threshold='fixed per-cell threshold; margin > 0', positive_class='natural state1',
            scope='annotated in-crop points; support-outside points retained; not keypoint location F1')


class SupervisionAudit:
    """Small Kx5xHxW counters; no dense labels inferred from sparse annotations."""
    def __init__(self, k, h, w):
        self.positive = torch.zeros(k, 5, h, w, dtype=torch.int64)
        self.negative = torch.zeros_like(self.positive)
        self.conflicting_sample_cells = 0

    def add(self, selected, xy, states, support, *, input_hw):
        h, w = support.shape[-2:]
        cells = torch.arange(h*w, device=support.device).reshape(1, h, w).expand(len(support), -1, -1).float()
        for b in range(len(support)):
            labels_by_cell = {}
            for slot in range(5):
                values, labels, _ = sparse_point_samples(cells[b:b+1], xy[b:b+1, slot:slot+1],
                    states[b:b+1, slot:slot+1], support[b:b+1], input_hw=input_hw)
                for cell, label in zip(values.long().tolist(), labels.bool().tolist()):
                    counts = self.positive if label else self.negative
                    counts[int(selected[b]), slot, cell//w, cell%w] += 1
                    labels_by_cell.setdefault(cell, set()).add(label)
            self.conflicting_sample_cells += sum(len(v) == 2 for v in labels_by_cell.values())

    def compute(self):
        p, n = self.positive.sum(1), self.negative.sum(1)
        return dict(positive_counts=self.positive.tolist(), negative_counts=self.negative.tolist(),
            supervised_cells_per_center=((p+n)>0).sum((1, 2)).tolist(),
            unobserved_cells_per_center=((p+n)==0).sum((1, 2)).tolist(),
            conflicting_sample_cells=self.conflicting_sample_cells,
            coordinate_order='center, semantic slot, y, x')


class CenterUpdate:
    """Effective-window BCE normalization, center-only optimizer, precommit OOM replay."""
    def __init__(self, model, *, lr=1e-3, positive_weight, anchor_weight=.001, max_grad_norm=1.):
        self.model = model
        self.optimizer = model.optimizer(lr=lr)
        self.positive_weight = positive_weight; self.anchor_weight = anchor_weight
        self.max_grad_norm = max_grad_norm; self.steps = 0; self.needs_restore = False

    def update(self, images, geometry, support, xy, states, *, microbatch):
        if self.needs_restore:
            raise RuntimeError('Restore last complete checkpoint after a failed optimizer commit')
        if microbatch < 1 or not len(images):
            raise ValueError('Invalid batch')
        device = next(self.model.parameters()).device; input_hw = images.shape[-2:]
        _, _, counts = sparse_point_samples(torch.zeros_like(support, dtype=torch.float32), xy, states, support, input_hw=input_hw)
        denominator = counts['positives']*self.positive_weight+counts['negatives']
        self.optimizer.zero_grad(set_to_none=True)
        if not denominator:
            return dict(status='no_supervision_skipped', step=self.steps, counts=counts, microbatch=microbatch)
        saved_rng = rng_state(); retries = []; n = len(images)
        while True:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            total = 0.; committing = False
            try:
                for start in range(0, n, microbatch):
                    sl = slice(start, min(n, start+microbatch))
                    x, g, s, p, st = [v[sl].to(device) for v in (images, geometry, support, xy, states)]
                    _, _, c = sparse_point_samples(torch.zeros_like(s, dtype=torch.float32), p, st, s, input_hw=input_hw)
                    weight = (c['positives']*self.positive_weight+c['negatives'])/denominator
                    if not weight: continue
                    output = self.model.observe(x, geometry=g, support=s)
                    losses = prototype_loss(self.model.bank, output, p, st, s, input_hw=input_hw,
                        positive_weight=self.positive_weight, anchor_weight=self.anchor_weight)
                    if not torch.isfinite(losses['loss']): raise FloatingPointError('Nonfinite O loss')
                    (losses['loss']*weight).backward(); total += float(losses['loss'].detach())*weight
                for name, parameter in self.model.named_parameters():
                    if parameter is not self.model.bank.centers and (parameter.requires_grad or parameter.grad is not None):
                        raise AssertionError('Gradient/optimizer escaped centers: '+name)
                gradient_norm=float(torch.nn.utils.clip_grad_norm_([self.model.bank.centers], self.max_grad_norm, error_if_nonfinite=True))
                committing = True; self.optimizer.step(); self.steps += 1
                self.optimizer.zero_grad(set_to_none=True)
                return dict(status='updated', step=self.steps, loss=total, counts=counts, microbatch=microbatch,
                            retries=retries,gradient_norms={'o':gradient_norm})
            except torch.OutOfMemoryError:
                if committing:
                    self.needs_restore = True
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                output = losses = x = g = s = p = st = None
                restore_rng(saved_rng)
                if device.type == 'cuda': torch.cuda.empty_cache()
                if microbatch == 1: raise
                retries.append(microbatch); microbatch = max(1, microbatch//2)
            except BaseException:
                if committing: self.needs_restore = True
                raise


class F1Checkpoint:
    """Save maximum validation F1; exact ties retain earlier epoch.

    min_delta affects patience only, never suppresses a higher-F1 best snapshot.
    All module/optimizer/scheduler/EMA states plus RNG and loader RNG are saved.
    The caller's immutable contract must include source/data/parent hashes.
    """
    def __init__(self, *, contract, metric, modules, optimizers, schedulers,
                 loader_generator, minimum_epoch=10, patience=5, min_delta=1e-4):
        if metric not in (O_METRIC, T_METRIC): raise ValueError('Only declared F1 selection is supported')
        self.contract = copy.deepcopy(dict(experiment=contract, metric=metric, minimum_epoch=minimum_epoch,
            patience=patience, min_delta=min_delta, tie='earlier epoch', selection='maximum validation F1'))
        self.metric = metric; self.modules = modules; self.optimizers = optimizers; self.schedulers = schedulers
        self.loader_generator = loader_generator; self.minimum_epoch = minimum_epoch
        self.patience = patience; self.min_delta = min_delta
        self.epoch = 0; self.best = -math.inf; self.best_epoch = 0; self.patience_reference = -math.inf
        self.stale = 0; self.last_metrics = None

    def record(self, metrics, *, split='val'):
        if split != 'val': raise ValueError('Checkpoint selection is validation-only')
        value = float(metrics[self.metric])
        if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError('Invalid F1')
        self.epoch += 1
        improved = value > self.best
        if improved: self.best = value; self.best_epoch = self.epoch
        significant = value > self.patience_reference+self.min_delta
        if significant: self.patience_reference = value
        if self.epoch < self.minimum_epoch or significant: self.stale = 0
        else: self.stale += 1
        self.last_metrics = copy.deepcopy(metrics)
        return improved

    @property
    def should_stop(self):
        return self.epoch >= self.minimum_epoch and self.stale >= self.patience

    def state(self):
        return dict(schema='occlusion-f1-checkpoint-v1', contract=copy.deepcopy(self.contract),
            modules={k:v.state_dict() for k,v in self.modules.items()},
            optimizers={k:v.state_dict() for k,v in self.optimizers.items()},
            schedulers={k:v.state_dict() for k,v in self.schedulers.items()}, rng=rng_state(),
            loader_rng=self.loader_generator.get_state(), epoch=self.epoch, best=self.best,
            best_epoch=self.best_epoch, patience_reference=self.patience_reference, stale=self.stale,
            last_metrics=self.last_metrics)

    def restore(self, state):
        if state.get('schema') != 'occlusion-f1-checkpoint-v1' or state.get('contract') != self.contract:
            raise ValueError('Changed experiment/source/data/selection contract')
        for name, objects in [('modules',self.modules), ('optimizers',self.optimizers), ('schedulers',self.schedulers)]:
            if set(state[name]) != set(objects): raise ValueError('Incomplete resume components')
        for name, objects in [('modules',self.modules), ('optimizers',self.optimizers), ('schedulers',self.schedulers)]:
            for key, obj in objects.items(): obj.load_state_dict(state[name][key])
        restore_rng(state['rng']); self.loader_generator.set_state(state['loader_rng'])
        for key in ('epoch','best','best_epoch','patience_reference','stale','last_metrics'): setattr(self,key,copy.deepcopy(state[key]))

    def save(self, directory, *, improved, complete=False):
        if self.epoch < 1: raise ValueError('No validated epoch')
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        state = self.state()
        def write(name):
            tmp = directory/(name+'.tmp'); torch.save(state,tmp); tmp.replace(directory/name)
        if improved: write('best.pt')
        write('last.pt')
        if self.epoch % 5 == 0 and not complete:
            name = f'epoch_{self.epoch:03d}.pt'; write(name)
            for path in directory.glob('epoch_[0-9][0-9][0-9].pt'):
                if path.name != name: path.unlink()
        if complete:
            for path in directory.glob('epoch_[0-9][0-9][0-9].pt'): path.unlink()
