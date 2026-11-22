"""Opt-in center-only adaptation of Messenger's fixed k-means observer.

Deployed order stays HEAD-free FACE/ellipse -> crop DINO -> Messenger -> Query.
The hard mask is used for inference; sparse occlusion BCE trains the centers via
the continuous similarity margin. No gradient is estimated through the mask.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from keypoint_comparison1.landmark_detection.geometry_prior_recovery import (
    GeometryPriorBank, PriorMessengerModel, feature_similarity,
)


def bank_digest(bank):
    digest = hashlib.sha256(json.dumps(bank.get_extra_state(), sort_keys=True).encode())
    for name, value in sorted(bank.named_buffers()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LearnablePrototypeBank(nn.Module):
    """Only spatial feature centers are parameters; routing/calibration are fixed."""

    def __init__(self, bank: GeometryPriorBank, *, logit_temperature=0.1):
        super().__init__()
        if not isinstance(bank, GeometryPriorBank):
            raise TypeError("Initialize from an authenticated fixed GeometryPriorBank")
        if not math.isfinite(logit_temperature) or logit_temperature <= 0:
            raise ValueError("logit_temperature must be finite and positive")
        self.fixed = copy.deepcopy(bank).requires_grad_(False).eval()
        self.centers = nn.Parameter(bank.centers.detach().clone())
        self.logit_temperature = float(logit_temperature)
        self.origin_sha256 = bank_digest(bank)

    @property
    def thresholds(self):
        return self.fixed.thresholds

    def get_extra_state(self):
        return dict(schema="learnable-occlusion-prototypes-v1",
                    origin_sha256=self.origin_sha256,
                    logit_temperature=self.logit_temperature)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Prototype origin or logit temperature differs from checkpoint")

    def forward(self, features, geometry, support):
        # Existing fixed forward validates shapes and performs identical routing.
        # Detach geometry too: it can originate in a differentiable detector.
        prior = self.fixed(features.detach(), geometry.detach(), support.detach())
        selected = prior["selected_center"]
        score = feature_similarity(features.detach(), self.centers[selected], self.fixed.channel_scale)
        margin = self.thresholds[selected] - score
        return dict(prior, score=score, margin=margin,
                    occlusion_logits=margin / self.logit_temperature,
                    anomalous=(margin > 0) & support.detach())

    def anchor_loss(self):
        # Original k-means coordinates remain immutable in fixed.centers.
        delta = (self.centers - self.fixed.centers) / self.fixed.channel_scale[None, :, None, None]
        return delta.square().mean()


def sparse_point_samples(logits, keypoints, states, support, *, input_hw):
    """Gather individual annotated points, retaining conflicting same-cell labels.

    xy are normalized crop coordinates (pixel/(input_size-1)), x right/y down.
    state1 = occluded, state2 = visible, state0 = missing/ignored. No point is
    clipped into the crop/support and no unlabeled cell becomes a negative.
    """
    if logits.ndim != 3 or support.shape != logits.shape or support.dtype != torch.bool:
        raise ValueError("Expected BHW logits and boolean support")
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2 or states.shape != keypoints.shape[:2]:
        raise ValueError("Expected BxNx2 coordinates and BxN states")
    if len(keypoints) != len(logits) or not ((states == 0) | (states == 1) | (states == 2)).all():
        raise ValueError("Invalid batch or keypoint states")
    ih, iw = input_hw
    if ih < 2 or iw < 2:
        raise ValueError("Invalid crop size")
    xy = keypoints.detach()
    labels = states.detach()
    finite = torch.isfinite(xy).all(-1)
    # Pixel bounds mirror the crop convention, including subpixel last-edge points.
    pixels = xy * xy.new_tensor([iw - 1, ih - 1])
    inside = finite & (pixels >= 0).all(-1) & (pixels[..., 0] < iw) & (pixels[..., 1] < ih)
    safe = torch.where(inside[..., None], pixels, torch.zeros_like(pixels))
    h, w = logits.shape[-2:]
    x = (safe[..., 0] * w / iw).long()
    y = (safe[..., 1] * h / ih).long()
    b = torch.arange(len(logits), device=logits.device)[:, None].expand_as(states)
    valid = inside & (labels > 0) & support[b, y, x]
    return logits[b, y, x][valid], (labels[valid] == 1).to(logits.dtype), dict(
        supervised=int(valid.sum()), positives=int((valid & (labels == 1)).sum()),
        negatives=int((valid & (labels == 2)).sum()),
        missing=int((labels == 0).sum()),
        outside_crop=int(((labels > 0) & ~inside).sum()),
        outside_support=int(((labels > 0) & inside & ~support[b, y, x]).sum()))


def prototype_loss(bank, prior, keypoints, states, support, *, input_hw,
                   positive_weight=1.0, anchor_weight=0.001):
    """Weighted sparse BCE. Fix positive_weight from TRAIN inventory, never val/test."""
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("Invalid fixed positive weight")
    if not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError("Invalid anchor weight")
    values, labels, counts = sparse_point_samples(prior["occlusion_logits"], keypoints,
                                                states, support, input_hw=input_hw)
    if not len(values):
        # Caller must skip the optimizer step, including momentum, for this batch.
        raise ValueError("No supervised points; skip this optimizer step")
    weights = torch.where(labels.bool(), positive_weight, 1.0)
    bce = (F.binary_cross_entropy_with_logits(values, labels, reduction="none") * weights).sum() / weights.sum()
    anchor = bank.anchor_loss()
    return dict(loss=bce + anchor_weight * anchor, bce=bce, anchor=anchor, counts=counts)


class CenterOnlyObserver(nn.Module):
    """Frozen inference pipeline plus a separate differentiable O training route.

    PriorMessengerModel.forward deliberately runs O under no_grad. Therefore
    training calls observe()/loss(), outside that context. Inference still uses
    the original pipeline implementation and the updated bank's hard mask.
    """

    def __init__(self, pipeline, *, logit_temperature=0.1):
        super().__init__()
        if not isinstance(pipeline, PriorMessengerModel) or pipeline.fusion_mode != "prior":
            raise ValueError("Requires a prior-fusion Messenger pipeline")
        self.pipeline = copy.deepcopy(pipeline).requires_grad_(False).eval()
        self.pipeline.zero_grad(set_to_none=True)
        self.pipeline.bank = LearnablePrototypeBank(self.pipeline.bank, logit_temperature=logit_temperature)

    @property
    def bank(self):
        return self.pipeline.bank

    def train(self, mode=True):
        super().train(mode)
        # Dropout/BN/staged-unfreeze must never alter the frozen feature system.
        self.pipeline.eval()
        return self

    def observe(self, images, *, geometry, support):
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            features = self.pipeline.observer(images.detach().float())
        return self.bank(features, geometry, support)

    def forward(self, images, **kwargs):
        return self.pipeline(images, **kwargs)

    def optimizer(self, *, lr=0.001):
        trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if len(trainable) != 1 or trainable[0][1] is not self.bank.centers:
            raise ValueError("Only prototype centers may be trainable")
        # No weight decay toward zero; optional anchor is toward original centers.
        return torch.optim.Adam([self.bank.centers], lr=lr, weight_decay=0)


def attach_learnable_prototypes(pipeline, *, enabled=False, **kwargs):
    return CenterOnlyObserver(pipeline, **kwargs) if enabled else pipeline
