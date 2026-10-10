"""Continuous crop-DINO recovery; all inference inputs are student features.

Option 1 supervises adoption with a detached scale-normalized oracle. Option 3
conditions the same local logit on a soft latent structure (not pose angles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from itertools import combinations
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .heads import sine_position_encoding_2d
from .messenger_recovery import patch_occlusion_ratios


@dataclass(frozen=True)
class ContinuousRecoveryConfig:
    enabled: bool = False
    layers: int = 2
    heads: int = 6
    feedforward_ratio: float = 2.0
    adoption: str = "optimal"  # alpha, local, optimal
    structure_enabled: bool = False
    structure_dim: int = 96
    structure_heads: int = 4
    prototypes: int = 8
    temperature: float = 0.20
    initial_gate: float = 0.10
    mixing_warmup_epochs: int = 2
    beta: float = 0.10
    huber_delta: float = 1.0
    scale_floor: float = 0.05
    scale_samples: int = 256
    oracle_epsilon: float = 1.0e-6
    candidate_weight: float = 0.10
    output_weight: float = 0.10
    gram_weight: float = 0.0
    alpha_weight: float = 0.01
    first_alpha_weight: float = 0.5
    adoption_weight: float = 0.01
    descriptor_weight: float = 0.10
    consistency_weight: float = 0.0
    cosine_weight: float = 1.0
    amplitude_weight: float = 1.0
    multi_scale: bool = False
    multi_level: bool = False
    reference_layers: tuple[int, ...] = (3, 7, 11)
    reference_factors: tuple[int, ...] = (1, 2, 4)
    reference_encoding: str = "crop_rect_level_scale_linear_v1"
    reference_adapter_seed: int = 20261004

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, embed_dim=None):
        raw = dict(raw or {})
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown continuous recovery settings: {sorted(unknown)}")
        for key in ("reference_layers", "reference_factors"):
            if key in raw:
                raw[key] = tuple(raw[key])
        cfg = cls(**raw)
        if not isinstance(cfg.multi_scale, bool) or not isinstance(cfg.multi_level, bool):
            raise ValueError("multi_scale and multi_level must be boolean")
        if cfg.reference_layers != (3, 7, 11) or cfg.reference_factors != (1, 2, 4):
            raise ValueError("Reference layers/factors are locked to [3,7,11]/[1,2,4]")
        if cfg.reference_encoding != "crop_rect_level_scale_linear_v1":
            raise ValueError("Unsupported reference position encoding")
        if cfg.layers != 2 or cfg.heads < 1 or (cfg.enabled and embed_dim and embed_dim % cfg.heads):
            raise ValueError("Continuous recovery requires 2 layers and divisible attention heads")
        if cfg.adoption not in {"alpha", "local", "optimal"}:
            raise ValueError("adoption must be alpha/local/optimal; pretrained pose option 2 is deferred")
        if cfg.structure_enabled and cfg.adoption == "alpha":
            raise ValueError("Structure correction requires the learned adoption gate")
        if cfg.structure_dim < 4 or cfg.structure_heads < 1 or cfg.structure_dim % cfg.structure_heads:
            raise ValueError("structure_dim must be divisible by structure_heads")
        if cfg.prototypes < 2 or cfg.scale_samples < 2 or cfg.mixing_warmup_epochs < 0:
            raise ValueError("Invalid prototype/statistics/warmup settings")
        for key in ("temperature", "beta", "huber_delta", "scale_floor", "oracle_epsilon", "feedforward_ratio"):
            if not math.isfinite(float(getattr(cfg, key))) or getattr(cfg, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 < cfg.initial_gate < 1 or not 0 < cfg.beta < 1 or not 0 < cfg.first_alpha_weight <= 1:
            raise ValueError("Invalid gate/beta/first-layer weights")
        for field in fields(cls):
            if field.name.endswith("weight") and (not math.isfinite(float(getattr(cfg, field.name))) or getattr(cfg, field.name) < 0):
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if cfg.consistency_weight and not cfg.structure_enabled:
            raise ValueError("Consistency requires structure_enabled")
        return cfg


class RecoveryLayer(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.q_norm = nn.LayerNorm(channels)
        self.x_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, int(channels * cfg.feedforward_ratio)), nn.GELU(), nn.Linear(int(channels * cfg.feedforward_ratio), channels))
        self.alpha = nn.Sequential(nn.Linear(2 * channels + 1, max(16, channels // 4)), nn.GELU(), nn.Linear(max(16, channels // 4), 1))
        for layer in (self.out, self.ffn[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.alpha[-1].weight)
        nn.init.constant_(self.alpha[-1].bias, math.log(cfg.initial_gate / (1 - cfg.initial_gate)))

    def forward(self, state, original, previous_alpha, *, memory=None, key_position=None, forbidden=None, reliability=None):
        b, n, c = original.shape
        normalized = self.x_norm(original if memory is None else memory)
        def split(value):
            return value.reshape(b, value.shape[1], self.heads, c // self.heads).transpose(1, 2)
        # Explicit FP32 attention: diagonal is excluded and n=1 has no context.
        with torch.autocast(device_type=original.device.type, enabled=False):
            keys = self.k(normalized.float())
            if key_position is not None:
                keys = keys + key_position.float()
            logits = split(self.q(self.q_norm(state.float()))) @ split(keys).transpose(-1, -2) / math.sqrt(c // self.heads)
            if reliability is not None:
                logits = logits + reliability.float().clamp_min(1e-6).log()[:, None, None, :]
            elif previous_alpha is not None:
                logits = logits + (1 - previous_alpha.float()).clamp_min(1e-6).log()[:, None, None, :]
            if forbidden is not None:
                valid_row = (~forbidden).any(-1)[None, None, :, None]
                logits = logits.masked_fill(forbidden[None, None], -torch.inf)
                attention = torch.where(valid_row, logits, torch.zeros_like(logits)).softmax(-1)
                attention = attention.masked_fill(forbidden[None, None], 0)
            elif n == 1:
                attention = torch.zeros_like(logits)
            else:
                logits = logits.masked_fill(torch.eye(n, device=state.device, dtype=torch.bool)[None, None], -torch.inf)
                attention = logits.softmax(-1)
            context = (attention @ split(self.v(normalized.float()))).transpose(1, 2).reshape(b, n, c)
        state = state + self.out(context.to(state.dtype))
        state = state + self.ffn(self.ffn_norm(state))
        alpha_input = torch.cat((F.layer_norm(original, (c,)), F.layer_norm(state, (c,)), (1 - F.cosine_similarity(original.float(), state.float(), dim=-1))[..., None].to(state.dtype)), -1)
        return state, self.alpha(alpha_input).squeeze(-1).sigmoid(), attention


class StructureContext(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.structure_dim
        self.project = nn.Linear(channels, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attention = nn.MultiheadAttention(d, cfg.structure_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.prototypes = nn.Parameter(torch.randn(cfg.prototypes, d) * 0.02)
        self.descriptor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 20))

    def forward(self, tokens, position):
        memory = self.project(tokens + position)
        context, _ = self.attention(self.query.expand(len(tokens), -1, -1), memory, memory, need_weights=False)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            logits = F.normalize(self.norm(context[:, 0].float()), dim=-1) @ F.normalize(self.prototypes.float(), dim=-1).T / self.cfg.temperature
            pi = logits.softmax(-1)
            z = (pi @ self.prototypes.float()).to(tokens.dtype)
        return z, pi, self.descriptor(z).reshape(-1, 10, 2)


class ContinuousRecoveryBlock(nn.Module):
    def __init__(self, embed_dim: int, config):
        super().__init__()
        self.config = config if isinstance(config, ContinuousRecoveryConfig) else ContinuousRecoveryConfig.from_mapping(config, embed_dim=embed_dim)
        cfg = self.config
        self.embed_dim = embed_dim
        self.messenger_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.layers = nn.ModuleList([RecoveryLayer(embed_dim, cfg) for _ in range(cfg.layers)])
        self.decoder = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        gate_dim = 3 * embed_dim + 2
        self.local_gate = None
        if cfg.adoption != "alpha":
            self.local_gate = nn.Sequential(nn.Linear(gate_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 1))
            nn.init.zeros_(self.local_gate[-1].weight)
            nn.init.constant_(self.local_gate[-1].bias, math.log(cfg.initial_gate / (1 - cfg.initial_gate)))
        self.structure = StructureContext(embed_dim, cfg) if cfg.structure_enabled else None
        self.structure_gate = None
        if self.structure is not None:
            self.structure_gate = nn.Sequential(nn.Linear(gate_dim + cfg.structure_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 1))
            nn.init.zeros_(self.structure_gate[-1].weight)
            nn.init.zeros_(self.structure_gate[-1].bias)
        self.register_buffer("channel_scale", torch.ones(embed_dim))
        self.register_buffer("scale_ready", torch.tensor(False))
        # Training schedule only; inference always enables the learned mixture.
        self.mixing_enabled = True
        self.structure_ablation = None  # validation diagnostic only, never checkpointed
        self.reference_memory = None
        if cfg.multi_scale or cfg.multi_level:
            from .recovery_reference import RecoveryReferenceMemory
            # Preserve common-module initialization AND the caller's RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.reference_adapter_seed)
                self.reference_memory = RecoveryReferenceMemory(embed_dim, cfg)

    def forward(self, feature_map, *, reference_maps=None, return_diagnostics=False, structure_ablation=None):
        structure_ablation = structure_ablation or self.structure_ablation
        if structure_ablation not in {None, "zero", "shuffle"}:
            raise ValueError("Unknown structure diagnostic")
        b, c, h, w = feature_map.shape
        x = feature_map.flatten(2).transpose(1, 2)
        position = sine_position_encoding_2d(h, w, c, device=x.device, dtype=x.dtype)
        state = self.messenger_token.to(x.dtype).expand(b, h * w, -1) + position
        previous = None
        alphas, attentions = [], []
        reference = None
        if self.reference_memory is not None:
            reference = self.reference_memory(feature_map, reference_maps)
        for layer in self.layers:
            if reference is None:
                state, previous, attention = layer(state, x, previous)
            else:
                reliability = self.reference_memory.reliability(previous, h, w)
                state, previous, attention = layer(state, x, previous, memory=reference["tokens"],
                    key_position=reference["key_position"], forbidden=reference["forbidden"], reliability=reliability)
            alphas.append(previous)
            if return_diagnostics:
                attentions.append(attention)
        candidate = self.decoder(state)
        u = torch.cat((F.layer_norm(x, (c,)), F.layer_norm(candidate, (c,)), (1 - F.cosine_similarity(x.float(), candidate.float(), dim=-1))[..., None].to(x.dtype), previous[..., None], position.expand(b, -1, -1)), -1)
        diagnostics = {}
        if self.local_gate is None:
            gate = previous
        else:
            local_logit = self.local_gate(u).squeeze(-1)
            correction = torch.zeros_like(local_logit)
            if self.structure is not None:
                z, pi, descriptor = self.structure(x, position)
                if structure_ablation == "shuffle":
                    z = z.roll(1, 0)
                correction = self.structure_gate(torch.cat((u, z[:, None].expand(-1, h * w, -1)), -1)).squeeze(-1)
                if structure_ablation == "zero":
                    correction = correction * 0
                prototypes = F.normalize(self.structure.prototypes.float(), dim=-1)
                diagnostics.update(structure_pi=pi, structure_descriptor=descriptor, structure_correction=correction,
                                   prototype_similarity=(prototypes @ prototypes.T).expand(b, -1, -1))
            gate = (local_logit + correction).sigmoid()
            diagnostics["local_gate"] = local_logit.sigmoid()
        effective_gate = gate if (not self.training or self.mixing_enabled) else gate * 0
        y = x + effective_gate[..., None] * (candidate - x)
        diagnostics.update(original_tokens=x, messenger_tokens=state, candidate_tokens=candidate, recovered_tokens=y,
                           gate_alpha=previous, layer_alpha=torch.stack(alphas, 1), adoption_gate=gate)
        if return_diagnostics:
            diagnostics["attention"] = torch.stack(attentions, 1)
            if reference is not None:
                diagnostics["reference_metadata"] = reference["metadata"]
                diagnostics["reference_forbidden"] = reference["forbidden"]
                diagnostics["reference_groups"] = reference["groups"]
                diagnostics["reference_attention_mass"] = torch.stack([
                    diagnostics["attention"][..., group["start"]:group["stop"]].sum(-1).mean((-1, -2))
                    for group in reference["groups"]], -1)
        return y.transpose(1, 2).reshape(b, c, h, w).contiguous(), diagnostics


def oracle_adoption(x, candidate, clean, scale, epsilon):
    """Detached least-squares oracle in fixed teacher-standardized coordinates.

    mean squared candidate displacement <= epsilon chooses raw (g*=0).
    """
    with torch.no_grad():
        delta = (candidate.float() - x.float()) / scale.float()
        target = (clean.float() - x.float()) / scale.float()
        denominator = delta.square().mean(-1)
        oracle = (target * delta).mean(-1) / denominator.clamp_min(epsilon)
        return torch.where(denominator > epsilon, oracle.clamp(0, 1), torch.zeros_like(oracle))


def balanced_alpha_loss(alpha, ratio):
    groups = [mask for mask in (ratio > 0, ratio == 0) if mask.any()]
    return torch.stack([(alpha[mask] - ratio[mask]).square().mean() for mask in groups]).mean()


class ContinuousRecoveryLoss(nn.Module):
    def __init__(self, config, *, patch_size=16):
        super().__init__()
        self.config = config
        self.patch_size = patch_size

    def forward(self, diagnostics, clean_map, pixel_mask, *, channel_scale, targets=None, clean_pi=None, clean_descriptor=None):
        cfg = self.config
        x, r, y = [diagnostics[key].float() for key in ("original_tokens", "candidate_tokens", "recovered_tokens")]
        z = clean_map.detach().float().flatten(2).transpose(1, 2)
        ratios = patch_occlusion_ratios(pixel_mask, patch_size=self.patch_size).to(x.device)
        weights = ratios + cfg.beta * (1 - ratios)
        scale = channel_scale.detach().float().clamp_min(cfg.scale_floor)
        zero = y.sum() * 0
        def reconstruction(value):
            # torch Huber (not SmoothL1): exactly the design's delta scaling.
            amplitude = F.huber_loss((value - z) / scale, torch.zeros_like(z), delta=cfg.huber_delta, reduction="none").mean(-1)
            error = cfg.cosine_weight * (1 - F.cosine_similarity(value, z, dim=-1)) + cfg.amplitude_weight * amplitude
            return (weights * error).sum() / weights.sum().clamp_min(1e-8)
        candidate_loss, output_loss = reconstruction(r), reconstruction(y)
        # A diagnostic only when disabled: no direct alpha teacher remains in autograd.
        alpha_loss = sum((cfg.first_alpha_weight if i == 0 else 1.) * balanced_alpha_loss(
            a if cfg.alpha_weight else a.detach(), ratios)
            for i, a in enumerate(diagnostics["layer_alpha"].unbind(1)))
        gram_loss = zero
        if cfg.gram_weight:
            with torch.autocast(device_type=y.device.type, enabled=False):
                yn, zn = F.normalize(y, dim=-1), F.normalize(z, dim=-1)
                gram_loss = ((yn @ yn.transpose(1, 2)) - (zn @ zn.transpose(1, 2))).square().mean()
        adoption_loss = zero
        if cfg.adoption == "optimal":
            oracle = oracle_adoption(x, r, z, scale, cfg.oracle_epsilon)
            adoption_loss = F.mse_loss(diagnostics["adoption_gate"].float(), oracle)
        descriptor_loss, consistency_loss = zero, zero
        if cfg.structure_enabled:
            if targets is None:
                raise ValueError("Structure supervision needs train descriptor targets")
            truth = torch.stack([t["structure_pairs"] for t in targets]).to(y.device).detach()
            valid = torch.stack([t["structure_pairs_valid"] for t in targets]).to(y.device)
            error = F.huber_loss(diagnostics["structure_descriptor"].float(), torch.nan_to_num(truth), delta=cfg.huber_delta, reduction="none").mean(-1)
            eligible = valid.any(-1)
            if eligible.any():
                descriptor_loss = ((error * valid).sum(-1) / valid.sum(-1).clamp_min(1))[eligible].mean()
                if clean_descriptor is not None:
                    clean_error = F.huber_loss(clean_descriptor.float(), torch.nan_to_num(truth), delta=cfg.huber_delta, reduction="none").mean(-1)
                    descriptor_loss = 0.5 * (descriptor_loss + ((clean_error * valid).sum(-1) / valid.sum(-1).clamp_min(1))[eligible].mean())
            if cfg.consistency_weight:
                if clean_pi is None:
                    raise ValueError("Consistency requires a detached clean assignment")
                clean_pi = clean_pi.detach().float()
                kl = (clean_pi * (clean_pi.clamp_min(1e-8).log() - diagnostics["structure_pi"].float().clamp_min(1e-8).log())).sum(-1)
                consistency_loss = (kl * (1 - ratios.mean(-1)).square()).mean()
        total = cfg.candidate_weight * candidate_loss + cfg.output_weight * output_loss + cfg.alpha_weight * alpha_loss + cfg.gram_weight * gram_loss + cfg.adoption_weight * adoption_loss + cfg.descriptor_weight * descriptor_loss + cfg.consistency_weight * consistency_loss
        result = dict(loss_messenger_total=total, loss_recovery_candidate=candidate_loss, loss_recovery_output=output_loss,
                      loss_recovery_alpha=alpha_loss, loss_recovery_gram=gram_loss, loss_recovery_adoption=adoption_loss,
                      loss_structure_descriptor=descriptor_loss, loss_structure_consistency=consistency_loss,
                      recovery_feature_norm=y.norm(dim=-1).mean(), recovery_adoption_mean=diagnostics["adoption_gate"].mean())
        for i, alpha in enumerate(diagnostics["layer_alpha"].unbind(1)):
            result[f"recovery_alpha_{i+1}_mse"] = (alpha.detach().float() - ratios).square().mean()
        result["recovery_alpha_direct_weight"] = y.new_tensor(cfg.alpha_weight)
        if cfg.structure_enabled:
            pi = diagnostics["structure_pi"].float()
            result.update(structure_entropy=-(pi * pi.clamp_min(1e-8).log()).sum(-1).mean(),
                          structure_variation=pi.var(0, unbiased=False).mean(),
                          structure_correction_abs=diagnostics["structure_correction"].abs().mean(),
                          structure_gate_change=(diagnostics["adoption_gate"] - diagnostics["local_gate"]).abs().mean())
            for i, usage in enumerate(pi.mean(0)):
                result[f"structure_mode_{i}_usage"] = usage
            sim = diagnostics["prototype_similarity"][0]
            result["structure_prototype_similarity"] = sim[~torch.eye(len(sim), dtype=torch.bool, device=sim.device)].mean()
        return result


def relative_structure_targets(keypoints, labeled, inverse_affine, input_size, head_side):
    """Crop-axis x-right/y-down pairs, in image pixels / GT Head side.

    Dataset keypoints already share crop rotation and flip (semantic 1<->0,
    4<->3). Isotropic inverse-affine scale undoes resize, not orientation.
    """
    pairs = torch.tensor(list(combinations(range(5), 2)), device=keypoints.device)
    scale = torch.linalg.vector_norm(inverse_affine[:2, 0]) * (input_size - 1) / head_side
    values = (keypoints[pairs[:, 0]] - keypoints[pairs[:, 1]]) * scale
    valid_points = labeled.bool() & torch.isfinite(keypoints).all(-1)
    valid = valid_points[pairs[:, 0]] & valid_points[pairs[:, 1]]
    return torch.nan_to_num(values), valid
