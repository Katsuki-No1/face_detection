"""Opt-in geometry-conditioned Messenger research components.

No existing model/trainer imports this module. The prior is fitted/calibrated on
disjoint TRAIN videos in a frozen backbone's coordinates; the hard anomaly mask
is intentionally non-differentiable. The student path remains differentiable.
"""
from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from .continuous_recovery import ContinuousRecoveryBlock


def backbone_state_sha256(backbone):
    """Authenticate the actual tensor coordinate system, not only a path label."""
    digest = hashlib.sha256()
    for name, value in sorted(backbone.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def geometry_embedding(geometry: torch.Tensor) -> torch.Tensor:
    """[relative major diameter, major/minor >= 1, image-axis radians].

    Scale = full predicted major diameter / sqrt(source width * source height).
    x points right, y down; angle is clockwise modulo pi, after augmentation.
    Circular ellipses contribute zero orientation. No GT Head inference input.
    """
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError("geometry must be Bx3: relative scale, ratio, radians")
    if not torch.isfinite(geometry).all() or (geometry[:, 0] <= 0).any() or (geometry[:, 1] < 1).any():
        raise ValueError("Invalid predicted ellipse geometry")
    scale, ratio, angle = geometry.float().unbind(-1)
    eccentricity = 1 - ratio.reciprocal()
    return torch.stack((scale.log(), ratio.log(), eccentricity * (2 * angle).sin(),
                        eccentricity * (2 * angle).cos()), -1)


def _features(value):
    if value.ndim != 4 or min(value.shape) < 1 or value.shape[1] < 2:
        raise ValueError("Features must be nonempty BxCxHxW with C >= 2")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite features")
    return value.detach().float()


@torch.no_grad()
def _kmeans(values, k, seed, iterations=100):
    """Deterministic k-means++/Lloyd; reject empty/collapsed requested clusters."""
    if not isinstance(k, int) or not 1 <= k <= 7 or len(values) < k * 2:
        raise ValueError("k must be 1..7 with at least 2 fit examples per requested center")
    rng = torch.Generator(device=values.device).manual_seed(seed)
    centers = [values[torch.randint(len(values), (), generator=rng, device=values.device)]]
    for _ in range(1, k):
        distance = torch.cdist(values, torch.stack(centers)).square().min(1).values
        if distance.sum() <= 1e-10:
            raise ValueError("Requested k exceeds distinct feature support")
        centers.append(values[torch.multinomial(distance, 1, generator=rng).item()])
    centers = torch.stack(centers)
    previous = None
    for _ in range(iterations):
        labels = torch.cdist(values, centers).argmin(1)
        if any((labels == j).sum() < 2 for j in range(k)):
            raise ValueError("Under-supported cluster; do not silently reduce k")
        centers = torch.stack([values[labels == j].mean(0) for j in range(k)])
        if previous is not None and torch.equal(labels, previous):
            return labels
        previous = labels
    raise ValueError("k-means did not converge within the declared iteration budget")


def feature_similarity(features, center, channel_scale):
    """Per-position channel Pearson correlation / (1 + standardized mean L1).

    Zero channel variance has no correlation evidence and returns correlation 0.
    This is a similarity score, not an occlusion probability.
    """
    x = features.float() / channel_scale[None, :, None, None]
    c = center.float() / channel_scale[None, :, None, None]
    xc, cc = x - x.mean(1, keepdim=True), c - c.mean(1, keepdim=True)
    correlation = F.cosine_similarity(xc, cc, dim=1, eps=1e-6)
    return correlation / (1 + (x - c).abs().mean(1))


class GeometryPriorBank(nn.Module):
    """Fixed spatial centers with geometry routing and held-out TRAIN calibration."""

    def __init__(self, payload):
        super().__init__()
        if payload.get("schema") != "geometry-prior-v1":
            raise ValueError("Unsupported prior schema")
        self.metadata = copy.deepcopy(payload["metadata"])
        self.quantile = float(payload["quantile"])
        self.temperature = float(payload["temperature"])
        self._validate_metadata(self.metadata)
        if not 0 < self.quantile < .5 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid calibration quantile or geometry temperature")
        for name in ("centers", "channel_scale", "geometry_mean", "geometry_std",
                     "geometry_centers", "thresholds", "calibration_counts"):
            value = payload[name].detach().clone()
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite prior buffer: {name}")
            self.register_buffer(name, value)
        k, c, h, w = self.centers.shape
        if not 1 <= k <= 7 or self.thresholds.shape != (k, h, w) or self.channel_scale.shape != (c,):
            raise ValueError("Prior feature/threshold shape mismatch")
        if self.geometry_centers.shape != (k, 4) or self.geometry_mean.shape != (4,) or self.geometry_std.shape != (4,):
            raise ValueError("Prior geometry shape mismatch")
        if self.calibration_counts.shape != (k,) or (self.thresholds.abs() > 1.00001).any():
            raise ValueError("Invalid calibration counts or similarity thresholds")
        if (self.channel_scale <= 0).any() or (self.geometry_std <= 0).any() or (self.calibration_counts < 2).any():
            raise ValueError("Invalid normalization or unsupported calibration cluster")

    @staticmethod
    def _validate_metadata(metadata):
        required = ("split", "annotation_sha256", "backbone_sha256", "backbone_state_sha256", "detector_sha256",
                    "crop_convention", "clean_eligibility", "fit_video_ids", "calibration_video_ids", "seed")
        if any(key not in metadata for key in required) or metadata["split"] != "train":
            raise ValueError("Prior requires explicit train-only provenance")
        for key in ("annotation_sha256", "backbone_sha256", "backbone_state_sha256", "detector_sha256"):
            digest = metadata[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
                raise ValueError(f"Invalid {key}")
        fit, calibration = set(metadata["fit_video_ids"]), set(metadata["calibration_video_ids"])
        if not fit or not calibration or fit & calibration:
            raise ValueError("Fit and calibration must use disjoint nonempty train video groups")
        if not metadata["crop_convention"] or not metadata["clean_eligibility"]:
            raise ValueError("Explicit crop/clean eligibility audit required")

    @classmethod
    @torch.no_grad()
    def fit(cls, features, geometry, calibration_features, calibration_geometry, *,
            k, metadata, quantile=.05, temperature=1.):
        cls._validate_metadata(metadata)
        if not 0 < quantile < .5 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid calibration settings")
        # CPU-only fitting is bounded by caller cache size and never takes the training GPU.
        x, calibration = _features(features).cpu(), _features(calibration_features).cpu()
        if x.shape[1:] != calibration.shape[1:]:
            raise ValueError("Fit/calibration feature coordinate systems must match")
        g, cg = geometry_embedding(geometry.detach().cpu()), geometry_embedding(calibration_geometry.detach().cpu())
        if len(g) != len(x) or len(cg) != len(calibration):
            raise ValueError("Feature/geometry inventory mismatch")
        scale = x.permute(1, 0, 2, 3).flatten(1).std(1, unbiased=False).clamp_min(.05)
        normalized = (x / scale[None, :, None, None]).flatten(1)
        labels = _kmeans(F.normalize(normalized, dim=1), k, int(metadata["seed"]))
        centers = torch.stack([x[labels == j].mean(0) for j in range(k)])
        mean, std = g.mean(0), g.std(0, unbiased=False).clamp_min(.1)
        gc = torch.stack([((g[labels == j] - mean) / std).mean(0) for j in range(k)])
        calibration_labels = torch.cdist((cg - mean) / std, gc).argmin(1)
        counts = torch.bincount(calibration_labels, minlength=k)
        if (counts < 2).any():
            raise ValueError("Geometry routing has an unsupported calibration cluster; report failure, not success")
        thresholds = torch.stack([torch.quantile(feature_similarity(calibration[calibration_labels == j],
            centers[j:j+1], scale), quantile, dim=0) for j in range(k)])
        return cls(dict(schema="geometry-prior-v1", metadata=metadata, quantile=quantile,
                        temperature=temperature, centers=centers, channel_scale=scale,
                        geometry_mean=mean, geometry_std=std, geometry_centers=gc,
                        thresholds=thresholds, calibration_counts=counts))

    def get_extra_state(self):
        return dict(metadata=self.metadata, quantile=self.quantile, temperature=self.temperature)

    def payload(self):
        return dict(schema="geometry-prior-v1", **copy.deepcopy(self.get_extra_state()),
                    **{name: value.detach().cpu().clone() for name, value in self.named_buffers()})

    def set_extra_state(self, state):
        self._validate_metadata(state["metadata"])
        # A different bank is a different experiment, never a silent resume.
        if state != self.get_extra_state():
            raise ValueError("Prior provenance/calibration differs from checkpoint")

    @torch.no_grad()
    def forward(self, features, geometry, support):
        x = _features(features)
        if x.shape[1:] != self.centers.shape[1:]:
            raise ValueError("Prior grid/channels differ from observed features")
        if support.shape != (len(x), *x.shape[2:]) or support.dtype != torch.bool or not support.flatten(1).any(1).all():
            raise ValueError("Nonempty boolean predicted-ellipse support required per face")
        embedding = geometry_embedding(geometry)
        if len(embedding) != len(x):
            raise ValueError("Geometry batch differs from features")
        distance = torch.cdist((embedding - self.geometry_mean) / self.geometry_std, self.geometry_centers).square()
        weights = (-distance / self.temperature).softmax(-1)
        # v1 uses nearest GEOMETRY center, not feature nearest-neighbor routing.
        # Keep soft weights for diagnostics; do not average incompatible pose maps.
        selected = distance.argmin(1)
        score = feature_similarity(x, self.centers[selected], self.channel_scale)
        anomalous = (score < self.thresholds[selected]) & support
        return dict(score=score, anomalous=anomalous, selected_center=selected,
                    center_weights=weights, geometry_embedding=embedding)


@torch.no_grad()
def connected_occlusion(anomalous, support, minimum_fraction=.05):
    """4-connected components; face flag at >= fraction of ellipse-support cells.

    Local fusion retains all anomalous cells even when the face flag is false.
    CPU traversal is deliberate for this opt-in prototype; no production claim.
    """
    if not 0 < minimum_fraction <= 1 or anomalous.shape != support.shape or anomalous.ndim != 3:
        raise ValueError("Invalid component fraction or mask shape")
    if anomalous.dtype != torch.bool or support.dtype != torch.bool:
        raise ValueError("Component masks must be boolean")
    raw = (anomalous & support).cpu()
    totals = support.flatten(1).sum(1).cpu()
    if (totals == 0).any():
        raise ValueError("Empty ellipse support")
    largest, accepted = [], torch.zeros_like(raw)
    for b, mask in enumerate(raw):
        visited, maximum = set(), 0
        h, w = mask.shape
        for y, x in mask.nonzero().tolist():
            if (y, x) in visited:
                continue
            stack, component = [(y, x)], []
            visited.add((y, x))
            while stack:
                py, px = stack.pop()
                component.append((py, px))
                for ny, nx in ((py-1, px), (py+1, px), (py, px-1), (py, px+1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited and mask[ny, nx]:
                        visited.add((ny, nx))
                        stack.append((ny, nx))
            maximum = max(maximum, len(component))
            if len(component) / int(totals[b]) >= minimum_fraction:
                for py, px in component:
                    accepted[b, py, px] = True
        largest.append(maximum / int(totals[b]))
    fraction = torch.tensor(largest, device=anomalous.device)
    return dict(face_occluded=fraction >= minimum_fraction, largest_component_fraction=fraction,
                large_component_mask=accepted.to(anomalous.device))


class PriorMessengerModel(nn.Module):
    """Compose an existing crop DINO/Messenger/Query Refinement without modifying it.

    Prior observation is from a separately frozen copy of the authenticated
    backbone. Student updates cannot invalidate the fixed prior coordinates.
    Disabled construction adds no parameters: return the original model instead.
    """

    def __init__(self, base_model, bank, *, observation_backbone_sha256, minimum_fraction=.05, fusion_mode="prior"):
        super().__init__()
        block = base_model.messenger_recovery
        if base_model.variant != "query_refinement" or not isinstance(block, ContinuousRecoveryBlock):
            raise ValueError("Requires existing continuous Messenger and Query Refinement")
        if base_model.intermediate_layers or block.config.multi_level or block.config.multi_scale or base_model.face_overlap_enabled:
            raise ValueError("v1 supports final-level single-scale Messenger without FOR only")
        if observation_backbone_sha256 != bank.metadata["backbone_sha256"]:
            raise ValueError("Frozen observer/prior checkpoint identity differs")
        if backbone_state_sha256(base_model.backbone) != bank.metadata["backbone_state_sha256"]:
            raise ValueError("Actual observer tensor state differs from cached feature coordinates")
        if fusion_mode not in {"prior", "messenger"}:
            raise ValueError("fusion_mode must be prior or matched messenger control")
        if not 0 < minimum_fraction <= 1:
            raise ValueError("minimum_fraction must be in (0,1]")
        self.base = base_model
        self.bank = bank
        self.observer = copy.deepcopy(base_model.backbone).requires_grad_(False).eval()
        self.minimum_fraction = minimum_fraction
        self.fusion_mode = fusion_mode
        self.channel_adoption = nn.Parameter(torch.zeros(block.embed_dim))

    def get_extra_state(self):
        return dict(schema="prior-messenger-v1", fusion_mode=self.fusion_mode,
                    minimum_fraction=self.minimum_fraction,
                    parent_recovery_config=vars(self.base.messenger_recovery.config).copy())

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Refusing a different fusion/area/parent recovery checkpoint contract")

    def train(self, mode=True):
        super().train(mode)
        self.observer.eval()
        return self

    def forward(self, images, *, geometry, support, return_attention=False, augmentation_valid_mask=None):
        # Bank calibration was extracted in FP32. Student autocast must not
        # silently move the fixed observer into a different numeric system.
        with torch.no_grad(), torch.autocast(images.device.type, enabled=False):
            observation = self.observer(images.float())
            prior = self.bank(observation, geometry, support)
        original = self.base.backbone(images)
        existing_fused, diagnostics = self.base.messenger_recovery(original)
        b, c, h, w = original.shape
        candidate = diagnostics["candidate_tokens"].transpose(1, 2).reshape(b, c, h, w)
        gate = diagnostics["adoption_gate"].reshape(b, 1, h, w)
        # A bounded channel parameter adjusts adoption; non-anomalous values are exact identities.
        channel = self.channel_adoption.sigmoid()[None, :, None, None]
        mask = prior["anomalous"][:, None].to(original.dtype)
        if self.training and not self.base.messenger_recovery.mixing_enabled:
            gate = gate * 0
        fused = original + mask * gate * channel * (candidate - original)
        if self.fusion_mode == "messenger":
            fused = existing_fused
        refinement = self.base.rgb_refinement(images) if self.base.rgb_refinement is not None else None
        output = self.base.head(fused, return_attention=return_attention,
                               augmentation_valid_mask=augmentation_valid_mask, refinement_feature_maps=refinement)
        output["feature_map_shape"] = tuple(original.shape)
        output["prior_diagnostics"] = dict(prior, **connected_occlusion(prior["anomalous"], support, self.minimum_fraction),
                                           original=original, candidate=candidate, fused=fused,
                                           gate=gate, observation=observation)
        return output


class ConditionalFeatureDiscriminator(nn.Module):
    """Training-only spatial discriminator conditioned on geometry and visible context."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2 * channels + 5, hidden, 3, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(.2), nn.Conv2d(hidden, 1, 1))

    def forward(self, features, observation, mask, geometry):
        if features.shape != observation.shape or mask.shape != (len(features), 1, *features.shape[2:]):
            raise ValueError("Discriminator feature/context/mask mismatch")
        condition = geometry_embedding(geometry).detach()[:, :, None, None].expand(-1, -1, *features.shape[2:])
        # Same masked context for real/fake: no clean target or hidden pixels in the condition.
        context = observation.detach() * (1 - mask.detach())
        return self.net(torch.cat((features, context, mask.detach(), condition), 1))


def _masked_mean(value, mask):
    expanded = mask.expand_as(value).to(value.dtype)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


@contextmanager
def frozen_discriminator(discriminator):
    flags = [p.requires_grad for p in discriminator.parameters()]
    try:
        discriminator.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)


def discriminator_loss(discriminator, output, clean_features, geometry):
    """D update only; no gradients to student, generator, observer or clean teacher."""
    d = output["prior_diagnostics"]
    fake, clean = d["fused"].detach(), clean_features.detach()
    mask = d["anomalous"][:, None].to(fake.dtype)
    if clean.shape != fake.shape:
        raise ValueError("Clean teacher/fused shape mismatch")
    real_logits = discriminator(clean, d["observation"], mask, geometry)
    fake_logits = discriminator(fake, d["observation"], mask, geometry)
    return _masked_mean(F.softplus(-real_logits) + F.softplus(fake_logits), mask)


def generator_losses(output, clean_features, keypoint_loss, *, discriminator=None,
                     geometry=None, reconstruction_weight=.1, adversarial_weight=0.):
    """Task + paired reconstruction + optional adversarial objective.

    keypoint_loss is the caller's unchanged LandmarkLoss total, including state.
    Supply same-target, same-crop clean features from the fixed teacher, not centers.
    D has its own optimizer; it is never updated by the keypoint objective.
    """
    if any(not math.isfinite(w) or w < 0 for w in (reconstruction_weight, adversarial_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    d = output["prior_diagnostics"]
    clean = clean_features.detach().float()
    if clean.shape != d["fused"].shape or not torch.isfinite(clean).all():
        raise ValueError("Invalid clean teacher map")
    mask = d["anomalous"][:, None].to(clean.dtype)
    # Candidate supervision also trains recovery when the detector misses occlusion.
    candidate = F.smooth_l1_loss(d["candidate"].float(), clean)
    fused = F.smooth_l1_loss(d["fused"].float(), clean)
    adversarial = d["fused"].sum() * 0
    if adversarial_weight:
        if discriminator is None or geometry is None:
            raise ValueError("Adversarial loss requires D and predicted geometry")
        with frozen_discriminator(discriminator):
            logits = discriminator(d["fused"], d["observation"], mask, geometry)
            adversarial = _masked_mean(F.softplus(-logits), mask)
    total = keypoint_loss + reconstruction_weight * (candidate + fused) + adversarial_weight * adversarial
    return dict(total=total, keypoint=keypoint_loss, candidate=candidate, fused=fused, adversarial=adversarial)


def attach_geometry_prior(base_model, bank=None, *, enabled=False, **kwargs):
    """Disabled means the identical object, state_dict and inference API."""
    if not enabled:
        return base_model
    if bank is None:
        raise ValueError("An authenticated fitted bank is required")
    return PriorMessengerModel(base_model, bank, **kwargs)


def research_train_step(model, images, clean_images, geometry, support, targets,
                        criterion, generator_optimizer, *, discriminator=None,
                        discriminator_optimizer=None, reconstruction_weight=.1,
                        adversarial_weight=0.):
    """One FP32 research update, separate G/D optimizers, no production launcher.

    Production accumulation/AMP/OOM fallback and authenticated four-gate wiring
    must be supplied by a future isolated runner, not the active training queue.
    This function deliberately rejects autocast to avoid pretending AMP support.
    """
    if torch.is_autocast_enabled(images.device.type):
        raise ValueError("research_train_step is FP32-only")
    if images.shape != clean_images.shape:
        raise ValueError("Clean/occluded images must share the crop geometry")
    if not math.isfinite(adversarial_weight) or adversarial_weight < 0:
        raise ValueError("Invalid adversarial weight")
    if not math.isfinite(reconstruction_weight) or reconstruction_weight < 0:
        raise ValueError("Invalid reconstruction weight")
    if model.fusion_mode == "messenger" and adversarial_weight:
        raise ValueError("The initial matched Messenger control has adversarial weight zero")
    generator_ids = [id(p) for group in generator_optimizer.param_groups for p in group["params"]]
    generator_params = set(generator_ids)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if generator_params != expected or len(generator_ids) != len(generator_params):
        raise ValueError("Generator optimizer must contain exactly trainable model parameters")
    if adversarial_weight:
        if discriminator is None or discriminator_optimizer is None:
            raise ValueError("Adversarial updates require a separate discriminator optimizer")
        d_list = [id(p) for group in discriminator_optimizer.param_groups for p in group["params"]]
        d_ids = set(d_list)
        if d_ids & generator_params or len(d_ids) != len(d_list) or d_ids != {id(p) for p in discriminator.parameters() if p.requires_grad}:
            raise ValueError("Discriminator optimizer ownership mismatch")
    model.train()
    generator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        clean = model.observer(clean_images)
    output = model(images, geometry=geometry, support=support)
    d_value = output["prior_diagnostics"]["fused"].new_zeros(())

    def backward_and_check(loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite objective; optimizer was not stepped")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group["params"]):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite gradients; optimizer was not stepped")

    if adversarial_weight:
        discriminator.train()
        discriminator_optimizer.zero_grad(set_to_none=True)
        d_value = discriminator_loss(discriminator, output, clean, geometry)
        if output["prior_diagnostics"]["anomalous"].any():
            backward_and_check(d_value, discriminator_optimizer)
            discriminator_optimizer.step()
        discriminator_optimizer.zero_grad(set_to_none=True)
    task = criterion(output, targets)["loss_total"]
    losses = generator_losses(output, clean, task, discriminator=discriminator, geometry=geometry,
                              reconstruction_weight=reconstruction_weight, adversarial_weight=adversarial_weight)
    backward_and_check(losses["total"], generator_optimizer)
    generator_optimizer.step()
    return {name: float(value.detach()) for name, value in dict(losses, discriminator=d_value).items()}
