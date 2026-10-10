from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


SUPPORTED_QUERY_ERROR_MODES = ("jitter", "miss", "swap")
DEFAULT_SWAP_PAIRS = ((0, 1), (3, 4))


@dataclass(frozen=True)
class QueryErrorAugmentationConfig:
    """Training-only corruption of learned spatial-query reference points.

    The corruption is applied before the first Transformer decoder layer. GT
    coordinates are never used as query inputs and labels are never changed.
    """

    enabled: bool = False
    modes: tuple[str, ...] = ()
    jitter_probability: float = 0.50
    jitter_std: float = 0.02
    jitter_maximum: float = 0.05
    miss_probability: float = 0.50
    miss_radius_minimum: float = 0.08
    miss_radius_maximum: float = 0.12
    swap_probability: float = 0.50
    swap_pairs: tuple[tuple[int, int], ...] = DEFAULT_SWAP_PAIRS

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        num_landmarks: int = 5,
    ) -> "QueryErrorAugmentationConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError(
                "model.variants.query_refinement.input_keypoint_error_augmentation "
                "must be a mapping"
            )
        modes_value = value.get("modes", ())
        if isinstance(modes_value, str) or not isinstance(modes_value, Sequence):
            raise ValueError("input keypoint error augmentation modes must be a list")
        modes = tuple(str(mode).strip().lower() for mode in modes_value)
        if len(modes) != len(set(modes)):
            raise ValueError("input keypoint error augmentation modes must be unique")
        unsupported = sorted(set(modes).difference(SUPPORTED_QUERY_ERROR_MODES))
        if unsupported:
            raise ValueError(
                "unsupported input keypoint error augmentation modes: "
                f"{unsupported}; expected {SUPPORTED_QUERY_ERROR_MODES}"
            )

        jitter = value.get("jitter", {}) or {}
        miss = value.get("miss", {}) or {}
        swap = value.get("swap", {}) or {}
        for name, section in (("jitter", jitter), ("miss", miss), ("swap", swap)):
            if not isinstance(section, Mapping):
                raise ValueError(f"input keypoint error augmentation {name} must be a mapping")

        radius = miss.get("radius", (0.08, 0.12))
        if (
            isinstance(radius, (str, bytes))
            or not isinstance(radius, Sequence)
            or len(radius) != 2
        ):
            raise ValueError("miss.radius must contain [minimum, maximum]")
        pairs_value = swap.get("pairs", DEFAULT_SWAP_PAIRS)
        if isinstance(pairs_value, (str, bytes)) or not isinstance(pairs_value, Sequence):
            raise ValueError("swap.pairs must be a list of landmark-index pairs")
        pairs: list[tuple[int, int]] = []
        for pair in pairs_value:
            if (
                isinstance(pair, (str, bytes))
                or not isinstance(pair, Sequence)
                or len(pair) != 2
            ):
                raise ValueError("every swap pair must contain exactly two indices")
            left, right = int(pair[0]), int(pair[1])
            if left == right or min(left, right) < 0 or max(left, right) >= num_landmarks:
                raise ValueError(
                    f"invalid swap pair {(left, right)} for {num_landmarks} landmarks"
                )
            pairs.append((left, right))
        if "swap" in modes and not pairs:
            raise ValueError("swap mode requires at least one semantic pair")

        config = cls(
            enabled=bool(value.get("enabled", False)),
            modes=modes,
            jitter_probability=float(jitter.get("probability", 0.50)),
            jitter_std=float(jitter.get("std", 0.02)),
            jitter_maximum=float(jitter.get("maximum", 0.05)),
            miss_probability=float(miss.get("probability", 0.50)),
            miss_radius_minimum=float(radius[0]),
            miss_radius_maximum=float(radius[1]),
            swap_probability=float(swap.get("probability", 0.50)),
            swap_pairs=tuple(pairs),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.enabled and not self.modes:
            raise ValueError("enabled input keypoint error augmentation requires at least one mode")
        for name, probability in (
            ("jitter.probability", self.jitter_probability),
            ("miss.probability", self.miss_probability),
            ("swap.probability", self.swap_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.jitter_std <= 0.0 or self.jitter_maximum <= 0.0:
            raise ValueError("jitter std and maximum must be positive")
        if not 0.0 < self.miss_radius_minimum <= self.miss_radius_maximum:
            raise ValueError("miss radius must satisfy 0 < minimum <= maximum")
        if self.miss_radius_maximum > 1.0:
            raise ValueError("miss maximum radius must not exceed normalized coordinate range")


def _one_valid_landmark_per_selected_sample(
    valid_mask: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    batch, landmarks = valid_mask.shape
    selected_samples = torch.rand(batch, device=valid_mask.device) < probability
    selected_samples &= valid_mask.any(dim=1)
    scores = torch.rand((batch, landmarks), device=valid_mask.device)
    scores = scores.masked_fill(~valid_mask, 2.0)
    indices = scores.argmin(dim=1)
    selected = torch.zeros_like(valid_mask)
    selected.scatter_(1, indices[:, None], True)
    return selected & selected_samples[:, None]


def _straight_through(clean: torch.Tensor, corrupted: torch.Tensor) -> torch.Tensor:
    """Use corrupted forward values while preserving the clean-input gradient."""

    return clean + (corrupted - clean).detach()


def apply_query_error_augmentation(
    reference_points: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    config: QueryErrorAugmentationConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if reference_points.ndim != 3 or reference_points.shape[-1] != 2:
        raise ValueError(
            "reference_points must be BxKx2, got "
            f"{tuple(reference_points.shape)}"
        )
    batch, landmarks, _ = reference_points.shape
    if valid_mask is None:
        valid = torch.ones(
            (batch, landmarks),
            dtype=torch.bool,
            device=reference_points.device,
        )
    else:
        if valid_mask.shape != (batch, landmarks):
            raise ValueError("valid_mask must match the coordinate batch and landmark dimensions")
        valid = valid_mask.to(device=reference_points.device, dtype=torch.bool)

    corrupted_reference_points = reference_points
    masks = {
        mode: torch.zeros_like(valid)
        for mode in SUPPORTED_QUERY_ERROR_MODES
    }

    if "jitter" in config.modes:
        jitter_mask = _one_valid_landmark_per_selected_sample(
            valid,
            config.jitter_probability,
        )
        noise = torch.randn_like(corrupted_reference_points) * config.jitter_std
        noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        scale = torch.clamp(config.jitter_maximum / noise_norm, max=1.0)
        jittered = (
            corrupted_reference_points + noise * scale * jitter_mask[..., None]
        ).clamp(0.0, 1.0)
        corrupted_reference_points = _straight_through(
            corrupted_reference_points,
            jittered,
        )
        masks["jitter"] = jitter_mask

    if "miss" in config.modes:
        miss_mask = _one_valid_landmark_per_selected_sample(
            valid,
            config.miss_probability,
        )
        angle = torch.rand(
            (batch, landmarks),
            dtype=reference_points.dtype,
            device=reference_points.device,
        ) * (2.0 * math.pi)
        radius = config.miss_radius_minimum + torch.rand(
            (batch, landmarks),
            dtype=reference_points.dtype,
            device=reference_points.device,
        ) * (config.miss_radius_maximum - config.miss_radius_minimum)
        offset = torch.stack((angle.cos(), angle.sin()), dim=-1) * radius[..., None]
        missed = (
            corrupted_reference_points + offset * miss_mask[..., None]
        ).clamp(0.0, 1.0)
        corrupted_reference_points = _straight_through(
            corrupted_reference_points,
            missed,
        )
        masks["miss"] = miss_mask

    if "swap" in config.modes:
        eligible_pairs = torch.stack(
            [valid[:, left] & valid[:, right] for left, right in config.swap_pairs],
            dim=1,
        )
        selected_samples = (
            torch.rand(batch, device=reference_points.device)
            < config.swap_probability
        ) & eligible_pairs.any(dim=1)
        pair_scores = torch.rand(
            eligible_pairs.shape,
            device=reference_points.device,
        ).masked_fill(~eligible_pairs, 2.0)
        selected_pairs = pair_scores.argmin(dim=1)
        swapped_reference_points = corrupted_reference_points.clone()
        swap_mask = torch.zeros_like(valid)
        for pair_index, (left, right) in enumerate(config.swap_pairs):
            active = selected_samples & (selected_pairs == pair_index)
            if not bool(active.any()):
                continue
            swapped_reference_points[active, left] = corrupted_reference_points[
                active,
                right,
            ]
            swapped_reference_points[active, right] = corrupted_reference_points[
                active,
                left,
            ]
            swap_mask[active, left] = True
            swap_mask[active, right] = True
        corrupted_reference_points = _straight_through(
            corrupted_reference_points,
            swapped_reference_points,
        )
        masks["swap"] = swap_mask

    return corrupted_reference_points, masks
