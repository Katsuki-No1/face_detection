from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .face_crop_cache import canonicalize_ellipse, perturb_ellipse


SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "center_major_diameter",
    "center_minor_diameter",
    "log_major_radius_ratio",
    "log_minor_radius_ratio",
    "angle_delta_radians",
)


def wrap_ellipse_angle(angle: float) -> float:
    """Wrap an unoriented ellipse angle difference to [-pi/2, pi/2)."""

    return (float(angle) + math.pi * 0.5) % math.pi - math.pi * 0.5


def ellipse_residual(
    ground_truth: Sequence[float],
    prediction: Sequence[float],
) -> list[float]:
    """Encode a predicted ellipse relative to GT in the GT-local frame.

    Center offsets are normalized by the corresponding GT diameter. Axis
    errors are log radius ratios, and angle is periodic modulo pi.
    """

    gt_cx, gt_cy, gt_major, gt_minor, gt_theta = canonicalize_ellipse(
        ground_truth
    )
    pred_cx, pred_cy, pred_major, pred_minor, pred_theta = canonicalize_ellipse(
        prediction
    )
    dx, dy = pred_cx - gt_cx, pred_cy - gt_cy
    cosine, sine = math.cos(gt_theta), math.sin(gt_theta)
    local_major = dx * cosine + dy * sine
    local_minor = -dx * sine + dy * cosine
    return [
        local_major / (2.0 * gt_major),
        local_minor / (2.0 * gt_minor),
        math.log(pred_major / gt_major),
        math.log(pred_minor / gt_minor),
        wrap_ellipse_angle(pred_theta - gt_theta),
    ]


def apply_ellipse_residual(
    ground_truth: Sequence[float],
    residual: Sequence[float],
) -> list[float]:
    if len(residual) != len(FEATURE_NAMES):
        raise ValueError(
            f"Ellipse residual must contain {len(FEATURE_NAMES)} values: {residual}"
        )
    values = [float(value) for value in residual]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Ellipse residual contains non-finite values: {residual}")
    return perturb_ellipse(
        ground_truth,
        center_offset=(values[0], values[1]),
        axis_scale=(math.exp(values[2]), math.exp(values[3])),
        angle_offset_radians=values[4],
    )


def ellipse_log_area_fraction(
    ellipse: Sequence[float],
    *,
    image_width: float,
    image_height: float,
) -> float:
    _cx, _cy, major, minor, _theta = canonicalize_ellipse(ellipse)
    image_area = max(1.0, float(image_width) * float(image_height))
    return math.log(max(1.0e-12, math.pi * major * minor / image_area))


def _bin_index(value: float, upper_bounds: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(upper_bounds), value, side="right"))


def build_ellipse_error_model(
    residuals: Sequence[Sequence[float]],
    area_metrics: Sequence[float],
    *,
    source: Mapping[str, Any],
    matching_report: Mapping[str, Any],
    area_bin_count: int = 3,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    minimum_samples: int = 200,
    clean_probability: float = 0.10,
    severity_range: tuple[float, float] = (0.75, 1.25),
) -> dict[str, Any]:
    """Create a robust joint empirical residual distribution."""

    matrix = np.asarray(residuals, dtype=np.float64)
    metrics = np.asarray(area_metrics, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"residuals must have shape [N,{len(FEATURE_NAMES)}], got {matrix.shape}"
        )
    if metrics.shape != (matrix.shape[0],):
        raise ValueError("area_metrics must contain one value per residual")
    if matrix.shape[0] < int(minimum_samples):
        raise RuntimeError(
            f"Only {matrix.shape[0]} matched ellipses; at least {minimum_samples} are required"
        )
    if not np.isfinite(matrix).all() or not np.isfinite(metrics).all():
        raise ValueError("Residual calibration data contains non-finite values")
    if not 1 <= int(area_bin_count) <= matrix.shape[0]:
        raise ValueError("area_bin_count is invalid")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Robust quantiles must satisfy 0 <= lower < upper <= 1")
    if not 0.0 <= float(clean_probability) < 1.0:
        raise ValueError("clean_probability must be in [0,1)")
    severity_low, severity_high = (float(value) for value in severity_range)
    if severity_low < 0.0 or severity_high < severity_low:
        raise ValueError("severity_range must be non-negative and ordered")

    lower = np.quantile(matrix, lower_quantile, axis=0)
    upper = np.quantile(matrix, upper_quantile, axis=0)
    clipped = np.clip(matrix, lower, upper)
    if int(area_bin_count) == 1:
        boundaries = np.empty((0,), dtype=np.float64)
    else:
        boundaries = np.quantile(
            metrics,
            np.linspace(0.0, 1.0, int(area_bin_count) + 1)[1:-1],
        )
    bins: list[list[list[float]]] = [[] for _ in range(int(area_bin_count))]
    for residual, metric in zip(clipped, metrics, strict=True):
        bins[_bin_index(float(metric), boundaries)].append(residual.tolist())
    if any(not rows for rows in bins):
        raise RuntimeError(
            f"Area stratification produced an empty bin: {[len(rows) for rows in bins]}"
        )

    absolute = np.abs(clipped)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": "joint_empirical_gt_local_ellipse_residual",
        "feature_names": list(FEATURE_NAMES),
        "feature_definition": {
            "center": "R(-gt_theta)*(pred_center-gt_center)/(2*gt_radii)",
            "axes": "log(pred_radius/gt_radius)",
            "angle": "wrap_pi(pred_theta-gt_theta) in [-pi/2,pi/2)",
        },
        "source": dict(source),
        "matching": dict(matching_report),
        "robust_clip": {
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "clipped_value_count": int(np.count_nonzero(matrix != clipped)),
        },
        "area_stratification": {
            "metric": "log(pi*gt_major*gt_minor/(image_width*image_height))",
            "upper_bounds": boundaries.tolist(),
            "sample_counts": [len(rows) for rows in bins],
            "samples": bins,
        },
        "sampling": {
            "preserve_joint_residual": True,
            "clean_probability": float(clean_probability),
            "severity_range": [severity_low, severity_high],
            "independent_component_noise": False,
        },
        "statistics": {
            "sample_count": int(matrix.shape[0]),
            "raw_mean": matrix.mean(axis=0).tolist(),
            "raw_std": matrix.std(axis=0).tolist(),
            "raw_median": np.median(matrix, axis=0).tolist(),
            "absolute_p90": np.quantile(absolute, 0.90, axis=0).tolist(),
            "absolute_p95": np.quantile(absolute, 0.95, axis=0).tolist(),
        },
    }


def validate_ellipse_error_model(model: Mapping[str, Any]) -> None:
    if int(model.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ellipse error model schema: {model.get('schema_version')}"
        )
    if tuple(model.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Ellipse error model feature order is incompatible")
    stratification = model.get("area_stratification")
    sampling = model.get("sampling")
    if not isinstance(stratification, Mapping) or not isinstance(sampling, Mapping):
        raise ValueError("Ellipse error model is missing stratification or sampling")
    boundaries = [float(value) for value in stratification.get("upper_bounds", ())]
    bins = stratification.get("samples")
    if (
        not isinstance(bins, list)
        or len(bins) != len(boundaries) + 1
        or any(not isinstance(rows, list) or not rows for rows in bins)
    ):
        raise ValueError("Ellipse error model contains invalid or empty area bins")
    for rows in bins:
        matrix = np.asarray(rows, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(FEATURE_NAMES)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Ellipse error model contains invalid residual samples")
    probability = float(sampling.get("clean_probability", -1.0))
    severity = [float(value) for value in sampling.get("severity_range", ())]
    if not 0.0 <= probability < 1.0:
        raise ValueError("Ellipse error model clean_probability is invalid")
    if len(severity) != 2 or severity[0] < 0.0 or severity[1] < severity[0]:
        raise ValueError("Ellipse error model severity_range is invalid")
    if sampling.get("preserve_joint_residual") is not True:
        raise ValueError("Ellipse residual components must be sampled jointly")
    if sampling.get("independent_component_noise") is not False:
        raise ValueError("Independent residual noise is forbidden")


@dataclass(frozen=True)
class EllipseErrorSample:
    ellipse: list[float]
    residual: list[float]
    area_bin: int
    clean: bool
    severity: float


class EllipseErrorSampler:
    """Sample correlated detector localization errors for GT ellipse crops."""

    def __init__(self, model: Mapping[str, Any], *, source_path: Path | None = None):
        validate_ellipse_error_model(model)
        stratification = model["area_stratification"]
        sampling = model["sampling"]
        self.model = dict(model)
        self.source_path = source_path
        self.upper_bounds = tuple(
            float(value) for value in stratification["upper_bounds"]
        )
        self.bins = tuple(
            tuple(tuple(float(value) for value in row) for row in rows)
            for rows in stratification["samples"]
        )
        self.clean_probability = float(sampling["clean_probability"])
        self.severity_range = tuple(
            float(value) for value in sampling["severity_range"]
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "EllipseErrorSampler":
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(
                "Calibrated ellipse-DEIM error model is missing: "
                f"{source_path}. The detector calibration job must finish before "
                "top-down keypoint training starts."
            )
        model = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(model, dict):
            raise ValueError(f"Ellipse error model must be a JSON object: {source_path}")
        return cls(model, source_path=source_path)

    def sample(
        self,
        ground_truth: Sequence[float],
        *,
        image_width: float,
        image_height: float,
        rng: random.Random | Any = random,
    ) -> EllipseErrorSample:
        metric = ellipse_log_area_fraction(
            ground_truth,
            image_width=image_width,
            image_height=image_height,
        )
        area_bin = _bin_index(metric, self.upper_bounds)
        if rng.random() < self.clean_probability:
            residual = [0.0] * len(FEATURE_NAMES)
            return EllipseErrorSample(
                ellipse=canonicalize_ellipse(ground_truth),
                residual=residual,
                area_bin=area_bin,
                clean=True,
                severity=0.0,
            )
        rows = self.bins[area_bin]
        base = rows[rng.randrange(len(rows))]
        severity = rng.uniform(*self.severity_range)
        residual = [float(value) * severity for value in base]
        return EllipseErrorSample(
            ellipse=apply_ellipse_residual(ground_truth, residual),
            residual=residual,
            area_bin=area_bin,
            clean=False,
            severity=float(severity),
        )
