from __future__ import annotations

import json
import math
import random

import numpy as np

from src.data.ellipse_error_model import (
    EllipseErrorSampler,
    apply_ellipse_residual,
    build_ellipse_error_model,
    ellipse_residual,
)


def _angle_error(left: float, right: float) -> float:
    return (left - right + math.pi * 0.5) % math.pi - math.pi * 0.5


def test_ellipse_residual_round_trip_in_gt_local_coordinates() -> None:
    ground_truth = [120.0, 80.0, 40.0, 20.0, math.radians(30.0)]
    expected_residual = [0.10, -0.15, math.log(1.2), math.log(0.8), 0.20]
    prediction = apply_ellipse_residual(ground_truth, expected_residual)
    actual_residual = ellipse_residual(ground_truth, prediction)
    assert np.allclose(actual_residual, expected_residual, atol=1.0e-6)

    reconstructed = apply_ellipse_residual(ground_truth, actual_residual)
    assert np.allclose(reconstructed[:4], prediction[:4], atol=1.0e-6)
    assert abs(_angle_error(reconstructed[4], prediction[4])) < 1.0e-6


def test_sampler_uses_one_joint_residual_and_common_severity(tmp_path) -> None:
    residuals = [
        [-0.04, 0.01, -0.10, -0.05, -0.15],
        [-0.02, 0.02, -0.05, -0.02, -0.10],
        [0.01, -0.01, 0.02, 0.01, 0.04],
        [0.03, -0.02, 0.08, 0.04, 0.12],
        [0.05, -0.03, 0.12, 0.08, 0.18],
        [0.08, -0.05, 0.18, 0.11, 0.24],
    ]
    model = build_ellipse_error_model(
        residuals,
        [-8.0, -7.5, -7.0, -5.0, -4.5, -4.0],
        source={"test_split_used": False},
        matching_report={"coverage": 1.0},
        area_bin_count=2,
        lower_quantile=0.0,
        upper_quantile=1.0,
        minimum_samples=6,
        clean_probability=0.0,
        severity_range=(1.25, 1.25),
    )
    path = tmp_path / "error_model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    sampler = EllipseErrorSampler.from_path(path)
    sample = sampler.sample(
        [50.0, 40.0, 10.0, 5.0, 0.0],
        image_width=100.0,
        image_height=100.0,
        rng=random.Random(7),
    )
    base_rows = model["area_stratification"]["samples"][sample.area_bin]
    assert any(
        np.allclose(np.asarray(sample.residual) / 1.25, row)
        for row in base_rows
    )
    assert sample.severity == 1.25
    assert not sample.clean


def test_sampler_clean_mixture_keeps_exact_gt() -> None:
    model = build_ellipse_error_model(
        [[0.1, 0.1, 0.1, 0.1, 0.1]] * 3,
        [-6.0, -5.0, -4.0],
        source={},
        matching_report={},
        area_bin_count=1,
        lower_quantile=0.0,
        upper_quantile=1.0,
        minimum_samples=3,
        clean_probability=0.999,
        severity_range=(1.0, 1.0),
    )
    sampler = EllipseErrorSampler(model)
    ground_truth = [50.0, 40.0, 10.0, 5.0, 0.25]
    sample = sampler.sample(
        ground_truth,
        image_width=100.0,
        image_height=100.0,
        rng=random.Random(1),
    )
    assert sample.clean
    assert sample.ellipse == ground_truth
    assert sample.residual == [0.0] * 5
