from __future__ import annotations

import pytest

from tools.run_small_scale_training_gate import (
    _remove_older_archives,
    assess_improvement,
)


def test_small_scale_gate_accepts_sustained_loss_improvement() -> None:
    result = assess_improvement(
        [10.0, 9.5, 9.6, 8.8, 8.5],
        direction="lower",
        window=1,
        minimum_points=5,
        minimum_relative_improvement=0.05,
        minimum_improving_transition_fraction=0.50,
    )
    assert result["passed"] is True
    assert result["relative_improvement"] == pytest.approx(0.15)


def test_small_scale_gate_rejects_flat_or_unstable_history() -> None:
    result = assess_improvement(
        [10.0, 9.9, 10.1, 9.95, 10.0],
        direction="lower",
        window=1,
        minimum_points=5,
        minimum_relative_improvement=0.01,
        minimum_improving_transition_fraction=0.50,
    )
    assert result["passed"] is False


def test_small_scale_gate_supports_higher_is_better_metrics() -> None:
    result = assess_improvement(
        [0.40, 0.42, 0.41, 0.46, 0.48],
        direction="higher",
        window=1,
        minimum_points=5,
        minimum_relative_improvement=0.10,
        minimum_improving_transition_fraction=0.50,
    )
    assert result["passed"] is True


def test_small_scale_gate_keeps_only_latest_failed_archive(tmp_path) -> None:
    work_dir = tmp_path / "small_scale"
    old = tmp_path / "small_scale.previous_1"
    latest = tmp_path / "small_scale.previous_2"
    old.mkdir()
    latest.mkdir()

    _remove_older_archives(work_dir, keep=latest)

    assert not old.exists()
    assert latest.is_dir()
