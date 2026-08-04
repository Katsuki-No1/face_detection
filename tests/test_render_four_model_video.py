from __future__ import annotations

from pathlib import Path

import pytest

from scripts.render_four_model_video import _clock, load_config


def test_clock_format() -> None:
    assert _clock(3723.5) == "01:02:03.50"


def test_production_config_has_four_existing_models() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        config = load_config(
            root / "configs/video_comparison/four_face_detectors_recall_p90.yaml"
        )
    except FileNotFoundError:
        pytest.skip("Optional legacy comparison weights were pruned locally")
    assert len(config.models) == 4
    assert [model.confidence_threshold for model in config.models] == pytest.approx([0.13, 0.33, 0.36, 0.27])
    assert all(model.weights.is_file() for model in config.models)
    assert config.models[2].source_class_id == 1
