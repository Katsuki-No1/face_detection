from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.landmark_detection.config import load_landmark_config
from src.landmark_detection.model import VARIANTS, build_landmark_model


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.model_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real DINOv3 model preflight requires CUDA")
def test_all_real_landmark_variants_load_strict_deim_dino_weights() -> None:
    config = load_landmark_config(
        REPO_ROOT / "configs/landmark_detection/dinov3_vits16_face_pose_drive_v2.yaml",
        project_root=REPO_ROOT,
    )
    for variant in VARIANTS:
        model, report = build_landmark_model(config.raw, variant=variant)
        assert report is not None
        assert report["missing"] == []
        assert report["unexpected"] == []
        assert report["loaded_tensors"] > 0
        del model
