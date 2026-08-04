from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_yolo_with_dinov3_deimv2_metrics.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("eval_yolo_common_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_image_path_accepts_dataset_root(tmp_path: Path) -> None:
    image = tmp_path / "images" / "test" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    assert MODULE.resolve_image_path(tmp_path, "images/test/frame.jpg") == image


def test_resolve_image_path_deduplicates_root_suffix(tmp_path: Path) -> None:
    image_root = tmp_path / "images" / "test"
    image = image_root / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    assert MODULE.resolve_image_path(image_root, "images/test/frame.jpg") == image


def test_resolve_image_sources_fails_before_inference(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="2/2 COCO image paths"):
        MODULE.resolve_image_sources(
            tmp_path,
            [
                {"id": 1, "file_name": "images/test/missing-a.jpg"},
                {"id": 2, "file_name": "images/test/missing-b.jpg"},
            ],
        )
