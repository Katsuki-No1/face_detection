from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts_YOLO.yolo26_pose.inference import iter_batched_image_predictions


class FakeModel:
    def __init__(self, result_paths: list[Path]) -> None:
        self.result_paths = result_paths
        self.kwargs: dict[str, Any] | None = None

    def predict(self, **kwargs: Any):
        self.kwargs = kwargs
        return iter([SimpleNamespace(path=str(path)) for path in self.result_paths])


def _images(tmp_path: Path, count: int = 3) -> list[Path]:
    paths = [tmp_path / f"frame_{index:02d}.jpg" for index in range(count)]
    for path in paths:
        path.touch()
    return paths


def test_batched_inference_uses_manifest_and_requested_batch(tmp_path: Path) -> None:
    images = _images(tmp_path)
    model = FakeModel(images)

    rows = list(
        iter_batched_image_predictions(
            model,
            images,
            imgsz=960,
            conf=0.001,
            device="0",
            batch_size=8,
            max_det=100,
        )
    )

    assert [path for path, _ in rows] == [path.resolve() for path in images]
    assert model.kwargs is not None
    assert model.kwargs["batch"] == 8
    assert model.kwargs["stream"] is True
    assert isinstance(model.kwargs["source"], str)
    assert model.kwargs["source"].endswith("images.txt")


def test_batched_inference_rejects_result_order_mismatch(tmp_path: Path) -> None:
    images = _images(tmp_path)
    model = FakeModel(list(reversed(images)))

    with pytest.raises(RuntimeError, match="Prediction order mismatch"):
        list(
            iter_batched_image_predictions(
                model,
                images,
                imgsz=960,
                conf=0.001,
                device="0",
                batch_size=2,
            )
        )


def test_batched_inference_rejects_nonpositive_batch(tmp_path: Path) -> None:
    images = _images(tmp_path, count=1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(
            iter_batched_image_predictions(
                FakeModel(images),
                images,
                imgsz=960,
                conf=0.001,
                device="cpu",
                batch_size=0,
            )
        )
