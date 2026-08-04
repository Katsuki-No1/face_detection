from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def iter_batched_image_predictions(
    model: Any,
    images: Sequence[Path],
    *,
    imgsz: int,
    conf: float,
    device: str,
    batch_size: int,
    max_det: int | None = None,
) -> Iterator[tuple[Path, Any]]:
    """Run deterministic, bounded-memory Ultralytics inference over image paths.

    Passing a Python list of paths to ``YOLO.predict`` makes Ultralytics load the
    entire list as a single in-memory batch.  A temporary text manifest selects
    ``LoadImagesAndVideos`` instead, so the requested batch size is respected.
    The returned result order is checked against the manifest to prevent silent
    image/result misalignment in evaluation reports.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    ordered = [Path(path).resolve() for path in images]
    if not ordered:
        return
    missing = [path for path in ordered if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} inference images: {preview}")
    if ordered != sorted(ordered):
        raise ValueError("Inference images must be sorted for deterministic result alignment")

    with TemporaryDirectory(prefix="yolo_pose_eval_") as temporary_dir:
        manifest = Path(temporary_dir) / "images.txt"
        manifest.write_text("\n".join(str(path) for path in ordered) + "\n", encoding="utf-8")
        predict_kwargs: dict[str, Any] = {
            "source": str(manifest),
            "imgsz": imgsz,
            "conf": conf,
            "device": device,
            "batch": batch_size,
            "stream": True,
            "verbose": False,
        }
        if max_det is not None:
            predict_kwargs["max_det"] = max_det
        results = model.predict(**predict_kwargs)
        count = 0
        for result in results:
            if count >= len(ordered):
                raise RuntimeError("Ultralytics returned more results than manifest images")
            expected = ordered[count]
            actual = Path(str(result.path)).resolve()
            if actual != expected:
                raise RuntimeError(
                    f"Prediction order mismatch at index {count}: expected={expected}, actual={actual}"
                )
            yield expected, result
            count += 1
        if count != len(ordered):
            raise RuntimeError(
                f"Ultralytics returned {count} results for {len(ordered)} manifest images"
            )
