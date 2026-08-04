#!/usr/bin/env python3
"""Calibrate GT-crop perturbations from ellipse-DEIM validation residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_comparison2.config import load_config  # noqa: E402
from keypoint_comparison2.inference import DEIMv2LFaceDetector  # noqa: E402
from src.data.ellipse_error_model import (  # noqa: E402
    build_ellipse_error_model,
    ellipse_log_area_fraction,
    ellipse_residual,
)
from src.data.face_crop_cache import match_annotations_to_detector_crops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "keypoint_comparison2/configs/production.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_image(root: Path, file_name: str) -> Path:
    path = Path(file_name)
    candidates = (path,) if path.is_absolute() else (root / path, root / "images" / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve {file_name!r} below {root}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, project_root=REPO_ROOT)
    raw = config.raw
    calibration = raw["data"]["crop_error_calibration"]
    annotation_path = Path(calibration["annotation"])
    image_root = Path(calibration["image_root"])
    output_path = Path(calibration["error_model"])
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco.get("images", [])}
    configured_videos = tuple(str(value) for value in calibration["source_video_ids"])
    actual_videos = {
        str(row.get("metadata", {}).get("source_video_id", ""))
        for row in images.values()
    }
    if actual_videos != set(configured_videos):
        raise RuntimeError(
            "Detector-error calibration video set changed: "
            f"configured={sorted(configured_videos)}, actual={sorted(actual_videos)}"
        )
    expected_hash = str(calibration["annotation_sha256"])
    actual_hash = _sha256(annotation_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Calibration annotation SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    detector = DEIMv2LFaceDetector(raw["detector"], device=device)
    predictions: dict[int, list[dict[str, Any]]] = {}
    total_images = len(images)
    for position, (image_id, row) in enumerate(images.items(), start=1):
        image_path = _resolve_image(image_root, str(row["file_name"]))
        with Image.open(image_path) as image:
            predictions[image_id] = detector(image.convert("RGB"))
        if position == 1 or position % 25 == 0 or position == total_images:
            print(f"ellipse calibration inference: {position}/{total_images}", flush=True)

    category_ids = tuple(int(value) for value in calibration["category_ids"])
    assignments, matching = match_annotations_to_detector_crops(
        coco,
        predictions,
        category_ids=category_ids,
        minimum_iou=float(calibration["minimum_matching_iou"]),
        require_ellipse=True,
    )
    annotations = {
        int(row["id"]): row
        for row in coco.get("annotations", [])
        if int(row.get("category_id", -1)) in set(category_ids)
    }
    residuals: list[list[float]] = []
    area_metrics: list[float] = []
    eligible_by_video: Counter[str] = Counter()
    matched_by_video: Counter[str] = Counter()
    eligible_ellipses = 0
    for annotation_id, annotation in annotations.items():
        ellipse = annotation.get("ellipse")
        if not isinstance(ellipse, (list, tuple)) or len(ellipse) != 5:
            continue
        image = images[int(annotation["image_id"])]
        video = str(image.get("metadata", {}).get("source_video_id", ""))
        eligible_ellipses += 1
        eligible_by_video[video] += 1
        assignment = assignments.get(annotation_id)
        if assignment is None:
            continue
        prediction = assignment.get("ellipse")
        if not isinstance(prediction, (list, tuple)) or len(prediction) != 5:
            raise RuntimeError(
                f"Matched detector result lacks ellipse: annotation_id={annotation_id}"
            )
        residuals.append(ellipse_residual(ellipse, prediction))
        area_metrics.append(
            ellipse_log_area_fraction(
                ellipse,
                image_width=float(image["width"]),
                image_height=float(image["height"]),
            )
        )
        matched_by_video[video] += 1

    matching.update(
        {
            "eligible_ellipses": eligible_ellipses,
            "matched_ellipse_residuals": len(residuals),
            "ellipse_residual_coverage": len(residuals) / max(1, eligible_ellipses),
            "per_source_video": {
                video: {
                    "eligible_ellipses": eligible_by_video[video],
                    "matched_ellipse_residuals": matched_by_video[video],
                    "coverage": matched_by_video[video]
                    / max(1, eligible_by_video[video]),
                }
                for video in sorted(eligible_by_video)
            },
        }
    )
    detector_config_path = Path(raw["detector"]["config"])
    detector_checkpoint_path = Path(raw["detector"]["checkpoint"])
    source = {
        "calibration_annotation": str(annotation_path),
        "calibration_annotation_sha256": actual_hash,
        "calibration_source_video_ids": sorted(actual_videos),
        "detector_config": str(detector_config_path),
        "detector_config_sha256": _sha256(detector_config_path),
        "detector_checkpoint": str(detector_checkpoint_path),
        "detector_checkpoint_sha256": _sha256(detector_checkpoint_path),
        "detector_score_threshold": float(raw["detector"]["score_threshold"]),
        "test_split_used": False,
        "keypoint_selection_split_used": False,
    }
    robust_quantiles = tuple(
        float(value) for value in calibration["robust_quantiles"]
    )
    severity_range = tuple(float(value) for value in calibration["severity_range"])
    model = build_ellipse_error_model(
        residuals,
        area_metrics,
        source=source,
        matching_report=matching,
        area_bin_count=int(calibration["area_bins"]),
        lower_quantile=robust_quantiles[0],
        upper_quantile=robust_quantiles[1],
        minimum_samples=int(calibration["minimum_matched_samples"]),
        clean_probability=float(calibration["clean_probability"]),
        severity_range=(severity_range[0], severity_range[1]),
    )
    _atomic_json(output_path, model)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": _sha256(output_path),
                "matched_samples": len(residuals),
                "coverage": matching["ellipse_residual_coverage"],
                "area_bin_counts": model["area_stratification"]["sample_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
