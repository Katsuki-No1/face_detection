#!/usr/bin/env python3
"""Prepare frozen, predicted-face FOR context on explicit fixed splits.

No training or success markers. A cache is input data, never a passed gate.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
from keypoint_comparison1.landmark_detection.evaluation import load_or_create_detector_predictions
from keypoint_comparison1.landmark_detection.face_overlap import validate_context_cache, file_sha256
from src.data.face_crop_cache import load_detector_prediction_cache
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "keypoint_comparison1/configs/face_overlap_robustness_v1.yaml")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    raw = load_landmark_config(args.config, project_root=ROOT).raw
    for_cfg = raw["model"].get("face_overlap_robustness", {})
    if not for_cfg.get("enabled"):
        raise ValueError("FOR must be enabled")
    cfg = raw["detector"]
    detector = DeimFaceDetector(
        config_path=cfg["config"], checkpoint_path=cfg["checkpoint"], device=args.device,
        image_size=cfg["image_size"], resize_mode=cfg.get("resize_mode"),
        pad_color=cfg.get("pad_color"), size_divisor=cfg.get("size_divisor"),
        score_threshold=cfg["score_threshold"], maximum_detections=cfg["maximum_detections"],
        crop_geometry="ellipse", return_face_query_features=True,
    )
    data = raw["data"]
    for split in args.splits:
        annotation = Path(data["annotations"][split])
        expected_hash = data.get("expected", {}).get(split, {}).get("sha256")
        if not expected_hash or file_sha256(annotation) != expected_hash:
            raise ValueError(f"FOR {split} annotation differs from its locked split hash")
        path = Path(data["face_context_caches"][split])
        if path.exists():
            raise FileExistsError(f"Preserve existing cache; use a new explicit config/output: {path}")
        predictions = load_or_create_detector_predictions(
            detector=detector, annotation_path=annotation,
            image_root=Path(data.get("image_roots", {}).get(split, data["root"])), cache_path=path,
        )
        metadata, _ = load_detector_prediction_cache(path)
        coco = json.loads(annotation.read_text())
        validate_context_cache(metadata, predictions, annotation_path=annotation,
                               image_ids={int(row["id"]) for row in coco["images"]},
                               query_dim=int(for_cfg["query_dim"]))
        print(f"{split}: {len(predictions)} images, {sum(map(len, predictions.values()))} detections -> {path}")


if __name__ == "__main__":
    main()
