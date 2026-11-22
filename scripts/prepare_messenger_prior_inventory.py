#!/usr/bin/env python3
"""Freeze clean TRAIN inventory and disjoint-video prior calibration membership.

This is annotation preparation, not a fitted prior or model gate. Feature
extraction must still use authenticated predicted FACE ellipses and the frozen
initial crop backbone. GT annotations here only establish clean eligibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keypoint_comparison1.landmark_detection.config import load_landmark_config


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_inventory(coco, split, *, adapter=False):
    result = {}
    for image in coco["images"]:
        source = image.get("metadata", {}) if adapter else image
        video = source.get("source_video_id")
        if not video or (adapter and source.get("split") != split):
            raise ValueError("Explicit source-video and split provenance required")
        name = image["file_name"]
        if name in result:
            raise ValueError("Duplicate source image")
        result[name] = dict(video=video, width=image["width"], height=image["height"])
    if len({im["id"] for im in coco["images"]}) != len(coco["images"]):
        raise ValueError("Duplicate image ID")
    return result


def prepare_inventory(adapter, canonical, *, seed=42, calibration_fraction=.2):
    if not 0 < calibration_fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    inventories = {split: image_inventory(coco, split) for split, coco in canonical.items()}
    if set(inventories) != {"train", "val", "test"}:
        raise ValueError("All three canonical splits are required for leakage checks")
    groups = {s: {r["video"] for r in inv.values()} for s, inv in inventories.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        if groups[left] & groups[right] or inventories[left].keys() & inventories[right].keys():
            raise ValueError("Canonical split leakage")
    source = image_inventory(adapter, "train", adapter=True)
    if source != inventories["train"]:
        raise ValueError("Adapter image/video inventory differs from canonical train")
    images = {im["id"]: im for im in adapter["images"]}
    records, excluded, identifiers = [], Counter(), set()
    for annotation in adapter["annotations"]:
        identity = annotation["id"]
        if identity in identifiers:
            raise ValueError("Duplicate annotation ID")
        identifiers.add(identity)
        image = images.get(annotation["image_id"])
        if image is None:
            raise ValueError("Orphan annotation")
        reason = None
        points = annotation.get("keypoints", [])
        ellipse = annotation.get("ellipse", [])
        bbox = annotation.get("bbox", [])
        if annotation.get("category_id") != 2:
            reason = "not_face"
        elif (annotation.get("attributes") or {}).get("face_occluded") is not False:
            reason = "contour_occluded_or_unknown"
        elif len(points) != 15 or any(v != 2 for v in points[2::3]):
            reason = "not_all_five_visible"
        elif not all(math.isfinite(v) for v in points):
            reason = "nonfinite_keypoints"
        elif any(not (0 <= x < image["width"] and 0 <= y < image["height"])
                 for x, y in zip(points[0::3], points[1::3])):
            reason = "keypoint_outside_image"
        elif (len(ellipse) != 5 or not all(math.isfinite(v) for v in ellipse)
              or min(ellipse[2:4]) <= 0 or len(bbox) != 4
              or not all(math.isfinite(v) for v in bbox) or min(bbox[2:4]) <= 0):
            reason = "invalid_face_geometry"
        if reason:
            excluded[reason] += 1
            continue
        records.append(dict(record_id=str(identity), annotation_id=identity,
            image_id=image["id"], file_name=image["file_name"],
            source_video_id=source[image["file_name"]]["video"], split="train",
            clean_eligible=True))
    available = {r["source_video_id"] for r in records}
    if len(available) < 2:
        raise ValueError("At least two clean-eligible train videos required")
    ordered = sorted(available, key=lambda video: (hashlib.sha256(
        f"{seed}:geometry-prior-calibration:{video}".encode()).hexdigest(), video))
    count = max(1, min(len(ordered)-1, math.ceil(len(ordered)*calibration_fraction)))
    calibration = set(ordered[:count])
    for record in records:
        record["partition"] = "calibration" if record["source_video_id"] in calibration else "fit"
    records.sort(key=lambda record: record["annotation_id"])
    return dict(schema="geometry-prior-clean-inventory-v1", status="inventory_prepared",
        split="train", training="not run", gate_success=False,
        topology="HEAD-free full-image FACE/ellipse -> crop DINO -> Messenger geometry-prior fusion -> Query Refinement",
        clean_eligibility="Explicit contour-unoccluded, all five keypoints state2, finite in-image keypoints and valid FACE geometry; missing labels never imply clean",
        feature_geometry="Must use matched predicted FACE ellipses; GT eligibility does not authorize GT feature crops",
        partition=dict(seed=seed, calibration_fraction=calibration_fraction,
            rule="First ceil(fraction * eligible_train_videos) by SHA256(seed:geometry-prior-calibration:video); no val/test tuning",
            fit_video_ids=sorted(available-calibration), calibration_video_ids=sorted(calibration)),
        counts=dict(canonical_videos={s:len(g) for s,g in groups.items()},
            annotations=len(adapter["annotations"]), clean_records=len(records),
            excluded=dict(excluded), partitions=dict(Counter(r["partition"] for r in records))),
        records=records,
        limitations=["Annotation-defined clean eligibility is not a claim that every face pixel is visible.",
            "Predicted-crop matching, post-crop eligibility and frozen-observer feature extraction remain required."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"keypoint_comparison1/configs/continuous_recovery_abc_a.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=.2)
    args = parser.parse_args()
    raw = load_landmark_config(args.config).raw
    adapter_path = Path(raw["data"]["annotations"]["train"])
    if digest(adapter_path) != raw["data"]["expected"]["train"]["sha256"]:
        raise ValueError("Train adapter hash differs from pinned parent")
    canonical, provenance = {}, {}
    for split, specification in raw["data"]["continuous_source_inventory"].items():
        path = Path(specification["path"])
        if not path.is_absolute():
            path = ROOT/path
        actual = digest(path)
        if actual != specification["sha256"]:
            raise ValueError(f"Canonical {split} annotation hash differs from pinned parent")
        canonical[split] = json.loads(path.read_text())
        provenance[split] = dict(path=str(path), sha256=actual)
    result = prepare_inventory(json.loads(adapter_path.read_text()), canonical,
        seed=args.seed, calibration_fraction=args.calibration_fraction)
    result["provenance"] = dict(canonical=provenance,
        adapter=dict(path=str(adapter_path), sha256=digest(adapter_path)),
        resolved_config_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
        implementation_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(path=str(args.output), sha256=digest(args.output), counts=result["counts"])))


if __name__ == "__main__":
    main()
