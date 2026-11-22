#!/usr/bin/env python3
"""Fit k=1..7 spatial priors from an audited TRAIN feature cache, no GPU work.

The input cache has features[N,C,H,W], geometry[N,3], records (split,
source_video_id, record_id, clean_eligible), and metadata matching the design memo.
Features must come from the same authenticated frozen backbone/crop convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import GeometryPriorBank


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    metadata = dict(cache["metadata"])
    if metadata.get("split") != "train":
        raise ValueError("Feature cache metadata must declare train")
    for key, path in (("annotation_sha256", args.annotations), ("backbone_sha256", args.backbone_checkpoint),
                      ("detector_sha256", args.detector_checkpoint)):
        if metadata.get(key) != sha256(path):
            raise ValueError(f"Cache provenance mismatch: {key}")
    records = cache["records"]
    if len(records) != len(cache["features"]) or len(records) != len(cache["geometry"]):
        raise ValueError("Cache inventory mismatch")
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Duplicate cache record IDs")
    if any(r["split"] != "train" or r["clean_eligible"] is not True for r in records):
        raise ValueError("Every cache record must be audited clean-eligible TRAIN data")
    inventory = json.loads(args.inventory.read_text())
    if (inventory.get("schema") != "geometry-prior-clean-inventory-v1"
            or inventory.get("status") != "inventory_prepared" or inventory.get("split") != "train"
            or inventory.get("provenance", {}).get("adapter", {}).get("sha256") != sha256(args.annotations)
            or metadata.get("clean_inventory_sha256") != sha256(args.inventory)):
        raise ValueError("Clean inventory provenance differs from feature cache/annotations")
    expected_records = {row["record_id"]: row for row in inventory["records"]}
    if len(expected_records) != len(inventory["records"]):
        raise ValueError("Duplicate clean inventory record IDs")
    calibration_videos = set(args.calibration_video)
    partition = inventory["partition"]
    if (calibration_videos != set(partition["calibration_video_ids"])
            or calibration_videos & set(partition["fit_video_ids"])):
        raise ValueError("Calibration partition differs from frozen clean inventory")
    for row in records:
        expected = expected_records.get(row["record_id"])
        if (expected is None or expected.get("clean_eligible") is not True
                or expected.get("split") != "train"
                or row["source_video_id"] != expected["source_video_id"]
                or expected["source_video_id"] not in set(partition["fit_video_ids"]) | calibration_videos
                or expected["partition"] != ("calibration" if row["source_video_id"] in calibration_videos else "fit")):
            raise ValueError("Feature record is not a member of the frozen clean TRAIN inventory")
    available = {r["source_video_id"] for r in records}
    if not calibration_videos or not calibration_videos < available:
        raise ValueError("Calibration videos must be a nonempty proper subset of train cache videos")
    cal = torch.tensor([r["source_video_id"] in calibration_videos for r in records])
    metadata.update(split="train", seed=args.seed, fit_video_ids=sorted(available - calibration_videos),
                    calibration_video_ids=sorted(calibration_videos), feature_cache_sha256=sha256(args.cache))
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="prepared_not_evaluated", metadata=metadata, candidates=[],
                  inventory_coverage=dict(clean_records=len(expected_records), cached_records=len(records),
                      uncached_record_ids=sorted(set(expected_records)-{r["record_id"] for r in records})),
                  training="not run", validation="not run", test="not run")
    for k in range(1, 8):
        try:
            bank = GeometryPriorBank.fit(cache["features"][~cal], cache["geometry"][~cal],
                cache["features"][cal], cache["geometry"][cal], k=k, metadata=metadata, quantile=args.quantile)
            path = args.output / f"prior_k{k}.pt"
            torch.save(bank.payload(), path)
            report["candidates"].append(dict(k=k, status="prepared", path=str(path), sha256=sha256(path),
                                             calibration_counts=bank.calibration_counts.tolist()))
        except ValueError as error:
            report["candidates"].append(dict(k=k, status="unsupported", reason=str(error)))
    if not any(row["status"] == "prepared" for row in report["candidates"]):
        report["status"] = "failed"
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="Frozen clean TRAIN membership from prepare_messenger_prior_inventory.py")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-video", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile", type=float, default=.05)
    args = parser.parse_args()
    report = prepare(args)
    print(json.dumps(report, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
