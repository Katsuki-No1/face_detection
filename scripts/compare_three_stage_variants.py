#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_numeric(value: Mapping[str, Any], *, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            output.update(flatten_numeric(item, prefix=name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            output[name] = float(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two matched three-stage detector runs.")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("val", "test"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": f"{args.candidate_label} minus baseline",
        "baseline_root": str(args.baseline_root),
        "candidate_root": str(args.candidate_root),
        "candidate_label": str(args.candidate_label),
        "splits": {},
        "selection_policy": "Configuration fixed before test; no setting is selected from test results.",
    }
    markdown = [
        f"# Three-stage comparison: {args.candidate_label}",
        "",
        "Both arms use the same source-video split, Co-DINO HEAD path, seed, "
        "epoch budget, and validation-frozen thresholds. Architecture, "
        "initialization, and augmentation differences are defined by each "
        "run's recorded configuration.",
        "",
        "| split | metric | baseline | candidate | candidate - baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    preferred = (
        "head.ap",
        "head.ap50",
        "head.ap75",
        "head.ar100",
        "gt_head.face.macro_f1",
        "gt_head.face.face_present_recall",
        "gt_head.face.no_face_recall",
        "gt_head.ellipse.mean_iou",
        "gt_head.ellipse.p10_iou",
        "gt_head.keypoint.macro_f1_0.05",
        "gt_head.keypoint.macro_f1_0.10",
        "gt_head.keypoint.matched_point_nme_0.10",
        "gt_head.state.state_accuracy",
        "gt_head.state.state_macro_f1",
        "gt_head.state.occluded_e2e_f1",
        "matched_head.face.macro_f1",
        "matched_head.ellipse.mean_iou",
        "matched_head.keypoint.macro_f1_0.10",
        "runtime.model_inference_milliseconds_per_image",
        "runtime.peak_vram_mib",
    )
    for split in args.splits:
        baseline_path = (
            args.baseline_root
            / "evaluation"
            / split
            / "baseline_aligned"
            / "metrics.json"
        )
        candidate_path = (
            args.candidate_root
            / "evaluation"
            / split
            / "baseline_aligned"
            / "metrics.json"
        )
        if not baseline_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(
                f"Missing comparison metrics for {split}: {baseline_path}, {candidate_path}"
            )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        invariants = {
            "schema_version": baseline.get("schema_version"),
            "split": baseline.get("split"),
            "inventory": baseline.get("inventory"),
            "thresholds": baseline.get("thresholds"),
            "annotation_sha256": baseline.get("provenance", {}).get(
                "annotation_sha256"
            ),
        }
        candidate_invariants = {
            "schema_version": candidate.get("schema_version"),
            "split": candidate.get("split"),
            "inventory": candidate.get("inventory"),
            "thresholds": candidate.get("thresholds"),
            "annotation_sha256": candidate.get("provenance", {}).get(
                "annotation_sha256"
            ),
        }
        for invariant in invariants:
            if invariants[invariant] != candidate_invariants[invariant]:
                raise ValueError(
                    f"Unfair comparison: {split}.{invariant} differs: "
                    f"{invariants[invariant]!r} != "
                    f"{candidate_invariants[invariant]!r}"
                )
        baseline_flat = flatten_numeric(baseline)
        candidate_flat = flatten_numeric(candidate)
        common = sorted(set(baseline_flat) & set(candidate_flat))
        deltas = {key: candidate_flat[key] - baseline_flat[key] for key in common}
        result["splits"][split] = {
            "baseline_metrics": str(baseline_path),
            "baseline_metrics_sha256": sha256_file(baseline_path),
            "candidate_metrics": str(candidate_path),
            "candidate_metrics_sha256": sha256_file(candidate_path),
            "invariants": invariants,
            "deltas_candidate_minus_baseline": deltas,
        }
        for metric in preferred:
            if metric in deltas:
                markdown.append(
                    f"| {split} | {metric} | {baseline_flat[metric]:.6g} | "
                    f"{candidate_flat[metric]:.6g} | {deltas[metric]:+.6g} |"
                )
    markdown.extend(
        (
            "",
            "Positive deltas only mean a larger numeric value. "
            "NME/error/runtime/VRAM are lower-is-better.",
            "",
            "This is a one-seed matched comparison; test results are reporting-only.",
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "comparison.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
