#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-only Messenger baseline adoption criteria.")
    parser.add_argument("--messenger-native", type=Path, required=True)
    parser.add_argument("--control-native", type=Path, required=True)
    parser.add_argument("--messenger-common", type=Path, required=True)
    parser.add_argument("--baseline-common", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    return parser.parse_args()


def _common_keypoint(value: dict[str, Any]) -> dict[str, Any]:
    keypoint = value.get("gt_head", {}).get("keypoint")
    if isinstance(keypoint, dict):
        return keypoint
    raise KeyError("Common metrics do not contain gt_head.keypoint")


def main() -> int:
    args = _args()
    messenger_native = _load(args.messenger_native)
    control_native = _load(args.control_native)
    messenger_common = _common_keypoint(_load(args.messenger_common))
    baseline_common = _common_keypoint(_load(args.baseline_common))
    if messenger_native.get("split") != "val" or control_native.get("split") != "val":
        raise RuntimeError("Promotion is validation-only; test metrics cannot select a model")

    messenger_occ = float(
        messenger_native["ordinary"]["recovered"][
            "natural_occluded_semantic_macro_f1_nme_0.10"
        ]
    )
    control_occ = float(
        control_native["ordinary"]["recovered"][
            "natural_occluded_semantic_macro_f1_nme_0.10"
        ]
    )
    messenger_macro = float(messenger_common["macro_f1_0.10"])
    baseline_macro = float(baseline_common["macro_f1_0.10"])
    messenger_visible_nme = float(messenger_common["visible_point_nme_0.10"])
    baseline_visible_nme = float(baseline_common["visible_point_nme_0.10"])
    improved_fraction = float(
        messenger_native["feature_recovery"]["recovered_improved_patch_fraction"]
    )
    values = (messenger_occ, control_occ, messenger_macro, baseline_macro,
              messenger_visible_nme, baseline_visible_nme, improved_fraction)
    if not all(math.isfinite(value) for value in values) or baseline_visible_nme <= 0:
        raise ValueError("Baseline adoption requires finite metrics and positive baseline NME")
    checks = {
        "natural_occluded_macro_f1_exceeds_matched_control": {
            "passed": messenger_occ > control_occ,
            "messenger": messenger_occ,
            "control": control_occ,
        },
        "overall_macro_f1_within_absolute_0.002_of_local_baseline": {
            "passed": abs(messenger_macro - baseline_macro) <= 0.002,
            "messenger": messenger_macro,
            "local_baseline": baseline_macro,
            "absolute_delta": abs(messenger_macro - baseline_macro),
        },
        "visible_nme_relative_degradation_at_most_3_percent": {
            "passed": messenger_visible_nme <= baseline_visible_nme * 1.03,
            "messenger": messenger_visible_nme,
            "local_baseline": baseline_visible_nme,
            "relative_delta": messenger_visible_nme / baseline_visible_nme - 1.0,
        },
        "synthetic_occluded_recovered_patch_improvement_above_50_percent": {
            "passed": improved_fraction > 0.50,
            "value": improved_fraction,
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    payload = {
        "schema_version": 2,
        "status": "complete",
        "outcome_type": "baseline_adoption",
        "adopted": passed,
        "decision": "adopted" if passed else "not_adopted",
        "selection_split": "val",
        "test_results_used_for_selection": False,
        "checks": checks,
        "inputs": {
            "messenger_native": str(args.messenger_native),
            "control_native": str(args.control_native),
            "messenger_common": str(args.messenger_common),
            "baseline_common": str(args.baseline_common),
        },
    }
    payload["input_sha256"] = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in payload["inputs"].items()
    }
    payload["decision_script"] = str(Path(__file__).resolve())
    payload["decision_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if passed:
        args.success_marker.parent.mkdir(parents=True, exist_ok=True)
        args.success_marker.write_text(
            json.dumps(
                {"status": "success", "selection_split": "val", "report": str(args.output)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    # A valid evaluation with no baseline adoption is distinct from an error.
    # Do not create an adoption success marker for this outcome.
    args.success_marker.unlink(missing_ok=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
