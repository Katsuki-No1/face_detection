#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the locked Messenger validation/test report.")
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--control-test-common", type=Path, required=True)
    parser.add_argument("--messenger-test-common", type=Path, required=True)
    parser.add_argument("--messenger-test-native", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _delta(recovered: dict[str, Any], raw: dict[str, Any], key: str) -> float | None:
    if recovered.get(key) is None or raw.get(key) is None:
        return None
    return float(recovered[key]) - float(raw[key])


def main() -> int:
    args = _args()
    promotion = _load(args.promotion)
    if promotion.get("status") != "success" or promotion.get("selection_split") != "val":
        raise RuntimeError("Final test report requires validation-only promotion success")
    control_common = _load(args.control_test_common)
    messenger_common = _load(args.messenger_test_common)
    native = _load(args.messenger_test_native)
    recovered = native["ordinary"]["recovered"]
    raw = native["ordinary"]["raw"]
    payload = {
        "schema_version": 1,
        "status": "success",
        "selection": {
            "source": str(args.promotion),
            "split": "val",
            "test_results_used_for_selection": False,
        },
        "test": {
            "control_common": str(args.control_test_common),
            "messenger_common": str(args.messenger_test_common),
            "common_macro_f1_0.10_delta": float(
                messenger_common["gt_head"]["keypoint"]["macro_f1_0.10"]
            )
            - float(control_common["gt_head"]["keypoint"]["macro_f1_0.10"]),
            "raw_to_recovered": {
                key: _delta(recovered, raw, key)
                for key in (
                    "conditional_semantic_macro_f1_nme_0.10",
                    "visible_semantic_macro_f1_nme_0.10",
                    "natural_occluded_semantic_macro_f1_nme_0.10",
                    "natural_visible_nme",
                    "natural_occluded_nme",
                    "pck_0.05",
                    "pck_0.10",
                )
            },
            "truncated": native["boundary"]["truncated_recovered"],
            "feature_recovery": native["feature_recovery"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = [
        "# Messenger Recovery final report",
        "",
        "Model selection used validation only; test values are report-only.",
        "",
        f"- Test common Macro F1@0.10 delta vs matched control: {payload['test']['common_macro_f1_0.10_delta']:+.6f}",
        f"- Synthetic boundary truncated PCK@0.10: {payload['test']['truncated']['pck_0.10']}",
        f"- Recovered-improved masked patch fraction: {payload['test']['feature_recovery']['recovered_improved_patch_fraction']}",
        "",
        "Raw-to-recovered paired deltas:",
    ]
    rows.extend(
        f"- {key}: {value}" for key, value in payload["test"]["raw_to_recovered"].items()
    )
    (args.output_dir / "comparison_table.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (args.output_dir / "success.json").write_text(
        json.dumps({"status": "success", "metrics": str(metrics_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
