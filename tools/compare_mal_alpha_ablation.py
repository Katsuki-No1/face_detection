#!/usr/bin/env python3
"""Select and report the DINO-unfreeze MAL-alpha ablation on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "report"), required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="ALPHA=/absolute/or/relative/path/to/bbox_iou50/metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-selection", type=Path)
    parser.add_argument("--precision-tolerance", type=float, default=0.0)
    parser.add_argument("--recall-tolerance", type=float, default=0.0)
    return parser.parse_args(argv)


def _parse_results(values: Sequence[str]) -> dict[float, Path]:
    output: dict[float, Path] = {}
    for value in values:
        alpha_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"--result must be ALPHA=PATH, got {value!r}")
        alpha = float(alpha_text)
        if alpha in output:
            raise ValueError(f"duplicate alpha {alpha}")
        output[alpha] = Path(path_text).expanduser().resolve()
    expected = {1.0, 1.25, 1.5, 2.0}
    if set(output) != expected:
        raise ValueError(f"expected alpha set {sorted(expected)}, got {sorted(output)}")
    return output


def _row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(metrics["validation_selected"])
    return {
        "alpha": float(metrics["mal_alpha"]),
        "threshold": float(selected["threshold"]),
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f1": float(selected["f1"]),
        "tp": int(selected["tp"]),
        "fp": int(selected["fp"]),
        "fn": int(selected["fn"]),
        "duplicate_extra_prediction_count": int(
            selected["duplicate_extra_prediction_count"]
        ),
        "duplicate_prediction_rate": float(selected["duplicate_prediction_rate"]),
        "duplicated_gt_count": int(selected["duplicated_gt_count"]),
        "duplicated_gt_rate": float(selected["duplicated_gt_rate"]),
        "images_with_duplicates": int(selected["images_with_duplicates"]),
        "duplicate_image_rate": float(selected["duplicate_image_rate"]),
        "duplicate_confidence_distribution": selected[
            "duplicate_confidence_distribution"
        ],
    }


def select_optimal_alpha(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision_tolerance: float = 0.0,
    recall_tolerance: float = 0.0,
) -> tuple[float | None, list[dict[str, Any]]]:
    by_alpha = {float(row["alpha"]): dict(row) for row in rows}
    control = by_alpha[1.0]
    candidates: list[dict[str, Any]] = []
    annotated: list[dict[str, Any]] = []
    for alpha in sorted(by_alpha):
        row = by_alpha[alpha]
        precision_maintained = (
            float(row["precision"]) + precision_tolerance
            >= float(control["precision"])
        )
        recall_maintained = (
            float(row["recall"]) + recall_tolerance >= float(control["recall"])
        )
        duplication_reduced = (
            int(row["duplicate_extra_prediction_count"])
            < int(control["duplicate_extra_prediction_count"])
            and float(row["duplicate_prediction_rate"])
            < float(control["duplicate_prediction_rate"])
        )
        eligible = (
            alpha != 1.0
            and precision_maintained
            and recall_maintained
            and duplication_reduced
        )
        enriched = {
            **row,
            "delta_precision_vs_alpha1": float(row["precision"])
            - float(control["precision"]),
            "delta_recall_vs_alpha1": float(row["recall"])
            - float(control["recall"]),
            "delta_duplicate_prediction_rate_vs_alpha1": float(
                row["duplicate_prediction_rate"]
            )
            - float(control["duplicate_prediction_rate"]),
            "precision_maintained": precision_maintained,
            "recall_maintained": recall_maintained,
            "duplication_reduced": duplication_reduced,
            "eligible": eligible,
        }
        annotated.append(enriched)
        if eligible:
            candidates.append(enriched)
    if not candidates:
        return None, annotated
    best = min(
        candidates,
        key=lambda row: (
            int(row["duplicate_extra_prediction_count"]),
            float(row["duplicate_prediction_rate"]),
            -float(row["recall"]),
            -float(row["precision"]),
            float(row["alpha"]),
        ),
    )
    return float(best["alpha"]), annotated


def _markdown(rows: Sequence[Mapping[str, Any]], selected: float | None) -> str:
    lines = [
        "# MAL alpha ablation — bbox IoU 0.50",
        "",
        "| alpha | threshold | precision | recall | F1 | duplicate extras | duplicate prediction rate | duplicate conf median | eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        median = row["duplicate_confidence_distribution"]["median"]
        median_text = "N/A" if median is None else f"{float(median):.4f}"
        lines.append(
            "| {alpha:g} | {threshold:.2f} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {duplicates:d} | {rate:.4f} | {median} | {eligible} |".format(
                alpha=float(row["alpha"]),
                threshold=float(row["threshold"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                duplicates=int(row["duplicate_extra_prediction_count"]),
                rate=float(row["duplicate_prediction_rate"]),
                median=median_text,
                eligible="yes" if row.get("eligible") else "no",
            )
        )
    lines.extend(
        [
            "",
            (
                f"Validation-selected alpha: `{selected:g}`."
                if selected is not None
                else "No alpha satisfied the validation selection contract."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _parse_results(args.result)
    loaded = {
        alpha: json.loads(path.read_text(encoding="utf-8"))
        for alpha, path in paths.items()
    }
    rows = [_row(loaded[alpha]) for alpha in sorted(loaded)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "select":
        selected, rows = select_optimal_alpha(
            rows,
            precision_tolerance=float(args.precision_tolerance),
            recall_tolerance=float(args.recall_tolerance),
        )
        payload = {
            "stage": "validation_selection",
            "selection_population": "validation only",
            "control_alpha": 1.0,
            "selected_alpha": selected,
            "criteria": {
                "precision": "candidate >= alpha-1 control",
                "recall": "candidate >= alpha-1 control",
                "duplication": (
                    "both duplicate-extra count and duplicate-prediction rate "
                    "must be strictly lower than alpha-1 control"
                ),
                "precision_tolerance": float(args.precision_tolerance),
                "recall_tolerance": float(args.recall_tolerance),
            },
            "rows": rows,
        }
        (args.output_dir / "validation_selection.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        marker_name = "validation_selection_success.json"
    else:
        if args.validation_selection is None:
            raise ValueError("--validation-selection is required for --stage report")
        selection = json.loads(args.validation_selection.read_text(encoding="utf-8"))
        selected = selection.get("selected_alpha")
        payload = {
            "stage": "locked_test_report",
            "validation_selection": str(args.validation_selection.resolve()),
            "selected_alpha": selected,
            "selection_was_not_changed_on_test": True,
            "rows": rows,
        }
        marker_name = "success.json"
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "comparison_table.md").write_text(
        _markdown(rows, selected), encoding="utf-8"
    )
    (args.output_dir / marker_name).write_text(
        json.dumps(
            {
                "status": "success",
                "stage": payload["stage"],
                "selected_alpha": selected,
                "selection_complete": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

