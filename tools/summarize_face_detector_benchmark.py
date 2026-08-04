from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize face detector test metrics and write JSON, CSV and Markdown benchmark tables."
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="LABEL=METRICS_JSON",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Face detector fixed-dataset benchmark")
    return parser.parse_args()


def normalized_metrics(label: str, path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if "fixed_metrics" in metrics:
        fixed = metrics["fixed_metrics"]
        ap_by_iou = metrics.get("ap_by_iou", {})
        row = {
            "model": label,
            "num_images": int(metrics["num_images"]),
            "num_gt": int(metrics["num_gt"]),
            "mAP50_95": float(metrics["map50_95"]),
            "AP50": float(metrics["ap50"]),
            "AP75": float(ap_by_iou.get("0.75", 0.0)),
            "threshold": float(metrics["fixed_threshold"]),
            "tp": int(fixed["tp"]),
            "fp": int(fixed["fp"]),
            "fn": int(fixed["fn"]),
            "precision": float(fixed["precision"]),
            "recall": float(fixed["recall"]),
            "f1": float(fixed["f1"]),
            "metrics_path": str(path),
            "checkpoint": metrics.get("checkpoint"),
        }
    else:
        best = metrics.get("best_threshold_by_f1", {})
        tp = metrics.get("tp_fixed_threshold", metrics.get("tp", None))
        fp = metrics.get("fp_fixed_threshold", metrics.get("fp", None))
        fn = metrics.get("fn_fixed_threshold", metrics.get("fn", None))
        if tp is None and isinstance(metrics.get("fixed_metrics"), dict):
            tp = metrics["fixed_metrics"].get("tp")
            fp = metrics["fixed_metrics"].get("fp")
            fn = metrics["fixed_metrics"].get("fn")
        if tp is None and all(key in best for key in ("tp", "fp", "fn")) and float(best.get("threshold", -1)) == float(metrics["fixed_threshold"]):
            tp, fp, fn = best["tp"], best["fp"], best["fn"]
        # The official-style evaluator always records P/R/F1; infer counts when
        # older metrics files did not persist them explicitly.
        num_gt = int(metrics["num_gt"])
        if tp is None:
            tp = round(float(metrics["recall_fixed_threshold"]) * num_gt)
            fn = num_gt - int(tp)
            precision = float(metrics["precision_fixed_threshold"])
            fp = round(int(tp) / precision - int(tp)) if precision > 0 else 0
        row = {
            "model": label,
            "num_images": int(metrics["num_images"]),
            "num_gt": num_gt,
            "mAP50_95": float(metrics["mAP"]),
            "AP50": float(metrics["AP50"]),
            "AP75": float(metrics["AP75"]),
            "threshold": float(metrics["fixed_threshold"]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(metrics["precision_fixed_threshold"]),
            "recall": float(metrics["recall_fixed_threshold"]),
            "f1": float(metrics["f1_fixed_threshold"]),
            "metrics_path": str(path),
            "checkpoint": metrics.get("checkpoint"),
        }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown(title: str, rows: list[dict[str, Any]]) -> str:
    best_map = max(rows, key=lambda row: float(row["mAP50_95"]))["model"]
    best_f1 = max(rows, key=lambda row: float(row["f1"]))["model"]
    lines = [
        f"# {title}",
        "",
        "All rows use the same independent test split. The score threshold was selected on validation only.",
        "",
        "| model | mAP50:95 | AP50 | AP75 | threshold | P | R | F1 | TP / FP / FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['mAP50_95']:.4f} | {row['AP50']:.4f} | {row['AP75']:.4f} | "
            f"{row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['tp']} / {row['fp']} / {row['fn']} |"
        )
    lines.extend(
        [
            "",
            f"- Best mAP50:95: **{best_map}**",
            f"- Best fixed-threshold F1: **{best_f1}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = [normalized_metrics(label, path) for label, path in args.result]
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError("Result labels must be unique")
    test_shapes = {(row["num_images"], row["num_gt"]) for row in rows}
    if len(test_shapes) != 1:
        raise ValueError(f"Results do not use the same test set shape: {sorted(test_shapes)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": args.title,
        "selection_policy": "checkpoint and threshold selected on validation only",
        "test_shape": {"num_images": rows[0]["num_images"], "num_gt": rows[0]["num_gt"]},
        "best_mAP50_95_model": max(rows, key=lambda row: float(row["mAP50_95"]))["model"],
        "best_f1_model": max(rows, key=lambda row: float(row["f1"]))["model"],
        "results": rows,
    }
    (args.output_dir / "benchmark_test_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "benchmark_test_metrics.csv", rows)
    (args.output_dir / "benchmark_test_metrics.md").write_text(
        markdown(args.title, rows), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
