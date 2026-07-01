from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config  # noqa: E402


METRIC_FIELDS = [
    "model",
    "experiment",
    "status",
    "threshold",
    "precision",
    "recall",
    "f1",
    "bbox_map",
    "bbox_map50",
    "ellipse_center_error",
    "ellipse_rx_error",
    "ellipse_ry_error",
    "ellipse_iou",
    "keypoint_nme",
    "keypoint_pck",
    "small_face_recall",
    "occluded_face_recall",
    "angled_face_recall",
    "dark_low_quality_face_recall",
    "gt_instances",
    "images",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DINO CenterFace augmentation ablation summaries.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/dino_centerface_ellipse_aug_ablation"))
    parser.add_argument("--eval-root", type=Path, default=Path("outputs_experiment/dino_centerface_ellipse_aug_ablation_eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_experiment/dino_centerface_ellipse_aug_ablation_eval/_summary"))
    parser.add_argument(
        "--yolo11s-summary",
        type=Path,
        default=Path("outputs_experiment/dino_scrfd_vs_yolo11s_20260628/comparison_summary.json"),
    )
    parser.add_argument("--include-inactive", action="store_true", help="Include configs marked experiment.active: false.")
    return parser.parse_args()


def config_paths(config_dir: Path, *, include_inactive: bool = False) -> list[Path]:
    paths = []
    for path in sorted(config_dir.glob("E*.yaml")):
        if not path.is_file():
            continue
        if not include_inactive and load_yaml_config(path).get("experiment", {}).get("active", True) is False:
            continue
        paths.append(path)
    return paths


def experiment_name(path: Path) -> str:
    config = load_yaml_config(path)
    return str(config.get("experiment", {}).get("name") or config.get("train", {}).get("name") or path.stem)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dino_row(config_path: Path, eval_root: Path) -> dict[str, Any]:
    name = experiment_name(config_path)
    summary_path = eval_root / name / "summary.json"
    summary = read_json(summary_path)
    if summary is None:
        return {"model": "DINO-CenterFace-ellipse", "experiment": name, "status": "missing", "summary_path": str(summary_path)}
    best = summary.get("best_threshold", {})
    condition = best.get("condition_recall", {}) if isinstance(best.get("condition_recall"), dict) else {}
    ap = summary.get("bbox_ap", {}) if isinstance(summary.get("bbox_ap"), dict) else {}
    return {
        "model": "DINO-CenterFace-ellipse",
        "experiment": name,
        "status": "ok",
        "threshold": best.get("threshold"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "f1": best.get("f1"),
        "bbox_map": summary.get("bbox_map"),
        "bbox_map50": ap.get("AP50"),
        "ellipse_center_error": best.get("ellipse_center_error"),
        "ellipse_rx_error": best.get("ellipse_rx_error"),
        "ellipse_ry_error": best.get("ellipse_ry_error"),
        "ellipse_iou": best.get("ellipse_iou"),
        "keypoint_nme": best.get("keypoint_nme"),
        "keypoint_pck": best.get("keypoint_pck"),
        "small_face_recall": condition.get("small_face"),
        "occluded_face_recall": condition.get("occluded_face"),
        "angled_face_recall": condition.get("angled_face"),
        "dark_low_quality_face_recall": condition.get("dark_low_quality_face"),
        "gt_instances": summary.get("gt_instances"),
        "images": summary.get("images"),
        "summary_path": str(summary_path),
    }


def yolo_row(path: Path) -> dict[str, Any] | None:
    data = read_json(path)
    if data is None:
        return None
    yolo = data.get("yolo11s", {})
    return {
        "model": "YOLO11s",
        "experiment": yolo.get("selected_candidate", "baseline"),
        "status": "baseline",
        "threshold": yolo.get("threshold"),
        "precision": yolo.get("precision"),
        "recall": yolo.get("recall"),
        "f1": yolo.get("f1"),
        "bbox_map": yolo.get("box_map50_95"),
        "bbox_map50": yolo.get("box_map50"),
        "ellipse_center_error": "",
        "ellipse_rx_error": "",
        "ellipse_ry_error": "",
        "ellipse_iou": "",
        "keypoint_nme": yolo.get("keypoint_nme"),
        "keypoint_pck": yolo.get("pck_0.05"),
        "small_face_recall": "",
        "occluded_face_recall": "",
        "angled_face_recall": "",
        "dark_low_quality_face_recall": "",
        "gt_instances": data.get("dataset", {}).get("gt_instances"),
        "images": data.get("dataset", {}).get("images"),
        "summary_path": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DINO CenterFace Ellipse Augmentation Ablation",
        "",
        "| model | experiment | status | precision | recall | F1 | bbox mAP | AP50 | ellipse IoU | kpt NME | PCK | small recall | occluded recall | angled recall | dark recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("model")),
                    fmt(row.get("experiment")),
                    fmt(row.get("status")),
                    fmt(row.get("precision")),
                    fmt(row.get("recall")),
                    fmt(row.get("f1")),
                    fmt(row.get("bbox_map")),
                    fmt(row.get("bbox_map50")),
                    fmt(row.get("ellipse_iou")),
                    fmt(row.get("keypoint_nme")),
                    fmt(row.get("keypoint_pck")),
                    fmt(row.get("small_face_recall")),
                    fmt(row.get("occluded_face_recall")),
                    fmt(row.get("angled_face_recall")),
                    fmt(row.get("dark_low_quality_face_recall")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = [dino_row(path, args.eval_root) for path in config_paths(args.config_dir, include_inactive=bool(args.include_inactive))]
    baseline = yolo_row(args.yolo11s_summary)
    if baseline is not None:
        rows.append(baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "augmentation_ablation_comparison.csv", rows)
    write_markdown(args.output_dir / "augmentation_ablation_comparison.md", rows)
    (args.output_dir / "augmentation_ablation_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
