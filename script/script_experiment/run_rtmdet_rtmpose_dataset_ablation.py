from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VARIANTS = ("old_repro", "modified_50", "modified_100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare RTMDet + RTMPose on fixed-evaluation dataset ablations."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--yolo-ablation-dir", type=Path, default=Path("outputs_experiment/yolo_pose_dataset_ablation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_experiment/rtmdet_rtmpose_dataset_ablation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-epochs", type=int, default=50)
    parser.add_argument("--keypoint-epochs", type=int, default=100)
    parser.add_argument("--detector-batch", type=int, default=8)
    parser.add_argument("--keypoint-batch", type=int, default=32)
    parser.add_argument("--thresholds", default="0.01:0.95:0.01")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-visualizations", type=int, default=120)
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    process.elapsed_seconds = time.monotonic() - started  # type: ignore[attr-defined]
    return process


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def image_names(root: Path, split: str) -> set[str]:
    split_root = root / "images" / split
    if not split_root.exists():
        return set()
    return {path.name for path in split_root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}


def metadata_source_for_variant(args: argparse.Namespace, variant: str) -> Path:
    if variant == "old_repro":
        return args.yolo_ablation_dir / "datasets" / "old" / "metadata" / "conversion_metadata.json"
    return args.yolo_ablation_dir / "datasets" / "_modified_all_train" / "metadata" / "conversion_metadata.json"


def split_names_for_variant(args: argparse.Namespace, variant: str) -> dict[str, set[str]]:
    root = args.yolo_ablation_dir / "datasets" / ("old" if variant == "old_repro" else variant)
    return {split: image_names(root, split) for split in ("train", "val", "test")}


def record_output_name(record: dict[str, Any]) -> str:
    output = record.get("output_image") or record.get("image_output") or record.get("image_name") or ""
    return Path(str(output)).name


def create_variant_metadata(args: argparse.Namespace, variant: str) -> Path:
    output_path = args.output_dir / "metadata" / f"{variant}_conversion_metadata.json"
    if output_path.exists() and args.skip_existing:
        return output_path
    source_path = metadata_source_for_variant(args, variant)
    source = read_json(source_path)
    split_names = split_names_for_variant(args, variant)
    selected_names = set().union(*split_names.values())
    records = []
    missing = []
    for raw_record in source.get("records", []):
        name = record_output_name(raw_record)
        if name not in selected_names:
            continue
        split = next((candidate for candidate, names in split_names.items() if name in names), None)
        if split is None:
            missing.append(name)
            continue
        record = dict(raw_record)
        record["split"] = split
        records.append(record)
    summary = {
        "source_metadata": str(source_path),
        "variant": variant,
        "records": len(records),
        "splits": {split: sum(1 for record in records if record.get("split") == split) for split in ("train", "val", "test")},
        "missing": sorted(set(missing)),
    }
    data = {
        **{key: value for key, value in source.items() if key != "records"},
        "schema_version": source.get("schema_version", "1.0"),
        "variant_summary": summary,
        "records": records,
    }
    write_json(output_path, data)
    return output_path


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in first]
    bx1, by1, bw, bh = [float(value) for value in second]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(first) + bbox_area(second) - inter
    return inter / union if union > 0 else 0.0


def parse_thresholds(value: str) -> list[float]:
    if ":" in value:
        start, end, step = [float(item) for item in value.split(":", 2)]
        thresholds = []
        current = start
        while current <= end + 1e-12:
            thresholds.append(round(current, 6))
            current += step
        return thresholds
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def face_detection_counts(
    coco: dict[str, Any],
    predictions: dict[str, Any],
    threshold: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", 0)) == 2:
            gt_by_image[int(ann["image_id"])].append(ann)
    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image_row in predictions.get("images", []):
        image_id = int(image_row["image_id"])
        for det in image_row.get("detections", []):
            if int(det.get("category_id", 0)) == 2 and float(det.get("score", 0.0)) >= threshold:
                pred_by_image[image_id].append(det)
    tp = fp = fn = 0
    for image in coco.get("images", []):
        image_id = int(image["id"])
        gts = gt_by_image.get(image_id, [])
        preds = sorted(pred_by_image.get(image_id, []), key=lambda row: float(row.get("score", 0.0)), reverse=True)
        candidates = []
        for gt_index, gt in enumerate(gts):
            for pred_index, pred in enumerate(preds):
                iou = iou_xywh(gt["bbox"], pred["bbox"])
                if iou >= iou_threshold:
                    candidates.append((iou, gt_index, pred_index))
        candidates.sort(reverse=True)
        matched_gt = set()
        matched_pred = set()
        for _, gt_index, pred_index in candidates:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
        tp += len(matched_gt)
        fp += len(preds) - len(matched_pred)
        fn += len(gts) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def optimize_detector_threshold(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    coco = read_json(run_dir / "data" / "converted" / "detection_coco" / "instances_val.json")
    predictions = read_json(run_dir / "predictions" / "detector" / "predictions.json")
    rows = []
    for threshold in parse_thresholds(args.thresholds):
        rows.append(
            {
                "threshold": threshold,
                "iou_threshold": args.iou_threshold,
                **face_detection_counts(coco, predictions, threshold, args.iou_threshold),
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"], -row["threshold"]))
    output_dir = run_dir / "metrics" / "detector_threshold"
    write_csv(output_dir / "val_face_threshold_sweep.csv", rows)
    write_json(output_dir / "val_face_best_threshold.json", best)
    shutil.copy2(
        run_dir / "predictions" / "detector" / "predictions.json",
        run_dir / "predictions" / "detector" / "val_raw_predictions.json",
    )
    return best


def filter_detector_predictions(run_dir: Path, threshold: float, split: str) -> None:
    path = run_dir / "predictions" / "detector" / "predictions.json"
    predictions = read_json(path)
    raw_path = run_dir / "predictions" / "detector" / f"{split}_raw_predictions.json"
    shutil.copy2(path, raw_path)
    kept = removed = 0
    for image_row in predictions.get("images", []):
        filtered = []
        for det in image_row.get("detections", []):
            if float(det.get("score", 0.0)) >= threshold:
                filtered.append(det)
                kept += 1
            else:
                removed += 1
        image_row["detections"] = filtered
    predictions["filter"] = {"threshold": threshold, "kept": kept, "removed": removed, "source": str(raw_path)}
    write_json(path, predictions)


def run_rtmdet_command(args: argparse.Namespace, run_dir: Path, command: str, extra: list[str], log_name: str) -> None:
    cmd = [
        sys.executable,
        "script/script_experiment/rtmdet_rtmpose_experiment.py",
        command,
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(run_dir),
        *extra,
    ]
    result = run_command(cmd, args.output_dir / "logs" / log_name)
    if result.returncode != 0:
        raise RuntimeError(f"{command} failed for {run_dir.name}. See {args.output_dir / 'logs' / log_name}")


def checkpoint_candidates(run_dir: Path, work_dir_keywords: tuple[str, ...], filename_prefixes: tuple[str, ...]) -> list[Path]:
    candidates = sorted((run_dir / "logs").glob("**/*.pth"), key=lambda path: (path.stat().st_mtime, path.as_posix()))
    filtered = []
    for path in candidates:
        work_dir_name = path.parent.name.lower()
        filename = path.name.lower()
        if any(keyword in work_dir_name for keyword in work_dir_keywords) and filename.startswith(filename_prefixes):
            filtered.append(path)
    return filtered


def default_detector_checkpoint(run_dir: Path) -> Path:
    candidates = checkpoint_candidates(run_dir, ("rtmdet", "detector"), ("epoch_",))
    if not candidates:
        raise FileNotFoundError(f"No detector checkpoint found under {run_dir / 'logs'}")
    return candidates[-1]


def default_keypoint_checkpoint(run_dir: Path) -> Path:
    candidates = checkpoint_candidates(run_dir, ("rtmpose", "keypoint"), ("best",))
    if candidates:
        return candidates[-1]
    candidates = checkpoint_candidates(run_dir, ("rtmpose", "keypoint"), ("epoch_",))
    if not candidates:
        raise FileNotFoundError(f"No keypoint checkpoint found under {run_dir / 'logs'}")
    return candidates[-1]


def prepare_variant(args: argparse.Namespace, variant: str) -> Path:
    metadata = create_variant_metadata(args, variant)
    run_dir = args.output_dir / "runs" / variant
    if (run_dir / "data" / "converted" / "conversion_summary.json").exists() and args.skip_existing:
        return run_dir
    run_rtmdet_command(
        args,
        run_dir,
        "run-all",
        ["--metadata", str(metadata), "--mode", "prepare", "--device", args.device],
        f"{variant}_prepare.log",
    )
    return run_dir


def train_and_evaluate_variant(args: argparse.Namespace, variant: str, run_dir: Path) -> dict[str, Any]:
    det_checkpoint = None
    kp_checkpoint = None
    if not args.eval_only:
        if not (run_dir / "logs" / "train_detector_status.json").exists() or not args.skip_existing:
            run_rtmdet_command(
                args,
                run_dir,
                "train-detector",
                [
                    "--epochs",
                    str(args.detector_epochs),
                    "--batch",
                    str(args.detector_batch),
                ],
                f"{variant}_train_detector.log",
            )
        if not (run_dir / "logs" / "train_keypoint_status.json").exists() or not args.skip_existing:
            run_rtmdet_command(
                args,
                run_dir,
                "train-keypoint",
                [
                    "--epochs",
                    str(args.keypoint_epochs),
                    "--batch",
                    str(args.keypoint_batch),
                ],
                f"{variant}_train_keypoint.log",
            )
    det_checkpoint = default_detector_checkpoint(run_dir)
    kp_checkpoint = default_keypoint_checkpoint(run_dir)

    run_rtmdet_command(
        args,
        run_dir,
        "infer-detector",
        ["--checkpoint", str(det_checkpoint), "--split", "val", "--device", args.device],
        f"{variant}_infer_detector_val.log",
    )
    best_threshold = optimize_detector_threshold(run_dir, args)
    run_rtmdet_command(
        args,
        run_dir,
        "infer-detector",
        ["--checkpoint", str(det_checkpoint), "--split", "test", "--device", args.device],
        f"{variant}_infer_detector_test.log",
    )
    filter_detector_predictions(run_dir, float(best_threshold["threshold"]), "test")
    run_rtmdet_command(
        args,
        run_dir,
        "infer-keypoint-gt-bbox",
        ["--checkpoint", str(kp_checkpoint), "--split", "test", "--device", args.device],
        f"{variant}_infer_keypoint_gt_bbox.log",
    )
    run_rtmdet_command(
        args,
        run_dir,
        "infer-two-stage",
        ["--checkpoint", str(kp_checkpoint), "--split", "test", "--device", args.device],
        f"{variant}_infer_two_stage.log",
    )
    for command, log_name in [
        ("evaluate-detector", "evaluate_detector"),
        ("evaluate-keypoint", "evaluate_keypoint"),
        ("evaluate-two-stage", "evaluate_two_stage"),
    ]:
        run_rtmdet_command(args, run_dir, command, ["--split", "test"], f"{variant}_{log_name}.log")
    run_rtmdet_command(
        args,
        run_dir,
        "visualize-predictions",
        ["--split", "test", "--max-images", str(args.max_visualizations)],
        f"{variant}_visualize.log",
    )
    return {
        "detector_checkpoint": str(det_checkpoint),
        "keypoint_checkpoint": str(kp_checkpoint),
        "best_val_face_threshold": best_threshold,
    }


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def comparison_row(variant: str, run_dir: Path, run_info: dict[str, Any]) -> dict[str, Any]:
    detector = read_json(run_dir / "metrics" / "detector_metrics.json")
    keypoint = read_json(run_dir / "metrics" / "keypoint_gt_bbox_metrics.json")
    two_stage = read_json(run_dir / "metrics" / "two_stage_metrics.json")
    two_stage_kp = read_json(run_dir / "metrics" / "two_stage_keypoint_metrics.json")
    conversion = read_json(run_dir / "data" / "converted" / "conversion_summary.json")
    return {
        "model": variant,
        "train_images": nested_get(conversion, ["images_by_split", "train"]),
        "val_images": nested_get(conversion, ["images_by_split", "val"]),
        "test_images": nested_get(conversion, ["images_by_split", "test"]),
        "detector_checkpoint": run_info.get("detector_checkpoint", ""),
        "keypoint_checkpoint": run_info.get("keypoint_checkpoint", ""),
        "val_face_best_threshold": nested_get(run_info, ["best_val_face_threshold", "threshold"]),
        "det_mAP50": detector.get("mAP@0.5"),
        "det_mAP50_95": detector.get("mAP@0.5:0.95"),
        "det_precision": detector.get("precision"),
        "det_recall": detector.get("recall"),
        "det_f1": detector.get("F1"),
        "det_tp": detector.get("TP"),
        "det_fp": detector.get("FP"),
        "det_fn": detector.get("FN"),
        "face_recall": nested_get(detector, ["class_recall", "face"]),
        "gt_bbox_nme": keypoint.get("NME"),
        "gt_bbox_pck_0.05": keypoint.get("PCK@0.05"),
        "gt_bbox_pck_0.10": keypoint.get("PCK@0.10"),
        "two_stage_face_recall": two_stage.get("face_detection_recall"),
        "two_stage_keypoint_success": two_stage.get("keypoint_success_rate"),
        "two_stage_nme": two_stage_kp.get("NME"),
        "two_stage_pck_0.05": two_stage_kp.get("PCK@0.05"),
        "two_stage_pck_0.10": two_stage_kp.get("PCK@0.10"),
        "two_stage_failures": json.dumps(two_stage.get("failure_counts", {}), ensure_ascii=False),
    }


def write_failure_category_comparison(args: argparse.Namespace, run_dirs: dict[str, Path]) -> None:
    categories = set()
    counts_by_variant = {}
    for variant, run_dir in run_dirs.items():
        two_stage = read_json(run_dir / "metrics" / "two_stage_metrics.json")
        counts = two_stage.get("failure_counts", {}) or {}
        counts_by_variant[variant] = counts
        categories.update(counts)
    rows = []
    for category in sorted(categories):
        row = {"category": category}
        for variant in VARIANTS:
            row[variant] = counts_by_variant.get(variant, {}).get(category, 0)
        rows.append(row)
    write_csv(args.output_dir / "failure_category_comparison.csv", rows)


def write_next_targets(args: argparse.Namespace, comparison_rows: list[dict[str, Any]]) -> None:
    old = next((row for row in comparison_rows if row["model"] == "old_repro"), {})
    full = next((row for row in comparison_rows if row["model"] == "modified_100"), {})
    lines = [
        "# RTMDet + RTMPose Next Data Collection Targets",
        "",
        "Fixed val/test splits are shared with the YOLO ablation. Detector confidence threshold is optimized on val face F1 and applied to test.",
        "",
    ]
    if old and full:
        lines.extend(
            [
                f"- Face recall: old={old.get('face_recall')} modified_100={full.get('face_recall')}",
                f"- Two-stage PCK@0.05: old={old.get('two_stage_pck_0.05')} modified_100={full.get('two_stage_pck_0.05')}",
                f"- GT-bbox NME: old={old.get('gt_bbox_nme')} modified_100={full.get('gt_bbox_nme')}",
                "",
            ]
        )
    lines.extend(
        [
            "Prioritize cases that hurt both detector recall and keypoint quality:",
            "",
            "- small or crowded faces",
            "- missing or sparse keypoints",
            "- medium faces with high keypoint error",
            "- edge/partial faces that create localization errors",
            "",
            "See `failure_category_comparison.csv` and each run's `visualizations/errors/` directory for concrete examples.",
        ]
    )
    (args.output_dir / "next_data_collection_targets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: dict[str, Path] = {}
    for variant in VARIANTS:
        run_dirs[variant] = prepare_variant(args, variant)
    if args.prepare_only:
        print(json.dumps({"runs": {key: str(value) for key, value in run_dirs.items()}}, ensure_ascii=False, indent=2))
        return 0
    run_info_by_variant = {}
    comparison_rows = []
    for variant in VARIANTS:
        run_info = train_and_evaluate_variant(args, variant, run_dirs[variant])
        run_info_by_variant[variant] = run_info
        comparison_rows.append(comparison_row(variant, run_dirs[variant], run_info))
    write_json(args.output_dir / "run_info.json", run_info_by_variant)
    write_json(args.output_dir / "model_comparison.json", comparison_rows)
    write_csv(args.output_dir / "model_comparison.csv", comparison_rows)
    write_failure_category_comparison(args, run_dirs)
    write_next_targets(args, comparison_rows)
    print(json.dumps({"comparison": comparison_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
