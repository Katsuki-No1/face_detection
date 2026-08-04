from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINO-DEIM-style bbox detector on a COCO split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--category-id", type=int, default=None)
    parser.add_argument("--conf-min", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-overlays", type=int, default=30)
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def evaluate_threshold(
    gt_by_image: dict[str, list[list[float]]],
    pred_by_image: dict[str, list[dict[str, Any]]],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    per_image = []
    for image_id, gt_boxes in gt_by_image.items():
        predictions = [pred for pred in pred_by_image.get(image_id, []) if float(pred["score"]) >= score_threshold]
        predictions.sort(key=lambda pred: float(pred["score"]), reverse=True)
        matched: set[int] = set()
        image_tp = image_fp = 0
        for prediction in predictions:
            best_iou = 0.0
            best_index = -1
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched:
                    continue
                overlap = iou_xyxy(prediction["bbox_xyxy"], gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            if best_iou >= iou_threshold and best_index >= 0:
                matched.add(best_index)
                tp += 1
                image_tp += 1
            else:
                fp += 1
                image_fp += 1
        image_fn = max(0, len(gt_boxes) - len(matched))
        fn += image_fn
        per_image.append({"image_id": image_id, "tp": image_tp, "fp": image_fp, "fn": image_fn, "gt": len(gt_boxes), "pred": len(predictions)})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": score_threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "per_image": per_image}


def average_precision(
    gt_by_image: dict[str, list[list[float]]],
    pred_by_image: dict[str, list[dict[str, Any]]],
    *,
    iou_threshold: float,
) -> float:
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    flat_predictions = []
    for image_id, predictions in pred_by_image.items():
        for prediction in predictions:
            flat_predictions.append((image_id, float(prediction["score"]), prediction["bbox_xyxy"]))
    flat_predictions.sort(key=lambda item: item[1], reverse=True)
    if not flat_predictions:
        return 0.0
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    tps = []
    fps = []
    for image_id, _score, box in flat_predictions:
        gt_boxes = gt_by_image.get(image_id, [])
        best_iou = 0.0
        best_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched.setdefault(image_id, set()):
                continue
            overlap = iou_xyxy(box, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            tps.append(1.0)
            fps.append(0.0)
        else:
            tps.append(0.0)
            fps.append(1.0)
    tp_cum = np.cumsum(np.asarray(tps, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fps, dtype=np.float64))
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(1e-9, tp_cum + fp_cum)
    ap = 0.0
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        keep = recalls >= recall_threshold
        ap += float(precisions[keep].max()) if keep.any() else 0.0
    return ap / 101.0


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)


def infer_image(model: torch.nn.Module, image: Image.Image, *, image_size: int, device: str) -> list[dict[str, Any]]:
    original_w, original_h = image.size
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
        detections = model(tensor)[0]
    boxes = detections["boxes"].detach().float().cpu().numpy()
    scores = detections["scores"].detach().float().cpu().numpy()
    labels = detections.get("labels")
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.zeros((len(boxes),), dtype=np.int64)
    scale_x = original_w / float(image_size)
    scale_y = original_h / float(image_size)
    predictions = []
    for box, score, label in zip(boxes, scores, labels_np):
        x1, y1, x2, y2 = [float(value) for value in box]
        scaled = [
            max(0.0, min(float(original_w), x1 * scale_x)),
            max(0.0, min(float(original_h), y1 * scale_y)),
            max(0.0, min(float(original_w), x2 * scale_x)),
            max(0.0, min(float(original_h), y2 * scale_y)),
        ]
        if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
            continue
        predictions.append({"score": float(score), "label": int(label), "bbox_xyxy": scaled})
    return predictions


def draw_overlay(path: Path, image: Image.Image, ground_truth: list[list[float]], predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for box in ground_truth:
        draw.rectangle(tuple(box), outline="lime", width=2)
        draw.text((box[0], max(0.0, box[1] - 12.0)), "GT", fill="lime")
    for prediction in predictions:
        box = prediction["bbox_xyxy"]
        draw.rectangle(tuple(box), outline="red", width=2)
        draw.text((box[0], max(0.0, box[1] - 12.0)), f"{prediction['score']:.2f}", fill="red")
    image.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_curves(metrics_path: Path, threshold_rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"plot skipped: {error}")
        return
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if rows:
            epochs = [int(float(row.get("epoch", index + 1))) for index, row in enumerate(rows)]
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for key in ("loss_total", "loss_cls", "loss_bbox", "loss_giou"):
                if key in rows[0]:
                    axes[0].plot(epochs, [float(row[key]) for row in rows], label=key)
            axes[0].set_ylabel("loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            for key in ("match_quality", "num_pos"):
                if key in rows[0]:
                    axes[1].plot(epochs, [float(row[key]) for row in rows], label=key)
            axes[1].set_xlabel("epoch")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(output_dir / "training_curves.png", dpi=150)
            plt.close(fig)
    if threshold_rows:
        fig, axis = plt.subplots(figsize=(9, 5))
        thresholds = [float(row["threshold"]) for row in threshold_rows]
        for key in ("precision", "recall", "f1"):
            axis.plot(thresholds, [float(row[key]) for row in threshold_rows], label=key)
        axis.set_xlabel("score threshold")
        axis.set_ylabel("metric")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "threshold_sweep.png", dpi=150)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    image_size = int(args.image_size or config.get("data", {}).get("image_size") or config.get("train", {}).get("imgsz", 320))
    category_ids = config.get("data", {}).get("category_ids") or []
    category_id = int(args.category_id if args.category_id is not None else (category_ids[0] if category_ids else 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(args.annotation.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns_by_image: dict[int, list[list[float]]] = {int(image["id"]): [] for image in images}
    for annotation in coco.get("annotations", []):
        if int(annotation.get("category_id", category_id)) != category_id:
            continue
        if annotation.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(int(annotation["image_id"]), []).append(xywh_to_xyxy(annotation["bbox"]))
    gt_by_image = {str(image["id"]): anns_by_image.get(int(image["id"]), []) for image in images}

    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    model = build_detector_from_config(config).to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint)
    if hasattr(model, "postprocessor"):
        model.postprocessor.score_threshold = float(args.conf_min)
        model.postprocessor.detections_per_image = int(getattr(model.postprocessor, "detections_per_image", 100) or 100)
    model.eval()

    pred_by_image: dict[str, list[dict[str, Any]]] = {}
    records = []
    overlay_dir = args.output_dir / f"overlays_thr{int(args.fixed_threshold * 1000):03d}"
    started = time.perf_counter()
    for index, image_info in enumerate(images):
        image_id = str(image_info["id"])
        image_path = args.image_root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        predictions = infer_image(model, image, image_size=image_size, device=device)
        pred_by_image[image_id] = predictions
        for prediction in predictions:
            records.append({"image_id": image_id, "file_name": image_info["file_name"], **prediction})
        if index < args.max_overlays:
            draw_overlay(
                overlay_dir / Path(image_info["file_name"]).name,
                image.copy(),
                gt_by_image[image_id],
                [prediction for prediction in predictions if float(prediction["score"]) >= args.fixed_threshold],
            )
    elapsed = time.perf_counter() - started

    threshold_rows = []
    for threshold in [round(float(value), 2) for value in np.arange(0.01, 0.96, 0.01)]:
        row = evaluate_threshold(gt_by_image, pred_by_image, score_threshold=threshold, iou_threshold=args.iou_threshold)
        row.pop("per_image")
        threshold_rows.append(row)
    best = max(threshold_rows, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])))
    fixed = evaluate_threshold(gt_by_image, pred_by_image, score_threshold=args.fixed_threshold, iou_threshold=args.iou_threshold)
    per_image = fixed.pop("per_image")
    ap_by_iou = {f"{threshold:.2f}": average_precision(gt_by_image, pred_by_image, iou_threshold=float(threshold)) for threshold in np.arange(0.50, 0.96, 0.05)}
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "annotation": str(args.annotation),
        "image_root": str(args.image_root),
        "device": device,
        "image_size": image_size,
        "category_id": category_id,
        "num_images": len(images),
        "num_gt": sum(len(value) for value in gt_by_image.values()),
        "num_predictions_conf_min": sum(len(value) for value in pred_by_image.values()),
        "fixed_threshold": args.fixed_threshold,
        "fixed_iou_threshold": args.iou_threshold,
        "fixed_metrics": fixed,
        "best_threshold_by_f1": best,
        "ap50": ap_by_iou.get("0.50", 0.0),
        "map50_95": float(np.mean(list(ap_by_iou.values()))) if ap_by_iou else 0.0,
        "ap_by_iou": ap_by_iou,
        "inference_elapsed_sec": elapsed,
        "inference_images_per_sec": len(images) / elapsed if elapsed > 0 else None,
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
        "overlay_dir": str(overlay_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "threshold_metrics.csv", threshold_rows, ["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
    write_csv(args.output_dir / "per_image_metrics.csv", per_image, ["image_id", "tp", "fp", "fn", "gt", "pred"])
    maybe_plot_curves(args.checkpoint.parents[1] / "metrics.csv", threshold_rows, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
