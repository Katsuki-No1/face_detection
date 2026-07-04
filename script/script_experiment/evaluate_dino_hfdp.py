from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.script_experiment.run_fdp_series_eval import evaluate_keypoints  # noqa: E402
from src.config_utils import load_yaml_config  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def expand_xyxy(box: list[float], width: int, height: int, margin: float) -> list[float] | None:
    clipped = clip_xyxy(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    expanded = [
        cx - bw * margin * 0.5,
        cy - bh * margin * 0.5,
        cx + bw * margin * 0.5,
        cy + bh * margin * 0.5,
    ]
    return clip_xyxy(expanded, width, height)


def annotations_by_image(coco: dict[str, Any], *, category_id: int | None = None) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if category_id is not None and int(ann.get("category_id", 0)) != int(category_id):
            continue
        out[int(ann["image_id"])].append(ann)
    return out


def load_model(config_path: Path, checkpoint_path: Path, device: str, score_threshold: float):
    config = load_yaml_config(config_path)
    model = build_detector_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "postprocessor"):
        model.postprocessor.score_threshold = float(score_threshold)
    model.eval()
    image_size = int(config.get("data", {}).get("image_size") or config.get("train", {}).get("imgsz") or 896)
    return model, image_size, config


def image_to_tensor(image: Image.Image, image_size: int, device: str) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


@torch.no_grad()
def predict_pil(model: torch.nn.Module, image: Image.Image, *, image_size: int, device: str) -> list[dict[str, Any]]:
    width, height = image.size
    tensor = image_to_tensor(image, image_size, device)
    detections = model(tensor)[0]
    boxes = detections["boxes"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    keypoints = detections["keypoints"].detach().cpu()
    scale_x = width / float(image_size)
    scale_y = height / float(image_size)
    rows = []
    for index in range(boxes.shape[0]):
        box = boxes[index].clone()
        box[0::2] *= scale_x
        box[1::2] *= scale_y
        kpts = keypoints[index].clone()
        if kpts.numel() > 0:
            kpts[:, 0] *= scale_x
            kpts[:, 1] *= scale_y
        clipped = clip_xyxy([float(value) for value in box.tolist()], width, height)
        if clipped is None:
            continue
        rows.append(
            {
                "bbox_xyxy": clipped,
                "score": float(scores[index]),
                "keypoints": [[float(x), float(y)] for x, y in kpts.tolist()],
            }
        )
    return rows


def detection_metrics(
    gt_by_image: dict[int, list[dict[str, Any]]],
    pred_by_image: dict[int, list[dict[str, Any]]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    for image_id, gt_rows in gt_by_image.items():
        pred_rows = sorted(pred_by_image.get(image_id, []), key=lambda row: float(row.get("score", 0.0)), reverse=True)
        used: set[int] = set()
        for pred in pred_rows:
            pred_box = pred.get("bbox_xyxy") or xywh_to_xyxy(pred.get("bbox", [0, 0, 0, 0]))
            best_iou = 0.0
            best_index = None
            for index, gt in enumerate(gt_rows):
                if index in used:
                    continue
                iou = box_iou(pred_box, xywh_to_xyxy(gt["bbox"]))
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is not None and best_iou >= iou_threshold:
                tp += 1
                used.add(best_index)
            else:
                fp += 1
        fn += len(gt_rows) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "iou_threshold": iou_threshold}


def run_head_detector(
    model: torch.nn.Module,
    image_size: int,
    coco: dict[str, Any],
    dataset_root: Path,
    *,
    device: str,
    topk: int,
) -> tuple[dict[int, list[dict[str, Any]]], float]:
    started = time.perf_counter()
    pred_by_image: dict[int, list[dict[str, Any]]] = {}
    for image in coco.get("images", []):
        path = dataset_root / image["file_name"]
        with Image.open(path) as pil:
            preds = predict_pil(model, pil.convert("RGB"), image_size=image_size, device=device)
        preds = sorted(preds, key=lambda row: row["score"], reverse=True)[:topk]
        pred_by_image[int(image["id"])] = preds
    return pred_by_image, time.perf_counter() - started


def gt_heads_from_coco(head_coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [{"bbox_xyxy": xywh_to_xyxy(ann["bbox"]), "bbox": ann["bbox"], "score": 1.0} for ann in anns]
        for image_id, anns in annotations_by_image(head_coco).items()
    }


def cascade_predictions(
    *,
    stage: str,
    face_model: torch.nn.Module,
    face_image_size: int,
    gt_face_coco: dict[str, Any],
    dataset_root: Path,
    heads_by_image: dict[int, list[dict[str, Any]]],
    device: str,
    crop_margin: float,
    topk: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    rows = []
    for image in gt_face_coco.get("images", []):
        image_id = int(image["id"])
        image_path = dataset_root / image["file_name"]
        full = Image.open(image_path).convert("RGB")
        width, height = full.size
        instances: list[dict[str, Any]] = []
        face_detections: list[dict[str, Any]] = []
        for head_index, head in enumerate(heads_by_image.get(image_id, [])):
            head_box = head.get("bbox_xyxy") or xywh_to_xyxy(head.get("bbox", [0, 0, 0, 0]))
            crop_xyxy = expand_xyxy(head_box, width, height, crop_margin)
            if crop_xyxy is None:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in crop_xyxy]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = full.crop((x1, y1, x2, y2))
            crop_preds = predict_pil(face_model, crop, image_size=face_image_size, device=device)
            for pred in crop_preds:
                bx1, by1, bx2, by2 = pred["bbox_xyxy"]
                mapped = clip_xyxy([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1], width, height)
                if mapped is None:
                    continue
                mapped_kpts = [[float(px) + x1, float(py) + y1] for px, py in pred.get("keypoints", [])]
                face_detections.append(
                    {
                        "bbox": xyxy_to_xywh(mapped),
                        "bbox_xyxy": mapped,
                        "score": pred["score"],
                        "source_head_index": head_index,
                        "source_head_score": head.get("score", 1.0),
                        "crop_xyxy": [x1, y1, x2, y2],
                        "keypoints": mapped_kpts,
                    }
                )
        face_detections.sort(key=lambda row: float(row["score"]), reverse=True)
        face_detections = face_detections[:topk]
        for det in face_detections:
            instances.append(
                {
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "detector_score": det["score"],
                    "source_head_score": det.get("source_head_score"),
                    "crop_xyxy": det.get("crop_xyxy"),
                    "keypoints": det.get("keypoints", []),
                    "keypoint_scores": [det["score"]] * len(det.get("keypoints", [])),
                }
            )
        rows.append({"image_id": image_id, "file_name": image["file_name"], "instances": instances, "face_detections": face_detections})
    predictions = {"stage": stage, "status": "completed", "images": rows, "crop_margin": crop_margin}
    return predictions, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 head->face->keypoint cascade.")
    parser.add_argument("--data-root", type=Path, default=Path("outsput/experience/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outsput/experience/dino_hfdp/eval"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--head-config", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--face-config", type=Path, required=True)
    parser.add_argument("--face-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head-score-thr", type=float, default=0.05)
    parser.add_argument("--face-score-thr", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--crop-margin", type=float, default=1.2)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    data_root = args.data_root.resolve()
    head_model, head_image_size, head_config = load_model(args.head_config, args.head_checkpoint, args.device, args.head_score_thr)
    face_model, face_image_size, face_config = load_model(args.face_config, args.face_checkpoint, args.device, args.face_score_thr)

    hf_coco = read_json(data_root / "mmdet_hf" / f"{args.split}.json")
    face_coco = read_json(data_root / "mmpose_face" / f"{args.split}.json")
    head_pose_coco = read_json(data_root / "mmpose_head" / f"{args.split}.json")
    dataset_root = data_root / "mmdet_hf"

    head_pred_by_image, head_seconds = run_head_detector(
        head_model,
        head_image_size,
        hf_coco,
        dataset_root,
        device=args.device,
        topk=args.topk,
    )
    head_gt_by_image = annotations_by_image(hf_coco, category_id=1)
    head_metrics = detection_metrics(head_gt_by_image, head_pred_by_image, iou_threshold=args.iou_threshold)

    hfdp_predictions, hfdp_seconds = cascade_predictions(
        stage="DINO_HFDP",
        face_model=face_model,
        face_image_size=face_image_size,
        gt_face_coco=face_coco,
        dataset_root=dataset_root,
        heads_by_image=head_pred_by_image,
        device=args.device,
        crop_margin=args.crop_margin,
        topk=args.topk,
    )
    gthfdp_predictions, gthfdp_seconds = cascade_predictions(
        stage="DINO_GTHFDP",
        face_model=face_model,
        face_image_size=face_image_size,
        gt_face_coco=face_coco,
        dataset_root=dataset_root,
        heads_by_image=gt_heads_from_coco(head_pose_coco),
        device=args.device,
        crop_margin=args.crop_margin,
        topk=args.topk,
    )

    write_json(output_dir / "predictions" / "head_detector_predictions.json", {"images": [{"image_id": key, "detections": value} for key, value in head_pred_by_image.items()]})
    write_json(output_dir / "predictions" / "DINO_HFDP" / "predictions.json", hfdp_predictions)
    write_json(output_dir / "predictions" / "DINO_GTHFDP" / "predictions.json", gthfdp_predictions)
    hfdp_metrics = evaluate_keypoints(face_coco, hfdp_predictions, "DINO_HFDP", output_dir)
    gthfdp_metrics = evaluate_keypoints(face_coco, gthfdp_predictions, "DINO_GTHFDP", output_dir)

    rows = [
        {
            "stage": "DINO_HFDP",
            "NME": hfdp_metrics.get("NME"),
            "PCK@0.03": hfdp_metrics.get("PCK@0.03"),
            "PCK@0.05": hfdp_metrics.get("PCK@0.05"),
            "PCK@0.10": hfdp_metrics.get("PCK@0.10"),
            "gt_instances": hfdp_metrics.get("gt_instances"),
            "predicted_instances": hfdp_metrics.get("predicted_instances"),
            "evaluated_instances": hfdp_metrics.get("evaluated_instances"),
            "missed_instances": hfdp_metrics.get("missed_instances"),
        },
        {
            "stage": "DINO_GTHFDP",
            "NME": gthfdp_metrics.get("NME"),
            "PCK@0.03": gthfdp_metrics.get("PCK@0.03"),
            "PCK@0.05": gthfdp_metrics.get("PCK@0.05"),
            "PCK@0.10": gthfdp_metrics.get("PCK@0.10"),
            "gt_instances": gthfdp_metrics.get("gt_instances"),
            "predicted_instances": gthfdp_metrics.get("predicted_instances"),
            "evaluated_instances": gthfdp_metrics.get("evaluated_instances"),
            "missed_instances": gthfdp_metrics.get("missed_instances"),
        },
    ]
    write_csv(output_dir / "dino_hfdp_keypoint_summary.csv", rows)
    runtime = [
        {"name": "dino_head_detector", "kind": "inference", "elapsed_seconds": round(head_seconds, 3), "images": len(hf_coco.get("images", []))},
        {"name": "DINO_HFDP", "kind": "inference", "elapsed_seconds": round(head_seconds + hfdp_seconds, 3), "images": len(face_coco.get("images", []))},
        {"name": "DINO_GTHFDP", "kind": "inference", "elapsed_seconds": round(gthfdp_seconds, 3), "images": len(face_coco.get("images", []))},
    ]
    write_csv(output_dir / "runtime_summary.csv", runtime)
    manifest = {
        "split": args.split,
        "device": args.device,
        "head_config": str(args.head_config),
        "head_checkpoint": str(args.head_checkpoint),
        "face_config": str(args.face_config),
        "face_checkpoint": str(args.face_checkpoint),
        "head_image_size": head_image_size,
        "face_image_size": face_image_size,
        "head_model_type": head_config.get("model", {}).get("type"),
        "face_model_type": face_config.get("model", {}).get("type"),
        "score_thresholds": {"head": args.head_score_thr, "face": args.face_score_thr},
        "crop_margin": args.crop_margin,
        "head_detection_metrics": head_metrics,
        "keypoint_summary": rows,
        "runtime": runtime,
    }
    write_json(output_dir / "summary.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
