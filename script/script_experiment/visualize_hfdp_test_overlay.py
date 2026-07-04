from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEYPOINT_NAMES = ("left_eye", "right_eye", "nose_tip", "left_mouth_corner", "right_mouth_corner")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    return [x, y, x + w, y + h]


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def draw_xywh(image: Any, bbox: list[float], color: tuple[int, int, int], cv2: Any, thickness: int = 2) -> None:
    x, y, w, h = [float(value) for value in bbox]
    cv2.rectangle(
        image,
        (int(round(x)), int(round(y))),
        (int(round(x + w)), int(round(y + h))),
        color,
        thickness,
    )


def draw_xyxy(image: Any, bbox: list[float], color: tuple[int, int, int], cv2: Any, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cv2.rectangle(
        image,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        color,
        thickness,
    )


def draw_label(image: Any, text: str, x: float, y: float, cv2: Any, scale: float = 0.48) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    x_i = max(0, int(round(x)))
    y_i = max(16, int(round(y)))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x_i, y_i - height - baseline - 4), (x_i + width + 6, y_i + 3), (20, 20, 20), -1)
    cv2.putText(image, text, (x_i + 3, y_i - baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_keypoints(image: Any, keypoints: list[Any], color: tuple[int, int, int], cv2: Any, prefix: str) -> None:
    for index, point in enumerate(keypoints):
        if isinstance(point, list):
            x, y = float(point[0]), float(point[1])
            visible = True
        else:
            offset = index * 3
            if offset + 2 >= len(keypoints):
                continue
            x, y = float(keypoints[offset]), float(keypoints[offset + 1])
            visible = int(keypoints[offset + 2]) > 0
        if not visible:
            continue
        cv2.circle(image, (int(round(x)), int(round(y))), 4, color, -1)
        label = KEYPOINT_NAMES[index] if index < len(KEYPOINT_NAMES) else str(index)
        draw_label(image, f"{prefix}:{label}", x + 5, y - 5, cv2, scale=0.36)


def build_index(rows: list[dict[str, Any]], key: str = "image_id") -> dict[int, list[dict[str, Any]]]:
    index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault(int(row[key]), []).append(row)
    return index


def best_face_iou(face_bbox: list[float], gt_faces: list[dict[str, Any]]) -> float:
    return max((iou_xywh(face_bbox, gt["bbox"]) for gt in gt_faces), default=0.0)


def summarize_head_metrics(detector_predictions: dict[str, Any], det_coco: dict[str, Any], score_thr: float, iou_thr: float) -> dict[str, Any]:
    gt_heads = build_index([ann for ann in det_coco.get("annotations", []) if int(ann.get("category_id", 0)) == 1])
    pred_by_image = {int(row["image_id"]): row.get("detections", []) for row in detector_predictions.get("images", [])}
    tp = fp = fn = 0
    for image in det_coco.get("images", []):
        image_id = int(image["id"])
        preds = [
            pred
            for pred in pred_by_image.get(image_id, [])
            if int(pred.get("category_id", 0)) == 1 and float(pred.get("score", 0.0)) >= score_thr
        ]
        matched: set[int] = set()
        for pred in sorted(preds, key=lambda row: float(row.get("score", 0.0)), reverse=True):
            best_iou = 0.0
            best_index = None
            for index, gt in enumerate(gt_heads.get(image_id, [])):
                if index in matched:
                    continue
                iou = iou_xywh(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is not None and best_iou >= iou_thr:
                tp += 1
                matched.add(best_index)
            else:
                fp += 1
        fn += max(0, len(gt_heads.get(image_id, [])) - len(matched))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1}


def visualize(args: argparse.Namespace) -> int:
    import cv2  # type: ignore

    output_dir = args.output_dir.resolve()
    out_dir = args.overlay_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    det_coco = read_json(output_dir / "data" / "mmdet_hf" / "test.json")
    face_coco = read_json(output_dir / "data" / "mmpose_face" / "test.json")
    hfdp_predictions = read_json(output_dir / "predictions" / "HFDP" / "predictions.json")
    detector_predictions = read_json(output_dir / "predictions" / "detector" / "predictions.json")

    gt_heads_by_image = build_index([ann for ann in det_coco.get("annotations", []) if int(ann.get("category_id", 0)) == 1])
    gt_faces_by_image = build_index(face_coco.get("annotations", []))
    detector_by_image = {int(row["image_id"]): row.get("detections", []) for row in detector_predictions.get("images", [])}
    hfdp_by_image = {int(row["image_id"]): row for row in hfdp_predictions.get("images", [])}

    index_rows: list[dict[str, Any]] = []
    image_root = output_dir / "data" / "mmdet_hf"
    for image in det_coco.get("images", []):
        image_id = int(image["id"])
        image_path = image_root / image["file_name"]
        canvas = cv2.imread(image_path.as_posix())
        if canvas is None:
            index_rows.append({"image_id": image_id, "file_name": image["file_name"], "status": "missing_image"})
            continue

        gt_heads = gt_heads_by_image.get(image_id, [])
        gt_faces = gt_faces_by_image.get(image_id, [])
        hfdp_row = hfdp_by_image.get(image_id, {"instances": [], "face_detections": []})
        head_preds = [
            pred
            for pred in detector_by_image.get(image_id, [])
            if int(pred.get("category_id", 0)) == 1 and float(pred.get("score", 0.0)) >= args.head_score_thr
        ]

        for gt in gt_heads:
            draw_xywh(canvas, gt["bbox"], (0, 220, 220), cv2, 2)
            draw_label(canvas, "GT head", gt["bbox"][0], gt["bbox"][1] - 4, cv2)
        for gt in gt_faces:
            draw_xywh(canvas, gt["bbox"], (0, 220, 0), cv2, 2)
            draw_label(canvas, f"GT face id={gt.get('id')}", gt["bbox"][0], gt["bbox"][1] - 4, cv2)
            draw_keypoints(canvas, gt.get("keypoints", []), (0, 220, 0), cv2, "GT")

        for index, pred in enumerate(head_preds):
            draw_xywh(canvas, pred["bbox"], (255, 180, 0), cv2, 1)
            draw_label(canvas, f"det head {index} s={float(pred.get('score', 0.0)):.2f}", pred["bbox"][0], pred["bbox"][1] + pred["bbox"][3] + 16, cv2, scale=0.4)

        crop_boxes = []
        for face_det in hfdp_row.get("face_detections", []):
            crop = face_det.get("crop_xyxy")
            if crop and crop not in crop_boxes:
                crop_boxes.append(crop)
        for instance in hfdp_row.get("instances", []):
            crop = instance.get("crop_xyxy")
            if crop and crop not in crop_boxes:
                crop_boxes.append(crop)
        for crop_index, crop in enumerate(crop_boxes):
            draw_xyxy(canvas, crop, (255, 255, 0), cv2, 2)
            draw_label(canvas, f"HFDP crop {crop_index}", crop[0], crop[1] + 16, cv2, scale=0.42)

        face_ious = []
        for det_index, face_det in enumerate(hfdp_row.get("face_detections", [])):
            det_iou = best_face_iou(face_det["bbox"], gt_faces)
            face_ious.append(det_iou)
            draw_xywh(canvas, face_det["bbox"], (0, 0, 255), cv2, 2)
            draw_label(
                canvas,
                f"HFDP face {det_index} s={float(face_det.get('score', 0.0)):.2f} IoU={det_iou:.2f}",
                face_det["bbox"][0],
                face_det["bbox"][1] - 18,
                cv2,
                scale=0.42,
            )

        for pose_index, instance in enumerate(hfdp_row.get("instances", [])):
            if "bbox" in instance:
                draw_xywh(canvas, instance["bbox"], (255, 0, 255), cv2, 1)
            draw_keypoints(canvas, instance.get("keypoints", []), (255, 0, 255), cv2, "P")
            bbox = instance.get("bbox", [10, 10, 0, 0])
            draw_label(canvas, f"pose {pose_index} score={float(instance.get('score', 0.0)):.2f}", bbox[0], bbox[1] + bbox[3] + 30, cv2, scale=0.4)

        status_text = (
            f"HFDP test | GT face={len(gt_faces)} GT head={len(gt_heads)} "
            f"head det={len(head_preds)} crop face={len(hfdp_row.get('face_detections', []))} "
            f"best IoU={max(face_ious) if face_ious else 0.0:.2f}"
        )
        draw_label(canvas, status_text, 8, 22, cv2, scale=0.55)

        out_path = out_dir / f"{image_id:06d}_{Path(image['file_name']).name}"
        cv2.imwrite(out_path.as_posix(), canvas)
        index_rows.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "overlay": out_path.as_posix(),
                "gt_heads": len(gt_heads),
                "gt_faces": len(gt_faces),
                "head_detections": len(head_preds),
                "hfdp_face_detections": len(hfdp_row.get("face_detections", [])),
                "hfdp_pose_instances": len(hfdp_row.get("instances", [])),
                "best_face_iou": max(face_ious) if face_ious else 0.0,
                "status": "ok",
            }
        )

    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(index_rows[0].keys()) if index_rows else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    head_metrics = summarize_head_metrics(detector_predictions, det_coco, args.head_score_thr, args.iou_thr)
    (out_dir / "head_metrics.json").write_text(json.dumps(head_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overlay_dir": out_dir.as_posix(), "images": len(index_rows), "head_metrics": head_metrics}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize HFDP test overlays from saved predictions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outsput/experience"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("outsput/experience/visualizations/HFDP_debug/test"))
    parser.add_argument("--head-score-thr", type=float, default=0.3)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    return visualize(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
