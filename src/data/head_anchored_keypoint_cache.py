"""Explicit predicted-FACE-to-GT-Head assignments for conditional keypoint validation."""
from .face_crop_cache import crop_geometry_from_detection, iou_xyxy

CONTRACT = "predicted_face_center_in_gt_head_square_1.20_score_then_distance_one_to_one"


def assigned_head_crops(coco, predictions):
    annotations = {int(a["id"]): a for a in coco["annotations"]}
    assignments = {}
    for image_id, rows in predictions.items():
        used_predictions = set()
        for row in rows:
            annotation_id = int(row["assigned_annotation_id"])
            ann = annotations.get(annotation_id)
            if ann is None or int(ann["image_id"]) != int(image_id):
                raise ValueError("Head crop assignment references a different image/annotation")
            prediction_id = int(row["source_prediction_index"])
            if annotation_id in assignments or prediction_id in used_predictions:
                raise ValueError("Head crop assignment must be one-to-one")
            if ann.get("normalization_bbox_source") != "GT Head":
                raise ValueError("Head crop selection requires an authenticated GT Head bbox")
            used_predictions.add(prediction_id)
            box, rotation = crop_geometry_from_detection(row, require_ellipse=True)
            x, y, w, h = ann["bbox"]
            assignments[annotation_id] = {**row, "crop_bbox_xyxy": box,
                "crop_rotation_degrees": rotation,
                "matching_iou": iou_xyxy([x, y, x+w, y+h], row["bbox_xyxy"])}
    return assignments, {"matching_contract": CONTRACT, "eligible_annotations": len(annotations),
        "matched_annotations": len(assignments), "unmatched_annotations": len(annotations)-len(assignments),
        "crop_geometry": "predicted_ellipse", "normalization_bbox_source": "GT Head"}
