"""Primary local FACE contract for crop-only continuous recovery ablations.

All arms share the immutable historical A1/A2 full-image detector. Its exact
cached predictions can therefore be evaluated as the baseline FACE branch too.
This identity is not asserted for keypoints, nor for the global reference.
"""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

from .face_bbox_iou50 import threshold_metrics, select_validation_threshold


def ellipse_bbox(ellipse, width, height):
    cx, cy, a, b, theta = map(float, ellipse)
    rx = math.hypot(a * math.cos(theta), b * math.sin(theta))
    ry = math.hypot(a * math.sin(theta), b * math.cos(theta))
    return [max(0., cx - rx), max(0., cy - ry), min(float(width), cx + rx), min(float(height), cy + ry)]


def primary_face_metrics(coco, predictions_by_image, *, split, output_dir, candidate_floor, detector_identity):
    output_dir = Path(output_dir)
    images = {int(image["id"]): image for image in coco["images"]}
    gt = {image_id: [] for image_id in images}
    for annotation in coco["annotations"]:
        if not annotation.get("attribute_valid", False):
            continue
        image_id = int(annotation["image_id"])
        image = images[image_id]
        gt[image_id].append(dict(annotation_id=annotation["id"], bbox_xyxy=ellipse_bbox(annotation["ellipse"], image["width"], image["height"])))
    predictions = [dict(image_id=image_id, **face) for image_id in images for face in predictions_by_image.get(image_id, {}).get("faces", [])]
    if int(detector_identity.get("prediction_cap", 100)) != 100 or any(len(predictions_by_image.get(image_id, {}).get("faces", [])) > 100 for image_id in images):
        raise ValueError("Continuous FACE contract requires the common cap of 100")
    def measure(threshold):
        return {key: value for key, value in threshold_metrics(gt, predictions, score_threshold=threshold, num_images=len(images)).items() if key != "duplicate_confidences"}
    lock = output_dir.parent.parent / "val" / output_dir.name / "face_bbox_threshold_selection.json"
    if split == "val":
        # All score breakpoints: no grid can accidentally hide the recall optimum.
        thresholds = sorted({float(candidate_floor), .33, *[float(prediction["score"]) for prediction in predictions if float(prediction["score"]) >= candidate_floor]})
        try:
            selected = select_validation_threshold([measure(threshold) for threshold in thresholds])
            selection = dict(status="success", selected_on_split="val", threshold=selected["threshold"])
        except RuntimeError:
            selected = None
            selection = dict(status="unsupported", selected_on_split="val", threshold=None, unsupported_reason="no_validation_bbox_threshold_satisfies_strict_precision_gt_0.90")
        selection.update(detector_identity=detector_identity, candidate_floor=candidate_floor)
        output_dir.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(selection, indent=2) + "\n")
    else:
        selection = json.loads(lock.read_text())
        if selection.get("selected_on_split") != "val" or selection.get("detector_identity") != detector_identity:
            raise ValueError("FACE bbox validation lock belongs to a different detector")
        selected = measure(float(selection["threshold"])) if selection["threshold"] is not None else None
    common = measure(.33)
    # Both values are recomputed with the same matcher, exact prediction cap and
    # cache. Only the unchanged detector branch has zero deltas by construction.
    baseline = {"common_0_33": measure(.33), "validation_selected": measure(float(selection["threshold"])) if selection["threshold"] is not None else None}
    points = {"common_0_33": common, "validation_selected": selected}
    comparisons = []
    # The shared module's exact count names are retained rather than renamed.
    for point, values in points.items():
        if values is None:
            comparisons.append(dict(operating_point=point, unsupported_reason=selection["unsupported_reason"]))
            continue
        for key, value in values.items():
            if isinstance(value, (float, int)):
                ref = baseline[point][key]
                comparisons.append(dict(operating_point=point, metric=key, candidate=value, local_baseline=ref, absolute_delta=value-ref, relative_delta=(value-ref)/ref if ref else None,
                                        relative_delta_unsupported_reason=None if ref else "local_baseline_value_is_zero"))
    return dict(schema_version="continuous-local-face-bbox-iou50-v1", table_type="primary_local_FACE_bbox_IoU_0.50",
                denominator="all canonical attribute-valid GT FACE ellipse bounding boxes, one-to-one within image",
                gt_bbox_convention="tight axis-aligned GT ellipse bounding box clipped to image bounds",
                matching_implementation="src.evaluation.face_bbox_iou50.threshold_metrics; score-descending one-to-one; every unmatched prediction FP",
                matching_sha256=hashlib.sha256(Path(__file__).with_name("face_bbox_iou50.py").read_bytes()).hexdigest(),
                prediction_cap=100, num_images=len(images), face_gt_count=sum(map(len, gt.values())), prediction_count=len(predictions),
                selection={**selection, "criterion": "max_recall_subject_to_strict_bbox_precision_gt_0.90", "test_selection_forbidden": True},
                **points, local_baseline=dict(role="historical_local_control", architecture_id="deimv2_l_query_refinement_landmark_occlusion_a1_a2",
                reuse_reason="identical immutable full-image detector and exact same cached predictions; crop recovery does not modify FACE detections",
                detector_identity=detector_identity, scope="FACE bbox metrics only", metrics=baseline), comparison=comparisons)


def markdown_rows(metrics):
    lines = ["## Primary local FACE bbox IoU ≥ 0.50", "", "Historical local control: DEIMv2 + Query Refinement A1/A2; same immutable FACE detector. This is not a comparison to the current alpha2 baseline.", "",
             "| Operating point | Metric | Local baseline | Candidate | Absolute delta | Relative delta |", "|---|---|---:|---:|---:|---:|"]
    for row in metrics["comparison"]:
        if "unsupported_reason" in row:
            lines.append(f"| {row['operating_point']} | {row['unsupported_reason']} | N/A | N/A | N/A | N/A |")
        else:
            values = [row[key] for key in ("local_baseline", "candidate", "absolute_delta", "relative_delta")]
            lines.append(f"| {row['operating_point']} | {row['metric']} | " + " | ".join("N/A" if value is None else f"{value:.6g}" for value in values) + " |")
    return [*lines, "", "## Legacy Head-anchored diagnostic / global reference comparison", "", "The reference report is compared only within this legacy table, never with the local FACE bbox table.", ""]
