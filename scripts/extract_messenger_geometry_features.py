#!/usr/bin/env python3
"""Authenticated TRAIN-only predicted-crop observer cache and k1..7 preparation.

No optimizer, training gate, validation selection or test inference is run here.
Use the canonical idle/lock runner for GPU preparation. Completed stages are
reusable only with identical input/code hashes; unverified outputs fail closed.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from keypoint_comparison1.landmark_detection.config import (
    load_landmark_config, configured_keypoint_target_policy, configured_state_head_mode,
)
from scripts.prepare_messenger_geometry_priors import sha256, prepare as fit_priors


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def crop_predictions(rows, image_ids, threshold):
    """Keep the inherited fixed operating point and retain empty-image entries."""
    result = {str(i): [] for i in image_ids}
    for row in rows:
        if str(row["image_id"]) not in result:
            raise ValueError("Prediction outside train image inventory")
        if row["score"] >= threshold:
            if len(row.get("ellipse", [])) != 5:
                raise ValueError("Predicted FACE ellipse is required; no GT fallback")
            result[str(row["image_id"])].append(dict(ellipse=row["ellipse"],
                bbox_xyxy=row["bbox_xyxy"], score=row["score"]))
    return result


def reusable(output, name, identity):
    marker = output / f"{name}.complete.json"
    if not marker.exists():
        return None
    value = json.loads(marker.read_text())
    if value.get("identity") != identity:
        raise ValueError(f"Stale {name} inputs or code")
    for file, expected in value["artifacts"].items():
        if sha256(output / file) != expected:
            raise ValueError(f"Corrupt {name} artifact: {file}")
    return value


def finish(output, name, identity, artifacts, **details):
    result = dict(identity=identity, artifacts={p.name:sha256(p) for p in artifacts}, **details)
    write_json(output / f"{name}.complete.json", result)
    return result


def run(args):
    from tools.dinov3_deimv2_common import load_config
    plan = yaml.safe_load(args.plan.read_text())
    raw = load_landmark_config(ROOT / plan["initialization"]["parent_training_config"]).raw
    inventory_path = ROOT / plan["clean_inventory_integration_20261012"]["artifact"]
    if sha256(inventory_path) != plan["clean_inventory_integration_20261012"]["artifact_sha256"]:
        raise ValueError("Clean train inventory changed")
    inventory = json.loads(inventory_path.read_text())
    annotation = Path(raw["data"]["annotations"]["train"])
    if sha256(annotation) != inventory["provenance"]["adapter"]["sha256"]:
        raise ValueError("Training annotations changed")
    parent = Path(raw["train"]["finetune_checkpoint"]["path"])
    detector = Path(raw["detector"]["checkpoint"])
    if sha256(parent) != plan["initialization"]["expected_sha256"]:
        raise ValueError("Frozen observer parent checkpoint changed")
    if sha256(detector) != plan["feature_preparation"]["detector_checkpoint_sha256"]:
        raise ValueError("Pinned matched-control detector changed")
    if sha256(Path(raw["detector"]["config"])) != plan["feature_preparation"]["detector_config_sha256"]:
        raise ValueError("Pinned detector config changed")
    cfg = load_config(Path(raw["detector"]["config"]))
    if Path(cfg["data"]["annotation_paths"]["train"]).resolve() != annotation.resolve():
        raise ValueError("Detector and observer training inventories differ")
    coco = json.loads(annotation.read_text())
    image_hashes = []
    for image in coco["images"]:
        actual = sha256(Path(raw["data"]["root"]) / image["file_name"])
        if image["metadata"].get("sha256") != actual:
            raise ValueError("Source train image changed: " + image["file_name"])
        image_hashes.append((image["file_name"], actual))
    files = [Path(__file__), ROOT/"scripts/prepare_messenger_geometry_priors.py",
        ROOT/"src/data/face_crop_cache.py", ROOT/"scripts/train_dino.py"]
    files += sorted((ROOT/"keypoint_comparison1/landmark_detection").glob("*.py"))
    identity = dict(split="train", clean_inventory_sha256=sha256(inventory_path),
        annotation_sha256=sha256(annotation), backbone_sha256=sha256(parent),
        detector_sha256=sha256(detector), detector_config_sha256=sha256(Path(raw["detector"]["config"])),
        resolved_parent_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),
        image_inventory_sha256=hashlib.sha256(json.dumps(sorted(image_hashes)).encode()).hexdigest(),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
        score_threshold=raw["detector"]["score_threshold"], prediction_cap=100,
        detector_weight_branch="ema", observer_weight_branch="ema", inference_batch=1,
        feature_batch=4, feature_precision="float32", seed=42)
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output/"preparation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Refusing to replace a different preparation")
    write_json(identity_path, identity)
    torch.set_num_threads(1)
    from keypoint_comparison1.landmark_detection.training import seed_everything
    seed_everything(42)
    predictions_path = args.output/"train_predictions.json"
    crop_path = args.output/"train_predicted_crops.json"
    if reusable(args.output, "predictions", identity) is None:
        if predictions_path.exists() or crop_path.exists():
            raise ValueError("Unverified partial prediction output; inspect before recovery")
        from src.models import build_detector_from_config
        from scripts.train_dino import validate_detection_split
        cfg = deepcopy(cfg)
        cfg["model"].pop("init_checkpoint", None)
        cfg["inference"].update(score_thr=0., detections_per_image=100)
        payload = torch.load(detector, map_location="cpu", weights_only=False)
        if not isinstance(payload.get("ema"), dict):
            raise ValueError("Authenticated detector EMA is required")
        model = build_detector_from_config(cfg).to(args.device).eval()
        model.load_state_dict(payload["ema"].get("module",payload["ema"]),strict=True)
        del payload
        print("Preparing full canonical train predictions",flush=True)
        result = validate_detection_split(model, cfg, device=args.device, split="train",
            image_size=960, fixed_threshold=identity["score_threshold"], score_min=0.,
            max_images=None,batch_size=1,predictions_output=predictions_path,compute_validation_loss=False)
        if result["num_images"] != len(coco["images"]):
            raise ValueError("Incomplete train prediction inventory")
        grouped = crop_predictions(json.loads(predictions_path.read_text()),
            [im["id"] for im in coco["images"]],identity["score_threshold"])
        write_json(crop_path,dict(metadata=dict(crop_geometry="ellipse",**identity),predictions=grouped))
        del model
        torch.cuda.empty_cache()
        finish(args.output,"predictions",identity,[predictions_path,crop_path],images=result["num_images"])
    features_path = args.output/"train_observer_features.pt"
    if reusable(args.output,"features",identity) is None:
        if features_path.exists():
            raise ValueError("Unverified feature cache; inspect before recovery")
        from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset, CropAugmentation
        from keypoint_comparison1.landmark_detection.model import build_landmark_model
        from keypoint_comparison1.landmark_detection.training import initialize_from_finetune_checkpoint
        from keypoint_comparison1.landmark_detection.geometry_prior_recovery import backbone_state_sha256
        model,_ = build_landmark_model(raw,variant="query_refinement",initialize_from_detector=False)
        initialization = initialize_from_finetune_checkpoint(model, raw["train"]["finetune_checkpoint"],
            variant="query_refinement",expected_target_policy=configured_keypoint_target_policy(raw),
            expected_state_head_mode=configured_state_head_mode(raw))
        observer = model.backbone.requires_grad_(False).eval().to(args.device)
        del model
        observer_sha = backbone_state_sha256(observer)
        ds = TopDownFaceLandmarkDataset(annotation_path=annotation,image_root=raw["data"]["root"],
            training=False,input_size=256,category_ids=(2,),
            augmentation=CropAugmentation(scale_factor=raw["data"]["augmentation"]["scale_factor"]),
            mean=raw["data"]["mean"],std=raw["data"]["std"],detector_crop_cache=crop_path,
            detector_matching_iou=.5,geometry_prior_grid_size=(16,16))
        expected={r["record_id"]:r for r in inventory["records"]}
        inputs, pending, records, feature_parts, geometry_parts, support_parts = [],[],[],[],[],[]
        rejected=Counter(); seen=set()
        def flush():
            if not inputs:return
            with torch.inference_mode():
                features=observer(torch.stack(inputs).to(args.device)).float().cpu()
            if features.shape[1:] != (384,16,16) or not torch.isfinite(features).all():
                raise ValueError("Unexpected/non-finite frozen-observer feature map")
            feature_parts.append(features)
            for row,target in pending:
                records.append(dict(row, predicted_ellipse=target['geometry_prior']['predicted_ellipse_source'].tolist(),
                    matching_iou=target['detector_matching_iou']))
                geometry_parts.append(target['geometry_prior']['geometry'])
                support_parts.append(target['geometry_prior']['support'])
            inputs.clear();pending.clear()
        for i,sample in enumerate(ds.samples):
            rid=str(sample["annotation"]["id"])
            if rid not in expected:continue
            seen.add(rid)
            image,target=ds[i]
            if not bool((target["keypoint_visibility"]==2).all()):
                rejected["visible_point_outside_predicted_crop"]+=1;continue
            if not torch.isfinite(image).all() or target['occlusion_mask'].any():
                raise ValueError("Unclean/non-finite observer input")
            inputs.append(image);pending.append((expected[rid],target))
            if len(inputs)==4:flush()
            if len(records) and len(records)%100==0:print(f"Observer features {len(records)}",flush=True)
        flush()
        rejected["no_matched_predicted_face"]=len(set(expected)-seen)
        if not records:raise ValueError("No actual clean predicted-crop features")
        metadata=dict(identity,backbone_state_sha256=observer_sha,
            crop_convention="predicted ellipse aligned long_axis_square scale1.25 input256; deterministic no augmentation",
            clean_eligibility=inventory["clean_eligibility"],
            crop_cache_sha256=sha256(crop_path),clean_inventory_sha256=sha256(inventory_path))
        temp=features_path.with_suffix('.tmp')
        torch.save(dict(metadata=metadata,features=torch.cat(feature_parts),geometry=torch.stack(geometry_parts),
            support=torch.stack(support_parts),records=records),temp)
        temp.replace(features_path)
        del observer,feature_parts
        torch.cuda.empty_cache()
        finish(args.output,"features",identity,[features_path],records=len(records),excluded=dict(rejected),initialization=initialization)
    prior_output=args.output/"priors"
    if not prior_output.exists():
        fit_priors(Namespace(cache=features_path,inventory=inventory_path,annotations=annotation,
            backbone_checkpoint=parent,detector_checkpoint=detector,
            calibration_video=inventory["partition"]["calibration_video_ids"],
            output=prior_output,seed=42,quantile=.05))
    report=json.loads((prior_output/"preparation_report.json").read_text())
    if report["metadata"]["feature_cache_sha256"] != sha256(features_path):
        raise ValueError("Prior cache identity mismatch")
    if any(sha256(ROOT/f)!=h for f,h in identity["code_sha256"].items()):
        raise ValueError("Preparation code changed while running")
    write_json(args.output/"preparation_complete.json",dict(status="prepared_not_evaluated",
        identity=identity,training="not run",model_gate_success=False,
        prior_report_sha256=sha256(prior_output/"preparation_report.json"),candidates=report["candidates"]))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=ROOT/"configs/training_plans/messenger_geometry_prior_v1.yaml")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    run(parser.parse_args())
