#!/usr/bin/env python3
"""CPU-only, train-only audit of conditional object augmentation distributions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.training import make_dataset
from keypoint_comparison1.landmark_detection.occlusion_policy import ObjectOverlapPolicy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(records, policy):
    applied = [r for r in records if r.get("applied")]
    overlaps = [r["face_overlap"] for r in applied]
    counts = dict(light=sum(x < .1 for x in overlaps), medium=sum(.1 <= x < .2 for x in overlaps), strong=sum(.2 <= x <= .4 for x in overlaps))
    n = len(applied)
    return dict(requested=len(records), applied=n, skipped=len(records)-n,
        skip_fraction=(len(records)-n)/max(1,len(records)),
        severity_counts=counts, severity_fractions={k:v/max(1,n) for k,v in counts.items()},
        overlap_quantiles=dict(zip(["min","p10","p50","p90","max"],np.quantile(overlaps,[0,.1,.5,.9,1]).tolist())) if overlaps else {},
        requested_routes=dict(Counter(r.get("requested_route","error") for r in records)),
        applied_routes=dict(Counter(r.get("requested_route","error") for r in applied)),
        acceptance_reasons=dict(Counter(r.get("acceptance_reason") for r in applied)),
        skip_reasons=dict(Counter(r.get("reason", "error") for r in records if not r.get("applied"))),
        qualifying_keypoint_counts=dict(Counter(slot for r in applied for slot in r.get("qualifying_keypoint_slots",[]))))


def distribution_checks(summary, policy):
    return dict(nonempty=summary["applied"] > 0,
        skip_fraction_within_budget=summary["skip_fraction"] <= policy.maximum_skip_fraction,
        light_fraction_within_budget=summary["severity_fractions"]["light"] <= policy.maximum_light_fraction,
        strong_fraction_sufficient=summary["severity_fractions"]["strong"] >= policy.minimum_strong_fraction)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"keypoint_comparison1/configs/occlusion_conditional_object_audit.yaml")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,nargs="+",default=[1,12,40])
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43])
    args=parser.parse_args()
    torch.set_num_threads(1);cv2.setNumThreads(1)
    raw=load_landmark_config(args.config.resolve()).raw
    policy=ObjectOverlapPolicy.from_mapping(raw["data"]["augmentation"]["paired_occlusion"]["object_overlap_policy"])
    canonical=ROOT/"outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/train.json"
    data=json.loads(canonical.read_text())
    assert (len(data["images"]),len(data["annotations"]),len({i["source_video_id"] for i in data["images"]})) == (4237,7129,44)
    assert digest(canonical) == raw["data"]["continuous_source_inventory"]["train"]["sha256"]
    annotation=Path(raw["data"]["annotations"]["train"])
    assert digest(annotation) == raw["data"]["expected"]["train"]["sha256"]
    args.output.mkdir(parents=True,exist_ok=True)
    report_path=args.output/"distribution.json"
    if report_path.exists():raise FileExistsError("Use a fresh audit output directory")
    records=[];groups={};all_count=0;errors=[];start=time.monotonic()
    with (args.output/"samples.jsonl").open("w") as log:
        for seed in args.seeds:
            raw["train"]["seed"]=seed
            ds=make_dataset(raw,split="train",training=True)
            assert {s["image"]["file_name"] for s in ds.samples} <= {i["file_name"] for i in data["images"]}
            for epoch in args.epochs:
                ds.set_epoch(epoch)
                part=[]
                for index in range(len(ds)):
                    all_count+=1
                    if ds._paired_occlusion_rng(index) is None:continue
                    sample_seed=int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{index}:audit".encode()).digest()[:4],"big")
                    random.seed(sample_seed);np.random.seed(sample_seed);torch.manual_seed(sample_seed)
                    base=dict(seed=seed,epoch=epoch,sample_index=index,sample_seed=sample_seed,
                        annotation_id=ds.samples[index]["annotation"]["id"],image_id=ds.samples[index]["annotation"]["image_id"])
                    try:
                        image,target=ds[index]
                        meta=target["paired_occlusion"]
                        assert torch.isfinite(image).all() and torch.isfinite(target["clean_image"]).all()
                        if meta.get("applied"):
                            assert 0 < meta["face_overlap"] <= .40
                            assert meta["acceptance_reason"] in {"standard","exempt_material","keypoint_neighborhood"}
                            assert not target["synthetic_truncated"].any()
                        row=dict(base,**meta)
                    except Exception as e:
                        row=dict(base,applied=False,reason="exception",error=repr(e));errors.append(row)
                    records.append(row);part.append(row)
                    log.write(json.dumps(row,ensure_ascii=False)+"\n")
                summary=summarize(part,policy);groups[f"seed{seed}_epoch{epoch}"]=summary
                log.flush()
                print(json.dumps(dict(group=f"seed{seed}_epoch{epoch}",**summary)),flush=True)
    summary=summarize(records,policy)
    material_groups=defaultdict(list)
    for r in records:material_groups[r.get("material_id","unknown")].append(r)
    by_material={k:summarize(v,policy) for k,v in material_groups.items()}
    checks=distribution_checks(summary,policy)
    checks["no_generation_errors"]=not errors
    checks["every_seed_epoch_distribution_safe"]=all(all(distribution_checks(s,policy).values()) for s in groups.values())
    checks["every_material_skip_within_budget"]=all(s["skip_fraction"] <= policy.maximum_skip_fraction for s in by_material.values())
    # Per-material light ratios reflect the deliberate glasses exemption; report them, do not hide in aggregate.
    files=[args.config,Path(__file__),ROOT/"keypoint_comparison1/landmark_detection/occlusion_policy.py",ROOT/"keypoint_comparison1/landmark_detection/dataset.py",ROOT/"keypoint_comparison1/landmark_detection/occlusion_materials.py"]
    output=dict(status="passed" if all(checks.values()) else "failed",audit_type="train_augmentation_distribution_only",
        training_run=False,production_gate_success=False,epochs=args.epochs,seeds=args.seeds,
        coordinate_convention="pre-flip crop pixels xy; ellipse radii in pixels; local disk clipped to face ellipse",
        local_radius_face_fraction=policy.local_radius_face_fraction,minimum_local_coverage=policy.minimum_local_coverage,
        policy=raw["data"]["augmentation"]["paired_occlusion"]["object_overlap_policy"],
        provenance=dict(canonical_train=str(canonical),canonical_sha256=digest(canonical),train_adapter=str(annotation),train_adapter_sha256=digest(annotation),
            manifest=raw["data"]["augmentation"]["paired_occlusion"]["material_manifest"],manifest_sha256=raw["data"]["augmentation"]["paired_occlusion"]["material_manifest_sha256"],
            config_and_code_sha256={str(p):digest(p) for p in files}),
        total_instance_epoch_visits=all_count,requested_fraction_per_instance_epoch=len(records)/max(1,all_count),
        summary=summary,by_seed_epoch=groups,by_material=by_material,checks=checks,errors=errors,elapsed_seconds=time.monotonic()-start,
        limitations=["No model training or accuracy claim", "Not a production schema/gpu/overfit/small-scale success marker", "Binary support only; no semitransparent lens simulation", "Deterministic audit crop RNG differs from training worker order", "Request probability30% is per source image with at most one selected face; per-instance fraction has a different denominator"])
    report_path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(report_path,flush=True)
    return 0 if all(checks.values()) else 1


if __name__=="__main__":raise SystemExit(main())
