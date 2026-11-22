#!/usr/bin/env python3
"""CPU actual-loader audit for face-only training; never emits a model gate success."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.training import make_dataset
from keypoint_comparison1.landmark_detection.real_occlusion import digest


def summarize(rows, maximum_skip_fraction=.05):
    applied = [r for r in rows if r["applied"]]
    routes = Counter(r["requested_route"] for r in applied)
    by_route = {route: dict(requested=sum(r["requested_route"]==route for r in rows),
        applied=routes[route], skipped=sum(r["requested_route"]==route and not r["applied"] for r in rows))
        for route in ("light", "medium", "strong")}
    return dict(requested=len(rows), applied=len(applied), routes=dict(routes), by_route=by_route,
        skip_reasons=dict(Counter(r["reason"] for r in rows if not r["applied"])),
        checks=dict(nonempty=bool(applied),
            skip_within_limit=(len(rows)-len(applied))/max(1,len(rows))<=maximum_skip_fraction,
            light_at_most_30_percent=routes["light"]/max(1,len(applied))<=.30,
            strong_at_least_25_percent=routes["strong"]/max(1,len(applied))>=.25))


def distribution_passed(summary, groups, scope):
    # User 2026-10-12: the quoted aggregate request denominator has a 7% limit.
    # Keep subgroup skip outcomes visible; other distribution guards remain per group.
    return (len(groups)==6 and all(summary["checks"].values())
        and all(all(v for k,v in g["checks"].items()
            if scope != "aggregate" or k != "skip_within_limit") for g in groups.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"keypoint_comparison1/configs/continuous_recovery_abc_b_real_face_w48_s2.yaml")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-pairs",type=int,default=0,help="Bounded implementation smoke only; cannot pass full audit")
    args=parser.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    raw=load_landmark_config(args.config).raw
    assert raw["data"]["augmentation"]["paired_occlusion"]["appearance"]=="real_face"
    policy=raw.get("distribution_audit", dict(maximum_skip_fraction=.05, skip_fraction_scope="each_seed_epoch"))
    limit=float(policy["maximum_skip_fraction"])
    scope=policy["skip_fraction_scope"]
    if not 0 <= limit <= 1 or scope not in {"aggregate", "each_seed_epoch"}:
        raise ValueError("Invalid distribution audit policy")
    rows=[];groups={};stop=False
    for seed in [42,43]:
        raw["train"]["seed"]=seed
        ds=make_dataset(raw,split="train",training=True)
        for epoch in [1,12,40]:
            ds.set_epoch(epoch);group=[]
            for i in range(len(ds)):
                if ds._paired_occlusion_rng(i) is None:continue
                crop_seed=int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{i}:face-audit".encode()).digest()[:4],"big")
                random.seed(crop_seed);np.random.seed(crop_seed);torch.manual_seed(crop_seed)
                x,t=ds[i];meta=t["paired_occlusion"]
                assert torch.isfinite(x).all() and torch.isfinite(t["clean_image"]).all()
                assert torch.equal(x[:,~t["occlusion_mask"]],t["clean_image"][:,~t["occlusion_mask"]])
                assert not t["synthetic_truncated"].any()
                assert ds.real_occlusion_bank.eligible_sample(ds.samples[i])
                if meta["applied"]:
                    assert meta["donor_video"]!=meta["recipient_video"]
                    if meta["requested_route"] == "light":
                        assert 0<meta["overlap"]<.10 and meta["donor_face_overlap"]>0
                    else:
                        lo,hi = {"medium":(.10,.20), "strong":(.20,.40)}[meta["requested_route"]]
                        assert lo<meta["overlap"]<=hi and meta["donor_face_overlap"]>=.01
                    delta=meta["context_delta"]
                    assert abs(delta[0])<=12 and np.linalg.norm(delta[1:3])<=12 and abs(delta[3])<=15
                row=dict(seed=seed,epoch=epoch,index=i,crop_seed=crop_seed,**meta)
                rows.append(row);group.append(row)
                with (args.output/"samples.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
                if len(rows)%100==0:print(f"pairs {len(rows)}",flush=True)
                if args.max_pairs and len(rows)>=args.max_pairs:stop=True;break
            groups[f"{seed}:{epoch}"]=summarize(group, limit)
            if stop:break
        if stop:break
    summary=summarize(rows, limit)
    passed=not args.max_pairs and distribution_passed(summary, groups, scope)
    result=dict(status="passed" if passed else "failed" if not args.max_pairs else "bounded_smoke",
        audit_only=True,production_gate_success=False,training="not run",scope="full_rgb_actual_loader",
        distribution_policy=policy,by_seed_epoch=groups,**summary,augmentation_contract=raw.get("augmentation_contract"),config_sha256=digest(args.config),
        manifest_sha256=raw["data"]["augmentation"]["paired_occlusion"]["material_manifest_sha256"],
        implementation_sha256={name:digest(ROOT/name) for name in ["scripts/audit_real_face_occlusion.py",
            "keypoint_comparison1/landmark_detection/dataset.py","keypoint_comparison1/landmark_detection/real_face_occlusion.py",
            "scripts/prototype_cross_video_face_occlusion.py","scripts/opaque_boundary_gradient.py"]},
        limitations=["Deterministic audit RNG differs from production worker ordering.","Face pastes only; not all natural occlusion."])
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return 0 if passed or args.max_pairs else 1


if __name__=="__main__":raise SystemExit(main())
