#!/usr/bin/env python3
"""Real local DINO/checkpoint acceptance; CPU default to avoid active GPU jobs."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from keypoint_comparison1.landmark_detection.config import load_landmark_config, configured_keypoint_target_policy, configured_state_head_mode
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import initialize_from_finetune_checkpoint, load_landmark_checkpoint, make_dataset, seed_everything
from keypoint_comparison1.landmark_detection.continuous_teacher import file_hash
from keypoint_comparison1.landmark_detection.recovery_acceptance import common_state_hash
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss

ARMS=("a","b_texture","b_object","c1","c2","c3")


def config_path(arm):
    return ROOT/f"keypoint_comparison1/configs/continuous_recovery_abc_{arm}.yaml"


def run(device):
    seed_everything(42)
    images=torch.randn(2,3,256,256,device=device)
    records={}
    hashes=[]
    for arm in ARMS:
        loaded=load_landmark_config(config_path(arm))
        raw=loaded.raw
        seed_everything(42)
        model,_=build_landmark_model(raw,variant="query_refinement",initialize_from_detector=False)
        initialization=initialize_from_finetune_checkpoint(model,raw["train"]["finetune_checkpoint"],variant="query_refinement",
            expected_target_policy=configured_keypoint_target_policy(raw),expected_state_head_mode=configured_state_head_mode(raw))
        digest=common_state_hash(model)
        hashes.append(digest)
        model.to(device).eval()
        counts=[0]*12
        def hook(index):
            def count(module,args,out):
                counts[index]+=1
            return count
        handles=[block.register_forward_hook(hook(i)) for i,block in enumerate(model.backbone.backbone.blocks)]
        with torch.no_grad():
            recovered=model(images,return_recovery_diagnostics=True,return_recovery_attention=True)
        for handle in handles:
            handle.remove()
        assert counts == [1]*12,counts
        with torch.no_grad():
            raw_out=model(images,recovery_mode="raw")
            final=model.extract_backbone_feature_map(images)
            maps=model.backbone.forward_intermediate_feature_maps(images,layer_indexes=(3,7,11))
            torch.testing.assert_close(final,maps[-1],atol=1e-6,rtol=1e-5)
            direct_head=model.head(final)
            torch.testing.assert_close(raw_out["decoded_coordinates"],direct_head["decoded_coordinates"],atol=1e-6,rtol=1e-5)
        assert final.shape == (2,384,16,16)
        assert all(m.shape == final.shape for m in maps)
        diag=recovered["recovery_diagnostics"]
        assert torch.isfinite(recovered["decoded_coordinates"]).all()
        record=dict(common_initialization_sha256=digest,initialization=initialization,
            final_map_max_abs_error=float((final-maps[-1]).abs().max()),one_forward_block_counts=counts,
            reference_tokens=diag["attention"].shape[-1],configuration=raw["model"]["continuous_recovery"],
            parameters=sum(p.numel() for p in model.parameters()),
            reference_parameters=sum(p.numel() for p in model.messenger_recovery.reference_memory.parameters()) if model.messenger_recovery.reference_memory else 0)
        if arm in {"c2","c3"}:
            # Prime the zero-initialized recovery projections, then test ONLY final
            # keypoint loss (not candidate/Gram/alpha auxiliary losses) on real DINO.
            model.train()
            model.backbone.requires_grad_(False)
            optim=torch.optim.Adam(model.messenger_recovery.parameters(),lr=.002)
            for _ in range(3):
                out=model(images,return_recovery_diagnostics=True)
                d=out["recovery_diagnostics"]
                (d["recovered_tokens"]-final.detach().flatten(2).transpose(1,2)).square().mean().backward()
                optim.step(); optim.zero_grad(set_to_none=True)
            model.backbone.requires_grad_(True)
            truth=[dict(keypoints=torch.rand(5,2,device=device),labeled=torch.ones(5,dtype=torch.bool,device=device),
                        visible=torch.tensor([1,1,0,1,0],dtype=torch.bool,device=device)) for _ in range(2)]
            out=model(images)
            LandmarkLoss()(out,truth)["loss_coordinates"].backward()
            groups={"adapter":list(model.messenger_recovery.reference_memory.parameters()),
                    "crop_dino":list(model.backbone.parameters()),"gate":list(model.messenger_recovery.local_gate.parameters())}
            for i,layer in enumerate(model.messenger_recovery.layers):
                groups[f"alpha_{i+1}"]=list(layer.alpha.parameters())
                groups[f"kv_{i+1}"]=list(layer.k.parameters())+list(layer.v.parameters())
            grads={}
            for key,params in groups.items():
                values=[p.grad for p in params if p.grad is not None]
                assert values and all(torch.isfinite(g).all() for g in values),key
                grads[key]=sum(float(g.abs().sum()) for g in values)
                assert grads[key]>0,(key,grads[key])
            record["final_coordinate_loss_only_grad_l1"]=grads
        if arm == "a":
            disabled_raw=copy.deepcopy(raw)
            disabled_raw["model"]["continuous_recovery"]["enabled"]=False
            disabled,_=build_landmark_model(disabled_raw,variant="query_refinement",initialize_from_detector=False)
            load_landmark_checkpoint(disabled,raw["train"]["finetune_checkpoint"]["path"],strict=True)
            disabled.to(device).eval()
            with torch.no_grad():
                disabled_output=disabled(images)["decoded_coordinates"]
            torch.testing.assert_close(disabled_output,raw_out["decoded_coordinates"],atol=0,rtol=0)
            record["disabled_baseline_strict_load_and_raw_output_exact"]=True
            del disabled
            # Same checkpoint strict-loads under C0 and its historical config.
            checkpoint=ROOT/"outputs/landmark_detection/continuous_recovery_combined_option1/query_refinement/checkpoints/best_conditional_semantic_macro_f1.pt"
            load_landmark_checkpoint(model,checkpoint,strict=True)
            model.eval()
            with torch.no_grad():
                c0=model(images)["decoded_coordinates"].cpu()
            old=load_landmark_config(ROOT/"keypoint_comparison1/configs/continuous_recovery_combined_option1.yaml").raw
            legacy,_=build_landmark_model(old,variant="query_refinement",initialize_from_detector=False)
            load_landmark_checkpoint(legacy,checkpoint,strict=True)
            legacy.to(device).eval()
            with torch.no_grad():
                old_out=legacy(images)["decoded_coordinates"].cpu()
            torch.testing.assert_close(c0,old_out,atol=0,rtol=0)
            record["old_combined_strict_checkpoint_sha256"]=file_hash(checkpoint)
            record["c0_old_output_exact"]=True
            del legacy
        records[arm]=record
        print(f"{arm}: passed; keys={record['reference_tokens']} common={digest}",flush=True)
        del model,recovered,raw_out,diag,maps,final
        gc.collect()
    assert len(set(hashes)) == 1,"Common initialization diverged across arms"
    return dict(status="passed",device=str(device),seed=42,arms=records,
        topology="HEAD-free ordered cascade: full-image FACE/ellipse -> Face-crop DINO -> recovery -> Query Refinement",
        production_training="not run",gpu_safety="not measured by this CPU acceptance audit")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--success-marker",type=Path)
    args=parser.parse_args()
    if args.success_marker:
        args.success_marker.unlink(missing_ok=True)
    torch.set_num_threads(2)
    started=time.time()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(torch.device(args.device))
    except Exception as error:
        args.output.write_text(json.dumps(dict(status="failed",error=str(error)),indent=2)+"\n")
        raise
    result["elapsed_seconds"]=time.time()-started
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    if args.success_marker:
        args.success_marker.parent.mkdir(parents=True,exist_ok=True)
        args.success_marker.write_text(json.dumps(dict(status="passed",report=str(args.output),report_sha256=file_hash(args.output)),indent=2)+"\n")


if __name__=="__main__":
    main()
