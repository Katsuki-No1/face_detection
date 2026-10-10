#!/usr/bin/env python3
"""Validation-only ABC diagnostics, separate from common deployed bbox metrics.

Uses the model's complete student forward for every reference configuration.
Artificial errors use the immutable baseline teacher, never the current student
as the clean target. Natural point groups exclude artificially covered points.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.audit_messenger_feature_recovery import errors
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import PairedOcclusionConfig, image_to_tensor
from keypoint_comparison1.landmark_detection.evaluation import evaluate_gt_boxes
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import load_landmark_checkpoint, make_dataset, seed_everything
from keypoint_comparison1.landmark_detection.continuous_teacher import file_hash
from keypoint_comparison1.landmark_detection.messenger_recovery import patch_occlusion_ratios
from keypoint_comparison1.landmark_detection.occlusion_materials import apply_material_occlusion
from src.data.landmark_occlusion import apply_landmark_occlusion_numpy, transform_ellipse_affine


def summarize(values):
    a=np.asarray(values,dtype=np.float64).ravel()
    if not len(a):
        return dict(count=0,mean=None,unsupported_reason="empty diagnostic group")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite diagnostic values")
    return dict(count=len(a),mean=float(a.mean()),p05=float(np.quantile(a,.05)),p50=float(np.quantile(a,.5)),
                p95=float(np.quantile(a,.95)),fraction_below_001=float((a<.01).mean()),fraction_above_099=float((a>.99).mean()))


def run(config_path, device, batch_size=8, limit=128, natural_limit=None):
    seed_everything(42)
    loaded=load_landmark_config(config_path)
    raw=copy.deepcopy(loaded.raw)
    inventory=raw["data"]["continuous_source_inventory"]["val"]
    if file_hash(ROOT/inventory["path"]) != inventory["sha256"]:
        raise ValueError("Canonical validation source inventory changed")
    checkpoint=loaded.output_root/"query_refinement/checkpoints"/raw["train"]["checkpoint_selection"]["filename"]
    model,_=build_landmark_model(raw,variant="query_refinement",initialize_from_detector=False)
    load_landmark_checkpoint(model,checkpoint,strict=True)
    model.to(device).eval()
    teacher=copy.deepcopy(model.backbone).eval().requires_grad_(False)
    baseline=raw["train"]["finetune_checkpoint"]
    if file_hash(baseline["path"]) != baseline["sha256"]:
        raise ValueError("Fixed teacher identity changed")
    state=torch.load(baseline["path"],map_location="cpu",weights_only=False)
    state=state.get("ema") or state["model"]
    teacher.load_state_dict({k[len("backbone."):]:v for k,v in state.items() if k.startswith("backbone.")},strict=True)
    val=raw["data"]["evaluation_sets"]["v2_val"]
    if file_hash(val["annotation"]) != val["sha256"]:
        raise ValueError("Validation annotation changed")
    raw["data"]["annotations"]["val"]=val["annotation"]
    raw["data"].setdefault("image_roots",{})["val"]=val["root"]
    raw["data"].pop("validation_detector_crop_cache",None)
    source=make_dataset(raw,split="val",training=False)
    natural_source=source if natural_limit is None else torch.utils.data.Subset(source,range(min(natural_limit,len(source))))
    natural={}
    for mode in ("raw","recovered"):
        metrics,_=evaluate_gt_boxes(model,natural_source,device=device,batch_size=batch_size,num_workers=0,
            input_size=256,attention_batches=0,recovery_mode=mode)
        natural[mode]={k:v for k,v in metrics.items() if isinstance(v,(int,float)) and np.isfinite(v)}
    populations={};feature_errors={};groups={};sample_manifest=[]
    def collect(destination,key,values):
        destination.setdefault(key,[]).extend(values.detach().float().cpu().flatten().tolist())
    cfg=PairedOcclusionConfig.from_mapping(raw["data"]["augmentation"]["paired_occlusion"])
    with torch.inference_mode():
        for index in range(min(limit,len(source))):
            clean,target=source[index]
            pixels=np.rint(np.clip(clean.permute(1,2,0).numpy()*np.asarray(source.std)+np.asarray(source.mean),0,1)*255).astype(np.uint8)
            points=target["keypoints"].numpy()*255
            states=target["keypoint_visibility"].numpy()
            ellipse=transform_ellipse_affine(target["gt_ellipse"].numpy(),target["affine"].numpy())
            z=teacher(clean[None].to(device)).flatten(2).transpose(1,2)
            for kind in ("natural","rectangle","irregular","boundary"):
                seed=int.from_bytes(hashlib.sha256(f"42:{int(target['annotation_id'])}:abc:{kind}".encode()).digest()[:8],"big")
                if kind == "natural":
                    image=clean
                    mask=np.zeros((256,256),dtype=bool)
                    meta=dict(applied=False)
                else:
                    kwargs=dict(face_ellipse=ellipse,keypoints=points,keypoint_visibility=states,rng=random.Random(seed))
                    if cfg.appearance == "solid":
                        augmented,mask,_,meta=apply_landmark_occlusion_numpy(pixels,**kwargs,config=cfg.landmark_mapping(),force=True,force_kind=kind)
                    else:
                        augmented,mask,_,meta=apply_material_occlusion(pixels,**kwargs,config=cfg,force_kind=kind)
                    if not meta.get("applied"):
                        raise ValueError("Diagnostic augmentation did not produce a valid mask")
                    image=image_to_tensor(augmented,mean=source.mean,std=source.std)
                out=model(image[None].to(device),return_recovery_diagnostics=True,return_recovery_attention=True)
                d=out["recovery_diagnostics"]
                ratio=patch_occlusion_ratios(torch.from_numpy(mask)[None].to(device),patch_size=16)[0]
                px=np.clip(np.rint(points[:,0]).astype(int),0,255);py=np.clip(np.rint(points[:,1]).astype(int),0,255)
                point_patches=torch.as_tensor((py//16)*16+px//16,device=device)
                point_artificial=mask[py,px]
                values={f"alpha_{i+1}":a for i,a in enumerate(d["layer_alpha"][0])}
                values.update(g=d["adoption_gate"][0],correction=d["structure_correction"][0])
                for key,value in values.items():
                    collect(populations,f"{kind}/artificial/{key}",value[ratio>0])
                    for label,state_value in (("natural_occluded",1),("visible",2)):
                        eligible=torch.as_tensor((states==state_value)&~point_artificial,device=device)
                        collect(populations,f"{kind}/{label}/{key}",value[point_patches[eligible]])
                if kind != "natural":
                    for branch,key in (("X","original_tokens"),("R","candidate_tokens"),("Y","recovered_tokens")):
                        for metric,error in errors(d[key],z,model.messenger_recovery.channel_scale).items():
                            collect(feature_errors,f"{kind}/{branch}/{metric}",error[0,ratio>0])
                for group,mass in zip(d.get("reference_groups",[]),d.get("reference_attention_mass",torch.empty(1,2,0))[0].unbind(-1)):
                    label=f"{kind}/level{group['level']}/factor{group['factor']}"
                    collect(groups,label,mass)
                sample_manifest.append({**meta,"index":index,"annotation_id":int(target["annotation_id"]),"kind":kind,"seed":seed})
    return dict(status="passed",diagnostic_only=True,split="val",seed=42,
        denominator="GT ellipse crops, unmodified validation for natural metrics; not deployed FACE bbox detection",
        interpretation="Raw/recovered is within-checkpoint inference contribution; teacher retains natural occlusion; no selection or promotion",
        canonical_annotation_sha256=raw["data"]["continuous_source_inventory"]["val"]["sha256"],
        annotation=val,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        teacher=baseline,config=str(config_path),config_sha256=file_hash(config_path),
        reference=raw["model"]["continuous_recovery"],appearance=cfg.appearance,
        reference_groups=d.get("reference_groups",[dict(level=11,factor=1,tokens=256)]),
        natural_samples=len(natural_source),feature_samples=min(limit,len(source)),partial_diagnostic=natural_limit is not None,
        coordinate_convention="crop x right/y down; point patch floor(pixel/16); natural states exclude artificial support",
        natural_raw_recovered=natural,populations={k:summarize(v) for k,v in populations.items()},
        artificial_feature_errors={k:summarize(v) for k,v in feature_errors.items()},
        reference_attention_mass={k:summarize(v) for k,v in groups.items()},sample_manifest=sample_manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    success=args.output.parent/"success.json"
    success.unlink(missing_ok=True)
    started=time.time()
    result=run(args.config,torch.device(args.device))
    result["elapsed_seconds"]=time.time()-started
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    success.write_text(json.dumps(dict(status="passed",report=str(args.output),sha256=file_hash(args.output)),indent=2)+"\n")


if __name__=="__main__":
    main()
