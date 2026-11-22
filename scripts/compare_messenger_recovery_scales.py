"""Matched post-crop recovery experiment on train-internal artificial pairs."""
from collections import defaultdict
from dataclasses import replace
import argparse
import copy
import fcntl
import hashlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from keypoint_comparison1.landmark_detection.dataset import PairedOcclusionConfig
from keypoint_comparison1.landmark_detection.recovery_reference import RecoveryReferenceMemory
from scripts.train_natural_feature_gate import (NaturalFeatureGate, load_model, build_crop_transform,
    warp_crop, image_to_tensor, ellipse_aligned_rectangle, run_head, digest, write_json)
from src.data.landmark_occlusion import apply_landmark_occlusion_numpy, transform_ellipse_affine


KINDS = ("rectangle","irregular","boundary")
PREVIOUS = ROOT/"outputs/diagnostics/natural_feature_gate_20261107"
AUDIT = ROOT/"outputs/diagnostics/natural_gate_artificial_restoration_20261107/results.json"
IMAGES = ROOT/"outputs/dataset_cache/face_pose_condition_balanced_811_latest_sources_r3_20261009"


@torch.no_grad()
def pairs(records,images,data,model,device):
    cfg=PairedOcclusionConfig.from_mapping(data["augmentation"]["paired_occlusion"]).landmark_mapping()
    columns=defaultdict(list)
    manifest=[]
    for a in records:
        im=images[a["image_id"]]
        raw=(IMAGES/im["file_name"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==im["metadata"]["sha256"]
        box,angle=ellipse_aligned_rectangle(a["ellipse"])
        crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        clean=warp_crop(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")),crop)
        ellipse=transform_ellipse_affine(a["ellipse"],crop.matrix)
        cx,cy,rx,ry,theta=ellipse
        yy,xx=np.mgrid[:16,:16]*16+7.5
        dx,dy=xx-cx,yy-cy
        support=torch.from_numpy(((dx*np.cos(theta)+dy*np.sin(theta))/rx)**2+((-dx*np.sin(theta)+dy*np.cos(theta))/ry)**2<=1)
        xy=crop.original_to_crop(np.array(a["keypoints"]).reshape(5,3)[:,:2])
        pixel=np.rint(xy).astype(int)
        valid=((pixel>=0)&(pixel<256)).all(1)
        ix,iy=np.clip(pixel,0,255).T
        def rgb(im):
            return image_to_tensor(im,mean=data["mean"],std=data["std"])
        variants=[]
        for kind in KINDS:
            seed=int.from_bytes(hashlib.sha256(f'42:{a["id"]}:O:{kind}'.encode()).digest()[:8],"big")
            image,mask,_,meta=apply_landmark_occlusion_numpy(clean,face_ellipse=ellipse,keypoints=xy,
                keypoint_visibility=np.full(5,2),config=cfg,rng=random.Random(seed),force=True,force_kind=kind)
            if not meta["applied"]:
                continue
            assert np.array_equal(clean[~mask],image[~mask])
            ratio=torch.from_numpy(mask.copy()).float().reshape(16,16,16,16).mean((1,3))
            variants.append(rgb(image))
            columns["covered"].append((ratio>0)&support)
            columns["outside"].append((ratio==0)&support)
            columns["point"].append(torch.from_numpy(valid & mask[iy,ix]))
            columns["xy"].append(torch.from_numpy(xy/255))
            manifest.append(dict(face=a["id"],video=im["metadata"]["source_video_id"],kind=kind,seed=seed))
        if variants:
            z=model.observer(rgb(clean)[None].to(device)).cpu()[0]
            columns["teacher"].extend([z]*len(variants))
            columns["raw"].extend(model.base.backbone(torch.stack(variants).to(device)).cpu().unbind())
    return {k:torch.stack(v) for k,v in columns.items()},manifest


def candidate(block,x):
    _,aux=block(x)
    return aux["candidate_tokens"].transpose(1,2).reshape_as(x)


def masked_mean(values,mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(block,model,gate,cache,manifest,device):
    rows=[]
    for ids in torch.arange(len(manifest)).split(16):
        x,z,cov,out,p,xy=[cache[k][ids].to(device) for k in ("raw","teacher","covered","outside","point","xy")]
        r=candidate(block,x)
        _,fused=gate(x,r,model.channel_adoption.sigmoid()[None,:,None,None])
        pred=run_head(model.base.head,fused)
        err=(pred["decoded_coordinates"]-xy).norm(dim=-1)*1.25
        success=(err<=.1)&(pred["presence_logits"].sigmoid()>=.7)&p
        cols=[cov.sum((1,2)),out.sum((1,2)),p.sum(1),success.sum(1),(err*p).sum(1)]
        for branch in (x,r,fused):
            cosine=1-F.cosine_similarity(branch,z,dim=1)
            mse=(branch-z).square().mean(1)
            cols.extend([(cosine*cov).sum((1,2)),(mse*cov).sum((1,2)),(cosine*out).sum((1,2))])
        rows.extend(torch.stack(cols,1).cpu().numpy())
    return np.array(rows,dtype=float)


def run(out,device):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(42)
    old=json.loads(AUDIT.read_text())
    gate_ck=torch.load(PREVIOUS/"gate.pt",map_location="cpu",weights_only=False)
    spec=json.loads((ROOT/"configs/training_plans/messenger_geometry_prior_runs/k3.json").read_text())
    parent=ROOT/spec["output_root"]
    model,prov=load_model(spec,json.loads((parent/"training_success.json").read_text())["identity"],parent/"checkpoints/best.pt",device)
    assert prov["sha256"]==old["parent_sha"] and digest(PREVIOUS/"gate.pt")==old["gate_sha"]
    model.requires_grad_(False).eval()
    gate=NaturalFeatureGate(model.channel_adoption.numel()).to(device).eval().requires_grad_(False)
    gate.load_state_dict(gate_ck["state_dict"])
    path=ROOT/"outputs/diagnostics/keypoint_side_revision_v3_20261031/annotations/deim_dino/face_keypoints_train.json"
    assert digest(path)==gate_ck["contract"]["annotations"]
    corpus=json.loads(path.read_text())
    faces={a["id"]:a for a in corpus["annotations"]}
    images={im["id"]:im for im in corpus["images"]}
    roles=gate_ck["contract"]["roles"]
    fit_ids=[i for i in roles["fit"]["ids"] if faces[i]["keypoints"][2::3]==[2]*5 and faces[i]["attributes"]["face_occluded"] is False]
    fit_ids=sorted(fit_ids,key=lambda i:hashlib.sha256(f'42:paired:{i}'.encode()).digest())[:256]
    assert set(old["ids"])<=set(roles["evaluation"]["ids"]) and not set(roles["fit"]["videos"])&set(roles["evaluation"]["videos"])
    data=spec["parent_config"]["data"]
    train,train_manifest=pairs([faces[i] for i in fit_ids],images,data,model,device)
    ev,manifest=pairs([faces[i] for i in old["ids"]],images,data,model,device)
    assert len(manifest)==sum(old["applied"].values())
    native=model.base.messenger_recovery
    results={"native":evaluate(native,model,gate,ev,manifest,device)}
    write_json(out/"contract.json",dict(parent=prov["sha256"],gate_sha=old["gate_sha"],crop="GT",fit=train_manifest,evaluation=manifest,
        seed=42,epochs=4,batch=16,lr=.0001,raw_scale=1,multi_scales=[1,2,4],outside_weight=.1,
        error="cosine distance + raw channel MSE, fixed clean observer",head_gate="frozen previously trained soft BCE + position"))
    for multi in (False,True):
        name="multiscale" if multi else "single"
        block=copy.deepcopy(native)
        cfg=replace(block.config,multi_scale=multi)
        block.config=cfg
        memory=RecoveryReferenceMemory(block.embed_dim,cfg)
        # Identical extra parameters in both arms; no positional offset at the start.
        torch.nn.init.zeros_(memory.key_position.weight);torch.nn.init.zeros_(memory.key_position.bias)
        block.reference_memory=memory.to(device)
        for param_name,p in block.named_parameters():
            p.requires_grad_(param_name.startswith(("layers.","decoder.","reference_memory.")) or param_name=="messenger_token")
        if not multi:
            with torch.no_grad():
                x=train["raw"][:2].to(device)
                assert torch.allclose(candidate(block,x),candidate(native,x),rtol=1.e-4,atol=1.e-5)
        opt=torch.optim.AdamW([p for p in block.parameters() if p.requires_grad],lr=.0001)
        rng=torch.Generator().manual_seed(42)
        for epoch in range(4):
            for ids in torch.randperm(len(train_manifest),generator=rng).split(16):
                x,z,cov,outside=[train[k][ids].to(device) for k in ("raw","teacher","covered","outside")]
                r=candidate(block,x)
                error=1-F.cosine_similarity(r,z,dim=1)+(r-z).square().mean(1)
                loss=masked_mean(error,cov)+.1*masked_mean(error,outside)
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(block.parameters(),5,error_if_nonfinite=True)
                opt.step()
            print(name,epoch+1,flush=True)
        torch.save({"state":block.state_dict(),"factors":block.reference_memory.factors,"common_parent":prov["sha256"]},out/(name+".pt"))
        results[name]=evaluate(block.eval(),model,gate,ev,manifest,device)
    np.savez_compressed(out/"paired_rows.npz",**results,columns=["covered","outside","covered_points","correct","error_sum",
        "X_cos","X_mse","X_outside_cos","R_cos","R_mse","R_outside_cos","Y_cos","Y_mse","Y_outside_cos"])
    summary={}
    for name,rows in results.items():
        n=rows.sum(0)
        summary[name]=dict(covered_patches=n[0],covered_points=n[2],correct=n[3],point_error=n[4]/n[2],
                           raw_distance=n[5]/n[0],candidate_distance=n[8]/n[0],candidate_mse=n[9]/n[0],
                           candidate_outside_distance=n[10]/n[1],fused_distance=n[11]/n[0])
    write_json(out/"evaluation.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    if args.device=="cuda":
        with (ROOT/"outputs/training_queues/pending_models_tests_then_training/queue.lock").open("a") as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.output,torch.device("cuda"))
    else:
        run(args.output,torch.device("cpu"))
