"""Train and evaluate the post-crop adoption gate on a video split within V3 train."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.evaluate_messenger_geometry_prior import load_model, image_to_tensor, build_crop_transform, warp_crop
from src.data.face_crop_cache import ellipse_aligned_rectangle
from src.data.natural_gate_teacher import patch_teacher
from src.models.natural_feature_gate import NaturalFeatureGate, supervised_losses
from src.staged_occlusion.core import digest, tensor_digest, write_json


class NaturalCrops:
    def __init__(self, records, images, data, *, need_map=True):
        self.records, self.images, self.data = records, images, data
        self.need_map = need_map

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        a = self.records[i]
        im = self.images[a["image_id"]]
        raw = (ROOT/"outputs/dataset_cache/face_pose_condition_balanced_811_latest_sources_r3_20261009"/im["file_name"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != im["metadata"]["sha256"]:
            raise ValueError("Image digest mismatch")
        photo = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        box, rotation = ellipse_aligned_rectangle(a["ellipse"])
        crop = build_crop_transform(box, output_size=256, scale_factor=self.data["augmentation"]["scale_factor"], rotation_deg=rotation)
        k = np.asarray(a["keypoints"]).reshape(5,3)
        xy = crop.original_to_crop(k[:,:2]) / 255
        states = k[:,2].astype(int)
        states[~((xy>=0)&(xy<=1)&np.isfinite(xy)).all(1)] = 0
        image = image_to_tensor(warp_crop(photo,crop),mean=self.data["mean"],std=self.data["std"])
        coordinates = np.where(states[:,None]>0,xy,0).astype("float32")
        if not self.need_map:
            return image, coordinates, states
        target, weight = patch_teacher(a["ellipse"],k[:,:2],states,crop)
        return image, target, weight, coordinates, states


@torch.no_grad()
def cache_features(dataset, base, device):
    cache = [[] for _ in range(6 if dataset.need_map else 4)]
    for image, *targets in DataLoader(dataset,batch_size=16,num_workers=2):
        x = base.backbone(image.to(device))
        _, aux = base.messenger_recovery(x)
        r = aux["candidate_tokens"].transpose(1,2).reshape_as(x)
        for col,arr in zip(cache,[x.cpu(),r.cpu(),*targets]):
            col.append(arr)
    return [torch.cat(col) for col in cache]


def batches(cache, order, device):
    for ids in order.split(16):
        yield [x.index_select(0,ids).to(device) for x in cache]


def run_head(head, fmap):
    return head(fmap,return_attention=False,augmentation_valid_mask=None,refinement_feature_maps=None)


@torch.no_grad()
def evaluate(gate, model, channel, cache, device, *, return_rows=False):
    rows = []
    ce_sum = mass = squared = 0.
    for x,r,target,weight,xy,state in batches(cache,torch.arange(len(cache[0])),device):
        logits, fused = gate(x,r,channel)
        p = logits.sigmoid()
        ce_sum += float((F.binary_cross_entropy_with_logits(logits[:,0],target,reduction="none")*weight).sum())
        squared += float(((p[:,0]-target).square()*weight).sum())
        mass += float(weight.sum())
        # Nearest containing feature patch, consistent with DINO's 16-pixel cells.
        ij = ((xy*255+.5)/16).floor().long().clamp(0,15)
        scores = p[torch.arange(len(x),device=device)[:,None],0,ij[...,1],ij[...,0]]
        metrics=[state,scores]
        for features in (x,fused):
            out=run_head(model.base.head,features)
            # Denominator is the GT ellipse major diameter (square crop side / 1.25).
            error=(out["decoded_coordinates"]-xy).norm(dim=-1)*1.25
            metrics.extend([error,out["presence_logits"].sigmoid()])
        rows.append(torch.stack(metrics,-1).cpu().numpy())
    data=np.concatenate(rows)
    points=data[data[:,:,0]>0]
    truth=points[:,0]==1
    g=points[:,1]
    pos=int(truth.sum()); neg=len(g)-pos
    tp=int(((g>=.5)&truth).sum()); fp=int(((g>=.5)&~truth).sum())
    result=dict(faces=len(data),soft_bce=ce_sum/mass,soft_rmse=(squared/mass)**.5,
                point_auc=float((rankdata(g)[truth].sum()-pos*(pos+1)/2)/(pos*neg)),
                confusion=dict(tp=tp,fp=fp,fn=pos-tp,tn=neg-fp))
    for name,mask in [("occluded",truth),("visible",~truth)]:
        z=points[mask]
        result[name]=dict(points=len(z),mean_gate=float(z[:,1].mean()))
        for mode,idx in [("zero",2),("learned",4)]:
            success=(z[:,idx]<=.1)&(z[:,idx+1]>=.7)
            result[name][mode]=dict(mean_error_diameter=float(z[:,idx].mean()),
                                    correct=int(success.sum()),recall=float(success.mean()))
    return (result,data) if return_rows else result


def train_position_ablation(out, model, channel, fit, ev, contract, device):
    # Both runs start with the same RNG state as the original gate run after caching.
    initial_rng = torch.get_rng_state()
    for name in ("all_points", "balanced_control"):
        torch.set_rng_state(initial_rng)
        run = out / name
        run.mkdir()
        gate = NaturalFeatureGate(fit[0].shape[1]).to(device)
        opt = torch.optim.AdamW(gate.parameters(), lr=contract["lr"])
        history = []
        for epoch in range(contract["epochs"]):
            total = 0.
            for i, (x, r, xy, state) in enumerate(batches(fit, torch.randperm(len(fit[0])), device)):
                logits, fused = gate(x, r, channel)
                coordinates = run_head(model.base.head, fused)["decoded_coordinates"]
                error = F.smooth_l1_loss(coordinates, xy, reduction="none", beta=.02).mean(-1)
                if name == "all_points":
                    valid = state > 0
                    loss = (error * valid).sum() / valid.sum().clamp_min(1)
                else:
                    groups = [error[state == v].mean() for v in (1, 2) if (state == v).any()]
                    loss = torch.stack(groups).mean() if groups else error.sum() * 0
                if epoch == 0 and i == 0:
                    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0].norm().item()
                    assert np.isfinite(grad) and grad > 0
                    write_json(run/"gradient_check.json", {"position_to_gate_logit":grad})
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 5, error_if_nonfinite=True)
                opt.step()
                total += loss.item() * len(x)
            history.append(total / len(fit[0]))
            write_json(run/"history.json",history)
            print(name, epoch+1, history[-1], flush=True)
        torch.save(dict(state_dict=gate.state_dict(),contract=dict(contract,point_weighting=name)),run/"gate.pt")
        metrics,rows=evaluate(gate.eval(),model,channel,ev,device,return_rows=True)
        write_json(run/"evaluation.json",metrics)
        np.savez_compressed(run/"paired_points.npz",ids=contract["roles"]["evaluation"]["ids"],rows=rows)
    assert tensor_digest(model) == contract["frozen_hash"] and all(p.grad is None for p in model.parameters())


def main(out, device, *, position_only=False):
    torch.manual_seed(42)
    torch.set_num_threads(6 if device.type=="cpu" else 2)
    specfile=ROOT/"configs/training_plans/messenger_geometry_prior_runs/k3.json"
    spec=json.loads(specfile.read_text())
    parent=ROOT/spec["output_root"]; ckpt=parent/"checkpoints/best.pt"
    identity=json.loads((parent/"training_success.json").read_text())["identity"]
    if digest(ckpt)!="ef7dc88c1395e1871d432114003edab526f5ea61061a82d7bcad195934d81fa7" or digest(specfile)!=identity["config_sha256"]:
        raise ValueError("Parent identity mismatch")
    model,_=load_model(spec,identity,ckpt,device)
    model.requires_grad_(False).eval()
    assert model.base.rgb_refinement is None
    channel=model.channel_adoption.sigmoid()[None,:,None,None].detach()
    frozen_hash=tensor_digest(model)
    annotation=ROOT/"outputs/diagnostics/keypoint_side_revision_v3_20261031/annotations/deim_dino/face_keypoints_train.json"
    if digest(annotation)!=json.loads((annotation.parents[2]/"success.json").read_text())["files"]["annotations/deim_dino/face_keypoints_train.json"]:
        raise ValueError("Train annotation identity mismatch")
    corpus=json.loads(annotation.read_text())
    images={im["id"]:im for im in corpus["images"]}
    assert all(im["metadata"]["split"]=="train" for im in images.values())
    def video(a):
        return images[a["image_id"]]["metadata"]["source_video_id"]
    rng=np.random.default_rng(42)
    faces=[a for a in corpus["annotations"] if a.get("num_keypoints",0)>0 and a.get("ellipse")]
    videos=rng.permutation(sorted({video(a) for a in faces}))
    held=set(videos[:max(1,len(videos)//5)])
    groups=[]
    for cutoff,scope in [(2048,False),(768,True)]:
        pool=[a for a in faces if (video(a) in held)==scope]
        groups.append([pool[i] for i in rng.choice(len(pool),min(len(pool),cutoff),replace=False)])
    contract=dict(seed=42,epochs=8,batch=16,lr=.0005,bce_weight=.1,coordinate_weight=1.,smooth_l1_beta=.02,
                  crop="GT ellipse aligned, scale 1.25",device=str(device),parent_sha=digest(ckpt),
                  annotations=digest(annotation),frozen_hash=frozen_hash,
                  heldout_videos=sorted(held),roles={})
    for name,records in zip(("fit","evaluation"),groups):
        contract["roles"][name]=dict(videos=sorted({video(a) for a in records}),
                                     ids=[a["id"] for a in records])
    if position_only:
        contract.update(bce_weight=0., objective="position_only", fit_patch_teacher=False)
        previous=json.loads((ROOT/"outputs/diagnostics/natural_feature_gate_20261107/contract.json").read_text())
        assert all(contract[k]==previous[k] for k in ("roles","frozen_hash","annotations","batch","epochs","seed","lr"))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"contract.json",contract)
    cached=[]
    for i,records in enumerate(groups):
        dataset=NaturalCrops(records,images,spec["parent_config"]["data"],need_map=(not position_only or i==1))
        cached.append(cache_features(dataset,model.base,device))
        print("Cached",len(records),"faces",flush=True)
    fit,ev=cached
    if position_only:
        return train_position_ablation(out,model,channel,fit,ev,contract,device)
    gate=NaturalFeatureGate(fit[0].shape[1]).to(device)
    opt=torch.optim.AdamW(gate.parameters(),lr=contract["lr"])
    log=[]
    for epoch in range(contract["epochs"]):
        sums=np.zeros(2); seen=0
        for x,r,t,w,xy,state in batches(fit,torch.randperm(len(fit[0])),device):
            logits,fused=gate(x,r,channel)
            coords=run_head(model.base.head,fused)["decoded_coordinates"]
            ce,position=supervised_losses(logits,coords,t,w,xy,state)
            if not log and seen==0:
                norms=[float(torch.autograd.grad(loss,logits,retain_graph=True)[0].norm()) for loss in (ce,position)]
                assert all(np.isfinite(n) and n>0 for n in norms)
                write_json(out/"gradient_check.json",dict(bce=norms[0],position=norms[1]))
            loss=contract["bce_weight"]*ce+position
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),5,error_if_nonfinite=True)
            opt.step()
            sums+=np.array([ce.item(),position.item()])*len(x);seen+=len(x)
        log.append(dict(epoch=epoch+1,bce=sums[0]/seen,position=sums[1]/seen))
        write_json(out/"history.json",log)
        print(log[-1],flush=True)
    torch.save(dict(state_dict=gate.state_dict(),contract=contract),out/"gate.pt")
    metrics=evaluate(gate.eval(),model,channel,ev,device)
    assert tensor_digest(model)==frozen_hash and all(p.grad is None for p in model.parameters())
    write_json(out/"evaluation.json",metrics)
    print(json.dumps(metrics),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--position-only",action="store_true")
    args=parser.parse_args()
    if args.device=="cpu":
        main(args.output,torch.device("cpu"),position_only=args.position_only)
    else:
        with (ROOT/"outputs/training_queues/pending_models_tests_then_training/queue.lock").open("a") as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main(args.output,torch.device("cuda"),position_only=args.position_only)
