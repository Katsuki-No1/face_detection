"""V3 post-crop research training. Select R before fitting the position-only gate."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import load_landmark_checkpoint
from keypoint_comparison1.landmark_detection.continuous_recovery import ContinuousRecoveryBlock
from scripts.train_natural_feature_gate import (build_crop_transform,ellipse_aligned_rectangle,warp_crop,
    NaturalFeatureGate,run_head,digest,write_json,tensor_digest)
from scripts.compare_messenger_recovery_scales import candidate,masked_mean
from src.data.v3_shape_occlusion import SHAPES,occlude


CONFIG=ROOT/"outputs/experiments/mal_alpha2_dataset_v3_query_pipeline/configs/query.yaml"
BASE=ROOT/"outputs/landmark_detection/mal_alpha2_dataset_v3_query_refinement/query_refinement/checkpoints/best_conditional_semantic_macro_f1.pt"
ANN=ROOT/"outputs/diagnostics/keypoint_side_revision_v3_20261031/annotations/deim_dino/face_keypoints_train.json"
OLD_SPLIT=ROOT/"outputs/diagnostics/natural_feature_gate_20261107/contract.json"


def stable(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"big")


class Crops:
    def __init__(self,records,images,root):
        self.records=records
        self.images=images
        self.root=root

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        a=self.records[index]
        im=self.images[a["image_id"]]
        raw=(self.root/im["file_name"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=im["metadata"]["sha256"]:
            raise ValueError("Changed V3 frame")
        rgb=np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        box,angle=ellipse_aligned_rectangle(a["ellipse"])
        crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        original=warp_crop(rgb,crop)
        kp=np.asarray(a["keypoints"]).reshape(5,3)
        coord=crop.original_to_crop(kp[:,:2])
        states=kp[:,2].astype(np.int64)
        valid=np.isfinite(coord).all(1)&((coord>=0)&(coord<=255)).all(1)
        states[~valid]=0
        coord[states==0]=0
        item=dict(i=index,image=original,xy=coord/255,state=states,synthetic=[])
        if (states==2).all() and a["attributes"]["face_occluded"] is False:
            cx,cy=crop.original_to_crop(np.array([a["ellipse"][:2]]))[0]
            rx,ry=np.array(a["ellipse"][2:4])*255/crop.side
            yy,xx=np.ogrid[:256,:256]
            face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
            support=face.reshape(16,16,16,16).mean((1,3))>=.5
            ix,iy=np.rint(coord).astype(int).T
            for typ,kind in enumerate(SHAPES):
                made=occlude(original,face,coord,kind,stable(f'v3:42:{a["id"]}:{kind}'))
                if made is None:
                    continue
                image,mask=made
                covered=mask.reshape(16,16,16,16).any((1,3))
                item["synthetic"].append((typ,image,support&covered,support&~covered,mask[iy,ix],(face&mask).sum()/face.sum()))
        return item


def collate(rows):
    return rows


def new_array(path,shape):
    return np.lib.format.open_memmap(path,"w+",dtype="float16",shape=shape)


@torch.no_grad()
def prepare(out, ds, eligible, model, data):
    storage=out/"features"
    storage.mkdir()
    n=len(ds)
    clean=new_array(storage/"clean.npy",(n,384,16,16))
    synth=new_array(storage/"synthetic.npy",(eligible*3,384,16,16))
    xy=np.empty((n,5,2),np.float32);state=np.empty((n,5),np.int64)
    mean=torch.tensor(data["mean"],device="cuda").view(1,3,1,1)
    std=torch.tensor(data["std"],device="cuda").view(1,3,1,1)
    def encode(images):
        value=torch.from_numpy(np.stack(images)).cuda().permute(0,3,1,2).float()/255
        return model.backbone((value-mean)/std).half().cpu().numpy()
    parents=[];types=[];covered=[];outside=[];points=[];fractions=[]
    for batch_id,batch in enumerate(DataLoader(ds,batch_size=16,collate_fn=collate,num_workers=6)):
        ids=[row["i"] for row in batch]
        clean[ids]=encode([row["image"] for row in batch])
        variants=[]
        for row in batch:
            i=row["i"];xy[i]=row["xy"];state[i]=row["state"]
            for typ,image,cov,ext,pts,frac in row["synthetic"]:
                parents.append(i);types.append(typ);covered.append(cov);outside.append(ext);points.append(pts);fractions.append(frac)
                variants.append(image)
        offset=len(parents)-len(variants)
        for j in range(0,len(variants),16):
            sub=variants[j:j+16];synth[offset+j:offset+j+len(sub)]=encode(sub)
        if batch_id%100==0:
            print("prepare",ids[-1]+1,"/",n,flush=True)
    clean.flush();synth.flush()
    np.savez(storage/"metadata.npz",xy=xy,state=state,parent=parents,kind=types,covered=covered,outside=outside,point=points,fraction=fractions)


def tensor(array,ids):
    v=torch.from_numpy(array[ids]).cuda()
    return v.float() if v.is_floating_point() else v


def chunks(ids,batch=32):
    for start in range(0,len(ids),batch):
        yield ids[start:start+batch]


def synthetic_errors(r,z):
    return 1-F.cosine_similarity(r,z,dim=1), (r-z).square().mean(1)


def run(out,refine_from=None):
    torch.manual_seed(42);torch.set_num_threads(2);cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32=True
    raw=yaml.safe_load(CONFIG.read_text()); data=raw["data"]
    assert digest(BASE)=="11b430b595e56395a14c266cb364bd50848b1a51b5b689548eb6749de0a23856"
    assert digest(ANN)==json.loads((ANN.parents[2]/"success.json").read_text())["files"]["annotations/deim_dino/face_keypoints_train.json"]
    model,_=build_landmark_model(raw,variant="query_refinement",initialize_from_detector=False)
    load_landmark_checkpoint(model,BASE,prefer_ema=True,strict=True)
    assert model.rgb_refinement is None and not model.intermediate_layers
    model.eval().requires_grad_(False).cuda()
    parent_hash=tensor_digest(model)
    corpus=json.loads(ANN.read_text()); images={i["id"]:i for i in corpus["images"]}
    faces=[a for a in corpus["annotations"] if a.get("ellipse") and a["num_keypoints"]>0]
    videos=np.array([images[a["image_id"]]["metadata"]["source_video_id"] for a in faces])
    held=set(json.loads(OLD_SPLIT.read_text())["heldout_videos"])
    dev=set(sorted(set(videos)-held,key=lambda v:stable("dev:42:"+v))[:16])
    roles=np.array(["eval" if v in held else "dev" if v in dev else "fit" for v in videos])
    assert all(images[a["image_id"]]["metadata"]["split"]=="train" for a in faces)
    assert set(roles)=={"eval","dev","fit"}
    eligible=sum(a["keypoints"][2::3]==[2]*5 and a["attributes"]["face_occluded"] is False for a in faces)
    conf=dict(seed=42,checkpoint=digest(BASE),annotation=digest(ANN),query_config=digest(CONFIG),
              encoder_head_initial_hash=parent_hash,recovery_init="random single-scale two-layer Messenger",
              channel=1.,epochs_R=12,epochs_g=8,batch=32,lr_R=.0003,lr_g=.0005,
              R_selection="minimum dev covered cosine+MSE, all three shapes",gate_selection="minimum dev all-point SmoothL1 beta=.02",
              max_R_patience=3,minimum_R_epochs=6,shape_coverage=[.1,.4],shapes=SHAPES,
              scope="GT crop, V3 train internal video split, historical V3 backbone exposed to train",
              feature_storage="FP16; FP32 optimization, TF32 matmul",roles={r:sorted(set(videos[roles==r])) for r in set(roles)})
    warm=None
    if refine_from is not None:
        prior=json.loads((refine_from/"contract.json").read_text())
        for key in ("checkpoint","annotation","query_config","encoder_head_initial_hash","roles","shapes","feature_storage"):
            if json.dumps(conf[key],sort_keys=True)!=json.dumps(prior[key],sort_keys=True):
                raise ValueError("Inconsistent refinement source: "+key)
        prior_faces=np.load(refine_from/"faces.npz")
        assert np.array_equal(prior_faces["ids"],[a["id"] for a in faces]) and np.array_equal(prior_faces["role"],roles)
        warm=torch.load(refine_from/"recovery.pt",weights_only=True,map_location="cpu")
        assert warm["parent"]==conf["checkpoint"]
        conf.update(epochs_R=36,lr_R=.0001,min_delta_R=.0001,max_R_patience=6,minimum_R_epochs=6,
                    warmstart_epoch=warm["epoch"],warmstart_sha=digest(refine_from/"recovery.pt"),
                    shared_features=str(refine_from/"features"),optimizer_start="fresh AdamW")
    out.mkdir(parents=True,exist_ok=False);write_json(out/"contract.json",conf)
    np.savez(out/"faces.npz",ids=[a["id"] for a in faces],videos=videos,role=roles)
    print("roles",np.unique(roles,return_counts=True),"eligible",eligible,flush=True)
    if refine_from is None:
        prepare(out,Crops(faces,images,Path(data["root"])),eligible,model,data)
    storage=(refine_from or out)/"features"; meta=dict(np.load(storage/"metadata.npz"))
    x=np.load(storage/"clean.npy",mmap_mode="r");src=np.load(storage/"synthetic.npy",mmap_mode="r")
    parent=meta["parent"];kind=meta["kind"]; synrole=roles[parent]
    rng=np.random.default_rng(42)
    block=ContinuousRecoveryBlock(384,dict(enabled=True,adoption="alpha")).cuda()
    if warm is not None:
        block.load_state_dict(warm["state"])
    opt=torch.optim.AdamW(block.parameters(),lr=conf["lr_R"])
    def recovery_epoch(ids,train):
        count=0.;total=0.
        with torch.set_grad_enabled(train):
            for b in chunks(ids):
                xx=tensor(src,b);z=tensor(x,parent[b]);cov=tensor(meta["covered"],b);ext=tensor(meta["outside"],b)
                rr=candidate(block,xx)
                cos,mse=synthetic_errors(rr,z); error=cos+mse
                if train:
                    loss=masked_mean(error,cov)+.1*masked_mean(error,ext)
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(block.parameters(),5,error_if_nonfinite=True);opt.step()
                total+=float((cov*error.detach()).sum());count+=int(cov.sum())
        return total/count
    rid=np.flatnonzero(synrole=="fit");did=np.flatnonzero(synrole=="dev")
    history=[];best=float("inf");best_epoch=0;meaningful_epoch=0;meaningful_score=float("inf")
    if warm is not None:
        best=recovery_epoch(did,False);meaningful_score=best
        torch.save(dict(state=block.state_dict(),epoch=0,config=vars(block.config),parent=conf["checkpoint"]),out/"recovery.pt")
        history.append(dict(epoch=0,dev=best,examples=0))
    for epoch in range(1,conf["epochs_R"]+1):
        # Each available type appears once per clean face every three epochs.
        use=rid[(parent[rid]+epoch-1)%3==kind[rid]]
        train_loss=recovery_epoch(rng.permutation(use),True)
        score=recovery_epoch(did,False)
        history.append(dict(epoch=epoch,fit=train_loss,dev=score,examples=len(use)))
        if score<best:
            best=score;best_epoch=epoch
            torch.save(dict(state=block.state_dict(),epoch=epoch,config=vars(block.config),parent=conf["checkpoint"]),out/"recovery.pt")
        if score<meaningful_score-conf.get("min_delta_R",0.):
            meaningful_score=score;meaningful_epoch=epoch
        write_json(out/"recovery_history.json",history)
        print("R",epoch,train_loss,score,flush=True)
        if epoch>=conf["minimum_R_epochs"] and epoch-meaningful_epoch>=conf["max_R_patience"]:break
    block.load_state_dict(torch.load(out/"recovery.pt",weights_only=True)["state"])
    block.eval().requires_grad_(False)
    rmap=new_array((storage if refine_from is None else out)/"natural_candidate.npy",x.shape)
    with torch.no_grad():
        for b in chunks(np.arange(len(x))):
            rmap[b]=candidate(block,tensor(x,b)).half().cpu().numpy()
    rmap.flush()
    gate=NaturalFeatureGate(384).cuda()
    opt=torch.optim.AdamW(gate.parameters(),lr=conf["lr_g"])
    def gate_epoch(ids,train):
        total=0.;count=0
        with torch.set_grad_enabled(train):
            for b in chunks(ids):
                xx=tensor(x,b);rr=tensor(rmap,b)
                logits,y=gate(xx,rr,1.)
                coords=run_head(model.head,y)["decoded_coordinates"]
                err=F.smooth_l1_loss(coords,tensor(meta["xy"],b),reduction="none",beta=.02).mean(-1)
                valid=tensor(meta["state"],b)>0
                loss=masked_mean(err,valid)
                if train:
                    if not (out/"gradient.json").exists():
                        norm=float(torch.autograd.grad(loss,logits,retain_graph=True)[0].norm())
                        assert np.isfinite(norm) and norm>0
                        write_json(out/"gradient.json",dict(position_to_logit_norm=norm))
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(gate.parameters(),5,error_if_nonfinite=True);opt.step()
                total+=float((err.detach()*valid).sum());count+=int(valid.sum())
        return total/count
    best=float("inf");history=[]
    for epoch in range(1,conf["epochs_g"]+1):
        train_loss=gate_epoch(rng.permutation(np.flatnonzero(roles=="fit")),True)
        score=gate_epoch(np.flatnonzero(roles=="dev"),False)
        history.append(dict(epoch=epoch,fit=train_loss,dev=score))
        if score<best:
            best=score
            torch.save(dict(state=gate.state_dict(),epoch=epoch,parent=conf["checkpoint"],R=digest(out/"recovery.pt")),out/"gate.pt")
        write_json(out/"gate_history.json",history);print("g",epoch,train_loss,score,flush=True)
    gate.load_state_dict(torch.load(out/"gate.pt",weights_only=True)["state"])
    gate.eval().requires_grad_(False)
    @torch.no_grad()
    def outputs(xx,rr,xy):
        logits,y=gate(xx,rr,1.)
        result=[]
        for branch in (xx,y):
            pred=run_head(model.head,branch)
            err=(pred["decoded_coordinates"]-xy).norm(dim=-1)*1.25
            result.extend([err,(err<=.1)&(pred["presence_logits"].sigmoid()>=.7)])
        cell=((xy*255+.5)/16).floor().long().clamp(0,15)
        result.append(logits[:,0].sigmoid()[torch.arange(len(xx),device="cuda")[:,None],cell[...,1],cell[...,0]])
        return y,torch.stack(result,-1).cpu().numpy()
    def summary(rows):
        return dict(points=len(rows),X_error=float(rows[:,0].mean()),X_correct=int(rows[:,1].sum()),
                    Y_error=float(rows[:,2].mean()),Y_correct=int(rows[:,3].sum()),g=float(rows[:,4].mean()))
    natural=[];ev=np.flatnonzero(roles=="eval")
    for b in chunks(ev):
        _,rows=outputs(tensor(x,b),tensor(rmap,b),tensor(meta["xy"],b));natural.extend(rows)
    natural=np.asarray(natural);state=meta["state"][ev]
    metrics=dict(natural={str(s):summary(natural[state==s]) for s in (1,2)},synthetic={})
    np.savez_compressed(out/"natural_points.npz",rows=natural,indices=ev,states=state)
    for typ,name in enumerate(SHAPES):
        ids=np.flatnonzero((synrole=="eval")&(kind==typ));sums=np.zeros(4);pts=[]
        for b in chunks(ids):
            xx=tensor(src,b);rr=candidate(block,xx);z=tensor(x,parent[b])
            y,rows=outputs(xx,rr,tensor(meta["xy"],parent[b]))
            pts.extend(rows[meta["point"][b]])
            cov=tensor(meta["covered"],b)
            sums+=np.array([int(cov.sum()),*[float(((1-F.cosine_similarity(f,z,dim=1))*cov).sum()) for f in (xx,rr,y)]])
        metrics["synthetic"][name]=dict(pairs=len(ids),patches=int(sums[0]),X_R_Y_cosine=(sums[1:]/sums[0]).tolist(),
            covered_points=summary(np.array(pts)),mean_coverage=float(meta["fraction"][ids].mean()))
    metrics["counts"]={r:dict(natural=int((roles==r).sum()),synthetic={k:int(np.sum((synrole==r)&(kind==i))) for i,k in enumerate(SHAPES)}) for r in ("fit","dev","eval")}
    assert tensor_digest(model)==parent_hash
    write_json(out/"evaluation.json",metrics)
    print(metrics,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--refine-from",type=Path)
    args=parser.parse_args()
    with (ROOT/"outputs/training_queues/pending_models_tests_then_training/queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.output,args.refine_from)
