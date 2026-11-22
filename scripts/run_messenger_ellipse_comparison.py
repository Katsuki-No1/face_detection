"""Frozen baseline FACE DEIM: raw crop DINO versus ellipse-trained ViT/DF."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_messenger_multiaxis import sha,write,status,batches,gpu,ANN,PRIOR,stable
from scripts.run_messenger_multiaxis_convergence import Plateau,save,signature,set_lr
from scripts.train_natural_feature_gate import tensor_digest
from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import load_checkpoint,load_config
from src.models.messenger_multiaxis import MultiAxisRecovery,region_mean
from src.models.messenger_feature_diffusion import FeatureDiffusion
from src.models.messenger_ellipse_probe import FrozenEllipseProbe,ellipse_parts
from src.data.v3_shape_occlusion import SHAPES,occlude

OUT=ROOT/'outputs/diagnostics/messenger_ellipse_comparison_20261119'
CROPS=ROOT/'outputs/conditional_asm_v2/train_predicted_crops.json'
PRED=ROOT/'outputs/experiments/mal_alpha2_dataset_v3_query_pipeline/preparation/train_predictions.json'
REGISTRY=ROOT/'configs/model_registry/local_baseline.yaml'
PLAN=dict(seed=42,arms=['raw','regression','diffusion'],trainable='recovery only; no g, fixed DINO/STA/encoder/DEIM decoder',
    topology='baseline full-image FACE predictions -> axis-aligned predicted-box crop DINO -> optional final-block recovery -> fixed baseline FACE DEIM',
    scope='conditional crop research on matched detected train faces, no production/global val/test',
    crop_side=256,crop_expansion=1.5,crop_rotation=0,levels=[5,8,11],replacement='entire block11 map only; shallow two DINO paths and STA detail unchanged',
    main='baseline final ellipse center/axes SmoothL1, axial angle cosine, contour Chamfer; top FACE-score query without GT selection',
    mse_weight=.01,cosine_weight=.01,feature_region='covered + 0.1 ellipse-uncovered',
    ellipse_angle_reduction='mean over all faces; circular targets contribute zero, fixed denominator',
    fit_faces=256,dev_faces=128,eval_faces=128,natural_faces=dict(fit=512,dev=128,eval=512),
    microbatch=4,effective_batch=16,lr=1e-4,weight_decay=.01,clip=5.,steps=16,
    max_epochs=600,min_epochs=40,lr_floor=1e-6,reduce_patience=8,stop_patience=15,
    selection='minimum synthetic-dev ellipse loss; no feature error for checkpoint selection',
    stopping='ellipse and total-objective plateau at minimum learning rate, not mathematical convergence',
    sample_seeds=[42,1042,2042,3042],clean_head=False,retention='all best/last until all numerical evaluations complete; no deletion')


def read(p):return json.loads(Path(p).read_text())


def predicted_crops_verified():
    data=read(CROPS);meta=data['metadata'];registry=yaml.safe_load(REGISTRY.read_text())
    provenance=read(PRED.with_suffix('.provenance.json'))
    assert meta['detector_checkpoint_sha256']==registry['checkpoint']['sha256']==provenance['identity']['checkpoint_sha256']
    assert sha(PRED)==provenance['predictions_sha256']
    rows={r['assigned_annotation_id']:r for group in data['predictions'].values() for r in group}
    wanted={r['source_prediction_index']:r for r in rows.values()}
    # Stream the 895 MiB pretty-printed array. Never materialize all predictions.
    index=-1;buffer=None;verified=0
    with PRED.open() as f:
        for line in f:
            if line.rstrip()=='  {':
                index+=1;buffer=[line] if index in wanted else None
            elif buffer is not None:
                buffer.append(line)
                if line.rstrip() in ('  },','  }'):
                    original=json.loads(''.join(buffer).rstrip().rstrip(','));r=wanted[index]
                    for key in ('bbox_xyxy','ellipse','score'):
                        np.testing.assert_allclose(original[key],r[key],atol=0,rtol=0)
                    r['image_id']=original['image_id'];verified+=1;buffer=None
    assert verified==len(rows)
    return rows


def make_crop(annotation,image_info,prediction,root):
    raw=(root/image_info['file_name']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==image_info['metadata']['sha256']
    rgb=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
    x1,y1,x2,y2=prediction['bbox_xyxy'];side=max(x2-x1,y2-y1)*PLAN['crop_expansion']
    assert side>0
    center=np.array([(x1+x2)/2,(y1+y2)/2]);scale=256/side
    matrix=np.array([[scale,0,128-center[0]*scale],[0,scale,128-center[1]*scale]],np.float64)
    crop=cv2.warpAffine(rgb,matrix,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(123,116,103))
    ell=np.array(annotation['ellipse'],np.float64);ell[:2]=ell[:2]*scale+matrix[:,2];ell[2:4]*=scale
    if ell[2]<ell[3]:ell[2:4]=ell[[3,2]];ell[4]+=np.pi/2
    ell[4]=(ell[4]+np.pi/2)%np.pi-np.pi/2
    yy,xx=np.mgrid[:256,:256];dx=xx-ell[0];dy=yy-ell[1];c=np.cos(ell[4]);s=np.sin(ell[4])
    face=((c*dx+s*dy)/ell[2])**2+((-s*dx+c*dy)/ell[3])**2<=1
    kp=np.array(annotation['keypoints']).reshape(5,3);xy=kp[:,:2]*scale+matrix[:,2]
    synth=[]
    if (kp[:,2]==2).all() and annotation['attributes']['face_occluded'] is False:
        for kind in SHAPES:
            made=occlude(crop,face,xy,kind,stable(f'ellipse:42:{annotation["id"]}:{kind}'))
            if made is None:break
            im,mask=made;support=face.reshape(16,16,16,16).mean((1,3))>=.5
            covered=mask.reshape(16,16,16,16).any((1,3))
            if not (covered&support).any() or not (~covered&support).any():break
            synth.append((im,covered&support,~covered&support))
    return dict(rgb=crop,target=(ell/np.array([256,256,256,256,1])).astype(np.float32),synth=synth,
                crop_box=[*(center-side/2),*(center+side/2)],image_id=annotation['image_id'])


def choose_records(cfg):
    predictions=predicted_crops_verified();corpus=read(ANN);images={x['id']:x for x in corpus['images']}
    split={video:role for role,videos in read(PRIOR)['roles'].items() for video in videos}
    pool=sorted([a for a in corpus['annotations'] if a.get('ellipse') and a.get('num_keypoints',0)>0],key=lambda a:stable(f'multiaxis:42:{a["id"]}'))
    selected={};roles={};skips={};root=Path(cfg['data']['image_root'])
    for role in ('fit','dev','eval'):
        syn=[];natural=[];missing=0;invalid=0
        target=PLAN[role+'_faces'];ntarget=PLAN['natural_faces'][role]
        for a in pool:
            info=images[a['image_id']]
            if split.get(info['metadata']['source_video_id'])!=role:continue
            assert info['metadata']['split']=='train'
            if a['id'] not in predictions:missing+=1;continue
            assert predictions[a['id']]['image_id']==a['image_id']
            need_natural=len(natural)<ntarget
            need_syn=len(syn)<target and a['keypoints'][2::3]==[2]*5 and a['attributes']['face_occluded'] is False
            if not need_natural and not need_syn:continue
            crop=make_crop(a,info,predictions[a['id']],root)
            if need_natural:natural.append(a['id'])
            if need_syn:
                if len(crop['synth'])==3:syn.append(a['id'])
                else:invalid+=1
            if a['id'] in natural or a['id'] in syn:selected[a['id']]=(a,crop)
            if len(syn)==target and len(natural)==ntarget:break
        assert len(syn)==target and len(natural)==ntarget
        roles[role]=dict(synthetic=syn,natural=natural);skips[role]=dict(no_matched_detection=missing,invalid_synthetic=invalid)
    return selected,roles,skips,images


@torch.no_grad()
def make_cache(probe,cfg):
    selected,roles,skips,images=choose_records(cfg);ids=sorted(selected);index={fid:i for i,fid in enumerate(ids)}
    synids=set(x for r in roles.values() for x in r['synthetic'])
    clean=[];synthetic=[];parents=[];covered=[];outside=[];kinds=[]
    for fid in ids:
        a,crop=selected[fid];clean.append(crop['rgb'])
        if fid in synids:
            for kind,(im,c,e) in enumerate(crop['synth']):
                synthetic.append(im);parents.append(index[fid]);covered.append(c);outside.append(e);kinds.append(kind)
    cache=dict(target=torch.tensor(np.stack([selected[f][1]['target'] for f in ids])),parent=np.array(parents),
        covered=torch.tensor(np.stack(covered)),outside=torch.tensor(np.stack(outside)),kind=np.array(kinds),ids=np.array(ids),
        videos=np.array([images[selected[f][0]['image_id']]['metadata']['source_video_id'] for f in ids]),
        occluded=np.array([bool(selected[f][0]['attributes']['face_occluded']) for f in ids]),
        point_occluded=np.array([1 in selected[f][0]['keypoints'][2::3] for f in ids]))
    parity=None
    for name,rgbs in (('natural',clean),('synthetic',synthetic)):
        maps=[];level0=[];level1=[];details=[]
        for b in batches(len(rgbs),4):
            x=torch.from_numpy(np.stack([rgbs[int(i)] for i in b])).cuda().permute(0,3,1,2).float()/255
            m,l,d=probe.encode(x)
            if parity is None:
                direct=probe.detector.decoder(probe.detector.encoder(probe.detector.backbone(x)))
                injected=probe(m[:,2],l,d)
                parity={k:float((direct[k]-injected[k]).abs().max()) for k in ('pred_logits','pred_ellipses')}
                for key in parity:torch.testing.assert_close(direct[key],injected[key],atol=2e-5,rtol=2e-5)
            maps.append(m.half().cpu());level0.append(l[0].half().cpu());level1.append(l[1].half().cpu());details.append(d.half().cpu())
            if int(b[0])%128==0:status(OUT,'cache',domain=name,done=int(b[-1])+1,total=len(rgbs))
        cache[name]=dict(maps=torch.cat(maps),levels=[torch.cat(level0),torch.cat(level1)],detail=torch.cat(details))
    for role in roles:
        cache['natural_'+role]=np.array([index[f] for f in roles[role]['natural']])
        cache['synthetic_'+role]=np.flatnonzero(np.isin(cache['parent'],[index[f] for f in roles[role]['synthetic']]))
    vsets=[set(cache['videos'][cache['natural_'+r]])|set(cache['videos'][cache['parent'][cache['synthetic_'+r]]]) for r in roles]
    assert not any(vsets[i]&vsets[j] for i in range(3) for j in range(i+1,3))
    manifest=dict(roles=roles,faces=ids,skips=skips,raw_injection_parity=parity,
        crop_geometry={str(f):selected[f][1]['crop_box'] for f in ids},
        image_hashes={str(selected[f][0]['image_id']):images[selected[f][0]['image_id']]['metadata']['sha256'] for f in ids},
        data_hashes=dict(annotation=sha(ANN),predicted_crops=sha(CROPS),source_predictions=sha(PRED)),clean_features_never_decoded=True)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    else:write(OUT/'manifest.json',manifest)
    return cache


def batch(cache,domain,j):
    data=cache[domain];parents=cache['parent'][j] if domain=='synthetic' else j
    return gpu(data['maps'][j]),[gpu(x[j]) for x in data['levels']],gpu(data['detail'][j]),gpu(cache['target'][parents])


def generate(net,maps,ids,seed=42):
    return net.sample(maps,ids,seed,PLAN['steps']) if isinstance(net,FeatureDiffusion) else net(maps)[0]


def objective(net,probe,cache,j,seed):
    maps,levels,detail,target=batch(cache,'synthetic',j);r=generate(net,maps,j,seed)
    pred,score,_=probe.select(probe(r,levels,detail));criterion=probe.detector.criterion
    shape,parts,eligible=ellipse_parts(pred,target,criterion.ellipse_weights,criterion.circle_ratio_threshold,criterion.contour_points)
    z=gpu(cache['natural']['maps'][cache['parent'][j],2]);c=gpu(cache['covered'][j]);e=gpu(cache['outside'][j])
    mse=region_mean((r-z).square().mean(1),c,e)
    cosine=region_mean(1-F.cosine_similarity(r,z,dim=1),c,e)
    total=shape.mean()+PLAN['mse_weight']*mse+PLAN['cosine_weight']*cosine
    return total,dict(shape=shape.mean(),mse=mse,cosine=cosine,objective=total,score=score.mean())


@torch.no_grad()
def validate(net,probe,cache):
    net.eval();sums={}
    ids=cache['synthetic_dev']
    for b in batches(len(ids),PLAN['microbatch']):
        j=ids[b];_,parts=objective(net,probe,cache,j,42)
        for key,value in parts.items():sums[key]=sums.get(key,0)+float(value)*len(j)
    return {k:v/len(ids) for k,v in sums.items()}


def train(arm,probe,cache,identity):
    folder=OUT/arm;folder.mkdir(exist_ok=True);torch.manual_seed(42)
    fit=np.unique(cache['parent'][cache['synthetic_fit']]);f=cache['natural']['maps'][fit,2].float()
    stats=f.mean((0,2,3)),f.std((0,2,3),unbiased=False).clamp_min(.01);del f
    net=(FeatureDiffusion(*stats) if arm=='diffusion' else MultiAxisRecovery('scale')).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=PLAN['lr'],weight_decay=PLAN['weight_decay'])
    ctrl=Plateau(PLAN['lr'],PLAN['lr_floor'],PLAN['min_epochs'])
    rng=np.random.default_rng(42);history=[];best=float('inf');start=0;done=False
    if (folder/'last_R.pt').exists():
        ck=torch.load(folder/'last_R.pt',weights_only=True,map_location='cpu');assert ck['identity']==identity
        net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);ctrl=Plateau(**ck['controller'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);start=ck['epoch']+1;done=ck['converged'];best=ck['best']
        history=read(folder/'history.json')[:start]
    if not (folder/'technical.json').exists():
        j=cache['synthetic_fit'][:4];loss,parts=objective(net,probe,cache,j,42)
        grad=torch.autograd.grad(parts['shape'],tuple(net.parameters()),retain_graph=True,allow_unused=True)
        gn=float(sum(g.square().sum() for g in grad if g is not None).sqrt());assert gn>0 and np.isfinite(gn)
        loss.backward();assert all(p.grad is None for p in probe.parameters());opt.zero_grad(set_to_none=True)
        write(folder/'technical.json',dict(ellipse_gradient_norm=gn,finite=True,fixed_parent_no_grad=True));del loss,parts,grad
    for epoch in range(start,PLAN['max_epochs']+1) if not done else ():
        tick=time.monotonic();net.train();train_loss=0.
        if epoch:
            ids=rng.permutation(cache['synthetic_fit'])
            for groupids in batches(len(ids),PLAN['effective_batch']):
                group=ids[groupids];opt.zero_grad(set_to_none=True)
                for b in batches(len(group),PLAN['microbatch']):
                    j=group[b];loss,_=objective(net,probe,cache,j,42+1009*epoch)
                    (loss*len(j)/len(group)).backward();train_loss+=float(loss.detach())*len(j)
                torch.nn.utils.clip_grad_norm_(net.parameters(),PLAN['clip'],error_if_nonfinite=True);opt.step()
            train_loss/=len(ids)
        dev=validate(net,probe,cache)
        if dev['shape']<best:
            best=dev['shape'];save(folder/'best_R.pt',dict(state=net.state_dict(),epoch=epoch,dev=dev,identity=identity))
        done=ctrl.step(epoch,[dev['shape'],dev['objective']]);set_lr(opt,ctrl.lr)
        row=dict(epoch=epoch,dev=dev,train=train_loss if epoch else None,lr=ctrl.lr,converged=done,seconds=time.monotonic()-tick)
        history.append(row);write(folder/'history.json',history)
        save(folder/'last_R.pt',dict(state=net.state_dict(),optimizer=opt.state_dict(),controller=asdict(ctrl),rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),epoch=epoch,best=best,converged=done,identity=identity))
        status(OUT,'training',arm=arm,**row)
        if done:break
    if not done:raise RuntimeError('Epoch cap reached without validation convergence')
    ck=torch.load(folder/'best_R.pt',weights_only=True);net.load_state_dict(ck['state']);net.eval().requires_grad_(False)
    write(folder/'convergence.json',dict(converged=True,selected_epoch=ck['epoch'],stop_epoch=history[-1]['epoch']))
    return net


@torch.no_grad()
def evaluate(arm,net,probe,cache,seed=42):
    folder=OUT/arm;folder.mkdir(exist_ok=True)
    for domain in ('natural','synthetic'):
        ids=cache[domain+'_eval'];predictions=[];scores=[];norms=[];components=[];raw_predictions=[];raw_scores=[]
        for b in batches(len(ids),4):
            j=ids[b];maps,levels,detail,target=batch(cache,domain,j)
            r=maps[:,2] if net is None else generate(net,maps,j,seed)
            pred,score,_=probe.select(probe(r,levels,detail));predictions.append(pred.cpu().numpy());scores.append(score.cpu().numpy())
            raw,rs,_=probe.select(probe(maps[:,2],levels,detail));raw_predictions.append(raw.cpu().numpy());raw_scores.append(rs.cpu().numpy())
            norms.append(torch.stack((maps[:,2].norm(dim=1),r.norm(dim=1)),1).cpu().numpy())
            if domain=='synthetic':
                z=gpu(cache['natural']['maps'][cache['parent'][j],2]);c=gpu(cache['covered'][j]);e=gpu(cache['outside'][j])
                components.append(torch.stack([torch.stack([(metric*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1) for metric in
                    ((r-z).square().mean(1),1-F.cosine_similarity(r,z,dim=1),z.norm(dim=1),r.norm(dim=1)/z.norm(dim=1).clamp_min(1e-8))],1)
                    for mask in (c,e)],1).cpu().numpy())
        parents=cache['parent'][ids] if domain=='synthetic' else ids
        payload=dict(pred=np.concatenate(predictions),score=np.concatenate(scores),raw_pred=np.concatenate(raw_predictions),raw_score=np.concatenate(raw_scores),
            target=cache['target'][parents].numpy(),ids=ids,face_ids=cache['ids'][parents],videos=cache['videos'][parents],
            occluded=cache['occluded'][parents],point_occluded=cache['point_occluded'][parents],norms=np.concatenate(norms))
        if domain=='synthetic':payload.update(components=np.concatenate(components),kind=cache['kind'][ids])
        np.savez_compressed(folder/f'{domain}_seed{seed}.npz',**payload)


def main(resume=False,wait=False):
    if resume:assert OUT.is_dir()
    else:OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    registry=yaml.safe_load(REGISTRY.read_text());checkpoint=ROOT/registry['checkpoint']['path'];assert sha(checkpoint)==registry['checkpoint']['sha256']
    cfgpath=ROOT/registry['config']['path'];cfg=load_config(cfgpath);cfg['model'].pop('init_checkpoint',None)
    paths=[__file__,ROOT/'src/models/messenger_ellipse_probe.py',ROOT/'tests/test_messenger_ellipse_probe.py',
        ROOT/'src/models/messenger_multiaxis.py',ROOT/'src/models/messenger_feature_diffusion.py',
        ROOT/'scripts/run_messenger_multiaxis.py',ROOT/'scripts/run_messenger_multiaxis_convergence.py',
        ROOT/'src/models/dinov3_deimv2/official.py',ROOT/'src/models/dinov3_deimv2_ellipse/official.py',ANN,PRIOR,CROPS,REGISTRY,cfgpath,checkpoint]
    plan=dict(**PLAN,detector_checkpoint=registry['checkpoint'],resolved_config=cfg,
              hashes={str(Path(p).relative_to(ROOT)):sha(p) for p in paths})
    if resume:assert read(OUT/'plan.json')==plan
    else:
        write(OUT/'plan.json',plan)
        for p in paths:
            p=Path(p)
            if p.suffix=='.py':
                dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);dest.chmod(0o444)
    identity=signature(plan)
    with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        while True:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if not wait:raise
                status(OUT,'waiting_for_gpu_lease');time.sleep(30)
        try:
            model=build_detector_from_config(cfg);load_checkpoint(model,checkpoint,strict=True)
            probe=FrozenEllipseProbe(model).cuda().eval();frozen=tensor_digest(probe)
            cache=make_cache(probe,cfg);status(OUT,'evaluating',arm='raw');evaluate('raw',None,probe,cache)
            for arm in ('regression','diffusion'):
                net=train(arm,probe,cache,signature(dict(plan=identity,arm=arm)));status(OUT,'evaluating',arm=arm)
                for seed in (PLAN['sample_seeds'] if arm=='diffusion' else [42]):evaluate(arm,net,probe,cache,seed)
                del net;gc.collect();torch.cuda.empty_cache();assert tensor_digest(probe)==frozen
                assert all(p.grad is None for p in probe.parameters())
            for p,h in plan['hashes'].items():assert sha(ROOT/p)==h,p
            weights=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*.pt')];assert len(weights)==4
            write(OUT/'training_completion.json',dict(all_converged=True,fixed_parent=frozen,source_verified=True,clean_head=False,weights=weights))
            status(OUT,'training_and_inference_completed',weights_retained=True)
        except Exception as error:status(OUT,'failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();main(args.resume,args.wait)
