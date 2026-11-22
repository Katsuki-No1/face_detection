"""Bounded nine-condition V3 research comparison, not production training."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_v3_shape_recovery_gate import CONFIG,BASE,ANN,Crops,stable
from scripts.train_natural_feature_gate import run_head,tensor_digest
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import load_landmark_checkpoint
from src.models.messenger_multiaxis import MultiAxisRecovery,reconstruction_loss,feature_error
from src.models.natural_feature_gate import NaturalFeatureGate

PRIOR=ROOT/'outputs/diagnostics/v3_shape_recovery_gate_20261107_refined/contract.json'
DEFAULT_OUT=ROOT/'outputs/diagnostics/messenger_multiaxis_20261114'
PLAN=dict(seed=42,levels=[3,7,11],patch_factors=[1,2,4],width=96,attention_heads=6,layers=2,
          families=['scale','level','level_scale'],conditions=['A','B','C'],
          synthetic_faces=dict(fit=256,dev=128,eval=128),natural_faces=dict(fit=512,dev=128,eval=512),
          epochs=4,gate_epochs=4,batch=16,lr=1e-4,gate_lr=5e-4,weight_decay=.01,
          outside_weight=.1,branch_weight=1.,consistency_weight=.1,overfit_steps=24,
          R_selection='minimum dev covered cosine+raw MSE, including epoch 0',
          g_selection='minimum dev all-point position SmoothL1 beta .02',
          scope='V3 train-internal disjoint videos; GT crop; historical backbone train exposure; no val/test',
          patch_encoding='learned strided patch projection, all within-patch cells retained before projection',
          attention_mask='none: target-containing patches permitted',
          retention='delete newly generated best R/g after completed evaluation; no feature cache on disk',
          capacity='width fixed per branch; 3 vs 9 branches, not parameter-matched',
          production=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def status(out,phase,**kwargs):
    write(out/'status.json',dict(phase=phase,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kwargs))
    print(phase,kwargs,flush=True)


def batches(n,size=16):
    for start in range(0,n,size):yield np.arange(start,min(start+size,n))


def gpu(x):return x.cuda().float() if x.is_floating_point() else x.cuda()


def prune_checkpoints(folder):
    root=folder.resolve();audit=[]
    if not (folder/'evaluation.json').is_file() or not (folder/'natural_points.npz').is_file():
        raise RuntimeError('Evaluation must be durable before deleting weights')
    for name in ('best_recovery.pt','best_gate.pt'):
        p=folder/name
        if p.is_symlink() or p.resolve().parent!=root:raise RuntimeError('Unsafe checkpoint target')
        if p.exists():
            audit.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    write(folder/'checkpoint_deletion.json',dict(files=audit,authorized=True,phase='prepared'))
    for row in audit:(ROOT/row['path']).unlink()
    write(folder/'checkpoint_deletion.json',dict(files=audit,authorized=True,phase='completed'))


@torch.no_grad()
def make_cache(out,model,data):
    prior=json.loads(PRIOR.read_text());corpus=json.loads(ANN.read_text())
    images={a['id']:a for a in corpus['images']}
    allfaces=[a for a in corpus['annotations'] if a.get('ellipse') and a['num_keypoints']>0]
    roles={v:r for r,vs in prior['roles'].items() for v in vs}
    selected={};synth_ids=set();natural_ids={};manifest={}
    for role in ('fit','dev','eval'):
        pool=[a for a in allfaces if roles[images[a['image_id']]['metadata']['source_video_id']]==role]
        assert all(images[a['image_id']]['metadata']['split']=='train' for a in pool)
        pool=sorted(pool,key=lambda a:stable(f'multiaxis:42:{a["id"]}'))
        eligible=[a for a in pool if a['keypoints'][2::3]==[2]*5 and a['attributes']['face_occluded'] is False]
        sy=eligible[:PLAN['synthetic_faces'][role]];na=pool[:PLAN['natural_faces'][role]]
        assert len(sy)==PLAN['synthetic_faces'][role] and len(na)==PLAN['natural_faces'][role]
        synth_ids.update(a['id'] for a in sy);natural_ids[role]=[a['id'] for a in na]
        selected.update({a['id']:a for a in sy+na})
        manifest[role]=dict(synthetic=[a['id'] for a in sy],natural=natural_ids[role])
    records=sorted(selected.values(),key=lambda a:a['id']);ds=Crops(records,images,Path(data['root']))
    index={a['id']:i for i,a in enumerate(records)}
    clean=[];syn=[];parent=[];cov=[];ext=[];points=[];kinds=[];fractions=[];xy=[];states=[]
    mean=torch.tensor(data['mean'],device='cuda')[None,:,None,None]
    std=torch.tensor(data['std'],device='cuda')[None,:,None,None]
    parity=None
    def encode(rgb):
        x=torch.from_numpy(np.stack(rgb)).cuda().permute(0,3,1,2).float()/255
        x=(x-mean)/std
        maps=model.backbone.forward_intermediate_feature_maps(x,layer_indexes=PLAN['levels'])
        nonlocal parity
        if parity is None:
            direct=model.backbone(x)
            parity=float((maps[-1]-direct).abs().max())
            torch.testing.assert_close(maps[-1],direct,rtol=1e-4,atol=1e-4)
        return torch.stack(maps,1).half().cpu()
    for ids in batches(len(records),8):
        rows=[ds[int(i)] for i in ids]
        clean.append(encode([r['image'] for r in rows]))
        rgb=[]
        for i,row in zip(ids,rows):
            xy.append(row['xy']);states.append(row['state'])
            if records[int(i)]['id'] not in synth_ids:continue
            assert len(row['synthetic'])==3,'All three shapes must be available'
            for kind,img,c,e,p,f in row['synthetic']:
                rgb.append(img);parent.append(i);cov.append(c);ext.append(e);points.append(p);kinds.append(kind);fractions.append(f)
        for b in batches(len(rgb),8):syn.append(encode([rgb[int(j)] for j in b]))
        if int(ids[0])%128==0:status(out,'cache',faces=int(ids[-1])+1,total=len(records))
    result=dict(clean=torch.cat(clean),syn=torch.cat(syn),parent=np.array(parent),cov=torch.tensor(np.stack(cov)),
                ext=torch.tensor(np.stack(ext)),point=torch.tensor(np.stack(points)),kind=np.array(kinds),fraction=np.array(fractions),
                xy=torch.tensor(np.stack(xy),dtype=torch.float32),state=torch.tensor(np.stack(states)),
                videos=np.array([images[a['image_id']]['metadata']['source_video_id'] for a in records]),
                contour=np.array([a['attributes']['face_occluded'] for a in records]),
                ratio=np.array([max(a['ellipse'][2:4])/min(a['ellipse'][2:4]) for a in records]))
    for role in ('fit','dev','eval'):
        result['natural_'+role]=np.array([index[i] for i in manifest[role]['natural']])
        result['syn_'+role]=np.flatnonzero(np.isin(result['parent'],[index[i] for i in manifest[role]['synthetic']]))
    video_sets=[set(result['videos'][result['natural_'+r]]) | set(result['videos'][result['parent'][result['syn_'+r]]]) for r in ('fit','dev','eval')]
    assert not any(video_sets[i]&video_sets[j] for i in range(3) for j in range(i+1,3))
    write(out/'manifest.json',dict(roles=manifest,faces=[a['id'] for a in records],
          image_hashes={str(a['image_id']):images[a['image_id']]['metadata']['sha256'] for a in records},
          final_map_parity_max=parity,cache_bytes=sum(v.numel()*v.element_size() for v in result.values() if isinstance(v,torch.Tensor)),
          synthetic_pairs=len(parent),natural_counts={r:len(result['natural_'+r]) for r in ('fit','dev','eval')}))
    return result


def syn_batch(cache,ids):
    return (gpu(cache['syn'][ids]),gpu(cache['clean'][cache['parent'][ids],2]),gpu(cache['cov'][ids]),gpu(cache['ext'][ids]))


@torch.no_grad()
def validation(net,cache,role):
    ids=cache['syn_'+role];sums=None;count=0
    net.eval()
    for b in batches(len(ids)):
        x,z,c,e=syn_batch(cache,ids[b]);y,r,w=net(x)
        vals=torch.stack([(feature_error(v,z)*c).sum() for v in [y,*r.unbind(1),x[:,2]]]).cpu().numpy()
        sums=vals if sums is None else sums+vals;count+=int(c.sum())
    return dict(fused=float(sums[0]/count),branches=(sums[1:-1]/count).tolist(),raw=float(sums[-1]/count),covered_patches=count)


@torch.no_grad()
def natural_candidates(net,cache):
    rows=[];net.eval()
    for ids in batches(len(cache['clean'])):rows.append(net(gpu(cache['clean'][ids]))[0].half().cpu())
    return torch.cat(rows)


def fit_gate(folder,model,cache,recovered):
    torch.manual_seed(42);gate=NaturalFeatureGate(384).cuda();opt=torch.optim.AdamW(gate.parameters(),lr=PLAN['gate_lr'])
    history=[];best=float('inf');rng=np.random.default_rng(42)
    for epoch in range(PLAN['gate_epochs']+1):
        record=dict(epoch=epoch)
        for role in (('dev',) if epoch==0 else ('fit','dev')):
            ids=cache['natural_'+role].copy();training=role=='fit'
            if training:rng.shuffle(ids)
            num=0.;den=0
            with torch.set_grad_enabled(training):
                for b in batches(len(ids)):
                    j=ids[b];_,y=gate(gpu(cache['clean'][j,2]),gpu(recovered[j]),1.)
                    coords=run_head(model.head,y)['decoded_coordinates'];valid=gpu(cache['state'][j])>0
                    error=F.smooth_l1_loss(coords,gpu(cache['xy'][j]),beta=.02,reduction='none').mean(-1)
                    loss=(error*valid).sum()/valid.sum().clamp_min(1)
                    if training:
                        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),5,error_if_nonfinite=True);opt.step()
                    num+=float((error.detach()*valid).sum());den+=int(valid.sum())
            record[role]=num/den
        if record['dev']<best:
            best=record['dev'];torch.save(dict(state=gate.state_dict(),epoch=epoch),folder/'best_gate.pt')
        history.append(record)
    selected=torch.load(folder/'best_gate.pt',weights_only=True);gate.load_state_dict(selected['state']);gate.eval()
    write(folder/'gate_history.json',dict(selected_epoch=selected['epoch'],history=history))
    return gate


def summarize_points(error,presence,state):
    result={}
    for s,name in ((1,'occluded'),(2,'visible')):
        valid=state==s;n=int(valid.sum())
        result[name]=dict(points=n,correct=int(((error<=.1)&(presence>=.7)&valid).sum()),
                          mean_error=float(error[valid].mean()) if n else None)
    return result


@torch.no_grad()
def head_evaluate(folder,model,net,gate,cache,recovered):
    ids=cache['natural_eval'];rows=[]
    for b in batches(len(ids)):
        j=ids[b];x=gpu(cache['clean'][j,2]);r=gpu(recovered[j]);_,y=gate(x,r,1.)
        values=[]
        for f in (x,r,y):
            pred=run_head(model.head,f)
            values.append(torch.stack(((pred['decoded_coordinates']-gpu(cache['xy'][j])).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
        rows.append(torch.stack(values,1).cpu().numpy())
    rows=np.concatenate(rows);state=cache['state'][ids].numpy()
    result={name:summarize_points(rows[:,i,:,0],rows[:,i,:,1],state) for i,name in enumerate(('raw','full_recovery','learned_gate'))}
    slices={}
    for contour in (False,True):
        for n in range(6):
            mask=(cache['contour'][ids]==contour)&((state==1).sum(1)==n)
            slices[f'contour={contour}/occluded_points={n}']=dict(faces=int(mask.sum()),
                metrics={name:summarize_points(rows[mask,i,:,0],rows[mask,i,:,1],state[mask]) for i,name in enumerate(('raw','full_recovery','learned_gate'))})
    np.savez_compressed(folder/'natural_points.npz',rows=rows,state=state,ids=ids,videos=cache['videos'][ids])
    result['occlusion_slices']=slices
    # Individual branches and leave-one-branch-out restoration, held-out artificial images.
    allids=cache['syn_eval'];sumerr=None;count=0;weightsum=None;strata={}
    for b in batches(len(allids)):
        j=allids[b];x,z,c,e=syn_batch(cache,j);y,branches,w=net(x)
        ablations=[(y-w[:,k]*branches[:,k])/(1-w[:,k]).clamp_min(1e-6) for k in range(len(net.specs))]
        vals=[y,*branches.unbind(1),*ablations,x[:,2]]
        per=torch.stack([feature_error(v,z) for v in vals],1)
        v=(per*c[:,None]).sum((0,2,3)).cpu().numpy();sumerr=v if sumerr is None else sumerr+v;count+=int(c.sum())
        v=w.sum((0,2,3,4)).cpu().numpy();weightsum=v if weightsum is None else weightsum+v
        for kind in range(3):
            for lo,hi in ((.1,.2),(.2,.3),(.3,.400001)):
                m=(cache['kind'][j]==kind)&(cache['fraction'][j]>=lo)&(cache['fraction'][j]<hi)
                if not m.any():continue
                key=f'{kind}/{lo:.1f}-{min(hi,.4):.1f}';old=strata.setdefault(key,[0.,0.,0])
                old[0]+=float((per[m,0]*c[m]).sum());old[1]+=float((per[m,-1]*c[m]).sum());old[2]+=int(c[m].sum())
    n=len(net.specs);v=sumerr/count
    result['synthetic']=dict(fused=float(v[0]),branches=v[1:1+n].tolist(),leave_one_out=v[1+n:1+2*n].tolist(),raw=float(v[-1]),
                             fusion_weights=(weightsum/weightsum.sum()).tolist(),
                             shape_area={k:dict(fused=a/n,raw=b/n,covered_patches=n) for k,(a,b,n) in strata.items()})
    return result


def run_arm(out,family,condition,model,cache):
    folder=out/f'{family}_{condition}';folder.mkdir()
    torch.manual_seed(42);net=MultiAxisRecovery(family).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=PLAN['lr'],weight_decay=PLAN['weight_decay'])
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    ids=cache['syn_fit'];probe=ids[:8];x,z,c,e=syn_batch(cache,probe)
    report={};start=None
    # Technical real-data GPU/overfit preflight; reset before the comparison.
    for step in range(PLAN['overfit_steps']):
        loss,parts=reconstruction_loss(net(x),z,c,e,condition)
        if start is None:start=float(parts['fused'].detach())
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
    with torch.no_grad():end=float(reconstruction_loss(net(x),z,c,e,condition)[1]['fused'])
    assert end<start, f'Overfit did not improve: {start} -> {end}'
    report.update(initial=start,final=end,steps=PLAN['overfit_steps'],passed=True)
    write(folder/'overfit.json',report)
    net.load_state_dict(initial);del initial,x,z,c,e
    opt=torch.optim.AdamW(net.parameters(),lr=PLAN['lr'],weight_decay=PLAN['weight_decay'])
    best=float('inf');history=[];rng=np.random.default_rng(42);started=time.monotonic()
    for epoch in range(PLAN['epochs']+1):
        if epoch:
            net.train();order=rng.permutation(ids);totals=[]
            for b in batches(len(order)):
                x,z,c,e=syn_batch(cache,order[b]);loss,parts=reconstruction_loss(net(x),z,c,e,condition)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
                totals.append([float(loss.detach()),*[float(v.detach()) for v in parts.values()]])
        metrics=validation(net,cache,'dev');row=dict(epoch=epoch,dev=metrics)
        if epoch:row['train_mean']=np.mean(totals,0).tolist()
        if metrics['fused']<best:
            best=metrics['fused'];torch.save(dict(state=net.state_dict(),epoch=epoch),folder/'best_recovery.pt')
        history.append(row);write(folder/'history.json',history)
        status(out,'training',arm=folder.name,epoch=epoch,dev=metrics['fused'])
    selected=torch.load(folder/'best_recovery.pt',weights_only=True);net.load_state_dict(selected['state']);net.eval()
    recovered=natural_candidates(net,cache)
    status(out,'gate_training',arm=folder.name)
    gate=fit_gate(folder,model,cache,recovered)
    metrics=head_evaluate(folder,model,net,gate,cache,recovered)
    metrics.update(selected_epoch=selected['epoch'],parameters=sum(p.numel() for p in net.parameters()),
                   branch_specs=net.specs,seconds=time.monotonic()-started,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(),weights=dict(recovery=sha(folder/'best_recovery.pt'),gate=sha(folder/'best_gate.pt')))
    write(folder/'evaluation.json',metrics)
    prune_checkpoints(folder)
    del recovered,net,gate,opt,selected;gc.collect();torch.cuda.empty_cache()
    return metrics


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (BASE,CONFIG,ANN,PRIOR,Path(__file__),ROOT/'src/models/messenger_multiaxis.py',ROOT/'src/models/natural_feature_gate.py',ROOT/'scripts/train_v3_shape_recovery_gate.py',ROOT/'src/data/v3_shape_occlusion.py')}
    write(out/'plan.json',dict(**PLAN,hashes=hashes))
    status(out,'preparing')
    try:
        with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            raw=yaml.safe_load(CONFIG.read_text());model,_=build_landmark_model(raw,variant='query_refinement',initialize_from_detector=False)
            load_landmark_checkpoint(model,BASE,prefer_ema=True,strict=True)
            assert sha(BASE)=='11b430b595e56395a14c266cb364bd50848b1a51b5b689548eb6749de0a23856'
            model.requires_grad_(False).eval().cuda();frozen=tensor_digest(model)
            cache=make_cache(out,model,raw['data']);results={}
            for family in PLAN['families']:
                for condition in PLAN['conditions']:
                    torch.cuda.reset_peak_memory_stats()
                    metrics=run_arm(out,family,condition,model,cache)
                    assert tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
                    results[f'{family}_{condition}']=metrics;write(out/'comparison.json',results)
                    status(out,'arm_completed',arm=f'{family}_{condition}',completed=len(results),total=9)
            assert all(sha(ROOT/p)==v for p,v in hashes.items()),'Execution inputs changed'
            write(out/'completion.json',dict(completed_arms=list(results),frozen_model_hash=frozen,source_hashes_verified=True,
                  generated_checkpoints_remaining=[str(p) for p in out.rglob('*.pt')],production=False))
            status(out,'completed',completed=9)
    except Exception as error:
        status(out,'failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().output.resolve())
