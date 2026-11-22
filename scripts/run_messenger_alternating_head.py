"""Synthetic-only alternating R/head diagnostic. Run to dev plateau, evaluate, then delete own weights."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_messenger_multiaxis as base
from scripts.run_messenger_multiaxis_convergence import Plateau, signature, set_lr
from scripts.train_natural_feature_gate import build_crop_transform, ellipse_aligned_rectangle
from src.models.messenger_multiaxis import MultiAxisRecovery, region_mean
from src.data.messenger_diverse_occlusion import make_occlusion
OUT=ROOT/'outputs/diagnostics/messenger_alternating_head_20261119'
PLAN=dict(seed=42,faces=dict(fit=256,dev=128,eval=128),variants=9,
    masks='5-20/20-40/40-70 percent ellipse pixels; rectangle/ellipse/curved; 3x3 location grid; solid/noise/gradient',
    training='3 of 9 variants per face per epoch; rotate banks every 3 epochs; identical across arms',
    backbone='fixed V3 DINO, final map; scale1/2/4 width96 independent attention',
    g='exact binary patch occupancy from generated pixel mask, any masked pixel; no learned g',
    stages=['warm_R_fixed_head','fixed_R_adapt_head','alternating','raw_adapt_head'],
    alternate='one H-only epoch with detached R, then one R-only epoch through frozen updated H; paired checkpoint',
    loss='balanced covered/uncovered keypoint mean Euclidean error / GT ellipse major diameter; R adds .01 masked MSE + .01 cosine distance, outside coefficient .1',
    head_loss='same position objective only; no feature loss gradients to H',
    lr_R=1e-4,lr_H=1e-5,weight_decay=.01,batch=16,micro=4,
    convergence='dev position and stationary position+.01MSE+.01cosine; min40, reduce8, floor/100, stop15; cap600 is failure',
    selection='minimum dev position, epoch0 eligible; full R+H pair saved',
    data_scope='same 256/128/128 clean faces as prior, train-internal video split; historical pretrained backbone exposure; no global val/test',
    natural=False,clean_teacher_into_head=False,
    retention='delete all newly generated model/optimizer weights after all numerical audits and saved artifacts, before user review')

def read(p):return json.loads(p.read_text())
def write(p,obj):
    base.write(p,obj)
    with Path(p).open('rb') as f:os.fsync(f.fileno())
    fd=os.open(Path(p).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def save(p,obj):
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def status(phase,**kw):
    write(OUT/'status.json',dict(phase=phase,pid=os.getpid(),**kw))
    print(phase,json.dumps(kw),flush=True)

def recovery():return MultiAxisRecovery('scale').cuda()
def gpu(t):return base.gpu(t)
def infer(r,x):return r(x[:,None].expand(-1,3,-1,-1,-1))[0]

@torch.no_grad()
def cache_data(model,data):
    prior=read(ROOT/'outputs/diagnostics/messenger_position_cosine_20261119/manifest.json')
    corpus=read(base.ANN);anns={a['id']:a for a in corpus['annotations']};images={a['id']:a for a in corpus['images']}
    records=[];roles=[]
    for role,n in PLAN['faces'].items():
        ids=prior['roles'][role]['synthetic'];assert len(ids)==n
        for fid in ids:
            a=anns[fid]
            assert a['keypoints'][2::3]==[2]*5 and a['attributes']['face_occluded'] is False
            assert images[a['image_id']]['metadata']['split']=='train'
            records.append(a);roles.append(role)
    videos=np.array([images[a['image_id']]['metadata']['source_video_id'] for a in records])
    roles=np.array(roles)
    sets=[set(videos[roles==role]) for role in PLAN['faces']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    # Crops' built-in historical masks are skipped; the original records stay unchanged.
    crop_records=copy.deepcopy(records)
    for a in crop_records:a['attributes']['face_occluded']=True
    ds=base.Crops(crop_records,images,Path(data['root']))
    n=len(records);clean=torch.empty(n,384,16,16,dtype=torch.float16)
    syn=torch.empty(n*9,384,16,16,dtype=torch.float16)
    xy=[];patches=[];outside=[];points=[];meta=[];pixel_hashes=[]
    parent=np.repeat(np.arange(n),9);variant=np.tile(np.arange(9),n)
    mean=torch.tensor(data['mean'],device='cuda')[None,:,None,None]
    std=torch.tensor(data['std'],device='cuda')[None,:,None,None]
    def encode(rgb):
        x=torch.from_numpy(np.stack(rgb)).cuda().permute(0,3,1,2).float()/255
        return model.backbone((x-mean)/std).half().cpu()
    import hashlib
    for i,a in enumerate(records):
        row=ds[i];assert (row['state']==2).all()
        clean[i]=encode([row['image']])[0];xy.append(row['xy'])
        box,angle=ellipse_aligned_rectangle(a['ellipse'])
        tr=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        cx,cy=tr.original_to_crop(np.array([a['ellipse'][:2]]))[0]
        rx,ry=np.array(a['ellipse'][2:4])*255/tr.side
        yy,xx=np.mgrid[:256,:256];face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
        rgb=[]
        for v in range(9):
            image,mask,patch,ext,point,m=make_occlusion(row['image'],face,row['xy'],a['id'],v)
            rgb.append(image);patches.append(patch);outside.append(ext);points.append(point);meta.append(m)
            pixel_hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
        for j in range(0,9,3):syn[i*9+j:i*9+min(j+3,9)]=encode(rgb[j:j+3])
        if i%32==0:status('cache',faces=i+1,total=n)
    cache=dict(clean=clean,syn=syn,xy=torch.tensor(np.stack(xy),dtype=torch.float32),
        patch=torch.tensor(np.stack(patches)),outside=torch.tensor(np.stack(outside)),
        point=torch.tensor(np.stack(points)),parent=parent,variant=variant,roles=roles,
        videos=videos,meta=meta)
    manifest=dict(faces=[a['id'] for a in records],roles=roles.tolist(),videos=videos.tolist(),
        images={str(a['image_id']):images[a['image_id']]['metadata']['sha256'] for a in records},
        masks=meta,pixel_mask_sha256=pixel_hashes,cache_dtype='float16',cache_bytes=clean.numel()*2+syn.numel()*2,
        cache_tensor_hash=base.tensor_digest(torch.nn.ParameterList([torch.nn.Parameter(clean,requires_grad=False),torch.nn.Parameter(syn,requires_grad=False)])))
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    else:write(OUT/'manifest.json',manifest)
    np.savez_compressed(OUT/'mask_truth.npz',patch=cache['patch'].numpy(),point=cache['point'].numpy(),
        outside=cache['outside'].numpy(),parent=parent,variant=variant,xy=cache['xy'].numpy())
    return cache

def batch(cache,ids):
    par=cache['parent'][ids]
    return (gpu(cache['syn'][ids]),gpu(cache['clean'][par]),gpu(cache['patch'][ids]),
            gpu(cache['outside'][ids]),gpu(cache['point'][ids]),gpu(cache['xy'][par]))

def forward(r,h,cache,ids,raw=False,detach=False):
    x,z,mask,out,point,xy=batch(cache,ids)
    recovered=x if raw else infer(r,x)
    used=x if raw else torch.where(mask[:,None],recovered,x)
    if detach:used=used.detach()
    pred=base.run_head(h,used)
    err=(pred['decoded_coordinates']-xy).norm(dim=-1)*1.25
    mse=region_mean((recovered-z).square().mean(1),mask,out)
    cos=region_mean(1-F.cosine_similarity(recovered,z,dim=1),mask,out)
    return err,point,mse,cos,recovered,z,pred

@torch.no_grad()
def validate(r,h,cache,role='dev',raw=False,details=False):
    r.eval();h.eval()
    ids=np.flatnonzero(cache['roles'][cache['parent']]==role);s=np.zeros(6);rows=[];features=[]
    for bi in base.batches(len(ids),4):
        j=ids[bi];err,point,mse,cos,rr,z,pred=forward(r,h,cache,j,raw=raw)
        for k,m in enumerate((point,~point)):s[k*2:k*2+2]+=[float(err[m].sum()),int(m.sum())]
        s[4:]+=[float(mse)*len(j),float(cos)*len(j)]
        if details:
            mask=gpu(cache['patch'][j]);den=mask.sum((1,2)).clamp_min(1)
            fm=((rr-z).square().mean(1)*mask).sum((1,2))/den
            fc=((1-F.cosine_similarity(rr,z,dim=1))*mask).sum((1,2))/den
            norm=((rr.norm(dim=1)*mask).sum((1,2))/(z.norm(dim=1)*mask).sum((1,2)).clamp_min(1e-8))
            xy=gpu(cache['xy'][cache['parent'][j]])
            def sample(v):return F.grid_sample(v,(xy*2-1)[:,:,None],align_corners=False)[:,:,:,0].transpose(1,2)
            rs,zs=sample(rr),sample(z)
            local=(rs-zs).square().mean(-1)+(1-F.cosine_similarity(rs,zs,dim=-1))
            rows.append(torch.stack([err,pred['presence_logits'].sigmoid(),local],-1).cpu().numpy())
            features.append(torch.stack([fm,fc,norm],-1).cpu().numpy())
    pos=np.mean([s[0]/s[1],s[2]/s[3]])
    result=dict(position=float(pos),covered=float(s[0]/s[1]),uncovered=float(s[2]/s[3]),
        mse=float(s[4]/len(ids)),cosine=float(s[5]/len(ids)),examples=len(ids))
    result['objective']=result['position']+.01*(result['mse']+result['cosine'])
    return (result,ids,np.concatenate(rows),np.concatenate(features)) if details else result

def train_epoch(r,h,cache,stage,epoch,opt):
    is_R=stage=='R'
    r.eval().requires_grad_(is_R);h.eval().requires_grad_(not is_R)
    r.zero_grad(set_to_none=True);h.zero_grad(set_to_none=True)
    ids=np.flatnonzero((cache['roles'][cache['parent']]=='fit') &
        ((cache['variant']//3)==((epoch-1+cache['parent'])%3)))
    ids=np.random.default_rng(42+epoch).permutation(ids)
    losses=[]
    for b in base.batches(len(ids),16):
        group=ids[b];opt.zero_grad(set_to_none=True)
        points=cache['point'][group];den=[int(points.sum()),int((~points).sum())]
        for mb in base.batches(len(group),4):
            j=group[mb];err,point,mse,cos,*_=forward(r,h,cache,j,raw=stage=='raw_H',detach=not is_R)
            position=sum(err[m].sum()/max(d,1) for m,d in zip((point,~point),den))/sum(d>0 for d in den)
            loss=position+(.01*(mse+cos)*len(j)/len(group) if is_R else 0)
            loss.backward();losses.append(float(loss.detach()))
        updated=r if is_R else h
        torch.nn.utils.clip_grad_norm_(updated.parameters(),5,error_if_nonfinite=True);opt.step()
    frozen=h if is_R else r
    assert all(p.grad is None for p in frozen.parameters())
    return float(sum(losses)/(len(ids)/16))

def train_stage(name,initial_R,initial_H,model,cache,identity):
    folder=OUT/name;folder.mkdir(exist_ok=True)
    r=recovery();r.load_state_dict(initial_R);h=copy.deepcopy(model.head).cuda();h.load_state_dict(initial_H)
    opts={'R':torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01),
          'H':torch.optim.AdamW(h.parameters(),lr=1e-5,weight_decay=.01)}
    ctrl=Plateau(1e-4,1e-6,40);history=[];start=0;best=float('inf');done=False
    last=folder/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cuda',weights_only=True);assert ck['identity']==identity
        r.load_state_dict(ck['R']);h.load_state_dict(ck['H'])
        for k in opts:opts[k].load_state_dict(ck['optimizers'][k])
        ctrl=Plateau(**ck['controller']);history=ck['history'];best=ck['best'];start=ck['epoch']+1;done=ck['converged']
    frozen_H=base.tensor_digest(h);frozen_R=base.tensor_digest(r)
    raw=name=='raw_adapt_head'
    for epoch in range(start,601) if not done else ():
        tick=time.monotonic();train={}
        if epoch:
            steps=['R'] if name=='warm_R_fixed_head' else ['H','R'] if name=='alternating' else ['raw_H'] if raw else ['H']
            for step in steps:train[step]=train_epoch(r,h,cache,step,epoch,opts['R' if step=='R' else 'H'])
        dev=validate(r,h,cache,raw=raw)
        if dev['position']<best:
            best=dev['position'];save(folder/'best.pt',dict(R=r.state_dict(),H=h.state_dict(),epoch=epoch,dev=dev,identity=identity))
        monitored=[dev['position'],dev['objective']] if name in ('warm_R_fixed_head','alternating') else [dev['position']]
        done=ctrl.step(epoch,monitored)
        set_lr(opts['R'],ctrl.lr);set_lr(opts['H'],ctrl.lr*.1)
        row=dict(epoch=epoch,dev=dev,train=train,lr_R=ctrl.lr,lr_H=ctrl.lr*.1,seconds=time.monotonic()-tick,converged=done)
        history.append(row);write(folder/'history.json',history)
        save(last,dict(R=r.state_dict(),H=h.state_dict(),epoch=epoch,history=history,best=best,
            optimizers={k:o.state_dict() for k,o in opts.items()},controller=asdict(ctrl),identity=identity,converged=done))
        if name=='warm_R_fixed_head':assert base.tensor_digest(h)==frozen_H
        if name in ('fixed_R_adapt_head','raw_adapt_head'):assert base.tensor_digest(r)==frozen_R
        status('training',arm=name,epoch=epoch,dev=dev,converged=done,seconds=row['seconds'])
        if done:break
    if not done:raise RuntimeError(name+' reached cap without plateau')
    ck=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
    r.load_state_dict(ck['R']);h.load_state_dict(ck['H'])
    write(folder/'convergence.json',dict(converged=True,stop_epoch=history[-1]['epoch'],selected_epoch=ck['epoch'],selected_dev=ck['dev']))
    return r.eval().requires_grad_(False),h.eval().requires_grad_(False)

def technical(model,cache):
    r=recovery();h=copy.deepcopy(model.head).cuda();ids=np.flatnonzero(cache['roles'][cache['parent']]=='fit')[:4]
    x,z,mask,out,point,xy=batch(cache,ids)
    h.requires_grad_(False);h0=base.tensor_digest(h);r0=base.tensor_digest(r)
    opt=torch.optim.AdamW(r.parameters(),lr=1e-4)
    err,_,mse,cos,*_=forward(r,h,cache,ids)
    loss=err.mean()+.01*(mse+cos);loss.backward()
    grad=float(sum(p.grad.norm() for p in r.parameters() if p.grad is not None))
    assert grad>0 and np.isfinite(grad) and all(p.grad is None for p in h.parameters())
    opt.step();assert base.tensor_digest(h)==h0 and base.tensor_digest(r)!=r0
    r.zero_grad(set_to_none=True);r.requires_grad_(False);h.requires_grad_(True)
    r1=base.tensor_digest(r);opt=torch.optim.AdamW(h.parameters(),lr=1e-5)
    err,*_=forward(r,h,cache,ids,detach=True);err.mean().backward();opt.step()
    assert base.tensor_digest(r)==r1 and base.tensor_digest(h)!=h0
    assert all(p.grad is None for p in r.parameters())
    with torch.no_grad():
        recovered=infer(r,x);mixed=torch.where(mask[:,None],recovered,x)
        assert torch.equal(mixed.permute(0,2,3,1)[~mask],x.permute(0,2,3,1)[~mask])
    # A real fixed mini-batch must fit before authorizing the long diagnostic.
    h.requires_grad_(False);r.requires_grad_(True)
    opt=torch.optim.AdamW(r.parameters(),lr=1e-4);values=[]
    for step in range(25):
        err,_,mse,cos,*_=forward(r,h,cache,ids)
        v=err.mean()+.01*(mse+cos);values.append(float(v.detach()))
        opt.zero_grad();v.backward();torch.nn.utils.clip_grad_norm_(r.parameters(),5,error_if_nonfinite=True);opt.step()
    assert np.mean(values[-3:])<np.mean(values[:3]),values
    write(OUT/'technical_checks.json',dict(R_grad=grad,R_update_H_frozen=True,H_update_R_frozen=True,
        exact_outside_identity=True,mini_fit_first3=float(np.mean(values[:3])),mini_fit_last3=float(np.mean(values[-3:])),
        no_clean_feature_head_input=True))
    del r,h;gc.collect();torch.cuda.empty_cache()

def metrics(rows,point):
    result={}
    for label,m in [('all',np.ones(point.shape,bool)),('covered',point),('uncovered',~point)]:
        e=rows[...,0][m];pr=rows[...,1][m]
        result[label]=dict(points=len(e),mean=float(e.mean()) if len(e) else None,
            median=float(np.median(e)) if len(e) else None,p90=float(np.quantile(e,.9)) if len(e) else None,
            location_success={str(t):int((e<=t).sum()) for t in (.02,.05,.1)},
            with_presence_success={str(t):int(((e<=t)&(pr>=.7)).sum()) for t in (.02,.05,.1)})
    return result

def evaluate(name,r,h,cache,raw=False):
    result,ids,rows,features=validate(r,h,cache,'eval',raw=raw,details=True)
    point=cache['point'][ids].numpy();videos=cache['videos'][cache['parent'][ids]]
    strata={}
    for key in ('band','shape','location','fill'):
        v=np.array([cache['meta'][i][key] for i in ids])
        strata[key]={str(k):metrics(rows[v==k],point[v==k]) for k in np.unique(v)}
    # Clean input is a no-occlusion control only; never substitute teacher features into a masked face.
    clean_ids=np.flatnonzero(cache['roles']=='eval');clean_rows=[]
    with torch.no_grad():
        for bi in base.batches(len(clean_ids),4):
            j=clean_ids[bi];p=base.run_head(h,gpu(cache['clean'][j]))
            clean_rows.append(((p['decoded_coordinates']-gpu(cache['xy'][j])).norm(dim=-1)*1.25).cpu().numpy())
    clean_rows=np.concatenate(clean_rows)
    np.savez_compressed(OUT/(name+'_evaluation.npz'),rows=rows,features=features,point=point,ids=ids,videos=videos,clean_error=clean_rows)
    result.update(metrics=metrics(rows,point),strata=strata,
        feature_columns=['covered_MSE','covered_cosine_distance','covered_norm_ratio'],
        feature_means=features.mean(0).tolist(),clean_mean=float(clean_rows.mean()))
    write(OUT/(name+'_evaluation.json'),result)
    return result

def paired_bootstrap(a,b,point,videos):
    rng=np.random.default_rng(42);uv=np.unique(videos);out={}
    for label,m in [('all',np.ones(point.shape,bool)),('covered',point),('uncovered',~point)]:
        sums=np.array([((a[...,0]-b[...,0])[videos==v]*m[videos==v]).sum() for v in uv])
        counts=np.array([m[videos==v].sum() for v in uv])
        ix=rng.integers(0,len(uv),(2000,len(uv)));den=counts[ix].sum(1);sample=sums[ix].sum(1)/den.clip(1)
        out[label]=dict(mean=float(sums.sum()/counts.sum()),ci95=np.quantile(sample[den>0],[.025,.975]).tolist(),videos=len(uv))
    return out

def finish(results,plan):
    pairs={}
    for a,b in [('warm_R_fixed_head','original_head'),('fixed_R_adapt_head','warm_R_fixed_head'),
                ('alternating','warm_R_fixed_head'),('alternating','fixed_R_adapt_head'),
                ('alternating','raw_adapt_head'),('raw_adapt_head','original_head')]:
        aa=np.load(OUT/(a+'_evaluation.npz'));bb=np.load(OUT/(b+'_evaluation.npz'))
        assert np.array_equal(aa['ids'],bb['ids'])
        pairs[a+' minus '+b]=paired_bootstrap(aa['rows'],bb['rows'],aa['point'],aa['videos'])
        # Independently recompute headline values from saved predictions.
        for name,dat in [(a,aa),(b,bb)]:
            saved=results[name]
            actual=metrics(dat['rows'],dat['point'])
            assert actual==saved['metrics']
            np.testing.assert_allclose(dat['features'].mean(0),saved['feature_means'])
    for p,h in plan['hashes'].items():assert base.sha(ROOT/p)==h,p
    checkpoints=[]
    for stage in PLAN['stages']:
        d=OUT/stage;con=read(d/'convergence.json');assert con['converged']
        history=read(d/'history.json')
        assert con['selected_epoch']==min(history,key=lambda x:x['dev']['position'])['epoch']
        for name in ['best.pt','last.pt']:
            p=d/name;ck=torch.load(p,map_location='cpu',weights_only=True)
            assert ck['identity']==signature(dict(plan=signature(plan),stage=stage))
            assert ck['epoch']==con['selected_epoch' if name=='best.pt' else 'stop_epoch']
            checkpoints.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=base.sha(p)))
            del ck
    write(OUT/'summary.json',dict(conditions=results,paired_video_bootstrap=pairs,checkpoints=checkpoints,
        all_numerical_evaluations_complete=True,limitations=PLAN['data_scope']))
    lines=['# 人工遮蔽での復元器・head交互更新','',
        '5点位置head。gは人工maskのpatch占有。自然遮蔽・正解特徴への置換比較なし。','',
        '|条件|遮蔽点誤差%|非遮蔽点誤差%|特徴MSE|cosine距離|遮蔽なし誤差%|',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in results.items():
        lines.append(f"|{name}|{m['covered']*100:.5f}|{m['uncovered']*100:.5f}|{m['feature_means'][0]:.5f}|{m['feature_means'][1]:.5f}|{m['clean_mean']*100:.5f}|")
    lines+=['', '動画対応差の95%区間はsummary.json。誤差は楕円長径正規化で、従来のHead矩形正規化NMEとは区別。',
        'GT crop・train内動画分割・単一seedの比較。既存DINOには過去のtrain露出がある。独立global val/testや自然遮蔽での実用性の検証ではない。',
        '全数値評価・監査・結果保存後に新規重みを削除。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for stage in PLAN['stages']:
        hist=read(OUT/stage/'history.json')
        for k,key in enumerate(['position','mse']):
            ax[k].plot([x['epoch'] for x in hist],[x['dev'][key] for x in hist],label=stage);ax[k].set_title('dev '+key)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'learning_curves.png');plt.close(fig)
    artifacts={str(p.relative_to(OUT)):base.sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.pt','.tmp') and p.name!='status.json'}
    write(OUT/'numeric_completion.json',dict(all_numerical_evaluations_complete=True,artifacts=artifacts,
        checkpoint_count=len(checkpoints),weights_retained=True,source_verified=True))
    os.sync()
    cleanup()

def cleanup():
    completion=read(OUT/'numeric_completion.json');assert completion['all_numerical_evaluations_complete']
    for f,h in completion['artifacts'].items():assert base.sha(OUT/f)==h,f
    files=read(OUT/'summary.json')['checkpoints'];log=OUT/'checkpoint_deletion.json'
    if log.exists() and read(log)['phase']=='completed':
        assert all(not (ROOT/e['path']).exists() for e in files);return
    for e in files:
        p=ROOT/e['path'];assert p.resolve()==p and p.is_relative_to(OUT) and p.name in ('best.pt','last.pt')
        if p.exists():assert base.sha(p)==e['sha256']
        else:assert log.exists(),'Missing weight without deletion record'
    write(log,dict(phase='prepared',files=files));os.sync()
    for e in files:
        p=ROOT/e['path']
        if p.exists():p.unlink()
    os.sync();assert not list(OUT.rglob('*.pt'))
    write(log,dict(phase='completed',files=files,deleted_bytes=sum(e['bytes'] for e in files)))
    status('completed',all_numerical_evaluations_complete=True,weights_retained=False,checkpoint_count=0)

def main(resume=False):
    torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'numeric_completion.json').exists():cleanup();return
        if not resume:OUT.mkdir(exist_ok=False)
        else:assert OUT.is_dir()
        paths=['scripts/run_messenger_alternating_head.py','src/data/messenger_diverse_occlusion.py',
            'tests/test_messenger_alternating.py','src/models/messenger_multiaxis.py',
            'scripts/run_messenger_multiaxis.py','scripts/run_messenger_multiaxis_convergence.py',
            'scripts/train_v3_shape_recovery_gate.py','scripts/train_natural_feature_gate.py',
            'keypoint_comparison1/landmark_detection/heads.py','keypoint_comparison1/landmark_detection/model.py',
            str(base.CONFIG.relative_to(ROOT)),str(base.BASE.relative_to(ROOT)),str(base.ANN.relative_to(ROOT)),
            'outputs/diagnostics/messenger_position_cosine_20261119/manifest.json']
        plan=dict(**PLAN,hashes={p:base.sha(ROOT/p) for p in paths})
        if resume:assert read(OUT/'plan.json')==plan
        else:
            write(OUT/'plan.json',plan)
            for p in paths:
                if p.endswith('.py'):
                    dest=OUT/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
        config=yaml.safe_load(base.CONFIG.read_text())
        model,_=base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
        base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True);model.cuda().eval().requires_grad_(False)
        frozen=base.tensor_digest(model);cache=cache_data(model,config['data'])
        if not (OUT/'technical_checks.json').exists():technical(model,cache)
        original_H=copy.deepcopy(model.head.state_dict());net=recovery();original_R=copy.deepcopy(net.state_dict());del net
        def run(name,initial_R):return train_stage(name,initial_R,original_H,model,cache,signature(dict(plan=signature(plan),stage=name)))
        r,h=run('warm_R_fixed_head',original_R)
        warm_R=copy.deepcopy(r.state_dict())
        results={}
        results['original_head']=evaluate('original_head',r,model.head,cache,raw=True)
        results['warm_R_fixed_head']=evaluate('warm_R_fixed_head',r,h,cache)
        del r,h;gc.collect();torch.cuda.empty_cache()
        for name in ['fixed_R_adapt_head','alternating','raw_adapt_head']:
            r,h=run(name,warm_R)
            results[name]=evaluate(name,r,h,cache,raw=name=='raw_adapt_head')
            write(OUT/'comparison.json',results)
            assert base.tensor_digest(model)==frozen
            del r,h;gc.collect();torch.cuda.empty_cache()
        assert base.tensor_digest(model)==frozen
        finish(results,plan)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:main(args.resume)
    except Exception as e:
        if OUT.exists():status('failed',error=repr(e))
        raise
