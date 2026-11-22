"""Position-primary regression vs differentiable full-sampler DF, frozen V3 head."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_messenger_diffusion_comparison as old
from scripts.run_messenger_multiaxis_convergence import save, signature, set_lr
from src.models.natural_feature_gate import NaturalFeatureGate
from src.models.messenger_feature_diffusion import FeatureDiffusion
base = old.base
OUT = ROOT / 'outputs/diagnostics/messenger_position_primary_20261118'
ARMS = ('det_position', 'diff_position')
PLAN = dict(arms=ARMS, seed=42, sample_steps=16, microbatch=4, effective_batch=16,
    lr=1e-4, gate_lr=5e-4, mse_weight=.01, cosine_weight=.01, gate_bce_weight=.01,
    position='mean of covered and uncovered GT-point mean Euclidean errors / GT ellipse major diameter',
    recovery_selection='minimum dev position, full 16-step sampling for DF',
    gate_selection='minimum dev mean of occluded and visible coordinate error',
    feature_region='covered mean plus 0.1 times ellipse-uncovered mean',
    DF='end-to-end deterministic DDIM-like v-parameterized 16-step generator; no denoising objective; all steps differentiated',
    architecture='same independent scale1/2/4 attention, width96; no branch auxiliary or consistency in either arm',
    R_convergence='position and full objective plateau at lr floor1e-6, minimum40 epochs, reduce8, stop15, cap600 is not convergence',
    g_convergence='position and full objective plateau at lr floor5e-6, minimum20 epochs, reduce8, stop15, cap300 is not convergence',
    sampling_seeds=dict(train='42+1009*epoch, item-bound', dev=42, evaluation=[42,1042,2042,3042]),
    natural_teacher='unchanged old weak five-point interpolation; confidence-weighted soft BCE',
    initialization='fresh seed42; V3 backbone/head fixed; same historical small train-video split',
    clean_head=False, retention='all best/last R/g until every numerical evaluation complete; no deletion',
    production=False)


def read(p): return json.loads(p.read_text())


def balanced_mean(error, first, second):
    values = [(error*m).sum()/m.sum().clamp_min(1) for m in (first, second) if bool(m.any())]
    if not values: raise ValueError('No valid points')
    return torch.stack(values).mean()


def generate(net, maps, ids, seed=42):
    return net.sample(maps, ids, seed, PLAN['sample_steps']) if isinstance(net, FeatureDiffusion) else net(maps)[0]


def endpoint_loss(model, r, x, z, mask, outside, xy, covered, valid):
    used = torch.where(mask[:, None], r, x)
    prediction = base.run_head(model.head, used)
    errors = (prediction['decoded_coordinates']-xy).norm(dim=-1)*1.25
    pos = balanced_mean(errors, covered & valid, ~covered & valid)
    mse = old.region_mean((r-z).square().mean(1), mask, outside)
    cos = old.region_mean(1-F.cosine_similarity(r,z,dim=1), mask, outside)
    return pos+PLAN['mse_weight']*mse+PLAN['cosine_weight']*cos, dict(position=pos,mse=mse,cosine=cos), errors


@torch.no_grad()
def validate(net, model, cache):
    net.eval(); sums=np.zeros(6)
    for b in base.batches(len(cache['syn_dev']),PLAN['microbatch']):
        j=cache['syn_dev'][b];maps,z,c,e=base.syn_batch(cache,j)
        r=generate(net,maps,j);xy=base.gpu(cache['xy'][cache['parent'][j]])
        point=base.gpu(cache['point'][j]);valid=base.gpu(cache['state'][cache['parent'][j]])>0
        _,_,err=endpoint_loss(model,r,maps[:,2],z,c,e,xy,point,valid)
        for k,m in enumerate((point & valid, ~point & valid)):
            sums[k*2:k*2+2] += [float(err[m].sum()),int(m.sum())]
        mse=(r-z).square().mean(1);cos=1-F.cosine_similarity(r,z,dim=1)
        sums[4:] += [float(old.region_mean(mse,c,e))*len(j),float(old.region_mean(cos,c,e))*len(j)]
    position=(sums[0]/sums[1]+sums[2]/sums[3])/2
    mse,cos=sums[4:]/len(cache['syn_dev'])
    return dict(position=float(position),covered=float(sums[0]/sums[1]),uncovered=float(sums[2]/sums[3]),
                mse=float(mse),cosine=float(cos),objective=float(position+.01*mse+.01*cos))


def train_recovery(folder,kind,model,cache,stats,identity):
    net=old.make_net(kind,stats)
    opt,ctrl,rng,history,best,start,done=old.initial(net,folder,'R',identity)
    for epoch in range(start,601) if not done else ():
        tick=time.monotonic(); net.train(); totals=[]
        if epoch:
            ids=rng.permutation(cache['syn_fit'])
            for bb in base.batches(len(ids),PLAN['effective_batch']):
                group=ids[bb];opt.zero_grad(set_to_none=True)
                # Normalize both strata over the effective batch, independent of microbatch composition.
                gp=cache['point'][group];gv=cache['state'][cache['parent'][group]]>0
                denominators=[int((gp&gv).sum()),int((~gp&gv).sum())]
                for mb in base.batches(len(group),PLAN['microbatch']):
                    j=group[mb];maps,z,c,e=base.syn_batch(cache,j)
                    r=generate(net,maps,j,42+1009*epoch)
                    xy=base.gpu(cache['xy'][cache['parent'][j]])
                    point=base.gpu(cache['point'][j]);valid=base.gpu(cache['state'][cache['parent'][j]])>0
                    _,parts,err=endpoint_loss(model,r,maps[:,2],z,c,e,xy,point,valid)
                    pos=sum(err[m].sum()/max(d,1) for m,d in zip((point&valid,~point&valid),denominators))/sum(d>0 for d in denominators)
                    loss=pos+(.01*parts['mse']+.01*parts['cosine'])*len(j)/len(group)
                    loss.backward();totals.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        dev=validate(net,model,cache)
        if dev['position']<best:
            best=dev['position'];save(folder/'best_R.pt',dict(state=net.state_dict(),epoch=epoch,dev=dev,identity=identity))
        done=ctrl.step(epoch,[dev['position'],dev['objective']]);set_lr(opt,ctrl.lr)
        row=dict(epoch=epoch,dev=dev,train=sum(totals)/48 if totals else None,lr=ctrl.lr,seconds=time.monotonic()-tick,converged=done)
        history.append(row);old.persist(folder,'R',net,opt,ctrl,rng,history,best,identity,done)
        base.status(OUT,'training',arm=kind,stage='R',**row)
        print(kind,'R',epoch,json.dumps(dev),'seconds',round(row['seconds'],2),flush=True)
        if done:break
    if not done:raise RuntimeError('R cap reached without convergence')
    ck=torch.load(folder/'best_R.pt',weights_only=True);net.load_state_dict(ck['state']);net.eval().requires_grad_(False)
    base.write(folder/'R_convergence.json',dict(converged=True,selected_epoch=ck['epoch'],stop_epoch=history[-1]['epoch']))
    return net


@torch.no_grad()
def candidates(net,cache):
    return torch.cat([generate(net,base.gpu(cache['clean'][j]),j).float().cpu() for j in base.batches(len(cache['clean']),4)])


def gate_pass(gate,model,cache,recovered,ids,opt=None):
    sums=np.zeros(6)
    with torch.set_grad_enabled(opt is not None):
        for b in base.batches(len(ids),16):
            j=ids[b];logits,y=gate(base.gpu(cache['clean'][j,2]),base.gpu(recovered[j]),1.)
            pred=base.run_head(model.head,y);state=base.gpu(cache['state'][j])
            err=(pred['decoded_coordinates']-base.gpu(cache['xy'][j])).norm(dim=-1)*1.25
            pos=balanced_mean(err,state==1,state==2)
            weight=base.gpu(cache['confidence'][j]);target=base.gpu(cache['teacher'][j])
            ce=F.binary_cross_entropy_with_logits(logits[:,0],target,reduction='none')
            bce=(ce*weight).sum()/weight.sum().clamp_min(1e-8)
            if opt is not None:
                opt.zero_grad(set_to_none=True);(pos+.01*bce).backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(),5,error_if_nonfinite=True);opt.step()
            for k,m in enumerate((state==1,state==2)):
                sums[k*2:k*2+2]+=[float(err[m].detach().sum()),int(m.sum())]
            sums[4:]+=[float((ce.detach()*weight).sum()),float(weight.sum())]
    pos=(sums[0]/sums[1]+sums[2]/sums[3])/2;bce=sums[4]/sums[5]
    return dict(position=float(pos),occluded=float(sums[0]/sums[1]),visible=float(sums[2]/sums[3]),soft_bce=float(bce),objective=float(pos+.01*bce))


def train_gate(folder,kind,model,cache,recovered,identity):
    torch.manual_seed(42);gate=NaturalFeatureGate(384).cuda()
    opt,ctrl,rng,history,best,start,done=old.initial(gate,folder,'g',identity)
    for epoch in range(start,301) if not done else ():
        train=gate_pass(gate,model,cache,recovered,rng.permutation(cache['natural_fit']),opt) if epoch else None
        dev=gate_pass(gate,model,cache,recovered,cache['natural_dev'])
        if dev['position']<best:
            best=dev['position'];save(folder/'best_g.pt',dict(state=gate.state_dict(),epoch=epoch,dev=dev,identity=identity))
        done=ctrl.step(epoch,[dev['position'],dev['objective']]);set_lr(opt,ctrl.lr)
        row=dict(epoch=epoch,dev=dev,train=train,lr=ctrl.lr,converged=done);history.append(row)
        old.persist(folder,'g',gate,opt,ctrl,rng,history,best,identity,done)
        base.status(OUT,'training',arm=kind,stage='g',**row)
        if epoch%10==0 or done:print(kind,'g',epoch,json.dumps(dev),flush=True)
        if done:break
    if not done:raise RuntimeError('g cap reached without convergence')
    ck=torch.load(folder/'best_g.pt',weights_only=True);gate.load_state_dict(ck['state']);gate.eval().requires_grad_(False)
    base.write(folder/'g_convergence.json',dict(converged=True,selected_epoch=ck['epoch'],stop_epoch=history[-1]['epoch']))
    return gate


def metrics(rows,selected,videos):
    error=rows[...,0];presence=rows[...,1]>=.7
    result=dict(points=int(selected.sum()),mean_error=old.boot(error,selected,videos),median=float(np.median(error[selected])),
                p90=float(np.quantile(error[selected],.9)),presence=int(presence[selected].sum()),thresholds={})
    for t in (.02,.03,.05,.1):
        result['thresholds'][str(t)]=dict(location=int(((error<=t)&selected).sum()),joint=int(((error<=t)&presence&selected).sum()))
    return result


@torch.no_grad()
def evaluate(folder,net,gate,model,cache,seed=42):
    result={}
    for domain in ('natural','artificial'):
        rows=[];ids=cache['natural_eval' if domain=='natural' else 'syn_eval'];errs=[];counts=[];locals_=[];energies=[]
        for b in base.batches(len(ids),4):
            j=ids[b]
            if domain=='natural':
                maps=base.gpu(cache['clean'][j]);xy=base.gpu(cache['xy'][j]);x=maps[:,2]
            else:
                maps,z,c,e=base.syn_batch(cache,j);x=maps[:,2];xy=base.gpu(cache['xy'][cache['parent'][j]])
            r=generate(net,maps,j,seed);_,y=gate(x,r,1.)
            values=(x,r,y) if domain=='natural' else (x,r,y,torch.where(c[:,None],r,x))
            pr=[]
            for value in values:
                pred=base.run_head(model.head,value)
                pr.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
            rows.append(torch.stack(pr,1).cpu().numpy())
            if domain=='artificial':
                errs.append(torch.stack([(old.feature_error(v,z)*m).sum((1,2)) for v in (x,r) for m in (c,e)],1).cpu().numpy())
                counts.append(torch.stack((c.sum((1,2)),e.sum((1,2))),1).cpu().numpy())
                zc,zl,valid=old.point_features(z,xy);ll=[];ee=[]
                for v in (x,r,z):
                    vc,vl,_=old.point_features(v,xy)
                    ce=1-F.cosine_similarity(vc,zc,dim=-1)+(vc-zc).square().mean(-1)
                    le=1-F.cosine_similarity(vl,zl,dim=-1)+(vl-zl).square().mean(-1)
                    ll.append(torch.stack((ce,(le*valid).sum(-1)/valid.sum(-1)),-1))
                    avg=(vl*valid[...,None]).sum(-2)/valid.sum(-1)[...,None]
                    ee.append((((vl-avg[...,None,:]).square().mean(-1))*valid).sum(-1)/valid.sum(-1))
                locals_.append(torch.stack(ll,1).cpu().numpy());energies.append(torch.stack(ee,1).cpu().numpy())
        rows=np.concatenate(rows)
        if domain=='natural':
            state=cache['state'][ids].numpy();videos=cache['videos'][ids];masks={'occluded':state==1,'visible':state==2}
            np.savez_compressed(folder/f'natural_seed{seed}.npz',rows=rows,state=state,ids=ids,videos=videos)
        else:
            point=cache['point'][ids].numpy();videos=cache['videos'][cache['parent'][ids]]
            masks={'covered':point,'uncovered':~point};er=np.concatenate(errs);ct=np.concatenate(counts)
            np.savez_compressed(folder/f'artificial_seed{seed}.npz',rows=rows,point=point,ids=ids,videos=videos,error_sums=er,counts=ct,
                                local=np.concatenate(locals_),energy=np.concatenate(energies))
            result['feature']={'raw':float(er[:,0].sum()/ct[:,0].sum()),'R':float(er[:,2].sum()/ct[:,0].sum())}
        names=('raw','full_R','learned_g') if domain=='natural' else ('raw','full_R','learned_g','mask_R')
        result[domain]={name:{label:metrics(rows[:,i],mask,videos) for label,mask in masks.items()} for i,name in enumerate(names)}
    return result


def technical(model,cache,stats):
    results={};j=cache['syn_fit'][:4];maps,z,c,e=base.syn_batch(cache,j)
    xy=base.gpu(cache['xy'][cache['parent'][j]]);point=base.gpu(cache['point'][j]);valid=base.gpu(cache['state'][cache['parent'][j]])>0
    for kind in ARMS:
        net=old.make_net(kind,stats);opt=torch.optim.AdamW(net.parameters(),lr=1e-4);torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
        net.train();r=generate(net,maps,j)
        with torch.no_grad(): expected=old.restore(net,maps,j)
        torch.testing.assert_close(r,expected)
        loss,parts,_=endpoint_loss(model,r,maps[:,2],z,c,e,xy,point,valid)
        pgrad=torch.autograd.grad(parts['position'],tuple(net.parameters()),retain_graph=True,allow_unused=True)
        fgrad=torch.autograd.grad(.01*(parts['mse']+parts['cosine']),tuple(net.parameters()),retain_graph=True,allow_unused=True)
        norm=lambda gs:float(sum(g.square().sum() for g in gs if g is not None).sqrt())
        pg,fg=norm(pgrad),norm(fgrad);assert pg>0 and fg>0
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        with torch.no_grad():after=float(endpoint_loss(model,generate(net,maps,j),maps[:,2],z,c,e,xy,point,valid)[0])
        assert all(p.grad is None for p in model.parameters())
        results[kind]=dict(position_gradient=pg,weighted_feature_gradient=fg,feature_to_position_gradient=fg/pg,
            before=float(loss.detach()),after=after,peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-tick)
        del net,opt,r,loss,pgrad,fgrad;gc.collect();torch.cuda.empty_cache()
    base.write(OUT/'technical_checks.json',results);print('technical',json.dumps(results),flush=True)


def main(resume=False):
    if resume:assert OUT.is_dir()
    else:OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    prior=read(old.OUT/'plan.json');paths=list(prior['hashes'])+[str(Path(__file__).relative_to(ROOT)),
        'tests/test_messenger_position_primary.py','keypoint_comparison1/landmark_detection/heads.py']
    for p,h in prior['hashes'].items():assert base.sha(ROOT/p)==h,p
    plan=dict(**PLAN,hashes={p:base.sha(ROOT/p) for p in paths});identity=signature(plan)
    if resume:assert read(OUT/'plan.json')==json.loads(json.dumps(plan))
    else:
        base.write(OUT/'plan.json',plan)
        for p in paths:
            if p.endswith('.py'):
                dest=OUT/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest);dest.chmod(0o444)
    try:
        with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            base.status(OUT,'preparing')
            config=yaml.safe_load(base.CONFIG.read_text());model,_=base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
            base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True);model.cuda().eval().requires_grad_(False)
            frozen=base.tensor_digest(model);cache=base.make_cache(OUT,model,config['data']);manifest=read(OUT/'manifest.json')
            for k in ('roles','faces','image_hashes'):assert manifest[k]==read(old.OUT/'manifest.json')[k]
            base.write(OUT/'teacher_audit.json',old.teacher_cache(cache,manifest))
            fit=np.unique(cache['parent'][cache['syn_fit']]);f=cache['clean'][fit,2].float();stats=(f.mean((0,2,3)),f.std((0,2,3),unbiased=False).clamp_min(.01));del f
            base.write(OUT/'normalization.json',dict(fit=fit.tolist(),mean=stats[0].tolist(),std=stats[1].tolist()))
            if not (OUT/'technical_checks.json').exists():technical(model,cache,stats)
            results={}
            for kind in ARMS:
                folder=OUT/kind;folder.mkdir(exist_ok=True);aid=signature(dict(plan=identity,arm=kind))
                if (folder/'evaluation.json').exists():
                    m=read(folder/'evaluation.json');assert m['identity']==aid
                    for p,h in m['weights'].items():assert base.sha(folder/p)==h
                else:
                    net=train_recovery(folder,kind,model,cache,stats,aid);recovered=candidates(net,cache)
                    gate=train_gate(folder,kind,model,cache,recovered,aid);base.status(OUT,'evaluating',arm=kind)
                    m=evaluate(folder,net,gate,model,cache)
                    if kind.startswith('diff'):
                        m['sample_sensitivity']={str(seed):evaluate(folder,net,gate,model,cache,seed) for seed in (1042,2042,3042)}
                    m.update(identity=aid,R=read(folder/'R_convergence.json'),g=read(folder/'g_convergence.json'),weights={p.name:base.sha(p) for p in folder.glob('*.pt')})
                    base.write(folder/'evaluation.json',m);del net,gate,recovered;gc.collect();torch.cuda.empty_cache()
                results[kind]=m;base.write(OUT/'comparison.json',results)
                assert base.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
            for p,h in plan['hashes'].items():assert base.sha(ROOT/p)==h
            files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=base.sha(p)) for p in OUT.rglob('*.pt')];assert len(files)==8
            base.write(OUT/'summary.json',dict(conditions=results,checkpoints=files,checkpoint_bytes=sum(p['bytes'] for p in files)))
            base.write(OUT/'training_completion.json',dict(all_converged=True,weights_retained=True,clean_head=False,fixed_parent=frozen,source_verified=True))
            base.status(OUT,'training_and_inference_completed',arms=2,weights_retained=True)
    except Exception as error:
        base.status(OUT,'failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args();main(args.resume)
