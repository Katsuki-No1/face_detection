"""Nine-arm validation-plateau comparison; retain every arm until all evaluation ends."""
import argparse
import copy
from dataclasses import dataclass, asdict
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_messenger_multiaxis as base
from scripts.report_messenger_multiaxis import interval
from src.models.messenger_multiaxis import MultiAxisRecovery,reconstruction_loss,feature_error
from src.models.natural_feature_gate import NaturalFeatureGate

OUT=ROOT/'outputs/diagnostics/messenger_multiaxis_convergence_20261114'
PLAN=copy.deepcopy(base.PLAN)
PLAN.update(epochs=600,gate_epochs=300,R_min_epochs=40,g_min_epochs=20,
            lr_patience=8,stop_patience=15,R_min_lr=1e-6,g_min_lr=5e-6,
            relative_delta=.001,R_absolute_delta=1e-5,g_absolute_delta=1e-6,
            retention='retain best and optimizer-resumable last for all nine arms; automatic deletion disabled',
            convergence='validation plateau, not mathematical convergence: both R fused and full objective; g dev position; plateau at minimum LR for 15 epochs',
            cap_policy='budget cap is NOT convergence; no successful completion if any stage hits cap',
            initialization='fresh seed42 identical to initial four-epoch comparison; old weights were deleted',
            same_data='exact same manifest roles and face IDs as initial four-epoch comparison',
            eval_timing='only after validation plateau; eval excluded from scheduler/selection',
            storage_budget_mib=768)


@dataclass
class Plateau:
    lr: float
    floor: float
    minimum: int
    rel: float=.001
    absolute: float=1e-5
    reduce_patience: int=8
    stop_patience: int=15
    anchors: list | None=None
    last_improvement: int=0
    last_reduction: int=0

    def step(self,epoch,values):
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite validation')
        if self.anchors is None:
            self.anchors=list(values);self.last_improvement=epoch
        else:
            if len(values)!=len(self.anchors):raise ValueError('Changed monitored objectives')
            changed=False
            for i,(value,best) in enumerate(zip(values,self.anchors)):
                if value<best-max(self.absolute,abs(best)*self.rel):
                    self.anchors[i]=value;changed=True
            if changed:self.last_improvement=epoch
        stale=epoch-max(self.last_improvement,self.last_reduction)
        if stale>=self.reduce_patience and self.lr>self.floor:
            self.lr=max(self.floor,self.lr*.5);self.last_reduction=epoch
            stale=0
        return epoch>=self.minimum and self.lr<=self.floor and stale>=self.stop_patience


def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def load_checkpoint(path,identity):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['identity']!=identity:raise ValueError('Resume identity mismatch')
    return ck


def set_lr(opt,lr):
    for group in opt.param_groups:group['lr']=lr


def checkpoint(net,opt,epoch,controller,rng,identity,converged=False):
    return dict(state=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                controller=asdict(controller),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                identity=identity,converged=converged)


@torch.no_grad()
def validation(net,cache,condition):
    net.eval();ids=cache['syn_dev'];sums=np.zeros(6);nc=ne=0
    for b in base.batches(len(ids)):
        x,z,c,e=base.syn_batch(cache,ids[b]);y,r,w=net(x)
        errors=[feature_error(y,z),feature_error(r,z[:,None]).mean(1),r.var(1,unbiased=False).mean(1)*(2*r.shape[1]/(r.shape[1]-1))]
        sums+=np.array([float((err*m).sum()) for err in errors for m in (c,e)])
        nc+=int(c.sum());ne+=int(e.sum())
    vals=sums.reshape(3,2);region=vals[:,0]/nc+.1*vals[:,1]/max(ne,1)
    return dict(fused=float(vals[0,0]/nc),objective=float(region[0]+(region[1] if condition!='A' else 0)+(.1*region[2] if condition=='C' else 0)),
                branch=float(vals[1,0]/nc),consistency=float(vals[2,0]/nc))


def history_file(folder,stage):return folder/('history.json' if stage=='R' else 'gate_history.json')


def initialize_stage(net,folder,stage,identity):
    initial_lr=PLAN['lr'] if stage=='R' else PLAN['gate_lr']
    opt=torch.optim.AdamW(net.parameters(),lr=initial_lr,weight_decay=PLAN['weight_decay'])
    rng=np.random.default_rng(42);history=[];best=float('inf');start=0
    ctrl=Plateau(initial_lr,PLAN['R_min_lr'] if stage=='R' else PLAN['g_min_lr'],
                 PLAN['R_min_epochs'] if stage=='R' else PLAN['g_min_epochs'],
                 absolute=PLAN['R_absolute_delta'] if stage=='R' else PLAN['g_absolute_delta'])
    last=folder/f'last_{stage}.pt';converged=False
    if last.exists():
        ck=load_checkpoint(last,identity);net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer'])
        # Optimizer buffers are moved to parameter devices by load_state_dict.
        ctrl=Plateau(**ck['controller']);rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        start=ck['epoch']+1;converged=ck['converged']
        history=json.loads(history_file(folder,stage).read_text())
        history=[row for row in history if row['epoch']<=ck['epoch']]
        best=min(row['dev']['fused'] if stage=='R' else row['dev'] for row in history)
    return opt,rng,history,best,start,ctrl,converged


def train_recovery(out,folder,family,condition,cache,identity):
    torch.manual_seed(42);net=MultiAxisRecovery(family).cuda()
    opt,rng,history,best,start,ctrl,converged=initialize_stage(net,folder,'R',identity)
    if not converged:
        for epoch in range(start,PLAN['epochs']+1):
            used_lr=opt.param_groups[0]['lr']
            if epoch:
                net.train();ids=rng.permutation(cache['syn_fit']);totals=[]
                for b in base.batches(len(ids)):
                    x,z,c,e=base.syn_batch(cache,ids[b]);loss,parts=reconstruction_loss(net(x),z,c,e,condition)
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
                    totals.append([float(loss.detach()),*[float(v.detach()) for v in parts.values()]])
            dev=validation(net,cache,condition)
            row=dict(epoch=epoch,dev=dev,lr=used_lr)
            if epoch:row['train_mean']=np.mean(totals,0).tolist()
            if dev['fused']<best:
                best=dev['fused'];save(folder/'best_recovery.pt',dict(state=net.state_dict(),epoch=epoch,identity=identity,dev=dev))
            converged=ctrl.step(epoch,[dev['fused'],dev['objective']]);set_lr(opt,ctrl.lr)
            row['controller']=asdict(ctrl);row['converged']=converged;history.append(row)
            base.write(history_file(folder,'R'),history)
            save(folder/'last_R.pt',checkpoint(net,opt,epoch,ctrl,rng,identity,converged))
            base.status(out,'training',arm=folder.name,stage='R',epoch=epoch,dev=dev,lr=ctrl.lr,converged=converged)
            if converged:break
    stop=history[-1]['epoch']
    if not converged:raise RuntimeError(f'{folder.name} R hit {stop} epoch cap without validation convergence; last checkpoint retained')
    ck=load_checkpoint(folder/'best_recovery.pt',identity);net.load_state_dict(ck['state']);net.eval()
    base.write(folder/'R_convergence.json',dict(converged=True,stop_epoch=stop,selected_epoch=ck['epoch'],controller=asdict(ctrl)))
    del opt;return net


def train_gate(out,folder,model,cache,recovered,identity):
    torch.manual_seed(42);gate=NaturalFeatureGate(384).cuda()
    opt,rng,history,best,start,ctrl,converged=initialize_stage(gate,folder,'g',identity)
    if not converged:
        for epoch in range(start,PLAN['gate_epochs']+1):
            record=dict(epoch=epoch,lr=opt.param_groups[0]['lr'])
            for role in (('dev',) if epoch==0 else ('fit','dev')):
                ids=cache['natural_'+role].copy();training=role=='fit'
                gate.train(training)
                if training:rng.shuffle(ids)
                num=0.;den=0
                with torch.set_grad_enabled(training):
                    for b in base.batches(len(ids)):
                        j=ids[b];_,y=gate(base.gpu(cache['clean'][j,2]),base.gpu(recovered[j]),1.)
                        coords=base.run_head(model.head,y)['decoded_coordinates'];valid=base.gpu(cache['state'][j])>0
                        error=F.smooth_l1_loss(coords,base.gpu(cache['xy'][j]),beta=.02,reduction='none').mean(-1)
                        loss=(error*valid).sum()/valid.sum().clamp_min(1)
                        if training:
                            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),5,error_if_nonfinite=True);opt.step()
                        num+=float((error.detach()*valid).sum());den+=int(valid.sum())
                record[role]=num/den
            if record['dev']<best:
                best=record['dev'];save(folder/'best_gate.pt',dict(state=gate.state_dict(),epoch=epoch,identity=identity,dev=best))
            converged=ctrl.step(epoch,[record['dev']]);set_lr(opt,ctrl.lr)
            record['controller']=asdict(ctrl);record['converged']=converged;history.append(record)
            base.write(history_file(folder,'g'),history)
            save(folder/'last_g.pt',checkpoint(gate,opt,epoch,ctrl,rng,identity,converged))
            base.status(out,'training',arm=folder.name,stage='g',epoch=epoch,dev=record['dev'],lr=ctrl.lr,converged=converged)
            if converged:break
    stop=history[-1]['epoch']
    if not converged:raise RuntimeError(f'{folder.name} g hit {stop} epoch cap without validation convergence; last checkpoint retained')
    ck=load_checkpoint(folder/'best_gate.pt',identity);gate.load_state_dict(ck['state']);gate.eval()
    base.write(folder/'g_convergence.json',dict(converged=True,stop_epoch=stop,selected_epoch=ck['epoch'],controller=asdict(ctrl)))
    del opt;return gate


@torch.no_grad()
def extra_evaluation(folder,model,net,gate,cache,recovered):
    natural=cache['natural_eval'];gs=[]
    for b in base.batches(len(natural)):
        j=natural[b];logits,_=gate(base.gpu(cache['clean'][j,2]),base.gpu(recovered[j]),1.)
        ij=((base.gpu(cache['xy'][j])*255+.5)/16).floor().long().clamp(0,15)
        p=logits.sigmoid()[:,0];v=p[torch.arange(len(j),device='cuda')[:,None],ij[...,1],ij[...,0]]
        gs.append(v.cpu().numpy())
    gs=np.concatenate(gs);state=cache['state'][natural].numpy()
    result=dict(gate_at_points={str(s):dict(mean=float(gs[state==s].mean()),median=float(np.median(gs[state==s]))) for s in (1,2)})
    np.savez_compressed(folder/'gate_points.npz',gate=gs,state=state,ids=natural)
    ids=cache['syn_eval'];errors=[];counts=[];headrows=[]
    for b in base.batches(len(ids)):
        j=ids[b];x,z,c,e=base.syn_batch(cache,j);r,branches,w=net(x);_,fused=gate(x[:,2],r,1.)
        vals=[x[:,2],r,*branches.unbind(1)]
        errors.append(torch.stack([(feature_error(v,z)*c).sum((1,2)) for v in vals],1).cpu().numpy())
        counts.append(c.sum((1,2)).cpu().numpy());out=[]
        for v in (x[:,2],r,fused):
            pred=base.run_head(model.head,v);xy=base.gpu(cache['xy'][cache['parent'][j]])
            out.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
        headrows.append(torch.stack(out,1).cpu().numpy())
    err=np.concatenate(errors);ct=np.concatenate(counts);hr=np.concatenate(headrows);covered=cache['point'][ids].numpy()
    np.savez_compressed(folder/'synthetic_rows.npz',error_sums=err,counts=ct,rows=hr,point=covered,
                        videos=cache['videos'][cache['parent'][ids]],kind=cache['kind'][ids],fraction=cache['fraction'][ids])
    result['synthetic_covered_head']={}
    for i,name in enumerate(('raw','full_recovery','learned_gate')):
        result['synthetic_covered_head'][name]=dict(points=int(covered.sum()),
            correct=int(((hr[:,i,:,0]<=.1)&(hr[:,i,:,1]>=.7)&covered).sum()),mean_error=float(hr[:,i,:,0][covered].mean()))
    return result


def summary(out,results):
    rows={}
    for name,m in results.items():
        p=np.load(out/name/'natural_points.npz');a=np.load(out/(name.rsplit('_',1)[0]+'_A')/'natural_points.npz')
        assert np.array_equal(p['ids'],a['ids']) and np.array_equal(p['state'],a['state'])
        ok=(p['rows'][...,0]<=.1)&(p['rows'][...,1]>=.7);ao=(a['rows'][:,2,:,0]<=.1)&(a['rows'][:,2,:,1]>=.7)
        rows[name]=dict(R=m['R_convergence'],g=m['g_convergence'],artificial_fused=m['synthetic']['fused'],
            artificial_branches=m['synthetic']['branches'],natural=m['learned_gate'],full_recovery=m['full_recovery'],raw=m['raw'],
            gate_at_points=m['extra']['gate_at_points'],synthetic_head=m['extra']['synthetic_covered_head'],
            versus_raw={k:interval(ok[:,2],ok[:,0],p['state'],p['videos'],s) for k,s in [('occluded',1),('visible',2)]},
            versus_A={k:interval(ok[:,2],ao,p['state'],p['videos'],s) for k,s in [('occluded',1),('visible',2)]})
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=base.sha(p)) for p in sorted(out.rglob('*.pt'))]
    assert len(files)==36,'Expected best and resumable last R/g for nine conditions'
    base.write(out/'summary.json',dict(conditions=rows,all_evaluations_completed=True,retained_checkpoints=files,
         checkpoint_bytes=sum(p['bytes'] for p in files),limitations=['validation plateau criterion, not proof of a global optimum',
         'same small training subsets as four-epoch comparison; no full-data claim','single seed and unadjusted comparisons',
         'historical train exposure and reused evaluation; GT crops, no global val/test','family capacities not matched']))


def main(out,resume=False):
    if resume:
        if not out.is_dir():raise RuntimeError('No run to resume')
    else:out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
    files=(base.BASE,base.CONFIG,base.ANN,base.PRIOR,Path(__file__),Path(base.__file__),
           ROOT/'src/models/messenger_multiaxis.py',ROOT/'src/models/natural_feature_gate.py',
           ROOT/'scripts/train_v3_shape_recovery_gate.py',ROOT/'scripts/train_natural_feature_gate.py',
           ROOT/'src/data/v3_shape_occlusion.py',ROOT/'scripts/report_messenger_multiaxis.py')
    hashes={str(p.relative_to(ROOT)):base.sha(p) for p in files}
    plan=dict(**PLAN,hashes=hashes);identity=signature(plan)
    if resume:
        if json.loads((out/'plan.json').read_text())!=plan:raise RuntimeError('Plan/source/data changed; cannot resume')
    else:base.write(out/'plan.json',plan)
    try:
        with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            base.status(out,'preparing',resume=resume)
            raw=yaml.safe_load(base.CONFIG.read_text());model,_=base.build_landmark_model(raw,variant='query_refinement',initialize_from_detector=False)
            base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True)
            assert base.sha(base.BASE)=='11b430b595e56395a14c266cb364bd50848b1a51b5b689548eb6749de0a23856'
            model.requires_grad_(False).eval().cuda();frozen=base.tensor_digest(model)
            cache=base.make_cache(out,model,raw['data'])
            old=json.loads((base.DEFAULT_OUT/'manifest.json').read_text());new=json.loads((out/'manifest.json').read_text())
            for key in ('roles','faces','image_hashes'):assert old[key]==new[key],f'Changed data: {key}'
            results={}
            for family in PLAN['families']:
                for condition in PLAN['conditions']:
                    name=f'{family}_{condition}';folder=out/name;folder.mkdir(exist_ok=True)
                    arm_identity=signature(dict(plan=identity,arm=name))
                    if (folder/'evaluation.json').exists():
                        m=json.loads((folder/'evaluation.json').read_text());assert m['identity']==arm_identity
                        for n,h in m['weights'].items():assert base.sha(folder/n)==h
                        results[name]=m;continue
                    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
                    net=train_recovery(out,folder,family,condition,cache,arm_identity)
                    recovered=base.natural_candidates(net,cache)
                    gate=train_gate(out,folder,model,cache,recovered,arm_identity)
                    base.status(out,'evaluating',arm=name)
                    metrics=base.head_evaluate(folder,model,net,gate,cache,recovered)
                    metrics['extra']=extra_evaluation(folder,model,net,gate,cache,recovered)
                    metrics.update(identity=arm_identity,parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-started,
                        R_convergence=json.loads((folder/'R_convergence.json').read_text()),g_convergence=json.loads((folder/'g_convergence.json').read_text()),
                        weights={p.name:base.sha(p) for p in folder.glob('*.pt')},peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    assert base.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
                    base.write(folder/'evaluation.json',metrics);results[name]=metrics;base.write(out/'comparison.json',results)
                    base.status(out,'arm_completed',arm=name,completed=len(results),total=9)
                    del recovered,net,gate;gc.collect();torch.cuda.empty_cache()
            assert all(base.sha(ROOT/p)==v for p,v in hashes.items()),'Execution source changed'
            summary(out,results)
            base.write(out/'completion.json',dict(completed_arms=list(results),all_R_and_g_validation_plateaus=True,
                fixed_model_hash=frozen,sources_verified=True,weights_retained=True,production=False))
            base.status(out,'completed',completed=9,weights_retained=True)
    except Exception as error:
        base.status(out,'failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--resume',action='store_true')
    a=p.parse_args();main(a.output.resolve(),a.resume)
