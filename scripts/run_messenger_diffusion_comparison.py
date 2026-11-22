"""Four-arm convergence comparison with the established natural soft-mask teacher."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import shutil
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_messenger_multiaxis as base
from scripts.run_messenger_multiaxis_convergence import Plateau,save,signature,set_lr
from scripts.diagnose_messenger_local_recovery import point_features,boot
from scripts.train_natural_feature_gate import build_crop_transform,ellipse_aligned_rectangle
from src.data.natural_gate_teacher import patch_teacher
from src.models.messenger_multiaxis import MultiAxisRecovery,reconstruction_loss,feature_error,region_mean
from src.models.messenger_feature_diffusion import FeatureDiffusion,noise_for
from src.models.natural_feature_gate import NaturalFeatureGate

OUT=ROOT/'outputs/diagnostics/messenger_diffusion_comparison_20261114'
OLD=ROOT/'outputs/diagnostics/messenger_multiaxis_convergence_20261114'
ARMS=('det_feature','det_head','diff_feature','diff_head')
PLAN=dict(seed=42,arms=ARMS,batch=16,lr=1e-4,gate_lr=5e-4,weight_decay=.01,
    head_weight=10.,gate_soft_bce=.1,steps=16,primary_sample_seed=42,extra_sample_seeds=[1042,2042,3042],
    R_min_epochs=40,g_min_epochs=20,R_cap=600,g_cap=300,R_lr_floor=1e-6,g_lr_floor=5e-6,
    reduce_patience=8,stop_patience=15,relative_delta=.001,
    R_selection='minimum dev covered cosine+MSE; same criterion for all arms',
    g_selection='minimum dev position+.1 confidence-weighted soft BCE',
    R_stop='plateau in sampled dev feature, masked-input head position, and own objective at minimum LR',
    g_stop='plateau in dev joint objective, position and teacher BCE at minimum LR',
    data='same manifest as nine-arm convergence; synthetic fit256/dev128/eval128 x3; natural fit512/dev128/eval512',
    head_aux='10 x all-annotated-point SmoothL1 beta .02 on mask*R+(1-mask)*X through frozen head',
    diffusion='cosine VP v-prediction; independent scale branches; mean branch MSE + fused MSE + .1 pairwise MSE; region weights 1/.1',
    diffusion_head_aux='same head loss on predicted x0 from training noisy clean features; not a differentiated full sampling trajectory',
    normalization='clean synthetic-fit final maps only; per-channel mean/std, std floor .01',
    validation_noise='fixed per-item noise and times; no eval use for selection',
    natural_teacher='unchanged natural_gate_teacher.patch_teacher: ellipse, nose valley, confidence-weighted weak interpolation',
    inference='no GT point/state/teacher mask input; GT crop as in prior experiment',
    retention='all best and optimizer/RNG last checkpoints retained, no automatic deletion',production=False)


def read(p):return json.loads(p.read_text())


def make_net(kind,stats):
    torch.manual_seed(42)
    return (FeatureDiffusion(*stats) if kind.startswith('diff') else MultiAxisRecovery('scale')).cuda()


def restore(net,maps,ids,seed=42):
    return net.sample(maps,ids,seed,PLAN['steps']) if isinstance(net,FeatureDiffusion) else net(maps)[0]


def head_loss(model,value,xy,state):
    error=F.smooth_l1_loss(base.run_head(model.head,value)['decoded_coordinates'],xy,beta=.02,reduction='none').mean(-1)
    valid=state>0
    return (error*valid).sum()/valid.sum().clamp_min(1)


def objective(net,maps,z,c,e,ids,epoch):
    if not isinstance(net,FeatureDiffusion):
        outputs=net(maps);loss,_=reconstruction_loss(outputs,z,c,e,'C')
        return loss,outputs[0]
    # Identical noise/time per item and epoch across feature/head arms.
    t=torch.tensor([np.random.default_rng(71+epoch*100000+int(i)).uniform(.001,.999) for i in ids],device=z.device,dtype=z.dtype)
    noise=noise_for(ids,z.shape[1:],z.device,907+epoch*1009)
    (v,branches,w),target,r=net.training_pair(maps,z,t,noise)
    main=region_mean((v-target).square().mean(1),c,e)
    branch=region_mean((branches-target[:,None]).square().mean((1,2)),c,e)
    consistency=region_mean(branches.var(1,unbiased=False).mean(1)*3,c,e)
    return main+branch+.1*consistency,r


def teacher_cache(cache,manifest):
    anns={a['id']:a for a in read(base.ANN)['annotations']};targets=[];weights=[]
    for i,fid in enumerate(manifest['faces']):
        a=anns[fid];kp=np.array(a['keypoints']).reshape(5,3)
        box,angle=ellipse_aligned_rectangle(a['ellipse'])
        tr=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        state=cache['state'][i].numpy();valid=state>0
        np.testing.assert_allclose(tr.original_to_crop(kp[:,:2])[valid]/255,cache['xy'][i].numpy()[valid],atol=1e-6)
        y,w=patch_teacher(a['ellipse'],kp[:,:2],state,tr);targets.append(y);weights.append(w)
    y=torch.tensor(np.stack(targets));w=torch.tensor(np.stack(weights))
    assert torch.isfinite(y).all() and torch.isfinite(w).all() and y.min()>=0 and y.max()<=1.00001
    cache['teacher']=y;cache['confidence']=w
    return dict(nonzero_weight_cells=int((w>0).sum()),target_range=[float(y.min()),float(y.max())],
                source_sha=base.sha(ROOT/'src/data/natural_gate_teacher.py'),teacher_sum=float((y*w).sum()))


def initial(net,folder,stage,identity):
    lr=PLAN['lr'] if stage=='R' else PLAN['gate_lr'];floor=PLAN['R_lr_floor'] if stage=='R' else PLAN['g_lr_floor']
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=PLAN['weight_decay'])
    ctrl=Plateau(lr,floor,PLAN[f'{stage}_min_epochs'],absolute=1e-5 if stage=='R' else 1e-6)
    rng=np.random.default_rng(42);history=[];best=float('inf');start=0;done=False
    path=folder/f'last_{stage}.pt'
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['identity']==identity
        net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);ctrl=Plateau(**ck['controller'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        history=read(folder/f'{stage}_history.json')[:ck['epoch']+1]
        start=ck['epoch']+1;best=ck['best'];done=ck['converged']
    return opt,ctrl,rng,history,best,start,done


def persist(folder,stage,net,opt,ctrl,rng,history,best,identity,done):
    base.write(folder/f'{stage}_history.json',history)
    save(folder/f'last_{stage}.pt',dict(state=net.state_dict(),optimizer=opt.state_dict(),controller=asdict(ctrl),
        rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),epoch=history[-1]['epoch'],best=best,identity=identity,converged=done))


@torch.no_grad()
def validate_R(net,model,cache,aux):
    net.eval();vals=[]
    for b in base.batches(len(cache['syn_dev'])):
        j=cache['syn_dev'][b];x,z,c,e=base.syn_batch(cache,j)
        r=restore(net,x,j);loss,estimate=objective(net,x,z,c,e,j,0)
        xy=base.gpu(cache['xy'][cache['parent'][j]]);st=base.gpu(cache['state'][cache['parent'][j]])
        hp=head_loss(model,torch.where(c[:,None],r,x[:,2]),xy,st)
        train_hp=head_loss(model,torch.where(c[:,None],estimate,x[:,2]),xy,st) if aux else hp*0
        vals.append([float((feature_error(r,z)*c).sum()),int(c.sum()),float(hp)*len(j),
                     float(loss+PLAN['head_weight']*train_hp)*len(j),len(j)])
    a=np.sum(vals,0)
    return dict(feature=float(a[0]/a[1]),position=float(a[2]/a[4]),objective=float(a[3]/a[4]))


def train_R(out,folder,kind,model,cache,stats,identity):
    net=make_net(kind,stats);aux=kind.endswith('head')
    opt,ctrl,rng,history,best,start,done=initial(net,folder,'R',identity)
    if not done:
        for epoch in range(start,PLAN['R_cap']+1):
            tick=time.monotonic();train=[]
            if epoch:
                net.train();ids=rng.permutation(cache['syn_fit'])
                for b in base.batches(len(ids)):
                    j=ids[b];x,z,c,e=base.syn_batch(cache,j);loss,r=objective(net,x,z,c,e,j,epoch)
                    if aux:
                        loss=loss+PLAN['head_weight']*head_loss(model,torch.where(c[:,None],r,x[:,2]),
                            base.gpu(cache['xy'][cache['parent'][j]]),base.gpu(cache['state'][cache['parent'][j]]))
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();train.append(float(loss.detach()))
            dev=validate_R(net,model,cache,aux)
            if dev['feature']<best:
                best=dev['feature'];save(folder/'best_R.pt',dict(state=net.state_dict(),epoch=epoch,dev=dev,identity=identity))
            done=ctrl.step(epoch,list(dev.values()));set_lr(opt,ctrl.lr)
            row=dict(epoch=epoch,dev=dev,train=float(np.mean(train)) if train else None,lr=ctrl.lr,seconds=time.monotonic()-tick,converged=done)
            history.append(row);persist(folder,'R',net,opt,ctrl,rng,history,best,identity,done)
            base.status(out,'training',arm=kind,stage='R',**row)
            if done:break
    if not done:raise RuntimeError(f'{kind} R safety cap reached without convergence; retained for review')
    ck=torch.load(folder/'best_R.pt',weights_only=True);net.load_state_dict(ck['state']);net.requires_grad_(False).eval()
    base.write(folder/'R_convergence.json',dict(converged=True,stop_epoch=history[-1]['epoch'],selected_epoch=ck['epoch']))
    del opt;return net


@torch.no_grad()
def candidates(net,cache):
    return torch.cat([restore(net,base.gpu(cache['clean'][j]),j).half().cpu() for j in base.batches(len(cache['clean']))])


def gate_pass(gate,model,cache,recovered,ids,opt=None):
    totals=np.zeros(4)
    with torch.set_grad_enabled(opt is not None):
        for b in base.batches(len(ids)):
            j=ids[b];logits,y=gate(base.gpu(cache['clean'][j,2]),base.gpu(recovered[j]),1.)
            coords=base.run_head(model.head,y)['decoded_coordinates'];st=base.gpu(cache['state'][j]);valid=st>0
            error=F.smooth_l1_loss(coords,base.gpu(cache['xy'][j]),beta=.02,reduction='none').mean(-1)
            w=base.gpu(cache['confidence'][j]);target=base.gpu(cache['teacher'][j])
            ce=F.binary_cross_entropy_with_logits(logits[:,0],target,reduction='none')
            pos=(error*valid).sum()/valid.sum().clamp_min(1);bce=(ce*w).sum()/w.sum().clamp_min(1e-8)
            if opt is not None:
                opt.zero_grad(set_to_none=True);(pos+PLAN['gate_soft_bce']*bce).backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(),5,error_if_nonfinite=True);opt.step()
            totals += [float((error.detach()*valid).sum()),int(valid.sum()),float((ce.detach()*w).sum()),float(w.sum())]
    pos,bce=totals[0]/totals[1],totals[2]/totals[3]
    return dict(joint=float(pos+PLAN['gate_soft_bce']*bce),position=float(pos),soft_bce=float(bce))


def train_g(out,folder,kind,model,cache,recovered,identity):
    torch.manual_seed(42);gate=NaturalFeatureGate(384).cuda()
    opt,ctrl,rng,history,best,start,done=initial(gate,folder,'g',identity)
    if not done:
        for epoch in range(start,PLAN['g_cap']+1):
            train=gate_pass(gate,model,cache,recovered,rng.permutation(cache['natural_fit']),opt) if epoch else None
            dev=gate_pass(gate,model,cache,recovered,cache['natural_dev'])
            if dev['joint']<best:
                best=dev['joint'];save(folder/'best_g.pt',dict(state=gate.state_dict(),epoch=epoch,dev=dev,identity=identity))
            done=ctrl.step(epoch,list(dev.values()));set_lr(opt,ctrl.lr)
            row=dict(epoch=epoch,dev=dev,train=train,lr=ctrl.lr,converged=done);history.append(row)
            persist(folder,'g',gate,opt,ctrl,rng,history,best,identity,done)
            base.status(out,'training',arm=kind,stage='g',**row)
            if done:break
    if not done:raise RuntimeError(f'{kind} g safety cap without convergence')
    ck=torch.load(folder/'best_g.pt',weights_only=True);gate.load_state_dict(ck['state']);gate.eval().requires_grad_(False)
    base.write(folder/'g_convergence.json',dict(converged=True,stop_epoch=history[-1]['epoch'],selected_epoch=ck['epoch']))
    del opt;return gate


def summarize_rows(rows,selected,videos):
    ok=(rows[...,0]<=.1)&(rows[...,1]>=.7)
    return dict(points=int(selected.sum()),correct=int(ok[selected].sum()),mean_error=boot(rows[...,0],selected,videos),
                presence_pass=int((rows[...,1][selected]>=.7).sum()))


@torch.no_grad()
def evaluate(folder,net,gate,model,cache,recovered,seed=42):
    result={};nat=[];gs=[];ids=cache['natural_eval']
    for b in base.batches(len(ids)):
        j=ids[b];x=base.gpu(cache['clean'][j,2]);r=base.gpu(recovered[j]) if seed==42 else restore(net,base.gpu(cache['clean'][j]),j,seed)
        logits,y=gate(x,r,1.);preds=[];xy=base.gpu(cache['xy'][j])
        for value in (x,r,y):
            pred=base.run_head(model.head,value);preds.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
        nat.append(torch.stack(preds,1).cpu().numpy())
        ij=((xy*255+.5)/16).floor().long().clamp(0,15)
        gs.append(logits.sigmoid()[:,0][torch.arange(len(j),device='cuda')[:,None],ij[:,:,1],ij[:,:,0]].cpu().numpy())
    nat=np.concatenate(nat);gs=np.concatenate(gs);state=cache['state'][ids].numpy();videos=cache['videos'][ids]
    np.savez_compressed(folder/f'natural_seed{seed}.npz',rows=nat,gate=gs,state=state,ids=ids,videos=videos)
    result['natural']={name:{label:summarize_rows(nat[:,i],state==s,videos) for s,label in [(1,'occluded'),(2,'visible')]} for i,name in enumerate(('raw','full_R','learned_g'))}
    result['gate_mean']={label:float(gs[state==s].mean()) for s,label in [(1,'occluded'),(2,'visible')]}
    ids=cache['syn_eval'];heads=[];errors=[];counts=[];local=[];energies=[]
    for b in base.batches(len(ids)):
        j=ids[b];x,z,c,e=base.syn_batch(cache,j);r=restore(net,x,j,seed);_,y=gate(x[:,2],r,1.)
        xy=base.gpu(cache['xy'][cache['parent'][j]]);pr=[]
        for value in (x[:,2],r,y,torch.where(c[:,None],r,x[:,2]),torch.where(c[:,None],z,x[:,2]),z):
            pred=base.run_head(model.head,value);pr.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
        heads.append(torch.stack(pr,1).cpu().numpy())
        errors.append(torch.stack([(feature_error(v,z)*m).sum((1,2)) for v in (x[:,2],r) for m in (c,e)],1).cpu().numpy())
        counts.append(torch.stack((c.sum((1,2)),e.sum((1,2))),1).cpu().numpy())
        zc,zl,valid=point_features(z,xy);ll=[];ee=[]
        for value in (x[:,2],r,z):
            vc,vl,_=point_features(value,xy);er=1-F.cosine_similarity(vc,zc,dim=-1)+(vc-zc).square().mean(-1)
            le=1-F.cosine_similarity(vl,zl,dim=-1)+(vl-zl).square().mean(-1)
            ll.append(torch.stack((er,(le*valid).sum(-1)/valid.sum(-1)),-1))
            avg=(vl*valid[...,None]).sum(-2)/valid.sum(-1)[...,None]
            ee.append((((vl-avg[...,None,:]).square().mean(-1))*valid).sum(-1)/valid.sum(-1))
        local.append(torch.stack(ll,1).cpu().numpy());energies.append(torch.stack(ee,1).cpu().numpy())
    h=np.concatenate(heads);err=np.concatenate(errors);ct=np.concatenate(counts);lc=np.concatenate(local);energy=np.concatenate(energies)
    point=cache['point'][ids].numpy();videos=cache['videos'][cache['parent'][ids]]
    np.savez_compressed(folder/f'artificial_seed{seed}.npz',rows=h,error_sums=err,counts=ct,local=lc,energy=energy,point=point,videos=videos,ids=ids)
    result['artificial']={name:summarize_rows(h[:,i],point,videos) for i,name in enumerate(('raw','full_R','learned_g','mask_R','mask_clean','clean'))}
    result['feature']={name:dict(covered=float(err[:,2*i].sum()/ct[:,0].sum()),outside=float(err[:,2*i+1].sum()/ct[:,1].sum()),
        local_center=boot(lc[:,i,:,0],point,videos),local_3x3=boot(lc[:,i,:,1],point,videos)) for i,name in enumerate(('raw','R'))}
    result['local_variance_R_over_clean']=float(energy[:,1][point].sum()/energy[:,2][point].sum())
    return result


def smoke(out,model,cache,stats):
    """Technical gradient and optimizer sanity, not a production quality gate."""
    result={};j=cache['syn_fit'][:4];x,z,c,e=base.syn_batch(cache,j)
    xy=base.gpu(cache['xy'][cache['parent'][j]]);st=base.gpu(cache['state'][cache['parent'][j]])
    for kind in ('det_head','diff_head'):
        net=make_net(kind,stats);opt=torch.optim.AdamW(net.parameters(),lr=1e-4);losses=[]
        for step in range(24):
            loss,r=objective(net,x,z,c,e,j,0);aux=head_loss(model,torch.where(c[:,None],r,x[:,2]),xy,st)
            if not step:
                grad=torch.autograd.grad(aux,r,retain_graph=True)[0];assert float(grad.norm())>0
            total=loss+PLAN['head_weight']*aux;opt.zero_grad(set_to_none=True);total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(float(total.detach()))
        assert losses[-1]<losses[0],(kind,losses)
        result[kind]=dict(initial=losses[0],final=losses[-1],updates=24)
        del net,opt
    j=cache['natural_fit'][:4];torch.manual_seed(42);gate=NaturalFeatureGate(384).cuda()
    x=base.gpu(cache['clean'][j,2]);logits,_=gate(x,x*.9,1.)
    w=base.gpu(cache['confidence'][j]);target=base.gpu(cache['teacher'][j])
    ce=(F.binary_cross_entropy_with_logits(logits[:,0],target,reduction='none')*w).sum()/w.sum()
    grad=torch.autograd.grad(ce,logits)[0];assert float(grad.norm())>0
    result['teacher_gradient']=float(grad.norm());assert all(p.grad is None for p in model.parameters())
    base.write(out/'technical_checks.json',result)


def main(resume=False):
    if not resume:OUT.mkdir(exist_ok=False)
    else:assert OUT.is_dir()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    old_hashes=read(OLD/'plan.json')['hashes']
    assert all(base.sha(ROOT/p)==sha for p,sha in old_hashes.items()),'Changed historical parent/source'
    paths=list(old_hashes)+[str(Path(__file__).relative_to(ROOT)),
        'src/models/messenger_feature_diffusion.py','src/data/natural_gate_teacher.py','scripts/diagnose_messenger_local_recovery.py']
    hashes={p:base.sha(ROOT/p) for p in paths};plan=dict(**PLAN,hashes=hashes);identity=signature(plan)
    if resume:assert read(OUT/'plan.json')==json.loads(json.dumps(plan)),'Source/plan mismatch'
    else:
        base.write(OUT/'plan.json',plan)
        for name in hashes:
            if not name.endswith('.py'):continue
            dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest);dest.chmod(0o444)
    try:
        with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);base.status(OUT,'preparing')
            raw=yaml.safe_load(base.CONFIG.read_text());model,_=base.build_landmark_model(raw,variant='query_refinement',initialize_from_detector=False)
            base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True);model.requires_grad_(False).cuda().eval()
            frozen=base.tensor_digest(model);cache=base.make_cache(OUT,model,raw['data']);manifest=read(OUT/'manifest.json')
            for key in ('roles','faces','image_hashes'):assert manifest[key]==read(OLD/'manifest.json')[key]
            base.write(OUT/'teacher_audit.json',teacher_cache(cache,manifest))
            fit=np.unique(cache['parent'][cache['syn_fit']]);f=cache['clean'][fit,2].float()
            mean=f.mean((0,2,3));std=f.std((0,2,3),unbiased=False).clamp_min(.01);stats=(mean,std);del f
            base.write(OUT/'normalization.json',dict(face_indices=fit.tolist(),mean=mean.tolist(),std=std.tolist(),split='fit'))
            if not (OUT/'technical_checks.json').exists():smoke(OUT,model,cache,stats)
            results={}
            for kind in ARMS:
                folder=OUT/kind;folder.mkdir(exist_ok=True);aid=signature(dict(plan=identity,arm=kind))
                if (folder/'evaluation.json').exists():
                    m=read(folder/'evaluation.json');assert m['identity']==aid
                    for name,sha in m['weights'].items():assert base.sha(folder/name)==sha
                    results[kind]=m;continue
                net=train_R(OUT,folder,kind,model,cache,stats,aid);recovered=candidates(net,cache)
                gate=train_g(OUT,folder,kind,model,cache,recovered,aid)
                base.status(OUT,'evaluating',arm=kind)
                m=evaluate(folder,net,gate,model,cache,recovered)
                if kind.startswith('diff'):
                    m['sample_sensitivity']={str(seed):evaluate(folder,net,gate,model,cache,recovered,seed) for seed in PLAN['extra_sample_seeds']}
                m.update(identity=aid,R=read(folder/'R_convergence.json'),g=read(folder/'g_convergence.json'),
                    parameters=sum(p.numel() for p in net.parameters()),weights={p.name:base.sha(p) for p in folder.glob('*.pt')})
                assert base.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
                base.write(folder/'evaluation.json',m);results[kind]=m;base.write(OUT/'comparison.json',results)
                base.status(OUT,'arm_completed',arm=kind,completed=len(results));del net,gate,recovered;gc.collect();torch.cuda.empty_cache()
            assert all(base.sha(ROOT/p)==sha for p,sha in hashes.items())
            files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=base.sha(p)) for p in OUT.rglob('*.pt')]
            assert len(files)==16
            base.write(OUT/'summary.json',dict(conditions=results,checkpoints=files,checkpoint_bytes=sum(p['bytes'] for p in files),all_numeric_evaluations_complete=True))
            base.write(OUT/'completion.json',dict(arms=list(results),sources_verified=True,fixed_model=frozen,weights_retained=True,all_converged=True))
            base.status(OUT,'completed',arms=4,weights_retained=True)
    except Exception as error:
        base.status(OUT,'failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.resume)
