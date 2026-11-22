"""Full official-train E16/K8192 comparison; artificial occlusion only."""
import argparse,copy,gc,hashlib,io,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_messenger_code_consistency as core
from src.models.messenger_full_e import ARMS,arm_spec,make_recovery
from src.data import messenger_full_e_cache as data
from src.data.messenger_balanced_occlusion import PLAN as MASK_PLAN

OUT=ROOT/'outputs/diagnostics/messenger_full_e_20261122'
GATE=ROOT/'outputs/training_gates/messenger_full_e_20261122'
PREP=ROOT/'outputs/diagnostics/messenger_full_e_preparation_20261122'
BASE=core.base
PLAN=dict(version=1,arms=list(ARMS),specs={a:arm_spec(a) for a in ARMS},seed=42,
    splits=dict(train=8688,val=1160,test=1009),data='official train all eligible; official val selection; official test evaluation',
    selection='all five visible and face_occluded=false and five crop coordinates valid',
    teacher_g='generated artificial binary patch mask only; no g model or optimizer',natural_evaluation=False,
    frozen=['DINO','keypoint_head'],mse_weight=.01,cosine_weight=.01,vq_weight=.01,commitment_weight=.25,
    lr=1e-4,weight_decay=.01,effective_batch=16,microbatch=16,R_cap=600,R_min_epochs=40,
    lr_floors=[1e-6],lr_reduce_patience=8,stop_patience=15,R_selection='minimum official val balanced position',
    kmeans=dict(chunk_size=512,max_iter=30,relative_tolerance=1e-5,initialization='seed42 fit patches without replacement'),
    initialization='existing train-only sensitivity basis; mean and 8192 kmeans codes from ALL 8688 clean training faces',
    dictionary='unit lookup/loss/decode; covered-only per-effective-batch EMA decay .99; no dictionary gradients',
    architecture='Messenger complete F384 -> encoder16 -> normalized VQ -> decoder384 + spatial correction from quantized latent -> teacher-mask blend with corrupted X -> frozen head',
    js=dict(weight=.01,temperature=.1,dictionary_detached=True,mean_detached=False,region='covered',warmup='none',branch_vq=False),
    augmentation=MASK_PLAN,retention='remove four new best/last R and initial codes after complete numeric evaluation/audit; preserve baseline',
    test_history='previously observed in prior comparisons; no test-driven checkpoint/threshold selection in this run')
core.PLAN=PLAN;core.arm_spec=arm_spec;core.RUNTIME_MICROBATCH=PLAN['microbatch']
write=BASE.write;read=core.read;sha=BASE.sha;status=BASE.status;lease=core.lease
GATE_PLAN=dict(phases=['schema','gpu_step','overfit','small_scale'],overfit_steps=32,fit_faces=64,val_faces=32,epochs=12,
               min_improvement=1e-5,quality='overfit position+objective improve; small scale val position improves; no automatic quality retry')


def source_hashes():
    paths={Path(__file__).resolve(),ROOT/'src/models/messenger_full_e.py',ROOT/'src/data/messenger_full_e_cache.py',
        BASE.CONFIG,BASE.BASE,BASE.ANN,core.DIRECTIONS/'audit.json',core.DIRECTIONS/'conditions.npz',
        BASE.BASE.parent.parent/'training_success.json'}
    paths.update(data.ANNOTATIONS.values())
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f:
            p=Path(f).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and p.suffix=='.py' and '.venv' not in p.parts:paths.add(p)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def load_model():
    config=yaml.safe_load(BASE.CONFIG.read_text());model,_=BASE.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
    BASE.load_landmark_checkpoint(model,BASE.BASE,prefer_ema=True,strict=True)
    model.requires_grad_(False).eval().cuda();return model,config

def setup_runtime():
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True

@torch.no_grad()
def prepare(model,config):
    PREP.mkdir(parents=True,exist_ok=True);manifest=data.build_manifest()
    path=PREP/'manifest.json'
    if path.exists():assert read(path)==manifest
    else:write(path,manifest)
    cache=data.make_cache(PREP,model,config['data'],manifest)
    normalization=PREP/'normalization.json';codespath=PREP/'initial_codes.npy'
    if (PREP/'initialization.json').exists():
        init=read(PREP/'initialization.json');assert sha(codespath)==init['codes_sha256'] and sha(normalization)==init['normalization_sha256']
        n=read(normalization);basis=torch.tensor(n['basis']);mean=torch.tensor(n['mean']);codes=torch.from_numpy(np.load(codespath))
    else:
        da=read(core.DIRECTIONS/'audit.json');assert sha(core.DIRECTIONS/'conditions.npz')==da['sha256']
        assert sha(BASE.BASE)==da['hashes'][str(BASE.BASE.relative_to(ROOT))]
        assert sha(BASE.ANN)==da['hashes'][str(BASE.ANN.relative_to(ROOT))]
        with np.load(core.DIRECTIONS/'conditions.npz') as v:
            fit_ids=set(map(int,v['fit_face_id']));assert fit_ids<=set(a['id'] for a in json.loads(BASE.ANN.read_text())['annotations'])
            valvid={a['video'] for a in manifest['records'] if a['split']!='train'}
            assert set(v['fit_video']).isdisjoint(valvid)
            basis=core.basis_from_fit(v['fit_grams'][0],16)
        parents=np.unique(cache['parent'][cache['syn_fit']]);total=torch.zeros(384,dtype=torch.float64)
        for start in range(0,len(parents),32):total+=cache['clean'][parents[start:start+32],2].double().sum((0,2,3))
        mean=(total/(len(parents)*256)).float();bg=basis.cuda();mg=mean.cuda();latent=[]
        for start in range(0,len(parents),32):
            f=cache['clean'][parents[start:start+32],2].cuda().float()
            latent.append(torch.einsum('cr,bchw->bhwr',bg,f-mg[None,:,None,None]).reshape(-1,16))
        latent=torch.cat(latent);status(PREP,'initializing_dictionary',faces=len(parents),patches=len(latent),codes=8192)
        centers,counts,inertia,iterations=core.kmeans_codes(latent,8192);codes=centers.cpu()
        np.save(codespath,codes.numpy());write(normalization,dict(mean=mean.tolist(),basis=basis.tolist(),direction_train_faces=len(fit_ids)))
        write(PREP/'initialization.json',dict(codes_sha256=sha(codespath),normalization_sha256=sha(normalization),fit_faces=len(parents),fit_patches=len(latent),codes=8192,used_codes=int((counts>0).sum()),inertia=inertia,iterations=iterations))
        del latent,bg,mg;gc.collect();torch.cuda.empty_cache()
    return cache,basis,mean,codes

def new_net(arm,basis,mean,codes):return make_recovery(arm,basis,mean,codes).cuda()

def authenticate():
    c=read(GATE/'completion.json');p=read(GATE/'plan.json')
    assert c['passed'] and p['plan']==PLAN and p['gate_plan']==GATE_PLAN
    core.verify_hashes(p['hashes'])
    for name,h in c['artifacts'].items():assert sha(GATE/name)==h,name
    assert sha(PREP/'manifest.json')==p['manifest_sha256']
    assert sha(PREP/'initialization.json')==p['initialization_sha256']
    for arm in ARMS:
        for phase in GATE_PLAN['phases']:
            x=read(GATE/arm/phase/'success.json');assert x['passed'] and x['context']==c['context']
            assert sha(GATE/arm/phase/'report.json')==x['report_sha256']
    return dict(passed=True,arms=list(ARMS),success_markers=8,completion_sha256=sha(GATE/'completion.json'))

def finish(arm,phase,context,result):
    folder=GATE/arm/phase;write(folder/'report.json',result)
    if not result['passed']:raise RuntimeError(f'Quality gate failed: {arm}/{phase}; no automatic retry/main')
    write(folder/'success.json',dict(passed=True,context=context,report_sha256=sha(folder/'report.json')))
    status(GATE,'gate_passed',arm=arm,stage=phase)

def selected_ids(cache,role,n):
    # Stable spread over the official split; same chosen faces in both arms.
    ids=cache['new_manifest']['roles'][role]
    faces=sorted(ids,key=lambda i:hashlib.sha256(f'gate42:{i}'.encode()).hexdigest())[:n]
    selected=[i for i,a in enumerate(cache['new_manifest']['records']) if a['uid'] in set(faces)]
    return cache['syn_'+role][np.isin(cache['parent'][cache['syn_'+role]],selected)],faces

def gate_fit(net,arm,model,cache,basis,trainids,valids,faces,overfit):
    optimizer=torch.optim.AdamW(net.parameters(),lr=PLAN['lr'],weight_decay=PLAN['weight_decay'])
    view=dict(cache,syn_dev=valids);history=[core.validate(net,model,view,basis,arm)];rng=np.random.default_rng(42)
    for epoch in range(1,(GATE_PLAN['overfit_steps'] if overfit else GATE_PLAN['epochs'])+1):
        if not overfit:data.refresh(cache,model,GATE/'small_scale_data','fit',epoch,faces)
        core.train_epoch(net,model,cache,basis,arm,optimizer,rng.permutation(trainids))
        row=core.validate(net,model,view,basis,arm);assert all(np.isfinite(v) for v in row.values());history.append(row)
        if epoch%4==0:status(GATE,'gating',arm=arm,stage='overfit' if overfit else 'small_scale',epoch=epoch,position=row['position'])
    first=history[0];passed=any(h['position']<first['position']-1e-5 and (not overfit or h['objective']<first['objective']-1e-5) for h in history[1:])
    return dict(passed=passed,history=history,fit_pair_ids=trainids.tolist(),val_pair_ids=valids.tolist())

@torch.no_grad()
def evaluate(net,model,cache,ids,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);before=BASE.tensor_digest(net);net.eval()
    rows=[];predictions=[];teacher_g=[];feature=[];codecounts=np.zeros(8192,dtype=np.int64)
    for start in range(0,len(ids),8):
        j=ids[start:start+8];maps,z,c,e=BASE.syn_batch(cache,j);xy=BASE.gpu(cache['xy'][cache['parent'][j]])
        restored,detail=core.generate(net,maps);fused=core.teacher_blend(maps[:,2],restored,c)
        values=[];coords=[]
        for f in (maps[:,2],fused):
            pred=BASE.run_head(model.head,f);values.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1));coords.append(pred['decoded_coordinates'])
        rows.append(torch.stack(values,1).cpu().numpy());predictions.append(torch.stack(coords,1).cpu().numpy());teacher_g.append(c.cpu().numpy())
        codecounts+=np.bincount(detail['indices'][c].cpu().numpy(),minlength=8192)
        feature.append(torch.stack([torch.stack((((f-z).square().mean(1)*c).sum((1,2)),((1-F.cosine_similarity(f,z,dim=1))*c).sum((1,2)),c.sum((1,2))),-1) for f in (maps[:,2],restored)],1).cpu().numpy())
    arr=np.concatenate(rows);point=cache['point'][ids].numpy();severity=cache['severity'][ids];videos=cache['videos'][cache['parent'][ids]]
    np.savez_compressed(folder/'evaluation.npz',rows=arr,coordinates=np.concatenate(predictions),point=point,severity=severity,videos=videos,
        face_uid=cache['uids'][cache['parent'][ids]],teacher_g=np.concatenate(teacher_g),feature_sums=np.concatenate(feature),
        code_counts=codecounts,gt_xy=cache['xy'][cache['parent'][ids]].numpy())
    summaries=[]
    for b in [-1,0,1,2,3,4]:
        pairmask=np.ones(len(ids),dtype=bool) if b<0 else severity==b;row=dict(bin=b,pairs=int(pairmask.sum()),methods={})
        for k,name in enumerate(['raw','restored']):
            row['methods'][name]={}
            for region,m in [('covered',point),('uncovered',~point)]:
                mask=m&pairmask[:,None];err=arr[:,k,:,0][mask];pres=arr[:,k,:,1][mask]
                row['methods'][name][region]=dict(points=len(err),mean_percent=float(err.mean()*100) if len(err) else None,
                    within5_count=int((err<.05).sum()),joint5_count=int(((err<.05)&(pres>=.7)).sum()),over20_count=int((err>.2).sum()))
        summaries.append(row)
    prob=codecounts[codecounts>0]/codecounts.sum()
    result=dict(conditions=summaries,code_usage=dict(used_codes=len(prob),perplexity=float(np.exp(-(prob*np.log(prob)).sum()))),
        model_unchanged=BASE.tensor_digest(net)==before,no_clean_head_input=True,teacher_g_matches_mask=bool(np.array_equal(np.concatenate(teacher_g),cache['cov'][ids].numpy())))
    assert result['model_unchanged'] and result['teacher_g_matches_mask'];write(folder/'evaluation.json',result)
    return result

def gates(wait):
    with lease(wait):
        if (GATE/'completion.json').exists():return authenticate()
        if (GATE/'plan.json').exists():raise ValueError('Incomplete gate attempt exists; inspect before retry')
        core.gpu_idle(wait);setup_runtime();GATE.mkdir(parents=True,exist_ok=True)
        hashes=source_hashes();write(GATE/'preparation_started.json',dict(hashes=hashes,plan=PLAN))
        try:
            model,config=load_model();frozen=BASE.tensor_digest(model);cache,basis,mean,codes=prepare(model,config)
            trainids,faces=selected_ids(cache,'fit',64);valids,valfaces=selected_ids(cache,'dev',32)
            data.refresh(cache,model,GATE/'gate_data','dev',0,valfaces)
            # Full epoch0 distribution audit before releasing main learning.
            fullaudit=data.refresh(cache,model,PREP,'fit',0)
            write(GATE/'full_train_distribution.json',fullaudit)
            plan=dict(plan=PLAN,gate_plan=GATE_PLAN,hashes=hashes,manifest_sha256=sha(PREP/'manifest.json'),initialization_sha256=sha(PREP/'initialization.json'))
            write(GATE/'plan.json',plan);context=core.signature(plan)
            initial={}
            for arm in ARMS:
                data.refresh(cache,model,GATE/'gate_data','fit',0,faces)
                net=new_net(arm,basis,mean,codes).eval();initial[arm]=BASE.tensor_digest(net)
                j=trainids[:16];maps,z,c,e=BASE.syn_batch(cache,j[:4]);restored,details=core.generate(net,maps,c)
                assert restored.shape==z.shape and torch.isfinite(restored).all()
                assert net.spatial is not None and net.codebook.update=='ema' and not net.codebook.embedding.requires_grad
                if arm==ARMS[0]:
                    from src.models.messenger_codebook_reconstruction import make_recovery as original_E
                    original=original_E('E_d16_k8192',basis,mean,codes).cuda().eval()
                    assert BASE.tensor_digest(original)==initial[arm]
                    with torch.no_grad():torch.testing.assert_close(original(maps,c)[0],restored.detach(),rtol=0,atol=0)
                    del original
                torch.testing.assert_close(core.teacher_blend(maps[:,2],restored,torch.zeros_like(c)),maps[:,2],rtol=0,atol=0)
                torch.testing.assert_close(core.teacher_blend(maps[:,2],restored,torch.ones_like(c)),restored,rtol=0,atol=0)
                finish(arm,'schema',context,dict(passed=True,initial_model_sha256=initial[arm],parameters=sum(p.numel() for p in net.parameters()),spec=arm_spec(arm)))
                xy=BASE.gpu(cache['xy'][cache['parent'][j[:4]]]);point=BASE.gpu(cache['point'][j[:4]]);valid=torch.ones_like(point)
                parts,_=core.loss_parts(model,restored,maps[:,2],z,c,e,xy,point,valid,basis.cuda(),arm,details)
                gradients=torch.autograd.grad(parts['position'],[net.messenger.branches[0].query,net.encoder.weight,net.spatial.output.weight],retain_graph=True)
                assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)
                js_norms=[]
                if arm_spec(arm)['js_weight']:
                    jsgrads=torch.autograd.grad(details['js'],[net.encoder.weight]+[b.query for b in net.messenger.branches],retain_graph=True)
                    assert all(torch.isfinite(g).all() and g.norm()>0 for g in jsgrads);js_norms=[float(g.norm()) for g in jsgrads]
                del restored,details,parts,gradients
                opt=torch.optim.AdamW(net.parameters(),lr=PLAN['lr'],weight_decay=PLAN['weight_decay']);old=net.codebook.embedding.detach().clone()
                losses=core.train_epoch(net,model,cache,basis.cuda(),arm,opt,j);assert np.isfinite(losses).all()
                assert net.codebook.embedding.grad is None and int(net.codebook.ema_steps)==1 and not torch.equal(old,net.codebook.embedding)
                net.eval();technical=evaluate(net,model,cache,j,GATE/arm/'gpu_step'/'evaluation')
                assert BASE.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
                finish(arm,'gpu_step',context,dict(passed=True,ema_steps=1,dictionary_changed=True,js_gradient_norms=js_norms,
                    dictionary_requires_grad=False,fixed_parent_unchanged=True,teacher_g_fixed=True,technical_evaluation=technical,
                    microbatch=core.RUNTIME_MICROBATCH,max_cuda_allocated=torch.cuda.max_memory_allocated()))
                del net,opt,old;gc.collect();torch.cuda.empty_cache()
                net=new_net(arm,basis,mean,codes);finish(arm,'overfit',context,gate_fit(net,arm,model,cache,basis.cuda(),j,j,faces,True))
                del net;gc.collect();torch.cuda.empty_cache()
                net=new_net(arm,basis,mean,codes);finish(arm,'small_scale',context,gate_fit(net,arm,model,cache,basis.cuda(),trainids,valids,faces,False))
                del net;gc.collect();torch.cuda.empty_cache()
            assert BASE.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters());core.verify_hashes(hashes)
            write(GATE/'summary.json',dict(initial_model_hashes=initial,parent_unchanged=True,success_markers=8))
            artifacts={str(p.relative_to(GATE)):sha(p) for p in GATE.rglob('*.json') if p.name!='status.json'}
            write(GATE/'completion.json',dict(passed=True,context=context,artifacts=artifacts,gate_weights_saved=False))
            status(GATE,'all_gates_passed',success_markers=8);return authenticate()
        except Exception as ex:status(GATE,'failed',error=repr(ex));raise


def interval(delta,videos):
    unique=np.unique(videos);sums=np.array([delta[videos==v].sum() for v in unique]);counts=np.array([(videos==v).sum() for v in unique]);rng=np.random.default_rng(42)
    ix=rng.integers(0,len(unique),(4000,len(unique)));samples=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(difference_pp=float(delta.mean()),video_bootstrap_95_pp=np.quantile(samples,[.025,.975]).tolist(),videos=len(unique))

def complete_and_cleanup():
    results={a:read(OUT/a/'test/evaluation.json') for a in ARMS};arrays={a:np.load(OUT/a/'test/evaluation.npz') for a in ARMS}
    first,second=[arrays[a] for a in ARMS]
    for k in ['point','severity','videos','face_uid','teacher_g','gt_xy']:np.testing.assert_array_equal(first[k],second[k])
    np.testing.assert_allclose(first['rows'][:,0],second['rows'][:,0],rtol=0,atol=0)
    contrasts={}
    for b in [-1,0,1,2,3,4]:
        mask=first['point'].copy()
        if b>=0:mask&=(first['severity']==b)[:,None]
        delta=(second['rows'][:,1,:,0].astype(float)-first['rows'][:,1,:,0])[mask]*100
        videos=np.broadcast_to(first['videos'][:,None],mask.shape)[mask];contrasts[str(b)]=interval(delta,videos)
    write(OUT/'comparison.json',dict(results=results,second_minus_first=contrasts,scope=PLAN['test_history']))
    for a,v in arrays.items():v.close()
    plan=read(OUT/'plan.json');core.verify_hashes(plan['hashes'])
    assert all(read(OUT/a/'R_convergence.json')['converged'] for a in ARMS)
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.npz'] and p.name not in ['status.json','numeric_completion.json','checkpoint_deletion.json']}
    write(OUT/'numeric_completion.json',dict(all_numeric_evaluations_complete=True,artifacts=artifacts,natural_evaluation=False))
    paths=[OUT/a/n for a in ARMS for n in ['best_R.pt','last_R.pt']]+[PREP/'initial_codes.npy'];audit=[]
    for p in paths:
        assert not p.is_symlink() and (p.resolve().is_relative_to(OUT.resolve()) or p.resolve()==(PREP/'initial_codes.npy').resolve())
        if p.exists():audit.append(dict(absolute_path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size))
    write(OUT/'checkpoint_deletion.json',dict(phase='prepared',weights=audit))
    for row in audit:Path(row['absolute_path']).unlink()
    write(OUT/'checkpoint_deletion.json',dict(phase='completed',weights=audit))
    status(OUT,'completed',all_numeric_evaluations_complete=True,new_weights_remaining=0)

def train(wait,resume=False):
    with lease(wait):
        if (OUT/'numeric_completion.json').exists():return
        if (OUT/'plan.json').exists() and not resume:raise ValueError('Existing training run: resume requires identity audit')
        core.gpu_idle(wait);setup_runtime();auth=authenticate();OUT.mkdir(parents=True,exist_ok=True)
        hashes=source_hashes()
        if resume:
            plan=read(OUT/'plan.json');assert plan['plan']==PLAN;core.verify_hashes(plan['hashes'])
        else:write(OUT/'plan.json',dict(plan=PLAN,hashes=hashes,gate=auth))
        write(OUT/'gate_authentication.json',auth)
        model,config=load_model();frozen=BASE.tensor_digest(model);cache,basis,mean,codes=prepare(model,config)
        data.refresh(cache,model,PREP,'dev',0)
        wb=None
        try:
            import wandb
            wb=wandb.init(project='face_detection',group='messenger_full_e_20261122',name='Full train E16K8192 vs scale8+JS',
                dir=str(OUT),config=PLAN,id='full-e-20261122',resume='allow',settings=wandb.Settings(init_timeout=30))
        except Exception as ex:write(OUT/'wandb_warning.json',dict(error=repr(ex)))
        try:
            for arm in ARMS:
                folder=OUT/arm;folder.mkdir(exist_ok=True);net=new_net(arm,basis,mean,codes)
                identity=core.signature(dict(plan=PLAN,hashes=hashes,arm=arm,initial=read(PREP/'initialization.json')))
                init=BASE.tensor_digest(net);write(folder/'initialization.json',dict(model_sha256=init,code_sha256=read(PREP/'initialization.json')['codes_sha256']))
                opt,controller,rng,history,best,start,done=core.initial(net,folder,'R',identity)
                for epoch in range(start,PLAN['R_cap']+1) if not done else ():
                    tick=time.monotonic();losses=[]
                    if epoch:
                        data.refresh(cache,model,PREP,'fit',epoch)
                        losses=core.train_epoch(net,model,cache,basis.cuda(),arm,opt,rng.permutation(cache['syn_fit']))
                    dev=core.validate(net,model,cache,basis.cuda(),arm)
                    if dev['position']<best:
                        best=dev['position'];core.save(folder/'best_R.pt',dict(state=net.state_dict(),epoch=epoch,dev=dev,identity=identity))
                    done=controller.step(epoch,[dev['position'],dev['objective']]);core.set_lr(opt,controller.lr)
                    row=dict(epoch=epoch,dev=dev,train=float(np.sum(losses)),lr=controller.lr,seconds=time.monotonic()-tick,converged=done)
                    history.append(row);core.legacy.persist(folder,'R',net,opt,controller,rng,history,best,identity,done);status(OUT,'training',arm=arm,**row)
                    if wb:wb.log({f'{arm}/epoch':epoch,f'{arm}/lr':controller.lr,**{f'{arm}/val_{k}':v for k,v in dev.items()}})
                    if done:break
                if not done:raise RuntimeError('Epoch cap reached before plateau')
                ck=torch.load(folder/'best_R.pt',map_location='cpu',weights_only=True)
                write(folder/'R_convergence.json',dict(converged=True,stop_epoch=history[-1]['epoch'],selected_epoch=ck['epoch']))
                del net,opt,ck;gc.collect();torch.cuda.empty_cache()
            assert BASE.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
            write(OUT/'training_completion.json',dict(all_converged=True,arms=list(ARMS),fixed_parent_unchanged=True,natural_evaluation=False))
            data.refresh(cache,model,PREP,'eval',0)
            for arm in ARMS:
                net=new_net(arm,basis,mean,codes);ck=torch.load(OUT/arm/'best_R.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['state']);net.eval().requires_grad_(False)
                status(OUT,'evaluating',arm=arm,split='test');result=evaluate(net,model,cache,cache['syn_eval'],OUT/arm/'test')
                if wb:
                    for b in result['conditions']:
                        for method in b['methods']:
                            wb.log({f'{arm}/test_bin{b["bin"]}_{method}_covered_nme':b['methods'][method]['covered']['mean_percent']})
                del net,ck;gc.collect();torch.cuda.empty_cache()
            assert BASE.tensor_digest(model)==frozen
            complete_and_cleanup()
        except Exception as ex:status(OUT,'failed',error=repr(ex));raise
        finally:
            if wb:wb.finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','gate','train']);p.add_argument('--wait',action='store_true');p.add_argument('--then-train',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.command=='plan':print(json.dumps(PLAN,ensure_ascii=False,indent=2))
    elif args.command=='gate':
        gates(args.wait)
        if args.then_train:os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'train','--wait'])
    else:train(args.wait,args.resume)
if __name__=='__main__':main()
