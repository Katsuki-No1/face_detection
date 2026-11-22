"""Check only previously unattempted arms after a recorded quality failure.
Never rerun a failed arm, change a threshold, or start convergence training.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gate_messenger_codebook_factorial import *
from scripts import gate_messenger_codebook_factorial as original


def check_arm(arm, model, cache, basis, mean, codes, context, frozen, initial_hashes, summaries):
    net=make_net(arm,basis,mean,codes).eval()
    initial_hashes[arm]=run.base.tensor_digest(net)
    ids=cache['syn_fit'][:16]
    maps,z,c,e=run.base.syn_batch(cache,ids[:4])
    r,details=run.generate(net,maps)
    assert r.shape==z.shape and torch.isfinite(r).all()
    finish(arm,'schema',context,dict(passed=True,shape=list(r.shape),initial_model_sha256=initial_hashes[arm],
        parameter_count=sum(p.numel() for p in net.parameters()),fixed_parent_sha256=frozen))
    xy=run.base.gpu(cache['xy'][cache['parent'][ids[:4]]])
    point=run.base.gpu(cache['point'][ids[:4]])
    valid=run.base.gpu(cache['state'][cache['parent'][ids[:4]]])>0
    parts,_=run.loss_parts(model,r,maps[:,2],z,c,e,xy,point,valid,basis,arm,details)
    code_gradient=None
    if getattr(net,'codebook',None) is not None and net.codebook.embedding.requires_grad:
        gradient=torch.autograd.grad(parts['position'],net.codebook.embedding)[0]
        code_gradient=float(gradient.norm());assert code_gradient>0 and torch.isfinite(gradient).all()
    del r,details,parts
    opt=optimizer(net)
    losses=run.train_epoch(net,model,cache,basis,arm,opt,ids)
    assert all(np.isfinite(loss) for loss in losses) and run.base.tensor_digest(net)!=initial_hashes[arm]
    if getattr(net,'codebook',None) is not None and net.codebook.update == 'ema':
        assert net.codebook.embedding.grad is None and int(net.codebook.ema_steps) == 1
        spec=arm_spec(arm)
        key=f'd{spec["dimension"]}_k{spec["count"]}'
        assert not torch.equal(net.codebook.embedding.cpu(),codes[key])
    natural=natural_ids(cache,'fit',32)
    gpu_g=fit_g(net,model,cache,natural,natural,1)
    finish(arm,'gpu_step',context,dict(passed=gpu_g['weights_changed'],position_code_gradient_norm=code_gradient,
        loss=sum(losses),g=gpu_g,fixed_parent_unchanged=run.base.tensor_digest(model)==frozen))
    del net,opt;gc.collect();torch.cuda.empty_cache()
    net=make_net(arm,basis,mean,codes)
    R=fit_R(net,arm,model,cache,basis,ids,ids,GATE_PLAN['overfit_steps'],True)
    natural=natural_ids(cache,'fit',GATE_PLAN['overfit_natural_faces'])
    g=fit_g(net,model,cache,natural,natural,GATE_PLAN['overfit_steps'],True)
    finish(arm,'overfit',context,dict(passed=R['passed'] and g['passed'],R=R,g=g))
    del net;gc.collect();torch.cuda.empty_cache()
    net=make_net(arm,basis,mean,codes)
    fit=subset(cache,'fit',GATE_PLAN['small_fit_faces']);dev=subset(cache,'dev',GATE_PLAN['small_dev_faces'])
    R=fit_R(net,arm,model,cache,basis,fit,dev,GATE_PLAN['small_epochs'],False)
    g=fit_g(net,model,cache,natural_ids(cache,'fit',GATE_PLAN['small_natural_fit']),
            natural_ids(cache,'dev',GATE_PLAN['small_natural_dev']),GATE_PLAN['small_epochs'])
    passed=R['passed'] and g['passed']
    finish(arm,'small_scale',context,dict(passed=passed,R=R,g=g))
    summaries[arm]=dict(R_initial=R['history'][0]['position'],R_best=min(x['position'] for x in R['history'][1:]),
        g_initial=g['history'][0]['position'],g_best=min(x['position'] for x in g['history'][1:]))
    del net;gc.collect();torch.cuda.empty_cache()
    assert run.base.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())


def main():
    with run.lease(False):
        run.gpu_idle(False)
        plan=run.read(OUT/'plan.json')
        run.verify_hashes(plan['hashes'])
        if plan['training_plan'] != run.PLAN or plan['gate_plan'] != GATE_PLAN:
            raise ValueError('Original gate contract changed')
        if plan['runtime'] != dict(python=sys.version,torch=torch.__version__,numpy=np.__version__):
            raise ValueError('Runtime changed')
        if (OUT/'completion.json').exists() or (OUT/'continuation_plan.json').exists():
            raise ValueError('Already complete or continuation already attempted; inspect manually')
        attempted=[];remaining=[]
        for arm in ARMS:
            if (OUT/arm/'small_scale/report.json').exists():
                attempted.append(arm)
            elif (OUT/arm).exists():
                raise ValueError(f'Incomplete technical arm needs inspection: {arm}')
            else:
                remaining.append(arm)
        own_hash=run.base.sha(Path(__file__))
        run.base.write(OUT/'continuation_plan.json',dict(script=str(Path(__file__).resolve()),sha256=own_hash,
            original_plan_sha256=run.base.sha(OUT/'plan.json'),skipped_attempted=attempted,remaining=remaining,
            retry_failed=False,main_training=False,pid=os.getpid()))
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        config=yaml.safe_load(run.base.CONFIG.read_text())
        model,_=run.base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
        run.base.load_landmark_checkpoint(model,run.base.BASE,prefer_ema=True,strict=True)
        model.requires_grad_(False).eval().cuda();frozen=run.base.tensor_digest(model)
        old_manifest=run.read(OUT/'manifest.json')
        cache=run.base.make_cache(OUT,model,config['data'])
        manifest=run.read(OUT/'manifest.json')
        run.manifest_contract(manifest,old_manifest)
        basis=run.load_basis(manifest).cuda()
        parents=np.unique(cache['parent'][cache['syn_fit']])
        mean=cache['clean'][parents,2].float().mean((0,2,3)).cuda()
        run.legacy.teacher_cache(cache,manifest)
        codes={};code_hashes={}
        for dimension in DIMENSIONS:
            for count in COUNTS:
                key=f'd{dimension}_k{count}'
                centers,_=run.initial_codes(cache,basis[:,:dimension],mean,count)
                codes[key]=torch.from_numpy(centers);code_hashes[key]=array_sha(centers)
        if code_hashes != run.read(OUT/'initialization.json')['code_sha256']:
            raise ValueError('Continuation dictionary differs')
        context=run.signature(dict(plan=plan,manifest=manifest,code_hashes=code_hashes))
        for marker in OUT.glob('*/*/success.json'):
            proof=run.read(marker)
            if proof['context']!=context or run.base.sha(marker.parent/'report.json')!=proof['report_sha256']:
                raise ValueError('Prior evidence invalid')
        initial_hashes={};summaries={}
        for arm in remaining:
            try:
                check_arm(arm,model,cache,basis,mean,codes,context,frozen,initial_hashes,summaries)
            except RuntimeError as error:
                if not str(error).startswith('Quality gate failed:'):
                    raise
                print(str(error),flush=True)
                gc.collect();torch.cuda.empty_cache()
        reports={};failures=[]
        for arm in ARMS:
            phases={}
            for phase in PHASES:
                p=OUT/arm/phase/'report.json'
                if p.exists():
                    phases[phase]=run.read(p)['passed']
            reports[arm]=phases
            if set(phases)!=set(PHASES) or not all(phases.values()): failures.append(arm)
        if run.base.tensor_digest(model)!=frozen or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Fixed parent changed')
        run.verify_hashes(plan['hashes'])
        assert run.base.sha(Path(__file__))==own_hash
        artifacts={str(p.relative_to(OUT)):run.base.sha(p) for p in OUT.rglob('*.json') if p.name!='status.json'}
        run.base.write(OUT/'gate_review.json',dict(all_arms_attempted=True,all_gates_passed=not failures,
            failed_arms=failures,phases=reports,artifacts=artifacts,source_unchanged=True,parent_unchanged=True,
            retry_failed=False,main_training_started=False,new_weights_created=False))
        run.base.status(OUT,'gate_review_complete',failed_arms=failures,main_training_started=False,
            passed_markers=len(list(OUT.glob('*/*/success.json'))))


if __name__=='__main__':
    main()
