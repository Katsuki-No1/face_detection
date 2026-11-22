"""Authenticated four-stage GPU gates for the factorial codebook comparison arms."""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_messenger_codebook_reconstruction as run
from src.models.messenger_codebook_reconstruction import ARMS, DIMENSIONS, COUNTS, arm_spec, make_recovery

OUT=ROOT/'outputs/training_gates/messenger_codebook_reconstruction_20261121'
PHASES=('schema','gpu_step','overfit','small_scale')
GATE_PLAN=dict(version=4,overfit_pairs=16,overfit_steps=32,
    small_fit_faces=64,small_dev_faces=32,small_epochs=12,R_min_improvement=1e-5,
    quality='R only: overfit position and objective improve; small-scale dev position improves over epoch0',
    teacher_g='fixed generated mask for R train/dev; fixed natural weak teacher for natural evaluation; no learned g',
    initialization='fresh identical main-run initialization for every gate stage; no gate weight transfer',
    training='same FP32/TF32 R loss, optimizer, microbatch4/effective16; no eval use',
    dictionary='same full synthetic-fit256 k-means initialization as main comparison',
    augmentation='five strata; fixed overfit pairs; fresh severity-balanced small-fit RGB and features every epoch',
    retention='gate models in memory only; numeric reports and success markers saved')


def array_sha(array):
    buffer=io.BytesIO();np.save(buffer,array)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def authenticate(manifest=None,code_hashes=None):
    completion=run.read(OUT/'completion.json')
    plan=run.read(OUT/'plan.json')
    if not completion.get('passed') or plan['gate_plan']!=GATE_PLAN or plan['training_plan']!=run.PLAN:
        raise ValueError('Gates are incomplete or the plan changed')
    run.verify_hashes(plan['hashes'])
    if plan['runtime']!=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__):
        raise ValueError('Gate runtime changed')
    for name,digest in completion['artifacts'].items():
        if run.base.sha(OUT/name)!=digest:
            raise ValueError(f'Gate evidence changed: {name}')
    for arm in ARMS:
        for phase in PHASES:
            marker=run.read(OUT/arm/phase/'success.json')
            if not marker.get('passed') or marker['context']!=completion['context']:
                raise ValueError(f'Invalid success marker: {arm}/{phase}')
            if run.base.sha(OUT/arm/phase/'report.json')!=marker['report_sha256']:
                raise ValueError('Gate report/marker mismatch')
    if manifest is not None:
        run.manifest_contract(manifest,run.read(OUT/'manifest.json'))
    if code_hashes is not None and code_hashes!=run.read(OUT/'initialization.json')['code_sha256']:
        raise ValueError('Main-run initial dictionaries differ from authenticated gates')
    return dict(passed=True,context=completion['context'],completion_sha256=run.base.sha(OUT/'completion.json'),
                phases=list(PHASES),arms=list(ARMS),evidence=str(OUT))


def finish(arm,phase,context,report):
    folder=OUT/arm/phase
    run.base.write(folder/'report.json',report)
    if not report['passed']:
        raise RuntimeError(f'Quality gate failed: {arm}/{phase}; no automatic retry or main training')
    run.base.write(folder/'success.json',dict(passed=True,context=context,report_sha256=run.base.sha(folder/'report.json')))
    run.base.status(OUT,'gate_passed',arm=arm,stage=phase)


def improve(history,kind,overfit=False):
    margin=GATE_PLAN[f'{kind}_min_improvement']
    initial=history[0]
    return any(row['position']<initial['position']-margin and
               (not overfit or row['objective']<initial['objective']-margin) for row in history[1:])


def subset(cache,role,n):
    parents=np.unique(cache['parent'][cache['syn_'+role]])[:n]
    return cache['syn_'+role][np.isin(cache['parent'][cache['syn_'+role]],parents)]


def make_net(arm,basis,mean,codes):
    spec=arm_spec(arm)
    key=f'd{spec["dimension"]}_k{spec["count"]}'
    return make_recovery(arm,basis[:,:spec['dimension']].cpu(),mean.cpu(),codes.get(key)).cuda()


def optimizer(net,kind='R'):
    return torch.optim.AdamW(net.parameters(),lr=run.PLAN['lr'],
                             weight_decay=run.PLAN['weight_decay'])


def checkpoint_state(net):
    return {key:value.detach().cpu().clone() for key,value in net.state_dict().items()}


def fit_R(net,arm,model,cache,basis,ids,dev,epochs,overfit):
    opt=optimizer(net)
    view=dict(cache,syn_dev=dev)
    history=[run.validate(net,model,view,basis,arm)]
    best=history[0]['position'];state=checkpoint_state(net)
    rng=np.random.default_rng(42)
    for epoch in range(1,epochs+1):
        if not overfit:
            parents=np.unique(cache['parent'][ids])
            faces=[cache['new_manifest']['faces'][i] for i in parents]
            run.dynamic.refresh(cache,model,OUT/'small_scale_data','fit',epoch,faces)
        run.train_epoch(net,model,cache,basis,arm,opt,rng.permutation(ids))
        row=run.validate(net,model,view,basis,arm)
        if not all(np.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite R validation')
        history.append(row)
        if row['position']<best:
            best=row['position'];state=checkpoint_state(net)
        if epoch%4==0:
            run.base.status(OUT,'gating',arm=arm,stage='overfit_R' if overfit else 'small_scale_R',epoch=epoch,position=row['position'])
    net.load_state_dict(state)
    return dict(history=history,passed=improve(history,'R',overfit),fit_pair_ids=ids.tolist(),dev_pair_ids=dev.tolist())


def gates(wait=False):
    with run.lease(wait):
        if (OUT/'completion.json').exists():
            return authenticate()
        if (OUT/'plan.json').exists():
            raise ValueError('An incomplete gate attempt exists; inspect its evidence before any retry')
        run.gpu_idle(wait)
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        OUT.mkdir(parents=True,exist_ok=True)
        plan=dict(gate_plan=GATE_PLAN,training_plan=run.PLAN,hashes=run.source_hashes(),
                  runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))
        run.base.write(OUT/'plan.json',plan)
        try:
            config=yaml.safe_load(run.base.CONFIG.read_text())
            model,_=run.base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
            run.base.load_landmark_checkpoint(model,run.base.BASE,prefer_ema=True,strict=True)
            model.requires_grad_(False).eval().cuda();frozen=run.base.tensor_digest(model)
            cache=run.dynamic.make_cache(OUT,model,config['data'])
            manifest=run.read(OUT/'manifest.json')
            run.manifest_contract(manifest,run.read(run.BASELINE/'manifest.json'))
            basis=run.load_basis(manifest).cuda()
            parents=np.unique(cache['parent'][cache['syn_fit']])
            mean=cache['clean'][parents,2].float().mean((0,2,3)).cuda()
            teacher=run.legacy.teacher_cache(cache,manifest)
            run.base.write(OUT/'teacher_audit.json',teacher)
            codes={};code_audits={};code_hashes={}
            for dimension in DIMENSIONS:
                for count in COUNTS:
                    key=f"d{dimension}_k{count}"
                    centers,audit=run.initial_codes(cache,basis[:,:dimension],mean,count)
                    codes[key]=torch.from_numpy(centers)
                    code_audits[key]=audit;code_hashes[key]=array_sha(centers)
            run.base.write(OUT/'initialization.json',dict(code_sha256=code_hashes,audits=code_audits))
            context=run.signature(dict(plan=plan,manifest=manifest,code_hashes=code_hashes))
            initial_hashes={};summaries={}
            for arm in ARMS:
                run.dynamic.refresh(cache,model,OUT,'fit',0)
                net=make_net(arm,basis,mean,codes).eval()
                initial_hashes[arm]=run.base.tensor_digest(net)
                ids=cache['syn_fit'][:16]
                maps,z,c,e=run.base.syn_batch(cache,ids[:4])
                r,details=run.generate(net,maps,c)
                torch.testing.assert_close(run.teacher_blend(maps[:,2],r,torch.zeros_like(c)),maps[:,2],rtol=0,atol=0)
                torch.testing.assert_close(run.teacher_blend(maps[:,2],r,torch.ones_like(c)),r,rtol=0,atol=0)
                assert r.shape==z.shape and torch.isfinite(r).all()
                finish(arm,'schema',context,dict(passed=True,shape=list(r.shape),initial_model_sha256=initial_hashes[arm],
                    parameter_count=sum(p.numel() for p in net.parameters()),fixed_parent_sha256=frozen))
                xy=run.base.gpu(cache['xy'][cache['parent'][ids[:4]]])
                point=run.base.gpu(cache['point'][ids[:4]])
                valid=run.base.gpu(cache['state'][cache['parent'][ids[:4]]])>0
                parts,_=run.loss_parts(model,r,maps[:,2],z,c,e,xy,point,valid,basis,arm,details)
                messenger_gradient=torch.autograd.grad(parts['position'],net.messenger.branches[0].query,retain_graph=True)[0]
                assert torch.isfinite(messenger_gradient).all() and messenger_gradient.norm()>0
                messenger_gradient_norm=float(messenger_gradient.norm())
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
                    assert not torch.equal(net.codebook.embedding.cpu(),torch.nn.functional.normalize(codes[key],dim=1))
                net.eval()
                before_eval=run.base.tensor_digest(net)
                with torch.no_grad():
                    natural=cache['natural_fit'][:4]
                    natural_maps=run.base.gpu(cache['clean'][natural])
                    natural_r,_=run.generate(net,natural_maps)
                    fixed_g=run.base.gpu(cache['teacher'][natural])
                    fused=run.teacher_blend(natural_maps[:,2],natural_r,fixed_g)
                    prediction=run.base.run_head(model.head,fused)
                    assert torch.isfinite(prediction['decoded_coordinates']).all()
                assert run.base.tensor_digest(net)==before_eval
                # Exercise the real final evaluator before long training, using fit faces only.
                technical_folder=OUT/arm/'gpu_step'/'teacher_evaluation'
                technical_folder.mkdir(parents=True,exist_ok=True)
                technical_cache=dict(cache,natural_eval=cache['natural_fit'][:32],syn_eval=cache['syn_fit'][:16])
                technical_result=run.evaluate(technical_folder,net,model,technical_cache)
                run.base.write(technical_folder/'metrics.json',technical_result)
                for domain in ('natural','artificial'):
                    with np.load(technical_folder/f'{domain}_seed42.npz') as saved:
                        target=cache['teacher'][technical_cache['natural_eval']].numpy() if domain=='natural' else cache['cov'][technical_cache['syn_eval']].numpy()
                        np.testing.assert_array_equal(saved['teacher_g'],target)
                assert run.base.tensor_digest(net)==before_eval
                finish(arm,'gpu_step',context,dict(passed=True,position_code_gradient_norm=code_gradient,
                    loss=sum(losses),messenger_position_gradient_norm=messenger_gradient_norm,teacher_g_fixed=True,no_g_model=True,
                    fixed_parent_unchanged=run.base.tensor_digest(model)==frozen))
                del net,opt;gc.collect();torch.cuda.empty_cache()
                net=make_net(arm,basis,mean,codes)
                R=fit_R(net,arm,model,cache,basis,ids,ids,GATE_PLAN['overfit_steps'],True)
                finish(arm,'overfit',context,dict(passed=R['passed'],R=R,teacher_g_fixed=True))
                del net;gc.collect();torch.cuda.empty_cache()
                net=make_net(arm,basis,mean,codes)
                fit=subset(cache,'fit',GATE_PLAN['small_fit_faces']);dev=subset(cache,'dev',GATE_PLAN['small_dev_faces'])
                R=fit_R(net,arm,model,cache,basis,fit,dev,GATE_PLAN['small_epochs'],False)
                finish(arm,'small_scale',context,dict(passed=R['passed'],R=R,teacher_g_fixed=True))
                summaries[arm]=dict(R_initial=R['history'][0]['position'],R_best=min(x['position'] for x in R['history'][1:]))
                del net;gc.collect();torch.cuda.empty_cache()
                assert run.base.tensor_digest(model)==frozen and all(p.grad is None for p in model.parameters())
            for dimension in DIMENSIONS:
                for count in COUNTS:
                    names=[f'{mode}_d{dimension}_k{count}' for mode in 'BD']
                    assert len({initial_hashes[name] for name in names}) == 1
            run.verify_hashes(plan['hashes'])
            run.base.write(OUT/'summary.json',dict(arms=summaries,initial_model_hashes=initial_hashes,parent_unchanged=True))
            artifacts={str(path.relative_to(OUT)):run.base.sha(path) for path in OUT.rglob('*.json') if path.name!='status.json'}
            run.base.write(OUT/'completion.json',dict(passed=True,context=context,artifacts=artifacts,gate_weights_saved=False))
            run.base.status(OUT,'all_gates_passed',arms=len(ARMS),success_markers=4*len(ARMS))
            return authenticate()
        except Exception as error:
            run.base.status(OUT,'failed',error=repr(error))
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--then-train',action='store_true')
    args=parser.parse_args()
    gates(args.wait)
    if args.then_train:
        print('All authenticated gates passed; starting fresh convergence comparison',flush=True)
        os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/run_messenger_codebook_reconstruction.py'),'train','--wait'])


if __name__=='__main__':
    main()
