#!/usr/bin/env python3
"""Fresh image-policy gates and guarded production, with a single-run F1 override."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.messenger_natural_policy_runtime import *
from src.models.messenger_policy_quality_override import assess_once
from scripts.messenger_natural_occlusion_common import verify_gates
from src.models.messenger_occlusion_training import rng_state,restore_rng

STAGES=('schema','gpu_step','overfit','small_scale')


class QualityFailure(RuntimeError):pass


def learning_assessment(context,rows,initial):
    cfg=context[0];budget=cfg['small_scale']
    if len(rows)!=budget['epochs'] or rows[-1]['step']!=budget['updates']:
        raise QualityFailure('Incomplete predeclared learning budget')
    values=[initial[T_METRIC]]+[r['validation'][T_METRIC] for r in rows]
    if not all(math.isfinite(v) and 0<=v<=1 for v in values):raise QualityFailure('Nonfinite F1 trajectory')
    if not all(r['finite'] and r['full_validation_crops']==768 for r in rows):
        raise QualityFailure('Incomplete/nonfinite full validation')
    requested=[sum(r['composition']['requested_per_bin'][b] for r in rows) for b in range(4)]
    applied=[sum(r['composition']['applied_per_bin'][b] for r in rows) for b in range(4)]
    if (not min(applied) or len(set(applied))!=1 or 1-sum(applied)/sum(requested)>cfg['audit']['maximum_skip']
            or max(1-a/n for a,n in zip(applied,requested))>cfg['audit']['maximum_stratum_skip']):
        raise QualityFailure('Observed training composition distribution/skip guard failed')
    report=assess_once(json.loads((ROOT/cfg['override']).read_text()),
        gate_run_id=cfg['small_scale_run_id'],config_path=ROOT/cfg['policy_config'],stage='small_scale',split='val',
        initial_f1=values[0],candidate_f1=max(values[1:]),
        initial_validation_contract=validation_contract(context),candidate_validation_contract=validation_contract(context))
    report.update(values=values,best_epoch=1+values[1:].index(max(values[1:])),updates=rows[-1]['step'])
    return report


def audit(context,output):
    cfg,policy,base,spec,*_=context;dataset,_=build_data(policy,base,spec);results=[]
    for epoch in cfg['audit']['epochs']:
        dataset.set_epoch(epoch);indices=sorted(dataset.requests)
        def check(i):
            r=dataset[i];m=dict(r['metadata'],feasible=bool(r['candidates']),real_available=r['real'] is not None)
            m['candidates']=[c['metadata'] for c in r['candidates']]
            return m
        with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(check,indices))
        requested=[sum(r['bin']==b for r in records) for b in range(4)]
        feasible=[sum(r['bin']==b and r['feasible'] for r in records) for b in range(4)]
        # Exact successful equal-bin balancing; count discarded excess as skips.
        accepted=[min(feasible)]*4
        skip=1-sum(accepted)/sum(requested)
        strata=[1-a/n for a,n in zip(accepted,requested)]
        real=sum(r['real_available'] for r in records)/len(records)
        passed=(skip<=cfg['audit']['maximum_skip'] and max(strata)<=cfg['audit']['maximum_stratum_skip']
                and min(requested)>=cfg['audit']['minimum_requests_per_stratum'] and real>=cfg['audit']['minimum_real_pair_fraction'])
        result=dict(epoch=epoch,train_crops=len(dataset),eligible=len(dataset.eligible),requested=requested,
            feasible=feasible,balanced_accepted=accepted,skip_fraction=skip,stratum_skip=strata,
            real_pair_fraction=real,passed=passed,records=records)
        write_json(output/f'audit_epoch_{epoch}.json',result)
        results.append({k:v for k,v in result.items() if k!='records'})
        print(json.dumps(dict(stage='schema_audit',**results[-1])),flush=True)
    write_json(output/'audit.json',dict(results=results,passed=all(r['passed'] for r in results)))
    if not all(r['passed'] for r in results):raise QualityFailure('Full-train composition audit failed')


def schema(context,output):
    tests=['tests/test_messenger_natural_policy_gate.py','tests/test_messenger_occlusion_policy.py',
           'tests/test_messenger_gate_metrics.py','tests/test_messenger_policy_quality_override.py',
           'tests/test_messenger_natural_occlusion.py','tests/test_messenger_prototype_observer.py',
           'tests/test_messenger_observer_diagnostic.py','keypoint_comparison1/tests/test_geometry_prior_recovery.py']
    with (output/'tests.txt').open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*tests],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    audit(context,output)
    write_json(output/'schema_report.json',dict(topology=context[-1]['topology'],tests=tests,
        inheritance='pinned v2 model and composition; separate prospective execution contract',
        production='requires all four fresh independent gates; gate weights never production initialization'))


def candidate(context,output,batch):
    runtime=Runtime(context,stage='gpu_step',microbatch=batch,output=output)
    indices=subset_indices(len(runtime.dataset),128)
    batches=epoch_batches(runtime.dataset,indices,1)
    rows=[read_rows(runtime.dataset,ids) for ids in batches]
    for name,opt in runtime.opts.items():
        for group in opt.param_groups:group['lr']=group['initial_lr']
    runtime.freeze_epoch(1,full=True)
    observations=[];torch.cuda.reset_peak_memory_stats()
    with SafetyMonitor() as safety:
        for cycle in range(2):
            torch.cuda.synchronize();start=time.perf_counter();updates=[]
            for sub in rows:updates.append(runtime.update.update(sub,microbatch=batch))
            torch.cuda.synchronize();train_seconds=time.perf_counter()-start
            start=time.perf_counter();metrics=runtime.validation();torch.cuda.synchronize()
            validation_seconds=time.perf_counter()-start;safety.check()
            if any(u['retries'] or u['microbatch']!=batch or not u['g_updated'] or not u['d_updated'] for u in updates) or runtime.validation_retries:
                raise RuntimeError('Candidate OOM fallback or missing G/D updates')
            if not all(p.grad is None for p in runtime.model.observer.parameters()):raise AssertionError('Observer gradient')
            observations.append(dict(step=runtime.update.step,train_seconds=train_seconds,
                validation_seconds=validation_seconds,validation=metrics,updates=updates))
        measured=observations[-1]
        result=dict(status='safe',contract=context[-1],microbatch=batch,optimizer_updates=runtime.update.step,
            validation_crops=len(runtime.val),finite=True,oom=False,full_backbone=True,
            projected_seconds=(measured['train_seconds']/128*len(runtime.dataset)+measured['validation_seconds'])*40,
            warmup_updates_excluded=2,rows=observations,**safety.metrics())
    if result['peak_reserved_mib']>safety.rows[0]['total']*.9:raise RuntimeError('Reserved VRAM limit exceeded')
    write_json(output,result)


def preflight(context,output,config):
    attempts=[]
    for batch in context[0]['preflight']['candidates']:
        p=output/f'batch_{batch}.json'
        with (output/f'batch_{batch}.log').open('w') as log:
            process=subprocess.run([str(ROOT/'.venv/bin/python'),__file__,'--config',str(config),
                '--stage','gpu_step','--candidate',str(batch),'--candidate-output',str(p)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(p.read_text()) if process.returncode==0 else dict(status='unsafe',microbatch=batch,returncode=process.returncode)
        attempts.append(result);write_json(output/'attempts.json',dict(attempts=attempts))
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','contract')}),flush=True)
    safe=[a for a in attempts if a['status']=='safe']
    if not safe:raise QualityFailure('No safe preflight candidate')
    fastest=min(a['projected_seconds'] for a in safe)
    chosen=min(a['microbatch'] for a in safe if a['projected_seconds']<=fastest*1.03)
    write_json(output/'batch_plan.json',dict(contract=context[-1],attempts=attempts,selected_microbatch=chosen))


def checked_marker(context,stage):
    path=ROOT/context[0]['gate_root']/stage/'success.json'
    marker=json.loads(path.read_text())
    if marker['status']!='passed' or marker['stage']!=stage or marker['contract']!=context[-1]:
        raise ValueError('Stale/missing prerequisite '+stage)
    if not marker['evidence'] or any(file_hash(ROOT/p)!=h for p,h in marker['evidence'].items()):
        raise ValueError('Changed gate evidence '+stage)


def issue_marker(context,output,stage):
    files=[p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.txt')
           and 'wandb' not in p.parts and p.name not in ('success.json','failure.json')]
    evidence={str(p.relative_to(ROOT)):file_hash(p) for p in files}
    write_json(output/'success.json',dict(status='passed',stage=stage,contract=context[-1],evidence=evidence))


def learning(context,output,stage,*,resume=False,stop_after_epoch=None):
    cfg=context[0];plan=json.loads((ROOT/cfg['gate_root']/'gpu_step/batch_plan.json').read_text())
    if plan['contract']!=context[-1]:raise ValueError('Stale batch plan')
    batch=plan['selected_microbatch'];runtime=Runtime(context,stage=stage,microbatch=batch,output=output)
    probe_rows,scanned=prepare_rows(runtime.dataset,cfg['probe'])
    write_json(output/'probe_manifest.json',dict(scanned=scanned,indices=[r['metadata']['index'] for r in probe_rows],
        candidates=[[c['metadata'] for c in r['candidates']] for r in probe_rows]))
    fixed=None
    if stage=='overfit':
        fixed,_=prepare_rows(runtime.dataset,dict(requested_per_bin=1,natural_crops=4,candidate_scan_per_bin=128))
        fixed=[dict(r,_sampled_action=0) for r in fixed]
        runtime.indices=[r['metadata']['index'] for r in fixed]
    epochs=cfg[stage]['epochs']
    history_path=output/'history.jsonl'
    history=[json.loads(s) for s in history_path.read_text().splitlines()] if resume else []
    if resume:
        runtime.policy.restore(torch.load(output/'checkpoints/last.pt',map_location='cpu',weights_only=False))
        runtime.update.step=runtime.policy.last_metrics['optimizer_step']
        if len(history)!=runtime.policy.epoch:raise ValueError('Checkpoint/history boundary mismatch')
        runtime.microbatch=history[-1]['microbatch']
        runtime.validation_batch=history[-1]['validation_batch']
    run=start_wandb(context,output,stage,resume)
    try:
        if not resume:
            initial=runtime.validation();probe=fixed_gate_probe(runtime.task,probe_rows,batch_size=batch)
            write_json(output/'epoch0_validation.json',initial);write_json(output/'gate_reference.json',probe)
            write_json(output/'validation_contract.json',validation_contract(context))
            write_json(output/'training_indices.json',dict(indices=runtime.indices,selection=cfg['sampling']))
            if fixed:write_json(output/'overfit_initial.json',dict(loss=runtime.fixed_loss(fixed)))
            run.log(dict(epoch=0,optimizer_step=0,**scalar_report(initial,probe)),step=0)
        initial=json.loads((output/'epoch0_validation.json').read_text())
        reference=json.loads((output/'gate_reference.json').read_text())
        with SafetyMonitor() as safety:
            while runtime.policy.epoch<epochs and not (stage=='production' and runtime.policy.should_stop):
                epoch=runtime.policy.epoch+1;runtime.freeze_epoch(epoch)
                batches=epoch_batches(runtime.dataset,runtime.indices,epoch) if not fixed else [None]
                updates=[];torch.cuda.synchronize();start=time.perf_counter()
                for ids in batches:
                    rows=read_rows(runtime.dataset,ids) if ids is not None else fixed
                    update=runtime.train_window(rows);updates.append(update);safety.check()
                torch.cuda.synchronize();train_seconds=time.perf_counter()-start
                metrics=runtime.validation();metrics['optimizer_step']=runtime.update.step
                probe=fixed_gate_probe(runtime.task,probe_rows,batch_size=runtime.validation_batch)
                improved=runtime.policy.record(metrics)
                complete=epoch==epochs or (stage=='production' and runtime.policy.should_stop)
                runtime.policy.save(output/'checkpoints',improved=improved,complete=complete)
                report=dict(epoch=epoch,step=runtime.update.step,validation=metrics,probe=probe,
                    gate_delta=gate_deltas(probe['synthetic'],reference['synthetic']),
                    validation_gate_delta=gate_deltas(metrics['gates'],initial['gates']),
                    finite=True,full_validation_crops=len(runtime.val),updates=updates,
                    composition=audit_actions([r for u in updates for r in u['records']]),
                    train_seconds=train_seconds,microbatch=runtime.microbatch,validation_batch=runtime.validation_batch,
                    backbone_frozen=epoch<=cfg['freeze_backbone_epochs'],safety=safety.metrics())
                if fixed:report['fixed_task_loss']=runtime.fixed_loss(fixed)
                with history_path.open('a') as stream:stream.write(json.dumps(report)+'\n')
                history.append(report);write_json(output/'progress.json',report)
                scalars=scalar_report(metrics,probe)
                run.log(dict(epoch=epoch,optimizer_step=runtime.update.step,**scalars),step=epoch)
                print(json.dumps(dict(stage=stage,epoch=epoch,step=runtime.update.step,**scalars)),flush=True)
                if stop_after_epoch==epoch:
                    write_json(output/'resume_boundary.json',dict(epoch=epoch,step=runtime.update.step,
                        checkpoint_sha256=file_hash(output/'checkpoints/last.pt')))
                    return False
        if stage=='small_scale':
            assessment=learning_assessment(context,history,initial)
            write_json(output/'assessment.json',assessment)
            if not assessment['criterion_passed']:raise QualityFailure('F1 did not improve under the one-off criterion')
        elif stage=='overfit':
            values=[json.loads((output/'overfit_initial.json').read_text())['loss']]+[r['fixed_task_loss'] for r in history]
            assessment=dict(initial=values[0],best=min(values[1:]),values=values,
                passed=all(math.isfinite(v) for v in values) and min(values[1:])<values[0],resumed=resume,
                checkpoint_selection=T_METRIC,loss_used_only_for_gate=True)
            write_json(output/'assessment.json',assessment)
            if not assessment['passed'] or not resume:raise QualityFailure('Fixed-input loss/resume gate failed')
        if stage in ('overfit','small_scale'):
            for p in (output/'checkpoints').glob('*.pt'):
                if p.name!='best.pt':p.unlink()
        write_json(output/'completion.json',dict(status='complete',stage=stage,epochs=runtime.policy.epoch,
            updates=runtime.update.step,best_f1=runtime.policy.best,best_epoch=runtime.policy.best_epoch,
            initial_f1=initial[T_METRIC],wandb_url=run.url,contract=context[-1],
            checkpoint_sha256=file_hash(output/'checkpoints/best.pt'),test_used=False))
        return True
    finally:run.finish()


def run(args):
    context=authenticate_execution(args.config);cfg=context[0];stage=args.stage
    if args.candidate:
        candidate(context,args.candidate_output,args.candidate);return
    output=ROOT/(cfg['production_root'] if stage=='production' else cfg['gate_root']+'/'+stage)
    if stage=='production':verify_gates(ROOT/cfg['gate_root'],context[-1])
    elif stage!='schema':checked_marker(context,STAGES[STAGES.index(stage)-1])
    if args.resume:
        if stage not in ('overfit','small_scale','production') or (output/'completion.json').exists():raise ValueError('Invalid/completed resume')
        if json.loads((output/'contract.json').read_text())!=context[-1]:raise ValueError('Source/contract drift on resume')
    else:
        output.mkdir(parents=True,exist_ok=False);write_json(output/'contract.json',context[-1])
        for name in context[-1]['identity']['sources']:
            target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    try:
        completed=True
        if stage=='schema':schema(context,output)
        elif stage=='gpu_step':preflight(context,output,args.config)
        else:completed=learning(context,output,stage,resume=args.resume,stop_after_epoch=args.stop_after_epoch)
        if authenticate_execution(args.config)[-1]!=context[-1]:raise ValueError('Sources changed during gate')
        if completed and stage in STAGES:issue_marker(context,output,stage)
    except Exception as error:
        write_json(output/'failure.json',dict(status='failed',stage=stage,classification='quality_failure' if isinstance(error,QualityFailure) else 'execution_error',
            error=str(error),traceback=traceback.format_exc(),contract=context[-1]))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training_plans/messenger_natural_policy_v2_execution_20261018.json')
    p.add_argument('--stage',choices=[*STAGES,'production'],required=True)
    p.add_argument('--candidate',type=int,choices=[4,8,16]);p.add_argument('--candidate-output',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-epoch',type=int)
    a=p.parse_args()
    if a.stop_after_epoch is not None and (a.stage!='overfit' or a.stop_after_epoch!=1):p.error('Only overfit epoch1 resume certification may stop early')
    run(a)
