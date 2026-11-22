"""Finish the high-occlusion evaluation, authenticate artifacts, then delete its source experiment weights."""
import argparse, fcntl, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_messenger_multiaxis import sha
from scripts.update_development_journal import main as journal
SOURCE=ROOT/'outputs/diagnostics/messenger_adversarial_comparison_20261119'
OUT=ROOT/'outputs/diagnostics/messenger_high_occlusion_20261119_run02'
LOW=ROOT/'outputs/diagnostics/messenger_known_mask_patterns_20261119_run02'
NAMES=['raw','baseline','adversarial','adversarial_fm']

def read(p):return json.loads(p.read_text())
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def status(phase,**kw):
    write(OUT/'finalization_status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw))
    print(phase,kw,flush=True)
def verify_completion(folder,marker):
    obj=read(folder/marker)
    assert obj['complete']
    for name,h in obj['artifacts'].items():
        p=ROOT/name if name.startswith('outputs/') else folder/name
        if folder == OUT and name == 'finalization_status.json':
            # A live monitor is not an immutable numerical artifact. Preserve the
            # original completion manifest and explicitly audit this sole exclusion.
            continue
        assert sha(p)==h,str(p)
    return obj
def finish(wait):
    with (OUT/'finalizer.lock').open('a') as single:
        fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OUT/'completion.json').exists():
            phase=read(OUT/'status.json')['phase']
            if phase=='failed':raise RuntimeError('Evaluator failed; all checkpoints retained')
            if not wait:raise RuntimeError('Evaluation not complete')
            status('waiting_for_evaluation',evaluation_phase=phase);time.sleep(30)
        status('verifying')
        verify_completion(OUT,'completion.json')
        verify_completion(LOW,'completion.json')
        verify_completion(SOURCE,'numeric_completion.json')
        for folder in [OUT,LOW]:
            for p,h in read(folder/'plan.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
        bins=[(LOW,'area_0.1_0.2','10-20%'),(LOW,'area_0.2_0.3','20-30%'),
              (LOW,'area_0.3_0.4','30-40%'),(OUT,'area_0.4_0.6','40-60%'),
              (OUT,'area_0.6_0.8','60-80%'),(OUT,'area_0.8_0.9','80-90%')]
        combined=[]
        for folder,key,label in bins:
            group=read(folder/'summary.json')[key]
            a=np.load(folder/'predictions.npz')
            _,lo,hi=key.split('_')
            sel=(a['fraction']>=float(lo))&(a['fraction']<float(hi))
            assert int(sel.sum())==group['patterns']
            for name,mask in [('covered',a['point']),('uncovered',~a['point']),('all',np.ones_like(a['point']))]:
                m=mask&sel[:,None]
                if not m.any():continue
                for i,arm in enumerate(NAMES):
                    er=a['rows'][:,i,:,0][m].astype(float);pr=a['rows'][:,i,:,1][m]
                    pub=group['metrics'][name][arm]
                    assert len(er)==pub['points']
                    assert abs(float(er.mean())-pub['mean'])<1e-12
                    assert int(((er<=.05)&(pr>=.7)).sum())==pub['joint5']
            combined.append(dict(coverage=label,**group))
        write(OUT/'coverage_comparison.json',combined)
        lines=['Frozen head + known binary mask gate; position error / GT ellipse major diameter.',
               'Raw means DINO features from the OCCLUDED image. No clean-feature-to-head evaluation.',
               'Same128faces/35videos, low7680 + high11520 artificial masks. Video bootstrap5000.',
               'Training occlusion was10-40%; >40% is extrapolation, no retraining.',
               '| Occluded face area | cases | points covered | raw | baseline R | adversarial R | adversarial+FM R |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for g in combined:
            m=g['metrics']['covered']
            values=' | '.join(f'{100*m[n]["mean"]:.5f}%' for n in NAMES)
            lines.append(f'| {g["coverage"]} | {g["patterns"]} | {m["raw"]["points"]} | {values} |')
            for n in NAMES[1:]:
                d=m[n]['delta']
                lines.append(f'  {n} minus raw: {100*d["difference"]:.5f}pp;95%CI={[100*x for x in d["video_bootstrap_95"]]}')
        (OUT/'coverage_comparison.txt').write_text('\n'.join(lines)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for n in NAMES:
            means=[100*g['metrics']['covered'][n]['mean'] for g in combined]
            axes[0].plot(range(6),means,marker='o',label=n)
        for i,n in enumerate(NAMES[1:]):
            stats=[g['metrics']['covered'][n]['delta'] for g in combined]
            means=np.array([x['difference'] for x in stats])*100
            ci=np.array([x['video_bootstrap_95'] for x in stats])*100
            axes[1].errorbar(np.arange(6)+(i-1)*.12,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o-',capsize=2,label=n)
        for ax in axes:
            ax.set_xticks(range(6),[g['coverage'] for g in combined]);ax.set_xlabel('Occluded fraction of GT ellipse area');ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[0].set_ylabel('Covered point error (%)')
        axes[1].set_ylabel('R minus raw error (percentage points)');axes[1].axhline(0,color='gray')
        fig.tight_layout();fig.savefig(OUT/'coverage_comparison.png',dpi=150);plt.close(fig)
        # Save the full numerical audit before deleting anything.
        write(OUT/'numerical_audit.json',dict(complete=True,source_and_weights_verified=True,all_bins_recomputed=True,
            original_384_anchor=read(OUT/'anchor_parity.json'),training=False,weights_generated=0,
            mutable_artifact_exclusions=[dict(path='finalization_status.json',reason='Live finalizer monitor accidentally included by evaluator manifest; original manifest unchanged. All numeric artifacts remain strictly authenticated.')],
            finalizer_source_sha256=sha(Path(__file__)),
            artifacts={p.name:sha(p) for p in [OUT/'coverage_comparison.json',OUT/'coverage_comparison.txt',OUT/'coverage_comparison.png']}))
        status('waiting_for_deletion_lease')
        with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
            while True:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if not wait:raise
                    time.sleep(15)
            # The evaluator may be closing its process; wait for its exit and reject any other user of these files.
            for attempt in range(12):
                running=[]
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    except (OSError,PermissionError):continue
                    if ('python' in cmd and any(token in cmd for token in
                        ['scripts.evaluate_messenger_high_occlusion','scripts/evaluate_messenger_high_occlusion.py',
                         'scripts/run_messenger_adversarial_comparison.py','scripts/resume_messenger_adversarial_durable.py'])):
                        running.append(dict(pid=int(proc.name),cmd=cmd))
                if not running:break
                time.sleep(5)
            assert not running,running
            expected=read(SOURCE/'summary.json')['checkpoints']
            assert len(expected)==12
            targets=[]
            for arm in NAMES[1:]:
                for name in ['best_R.pt','last_R.pt','best_g.pt','last_g.pt']:
                    p=SOURCE/arm/name
                    assert not p.is_symlink() and p.resolve().parent==(SOURCE/arm).resolve()
                    assert p.resolve().is_relative_to(SOURCE.resolve()) and p.is_file()
                    recorded=next(x for x in expected if ROOT/x['path']==p)
                    assert sha(p)==recorded['sha256']
                    targets.append(dict(path=str(p.resolve()),sha256=recorded['sha256'],bytes=p.stat().st_size))
            paths={x['path'] for x in targets}
            open_users=[]
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                try:fds=list((proc/'fd').iterdir())
                except (OSError,PermissionError):continue
                for fd in fds:
                    try:dest=str(fd.resolve())
                    except (OSError,RuntimeError):continue
                    if dest in paths:open_users.append(dict(pid=int(proc.name),path=dest))
            assert not open_users,open_users
            audit=dict(phase='prepared',authorized_by='2026-11-19 user instruction: delete after numeric evaluation, before final report',
                files=targets,total_bytes=sum(x['bytes'] for x in targets),checked_processes=True,canonical_lease=True,
                numerical_audit_sha=sha(OUT/'numerical_audit.json'),preserved='fixed V3 baseline and selected scale C/level+scale C, all logs/results/hashes',
                historical_completion_flags_unchanged=True)
            deletion=SOURCE/'checkpoint_deletion_20261119.json'
            assert not deletion.exists(),'Deletion audit already exists; inspect before retry'
            write(deletion,audit)
            for item in targets:Path(item['path']).unlink()
            assert all(not Path(x['path']).exists() for x in targets)
            audit.update(phase='completed',utc=datetime.now(timezone.utc).isoformat(),remaining_target_weights=0)
            write(deletion,audit)
            previous=read(SOURCE/'status.json')
            write(SOURCE/'status_before_deletion_20261119.json',previous)
            previous.update(phase='completed_weights_deleted',weights_retained=False,checkpoint_count=0,
                deletion_audit=deletion.name,utc=datetime.now(timezone.utc).isoformat())
            write(SOURCE/'status.json',previous)
        metrics='; '.join(g['coverage']+': '+ '/'.join(f'{100*g["metrics"]["covered"][n]["mean"]:.5f}' for n in NAMES) for g in combined)
        journal(['add','--entry-id','messenger-high-occlusion-20261119','--title','High-area artificial occlusion with known gate',
            '--status','completed','--source','side conversation user request',
            '--design','Evaluate whether raw features remain best at40-90% occlusion; freeze all3 learned R and head, exact known mask gate.',
            '--implementation','High-area generator targets uniform ellipse-area coverage bins40-60/60-80/80-90, three silhouettes,10seeds each per128faces. No model-dependent mask selection.',
            '--training-condition','No training. 11520high-area masks + prior7680low-area masks. Reused eval128faces/35videos, unseen masks; >40% outside R training support.',
            '--training-result',metrics,
            '--artifact','outputs/diagnostics/messenger_high_occlusion_20261119_run02/coverage_comparison.txt',
            '--artifact','outputs/diagnostics/messenger_high_occlusion_20261119_run02/numerical_audit.json',
            '--artifact','outputs/diagnostics/messenger_adversarial_comparison_20261119/checkpoint_deletion_20261119.json',
            '--verification','180geometry checks passed; historical384masks reproduced; all artifact/source/weight SHA checked, all bins independently recomputed; no clean-feature head. Original evaluator interrupted by Windows reboot before evaluation; preserved. All12 adversarial-comparison weights deleted after final numeric audit under canonical lease; fixed baseline and selected candidates excluded.'])
        os.sync()
        status('completed',evaluation_complete=True,source_experiment_weights_deleted=12,deletion_bytes=audit['total_bytes'])
        print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true')
    try:finish(p.parse_args().wait)
    except Exception as e:
        status('failed',error=repr(e),automatic_retry=False)
        raise
