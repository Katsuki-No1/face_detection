"""Compare validation-plateau results with the previous four-epoch experiment."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/diagnostics/messenger_multiaxis_convergence_20261114'
OLD=ROOT/'outputs/diagnostics/messenger_multiaxis_20261114'


def read(p):return json.loads(p.read_text())


def feature_interval(path):
    p=np.load(path);videos=np.unique(p['videos']);sums=[];counts=[]
    for v in videos:
        m=p['videos']==v;sums.append(p['error_sums'][m,:2].sum(0));counts.append(p['counts'][m].sum())
    sums=np.stack(sums);counts=np.array(counts);rng=np.random.default_rng(42)
    ix=rng.integers(0,len(videos),size=(2000,len(videos)))
    differences=(sums[ix,1]-sums[ix,0]).sum(1)/counts[ix].sum(1)
    return dict(fused_minus_raw=float((sums[:,1]-sums[:,0]).sum()/counts.sum()),
                video_bootstrap_95=np.quantile(differences,[.025,.975]).tolist(),videos=len(videos))


def paired_feature_interval(first, second):
    a=np.load(first);b=np.load(second)
    for key in ('videos','counts','kind','fraction'):
        assert np.array_equal(a[key],b[key]),key
    videos=np.unique(a['videos']);sums=[];counts=[]
    for v in videos:
        mask=a['videos']==v
        sums.append((a['error_sums'][mask,1]-b['error_sums'][mask,1]).sum())
        counts.append(a['counts'][mask].sum())
    sums=np.array(sums);counts=np.array(counts)
    ix=np.random.default_rng(42).integers(0,len(videos),size=(2000,len(videos)))
    differences=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(first_minus_second=float(sums.sum()/counts.sum()),
                video_bootstrap_95=np.quantile(differences,[.025,.975]).tolist(),videos=len(videos))


def main():
    assert read(OUT/'status.json')['phase']=='completed'
    summary=read(OUT/'summary.json')
    assert summary['all_evaluations_completed']
    checkpoints=summary['retained_checkpoints']
    assert len(checkpoints)==36
    for checkpoint in checkpoints:
        path=ROOT/checkpoint['path']
        assert path.stat().st_size==checkpoint['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint['sha256']
    for condition in summary['conditions'].values():
        assert condition['R']['converged'] and condition['g']['converged']
    (OUT/'completion_audit.json').write_text(json.dumps(dict(all_nine_R_and_g_converged=True,
        retained_checkpoint_count=len(checkpoints),all_checkpoint_hashes_verified=True,
        checkpoint_bytes=sum(p['bytes'] for p in checkpoints)),indent=2)+'\n')
    current=read(OUT/'comparison.json');old=read(OLD/'comparison.json');table={}
    for name,m in current.items():
        p=np.load(OUT/name/'natural_points.npz');rows=p['rows'];state=p['state'];n=state==1
        full=rows[:,1];position_ok=(full[:,:,0]<=.1)&n;present=(full[:,:,1]>=.7)&n
        table[name]=dict(R_stop=m['R_convergence']['stop_epoch'],R_best=m['R_convergence']['selected_epoch'],
            g_stop=m['g_convergence']['stop_epoch'],g_best=m['g_convergence']['selected_epoch'],
            four_epoch_fused=old[name]['synthetic']['fused'],converged_fused=m['synthetic']['fused'],
            artificial_vs_raw=feature_interval(OUT/name/'synthetic_rows.npz'),
            natural=m['learned_gate'],full_recovery=m['full_recovery'],raw=m['raw'],
            full_occluded_presence_pass=int(present.sum()),full_occluded_position_pass=int(position_ok.sum()),
            gate=m['extra']['gate_at_points'],parameters=m['parameters'])
    (OUT/'comparison_with_four_epochs.json').write_text(json.dumps(table,indent=2)+'\n')
    pairs={}
    for family in ('scale','level','level_scale'):
        for first,second in (('B','A'),('C','A'),('C','B')):
            pairs[f'{family}_{first}_minus_{second}']=paired_feature_interval(
                OUT/f'{family}_{first}'/'synthetic_rows.npz',
                OUT/f'{family}_{second}'/'synthetic_rows.npz')
    (OUT/'paired_feature_comparisons.json').write_text(json.dumps(pairs,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for row,family in enumerate(('scale','level','level_scale')):
        for condition in ('A','B','C'):
            name=f'{family}_{condition}';h=read(OUT/name/'history.json');g=read(OUT/name/'gate_history.json')
            axes[row,0].plot([x['epoch'] for x in h[1:]],[x['dev']['fused'] for x in h[1:]],label=condition)
            axes[row,1].plot([x['epoch'] for x in g],[x['dev'] for x in g],label=condition)
        for col in (0,1):
            axes[row,col].axvline(4,color='gray',linestyle=':',label='previous cutoff')
            axes[row,col].set_title(family+' / '+('recovery validation' if col==0 else 'gate validation'))
            axes[row,col].set_xlabel('epoch');axes[row,col].grid(alpha=.2);axes[row,col].legend()
        axes[row,0].set_ylabel('covered cosine distance + MSE');axes[row,1].set_ylabel('position Smooth L1')
    fig.suptitle('Nine conditions trained to validation plateaus (fixed data, seed 42)')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
    for name,m in table.items():
        print(name,'R',m['R_stop'],m['R_best'],'g',m['g_stop'],m['g_best'],
              'error',round(m['converged_fused'],5),'occ',m['natural']['occluded']['correct'],
              'vis',m['natural']['visible']['correct'],'full-occ',m['full_recovery']['occluded']['correct'])


if __name__=='__main__':main()
