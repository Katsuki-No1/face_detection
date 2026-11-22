"""CPU-only paired threshold curves for frozen natural-occlusion predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/diagnostics/messenger_diffusion_comparison_20261114'
OUT=ROOT/'outputs/diagnostics/messenger_success_thresholds_20261118'
ARMS=('det_feature','det_head','diff_feature','diff_head')
NAMES=('raw',*ARMS)
THRESHOLDS=np.array([.01,.02,.03,.04,.05,.06,.075,.08,.09,.10,.125,.15,.20])


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def paired_interval(differences,videos):
    unique=np.unique(videos)
    sums=np.array([differences[videos==v].sum() for v in unique]);counts=np.array([(videos==v).sum() for v in unique])
    ix=np.random.default_rng(42).integers(0,len(unique),(5000,len(unique)))
    values=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(difference=float(differences.mean()),video_bootstrap_95=np.quantile(values,[.025,.975]).tolist(),videos=len(unique))


def summarize(rows,videos):
    # rows: methods x points x (normalized coordinate error, presence).
    errors=rows[:,:,0];present=rows[:,:,1]>=.7
    joint=(errors[:,:,None]<=THRESHOLDS)&present[:,:,None]
    location=errors[:,:,None]<=THRESHOLDS
    assert np.all(np.diff(joint.sum(1),axis=1)>=0)
    assert np.all(np.diff(location.sum(1),axis=1)>=0)
    result={}
    for k,name in enumerate(NAMES):
        result[name]=dict(points=rows.shape[1],presence_pass=int(present[k].sum()),
            mean_error=float(errors[k].mean()),median_error=float(np.median(errors[k])),
            quantiles_90_95=np.quantile(errors[k],[.9,.95]).tolist(),thresholds=[])
        for t,threshold in enumerate(THRESHOLDS):
            j=joint[k,:,t];r=joint[0,:,t];l=location[k,:,t];rl=location[0,:,t]
            # Exact counterfactual accounting, not a causal intervention on the network.
            location_only_change=l&present[0]
            presence_only_change=rl&present[k]
            loc_contribution=.5*(int(location_only_change.sum()-r.sum())+int(j.sum()-presence_only_change.sum()))
            presence_contribution=.5*(int(presence_only_change.sum()-r.sum())+int(j.sum()-location_only_change.sum()))
            assert loc_contribution+presence_contribution==int(j.sum()-r.sum())
            result[name]['thresholds'].append(dict(tolerance=float(threshold),joint_correct=int(j.sum()),
                location_only_correct=int(l.sum()),joint_wins=int((j&~r).sum()),joint_losses=int((r&~j).sum()),
                joint_delta=paired_interval(j.astype(float)-r,videos),
                location_delta=paired_interval(l.astype(float)-rl,videos),
                raw_presence_new_location_correct=int(location_only_change.sum()),
                new_presence_raw_location_correct=int(presence_only_change.sum()),
                symmetric_location_contribution_count=loc_contribution,
                symmetric_presence_contribution_count=presence_contribution))
        if k:
            delta=errors[k]-errors[0]
            result[name]['coordinate_comparison']=dict(improved=int((delta<0).sum()),worsened=int((delta>0).sum()),
                unchanged=int((delta==0).sum()),mean_difference=paired_interval(delta,videos),
                joint_pass_both_10pct_mean_error_raw=float(errors[0][joint[k,:,-4]&joint[0,:,-4]].mean())
                if (joint[k,:,-4]&joint[0,:,-4]).any() else None)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    completion=read(SOURCE/'numeric_completion.json')
    assert completion['all_numeric_evaluations_completed']
    for name,expected in completion['artifacts'].items():assert sha(SOURCE/name)==expected,name
    summary=read(SOURCE/'summary.json');old=read(SOURCE/'comparison_audit.json')
    loaded={arm:np.load(SOURCE/arm/'natural_seed42.npz') for arm in ARMS}
    ref=loaded[ARMS[0]]
    for a in loaded.values():
        for key in ('ids','state','videos'):assert np.array_equal(a[key],ref[key]),key
        np.testing.assert_allclose(a['rows'][:,0],ref['rows'][:,0],atol=1e-7,rtol=0)
        assert np.isfinite(a['rows']).all() and (a['rows'][...,0]>=0).all()
    rows=np.stack([ref['rows'][:,0],*[loaded[arm]['rows'][:,2] for arm in ARMS]])
    videos=np.broadcast_to(ref['videos'][:,None],ref['state'].shape)
    result={}
    for state,label,expected in [(1,'occluded',349),(2,'visible',1852)]:
        selected=ref['state']==state;assert int(selected.sum())==expected
        result[label]=summarize(rows[:,selected],videos[selected])
        for arm in ARMS:
            stats=result[label][arm];m=summary['conditions'][arm]['natural']['learned_g'][label]
            at10=next(r for r in stats['thresholds'] if r['tolerance']==.1)
            at5=next(r for r in stats['thresholds'] if r['tolerance']==.05)
            assert at10['joint_correct']==m['correct']
            assert at5['joint_correct']==old['exploratory_position_distributions'][arm]['learned_g'][label]['presence_and_error_under_5pct']
            assert abs(stats['mean_error']-m['mean_error']['mean'])<1e-7
    hashes={str((SOURCE/arm/'natural_seed42.npz').relative_to(ROOT)):sha(SOURCE/arm/'natural_seed42.npz') for arm in ARMS}
    report=dict(conditions=result,tolerances=THRESHOLDS.tolist(),presence_threshold=.7,source_hashes=hashes,
        primary_generation_seed=42,training=False,inference=False,denominator='fixed GT-annotated points, state1 occluded/state2 visible',
        normalization='coordinate distance / GT ellipse major diameter; same as original diagnostic',
        verification='Published mean error, 5%/10% success, identities and raw predictions reproduced; monotonicity and attribution sum verified',
        limitations=['same reused development evaluation, single training seed and fixed primary generation seed',
            'threshold sweep is exploratory, not new threshold/model selection for deployment',
            '5000 video-cluster paired bootstrap replicates; pointwise intervals unadjusted for multiple thresholds',
            'counterfactual presence/location attribution is arithmetic, not a causal separation of network behavior'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid=np.linspace(0,.2,401);selected=ref['state']==1;e=rows[:,selected,0];p=rows[:,selected,1]>=.7
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    labels=['Original','Regression','Regression + head','Diffusion','Diffusion + head']
    colors=['black','#2563eb','#60a5fa','#dc2626','#f97316']
    for k,label in enumerate(labels):
        for i in (0,1):
            successes=(e[k,:,None]<=grid)&(p[k,:,None] if i==0 else True)
            axes[i].plot(grid*100,successes.mean(0)*100,label=label,color=colors[k],linewidth=2 if not k else 1.4)
    for i,title in enumerate(['Joint success: presence >= 0.7','Coordinate-only: ignore presence']):
        axes[i].set_title(title);axes[i].set_xlabel('Allowed error / GT ellipse major diameter (%)')
        axes[i].set_ylabel('Success / all 349 occluded points (%)');axes[i].grid(alpha=.2)
        for x in (3,5,10):axes[i].axvline(x,color='gray',linestyle=':',alpha=.5)
        axes[i].legend(fontsize=8);axes[i].set_xlim(0,20);axes[i].set_ylim(0,100)
    fig.suptitle('Fixed predictions: stricter location tolerance and presence effects')
    fig.tight_layout();fig.savefig(OUT/'threshold_curves.png',dpi=150);plt.close(fig)
    lines=['# 自然遮蔽の成功閾値検証（2026-11-18）','',
        '同じ自然遮蔽349点、presence≥0.7固定。数値は成功数。許容誤差はGT顔楕円長径比。再学習・再推論なし。','',
        '| 許容誤差 | 元特徴 | 回帰 | 回帰＋head | DF | DF＋head |','|---|---:|---:|---:|---:|---:|']
    for i,t in enumerate(THRESHOLDS):
        counts=[result['occluded'][name]['thresholds'][i]['joint_correct'] for name in NAMES]
        lines.append('| '+f'{t*100:g}%'+' | '+' | '.join(map(str,counts))+' |')
        print(f'{t*100:g}%',*counts)
    lines+=['','位置だけの判定、可視点、各閾値の動画対応区間、元特徴に対する成功／失敗の入れ替わり、presenceと位置の算術的な寄与分解はsummary.jsonに保存。探索的再集計であり、運用閾値・モデル選択を変更しない。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('10% decomposition:')
    for name in NAMES:
        r=next(x for x in result['occluded'][name]['thresholds'] if x['tolerance']==.1)
        print(name,'joint',r['joint_correct'],'location_only',r['location_only_correct'],'presence',result['occluded'][name]['presence_pass'],
              'loc_effect',r['symmetric_location_contribution_count'],'presence_effect',r['symmetric_presence_contribution_count'])


if __name__=='__main__':main()
