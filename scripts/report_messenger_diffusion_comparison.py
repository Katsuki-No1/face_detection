"""Final paired comparisons and source/checkpoint audit, no model updates."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_messenger_local_recovery import boot
OUT=ROOT/'outputs/diagnostics/messenger_diffusion_comparison_20261114'
ARMS=('det_feature','det_head','diff_feature','diff_head')


def read(path):return json.loads(path.read_text())


def feature_difference(a,b):
    for key in ('ids','videos','counts'):assert np.array_equal(a[key],b[key])
    sums=[];counts=[]
    for v in np.unique(a['videos']):
        m=a['videos']==v;sums.append((a['error_sums'][m,2]-b['error_sums'][m,2]).sum());counts.append(a['counts'][m,0].sum())
    sums=np.array(sums);counts=np.array(counts);ix=np.random.default_rng(42).integers(0,len(sums),(2000,len(sums)))
    d=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(difference=float(sums.sum()/counts.sum()),video_bootstrap_95=np.quantile(d,[.025,.975]).tolist())


def natural_difference(a,b):
    for key in ('ids','videos','state'):assert np.array_equal(a[key],b[key])
    ar=a['rows'][:,2];br=b['rows'][:,2];ok=lambda x:((x[...,0]<=.1)&(x[...,1]>=.7)).astype(float)
    return {label:dict(success=boot(ok(ar)-ok(br),a['state']==s,a['videos']),
        mean_error=boot(ar[...,0]-br[...,0],a['state']==s,a['videos'])) for s,label in [(1,'occluded'),(2,'visible')]}


def main():
    assert read(OUT/'status.json')['phase']=='completed'
    summary=read(OUT/'summary.json');conditions=summary['conditions'];assert set(conditions)==set(ARMS)
    assert len(summary['checkpoints'])==16
    for f in summary['checkpoints']:
        path=ROOT/f['path'];assert path.stat().st_size==f['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256']
    nat={arm:np.load(OUT/arm/'natural_seed42.npz') for arm in ARMS}
    art={arm:np.load(OUT/arm/'artificial_seed42.npz') for arm in ARMS}
    comparisons={};versus_raw={};distributions={}
    for arm in ARMS:
        p=nat[arm];g=p['rows'][:,2];raw=p['rows'][:,0]
        ok=lambda x:((x[...,0]<=.1)&(x[...,1]>=.7)).astype(float)
        versus_raw[arm]={label:dict(success=boot(ok(g)-ok(raw),p['state']==s,p['videos']),
            mean_error=boot(g[...,0]-raw[...,0],p['state']==s,p['videos'])) for s,label in [(1,'occluded'),(2,'visible')]}
        distributions[arm]={}
        for i,name in [(0,'raw'),(2,'learned_g')]:
            distributions[arm][name]={}
            for s,label in [(1,'occluded'),(2,'visible')]:
                rows=p['rows'][:,i][p['state']==s]
                distributions[arm][name][label]=dict(median_error=float(np.median(rows[:,0])),
                    p90_error=float(np.quantile(rows[:,0],.9)),p95_error=float(np.quantile(rows[:,0],.95)),
                    presence_and_error_under_5pct=int(((rows[:,1]>=.7)&(rows[:,0]<=.05)).sum()))
    for first,second in [('det_head','det_feature'),('diff_head','diff_feature'),('diff_feature','det_feature'),('diff_head','det_head')]:
        comparisons[f'{first}_minus_{second}']=dict(feature=feature_difference(art[first],art[second]),
            natural=natural_difference(nat[first],nat[second]))
    sensitivity={}
    precision=read(OUT/'sampling_precision_audit.json')['conditions']
    for arm in ('diff_feature','diff_head'):
        m=conditions[arm];trials=[m,*m['sample_sensitivity'].values()]
        sensitivity[arm]=dict(seeds=[42,*map(int,m['sample_sensitivity'])],
            feature=[v['feature']['R']['covered'] for v in trials],
            natural_occluded_correct=[precision[arm][str(seed)]['half_cache']['occluded']['correct'] for seed in [42,1042,2042,3042]],
            natural_occluded_mean_error=[precision[arm][str(seed)]['half_cache']['occluded']['mean_error'] for seed in [42,1042,2042,3042]],
            candidate_precision='uniform fp16 cache, separately audited')
    audit=dict(all_four_converged=all(m['R']['converged'] and m['g']['converged'] for m in conditions.values()),
        all_16_checkpoint_hashes_verified=True,checkpoint_bytes=summary['checkpoint_bytes'],
        primary_seed_fixed_before_training=42,paired_comparisons=comparisons,versus_raw=versus_raw,sample_sensitivity=sensitivity,
        exploratory_position_distributions=distributions)
    assert audit['all_four_converged'];(OUT/'comparison_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for arm in ARMS:
        r=read(OUT/arm/'R_history.json');g=read(OUT/arm/'g_history.json')
        axes[0].plot([h['epoch'] for h in r[1:]],[h['dev']['feature'] for h in r[1:]],label=arm)
        axes[1].plot([h['epoch'] for h in r],[h['dev']['position'] for h in r],label=arm)
        axes[2].plot([h['epoch'] for h in g],[h['dev']['soft_bce'] for h in g],label=arm)
    for ax,title in zip(axes,['Recovery validation feature error','Recovery validation head position loss','Natural-gate validation soft BCE']):
        ax.set_title(title);ax.set_xlabel('epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=150)
    lines=['# 4条件の収束比較結果','','全条件で自然gは以前の遮蔽領域の弱い教師を使用。全16重みを保持。',
        '', '| 条件 | R 停止/最良 | g 停止/最良 | 特徴誤差 | GT点パッチ誤差 | 人工g成功 /663 | 自然遮蔽g成功 /349 | 自然可視g成功 /1852 | 自然遮蔽平均誤差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in conditions.items():
        n=m['natural']['learned_g'];r=m['R'];g=m['g'];f=m['feature']['R']
        lines.append(f"| {arm} | {r['stop_epoch']}/{r['selected_epoch']} | {g['stop_epoch']}/{g['selected_epoch']} | {f['covered']:.5f} | {f['local_center']['mean']:.5f} | {m['artificial']['learned_g']['correct']} | {n['occluded']['correct']} | {n['visible']['correct']} | {100*n['occluded']['mean_error']['mean']:.3f}% |")
        print(arm,'R',r,'g',g,'feature',f['covered'],'natural',n['occluded']['correct'],n['visible']['correct'])
    lines+=['','同一人工fit256顔／自然fit512顔の単一seed比較。自然遮蔽の正解特徴はなく、自然の評価はhead位置とpresence。特徴誤差はcosine＋MSE。平均位置誤差は全対象点のGT楕円長径比でpresenceによる除外なし。成功はpresence≥0.7かつ誤差≤10%。',
        '', '停止は検証plateauであり大域最適の保証ではない。評価動画の再利用、4条件の探索的・未調整比較、Diffusionの追加パラメータと16回denoising費用を含む。拡散の複数seedをGTで選択せず、comparison_audit.jsonに全結果を記録。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
