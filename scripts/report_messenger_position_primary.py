"""Independent CPU paired evaluation and retained-checkpoint audit."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_messenger_success_thresholds import paired_interval
OUT=ROOT/'outputs/diagnostics/messenger_position_primary_20261118'
ARMS=('det_position','diff_position')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    completion=read(OUT/'training_completion.json');assert completion['all_converged'] and not completion['clean_head']
    precision=read(OUT/'batch_precision_audit/summary.json')
    assert precision['frozen_parent_unchanged'] and not precision['clean_head']
    summary=read(OUT/'summary.json');hashes={};audit={}
    assert len(summary['checkpoints'])==8
    for entry in summary['checkpoints']:
        assert sha(ROOT/entry['path'])==entry['sha256']
    for name,h in read(OUT/'plan.json')['hashes'].items():assert sha(ROOT/name)==h
    for arm in ARMS:
        for stage in ('R','g'):
            history=read(OUT/arm/f'{stage}_history.json');end=read(OUT/arm/f'{stage}_convergence.json')
            assert history[-1]['converged'] and history[-1]['epoch']==end['stop_epoch']
            best=min(history,key=lambda r:r['dev']['position'])
            assert best['epoch']==end['selected_epoch']
        audit[arm]={}
        for seed in ((42,1042,2042,3042) if arm.startswith('diff') else (42,)):
            entry=summary['conditions'][arm] if seed==42 else summary['conditions'][arm]['sample_sensitivity'][str(seed)]
            audit[arm][str(seed)]={}
            for domain in ('natural','artificial'):
                path=OUT/arm/f'{domain}_seed{seed}.npz';a=np.load(path);hashes[str(path.relative_to(ROOT))]=sha(path)
                rows=a['rows']; assert np.isfinite(rows).all()
                masks={'occluded':a['state']==1,'visible':a['state']==2} if domain=='natural' else {'covered':a['point'],'uncovered':~a['point']}
                assert rows.shape[1]==(3 if domain=='natural' else 4)
                names=('raw','full_R','learned_g') if domain=='natural' else ('raw','full_R','learned_g','mask_R')
                oldpath=ROOT/'outputs/diagnostics/messenger_diffusion_comparison_20261114/det_feature'/f'{domain}_seed42.npz'
                old=np.load(oldpath)
                for key in ('ids','videos','state' if domain=='natural' else 'point'):np.testing.assert_array_equal(a[key],old[key])
                # Exact current-arm parity; historical batching is separately reproduced on GPU.
                current=np.load(OUT/'det_position'/f'{domain}_seed42.npz')
                np.testing.assert_allclose(rows[:,0],current['rows'][:,0],rtol=0,atol=1e-7)
                assert precision['batch_precision'][domain]['batch4_reproduces_current']
                assert precision['batch_precision'][domain]['batch16_reproduces_historical']
                groups={}
                for group,mask in masks.items():
                    videos=np.broadcast_to(a['videos'][:,None],mask.shape)[mask];methods={}
                    for i,name in enumerate(names):
                        er=rows[:,i,:,0][mask];pr=rows[:,i,:,1][mask];raw=rows[:,0,:,0][mask];rp=rows[:,0,:,1][mask]
                        published=entry[domain][name][group]
                        assert len(er)==published['points'] and abs(float(er.mean())-published['mean_error']['mean'])<1e-7
                        methods[name]=dict(mean_error=float(er.mean()),median=float(np.median(er)),p90=float(np.quantile(er,.9)),
                            error_difference=paired_interval(er.astype(float)-raw,videos),thresholds={})
                        for t in (.02,.03,.05,.1):
                            joint=(er<=t)&(pr>=.7);base=(raw<=t)&(rp>=.7)
                            assert int(joint.sum())==published['thresholds'][str(t)]['joint']
                            methods[name]['thresholds'][str(t)]=dict(joint=int(joint.sum()),location=int((er<=t).sum()),
                                difference=paired_interval(joint.astype(float)-base,videos))
                    groups[group]=dict(points=int(mask.sum()),methods=methods)
                audit[arm][str(seed)][domain]=groups
    cross={}
    for domain in ('natural','artificial'):
        aa=np.load(OUT/ARMS[0]/f'{domain}_seed42.npz');bb=np.load(OUT/ARMS[1]/f'{domain}_seed42.npz')
        masks={'occluded':aa['state']==1,'visible':aa['state']==2} if domain=='natural' else {'covered':aa['point'],'uncovered':~aa['point']}
        cross[domain]={}
        for label,mask in masks.items():
            videos=np.broadcast_to(aa['videos'][:,None],mask.shape)[mask]
            cross[domain][label]={}
            for mode,i in [('learned_g',2)]+([('mask_R',3)] if domain=='artificial' else []):
                a=aa['rows'][:,i][mask];b=bb['rows'][:,i][mask]
                cross[domain][label][mode]=dict(error=paired_interval(b[:,0].astype(float)-a[:,0],videos),
                    success5=paired_interval(((b[:,0]<=.05)&(b[:,1]>=.7)).astype(float)-((a[:,0]<=.05)&(a[:,1]>=.7)),videos))
    result=dict(within_arm=audit,DF_minus_regression=cross,source_hashes=hashes,precision_and_gate_use=precision,
                limitations='small reused train-derived split, one training seed; 5000 paired video resamples, unadjusted exploratory intervals')
    (OUT/'comparison_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 位置主目的の回帰・DF比較結果','','同一小規模train内分割。位置誤差はGT顔楕円長径比。成功はpresence>=0.7も要求。正解特徴head比較なし。','',
        '| 条件 | R停止 / 選択epoch | g停止 / 選択epoch | 人工特徴誤差 | 自然遮蔽平均誤差 | 自然遮蔽5% / 10%成功 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r=summary['conditions'][arm];m=r['natural']['learned_g']['occluded']
        lines.append(f"| {arm} | {r['R']['stop_epoch']} / {r['R']['selected_epoch']} | {r['g']['stop_epoch']} / {r['g']['selected_epoch']} | {r['feature']['R']:.6f} | {m['mean_error']['mean']*100:.4f}% | {m['thresholds']['0.05']['joint']} / {m['thresholds']['0.1']['joint']} |")
    lines+=['','詳細な人工・可視・全置換・生成seed別数値はsummary.json、対応動画区間はcomparison_audit.json。全8重みのSHAと最良位置epochを確認。元特徴の微小差は推論batch4対16でそれぞれ再現。今回2条件は共通batch4で一致。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in ARMS:
        for i,stage in enumerate(('R','g')):
            h=read(OUT/arm/f'{stage}_history.json');axes[i].plot([r['epoch'] for r in h],[r['dev']['position']*100 for r in h],label=arm)
    for ax,title in zip(axes,('Recovery: sampled endpoint position','Natural fusion: position')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Balanced normalized coordinate error (%)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=140);plt.close(fig)
    artifacts={name:sha(OUT/name) for name in ('summary.json','comparison_audit.json','report.md','learning_curves.png','batch_precision_audit/summary.json')}
    (OUT/'numeric_completion.json').write_text(json.dumps(dict(all_numeric_evaluations_complete=True,weights_retained=True,
        checkpoint_count=8,checkpoint_bytes=summary['checkpoint_bytes'],artifacts=artifacts),indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
