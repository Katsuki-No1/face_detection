"""Summarize paired local-feature diagnostics without training or GPU work."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_messenger_local_recovery import OUT,ARMS,boot


def main():
    summary=json.loads((OUT/'summary.json').read_text());derived={}
    for arm in ARMS:
        a=np.load(OUT/f'{arm}.npz');p=a['point'];v=a['videos'];h=a['head'];local=a['local']
        improved=local[:,1,:,:2].sum(-1)<local[:,0,:,:2].sum(-1)
        worse=h[:,1,:,0]>h[:,0,:,0]
        pair={name:boot(h[:,i,:,0]-h[:,0,:,0],p,v) for i,name in enumerate(summary['variants'])}
        sums=[]
        for video in np.unique(v):
            m=(v==video)[:,None]&p
            sums.append([a['spatial_energy'][:,k][m].sum() for k in (1,2)])
        sums=np.array(sums);ix=np.random.default_rng(42).integers(0,len(sums),(2000,len(sums)))
        ratios=sums[ix,:,].sum(1);ratios=ratios[:,0]/ratios[:,1]
        derived[arm]=dict(paired_head_mean_error_difference=pair,
            local_energy_R_over_clean=dict(ratio=float(sums[:,0].sum()/sums[:,1].sum()),video_bootstrap_95=np.quantile(ratios,[.025,.975]).tolist()),
            improved_center_features=int(improved[p].sum()),improved_features_but_full_R_position_worse=int((improved&worse&p).sum()),
            points=int(p.sum()))
    (OUT/'paired_analysis.json').write_text(json.dumps(derived,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    colors=['#64748b','#0284c7','#d97706']
    names=['Original','Scale C','Level+scale C']
    scale=summary['conditions']['scale_C'];level=summary['conditions']['level_scale_C']
    vals=[scale['local']['raw']['center_combined']['mean'],scale['local']['R']['center_combined']['mean'],level['local']['R']['center_combined']['mean']]
    axes[0].bar(names,vals,color=colors);axes[0].set_ylabel('Cosine distance + MSE (lower is better)')
    axes[0].set_title('Feature error at covered keypoint tokens')
    for i,x in enumerate(vals):axes[0].text(i,x+.006,f'{x:.3f}',ha='center')
    vals=[1,derived['scale_C']['local_energy_R_over_clean']['ratio'],derived['level_scale_C']['local_energy_R_over_clean']['ratio']]
    axes[1].bar(['Clean target',*names[1:]],vals,color=colors);axes[1].set_ylim(0,1.15)
    axes[1].set_title('3x3 local feature variation / clean');axes[1].set_ylabel('Ratio (not accuracy)')
    for i,x in enumerate(vals):axes[1].text(i,x+.02,f'{x:.2f}',ha='center')
    labels=['Original','Full R','R inside mask','Clean inside mask']
    keys=['raw','full_R','oracle_region_R','oracle_region_clean'];x=np.arange(4)
    for i,(name,m) in enumerate([('Scale C',scale),('Level+scale C',level)]):
        vals=[m['head'][k]['correct'] for k in keys]
        bars=axes[2].bar(x+(i-.5)*.36,vals,.36,label=name,color=colors[i+1])
        axes[2].bar_label(bars,padding=2,fontsize=8)
    axes[2].set_xticks(x,labels,rotation=18);axes[2].set_ylim(600,670)
    axes[2].set_ylabel('Successful covered keypoints / 663');axes[2].set_title('Fixed-head intervention (axis starts at 600)');axes[2].legend(loc='lower right')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Artificial occlusion: 128 faces, 384 pairs, 35 videos; reused evaluation, frozen models')
    fig.tight_layout();fig.savefig(OUT/'local_recovery_diagnostic.png',dpi=150)
    print(json.dumps(derived,indent=2))


if __name__=='__main__':main()
