"""Paired video-bootstrap summaries for the bounded nine-arm comparison."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/diagnostics/messenger_multiaxis_20261114'


def interval(success,reference,state,videos,label):
    groups=np.unique(videos);num=[];den=[]
    for v in groups:
        mask=(videos==v)[:,None] & (state==label)
        num.append(float(((success.astype(float)-reference)*mask).sum()))
        den.append(int(mask.sum()))
    num=np.array(num);den=np.array(den)
    rng=np.random.default_rng(42);ids=rng.integers(0,len(groups),size=(2000,len(groups)))
    n=den[ids].sum(1);valid=n>0
    vals=num[ids].sum(1)[valid]/n[valid]
    return dict(difference_pp=float(num.sum()/den.sum()*100),
                video_bootstrap_95_pp=(np.quantile(vals,[.025,.975])*100).tolist(),
                points=int(den.sum()),videos=int((den>0).sum()))


def main():
    assert json.loads((OUT/'status.json').read_text())['phase']=='completed'
    comparison=json.loads((OUT/'comparison.json').read_text())
    summary={};deleted=0
    for name,metric in comparison.items():
        p=np.load(OUT/name/'natural_points.npz');a=np.load(OUT/(name.rsplit('_',1)[0]+'_A')/'natural_points.npz')
        assert np.array_equal(p['ids'],a['ids']) and np.array_equal(p['state'],a['state'])
        rows=p['rows'];ok=(rows[...,0]<=.1)&(rows[...,1]>=.7)
        ar=a['rows'];aok=(ar[:,2,:,0]<=.1)&(ar[:,2,:,1]>=.7)
        st=p['state'];videos=p['videos'];nat=metric['learned_gate']
        summary[name]=dict(selected_epoch=metric['selected_epoch'],parameters=metric['parameters'],seconds=metric['seconds'],
             artificial_fused_error=metric['synthetic']['fused'],artificial_raw_error=metric['synthetic']['raw'],
             artificial_branch_errors=metric['synthetic']['branches'],
             full_recovery=metric['full_recovery'],raw=metric['raw'],
             occluded_correct=nat['occluded']['correct'],occluded_points=nat['occluded']['points'],
             visible_correct=nat['visible']['correct'],visible_points=nat['visible']['points'],
             occluded_mean_error=nat['occluded']['mean_error'],visible_mean_error=nat['visible']['mean_error'],
             learned_vs_raw={key:interval(ok[:,2],ok[:,0],st,videos,s) for key,s in [('occluded',1),('visible',2)]},
             learned_vs_family_A={key:interval(ok[:,2],aok,st,videos,s) for key,s in [('occluded',1),('visible',2)]})
        deletion=json.loads((OUT/name/'checkpoint_deletion.json').read_text());assert deletion['phase']=='completed'
        deleted+=sum(v['bytes'] for v in deletion['files'])
    report=dict(conditions=summary,deleted_checkpoint_bytes=deleted,remaining_checkpoints=[str(p) for p in OUT.rglob('*.pt')],
                limitations=['single seed, bounded 4-epoch comparison','all selected R at final epoch and still improving: not convergence evidence',
                 '3 vs 9 branches not capacity-matched','train-internal GT crop; historical backbone train exposure',
                 'bootstrap intervals unadjusted for nine comparisons; no val/test or production promotion'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for name,m in summary.items():
        print(name,round(m['artificial_fused_error'],5),f'{m["occluded_correct"]}/{m["occluded_points"]}',
              f'{m["visible_correct"]}/{m["visible_points"]}',m['learned_vs_raw']['occluded'])
    print('deleted checkpoint MiB',deleted/1024**2)


if __name__=='__main__':main()
