#!/usr/bin/env python3
"""Summarize a completed O diagnostic; no inference or threshold selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    d=json.loads((output/'metrics.json').read_text());marker=json.loads((output/'diagnostic_complete.json').read_text())
    assert marker['metrics_sha256']==sha(output/'metrics.json')
    assert marker['points_sha256']==sha(output/'points.jsonl')
    rows=[json.loads(s) for s in (output/'points.jsonl').read_text().splitlines()]
    assert len({(r['annotation_id'],r['slot']) for r in rows})==len(rows)
    assert all(not r['m'] or r['inside_support'] for r in rows)
    assert all(r['coefficient']==0 and r['relative_feature_change']==0 for r in rows if not r['m'])
    assert d['counts']['crops']==768
    n=d['natural_point_detection']['all_in_crop']
    assert n['count']==len(rows) and n['tp']+n['fn']==sum(r['state']==1 for r in rows)
    details={}
    for state in [1,2]:
        x=[r for r in rows if r['state']==state];detected=[r for r in x if r['m']]
        details[f'state{state}']=dict(points=len(x),O_positive=len(detected),
            g_below_001=sum(r['g']<.01 for r in x),O_positive_g_below_001=sum(r['g']<.01 for r in detected),
            O_positive_mean_g=float(np.mean([r['g'] for r in detected])) if detected else None,
            coefficient_below_0001=sum(r['coefficient']<.001 for r in x))
    expected=json.loads((ROOT/'outputs/landmark_detection/messenger_geometry_prior/k3/training_success.json').read_text())['best_value']
    actual=d['conditional_five_slot']['actual']['conditional_semantic_macro_f1_nme_0.10']
    summary=dict(invariants='passed',details=details,historical_cuda_conditional_f1=expected,current_cpu_conditional_f1=actual,
                 difference_percentage_points=100*(actual-expected),reporter_sha256=sha(__file__))
    (output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    pc=lambda v:'N/A' if v is None else f'{100*v:.2f}%'
    lines=['# 現行k=3の遮蔽判定O：Validation診断','',
        '固定canonical-v2 Val、HEAD-free FACE/ellipse → predicted crop DINO + frozen observer → Messenger prior fusion → Query。',
        '新規学習・Test使用・閾値変更なし。既存k=3 best EMAをCPU FP32で推論。現行V3 baselineの評価ではない。','',
        '## 対象と分母','',
        f"- 全768予測FACE crop。FACE未対応は{d['missing_FACE']['faces']}顔、state別点数={d['missing_FACE']['state_counts']}。",
        f"- O診断: crop内state1={n['tp']+n['fn']}点、state2={n['fp']+n['tn']}点。state0は除外。",
        f"- crop外: state1={d['counts'].get('state1_crop_outside',0)}、state2={d['counts'].get('state2_crop_outside',0)}。",
        f"- crop内だが予測楕円support外: state1={d['counts'].get('state1_outside_ellipse_support',0)}、state2={d['counts'].get('state2_outside_ellipse_support',0)}。Oはこの領域で常にfalse。",
        '- 自然点位置の疎な検知評価。密な遮蔽物maskのIoUではない。APは順位診断で、閾値の選び直しはしていない。','',
        '## 自然遮蔽のO検知','',
        '| 集合 | TP | FP | FN | TN | Precision | Recall | 可視点FPR | AP | AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,m in d['natural_point_detection'].items():
        lines.append(f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {pc(m['precision'])} | {pc(m['recall'])} | {pc(m['visible_false_positive_rate'])} | {pc(m['average_precision'])} | {pc(m['auroc'])} |")
    lines+=['','## 採用係数と実際の特徴変更','',
        '| 集合 | g平均 | g中央値 | 有効係数平均 | 特徴変更量/元特徴norm 平均 | g<0.01 |',
        '|---|---:|---:|---:|---:|---:|']
    for state in [1,2]:
        p=d['natural_distributions'];g=p[f'state{state}/g'];co=p[f'state{state}/coefficient'];de=p[f'state{state}/relative_feature_change']
        lines.append(f"| state{state} | {g['mean']:.8f} | {g['p50']:.8f} | {co['mean']:.8f} | {de['mean']:.8f} | {details[f'state{state}']['g_below_001']}/{g['count']} |")
    lines+=['','有効係数はchannel平均のm×g×sigmoid(a)。gも有効係数も遮蔽確率ではない。','',
        '## 同じcheckpointでの介入診断','',
        '| 推論モード | 条件付き5slot Macro F1 NME0.10 | 自然遮蔽GT | 出力 | 位置成功0.10 | 可視点成功0.10 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode,m in d['conditional_five_slot'].items():
        occ=d['natural_location_counts'][mode+'/state1'];vis=d['natural_location_counts'][mode+'/state2']
        lines.append(f"| {mode} | {pc(m['conditional_semantic_macro_f1_nme_0.10'])} | {occ['gt']} | {occ['emitted']} | {occ['tp010']} | {vis['tp010']} |")
    lines+=['','rawは復元差分なし。mask_all_supportはOの判定だけを予測楕円全体で有効化し、gとchannel係数を保持。',
        'これらは再学習した対照群ではなく、採用・モデル選択には使わない。',
        f'既存CUDA validation={expected:.9f}、今回CPU={actual:.9f}。差={100*(actual-expected):+.5f}ポイント。','',
        '## 人工遮蔽：既知maskとの比較','',
        f"全5点visibleの候補{d['artificial_available_clean_crops']} cropからseed42のhash順で{d['artificial_requested_crops']} cropを選択。",
        f"3形状で合成成功{d['counts']['artificial_applied']}件、見送り{d['counts'].get('artificial_skipped',0)}件。輪郭非遮蔽は元Head注釈から認証できない。",
        'any_overlapはpatch内1pixel以上のpaste、majority_overlapは50%以上。評価領域は予測楕円support内。','',
        '| 形状/定義 | Precision | Recall | 非対象patch FPR | AP |',
        '|---|---:|---:|---:|---:|']
    for label,m in d['artificial_detection'].items():
        lines.append(f"| {label} | {pc(m['precision'])} | {pc(m['recall'])} | {pc(m['visible_false_positive_rate'])} | {pc(m['average_precision'])} |")
    lines+=['','| 形状 | X cosine距離 | R | Y | 既知maskでのY | R誤差相対低下 | Y誤差相対低下 | 人工領域g平均 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    f=d['artificial_features']
    for kind in ['rectangle','irregular','boundary']:
        vals=[f[f'{kind}/occluded/{b}/cosine_distance']['mean'] for b in ['X','R','Y','known_mask_Y']]
        lines.append(f"| {kind} | "+' | '.join(f'{v:.6f}' for v in vals)+f" | {pc((vals[0]-vals[1])/vals[0])} | {100*(vals[0]-vals[2])/vals[0]:.5f}% | {f[kind+'/occluded/g']['mean']:.6f} |")
    lines+=['','X/R/Yは固定observerの合成前特徴との比較。known_mask_Yは既知の追加遮蔽maskへ置換するだけでgを維持。',
        '人工遮蔽の再現精度を自然遮蔽の正解特徴復元と読み替えない。自然点と人工patchは異なる母集団。','',
        '## 診断上の判断','',
        '- Oには自然遮蔽との弱い関連があるが、約半数を見逃し可視点にも多く反応する。単なる無情報ではないが、遮蔽専用の判定器とはいえない。',
        '- crop内の見逃し218点のうち39点は予測楕円support外、179点はsupport内の異常判定不成立。境界・supportとスコア判定を分けて改善する必要がある。',
        '- 自然遮蔽433点すべてでg<0.01。復元を無効にしても、Oだけを全supportに広げても、今回の5slot F1と自然遮蔽位置成功数は変わらない。',
        '- 人工遮蔽で復元候補Rは教師特徴へのcosine距離を約40〜43%下げるが、通常融合Yの低下は0.004%未満。既知maskへの置換だけでもほぼ改善が伝わらない。',
        '- Oの誤検知・見落としに加え、採用係数の小ささが直接観測されたボトルネック。gを強制的に上げることの有効性や自然遮蔽の復元正解は未検証。',
        '- k3と対照群の過去の性能差を、現checkpointの推論時融合の効果としては説明できない。学習中の影響はこの介入診断からは分からない。',
        '- 新しい敵対的augmentationへ進む前に、gの教師・損失・勾配経路と、人工遮蔽でもO recallが低い理由を確認する価値が高い。正式ゲートの合否やbaseline昇格判定ではない。','',
        '## 再現性と制限','',
        '- metrics.json: checkpoint・prior・config・注釈・cache・入力画像・コードhash、全件数、PR曲線。',
        '- points.jsonl: 自然点ごとのO、g、係数、特徴変化、presence、NME、slot、動画、参照center。',
        '- artificial_manifest.json: 対象annotation、形状、seed、合成結果。',
        '- verification.json: 重複・マスク外変更なし・集計分母・hashの整合確認。',
        f"- 実行時間{d['runtime']['elapsed_seconds']:.1f}秒、CPU {d['runtime']['threads']}スレッド。既存GPU学習を停止・変更していない。",
        '- 学習当時から変更された周辺ソースはmetrics.jsonのsource_driftに記録。prior/融合の中心モジュールは学習時hash一致。',
        '- 単一checkpoint。動画内・同じpatchの点は相関する。統計的有意差やV3への改善は主張しない。','',
        '![診断図](observer_diagnostic.png)','']
    (output/'report.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    ax=axes[0];curve=n['pr_curve'];ax.plot([c['recall'] for c in curve],[c['precision'] for c in curve],label=f"Natural AP={n['average_precision']:.3f}")
    ax.axhline(n['prevalence'],color='gray',ls='--',label=f"Prevalence={n['prevalence']:.3f}");ax.scatter([n['recall']],[n['precision']],color='red',label='Current O threshold');ax.set(xlabel='Recall',ylabel='Precision',title='Natural occluded vs visible points',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    ax=axes[1]
    for state,label in [(1,'Natural occluded'),(2,'Visible')]:
        values=np.array([r['g'] for r in rows if r['state']==state]);ax.hist(np.log10(np.maximum(values,1e-12)),bins=35,density=True,histtype='step',label=label)
    ax.set(xlabel='log10(g)',ylabel='Density',title='Adoption gate on natural points');ax.legend(fontsize=8)
    ax=axes[2];labels=['rectangle','irregular','boundary'];x=np.arange(3)
    for j,branch in enumerate(['X','R','Y','known_mask_Y']):
        ax.bar(x+(j-1.5)*.2,[f[f'{kind}/occluded/{branch}/cosine_distance']['mean'] for kind in labels],width=.2,label=branch)
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean cosine distance (lower better)',title='Artificial occlusion feature error');ax.legend(fontsize=8)
    fig.savefig(output/'observer_diagnostic.png',dpi=170);plt.close(fig)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    source=ROOT/'scripts/diagnose_messenger_occlusion_observer.py'
    assert sha(source)==d['provenance']['script_sha256']
    shutil.copy2(source,snapshot/source.name);shutil.copy2(__file__,snapshot/Path(__file__).name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
