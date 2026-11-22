"""CPU-only paired audit of the feature-MSE ablation and saved norm/g maps."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_messenger_success_thresholds import paired_interval

OUT = ROOT / 'outputs/diagnostics/messenger_position_cosine_20261119'
PREV = ROOT / 'outputs/diagnostics/messenger_position_primary_20261118'
ARMS = ('det_position', 'diff_position')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    assert len(x) and np.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p05=float(np.quantile(x, .05)), p90=float(np.quantile(x, .9)),
                p95=float(np.quantile(x, .95)), min=float(x.min()), max=float(x.max()))


def norm_stats(maps, mask, names, reference):
    result = {name: stats(maps[:, i][mask]) for i, name in enumerate(names)}
    denominator = maps[:, names.index(reference)][mask].astype(float)
    positive = denominator > 0
    result['ratio_reference'] = reference
    result['zero_reference_count'] = int((~positive).sum())
    result['elementwise_ratios'] = {
        name: stats(maps[:, i][mask][positive] / denominator[positive])
        for i, name in enumerate(names) if name != reference
    }
    return result


def main():
    done = read(OUT / 'training_completion.json')
    assert done['all_converged'] and done['source_verified'] and not done['clean_head']
    summary = read(OUT / 'summary.json')
    previous = read(PREV / 'summary.json')
    plan = read(OUT / 'plan.json'); prior = read(PREV / 'plan.json')
    assert {k for k in prior if k != 'hashes' and plan[k] != prior[k]} == {'mse_weight'}
    assert plan['mse_weight'] == 0 and prior['mse_weight'] == .01
    for path, expected in plan['hashes'].items():
        assert sha(ROOT / path) == expected, path
    assert len(summary['checkpoints']) == 8
    for entry in summary['checkpoints']:
        assert sha(ROOT / entry['path']) == entry['sha256']
        assert (ROOT / entry['path']).stat().st_size == entry['bytes']
    prior_completion = read(PREV / 'numeric_completion.json')
    assert prior_completion['all_numeric_evaluations_complete']
    for path, expected in prior_completion['artifacts'].items():
        assert sha(PREV / path) == expected, path
    prior_audit = read(PREV / 'comparison_audit.json')
    for path, expected in prior_audit['source_hashes'].items():
        assert sha(ROOT / path) == expected, path

    audit = {}; hashes = {}; convergence = {}
    for arm in ARMS:
        convergence[arm] = {}
        for stage in ('R', 'g'):
            hist = read(OUT / arm / f'{stage}_history.json')
            end = read(OUT / arm / f'{stage}_convergence.json')
            assert hist[-1]['converged'] and hist[-1]['epoch'] == end['stop_epoch']
            assert min(hist, key=lambda row: row['dev']['position'])['epoch'] == end['selected_epoch']
            convergence[arm][stage] = end
        audit[arm] = {}
        for seed in ((42, 1042, 2042, 3042) if arm.startswith('diff') else (42,)):
            published = summary['conditions'][arm]
            old_published = previous['conditions'][arm]
            if seed != 42:
                published = published['sample_sensitivity'][str(seed)]
                old_published = old_published['sample_sensitivity'][str(seed)]
            domains = {}
            for domain in ('natural', 'artificial'):
                path = OUT / arm / f'{domain}_seed{seed}.npz'
                before_path = PREV / arm / f'{domain}_seed{seed}.npz'
                a = np.load(path); before = np.load(before_path)
                hashes[str(path.relative_to(ROOT))] = sha(path)
                hashes[str(before_path.relative_to(ROOT))] = sha(before_path)
                rows = a['rows']; norms = a['norms']; pn = a['point_norms']; gate = a['gate']
                for key in ('ids', 'videos', 'state' if domain == 'natural' else 'point'):
                    np.testing.assert_array_equal(a[key], before[key])
                np.testing.assert_allclose(rows[:, 0], before['rows'][:, 0], rtol=0, atol=1e-7)
                for value in (rows, norms, pn, gate):
                    assert np.isfinite(value).all()
                assert (norms >= 0).all() and ((gate >= 0) & (gate <= 1)).all()
                masks = ({'occluded': a['state'] == 1, 'visible': a['state'] == 2} if domain == 'natural'
                         else {'covered': a['point'], 'uncovered': ~a['point']})
                names = ('raw', 'full_R', 'learned_g') + (('mask_R',) if domain == 'artificial' else ())
                norm_names = ['X', 'R', 'fused'] if domain == 'natural' else ['X', 'R', 'Z', 'fused']
                np.testing.assert_array_equal(a['norm_order'], norm_names)
                assert rows.shape[1] == len(names) and norms.shape[1:] == (len(norm_names), 16, 16)
                assert pn.shape == (len(rows), len(norm_names), 5) and gate.shape == (len(rows), 5)
                groups = {}
                for group, mask in masks.items():
                    videos = np.broadcast_to(a['videos'][:, None], mask.shape)[mask]
                    methods = {}
                    for i, name in enumerate(names):
                        er = rows[:, i, :, 0][mask].astype(float)
                        pr = rows[:, i, :, 1][mask]
                        old_er = before['rows'][:, i, :, 0][mask].astype(float)
                        old_pr = before['rows'][:, i, :, 1][mask]
                        raw_er = rows[:, 0, :, 0][mask].astype(float)
                        raw_pr = rows[:, 0, :, 1][mask]
                        m = published[domain][name][group]
                        om = old_published[domain][name][group]
                        assert len(er) == m['points']
                        assert abs(er.mean() - m['mean_error']['mean']) < 1e-7
                        assert abs(old_er.mean() - om['mean_error']['mean']) < 1e-7
                        thresholds = {}
                        for t in (.02, .03, .05, .1):
                            joint = (er <= t) & (pr >= .7)
                            old_joint = (old_er <= t) & (old_pr >= .7)
                            raw_joint = (raw_er <= t) & (raw_pr >= .7)
                            assert int(joint.sum()) == m['thresholds'][str(t)]['joint']
                            assert int((er <= t).sum()) == m['thresholds'][str(t)]['location']
                            thresholds[str(t)] = dict(joint=int(joint.sum()), location=int((er <= t).sum()),
                                previous_joint=int(old_joint.sum()),
                                versus_previous=paired_interval(joint.astype(float) - old_joint, videos),
                                versus_raw=paired_interval(joint.astype(float) - raw_joint, videos))
                        methods[name] = dict(error=stats(er), previous_error=stats(old_er),
                            versus_raw=paired_interval(er - raw_er, videos),
                            versus_previous=paired_interval(er - old_er, videos), thresholds=thresholds)
                    g = gate[mask]
                    gstats = stats(g)
                    gstats['fraction_at_least_half'] = float((g >= .5).mean())
                    assert abs(g.mean() - published[domain + '_gate'][group]['mean']) < 1e-7
                    if domain == 'natural' and seed == 42:
                        old_gate_path = PREV / 'batch_precision_audit' / f'{arm}_natural_gate.npz'
                        og = np.load(old_gate_path)
                        np.testing.assert_array_equal(og['ids'], a['ids'])
                        np.testing.assert_array_equal(og['state'], a['state'])
                        gstats['previous'] = stats(og['gate'][mask])
                        gstats['versus_previous'] = paired_interval(g.astype(float) - og['gate'][mask], videos)
                        hashes[str(old_gate_path.relative_to(ROOT))] = sha(old_gate_path)
                    groups[group] = dict(points=int(mask.sum()), videos=len(np.unique(videos)), methods=methods,
                        gate=gstats, point_norms=norm_stats(pn, mask, norm_names, 'X' if domain == 'natural' else 'Z'))
                domains[domain] = dict(groups=groups)
                if domain == 'artificial':
                    ct = a['counts']; comp = a['components']
                    np.testing.assert_array_equal(ct[:, 0], a['covered'].sum((1, 2)))
                    np.testing.assert_array_equal(ct[:, 1], a['outside'].sum((1, 2)))
                    assert not np.any(a['covered'] & a['outside'])
                    np.testing.assert_allclose(comp.sum(-1).reshape(len(rows), 4), a['error_sums'], rtol=2e-6, atol=2e-5)
                    regions = {}
                    for k, region in enumerate(('covered', 'outside')):
                        f = {name: {metric: float(comp[:, i, k, j].sum() / ct[:, k].sum())
                              for j, metric in enumerate(('mse', 'cosine'))} for i, name in enumerate(('raw', 'R'))}
                        for name in f:
                            for metric in f[name]:
                                assert abs(f[name][metric] - published['feature_components'][name][region][metric]) < 1e-7
                        regions[region] = dict(feature=f, norms=norm_stats(norms, a[region], norm_names, 'Z'))
                    for name in ('raw', 'R'):
                        assert abs(sum(regions['covered']['feature'][name].values()) - published['feature'][name]) < 1e-6
                    domains[domain]['regions'] = regions
            audit[arm][str(seed)] = domains

    cross = {}
    for domain in ('natural', 'artificial'):
        a = np.load(OUT / ARMS[0] / f'{domain}_seed42.npz')
        b = np.load(OUT / ARMS[1] / f'{domain}_seed42.npz')
        masks = ({'occluded': a['state'] == 1, 'visible': a['state'] == 2} if domain == 'natural'
                 else {'covered': a['point'], 'uncovered': ~a['point']})
        cross[domain] = {}
        for group, mask in masks.items():
            videos = np.broadcast_to(a['videos'][:, None], mask.shape)[mask]
            cross[domain][group] = {}
            for name, i in [('learned_g', 2)] + ([('mask_R', 3)] if domain == 'artificial' else []):
                x = a['rows'][:, i][mask]; y = b['rows'][:, i][mask]
                cross[domain][group][name] = dict(error=paired_interval(y[:, 0].astype(float) - x[:, 0], videos),
                    success5=paired_interval(((y[:, 0] <= .05) & (y[:, 1] >= .7)).astype(float)
                        - ((x[:, 0] <= .05) & (x[:, 1] >= .7)), videos))
    result = dict(conditions=audit, convergence=convergence, DF_minus_regression=cross, source_hashes=hashes,
        configuration_difference={'mse_weight': [.01, 0.]}, raw_predictions_match_previous=True,
        norm_definition='channel L2; point statistics at containing GT patch; ratios are mean of elementwise ratios, not ratio of means',
        limitations=['same small train-video split, single training seed; exploratory unadjusted 5000 paired video bootstrap',
                     'previous recovery norms unavailable after authorized checkpoint deletion; no before/after norm claim',
                     'natural R/X measures magnitude change, not accuracy against unoccluded ground truth',
                     'g is a soft mixing coefficient; its mean is not a binary adoption percentage'])
    (OUT / 'comparison_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 特徴MSE除外の比較結果', '',
        '位置主損失を維持し、特徴補助を0.01×cosine距離に変更。他の学習設定は前回と同一。', '',
        '| 条件 | 自然遮蔽平均誤差 | 5%成功 | 10%成功 | 可視平均誤差 | 遮蔽点平均g |',
        '|---|---:|---:|---:|---:|---:|']
    raw = summary['conditions'][ARMS[0]]['natural']['raw']
    def table_row(label, d, g):
        m = d['occluded']; v = d['visible']
        return f"| {label} | {100*m['mean_error']['mean']:.5f}% | {m['thresholds']['0.05']['joint']} | {m['thresholds']['0.1']['joint']} | {100*v['mean_error']['mean']:.5f}% | {g} |"
    lines.append(table_row('元特徴', raw, '—'))
    for arm, label in zip(ARMS, ('回帰', 'DF')):
        groups = audit[arm]['42']['natural']['groups']
        lines.append(table_row(label + '・前回MSEあり', previous['conditions'][arm]['natural']['learned_g'], f"{100*groups['occluded']['gate']['previous']['mean']:.3f}%"))
        lines.append(table_row(label + '・今回MSEなし', summary['conditions'][arm]['natural']['learned_g'], f"{100*groups['occluded']['gate']['mean']:.3f}%"))
    lines += ['', '自然遮蔽349点／可視1852点。誤差はGT顔楕円長径比、成功にはpresence≥0.7も要求。gは連続的な混合係数。', '',
        '| 条件 | R停止/選択 | g停止/選択 | 人工mask_R位置誤差 | 人工MSE | 人工cosine距離 | 遮蔽patch R/Zノルム比平均 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS, ('回帰', 'DF')):
        r = summary['conditions'][arm]; region = audit[arm]['42']['artificial']['regions']['covered']
        lines.append(f"| {label} | {r['R']['stop_epoch']}/{r['R']['selected_epoch']} | {r['g']['stop_epoch']}/{r['g']['selected_epoch']} | {100*r['artificial']['mask_R']['covered']['mean_error']['mean']:.5f}% | {region['feature']['R']['mse']:.6f} | {region['feature']['R']['cosine']:.6f} | {region['norms']['elementwise_ratios']['R']['mean']:.5f} |")
    lines += ['', '同一小規模train内分割の検証plateauまで実行。本学習・独立val/test・複数学習seedの比較ではない。正解特徴をheadへ渡す比較なし。',
        '全8重みのSHA、実行ソース不変、前回との差がMSE係数だけ、元特徴の推論一致、最良epoch選択、数値の独立再集計を確認。',
        '前回Rノルムは未保存・重み削除済みのため比較不能。自然遮蔽のR/Xノルム比は振幅の変化であり正解への近さではない。',
        '詳細な対応動画区間・ノルム分布・g分布・生成seed別値はcomparison_audit.jsonに保存。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, color in zip(ARMS, ('#2563eb', '#dc2626')):
        for i, stage in enumerate(('R', 'g')):
            for directory, suffix, style in ((PREV, 'MSE+cos', '--'), (OUT, 'cos', '-')):
                h = read(directory / arm / f'{stage}_history.json')
                axes[i].plot([x['epoch'] for x in h], [100*x['dev']['position'] for x in h],
                             label=f'{arm}: {suffix}', color=color, linestyle=style, linewidth=1.2)
    for ax, title in zip(axes, ('Recovery: endpoint dev position', 'Natural fusion: dev position')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('Balanced position error (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'learning_curves.png', dpi=140); plt.close(fig)
    artifact_names = ('summary.json', 'comparison_audit.json', 'report.md', 'learning_curves.png', 'training_completion.json', 'plan.json', 'teacher_audit.json', 'technical_checks.json')
    complete = dict(all_numeric_evaluations_complete=True, weights_retained=True, checkpoint_count=8,
        checkpoint_bytes=summary['checkpoint_bytes'], artifacts={name: sha(OUT / name) for name in artifact_names},
        report_source_sha256=sha(Path(__file__)), no_clean_head_comparison=True)
    (OUT / 'numeric_completion.json').write_text(json.dumps(complete, indent=2) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
