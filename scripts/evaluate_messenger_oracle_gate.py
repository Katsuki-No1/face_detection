"""CPU reanalysis of saved artificial-occlusion oracle-mask head interventions."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/diagnostics/messenger_diffusion_comparison_20261114'
OUT = ROOT / 'outputs/diagnostics/messenger_oracle_gate_20261118'
ARMS = ('det_feature', 'det_head', 'diff_feature', 'diff_head')
MODES = ('raw', 'full_R', 'learned_g', 'mask_R', 'mask_clean', 'clean')
TOLERANCES = (.02, .03, .05, .1)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def interval(delta, videos):
    unique = np.unique(videos)
    sums = np.array([delta[videos == v].sum() for v in unique])
    counts = np.array([(videos == v).sum() for v in unique])
    ix = np.random.default_rng(42).integers(0, len(unique), (5000, len(unique)))
    draws = sums[ix].sum(1) / counts[ix].sum(1)
    return dict(mean=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(), videos=len(unique))


def summarize(rows, videos):
    result = {}
    for i, mode in enumerate(MODES):
        error = rows[i, :, 0].astype(float)
        present = rows[i, :, 1] >= .7
        stats = dict(points=len(error), mean_error=float(error.mean()), median_error=float(np.median(error)),
                     p90_error=float(np.quantile(error, .9)), presence_pass=int(present.sum()), thresholds={})
        for t in TOLERANCES:
            loc = error <= t
            joint = loc & present
            stats['thresholds'][str(t)] = dict(location_only=int(loc.sum()), joint=int(joint.sum()),
                                               excluded_by_presence=int((loc & ~present).sum()))
        result[mode] = stats
    comparisons = {}
    for target, baseline in [('mask_R', 'raw'), ('mask_R', 'learned_g'), ('mask_R', 'mask_clean'),
                              ('mask_clean', 'raw'), ('learned_g', 'raw'), ('full_R', 'raw')]:
        a, b = rows[MODES.index(target)], rows[MODES.index(baseline)]
        delta = a[:, 0].astype(float) - b[:, 0]
        comparison = dict(mean_error_difference=interval(delta, videos), improved=int((delta < 0).sum()),
                          worsened=int((delta > 0).sum()), unchanged=int((delta == 0).sum()), thresholds={})
        for t in TOLERANCES:
            aj = (a[:, 0] <= t) & (a[:, 1] >= .7)
            bj = (b[:, 0] <= t) & (b[:, 1] >= .7)
            comparison['thresholds'][str(t)] = dict(joint_difference=interval(aj.astype(float) - bj, videos),
                gains=int((aj & ~bj).sum()), losses=int((bj & ~aj).sum()),
                coordinate_difference=interval((a[:, 0] <= t).astype(float) - (b[:, 0] <= t), videos))
        comparisons[target + '_minus_' + baseline] = comparison
    return dict(modes=result, paired=comparisons)


def main():
    OUT.mkdir(exist_ok=False)
    completion = read(SOURCE / 'numeric_completion.json')
    assert completion['all_numeric_evaluations_completed']
    for name, expected in completion['artifacts'].items():
        assert sha(SOURCE / name) == expected, name
    plan = read(SOURCE / 'plan.json')
    for name, expected in plan['hashes'].items():
        assert sha(ROOT / name) == expected, name
    old = read(SOURCE / 'summary.json')
    results, hashes = {}, {}
    reference = None
    for arm in ARMS:
        results[arm] = {}
        seeds = (42, 1042, 2042, 3042) if arm.startswith('diff') else (42,)
        for seed in seeds:
            path = SOURCE / arm / f'artificial_seed{seed}.npz'
            hashes[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path) as archive:
                a = {k: archive[k] for k in archive.files}
            if reference is None:
                reference = a
            for key in ('ids', 'point', 'videos', 'counts'):
                np.testing.assert_array_equal(a[key], reference[key])
            for i in (0, 4, 5):
                np.testing.assert_allclose(a['rows'][:, i], reference['rows'][:, i], rtol=0, atol=1e-7)
            assert a['rows'].shape == (384, 6, 5, 2)
            assert np.isfinite(a['rows']).all() and (a['rows'][..., 0] >= 0).all()
            assert ((a['rows'][..., 1] >= 0) & (a['rows'][..., 1] <= 1)).all()
            selected = a['point']
            assert selected.dtype == bool and selected.sum() == 663
            videos = np.broadcast_to(a['videos'][:, None], selected.shape)
            rows = a['rows'].transpose(1, 0, 2, 3)
            entry = {name: summarize(rows[:, mask], videos[mask]) for name, mask in
                     [('covered', selected), ('uncovered', ~selected), ('all', np.ones_like(selected))]}
            expected = old['conditions'][arm] if seed == 42 else old['conditions'][arm]['sample_sensitivity'][str(seed)]
            for mode, stats in entry['covered']['modes'].items():
                prior = expected['artificial'][mode]
                assert stats['thresholds']['0.1']['joint'] == prior['correct']
                assert abs(stats['mean_error'] - prior['mean_error']['mean']) < 1e-7
            raw_local = a['local'][:, 0, :, 0][selected]
            restored_local = a['local'][:, 1, :, 0][selected]
            improved = restored_local < raw_local
            worsened_head = a['rows'][:, 3, :, 0][selected] > a['rows'][:, 0, :, 0][selected]
            entry['features'] = dict(covered_raw=float(a['error_sums'][:, 0].sum() / a['counts'][:, 0].sum()),
                covered_R=float(a['error_sums'][:, 2].sum() / a['counts'][:, 0].sum()),
                center_raw=float(raw_local.mean()), center_R=float(restored_local.mean()),
                center_improved=int(improved.sum()), center_improved_but_oracle_head_worse=int((improved & worsened_head).sum()))
            assert abs(entry['features']['covered_R'] - expected['feature']['R']['covered']) < 1e-7
            results[arm][str(seed)] = entry
            m = entry['covered']['modes']
            print(arm, seed, {k: (round(v['mean_error'] * 100, 4), v['thresholds']['0.05']['joint'],
                                  v['thresholds']['0.1']['joint']) for k, v in m.items()}, flush=True)
    report = dict(results=results, source_hashes=hashes, analysis_source_sha256=sha(Path(__file__)),
        primary_generation_seed=42, extra_diffusion_seeds=[1042, 2042, 3042], training=False, inference=False,
        protocol=dict(faces=128, shapes_per_face=3, pairs=384, covered_points=663, uncovered_points=1257,
            videos=len(np.unique(reference['videos'])), presence_threshold=.7, tolerances=TOLERANCES,
            g_oracle='g=c: binary known artificial coverage on 16x16 feature grid intersected with ellipse support',
            input='(1-c)*X+c*R; clean control=(1-c)*X+c*Z',
            normalization='coordinate distance / GT ellipse major diameter; NOT production Head-normalized NME',
            feature_metric='cosine distance + channel mean squared error; NOT coordinate NME',
            bootstrap='5000 paired video-cluster resamples, point-weighted, exploratory unadjusted intervals'),
        limitations=['Fixed existing head; cannot separate intrinsic feature information loss from recoverable head mismatch.',
            'Known coverage is not an optimal blending coefficient and cannot remove global ViT contamination outside the mask.',
            'Reused small train-derived development cohort; no independent val/test or production claim.',
            'Diffusion sensitivity uses four generation seeds of the same trained checkpoint, not four training runs.',
            'Saved raw predictions and published primary/secondary metrics reproduced; no new model forward pass.'])
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 人工遮蔽の正解gによるhead評価', '',
        '同一128顔×3形状、遮蔽663点。主seed42。平均誤差はGT顔楕円長径比。成功数はpresence≥0.7も要求。', '',
        '| 条件 | 入力方式 | 平均位置誤差 | 5%成功 | 10%成功 |', '|---|---|---:|---:|---:|']
    for arm in ARMS:
        for mode, m in results[arm]['42']['covered']['modes'].items():
            lines.append(f"| {arm} | {mode} | {m['mean_error']*100:.4f}% | {m['thresholds']['0.05']['joint']} | {m['thresholds']['0.1']['joint']} |")
    lines += ['', '未遮蔽1257点・全1920点、座標のみの成功、2/3/5/10%閾値、対応動画区間、DF生成4seed、局所特徴改善と位置悪化の同時発生数はsummary.jsonに保存。',
              '', '人工maskを直接使う推論済み条件の再集計。gの学習やhead再学習は行っていない。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
