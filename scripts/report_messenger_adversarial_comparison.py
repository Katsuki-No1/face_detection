"""Recompute comparison from point records, paired by video, and authenticate completion."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_messenger_success_thresholds import paired_interval
from scripts.run_messenger_multiaxis import sha, write
ARMS = ('baseline', 'adversarial', 'adversarial_fm')


def read(path):
    return json.loads(Path(path).read_text())


def report(out):
    out = Path(out)
    summary = read(out/'summary.json')
    complete = read(out/'training_completion.json')
    assert complete['all_converged'] and complete['clean_head'] is False
    for p, h in read(out/'plan.json')['hashes'].items():
        assert sha(ROOT/p) == h, p
    assert len(summary['checkpoints']) == 12
    for row in summary['checkpoints']:
        assert sha(ROOT/row['path']) == row['sha256']
    initialization = [read(out/a/'initialization.json') for a in ARMS]
    assert len({x['R'] for x in initialization}) == 1
    assert initialization[1]['D'] == initialization[2]['D']
    audits = {}; comparisons = {}; artifacts = {}
    for arm in ARMS:
        for stage in ('R', 'g'):
            h = read(out/arm/f'{stage}_history.json'); end = read(out/arm/f'{stage}_convergence.json')
            assert h[-1]['converged'] and h[-1]['epoch'] == end['stop_epoch']
            assert min(h, key=lambda r: r['dev']['position'])['epoch'] == end['selected_epoch']
        audits[arm] = {}
        for domain in ('natural', 'artificial'):
            path = out/arm/f'{domain}_seed42.npz'; a = np.load(path)
            ref = np.load(out/'baseline'/f'{domain}_seed42.npz')
            key = 'state' if domain == 'natural' else 'point'
            for k in ('ids', 'videos', key):
                np.testing.assert_array_equal(a[k], ref[k])
            np.testing.assert_allclose(a['rows'][:, 0], ref['rows'][:, 0], rtol=0, atol=1e-7)
            assert np.isfinite(a['rows']).all()
            assert a['rows'].shape[1] == (3 if domain == 'natural' else 4)
            artifacts[str(path.relative_to(ROOT))] = sha(path)
            masks = {'occluded': a['state'] == 1, 'visible': a['state'] == 2} if domain == 'natural' else {
                'covered': a['point'], 'uncovered': ~a['point']}
            names = ('raw', 'full_R', 'learned_g') if domain == 'natural' else ('raw', 'full_R', 'learned_g', 'mask_R')
            audits[arm][domain] = {}
            for label, mask in masks.items():
                methods = {}; videos = np.broadcast_to(a['videos'][:, None], mask.shape)[mask]
                for i, name in enumerate(names):
                    er = a['rows'][:, i, :, 0][mask]; pr = a['rows'][:, i, :, 1][mask]
                    pub = summary['conditions'][arm][domain][name][label]
                    assert len(er) == pub['points']
                    assert abs(float(er.mean())-pub['mean_error']['mean']) < 1e-7
                    thresholds = {}
                    for t in (.02, .03, .05, .1):
                        joint = (er <= t) & (pr >= .7)
                        assert int(joint.sum()) == pub['thresholds'][str(t)]['joint']
                        thresholds[str(t)] = dict(joint=int(joint.sum()), location=int((er <= t).sum()))
                    emitted = pr >= .7
                    methods[name] = dict(mean_error_all_GT=float(er.mean()), median=float(np.median(er)),
                        p90=float(np.quantile(er, .9)), emitted=int(emitted.sum()),
                        emitted_mean_error=float(er[emitted].mean()) if emitted.any() else None,
                        thresholds=thresholds, versus_raw_error=paired_interval(
                            er.astype(float)-a['rows'][:, 0, :, 0][mask], videos))
                audits[arm][domain][label] = dict(points=int(mask.sum()), methods=methods)
            # Persist semantic and requested GT-condition slices without selecting them by prediction.
            extras = np.load(out/arm/f'{domain}_extras.npz')
            slice_rows = []
            for contour in (False, True):
                for n in range(6):
                    for pose, lo, hi in [('all', 0, float('inf')), ('lt3', 0, 3), ('3to4', 3, 4), ('4to5', 4, 5), ('ge5', 5, float('inf'))]:
                        face = (extras['contour'] == contour) & ((extras['state'] == 1).sum(1) == n) & (extras['ratio'] >= lo) & (extras['ratio'] < hi)
                        for label, mask in masks.items():
                            selected = mask & face[:, None]
                            for i, name in enumerate(names):
                                er = a['rows'][:, i, :, 0]; pr = a['rows'][:, i, :, 1]
                                for slot in range(5):
                                    m = selected[:, slot]; count = int(m.sum())
                                    slice_rows.append(dict(contour=contour, occluded_count=n, pose=pose, group=label, mode=name,
                                        slot=slot, points=count, mean_error=float(er[:, slot][m].mean()) if count else None,
                                        joint5=int(((er[:, slot] <= .05) & (pr[:, slot] >= .7) & m).sum())))
            write(out/arm/f'{domain}_slices.json', slice_rows)
    for left, right in [('baseline', 'adversarial'), ('baseline', 'adversarial_fm'), ('adversarial', 'adversarial_fm')]:
        key = right+'_minus_'+left; comparisons[key] = {}
        for domain in ('natural', 'artificial'):
            a = np.load(out/left/f'{domain}_seed42.npz'); b = np.load(out/right/f'{domain}_seed42.npz')
            masks = {'occluded': a['state'] == 1, 'visible': a['state'] == 2} if domain == 'natural' else {
                'covered': a['point'], 'uncovered': ~a['point']}
            comparisons[key][domain] = {}
            for label, mask in masks.items():
                videos = np.broadcast_to(a['videos'][:, None], mask.shape)[mask]
                modes = [('learned_g', 2)] + ([('mask_R', 3)] if domain == 'artificial' else [])
                comparisons[key][domain][label] = {}
                for mode, i in modes:
                    aa = a['rows'][:, i][mask]; bb = b['rows'][:, i][mask]
                    comparisons[key][domain][label][mode] = dict(
                        error=paired_interval(bb[:, 0].astype(float)-aa[:, 0], videos),
                        success5=paired_interval(((bb[:, 0] <= .05) & (bb[:, 1] >= .7)).astype(float)-((aa[:, 0] <= .05) & (aa[:, 1] >= .7)), videos),
                        success10=paired_interval(((bb[:, 0] <= .1) & (bb[:, 1] >= .7)).astype(float)-((aa[:, 0] <= .1) & (aa[:, 1] >= .7)), videos))
    write(out/'comparison_audit.json', dict(within_arm=audits, paired_comparisons=comparisons,
        limitations='single training seed; reused small train-derived split; 5000 video bootstrap resamples; unadjusted exploratory intervals',
        identical_initial_R=True, identical_initial_D=True, position_selected=True, clean_head=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm in ARMS:
        for i, stage in enumerate(('R', 'g')):
            h = read(out/arm/f'{stage}_history.json')
            axes[i].plot([r['epoch'] for r in h], [100*r['dev']['position'] for r in h], label=arm)
        if arm != 'baseline':
            h = read(out/arm/'R_history.json')
            axes[2].plot([r['epoch'] for r in h], [r['dev']['D_loss'] for r in h], label=arm)
    for ax, title in zip(axes, ('R dev position (%)', 'g dev position (%)', 'D dev loss (moving critic)')):
        ax.set_title(title); ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'learning_curves.png'); plt.close(fig)
    lines = ['# Position-primary adversarial feature comparison', '',
        'Same small train-video split; regression only; checkpoints selected by dev position.',
        'No clean-feature input to head. Natural results use separately trained g.', '',
        '| Arm | R stop / best | g stop / best | Artificial mask-R error % | Natural occluded error % | Natural visible error % | Occluded joint5 / joint10 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        v = summary['conditions'][arm]; oc = v['natural']['learned_g']['occluded']; vi = v['natural']['learned_g']['visible']
        lines.append(f"| {arm} | {v['R']['stop_epoch']} / {v['R']['selected_epoch']} | {v['g']['stop_epoch']} / {v['g']['selected_epoch']} | {100*v['artificial']['mask_R']['covered']['mean_error']['mean']:.4f} | {100*oc['mean_error']['mean']:.4f} | {100*vi['mean_error']['mean']:.4f} | {oc['thresholds']['0.05']['joint']} / {oc['thresholds']['0.1']['joint']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in ('.npz', '.json', '.png', '.md') and p.name not in ('status.json', 'numeric_completion.json'):
            artifacts[str(p.relative_to(ROOT))] = sha(p)
    write(out/'numeric_completion.json', dict(complete=True, arms=list(ARMS), position_selected=True,
        all_validation_plateaus=True, clean_head=False, source_and_weights_verified=True, checkpoints_retained=12, artifacts=artifacts))


if __name__ == '__main__':
    report(ROOT/'outputs/diagnostics/messenger_adversarial_comparison_20261119')
