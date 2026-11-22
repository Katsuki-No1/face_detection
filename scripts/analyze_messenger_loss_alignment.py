"""Saved feature statistics and checkpoint-selection audit; no head inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/diagnostics/messenger_diffusion_comparison_20261114'
OUT = ROOT / 'outputs/diagnostics/messenger_loss_alignment_20261118'
ARMS = ('det_feature', 'det_head', 'diff_feature', 'diff_head')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    OUT.mkdir(exist_ok=False)
    completion = read(SOURCE / 'numeric_completion.json')
    for name, expected in completion['artifacts'].items():
        assert sha(SOURCE / name) == expected
    old = read(SOURCE / 'summary.json')
    results, hashes = {}, {}
    for arm in ARMS:
        p = SOURCE / arm / 'artificial_seed42.npz'
        h = SOURCE / arm / 'R_history.json'
        hashes[str(p.relative_to(ROOT))] = sha(p)
        hashes[str(h.relative_to(ROOT))] = sha(h)
        history = read(h)
        best_f = min(history, key=lambda row: row['dev']['feature'])
        best_p = min(history, key=lambda row: row['dev']['position'])
        convergence = read(SOURCE / arm / 'R_convergence.json')
        assert best_f['epoch'] == convergence['selected_epoch']
        with np.load(p) as a:
            mask = a['point']
            assert mask.sum() == 663
            # Only raw, learned-g and restored-mask head predictions are analyzed.
            # No clean-feature head comparison; clean FEATURE statistics remain targets.
            raw_error = a['rows'][:, 0, :, 0][mask]
            restored_error = a['rows'][:, 3, :, 0][mask]
            learned_error = a['rows'][:, 2, :, 0][mask]
            assert abs(float(raw_error.mean()) - old['conditions'][arm]['artificial']['raw']['mean_error']['mean']) < 1e-7
            local_raw = a['local'][:, 0][mask]
            local_R = a['local'][:, 1][mask]
            energy = a['energy']
            ratio = float(energy[:, 1][mask].sum() / energy[:, 2][mask].sum())
            assert abs(ratio - old['conditions'][arm]['local_variance_R_over_clean']) < 1e-6
            delta_f = local_R[:, 0] - local_raw[:, 0]
            delta_p = restored_error - raw_error
            feature_improved = delta_f < 0
            result = dict(feature_center_raw=float(local_raw[:, 0].mean()), feature_center_R=float(local_R[:, 0].mean()),
                feature_3x3_raw=float(local_raw[:, 1].mean()), feature_3x3_R=float(local_R[:, 1].mean()),
                local_energy_R_over_target=ratio,
                local_energy_X_over_target=float(energy[:, 0][mask].sum() / energy[:, 2][mask].sum()),
                feature_center_improved=int(feature_improved.sum()),
                feature_improved_position_worse=int((feature_improved & (delta_p > 0)).sum()),
                feature_improved_position_better=int((feature_improved & (delta_p < 0)).sum()),
                spearman_feature_error_delta_vs_position_error_delta=float(spearmanr(delta_f, delta_p).statistic),
                mean_position_raw=float(raw_error.mean()), mean_position_mask_R=float(restored_error.mean()),
                mean_position_learned_g=float(learned_error.mean()),
                feature_selected_epoch=best_f['epoch'], position_best_epoch=best_p['epoch'],
                dev_at_feature_selected=best_f['dev'], dev_at_position_best=best_p['dev'],
                selected_dev_position_loss_relative_excess=float(best_f['dev']['position'] / best_p['dev']['position'] - 1))
        results[arm] = result
        print(arm, json.dumps(result), flush=True)
    report = dict(results=results, hashes=hashes, training=False, inference=False,
        no_clean_feature_head_comparison=True,
        limitations=['Feature targets may use paired unoccluded features; no unoccluded feature is passed to a head.',
            'Spatial energy is not gradient/phase alignment and cannot identify which semantic information is missing.',
            'Spearman is descriptive on repeated points; no independent-point significance test.',
            'Dev position is all-point coordinate SmoothL1, not NME; logged minimum is not a newly evaluated checkpoint.',
            'Feature information loss vs head mismatch remains unresolved.'])
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
