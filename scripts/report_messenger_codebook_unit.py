"""Matched before/after teacher-g comparison and scoped checkpoint cleanup."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from scripts import run_messenger_codebook_unit as experiment
from scripts.report_messenger_codebook_teacher import pair_comparison, semantic_metrics, summarize, verify_artifacts
from scripts.run_messenger_codebook_teacher import base, read
from src.models.messenger_codebook_factorial import COUNTS, DIMENSIONS, arm_spec
from src.models.messenger_codebook_unit import ARMS


def arrays(folder):
    result = {}
    for domain in ('natural', 'artificial'):
        with np.load(folder/f'{domain}_seed42.npz') as data:
            result[domain] = {key: data[key] for key in data.files}
    return result


def parity(current, before):
    for domain, a in current.items():
        b = before[domain]
        for key in ('ids', 'videos', 'teacher_g', 'state' if domain == 'natural' else 'point'):
            np.testing.assert_array_equal(a[key], b[key])
        for key in ('rows', 'norms', 'gate', 'teacher_g'):
            if not np.isfinite(a[key]).all():
                raise ValueError(f'Nonfinite {domain} {key}')
        np.testing.assert_allclose(a['rows'][:,0], b['rows'][:,0], rtol=0, atol=1e-6)
        if domain == 'artificial':
            np.testing.assert_array_equal(a['covered'], a['teacher_g'])


def keys_summary(data, previous):
    result = {}
    for domain, a in data.items():
        masks = {'occluded':a['state']==1,'visible':a['state']==2} if domain == 'natural' else {'covered':a['point'],'uncovered':~a['point']}
        methods = {}
        for i, method in enumerate(('raw','full_R','teacher_g')):
            methods[method] = {}
            for group, mask in masks.items():
                value = summarize(a['rows'][:,i], mask, a['videos'], previous[domain]['rows'][:,i])
                value['versus_prior_condition'] = value.pop('versus_fresh_regression384', None)
                value['semantic'] = semantic_metrics(a['rows'][:,i], mask)
                methods[method][group] = value
        result[domain] = methods
    return result


def check_convergence(folder, published):
    hist = read(folder/'R_history.json')
    selected = min(hist, key=lambda row: row['dev']['position'])['epoch']
    if not hist[-1]['converged'] or selected != published['R']['selected_epoch'] or hist[-1]['epoch'] != published['R']['stop_epoch']:
        raise ValueError('Convergence or best-epoch inconsistency')
    for kind in ('best','last'):
        checkpoint = torch.load(folder/f'{kind}_R.pt', map_location='cpu', weights_only=True)
        norm_error = (checkpoint['state']['codebook.embedding'].norm(dim=1)-1).abs().max().item()
        if (checkpoint['identity'] != published['identity'] or norm_error > 2e-6
                or not np.isfinite(norm_error)):
            raise ValueError('Checkpoint identity or unit norm mismatch')
        if kind == 'best' and checkpoint['epoch'] != selected:
            raise ValueError('Best weights differ from selected epoch')
        if kind == 'last' and not checkpoint['converged']:
            raise ValueError('Last checkpoint is incomplete')


def report(out):
    out = Path(out)
    complete = out/'numeric_completion.json'
    if complete.exists():
        verify_artifacts(out, read(complete)['artifacts'])
        return
    status = read(out/'training_completion.json')
    if not status['all_converged'] or status['arms'] != list(ARMS) or status['learned_g'] or status['controls_retrained']:
        raise ValueError('Incomplete or incompatible training contract')
    plan = read(out/'plan.json')
    experiment.verify_hashes(plan['hashes'])
    prior = experiment.BASELINE
    experiment.manifest_contract(read(out/'manifest.json'), read(prior/'manifest.json'))
    previous_completion = read(prior/'numeric_completion.json')
    if not previous_completion['all_numeric_evaluations_complete']:
        raise ValueError('Previous controls incomplete')
    verify_artifacts(prior, previous_completion['artifacts'])
    continuous = {name: arrays(prior/name) for name in ('continuous16','continuous32','regression384')}
    result = dict(conditions={},before_after={},versus_matching_continuous={},versus_continuous32={},compared_to={},attention_pairs={})
    files = [out/name for name in ('plan.json','manifest.json','teacher_audit.json','normalization.json','gate_authentication.json',
                                   'comparison.json','training_completion.json')]
    files.extend(out/f'initial_codes_d{d}_k{k}.json' for d in DIMENSIONS for k in COUNTS)
    lines = ['# 教師g固定の正規化修正比較', '', '教師gと固定headを共通にし、B・D・Eの12条件を新規学習した。', '',
             '| 条件 | 自然遮蔽 NME 旧 → 新 | 5%以内（自然） | 人工遮蔽 NME 旧 → 新 | 5%以内（人工） |',
             '|---|---:|---:|---:|---:|']
    current = {}
    for arm in ARMS:
        folder = out/arm
        published = read(folder/'evaluation.json')
        verify_artifacts(folder, published['artifacts'])
        check_convergence(folder, published)
        data = arrays(folder)
        before = arrays(prior/arm)
        parity(data,before)
        for control in continuous.values():
            parity(data,control)
        current[arm] = data
        spec = arm_spec(arm)
        selected = keys_summary(data,before)
        selected.update(R=published['R'], feature_components=published['feature_components'], code_usage=published['code_usage'],
                        codebook_change_l2=published['codebook_change_l2'], ema_steps=published['ema_steps'],
                        unit_code_norm_max_error=published['unit_code_norm_max_error'])
        result['conditions'][arm] = selected
        result['before_after'][arm] = pair_comparison(data,before)
        result['versus_matching_continuous'][arm] = pair_comparison(data,continuous[f'continuous{spec["dimension"]}'])
        result['versus_continuous32'][arm] = pair_comparison(data,continuous['continuous32'])
        control_mode = 'A' if spec['mode']=='B' else 'C'
        control_name = f'{control_mode}_d{spec["dimension"]}_k{spec["count"]}'
        control = arrays(prior/control_name)
        parity(data,control)
        result['compared_to'][arm] = dict(arm=control_name, differences=pair_comparison(data,control))
        previous_metrics = read(prior/arm/'evaluation.json')
        cells = []
        for domain, group in (('natural','occluded'),('artificial','covered')):
            old = previous_metrics[domain]['teacher_g'][group]['mean_error']['mean']
            new = selected[domain]['teacher_g'][group]
            actual = published[domain]['teacher_g'][group]
            if abs(new['mean_error']-actual['mean_error']['mean'])>1e-6 or new['thresholds']['0.05']['location']!=actual['thresholds']['0.05']['location']:
                raise ValueError('Published evaluation disagrees with saved predictions')
            cells.extend([f'{old*100:.4f}% → {new["mean_error"]*100:.4f}%',
                          f'{new["thresholds"]["0.05"]["location"]}/{new["points"]}'])
        lines.append('| '+' | '.join([arm,*cells])+' |')
        files.extend(p for p in folder.iterdir() if p.suffix in ('.json','.npz'))
    for d in DIMENSIONS:
        for k in COUNTS:
            key = f'd{d}_k{k}'
            result['attention_pairs'][key] = pair_comparison(current[f'E_{key}'],current[f'D_{key}'])
    result['limits'] = ('Teacher g fixed, natural five-point weak teacher, artificial generated mask. Historical train-internal video split, '
                        'one seed. New quantization MSE is computed on unit vectors; historical raw MSE is on a different scale.')
    lines += ['', '旧条件・辞書なし対照との動画対応95%区間を `comparison_audit.json` に収録。',
              '修正後の量子化MSEは単位ベクトル間の値。旧実験は正規化前の値で計算している。']
    base.write(out/'comparison_audit.json',result)
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    files.extend((out/'comparison_audit.json',out/'report.md'))
    weights = [dict(path=str(p.relative_to(out)),absolute_path=str(p.resolve()),sha256=base.sha(p),bytes=p.stat().st_size)
               for p in weight_paths(out)]
    base.write(complete,dict(all_numeric_evaluations_complete=True,artifacts={str(p.relative_to(out)):base.sha(p) for p in files},
                            weights=weights, learned_g=False, controls_retrained=False, weight_state_at_numeric_completion='retained'))


def weight_paths(out):
    root=Path(out).resolve()
    if Path(out).is_symlink() or Path(out).absolute()!=root:
        raise ValueError("Output root is a symlink")
    paths=[root/arm/f"{which}_{stage}.pt" for arm in ARMS for stage in ("R",) for which in ("best","last")]
    paths.extend(root/f"initial_codes_d{dim}_k{count}.npy" for dim in DIMENSIONS for count in COUNTS)
    for path in paths:
        if path.is_symlink() or path.parent.resolve()!=path.parent or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe checkpoint path")
    return paths


def assert_not_in_use(out):
    targets={str(path) for path in weight_paths(out)}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            args=(entry/"cmdline").read_bytes().split(b"\0")
            if args and b"python" in args[0] and any(b"run_messenger_codebook_unit.py" in arg for arg in args):
                raise RuntimeError(f"Another comparison process is alive: {entry.name}")
            for descriptor in (entry/"fd").iterdir():
                try:
                    if os.readlink(descriptor) in targets:
                        raise RuntimeError(f"Process {entry.name} has a generated weight open")
                except (FileNotFoundError,PermissionError,OSError):
                    continue
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue


def cleanup(out):
    """Call while holding the canonical lease. Supports recovery after partial deletion."""
    out=Path(out)
    completion=read(out/"numeric_completion.json")
    if not completion["all_numeric_evaluations_complete"]:
        raise ValueError("Numeric evaluation is incomplete")
    verify_artifacts(out,completion["artifacts"])
    expected={str(p.relative_to(out.resolve())):p for p in weight_paths(out)}
    saved={row["path"]:row for row in completion["weights"]}
    if set(expected)!=set(saved):
        raise ValueError("Unexpected deletion scope")
    deletion=out/"checkpoint_deletion.json"
    prepared=deletion.exists()
    if prepared and read(deletion)["weights"]!=completion["weights"]:
        raise ValueError("Deletion ledger changed")
    assert_not_in_use(out)
    for name,path in expected.items():
        if saved[name].get("absolute_path",str(path)) != str(path):
            raise ValueError("Absolute checkpoint scope changed")
        if path.exists():
            if base.sha(path)!=saved[name]["sha256"] or path.stat().st_size!=saved[name]["bytes"]:
                raise ValueError("Checkpoint changed before deletion")
        elif not prepared:
            raise ValueError("Checkpoint disappeared before deletion was prepared")
    base.write(deletion,dict(phase="prepared",output_root=str(out.resolve()),weights=completion["weights"],process_audit_pid=os.getpid()))
    for path in expected.values():
        path.unlink(missing_ok=True)
    if any(path.exists() for path in expected.values()):
        raise RuntimeError("Some generated weights remain")
    base.write(deletion,dict(phase="completed",output_root=str(out.resolve()),weights=completion["weights"],
               deleted_bytes=sum(row["bytes"] for row in saved.values()),process_audit_pid=os.getpid()))
    base.status(out,"completed",all_numeric_evaluations_complete=True,new_weights_remaining=0)
