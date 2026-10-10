#!/usr/bin/env python3
"""Authenticated gates and production launch for existing-WIDER -> MAL2 ellipse."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools.dinov3_deimv2_common import load_config, resolved_config_sha256
from src.wandb_utils import write_json_atomic
from scripts.preflight_dinov3_deimv2_batch_plan import _run_candidate, _fallback_plans

ARCH = 'mal_alpha2_ellipse_from_existing_wider'
CONFIG = ROOT / f'configs/dinov3_deimv2/{ARCH}.yaml'
GATES = ROOT / f'outputs/training_gates/{ARCH}'
OUTPUT = ROOT / f'outputs/experiments/{ARCH}'
PYTHON = str(ROOT / '.venv/bin/python')
STAGES = ('schema', 'gpu_step', 'overfit', 'small_scale')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def variant(suffix):
    return CONFIG.with_name(ARCH + suffix + '.yaml')


def identity():
    config = load_config(CONFIG)
    init = Path(config['model']['init_checkpoint']['path'])
    digest = sha(init)
    if digest != config['verification']['expected_init_checkpoint_sha256']:
        raise RuntimeError('WIDER initialization checkpoint changed')
    if config['deployment']['topology'] != 'non-cascade' or config['deployment']['stages'] != ['full-image FACE/ellipse']:
        raise RuntimeError('Expected HEAD-free full-image FACE/ellipse topology')
    for group in config['train']['param_groups']:
        if group['name'].startswith('dinov3') and group['lr'] != config['train']['lr_backbone']:
            raise RuntimeError('Backbone learning-rate configuration disagrees')
    files = [ROOT / p for p in ('tools/run_mal_existing_wider.py', 'scripts/train_dino.py',
        'tools/train_dinov3_deimv2.py', 'tools/run_small_scale_training_gate.py',
        'tools/preflight_dinov3_deimv2_l.py', 'scripts/preflight_dinov3_deimv2_batch_plan.py',
        'scripts/run_dinov3_deimv2_with_batch_plan.py', 'src/data/ellipse_augmentation.py',
        'src/models/dinov3_deimv2/official.py', 'src/models/dinov3_deimv2_ellipse/official.py',
        'src/third_party/deimv2/engine/deim/deim_criterion.py')]
    return dict(config_sha256=resolved_config_sha256(config), initialization_sha256=digest,
        variants={s: resolved_config_sha256(load_config(variant(s))) for s in ('_gate', '_overfit', '_probe')},
        implementation={str(p.relative_to(ROOT)): sha(p) for p in files},
        annotations={split: sha(path) for split, path in config['data']['annotation_paths'].items()})


def require_gates(stages, expected):
    for stage in stages:
        evidence = json.loads((GATES / stage / 'success.json').read_text())
        if evidence.get('status') != 'passed' or evidence.get('identity') != expected or evidence.get('stage') != stage:
            raise RuntimeError(f'Missing or stale authenticated gate: {stage}')
        if stage == 'gpu_step' and evidence.get('batch_plan_sha256') != sha(GATES / 'gpu_step/batch_plan.json'):
            raise RuntimeError('Batch plan changed after gate authentication')


def run(command):
    subprocess.run([str(v) for v in command], cwd=ROOT, check=True)


def gpu_gate(config):
    work = GATES / 'gpu_step'
    args = SimpleNamespace(output_dir=work, probe_config=variant('_probe'),
        data_root=Path(config['data']['data_root']), device_index=0)
    attempts = []
    for batch in (4, 8, 16):
        result = _run_candidate(args=args, candidate=batch, accumulation=16 // batch,
            production_config=config, checkpoint_validator=lambda cp: {
                'model_parameters_finite': all(torch.isfinite(v).all().item() for v in cp['model'].values() if torch.is_floating_point(v))})
        attempts.append(result)
        write_json_atomic(work / 'attempts.json', dict(attempts=attempts))
    safe = [a for a in attempts if a['safe']]
    if not safe:
        raise RuntimeError('No safe microbatch candidate')
    fastest = min(a['projected_production_seconds'] for a in safe)
    selected = min((a for a in safe if a['projected_production_seconds'] <= fastest * 1.03), key=lambda a: a['microbatch'])
    plan = dict(status='passed', resolved_config_sha256=resolved_config_sha256(config),
        probe_resolved_config_sha256=resolved_config_sha256(load_config(variant('_probe'))),
        attempts=attempts, selected={k: selected[k] for k in ('microbatch', 'accumulation_steps', 'effective_batch_size', 'projected_production_seconds')},
        fallback_plans=_fallback_plans(safe, selected), runtime_oom_fallback=True,
        selection_rule='fastest synchronized post-warmup projection; within 3% choose smaller batch')
    write_json_atomic(work / 'batch_plan.json', plan)


def learning_gate(stage, config):
    work = GATES / stage
    plan = json.loads((GATES / 'gpu_step/batch_plan.json').read_text())
    batch = plan['selected']
    policy = config['gate_policy'][stage]
    overfit = stage == 'overfit'
    metrics = work / ('metrics.csv' if overfit else 'val_metrics.csv')
    # Keep the published success marker absent until both improvement and actual
    # optimizer-update budget are verified. The generic report is diagnostic.
    temporary_marker = work / 'metric_pass.json'
    cmd = [PYTHON, ROOT / 'tools/run_small_scale_training_gate.py', '--work-dir', work,
        '--metrics', metrics, '--metric', 'loss_total' if overfit else 'mAP',
        '--direction', 'lower' if overfit else 'higher', '--minimum-points', str(policy['window'] * 2),
        '--window', policy['window'], '--minimum-relative-improvement', policy['minimum_relative_improvement'],
        '--minimum-improving-transition-fraction', policy['minimum_improving_transition_fraction'],
        '--success-marker', temporary_marker, '--report', work / 'report.json', '--',
        PYTHON, ROOT / 'tools/train_dinov3_deimv2.py', '--config', variant('_overfit' if overfit else '_gate'),
        '--data-root', config['data']['data_root'], '--output-dir', work, '--device', 'cuda:0',
        '--batch-size', batch['microbatch'], '--gradient-accumulation-steps', batch['accumulation_steps'],
        '--epochs', policy['epochs'], '--max-train-samples', policy['samples'], '--val-every', '1',
        '--val-max-images', policy.get('validation_images', 16), '--num-workers', '2',
        '--save-every', '0', '--wandb-mode', 'disabled']
    try:
        run(cmd)
        cp = torch.load(work / 'checkpoints/last.pt', map_location='cpu', weights_only=False, mmap=True)
        updates = int(cp['step'])
        del cp
        if updates < policy['minimum_optimizer_updates']:
            raise RuntimeError(f'Insufficient actual optimizer updates: {updates}')
        result = dict(optimizer_updates=updates, batch_plan_sha256=sha(GATES / 'gpu_step/batch_plan.json'), policy=policy)
        keep = 'best_loss.pt' if overfit else 'best_val_map.pt'
        if not (work / 'checkpoints' / keep).is_file():
            raise RuntimeError('Missing validation/loss-selected gate checkpoint')
        for checkpoint in (work / 'checkpoints').glob('*.pt'):
            if checkpoint.name != keep:
                checkpoint.unlink()
        return result
    finally:
        temporary_marker.unlink(missing_ok=True)


def main(*, dataset_version='face_pose_drive_v2_balanced_811', wandb_group='mal_alpha2_existing_wider'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'train'), required=True)
    args = parser.parse_args()
    stage = args.stage
    before = identity()
    config = load_config(CONFIG)
    require_gates(STAGES if stage == 'train' else STAGES[:STAGES.index(stage)], before)
    marker = OUTPUT / 'training_success.json' if stage == 'train' else GATES / stage / 'success.json'
    marker.unlink(missing_ok=True)
    extra = {}
    if stage == 'schema':
        run([PYTHON, ROOT / 'tools/preflight_dinov3_deimv2_l.py', '--config', CONFIG,
            '--skip-forward', '--output', GATES / 'schema/report.json'])
    elif stage == 'gpu_step':
        gpu_gate(config)
        extra['batch_plan_sha256'] = sha(GATES / 'gpu_step/batch_plan.json')
    elif stage in ('overfit', 'small_scale'):
        extra = learning_gate(stage, config)
    else:
        run([PYTHON, ROOT / 'scripts/run_dinov3_deimv2_with_batch_plan.py', '--config', CONFIG,
            '--data-root', config['data']['data_root'], '--output-dir', OUTPUT,
            '--batch-plan', GATES / 'gpu_step/batch_plan.json', '--device', 'cuda:0',
            '--wandb-entity', 'sato-waseda-university', '--wandb-project', 'face_detection',
            '--wandb-run-name', ARCH, '--wandb-group', wandb_group,
            '--dataset-version', dataset_version])
    if identity() != before:
        raise RuntimeError('Configuration, code, data or initialization changed during execution')
    write_json_atomic(marker, dict(status='passed', stage=stage, identity=before, **extra))


if __name__ == '__main__':
    main()
