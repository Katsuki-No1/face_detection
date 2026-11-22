"""Authenticated launcher contracts; production remains closed without four gates."""
import hashlib
import json
from pathlib import Path

from .geometry_prior_training import file_hash

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('schema', 'gpu_step', 'overfit', 'small_scale')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_hashes():
    files = list((ROOT/'keypoint_comparison1/landmark_detection').glob('*.py'))
    files += [ROOT/'scripts/train_messenger_geometry_prior.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def authenticate(config):
    config = Path(config)
    spec = json.loads(config.read_text())
    if spec['schema'] != 'geometry-prior-launch-v1':
        raise ValueError('Unknown launcher configuration')
    if spec['adversarial']:
        raise ValueError('Adversarial production requires a validation-selected k manifest and separate integration')
    if spec['arm'] not in ['control'] + [f'k{k}' for k in range(1, 8)]:
        raise ValueError('Unknown prior arm')
    raw = spec['parent_config']
    if raw['train']['seed'] != 42 or raw['train']['intended_effective_batch_size'] != 64:
        raise ValueError('Seed/effective batch contract changed')
    for path, expected in spec['artifacts'].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError(f'Artifact changed: {path}')
    prior = spec['prior']
    if spec['artifacts'].get(prior['path']) != prior['sha256']:
        raise ValueError('Prior is not bound to artifact inventory')
    expected_k = 1 if spec['arm'] == 'control' else int(spec['arm'][1:])
    if prior['k'] != expected_k:
        raise ValueError('Prior k differs from declared arm')
    return spec, dict(config_sha256=file_hash(config), artifacts=spec['artifacts'], sources=source_hashes())


def verify_gates(spec, identity, root=ROOT):
    directory = Path(root)/'outputs/training_gates'/spec['architecture']
    for stage in STAGES:
        marker = json.loads((directory/stage/'success.json').read_text())
        if (marker.get('status') != 'passed' or marker.get('architecture') != spec['architecture']
                or marker.get('gate_stage') != stage or marker.get('identity') != identity):
            raise ValueError(f'Missing or stale {stage} gate')
    plan = json.loads((directory/'gpu_step/batch_plan.json').read_text())
    if (plan.get('identity') != identity or plan.get('status') != 'passed'
            or plan.get('candidates') != [4, 8, 16] or plan.get('effective_batch') != 64):
        raise ValueError('Invalid production preflight')
    batch = plan['selected_microbatch']
    if batch not in [a['microbatch'] for a in plan['attempts'] if a['status'] == 'safe']:
        raise ValueError('Selected microbatch was not safe')
    return batch


def bind_run(output, contract, *, resume):
    output = Path(output)
    path = output/'run_identity.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if not resume or existing['contract'] != contract:
            raise ValueError('Existing output requires resume with identical contract')
        return existing
    if resume:
        raise ValueError('Cannot resume an unregistered run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse nonempty unregistered output')
    value = dict(contract=contract, wandb_id='gp-' + digest(contract)[:20])
    write_json(path, value)
    return value


def start_wandb(spec, binding, output, *, resume, diagnostic):
    import wandb
    cfg = spec['parent_config']['wandb']
    run = wandb.init(entity=cfg['entity'], project=cfg['project'],
        group='messenger_geometry_prior_launcher_checks' if diagnostic else 'messenger_geometry_prior_v1',
        name=spec['architecture'] + ('-launcher-check' if diagnostic else ''),
        id=binding['wandb_id'], resume='allow' if resume else 'never', mode='online',
        dir=str(output), config=binding['contract'],
        tags=['diagnostic-not-gate' if diagnostic else 'production', spec['arm']])
    if run is None or run.settings.mode != 'online':
        raise RuntimeError('Online W&B required')
    if resume and (Path(output)/'checkpoints/last.pt').exists() and not run.resumed:
        run.finish(exit_code=1)
        raise RuntimeError('Existing checkpoint requires the original online W&B run')
    write_json(Path(output)/'wandb_run.json', dict(id=run.id, url=run.url, resumed=run.resumed))
    return run
