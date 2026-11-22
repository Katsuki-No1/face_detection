"""Stage ownership, differentiable objectives and authenticated transitions."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

STAGES = ('S1', 'S2', 'S3a', 'S3b', 'S3c')
TOPOLOGY = ['full_image_face_ellipse', 'face_crop_dino', 'messenger_prior_fusion', 'query_refinement']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configure_stage(pipeline, stage):
    """Return exact optimizer names. Frozen head input autograd remains enabled."""
    if stage not in STAGES:
        raise ValueError('Unknown stage')
    pipeline.requires_grad_(False).eval()
    pipeline.zero_grad(set_to_none=True)
    selected = []
    for name, parameter in pipeline.named_parameters():
        recovery = name.removeprefix('base.messenger_recovery.')
        active = (
            (stage == 'S2' and name == 'bank.centers') or
            (stage == 'S3a' and name.startswith('base.messenger_recovery.') and
             (recovery == 'messenger_token' or recovery.startswith(('layers.', 'decoder.')))) or
            (stage == 'S3b' and name.startswith('base.messenger_recovery.') and
             recovery.startswith(('local_gate.', 'structure_gate.', 'structure.')) and
             not recovery.startswith('structure.descriptor.')) or
            (stage == 'S3c' and name.startswith('base.head.')))
        if active:
            parameter.requires_grad_(True)
            selected.append(name)
    if stage != 'S1' and not selected:
        raise ValueError('No trainable parameters for stage; install the declared module first')
    block = pipeline.base.messenger_recovery
    if getattr(block, 'reference_memory', None) is not None:
        raise ValueError('Reference adapter ownership is not defined')
    # All frozen modules stay in eval. Only owned modules enable training mode.
    if stage == 'S3a':
        block.layers.train(); block.decoder.train()
    elif stage == 'S3b':
        block.local_gate.train()
        if block.structure is not None:
            block.structure.train(); block.structure.descriptor.eval(); block.structure_gate.train()
    elif stage == 'S3c':
        pipeline.base.head.train()
    return selected


def weighted_mean(error, weight):
    weight = weight.detach().to(error)
    if weight.shape != error.shape:
        weight = weight.expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1)


def recovery_objective(candidate, original, teacher, coverage, paired, visible):
    """S3a: paired coverage/outside + sparse unaugmented visible preservation."""
    original, teacher = original.detach(), teacher.detach()
    paired = paired.detach()[:, None, None, None]
    error = F.smooth_l1_loss(candidate, teacher, reduction='none').mean(1, keepdim=True)
    covered = weighted_mean(error, coverage * paired)
    outside = weighted_mean(error, (1 - coverage) * paired)
    identity = weighted_mean(F.smooth_l1_loss(candidate, original, reduction='none').mean(1, keepdim=True),
                             visible * ~paired)
    return dict(total=covered + outside + identity, covered=covered, outside=outside, identity=identity)


def adoption_target(original, candidate, teacher, mask, channel, coverage, epsilon=1e-8):
    """Oracle for the actual channel-scaled fusion, on artificial covered tokens only."""
    with torch.no_grad():
        v = mask * channel * (candidate - original)
        q = teacher - original
        norm = v.square().sum(1, keepdim=True)
        valid = (norm > epsilon) & (coverage > 0) & mask.bool()
        target = ((v * q).sum(1, keepdim=True) / (norm + epsilon)).clamp(0, 1)
    return target, valid


def adoption_objective(gate, original, candidate, teacher, mask, channel, coverage,
                       paired, visible, position_loss):
    """S3b: position_loss must use frozen H, and remain differentiable to gate."""
    original, candidate, teacher = (v.detach() for v in (original, candidate, teacher))
    mask, channel = mask.detach(), channel.detach()
    paired = paired.detach()[:, None, None, None]
    target, valid = adoption_target(original, candidate, teacher, mask, channel, coverage * paired)
    fused = original + mask * gate * channel * (candidate - original)
    adopt = weighted_mean((gate - target).square(), valid * coverage)
    pair = weighted_mean(F.smooth_l1_loss(fused, teacher, reduction='none').mean(1, keepdim=True), coverage * paired)
    identity = weighted_mean(F.smooth_l1_loss(fused, original, reduction='none').mean(1, keepdim=True), visible * ~paired)
    return dict(total=.01 * adopt + .1 * pair + .1 * position_loss + .1 * identity,
                adoption=adopt, paired=pair, identity=identity, position=position_loss,
                valid_teacher_tokens=int(valid.sum()), fused=fused)


def require_transition(root, requested_stage, contract):
    """No downstream launch without every preceding authenticated quality marker."""
    if requested_stage not in STAGES:
        raise ValueError('Unknown requested stage')
    root = Path(root).resolve()
    for stage in STAGES[:STAGES.index(requested_stage)]:
        marker_path = root / stage / 'transition_success.json'
        if not marker_path.exists():
            raise RuntimeError(f'{requested_stage} blocked: {stage} has no authenticated transition success')
        marker = json.loads(marker_path.read_text())
        if marker.get('status') != 'passed' or marker.get('stage') != stage or marker.get('contract') != contract:
            raise RuntimeError('Stale or mismatched transition: ' + stage)
        evidence = marker.get('evidence', {})
        if not evidence:
            raise RuntimeError('No authenticated transition evidence')
        required = [f'{stage}/assessment.json', f'{stage}/checkpoints/best.pt']
        required += [f'{stage}/gates/{g}/success.json' for g in ('schema','gpu_step','overfit','small_scale')]
        if any(item not in evidence for item in required):
            raise RuntimeError('Incomplete authenticated stage dependencies')
        for relative, expected in evidence.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
                raise RuntimeError('Changed transition evidence: ' + relative)
        quality = json.loads((root / stage / 'assessment.json').read_text())
        if quality.get('status') != 'passed' or not quality.get('checks') or not all(v is True for v in quality['checks'].values()):
            raise RuntimeError('Quality assessment is not passed')
        if str(Path(stage) / 'assessment.json') not in evidence:
            raise RuntimeError('Assessment is not authenticated')
        for gate in ('schema','gpu_step','overfit','small_scale'):
            gate_record = json.loads((root/stage/'gates'/gate/'success.json').read_text())
            if gate_record.get('status') != 'passed' or gate_record.get('contract') != contract or not gate_record.get('evidence'):
                raise RuntimeError('Invalid execution gate: '+gate)
            for relative, expected in gate_record['evidence'].items():
                path=(root/relative).resolve()
                if not path.is_relative_to(root) or not path.is_file() or digest(path)!=expected:
                    raise RuntimeError('Stale execution gate evidence: '+relative)


def issue_transition(root, stage, contract, evidence):
    root = Path(root)
    assessment = json.loads((root / stage / 'assessment.json').read_text())
    if assessment.get('status') != 'passed' or not assessment.get('checks') or not all(v is True for v in assessment['checks'].values()):
        raise RuntimeError('A failed/pending assessment cannot issue success')
    required = [f'{stage}/assessment.json', f'{stage}/checkpoints/best.pt']
    required += [f'{stage}/gates/{g}/success.json' for g in ('schema', 'gpu_step', 'overfit', 'small_scale')]
    for item in required:
        if item not in evidence:
            raise RuntimeError('Missing required transition evidence: ' + item)
    for gate in ('schema', 'gpu_step', 'overfit', 'small_scale'):
        item = json.loads((root / stage / 'gates' / gate / 'success.json').read_text())
        if item.get('status') != 'passed' or item.get('contract') != contract:
            raise RuntimeError('Unauthenticated execution gate')
    hashes = {}
    for item in evidence:
        path = (root / item).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Evidence outside stage run')
        hashes[item] = digest(path)
    write_json(root / stage / 'transition_success.json', dict(status='passed', stage=stage,
               contract=contract, evidence=hashes))
