"""Isolated selected-k adversarial contract; completed non-adversarial runs stay pinned."""
import json
from pathlib import Path
import torch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import (
    ROOT, authenticate as authenticate_parent, source_hashes,
)
from keypoint_comparison1.landmark_detection.geometry_prior_training import PriorUpdateRuntime, file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import ConditionalFeatureDiscriminator

D_CONTRACT=dict(channels=384,hidden=64,seed=42,lr=1e-4,betas=[.9,.999],weight_decay=.01,
                adversarial_weight=.01,reconstruction_weight=.1)


def authenticate(config):
    spec=json.loads(Path(config).read_text())
    if spec.get('schema')!='geometry-prior-adversarial-launch-v1' or not spec.get('adversarial'):
        raise ValueError('Expected separate adversarial configuration')
    parent,parent_identity=authenticate_parent(ROOT/spec['parent_launch_config'])
    selection_path=ROOT/spec['selection_manifest']
    if file_hash(selection_path)!=spec['selection_sha256']:
        raise ValueError('Frozen validation selection changed')
    selection=json.loads(selection_path.read_text())
    if (selection.get('status')!='success' or selection.get('selected_on_split')!='val'
            or selection.get('test_used') is not False or selection['selected_k']!=parent['prior']['k']
            or selection['selected_prior']!=parent['prior']):
        raise ValueError('Adversarial k must be the frozen validation-only winner')
    winner=next(row for row in selection['candidates'] if row['k']==selection['selected_k'])
    if winner['identity']!=parent_identity:
        raise ValueError('Selected parent identity changed')
    if (spec['parent_config']!=parent['parent_config'] or spec['prior']!=parent['prior']
            or spec['train_crops']!=parent['train_crops'] or spec['val_crops']!=parent['val_crops']
            or spec['val_annotations']!=parent['val_annotations'] or spec['discriminator']!=D_CONTRACT
            or spec['architecture']!='messenger_geometry_prior_selected_k_adversarial_seed42'
            or spec['arm']!='selected_k_adversarial'):
        raise ValueError('Matched initialization/data or discriminator contract changed')
    for p,expected in spec['artifacts'].items():
        if file_hash(ROOT/p)!=expected:raise ValueError(f'Changed artifact: {p}')
    if not all(spec['artifacts'].get(p)==h for p,h in parent['artifacts'].items()):
        raise ValueError('Parent artifact inventory omitted')
    sources=source_hashes()
    for name in ['messenger_geometry_adversarial_contract.py','train_messenger_geometry_adversarial.py',
                 'preflight_messenger_geometry_adversarial.py','run_messenger_geometry_adversarial_gate.py']:
        p=ROOT/'scripts'/name;sources[str(p.relative_to(ROOT))]=file_hash(p)
    return spec,dict(config_sha256=file_hash(config),artifacts=spec['artifacts'],sources=sources,
        selection_sha256=spec['selection_sha256'])


class AdversarialRuntime(PriorUpdateRuntime):
    def __init__(self,model,criterion,optimizer,**kwargs):
        # D initialization must not alter the matched generator/data RNG stream.
        device=next(model.parameters()).device
        devices=[device.index or 0] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(D_CONTRACT['seed'])
            d=ConditionalFeatureDiscriminator(D_CONTRACT['channels'],D_CONTRACT['hidden']).to(device)
        dopt=torch.optim.AdamW(d.parameters(),lr=D_CONTRACT['lr'],betas=tuple(D_CONTRACT['betas']),
            weight_decay=D_CONTRACT['weight_decay'])
        super().__init__(model,criterion,optimizer,discriminator=d,discriminator_optimizer=dopt,
            adversarial_weight=D_CONTRACT['adversarial_weight'],reconstruction_weight=D_CONTRACT['reconstruction_weight'],**kwargs)
