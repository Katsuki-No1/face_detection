#!/usr/bin/env python3
"""Bounded train-only center optimization, followed by frozen validation diagnostics.

Not a production launcher, four-gate marker, or model promotion. CPU only; no
test samples, no threshold/epoch selection. Keeps one final prototype snapshot.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from keypoint_comparison1.landmark_detection.dataset import CropAugmentation, TopDownFaceLandmarkDataset
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from scripts.evaluate_messenger_geometry_prior import load_model
from scripts.diagnose_messenger_occlusion_observer import binary_metrics, write
from src.models.messenger_prototype_observer import (
    attach_learnable_prototypes, prototype_loss, sparse_point_samples,
)


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == "pipeline.bank.centers" or not isinstance(value, torch.Tensor):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def target_fields(target):
    states = torch.where(target['original_labeled'],
                         torch.where(target['original_visible'], 2, 1), 0)
    return target['keypoints'], states, target['geometry_prior']['geometry'], target['geometry_prior']['support']


def run(args):
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    if cfg['schema'] != 'messenger-prototype-smoke-v1' or not 1 <= cfg['train_crops'] <= 256 or not 1 <= cfg['updates'] <= 100:
        raise ValueError('Only a bounded implementation smoke is supported')
    if cfg['device'] != 'cpu' or not 1 <= cfg['threads'] <= 2:
        raise ValueError('Smoke must not compete for the training GPU')
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed'])
    args.output.mkdir(parents=True, exist_ok=False)
    spec_path = ROOT / cfg['parent_config']; spec = json.loads(spec_path.read_text())
    production = ROOT / spec['output_root']
    done = json.loads((production / 'training_success.json').read_text()); identity = done['identity']
    if done['status'] != 'complete' or identity['config_sha256'] != file_hash(spec_path):
        raise ValueError('Parent completion/config identity changed')
    for path, expected in identity['artifacts'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError('Changed parent input: ' + path)
    drift = {p: dict(historical=h, current=file_hash(ROOT/p) if (ROOT/p).is_file() else None)
             for p, h in identity['sources'].items() if not (ROOT/p).is_file() or file_hash(ROOT/p) != h}
    if any(p.endswith(('/geometry_prior_recovery.py', '/geometry_prior_inputs.py', '/continuous_recovery.py')) for p in drift):
        raise ValueError('Core inference source drift requires review')
    checkpoint = production / 'checkpoints/best.pt'
    if file_hash(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    parent, provenance = load_model(spec, identity, checkpoint, torch.device('cpu'))
    model = attach_learnable_prototypes(parent, enabled=True, logit_temperature=cfg['logit_temperature'])
    del parent
    model.train()
    if not torch.equal(model.bank.centers, model.bank.fixed.centers):
        raise AssertionError('Initialization must equal original k-means centers')
    data = spec['parent_config']['data']
    shared = dict(image_root=data['root'], training=False, input_size=data['input_size'],
                  category_ids=data['category_ids'], require_visible_keypoint=False,
                  augmentation=CropAugmentation(**data['augmentation']), mean=data['mean'], std=data['std'],
                  geometry_prior_grid_size=(16, 16), seed=cfg['seed'])
    train = TopDownFaceLandmarkDataset(annotation_path=data['annotations']['train'],
        detector_crop_cache=ROOT/spec['train_crops'], **shared)
    val = TopDownFaceLandmarkDataset(annotation_path=ROOT/spec['val_annotations'],
        detector_crop_cache=ROOT/spec['val_crops'], **shared)
    video_maps = {}
    for split in ('train', 'val'):
        source = data['continuous_source_inventory'][split]
        if file_hash(ROOT/source['path']) != source['sha256']:
            raise ValueError('Source annotation identity changed')
        native = json.loads((ROOT/source['path']).read_text())
        video_maps[split] = {im['id']: im['source_video_id'] for im in native['images']}
    if set(video_maps['train'].values()) & set(video_maps['val'].values()):
        raise ValueError('Train/validation video overlap')
    indices = sorted(range(len(train)), key=lambda i: hashlib.sha256(
        f"{cfg['seed']}:{train.samples[i]['annotation']['id']}:prototype-smoke".encode()).hexdigest())[:cfg['train_crops']]
    tensors = [[] for _ in range(5)]; manifest = []; image_hashes = {}
    for n, index in enumerate(indices):
        image, target = train[index]; xy, states, geometry, support = target_fields(target)
        with torch.no_grad():
            features = model.pipeline.observer(image[None]).squeeze(0)
        for rows, value in zip(tensors, (features, xy, states, geometry, support)):
            rows.append(value)
        path = Path(data['root']) / target['file_name']
        image_hashes[str(path)] = file_hash(path)
        manifest.append(dict(index=index, annotation_id=int(target['annotation_id']), image_id=int(target['image_id']),
                             video=video_maps['train'][int(target['image_id'])]))
        if (n+1) % 32 == 0:
            print(json.dumps(dict(phase='train_features', crops=n+1)), flush=True)
    features, xy, states, geometry, support = [torch.stack(rows) for rows in tensors]
    input_hw = (data['input_size'], data['input_size'])
    with torch.no_grad():
        initial = model.bank(features, geometry, support)
        _, _, counts = sparse_point_samples(initial['occlusion_logits'], xy, states, support, input_hw=input_hw)
    if counts['positives'] == 0 or counts['negatives'] == 0:
        raise ValueError('Both natural classes required')
    positive_weight = counts['negatives'] / counts['positives']
    loss_options = dict(input_hw=input_hw, positive_weight=positive_weight, anchor_weight=cfg['anchor_weight'])
    frozen_before = frozen_digest(model)
    optimizer = model.optimizer(lr=cfg['learning_rate'])
    with torch.no_grad():
        initial_loss = float(prototype_loss(model.bank, initial, xy, states, support, **loss_options)['loss'])
    rng = torch.Generator().manual_seed(cfg['seed']); history = []; gradient_max = 0.
    for step in range(cfg['updates']):
        batch = torch.randperm(len(features), generator=rng)[:cfg['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        prior = model.bank(features[batch], geometry[batch], support[batch])
        losses = prototype_loss(model.bank, prior, xy[batch], states[batch], support[batch], **loss_options)
        losses['loss'].backward()
        for name, parameter in model.named_parameters():
            if name != 'pipeline.bank.centers' and (parameter.requires_grad or parameter.grad is not None):
                raise AssertionError('Gradient escaped centers: ' + name)
        grad = model.bank.centers.grad
        if grad is None or not torch.isfinite(grad).all() or not torch.isfinite(losses['loss']):
            raise AssertionError('Missing/nonfinite center gradient or loss')
        gradient_max = max(gradient_max, float(grad.abs().max()))
        optimizer.step()
        history.append(dict(step=step+1, loss=float(losses['loss'].detach()), counts=losses['counts']))
    with torch.no_grad():
        final_loss = float(prototype_loss(model.bank, model.bank(features, geometry, support), xy, states, support, **loss_options)['loss'])
    if frozen_before != frozen_digest(model) or gradient_max == 0 or torch.equal(model.bank.centers, model.bank.fixed.centers):
        raise AssertionError('Freeze/update verification failed')
    print(json.dumps(dict(phase='optimized', initial_loss=initial_loss, final_loss=final_loss)), flush=True)
    metrics = {group: {arm: [[], [], []] for arm in ('fixed', 'learned')} for group in ('all_in_crop', 'inside_support')}
    val_counts = Counter()
    with torch.no_grad():
        for index in range(len(val)):
            image, target = val[index]; point, st, geo, sup = target_fields(target)
            feature = model.pipeline.observer(image[None])
            outputs = {'fixed': model.bank.fixed(feature, geo[None], sup[None]),
                       'learned': model.bank(feature, geo[None], sup[None])}
            for arm, prior in outputs.items():
                margin = model.bank.thresholds[prior['selected_center']] - prior['score']
                effective_margin = torch.where(sup[None], margin, -3.)
                for group, allowed in [('all_in_crop', torch.ones_like(sup)), ('inside_support', sup)]:
                    values, labels, c = sparse_point_samples(effective_margin, point[None], st[None], allowed[None], input_hw=input_hw)
                    rows = metrics[group][arm]
                    rows[0].extend(labels.tolist()); rows[1].extend(values.tolist()); rows[2].extend((values > 0).tolist())
                    if arm == 'learned' and group == 'inside_support': val_counts.update(c)
            path = Path(data['root']) / target['file_name']; image_hashes[str(path)] = file_hash(path)
            if (index+1) % 128 == 0:
                print(json.dumps(dict(phase='validation', crops=index+1, total=len(val))), flush=True)
    results = {group: {arm: binary_metrics(*rows) for arm, rows in arms.items()} for group, arms in metrics.items()}
    source_paths = [args.config, Path(__file__), ROOT/'src/models/messenger_prototype_observer.py',
                    ROOT/'tests/test_messenger_prototype_observer.py']
    contract = dict(config=cfg, config_sha256=file_hash(args.config), parent=provenance,
                    prototype_origin=model.bank.origin_sha256, positive_weight=positive_weight,
                    source_sha256={str(p.resolve()): file_hash(p) for p in source_paths},
                    historical_parent_source_drift=drift,
                    current_parent_source_sha256={p: file_hash(ROOT/p) for p in identity['sources']},
                    annotation_inventory={s: data['continuous_source_inventory'][s] for s in ('train', 'val')},
                    parent_artifacts=identity['artifacts'])
    torch.save(dict(schema='prototype-smoke-checkpoint-v1', contract=contract, step=cfg['updates'],
                    prototypes=model.bank.state_dict(), optimizer=optimizer.state_dict(), rng=rng.get_state()),
               args.output/'prototypes_last.pt')
    write(args.output/'train_manifest.json', manifest)
    write(args.output/'image_hashes.json', image_hashes)
    write(args.output/'loss_history.json', history)
    report = dict(status='completed_implementation_smoke_not_gate', production_training='not run',
        test_used=False, validation_used_for_selection=False, contract=contract,
        topology=cfg['topology'], train_population=len(train), train_crops=len(indices), train_counts=counts,
        train_videos=sorted({row['video'] for row in manifest}), validation_crops=len(val), validation_counts=dict(val_counts),
        train_loss_before=initial_loss, train_loss_after=final_loss, validation_observer=results,
        trainable_parameters=model.bank.centers.numel(), optimized_names=['pipeline.bank.centers'],
        frozen_tensors_sha256_before=frozen_before, frozen_tensors_sha256_after=frozen_digest(model),
        gradient_max=gradient_max, center_delta_l2=float((model.bank.centers-model.bank.fixed.centers).detach().norm()),
        checkpoint_sha256=file_hash(args.output/'prototypes_last.pt'), elapsed_seconds=time.perf_counter()-started,
        limitations=['Sparse point O diagnostics, not dense natural occlusion segmentation',
                     'Bounded 128-crop fit is not production or a current V3 baseline comparison',
                     'g remains frozen; improved O alone does not establish downstream benefit'])
    write(args.output/'metrics.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'train_loss_before', 'train_loss_after', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/training_plans/messenger_prototype_observer_smoke.json')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
