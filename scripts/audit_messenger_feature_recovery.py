#!/usr/bin/env python3
"""Paired validation feature audit; diagnostic GT crops, never a deployment score."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import PairedOcclusionConfig
from keypoint_comparison1.landmark_detection.messenger_recovery import patch_occlusion_ratios
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import load_landmark_checkpoint, make_dataset
from src.data.landmark_occlusion import apply_landmark_occlusion_numpy, transform_ellipse_affine


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def errors(value, target, scale):
    """All distances use FP32 even if callers used autocast."""
    value, target = value.float(), target.float()
    return {
        'cosine_distance': 1 - F.cosine_similarity(value, target, dim=-1),
        'mse': (value - target).square().mean(-1),
        'scaled_huber': F.huber_loss((value - target) / scale.float(),
                                    torch.zeros_like(value), reduction='none', delta=1.).mean(-1),
    }


class PairedCrops(Dataset):
    """Every model sees the same 3 corruptions for each validation crop."""
    kinds = ('rectangle', 'irregular', 'boundary')

    def __init__(self, source, paired):
        self.source = source
        self.cfg = PairedOcclusionConfig.from_mapping(paired).landmark_mapping()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        clean, target = self.source[index]
        mean = np.asarray(self.source.mean, dtype=np.float32)
        std = np.asarray(self.source.std, dtype=np.float32)
        pixels = clean.permute(1, 2, 0).numpy() * std + mean
        # Recover the actual uint8 crop; preserve exact clean input independently.
        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        ellipse = transform_ellipse_affine(target['gt_ellipse'].numpy(), target['affine'].numpy())
        points = target['keypoints'].numpy() * (self.source.input_size - 1)
        outputs, masks, applied = [], [], []
        for kind in self.kinds:
            seed_text = f"42:{target['annotation_id']}:feature_audit:{kind}"
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'big')
            image, mask, _, meta = apply_landmark_occlusion_numpy(
                pixels, face_ellipse=ellipse, keypoints=points,
                keypoint_visibility=target['keypoint_visibility'].numpy(),
                config=self.cfg, rng=random.Random(seed), force=True, force_kind=kind)
            normalized = (image.astype(np.float32) / 255 - mean) / std
            outputs.append(torch.from_numpy(normalized.transpose(2, 0, 1).copy()))
            masks.append(torch.from_numpy(mask.copy()))
            applied.append(bool(meta['applied']))
        return clean, torch.stack(outputs), torch.stack(masks), torch.tensor(applied), target['annotation_id']


class Accumulator:
    def __init__(self):
        self.data = {}

    def add(self, prefix, raw, alternatives, mask):
        count = int(mask.sum())
        if not count:
            return
        for metric, values in raw.items():
            base = values[mask].double()
            for branch, result in alternatives.items():
                after = result[metric][mask].double()
                delta = base - after
                key = '/'.join((*prefix, branch, metric))
                row = self.data.setdefault(key, dict(count=0, raw_sum=0., after_sum=0.,
                                                     improved=0, worsened=0, tied=0,
                                                     delta_sum=0., delta_square_sum=0.))
                row['count'] += count
                row['raw_sum'] += base.sum().item()
                row['after_sum'] += after.sum().item()
                row['delta_sum'] += delta.sum().item()
                row['delta_square_sum'] += delta.square().sum().item()
                row['improved'] += int((delta > 1e-7).sum())
                row['worsened'] += int((delta < -1e-7).sum())
                row['tied'] += int((delta.abs() <= 1e-7).sum())

    def finish(self):
        out = {}
        for key, row in self.data.items():
            n = row['count']
            before, after = row['raw_sum'] / n, row['after_sum'] / n
            out[key] = dict(patches=n, raw_error=before, branch_error=after,
                            error_reduction=before-after,
                            relative_error_reduction=(before-after)/before if before > 1e-12 else None,
                            improved_fraction=row['improved']/n,
                            worsened_fraction=row['worsened']/n,
                            tied_fraction=row['tied']/n)
        return out


def branches(block, feature):
    recovered, diag = block(feature)
    original = diag['original_tokens']
    if 'candidate_tokens' in diag:
        candidate = diag['candidate_tokens']
    else:
        # v1 has a residual projection, not a standalone decoder. Report its
        # equivalent full-gate proposal, not unprojected Messenger state as R.
        candidate = original + block.correction_projection(diag['messenger_tokens'] - original)
    return original, {'candidate': candidate, 'output': diag['recovered_tokens']}, diag


@torch.inference_mode()
def evaluate(model, teacher, loader, scale, device):
    stats = Accumulator()
    input_digest = hashlib.sha256()
    counts = dict(crops=0, applied={k: 0 for k in PairedCrops.kinds})
    gate_sum = gate_n = 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch_idx, (clean, occ, masks, applied, ids) in enumerate(loader):
        input_digest.update(ids.numpy().tobytes())
        input_digest.update(clean.numpy().tobytes())
        input_digest.update(occ.numpy().tobytes())
        input_digest.update(masks.numpy().tobytes())
        counts['crops'] += len(clean)
        clean = clean.to(device)
        teacher_clean = teacher(clean).flatten(2).transpose(1, 2).float()
        student_clean_map = model.backbone(clean)
        student_clean = student_clean_map.flatten(2).transpose(1, 2).float()
        _, clean_branches, _ = branches(model.messenger_recovery, student_clean_map)
        for ref, target in [('fixed_teacher', teacher_clean), ('student_clean', student_clean)]:
            raw = errors(student_clean, target, scale)
            outputs = {k: errors(v, target, scale) for k, v in clean_branches.items()}
            stats.add((ref, 'clean', 'all'), raw, outputs, torch.ones_like(raw['mse'], dtype=torch.bool))
        for kind_idx, kind in enumerate(PairedCrops.kinds):
            counts['applied'][kind] += int(applied[:, kind_idx].sum())
            features = model.backbone(occ[:, kind_idx].to(device))
            original, proposals, diag = branches(model.messenger_recovery, features)
            ratios = patch_occlusion_ratios(masks[:, kind_idx].to(device), patch_size=16)
            subsets = {'all': torch.ones_like(ratios, dtype=torch.bool),
                       'masked': ratios > 0, 'unmasked': ratios == 0,
                       'fully_masked': ratios >= 1 - 1e-7}
            gate = diag.get('adoption_gate', diag['gate_alpha'])
            gate_sum += gate[ratios > 0].double().sum().item()
            gate_n += int((ratios > 0).sum())
            for ref, target in [('fixed_teacher', teacher_clean), ('student_clean', student_clean)]:
                raw = errors(original, target, scale)
                outputs = {k: errors(v, target, scale) for k, v in proposals.items()}
                for subset, mask in subsets.items():
                    stats.add((ref, kind, subset), raw, outputs, mask)
                    stats.add((ref, 'pooled', subset), raw, outputs, mask)
        if batch_idx % 20 == 0:
            print(f'  crops {counts["crops"]}/{len(loader.dataset)}', flush=True)
    torch.cuda.synchronize(device)
    return dict(metrics=stats.finish(), counts=counts, paired_inputs_sha256=input_digest.hexdigest(),
                masked_adoption_mean=gate_sum/max(1, gate_n),
                runtime_seconds=time.perf_counter()-started,
                peak_vram_mib=torch.cuda.max_memory_allocated(device)/1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['v1', 'direct', 'gram', 'option1', 'local', 'option3', 'option3_consistency', 'combined_option1'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    config_root = ROOT/'keypoint_comparison1/configs'
    direct = load_landmark_config(config_root/'continuous_recovery_direct.yaml')
    canonical = ROOT/'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/val.json'
    assert sha(canonical) == '0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0'
    canonical_data = json.loads(canonical.read_text())
    assert len(canonical_data['images']) == 530 and len(canonical_data['annotations']) == 882
    raw = copy.deepcopy(direct.raw)
    ev = raw['data']['evaluation_sets']['v2_val']
    raw['data']['annotations']['val'] = ev['annotation']
    raw['data'].setdefault('image_roots', {})['val'] = ev['root']
    raw['data'].pop('validation_detector_crop_cache', None)
    source = make_dataset(raw, split='val', training=False)
    allowed = {im['file_name'] for im in canonical_data['images']}
    assert all(s['image']['file_name'] in allowed for s in source.samples)
    assert len(source) == 840, f'Unexpected validation GT-crop count: {len(source)}'
    pairs = PairedCrops(source, raw['data']['augmentation']['paired_occlusion'])
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    teacher_cfg = copy.deepcopy(direct.raw)
    teacher_cfg['model']['continuous_recovery']['enabled'] = False
    teacher_model, _ = build_landmark_model(teacher_cfg, variant='query_refinement', initialize_from_detector=False)
    init = direct.raw['train']['finetune_checkpoint']
    assert sha(init['path']) == init['sha256']
    load_landmark_checkpoint(teacher_model, init['path'], prefer_ema=True, strict=True)
    teacher = teacher_model.backbone.to(device).eval()
    del teacher_model
    scale_path = direct.output_root/'query_refinement/continuous_teacher_provenance.json'
    scale_prov = json.loads(scale_path.read_text())
    assert scale_prov['split'] == 'train' and sha(scale_prov['annotation_path']) == scale_prov['annotation_sha256']
    scale = torch.tensor(scale_prov['channel_scale'], device=device)
    common = dict(schema_version=1, split='val', seed=42, diagnostic_only=True,
                  topology='HEAD-free ordered full-image FACE/ellipse -> Face-crop DINO -> Messenger -> Query Refinement',
                  scope='GT-ellipse-crop feature diagnostic; no detector or HEAD added; not primary deployment evaluation',
                  clean_definition='same validation crop before synthetic occlusion; existing natural occlusion is retained',
                  precision='float32, no AMP, no TF32', input_size=256, patch_size=16,
                  coordinate_convention='same 256x256 long-axis-square crop, x right/y down; no geometric or color augmentation',
                  annotation=dict(canonical_path=str(canonical), canonical_sha256=sha(canonical),
                                  adapter_path=str(source.annotation_path), adapter_sha256=sha(source.annotation_path),
                                  canonical_images=530, canonical_heads=882, diagnostic_face_crops=len(source)),
                  teacher=dict(path=init['path'], sha256=init['sha256'], state='EMA-preferred baseline warm-start backbone'),
                  train_only_scale=dict(path=str(scale_path), sha256=sha(scale_path), method=scale_prov['method']),
                  metric_definition='lower is better; improved if raw_error - branch_error > 1e-7; pooled is patch-weighted across all three corruption kinds',
                  reference_definition=dict(fixed_teacher='baseline clean features used by v2 training',
                                            student_clean='same selected student backbone on clean input; controls feature-coordinate drift'),
                  candidate_definition=dict(v1='X + correction_projection(M-X), equivalent proposal at gate=1',
                                            v2='decoder(M), standalone pre-gate proposal'),
                  actual_output_definition='features actually passed to unchanged Query Refinement',
                  paired_augmentation=raw['data']['augmentation']['paired_occlusion'],
                  corruption_protocol='each GT crop gets rectangle, irregular and boundary with force=True; seed sha256(42:annotation_id:feature_audit:kind)',
                  thresholds='no Face/keypoint threshold used in feature distances; no selection on these diagnostics',
                  checkpoint_selection='existing validation-selected best_conditional_semantic_macro_f1; no reselection',
                  batch_size=args.batch_size, script_sha256=sha(__file__),
                  git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()))
    all_results = {}
    expected_digest = None
    for name in args.models:
        cfg_path = config_root/('dinov3_vits16_face_pose_drive_v2_messenger_recovery.yaml' if name == 'v1' else f'continuous_recovery_{name}.yaml')
        cfg = load_landmark_config(cfg_path)
        checkpoint = cfg.output_root/'query_refinement/checkpoints/best_conditional_semantic_macro_f1.pt'
        print(f'START {name}', flush=True)
        model, _ = build_landmark_model(cfg.raw, variant='query_refinement', initialize_from_detector=False)
        info = load_landmark_checkpoint(model, checkpoint, prefer_ema=True, strict=True)
        model.to(device).eval()
        if name != 'v1':
            assert torch.allclose(model.messenger_recovery.channel_scale, scale, atol=1e-6, rtol=1e-6)
        result = evaluate(model, teacher, loader, scale, device)
        assert all(n > 0 for n in result['counts']['applied'].values())
        assert result['counts']['crops'] == len(source)
        json.dumps(result, allow_nan=False)
        if expected_digest is None:
            expected_digest = result['paired_inputs_sha256']
        assert result['paired_inputs_sha256'] == expected_digest
        payload = dict(**common, status='complete', model=name,
                       config=dict(path=str(cfg_path), sha256=sha(cfg_path), resolved_sha256=hashlib.sha256(json.dumps(cfg.raw,sort_keys=True,default=str).encode()).hexdigest()),
                       checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint), epoch=info['epoch'], prefer_ema=True), **result)
        out = args.output/f'{name}.json'
        out.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        all_results[name] = dict(path=str(out), sha256=sha(out), epoch=info['epoch'],
                                 metrics=result['metrics'], masked_adoption_mean=result['masked_adoption_mean'])
        print('DONE', name, json.dumps(result['metrics']['fixed_teacher/pooled/masked/output/cosine_distance']), flush=True)
        del model
        torch.cuda.empty_cache()
    comparison = args.output/'comparison.json'
    comparison.write_text(json.dumps(dict(**common, status='complete', paired_inputs_sha256=expected_digest, models=all_results),indent=2,allow_nan=False)+'\n')
    (args.output/'success.json').write_text(json.dumps(dict(status='success', meaning='audit completed and finite; does not assert feature improvement', models=list(all_results), comparison_sha256=sha(comparison), identical_paired_inputs=True),indent=2)+'\n')


if __name__ == '__main__':
    main()
