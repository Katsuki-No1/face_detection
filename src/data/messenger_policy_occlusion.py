"""Train-only proposal pool for learned opaque material/scale/placement choices.

The policy chooses a JOINT action from freshly sampled feasible proposals. It
does not generate arbitrary pixels or select the severity stratum. One donor
per action; multiple donors and rotation are deliberately not in this version.
"""
import copy
import hashlib
import math
import random
from collections import Counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.data.landmark_occlusion import ellipse_region_mask
from src.data.messenger_natural_occlusion import select_material, source_geometry_bin
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank


POLICY_BINS = ((.05, .20), (.20, .40), (.60, .80), (.80, 1.00))


def in_bin(fraction, bin_id):
    low, high = POLICY_BINS[bin_id]
    return low <= fraction and (fraction < high or (bin_id == 3 and fraction <= high))


def balanced_bins(indices, *, seed, epoch, probability):
    if not 0 <= probability <= 1:
        raise ValueError('Invalid request probability')
    rng = random.Random(f'{seed}:{epoch}:policy-bins')
    ids = sorted(indices); rng.shuffle(ids)
    ids = ids[:round(len(ids) * probability)]
    order = list(range(4)); rng.shuffle(order)
    bins = [order[i % 4] for i in range(len(ids))]; rng.shuffle(bins)
    return dict(zip(ids, bins))


def propose(patch, foreground, face, bin_id, rng, *, attempts=256):
    """Hard rectangular donor support; clipping and overlap are measured in pixels."""
    if face.dtype != np.bool_ or not face.any():
        return None
    h, w = face.shape; area = int(face.sum())
    yy, xx = np.where(face); cy, cx = yy.mean(), xx.mean()
    low, high = POLICY_BINS[bin_id]
    aspect = patch.shape[1] / patch.shape[0]
    for _ in range(attempts):
        desired = rng.uniform(low, high) * area
        ph = max(2, round(math.sqrt(desired / aspect) * rng.uniform(1., 2.5)))
        pw = max(2, round(ph * aspect))
        if ph > 4*h or pw > 4*w:
            continue
        x = round(cx + rng.uniform(-.65, .65)*(xx.max()-xx.min()+1) - pw/2)
        y = round(cy + rng.uniform(-.65, .65)*(yy.max()-yy.min()+1) - ph/2)
        a, b, c, d = max(0, x), max(0, y), min(w, x+pw), min(h, y+ph)
        if c <= a or d <= b:
            continue
        mask = np.zeros((h, w), bool); mask[b:d, a:c] = True
        fraction = int((mask & face).sum()) / area
        if not in_bin(fraction, bin_id):
            continue
        if foreground is not None:
            fg = cv2.resize(foreground.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)
            if not (fg[b-y:d-y, a-x:c-x].astype(bool) & face[b:d, a:c]).any():
                continue
        resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[b:d, a:c] = resized[b-y:d-y, a-x:c-x]
        return canvas, mask, dict(overlap=fraction, bbox=[x, y, x+pw, y+ph],
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return None


def compose(clean, donor, mask, target):
    """No alpha; preserve labeled locations even at 100% occlusion."""
    if mask.dtype != torch.bool or mask.shape != (1, *clean.shape[-2:]):
        raise ValueError('Binary support required')
    result = torch.where(mask, donor.detach(), clean.detach())
    t = copy.deepcopy(target)
    xy = t['keypoints']; h, w = clean.shape[-2:]
    inside = torch.isfinite(xy).all(-1) & (xy >= 0).all(-1) & (xy <= 1).all(-1)
    states = t['keypoint_visibility'].clone(); states[~inside] = 0
    pixels = torch.nan_to_num(xy).mul(xy.new_tensor([w-1, h-1])).round().long()
    covered = torch.zeros_like(states, dtype=torch.bool)
    ids = (inside & (states > 0)).nonzero().flatten()
    covered[ids] = mask[0, pixels[ids, 1], pixels[ids, 0]]
    states[covered] = 1
    t.update(keypoint_visibility=states, labeled=states > 0, visible=states == 2)
    return result, t, covered


class PolicyOcclusionDataset(Dataset):
    """Only an authenticated TRAIN bank may supply recipients, real crops or donors."""
    def __init__(self, clean, bank, *, split, seed=42, probability=.5,
                 material_candidates=4, placements_per_material=2, attempts=256,
                 real_matching='geometry'):
        if split != 'train' or clean.training or not isinstance(bank, RealOcclusionBank):
            raise ValueError('Policy augmentation is train-only with an authenticated bank')
        if bank.data['adapter_sha256'] != hashlib.sha256(clean.annotation_path.read_bytes()).hexdigest():
            raise ValueError('Train annotation identity differs from bank')
        if material_candidates < 2 or placements_per_material < 2 or attempts < 1:
            raise ValueError('Need multiple materials and placements')
        self.clean = clean; self.bank = bank; self.seed = seed; self.probability = probability
        if real_matching not in ('geometry', 'partition_video_only'):
            raise ValueError('Unknown natural-image matching policy')
        self.real_matching = real_matching
        self.material_candidates = material_candidates; self.placements = placements_per_material
        self.attempts = attempts
        self.eligible = [i for i, s in enumerate(clean.samples) if bank.eligible_sample(s)]
        self.reals = {}
        for i, sample in enumerate(clean.samples):
            if sample['image']['file_name'] not in bank.images:
                raise ValueError('Non-train image in policy dataset')
            if np.any(sample['labeled'] & ~sample['visible']):
                self.reals.setdefault(source_geometry_bin(sample), []).append(i)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.requests = balanced_bins(self.eligible, seed=self.seed, epoch=epoch, probability=self.probability)

    def __len__(self):
        return len(self.clean)

    def pixels(self, i):
        sample = self.clean.samples[i]
        self.bank.read(self.bank.images[sample['image']['file_name']])
        image, target = self.clean[i]
        image = image * torch.tensor(self.clean.std)[:, None, None] + torch.tensor(self.clean.mean)[:, None, None]
        return image.clamp(0, 1), target

    def __getitem__(self, index):
        clean, target = self.pixels(index); sample = self.clean.samples[index]
        recipient = self.bank.images[sample['image']['file_name']]
        row = dict(clean=clean, target=target, candidates=[], real=None,
                   metadata=dict(index=index, recipient_video=recipient['video'],
                                 requested=index in self.requests, reason='unaugmented'))
        if index not in self.requests:
            return row
        bin_id = self.requests[index]; row['metadata']['bin'] = bin_id
        rng = random.Random(f'{self.seed}:{self.epoch}:{index}:policy')
        choices = self.real_choices(sample, recipient['video'])
        if not choices:
            row['metadata']['reason'] = 'no_matched_train_real'; return row
        real_id = rng.choice(choices); row['real'], _ = self.pixels(real_id)
        row['metadata']['real_index'] = real_id
        row['metadata']['real_video'] = self.bank.images[self.clean.samples[real_id]['image']['file_name']]['video']
        face = ellipse_region_mask(*clean.shape[-2:], target['geometry_prior']['predicted_ellipse_crop'].numpy()).astype(bool)
        seen = set()
        for material_id in range(self.material_candidates):
            # Both families are PROPOSED; G chooses the family as part of its action.
            family = ('face', 'non_face')[material_id % 2]
            material = select_material(self.bank, family, recipient, rng)
            if material is None:
                continue
            patch, foreground, provenance = material
            if provenance['donor_video'] == recipient['video'] or provenance['donor_file'] not in self.bank.images:
                raise ValueError('Donor must be a different authenticated train video')
            for _ in range(self.placements):
                placed = propose(patch, foreground, face, bin_id, rng, attempts=self.attempts)
                if placed is None:
                    continue
                donor, mask, placement = placed
                key = (provenance['donor_file'], str(provenance.get('donor_source_box')), placement['mask_sha256'])
                if key in seen:
                    continue
                seen.add(key)
                donor = torch.from_numpy(donor.copy()).permute(2, 0, 1).float()/255
                mask = torch.from_numpy(mask[None].copy())
                h, w = face.shape; x0, y0, x1, y1 = placement['bbox']
                # Recipient RGB is NOT an input: geometry + donor preview avoid
                # leaking recipient pixels hidden by a different candidate.
                features = torch.tensor([x0/w, y0/h, x1/w, y1/h, placement['overlap'],
                    float(family == 'face'), *[float(bin_id == b) for b in range(4)]])
                row['candidates'].append(dict(donor=donor, mask=mask, features=features,
                    preview=F.adaptive_avg_pool2d(donor, (16, 16)),
                    metadata=dict(material_id=material_id, family=family, **provenance, **placement)))
        if len(row['candidates']) < 2:
            row['candidates'] = []; row['metadata']['reason'] = 'insufficient_feasible_actions'
        else:
            row['metadata']['reason'] = None
        return row

    def real_choices(self, sample, recipient_video):
        # RestrictedPolicy has already filtered self.reals to the permitted
        # train-video partition. Neither mode can cross that boundary.
        if getattr(self, 'real_matching', 'geometry') == 'partition_video_only':
            candidates = sorted({j for group in self.reals.values() for j in group})
        else:
            candidates = self.reals.get(source_geometry_bin(sample), [])
        return [j for j in candidates if
                self.bank.images[self.clean.samples[j]['image']['file_name']]['video'] != recipient_video]


def audit_actions(records):
    requested = Counter(); applied = Counter(); reasons = Counter()
    for r in records:
        if not r['requested']:
            continue
        requested[r['bin']] += 1
        if r.get('applied'):
            if not in_bin(r['overlap'], r['bin']):
                raise ValueError('Selected action escaped severity stratum')
            applied[r['bin']] += 1
        else:
            reasons[r['reason']] += 1
    counts = [applied[i] for i in range(4)]
    return dict(requested_per_bin=[requested[i] for i in range(4)], applied_per_bin=counts,
                skips=dict(reasons), equal_applied_distribution=bool(min(counts)>0 and max(counts)-min(counts)<=1),
                production_acceptance='not established; no production gate issued')
