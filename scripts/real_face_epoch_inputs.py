"""Deterministic full-image epoch inputs shared by both pending FOR arms.

No model, training, or gate authorization is implied by this preparation layer.
Recipe identities are independent of iteration order, worker count and model RNG.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scripts.real_face_full_image import FullImageFaceBank, compose_full_image

MANIFEST_SHA = '3f9f1cd404ac672225373b797c3b4bfc7086404cf686e8852713345c34951364'
ADAPTER_SHA = '2fac1a20b089c801099bed0b439db3ed612d2ea812b5eab21a574f473f981cfa'
RECIPE_VERSION = 'real-face-shared-epoch-v1'


def recipe_seed(seed, epoch, image_id):
    if type(seed) is not int or type(epoch) is not int or epoch < 1 or type(image_id) is not int:
        raise ValueError('Integer seed/image ID and positive epoch required')
    payload = json.dumps([RECIPE_VERSION, seed, epoch, image_id], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], 'big')


def normalized_annotations(annotations):
    result = deepcopy(annotations)
    names = ['left_eye', 'right_eye', 'nose_tip', 'left_mouth_corner', 'right_mouth_corner']
    for ann in result:
        states = np.asarray(ann['keypoints']).reshape(5, 3)[:, 2]
        attrs = ann.setdefault('attributes', {})
        attrs['keypoint_occluded'] = (states == 1).tolist()
        attrs['occluded_keypoint_names'] = [n for n, s in zip(names, states) if s == 1]
        attrs['occluded_keypoint_count'] = int((states == 1).sum())
        ann['num_keypoints'] = int((states > 0).sum())
    return result


class SharedEpochInputs:
    def __init__(self, manifest):
        manifest = Path(manifest)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA:
            raise ValueError('Scheduled face bank identity changed')
        data = json.loads(manifest.read_text())
        adapter = Path(data['adapter_path'])
        if hashlib.sha256(adapter.read_bytes()).hexdigest() != ADAPTER_SHA:
            raise ValueError('Scheduled train adapter changed')
        self.bank = FullImageFaceBank(manifest, MANIFEST_SHA,
            annotation_path=adapter, image_root=data['image_root'])
        self.coco = json.loads(adapter.read_text())
        self.images = sorted(self.coco['images'], key=lambda row: row['id'])
        self.annotations = defaultdict(list)
        for ann in self.coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        if len(self.images) != 4237 or len({i['id'] for i in self.images}) != 4237:
            raise ValueError('Full train image inventory changed')
        if len({i['metadata']['source_video_id'] for i in self.images}) != 44:
            raise ValueError('Full train group inventory changed')
        for image in self.images:
            if image['metadata']['split'] != 'train' or image['file_name'] not in self.bank.images:
                raise ValueError('Non-train image in epoch inputs')

    def render(self, image, *, seed, epoch, load_unmodified=True):
        """One input; clean/skipped inputs remain in the training population."""
        local_seed = recipe_seed(seed, epoch, image['id'])
        rng = random.Random(local_seed)
        anns = self.annotations[image['id']]
        eligible = [a for a in anns if self.bank.eligible_sample(dict(image=image, annotation=a))]
        requested = bool(eligible) and rng.random() < .30
        meta = dict(recipe_version=RECIPE_VERSION, seed=seed, epoch=epoch,
            image_id=image['id'], image_file=image['file_name'],
            source_video=image['metadata']['source_video_id'],
            source_sha256=image['metadata']['sha256'], recipe_seed=str(local_seed),
            eligible_recipients=len(eligible), requested=requested, applied=False,
            request_probability=.30, post_composition_queries_required=True)
        rgb = None
        if requested or load_unmodified:
            rgb = self.bank.read(self.bank.images[image['file_name']])
        if requested:
            target = rng.choice(eligible)
            output, updated, mask, composition = compose_full_image(rgb, image=image,
                annotations=anns, target_id=target['id'], bank=self.bank, rng=rng)
            if not np.array_equal(output[~mask], rgb[~mask]):
                raise ValueError('Composition changed exterior pixels')
            if not composition['applied'] and (mask.any() or updated != anns):
                raise ValueError('Skipped composition changed supervision')
            meta.update(applied=composition['applied'], composition=composition)
            rgb, anns = output, updated
        else:
            meta['reason'] = 'not_requested' if eligible else 'no_eligible_recipient'
        anns = normalized_annotations(anns)
        if meta['applied']:
            target_ann = next(a for a in anns if a['id'] == meta['composition']['recipient_id'])
            meta['covered_target_keypoints'] = target_ann['attributes']['occluded_keypoint_count']
        meta['recipient_annotations_sha256'] = hashlib.sha256(
            json.dumps(anns, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if rgb is not None:
            meta['rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
        return rgb, anns, meta
