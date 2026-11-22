"""HEAD-free prior validation: predicted FACE crops, GT Head normalization.

Only constant-size semantic counters are retained during epoch validation.
Detailed deployment reports and test evaluation remain separate jobs.
"""
from collections import Counter, defaultdict
import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.head_anchored_keypoint_cache import CONTRACT
from .dataset import collate_landmark_batch
from .evaluation import predictions_to_original


def head_validation_annotations(native):
    """Preserve full image inventory and audit excluded native Head attributes."""
    counts = Counter()
    annotations = []
    for head in native['annotations']:
        if not head.get('attribute_valid', False):
            counts['invalid_or_ambiguous_attributes'] += 1
            continue
        ellipse = np.asarray(head['ellipse'], dtype=float)
        points = np.asarray(head['keypoints'], dtype=float)
        states = np.asarray(head['keypoint_states'])
        if ellipse.shape != (5,) or not np.isfinite(ellipse).all() or min(ellipse[2:4]) <= 0:
            counts['not_face_positive'] += 1
            continue
        if points.shape != (5, 2) or states.shape != (5,) or not np.isin(states, [0, 1, 2]).all():
            raise ValueError('Invalid native keypoint schema')
        if not np.isfinite(points[states > 0]).all():
            raise ValueError('Non-finite supervised native point')
        if not (states > 0).any():
            counts['all_missing'] += 1
            continue
        row = copy.deepcopy(head)
        row.update(category_id=2, normalization_bbox_source='GT Head',
                   keypoints=np.column_stack([points, states]).reshape(-1).tolist(),
                   num_keypoints=int((states > 0).sum()))
        annotations.append(row)
    if len({a['id'] for a in native['annotations']}) != len(native['annotations']):
        raise ValueError('Duplicate native Head IDs')
    report = dict(images=len(native['images']), native_heads=len(native['annotations']),
                  eligible_before_FACE_matching=len(annotations), excluded=dict(counts))
    return dict(images=copy.deepcopy(native['images']), annotations=annotations,
                categories=[dict(id=2, name='face')]), report


def assign_validation_crops(native, adapted, predictions):
    from tools.evaluate_deim_query_baseline_aligned import associate_predictions_to_heads
    image_ids = {int(i['id']) for i in native['images']}
    if set(map(int, predictions)) != image_ids:
        raise ValueError('Incomplete full validation prediction image inventory')
    heads = defaultdict(list)
    for row in native['annotations']:
        heads[int(row['image_id'])].append(row)
    eligible = {int(a['id']) for a in adapted['annotations']}
    result = {}
    for iid in sorted(image_ids):
        detections = predictions.get(iid, predictions.get(str(iid)))
        matches = associate_predictions_to_heads(heads[iid], detections, head_roi_scale=1.20)
        result[str(iid)] = [dict(detections[j], assigned_annotation_id=heads[iid][h]['id'],
                                source_prediction_index=j)
                            for h, j in matches.items() if int(heads[iid][h]['id']) in eligible]
    return result


class ConditionalSemanticCounters:
    """Equal-weight mean of five F1s, inclusive presence .70 and NME .10."""
    def __init__(self):
        self.tp = np.zeros(5, dtype=np.int64)
        self.predicted = np.zeros(5, dtype=np.int64)
        self.gt = np.zeros(5, dtype=np.int64)
        self.instances = 0

    def add(self, points, scores, targets):
        for predicted, score, target in zip(points, scores, targets, strict=True):
            labeled = np.asarray(target['original_labeled'], dtype=bool)
            if not labeled.any():
                continue
            box = np.asarray(target['bbox_xyxy'], dtype=float)
            side = 1.20 * max(box[2] - box[0], box[3] - box[1])
            truth = np.asarray(target['original_keypoints'], dtype=float)
            if side <= 0 or not np.isfinite(side) or not np.isfinite(predicted).all() or not np.isfinite(score).all():
                raise ValueError('Invalid validation prediction or GT Head normalization')
            present = score >= .70 - 1e-7
            correct = np.linalg.norm(predicted - truth, axis=1) / side <= .10
            self.tp += present & labeled & correct
            self.predicted += present
            self.gt += labeled
            self.instances += 1

    def compute(self):
        denominator = self.predicted + self.gt
        f1 = np.divide(2 * self.tp, denominator, out=np.zeros(5), where=denominator > 0)
        return dict(**{'conditional_semantic_macro_f1_nme_0.10': float(f1.mean())},
                    semantic_tp=self.tp.tolist(), semantic_fp=(self.predicted-self.tp).tolist(),
                    semantic_fn=(self.gt-self.tp).tolist(), semantic_f1=f1.tolist(),
                    eligible_face_crops=self.instances, point_presence_threshold=.70,
                    nme_threshold=.10, normalization='1.20 * max(GT Head width, GT Head height)',
                    metric_scope='conditional on matched predicted FACE and valid face-positive GT Head with labeled points')


@torch.inference_mode()
def validate_prior(model, dataset, *, device, batch_size, num_workers=0, diagnostics=None):
    report = dataset.detector_crop_report
    if not report or report.get('matching_contract') != CONTRACT or report.get('normalization_bbox_source') != 'GT Head':
        raise ValueError('Validation requires authenticated predicted-FACE-to-GT-Head assignments')
    if dataset.training or dataset.geometry_prior_grid_size is None:
        raise ValueError('Validation requires unaugmented predicted geometry inputs')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_landmark_batch)
    counters = ConditionalSemanticCounters()
    model.eval()
    for images, targets in loader:
        images = images.to(device)
        geometry = torch.stack([t['geometry_prior']['geometry'] for t in targets]).to(device)
        support = torch.stack([t['geometry_prior']['support'] for t in targets]).to(device)
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            output = model(images, geometry=geometry, support=support)
        if diagnostics is not None:
            diagnostics(output, targets, images.shape[-2:])
        inverse = torch.stack([t['inverse_affine'] for t in targets])
        points = predictions_to_original(output['decoded_coordinates'], inverse, input_size=dataset.input_size)
        if 'presence_logits' not in output:
            raise ValueError('Declared Query presence branch is missing')
        scores = output['presence_logits'].sigmoid().float().cpu().numpy()
        counters.add(points, scores, targets)
    if counters.instances != len(dataset) or not counters.instances:
        raise ValueError('Incomplete or empty conditional validation')
    return dict(counters.compute(), detector_crop_report=report)
