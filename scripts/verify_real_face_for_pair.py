"""Bounded CPU verification of shared real-face FOR inputs and initialization."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.face_overlap import face_context_kwargs
from scripts.real_face_for_pair import prepare_bundle, shared_dataset, initialize_matched_arm, sha


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_landmark_config(ROOT/'keypoint_comparison1/configs/face_overlap_real_face_w48_s2.yaml').raw
    bundle = prepare_bundle(args.source, args.output)
    checkpoint = cfg['model']['backbone']['initialize_from_deim_checkpoint']
    if sha(checkpoint) != bundle['identity']['checkpoint_sha256']:
        raise ValueError('Shared initialization checkpoint differs from pinned detector')
    for k in ('checkpoint', 'config'):
        if sha(bundle['identity'][k]) != bundle['identity'][k+'_sha256']:
            raise ValueError('Frozen detector source changed')
    off_data = shared_dataset(bundle, cfg['data'], training=True)
    on_data = shared_dataset(bundle, cfg['data'], training=True)
    if not len(off_data):
        raise ValueError('No matched recipient crops')
    for index in range(len(off_data)):
        seed(); a, ta = off_data[index]
        seed(); b, tb = on_data[index]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert ta.keys() == tb.keys()
        for key in ta:
            if isinstance(ta[key], torch.Tensor):
                torch.testing.assert_close(ta[key], tb[key], rtol=0, atol=0)
            else:
                assert ta[key] == tb[key], key
    off_cfg = deepcopy(cfg)
    off_cfg['model']['face_overlap_robustness']['enabled'] = False
    seed(); off, off_init = build_landmark_model(off_cfg, variant='query_refinement')
    seed(); on, on_init = build_landmark_model(cfg, variant='query_refinement')
    # Construction order can consume RNG differently in downstream refinement.
    before = [k for k, v in off.state_dict().items() if not torch.equal(v, on.state_dict()[k])]
    init = initialize_matched_arm(off, on)
    off.eval(); on.eval()
    with torch.no_grad():
        x, target = off_data[0]
        a = off(x[None], **face_context_kwargs(off, [target], torch.device('cpu')))
        b = on(x[None], **face_context_kwargs(on, [target], torch.device('cpu')))
        for key in ['decoded_coordinates', 'visibility_logits']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    report = dict(status='bounded_pair_preparation_passed_not_gate', training='not run',
        topology=['full_image_face_ellipse', 'face_crop_dino', 'query_refinement'],
        external_head=False, seed=42, source_composites_sha256=bundle['source_sha256'],
        images=bundle['images'], recipient_annotations=bundle['annotations'],
        matched_crops=len(off_data), matcher_report=off_data.detector_crop_report,
        annotation_sha256=sha(bundle['annotation']), cache_sha256=sha(bundle['cache']),
        detector_identity=bundle['identity'], initial_shared_differences=before,
        matched_initialization=init, backbone_initialization=off_init,
        paired_input_and_zero_residual_forward_exact=True,
        limitations=['bounded four-image sample; not full-train distribution audit',
            'production/resume/W&B and four gates not integrated',
            'natural val/test and frozen overlap/misattribution evaluation still required'])
    for name in ['real_face_for_pair.py', 'verify_real_face_for_pair.py']:
        report.setdefault('source_hashes', {})[name] = sha(ROOT/'scripts'/name)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'images', 'recipient_annotations',
        'matched_crops', 'initial_shared_differences', 'matched_initialization']}, indent=2))


if __name__ == '__main__':
    main()
