"""Bounded CPU audit of full-image transport; never writes gate success markers."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from keypoint_comparison1.landmark_detection.real_occlusion import digest
from scripts.real_face_full_image import FullImageFaceBank, compose_full_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'outputs/dataset_adapters/real_face_occlusion_w48_s2/manifest.json')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError('Positive sample count required')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text())
    bank = FullImageFaceBank(args.manifest, digest(args.manifest),
        annotation_path=manifest['adapter_path'], image_root=manifest['image_root'])
    adapter = json.loads(Path(manifest['adapter_path']).read_text())
    by_image = defaultdict(list)
    for ann in adapter['annotations']:
        by_image[ann['image_id']].append(ann)
    rng = random.Random(args.seed)
    images = list(adapter['images']); rng.shuffle(images)
    rows = []
    for image in images:
        anns = by_image[image['id']]
        eligible = [a for a in anns if bank.eligible_sample(dict(image=image, annotation=a))]
        if not eligible:
            continue
        target = rng.choice(eligible)
        rgb = bank.read(bank.images[image['file_name']])
        originals = deepcopy(anns)
        output, updated, mask, meta = compose_full_image(rgb, image=image, annotations=anns,
            target_id=target['id'], bank=bank, rng=rng)
        assert np.array_equal(rgb[~mask], output[~mask])
        assert anns == originals
        assert len(updated) == len(anns)  # donor is separately and explicitly tracked
        if meta['applied']:
            assert meta['donor']['support_pixels'] > 0
            assert meta['donor_video'] != meta['recipient_video']
            assert meta['post_composition_queries_required']
        else:
            assert not mask.any() and np.array_equal(rgb, output) and updated == anns
        rows.append(meta)
        if len(rows) % 16 == 0:
            print(f'audited {len(rows)}/{args.samples}', flush=True)
        if len(rows) >= args.samples:
            break
    applied = [r for r in rows if r['applied']]
    report = dict(status='preparation_audit_only_not_gate', seed=args.seed, requested=len(rows),
        applied=len(applied), skips=len(rows)-len(applied),
        skip_reasons=dict(Counter(r['reason'] for r in rows if not r['applied'])),
        requested_routes=dict(Counter(r.get('requested_route') for r in rows)),
        applied_routes=dict(Counter(r['requested_route'] for r in applied)),
        donor_support_truncated=sum(r['donor']['support_truncated'] for r in applied),
        additional_touched_recipients=sum(len(r['touched_recipient_ids'])-1 for r in applied),
        checks=dict(exterior_pixels_unchanged=True, source_annotations_unchanged=True,
                    train_only_cross_video=True, skipped_inputs_unchanged=True),
        manifest_sha256=digest(args.manifest), adapter_sha256=manifest['adapter_sha256'],
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'scripts/real_face_full_image.py']},
        production_training='not run', formal_gates='not run',
        remaining=['Explicit truncated-donor detector label contract',
                   'Actual detector loader and production jitter/distribution audit',
                   'FOR post-composition query extraction and shared on/off manifests',
                   'Frozen FOR overlap/attribution validation protocol'])
    (args.output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
