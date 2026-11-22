"""Bounded real-data FOR context preparation, not training or gate certification."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from PIL import Image
from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
from keypoint_comparison1.landmark_detection.real_occlusion import digest
from scripts.real_face_full_image import FullImageFaceBank, compose_full_image
from scripts.real_face_composite_context import extract_composite_context
from tools.dinov3_deimv2_common import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.samples <= 64:
        raise ValueError('Bounded preparation requires 1..64 composites')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    config = ROOT/'configs/dinov3_deimv2/mal_alpha2_ellipse_from_existing_wider.yaml'
    checkpoint = ROOT/'outputs/experiments/mal_alpha2_ellipse_from_existing_wider/checkpoints/best_val_map.pt'
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != '3bd6e40517bc6f6e276c47e1dfdf639173e8b9c04d32486adbb6f4b407f006a0':
        raise ValueError('Scheduled FOR frozen detector identity changed')
    query_dim = int(load_config(config)['model']['decoder']['hidden_dim'])
    if query_dim != 224:
        raise ValueError('Scheduled DEIMv2-L query dimension differs from authenticated 224')
    detector_identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        config=str(config), config_sha256=digest(config), score_threshold=.33,
        maximum_detections=100, image_size=960, query_dim=query_dim, external_head=False)
    detector = DeimFaceDetector(config_path=config, checkpoint_path=checkpoint,
        image_size=960, score_threshold=.33, maximum_detections=100,
        crop_geometry='ellipse', return_face_query_features=True)
    manifest = ROOT/'outputs/dataset_adapters/real_face_occlusion_w48_s2/manifest.json'
    source = json.loads(manifest.read_text())
    bank = FullImageFaceBank(manifest, digest(manifest),
        annotation_path=source['adapter_path'], image_root=source['image_root'])
    data = json.loads(Path(source['adapter_path']).read_text())
    annotations = defaultdict(list)
    for ann in data['annotations']:
        annotations[ann['image_id']].append(ann)
    rng = random.Random(42)
    images = list(data['images']); rng.shuffle(images)
    rows = []
    for image in images:
        anns = annotations[image['id']]
        eligible = [a for a in anns if bank.eligible_sample(dict(image=image, annotation=a))]
        if not eligible:
            continue
        target = rng.choice(eligible)
        rgb = bank.read(bank.images[image['file_name']])
        composite, updated, mask, meta = compose_full_image(rgb, image=image,
            annotations=anns, target_id=target['id'], bank=bank, rng=rng)
        if not meta['applied']:
            continue
        context = extract_composite_context(detector, composite, meta,
                                             detector_identity=detector_identity, query_dim=query_dim)
        filename = f'composite_{len(rows):04d}.png'
        Image.fromarray(composite).save(args.output/filename)
        row = dict(image=image, composite_file=filename, png_sha256=digest(args.output/filename),
                   composition=meta, recipient_annotations=updated, context=context)
        rows.append(row)
        print(f'composite {len(rows)}/{args.samples}: {len(context["predictions"])} predicted FACE queries', flush=True)
        if len(rows) == args.samples:
            break
    if len(rows) != args.samples:
        raise RuntimeError('Insufficient valid composites; not a completed preparation')
    (args.output/'composites.json').write_text(json.dumps(rows, indent=2)+'\n')
    report = dict(status='bounded_context_preparation_only_not_gate', seed=42,
        composite_count=len(rows), detector_identity=detector_identity,
        manifest_sha256=digest(manifest), split='train',
        composite_manifest_sha256=digest(args.output/'composites.json'),
        production_training='not run', formal_gates='not run',
        common_for_on_off_input=True, stale_clean_queries_used=False,
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT/'scripts/real_face_full_image.py', ROOT/'scripts/real_face_composite_context.py']})
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
