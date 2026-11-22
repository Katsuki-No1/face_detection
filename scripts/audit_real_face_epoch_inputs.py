"""Audit scheduled full-train epoch synthesis recipes; never a formal gate.

No PNGs, checkpoints, or prediction tensors are stored. Recipes support exact
replay; production crop jitter and fresh detector context still require audit.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from scripts.real_face_epoch_inputs import SharedEpochInputs, MANIFEST_SHA, ADAPTER_SHA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--epochs', nargs='+', type=int, default=[1, 12, 40])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.epochs)) != len(args.epochs):
        raise ValueError('Duplicate seed/epoch groups')
    if any(e < 1 for e in args.epochs) or (args.limit is not None and not 1 <= args.limit <= 4237):
        raise ValueError('Invalid epoch/limit')
    cv2.setNumThreads(1)
    inputs = SharedEpochInputs(ROOT/'outputs/dataset_adapters/real_face_occlusion_w48_s2/manifest.json')
    images = inputs.images[:args.limit]
    source_files = [Path(__file__), ROOT/'scripts/real_face_epoch_inputs.py',
        ROOT/'scripts/real_face_full_image.py', ROOT/'scripts/prototype_cross_video_face_occlusion.py',
        ROOT/'scripts/opaque_boundary_gradient.py',
        ROOT/'keypoint_comparison1/landmark_detection/real_face_occlusion.py',
        ROOT/'keypoint_comparison1/landmark_detection/real_occlusion.py',
        ROOT/'src/data/face_keypoint_dataset.py', ROOT/'src/data/landmark_occlusion.py']
    contract = dict(schema='real-face-epoch-recipe-audit-v1', seeds=args.seeds,
        epochs=args.epochs, image_ids=[r['id'] for r in images],
        manifest_sha256=MANIFEST_SHA, adapter_sha256=ADAPTER_SHA,
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        preparation_only=True, production_gate=False)
    if args.output.exists():
        if not args.resume or json.loads((args.output/'contract.json').read_text()) != contract:
            raise ValueError('Resume requires identical audit contract')
    else:
        args.output.mkdir(parents=True)
        (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    expected = [(s, e, i['id']) for s in args.seeds for e in args.epochs for i in images]
    records = args.output/'recipes.jsonl'
    prior = []
    if records.exists():
        # Only complete durable lines are checkpointed; discard a torn final line.
        data = records.read_bytes(); end = data.rfind(b'\n')+1
        prior = [json.loads(line) for line in data[:end].splitlines()]
        if len(prior) > len(expected) or [(r['seed'],r['epoch'],r['image_id']) for r in prior] != expected[:len(prior)]:
            raise ValueError('Audit resume inventory mismatch')
        with records.open('r+b') as stream:
            stream.truncate(end)
    with records.open('a') as stream:
        index = 0
        for seed in args.seeds:
            for epoch in args.epochs:
                for image in images:
                    if index < len(prior):
                        index += 1
                        continue
                    _, _, row = inputs.render(image, seed=seed, epoch=epoch, load_unmodified=False)
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                    stream.flush()
                    index += 1
                    if index % 100 == 0:
                        os.fsync(stream.fileno())
                        print(f'recipes {index}/{len(expected)}', flush=True)
        os.fsync(stream.fileno())
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    requested = [r for r in rows if r['requested']]
    applied = [r for r in requested if r['applied']]
    def summarize(group):
        req = [r for r in group if r['requested']]; app = [r for r in req if r['applied']]
        return dict(images=len(group), eligible_images=sum(r['eligible_recipients'] > 0 for r in group),
            requested=len(req), applied=len(app), skipped=len(req)-len(app),
            skip_fraction=(len(req)-len(app))/len(req) if req else None,
            skip_reasons=dict(Counter(r['composition']['reason'] for r in req if not r['applied'])),
            applied_routes=dict(Counter(r['composition']['requested_route'] for r in app)),
            requested_routes=dict(Counter(r['composition']['requested_route'] for r in req)),
            requested_profiles=dict(Counter(str(r['composition']['requested_profile']) for r in req)),
            applied_profiles=dict(Counter(str(r['composition']['requested_profile']) for r in app)),
            covered_target_keypoint_counts=dict(Counter(str(r['covered_target_keypoints']) for r in app)))
    report = dict(status='epoch_recipe_audit_complete_not_gate', finished_at=datetime.now(timezone.utc).isoformat(),
        full_inventory=args.limit is None or args.limit == 4237, **summarize(rows),
        groups={f'{s}/{e}': summarize([r for r in rows if r['seed']==s and r['epoch']==e])
            for s in args.seeds for e in args.epochs},
        videos={v:summarize([r for r in rows if r['source_video']==v]) for v in sorted({r['source_video'] for r in rows})},
        contract_sha256=digest(args.output/'contract.json'), recipes_sha256=digest(records),
        storage_bytes=records.stat().st_size,
        production_training='not run', formal_gates='not run',
        limitations=['full-image synthesis before predicted crop jitter/flip/photometric',
            'no fresh detector queries or production crop audit in this CPU job',
            'no quality gate acceptance inferred from preparation completion'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','images','requested','applied','skipped','skip_fraction','skip_reasons']}, indent=2))


if __name__ == '__main__':
    main()
