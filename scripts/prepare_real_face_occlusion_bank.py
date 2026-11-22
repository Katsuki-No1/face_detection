#!/usr/bin/env python3
"""Freeze strict train-only context descriptors for planned face-only w48/s2 training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank, digest, identity
from scripts.prototype_cross_video_face_occlusion import context_descriptor, recipient_eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    assert digest(args.source) == args.source_sha256
    data = json.loads(args.source.read_text())
    assert data["canonical_sha256"] == "b8a81c114c26a9476907ecb4611ee450e49a5f21a5683021841c8ddcb4a32ecb"
    bank = RealOcclusionBank(args.source, args.source_sha256,
        annotation_path=data["adapter_path"], image_root=data["image_root"])
    canonical = json.loads(Path(data["canonical_path"]).read_text())
    adapter = json.loads(Path(data["adapter_path"]).read_text())
    images = {r["id"]: r for r in canonical["images"]}
    adapter_images = {r["id"]: r for r in adapter["images"]}
    source = defaultdict(list)
    for ann in adapter["annotations"]:
        source[identity(adapter_images[ann["image_id"]]["file_name"], ann["ellipse"])].append(ann)
    heads = defaultdict(list)
    for ann in canonical["annotations"]:
        heads[images[ann["image_id"]]["file_name"]].append(ann)
    contexts = {}; eligible = []; strict_faces = []
    face_by_key = {identity(f["file"], f["ellipse"]): f for f in data["faces"]}
    for number, (file, annotations) in enumerate(heads.items()):
        rgb = bank.read(bank.images[file])
        for ann in annotations:
            key = identity(file, ann["ellipse"])
            matches = source[key]
            if len(matches) != 1:
                continue
            if recipient_eligible(ann, matches[0]):
                eligible.append(key)
            descriptor = context_descriptor(rgb, ann["bbox"], [a["bbox"] for a in annotations])
            if descriptor is None:
                continue  # Never replace unavailable background with whole-image color.
            contexts[key] = dict(descriptor=descriptor, video=bank.images[file]["video"])
            if key in face_by_key:
                row = dict(face_by_key[key], descriptor=descriptor, video=bank.images[file]["video"],
                    ratio=max(ann["ellipse"][2:4])/min(ann["ellipse"][2:4]))
                strict_faces.append(row)
        if number % 600 == 0:
            print(f"context images {number}/{len(heads)}", flush=True)
    data.update(face_only_policy="real_face_w48_s2_v1", recipient_contexts=contexts,
        strict_faces=strict_faces, eligible_recipients=eligible,
        strict_builder_sha256=digest(__file__), source_manifest_sha256=args.source_sha256,
        context_missing_policy="skip_without_fallback")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(data, stream, separators=(",", ":"))
    print(json.dumps(dict(path=str(args.output), sha256=digest(args.output),
        eligible=len(eligible), contexts=len(contexts), strict_donors=len(strict_faces))), flush=True)


if __name__ == "__main__":
    main()
