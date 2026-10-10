#!/usr/bin/env python3
"""Train-only CPU feasibility prototype; deliberately not a training data loader."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from src.data.landmark_occlusion import ellipse_region_mask
from scripts.opaque_boundary_gradient import opaque_boundary_gradient, DEFAULT_WIDTH, DEFAULT_STRENGTH

CANONICAL = ROOT / 'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/train.json'
ADAPTER = ROOT / 'outputs/dataset_adapters/face_pose_drive_v2_balanced_811/face_keypoints_train.json'
IMAGE_ROOT = ROOT / 'outputs/dataset_cache/immutable_f493be94f9c30e54/face_pose_drive_v2_balanced_811'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(image, ann):
    return image['file_name'], tuple(round(x, 4) for x in ann['ellipse'])


def recipient_eligible(ann, source):
    """Missing/unknown is not a negative occlusion label."""
    return (ann.get('attribute_valid') is True
            and ann.get('source_face_index') is not None
            and source.get('attributes', {}).get('face_occluded') is False
            and ann.get('keypoint_states') == [2] * 5
            and np.asarray(ann.get('keypoints')).shape == (5, 2)
            and np.isfinite(ann['keypoints']).all())


def context_descriptor(rgb, bbox, all_boxes):
    """Lab moments in 1.8x HEAD box minus all HEAD boxes; background proxy."""
    h, w = rgb.shape[:2]
    tiny = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    sx, sy = 320 / w, 180 / h
    mask = np.zeros((180, 320), np.uint8)
    x, y, bw, bh = bbox
    x0, x1 = max(0, int((x-.4*bw)*sx)), min(320, int((x+1.4*bw)*sx))
    y0, y1 = max(0, int((y-.4*bh)*sy)), min(180, int((y+1.4*bh)*sy))
    mask[y0:y1, x0:x1] = 1
    for x, y, bw, bh in all_boxes:
        mask[max(0, int(y*sy)):min(180, int(np.ceil((y+bh)*sy))),
             max(0, int(x*sx)):min(320, int(np.ceil((x+bw)*sx)))] = 0
    if mask.sum() < 64:
        return None
    lab = cv2.cvtColor(tiny.astype(np.float32)/255, cv2.COLOR_RGB2LAB)[mask > 0]
    return np.r_[lab.mean(0), lab[:, 0].std()].astype(float).tolist()


def choose_donor(target, donors, rng, profile):
    """Balance by source video, not number of neighboring frames."""
    best = {}
    t = np.array(target['descriptor'])
    for d in donors:
        if d['video'] == target['video'] or (d['ratio'] >= 3) != profile:
            continue
        delta = np.array(d['descriptor']) - t
        if abs(delta[0]) > 12 or np.linalg.norm(delta[1:3]) > 12 or abs(delta[3]) > 15:
            continue
        distance = float(np.linalg.norm(delta))
        if d['video'] not in best or distance < best[d['video']][0]:
            best[d['video']] = distance, d
    top = sorted(best.values(), key=lambda x: x[0])[:5]
    return rng.choice(top) if top else None


def crop_record(record, *, target):
    rgb = np.array(Image.open(IMAGE_ROOT / record['file']).convert('RGB'))
    cx, cy, a, b, angle = record['ellipse']
    if target:
        side = 2.5 * max(a, b)
        box = [cx-side/2, cy-side/2, cx+side/2, cy+side/2]
    else:
        x, y, w, h = record['bbox']
        box = [x-.05*w, y-.05*h, x+1.05*w, y+1.05*h]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    scale = 256 / max(x1-x0, y1-y0)
    matrix = np.array([[scale, 0, -x0*scale], [0, scale, -y0*scale]])
    size = (round((x1-x0)*scale), round((y1-y0)*scale))
    out = cv2.warpAffine(rgb, matrix, size, flags=cv2.INTER_LINEAR)
    ellipse = [(cx-x0)*scale, (cy-y0)*scale, a*scale, b*scale, angle]
    points = (np.array(record['points'])-[x0,y0])*scale
    states = np.array(record.get('states',[2]*5),dtype=np.int64)
    inside = np.isfinite(points).all(axis=1) & (points>=0).all(axis=1)
    inside &= (points[:,0]<out.shape[1]-.5)&(points[:,1]<out.shape[0]-.5)
    states[~inside] = 0
    points[states==0] = 0  # Missing slots carry no coordinate supervision.
    return out, ellipse, points, states


def place(target, donor, rng, route):
    image, ellipse, points = target[:3]
    states = np.array(target[3] if len(target)>3 else [2]*5,dtype=np.int64)
    drgb, de = donor[:2]
    h, w = image.shape[:2]
    face = ellipse_region_mask(h, w, ellipse)
    dface = ellipse_region_mask(*drgb.shape[:2], de)
    lo, hi = {'light': (0, .1), 'medium': (.1, .2), 'strong': (.2, .4)}[route]
    for attempt in range(128):
        scale = rng.uniform(.65, 1.35) * 2*max(ellipse[2:4])/max(drgb.shape[:2])
        dh, dw = [max(1, round(n*scale)) for n in drgb.shape[:2]]
        side = rng.randrange(4)
        # The pasted rectangle enters from a crop edge. No rotation or warping.
        x = rng.randint(-dw//2, w-1); y = rng.randint(-dh//2, h-1)
        if side == 0: x = -rng.randint(1, max(1,dw-1))
        if side == 1: x = w-rng.randint(1,max(1,dw-1))
        if side == 2: y = -rng.randint(1,max(1,dh-1))
        if side == 3: y = h-rng.randint(1,max(1,dh-1))
        x0,y0,x1,y1 = max(0,x),max(0,y),min(w,x+dw),min(h,y+dh)
        if x1 <= x0 or y1 <= y0: continue
        mask = np.zeros((h,w),bool); mask[y0:y1,x0:x1] = True
        overlap = float((mask & face).sum()/face.sum())
        xy = np.rint(points).astype(int)
        inside = (states>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        covered = np.zeros(len(points),bool)
        covered[inside] = mask[xy[inside,1],xy[inside,0]]
        if not (lo < overlap <= hi) or (route == 'light' and not covered.any()): continue
        df = cv2.resize(dface.astype(np.uint8),(dw,dh),interpolation=cv2.INTER_NEAREST)
        foreground = np.zeros((h,w),bool)
        foreground[y0:y1,x0:x1] = df[y0-y:y1-y,x0-x:x1-x] > 0
        actual_face_overlap = float((foreground & face).sum()/face.sum())
        if actual_face_overlap < .01: continue  # Reject background-only occluders.
        out = image.copy(); resized = cv2.resize(drgb,(dw,dh),interpolation=cv2.INTER_LINEAR)
        out[y0:y1,x0:x1] = resized[y0-y:y1-y,x0-x:x1-x]
        updated_states = np.where(covered,1,states).tolist()
        return out, mask, dict(applied=True, attempts=attempt+1, overlap=overlap,
            donor_face_overlap=actual_face_overlap, states=updated_states,
            pre_paste_states=states.tolist(),missing_keypoints=int((states==0).sum()),
            occluded_keypoints=int(covered.sum()), bbox_xyxy=[x,y,x+dw,y+dh],
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return image.copy(), np.zeros((h,w),bool), dict(applied=False,reason='placement_budget',attempts=128)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=240)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--opaque-gradient',action='store_true')
    args=p.parse_args(); cv2.setNumThreads(1); rng=random.Random(args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(CANONICAL.read_text()); a=json.loads(ADAPTER.read_text())
    ci={x['id']:x for x in c['images']}; ai={x['id']:x for x in a['images']}
    assert (len(ci),len(c['annotations']),len({i['source_video_id'] for i in ci.values()})) == (4237,7129,44)
    assert sha(CANONICAL) == 'b8a81c114c26a9476907ecb4611ee450e49a5f21a5683021841c8ddcb4a32ecb'
    lookup=defaultdict(list)
    for an in a['annotations']:
        assert ai[an['image_id']]['metadata']['split'] == 'train'
        lookup[identity(ai[an['image_id']],an)].append(an)
    by_image=defaultdict(list)
    for an in c['annotations']: by_image[an['image_id']].append(an)
    counts=Counter(); records=[]
    for n,(iid,annotations) in enumerate(by_image.items()):
        im=ci[iid]; rgb=None
        for an in annotations:
            if an.get('attribute_valid') is not True or an.get('source_face_index') is None:
                counts['invalid_or_no_face']+=1;continue
            match=lookup.get(identity(im,an),[])
            if len(match)!=1: counts['ambiguous_or_absent_adapter']+=1;continue
            src=match[0]
            if src.get('attributes',{}).get('face_occluded') is not False:
                counts['contour_occluded_or_unknown']+=1;continue
            eligible=recipient_eligible(an,src)
            counts['eligible' if eligible else 'not_all_five_visible']+=1
            if rgb is None:
                path=IMAGE_ROOT/im['file_name']
                adapter_image=ai[src['image_id']]
                assert adapter_image['metadata']['source_video_id']==im['source_video_id']
                assert sha(path)==adapter_image['metadata']['sha256']
                rgb=np.array(Image.open(path).convert('RGB'))
            desc=context_descriptor(rgb,an['bbox'],[b['bbox'] for b in annotations])
            if desc is None: counts['insufficient_context']+=1;continue
            records.append(dict(id=an['id'],file=im['file_name'],video=im['source_video_id'],
                ellipse=an['ellipse'],bbox=an['bbox'],points=an['keypoints'],states=an['keypoint_states'],descriptor=desc,
                ratio=max(an['ellipse'][2:4])/min(an['ellipse'][2:4]),eligible=eligible))
        if n%500==0: print(f'descriptors {n}/{len(by_image)}',flush=True)
    recipients=[r for r in records if r['eligible']];rng.shuffle(recipients)
    rows=[]; panels=[]
    for target in recipients[:args.samples]:
        route=rng.choices(['light','medium','strong'],[.2,.4,.4])[0]
        profile=rng.random()<.5
        base=dict(recipient_id=target['id'],recipient_video=target['video'],route=route,profile_requested=profile)
        tc=crop_record(target,target=True)
        if tc is None:rows.append(dict(base,applied=False,reason='empty_recipient_crop'));continue
        picked=choose_donor(target,records,rng,profile)
        if picked is None:rows.append(dict(base,applied=False,reason='no_photometric_donor'));continue
        distance,donor=picked; assert donor['video']!=target['video']
        dc=crop_record(donor,target=False)
        if dc is None:rows.append(dict(base,applied=False,reason='empty_donor_crop'));continue
        out,mask,meta=place(tc,dc,rng,route)
        row=dict(base,**meta,donor_id=donor['id'],donor_video=donor['video'],
            photometric_distance=distance,donor_ratio=donor['ratio'])
        rows.append(row)
        if meta['applied'] and len(panels)<8:
            corrected=opaque_boundary_gradient(tc[0],out,mask) if args.opaque_gradient else out
            arrays=[tc[0],dc[0],out,corrected] if args.opaque_gradient else [tc[0],dc[0],out]
            panel=Image.new('RGB',(256*len(arrays),302),'white')
            labels=['Target','Donor (other video)','Hard paste','Opaque RGB gradient']
            for j,arr in enumerate(arrays):
                thumb=Image.fromarray(arr);thumb.thumbnail((256,256));panel.paste(thumb,(j*256,28))
                ImageDraw.Draw(panel).text((j*256+4,286),labels[j],fill='black')
            draw=ImageDraw.Draw(panel)
            draw.text((4,4),f"Target {target['id']} | Donor {donor['id']} | {route} {meta['overlap']:.1%}, occluded {meta['occluded_keypoints']}/5, missing {meta['missing_keypoints']}/5",fill='black')
            example=args.output/f'example_{len(panels)+1:02d}'
            example.mkdir()
            for name,arr in zip(['target','donor','hard_paste','opaque_gradient'],arrays):
                Image.fromarray(arr).save(example/f'{name}.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(example/'mask.png')
            panel.save(example/'comparison.png')
            (example/'metadata.json').write_text(json.dumps(dict(row,
                recipient_file=target['file'],donor_file=donor['file'],
                target_keypoints=tc[2].tolist(),target_crop_shape=list(tc[0].shape),
                correction='additive RGB offset from exterior seam only; no alpha',band_width=DEFAULT_WIDTH,
                strength=DEFAULT_STRENGTH,maximum_offset=None),indent=2)+'\n')
            panels.append(panel)
    applied=[r for r in rows if r['applied']]
    montage=Image.new('RGB',(panels[0].width,302*len(panels)),'white') if panels else None
    if montage:
        for i,panel in enumerate(panels):montage.paste(panel,(0,i*302))
        montage.save(args.output/'preview.png')
    report=dict(status='prototype_only',training='not run',production_ready=False,seed=args.seed,
        topology='HEAD-free full-image FACE/ellipse -> Face-crop DINO -> Messenger -> Query Refinement',
        canonical_sha256=sha(CANONICAL),adapter_sha256=sha(ADAPTER),code_sha256=sha(__file__),
        gradient_code_sha256=sha(ROOT/'scripts/opaque_boundary_gradient.py'),
        opaque_gradient_examples=args.opaque_gradient,
        gradient_settings=dict(width=DEFAULT_WIDTH,strength=DEFAULT_STRENGTH,alpha=False),
        crop_policy='clip to source bounds; isotropic resize; out-of-crop keypoints state0; donor may be truncated',
        image_root=str(IMAGE_ROOT),canonical=str(CANONICAL),adapter=str(ADAPTER),
        coordinates='source/crop pixels xy; ellipse center, radii, angle radians; integer binary paste support',
        inventory=dict(counts),donor_records=len(records),recipient_records=len(recipients),
        requested=len(rows),applied=len(applied),skip_reasons=dict(Counter(r['reason'] for r in rows if not r['applied'])),
        requested_routes=dict(Counter(r['route'] for r in rows)),applied_routes=dict(Counter(r['route'] for r in applied)),
        occluded_keypoint_counts=dict(Counter(r['occluded_keypoints'] for r in applied)),
        post_crop_missing_counts=dict(Counter(r['missing_keypoints'] for r in applied)),
        donor_video_counts=dict(Counter(r['donor_video'] for r in applied)),
        donor_profile_counts=dict(Counter(str(r['profile_requested']) for r in applied)),
        provisional_settings=dict(route_probabilities=[.2,.4,.4],profile_probability=.5,
            max_delta_L=12,max_chroma_distance=12,max_delta_L_std=15,top_videos=5,attempts=128),
        limitations=['Context ring is a background proxy, not a segmentation mask.',
            'Rectangular source background also occludes. Donor ellipse is not the opacity mask.',
            'Prototype fixed GT crop differs from production crop jitter; not a production distribution gate.',
            'No train/validation/test model evaluation or training integration. No success marker.',
            'Binary rectangle seams remain; realism requires visual review and a future silhouette-mask comparison.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
