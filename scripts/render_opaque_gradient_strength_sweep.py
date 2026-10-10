#!/usr/bin/env python3
"""Compare gradient widths and strengths on identical train-only placements."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scripts.opaque_boundary_gradient import opaque_boundary_gradient
from scripts.render_opaque_boundary_gradient_examples import seam_errors

VARIANTS=[(24,1),(48,1),(48,2),(72,2)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];panels=[]
    for source in sorted(args.source.glob('example_*')):
        target=np.array(Image.open(source/'target.png').convert('RGB'))
        hard=np.array(Image.open(source/'hard_paste.png').convert('RGB'))
        mask=np.array(Image.open(source/'mask.png'))>0
        meta=json.loads((source/'metadata.json').read_text())
        dest=args.output/source.name;dest.mkdir()
        panel=Image.new('RGB',(1280,302),'white');draw=ImageDraw.Draw(panel)
        draw.text((4,4),f"{source.name} | Target {meta['recipient_id']} | Donor {meta['donor_id']} | Same placement, binary mask, no alpha",fill='black')
        panel.paste(Image.fromarray(target),(0,28));draw.text((4,286),'Original target',fill='black')
        variants=[]
        for j,(width,strength) in enumerate(VARIANTS,1):
            out=opaque_boundary_gradient(target,hard,mask,width=width,strength=strength)
            assert np.array_equal(out[~mask],target[~mask])
            name=f'width{width}_strength{strength}'
            Image.fromarray(out).save(dest/f'{name}.png')
            panel.paste(Image.fromarray(out),(j*256,28))
            draw.text((j*256+4,286),f'Width {width}px / Strength {strength}',fill='black')
            variants.append(dict(width=width,strength=strength,seam_rgb_mae=seam_errors(out,mask),
                mean_crop_rgb_change=float(np.abs(out.astype(float)-hard.astype(float))[mask].mean()),
                outside_changed_pixels=int(np.any(out!=target,axis=-1)[~mask].sum())))
        panel.save(dest/'comparison.png');panels.append(panel)
        row=dict(example=source.name,recipient_id=meta['recipient_id'],donor_id=meta['donor_id'],
            overlap=meta['overlap'],states=meta['states'],variants=variants,
            source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.png')})
        (dest/'metadata.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(source.name,flush=True)
    if not panels:raise ValueError('No source examples')
    preview=Image.new('RGB',(1280,302*len(panels)),'white')
    for i,panel in enumerate(panels):preview.paste(panel,(0,302*i))
    preview.save(args.output/'preview.png')
    report=dict(status='preview_only',training='not run',source=str(args.source),
        variants=[dict(width=w,strength=s) for w,s in VARIANTS],examples=rows,
        method='Crop-side additive RGB correction; strength extends inward influence without changing seam endpoint.',
        strength_definition='gain=s/(1+(s-1)*u), u=diffused unit seam field; s=1 is the prior implementation',
        original_side_changed_pixels=0,
        limitations=['Widths are output-crop pixels, not source-image pixels.',
            'Large widths may cover most of a small donor; no fixed untouched core is guaranteed.',
            'Preview settings are not production training settings or an accuracy claim.'],
        code_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),ROOT/'scripts/opaque_boundary_gradient.py']})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
