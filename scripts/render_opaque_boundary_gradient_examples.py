#!/usr/bin/env python3
"""Re-render identical saved placements with recipient-aware opaque boundaries."""
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
from scripts.opaque_boundary_gradient import opaque_boundary_gradient, DEFAULT_WIDTH, DEFAULT_STRENGTH


def seam_errors(rgb,mask):
    values=[]
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy()
        a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        edge=mask[a]!=mask[b]
        values.extend(np.abs(rgb[a].astype(float)-rgb[b].astype(float))[edge].ravel())
    return float(np.mean(values)) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=DEFAULT_WIDTH)
    parser.add_argument('--strength',type=float,default=DEFAULT_STRENGTH)
    args=parser.parse_args();cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    panels=[];rows=[]
    for source in sorted(args.source.glob('example_*')):
        target=np.array(Image.open(source/'target.png').convert('RGB'))
        hard=np.array(Image.open(source/'hard_paste.png').convert('RGB'))
        previous=np.array(Image.open(source/'opaque_gradient.png').convert('RGB'))
        mask=np.array(Image.open(source/'mask.png'))>0
        result=opaque_boundary_gradient(target,hard,mask,width=args.width,strength=args.strength)
        assert np.array_equal(result[~mask],target[~mask])
        hidden=target.copy();hidden[mask]=255-target[mask]
        assert np.array_equal(result,opaque_boundary_gradient(hidden,hard,mask,width=args.width,strength=args.strength))
        meta=json.loads((source/'metadata.json').read_text())
        row=dict(example=source.name,recipient_id=meta['recipient_id'],donor_id=meta['donor_id'],
            overlap=meta['overlap'],states=meta['states'],outside_changed_pixels=0,
            hard_seam_rgb_mae=seam_errors(hard,mask),previous_seam_rgb_mae=seam_errors(previous,mask),
            current_seam_rgb_mae=seam_errors(result,mask),
            source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.png')})
        dest=args.output/source.name;dest.mkdir()
        Image.fromarray(result).save(dest/'opaque_gradient.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mask.png')
        panel=Image.new('RGB',(1024,302),'white');draw=ImageDraw.Draw(panel)
        draw.text((4,4),f"{source.name} | Target {meta['recipient_id']} | Donor {meta['donor_id']} | Occlusion {meta['overlap']:.1%}",fill='black')
        for j,(array,label) in enumerate(zip([target,hard,previous,result],
                    ['Original target','Hard paste','Previous correction','New: crop side only'])):
            panel.paste(Image.fromarray(array),(j*256,28))
            draw.text((j*256+4,286),label,fill='black')
        panel.save(dest/'comparison.png');panels.append(panel);rows.append(row)
        (dest/'metadata.json').write_text(json.dumps(row,indent=2)+'\n')
    if not panels:raise ValueError('No source examples')
    montage=Image.new('RGB',(1024,302*len(panels)),'white')
    for i,panel in enumerate(panels):montage.paste(panel,(0,302*i))
    montage.save(args.output/'preview.png')
    report=dict(status='preview_only',training='not run',source=str(args.source),
        method='Exact exterior recipient boundary colors; additive correction inside crop band; no alpha',
        width=args.width,strength=args.strength,maximum_offset=None,original_side_changed_pixels=0,examples=rows,
        code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'scripts/opaque_boundary_gradient.py']},
        limitation='Adjacent-pixel RGB seam error is not a measure of natural appearance or model accuracy.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'example':r['example'],'before':r['previous_seam_rgb_mae'],'after':r['current_seam_rgb_mae']} for r in rows],indent=2))


if __name__=='__main__':main()
