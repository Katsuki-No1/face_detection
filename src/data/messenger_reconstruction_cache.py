"""Epoch-specific RGB/mask/feature cache with audited deterministic alignment."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts import run_messenger_multiaxis as base
from scripts.train_v3_shape_recovery_gate import Crops,ANN
from scripts.train_natural_feature_gate import build_crop_transform,ellipse_aligned_rectangle
from src.data.messenger_balanced_occlusion import schedule,render,SHAPES,PLAN


def sources(manifest,data):
    corpus=json.loads(ANN.read_text());anns={a['id']:a for a in corpus['annotations']};images={i['id']:i for i in corpus['images']}
    ids=sorted({i for v in manifest['roles'].values() for i in v['synthetic']})
    ds=Crops([anns[i] for i in ids],images,Path(data['root']));result={}
    yy,xx=np.ogrid[:256,:256]
    for k,i in enumerate(ids):
        row=ds[k];a=anns[i];box,angle=ellipse_aligned_rectangle(a['ellipse'])
        crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        cx,cy=crop.original_to_crop(np.array([a['ellipse'][:2]]))[0]
        rx,ry=np.array(a['ellipse'][2:4])*255/crop.side
        face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
        assert (row['state']==2).all()
        result[i]=dict(image=row['image'],face=face,xy=row['xy']*255,
                       ratio=float(max(rx,ry)/min(rx,ry)),size=float(max(a['ellipse'][2:4])*2))
    return result


def audit_records(rows,masks,face_ids,out,role,epoch,preview=None):
    n=len(face_ids);fractions=np.array([x['fraction'] for x in rows]);bins=np.array([x['bin'] for x in rows])
    counts=np.bincount((fractions*5).astype(int).clip(0,4),minlength=5)
    if not np.array_equal(counts,np.full(5,n)):raise ValueError('Five strata are not equally represented')
    error=max(abs(x['fraction']-x['target']) for x in rows)
    if error>PLAN['target_tolerance']:raise ValueError('Area target tolerance exceeded')
    shape_counts=[[sum(x['bin']==b and x['shape']==s for x in rows) for s in SHAPES] for b in range(5)]
    position_counts=[[sum(x['bin']==b and x['anchor']==a for x in rows) for a in range(9)] for b in range(5)]
    if n>=54 and (min(map(min,shape_counts))==0 or min(map(min,position_counts))==0):
        raise ValueError('Missing assigned shape or position in a severity bin')
    unique=len(set(x['mask_sha256'] for x in rows))
    if unique/len(rows)<.98:raise ValueError('Repeated masks exceed 2%')
    summary=dict(passed=True,role=role,epoch=epoch,pairs=len(rows),counts=counts.tolist(),
        fraction_min=float(fractions.min()),fraction_max=float(fractions.max()),max_area_error=error,
        shape_counts=shape_counts,anchor_counts=position_counts,shape_order=list(SHAPES),
        covered_point_counts=np.bincount([x['covered_points'] for x in rows],minlength=6).tolist(),
        point_patterns={p:sum(x['point_pattern']==p for x in rows) for p in sorted(set(x['point_pattern'] for x in rows))},
        unique_masks=unique,mean_patch_fraction=float(np.mean([x['patch_fraction'] for x in rows])),
        within_bin_deciles=[np.histogram((fractions[bins==b]-.2*b)/.2,np.linspace(0,1,11))[0].tolist() for b in range(5)],
        manifest_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    stem=f'{role}_epoch{epoch:04d}_faces{n}'
    path=out/(stem+'.json')
    if path.exists():
        old=json.loads(path.read_text())
        if old!=summary:raise ValueError('Regenerated distribution differs from prior epoch identity')
    base.write(path,summary)
    base.write(out/(stem+'_records.json'),rows)
    heat=np.stack([masks[bins==b].mean(0) for b in range(5)])
    np.savez_compressed(out/(stem+'_coverage.npz'),coverage=heat)
    if preview:
        canvas=Image.new('RGB',(6*192,5*216),'white');draw=ImageDraw.Draw(canvas)
        for (b,s),(im,meta) in preview.items():
            canvas.paste(Image.fromarray(im).resize((192,192)),(s*192,b*216))
            draw.text((s*192+3,b*216+193),f'{SHAPES[s]} {meta["fraction"]:.1%}',fill='black')
        canvas.save(out/(stem+'_examples.jpg'))
    return summary


def render_role(source,face_ids,role,epoch):
    rows=[];rgb=[];cov=[];ext=[];points=[];masks=[];preview={}
    for spec in schedule(face_ids,role,epoch):
        a=source[spec['face_id']]
        im,c,e,p,meta,mask=render(a['image'],a['face'],a['xy'],spec)
        meta.update(face_ratio=a['ratio'],face_size=a['size'])
        rows.append(meta);rgb.append(im);cov.append(c);ext.append(e);points.append(p);masks.append(mask)
        preview.setdefault((spec['bin'],SHAPES.index(spec['shape'])),(im,meta))
    return rows,np.stack(rgb),np.stack(cov),np.stack(ext),np.stack(points),np.stack(masks),preview


@torch.no_grad()
def encode(model,rgb,data):
    mean=torch.tensor(data['mean'],device='cuda')[None,:,None,None];std=torch.tensor(data['std'],device='cuda')[None,:,None,None]
    maps=[]
    for ids in base.batches(len(rgb),8):
        x=torch.from_numpy(rgb[ids]).cuda().permute(0,3,1,2).float()/255
        maps.append(torch.stack(model.backbone.forward_intermediate_feature_maps((x-mean)/std,layer_indexes=[3,7,11]),1).half().cpu())
    return torch.cat(maps)


def make_cache(out,model,data):
    cache=base.make_cache(out,model,data)
    manifest=json.loads((Path(out)/'manifest.json').read_text())
    cache['source']=sources(manifest,data);cache['data_config']=data;cache['new_manifest']=manifest
    parent=[];roles={};index={i:k for k,i in enumerate(manifest['faces'])}
    for role in ('fit','dev','eval'):
        begin=len(parent)
        ids=sorted(manifest['roles'][role]['synthetic'])
        parent.extend([index[i] for i in ids for _ in range(5)])
        roles[role]=np.arange(begin,len(parent))
    cache.update(parent=np.array(parent),syn=torch.empty((len(parent),3,384,16,16),dtype=torch.float16),
                 cov=torch.empty((len(parent),16,16),dtype=torch.bool),ext=torch.empty((len(parent),16,16),dtype=torch.bool),
                 point=torch.empty((len(parent),5),dtype=torch.bool),kind=np.empty(len(parent),dtype=int),
                 fraction=np.empty(len(parent)),severity=np.empty(len(parent),dtype=int))
    for role in roles:cache['syn_'+role]=roles[role]
    cache['dynamic_epoch']={}
    for role in roles:refresh(cache,model,out,role,0)
    manifest['synthetic_pairs']=len(parent);manifest['augmentation']=PLAN
    base.write(Path(out)/'manifest.json',manifest)
    return cache


def refresh(cache,model,out,role,epoch,face_ids=None):
    epoch=epoch if role=='fit' else 0
    face_ids=sorted(face_ids if face_ids is not None else cache['new_manifest']['roles'][role]['synthetic'])
    rows,rgb,c,e,p,masks,preview=render_role(cache['source'],face_ids,role,epoch)
    summary=audit_records(rows,masks,face_ids,Path(out)/'occlusion_audit',role,epoch,preview if epoch==0 else None)
    maps=encode(model,rgb,cache['data_config'])
    actual_sha=hashlib.sha256(maps.numpy().tobytes()).hexdigest()
    positions={i:k for k,i in enumerate(sorted(cache['new_manifest']['roles'][role]['synthetic']))}
    indices=np.concatenate([cache['syn_'+role][positions[i]*5:positions[i]*5+5] for i in face_ids])
    cache['syn'][indices]=maps;cache['cov'][indices]=torch.from_numpy(c);cache['ext'][indices]=torch.from_numpy(e);cache['point'][indices]=torch.from_numpy(p)
    cache['kind'][indices]=[SHAPES.index(x['shape']) for x in rows];cache['fraction'][indices]=[x['fraction'] for x in rows];cache['severity'][indices]=[x['bin'] for x in rows]
    proof=dict(epoch=epoch,role=role,faces=len(face_ids),manifest_sha256=summary['manifest_sha256'],feature_sha256=actual_sha)
    path=Path(out)/'occlusion_audit'/f'{role}_epoch{epoch:04d}_faces{len(face_ids)}_feature.json'
    if path.exists() and json.loads(path.read_text())!=proof:raise ValueError('Feature cache identity differs across model conditions')
    base.write(path,proof);cache['dynamic_epoch'][role]=proof
    return proof
