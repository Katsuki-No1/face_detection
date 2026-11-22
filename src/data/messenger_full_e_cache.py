"""Official split, all clean faces; bounded RGB batches and disk-backed features."""
import gzip,hashlib,json,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from scripts.train_natural_feature_gate import build_crop_transform,ellipse_aligned_rectangle
from scripts.evaluate_messenger_geometry_prior import warp_crop
from src.data.messenger_balanced_occlusion import schedule,render,SHAPES,PLAN as MASK_PLAN
from scripts.run_messenger_multiaxis import write,sha,status

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'outputs/dataset_cache/face_pose_condition_balanced_811_latest_sources_r3_20261009'
ANNOTATIONS={s:DATA/f'annotations/deim_dino/face_keypoints_{s}.json' for s in ('train','val','test')}
# Retain the corrected train labels already used for recovery; official video/image split unchanged.
ANNOTATIONS['train']=ROOT/'outputs/diagnostics/keypoint_side_revision_v3_20261031/annotations/deim_dino/face_keypoints_train.json'
ROLE_SPLIT={'fit':'train','dev':'val','eval':'test'}

def build_manifest():
    records=[];roles={};images_by_role={};videos_by_role={};offset=0
    for role,split in ROLE_SPLIT.items():
        corpus=json.loads(ANNOTATIONS[split].read_text());ims={i['id']:i for i in corpus['images']}
        images_by_role[role]={i['metadata']['sha256'] for i in ims.values()}
        videos_by_role[role]={i['metadata']['source_video_id'] for i in ims.values()}
        ids=[]
        for a in sorted(corpus['annotations'],key=lambda a:a['id']):
            if not(a.get('category_id')==2 and a.get('ellipse') and a['keypoints'][2::3]==[2]*5 and a.get('attributes',{}).get('face_occluded') is False):continue
            box,angle=ellipse_aligned_rectangle(a['ellipse']);crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
            xy=crop.original_to_crop(np.asarray(a['keypoints']).reshape(5,3)[:,:2])
            if not(np.isfinite(xy).all() and ((xy>=0)&(xy<=255)).all()):continue
            im=ims[a['image_id']];uid=offset+int(a['id']);ids.append(uid)
            records.append(dict(uid=uid,annotation_id=a['id'],split=split,role=role,ellipse=a['ellipse'],xy=xy.tolist(),
                file=str(DATA/im['file_name']),image_sha256=im['metadata']['sha256'],video=im['metadata']['source_video_id']))
        roles[role]=ids;offset+=1000000000
    for i,a in enumerate(roles):
        for b in list(roles)[i+1:]:
            assert not(images_by_role[a]&images_by_role[b] or videos_by_role[a]&videos_by_role[b]),'Official split leakage'
    assert {k:len(v) for k,v in roles.items()}==dict(fit=8688,dev=1160,eval=1009)
    return dict(roles=roles,faces=[a['uid'] for a in records],records=records,annotations={s:dict(path=str(p),sha256=sha(p)) for s,p in ANNOTATIONS.items()},
        selection='all five points visible, face_occluded=false, all five coordinates finite inside GT crop',
        crop='GT ellipse square x1.25, 256 RGB, head coordinates normalized by 255',
        split_overlap=dict(video=0,image_sha256=0),augmentation=MASK_PLAN)

class FeatureView:
    """Store block11 only; expose historical three-level interface without storing unused maps."""
    def __init__(self,arr):self.arr=arr
    def __getitem__(self,key):
        if isinstance(key,tuple):
            ids,level=key
            if level!=2:raise ValueError('Only block11 is used in this scale study')
            return torch.from_numpy(np.array(self.arr[ids],copy=True))
        value=torch.from_numpy(np.array(self.arr[key],copy=True))
        return value[:,None].expand(-1,3,-1,-1,-1)

@torch.no_grad()
def encode(model,rgb,data):
    x=torch.from_numpy(np.stack(rgb)).cuda().permute(0,3,1,2).float()/255
    mean=torch.tensor(data['mean'],device='cuda')[None,:,None,None];std=torch.tensor(data['std'],device='cuda')[None,:,None,None]
    # Same layer call as prior runs for parity; discard unused intermediate layers.
    return model.backbone.forward_intermediate_feature_maps((x-mean)/std,layer_indexes=[3,7,11])[-1].half().cpu().numpy()

def disk_array(path,shape,dtype):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        arr=np.load(path,mmap_mode='r+')
        if arr.shape!=shape or arr.dtype!=np.dtype(dtype):raise ValueError('Cache shape/type changed')
        return arr
    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

@torch.no_grad()
def make_cache(out,model,data,manifest):
    out=Path(out);records=manifest['records'];n=len(records);index={a['uid']:i for i,a in enumerate(records)}
    rgb=disk_array(out/'cache/rgb.npy',(n,256,256,3),'uint8')
    clean=disk_array(out/'cache/clean.npy',(n,384,16,16),'float16')
    marker=out/'cache/clean_complete.json'
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if marker.exists():
        old=json.loads(marker.read_text())
        assert old['manifest_identity']==identity
        assert old['clean_sha256']==sha(out/'cache/clean.npy') and old['rgb_sha256']==sha(out/'cache/rgb.npy')
    else:
        for start in range(0,n,8):
            batch=[]
            for i in range(start,min(start+8,n)):
                a=records[i];raw=Path(a['file']).read_bytes()
                assert hashlib.sha256(raw).hexdigest()==a['image_sha256']
                photo=np.asarray(Image.open(__import__('io').BytesIO(raw)).convert('RGB'))
                box,angle=ellipse_aligned_rectangle(a['ellipse']);crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
                batch.append(warp_crop(photo,crop))
            rgb[start:start+len(batch)]=np.stack(batch);clean[start:start+len(batch)]=encode(model,batch,data)
            if start%512==0:status(out,'preparing_clean',faces=start+len(batch),total=n)
        rgb.flush();clean.flush()
        write(marker,dict(manifest_identity=identity,clean_sha256=sha(out/'cache/clean.npy'),rgb_sha256=sha(out/'cache/rgb.npy'),faces=n))
    parent=[];role_indices={}
    for role,ids in manifest['roles'].items():
        begin=len(parent);parent.extend(index[i] for i in ids for _ in range(5));role_indices[role]=np.arange(begin,len(parent))
    total=len(parent)
    syn=disk_array(out/'cache/syn.npy',(total,384,16,16),'float16')
    sources={};yy,xx=np.ogrid[:256,:256]
    for i,a in enumerate(records):
        box,angle=ellipse_aligned_rectangle(a['ellipse']);crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        cx,cy=crop.original_to_crop(np.array([a['ellipse'][:2]]))[0];rx,ry=np.array(a['ellipse'][2:4])*255/crop.side
        sources[a['uid']]=dict(image=rgb[i],face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1,xy=np.array(a['xy']))
    cache=dict(clean=FeatureView(clean),syn=FeatureView(syn),syn_array=syn,parent=np.array(parent),
        xy=torch.tensor(np.array([a['xy'] for a in records])/255,dtype=torch.float32),state=torch.full((n,5),2,dtype=torch.long),
        videos=np.array([a['video'] for a in records]),uids=np.array([a['uid'] for a in records]),
        cov=torch.zeros((total,16,16),dtype=torch.bool),ext=torch.zeros((total,16,16),dtype=torch.bool),point=torch.zeros((total,5),dtype=torch.bool),
        severity=np.empty(total,dtype=np.int64),fraction=np.empty(total),source=sources,data_config=data,new_manifest=manifest)
    for role,ids in role_indices.items():cache['syn_'+role]=ids
    return cache

@torch.no_grad()
def refresh(cache,model,out,role,epoch,face_ids=None):
    epoch=epoch if role=='fit' else 0
    all_ids=cache['new_manifest']['roles'][role];face_ids=list(all_ids if face_ids is None else face_ids)
    positions={uid:i for i,uid in enumerate(all_ids)};rows=[];coverage=np.zeros((5,256,256),dtype=np.int64)
    specs=list(schedule(face_ids,role,epoch));feature_hash=hashlib.sha256()
    for start in range(0,len(specs),8):
        batch=[];idx=[]
        for spec in specs[start:start+8]:
            source=cache['source'][spec['face_id']];im,c,e,p,meta,mask=render(source['image'],source['face'],source['xy'],spec)
            j=int(cache['syn_'+role][positions[spec['face_id']]*5+spec['bin']]);idx.append(j);batch.append(im);rows.append(meta)
            cache['cov'][j]=torch.from_numpy(c);cache['ext'][j]=torch.from_numpy(e);cache['point'][j]=torch.from_numpy(p)
            cache['severity'][j]=spec['bin'];cache['fraction'][j]=meta['fraction'];coverage[spec['bin']]+=mask
        encoded=encode(model,batch,cache['data_config']);cache['syn_array'][idx]=encoded;feature_hash.update(encoded.tobytes())
        if len(specs)>10000 and start%4096==0:status(Path(out),'rendering_occlusion',role=role,epoch=epoch,pairs=start+len(batch),total=len(specs))
    cache['syn_array'].flush()
    n=len(face_ids);counts=np.bincount([x['bin'] for x in rows],minlength=5)
    assert counts.tolist()==[n]*5
    fractions=np.array([x['fraction'] for x in rows]);bins=np.array([x['bin'] for x in rows]);assert np.array_equal((fractions*5).astype(int),bins)
    error=max(abs(x['fraction']-x['target']) for x in rows);assert error<=MASK_PLAN['target_tolerance']
    shapes=[[sum(x['bin']==b and x['shape']==s for x in rows) for s in SHAPES] for b in range(5)]
    anchors=[[sum(x['bin']==b and x['anchor']==s for x in rows) for s in range(9)] for b in range(5)]
    appearance=[[sum(x['bin']==b and x['appearance']==s for x in rows) for s in range(3)] for b in range(5)]
    unique=len({x['mask_sha256'] for x in rows});assert unique/len(rows)>=.98
    if n>=54:assert min(map(min,shapes))>0 and min(map(min,anchors))>0
    if n>=162:assert min(map(min,appearance))>0
    audit=dict(passed=True,role=role,epoch=epoch,faces=n,pairs=len(rows),counts=counts.tolist(),shape_order=list(SHAPES),
        shape_counts=shapes,anchor_counts=anchors,appearance_counts=appearance,unique_masks=unique,max_area_error=error,
        covered_point_counts=np.bincount([x['covered_points'] for x in rows],minlength=6).tolist(),
        within_bin_deciles=[np.histogram((fractions[bins==b]-.2*b)/.2,np.linspace(0,1,11))[0].tolist() for b in range(5)],
        feature_sha256=feature_hash.hexdigest(),records_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    folder=Path(out)/'occlusion_audit';folder.mkdir(parents=True,exist_ok=True);stem=f'{role}_epoch{epoch:04d}_faces{n}';path=folder/(stem+'.json')
    if path.exists() and json.loads(path.read_text())!=audit:raise ValueError('Occlusion/feature identity changed across conditions')
    write(path,audit)
    raw=json.dumps(rows,ensure_ascii=False).encode();target=folder/(stem+'_records.json.gz');tmp=target.with_suffix('.tmp');tmp.write_bytes(gzip.compress(raw,mtime=0));tmp.replace(target)
    np.savez_compressed(folder/(stem+'_coverage.npz'),coverage=coverage/max(n,1))
    return audit
