"""Balanced five-bin opaque occlusions, deterministic per epoch and face."""
import hashlib
import cv2
import numpy as np
from src.data.v3_shape_occlusion import silhouette

SHAPES=('line','face','hand','ellipse','irregular','multiple')
PLAN=dict(bins=[0,.2,.4,.6,.8,1],per_face=5,shapes=list(SHAPES),positions='3x3 ellipse-local anchor grid plus jitter',
    area='pixel mask intersection / GT ellipse area',target_tolerance=.001,
    sampling='stratified within-bin targets; exact per-face five bins; balanced shape-position slots',
    mask='opaque; patch g is ANY occluded RGB pixel, no face-support threshold',
    fit='epoch-specific seed; regenerate RGB and DINO features',dev_eval='fixed independent seeds',
    appearance=['flat','smooth_texture','pixel_texture'],shape_method='signed-distance silhouette level sets calibrated to actual ellipse intersection',
    shape_limit='high-coverage dilation can merge components; actual topology and coverage are recorded')

def seed_for(*values):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,values)).encode()).digest()[:8],'big')

def schedule(face_ids,role,epoch):
    epoch=epoch if role=='fit' else 0
    n=len(face_ids);rng=np.random.default_rng(seed_for(42,role,epoch,'schedule'))
    ranks=np.empty(n,dtype=int);ranks[rng.permutation(n)]=np.arange(n)
    target_ranks=[rng.permutation(n) for _ in range(5)]
    for i,face_id in enumerate(face_ids):
        for bin_id in range(5):
            seed=seed_for(42,role,epoch,face_id,bin_id)
            rr=np.random.default_rng(seed)
            # Keep a one-pixel safety margin at boundaries while stratifying the interior.
            unit=(target_ranks[bin_id][i]+rr.uniform(.05,.95))/n
            target=(bin_id+(.005+.99*unit))*.2
            slot=int(ranks[i])+bin_id*17+epoch*7
            yield dict(face_id=int(face_id),bin=bin_id,target=float(target),seed=seed,
                       shape=SHAPES[slot%6],anchor=(slot//6)%9,appearance=(int(ranks[i])//54+bin_id+epoch)%3,epoch=epoch,role=role)

def make_mask(face,spec):
    rng=np.random.default_rng(spec['seed']);h,w=face.shape
    yy,xx=np.nonzero(face);cy,cx=yy.mean(),xx.mean();ry=(yy.max()-yy.min()+1)/2;rx=(xx.max()-xx.min()+1)/2
    ay,ax=divmod(spec['anchor'],3)
    center=(cx+rx*((ax-1)*.55+rng.uniform(-.18,.18)),cy+ry*((ay-1)*.55+rng.uniform(-.18,.18)))
    kind=spec['shape']
    if kind in ('line','face','hand'):
        stamp=silhouette(kind,rng)
    else:
        stamp=np.zeros((120,120),np.uint8)
        if kind=='ellipse':
            cv2.ellipse(stamp,(60,60),(45,28),0,0,360,1,-1)
        elif kind=='irregular':
            angles=np.arange(12)*2*np.pi/12;rad=rng.uniform(23,53,12)
            poly=np.stack((60+rad*np.cos(angles),60+rad*np.sin(angles)),1).astype(np.int32)
            cv2.fillPoly(stamp,[poly],1)
        else:
            for p in ((28,32),(88,35),(55,87)):
                cv2.ellipse(stamp,p,(16,22),float(rng.uniform(0,180)),0,360,1,-1)
    angle=rng.uniform(-180,180);mat=cv2.getRotationMatrix2D((60.,60.),angle,rng.uniform(.65,1.1)*min(rx,ry)/55)
    mat[:,:2]=mat[:,:2]@np.diag([rng.uniform(.7,1.4),rng.uniform(.7,1.4)])
    mat[:,2]=np.array(center)-mat[:,:2]@np.array([60.,60.])
    core=cv2.warpAffine(stamp,mat,(w,h),flags=cv2.INTER_NEAREST).astype(np.uint8)
    if not core.any():
        raise ValueError('Empty shape; do not replace its assigned bin')
    score=cv2.distanceTransform(1-core,cv2.DIST_L2,5)-cv2.distanceTransform(core,cv2.DIST_L2,5)
    score=score.astype(np.float64)+rng.uniform(-1e-5,1e-5,(h,w))
    target_count=int(round(spec['target']*int(face.sum())))
    ordered=np.sort(score[face])
    threshold=(ordered[target_count-1]+ordered[target_count])/2
    mask=score<=threshold
    measured=float((mask&face).sum()/face.sum())
    if int(measured*5)!=spec['bin'] or abs(measured-spec['target'])>PLAN['target_tolerance']:
        raise ValueError('Realized mask violates assigned area stratum')
    return mask,dict(fraction=measured,angle=float(angle),center=list(map(float,center)),
                     components=int(cv2.connectedComponents((mask&face).astype(np.uint8))[0]-1),
                     mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())

def render(image,face,xy,spec):
    mask,meta=make_mask(face,spec);h,w=face.shape;rng=np.random.default_rng(seed_for(spec['seed'],'appearance'))
    color=rng.uniform(28,225,3)
    if spec['appearance']==0: texture=np.zeros((h,w,3))
    elif spec['appearance']==1: texture=cv2.resize(rng.normal(0,22,(5,5,3)).astype('float32'),(w,h),interpolation=cv2.INTER_CUBIC)
    else: texture=rng.normal(0,30,(h,w,3))
    result=image.copy();result[mask]=np.clip(color+texture[mask],0,255).astype(np.uint8)
    covered=mask.reshape(h//16,16,w//16,16).any((1,3))
    support=face.reshape(h//16,16,w//16,16).any((1,3))
    ij=np.rint(xy).astype(int).clip([0,0],[w-1,h-1]);point=mask[ij[:,1],ij[:,0]]
    patch=(ij//16);assert covered[patch[:,1],patch[:,0]][point].all()
    meta.update(spec,covered_points=int(point.sum()),point_pattern=''.join(map(str,point.astype(int))),
                crop_mask_fraction=float(mask.mean()),patch_fraction=float(covered.mean()),
                rgb_sha256=hashlib.sha256(result.tobytes()).hexdigest())
    return result,covered,support&~covered,point,meta,mask
