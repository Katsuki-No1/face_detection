"""Opaque line, face, and hand silhouettes for paired face-crop restoration."""
import cv2
import numpy as np


SHAPES = ("line", "face", "hand")


def silhouette(kind, rng):
    mask = np.zeros((120, 120), np.uint8)
    if kind == "line":
        cv2.line(mask, (6, 59), (114, 61), 1, int(rng.integers(9, 23)))
    elif kind == "face":
        # Forehead, temples, cheeks and a tapering jaw; ears join the sides.
        outline = [[60,5],[78,7],[92,17],[99,34],[98,58],[92,81],[80,99],
                   [66,111],[55,113],[40,104],[28,89],[21,68],[18,43],[21,26],[34,11]]
        cv2.fillPoly(mask,[np.asarray(outline,np.int32)],1)
        for x in (19,97):
            cv2.ellipse(mask,(x,60),(6,12),0,0,360,1,-1)
    elif kind == "hand":
        cv2.ellipse(mask,(62,79),(28,30),-8,0,360,1,-1)
        cv2.fillPoly(mask,[np.int32([[43,91],[77,95],[77,119],[42,119]])],1)
        # A lateral thumb and four independently flexed fingers.
        for base,tip,thickness in [((42,83),(12,57),12),((44,65),(33,18),10),
                                   ((56,58),(52,7),11),((69,59),(70,14),10),((80,68),(90,32),9)]:
            tip=np.asarray(tip)+rng.integers([-4,-4],[5,6])
            joint=np.rint((np.asarray(base)+tip)/2+rng.integers(-3,4,2)).astype(int)
            cv2.line(mask,base,tuple(joint),1,thickness)
            cv2.line(mask,tuple(joint),tuple(tip),1,thickness)
        if rng.random()<.5:
            mask=cv2.flip(mask,1)
    else:
        raise ValueError(f"Unknown occluder {kind}")
    return mask


def occlude(image, face, xy, kind, seed):
    """Match support area on the recipient ellipse and cover an annotated point."""
    rng=np.random.default_rng(seed)
    h,w=image.shape[:2]
    ys,xs=np.nonzero(face)
    if not len(xs):
        raise ValueError("Empty recipient ellipse")
    pixel=np.rint(xy).astype(int)
    valid=((pixel>=0)&(pixel<[w,h])).all(1)
    px,py=np.clip(pixel,[0,0],[w-1,h-1]).T
    candidates=np.flatnonzero(valid & face[py,px])
    if not len(candidates):
        return None
    stamp=silhouette(kind,rng)
    sy,sx=np.nonzero(stamp)
    for _ in range(80):
        slot=int(rng.choice(candidates))
        j=rng.integers(len(sx))
        scale=np.sqrt(len(xs)*rng.uniform(.13,.56)/len(sx))
        matrix=cv2.getRotationMatrix2D((float(sx[j]),float(sy[j])),rng.uniform(-180,180),scale)
        matrix[:,2]+=xy[slot]-[sx[j],sy[j]]
        mask=cv2.warpAffine(stamp,matrix,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
        overlap=np.count_nonzero(mask & face)/len(xs)
        if .10<=overlap<=.40 and mask[py[slot],px[slot]]:
            break
    else:
        return None
    # Vary appearance independently of shape. Preserve every pixel outside the paste.
    texture=cv2.resize(rng.normal(0,18,(5,5,3)).astype("float32"),(w,h),interpolation=cv2.INTER_CUBIC)
    color=rng.uniform(28,225,3)
    result=image.copy()
    result[mask]=np.clip(color+texture[mask],0,255).astype(np.uint8)
    return result,mask
