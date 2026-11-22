"""Deterministic synthetic masks with recorded pixel truth; no natural-mask estimation."""
import numpy as np

BANDS = ((.05, .2), (.2, .4), (.4, .7))

def make_occlusion(rgb, face, xy, face_id, variant):
    """Nine variants per face: all nine locations, balanced sizes/shapes across faces."""
    seed = int(face_id)*1009 + int(variant)*104729 + 42
    rng = np.random.default_rng(seed)
    band = variant % 3
    shape = (variant // 3 + int(face_id)) % 3
    location = (variant + int(face_id)) % 9
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    fy, fx = np.nonzero(face)
    if len(fx) < 100:
        raise ValueError("Insufficient face support")
    cx = fx.min() + (location % 3 + rng.uniform(.15, .85))/3 * (fx.max()-fx.min())
    cy = fy.min() + (location // 3 + rng.uniform(.15, .85))/3 * (fy.max()-fy.min())
    theta = rng.uniform(0, np.pi)
    dx, dy = (xx-cx)/(fx.max()-fx.min()), (yy-cy)/(fy.max()-fy.min())
    u, v = dx*np.cos(theta)+dy*np.sin(theta), -dx*np.sin(theta)+dy*np.cos(theta)
    aspect = rng.uniform(.5, 2.)
    if shape == 0:
        distance = np.maximum(np.abs(u)*aspect, np.abs(v)/aspect)
    elif shape == 1:
        distance = (u*aspect)**2 + (v/aspect)**2
    else:
        distance = (u*aspect)**2 + (v/aspect+.12*np.sin(u*14))**2
    fraction = rng.uniform(*BANDS[band])
    count = max(1, round(fraction*int(face.sum())))
    # Select exactly count face pixels, independent of image values and keypoints.
    indexes = np.flatnonzero(face)
    chosen = indexes[np.argsort(distance.flat[indexes], kind='stable')[:count]]
    mask = np.zeros(face.shape, bool); mask.flat[chosen] = True
    fill = (int(face_id)+variant) % 3
    if fill == 0:
        value = np.broadcast_to(rng.integers(0, 256, (1, 1, 3), dtype=np.uint8), rgb.shape)
    elif fill == 1:
        value = rng.integers(0, 256, rgb.shape, dtype=np.uint8)
    else:
        color = rng.uniform(0, 255, (1, 1, 3))
        value = np.clip(color + (xx[..., None]/rgb.shape[1]-.5)*rng.uniform(-160,160,(1,1,3)),0,255).astype(np.uint8)
    image = np.where(mask[..., None], value, rgb).astype(np.uint8)
    patch = mask.reshape(16,16,16,16).any((1,3))
    support = face.reshape(16,16,16,16).any((1,3))
    ix, iy = np.rint(np.asarray(xy)*255).astype(int).T
    if ((ix<0)|(ix>255)|(iy<0)|(iy>255)).any():
        raise ValueError("GT outside crop")
    return image, mask, patch, support & ~patch, mask[iy,ix], dict(
        band=band, shape=shape, location=location, fill=fill,
        fraction=float(mask.sum()/face.sum()), seed=seed,
        center=[float(cx),float(cy)])
