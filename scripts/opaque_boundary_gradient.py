"""Opaque RGB boundary correction, using only target pixels outside the paste.

This is a color-offset field, not alpha compositing. Target texture beneath the
paste is never read. A narrow inner band interpolates the boundary color offset
to zero; the donor interior and the target exterior are unchanged.
"""
import cv2
import numpy as np

DEFAULT_WIDTH = 48
DEFAULT_STRENGTH = 2.0


def opaque_boundary_gradient(target, pasted, mask, width=DEFAULT_WIDTH, strength=DEFAULT_STRENGTH):
    if target.shape != pasted.shape or target.shape[:2] != mask.shape:
        raise ValueError('Incompatible image and mask shapes')
    if mask.dtype != np.bool_ or type(width) is not int or width < 1:
        raise ValueError('Binary support and positive band width required')
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('Strength must be positive and finite')
    if not mask.any():
        return pasted.copy()
    h, w = mask.shape
    # Missing neighbors at the image edge are NOT treated as visible background.
    sums = np.zeros((*mask.shape, 3), np.float32)
    counts = np.zeros(mask.shape, np.float32)
    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
        yy, xx = np.where(mask)
        ny, nx = yy+dy, xx+dx
        valid = (ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        yy,xx,ny,nx = yy[valid],xx[valid],ny[valid],nx[valid]
        exposed = ~mask[ny,nx]
        yy,xx,ny,nx = yy[exposed],xx[exposed],ny[exposed],nx[exposed]
        sums[yy,xx] += target[ny,nx].astype(np.float32)-pasted[yy,xx].astype(np.float32)
        counts[yy,xx] += 1
    boundary = counts > 0
    if not boundary.any():
        return pasted.copy()
    # Match the actual recipient color immediately OUTSIDE the pasted crop.
    # Neither a correction cap nor smoothing may weaken this boundary value.
    # At corners with two exterior neighbors, their mean is the boundary color.
    fixed = sums/np.maximum(counts[...,None],1)
    distance = cv2.distanceTransform((~boundary).astype(np.uint8),cv2.DIST_L2,5)
    band = mask & (distance < width)
    # Fourth channel is an inward influence field: 1 at seam, 0 beyond band.
    field = np.zeros((*mask.shape,4),np.float32)
    field[boundary,:3] = fixed[boundary]
    field[boundary,3] = 1
    kernel=np.array([[0,.25,0],[.25,0,.25],[0,.25,0]],np.float32)
    # Dirichlet boundary at the exposed seam, zero at the inner edge of band.
    for _ in range(width*width//2):
        field = cv2.filter2D(field,-1,kernel,borderType=cv2.BORDER_REPLICATE)
        field[~band] = 0
        field[boundary,:3] = fixed[boundary]
        field[boundary,3] = 1
    # s*u/(1+(s-1)*u) retains exact endpoint colors and carries more correction
    # inward for s>1. This weights a color-offset field, not target-image alpha.
    influence=np.clip(field[...,3],0,1)
    gain=strength/(1+(strength-1)*influence)
    correction=field[...,:3]*gain[...,None]
    result=pasted.copy()
    result[band] = np.rint(np.clip(pasted[band].astype(np.float32)+correction[band],0,255)).astype(np.uint8)
    return result
