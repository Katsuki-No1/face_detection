"""Predicted FACE geometry in source-image and augmented crop coordinates.

No GT or HEAD fallback is allowed. Pixel centers use x-right/y-down; feature
cell centers are (index + 0.5) * crop_size / grid_size - 0.5. Source scale is
major diameter / sqrt(source width * source height); theta is crop-frame,
clockwise from image x, modulo pi, after affine and optional horizontal flip.
"""
import math
import numpy as np
import torch
from src.data.face_crop_cache import canonicalize_ellipse


def predicted_geometry_inputs(prediction, *, affine, flipped, source_size, crop_size, grid_size):
    if not isinstance(prediction, dict) or prediction.get("ellipse") is None:
        raise ValueError("Geometry prior requires an actual predicted FACE ellipse; GT fallback is forbidden")
    ellipse = np.asarray(canonicalize_ellipse(prediction["ellipse"]), dtype=np.float64)
    width, height = (int(v) for v in source_size)
    gh, gw = (int(v) for v in grid_size)
    ch, cw = (int(v) for v in crop_size)
    if min(width,height,gh,gw,ch,cw) <= 0:
        raise ValueError("Source/crop/grid dimensions must be positive")
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (2,3) or not np.isfinite(matrix).all():
        raise ValueError("Finite source-to-crop affine required")
    singular = np.linalg.svd(matrix[:,:2], compute_uv=False)
    if singular.min() <= 0 or not np.isclose(singular[0],singular[1],rtol=1e-5,atol=1e-8):
        raise ValueError("Geometry prior requires the isotropic long-axis-square crop contract")
    complete = np.eye(3);complete[:2] = matrix
    if flipped:
        complete = np.array([[-1.,0.,cw-1],[0.,1.,0.],[0.,0.,1.]]) @ complete
    center = complete @ np.array([ellipse[0], ellipse[1], 1.])
    major_direction = complete[:2,:2] @ np.array([math.cos(ellipse[4]), math.sin(ellipse[4])])
    crop_theta = math.atan2(major_direction[1], major_direction[0]) % math.pi
    crop_ellipse = np.array([center[0], center[1], ellipse[2]*singular[0], ellipse[3]*singular[0], crop_theta])
    # Test support against the source ellipse to avoid radius rescaling ambiguity.
    y,x = np.meshgrid((np.arange(gh)+.5)*ch/gh-.5,(np.arange(gw)+.5)*cw/gw-.5,indexing="ij")
    points = np.stack([x,y,np.ones_like(x)],axis=-1) @ np.linalg.inv(complete).T
    delta = points[...,:2]-ellipse[:2]
    c,s = math.cos(ellipse[4]),math.sin(ellipse[4])
    u = delta[...,0]*c+delta[...,1]*s
    v = -delta[...,0]*s+delta[...,1]*c
    support = (u/ellipse[2])**2+(v/ellipse[3])**2 <= 1
    if not support.any():
        raise ValueError("Predicted ellipse has no feature-cell support in this crop")
    geometry = [2*ellipse[2]/math.sqrt(width*height),ellipse[2]/ellipse[3],crop_ellipse[4]%math.pi]
    return dict(geometry=torch.tensor(geometry,dtype=torch.float32),
        support=torch.from_numpy(support),predicted_ellipse_source=torch.tensor(ellipse,dtype=torch.float32),
        predicted_ellipse_crop=torch.tensor(crop_ellipse,dtype=torch.float32))
