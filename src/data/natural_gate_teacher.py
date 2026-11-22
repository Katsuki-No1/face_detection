"""Natural five-point weak labels in the canonical ellipse coordinate frame."""
import numpy as np
from scipy.spatial import Delaunay


def smooth(t):
    return t * t * (3 - 2 * t)


def field(anchors, values, queries):
    """Interpolate inside and extend the closest hull edge with declining confidence."""
    if not len(anchors):
        return np.zeros(len(queries)), np.zeros(len(queries))
    if len(anchors) == 1:
        squared = ((queries - anchors[0]) ** 2).sum(1)
        return np.full(len(queries), values[0]), np.exp(-squared / (2 * .32**2))
    centered = anchors - anchors.mean(0)
    rank = np.linalg.matrix_rank(centered, tol=1.e-6)
    inside = np.zeros(len(queries), dtype=bool)
    result = np.zeros(len(queries))
    if rank == 2:
        tri = Delaunay(anchors)
        idx = tri.find_simplex(queries, tol=1.e-10)
        inside = idx >= 0
        tr = tri.transform[idx[inside]]
        b = np.einsum("ijk,ik->ij", tr[:, :2], queries[inside] - tr[:, 2])
        b = smooth(np.clip(np.column_stack([b, 1-b.sum(1)]), 0, 1))
        result[inside] = (b * values[tri.simplices[idx[inside]]]).sum(1) / b.sum(1)
        edges = tri.convex_hull
    else:
        order = np.argsort(anchors[:, np.argmax(np.ptp(anchors, axis=0))])
        edges = zip(order[:-1], order[1:])
    ids = np.flatnonzero(~inside)
    nearest = np.full(len(ids), np.inf)
    for a, b in edges:
        edge = anchors[b] - anchors[a]
        t = np.clip((queries[ids]-anchors[a]) @ edge / (edge @ edge), 0, 1)
        squared = ((queries[ids] - anchors[a] - t[:, None]*edge)**2).sum(1)
        use = squared < nearest
        nearest[use] = squared[use]
        result[ids[use]] = values[a] + smooth(t[use])*(values[b]-values[a])
    weight = np.ones(len(queries))
    weight[ids] = np.exp(-nearest/(2*.32**2))
    return result, weight


def patch_teacher(ellipse, points, states, transform, grid=16, samples=4):
    cx, cy, r, s, angle = ellipse
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def normalized(xy):
        return (xy - [cx, cy]) @ rot / [r,s]
    p = normalized(points)
    valid = (states > 0) & (np.linalg.norm(p, axis=1) <= 1)
    anchors = list(p[valid])
    labels = list((states[valid] == 1).astype(float))
    for eye, mouth in ((0,3), (1,4)):
        if not (valid[[eye,2,mouth]].all() and states[eye]==states[mouth]==1 and states[2]==2):
            continue
        span = p[mouth,0]-p[eye,0]
        if abs(span) < 1.e-6:
            continue
        u = (p[2,0]-p[eye,0])/span
        if 0 < u < 1:
            valley = p[eye] + u*(p[mouth]-p[eye])
            if np.linalg.norm(np.asarray(anchors)-valley, axis=1).min() > 1.e-5:
                anchors.append(valley)
                labels.append(0.)
    anchors = np.asarray(anchors).reshape(-1,2)
    distinct, groups = np.unique(anchors, axis=0, return_inverse=True)
    # Conflicting colocated annotations cannot supply a spatial target.
    values = [np.mean(np.asarray(labels)[groups==i]) for i in range(len(distinct))]
    keep = np.isin(values, [0.,1.])
    axis = (np.arange(grid*samples)+.5)*transform.output_size/(grid*samples)-.5
    xx, yy = np.meshgrid(axis, axis)
    q = normalized(transform.crop_to_original(np.column_stack([xx.ravel(), yy.ravel()])))
    y, w = field(distinct[keep], np.asarray(values)[keep], q)
    w *= (np.linalg.norm(q,axis=1)<=1)
    def pool(array):
        return array.reshape(grid,samples,grid,samples).mean((1,3))
    weight = pool(w)
    return (pool(w*y)/np.maximum(weight,1.e-12)).astype("float32"), weight.astype("float32")
