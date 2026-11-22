"""Versioned GT side correction using uniquely owned eye/nose/mouth geometry.

No predictions, Head boxes, or nearest-face decisions enter this correction.
Image x-right/y-down. Ellipses are [cx,cy,rx,ry,theta_radians].
"""
from __future__ import annotations

import math

V1_POLICY = "unique_ellipse_updown_v1"
V2_POLICY = "unique_ellipse_updown_v2"
POLICY = "unique_ellipse_rotating_axis_v3"
LEGACY_POLICY = "legacy_image_x"


def ellipse_contains(point, ellipse):
    if len(ellipse) != 5 or not all(math.isfinite(float(v)) for v in ellipse) or min(ellipse[2:4]) <= 0:
        raise ValueError("Finite ellipse with positive radii required")
    cx, cy, rx, ry, angle = ellipse
    dx, dy = point[0] - cx, point[1] - cy
    c, s = math.cos(angle), math.sin(angle)
    return ((c*dx+s*dy)/rx)**2 + ((-s*dx+c*dy)/ry)**2 <= 1.0 + 1e-9


def side_permutation(points, states, ellipses, face_index, *, policy=POLICY):
    """Return an idempotent slot permutation and an auditable decision.

    V3 uses a directed rotation-relative axis, with nose-origin fallback.
    Any complete pair not straddling the axis blocks the whole face. V1/V2
    remain explicit historical policies. Shared/outside observed points always
    block changes; coordinates and visibility must follow the returned slots.
    """
    identity = list(range(5))
    audit = dict(policy=policy, status="pending", inverted=None, permutation=identity.copy())
    if policy == LEGACY_POLICY:
        audit["status"] = "legacy_unchanged"
        return identity, audit
    if policy not in (POLICY, V1_POLICY, V2_POLICY):
        raise ValueError(f"Unknown keypoint side policy: {policy}")
    if len(points) != 5 or len(states) != 5 or any(s not in (0, 1, 2) for s in states):
        raise ValueError("Expected five points and semantic states 0/1/2")
    if not 0 <= face_index < len(ellipses):
        raise ValueError("Target ellipse index outside inventory")
    for p, state in zip(points, states):
        if len(p) != 2 or (state and not all(math.isfinite(float(v)) for v in p)):
            raise ValueError("Observed GT points must be finite 2D coordinates")
    # Validate even unused ellipses: an invalid competing ellipse makes unique
    # containment unprovable. Never silently omit it from ownership checks.
    for e in ellipses:
        ellipse_contains((0., 0.), e)
    owners = [[i for i, e in enumerate(ellipses) if ellipse_contains(p, e)] if state else []
              for p, state in zip(points, states)]
    audit["ellipse_memberships"] = owners
    if any(state and len(ids) > 1 for ids, state in zip(owners, states)):
        audit["status"] = "ambiguous_multiple_ellipses"
    elif any(state and ids != [face_index] for ids, state in zip(owners, states)):
        audit["status"] = "not_uniquely_inside_target"
    elif policy == POLICY:
        return _rotating_axis_permutation(points, states, audit)
    elif policy == V2_POLICY:
        return _two_type_permutation(points, states, audit)
    elif not all(states):
        audit["status"] = "incomplete_orientation_points"
    else:
        eye = [(points[0][j]+points[1][j])/2 for j in (0, 1)]
        mouth = [(points[3][j]+points[4][j])/2 for j in (0, 1)]
        axis = [mouth[j]-eye[j] for j in (0, 1)]
        length2 = sum(v*v for v in axis)
        audit.update(eye_center=eye, mouth_center=mouth, eye_to_mouth=axis)
        if length2 <= 1e-18:
            audit["status"] = "degenerate_orientation"
            return identity, audit
        t = sum((points[2][j]-eye[j])*axis[j] for j in (0, 1))/length2
        audit["nose_axis_fraction"] = t
        # Only numerical tolerances; no threshold selected on model metrics.
        epsilon = 1e-9 * max(1., math.sqrt(length2))
        if not 1e-9 < t < 1-1e-9:
            audit["status"] = "inconsistent_eye_nose_mouth_order"
        elif abs(axis[1]) <= epsilon:
            audit["status"] = "horizontal_orientation_undetermined"
        elif axis[1] > 0:
            audit.update(status="upright_unchanged", inverted=False)
        elif any(abs(points[a][0]-points[b][0]) <= epsilon for a, b in ((0,1), (3,4))):
            audit.update(status="side_order_undetermined", inverted=True)
        else:
            permutation = identity.copy()
            for a, b in ((0,1), (3,4)):
                if points[a][0] < points[b][0]:
                    permutation[a], permutation[b] = b, a
            audit.update(status="swapped" if permutation != identity else "already_oriented",
                         inverted=True, permutation=permutation)
            return permutation, audit
    return identity, audit


def _two_type_permutation(points, states, audit):
    """Abstain on contradictory vertical ordering; never infer an absent mate.

    Type centers average only observed points, regardless of their side labels.
    A single observed member of an eye/mouth pair can establish type direction,
    but cannot establish that pair's left/right order, so is never swapped.
    """
    identity = list(range(5))
    centers = {}
    for name, slots in (("eye", (0, 1)), ("nose", (2,)), ("mouth", (3, 4))):
        observed = [points[i] for i in slots if states[i]]
        if observed:
            centers[name] = [sum(p[j] for p in observed)/len(observed) for j in (0, 1)]
    audit["type_centers"] = centers
    if len(centers) < 2:
        audit["status"] = "insufficient_keypoint_types"
        return identity, audit
    values = list(centers.values())  # anatomical order: eye -> nose -> mouth
    scale = max(1., *(math.dist(a, b) for i, a in enumerate(values) for b in values[i+1:]))
    epsilon = 1e-9 * scale
    dy = [b[1]-a[1] for i, a in enumerate(values) for b in values[i+1:]]
    audit.update(vertical_differences=dy, numerical_epsilon=epsilon)
    if any(abs(d) <= epsilon for d in dy):
        audit["status"] = "horizontal_orientation_undetermined"
    elif all(d > epsilon for d in dy):
        audit.update(status="upright_unchanged", inverted=False)
    elif not all(d < -epsilon for d in dy):
        audit["status"] = "inconsistent_vertical_order"
    else:
        audit["inverted"] = True
        pairs = [(a, b) for a, b in ((0, 1), (3, 4)) if states[a] and states[b]]
        if not pairs:
            audit["status"] = "no_complete_side_pair"
        elif any(abs(points[a][0]-points[b][0]) <= epsilon for a, b in pairs):
            audit["status"] = "side_order_undetermined"
        else:
            permutation = identity.copy()
            for a, b in pairs:
                if points[a][0] < points[b][0]:
                    permutation[a], permutation[b] = b, a
            audit.update(status="swapped" if permutation != identity else "already_oriented",
                         permutation=permutation)
            return permutation, audit
    return identity, audit


def _rotating_axis_permutation(points, states, audit):
    """Skip the ENTIRE face on same-side eyes, even if mouth could swap."""
    from .keypoint_rotation_axis import directed_axis

    identity = list(range(5))
    axis = directed_axis(points, states, nose_fallback=True)
    audit["rotating_axis"] = axis
    if axis["status"] != "axis_available":
        audit["status"] = axis["status"]
        return identity, audit
    for name in ("eyes", "mouth"):
        if axis["pairs"][name]["status"] == "axis_not_between_pair":
            audit["status"] = "axis_not_between_" + name
            return identity, audit
    permutation = identity.copy()
    for name, (a, b) in (("eyes", (0, 1)), ("mouth", (3, 4))):
        if axis["pairs"][name]["status"] == "swap":
            permutation[a], permutation[b] = b, a
    audit.update(status="swapped" if permutation != identity else "already_oriented",
                 permutation=permutation)
    return permutation, audit
