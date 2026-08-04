# Head-pose geometry coordinate conventions

This file is normative for `src/head_pose_geometry`.

## Image and annotation coordinates

- Images use pixel coordinates with origin at the top left, `+x` right and
  `+y` down.
- HEAD and rendered boxes use `xyxy` pixels. `x2,y2` are the exclusive image
  extent in valid-region metadata.
- Direct model boxes use normalized `cx,cy,w,h`.
- FACE ellipses use
  `[cx, cy, semi_major, semi_minor, theta]`. The major axis is first and
  `theta` is canonicalized modulo pi into `[-pi/2,pi/2)`.
- Ellipse centers and contours are amodal and are not clamped to the image.
  Rendering may use a separate clipped polygon/AABB.
- Geometry ellipses normalize center by `(W,H)` and both axes by `max(W,H)`.
- Keypoints are ordered
  `left_eye,right_eye,nose_tip,left_mouth_corner,right_mouth_corner`.
  Their normalized coordinates divide by `(W,H)`.
- Keypoint state `0` is unsupervised, `1` is coordinate-supervised but
  occluded, and `2` is coordinate-supervised and visible. Missing never means
  rear view.

The crop transform is a column-vector homogeneous transform:

```text
[x_image, y_image, 1]^T = crop_to_image @ [x_crop, y_crop, 1]^T
```

`valid_mask=1` denotes a source pixel inside the valid image region. Padding
and synthetic boundary removal are zero. Border features are signed normalized
distances from the HEAD box to left/top/right/bottom valid-region edges.

## Rotation

`R` is a right-handed active rotation from canonical head coordinates into
camera coordinates:

```text
v_camera = R_camera_from_head @ v_head
```

Both frames use `+X` image-right, `+Y` image-down, and `+Z` into the scene.
The canonical outward face normal is `-Z`; identity is frontal.

Euler angles are diagnostics only, in `[pitch_x,yaw_y,roll_z]` order:

```text
R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
```

Near identity:

- positive pitch moves the nose down;
- positive yaw moves the nose image-left;
- positive roll is clockwise in the image.

The network target is always a proper `3x3` matrix. Six-value output stores
the first two matrix columns; stable Gram–Schmidt recovers SO(3). Rotation
error is `atan2(sin,cos)` geodesic distance, including near 180 degrees.

Horizontal semantic image flipping uses:

```text
F = diag(-1, 1, 1)
R_flipped = F @ R @ F
```

This remains in SO(3), preserves pitch, and changes yaw/roll signs away from
Euler singularities. FACE angle loss/fusion uses
`[cos(2 theta),sin(2 theta)]`; raw angles or raw axis parameters are never
averaged.
