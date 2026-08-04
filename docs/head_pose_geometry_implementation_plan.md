# Head pose and lightweight 3D geometry: repository audit and implementation plan

## Audit scope and result

Audit date: 2026-07-26.

There was no `AGENTS.md` inside this repository before this work. The only file
found by the parent-directory search was in a sibling repository and is not
applicable. A small root `AGENTS.md` now records only commands, experiment
rules, and leakage constraints.

The active integration target is
`src/models/face_keypoint_cooperative/FaceKeypointCooperativeDETR`. It already
has independently supervised:

- a full-frame DEIMv2-L direct FACE bbox/ellipse query branch;
- a full-frame five-semantic-keypoint query branch;
- learned one-to-one training matches and learned FACE/keypoint association;
- residual bidirectional cooperative refinement;
- independent and refined inference/evaluation outputs.

The new components will extend this model behind disabled-by-default switches.
The separate RTMDet-HF and Co-DINO HEAD detectors remain separate models.

The following entry points were inspected:

- model/build/checkpoint:
  `src/models/face_keypoint_cooperative/`,
  `src/models/dinov3_deimv2/`,
  `src/models/dinov3_deimv2_ellipse/`,
  `src/models/head_codino/`;
- data/augmentation:
  `keypoint_comparison2/data.py`,
  `src/data/face_keypoint_dataset.py`,
  `tools/face_keypoint_cooperative_common.py`;
- training/checkpoint resume:
  `tools/train_face_keypoint_cooperative.py`;
- evaluation/inference:
  `tools/eval_face_keypoint_cooperative.py`,
  `tools/infer_face_keypoint_cooperative.py`;
- split preparation and leakage audit:
  `configs/datasets/default_face_pose.yaml`,
  `scripts/prepare_default_face_pose_dataset.py`,
  `src/data/coco_split_audit.py`;
- external HEAD paths:
  `scripts/train_head_detector.py`,
  `src/models/head_codino/`,
  `configs/rtmdet/`,
  `configs/head_codino/`.

## Resolved annotation and coordinate contract

### Image, boxes, ellipses, and keypoints

- Original image coordinates are pixels with origin at the top left, `+x`
  right and `+y` down.
- COCO FACE bbox values are visible/clipped `xywh`. Dataset tensors use pixel
  `xyxy` before the detector adapter converts them to normalized `cxcywh`.
- FACE ellipse values are
  `[cx, cy, semi_major, semi_minor, theta_rad]`. `theta` is the major-axis
  angle modulo pi, canonicalized to `[-pi/2, pi/2)`.
- Full ellipse values are retained even when the center or contour is outside
  the image. On the fixed dataset, 284 ellipse centers and 1,503 contours are
  outside/truncated across all splits. The matching bbox is clipped.
- Direct ellipse targets normalize center by `(W,H)` and axes by
  `max(W,H)` without clamping the ellipse. The direct ellipse head predicts an
  additive center residual around the bounded bbox center, so its ellipse
  center is not itself sigmoid-clamped. Rendering AABBs are clipped separately.
- Existing direct ellipse angle loss already uses the pi-periodic
  `1-cos(2 delta_theta)` convention. The new geometry path will expose
  `[cos(2 theta), sin(2 theta)]` and keep amodal coordinates separate.
- Five keypoint identities are fixed:
  `left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner`.
  Horizontal flip order is `[1,0,2,4,3]`.
- Visibility is COCO-style: `0` missing/outside image, `1` annotated but
  occluded, `2` annotated and visible. Existing training correctly supervises
  coordinates for states 1 and 2 and visibility only as a binary target.
  State 0 receives no coordinate loss.
- Missing keypoints have no pose meaning. In particular they are not
  rear-view labels.

### HEAD and FACE association

The canonical metadata contains manual HEAD boxes and FACE ellipses. The
head-centric export uses deterministic Hungarian one-to-one assignment based
on containment, coverage, and normalized center distance. It records
`source_head_index`, optional `source_face_index`, and the association audit.
These indices are local to one image. There is no persistent person/track ID.

The head-centric fixed split contains:

| split | images | HEAD | accepted HEAD–FACE attributes | zero labeled keypoints |
|---|---:|---:|---:|---:|
| train | 4,237 | 7,129 | 6,447 | 1,063 |
| val | 530 | 882 | 834 | 95 |
| test | 528 | 806 | 766 | 86 |

Unmatched/invalid HEAD attributes remain valid HEAD samples but must not receive
FACE projection targets.

### Split and leakage contract

`face_pose_drive_v2_balanced_811` is fixed at 4,237/530/528 images and
44/6/6 source videos. The existing SHA-256/source-video/path leakage audit
passes with zero overlap. Pseudo-pose generation will reject any split other
than `train` by default and will store the source split and source-video ID in
every cache entry.

### Current validation behavior

The cooperative model currently evaluates the full image and does not consume a
HEAD crop. The head-centric dataset path uses ground-truth HEAD boxes. New
deployment-style pose/geometry validation and test therefore require a cached
or live predicted-HEAD source; tools will fail instead of silently falling back
to ground truth. Ground-truth and jittered-ground-truth crops remain permitted
for training and explicitly labeled oracle analysis.

## Rotation convention

The canonical internal pose is a proper `3x3` rotation matrix. It is an active
rotation from canonical head coordinates to camera coordinates and multiplies
column vectors:

`v_camera = R_camera_from_head @ v_head`.

Both coordinate frames are right-handed with `+X` image-right, `+Y` image-down,
and `+Z` forward into the scene. The canonical face looks toward the camera, so
its outward face normal is `-Z`. Identity is frontal.

Euler values are diagnostic only and use
`R = Rz(roll) @ Ry(yaw) @ Rx(pitch)`:

- positive pitch is a right-handed rotation about `+X` (nose moves down near
  identity);
- positive yaw is about `+Y` (nose moves image-left near identity);
- positive roll is about `+Z` (clockwise in the image near identity).

Horizontal image reflection uses `F = diag(-1,1,1)` and
`R_flipped = F @ R @ F`. This preserves determinant +1 and changes yaw and
roll signs while preserving pitch away from Euler singularities.

The official 6DRepNet360 code emits two 3-vectors, normalizes the first,
orthogonalizes/normalizes the second through a cross product, and concatenates
the resulting basis as matrix columns. The local implementation uses the
equivalent stable Gram-Schmidt form with explicit epsilon handling.

## Pretrained reference audit

- `thohemp/6DRepNet360`, commit
  `b06e4fed77f802fbf05319b85bb65215087d5c7a`, is MIT licensed. The official
  full-rotation architecture is ResNet-50 plus a six-value regression layer,
  224 crop input, ImageNet normalization, 6D-to-SO(3), and Euler conversion
  only for reporting.
- The official checkpoint URL in its README returns HTTP 404 as of the audit
  date. A full-rotation ONNX conversion was downloaded from the public
  PINTO model-zoo archive using the terminal. It is stored only under ignored
  `checkpoint/head_pose_geometry/`.
- Dynamic-batch ONNX SHA-256:
  `852ea8e28014cee888a779fa235082fe3424e0ef2dd786c5c5496745741a46d8`.
  Fixed-batch ONNX SHA-256:
  `422b69d17bf02e164bd1062f5e970f67f1688cadf7d700800fad5fa12a16fc82`.
  These are treated as converted teacher weights, not falsely described as the
  original `.pth`.
- The model graph contains the complete fused ResNet-50 convolution and linear
  initializers. A loader will diagnose the source and either restore them into
  the fused trainable adapter or use ONNX Runtime for frozen teacher inference.
- `KupynOrest/head_detector` source is MIT licensed. Its published VGGHeads
  model card is CC-BY-NC-4.0. It remains an optional second teacher and is not
  downloaded or enabled by default.

No third-party source file is copied into this repository. Local code follows
the published algorithms and records attribution.

## Implementation plan

1. Add a standalone `src/head_pose_geometry` package with:
   rotation math/losses; typed external HEAD interface; context crop builder
   with valid masks and affine transforms; official/fused 6DRepNet360 adapter;
   canonical template projection; differentiable covariance ellipse; visibility
   and uncertainty outputs; cache schemas and filtering.
2. Add deterministic derived-label and pseudo-pose tools. Derived labels use
   numerical ellipse/image intersection and preserve both full and visible
   forms. Pseudo acceptance uses confidence plus crop-jitter and flip
   geodesic consistency, with optional temporal/second-teacher agreement.
3. Add an optional query bridge. It consumes matched HEAD/pose instances and
   produces pose/geometry tokens, projected ellipse/keypoint references, and
   uncertainty/observability features. Gated residual fusion never overwrites
   independent direct outputs.
4. Extend the cooperative builder/forward/loss/optimizer/decode paths only when
   `model.head_pose_geometry.enabled: true`. With the switch absent/false, do
   not instantiate new parameters and preserve existing state keys and outputs.
5. Add stage and A0–A8 configs, training/evaluation documentation, stratified
   result templates, dependency/license report, tests, smoke/overfit tools, and
   deployment-source guards.
6. Run focused unit/integration tests, existing cooperative regression tests,
   forward/backward, checkpoint round-trip, deterministic cache/leakage tests,
   and a bounded tiny overfit. Do not run a full experiment or fabricate
   metrics.

## Remaining limitations, not invented conventions

- There is no manual ground-truth 3D rotation in the fixed real dataset.
  Accepted teacher labels are pseudo supervision; low-quality samples receive
  only 2D/consistency objectives.
- No persistent identity/track ID exists. Temporal filtering can be used only
  when a caller supplies a track ID or an external deterministic association.
- The converted ONNX source does not prove byte identity with the now-missing
  official `.pth`; provenance is reported accurately. A user-supplied official
  `.pth` remains the preferred trainable initialization and is loaded with
  explicit key/shape diagnostics.
- A real performance gain cannot be claimed until the staged experiments are
  trained and evaluated with validation-selected thresholds and predicted HEAD
  detections.
