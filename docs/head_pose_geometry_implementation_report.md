# Head-pose geometry implementation report

Implementation date: 2026-07-26.

## Delivered architecture

- Detector-independent HEAD result protocol, deterministic prediction cache,
  and val/test ground-truth guard.
- Enlarged differentiable HEAD crops carrying RGB, valid mask, signed border
  distances, original size, both affine transforms, confidence, ID, and source.
- Realistic crop center/scale/asymmetric/aspect/boundary perturbations.
- Official-key strict `.pth` adapter and complete fused-ONNX-to-trainable
  ResNet-50 adapter with per-node diagnostics and checksums.
- Stable 6D-to-SO(3), atan2 geodesic distance, diagnostic Euler conversion,
  and semantic flip transform.
- Frozen/domain-adapter/final-stage/full-explicit pose training modes, pose
  embedding, confidence, and uncertainty.
- Fixed curved canonical contour/five-keypoint template, weak-perspective
  scale/translation, differentiable visibility, observability, uncertainty,
  and optional bounded four-value shape code.
- Weighted covariance ellipse with regularization, ordered axes, near-circle
  fallback, and doubled-angle representation. Amodal centers are unclamped.
- Pose/geometry query injection followed by residual uncertainty-gated fusion.
  Direct independent outputs and losses remain intact.
- Masked real/synthetic loss path; rejected pseudo rotations never receive
  hard geodesic loss but retain real 2D supervision.
- Head-centric dataset, deterministic derived-label/pseudo-pose scripts,
  external COCO-prediction cache converter, stage 0–5 configs, A0–A8 configs,
  documentation, and result template.

## Verification performed

```text
python -m pytest -q tests/test_head_pose_geometry.py
22 passed

python -m pytest -q tests/test_face_keypoint_cooperative.py
18 passed

python -m pytest -q tests/test_face_pose_balanced_split.py \
  tests/test_unified_face_pose_drive_v2.py
10 passed

python -m pytest -q
297 passed, 1 skipped
```

These cover special/degenerate rotations, flips, transforms, padding, amodal
projection, differentiable ellipse/angle periodicity, annotation masks,
multi-signal pseudo filtering, deterministic cache and leakage rejection,
pose-only and geometry forward/backward, mixed pose masks, checkpoint
round-trip, existing-disabled state/output compatibility, and a bounded tiny
overfit with positive ordered axes.

The fixed ONNX model restored 53 fused convolutions plus the regression layer
(108 tensors). On identical random normalized input, PyTorch diagnostic
`[yaw,pitch,roll]` degrees matched ONNX:

```text
PyTorch: [-17.03979, 170.35805, 171.63900]
ONNX:    [-17.039791,170.35805, 171.63902]
```

No full training run or performance metric was fabricated.

## Known limitations and first experiment

- Real data has no manual 3D rotation; pseudo pose remains teacher-dependent.
- The downloaded artifact is a documented public ONNX conversion, not
  byte-identical evidence of the unavailable official `.pth`.
- No persistent track IDs exist, so temporal pseudo filtering is inactive
  until an external tracker supplies identities.
- Deployment evaluation requires external predicted-HEAD cache creation. It
  intentionally cannot run with GT HEAD fallback.
- VGGHeads remains optional because its published weights are non-commercial.

The recommended first full comparison is A0 vs A1 vs A3 vs A4 with three
seeds, a frozen pose backbone, fixed template, no shape code, and the same
RTMDet-HF predicted HEAD cache for val/test. Choose thresholds on validation,
freeze them, then evaluate test once. Advance to A7 only if A4 improves the
real boundary/low-visible-keypoint metrics without regressing overall direct
metrics.
