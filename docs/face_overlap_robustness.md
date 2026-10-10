# Face Overlap Robustness (FOR)

2026-10-09 ablation follow-up: schedule a matched FOR-off control alongside the
FOR-on real-face augmentation arm. Both use identical post-composition detections,
target crops, frozen detector, augmentation manifests, common initial tensors and
training contracts. Only query-feature/predicted-geometry conditioning is toggled.
This tests the whole FOR v1 design, not a person-ID embedding (v1 has none).
Use validation for the use/no-use decision; overlap/misattribution definitions
must be frozen before production. Details and arm IDs are in the training plan.
Both arms remain pending integration, with independent four-gate requirements.

2026-10-09 follow-up: the user scheduled face-overlap augmentation for both the
FACE/ellipse detector and crop-keypoint/FOR training, plus a separate Messenger
recovery experiment. The shared operation uses opaque crop-side width48/strength2
color correction. See `docs/real_face_occlusion_training_schedule.md` and
`configs/training_plans/real_face_occlusion_w48_s2.yaml`. The new FOR arm is
scheduled pending composite-image/query integration; it has not trained. The
original v1 config and implementation-only registry remain unchanged history.
Do not reuse clean cached full-image queries for a newly occluded crop.

2026-10-09 — implementation v1; training and accuracy evaluation: **not run**.

**FOR is the extensible project name for work aiming to improve robustness to
overlapping faces, especially keypoints attributed to the wrong face.** The name
states the intended purpose, not a measured performance claim. Future other-face
queries, overlap augmentation or attribution losses can be separately versioned
under this name. They are not implemented or authorized for training by v1.

## Topology and scope

HEAD-free ordered cascade: full-image FACE/ellipse → face-crop DINO →
target-conditioned keypoint Query Refinement. No external HEAD stage.
The current local baseline remains `mal_alpha2_ellipse_from_existing_wider`
(FACE/ellipse only); the global reference remains
`reference_model/face_dino_v2_distribution`. FOR is an independent candidate,
not a baseline promotion or an evaluated alpha2 + keypoint composition.

V1 transports the selected detection's final decoder query and predicted
bbox/ellipse to the crop model. A zero-initialized residual adds projected face
features and crop geometry to all five semantic queries before decoding.
The initial and refinement stages consequently share target-conditioned queries.
No ID classification, person recognition, tracking, depth ordering, RGB mask,
cross-face penalty or overlapping-face augmentation is introduced.

## Interfaces and compatibility

- Detector: `set_face_query_output(True)` opts into `face_query_index` and
  `face_query_features`. The default output/state-dict contract is unchanged.
  `DeimFaceDetector(return_face_query_features=True)` carries them through
  confidence sorting, validity filtering, detection caps and serialization.
- IDs are image-local original decoder slots, not stable identities across frames.
  Numeric IDs never enter a learned embedding; they identify the corresponding
  feature vector and geometry.
- Model config: `model.face_overlap_robustness: {enabled: true, query_dim: 256}`.
  Omitted/disabled creates no new parameter keys. Enabled parameters are under
  `head.query_head.face_overlap.*` for Query Refinement.
- Crop model inputs: `face_query_features[B,C]` and `face_geometry[B,10]`.
  `crop_face_context(detection, transform, flipped=...)` constructs these;
  `face_context_kwargs(model, targets, device)` batches the same fields for
  training and evaluation. Missing/invalid context is rejected when enabled.
- Existing checkpoint warm starts may allow **only** the new
  `head.query_head.face_overlap.` prefix as missing using the existing
  `finetune_checkpoint.allowed_missing_prefixes` interface with `strict: false`.
  Ordinary resumed FOR checkpoints still load strictly. No checkpoint was changed.
- Legacy ONNX/TensorRT output signatures remain unchanged. Query-bearing export
  via the legacy deploy tuple is rejected; a FOR export is future work.

## Coordinate convention

Source coordinates use x right/y down, bbox xyxy and ellipse
`[cx, cy, radius1, radius2, theta_radians]`. Geometry uses exactly the image crop's
affine transform, composed with horizontal reflection when applied. The bbox is
the envelope of all four transformed corners. The ellipse is transformed
analytically using its axis covariance and canonicalized to major/minor radii.

The ten values are `[x1,y1,x2,y2,cx,cy,major,minor,cos(2θ),sin(2θ)]`.
Coordinates/radii are divided by `crop_size - 1`; values outside the crop remain
unclipped. Double-angle encoding makes the ellipse orientation π-periodic.
All geometry comes from predictions, never the target's GT ellipse.

## Training data transport and limits

`keypoint_comparison1/configs/face_overlap_robustness_v1.yaml` is an independent
candidate config. It retains seed42, nominal max40 epochs, effective batch64,
staged unfreeze, immediate post-warmup decay, epoch validation and the existing
conditional semantic Macro F1 NME0.10 selection contract. These are prospective
settings, not an executed run or an authenticated batch plan. It uses the local
baseline detector with common score0.33; no threshold is selected using test.

The detector stays frozen through precomputed context. Training and validation
use their separate predicted-face crop caches. Cache metadata authenticates
annotation/config/checkpoint hashes, score/cap, image inventory and context schema;
query vectors/IDs are checked for dimensions, finiteness and per-image uniqueness.
Training supervision is accepted only from the train split. Validation uses the
full 530-image adapter (840 FACE annotation records, 790 with usable keypoints),
derived from the canonical split; these are annotation counts, not model results.
Missed detections are excluded from matched-crop training, and cache matching
coverage is reported. Primary evaluation still accounts for detector misses.

Preparation entry point (not executed in this task):

```bash
.venv/bin/python scripts/prepare_face_overlap_context.py \
  --config keypoint_comparison1/configs/face_overlap_robustness_v1.yaml \
  --splits train val
```

It checks the locked annotation hashes, preserves existing output files, and
produces no training-gate success markers. Test cache generation is separate and
explicit. Generated features stay under `outputs/`, never in source control.

Crop geometric/photometric augmentation is supported. Crop-only synthetic
occlusion is rejected: cached full-image queries would expose pre-occlusion
information. Future face-paste augmentation must run before full-image detector
feature extraction and update instance geometry/labels consistently. GT-crop
oracle diagnostics have no genuine corresponding detector query and are marked
unsupported for FOR, instead of fabricating context.

The canonical queue records FOR under `implementation_only_architectures`, with
no runnable job. Production requires explicit scheduling, four fresh gates,
synchronized 4/8/16 batch preflight, runtime OOM fallback, W&B and separately
registered common val/test evaluation. Existing jobs and baseline registries
remain unchanged.

## Verification

CPU tests in `tests/test_face_overlap_robustness.py` exercise actual decoder query
extraction, identical detector predictions, filtering/order/cap, empty output,
analytic geometry under rotation/flip, training/validation cache transport,
provenance rejection, gradient flow, conditioned-query response, strict disabled
checkpoint compatibility and identical warm-start predictions. These establish
implementation contracts, not occlusion robustness or real-GPU gate success.

Chronological record: `docs/development_journal.md`, entry
`face-overlap-robustness-v1-20261009`.
