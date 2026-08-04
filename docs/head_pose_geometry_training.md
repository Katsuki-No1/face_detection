# Head-pose geometry training

## Prerequisites

Use the repository virtual environment. The pretrained files are local and
git-ignored:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/verify_head_pose_weights.py \
  checkpoint/head_pose_geometry/sixdrepnet360_1x3x224x224_full.onnx
```

The fixed-batch converted model currently used by this workspace has SHA-256
`422b69d17bf02e164bd1062f5e970f67f1688cadf7d700800fad5fa12a16fc82`.
The loader restores all fused ResNet-50 convolutions and the six-value
regression layer and prints the exact ONNX-node-to-module mapping. A supplied
official `.pth` is also accepted and checked strictly.

Before deployment-style validation or test, create schema-v1 external HEAD
caches at the paths in `configs/head_pose_geometry/base.yaml`. Each
`entries[image_id]` row contains:

```json
{
  "box_xyxy": [10, 20, 100, 150],
  "confidence": 0.9,
  "instance_id": 1,
  "valid_image_region_xyxy": [0, 0, 1280, 720],
  "source": "rtmdet_hf"
}
```

Val/test loading deliberately fails if these predicted caches are missing or
if a ground-truth source is supplied.

For the existing Co-DINO evaluator, first write COCO HEAD predictions, then
convert them without rerunning or coupling the geometry model to that detector:

```bash
.venv/bin/python scripts/eval_head_codino.py \
  --config configs/head_codino/production.yaml \
  --checkpoint /path/to/head_detector.pth --split val \
  --output outputs/head_pose_geometry/raw_heads/val.json

.venv/bin/python scripts/build_predicted_head_cache.py \
  --annotations outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/val.json \
  --predictions outputs/head_pose_geometry/raw_heads/val.json \
  --checkpoint /path/to/head_detector.pth \
  --detector-name head_codino --split val \
  --output outputs/head_pose_geometry/head_detections/val.json
```

The converter preserves original-image coordinates and records annotation,
prediction, and detector-checkpoint hashes. The same COCO-result contract can
be produced by the existing RTMDet-HF evaluator.
Detector misses do not remove FACE/keypoint ground truth: they disable only
the per-instance geometry supervision/fusion term. Deployment inference still
uses every predicted HEAD and never inserts a GT replacement.

## Stage 0 and pseudo labels

Build derived labels and verify source-video separation:

```bash
.venv/bin/python scripts/build_head_pose_geometry_labels.py \
  --dataset-root outputs/dataset_cache/face_pose_drive_v2_balanced_811 \
  --annotation-dir outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1 \
  --output-dir outputs/head_pose_geometry/derived_labels
```

Generate pseudo rotations only for train:

```bash
.venv/bin/python scripts/generate_pose_pseudo_labels.py \
  --split train \
  --annotation outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/train.json \
  --image-root outputs/dataset_cache/face_pose_drive_v2_balanced_811 \
  --checkpoint checkpoint/head_pose_geometry/sixdrepnet360_1x3x224x224_full.onnx \
  --output outputs/head_pose_geometry/pseudo_pose/train.json \
  --device cuda --seed 42
```

Acceptance uses confidence, uncertainty, crop-jitter consistency, and
horizontal-flip consistency. Optional temporal/second-teacher values can be
added when real track IDs or an independently licensed teacher are available.
Rejected real rows still train through 2D ellipse/keypoint, HEAD-relative,
visibility, and consistency objectives; only their hard geodesic pose mask is
off.

## Training stages

The existing trainer is reused:

```bash
.venv/bin/python tools/train_face_keypoint_cooperative.py \
  --config configs/head_pose_geometry/stage1_pose_only.yaml
```

Use, in order:

1. `stage1_pose_only.yaml`: frozen pretrained encoder plus trainable pose token.
2. `stage2_fixed_geometry.yaml`: fixed template, scale/translation,
   observability/uncertainty, and gated residual fusion.
3. `stage3_pose_domain_adaptation.yaml`: final ResNet stage plus train-only
   accepted pseudo rotations. Mix synthetic full-range batches when supplied
   by the experiment environment.
4. `stage4_shape_code.yaml`: four-value strongly bounded shape ablation.
5. `stage5_joint_finetune.yaml`: low-LR joint pass retaining all direct losses.

Start a later stage from the previous best checkpoint without relaxing any
historical direct-model tensor:

```bash
.venv/bin/python tools/train_face_keypoint_cooperative.py \
  --config configs/head_pose_geometry/stage2_fixed_geometry.yaml \
  --initialize-from outputs/experiments/head_pose_geometry/stage1_pose_only/checkpoints/best_val_loss.pt \
  --no-resume
```

Only new/resized `head_pose_geometry.*` tensors may be absent or incompatible;
any direct-model mismatch is fatal and is written to the load audit.

Full pose fine-tuning requires both `training_mode: full` and
`allow_full_finetune: true`; it is never enabled by the supplied configs.
A0–A8 configs are under `configs/head_pose_geometry/ablations/`.
For Stage 3+, set `data.synthetic.enabled: true` with a licensed COCO-like
HEAD export containing per-instance `rotation_matrix`. The deterministic mixed
dataset retains every real sample and adds the configured synthetic fraction.
Optional camera-frame `template_vertices`, weak-perspective scale/translation,
and shape-code targets are masked and supervised when present. The repository
contains no licensed synthetic export, so the supplied default is explicitly
off rather than pointing at invented data.

## Evaluation

Keep face AP/ellipse errors and five-keypoint PCK/visibility as the selection
metrics. Select thresholds on validation once and freeze them for test.
Stratify using `docs/head_pose_geometry_results_template.md`. Teacher-pose
agreement is diagnostic only.

Run focused verification before a long job:

```bash
.venv/bin/python -m pytest -q \
  tests/test_head_pose_geometry.py \
  tests/test_face_keypoint_cooperative.py
```

The geometry output contains separate visible/amodal confidence,
observability, pose uncertainty, and geometry uncertainty. Do not collapse
these into one deployment threshold.
