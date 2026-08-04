# FaceKeypointCooperativeDETR

## Positioning

`FaceKeypointCooperativeDETR` is a performance-oriented full-frame model. It
preserves the trained DEIMv2-L face detector as an independently supervised
branch, adds a face-proposal-independent five-keypoint query branch, learns
face/pose association, and refines both directions with gated residual updates.
It is not a latency-optimized replacement for the baseline.

The implementation deliberately keeps three separately measurable outputs:

1. the original DEIM bbox/ellipse set prediction;
2. the independent full-frame pose-query prediction;
3. cooperative refined face and pose predictions.

This allows a failed cooperative experiment to be diagnosed without hiding a
regression in either independent branch.

## Data flow

```text
960 RGB
  -> DINOv3 ViT-S/16 + STA
  -> shared HybridEncoder (stride 8/16/32)
     |-> unchanged DEIMv2-L decoder -> face bbox/ellipse/query
     `-> independent encoder top-k pose proposals
         -> [pose instance + five semantic keypoint queries] x 4
         -> grouped self-attention
         -> multi-scale deformable attention
         -> iterative coordinates/confidence/visibility + local refinement
  -> learned appearance + normalized geometry association
  -> association-top-k bidirectional decoder x 3
     -> face attends associated pose/keypoint tokens
     -> pose/keypoint tokens attend associated face tokens
     -> confidence/visibility/association-conditioned sigmoid gates
     -> multi-scale deformable feature refresh
     -> bbox/ellipse/keypoint/confidence refinement
```

The association geometry includes normalized center error, face/pose size
agreement, keypoint spread, soft bbox/ellipse containment, ellipse/keypoint
orientation agreement, face and pose confidence, keypoint confidence, and
visibility. Association is supervised from the common GT instance assignment;
it is not an IoU threshold.

## Loss

The total objective is:

```text
original DEIM face loss
+ lambda_kp_independent * independent pose/keypoint/no-object/aux loss
+ lambda_association * association focal loss
+ lambda_face_refined * refined face bbox/GIoU/ellipse/no-object/aux loss
+ lambda_kp_refined * refined pose/keypoint/no-object/aux loss
+ lambda_visibility * independent and refined visibility loss
```

`v=0` is absent and is excluded from point losses. `v=1` has a coordinate
target and a negative visibility target. `v=2` has both coordinate and visible
targets. The fixed semantic order is left eye, right eye, nose tip, left mouth
corner, right mouth corner.

Train-time augmentation applies mosaic, mixup, affine rotation/translation/
scale, semantic horizontal flip, photometric perturbation, and synthetic
occlusion jointly to the image and all aligned targets. Rotated ellipses are
transformed through their covariance, left/right landmark identities are
permuted on flips, and synthetic occlusion changes visibility without erasing
the coordinate label. Mosaic is disabled for the configured final epochs.

## Initialization contract

The production training config starts directly from the public
DEIMv2-L/DINOv3 ViT-S/16 COCO-pretrained checkpoint. It does not initialize
from the face/ellipse-fine-tuned detector or the face-keypoint-fine-tuned
Query+refinement model. The compatible shared backbone, STA, encoder, and DEIM
decoder tensors are transferred from the public checkpoint; the one-class
classification layers, ellipse layers, independent pose branch, association,
and cooperative decoder are newly initialized.

The builder treats `pretrained`, `fine_tuned`, and `random` face initialization
as mutually exclusive and rejects a fine-tuned checkpoint in `pretrained`
mode. Pose initialization similarly rejects a checkpoint in `random` mode.
The module-level load audit saved by the training entry point records the
initialization kind, public checkpoint path and SHA-256, transfer coverage, and
the components that are random.

The strict fine-tuned face loader and explicit Query+refinement mapping remain
available for non-production ablations. A strict face load rejects any missing,
unexpected, or shape-mismatched tensor. Query+refinement transfer maps only
compatible semantic queries, grouped self-attention, FFN, normalization,
coordinate heads, visibility head, and local refinement stages.

## Commands

Architecture, checkpoint, one-minibatch gradient, numerical, empty-target,
branch-direction, VRAM, and inference-time audit:

```bash
.venv/bin/python tools/audit_face_keypoint_cooperative.py \
  --config configs/face_keypoint_cooperative/face_keypoint_cooperative_detr_l_960.yaml \
  --device cuda:0
```

Short real-data learnability check:

```bash
.venv/bin/python tools/train_face_keypoint_cooperative.py \
  --config configs/face_keypoint_cooperative/face_keypoint_cooperative_detr_l_960.yaml \
  --device cuda:0 \
  --overfit-smoke --max-samples 4 --overfit-steps 64
```

Full training (single GPU):

```bash
.venv/bin/python tools/train_face_keypoint_cooperative.py \
  --config configs/face_keypoint_cooperative/face_keypoint_cooperative_detr_l_960.yaml \
  --device cuda:0 --resume --wandb
```

The production config enables online W&B logging. The trainer derives a stable
run ID from the entity, project, experiment, dataset version, and seed, and
stores it in `wandb_run.json`. Resuming from `checkpoints/last.pt` reuses that
run. When W&B is enabled for an experiment that already has `metrics.csv`, any
completed epochs not yet recorded in the run are backfilled before training
continues. Use `--no-wandb` only for an intentional local-only run.

Before a fresh full training run, the trainer performs one complete throwaway
optimizer update with the real model and real training data. It tries
`batch 8 / accumulation 2`, then only on CUDA OOM falls back to `4 / 4`,
`2 / 8`, and finally `1 / 16`. Every candidate preserves an effective batch
size of 16. The probe includes backward, AdamW state creation, gradient
clipping, optimizer step, and EMA update. Its throwaway state is deleted and
the seed is reset before the actual model is constructed.
`batch_size_selection.json` records every OOM, the selected pair, and measured
peak allocated/reserved VRAM. A resumed checkpoint reuses its recorded pair so
the minibatch contract cannot silently change between epochs.

DDP:

```bash
torchrun --nproc-per-node=2 tools/train_face_keypoint_cooperative.py \
  --config configs/face_keypoint_cooperative/face_keypoint_cooperative_detr_l_960.yaml \
  --resume
```

Fixed test evaluation after selecting thresholds on validation:

```bash
.venv/bin/python tools/eval_face_keypoint_cooperative.py \
  --config configs/face_keypoint_cooperative/face_keypoint_cooperative_detr_l_960.yaml \
  --checkpoint outputs/experiments/face_keypoint_cooperative_detr_l_960/checkpoints/best_val_loss.pt \
  --split test --device cuda:0
```

Inference saves independent and refined JSONL plus separate overlays:

```bash
.venv/bin/python tools/infer_face_keypoint_cooperative.py \
  --config configs/face_keypoint_cooperative/face_keypoint_cooperative_detr_l_960.yaml \
  --checkpoint outputs/experiments/face_keypoint_cooperative_detr_l_960/checkpoints/best_val_loss.pt \
  --input path/to/images \
  --output-dir outputs/inference/face_keypoint_cooperative \
  --device cuda:0
```

## What is and is not established

The architecture and tests establish that both independent objectives remain
active, all new modules are differentiable, branch dropout is training-only,
and unilateral face/pose candidates are retained. The existing
Query+refinement result establishes that its local refinement design improved
the prior query baseline on the fixed test set.

They do not establish that the new cooperative model beats DEIMv2-L or the
Query+refinement baseline. That claim requires the full training and fixed
validation/test protocol above. Compare independent and refined outputs from
the same checkpoint; do not select thresholds or checkpoints on test.
