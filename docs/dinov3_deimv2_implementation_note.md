# DINOv3 Official-Style DEIMv2 Detector

This note describes the official-style head/face bbox detector exposed as
`dinov3_deimv2`.

## Scope

The implementation is an added detector path for bbox detection only:

```text
image
  -> DINOv3 ViT patch tokens
  -> [B, C, H/16, W/16] feature map
  -> official DEIMv2-style STA
  -> P3/P4/P5 multi-scale features
  -> official HybridEncoder
  -> official DEIMTransformer
  -> pred_logits [B, Q, 2] and pred_boxes [B, Q, 4]
```

`pred_boxes` are normalized `cx, cy, w, h` in `[0, 1]`. Postprocess converts
them to clipped `xyxy` boxes in image coordinates. `head` and `face` are COCO
category ids `1` and `2`; the criterion converts these 1-based ids to 0-based
class indices internally.

## Implementation

- `src/third_party/deimv2/` contains vendored upstream DEIMv2 detector
  components.
- `src/models/dinov3_deimv2/` connects the vendored official components to this
  repository's DINOv3 loader, datasets, training loop, and evaluator.
- `src/data/coco_detection.py` provides a COCO detection dataset entry point.
- `tools/train_dinov3_deimv2.py` reuses the existing config-driven training
  loop and adds `--data-root` / `--seed` compatibility.
- `tools/eval_dinov3_deimv2.py` writes `metrics.json`, `metrics.csv`,
  `threshold_metrics.csv`, `predictions.json`, visualization samples, and
  failure samples.
- `tools/infer_dinov3_deimv2.py` writes `predictions.json`, per-image counts,
  class counts, and colorized overlays.

This path is intended for official DEIMv2 evaluation. It uses the upstream
DEIMTransformer, DEIMCriterion, HungarianMatcher, PostProcessor, denoising
queries, aux losses, encoder aux losses, MAL, box/local losses, and deformable
attention. The older lightweight local implementation remains available as
`model.type: dino_deim_style`.

## Commands

Debug smoke training:

```bash
python tools/train_dinov3_deimv2.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_s_face_head_debug.yaml \
  --data-root dataset \
  --output-dir outputs/experiments/debug_dinov3_deimv2 \
  --amp \
  --seed 0
```

Full training:

```bash
python tools/train_dinov3_deimv2.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_s_face_head.yaml \
  --data-root dataset \
  --output-dir outputs/experiments/dinov3_deimv2_s_001 \
  --amp \
  --seed 0 \
  --wandb \
  --wandb-entity sato-waseda-university \
  --wandb-project face_detection \
  --dataset-version face_dataset_v7
```

W&Bにはepoch単位のtrain/validation曲線を記録し、job typeは`training`です。`--wandb-group`を省略した
場合、seedを除いた主要条件からgroupを自動生成するため、seed違いの反復実験をまとめて比較できます。

Evaluation:

```bash
python tools/eval_dinov3_deimv2.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_s_face_head.yaml \
  --checkpoint outputs/experiments/dinov3_deimv2_s_001/checkpoints/best_loss.pt \
  --data-root dataset \
  --split test \
  --output-dir outputs/experiments/dinov3_deimv2_eval
```

Inference:

```bash
python tools/infer_dinov3_deimv2.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_s_face_head.yaml \
  --checkpoint outputs/experiments/dinov3_deimv2_s_001/checkpoints/best_loss.pt \
  --input path/to/images \
  --output outputs/experiments/dinov3_deimv2_infer \
  --score-thr 0.3
```

## Data

Expected COCO layout:

```text
dataset/
  images/
    train/
    val/
    test/
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
```

For ellipse face annotations, convert to the outer bbox before writing COCO.
Clip boxes to image bounds and keep unclipped metadata in the converter output
when available.

## Risks

- The evaluator is a local COCO-like implementation, not pycocotools COCOeval.
  Use it consistently across YOLO11s, CenterFace-style, and this detector for
  fair local comparison.
- Backbone training is config-specific. The older M baseline freezes its
  distilled ViT-TinyPlus backbone. The production DEIMv2-L config at
  `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v1_960_full.yaml`
  fully fine-tunes the real DINOv3 ViT-S/16 backbone and uses EMA validation.
- Full training requires a CUDA machine and the configured local DINOv3 repo and
  weights. The debug config uses a dummy DINO backbone for CPU smoke tests.
