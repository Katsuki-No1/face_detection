# DINO-DEIM-Style Detector

This repository now has two separate DINO + DEIM-related detector paths:

- `model.type: dinov3_deimv2`: official-style DEIMv2 path. It vendors the
  upstream DEIMv2 decoder, matcher, criterion, postprocessor, denoising, and
  deformable-attention/local-loss components under `src/third_party/deimv2/`.
- `model.type: dino_deim_style`: legacy lightweight experimental path. This is
  inspired by DEIMv2 / RT-DETR, but is not a complete DEIMv2 reproduction.

Use `dinov3_deimv2` when evaluating whether DEIMv2 itself works on this data.
Use `dino_deim_style` only for continuity with older local ablations.

## Official-Style `dinov3_deimv2`

```text
image
  -> DINOv3/ViT backbone
  -> official DEIMv2-style STA
  -> official HybridEncoder
  -> official DEIMTransformer
  -> official DEIMCriterion / PostProcessor
```

The official-style path supports several explicitly configured backbone modes.
The pretrained DEIMv2-M mode uses the official distilled ViT-TinyPlus wrapper.
The production DEIMv2-L mode uses the real DINOv3 ViT-S/16 backbone and fully
fine-tunes it, matching the official L architecture rather than treating a
separate frozen feature extractor as part of the parameter count. Targets are
converted inside the model from pixel
`xyxy` boxes and COCO category ids to normalized `cxcywh` boxes and contiguous
0-based labels. For the default head/face config, `head=1 -> 0` and
`face=2 -> 1`.

The official-style path intentionally disables feature cache support because the
STA consumes raw images through its spatial prior branch. It also exposes
`set_epoch(epoch)` so the upstream matcher can switch matching cost according
to `matcher_change_epoch`.

Default configs:

```text
configs/dinov3_deimv2/dinov3_deimv2_s_face_head.yaml
configs/dinov3_deimv2/dinov3_deimv2_s_face_head_debug.yaml
configs/dinov3_deimv2/dinov3_deimv2_m_face_pretrained.yaml
configs/dinov3_deimv2/dinov3_deimv2_m_face_pose_drive_v1_960_aug.yaml
configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v1_960_full.yaml
```

DEIMv2-L asset preparation, parameter verification, preflight, and production
commands are documented in `docs/dinov3_deimv2_l_vits16.md`.

`dinov3_deimv2_m_face_pose_drive_v1_960_aug.yaml` は、動画単位で分割済みの3,090フレーム版COCOを
使用します。学習入口はモデル初期化前にtrain/val/testの元動画IDと実画像SHA-256を再監査し、
split間重複または変換後の画像変更を検出した場合は学習を開始しません。

```bash
.venv/bin/python tools/train_dinov3_deimv2.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_m_face_pose_drive_v1_960_aug.yaml \
  --output-dir outputs/experiments/dinov3_deimv2_m_face_pose_drive_v1_960_aug \
  --device cuda:0
```

設定と実データをforwardまで確認し、本学習を開始しない場合は `--dry-run` を追加します。監査成功時の
詳細はrun directoryの `data_leakage_audit.json` に保存されます。

`dinov3_deimv2_m_face_pretrained.yaml` is the preferred config when using the
official DEIMv2 DINOv3-M COCO checkpoint at
`checkpoint/DEIMv2/deimv2_dinov3_m_coco.pth`. The checkpoint architecture is
different from the local ViT-B/16 frozen-backbone config, so the weights are not
enough by themselves; the matching DEIMv2-M-style backbone/encoder/decoder
config is also required. COCO class heads are skipped and re-initialized for the
repository's face-only target (`category_ids: [2]`, `num_classes: 1`).

This face config intentionally differs from the COCO recipe in task-level
settings: it uses one face class instead of 80 COCO classes, freezes the DINO
backbone for the small local dataset, and keeps a higher 960px input size for
small face boxes. The model structure still matches the DEIMv2-M checkpoint
except for the re-initialized class heads.

Smoke command:

```bash
.venv/bin/python scripts/train_dino.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_s_face_head_debug.yaml \
  --dry-run \
  --synthetic \
  --device cpu
```

## Legacy `dino_deim_style`

The sections below describe the legacy lightweight detector. Existing E1-E5
results belong to this legacy path and should not be interpreted as official
DEIMv2 performance.

## Model

```text
image
  -> DINO / DINOv3 backbone
  -> SpatialTuningAdapter
  -> multi-scale P3/P4/P5 features
  -> HybridEncoder
  -> learned-query DETR-style decoder
  -> class logits + normalized cxcywh boxes
```

The initial adapter is intentionally lightweight:

- 1x1 projection from DINO channels to `hidden_dim`
- 3x3 refinement at the DINO patch resolution
- nearest upsample + 3x3 conv for P3
- stride-2 3x3 conv for P5

The default training configs use `encoder.type: identity`. A small
`lightweight_transformer` encoder exists for smoke tests and future ablations,
but full self-attention over 960px multi-scale tokens is expensive.

## Purpose

The legacy path was built to compare a stronger DINO-based detector against the
existing DINO CenterFace-style head while keeping DINO as the main feature
extractor. It remains useful as a local baseline, but it is no longer the path
to use for official DEIMv2 evaluation.

## E1-E5 Ablation Variants

The E1-E5 configs are intended to make each added idea measurable.

```text
E1: DINO + adapter + RT-DETR-style detector
    standard Hungarian matching
    standard cls / L1 / GIoU loss

E2: E1 + stronger augmentation
    controlled mosaic / mixup / geometry / photometric augmentation

E3: E2 + Dense O2O-style matching
    extra query-target positives per GT to increase positive samples

E4: E3 + MAL-style quality handling
    quality-aware target/weighting for low-quality matches

E5: E4 + STA/decoder refinement style toggles
    cross-scale adapter fusion, adapter refinement layers, and iterative bbox refinement
```

The E3/E4/E5 pieces are practical style implementations for this repository's
detector, not exact official DEIMv2 components. This distinction matters when
interpreting the result: an underperforming E5 does not automatically mean
DEIMv2 itself is ineffective.

## What This Is Not

The legacy implementation is not a full DEIMv2 reproduction. It does not include
RT-DETR top-k proposal initialization, deformable attention, denoising queries,
or the complete official DEIMv2 training recipe. The official-style
`dinov3_deimv2` path is the replacement for that purpose.

## Compared With Existing Models

- CenterFace-style: light, dense, ellipse/keypoint friendly, good for seeing
  whether DINO features are useful.
- DINO-DEIM-style: query-based bbox detector intended as a stronger DINO-first
  detector baseline.
- RTMDet: strong OpenMMLab baseline, but its standard recipe is CSPNeXt-based;
  using DINO as the main backbone requires an adapter/neck/head design.

## Configs

```text
configs/dino_deim_style_head.yaml
configs/dino_deim_style_face.yaml
configs/dino_deim_style_ablation/head/E1_baseline.yaml
configs/dino_deim_style_ablation/head/E2_stronger_aug.yaml
configs/dino_deim_style_ablation/head/E3_dense_o2o.yaml
configs/dino_deim_style_ablation/head/E4_mal.yaml
configs/dino_deim_style_ablation/head/E5_sta_decoder_refine.yaml
configs/dino_deim_style_ablation/face/E1_baseline.yaml
configs/dino_deim_style_ablation/face/E2_stronger_aug.yaml
configs/dino_deim_style_ablation/face/E3_dense_o2o.yaml
configs/dino_deim_style_ablation/face/E4_mal.yaml
configs/dino_deim_style_ablation/face/E5_sta_decoder_refine.yaml
```

The head config filters COCO annotations to category id `1`. The face config
uses head-crop face data and filters annotations to category id `2`.

## Smoke Commands

```bash
.venv/bin/python -m pytest tests/test_dino_deim_style_detector.py

.venv/bin/python scripts/train_dino.py \
  --config configs/dino_deim_style_ablation/face/E5_sta_decoder_refine.yaml \
  --dry-run \
  --synthetic \
  --tiny-model \
  --device cpu \
  --image-size 64 \
  --batch-size 2

.venv/bin/python scripts/run_dino_deim_style_ablation.py \
  --target face
```

## Training Command Examples

These commands are examples only; they start real training.

```bash
.venv/bin/python scripts/train_dino.py \
  --config configs/dino_deim_style_ablation/head/E1_baseline.yaml \
  --device cuda:0

.venv/bin/python scripts/train_dino.py \
  --config configs/dino_deim_style_ablation/face/E5_sta_decoder_refine.yaml \
  --device cuda:0

.venv/bin/python scripts/run_dino_deim_style_ablation.py \
  --target all \
  --execute \
  --allow-training \
  --device cuda:0
```

## Inference Output

`scripts/infer_dino_deim_style.py` writes records in this shape:

```json
{"image_id": "frame.jpg", "label": "face", "score": 0.91, "bbox_xyxy": [10.0, 20.0, 80.0, 100.0]}
```

Example:

```bash
.venv/bin/python scripts/infer_dino_deim_style.py \
  --config configs/dino_deim_style_face.yaml \
  --checkpoint runs/dino_deim_style/dino_deim_style_headcrop_face_v1/checkpoints/best_loss.pt \
  --source path/to/images \
  --output outputs_experiment/dino_deim_style/predictions.jsonl \
  --device cuda:0
```
