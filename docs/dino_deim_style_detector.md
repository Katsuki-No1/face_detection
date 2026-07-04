# DINO-DEIM-Style Detector

This is an experimental DINO-first bbox detector inspired by DEIMv2 and
RT-DETR. It is not a complete DEIMv2 reproduction.

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

The purpose is to compare a stronger DINO-based detector against the existing
DINO CenterFace-style head while keeping DINO as the main feature extractor.
The first target is bbox detection for `head` and `face`. Keypoint refinement
can stay in the existing RTMPose or downstream keypoint pipeline.

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

This implementation is still not a full DEIMv2 reproduction. It does not include
RT-DETR top-k proposal initialization, deformable attention, denoising queries,
or the complete official DEIMv2 training recipe.

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
