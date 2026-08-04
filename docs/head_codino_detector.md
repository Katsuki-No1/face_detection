# DINOv3 ViT-S+/16 + compact Co-DINO Head detector

This package is the native-resolution, one-class Head detector described in
`for codex/face_dino_current_technical_report_20260726.html`. It ports the
Co-DINO query path to the repository's current MMCV 2.1.0 / MMDetection 3.3.0
environment; it does not require or support downgrading MMDetection.

## Architecture and invariants

The model is registered as `model.type: head_codino`.

| component | parameters | contract |
|---|---:|---|
| official `dinov3_vits16plus` | 28,697,472 | final `x_norm_patchtokens` only |
| SFP P3–P6 | 4,699,264 | bias-free convs, affine channel-first LN |
| compact Co-DINO | 5,132,436 | query path 4,541,844 + encoded-P6 branch 590,592 |
| total | 38,529,172 | exact |

`HeadImageBatch` accepts RGB CHW `uint8` or floating `[0,1]` tensors, applies
ImageNet normalization, and pads only the right and bottom to a multiple of 32.
Each image's individual padded tensor is sent through the official DINOv3 list
forward API. Thus, a larger image in the batch never inserts extra patch tokens
into a smaller image's ViT self-attention.

SFP is also evaluated per image. P3/P4/P5/P6 are then padded per level with a
boolean mask (`true` means padding). Only P3, P4, P5, in that order, enter the
detector. During inference, different per-image padded shapes are isolated at
the MS-Deformable Attention boundary while equal shapes remain batched. This
makes one image's raw detector output invariant to differently sized batch
partners. `encoded_p6` remains connected for checkpoint/DDP compatibility but
does not enter the decoder or detector predictions.

The query path is fixed to hidden size 256, 3 feature levels, one encoder layer,
three decoder layers, 8 heads, four sampling points per level, ReLU FFN 1024,
100 matching queries, two-stage mixed selection, and iterative box refinement.
CDN uses dynamic query groups with a configured budget of 50, label noise 0.5,
and box noise 0.4. Every group contains positive and negative queries. Matching
queries are masked from all DN queries.

Hungarian matching uses focal/L1/GIoU weights 2/5/2. Training uses Quality
Focal/L1/GIoU weights 1/5/2. Loss values are returned for the final decoder
layer, two auxiliary layers, encoder proposals, and every DN decoder layer.
`loss_total` is their explicit sum. Under AMP, model computation remains fp16
while loss inputs are cast to fp32 to avoid the MMDetection 3.3 QFL empty-GT
dtype defect.

## Public outputs

Raw inference returns:

- `pred_logits [B,100,1]`
- `pred_boxes [B,100,4]`, normalized `cxcywh` in each original valid image
- two `aux_outputs`
- `enc_outputs`

Training additionally returns all three layers in `dn_outputs`, `dn_meta`, and
the attention mask in the diagnostic `_dn_mask` key. Passing
`return_features=True` adds P3/P4/P5/P6, level masks, original sizes, individual
padded sizes, and the batch padded size.

Production postprocessing applies sigmoid, score threshold 0.30, descending
stable sort, maximum 100 detections, original-pixel clipping, and invalid-box
removal. It deliberately performs no NMS. Labels are internal class 0. The
separate `expand_square_roi(..., scale=1.20)` utility is for downstream crops;
it never alters detection or COCO evaluation boxes.

## Data

Both supplied configs read:

```text
datasets/face_pose_drive_v2/annotations/rtmdet_hf/{train,val,test}.json
```

Only COCO category id 1 is loaded and mapped to internal label 0. Category id 2
and attribute targets are excluded. Images with no category-1 GT are retained.
No training resize is applied.

## Commands

The production training path intentionally requires the official ViT-S+
weight. The audit command reports a missing weight as an explicit skip and
still performs a random-initialized structural parameter audit:

```bash
.venv/bin/python scripts/audit_head_codino.py \
  --config configs/head_codino/production.yaml
```

Single-GPU and DDP training:

```bash
.venv/bin/python scripts/train_head_codino.py \
  --config configs/head_codino/production.yaml

torchrun --standalone --nproc_per_node=2 scripts/train_head_codino.py \
  --config configs/head_codino/production.yaml
```

Resume restores the strict model state, optimizer, scheduler, GradScaler, EMA,
epoch, and global step:

```bash
.venv/bin/python scripts/train_head_codino.py \
  --config configs/head_codino/production.yaml \
  --resume runs/head_codino_vits16plus/last.pth
```

The trainer uses AMP, gradient scaling, gradient clipping, fused AdamW,
gradient accumulation, EMA, and separate backbone/SFP/head learning-rate
groups. A bounded real-data overfit audit is available:

```bash
.venv/bin/python scripts/train_head_codino.py \
  --config configs/head_codino/production.yaml \
  --overfit-images 4 --epochs 20 --max-steps 80
```

Evaluation uses bbox COCOeval with `catIds=[1]` and `maxDets=[1,10,100]`:

```bash
.venv/bin/python scripts/eval_head_codino.py \
  --config configs/head_codino/production.yaml \
  --checkpoint runs/head_codino_vits16plus/last.pth --split val

.venv/bin/python scripts/eval_head_codino.py \
  --config configs/head_codino/production.yaml \
  --checkpoint runs/head_codino_vits16plus/last.pth --split test
```

Inference, exact checkpoint audit, and benchmark:

```bash
.venv/bin/python scripts/infer_head_codino.py IMAGE.jpg \
  --config configs/head_codino/production.yaml \
  --checkpoint runs/head_codino_vits16plus/last.pth

.venv/bin/python scripts/audit_head_codino.py \
  --config configs/head_codino/production.yaml \
  --checkpoint runs/head_codino_vits16plus/last.pth \
  --output outputs/head_codino/audit.json

.venv/bin/python scripts/benchmark_head_codino.py \
  --config configs/head_codino/production.yaml \
  --checkpoint runs/head_codino_vits16plus/last.pth
```

Benchmark JSON records resolution, batch size, GPU, warmup/iteration counts,
mean/p50/p95 latency, images/s, peak allocated memory, and Head count.

## Checkpoint policy and reference assets

Full checkpoint auditing recognizes `model`, `state_dict`, or `ema`, plus
`module.` and `_orig_mod.` prefixes. It reports the SHA-256 digest, every key
rename, missing and unexpected keys, shape mismatches, and component parameter
counts. Exact loading proceeds only if every check passes and always calls
`load_state_dict(..., strict=True)`; there is no silent `strict=False` path.

The required DINOv3 ViT-S+/16 initialization was downloaded from the
terminal on 2026-07-27:

```text
Dino/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
SHA-256 4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea
```

The source is Meta's public `facebook/boxer` Hugging Face repository:

```text
https://huggingface.co/facebook/boxer/resolve/main/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

The strict constructor loaded it successfully. It remains git-ignored. The
historical `runs/face_dino_v3_best_20260726/model.pth` detector checkpoint is
still absent, so historical reference AP is not reported as reproduced. The
existing
`checkpoint/DINOv3/dinov3_vits16_face_pose_drive_v2_deimv2_backbone.pth` is a
standard ViT-S/16 checkpoint with `fc1/fc2` MLP keys; the loader explicitly
rejects it as a ViT-S+ substitute, whose SwiGLU blocks require `w1/w2/w3`.

`configs/head_codino/smoke.yaml` uses random ViT-S+ weights only for structural
tests. Its output must not be used to claim the historical AP (including 0.755)
or production accuracy.

## Verification performed on 2026-07-26

Environment: Python 3.12.13, PyTorch 2.12.1+cu126, MMCV 2.1.0,
MMEngine 0.10.7, MMDetection 3.3.0, and NVIDIA GeForce RTX 4090. The compiled
CUDA `MultiScaleDeformableAttention` preflight passed.

- `pytest -q tests/test_head_codino.py`: 12 passed.
- repository-wide `pytest -q`: 267 passed, 3 skipped.
- `python -m compileall -q src scripts tests`: passed.
- ruff, mypy, and pyright were not installed, so no lint/type result is
  claimed.
- only one GPU was present, so the `torchrun`/DDP code path was implemented but
  not exercised with multiple ranks.
- mixed native-resolution inference and AMP forward/backward with one positive
  and one empty-GT image passed with finite losses and gradients.
- a saved smoke checkpoint strict-loaded from EMA with zero missing,
  unexpected, or shape-mismatched keys; its SHA-256 was
  `9a08323ca7fb86cbdc21f2d900ef4be11245beccb8f17ea4bb8b57ff2df39d44`.
- a four-real-image random-backbone smoke overfit reached 60 successful
  optimizer steps. Across the final continuation, QFL fell 0.4861→0.0382, L1
  0.4719→0.0618, GIoU 0.7665→0.0876, total loss 11.6933→6.8252, and normalized
  predicted boxes moved by a mean absolute 0.1271.
- Val and Test COCOeval entry points each processed all 530 images. The smoke
  checkpoint emitted no score-0.30 detections and therefore AP was 0.0. This
  validates the path only and is not a model-quality measurement.

Full-model fp16 inference benchmark (batch 1, three warmups, ten timed
iterations) used the smoke checkpoint:

| input H×W | mean latency | images/s | peak allocated GPU memory |
|---|---:|---:|---:|
| 480×854 | 40.23 ms | 24.86 | 239,219,712 bytes |
| 720×1280 | 38.69 ms | 25.84 | 330,628,608 bytes |
| 1080×1920 | 65.56 ms | 15.25 | 531,891,200 bytes |

The timing includes preprocessing, the model, and production postprocessing.
The smoke model returned zero Heads at threshold 0.30. Short-run timings can
show host scheduling variance; rerun the production benchmark for release
numbers after installing the official weight and training a production
checkpoint.

The corresponding validation commands were:

```bash
.venv/bin/python scripts/train_head_codino.py \
  --config configs/head_codino/smoke.yaml --output-dir /tmp/head_codino_overfit20 \
  --overfit-images 4 --epochs 10 --max-steps 20
.venv/bin/python scripts/train_head_codino.py \
  --config configs/head_codino/smoke.yaml \
  --resume /tmp/head_codino_overfit20/last.pth \
  --output-dir /tmp/head_codino_overfit60 \
  --overfit-images 4 --epochs 20 --max-steps 60
.venv/bin/python scripts/eval_head_codino.py \
  --config configs/head_codino/smoke.yaml \
  --checkpoint /tmp/head_codino_overfit60/last.pth --split val \
  --output /tmp/head_codino_overfit60/val_predictions.json
.venv/bin/python scripts/eval_head_codino.py \
  --config configs/head_codino/smoke.yaml \
  --checkpoint /tmp/head_codino_overfit60/last.pth --split test \
  --output /tmp/head_codino_overfit60/test_predictions.json
.venv/bin/python scripts/benchmark_head_codino.py \
  --config configs/head_codino/production.yaml \
  --checkpoint /tmp/head_codino_overfit60/last.pth \
  --resolution 480x854 --resolution 720x1280 --resolution 1080x1920 \
  --warmup 3 --iterations 10 \
  --output /tmp/head_codino_overfit60/native_benchmark.json
```
