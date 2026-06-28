# DINO-SCRFD Face Detector Design Note

This addition is a Phase 2 scaffold for learning face bbox + five facial keypoints on top of frozen DINO patch features.

## Model

The model path is:

```text
image -> DINO patch tokens -> 2D feature map -> adapter conv -> SimpleFPN -> SCRFD-style head
```

The DINO wrapper is DINOv2-oriented and returns a `[B, C, H_patch, W_patch]` tensor by removing prefix tokens and reshaping patch tokens. It supports a local `weights` path and a dummy local backbone for unit tests. It does not download pretrained weights.

## Freeze Policy

The backbone is frozen by default because the first objective is to test whether DINO features can support a detector head without destabilizing a large ViT. Trainable parameters are limited to the adapter, FPN, and SCRFD-style head. `unfreeze_last_n_blocks` exists as a later-phase option, but should stay at `0` for the initial experiments.

## Adapter and FPN

DINO outputs a single patch-token feature map, not CNN P3/P4/P5 maps. The adapter converts DINO channels to detector channels with 1x1 and 3x3 convolutions. `SimpleFPN` then creates a small pyramid by upsampling, keeping the same scale, and downsampling. Decode and assignment derive cell centers from actual feature shape and image shape instead of hard-coding strides.

## Head and Loss

The first head is anchor-free SCRFD-style for maintainability. It predicts:

- face logits
- bbox distances as `[l, t, r, b]`
- five keypoints as x/y offsets from the assigned cell center

Losses are focal classification loss, SmoothL1 bbox loss, and SmoothL1 keypoint loss. Keypoint loss is masked by visibility and becomes zero when keypoints are missing.

## Dataset Notes

FACE ellipse annotations are converted to their rotation-aware outer bbox for the face bbox target. The original ellipse metadata is preserved in `target["metadata"]["ellipse_metadata"]` for future ellipse-aware extensions. HEAD boxes are retained as optional metadata and are not the primary target in this scaffold.

## Current Status

No training has been run as part of this implementation. The intended first command is a dry-run smoke check, then a short controlled training run only after local data and local DINO weights are explicitly configured.
