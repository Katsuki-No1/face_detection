# DINO Patch FPN Experimental Design

## Purpose

The Phase 1 DINO checks showed that global crop embeddings can capture useful face/head information, but they are not a clean separator for labels such as occlusion, small, edge, and normal. This experiment keeps the spatial patch tokens instead of collapsing the image into one global embedding.

The intended path is:

```text
DINO patch tokens -> 2D token feature map -> CNN adapter -> SimpleFPN -> dense bbox/keypoint heads
```

No training, long inference, DINO weight download, or dataset-wide preprocessing was run for this implementation.

## Why Patch Tokens

DINO patch tokens preserve where the evidence is in the image. That is important for small faces, partial heads near image edges, occlusion, and landmarks such as eyes, nose, and mouth corners. A global embedding is useful for image-level retrieval and similarity checks, but it discards the grid structure needed by dense detection and keypoint heads.

The wrapper reshapes patch tokens from `(B, N, C)` into `(B, C, H, W)`. The grid size is inferred from the input image size and patch size. Prefix tokens such as CLS or register tokens can be removed before reshaping. If the remaining token count does not match the expected grid, the wrapper raises an explicit error.

## CNN Adapter And SimpleFPN

DINO channels are not arranged like a CNN backbone output, so the adapter uses a 1x1 projection followed by configurable 3x3 convolution blocks. This gives the detector head a local convolutional feature surface without changing the DINO backbone itself.

SimpleFPN creates same-channel multi-scale features from one DINO token map:

```text
base feature
├─ p2: upsample x2 + conv
├─ p3: conv
├─ p4: stride-2 conv
└─ p5: stride-2 conv from p4
```

For a `448x448` image and `patch_size=14`, the DINO token map is `32x32`, so the FPN shapes are `p2=64x64`, `p3=32x32`, `p4=16x16`, and `p5=8x8`.

## Head Design

The first head is an anchor-free dense head. For each FPN level it returns:

- `cls_logits`: class logits for `head` and `face`
- `bbox_reg`: 4 bbox regression values
- `centerness`: 1 objectness/centerness logit
- `keypoints`: 5 keypoints with `(x, y, visibility/logit)` values
- `ellipse`: optional 5-value regression `(cx, cy, rx, ry, theta)`

The main task is bbox plus 5 keypoints. Ellipse regression is kept optional because existing CVAT face annotations may originate as ellipses, but the current YOLO Pose training path already bbox-izes those ellipses.

## Dataset Assumptions

The existing YOLO Pose labels use class id `0` for face and keypoints in `e1, e2, n, m1, m2` order. The experimental model reserves class names `head, face` and keypoint names `left_eye, right_eye, nose, mouth_left, mouth_right` with `flip_idx=[1, 0, 2, 4, 3]`.

The added adapter can parse one YOLO Pose row into a target dict for lightweight tests. It does not perform dataset-wide conversion.

## Training Plan

Recommended next steps when moving beyond smoke tests:

1. Phase 1: freeze DINO and train only the CNN adapter, SimpleFPN, and dense heads.
2. Phase 2: unfreeze only the last few DINO blocks with a much smaller learning rate.
3. Phase 3: fine-tune on hard examples, especially small faces, image-edge cases, occlusion, and incomplete keypoints.

Small face detection needs care: with `patch_size=14`, very small faces may occupy only a few tokens before FPN upsampling. Higher input resolution, careful scale sampling, or a smaller-patch DINO variant may be needed.

Use full images for detection and pose localization. Face crops and head crops remain useful for analysis, hard-example mining, and auxiliary diagnostics, but a dense detector should learn from full-image spatial context first.
