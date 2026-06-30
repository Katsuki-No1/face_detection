# DINO CenterFace Ellipse Detector

This detector keeps DINO as a feature extractor and adds a compact
CenterFace/FCOS-style anchor-free head for ellipse-first face detection.

## Model

```text
image
  -> DINO patch tokens
  -> [B, N, C] to [B, C, Hf, Wf]
  -> 1x1 + 3x3 CNN adapter
  -> SimpleFPN
  -> CenterFaceEllipseHead
```

Head outputs per FPN level:

- `center_logits`: `[B, 1, H, W]`
- `ellipse_reg`: `[B, 5, H, W]`
- `kpt_reg`: `[B, K * 2, H, W]`
- `bbox_reg`: optional `[B, 4, H, W]`

`ellipse_reg` is `[cx_offset, cy_offset, rx, ry, theta]`.
Center offsets are in feature-cell coordinates, `rx` and `ry` are image
pixels, and `theta` is radians. The head applies sigmoid to center offsets and
softplus to radii so decoded radii are positive.

## Target Assignment

Each ellipse is assigned to one center cell on the selected FPN level. A
Gaussian is drawn on `center_targets`; regression losses are computed only at
the center cell.

Annotation handling:

- CVAT ellipse metadata is converted to `[cx, cy, rx, ry, theta]`.
- YOLO/COCO bbox-only samples derive an axis-aligned ellipse from the bbox.
- Fully outside boxes are filtered by the dataset clip step.
- Ellipses whose center is outside the image are ignored by the assigner.
- Keypoint losses use only visible keypoints.

## Loss

```text
loss_total =
    center * center_heatmap_focal_loss
  + ellipse * smooth_l1(ellipse)
  + keypoint * smooth_l1(visible keypoints)
  + bbox * smooth_l1(optional bbox)
```

Default weights:

- center: `1.0`
- ellipse: `5.0`
- keypoint: `1.0`
- bbox: `0.5`

## Decode

Inference uses sigmoid heatmap scores, local maximum pooling, top-K filtering,
score thresholding, ellipse/keypoint/bbox decode, and bbox IoU NMS. The NMS
entry point is separated as `ellipse_aware_nms` so it can later be replaced by
ellipse IoU.
