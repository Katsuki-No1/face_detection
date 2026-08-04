# Shared HEAD detector -> HEAD crop -> cooperative DINO

This architecture is a true two-pass cascade and is distinct from the
single-pass integrated Three-stage model.

1. The common DINOv3 ViT-S+/16 + compact Co-DINO model detects HEAD boxes on a
   960-square full image.
2. Each prediction is expanded to a square with side
   `1.20 * max(HEAD width, HEAD height)`. Out-of-image pixels are padded.
3. The square crop is resized to 960 and passed through the cooperative
   DINOv3/DEIMv2-L/landmark-query model.
4. Face, ellipse, keypoint, and state outputs are mapped back to original-image
   coordinates before common evaluation.

The upstream HEAD model is trained only once from
`configs/head_codino/shared_three_stage_960.yaml`. Its checkpoint has two
consumers:

- compatible SFP P3/P4/P5 and compact Co-DINO weights initialize every
  integrated Three-stage HEAD branch;
- locked score-0.30 predictions are cached once for train/validation/test and
  consumed by the HEAD-crop model.

This shares weights and predicted HEAD results. It does not share cached DINO
feature tensors across downstream training runs: the cooperative DINO
backbone is fine-tuned and training augmentation changes its inputs, so such a
cache would be stale and would silently change the optimization problem.

Production config:

```text
configs/face_keypoint_cooperative/head_crop_cooperative_detr_l_960.yaml
```

Gate-only config uses GT HEAD crops solely because the production HEAD
checkpoint cannot exist before the tests-first gate group finishes. It still
exercises the identical crop transform, cooperative model, losses, AMP, and
optimizer path. Production validation/test have no GT fallback.
