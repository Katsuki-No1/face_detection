# Head-pose geometry results template

Record dataset/export SHA-256, git commit, config, seed, checkpoint SHA-256,
HEAD detector/checkpoint, validation-selected thresholds, hardware, PyTorch,
CUDA, batch size, and whether results use EMA.

## Primary metrics

| run | face AP50 | ellipse center | axes | angle | PCK@.03 | PCK@.05 | visibility F1 | runtime ms | params M | VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 existing | | | | | | | | | | |
| A1 frozen pose | | | | | | | | | | |
| A2 observability | | | | | | | | | | |
| A3 geometry/no fusion | | | | | | | | | | |
| A4 gated fusion | | | | | | | | | | |
| A5 no valid/border | | | | | | | | | | |
| A6 no pseudo adaptation | | | | | | | | | | |
| A7 final pose stage | | | | | | | | | | |
| A8 shape code | | | | | | | | | | |

## Required strata

Repeat the primary real FACE/keypoint metrics for:

- visible-area ratio: `[0,.25), [.25,.5), [.5,.75), [.75,1]`;
- truncation direction: none, left, right, top, bottom, multiple;
- visible keypoints: `0,1–2,3–4,5`;
- HEAD size by validation-fixed small/medium/large area bins;
- direct ellipse uncertainty quartile;
- geometry uncertainty quartile;
- boundary and zero/low-keypoint intersection;
- validation-fixed manual extreme-profile and near-rear audit subsets.

Teacher-pose bins may be reported in a separate analysis table and must not be
used as the primary claim.

## Integrity checklist

- [ ] Same source-video split and test set for all rows.
- [ ] Pseudo labels contain train rows only.
- [ ] Val/test use predicted external HEADs, with detector/checkpoint recorded.
- [ ] Thresholds were fixed from validation before test.
- [ ] Direct independent metrics are reported beside refined metrics.
- [ ] Runtime includes HEAD crop + pose + geometry, and HEAD-detector runtime is
      separately stated.
- [ ] No run was selected from test performance.
- [ ] Failure examples include boundary, profile, rear/zero-keypoint, false
      HEAD, and unreliable-pose cases.

## Conclusions

State effect sizes and uncertainty/repeats. Remove the shape code unless A8
provides a repeatable primary-metric gain over A7. Do not claim improvement
from pose-teacher agreement alone.
