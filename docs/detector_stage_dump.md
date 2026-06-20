# Detector Stage Dump

Debug-only tools for inspecting SCRFD, standalone MediaPipe face detection, merge, and postprocess stages.

## Dump stages

```bash
MPLCONFIGDIR=/tmp/matplotlib python script/script_model/dump_detector_stages.py \
  --input ./data/sample_images \
  --output ./debug_outputs/sample_run \
  --score-min 0.01 \
  --save-overlays
```

Useful options:

- `--score-min`: lowest SCRFD score saved in `raw_scrfd.jsonl` before NMS.
- `--score-threshold`: postprocess score threshold after merge. Default: `0.5`.
- `--nms-threshold`: primary NMS IoU threshold. Default: `0.4`.
- `--min-box-size`: optional minimum `sqrt(width * height)` filter.
- `--post-nms-iou`: optional second-pass NMS after size filtering.
- `--max-detections`: optional per-image cap after sorting by score.
- `--recursive`: read image folders recursively.
- `--max-images`: limit images for smoke tests.

Output files:

- `raw_scrfd.jsonl`: SCRFD candidates before NMS, in original image coordinates.
- `raw_mediapipe.jsonl`: standalone MediaPipe face detection candidates, in original image coordinates, with normalized values in `raw_extra`.
- `merged_candidates.jsonl`: SCRFD and MediaPipe candidates after deterministic matching, before postprocessing.
- `postprocessed.jsonl`: final detections after score threshold, NMS, and optional filters.
- `removed_detections.jsonl`: detections removed during postprocessing, with reason and kept overlap when available.
- `summary.csv`: per-image counts and max scores.
- `skipped.csv`: unreadable images or per-image errors.
- `overlays/`: optional stage overlays when `--save-overlays` is enabled.

## Threshold sweep

```bash
python script/script_experiment/sweep_threshold_from_raw.py \
  --raw ./debug_outputs/sample_run/merged_candidates.jsonl \
  --output ./debug_outputs/sample_run/threshold_sweep.csv \
  --thresholds 0.01,0.05,0.10,0.15,0.20,0.30,0.50
```

The sweep uses saved raw merged candidates, so it does not rerun SCRFD or MediaPipe.
