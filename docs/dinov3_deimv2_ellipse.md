# DEIMv2-L direct ellipse detector

## Why this is a separate model

The existing `dinov3_deimv2` model predicts normalized axis-aligned
`[cx, cy, width, height]` boxes. An enclosing axis-aligned box does not retain
the source ellipse angle, so the conversion cannot be inverted uniquely.

`dinov3_deimv2_ellipse` keeps the existing DEIMv2 box proposal, matching, FDR,
and loss path, and adds a direct ellipse head to the final matched decoder
queries. Existing configs, checkpoints, result dictionaries, and tools are not
changed.

The public ellipse convention is:

```text
[cx, cy, semi_major, semi_minor, theta_rad]
semi_major >= semi_minor
theta_rad in [-pi/2, pi/2), modulo pi
```

Internally the angle is regressed as `(cos(2 theta), sin(2 theta))`.

## Prepare the read-only default dataset

The source ZIP and existing datasets are never modified. The preparation
command verifies the archive and its 10,617 checksummed members, materializes
an exact cache under `outputs/dataset_cache`, and writes the DEIM ellipse
adapter under `outputs/dataset_adapters`.

```bash
.venv/bin/python scripts/prepare_default_face_pose_dataset.py
```

The fixed balanced split contains 4,237 / 530 / 528 images and
6,529 / 840 / 769 face ellipses.

## Pre-training checks

```bash
.venv/bin/python scripts/train_dino.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml \
  --dry-run --device cuda:0 --batch-size 1

.venv/bin/python scripts/train_dino.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml \
  --overfit-smoke --device cuda:0 --max-samples 16 --steps 64 \
  --lr 1.0e-3 --batch-size 1 --num-workers 0
```

The ellipse overfit smoke disables augmentation and denoising, freezes shared
DEIMv2 parameters, and verifies that the new ellipse head can reduce its loss
on the fixed 16-image subset.

## Full training

If the GPU is already occupied by another model-training group, start the
generic queue watcher once. It waits indefinitely for all detected training
parent processes to exit and for GPU 0 to remain below 10% utilization and
2,500 MiB used memory for four consecutive samples. It then runs every enabled
job in the YAML `jobs` list in order:

```bash
.venv/bin/python scripts/run_training_queue_when_idle.py \
  --queue-config configs/training_queues/after_current_models.yaml \
  --detach
```

Current state and logs are written to:

```text
outputs/training_queues/after_current_models/status.json
outputs/training_queues/after_current_models/queue.log
outputs/training_queues/after_current_models/watcher.log
```

The watcher uses a lock to prevent duplicate launches. Each job declares its
completion marker and optional resume checkpoint; completed jobs are skipped,
and interrupted jobs automatically receive their configured resume arguments.
Check what it would do without waiting or launching anything:

```bash
.venv/bin/python scripts/run_training_queue_when_idle.py \
  --queue-config configs/training_queues/after_current_models.yaml \
  --dry-run
```

For a direct start without waiting, add `--no-wait`.

To add another model later, append another item to
`configs/training_queues/after_current_models.yaml`:

```yaml
  - name: unique_model_name
    enabled: true
    command:
      - "{project_root}/.venv/bin/python"
      - "{project_root}/path/to/training_script.py"
      - --config
      - "{project_root}/path/to/model_config.yaml"
    completion_marker: outputs/path/to/model/summary.md
    resume:
      checkpoint: outputs/path/to/model/checkpoints/last.pt
      arguments: [--resume-from, "{checkpoint}"]
```

No Python modification is required when adding a job.

The underlying full-training command is:

```bash
.venv/bin/python tools/train_dinov3_deimv2.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml \
  --data-root outputs/dataset_cache/face_pose_current \
  --output-dir outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_960_ellipse \
  --device cuda:0
```

The model warm-starts all shared parameters from the v2 `best_val_map.pt` EMA.
Only the six ellipse-head tensors are newly initialized. Validation and early
stopping use ellipse mAP50:95; the original bbox metrics remain in the report.

## Evaluation and inference

```bash
.venv/bin/python tools/eval_dinov3_deimv2_ellipse.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml \
  --checkpoint path/to/best_val_ellipse_map.pt \
  --split test --output-dir outputs/eval/deimv2_ellipse_test --device cuda:0

.venv/bin/python tools/infer_dinov3_deimv2_ellipse.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml \
  --checkpoint path/to/best_val_ellipse_map.pt \
  --input path/to/images --output outputs/infer/deimv2_ellipse --device cuda:0

.venv/bin/python tools/infer_dinov3_deimv2_ellipse_video.py \
  --config configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml \
  --checkpoint path/to/best_val_ellipse_map.pt \
  --input input.mp4 --output-video output.mp4 --device cuda:0
```

Evaluation reports ellipse AP50:95/AP50/AP75, matched ellipse IoU, center NME,
axis errors, angle MAE, the bbox-derived ellipse baseline, and the unchanged
bbox metrics. A checkpoint is eligible for promotion only when its ellipse mAP
beats the bbox-derived ellipse baseline and its bbox mAP drop is at most `0.01`.
