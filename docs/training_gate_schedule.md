# Tests-first training schedule

The canonical pending-model queue is
`configs/training_queues/pending_models_tests_then_training.yaml`.
Legacy queue files remain only as execution history.

Every new production architecture has four mandatory gates:

1. `schema`: resolve and validate the production configuration without training.
2. `gpu_step`: run the production dtype and model/loss path through GPU forward,
   backward, and the architecture audit.
3. `overfit`: reduce the required losses on a tiny real-data subset.
4. `small_scale`: train for several epochs on a bounded real train/validation
   subset and require a finite metric with both a minimum relative improvement
   and a majority of improving epoch transitions.

The queue loader rejects an incomplete `production_train` job unless exactly
one enabled gate of each stage exists for the same `architecture`, all four
success markers are dependencies, and every gate is positioned before the
first production job.

Queue order is deliberately:

1. all gate jobs for all pending architectures;
2. all long production training jobs without evaluation gaps;
3. validation, test evaluation, and comparisons;
4. independent active-learning video inference and frame-extraction jobs.

The active-learning jobs use queue order rather than cross-job dependencies.
They therefore run sequentially after the scheduled model work, while a
post-processing failure in one dataset does not prevent the next dataset from
being attempted.

The current canonical queue contains 24 gates and six contiguous production
jobs. Two belong to the newly restored cascade path: one shared upstream
DINOv3+Co-DINO HEAD component and one
`HEAD detection -> square HEAD crop -> cooperative DINO` model. The shared
HEAD checkpoint initializes the Three-stage HEAD branches and generates the
HEAD caches once, avoiding separate HEAD-detector training for each consumer.

This front-loads interactive failure discovery and leaves unattended GPU time
for continuous production training. A gate failure never writes its success
marker, so no dependent production job can start. Failed disposable
small-scale work directories are renamed with a timestamp before a retry,
preserving the diagnostic history.

Use a read-only schedule check before starting the watcher:

```bash
.venv/bin/python scripts/run_training_queue_when_idle.py --dry-run
```

Do not start the canonical queue until all gate jobs can be observed during an
operator-attended window.
