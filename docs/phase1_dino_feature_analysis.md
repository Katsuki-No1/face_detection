# Phase 1 DINO Feature Analysis

This phase diagnoses frozen DINO features only.

It does not fine-tune DINO, does not train a detection head, does not train a
keypoint head, and does not run YOLO training. Precision / Recall / mAP are not
the target metrics for this phase.

## Default Config

```bash
configs/dino_phase1_feature_analysis.yaml
```

The default config reads the current YOLO Pose `modified_100` dataset and writes
all artifacts under:

```bash
outputs/phase1_dino_feature_analysis
```

## CLI

Extract frozen DINO features:

```bash
python -m src.phase1_dino.extract_features \
  --config configs/dino_phase1_feature_analysis.yaml \
  --max-images 100
```

Analyze extracted features:

```bash
python -m src.phase1_dino.analyze_features \
  --config configs/dino_phase1_feature_analysis.yaml
```

Build the HTML report:

```bash
python -m src.phase1_dino.make_report \
  --config configs/dino_phase1_feature_analysis.yaml
```

Run all Phase 1 steps:

```bash
python -m src.phase1_dino.run_phase1 \
  --config configs/dino_phase1_feature_analysis.yaml \
  --max-images 50
```

For a fast smoke test that does not load a real DINO model:

```bash
python -m src.phase1_dino.run_phase1 \
  --config configs/dino_phase1_feature_analysis.yaml \
  --max-images 3 \
  --mock-features \
  --output-dir outputs/phase1_dino_feature_analysis_smoke
```

The shell wrapper runs the same full pipeline with `MAX_IMAGES=50` by default:

```bash
bash scripts/run_phase1_dino_feature_analysis.sh
```

## Outputs

```text
outputs/phase1_dino_feature_analysis/
  crops/
    crop_metadata.csv
    *.jpg
  features/
    crop_features.npy
    crop_features_meta.csv
    patch_features_sample.npz
    feature_run_info.json
  metrics/
    knn_metrics.json
    similarity_metrics.csv
    retrieval_metrics.json
    patch_similarity_metrics.csv
  figures/
    similarity_histogram.png
    pca_by_crop_type.png
    pca_by_label.png
    pca_by_face_size.png
  retrieval/
    knn_retrieval.html
  overlays/
    similarity_map_samples/
  report.html
```

`crop_features.npy` has shape `[num_crops, feature_dim]`. With real DINOv2
`dinov2_vits14`, `feature_dim` is expected to be 384. With `--mock-features`,
the default feature dimension is 64.

## Interpretation

Phase 1 can show whether face-like crops and non-face crops are separated in a
frozen DINO feature space, whether hard negatives mix into the face cluster, and
whether patch-token similarity is stronger around GT face regions.

Phase 1 cannot report detector Precision / Recall / mAP. To measure those,
Phase 2 needs a frozen DINO backbone plus a detection head and evaluation logic.
