# 顔検出と顔ランドマーク検出のためのスクリプト集

このリポジトリには、顔と頭部の検出、顔ランドマークの抽出、YOLO Pose 用のデータセットへの変換、学習、推論、可視化、評価を行う Python スクリプトが含まれています。

このリポジトリでは、個人情報や実データを Git に含めない運用を前提にしています。モデルの重み、入力動画、抽出したフレーム、アノテーション、生成したデータセット、学習結果、推論結果はローカルに保存し、`.gitignore` によって Git の管理対象から除外します。

## このリポジトリでできること

- YOLO Pose を使って、顔の矩形と 5 点の keypoint を推論できます。keypoint の名前は `e1`、`e2`、`n`、`m1`、`m2` です。
- YOLO Pose の推論結果を JSONL 形式で出力できます。
- CVAT のアノテーション、または変換済みの ground truth JSON から、YOLO Pose 形式のデータセットを作成できます。
- YOLO Pose のラベルが学習可能な形式になっているかを確認できます。
- Ultralytics を使って YOLO Pose モデルを学習、または fine-tune できます。
- confidence threshold を探索し、YOLO Pose の bbox detection の F1 を評価できます。
- OpenMMLab の RTMDet と RTMPose を使い、顔検出と 5 点ランドマーク推定を二段構成で fine-tune できます。
- SCRFD による顔検出と、必要に応じた MediaPipe による目のランドマーク抽出を実行できます。
- ラベルの確認、YOLO Pose の推論結果、SCRFD と MediaPipe の処理段階、検出エラーの確認用に overlay 画像を生成できます。
- DINOv3 / DINOv2 を fine-tune せず frozen backbone として使い、顔・頭・背景・hard negative・patch similarity の特徴分離を Phase 1 として診断できます。

## このリポジトリではできないこと

- 学習済みモデルの重みは含まれていません。
- 入力動画、顔画像、抽出フレーム、CVAT から export したファイル、アノテーション、生成済みデータセットは含まれていません。
- 認証情報や非公開ストレージへのアクセス情報は含まれていません。
- 公開用の再現可能なベンチマーク用データセットは含まれていません。評価や学習には、ローカルに用意したデータとモデルの重みを使用してください。

## ディレクトリ構成

```text
scripts_YOLO/
  convert_annotations_to_yolo_pose.py  # CVAT/JSON から YOLO Pose データセットを作成する
  check_yolo_pose_dataset.py           # YOLO Pose のラベルを検査する
  audit_yolo_pose_dataset.py           # 学習前の Phase 0 dataset audit と class-balanced train list を生成する
  refresh_yolo_pose_dataset_audit.py   # data/output 配下の CVAT zip を再探索し、dataset/audit を更新する
  visualize_yolo_pose_labels.py        # ラベル確認用の overlay 画像を生成する
  train_yolo_pose.py                   # Ultralytics YOLO Pose の学習を実行する
  predict_yolo_pose.py                 # YOLO Pose の推論結果を画像または動画として可視化する
  export_yolo_pose_jsonl.py            # YOLO Pose の推論結果を JSONL で出力する
  evaluate_yolo_pose_f1.py             # bbox detection F1 と threshold を評価する
  evaluate_yolo_pose_metrics.py        # mAP、固定 threshold、keypoint 品質、失敗種別をまとめて評価する
  run_yolo_pose_dataset_ablation.py    # 旧データと追加データの YOLO Pose 比較を実行する
  run_yolo11s_improvement_phase123.py  # 失敗分析、難例抽出、改善用データセット作成を補助する
  run_yolo11s_aug_phase1.py            # YOLO11s Pose の train-time augmentation 比較を実行する
  visualize_yolo_pose_predictions.py   # YOLO Pose 推論結果の overlay を生成する
  export_yolo_pose_failure_details.py  # YOLO Pose の失敗詳細を CSV/画像で確認しやすく出力する
  run_yolo_pose_gpu_compare.py         # 複数の YOLO Pose モデルを GPU で比較する

script/
  script_data/                         # アノテーションとフレームに関する補助スクリプト
  script_model/                        # SCRFD/MediaPipe の推論とデバッグ用スクリプト
    run_dinov3.py                      # DINOv3 の画像特徴量を JSONL に出力する
  script_experiment/                   # 評価、可視化、確認用の補助スクリプト
    rtmdet_rtmpose_experiment.py       # RTMDet + RTMPose の変換、学習、推論、評価をまとめた CLI
  script_process/                      # 前処理と後処理の補助スクリプト

src/
  phase1_dino/                         # frozen DINO feature analysis 用の crop 生成、特徴抽出、数値解析、HTML report

configs/
  dino_phase1_feature_analysis.yaml    # Phase 1 DINO 特徴診断の default config

docs/
  detector_stage_dump.md               # SCRFD/MediaPipe の debug 出力に関する説明
  phase1_dino_feature_analysis.md      # DINO Phase 1 の目的、CLI、成果物、解釈のメモ

scripts/
  run_phase1_dino_feature_analysis.sh  # Phase 1 DINO 特徴診断の一括実行ラッパー
  analyze_dinov3_cpu_features.py       # DINOv3 image-level embedding の可視化と近傍解析
  visualize_dinov3_crop_patch_cpu.py   # 実 DINOv3 CPU で crop gallery と patch similarity map を生成する
  profile_training_speed.py            # GPU 空き待ち、1 epoch 学習、nvidia-smi 計測をまとめて実行する

tests/
  test_dino_phase1_*.py                # DINO Phase 1 の crop/metrics/report の軽量 unittest

requirements.txt                       # Python の依存パッケージ
```

## セットアップ

Python 3.12、または使用する環境で互換性がある Python を用意してください。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

GPU で学習や推論を行う場合は、使用している NVIDIA driver と互換性がある PyTorch をインストールしてください。GPU が使用できるかどうかは、次のコマンドで確認できます。

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## モデルの重みを配置する方法

モデルの重みは Git に含めないでください。ローカルでは、次のような場所に置く運用を想定しています。

```text
models/scrfd/<scrfd_model>.onnx
weights/<face_pose_model>.pt
script/script_model/DINO/<dinov3_checkpoint>.pth
```

`models/`、`weights/`、DINO checkpoint、実験出力は Git に含めない運用です。実行時には、`--model` または `--weights` にローカルのパスを指定してください。

## DINO の導入と現在の段階

このリポジトリでは、DINO を **fine-tune せず frozen backbone** として使うための入口を追加しています。目的は、DINO 単体の特徴量が顔検出や keypoint 検出の前段として使えそうかを、学習前に確認することです。

現時点の位置づけは **Phase 1: DINO 特徴量診断** です。以下はまだ行っていません。

- DINO backbone の fine-tune
- detection head の学習
- keypoint head の学習
- YOLO 学習
- face detection の Precision / Recall / mAP 評価

そのため、DINO 関連の現在の出力は「検出性能」ではなく、特徴空間の分離傾向を見るための診断結果です。実際の Precision / Recall / mAP を評価するには、次の Phase 2 で frozen DINO backbone に detection head / keypoint head を接続して評価する必要があります。

### DINOv3 の画像特徴量抽出

`script/script_model/run_dinov3.py` は、DINOv3 公式 repo の PyTorch Hub entrypoint を使って画像ごとの CLS embedding を JSONL に出力します。ローカルに DINOv3 公式 repo の cache または clone がある場合は、それを `--dinov3-repo` に指定できます。

GPU を使わず CPU で確認する場合は、GPU 学習に干渉しないように `CUDA_VISIBLE_DEVICES=` と `--device cpu` を明示してください。

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python script/script_model/run_dinov3.py \
  --source path/to/images \
  --output outputs/dinov3_cpu_features.jsonl \
  --dinov3-repo /path/to/facebookresearch_dinov3 \
  --weights script/script_model/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --model-name dinov3_vitb16 \
  --device cpu \
  --batch-size 2 \
  --resize-size 256 \
  --include-patch-mean
```

30 枚程度の smoke test であれば CPU でも完了可能です。ただし、DINOv3 ViT-B/16 は CPU では軽くないため、全データへ広げる場合は実行時間に注意してください。

### Phase 1 DINO 特徴診断

`src/phase1_dino/` は、アノテーションから `face`、`head`、`eye_region`、`background`、`hard_negative` crop を作成し、DINO の frozen feature を抽出して数値解析と HTML report を出力します。

default config は次にあります。

```text
configs/dino_phase1_feature_analysis.yaml
```

一括実行は次のコマンドです。

```bash
.venv/bin/python -m src.phase1_dino.run_phase1 \
  --config configs/dino_phase1_feature_analysis.yaml \
  --max-images 50
```

実 DINO を使わずに pipeline だけ確認する場合は、mock feature で軽く実行できます。

```bash
.venv/bin/python -m src.phase1_dino.run_phase1 \
  --config configs/dino_phase1_feature_analysis.yaml \
  --max-images 3 \
  --mock-features \
  --output-dir outputs/phase1_dino_feature_analysis_smoke
```

主な成果物は次のとおりです。

```text
outputs/phase1_dino_feature_analysis/
  crops/
  features/crop_features.npy
  features/crop_features_meta.csv
  features/patch_features_sample.npz
  metrics/knn_metrics.json
  metrics/similarity_metrics.csv
  metrics/patch_similarity_metrics.csv
  figures/pca_by_crop_type.png
  figures/pca_by_label.png
  figures/pca_by_face_size.png
  retrieval/knn_retrieval.html
  overlays/similarity_map_samples/
  report.html
```

`hard_negative` は現時点では、GT face bbox とほぼ重ならない、顔サイズに近い random background crop として生成しています。より顔らしい hard negative 候補の抽出は、Phase 2 以降の改善候補です。

### 実 DINOv3 CPU での目視確認

`scripts/visualize_dinov3_crop_patch_cpu.py` は、manifest で指定したサンプルに対して実 DINOv3 を CPU で実行し、次の目視確認用ファイルを出力します。

- `face crop`
- `head crop`
- `background crop`
- `hard negative crop`
- face crop prototype に対する `patch similarity map`

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/visualize_dinov3_crop_patch_cpu.py \
  --manifest outputs/dinov3_cpu_stratified_30_20260625/sample_manifest.csv \
  --output-dir outputs/dinov3_cpu_stratified_30_20260625/crop_patch_analysis \
  --metadata outputs_experiment/yolo_pose_dataset_ablation_20260625_all_zips/datasets/_modified_all_train/metadata/conversion_metadata.json \
  --dinov3-repo /home/katsuki/.cache/torch/hub/facebookresearch_dinov3_main \
  --weights script/script_model/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

この可視化では、`crop_patch_gallery.html` を開いて、crop が意図通りか、patch similarity の高反応領域が GT face bbox 周辺に出ているかを目視確認します。数値としては `patch_similarity_metrics.csv` に bbox 内外の similarity 平均と margin を保存します。

### DINO detector 学習高速化の Priority B

`scripts/train_dino.py` は、DINO-SCRFD 系、DINO-CenterFace ellipse 系、DINO-DEIM-style 系の学習を同じ入口で扱います。学習は GPU 前提です。`--device cpu` を指定した full training は拒否されます。

Priority A として、AMP/TF32、DataLoader の `num_workers`、`persistent_workers`、`prefetch_factor`、batch size 調整、validation 頻度見直しを先に確認します。Priority B は条件付きで次を使います。

- `fused_optimizer`: CUDA 上の AdamW で `fused=True` を試します。失敗した場合は通常 AdamW に fallback し、`run_meta.json` に理由を残します。
- `torch.compile`: DINO backbone ではなく、学習対象の `adapter`、`fpn`、`head` だけを compile します。shape が安定し、複数 epoch 回す実験で効果を見ます。
- `feature_cache`: frozen DINO backbone の出力を `.pt` として保存し、以降は backbone forward を省略します。train-time augmentation、progressive resizing、backbone fine-tune とは同時に使いません。
- `progressive_resizing`: 序盤を小さい画像サイズで回し、warmup 中に最終サイズへ近づけます。feature cache とは同時に使いません。

### DINO-DEIM-style bbox detector

CenterFace-style head より強い DINO 主体の bbox detector を比較するため、
DEIMv2 / RT-DETR に inspired された実験用 detector を追加しています。完全な
DEIMv2 再現ではありません。

この実装の目的は、DINOv3 を特徴抽出 backbone として使った場合に、
`head` / `face` の bbox 検出性能を stage ごとに改善できるかを検証することです。
特に HFDP の流れである `image -> head detection -> head crop -> face detection -> face crop -> keypoint detection`
のうち、今回の DINO-DEIM-style は `head detection` と `face detection` の bbox detector を担当します。
keypoint 検出はこのモデルには含めず、既存の RTMPose などの下流 keypoint detector が担当します。

現在の結果は検証中です。E1〜E5 の smoke test では forward / loss 計算と
Dense O2O-style positive 増加が動作することを確認していますが、実データでの bbox 精度、
HFDP/GTHFDP に組み込んだ keypoint 指標、RTMDet 系 baseline との優劣はまだ確定していません。
そのため、この節の内容は「実装済みの実験条件」と「これから比較する仮説」を示すものであり、
性能改善を保証する結果ではありません。

```text
DINO backbone
  -> Spatial Tuning Adapter風 multi-scale adapter
  -> lightweight hybrid encoder
  -> DETR-style decoder
  -> bbox prediction
```

設定ファイルは次です。

```text
configs/dino_deim_style_head.yaml
configs/dino_deim_style_face.yaml
configs/dino_deim_style_ablation/head/E1_baseline.yaml
configs/dino_deim_style_ablation/head/E2_stronger_aug.yaml
configs/dino_deim_style_ablation/head/E3_dense_o2o.yaml
configs/dino_deim_style_ablation/head/E4_mal.yaml
configs/dino_deim_style_ablation/head/E5_sta_decoder_refine.yaml
configs/dino_deim_style_ablation/face/E1_baseline.yaml
configs/dino_deim_style_ablation/face/E2_stronger_aug.yaml
configs/dino_deim_style_ablation/face/E3_dense_o2o.yaml
configs/dino_deim_style_ablation/face/E4_mal.yaml
configs/dino_deim_style_ablation/face/E5_sta_decoder_refine.yaml
```

E1〜E5 は次の段階比較です。

```text
E1: DINO + adapter + RT-DETR-style detector
E2: E1 + stronger augmentation
E3: E2 + Dense O2O-style extra positives
E4: E3 + MAL-style quality-aware matching loss
E5: E4 + STA/decoder refinement style toggles
```

一括実験のコマンド表は次で生成します。デフォルトでは実学習は走らず、manifest だけを書きます。

```bash
.venv/bin/python scripts/run_dino_deim_style_ablation.py --target face

.venv/bin/python scripts/run_dino_deim_style_ablation.py \
  --target all \
  --execute \
  --allow-training \
  --device cuda:0
```

詳細は `docs/dino_deim_style_detector.md` を参照してください。

config では次の項目で管理します。

```yaml
train:
  fused_optimizer: true
  torch_compile:
    enabled: false
    mode: reduce-overhead
    fullgraph: false
    dynamic: false
  feature_cache:
    enabled: false
    dtype: float16
    reuse_existing: true
  progressive_resizing:
    enabled: false
    start_size: 672
    end_size: 896
    warmup_epochs: 20
```

GPU が空いたタイミングで 1 epoch だけ計測する場合は、次の profiler を使います。`--wait-for-free-gpu` は、GPU util と VRAM 使用量が連続して閾値以下になるまで待ってから学習を開始します。

```bash
.venv/bin/python scripts/profile_training_speed.py \
  --config configs/dino_centerface_ellipse_aug_ablation/E0_baseline.yaml \
  --name centerface_e0_b8_priority_b_fused_wait_20260630_0905 \
  --batch-size 8 \
  --epochs 1 \
  --device cuda:0 \
  --wait-for-free-gpu \
  --free-gpu-util 10 \
  --free-memory-used-mib 3000 \
  --free-samples 5 \
  --fused-optimizer \
  --no-torch-compile \
  --no-feature-cache \
  --no-progressive-resizing
```

出力は `outputs/gpu_profiles/<name>/` に保存されます。主に `gpu_profile_summary.json`、`gpu_samples.csv`、`train.log`、`train/run_meta.json`、`train/metrics.csv` を確認します。

### DINO augmentation ablation の blur 方針

DINO-SCRFD 系と DINO-CenterFace ellipse 系では、blur を他の photometric augmentation から分けて評価します。学習時の標準方針は **blur-only を medium 固定** とし、`E9_blur_medium` だけを blur 専用条件として使います。

- `E3_photometric_only`、`E5_mosaic_photometric`、`E6_geometry_photometric`、`E7_all` は HSV、brightness、contrast、gamma、noise を見る条件で、`blur: 0.0` にしています。
- `E9_blur_medium` は `blur: 0.20`、`blur_radius_min: 0.3`、`blur_radius_max: 1.5` の blur-only 条件です。
- `E8_blur_weak` と `E10_blur_strong` は archive 扱いで、`experiment.active: false` にしています。通常の manifest、preview、collect では対象から外れます。

学習前には、使用する config と manifest で blur 方針を確認してください。

```bash
.venv/bin/python scripts/run_dino_centerface_aug_ablation.py \
  --config-dir configs/dino_centerface_ellipse_aug_ablation \
  --manifest outputs_experiment/dino_centerface_ellipse_aug_ablation_manifest/commands.json
```

生成された manifest に `E9_blur_medium` が含まれ、`E8_blur_weak` と `E10_blur_strong` が含まれないことを確認します。学習後は各 run の `config_resolved.yaml` でも、photometric 系の `blur: 0.0` と `E9_blur_medium` の medium 設定を確認できます。

2026-07-01 時点の DINO-CenterFace ellipse `E9_blur_medium` 評価は `outputs_experiment/dino_centerface_ellipse_blur_ablation_real_dinov3_20260701_eval/E9_blur_medium/summary.json` に保存しています。test split 30 images / 63 instances で、best threshold `0.31`、precision `0.8600`、recall `0.6825`、F1 `0.7611`、bbox mAP `0.3159`、AP50 `0.6917` でした。既存の古い photometric checkpoint は blur `0.05` を含む設定で学習されているため、新方針で厳密に比較する場合は photometric no-blur 条件を再学習します。

2026-06-30 の batch size 8 計測では、既存GPUジョブが終了してから計測を開始するため、profiler の空き待ち条件を使いました。比較時は GPU util よりも、`epoch_seconds`、`images_per_second`、最大VRAM、loss が finite であることを優先して読みます。

実測結果は次のとおりです。

| profile | status | batch | epoch seconds | images/sec | active GPU util mean / max | active VRAM max | optimizer | loss finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `outputs/gpu_profiles/centerface_e0_b8_priority_b_fused_wait_20260630_0905` | 0 | 8 | 62.447 | 23.38 | 90.24% / 93% | 5280 MiB | AdamW fused | yes |

この計測では `torch.compile`、`feature_cache`、`progressive_resizing` はOFFにし、常用しやすい `fused_optimizer` だけをONにしています。`run_meta.json` では `optimizer.fused=true`、`amp=true`、`tf32=true`、`num_workers=4`、`persistent_workers=true`、`prefetch_factor=4` を確認できます。

## YOLO Pose の推論

### 画像または画像ディレクトリを可視化する

`scripts_YOLO/predict_yolo_pose.py` は、`--source` に画像ファイル、画像ディレクトリ、または動画ファイルを指定できます。可視化された推論結果は `--out-dir` に保存されます。

```bash
.venv/bin/python scripts_YOLO/predict_yolo_pose.py \
  --weights weights/<face_pose_model>.pt \
  --source input/images \
  --imgsz 960 \
  --conf 0.25 \
  --device 0 \
  --out-dir output/yolo_pose_predictions
```

CPU で実行する場合は、`--device cpu` を指定するか、`--device` を省略してください。

### 動画を可視化する

```bash
.venv/bin/python scripts_YOLO/predict_yolo_pose.py \
  --weights weights/<face_pose_model>.pt \
  --source input/videos/<video>.mp4 \
  --imgsz 960 \
  --conf 0.25 \
  --device 0 \
  --out-dir output/yolo_pose_video_predictions
```

### 推論結果を JSONL で出力する

```bash
.venv/bin/python scripts_YOLO/export_yolo_pose_jsonl.py \
  --weights weights/<face_pose_model>.pt \
  --source input/images \
  --output output/yolo_pose_predictions.jsonl \
  --imgsz 960 \
  --conf 0.25 \
  --device 0
```

JSONL の各行は、次のような構造になります。

```json
{
  "source": "input/images",
  "video": null,
  "image": "path/to/image.jpg",
  "frame_index": 0,
  "time_sec": null,
  "objects": [
    {
      "class_name": "face",
      "bbox": [0.0, 0.0, 100.0, 100.0],
      "score": 0.99,
      "keypoints": {
        "e1": [0.0, 0.0, 2],
        "e2": [0.0, 0.0, 2],
        "n": [0.0, 0.0, 2],
        "m1": [0.0, 0.0, 2],
        "m2": [0.0, 0.0, 2]
      }
    }
  ]
}
```

`bbox` と keypoint の座標は、pixel 単位の座標です。keypoint の visibility は、点が存在する場合は `2`、欠損している場合は `0` です。

## YOLO Pose 用データセットの作成と学習

CVAT から export したファイル、または変換済みの ground truth JSON から、YOLO Pose 形式のデータセットを作成できます。

```bash
.venv/bin/python scripts_YOLO/convert_annotations_to_yolo_pose.py \
  --annotation-zip annotations/<cvat_export>.zip \
  --image-root input/images \
  --output-dir datasets/face_pose \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42 \
  --clean
```

変換スクリプトは、次のファイルとディレクトリを出力します。

```text
datasets/face_pose/images/{train,val,test}/
datasets/face_pose/labels/{train,val,test}/
datasets/face_pose/metadata/conversion_metadata.json
datasets/face_pose/face_pose.yaml
```

変換時の主な方針は次のとおりです。

- Face の楕円は、rotation を反映した bbox に変換されます。その後、学習用の bbox は画像内に clip されます。
- Face の楕円がない画像は、顔がない画像、つまり negative sample として残されます。この場合は空の label ファイルが作成されます。
- keypoint は `e1`、`e2`、`n`、`m1`、`m2` の 5 点です。
- 画像内にある keypoint は visibility `2` になります。欠損している keypoint、または画像外にある keypoint は visibility `0` になります。

作成したデータセットは、学習前に確認してください。

```bash
.venv/bin/python scripts_YOLO/check_yolo_pose_dataset.py \
  --data-root datasets/face_pose \
  --yaml datasets/face_pose/face_pose.yaml \
  --output output/pretrain_dataset_check.json
```

ラベルを目視で確認する場合は、overlay 画像を生成してください。

```bash
.venv/bin/python scripts_YOLO/visualize_yolo_pose_labels.py \
  --data datasets/face_pose/face_pose.yaml \
  --metadata datasets/face_pose/metadata/conversion_metadata.json \
  --split all \
  --samples 50 \
  --out-dir output/label_check
```

YOLO Pose モデルを学習する場合は、次のように実行します。

```bash
.venv/bin/python scripts_YOLO/train_yolo_pose.py \
  --data datasets/face_pose/face_pose.yaml \
  --model yolo11s-pose.pt \
  --imgsz 960 \
  --epochs 200 \
  --batch 32 \
  --patience 40 \
  --project runs/face_pose \
  --name face_pose_baseline \
  --device 0
```

複数の YOLO Pose モデルを同じ条件で比較する場合は、次のスクリプトを使用できます。

```bash
.venv/bin/python scripts_YOLO/run_yolo_pose_gpu_compare.py \
  --data datasets/face_pose/face_pose.yaml \
  --data-root datasets/face_pose \
  --project runs/face_pose \
  --output-dir output/yolo_pose_compare \
  --imgsz 960 \
  --epochs 200 \
  --batch 32 \
  --fallback-batch 16 \
  --patience 40 \
  --device 0
```

### 追加アノテーションを継続的に学習対象へ入れる

`scripts_YOLO/run_yolo_pose_dataset_ablation.py` は、`--modified-zip` と `--modified-zip-manifest` を指定しない場合、`data/output` 配下を探索し、画像ファイルが直接入っているディレクトリ内の CVAT annotation zip を現在の追加学習データとして読み込みます。zip は中に `annotations.xml` を含むものだけが対象です。

新しいデータを追加する場合は、frame 画像が入っている `data/output/<dataset_name>/` に CVAT annotation zip を置いてから再実行してください。コードや manifest の書き換えは不要です。

```text
data/output/<dataset_name>/
  frame_00001.jpg
  frame_00002.jpg
  ...
  task_annotation.zip  # contains annotations.xml
```

一時的に別の zip 一覧で試したい場合は、従来通り `--modified-zip` を複数回指定すると自動探索より優先されます。探索先を変える場合は `--modified-zip-root <path>`、固定一覧を使う場合は `--modified-zip-manifest <path>` を指定します。

## YOLO Pose の評価

`scripts_YOLO/evaluate_yolo_pose_f1.py` では、`train`、`val`、`test` のいずれかの split に対して confidence threshold を探索できます。

```bash
.venv/bin/python scripts_YOLO/evaluate_yolo_pose_f1.py \
  --weights runs/face_pose/face_pose_baseline/weights/best.pt \
  --data-root datasets/face_pose \
  --split val \
  --imgsz 960 \
  --device 0 \
  --iou-threshold 0.5 \
  --prediction-conf-min 0.001 \
  --thresholds 0.01:0.95:0.01 \
  --output-dir output/yolo_pose_eval \
  --write-predictions
```

主な出力は次のとおりです。

- `<split>_threshold_sweep.csv`: threshold、IoU threshold、TP、FP、FN、precision、recall、F1 が記録されます。
- `<split>_best_threshold.json`: 最も F1 が高い threshold の行が保存されます。
- `<split>_raw_predictions.jsonl`: `--write-predictions` を指定した場合に、推論結果が保存されます。

### 現在の baseline 結果

現在の内部 test split に対する暫定的な評価結果です。この比較は最終的なベンチマーク性能を示すものではなく、今後の失敗分析で使う baseline model を選ぶためのものです。各手法では、val split で F1 が最も高くなった confidence threshold を選び、その値を test split に適用しています。評価は IoU threshold `0.5` の face bbox detection F1 で行っています。

| method | val threshold | test F1 | precision | recall | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s Pose | 0.51 | 0.8073 | 0.9565 | 0.6984 | 44 | 2 | 19 |
| YOLO11n Pose | 0.55 | 0.7778 | 0.9333 | 0.6667 | 42 | 3 | 21 |
| YOLO11m Pose | 0.52 | 0.7593 | 0.9111 | 0.6508 | 41 | 4 | 22 |
| SCRFD + MediaPipe | 0.09 | 0.5938 | 0.5846 | 0.6032 | 38 | 27 | 25 |

現時点では YOLO11s Pose を暫定 baseline とします。F1、precision、recall のバランスが最もよく、model size も YOLO11m Pose より扱いやすいためです。ただし、YOLO11s Pose と YOLO11m Pose では失敗するケースが一部異なるため、model ごとの失敗理由の確認は継続します。SCRFD + MediaPipe は比較用 baseline として残しますが、現在の test split では false positive が多く出ています。

### YOLO11s のデータ追加と高解像度比較

`outputs_experiment/yolo_pose_dataset_ablation/` では、旧データ、追加データ、今回の高解像度 YOLO11s を同じ固定 val/test split で比較しています。test split は 30 画像、63 件の face アノテーションです。各行では val split で F1 が最も高い confidence threshold を選び、その threshold を test split に適用しています。評価は IoU threshold `0.5` です。

| model | train data | imgsz / batch | epochs | val threshold | test box mAP50-95 | test pose mAP50-95 | test F1 | precision | recall | TP / FP / FN | keypoint NME | PCK@0.05 | PCK@0.10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s old repro | old | 960 / 32 | 200 | 0.51 | 0.4381 | 0.7476 | 0.8073 | 0.9565 | 0.6984 | 44 / 2 / 19 | 0.0554 | 0.6220 | 0.8804 |
| YOLO11s modified_50 | old + 50追加 | 960 / 32 | 200 | 0.53 | 0.4737 | 0.7455 | 0.8148 | 0.9778 | 0.6984 | 44 / 1 / 19 | 0.0585 | 0.5545 | 0.8389 |
| YOLO11s modified_100 | old + 100追加 | 960 / 32 | 104 | 0.40 | 0.5125 | 0.7525 | 0.7500 | 0.8571 | 0.6667 | 42 / 7 / 21 | 0.0634 | 0.5231 | 0.8718 |
| YOLO11s modified_100 high-res | old + 100追加 | 1280 / 16 | 200 | 0.52 | 0.4984 | 0.7703 | 0.7778 | 0.9333 | 0.6667 | 42 / 3 / 21 | 0.0489 | 0.6650 | 0.9188 |

今回の `imgsz=1280`、`batch=16` の YOLO11s は、同じ `modified_100` データを `imgsz=960` で学習したモデルと比べると、false positive が `7` から `3` に減り、test F1 は `0.7500` から `0.7778` に上がりました。keypoint も改善しており、NME は `0.0634` から `0.0489` に下がり、PCK@0.05 は `0.5231` から `0.6650` に上がっています。

F1 の差は、bbox/keypoint の質そのものよりも TP/FP/FN の差に強く影響されています。高解像度版は検出できた bbox の mean IoU と keypoint NME は良い一方で、旧 baseline より TP が少なく FN が多いため、F1 では不利になっています。

| model | TP | FP | FN | precision | recall | F1 | mean IoU | keypoint NME |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s old repro | 44 | 2 | 19 | 0.9565 | 0.6984 | 0.8073 | 0.8008 | 0.0554 |
| YOLO11s modified_50 | 44 | 1 | 19 | 0.9778 | 0.6984 | 0.8148 | 0.8198 | 0.0585 |
| YOLO11s modified_100 high-res | 42 | 3 | 21 | 0.9333 | 0.6667 | 0.7778 | 0.8276 | 0.0489 |

test split 上で後追いの threshold 最適化を行うと、高解像度版は `threshold=0.68` で `TP=42`、`FP=0`、`FN=21`、F1 `0.8000` まで上がります。ただし TP は増えません。逆に `threshold=0.01` まで下げると `TP=49` まで拾えますが、`FP=20` まで増えて F1 は `0.7424` に下がります。このため、現状の主な課題は bbox regression や keypoint regression よりも、難しい顔の confidence と false positive の score separation です。

一方で、bbox detection F1 だけを見ると、旧 baseline の `old_repro` と `modified_50` にはまだ届いていません。高解像度化は keypoint 品質には効いていますが、検出漏れは `42 TP / 21 FN` のままなので、総合 baseline を置き換えるには recall 改善が必要です。学習ログ上の val best pose mAP50-95 は `0.6879` at epoch `128` で、`modified_100` の `0.6631` より高くなっています。

関連する成果物は次にあります。

- `outputs_experiment/yolo_pose_dataset_ablation/model_comparison.csv`: 上の比較表の元データ。
- `outputs_experiment/yolo_pose_dataset_ablation/evaluations/modified_100_imgsz1280_b16/`: 今回追加した高解像度モデルの val threshold sweep と test 評価。
- `runs/face_pose_dataset_ablation/yolo11s_modified_100_imgsz1280_b16/`: 高解像度 YOLO11s の学習ログ、`results.csv`、`best.pt`、`last.pt`。

### YOLO11s の hard mining + keypoint-safe augmentation

`output_experiment/03_yolo_aug/yolo11s_hard_mining_balanced_v1_aug_keypoint_safe/` では、YOLO11s-pose に hard example を追加した balanced dataset と、keypoint への影響を抑える augmentation を適用し、顔 keypoint 精度が改善するかを確認しました。

主な条件は次のとおりです。

- model: `yolo11s-pose.pt`
- train: `exp_yolo11s_hard_mining_balanced_v1`
- val/test: old dataset 固定、val 100 画像 / test 30 画像
- imgsz: `960`
- batch: `8`
- confidence threshold: `0.51` 固定
- augmentation: `aug_keypoint_safe`
- max epochs: `200`
- early stopping: epoch `165` で停止、best epoch は `115`

この検証では、hard example 追加で small face、edge face、occluded candidate に強くなるか、また keypoint-safe augmentation で NME と PCK@0.05 が改善するかを見ています。ただし、この run は dataset と augmentation を同時に変更しているため、悪化や改善の原因を片方だけに分解するものではありません。

| split | pose mAP50-95 | baseline | NME | baseline | PCK@0.05 | baseline | fixed F1 | baseline | failures | baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.6186 | 0.6486 | 0.0977 | 0.0780 | 0.2623 | 0.3750 | 0.7719 | 0.8092 | 97 | 86 |
| test | 0.7405 | 0.7476 | 0.0640 | 0.0554 | 0.5467 | 0.6220 | 0.8257 | 0.8073 | 41 | 39 |

test の fixed F1 は `0.8073` から `0.8257` に少し改善しました。一方で、keypoint-first の重要指標は val/test ともに悪化しています。NME は val で `0.0780` から `0.0977`、test で `0.0554` から `0.0640` に上がり、PCK@0.05 は val で `0.3750` から `0.2623`、test で `0.6220` から `0.5467` に下がりました。この条件は、現時点では採用候補として弱いです。次に試すなら、失敗可視化を確認したうえで、hard example の比率または augmentation のどちらか一方だけを弱めた条件を切り分けます。

### YOLO11s の train-time augmentation Phase 1

`scripts_YOLO/run_yolo11s_aug_phase1.py` は、YOLO11s Pose に対して Ultralytics 標準の train-time augmentation だけを比較するための実験 runner です。Phase 0 としてデータセット監査を行い、その後 `E0_current_control`、`E1_cutmix_light`、`E2_mosaic_mid`、`E3_viewpoint_light` の 4 条件を同じデータ、同じ model、同じ学習条件で比較します。

監査だけを行う場合は次のように実行します。

```bash
.venv/bin/python scripts_YOLO/run_yolo11s_aug_phase1.py \
  --data-root outputs_experiment/yolo_pose_dataset_ablation_20260624_task2/datasets/modified_100 \
  --output-dir output_experiment/05_yolo_aug_phase1 \
  --manual-tags-csv output_experiment/01_failure_mining/failures_review.csv \
  --audit-only
```

4 条件の学習と評価をまとめて行う場合は次のように実行します。

```bash
.venv/bin/python scripts_YOLO/run_yolo11s_aug_phase1.py \
  --data-root outputs_experiment/yolo_pose_dataset_ablation_20260624_task2/datasets/modified_100 \
  --output-dir output_experiment/05_yolo_aug_phase1 \
  --manual-tags-csv output_experiment/01_failure_mining/failures_review.csv \
  --model yolo11s-pose.pt \
  --imgsz 960 \
  --epochs 200 \
  --batch 8 \
  --patience 50 \
  --device 0 \
  --save-error-images
```

主な出力は次のとおりです。

- `phase0_dataset_audit.json`: 画像数、instance 数、positive/negative 数、split 重複、近接 frame、bbox 分布、keypoint visibility、`flip_idx` をまとめた監査結果。
- `phase1_model_comparison.csv`: val で選んだ confidence threshold を test に適用した全体指標。
- `phase1_tag_fn_recall.csv`: `angle`、`occlusion`、`pixelated_blur`、`small`、`crowded`、`edge` のタグ別 FN/recall 集計。
- `phase1_adoption_check.csv`: E0 比での F1、NME、PCK@0.05 の変化と採用候補判定。
- `evaluations/<experiment>/`: threshold sweep、val/test metrics、失敗分析 CSV、推論可視化。
- `runs/<experiment>/`: Ultralytics の学習ログ、`args.yaml`、`results.csv`、`train_batch` 画像、`weights/best.pt`。

`small`、`crowded`、`edge` は bbox と画像内 object 数から自動集計されます。`angle`、`occlusion`、`pixelated_blur` は `--manual-tags-csv` に `image` または `image_path` と、`tags` / `manual_failure_type` / `auto_failure_type` などの列を持つ CSV を渡した場合に集計されます。

`mosaic` は Ultralytics YOLO の 4 画像合成 augmentation であり、pixelated / blur とは別問題として扱います。横顔や俯瞰などの yaw / pitch 問題も、2D 回転だけで作れるものとしては扱わず、Phase 1 では軽い viewpoint 耐性の確認に留めます。遮蔽や pixelated / blur を人工的に作る offline augmentation は、Phase 1 後に失敗タグとして残った場合の Phase 2 候補です。

### RTMDet + RTMPose-face の実験結果

YOLO Pose とは別系統の実験として、OpenMMLab の RTMDet と RTMPose を組み合わせた二段構成の fine-tune も行っています。RTMDet は `head` と `face` の 2 クラスを同時に検出し、そのうち `face` bbox だけを RTMPose に渡して 5 点 keypoint を推定します。つまり、現在の構成は「頭を検出してから顔を探す」cascade ではありません。

実験用の変換データ、checkpoint、推論結果、評価値、可視化画像は `output_experiment/` 以下に保存します。このディレクトリは Git 管理対象外です。実行の入口は `script/script_experiment/rtmdet_rtmpose_experiment.py` です。

```bash
.venv/bin/python script/script_experiment/rtmdet_rtmpose_experiment.py run-all \
  --data-root . \
  --output-dir output_experiment \
  --mode full \
  --device cuda:0
```

現在の test split は、YOLO11s Pose と SCRFD + MediaPipe の比較で使ったものと同じ 30 画像、63 件の face アノテーションです。次の表では、face bbox detection として公平に比較できる値だけを並べています。RTMDet は可視化を見やすくするため、スコアしきい値 `0.50` で評価しています。YOLO11s Pose と SCRFD + MediaPipe は、それぞれ val split で F1 が最も高くなったしきい値を test split に適用しています。

| method | threshold | test F1 | precision | recall | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s Pose | 0.51 | 0.8073 | 0.9565 | 0.6984 | 44 | 2 | 19 |
| RTMDet + RTMPose-face | 0.50 | 0.6931 | 0.9211 | 0.5556 | 35 | 3 | 28 |
| SCRFD + MediaPipe | 0.09 | 0.5938 | 0.5846 | 0.6032 | 38 | 27 | 25 |

この条件では、YOLO11s Pose が最もよい F1 と recall を出しています。RTMDet + RTMPose-face は YOLO11s Pose より検出漏れが多く、F1 は下がります。一方で、SCRFD + MediaPipe と比べると false positive が大きく減り、precision と F1 は上回っています。SCRFD + MediaPipe は recall では RTMDet + RTMPose-face より少し高いものの、false positive が多く、画面上の検出結果も散らかりやすい傾向があります。

RTMDet 単体の `head` と `face` をまとめた参考値は precision `0.9579`、recall `0.6842`、F1 `0.7982` です。ただし、これは `head` クラスも含む集計なので、YOLO11s Pose や SCRFD + MediaPipe の face bbox detection とは直接比較しません。two-stage の入口として見るべき値は、上の表にある RTMDet の face-only の値です。

keypoint については、現時点では RTMDet + RTMPose-face が YOLO11s Pose と SCRFD + MediaPipe に届いていません。RTMDet + RTMPose-face の two-stage 評価では NME `0.0740`、PCK@0.05 `0.4762`、PCK@0.10 `0.7619` でした。既存の probe では、YOLO11s Pose が NME `0.0352`、PCK@0.05 `0.8086`、PCK@0.10 `0.9522`、SCRFD + MediaPipe が NME `0.0352`、PCK@0.05 `0.8389`、PCK@0.10 `0.9667` です。ただし、この keypoint 比較は完全に同じ evaluator ではなく、既存 probe に基づく参考比較です。

現状の結論としては、総合的な baseline は引き続き YOLO11s Pose です。RTMDet + RTMPose-face は SCRFD + MediaPipe より誤検出を抑えやすい一方で、face recall と keypoint 精度の改善が必要です。次に進めるなら、RTMDet の face class の threshold 最適化、入力解像度、small face の補強、RTMPose の学習 schedule と augmentation を重点的に見直します。

`outputs_experiment/rtmdet_rtmpose_dataset_ablation/model_comparison.csv` には、RTMDet + RTMPose-face を旧データと追加データで比較した結果も保存されています。test split は同じ 30 画像です。

| model | train images | val threshold | detector F1 | face recall | two-stage NME | two-stage PCK@0.05 | two-stage PCK@0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old_repro | 260 | 0.43 | 0.7871 | 0.6825 | 0.0334 | 0.8732 | 0.9463 |
| modified_50 | 280 | 0.46 | 0.8163 | 0.6825 | 0.0259 | 0.9235 | 0.9643 |
| modified_100 | 560 | 0.42 | 0.8367 | 0.7460 | 0.0324 | 0.8378 | 0.9414 |

RTMDet 側では `modified_100` が detector F1 と face recall を改善しています。一方で、keypoint 品質は `modified_50` が最もよく、追加データを増やすだけでは RTMPose の精度が単調に上がるわけではありません。RTMPose の fixed-100 same-condition 比較は `output_experiment/learning_curves_fixed100_compare/` にあり、同一 trajectory 内では epoch100 / best92 が epoch20 / epoch50 より良いものの、learning rate schedule の見直し余地が残っています。

### RTMDet + RTMPose の stage 数比較

2026-07-04 に、DINO 系ではなく RTMDet / RTMPose 系のモデルで FDP 系列の stage 数比較を行いました。backbone は RTMDet / RTMPose の CSPNeXt 系であり、DINOv3 backbone はこの比較には含めていません。

比較した構成は次の通りです。

| alias | stage | pipeline |
| --- | --- | --- |
| FDP | 1stage | image -> RTMDet-HF(face on full image) -> RTMPose(face) |
| HDP | 2stage | image -> RTMDet-HF(head on full image) -> head crop -> RTMPose(head) |
| GTFP | 2stage_GT | GT face bbox -> RTMPose(face) |
| HFDP | 3stage | image -> RTMDet-HF(head on full image) -> head crop -> RTMDet-headcrop-face(face on crop) -> RTMPose(face) |
| GTHFDP | 3stage_GT | GT head bbox -> head crop -> RTMDet-headcrop-face(face on crop) -> RTMPose(face) |

test split は 30 画像、63 face instances です。今回の主比較では、1stage の `FDP` より、head crop を挟んで face detection と keypoint 推定を行う 3stage の `HFDP` の方が keypoint 精度は改善しました。

| stage | NME | PCK@0.05 | PCK@0.10 | missed |
| --- | ---: | ---: | ---: | ---: |
| FDP | 0.0489 | 0.8209 | 0.9005 | 20 |
| HDP | 0.0563 | 0.8390 | 0.9195 | 9 |
| GTFP | 0.0702 | 0.7148 | 0.8242 | 0 |
| HFDP | 0.0458 | 0.8482 | 0.9215 | 23 |
| GTHFDP | 0.0574 | 0.7982 | 0.8807 | 14 |

`HFDP` は `FDP` に対して NME が `0.0489 -> 0.0458` に下がり、PCK@0.05 が `0.8209 -> 0.8482` に上がりました。したがって、今回の RTMDet 系条件では stage を分け、head 検出から head crop 内の face 検出へ進む構成が keypoint の位置精度に有効でした。一方で missed instances は `20 -> 23` に増えているため、head 検出または crop 内 face 検出で落ちるケースの改善はまだ必要です。

今回の結果、checkpoint、学習曲線、推論時間は `outsput/experience/reports/rtm_hfdp_20260704_170447_summary.md`、学習曲線は `outsput/experience/reports/curves_rtm_hfdp_20260704_170447/` に保存しています。次の検証では、同じ FDP / HDP / GTFP / HFDP / GTHFDP の stage 比較を DINO 系モデルでも行い、RTMDet / RTMPose 系で得られた stage 分割の効果が DINO backbone でも再現するかを確認します。

### データ監査と追加データセット

ローカルの `output_experiment/` と `outputs_experiment/` には、README に未反映だったデータ監査とデータセット作成結果があります。公開 repo には実データを含めない前提なので、数値はローカル成果物がある環境での確認値です。

- `output_experiment/output_data_visual_check/`: 実際に学習へ渡す YOLO/COCO 出力を可視化して確認した結果です。YOLO label は 390 files、non-empty は 346 files、multi-object label は 142 files でした。COCO keypoint は train/val/test が 387 / 100 / 63 annotations です。
- `output_experiment/annotation_gap_check/`: 元 metadata の監査です。no-face/object records は 44 件で、全て negative sample として空 label が作成されています。non-negative で head 数が face ellipse 数より多い record は 33 件、推定 missing face slots は 47 件です。
- `output_experiment/02_datasets/dataset_compare.csv`: 改善用データセット候補を保存しています。`exp_yolo11s_base_repro` は train 260 images、`hard_mining_balanced_v1` は train 433 images / hard examples 173、`negative_v1` は train 291 images / negative 55 images です。
- `output_experiment/01_failure_mining/`: 失敗 mining の中間結果です。主な候補は `keypoint_ng` 69 images、`fn` 42 images、`occluded_candidate` 42 images、`small` 25 images、`edge` 22 images、`fp` 5 images です。
- `output_experiment/03_yolo_aug/`: hard mining + keypoint-safe augmentation の YOLO11s 実験ログがあります。`report.md`、`compare_to_baseline.csv`、`eval_val/`、`eval_test/` に固定 val/test での比較結果を保存しています。

主な観察結果は次のとおりです。

- YOLO11s Pose は precision が高い一方で、難しい face の検出漏れが残っています。
- 検出できていない face は、confidence threshold の調整だけではあまり改善しにくい傾向があります。
- 失敗しやすい条件には、斜め向きの顔、blur、occlusion、小さい顔、partial face、極端な close-up があります。
- 失敗理由の目視確認では、raw 出力にアノテーション bbox と一部重なるだけで顔を正しく囲えていない矩形が多く含まれていました。次の改善では、顔として確信を持てる矩形の confidence を上げる方針にします。具体的には、学習データの bbox 定義の一貫性、難例の追加、入力解像度、または学習設定を見直します。

顔検出では、次の観点を分けて確認することを推奨します。

- 顔が写っていない画像: false positive が過度に増えていないかを確認します。
- 遮蔽された顔、または画面端で見切れている顔: clip された bbox が画像端付近で検出できているかを確認します。
- 小さい顔: metadata で大きさを分類できる場合は、小さい顔の recall を別に確認します。
- 顔が密集している画像: 隣接する顔の検出漏れや重複検出を確認します。
- keypoint の visibility: 欠損点が `0` のままになっているか、存在する点が画像内に収まっているかを確認します。

## SCRFD と MediaPipe の推論

SCRFD による顔検出と、必要に応じた MediaPipe による目のランドマーク抽出を実行できます。

```bash
.venv/bin/python script/script_model/run_scrfd_mediapipe.py \
  --ground-truth annotations/ground_truth.json \
  --model models/scrfd/<scrfd_model>.onnx \
  --output output/scrfd_mediapipe_predictions.json \
  --input-size 640 \
  --score-threshold 0.5 \
  --nms-threshold 0.4 \
  --mediapipe-crop-margin 0.15
```

MediaPipe を使わず、SCRFD の検出結果と 5 点ランドマークだけを保存する場合は、`--skip-mediapipe` を指定してください。

出力される JSON は、次のような構造です。

```json
{
  "schema_version": "1.0",
  "model": {
    "detector": "SCRFD",
    "detector_path": "models/scrfd/<scrfd_model>.onnx",
    "landmark_refiner": "MediaPipe FaceMesh",
    "input_size": 640,
    "score_threshold": 0.5,
    "nms_threshold": 0.4,
    "mediapipe_crop_margin": 0.15
  },
  "records": [
    {
      "annotation_kind": "<kind>",
      "annotation_mode": "<mode>",
      "image_stem": "<stem>",
      "image_name": "<image>",
      "image_path": "<path>",
      "detections": [
        {
          "bbox": {"xtl": 0.0, "ytl": 0.0, "xbr": 100.0, "ybr": 100.0},
          "score": 0.99,
          "scrfd_landmarks_5pt": [],
          "mediapipe_eye_landmarks": []
        }
      ]
    }
  ]
}
```

SCRFD/MediaPipe の検出結果を評価する場合は、次のように実行します。

```bash
.venv/bin/python script/script_experiment/evaluate_face_detections.py \
  --ground-truth annotations/ground_truth.json \
  --predictions output/scrfd_mediapipe_predictions.json \
  --target-region head \
  --iou-threshold 0.5 \
  --match-policy iou \
  --summary-output output/detection_summary.json \
  --matches-output output/detection_matches.csv
```

検出結果の overlay 画像を生成する場合は、次のように実行します。

```bash
.venv/bin/python script/script_experiment/visualize_detections.py \
  --ground-truth annotations/ground_truth.json \
  --predictions output/scrfd_mediapipe_predictions.json \
  --output-dir output/detection_overlays \
  --target-region head \
  --bbox-scale 1.8 \
  --iou-threshold 0.5 \
  --only-errors
```

SCRFD と MediaPipe の処理段階を出力し、必要に応じて overlay も保存する場合は、次のように実行します。

```bash
.venv/bin/python script/script_model/dump_detector_stages.py \
  --input input/images \
  --output output/detector_stage_dump \
  --model models/scrfd/<scrfd_model>.onnx \
  --score-min 0.01 \
  --save-overlays
```

処理段階の出力には、`raw_scrfd.jsonl`、`raw_mediapipe.jsonl`、`merged_candidates.jsonl`、`postprocessed.jsonl`、`removed_detections.jsonl`、`summary.csv`、`skipped.csv`、および任意の `overlays/` が含まれます。

## YOLO11s Pose augmentation ablation

YOLO11s Pose の train-time augmentation を比較しました。ここでの augmentation は学習時の前処理であり、train/val/test split は変更していません。val/test には augmentation を入れず、評価時の `conf=0.25`、`iou=0.7` は固定です。cutmix は全条件で `0` にしています。

共通条件は `model=yolo11s-pose.pt`、`imgsz=960`、`batch=8`、`epochs=200`、`patience=50`、`seed=42`、`device=0` です。dataset yaml は `outputs_experiment/yolo_pose_dataset_ablation_20260624_task2/datasets/modified_100/face_pose.yaml` を使いました。

主な出力先は次のとおりです。

- 学習結果: `runs/pose_aug_ablation/<experiment>/`
- test 評価: `outputs/aug_ablation_test_eval/<experiment>/`
- 集計: `outputs/aug_ablation_test_eval/summary_metrics.csv`
- mosaic 比較: `outputs/aug_ablation_test_eval/mosaic_epoch100_comparison.csv`
- review HTML: `outputs/aug_ablation_test_eval/<experiment>/review_report.html`

test split の固定閾値評価では、次の結果になりました。

| experiment | box recall | box mAP50-95 | pose recall | pose mAP50-95 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| E0_current_control | 0.7143 | 0.4642 | 0.7302 | 0.7094 | baseline |
| E1_photo_light | 0.7302 | 0.4125 | 0.7460 | 0.7154 | light photometric |
| E2_photo_mid | 0.7619 | 0.4373 | 0.7778 | 0.7403 | medium photometric |
| E3_geo_mid | 0.7460 | 0.4515 | 0.7778 | 0.7460 | medium geometry |
| E4_mosaic_mid | 0.6984 | 0.4099 | 0.7619 | 0.7342 | interrupted after epoch 112 was saved |
| E5_combined_best | | | | | not run |

E2 と E3 は E0 より pose recall と pose mAP50-95 が改善しました。E1 は recall は上がりましたが、box mAP50-95 の低下が大きめです。E4 は pose recall と pose mAP50-95 は改善しましたが、box recall と box mAP50-95 は下がりました。

E4 は、mosaic を弱めにするか中程度にするかを早めに判断するため、100 epoch 付近で停止しました。厳密な epoch 100 の weight は保存していないため、epoch 100 の直接比較は `results.csv` の validation 行で行い、test は保存済み `best.pt` と停止時点の `last.pt` を補助的に評価しています。

| comparison | metric | E0 mosaic=0.2 | E4 mosaic=0.4 | delta |
| --- | --- | ---: | ---: | ---: |
| val epoch100 | box recall | 0.7085 | 0.7400 | +0.0315 |
| val epoch100 | pose recall | 0.6886 | 0.7500 | +0.0614 |
| val epoch100 | pose mAP50-95 | 0.6350 | 0.6840 | +0.0490 |
| test best | pose recall | 0.7302 | 0.7619 | +0.0317 |
| test best | pose mAP50-95 | 0.7094 | 0.7342 | +0.0248 |
| test best | box mAP50-95 | 0.4642 | 0.4099 | -0.0543 |

現時点では、keypoint と recall を優先するなら `mosaic=0.4, close_mosaic=30` は有効候補です。一方で box mAP50-95 が下がるため、次は `mosaic=0.3` または `mosaic=0.4, close_mosaic=20` のように中間条件を確認するのが妥当です。combined 条件を試す場合は、E2 の photometric と E3 の geometry を基準にし、mosaic は E4 より少し弱める候補を残します。

## Git に含めないもの

次のファイルやディレクトリは、Git に含めない運用にしてください。

- `.env`、仮想環境、Python のキャッシュ。
- `data/`、`datasets/`、`input/`、`output/`、`runs/`、`results/`、生成した出力、キャッシュ、実験結果。
- 抽出した `frames/`、入力用の `videos/`、入力用の `images/`。
- CVAT から export したファイル、アノテーション、label、生成済みの YOLO label。
- モデル用のディレクトリと重みファイル。例: `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.engine`、`*.tflite`。
- 動画ファイル、画像ファイル、JSONL ファイル、ローカルデータベース、実験管理ツールの出力。

GitHub に公開する前に、次のコマンドで追加予定のファイルを確認してください。

```bash
git status --ignored
git add --dry-run .
```

ソースコード、ドキュメント、例示用ファイル、依存パッケージの定義だけが追加対象になっていることを確認してください。

## Troubleshooting

- `SCRFD model not found` が出る場合: SCRFD の ONNX ファイルをローカルに配置し、`--model` で指定してください。
- `Missing ultralytics` が出る場合: `.venv/bin/python -m pip install -r requirements.txt` を実行してください。
- CUDA が使用できない場合: NVIDIA driver と互換性がある PyTorch をインストールするか、`--device cpu` を指定してください。
- YOLO の学習中に out of memory が出る場合: `--batch` を小さくするか、`--imgsz` を小さくするか、より小さいモデルを使用してください。
- 推論結果が出ない場合: `--conf` を下げるか、指定したモデルの重みが目的の task に対応しているかを確認してください。
- dataset checker が失敗する場合: label の欠損、bbox の範囲、keypoint の visibility、生成された YAML の `kpt_shape` を確認してください。
- overlay 生成時に import error が出る場合: リポジトリのルートからコマンドを実行してください。
