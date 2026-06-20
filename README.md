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
  visualize_yolo_pose_labels.py        # ラベル確認用の overlay 画像を生成する
  train_yolo_pose.py                   # Ultralytics YOLO Pose の学習を実行する
  predict_yolo_pose.py                 # YOLO Pose の推論結果を画像または動画として可視化する
  export_yolo_pose_jsonl.py            # YOLO Pose の推論結果を JSONL で出力する
  evaluate_yolo_pose_f1.py             # bbox detection F1 と threshold を評価する
  run_yolo_pose_gpu_compare.py         # 複数の YOLO Pose モデルを GPU で比較する

script/
  script_data/                         # アノテーションとフレームに関する補助スクリプト
  script_model/                        # SCRFD/MediaPipe の推論とデバッグ用スクリプト
  script_experiment/                   # 評価、可視化、確認用の補助スクリプト
    rtmdet_rtmpose_experiment.py       # RTMDet + RTMPose の変換、学習、推論、評価をまとめた CLI
  script_process/                      # 前処理と後処理の補助スクリプト

docs/
  detector_stage_dump.md               # SCRFD/MediaPipe の debug 出力に関する説明

requirements.txt                       # Python の依存パッケージ
```

現時点では、`src/`、`configs/`、`tests/`、`pyproject.toml` は存在しません。

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
```

`models/` と `weights/` は `.gitignore` によって除外されます。実行時には、`--model` または `--weights` にローカルのパスを指定してください。

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
