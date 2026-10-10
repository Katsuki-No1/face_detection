# FACE/楕円検出 + Query Refinement baseline

HEADを使わない2段構成の顔検出・5点keypoint検出モデルです。

```text
全画像 → MAL α=2 FACE/楕円検出 → Face crop → DINOv3 Query Refinement
```

顔検出部は既存WIDER事前学習済みDEIMv2-Lを目的データでfine-tuneした
`mal_alpha2_ellipse_from_existing_wider`、keypoint部は既存A1/A2版Query Refinementです。
定義・設定・checkpoint SHA・閾値は [local_baseline.yaml](configs/model_registry/local_baseline.yaml) に固定しています。

## 性能

以下は **FACE/楕円検出部のみ** の保存済み結果です。
一式の新しい組み合わせ全体は未評価であり、この表にはkeypoint性能を含めていません。

| Split | 画像数 | FACE GT | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| val | 530 | 840 | 93.53% | 91.19% | 92.34% |
| test | 528 | 769 | 95.74% | 93.50% | 94.61% |

固定 `face_pose_drive_v2_balanced_811`、FACE score 0.33、bbox IoU 0.50の一対一マッチ。
checkpointはvalidationで選んだepoch45 EMAです。
[性能の詳細・遮蔽と横顔の集計状況・出典](docs/baseline_performance.md) に、
validation固定閾値での結果、bbox/楕円AP、annotation hashを掲載しています。

## 準備

Python環境に `requirements.txt` の依存を用意します。
入力画像・固定splitのアノテーション・事前学習重み・2成分のcheckpointはローカルに配置してください。
Gitに重み、データセット、生成キャッシュは含めていません。
設定内の `outputs/` や初期化重みの絶対パスは学習時の配置を記録しています。
別環境でパスを変更する場合は別設定として保存し、登録済み設定と重みの識別情報を保全します。

## 実行入口

| 用途 | 入口 |
|---|---|
| FACE/楕円の画像推論 | `tools/infer_dinov3_deimv2_ellipse.py` |
| FACE/楕円の動画推論 | `tools/infer_dinov3_deimv2_ellipse_video.py` |
| FACE/楕円の評価 | `tools/eval_dinov3_deimv2_ellipse.py` |
| FACE/楕円のゲート・学習 | `tools/run_mal_existing_wider.py` |
| keypointの学習・評価 | `keypoint_comparison1/scripts/run_landmark_experiments.py`（`query_refinement`） |
| FACE予測からkeypointまでの診断評価 | `tools/evaluate_deim_query_baseline_aligned.py` |

顔検出部の画像推論例：

```bash
.venv/bin/python tools/infer_dinov3_deimv2_ellipse.py \
  --config configs/dinov3_deimv2/mal_alpha2_ellipse_from_existing_wider.yaml \
  --checkpoint outputs/experiments/mal_alpha2_ellipse_from_existing_wider/checkpoints/best_val_map.pt \
  --input data/example_images \
  --output outputs/baseline_inference \
  --score-thr 0.33 --device cuda:0
```

各入口の引数は `--help` で確認できます。この画像推論例はFACE/楕円検出部だけを実行します。
keypoint成分の学習時設定や過去の評価設定は、そのまま現在の一式の実行設定を意味しません。
一式を評価する際はregistryの2つの重みを明示し、HEAD-freeの構成とvalidation固定閾値を確認してください。

## 開発

```bash
.venv/bin/python -m pytest -q
```

`configs/` は現行成分と設定継承に必要なもの、`scripts/`・`tools/` は実行入口、
`src/`・`keypoint_comparison1/landmark_detection/` はモデル・データ処理の内部実装です。
内部の共有モジュールと学習時の設定はcheckpoint互換性を保つため保持しています。

このブランチの一覧はbaseline用に整理しています。削除前の実験ファイルはGit履歴に残り、
Messenger/遮蔽回復の追加開発は `occlusion` ブランチで管理します。
[文書一覧](docs/README.md) / [構成の記録](docs/model_baseline_history.md) / [開発記録](docs/development_journal.md)
