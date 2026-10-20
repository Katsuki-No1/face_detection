# FACE/楕円検出 + Query Refinement baseline

HEADを使わない2段構成の顔検出・5点keypoint検出モデルです。

```text
全画像 → MAL α=2 FACE/楕円検出 → Face crop → DINOv3 Query Refinement
```

性能欄の評価対象はV3学習済みの `mal_alpha2_ellipse_dataset_v3_r3` と
`mal_alpha2_dataset_v3_query_refinement` の組み合わせです。
使用したcheckpointと固定閾値は[dataset v3の評価記録](docs/baseline_dataset_v3_performance.md)に記載しています。

このブランチに同梱する[registry](configs/model_registry/local_baseline.yaml)と下記の実行例は旧モデルの保存設定です。
V3の性能値は評価記録に示す2つの重みの結果であり、旧設定・旧重みで再現した値ではありません。

## 性能（dataset v3）

**V3で学習したFACE/楕円検出とQuery Refinementを組み合わせた、HEAD-freeの2段パイプライン**の保存済み結果です。
データセットは `face_pose_condition_balanced_811_latest_sources_r3_20261009`。
動画単位の固定split（train 16,411枚／val 2,022枚／test 2,060枚）を使用しています。

validationでFACE bbox F1を最大化した後、FACE閾値を固定して5点keypointのsemantic Macro F1（NME ≤ 0.10）を最大化しています。
**FACE score = 0.314453125、point presence = 0.59716796875** をtestにも固定適用しました。

### FACE検出

| Split | 画像数 | FACE GT | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| val | 2,022 | 3,072 | 95.82% | 91.73% | 93.73% |
| test | 2,060 | 2,972 | 95.06% | 91.89% | 93.45% |

FACE bbox IoU ≥ 0.50の一対一マッチで、全native FACE GTを対象にします。
未対応予測は重複を含めFP、未対応GTはFNです。

| Split | bbox mAP50:95 | bbox AP50 | bbox AP75 | bbox平均IoU（対応済みのみ） | 楕円平均IoU（未検出=0） |
| --- | --- | --- | --- | --- | --- |
| val | 65.35% | 94.36% | 74.86% | 0.8395 | 0.6989 |
| test | 65.95% | 94.48% | 76.77% | 0.8445 | 0.7114 |

APは信頼度順の全予測によるCOCO評価（各画像上限100）です。
bbox平均IoUは検出成功例のみ、楕円平均IoUは全FACE GTが分母です。

### 5点keypoint（end-to-end）

| Split | Macro F1@NME0.05 | Macro F1@NME0.10 | 眼重視F1@0.05 | 眼重視F1@0.10 | 平均NME |
| --- | --- | --- | --- | --- | --- |
| val | 87.84% | 90.57% | 87.11% | 90.03% | 0.0213 |
| test | 87.37% | 90.10% | 86.61% | 89.45% | 0.0215 |

FACEの見逃しと未対応FACE予測に付随するkeypointのFPを含みます。
Macro F1は左右の目・鼻・左右の口の5スロットのF1の単純平均、眼重視版は固定重み `[2,2,1,1,1]` です。
NMEは元画像上の距離を `1.20 × max(GT Head幅, GT Head高さ)` で正規化します（推論にHEAD検出器は使用しません）。
平均NMEはGT点が存在し、点が出力された例を対象とするため、見逃しを含むF1と併記します。
属性対応が未解決のFACEはval/test各32件をkeypoint評価から除外し、FACE検出の分母には残しています。

[詳細：共通閾値0.33／0.70、点別性能、遮蔽・横顔、モデルSHA・評価出典](docs/baseline_dataset_v3_performance.md)
に条件と件数を掲載しています。旧530／528枚のFACE成分のみの結果は[過去の性能記録](docs/baseline_performance.md)に保持しています。

## 準備

別デバイスでは `baseline` ブランチを指定して取得します。

```bash
git clone --branch baseline --single-branch https://github.com/Katsuki-No1/face_detection.git
cd face_detection
git clone https://github.com/facebookresearch/dinov3.git models/source/dinov3
git -C models/source/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

DINOv3の実装はローカルの `models/source/dinov3` から読み込むため、checkpointとは別に必要です。
上記は元の開発環境で使用しているDINOv3ソースのrevisionです。
Python環境に `requirements.txt` の依存を用意し、GPUを使う場合はPyTorch・CUDA・ドライバを対応させます。
顔検出だけならFACE成分のcheckpoint、一式を使う場合はFACEとQuery Refinementの2つを用意します。
新しいデバイスでの実機動作は未検証です。

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

同梱する旧FACE成分の画像推論例（上記V3評価とは別の重み）：

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
V3の一式を再現する場合は評価記録の2つの重み・対応する設定を別途用意し、HEAD-freeの構成とvalidation固定閾値を確認してください。

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
