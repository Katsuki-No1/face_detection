# YOLO26 Pose production comparison

`face_pose_drive_v1` の固定3,090 frameを用い、YOLO26s-poseとYOLO26m-poseを
同一条件で学習・評価するための本番用パイプラインです。短縮学習、subset、demo用の
分岐はありません。両モデルのfull training、validation選択、独立test評価まで完了しています。

## 現在のモデル選択

同じsplitとprotocolで比較した結果、`m`は設定済みの品質gateをすべて通過しました。
最大性能を優先する場合は`m`、現在の2,460 train frameでparameter数・学習時間との効率を
優先する場合は`s`を選択します。将来datasetを拡大した際も同じrunnerで再比較できます。

`compare` stageは次のgateを評価します。

- `m` のpose mAP50:95が `s` より0.010以上改善する。
- `m` のbbox mAP50:95低下が0.005以下である。
- `m` のNME悪化が0.002以下である。

すべて満たす場合だけ現在のdatasetでも `m` を推奨します。Active Learningで学習データが
増えた後も同じ比較を再実行できるため、将来 `m` へ移行しても入口と評価方法は変わりません。

## 実測結果

checkpointと動作閾値はvalidationだけで選び、独立test 330画像 / 503 faceへ固定しました。
sはepoch 188で停止してbest epoch 138、mはepoch 193で停止してbest epoch 143です。

| variant | parameters | GFLOPs | training time | bbox mAP50:95 | pose mAP50:95 | NME | PCK@0.05 | fixed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s | 10,579,532 | 24.5 | 1.960 h | 0.4814 | 0.8173 | 0.0461 | 0.7613 | 0.8809 |
| m | 23,535,244 | 82.1 | 3.640 h | 0.5178 | 0.8385 | 0.0414 | 0.7924 | 0.8866 |

m - sはpose mAP50:95 +0.0213、bbox mAP50:95 +0.0363、NME -0.0048で、
3つの品質gateをすべて通過しました。比較結果は
`outputs/yolo/yolo26_pose/face_pose_drive_v1/comparison.{json,csv,md}`に保存しています。

## 実装上のproduction条件

- train / val / test = 2,460 / 300 / 330を固定し、`fraction=1.0`を強制する。
- COCO JSONのSHA-256、画像数、label数、全label構文を学習前に検査する。
- 元動画ID、画像SHA-256、画像pathのsplit間重複を検査する。
- official pose checkpointを使用し、17点human-pose headを5点face-pose headへ置換する。
- YOLO26のend-to-end modeを維持できない場合は学習を拒否する。
- 960 px、200 epochs、MuSGD、AMP、cosine schedule、全epoch validationを使う。
- `best.pt`とは別に5 epochごとのcheckpointを保存し、validationが50 epoch改善しない場合は
  正常終了するearly stoppingを使う（最大200 epoch）。
- sはbatch 16、mはbatch 8、両方ともnominal batch size 64で正規化する。
- test thresholdをtest上で探索しない。valのbest F1 thresholdをtestへ固定する。
- bbox / pose mAP、固定閾値P/R/F1、keypoint NME/PCK、failure分類を保存する。
- 既存runを上書きしない。完了runは再利用し、partial runは明示的な`--resume`だけを許す。
- W&Bは`sato-waseda-university/face_detection`へonline記録し、s/mを
  `face_pose_drive_v1_four_model_comparison` groupで比較する。
- preflightでW&B認証とprojectアクセスを検査し、失敗した場合は学習を開始しない。
- W&B run IDはexperiment・model・seedから決定的に生成し、`--resume`時は同じrunへ継続する。

augmentationは固定datasetのYOLO11s比較と同系統に揃え、`mosaic=0.30`、
`close_mosaic=20`、中程度のgeometry / photometric augmentationを使用します。
val / testへaugmentationは適用しません。

## 配置

```text
configs/yolo/yolo26_pose/
  face_pose_drive_v1_full.yaml
scripts_YOLO/yolo26_pose/
  pipeline.py
  run_yolo26_pose.py
models/yolo/yolo26_pose/pretrained/
  yolo26s-pose.pt
  yolo26m-pose.pt
runs/yolo/yolo26_pose/face_pose_drive_v1/
  yolo26s_pose_full/
  yolo26m_pose_full/
outputs/yolo/yolo26_pose/face_pose_drive_v1/
  preflight/
  yolo26s_pose_full/
  yolo26m_pose_full/
  comparison.{json,csv,md}
```

W&Bのローカル同期データは`runs/yolo/yolo26_pose/face_pose_drive_v1/_wandb/`へ集約します。

`models/`、`runs/`、`outputs/`は生成物なのでGit管理対象外です。設定・runner・testだけを
repositoryに保持します。

## 実行方法

リポジトリルートから実行します。

学習を開始しない互換性検査:

```bash
.venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py \
  --config configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml \
  --stage preflight \
  --models all
```

sをfull trainingする場合:

```bash
.venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py \
  --config configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml \
  --stage train \
  --models s \
  --device 0
```

学習コマンドは毎回preflightを実行し、W&B API keyと対象projectを確認してから学習へ進みます。
認証情報そのものをreportやlogへ保存することはありません。

mをfull trainingする場合:

```bash
.venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py \
  --config configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml \
  --stage train \
  --models m \
  --device 0
```

中断runは同じmodelだけを指定して再開します。

```bash
.venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py \
  --config configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml \
  --stage train \
  --models m \
  --device 0 \
  --resume
```

学習後の評価と比較:

```bash
.venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py \
  --config configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml \
  --stage evaluate \
  --models all \
  --device 0

.venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py \
  --config configs/yolo/yolo26_pose/face_pose_drive_v1_full.yaml \
  --stage compare
```

`--stage all`はpreflight後に実際のs/m学習を開始するため、準備だけを行う場合は使用しません。

## Preflight結果

2026-07-18にUltralytics 8.4.71 / PyTorch 2.12.1+cu126で確認しました。

| item | YOLO26s-pose | YOLO26m-pose |
| --- | ---: | ---: |
| official checkpoint | pass | pass |
| source keypoints | 17 x 3 | 17 x 3 |
| adapted face keypoints | 5 x 3 | 5 x 3 |
| end-to-end mode | pass | pass |
| adapted parameters | 10,579,532 | 23,535,244 |
| compatible state tensors | 795 / 879 | 855 / 939 |

dataset側も全3,090 image / label、5,033 objectを検査し、split間重複は全項目0でした。
詳細は `outputs/yolo/yolo26_pose/face_pose_drive_v1/preflight/preflight_report.json` に保存されます。

## W&B runs

- YOLO26s-pose: <https://wandb.ai/sato-waseda-university/face_detection/runs/e4b6f86bdebe7f35>
- YOLO26m-pose: <https://wandb.ai/sato-waseda-university/face_detection/runs/842e3dc1365fed60>

## 既知の制約

- 公式のCOCO pose値は17点human-poseの参考値であり、本datasetのface性能ではありません。
- 事前学習headのkeypoint固有層は形状が異なるため再初期化されます。backboneと互換層は転移します。
- dataset、model、Ultralytics versionが変わった場合はpreflightを再実行してください。

公式仕様: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)、
[Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/)
