# Baselineの性能

対象は `mal_alpha2_ellipse_from_existing_wider` の **FACE/楕円検出部のみ**。
HEAD-freeの一式 `全画像FACE/楕円 → Face-crop DINOv3 Query Refinement` は、
registry上で組み合わせ全体の推論・評価が `not_run` のため、keypoint性能を含む一式の数値は未掲載。

2026-10-09の保存済み評価から対象モデルの値だけを掲載した。今回の整理では再推論・再評価をしていない。
固定データセットは `face_pose_drive_v2_balanced_811`（val 530枚、test 528枚）。
最新ソースr3のval 2,022枚／test 2,060枚の評価とは異なる。

## 基本性能

| Split | 閾値 | FACE GT | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---|
| val | 0.33 | 840 | 93.53% | 91.19% | 92.34% | 766 / 53 / 74 |
| val | 0.2666015625 | 840 | 90.33% | 92.26% | 91.28% | 775 / 83 / 65 |
| test | 0.33 | 769 | 95.74% | 93.50% | 94.61% | 719 / 32 / 50 |
| test | 0.2666015625 | 769 | 93.77% | 94.02% | 93.90% | 723 / 48 / 46 |

FACE bbox IoU ≥ 0.50で画像内一対一マッチ。未対応予測は重複を含めすべてFP、未対応GTはFN。
予測上限100。0.33は共通運用点、0.2666015625はvalidationでPrecision > 0.90を満たすRecall最大の閾値。
testへ同じ閾値を固定適用し、testに基づく再選択は行っていない。

| Split | bbox mAP50:95 | bbox AP50 | bbox AP75 | ellipse mAP50:95 | ellipse AP50 |
|---|---:|---:|---:|---:|---:|
| val | 59.54% | 93.94% | 67.98% | 48.34% | 88.40% |
| test | 61.03% | 95.55% | 66.72% | 51.47% | 91.11% |

## 遮蔽条件別

採用済みの条件はGT `face_occluded` × GT keypoint state=1の個数。
保存済みの上記レポートには、この12条件に一致するFACE bbox性能表がないため未集計とする。
異なる旧区分やHead単位の診断値で代用しない。N/Aは性能ゼロや成功を意味しない。

| face_occluded | state=1の個数 | val件数 / FACE性能 | test件数 / FACE性能 |
|---|---:|---|---|
| false | 0 | N/A（未集計） | N/A（未集計） |
| false | 1 | N/A（未集計） | N/A（未集計） |
| false | 2 | N/A（未集計） | N/A（未集計） |
| false | 3 | N/A（未集計） | N/A（未集計） |
| false | 4 | N/A（未集計） | N/A（未集計） |
| false | 5 | N/A（未集計） | N/A（未集計） |
| true | 0 | N/A（未集計） | N/A（未集計） |
| true | 1 | N/A（未集計） | N/A（未集計） |
| true | 2 | N/A（未集計） | N/A（未集計） |
| true | 3 | N/A（未集計） | N/A（未集計） |
| true | 4 | N/A（未集計） | N/A（未集計） |
| true | 5 | N/A（未集計） | N/A（未集計） |

## 横顔条件別

ratioはGT FACE楕円の長軸／短軸。次の区分に一致する性能表も保存済みレポートにはなく、未集計。
遮蔽・横顔の頑健性はvalidationで判断する。testは固定条件での報告に限定する。

| GT楕円のratio | val件数 / FACE性能 | test件数 / FACE性能 |
|---|---|---|
| 3 ≤ ratio < 4 | N/A（未集計） | N/A（未集計） |
| 4 ≤ ratio < 5 | N/A（未集計） | N/A（未集計） |
| ratio ≥ 5 | N/A（未集計） | N/A（未集計） |

## モデルと出典

- Checkpoint: `outputs/experiments/mal_alpha2_ellipse_from_existing_wider/checkpoints/best_val_map.pt`
- Checkpoint SHA-256: `3bd6e40517bc6f6e276c47e1dfdf639173e8b9c04d32486adbb6f4b407f006a0`
- validation bbox mAP50:95で選択したepoch45 EMA。入力960、seed0、effective batch16。
- Keypoint成分はA1/A2版DINOv3 Query Refinementを保持。重みの識別情報は[registry](../configs/model_registry/local_baseline.yaml)。
- 以下はローカル一次資料の識別情報。重み・データ・予測キャッシュはGitに含めない。

### val

- FACE GT: 840。Head GT: 882（FACE主表の分母とは別）。
- Annotation SHA-256: `0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0`
- FACE adapter SHA-256: `906fd215ed30d056cc9dfe7666143c796e50f6810aa5b5a50cbffe28bf04a2ab`
- Metrics: `outputs/experiments/mal_alpha2_ellipse_from_existing_wider/evaluation/val/baseline_aligned/metrics.json`
- Metrics SHA-256: `0ca2e6d9b24a395eca960cf1175d99d8600d9c491918b0ac0f0b3503f956ec42`

### test

- FACE GT: 769。Head GT: 806（FACE主表の分母とは別）。
- Annotation SHA-256: `bd32bd1591686947bfcc79cbbcd79e1059bba1569e8bd06127d3327f2b776487`
- FACE adapter SHA-256: `bbab427fdddbfe2d4c0d669ea8d879db58c6f3bab13d6653738b060cecb3e760`
- Metrics: `outputs/experiments/mal_alpha2_ellipse_from_existing_wider/evaluation/test/baseline_aligned/metrics.json`
- Metrics SHA-256: `202d7c76f9efd668bfbd0554623f8891ae3be1e3235858e2f61f74c12c7424f5`
