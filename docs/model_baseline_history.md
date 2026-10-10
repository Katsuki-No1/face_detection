# Face detection / pose model baseline history

最終更新: 2026-10-09

## 目的と比較上の注意

この文書は、容量整理で学習生成物や推論結果を削除した後も、開発の道筋、主要モデル名、代表性能、baseline の変遷を追跡できるようにするための要約である。詳細な条件と履歴は W&B、各評価レポート、Git 履歴を正とする。

- project-wide/global baseline は `reference_model/face_dino_v2_distribution` の配布モデルであり、全体比較の固定基準である。
- current local baseline は **顔検出＋keypoint検出の一式**である。構成は `全画像FACE／楕円（MAL α=2） → Face crop DINOv3 Query Refinement（既存A1/A2版）`。`mal_alpha2_ellipse_from_existing_wider`はその顔検出部であり、顔検出部単独をbaseline全体と呼ばない。正規の定義は `configs/model_registry/local_baseline.yaml` の `baseline_pipeline`。新しい組み合わせ全体の評価は未実施で、既存のFACE比較値は顔検出部の証拠に限定する。
- 今後のlocalモデル間のFACE主比較は、予測bboxとGT FACE bboxをIoU 0.50以上で一対一対応させ、未対応予測（同一GTへの余剰重複を含む）をFPとする通常の検出Precision/Recall/F1および重複件数を用いる。Head単位のFace-presence表はdownstream診断とglobal/reference比較専用とし、local bbox表との数値を混在させない。
- global/referenceモデルはlocked reportだけが正確な比較源であるため、global/referenceとの比較時に限り従来のbaseline-aligned Head単位表を使用する。reference reportとlocal bbox-IoU-0.50主表の間では差分を計算しない。
- 検証データで閾値や checkpoint を選び、test 結果で設定を選ばない。
- 下表は実験時期によってデータ split、入力解像度、評価契約が異なるため、異なる節の数値を単純に横比較しない。

## 2026-10-09 既存WIDER → 目的データα=2へのlocal baseline更新

ユーザーの明示指示「ではα=2をbaselineとして更新しましょう」により更新。
同日のユーザー補足「顔検出とkeypoint検出はセットでbaselineになる」に従い、
α=2への更新はbaselineの顔検出部の更新として記載する。keypoint検出部は既存A1/A2版の
DINOv3 Query Refinementを引き継ぎ、baseline全体はHEAD-freeの順序付き2段構成とする。
顔検出部自体はnon-cascade full-image FACE/ellipseであり、以下のcheckpoint・指標はこの部品のみのもの。
新しい組み合わせ全体の推論・評価は未実施。旧A1/A2＋Query Refinementの実行構成と過去の評価値は保持する。
2026-10-09の5分動画は「新顔検出部のみ対旧baseline一式」の部分比較であり、
baseline一式同士の比較を完了した成果物ではない。説明の訂正で過去の動画にkeypointが存在したことにはしない。

- config: `configs/dinov3_deimv2/mal_alpha2_ellipse_from_existing_wider.yaml`
- checkpoint: `outputs/experiments/mal_alpha2_ellipse_from_existing_wider/checkpoints/best_val_map.pt`
- SHA-256: `3bd6e40517bc6f6e276c47e1dfdf639173e8b9c04d32486adbb6f4b407f006a0`
- validation bbox mAPで選んだepoch45 EMA。checkpoint・閾値の再選択は行っていない。
- 共通FACE閾値0.33、validation限定Precision > 0.90でRecall最大の閾値0.2666015625、予測上限100。
- 既存WIDER重みからseed0、960、microbatch8×蓄積2、occlusion適用確率0.30でfine-tune。65 epochで正常early stop。
- 固定val 530画像、test 528画像。4 gate・本学習・共通val/test成功markerと重み/config/annotation hashを再認証。

| bbox IoU ≥ 0.50・共通0.33 | 旧A1/A2検出部 | 新α=2 |
|---|---:|---:|
| validation Precision | 90.32% | 93.53% |
| validation Recall | 91.07% | 91.19% |
| validation F1 | 90.69% | 92.34% |
| test Precision（報告のみ） | 93.10% | 95.74% |
| test Recall（報告のみ） | 92.98% | 93.50% |
| test F1（報告のみ） | 93.04% | 94.61% |

位置・形状の回帰監視も継続する。test bbox AP75は69.34%→66.72%、test楕円AP80/85/90は改善だが、
val楕円AP85/90は低下している。αだけの効果を切り分けた実験ではない。
元の比較表は `outputs/reports/mal_alpha2_existing_wider_baseline_comparison_20261009/`。
更新後の `tools/eval_codino_official.py` は新registryから参照し、登録済み予測とvalidation閾値を再利用する。
比較先の異なる既存出力先への上書きは拒否する。歴史的な固定モデル比較やcrop-only対照を新baselineの比較と混同しない。

## 2026-09-14 遮蔽augmentation版へのlocal baseline更新（旧baseline）

ユーザー指定により、旧local baselineと同じDEIMv2-L FACE/ellipse + Query Refinement構成にlandmark-guided occlusion augmentationを加えたモデルを新local baselineとした。これはfull-image FACE/ellipse検出器であり、HEAD検出器は含まない。baseline-aligned評価内のHEAD値は共通の外部Co-DINO HEAD予測によるもので、今回の更新対象ではない。

augmentationはdetectorとQuery Refinementの双方で確率0.30、通常遮蔽とlandmark-anchor遮蔽（A1/A2）を概ね等分し、対象顔との重なりを0.10–0.40に制約した。detectorはtrain 4,237枚、seed 0、microbatch 4、gradient accumulation 4（effective batch 16）、最大70 epochで学習し、bestはvalidation ellipse mAP50-95で選択した。Query Refinementはtrain 6,529 instance、seed 42、batch 32、gradient accumulation 2、最大150 epoch、毎epoch validationで学習し、epoch 103のbest NME 0.037826を採用、epoch 128でearly stoppingした。実行時の契約を維持した既存runであり、後から導入されたprospective training defaultsを遡及適用していない。

固定split・同一annotation SHA・FACE threshold 0.33・point threshold 0.70で比較した。testは設定選択に使用していない。

| Test metric | Previous local baseline | Occlusion-augmented | Delta |
|---|---:|---:|---:|
| FACE recall | 94.78% | 95.95% | +1.17 pp |
| No-face recall | 89.74% | 87.18% | -2.56 pp |
| Ellipse mean IoU | 70.75% | 71.68% | +0.93 pp |
| Ellipse P10 IoU | 42.20% | 47.46% | +5.26 pp |
| Keypoint Macro F1 @ NME 0.10 | 89.74% | 90.04% | +0.30 pp |
| Occluded-point NME @ 0.10 | 0.02784 | 0.02641 | -0.00144 |
| Conditional occluded F1 | 64.86% | 65.44% | +0.58 pp |
| Occluded end-to-end F1 | 58.33% | 57.79% | -0.54 pp |
| Occluded-only keypoint Macro F1 @ 0.10 | 34.80% | 30.39% | -4.42 pp |

主な改善はFACE recall、低分位のellipse IoU、遮蔽点の位置誤差である。一方、no-face recall、occluded-only keypoint Macro F1、遮蔽end-to-end F1には回帰があり、次のlocal architecture比較では明示的な監視対象とする。完全なval/test比較は`outputs/evaluations/local_baseline_occlusion_augmentation_comparison_20260914/`を正とする。

## 1. 初期 detector / pose baseline

| 系列 | 代表設定 | 代表性能 | 位置付け |
|---|---|---:|---|
| SCRFD + MediaPipe | Face threshold 0.09 | F1 0.5938 / Precision 0.5846 / Recall 0.6032 | 初期の二段構成 |
| YOLO11n-pose | 初期比較 | F1 0.7778 | 軽量候補 |
| YOLO11m-pose | 初期比較 | F1 0.7593 | サイズ拡大の比較 |
| YOLO11s-pose | Face threshold 0.51 | F1 0.8073 / Precision 0.9565 / Recall 0.6984 | 一時的な local baseline |

## 2. YOLO11 データ・augmentation 開発

| 実験 | Box mAP50-95 | Pose mAP50-95 | Face F1 | NME / PCK@0.05 / PCK@0.10 |
|---|---:|---:|---:|---:|
| old reproduction | 0.4381 | 0.7476 | 0.8073 | 0.0554 / 0.6220 / 0.8804 |
| modified_50 | 0.4737 | 0.7455 | 0.8148 | N/A |
| modified_100 | 0.5125 | 0.7525 | 0.7500 | N/A |
| high resolution 1280 | 0.4984 | 0.7703 | 0.7778 | 0.0489 / N/A / N/A |
| hard mining + keypoint-safe augmentation | N/A | N/A | 0.8257 | 0.0640 / 0.5467 / N/A |

Hard mining 系は Face F1 は改善したが、NME と PCK@0.05 が悪化したため baseline には採用しなかった。augmentation 強度比較では、例として E0 が box recall 0.7143 / box mAP 0.4642 / pose recall 0.7302 / pose mAP 0.7094、E2 が 0.7619 / 0.4373 / 0.7778 / 0.7403、E3 が 0.7460 / 0.4515 / 0.7778 / 0.7460 だった。

## 3. RTMDet / RTMPose と二段構成

| 系列 | 代表性能 | 結論 |
|---|---:|---|
| RTMDet + RTMPose two-stage | Face threshold 0.50、F1 0.6931、Precision 0.9211、Recall 0.5556、NME 0.0740、PCK@0.05/0.10 0.4762/0.7619 | 初期二段構成の比較 |
| RTM data ablation modified_100 | detector F1 0.8367、Face recall 0.7460、NME 0.0324 | データ改善の有効性を確認 |
| FDP | NME 0.0489、PCK@0.05 0.8209、missed 20 | stage 比較 |
| HFDP | NME 0.0458、PCK@0.05 0.8482、missed 23 | 精度は改善、miss は増加 |

固定 3090-frame benchmark では、DINOv3+DEIMv2 freeze が mAP 0.4562 / AP50 0.8440 / F1 0.8170、unfreeze が 0.4760 / 0.8791 / 0.8654、Legacy E1 が 0.2560 / 0.6992 / 0.6871、YOLO11s Detect が 0.4789 / 0.8489 / 0.8556、YOLO11s Pose が 0.5332 / 0.8850 / 0.8877（pose mAP 0.8327）だった。RTMDet-S-HF の別契約での test は Head mAP 0.796 / AP50 0.974 / AP75 0.871、Face mAP 0.525 / AP50 0.900 / AP75 0.564 だった。

## 4. DINOv3 / DEIMv2 と landmark query refinement

DINOv3+DEIMv2 の代表比較では、freeze が mAP 0.5318 / AP50 0.9141 / AP75 0.5372 / threshold 0.35 / F1 0.8935、unfreeze が 0.5914 / 0.9480 / 0.6533 / threshold 0.48 / F1 0.9310 だった。

landmark branch は ViTPose から Landmark Query、Query Refinement へ進んだ。

| landmark model | Validation NME | 代表 test 結果 |
|---|---:|---|
| ViTPose | 0.035038 | 初期 landmark baseline |
| Landmark Query | 0.024031 | query-based branch の改善 |
| Query Refinement | 0.022516 | Condition E: NME 0.022853、PCK@0.05 0.918006、PCK@0.10 0.962283、coverage 0.941631、end-to-end 0.864422 |

この結果を受け、local baseline は `DEIMv2-L FACE/ellipse -> Query Refinement` へ移行した。

## 5. 旧local baseline と三段モデル比較

以下は同一の baseline-aligned comparison table から抜粋した代表値である。

| Metric | Global reference | Local baseline: DEIMv2 + Query Refinement | New 3-stage |
|---|---:|---:|---:|
| HEAD AP | 73.26% | N/A | 73.52% |
| FACE precision | 98.91% | 99.45% | 98.11% |
| FACE recall | 94.91% | 94.78% | 94.91% |
| No-face recall | 79.49% | 89.74% | 64.10% |
| Ellipse mean IoU | 69.49% | 70.75% | 63.89% |
| Ellipse center NME | 0.05091 | 0.03820 | 0.05712 |
| Keypoint semantic Macro F1 | 0.8871 | 0.8974 | 0.8791 |
| Matched keypoint NME | 0.01626 | 0.01219 | 0.02342 |
| Occluded F1 | 0.5732 | 0.5833 | 0.4798 |

三段モデルは HEAD AP がわずかに改善した一方、Face/no-face、ellipse、keypoint、occlusion の主要指標で local baseline を上回らず、baseline には採用しなかった。global reference と local baseline の役割は引き続き分離する。

## 2026-08-23 容量整理で削除する生成物

次の生成物は、上記の開発経路と代表性能をこの文書および既存レポートで保持したうえで削除対象とした。ソースコード、設定、固定 annotation、現行 global/local baseline、実行中 production run は対象外である。

- Git 参照外オブジェクト
- 正常完了済み run の冗長な `epoch_*` checkpoint（`best` と `last` は保持）
- 初期 YOLO11 / RTM / SCRFD の学習・推論生成物とルート直下の旧 weight コピー
- 成功済み retry に対応する `.previous_*`、invalid、stopped、failed attempt
- `normal_performance_10m_20260718` と active-learning round 1 の再生成可能な動画、画像、prediction cache

## 主な参照先

- `README.md`
- `docs/face_pose_drive_v1_benchmark_20260715.md`
- `reference_model/face_dino_v2_distribution/`
- `outputs/keypoint_comparison2/` 配下の baseline-aligned evaluation reports
- W&B の各 run（学習条件・metric history の durable store）
