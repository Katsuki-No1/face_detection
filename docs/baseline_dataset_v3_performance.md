# Dataset v3：FACE/楕円 + Query Refinementの性能

2026-10-11の保存済み評価を、2026-10-20にREADME向けに転記した記録です。今回の更新では学習・再推論・再評価は行っていません。
対象は現在の**local baseline**のV3学習済み2成分です。global/reference baselineの成績ではありません。
歴史的な評価フォルダ名は `candidate` のまま保持します。同じ出力内の `current_local_baseline` は当時の旧モデルを指すため、本ページには使用していません。

## モデルと評価条件

- 構成：`full-image FACE/ellipse → predicted-Face-crop DINOv3 Query Refinement`。HEAD検出器なし。
- FACE：`mal_alpha2_ellipse_dataset_v3_r3`、validation bbox mAP50:95で選択したepoch30 EMA。
- Query：`mal_alpha2_dataset_v3_query_refinement`、validation conditional semantic Macro F1@NME0.10で選択したepoch15 EMA。
- Dataset：`face_pose_condition_balanced_811_latest_sources_r3_20261009`。固定動画split、train 16,411枚。
- validationでFACE bbox F1最大化後、FACE閾値を固定してend-to-end 5点Macro F1@NME0.10最大化。各段で完全同値なら高い閾値を採用し、同一スコアの予測はまとめて扱う。
- 選択値：FACE `0.314453125`、全5点共通presence `0.59716796875`。testで再選択しない。
- 共通比較値：FACE `0.33`、presence `0.70`。選択値と別に掲載する。
- FACEはscore順greedyのbbox IoU ≥ 0.50一対一対応。重複を含む未対応予測をFP、未対応GTをFNとして数える。予測上限100/画像。
- NME：元画像座標の距離 / `1.20 × max(GT Head幅, GT Head高さ)`。GT Headは正規化・属性対応用であり推論ステージではない。
- semantic Macro F1：5スロットのF1の等重み平均。眼重視版：`[2,2,1,1,1]` の加重平均。眼重視値は閾値選択に使用しない。
- state 0=missing、1=occluded、2=visible。GT-presentはstate>0。空スロット・空条件はN/Aを保持し、Macro計算の重みを付け替えない。

| Split | 画像 | 全FACE GT | 属性評価対象FACE | 属性除外FACE | GT-present keypoints |
| --- | --- | --- | --- | --- | --- |
| val | 2022 | 3072 | 3040 | 32 | 11967 |
| test | 2060 | 2972 | 2940 | 32 | 11597 |

属性の除外理由はFace-to-Headリンク未解決。全FACE GTの検出分母は減らしていません。

## 基本性能

| Split | 閾値条件 | P | R | F1 | TP / FP / FN | 重複GT | 余分な重複予測 | 重複画像 | 重複なし画像率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | common_0_33 | 96.09% | 91.31% | 93.64% | 2805 / 114 / 267 | 3 | 3 | 3 | 99.85% |
| val | validation_selected | 95.82% | 91.73% | 93.73% | 2818 / 123 / 254 | 3 | 3 | 3 | 99.85% |
| test | common_0_33 | 95.45% | 91.79% | 93.58% | 2728 / 130 / 244 | 4 | 4 | 4 | 99.81% |
| test | validation_selected | 95.06% | 91.89% | 93.45% | 2731 / 142 / 241 | 5 | 5 | 5 | 99.76% |

`validation_selected` は上記のvalidation選択値、`common_0_33` は共通値です。

| Split | bbox mAP50:95 | AP50 | AP75 | bbox平均IoU（対応済み） | 楕円平均IoU（未検出=0） |
| --- | --- | --- | --- | --- | --- |
| val | 65.35% | 94.36% | 74.86% | 0.8395 | 0.6989 |
| test | 65.95% | 94.48% | 76.77% | 0.8445 | 0.7114 |

IoU列はvalidation選択値で集計。bbox IoUの母数はval 2,818／test 2,731対応、楕円IoUの母数はval 3,072／test 2,972 GTです。APはpycocotools.COCOeval、IoU 0.50:0.05:0.95、101 recall点、score floor 0、cap100による信頼度順評価です。

### Keypoint全体

| Split | 閾値条件 | 範囲 | GT点数 | Macro F1@.05 | Macro F1@.10 | 眼重視@.05 | 眼重視@.10 | 出力率 | 平均NME | NME点数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | common_0_33 | end-to-end | 11967 | 87.97% | 90.40% | 87.24% | 89.81% | 93.95% | 0.0209 | 11243 |
| val | common_0_33 | conditional | 11785 | 89.11% | 91.58% | 88.44% | 91.04% | 95.40% | 0.0209 | 11243 |
| val | validation_selected | end-to-end | 11967 | 87.84% | 90.57% | 87.11% | 90.03% | 95.43% | 0.0213 | 11420 |
| val | validation_selected | conditional | 11798 | 89.11% | 91.88% | 88.46% | 91.43% | 96.80% | 0.0213 | 11420 |
| test | common_0_33 | end-to-end | 11597 | 87.61% | 90.17% | 86.88% | 89.51% | 93.73% | 0.0210 | 10870 |
| test | common_0_33 | conditional | 11377 | 89.21% | 91.81% | 88.58% | 91.26% | 95.54% | 0.0210 | 10870 |
| test | validation_selected | end-to-end | 11597 | 87.37% | 90.10% | 86.61% | 89.45% | 95.29% | 0.0215 | 11051 |
| test | validation_selected | conditional | 11379 | 89.23% | 92.01% | 88.60% | 91.50% | 97.12% | 0.0215 | 11051 |

end-to-endはFACE見逃しと未対応予測の点FPを含む。conditionalはFACEが対応し、GT-present点のある対象に限定する。平均NMEは大きな誤差も含む出力済みGT-present点の件数重み平均で、見逃しを0として扱わない。

### 5点別（validation選択値、end-to-end）

| Split | Slot | GT点数 | GT上の出力点数 | 未出力 | P@.10 | R@.10 | F1@.10 | F1@.05 | 平均NME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | left_eye | 2625 | 2460 | 165 | 91.42% | 88.04% | 89.70% | 86.44% | 0.0251 |
| val | right_eye | 2394 | 2288 | 106 | 85.27% | 90.18% | 87.66% | 84.13% | 0.0255 |
| val | nose | 2463 | 2368 | 95 | 94.47% | 95.70% | 95.08% | 93.79% | 0.0133 |
| val | left_mouth | 2318 | 2212 | 106 | 91.72% | 90.34% | 91.02% | 87.98% | 0.0216 |
| val | right_mouth | 2167 | 2092 | 75 | 87.16% | 91.79% | 89.41% | 86.90% | 0.0209 |
| test | left_eye | 2533 | 2364 | 169 | 89.74% | 87.05% | 88.38% | 84.89% | 0.0261 |
| test | right_eye | 2336 | 2228 | 108 | 85.02% | 89.64% | 87.27% | 84.52% | 0.0245 |
| test | nose | 2430 | 2333 | 97 | 94.38% | 95.39% | 94.88% | 93.33% | 0.0153 |
| test | left_mouth | 2212 | 2112 | 100 | 90.70% | 90.42% | 90.56% | 87.30% | 0.0214 |
| test | right_mouth | 2086 | 2014 | 72 | 87.02% | 91.90% | 89.39% | 86.83% | 0.0200 |

## 遮蔽・横顔条件別

以下はvalidation選択値で固定した集計です。頑健性の判断にはvalidationを用い、testは同じ固定条件の報告として分離します。
FACEの条件別P/F1は背景FPの条件帰属が未定義のためN/A。keypointはGT対象集合のMacro F1で、FACE見逃しを含み、GTに対応しない予測の点FPは含みません。したがって全体end-to-end表とはFPの範囲が異なります。

### 遮蔽

GT face_occluded × state=1の点数（0〜5）の12条件。missing（state=0）は遮蔽点数に含めません。

#### val

| 条件 | FACE GT | 属性対象FACE | GT点数 | FACE Recall | 楕円IoU（未検出=0） | 点Macro F1@.10 | 眼重視F1@.10 | 平均NME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face_false_kp_0 | 1512 | 1487 | 6438 | 95.63% | 0.7658 | 93.31% | 92.91% | 0.0192 |
| face_false_kp_1 | 411 | 407 | 1664 | 97.08% | 0.7212 | 89.28% | 88.15% | 0.0218 |
| face_false_kp_2 | 382 | 382 | 1661 | 95.81% | 0.6905 | 93.19% | 92.60% | 0.0194 |
| face_false_kp_3 | 47 | 47 | 220 | 93.62% | 0.6391 | 83.53% | 83.28% | 0.0317 |
| face_false_kp_4 | 7 | 7 | 34 | 100.00% | 0.6714 | 89.17% | 90.22% | 0.0419 |
| face_false_kp_5 | 1 | 1 | 5 | 100.00% | 0.8324 | 100.00% | 100.00% | 0.0121 |
| face_true_kp_0 | 387 | 384 | 726 | 65.37% | 0.4573 | 81.87% | 81.98% | 0.0273 |
| face_true_kp_1 | 193 | 193 | 676 | 91.19% | 0.6674 | 83.17% | 83.25% | 0.0306 |
| face_true_kp_2 | 109 | 109 | 435 | 94.50% | 0.6611 | 86.65% | 85.97% | 0.0286 |
| face_true_kp_3 | 21 | 21 | 98 | 100.00% | 0.6702 | 84.22% | 85.16% | 0.0303 |
| face_true_kp_4 | 2 | 2 | 10 | 100.00% | 0.6997 | 66.67% | 66.67% | 0.0208 |
| face_true_kp_5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A |

#### test

| 条件 | FACE GT | 属性対象FACE | GT点数 | FACE Recall | 楕円IoU（未検出=0） | 点Macro F1@.10 | 眼重視F1@.10 | 平均NME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face_false_kp_0 | 1301 | 1276 | 5580 | 95.31% | 0.7740 | 93.08% | 92.67% | 0.0195 |
| face_false_kp_1 | 543 | 542 | 2192 | 97.97% | 0.7490 | 91.99% | 91.34% | 0.0196 |
| face_false_kp_2 | 306 | 306 | 1345 | 94.12% | 0.6919 | 91.87% | 90.92% | 0.0214 |
| face_false_kp_3 | 36 | 36 | 169 | 100.00% | 0.6706 | 94.17% | 93.90% | 0.0274 |
| face_false_kp_4 | 5 | 5 | 24 | 100.00% | 0.6863 | 89.78% | 88.25% | 0.0465 |
| face_false_kp_5 | 2 | 2 | 10 | 100.00% | 0.7068 | 60.00% | 57.14% | 0.0833 |
| face_true_kp_0 | 413 | 409 | 875 | 71.19% | 0.5218 | 83.36% | 82.84% | 0.0261 |
| face_true_kp_1 | 231 | 229 | 836 | 95.24% | 0.7059 | 85.34% | 84.71% | 0.0285 |
| face_true_kp_2 | 106 | 106 | 428 | 83.96% | 0.6035 | 82.55% | 81.44% | 0.0301 |
| face_true_kp_3 | 27 | 27 | 128 | 85.19% | 0.5920 | 76.77% | 76.70% | 0.0304 |
| face_true_kp_4 | 2 | 2 | 10 | 100.00% | 0.7107 | 86.67% | 90.48% | 0.0364 |
| face_true_kp_5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A |

### 横顔

GT FACE楕円の長軸/短軸比による区分。`3_to_4` は3以上4未満、`4_to_5` は4以上5未満、`5_to_inf` は5以上。

#### val

| 条件 | FACE GT | 属性対象FACE | GT点数 | FACE Recall | 楕円IoU（未検出=0） | 点Macro F1@.10 | 眼重視F1@.10 | 平均NME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3_to_4 | 238 | 233 | 740 | 86.97% | 0.5989 | 85.83% | 85.22% | 0.0187 |
| 4_to_5 | 84 | 83 | 186 | 76.19% | 0.4702 | 76.42% | 74.63% | 0.0230 |
| 5_to_inf | 129 | 128 | 173 | 65.89% | 0.3114 | 68.81% | 67.07% | 0.0197 |

#### test

| 条件 | FACE GT | 属性対象FACE | GT点数 | FACE Recall | 楕円IoU（未検出=0） | 点Macro F1@.10 | 眼重視F1@.10 | 平均NME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3_to_4 | 200 | 200 | 676 | 90.50% | 0.6249 | 87.50% | 86.48% | 0.0201 |
| 4_to_5 | 77 | 77 | 192 | 85.71% | 0.5323 | 79.92% | 78.15% | 0.0223 |
| 5_to_inf | 117 | 117 | 201 | 68.38% | 0.3442 | 69.29% | 68.68% | 0.0229 |

## 制限と過去の記録

- HEAD APとmatched-predicted-HEAD表はN/A（HEAD-free構成）。選択閾値での推論時間は未再計測のためN/A。
- 重複は上記基本表に掲載。画像端・重なり・小顔の診断、state別結果、点別NME0.05/0.10の全カウンタ、同一閾値の失敗例は元の `candidate/{val,test}/report.md`、`detailed_metrics.json`、`validation_selected_keypoints.csv`、`failure_review.json` と `failure_review/` に保存されています。これらの画像・予測キャッシュはGitには同梱していません。
- 元レポートの画像端・重なり・小顔区分は幾何学的診断です。cross-face誤帰属の正式基準は未確定であり、幾何学的重なりを誤帰属率と解釈しません。反射顔はGTラベル不足のため未評価です。
- この更新は保存済み成績の文書化です。同梱registryと実行例は旧モデルの保存設定であり、V3用設定やランタイムの移植・新規動作検証は行っていません。
- 旧モデル・旧splitの結果は[過去の性能記録](baseline_performance.md)に保持しています。

## 一次資料とSHA-256

以下のパスは評価を行った開発環境でのリポジトリ相対パスです。重み・データセット・生成キャッシュはGitに含めていません。

- FACE checkpoint: `outputs/experiments/mal_alpha2_ellipse_dataset_v3_r3/checkpoints/best_val_map.pt`
  - SHA-256: `5e17564ccf772a37b5c825614635ed0d93344e65fdb4757740a2ed782c9ba2d3`
- Query checkpoint: `outputs/landmark_detection/mal_alpha2_dataset_v3_query_refinement/query_refinement/checkpoints/best_conditional_semantic_macro_f1.pt`
  - SHA-256: `11b430b595e56395a14c266cb364bd50848b1a51b5b689548eb6749de0a23856`
- FACE config: `configs/dinov3_deimv2/mal_alpha2_ellipse_dataset_v3_r3.yaml`
  - SHA-256: `3d0af113b19360e26b3f959f146e68b344758f96ae5cbb6856ba410632118d13`

- `outputs/evaluations/dataset_v3_f1_selected_20261011/success.json`
  - SHA-256: `ef62d75ed9fe29e7b91a12893236e883e6bb677dd9c298c40807ecf0b0d48193`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/validation_lock.json`
  - SHA-256: `1ffd0636194d1c110760bee1a0d6916abe66f026330d0a0ade78c4e0a2472000`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/val/baseline_aligned/metrics.json`
  - SHA-256: `9ad33ba359a0023990f0816792ba34c15ea56bcf829e7eed29958749b2e64278`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/val/baseline_aligned/success.json`
  - SHA-256: `783f420142cce530af24ccb597e246d9f055fd7f784884ea7135efe2b8c58216`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/val/detailed_metrics.json`
  - SHA-256: `858c2d30d3a76abb69297220d18b7bde6a263c293150cddafc93cc3612f4c070`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/test/baseline_aligned/metrics.json`
  - SHA-256: `f86de497544735284ee084afd447b3b2744a686fdca817ff19184e0962d78b95`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/test/baseline_aligned/success.json`
  - SHA-256: `8770e15a8311c01f8789f673dbf317687c779ee6cf52332f1fbe16df69020930`
- `outputs/evaluations/dataset_v3_f1_selected_20261011/candidate/test/detailed_metrics.json`
  - SHA-256: `9f10ea4592209dc5d422656c20df599b57205ae0444b09d4226b87794e792be6`
- val native annotation SHA-256: `7728e75be9a17d0f91797c1375db9eee86279b7ae60b67ae0a33251735034016`
- test native annotation SHA-256: `b606e6d672451bb6963aee11b8cd83b683adf9007aecbf49ba4f3297fcdc22d6`

保存済みsuccessはpassedで、上記8件のレポート・閾値lockのSHA-256を今回照合しました。val/testのモデル識別、閾値の一致、FACEのTP/FP/FNとF1、5スロットMacro F1と固定眼重みの整合を確認しています。
