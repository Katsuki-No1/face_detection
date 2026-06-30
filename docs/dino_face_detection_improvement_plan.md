# DINO Backbone Face Detection Improvement Plan

このメモは、DINO を frozen backbone として用いる顔 bbox + 5 keypoint 検出を改善するための実験方針です。対象はこのリポジトリの `DINO -> adapter/FPN -> head` 系で、YOLO11 Pose や RTMDet + RTMPose は比較基準および上限確認として扱います。

## 結論

次に最優先で行うべきことは、DINO と head のデータ形式そのものの再設計ではありません。現状の実装では、DINOv3 の patch token を `[B, C, H, W]` の feature map に変換し、YOLO/CVAT/metadata 由来の bbox、ellipse、keypoint、visibility mask を学習 target に渡す経路は実験可能な水準まで到達しています。

ただし「完全に終了」とは言い切らない方が安全です。データ形式は基礎実装としては終了、次は head 比較と失敗分析を進める段階です。比較中に精度劣化の原因が target 形式、resize、visibility、flip、ellipse/bbox 変換にあると判明した箇所だけを直す、という進め方がよいです。

現状の実験値からは、DINO-SCRFD より DINO-CenterFace-ellipse が有望です。特に real DINOv3 の CenterFace ellipse 系は、test 30 images / 63 instances で F1 が YOLO11s Pose baseline に近い水準まで来ています。一方で keypoint NME は YOLO より悪く、keypoint head はまだ改善余地が大きいです。

## 1. DINO と head のデータ形式最適化は次に行うべきか

基礎的な接続は終わっています。

- `src/models/dino_backbone.py`: DINOv3 の `forward_features` と `x_norm_patchtokens` を受け取り、patch token を feature map に変換する実装があります。`patch_size=16`、`num_prefix_tokens=0`、`allow_non_divisible=false` も設定されています。
- `src/data/face_keypoint_dataset.py`: YOLO Pose label、COCO-like、CVAT/metadata を読み、`boxes`、`ellipses`、`keypoints`、`keypoint_mask` に統一しています。CVAT ellipse から rotation-aware bbox も作れます。
- `src/data/ellipse_augmentation.py`: bbox、ellipse、keypoint を同時に変換する augmentation が入っています。
- `configs/dino_scrfd_aug_ablation/base.yaml` と `configs/dino_centerface_ellipse.yaml`: real DINOv3 weight、local hub repo、frozen backbone、adapter/FPN/head の設定があります。

ただし、以下は「精度改善の途中で確認すべき残課題」です。

- 入力画像を正方形 resize しているため、アスペクト比歪みが bbox/keypoint に効いている可能性があります。YOLO と公平比較するなら letterbox resize 版を追加して比較します。
- 評価 threshold は test sweep で選ばれている出力が多いため、val で threshold を決めて test に固定する運用にします。
- README の DINO 段階説明は Phase 1 の記述が残っており、現在の Phase 2/real DINOv3 実験とはずれています。
- `runs/dino_scrfd_face/dino_scrfd_896_b4_e100_20260628` は dummy DINO backbone の注記があるため、DINO backbone の性能判断には使わず、実装 smoke/過去比較として扱います。
- keypoint の左右反転、欠損 keypoint mask、metadata 由来 target と YOLO label target の一致は、head 比較の前に小さな検証セットで再確認します。

次の作業順としては、データ形式を広く作り直すのではなく、同一 dataset、同一 split、同一 input size、同一 threshold 選定ルールで head を比較できる状態に固定することを優先します。

## 現状の主な実験値

| モデル | 条件 | Precision | Recall | F1 | bbox mAP50-95 | keypoint NME | メモ |
|---|---:|---:|---:|---:|---:|---:|---|
| DINO-SCRFD | real DINOv3, E0 baseline | 0.870 | 0.635 | 0.734 | 未出力 | 0.091 | `outputs_experiment/dino_scrfd_aug_ablation_real_dinov3_20260630_eval/E0_baseline/summary.json` |
| DINO-CenterFace-ellipse | E2 geometry only | 0.947 | 0.857 | 0.900 | 0.384 | 0.104 | F1 最高 |
| DINO-CenterFace-ellipse | E4 mosaic geometry | 0.945 | 0.825 | 0.881 | 0.453 | 0.095 | bbox AP と keypoint のバランスが良い |
| DINO-CenterFace-ellipse | E7 all | 0.871 | 0.857 | 0.864 | 0.465 | 0.095 | bbox mAP50-95 と keypoint PCK が良い |
| YOLO11s Pose | modified_100 baseline | 0.964 | 0.841 | 0.898 | 0.510 | 0.057 | keypoint はまだ強い比較基準 |

この表からの読み取りは次の通りです。

- DINO-CenterFace-ellipse は、bbox detection F1 では YOLO11s に近いところまで到達しています。
- DINO-SCRFD は real DINOv3 でも CenterFace ellipse より低く、現状の本命ではありません。
- keypoint NME は YOLO11s が明確に良いため、DINO 系は keypoint head または crop-based keypoint refinement の比較が必要です。

## 2. 比較すべき head モデル

まずは同時出力型を主戦場にし、二段型は keypoint 上限確認として比較します。

同時出力型を優先する理由は、顔検出では bbox と 5 keypoint が強く結びついており、CenterFace や RetinaFace でも detection と alignment の joint learning が有効な設計として使われているためです。また、このリポジトリの現在の結果でも DINO-CenterFace-ellipse が最も有望です。

二段型を比較する理由は、DINO 系の keypoint NME が YOLO11s より悪いからです。bbox が取れているのに keypoint だけ悪いのか、bbox の粗さが keypoint に伝播しているのかを切り分けるには、GT bbox crop と predicted bbox crop の両方で crop-based keypoint model を評価するのが早いです。

### 優先度 A: DINO-CenterFace-ellipse

これは現在の本命です。CenterFace は顔中心 heatmap から bbox と landmark を同時に出す anchor-free 系です。このリポジトリの実装では、顔 ellipse を中心 target にし、`center_logits`、`ellipse_reg`、`bbox_reg`、`kpt_reg` を同時に出します。

根拠:

- CVAT の face annotation が ellipse 由来なので、bbox だけより annotation 形状に合っています。
- real DINOv3 実験で F1 0.900 が出ており、DINO-SCRFD より良いです。
- CenterFace の思想は「顔を点として扱い、bbox と 5 landmarks を同時に推定する」ため、現状の 5 keypoint タスクと相性が良いです。

次に比較すべき variant:

- `bbox_source=bbox` と `bbox_source=ellipse` の固定比較。
- E2 geometry、E4 mosaic geometry、E7 all の再評価。F1、bbox mAP、keypoint NME/PCK のどれを主指標にするかで winner が変わるためです。
- ellipse IoU NMS。現状は entry point はありますが、実体は bbox IoU NMS です。
- keypoint loss を SmoothL1 から Wing Loss / Adaptive Wing Loss 系、または bbox scale 正規化 loss に変える。

### 優先度 A: DINO-SCRFD-KPS

これは現在の baseline head として残します。SCRFD は sample redistribution と computation redistribution を重視する高効率顔検出器で、5 keypoint 付き variant も一般的です。このリポジトリでは `cls_logits`、`center_offset`、`bbox_reg`、`kpt_reg` を出す anchor-free 版として実装されています。

根拠:

- DINO-CenterFace-ellipse との比較で、中心 heatmap型と SCRFD型の差を見られます。
- 現在の real DINOv3 baseline は F1 0.734 で低いため、改善候補というより比較基準として必要です。
- false positive が多い低 threshold 領域が見えるため、score calibration、quality branch、assigner の改善効果を測りやすいです。

改善して比較する variant:

- center assignment を 1 cell 固定から radius/center sampling にする。
- bbox loss を SmoothL1 から IoU/GIoU/DIoU 系にする。
- score を `cls * quality` にするための IoU quality または centerness branch を足す。
- size range を dataset の face size 分布に合わせて再調整する。

### 優先度 B: DINO-RetinaFace-style head

RetinaFace は single-stage dense face localization で、bbox、5 landmarks、追加 supervision を組み合わせる設計です。DINO backbone の上に、anchor/prior box を持つ RetinaFace 風 head を載せると、CenterFace/SCRFD と異なる帰納バイアスを比較できます。

根拠:

- 小顔や hard face で landmark supervision が detection を助けるという設計思想が、今回の「小顔、混雑、欠損 keypoint」課題に合います。
- anchor-free head だけで詰まった場合、anchor/prior による scale prior が効く可能性があります。

実装コストは CenterFace/SCRFD より高いので、先に CenterFace ellipse と二段 keypoint oracle を終えてからで十分です。

### 優先度 B: DINO-FCOS/CenterNet-style dense head

`src/dino_patch_fpn/head.py` には、class、bbox、centerness、keypoints、optional ellipse を出す scaffold があります。これを実験可能にするなら、FCOS/CenterNet 風の target assignment と loss を整備し、CenterFace ellipse と比較します。

根拠:

- centerness や quality score を明示的に持てるため、DINO-SCRFD の false positive/score calibration 問題を切り分けやすいです。
- class を `head` と `face` に分ける設計も試せます。頭部 bbox を context として使えば、顔が部分的に隠れている場合の recall 改善が期待できます。

### 優先度 A: 二段型 keypoint model

二段型は「本命 head」ではなく、keypoint 精度の上限確認に使います。比較セットは次がよいです。

| 検出器 | keypoint model | 目的 |
|---|---|---|
| GT bbox | RTMPose または PFLD | keypoint model 単体の上限を測る |
| DINO-CenterFace bbox | RTMPose または PFLD | DINO detector の bbox 誤差込み keypoint 精度を見る |
| YOLO11s bbox | RTMPose または PFLD | 既存強baseline bbox上での keypoint 上限を見る |
| RTMDet bbox | RTMPose | OpenMMLab二段構成の比較基準 |

候補名:

- RTMPose: top-down pose estimation。既に `script/script_experiment/rtmdet_rtmpose_experiment.py` 系の資産があります。顔 5 点用に fine-tune し、bbox crop 内で keypoint を出す比較に向いています。
- PFLD: face landmark 専用で軽量です。5点だけなら過剰に大きい heatmap model より有利な可能性があります。

判断基準:

- keypoint NME を最優先するなら、二段型が勝つ可能性があります。
- latency と実装単純性を重視するなら、同時出力型を改善する方がよいです。
- bbox miss は二段型では救えないため、detection recall は同時出力型または detector 側で先に改善します。

## 3. bbox 精度を上げる一般的な方法

### データと評価

- val で threshold を決め、test は一度だけ使います。test sweep の best threshold は研究メモとしては便利ですが、改善判断には楽観的です。
- false positive と false negative を `small_face`、`occluded_face`、`edge_face`、`crowded_image`、`low_quality_face` で分け、条件別 recall を主指標に入れます。
- hard negative mining を行います。高 score false positive を次の train set に negative として追加します。
- frame 抽出由来の近傍フレームが train/test にまたがらないよう、動画単位またはシーン単位で split を固定します。
- annotation の欠損、ellipse/bbox 不一致、左右 keypoint の入れ替わりを audit します。

### 入力と augmentation

- square resize と letterbox resize を比較します。keypoint まで見るなら letterbox の方が安全な可能性があります。
- small face 用に input size 896/1280、multi-scale train、random scale crop、mosaic の強さを比較します。
- 幾何 augmentation は過剰だと ellipse/keypoint を壊しやすいため、E2 geometry が強い理由を失敗画像で確認します。
- dark/blur/noise は実データ分布に合わせて軽く使い、強すぎる photometric augmentation は避けます。

### head と loss

- bbox regression を SmoothL1 だけでなく IoU/GIoU/DIoU loss で比較します。
- score quality branch を追加します。classification score と localization quality がずれていると NMS/threshold が不安定になります。
- CenterFace 系は heatmap radius、positive cell 数、level assignment の size range を dataset に合わせます。
- 小顔のために FPN の最も細かい level を強化します。DINOv3 ViT-B/16 では 896 input が 56x56 patch grid になるため、さらに細かい local detail は adapter/FPN 側で補う必要があります。
- frozen backbone で head が収束した後、DINO の最後 1から2 block だけ小さい learning rate で unfreeze します。まずは adapter/LoRA 的な軽い更新から試します。

## 4. keypoint 精度を上げる一般的な方法

### まず切り分けること

- GT bbox crop で keypoint model を評価し、bbox 誤差なしの keypoint 上限を測ります。
- predicted bbox crop で同じ keypoint model を評価し、bbox 誤差による劣化量を測ります。
- DINO 同時出力 head の keypoint NME と比較し、問題が DINO feature、keypoint loss、bbox、annotation noise のどこにあるかを切り分けます。

### loss と表現

- keypoint loss を顔サイズで正規化します。大きい顔が loss を支配しないようにします。
- SmoothL1 に加えて Wing Loss / Adaptive Wing Loss を比較します。顔 landmark では小から中程度の誤差を強く詰める設計が有効なことがあります。
- heatmap + offset 方式を検討します。座標直接回帰より、目や口角の局所的なピークを学習しやすい場合があります。
- keypoint visibility/confidence を明示的に出します。現在は mask で loss を無視していますが、推論時の keypoint 信頼度はありません。

### データ

- visible keypoint が多いサンプル、横顔、隠れ、暗所、小顔をタグ別に増やします。
- 欠損 keypoint が多い顔は bbox 学習には使い、keypoint loss には入れない、という分離を徹底します。
- fliplr 時の `e1/e2`、`m1/m2` swap を目視 preview で確認します。
- 目、鼻、口角の per-keypoint NME/PCK を毎回出し、悪い部位に対して loss weight または追加ラベルを検討します。

### crop refinement

- 同時出力 head の keypoint を初期値として、bbox crop 内で PFLD/RTMPose をもう一段かける refinement を試します。
- ellipse の `theta` を使って crop を正面寄せに正規化し、keypoint head に渡します。
- crop margin を `1.1x`、`1.25x`、`1.5x` で比較します。顔 bbox が tight すぎると口角や目が欠けやすいです。

## 5. 追加で工夫できること

- Ellipse-aware NMS を実装します。現状は `ellipse_aware_nms` が bbox NMS に fallback しているため、ellipse 同士の重なりを使えば重複検出が減る可能性があります。
- bbox と ellipse の両方を評価指標にします。annotation が ellipse 由来なら、bbox IoU だけでなく ellipse IoU、center error、radius error も見ます。
- YOLO11s Pose を teacher にした pseudo-labeling を使います。DINO が苦手な小顔/暗所/混雑に対して、高信頼 pseudo label と人手確認を組み合わせます。
- DINO patch similarity を hard negative mining に使います。Phase 1 の patch similarity で顔らしい背景 crop を集め、false positive 抑制用 negative として追加します。
- temporal smoothing を入れます。入力が動画由来なら、tracking により bbox/keypoint の一瞬の欠落を補えます。ただし評価は frame-level と video-level を分けます。
- head box を auxiliary context に使います。`head_boxes` は metadata に残っているため、face が隠れている場合に head center/context を補助 target として使えます。
- score calibration を別途行います。val set で reliability curve を見て、temperature scaling または threshold per condition を検討します。
- TTA は最後に使います。multi-scale/flip TTA は精度を上げる可能性がありますが、head 自体の改善判断を曇らせるため、基本比較が終わってからにします。

## 推奨する次の実験順

1. 評価 protocol を固定する。dataset、split、image size、resize方式、val threshold、test一回評価、出力指標を固定します。
2. DINO-CenterFace-ellipse の E2/E4/E7 を同じ protocol で再評価します。F1 winner と bbox mAP winner と keypoint winner を分けて記録します。
3. DINO-SCRFD real DINOv3 baseline を同じ protocol で再評価します。score calibration と assignment 改善前の基準にします。
4. GT bbox crop + PFLD/RTMPose を実行します。DINO同時出力 head の keypoint NME がどこまで改善可能かを見るためです。
5. DINO-CenterFace-ellipse に keypoint loss 改善、quality score、ellipse-aware NMS を順に入れます。
6. それでも小顔 recall が足りなければ、RetinaFace-style anchor/prior head または finer FPN level を追加します。
7. 最後に DINO last blocks の partial unfreeze または adapter fine-tuning を試します。

## 参照

- Local design: `docs/dino_centerface_ellipse_design.md`
- Local design: `notes/dino_scrfd_design.md`
- Local evaluation: `outputs_experiment/dino_centerface_ellipse_aug_ablation_real_dinov3_20260628_eval/_summary/augmentation_ablation_comparison.csv`
- Local evaluation: `outputs_experiment/dino_scrfd_aug_ablation_real_dinov3_20260630_eval/E0_baseline/summary.json`
- Local comparison: `outputs_experiment/dino_scrfd_vs_yolo11s_20260628/comparison_summary.json`
- DINOv3: <https://arxiv.org/abs/2508.10104>
- DINOv3 official repository: <https://github.com/facebookresearch/dinov3>
- DINOv2: <https://arxiv.org/abs/2304.07193>
- CenterFace: <https://arxiv.org/abs/1911.03599>
- SCRFD: <https://arxiv.org/abs/2105.04714>
- RetinaFace: <https://arxiv.org/abs/1905.00641>
- RTMDet: <https://arxiv.org/abs/2212.07784>
- RTMPose: <https://arxiv.org/abs/2303.07399>
- PFLD: <https://arxiv.org/abs/1902.10859>
- Wing Loss: <https://arxiv.org/abs/1711.06753>
- Adaptive Wing Loss: <https://arxiv.org/abs/1904.07399>
- Ultralytics Pose task: <https://docs.ultralytics.com/tasks/pose/>
