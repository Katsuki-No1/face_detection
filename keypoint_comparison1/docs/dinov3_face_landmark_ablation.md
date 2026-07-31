# DINOv3 face landmark A–G比較

## 目的

DEIMv2-Lで顔を検出し、各顔cropから5点landmark（左右の目、鼻先、左右の口角）を推定する本番品質の比較実装です。比較対象はViTPose-style CNN head、そのparameter-matched 376-channel版、Landmark Query、座標周辺の特徴を再取得するLandmark Query + refinementです。

```text
train/validation: GT ellipse ── calibrated geometry augmentation (train only)
deployment: ellipse DEIMv2-L ── long-axis-aligned square ROI with real background
                                              ↓ 1.25x context + affine normalization
                                       256×256 model input ── DINOv3 ViT-S/16
                                                                  ├─ ViTPose-style heatmap head
                                                                  ├─ parameter-matched ViTPose-376 heatmap head
                                                                  ├─ Landmark Query head
                                                                  └─ Query head → local resampling → residual refinement ×2
```

元画像上では楕円の長径を一辺とし、角度に沿って回転した正方形ROIを使います。
短径側の余白はpaddingではなく本番と同じ実画像背景です。この正方形を等方affineで
model inputへ変換し、予測点は逆affineで元画像座標へ戻します。学習時はGT楕円へ、学習済み
楕円DEIMとGTの実測5次元残差を顔面積別にjoint sampleして適用します。中心ずれ、軸比、
角度差の相関を維持し、成分ごとの独立noiseは使いません。validationは固定GT楕円、
実運用評価は固定DEIM予測を使います。誤差校正2動画とkeypoint選択4動画は動画単位で
分離し、testは校正に使いません。

デモ用の縮小データや短縮epochは使いません。production configは全train instance、150 epoch上限、AMP、EMA、backbone/head別learning rate、warmup + cosine schedule、段階的backbone unfreeze、early stopping、W&Bを使用します。

## 比較条件

7条件に対して学習するcheckpointは4個です。これによりbboxだけを変えるA/C、B/D、F/Gで、head重みの違いが混入しません。

| 条件 | 入力ROI | head / checkpoint | 確認すること |
| --- | --- | --- | --- |
| A | GT楕円 | ViTPose-style | 単純なheatmap modelのoracle-ROI上限 |
| B | GT楕円 | Landmark Query | Aとの差によるquery設計の効果 |
| C | DEIM予測楕円 | Aと同じViTPose-style | Aとの差によるROI誤差の影響 |
| D | DEIM予測楕円 | Bと同じLandmark Query | query modelの実運用性能 |
| E | DEIM予測楕円 | Query + refinement | Dに対する段階補正の追加利益 |
| F | GT楕円 | ViTPose-376 | Queryとhead parameter数を揃えたCNN decoderのoracle性能 |
| G | DEIM予測楕円 | Fと同じViTPose-376 | parameter-matched CNN decoderの実運用性能 |

Eのrefinementはdecoder layerを追加するだけではありません。現在座標の周囲を3×3でDINO projected feature mapからbilinear samplingし、query特徴・局所特徴・座標からbounded residualを推定します。2段の探索半径と最大補正量は設定から変更できます。

## 公平性を保つ固定条件

- 全headでDINOv3 ViT-S/16、初期DINO重み、GT楕円整列ROI、1.25倍context、256×256 model input、geometry augmentation、optimizer recipeを共通化。
- DINO初期値は、balanced 8:1:1 datasetをCOCO事前学習重みからfine-tuneした楕円DEIMv2-L checkpointの`backbone.dino.backbone.*`をstrictに抽出。
- DEIMv2-L内と同じImageNet mean/stdでcrop入力を正規化。
- A/Cは同じViTPose checkpoint、B/Dは同じQuery checkpointを使用。
- F/Gは同じViTPose-376 checkpointを使用。
- C/D/E/Gは同じDEIM prediction cacheとIoU matchingを使用。
- DEIM score thresholdはv2 validationだけで`precision >= 0.90`を満たす最大recallとして選択済みの0.33（validation P=0.9026、R=0.8750）に固定。test上で再選択しない。
- bbox影響のdeltaは、DEIMがmatchできた同一GT instanceだけに絞ったoracle resultと比較。全GTに対する運用性能はdetector missを失敗として含むend-to-end PCKで別途表示。
- fixed test annotationのSHA-256とsplit間のsource video/image hash漏洩をpreflightで検査。
- 外部`testdata_for_v1-2`はtrain/val/testとのsource videoおよびimage hash重複がないことを検査。

## データセット

学習・validationはcanonical metadataから直接生成した`face_pose_drive_v2`の
DEIM/DINO専用5点annotation（category id 2、ellipse保持）を使用します。

COCO keypoint visibilityは`v=0`をmissing/out-of-image、`v=1`を座標あり・遮蔽、
`v=2`を座標あり・可視として扱います。本命のQuery Refinementは各点に
`presence=P(v>0)`と`visibility=P(v=2 | v>0)`の2個のbinary logitを持ちます。
座標・heatmap lossは`v>0`だけ、presence BCEはmissingを含む全slot、
visibility BCEは`v>0`だけを対象にします。推論時の3状態確率は
`[1-p, p(1-v), pv]`の順にmissing / occluded / visibleへ復元します。
2段のrefinementも座標だけでなく両logitを残差更新します。学習時の人工遮蔽は
presenceと座標ラベルを維持したままconditional visibilityだけをnegativeへ更新します。

旧checkpointは単一logitしか持たないため、新しい`factorized_state`出力先へ
再学習します。`state_head_mode`を省略したlegacy modeでは旧modelのparameter名・
shape・出力keyを維持しますが、factorized runから旧checkpointをresumeすることは
拒否されます。

| split | images | state教師face | 全点missing face | 座標教師face | annotation SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 4,237 | 6,529 | 448 | 6,081 | `2fac1a20b089c801099bed0b439db3ed612d2ea812b5eab21a574f473f981cfa` |
| val（選択用固定subset） | 330 | 532 | 34 | 498 | `11148e4ec7240b0ec24cc9ade9530fc4fb231f874e23aab6aea7c0c31981a58e` |
| test | 528 | 769 | 49 | 720 | `bbab427fdddbfe2d4c0d669ea8d879db58c6f3bab13d6653738b060cecb3e760` |

独立確認には`testdata_for_v1-2`（600 images、870 annotations、828 usable instances、DEIM/DINO
SHA-256 `f9c3fe76cc948b43d31cf79c58a514cb3432855c2f666c58d39936f7346f1690`）も使用します。
test setで閾値や学習設定を再調整しないでください。

## Model details

### 共通DINO backbone

- DINOv3 ViT-S/16
- patch size 16、embedding 384
- 256×256 model inputから16×16 patch feature mapを生成
- v2でfull-unfreeze学習済みのDEIMv2-L DINO部分を初期値として使用
- 最初の5 epochはfreezeし、その後は低learning rateでfull unfreeze

実checkpointを用いたpreflightでのparameter実測値は次の通りです。DINO列は各modelに含まれる共通backboneで、同時運用時に共有しているという意味ではありません。

| variant | DINOv3 backbone | landmark head | total |
| --- | ---: | ---: | ---: |
| ViTPose-style | 21,601,152 | 2,627,599 | 24,228,751 |
| ViTPose-376 | 21,601,152 | 4,579,399 | 26,180,551 |
| Landmark Query | 21,601,152 | 4,579,338 | 26,180,490 |
| Query + refinement | 21,601,152 | 7,468,562 | 29,069,714 |

strict extractionではDEIM checkpointからDINOの188 tensor、21,614,992 state element（parameter以外のstateを含む）を読み込み、missing/unexpectedはいずれも0でした。

### ViTPose-style

DINOの16×16 featureを2段のConvTranspose2d + BatchNorm + ReLUで64×64へ復元し、landmarkごとのheatmapを生成します。学習はGaussian heatmap lossとsoft-argmax coordinate lossを併用し、評価はargmax + quarter-pixel補正です。これはDINO後段にCNNの局所性を与えた条件です。

### Parameter-matched ViTPose-376

backbone、loss、decode、学習recipeはViTPose-styleと同一で、2段のConvTranspose2dのhidden channelだけを256から376へ広げます。factorized state head込みで4,579,399 parametersとなり、Query headの4,579,338との差は61 parametersです。これは公式のS/B/L/H名ではなく、CNN decoder容量をQueryに一致させるためのablation専用variantです。

### Landmark Query

5個の意味固定queryを用い、query-query self-attentionとDINO patchへのcross-attentionを4層実行します。各decoder layerにauxiliary coordinate supervisionを与えます。query間attention、cross-attention entropy、landmark別誤差相関、shared-headに対するlandmark別gradient cosineを保存するため、landmark間の協調・干渉を観測できます。

### Query + refinement

Queryとrefinementを同じ150 epoch内でend-to-end学習し、次の2段を追加します。Bの学習済みQueryからのwarm-startは実装上選択できますが、Bの全学習に加えてEを再学習するとtraining budgetが倍になり、D対Eの差へ学習量が混入します。そのため既定のA–G比較では無効です。

1. 現在座標から半径2 feature cell、3×3点をsampleし、最大0.08 crop幅のresidualを推定。
2. 更新座標から半径1 feature cellで再sampleし、最大0.04 crop幅のresidualを推定。

各段にauxiliary lossを入れ、途中座標も監督します。

## 実行方法

設定とreal checkpoint/dataを検査し、4 headすべてで実forward/backward、DEIMで1画像推論を行います。

```bash
.venv/bin/python keypoint_comparison1/scripts/run_landmark_experiments.py \
  --config keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2.yaml \
  --stage preflight \
  --device cuda:0
```

4 checkpointを順にfull trainingします。全headを同じseed、同じ150 epoch上限で学習し、既定では`last.pt`があれば再開します。

```bash
.venv/bin/python keypoint_comparison1/scripts/run_landmark_experiments.py \
  --stage train \
  --variants all \
  --device cuda:0 \
  --resume \
  --wandb
```

既存3モデルを再学習せずViTPose-376だけを追加学習する場合は、次を実行します。

```bash
.venv/bin/python keypoint_comparison1/scripts/run_landmark_experiments.py \
  --stage train \
  --variants vitpose_376 \
  --device cuda:0 \
  --resume \
  --wandb
```

固定test 2種でA–Gを評価します。

```bash
.venv/bin/python keypoint_comparison1/scripts/run_landmark_experiments.py \
  --stage evaluate \
  --evaluation-set v2_test \
  --evaluation-set external_v1_v2 \
  --device cuda:0
```

`--stage all`でpreflight、training、evaluationを連続実行できます。DEIM cacheを作り直す場合だけ`--force-detector-cache`を指定します。

重みを必要としないunit testと、real weight testはmarkerで分離しています。

```bash
.venv/bin/python -m pytest -q keypoint_comparison1/tests/test_landmark_detection.py
.venv/bin/python -m pytest -q -m model_weights keypoint_comparison1/tests/test_landmark_detection_model_weights.py
```

## 出力

```text
outputs/landmark_detection/..._gt_ellipse_long_axis_square_coco_visibility/
├── config_resolved.yaml
├── logs/pipeline.log
├── preflight/{report.json,success.json}
├── vitpose/
│   ├── metrics.csv
│   ├── training_summary.json
│   └── checkpoints/{best_nme.pt,last.pt,epoch_*.pt}
├── vitpose_376/...
├── query/...
├── query_refinement/...
└── evaluation/
    ├── all_results.json
    ├── v2_test/
    │   ├── cache/deimv2_l_predictions.json
    │   ├── A_vitpose_gt/{metrics.json,predictions.jsonl}
    │   ├── B_query_gt/{metrics.json,predictions.jsonl}
    │   ├── C_vitpose_deim/{metrics.json,predictions.jsonl}
    │   ├── D_query_deim/{metrics.json,predictions.jsonl}
    │   ├── E_query_refinement_deim/{metrics.json,predictions.jsonl}
    │   ├── F_vitpose_376_gt/{metrics.json,predictions.jsonl}
    │   ├── G_vitpose_376_deim/{metrics.json,predictions.jsonl}
    │   ├── diagnostics.json
    │   └── comparison.{json,csv,md}
    └── external_v1_v2/...
```

## 判断指標

- GT bboxのA対B: CNN heatmapとquery設計の差。
- GT bboxのF対B: head parameter数を一致させたCNN decodeとQuery decodeの差。
- matched instanceに限定したA対C、B対D: detector bbox位置・大きさ誤差によるlandmark劣化。
- matched instanceに限定したF対G: parameter-matched ViTPoseのbbox誤差による劣化。
- C対D: 実運用box条件でのCNNとqueryの差。
- G対D: 実運用box条件でparameter-matched CNNとqueryの差。
- D対E: 局所再sampleと段階補正の追加利益。
- `NME`: GT bbox対角長で正規化。低い方が良い。
- `PCK@.03/.05/.10`: match済みfaceでのlandmark精度。
- `end_to_end_PCK`: DEIMの未検出を失敗に含めた全GT上の性能。
- `by_face_size`: original GT face areaによるsmall/medium/large別性能。
- `presence_missing_macro_f1`: missing / presentの2-class macro F1。
- `state_macro_f1`: missing / occluded / visibleの3-class macro F1。
- gradient cosineの負値: 2 landmarkの勾配がshared head上で競合している可能性。
- landmark error correlation: 同じsampleで複数点が同時に崩れる傾向。

parameter数は`preflight/report.json`へ、DINO backboneと各headを分けて実測保存します。学習・評価をまだ実行していない段階では性能値を記載せず、結果を捏造しません。

## 既知の制約

- landmark annotationは5点のみです。68/98点などへ拡張する場合はsemantic query数、左右反転index、category metadataを明示的に変更する必要があります。
- top-down方式なので、運用性能の上限はDEIMv2-Lのface recallに依存します。
- presence / conditional visibilityの運用閾値校正は別工程です。test setで閾値を選ばないでください。
- query attentionは因果的説明ではなく診断情報です。性能差、gradient cosine、誤差相関と合わせて解釈してください。
- 256×256 model inputは現dataset向けの固定比較条件です。元画像上のROIは楕円長径基準の回転正方形です。入力解像度を変更するとDINO token数とViTPose heatmap解像度が変わるため、新しい比較実験として扱います。
