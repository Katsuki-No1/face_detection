# DINOv3 face landmark A–E比較

## 目的

DEIMv2-Lで顔を検出し、各顔cropから5点landmark（左右の目、鼻先、左右の口角）を推定する本番品質の比較実装です。比較対象はViTPose-style CNN head、Landmark Query、座標周辺の特徴を再取得するLandmark Query + refinementです。

```text
original image ── DEIMv2-L ── face bbox ── 256×256 crop ── DINOv3 ViT-S/16
                                                              ├─ ViTPose-style heatmap head
                                                              ├─ Landmark Query head
                                                              └─ Query head → local resampling → residual refinement ×2
```

デモ用の縮小データや短縮epochは使いません。production configは全train instance、150 epoch上限、AMP、EMA、backbone/head別learning rate、warmup + cosine schedule、段階的backbone unfreeze、early stopping、W&Bを使用します。

## 比較条件

5条件に対して学習するcheckpointは3個です。これによりbboxだけを変えるA/CとB/Dで、head重みの違いが混入しません。

| 条件 | 入力bbox | head / checkpoint | 確認すること |
| --- | --- | --- | --- |
| A | GT | ViTPose-style | 単純なheatmap modelのoracle-box上限 |
| B | GT | Landmark Query | Aとの差によるquery設計の効果 |
| C | DEIM予測 | Aと同じViTPose-style | Aとの差によるbbox誤差の影響 |
| D | DEIM予測 | Bと同じLandmark Query | query modelの実運用性能 |
| E | DEIM予測 | Query + refinement | Dに対する段階補正の追加利益 |

Eのrefinementはdecoder layerを追加するだけではありません。現在座標の周囲を3×3でDINO projected feature mapからbilinear samplingし、query特徴・局所特徴・座標からbounded residualを推定します。2段の探索半径と最大補正量は設定から変更できます。

## 公平性を保つ固定条件

- 全headでDINOv3 ViT-S/16、初期DINO重み、256×256 crop、crop倍率、augmentation、optimizer recipeを共通化。
- DINO初期値は、学習済みDEIMv2-L v2 checkpointの`backbone.dino.backbone.*`をstrictに抽出。
- DEIMv2-L内と同じImageNet mean/stdでcrop入力を正規化。
- A/Cは同じViTPose checkpoint、B/Dは同じQuery checkpointを使用。
- C/D/Eは同じDEIM prediction cacheとIoU matchingを使用。
- DEIM score thresholdはv2 validationだけで`precision >= 0.90`を満たす最大recallとして選択済みの0.33（validation P=0.9026、R=0.8750）に固定。test上で再選択しない。
- bbox影響のdeltaは、DEIMがmatchできた同一GT instanceだけに絞ったoracle resultと比較。全GTに対する運用性能はdetector missを失敗として含むend-to-end PCKで別途表示。
- fixed test annotationのSHA-256とsplit間のsource video/image hash漏洩をpreflightで検査。
- 外部`testdata_for_v1-2`はtrain/val/testとのsource videoおよびimage hash重複がないことを検査。

## データセット

学習・validationは`face_pose_drive_v2`のRTMPose形式5点annotationを使用します。

| split | images | annotations | visible landmarkを1点以上持つface instance | annotation SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 4,235 | 6,542 | 6,126 | `ba59dba03c94361ece2ca4a59682bd52a9e25a272f9ab515e3f4b92c3a3c24cc` |
| val | 530 | 784 | 710 | `10208048c607ce32daea2715a73920c32ef9ca432e7080b6b34496c047ddf2d6` |
| test | 530 | 812 | 755 | `5e02ae76ac564f9bf4ee0cbecab623bba6f23ff790a5072a12cf13d6432e7894` |

独立確認には`testdata_for_v1-2`（600 images、870 annotations、828 usable instances、SHA-256 `98161a5a3e9ccaa18c34d24f5926b89aaba602b330cc8ab642b26a02bb703de6`）も使用します。test setで閾値や学習設定を再調整しないでください。

## Model details

### 共通DINO backbone

- DINOv3 ViT-S/16
- patch size 16、embedding 384
- 256 cropから16×16 patch feature mapを生成
- v2でfull-unfreeze学習済みのDEIMv2-L DINO部分を初期値として使用
- 最初の5 epochはfreezeし、その後は低learning rateでfull unfreeze

実checkpointを用いたpreflightでのparameter実測値は次の通りです。DINO列は各modelに含まれる共通backboneで、同時運用時に共有しているという意味ではありません。

| variant | DINOv3 backbone | landmark head | total |
| --- | ---: | ---: | ---: |
| ViTPose-style | 21,601,152 | 2,625,674 | 24,226,826 |
| Landmark Query | 21,601,152 | 4,579,081 | 26,180,233 |
| Query + refinement | 21,601,152 | 7,467,791 | 29,068,943 |

strict extractionではDEIM checkpointからDINOの188 tensor、21,614,992 state element（parameter以外のstateを含む）を読み込み、missing/unexpectedはいずれも0でした。

### ViTPose-style

DINOの16×16 featureを2段のConvTranspose2d + BatchNorm + ReLUで64×64へ復元し、landmarkごとのheatmapを生成します。学習はGaussian heatmap lossとsoft-argmax coordinate lossを併用し、評価はargmax + quarter-pixel補正です。これはDINO後段にCNNの局所性を与えた条件です。

### Landmark Query

5個の意味固定queryを用い、query-query self-attentionとDINO patchへのcross-attentionを4層実行します。各decoder layerにauxiliary coordinate supervisionを与えます。query間attention、cross-attention entropy、landmark別誤差相関、shared-headに対するlandmark別gradient cosineを保存するため、landmark間の協調・干渉を観測できます。

### Query + refinement

Queryとrefinementを同じ150 epoch内でend-to-end学習し、次の2段を追加します。Bの学習済みQueryからのwarm-startは実装上選択できますが、Bの全学習に加えてEを再学習するとtraining budgetが倍になり、D対Eの差へ学習量が混入します。そのため既定のA–E比較では無効です。

1. 現在座標から半径2 feature cell、3×3点をsampleし、最大0.08 crop幅のresidualを推定。
2. 更新座標から半径1 feature cellで再sampleし、最大0.04 crop幅のresidualを推定。

各段にauxiliary lossを入れ、途中座標も監督します。

## 実行方法

設定とreal checkpoint/dataを検査し、3 headすべてで実forward/backward、DEIMで1画像推論を行います。

```bash
.venv/bin/python scripts/landmarks/run_landmark_experiments.py \
  --config configs/landmark_detection/dinov3_vits16_face_pose_drive_v2.yaml \
  --stage preflight \
  --device cuda:0
```

3 checkpointを順にfull trainingします。全headを同じseed、同じ150 epoch上限で学習し、既定では`last.pt`があれば再開します。

```bash
.venv/bin/python scripts/landmarks/run_landmark_experiments.py \
  --stage train \
  --variants all \
  --device cuda:0 \
  --resume \
  --wandb
```

固定test 2種でA–Eを評価します。

```bash
.venv/bin/python scripts/landmarks/run_landmark_experiments.py \
  --stage evaluate \
  --evaluation-set v2_test \
  --evaluation-set external_v1_v2 \
  --device cuda:0
```

`--stage all`でpreflight、training、evaluationを連続実行できます。DEIM cacheを作り直す場合だけ`--force-detector-cache`を指定します。

重みを必要としないunit testと、real weight testはmarkerで分離しています。

```bash
.venv/bin/python -m pytest -q tests/test_landmark_detection.py
.venv/bin/python -m pytest -q -m model_weights tests/test_landmark_detection_model_weights.py
```

## 出力

```text
outputs/landmark_detection/dinov3_vits16_face_pose_drive_v2/
├── config_resolved.yaml
├── logs/pipeline.log
├── preflight/report.json
├── vitpose/
│   ├── metrics.csv
│   ├── training_summary.json
│   └── checkpoints/{best_nme.pt,last.pt,epoch_*.pt}
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
    │   ├── diagnostics.json
    │   └── comparison.{json,csv,md}
    └── external_v1_v2/...
```

## 判断指標

- GT bboxのA対B: CNN heatmapとquery設計の差。
- matched instanceに限定したA対C、B対D: detector bbox位置・大きさ誤差によるlandmark劣化。
- C対D: 実運用box条件でのCNNとqueryの差。
- D対E: 局所再sampleと段階補正の追加利益。
- `NME`: GT bbox対角長で正規化。低い方が良い。
- `PCK@.03/.05/.10`: match済みfaceでのlandmark精度。
- `end_to_end_PCK`: DEIMの未検出を失敗に含めた全GT上の性能。
- `by_face_size`: original GT face areaによるsmall/medium/large別性能。
- gradient cosineの負値: 2 landmarkの勾配がshared head上で競合している可能性。
- landmark error correlation: 同じsampleで複数点が同時に崩れる傾向。

parameter数は`preflight/report.json`へ、DINO backboneと各headを分けて実測保存します。学習・評価をまだ実行していない段階では性能値を記載せず、結果を捏造しません。

## 既知の制約

- landmark annotationは5点のみです。68/98点などへ拡張する場合はsemantic query数、左右反転index、category metadataを明示的に変更する必要があります。
- top-down方式なので、運用性能の上限はDEIMv2-Lのface recallに依存します。
- head visibility scoreの運用閾値校正は別工程です。test setで閾値を選ばないでください。
- query attentionは因果的説明ではなく診断情報です。性能差、gradient cosine、誤差相関と合わせて解釈してください。
- 256 cropは現dataset向けの固定比較条件です。変更するとDINO token数とViTPose heatmap解像度が変わるため、新しい比較実験として扱います。
