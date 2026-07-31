# 現行 DEIMv2-L + DINOv3 ViT-S/16 unfreeze アーキテクチャ

最終更新: 2026-07-22

## 1. 文書の対象

本書は、このリポジトリで顔検出に使用している **DEIMv2-L + DINOv3 ViT-S/16**
の構造、パラメータ数、初期化、学習時の unfreeze 方針を説明する。

現行の学習設定は次のファイルである。

- 最新学習: `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full.yaml`
- アーキテクチャ定義元: `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v1_960_full.yaml`
- 実装: `src/models/dinov3_deimv2/official.py`

v2 設定は v1 設定を `extends` で継承し、dataset の場所、件数検証値、実験名だけを
上書きしている。したがって v1 と v2 のモデル構造およびパラメータ数は同じである。

なお、`configs/model_registry/active_learning_face_models.yaml` の active-learning 用 entry は
現在 v1 の EMA checkpoint を参照している。一方、最新の学習済み run は v2 である。
両者は dataset と重みが異なるが、本書で説明するアーキテクチャは共通である。

## 2. 概要

本モデルは、DINOv3 ViT-S/16 を semantic backbone、DEIMv2 の Spatial and Token
Adapter（STA）を detail path、HybridEncoder を multi-scale neck、DEIMTransformer を
NMS-free な bbox decoder として用いる 1-class face detector である。

```text
RGB image [B, 3, 960, 960]
  |
  +-- DINOv3 ViT-S/16 --------------------------------------+
  |     block index 5, 8, 11 の patch token                 |
  |     [60x60, 384ch] -> [stride 8, 16, 32] へ再標本化     |
  |                                                          +--> scale別concat
  +-- STA spatial prior CNN ---------------------------------+      + 1x1 projection
        [stride 8: 64ch, stride 16: 128ch, stride 32: 128ch]       -> 各224ch
                                                                       |
                                                        HybridEncoder (FPN + PAN)
                                                        [120x120, 60x60, 30x30]
                                                                       |
                                                        flatten: 18,900 positions
                                                                       |
                                                        top-300 query selection
                                                                       |
                                                        DEIM decoder x 4 layers
                                                                       |
                                                        face score + bbox
                                                                       |
                                                        top-k / score filter
                                                        (NMSは使用しない)
```

主要設定は次のとおりである。

| 項目 | 現行値 |
| --- | ---: |
| task | face bbox detection、1 class |
| 入力 | 960 x 960 RGB |
| backbone | DINOv3 ViT-S/16 |
| DINO patch size | 16 x 16 |
| DINO embedding width | 384 |
| DINO block数 / attention head数 | 12 / 6 |
| DINO FFN ratio | 4 |
| DINO storage token数 | 4 |
| 取得する DINO block index | `[5, 8, 11]`（0-based） |
| STA `conv_inplane` | 32 |
| detector hidden width | 224 |
| feature stride | `[8, 16, 32]` |
| HybridEncoder | P5に1 Transformer layer、8 heads、FFN 896 |
| decoder | 4 layers、8 heads、FFN 1792 |
| object query | 300 |
| deformable attention points | `[3, 6, 3]` |
| bbox distribution | `reg_max=32`、各辺33 bin |
| output class | face 1 class（COCO category id 2を内部label 0へ変換） |

`[5, 8, 11]` は 0-based index であり、12 block のうち第6、第9、第12 blockを
通過した後の特徴を表す。CLS token と storage token は検出特徴には使用せず、各層の
正規化済み patch token を使用する。

## 3. 各コンポーネント

### 3.1 DINOv3 ViT-S/16 backbone

DINOv3 は 960 x 960 の画像を 16 x 16 patch に分割し、60 x 60 = 3,600 個の patch
tokenへ変換する。各 token は384次元である。第6、第9、第12 block後の token mapを取得し、
STAの3段構造に合わせて次の大きさへ bilinear interpolation する。

| level | DINO取得時 | 再標本化後 | 実効stride |
| --- | --- | --- | ---: |
| P3 semantic | `[B, 384, 60, 60]` | `[B, 384, 120, 120]` | 8 |
| P4 semantic | `[B, 384, 60, 60]` | `[B, 384, 60, 60]` | 16 |
| P5 semantic | `[B, 384, 60, 60]` | `[B, 384, 30, 30]` | 32 |

DINOv3 単体は本来 single-scale ViT だが、この再標本化と STA により detector が必要とする
3段の feature pyramid を作る。

### 3.2 STA spatial prior と projection

semantic path と並列に、正規化済みRGB画像を小さな CNN に入力する。`conv_inplane=32` の
SpatialPriorModuleV2 は、畳み込み、SyncBatchNorm、GELU、max pooling により局所的な輪郭や
位置情報を抽出する。

| level | STA detail feature | DINOとの結合後 | 1x1 projection後 |
| --- | --- | --- | --- |
| P3 | `[B, 64, 120, 120]` | 384 + 64 = 448 ch | `[B, 224, 120, 120]` |
| P4 | `[B, 128, 60, 60]` | 384 + 128 = 512 ch | `[B, 224, 60, 60]` |
| P5 | `[B, 128, 30, 30]` | 384 + 128 = 512 ch | `[B, 224, 30, 30]` |

各 scale で semantic feature と detail feature を channel方向に連結し、1 x 1 convolution と
SyncBatchNorm で224 channelへ射影する。DINOの意味特徴だけでなく、顔bboxの境界推定に必要な
細部をRGB側から補う役割である。

### 3.3 HybridEncoder

3段の224-channel featureを DEIMv2 の HybridEncoder に渡す。

1. 最も低解像度の P5（30 x 30）だけに、2D sin-cos positional embedding付きの
   Transformer encoderを1層適用する。
2. top-down FPNで P5 -> P4 -> P3 と特徴を伝播する。
3. bottom-up PANで P3 -> P4 -> P5 と再集約する。
4. scale間の融合は concat ではなく DEIMv2 設定の `sum` を使い、各融合部には
   `RepNCSPELAN5` / CSP2系 blockを使う。

出力shapeは入力と同じく `[B, 224, 120, 120]`、`[B, 224, 60, 60]`、
`[B, 224, 30, 30]` である。

### 3.4 DEIMTransformer decoder

3段の feature mapを flattenして連結すると、1画像あたり
`120*120 + 60*60 + 30*30 = 18,900` positionsになる。それぞれに scale-aware anchorを
生成し、encoder score上位300 positionsを object queryとして選ぶ。

`learn_query_content=false` のため、query contentは固定の learned query embeddingではなく、
選択された画像特徴から作られる。decoderは以下の処理を4層繰り返す。

- query間の multi-head self-attention
- 3 levelに対する multi-scale deformable cross-attention
- gateway と RMSNorm
- SwiGLU FFN
- face scoreとbboxの逐次 refinement

bboxは単純な4値回帰だけでなく、`reg_max=32` の分布表現（33 bin x 4辺）を使用する。
推論時の最終出力は300 query分の face logitと正規化 `cx, cy, w, h` である。

### 3.5 学習時だけ使う機構

- contrastive denoising: `num_denoising=100`、label noise 0.5、box noise scale 1.0。
  実際の denoising query数はbatch内最大GT数からpositive/negative pairとして動的に決まる。
- auxiliary loss: 各decoder layer、中間encoder出力、pre-bbox出力にもlossを適用する。
- Hungarian matching: epoch 0--49は class / L1 / GIoU costを `2 / 5 / 2` で使用し、
  epoch 50以降は `class_score * IoU^4` を用いるDEIM matchingへ切り替える。
- loss: MAL 1.0、bbox L1 5.0、GIoU 2.0、FGL 0.15、DDF 1.5。

criterion、Hungarian matcher、denoising生成器は学習時だけ必要で、学習可能parameterは持たない。

### 3.6 postprocess

最終bboxを `cxcywh` から `xyxy` へ変換し、sigmoid face scoreの上位候補を返す。
DETR/DEIMのset predictionであるため NMS は使用しない。

モデル設定上は top 300を生成後、wrapperで最大100件に制限する。`score_thr=0.001` は評価用に
候補を広く残すためのfloorであり、運用上の判定thresholdとは別である。例えば現行
active-learning registryは v1 test評価 artifactに保存された fixed thresholdを読み込む。

## 4. パラメータ数

パラメータ数は v1 preflightで実モデルを構築して集計した値と、同一architectureを用いた
v2 runの `run_meta.json` を照合した値である。

| component | 全parameter | 学習対象 | 全体比 |
| --- | ---: | ---: | ---: |
| DINOv3 ViT-S/16 | 21,601,152 | 21,601,152 | 66.58% |
| STA + 3 scale projection / norm | 572,256 | 572,256 | 1.76% |
| HybridEncoder | 5,147,296 | 5,147,296 | 15.86% |
| DEIMTransformer decoder / heads | 5,123,939 | 5,123,937 | 15.79% |
| criterion / matcher / postprocessor | 0 | 0 | 0.00% |
| **合計** | **32,444,643** | **32,444,641** | **100.00%** |

重要な点は次のとおりである。

- **総parameter数は32,444,643（約32.44M）**。
- **optimizerが更新するparameter数は32,444,641**。
- 差の2個は decoder の固定スカラー `up` と `reg_scale` であり、DINOの凍結parameterではない。
- DINOv3の21,601,152 parametersは、patch embedding、全12 Transformer blocks、norm、
  token parameterを含めてすべて学習対象である。
- DINO以外の検出部は合計10,843,491 parameters、そのうち10,843,489が学習対象である。
- FP32 parameter本体だけなら約123.8 MiBである。学習時は勾配、AdamW state、EMA、activationも
  必要なので、実際のGPU memory使用量はこの値より大きい。

EMAは32.44M parametersのshadow copyを保持するが、これは別architectureのparameterではない。
したがってモデルのparameter数をEMA分だけ2倍には数えない。

### 4.1 公式値との関係

| 比較対象 | element / parameter数 | 補足 |
| --- | ---: | --- |
| 公式model-zoo DEIMv2-L | 32.2M | 公式表の丸められたCOCO 80-class値 |
| local 80-class再構成モデル | 32,551,214 | 現行wrapperで80 class headを構築した値 |
| local face 1-classモデル | 32,444,643 | 本書の対象 |
| 公式HF checkpoint全tensor element | 32,631,627 | BN統計、anchor buffer等を含みparameter数とは定義が異なる |

local 80-class再構成値と公式32.2M表記との差は351,214（1.09%）で、設定された2%以内である。
80 classから1 classへ変更すると、class head関連で106,571 parameters減少する。

## 5. 「DINO unfreeze」の意味

現行設定では次の2条件を同時に指定している。

```yaml
model:
  backbone:
    freeze: false
    trainable_last_blocks: 0
train:
  backbone_policy: full_unfreeze
```

ここで `trainable_last_blocks: 0` は「DINOを0 blockだけ学習する」という意味ではない。
`freeze: false` と `backbone_policy: full_unfreeze` が優先され、DINO全体が学習対象になる。
production checkもこの組み合わせを検証している。

学習率は DINO と detectorで分ける。

| parameter group | 基準learning rate | weight decay |
| --- | ---: | ---: |
| DINO decay対象 | 1.25e-5 | 1.25e-4 |
| DINO norm / bias | 1.25e-5 | 0 |
| DEIM norm / bias | 5.0e-4 | 0 |
| その他 detector | 5.0e-4 | 1.25e-4 |

DINOのlearning rateは detectorの1/40である。事前学習済み表現を急激に壊さず、全backboneを
face datasetへ適応させるための設定である。backboneを更新するため feature cacheは無効である。

参考として、同じモデルをfreezeした場合も総parameter数は32,444,643のままだが、学習対象は
非DINO部分の10,843,489だけになる。freeze / unfreezeの違いはモデル容量ではなく、
backpropagationとoptimizer更新の対象範囲である。

## 6. 初期化とcheckpoint

モデルは公式 DEIMv2-L COCO checkpoint
`checkpoint/DEIMv2/deimv2_dinov3_l_coco.safetensors` から初期化する。このcheckpointには
DINOv3 backboneも含まれるため、`backbone.pretrained=false`、`weights=null` でもDINOが
random initializationになるわけではない。

face 1-classへ変更するため、COCO 80-classの次のheadだけを読み込まず再初期化する。

- denoising class embedding
- encoder score head
- 4層の decoder score head

preflightでの移植結果は、807 checkpoint tensors中794 tensorsをload、DINO prefixの188 keysを
local wrapper名へremap、shape mismatch 0である。local face model state elementの99.995%を
初期化している。checkpoint側だけにある640入力用の生成済み `anchors` と `valid_mask` は使わず、
現行実装で入力feature sizeから動的生成する。

学習中は full-model EMA（decay 0.9999、warmups 1000）も更新する。評価・推論loaderは
checkpoint内にEMAがあればEMA stateを優先する。

## 7. 現行学習条件と状態

最新 v2 runの主要条件は以下である。

| 項目 | 値 |
| --- | ---: |
| dataset | `face_pose_drive_v2` |
| train / val / test画像数 | 4,235 / 530 / 530 |
| train batch | 4 |
| gradient accumulation | 4 |
| effective batch | 16 |
| optimizer | fused AdamW |
| 最大epoch | 120 |
| warmup / flat | 3 / 60 epochs |
| scheduler | flat-cosine |
| gradient clipping | 0.1 |
| AMP / TF32 | enabled / enabled |
| validation | 5 epochsごと |
| early stopping開始可能epoch | 68 |

v2 runは2026-07-20に正常終了し、epoch 70でearly stoppingした。best validation mAP50:95は
epoch 5の0.5722で、`checkpoints/best_val_map.pt` に保存されている。本項の値は学習状態の記録で
あり、アーキテクチャのparameter数には影響しない。また、リポジトリ内では v2 best checkpointの
独立test評価artifactは本書更新時点で確認できていないため、test性能値とは扱わない。

## 8. 実装・検証資料

- 現行v2設定: `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full.yaml`
- 継承元の全モデル設定: `configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v1_960_full.yaml`
- model builder / STA wrapper: `src/models/dinov3_deimv2/official.py`
- HybridEncoder: `src/third_party/deimv2/engine/deim/hybrid_encoder.py`
- decoder: `src/third_party/deimv2/engine/deim/deim_decoder.py`
- matcher: `src/third_party/deimv2/engine/deim/matcher.py`
- v1 parameter preflight: `outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v1_960_full_preflight/preflight_report.json`
- v2実行時parameter group: `outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full/run_meta.json`
- v2 early stopping記録: `outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full/early_stop.json`

公式の対応構成は DEIMv2-L であり、DEIMv2-Mのdistilled ViT-TinyPlus構成とは異なる。

- <https://github.com/Intellindust-AI-Lab/DEIMv2>
- <https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/configs/deimv2/deimv2_dinov3_l_coco.yml>
- <https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO>
- <https://github.com/facebookresearch/dinov3>
