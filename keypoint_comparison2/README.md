# 顔5点 keypoint comparison 2

`keypoint_comparison2` は、固定した DEIMv2-L face detector と
`face_pose_drive_v2_balanced_811` を用いて、顔5点を4方式で比較するproduction実装です。
短縮epoch、dataset subset、demo model、自動生成ラベルへの置換は行いません。

比較対象は次の4方式です。

1. `rtmpose_face5`: face crop 256 → CSPNeXt-M → RTMCC Head → SimCCLabel → 5点
2. `dinov3_fpn_rtmcc`: face crop 320 → DINOv3 ViT-S/16 patch features → Simple FPN → RTMCC → SimCCLabel → 5点
3. `dinov3_orformer_rtmcc`: face crop 320 → DINOv3 ViT-S/16 patch features → patch遮蔽判定 → ORFormer型feature recovery → Simple FPN → RTMCC → SimCCLabel → 5点
4. `deimv2_joint`: DEIMv2 encoder/decoder → face object queries → faceごとの意味固定5 landmark queries → bbox + 5点 + visibility

モデル2とモデル3は同一のDINOv3 ViT-S/16初期値、同一crop、同一FPN/RTMCC条件を使います。
したがって、両者の主な差は遮蔽判定とfeature recoveryです。

## 実装上の重要点

### RTMPose-Face-5

公式の `rtmpose-m_8xb256-120e_face6-256x256` は「6点model」ではなく、
LaPa/WFLW/300W/COFW/Halpe/COCO-WholeBody-Faceの6 datasetを統合した106点modelです。
この実装はCSPNeXt-M、GAU、SimCC分類器の一致する全weightを読み込み、最終convolutionだけを
次の意味対応で5点へ転写してからfull fine-tuneします。

- left eye: 106点index 66–74の平均
- right eye: index 75–83の平均
- nose tip: index 54
- left mouth corner: index 84
- right mouth corner: index 90

単なるImageNet backbone初期化ではなく、公式face landmark headの学習済みweightも利用します。

### DINOv3 + Simple FPN

DINOv3 ViT-S/16のblock 2/5/8/11からpatch featureを取り、学習可能な重みで融合します。
ViTDet型Simple FPNで `4x/2x/1x/0.5x` の4 scaleを生成し、RTMCC Headへ渡します。
320入力、bf16、最初の5 epochのbackbone freeze、その後のlayer-wise
learning-rate decay付きfull fine-tuneを既定にしています。

### 遮蔽対応DINOv3

学習時はclean cropと矩形/noise occlusionを対で生成します。synthetic occluderに覆われた
landmarkはvisibility=falseへ更新します。patch-level occlusion classifierを
BCEで監督し、遮蔽tokenをlearnable mask tokenへ置換した後、visible tokenをmemoryとする
self/cross attention recoveryを6層実行します。遮蔽位置ではclean DINOv3 featureとの
Smooth L1 reconstruction lossも加えます。推論時はclean画像や遮蔽maskを必要としません。
回復器はSの384次元featureに対してidentity-preserving residualとして初期化し、未学習時に
pretrained featureを破壊しない構成です。

### DEIMv2 joint query model

既存のDEIMv2-L encoder/decoderと学習済みface object queryを維持し、confidence上位face query
（学習時はHungarian match済みqueryを必ず含む）から5個の意味固定queryを生成します。
各faceの5 query間だけself-attentionを許可し、DEIMv2 multi-scale encoder memoryへcross-attention
します。座標は対応face bbox内の相対座標として予測するため、別faceへの逸脱を抑えます。
DEIMv2のbbox loss、landmark coordinate loss、各点visibility BCE、全decoder層のauxiliary lossを
同時最適化します。

960px joint学習は有効batch 16を固定し、開始前にunfreeze状態でforward/backward、
AdamW初期state生成、optimizer step、EMA更新までを実測します。`batch 8 / accumulation 2`
から試し、CUDA OOMの場合だけ`4/4 → 2/8 → 1/16`へ移行します。選択結果とpeak GPU memoryは
`deimv2_joint/batch_size_selection.json`へ保存し、同じcheckpointのresumeでは再利用します。

joint学習にはbboxだけの旧`train.json`やRTMPose JSONではなく、canonical metadataから直接生成した
DEIM/DINO専用JSONを使用します。face categoryは最初から既存DEIMv2-Lと同じ`2`で、各annotationに
bbox、楕円、5点、visibility、遮蔽属性を保持します。

### 楕円DEIMとの互換性

DEIM/DINO専用JSONの`ellipse`は、既存の楕円DEIMと同じ
`[cx, cy, semi_major, semi_minor, theta_rad]`形式です。角度は長軸方向を
`[-pi/2, pi/2)`へ正規化します。既存`ellipse_annotations`との全8,138 annotation比較で
楕円値の完全一致を確認しています。

top-down detectorの`crop_geometry: ellipse`は、楕円DEIMの中心・長径・短径・角度から
長径基準の回転正方形ROIを作り、実背景を含む正方形を1回の等方affineで固定サイズの
model tensorへ変換します。顔の縦横比を歪めず、逆affineで予測点を元画像へ厳密に戻します。
画像境界を越えない限り人工paddingは入りません。joint datasetも
ellipse targetを常に渡し、joint推論も楕円model使用時は`ellipse`と`ellipse_bbox_xyxy`を追加出力します。

top-down 3方式の学習・validation cropはDEIM予測そのものを使いません。annotationのGT楕円
`[cx, cy, semi_major, semi_minor, theta_rad]`を基準に、trainだけ学習済み楕円DEIMとGTの
実測残差を毎sample再抽選します。残差はGT楕円の局所座標で、中心2成分、長短径の対数比2成分、
180度周期の角度差1成分として表し、5成分を同じ実測instanceから同時にsampleするため相関を
維持します。顔面積別3層、1–99 percentile winsorization、共通severity 0.75–1.25、
10% clean GTを固定条件とし、成分ごとの独立noiseは加えません。摂動後の長径を基準に
1.25倍context marginを持つ回転正方形を実画像から取得し、model tensorへ等方affine変換します。
validationでは摂動なしのGT楕円を使います。

残差校正には元validationのうち動画単位で固定した2動画（200画像）だけを使い、残り4動画
（330画像）をkeypoint checkpoint選択用validationにします。両者はsource video/image hashで
非重複を検査し、testは校正にも閾値選択にも使いません。DEIM未検出はcrop摂動として人工的に
表現せず、conditional localization residualとは別にcalibration coverageとして保存します。
balanced_811楕円DEIMの`best_val_ellipse_map.pt`は、固定testおよび実運用の
end-to-end `DEIM → crop → keypoint`評価と、上記のcrop誤差校正に使用します。

## 事前学習済み重みと権限

### 追加権限が不要

- RTMPose-M Face6/106: OpenMMLab公開checkpoint。`prepare_weights.py --rtmpose` で取得できます。
- DEIMv2-L COCO: Intellindust公開checkpoint。`prepare_weights.py --deimv2` で取得できます。
- このrepositoryのfine-tuned DEIMv2-L:
  `outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full/checkpoints/best_val_map.pt`
- 上記DEIMv2-L checkpoint内のDINOv3 ViT-S/16 EMA backbone。本比較では188 tensorをstrictに
  抽出し、モデル2・3の共通初期値として使用します。

```bash
.venv/bin/python keypoint_comparison2/prepare_weights.py --all-public
.venv/bin/python keypoint_comparison2/prepare_weights.py \
  --extract-dinov3-s-from-detector
```

現在のS構成では、手元の学習済みDEIMv2-Lから抽出できるため、新しいdownload権限は不要です。
抽出先は
`checkpoint/DINOv3/dinov3_vits16_face_pose_drive_v2_deimv2_backbone.pth`
です。元checkpointはtrain splitだけで学習されており、val/test keypoint labelは初期値へ入りません。

### 純粋な公式DINOv3-S初期値へ差し替える場合

DINOv3 code/weightは独自のDINOv3 Licenseです。公式download pageから申請し、承認mailに記載された
一時URLまたはdownload済みfileを使用してください。

1. <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/> でDINOv3 accessを申請する。
2. DINOv3 Licenseを組織・用途として確認し、必要なら法務確認を行う。
3. 承認後、ViT-S/16 LVD-1689M checkpointを取得する。
4. 次のいずれかで所定位置へ配置する。

```bash
.venv/bin/python keypoint_comparison2/prepare_weights.py \
  --dinov3-weights /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# または承認mailの一時URL
.venv/bin/python keypoint_comparison2/prepare_weights.py \
  --dinov3-url 'https://<temporary-approved-url>'
```

配置先は抽出weightと同じ上記S checkpoint pathです。
access tokenや一時URLをconfig、shell history、Gitへ保存しないでください。

公式sourceとlicense:

- DINOv3: <https://github.com/facebookresearch/dinov3>
- RTMPose/MMPose: <https://github.com/open-mmlab/mmpose>
- DEIMv2: <https://github.com/Intellindust-AI-Lab/DEIMv2>

## 環境

既存 `.venv` では次の組合せで実装検証しています。

- PyTorch 2.12.1 + CUDA 12.6
- mmengine 0.10.7
- mmcv 2.1.0
- mmdet 3.3.0
- mmpose 1.3.2

OpenMMLabを新規導入する場合はCUDA/PyTorchに合うmmcv wheelを使ってください。

```bash
.venv/bin/python -m pip install -U openmim
.venv/bin/mim install 'mmengine==0.10.7'
.venv/bin/mim install 'mmcv==2.1.0'
.venv/bin/mim install 'mmdet==3.3.0'
.venv/bin/mim install 'mmpose==1.3.2'
```

## モデル別データ変換

YOLO labelをRTMPose/DEIM/DINOの中間表現として使用しません。両exporterは
`metadata/conversion_metadata.json`のCVAT情報を直接読みます。

```bash
.venv/bin/python script/script_data/export_face_keypoints_rtmpose.py \
  outputs/dataset_cache/face_pose_current
.venv/bin/python script/script_data/export_face_keypoints_deim_dino.py \
  outputs/dataset_cache/face_pose_current \
  --output-root outputs/dataset_adapters/face_pose_drive_v2_balanced_811
```

- RTMPose用: `annotations/rtmpose/person_keypoints_{train,val,test}.json`、category id 1
- DEIM/DINO用:
  `outputs/dataset_adapters/face_pose_drive_v2_balanced_811/face_keypoints_{train,val,test}.json`、
  category id 2、
  bbox + ellipse + keypoints + occlusion attributes
- 共通意味規則: `v=0` missing/out-of-image、`v=1` annotated occluded、`v=2` visible

preflightは両exportの画像identity、件数、visibility分布が一致すること、DEIM/DINO側の全annotationが
有効なellipseを持つことも検査します。

## 学習入力の監査結果

固定annotationのSHA-256、件数、画像実体、keypoint順序、split間のsource video/image SHA重複を
preflightで検査します。trainは6,529 face annotation中6,081 instanceが1点以上の座標を持ち、
4,048 instanceが5点すべてを持ちます。学習DataLoaderは`drop_last=false`で、usable instanceを
意図的に捨てません。

trainのvisibility値は`v=0/1/2 = 6,785/3,521/22,339`です。valは
`847/457/2,896`、testは`811/437/2,597`で、CVATの`occluded`を`v=1`として保持しています。
`v=1`は座標lossの教師に含め、visibility targetはfalseにします。`v=0`は座標・visibility lossから
除外します。

この`occluded`は点単位の自然遮蔽教師として有効ですが、遮蔽物のpixel領域を示すmaskではありません。
そのためモデル3のpatch-level occlusion classifierとfeature reconstructionは引き続き
clean cropから生成する矩形/noise maskで監督し、自然遮蔽点はlandmark座標・visibilityと
`natural_occluded_*`評価に使用します。

## Preflight

preflightは次をfail-fastで検査します。

- train/val/testの画像件数
- source video IDと画像SHA-256のsplit間重複
- 5点annotationの存在
- 必須source/checkpoint
- pretrained weightのstrict load
- 選択modelの実forward/backwardとfinite loss

```bash
.venv/bin/python keypoint_comparison2/run.py \
  --stage preflight \
  --models rtmpose_face5 deimv2_joint \
  --device cuda:0

# DINOv3-S抽出後
.venv/bin/python keypoint_comparison2/run.py \
  --stage preflight \
  --models all \
  --device cuda:0
```

asset/data検査だけを行う場合は `--no-forward-preflight` を指定します。

## Full training

既定は全件、150 epoch、AMP bf16、TF32、AdamW、cosine schedule、gradient clipping、EMA、
validation-based best checkpoint、100 epoch以降のみearly stoppingです。
RTX 4090 24GB上の実forward/backwardで確認したうえで、モデル2・3ともphysical/effective
batchを64に固定しています。Sではactivation checkpointを使わず、学習速度を優先します。
top-down 3方式のbest checkpointはvalidation NMEで選択し、joint方式だけvalidation lossを使います。

```bash
.venv/bin/python keypoint_comparison2/run.py \
  --stage train \
  --models all \
  --device cuda:0 \
  --resume
```

個別modelも選択できます。

```bash
.venv/bin/python keypoint_comparison2/run.py \
  --stage train \
  --models dinov3_orformer_rtmcc \
  --device cuda:0
```

通常運用では[学習queue](../configs/training_queues/after_current_models.yaml)が、楕円DEIMの完了直後に
校正用200画像を推論して`ellipse_deim_val_residuals.json`を原子的に生成します。200件以上の
matched ellipseが得られなければqueueを停止し、成功後に全4モデルのpreflightを行って、
`rtmpose_face5`、`dinov3_fpn_rtmcc`、
`dinov3_orformer_rtmcc`、`deimv2_joint`の順で実行します。
各jobは`last.pt`から再開し、正常終了時だけ`training_complete.json`を生成します。

## 固定test評価

test上でthresholdを探索しません。DEIMv2-Lのscore threshold `0.33` とIoU `0.50` は
validationで固定済みです。出力する主指標はdetector precision/recall/F1、matched NME、
PCK@0.05/0.10、keypoint coverage、end-to-end PCK、visibility precision/recall/F1です。
加えて`natural_visible_nme/pck_*`と`natural_occluded_nme/pck_*`を別々に出力します。

```bash
.venv/bin/python keypoint_comparison2/run.py \
  --stage evaluate \
  --models all \
  --device cuda:0
```

4方式の集約結果は
`outputs/keypoint_comparison2/face_pose_drive_v2_balanced_811_gt_ellipse_long_axis_square/comparison.json`、
各modelの予測は `evaluation/test_predictions.jsonl` へ保存されます。

## 画像・動画推論

top-down 3方式では必ず `DEIMv2-L → face crop → keypoint model` を実行します。
joint方式は1 passでbbox/keypoints/visibilityを出力します。

```bash
.venv/bin/python keypoint_comparison2/infer.py input.mp4 \
  --model dinov3_orformer_rtmcc \
  --output outputs/keypoint_comparison2/inference/orformer.mp4 \
  --device cuda:0
```

## Test

download不要のunit testとreal weight testを分離しています。

```bash
.venv/bin/python -m pytest -q keypoint_comparison2/tests/test_models.py -m 'not model_weights'
.venv/bin/python -m pytest -q keypoint_comparison2/tests/test_models.py -m model_weights
```

## 現在の実装検証状況

- RTMPose-M Face6公式weightの362 tensorをloadし、CSPNeXt backbone 350/350 tensorの一致、
  5点final layer転写、GPU forward/backwardを確認済み。
- DINOv3 + Simple FPN + RTMCCとDINOv3 + recovery + RTMCCはDINOv3 ViT-S/16を共通初期値として
  使用します。fine-tuned DEIMv2-LのEMAから188 tensorをstrict抽出し、両方で実GPU
  forward/backward済みです。
- RTX 4090上のbatch 64 bf16計測では、activation checkpointなしのforward/backward peakは
  モデル2が約12.2 GiB、モデル3のoptimizer/EMA更新まで含むpeakは約16.0 GiBでした。
- 4方式すべてのreal-weight preflightとreal-weight pytestが成功しています。
- fine-tuned DEIMv2-Lを読み込んだjoint model（39,778,630 parameters）は、実960px frameで
  bbox + 100 face queries × 5 landmark queries + visibilityのGPU forward/backwardを確認済み。
