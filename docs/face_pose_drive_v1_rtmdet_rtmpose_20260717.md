# face_pose_drive_v1 RTMDet / RTMPose 学習・互換性確認（2026-07-17）

## 結論

固定した 3,090 frame の `face_pose_drive_v1` で RTMDet-S と RTMPose-S を
学習し、独立 test まで評価した。RTMDet-S は test bbox mAP50:95 `0.5366`、
RTMPose-S は GT bbox 入力で test NME `0.0345` だった。RTMDet の予測 bbox を
RTMPose へ渡す two-stage 推論も成功し、両モデルの互換性を確認した。

## 固定データセット

| split | images | face boxes | RTMPose有効instance |
| --- | ---: | ---: | ---: |
| train | 2,460 | 4,027 | 3,744 |
| val | 300 | 503 | 467 |
| test | 330 | 503 | 475 |
| total | 3,090 | 5,033 | 4,686 |

RTMPose有効instanceは、5点のうち少なくとも1点が可視のbboxである。bbox検出は
キーポイント可視性に関係なく全instanceを使用した。独自のtest evaluatorは
zero-visibleを含む503 bboxを入力し、可視な1,976点を評価した。

| split | COCO annotation SHA-256 |
| --- | --- |
| train | `f783ba5bf9f0d5da672b4e34ce46767c2180676c0046dfbf894bf65aff52f423` |
| val | `c2755ce5291c0ee8fdd617c3886e7f8a2f1b2c76f162a91b61cd346708378bd6` |
| test | `cce68a5fef9c7c386cd208cb9a18b2b0fa1e8f54a96e8a5484ccd2adc25eb950` |

## RTMDet-S

- input: 960 x 960
- batch size: 16
- seed: 0
- optimizer: AdamW、base learning rate `2e-4`
- initialization: official RTMDet-S COCO checkpoint
- augmentation: CachedMosaic、random resize/crop、HSV、horizontal flip
- checkpoint selection: validation COCO bbox mAP50:95
- selected epoch: 35（epoch 40で低下したためearly stop）
- parameter count: 8,856,351

初回のbase learning rate `1e-3`ではepoch 5のvalidation mAP `0.329`から
epoch 10で`0.078`へ不安定化した。この試行を停止し、testを見ずに`2e-4`へ
下げて再学習した。低学習率版のvalidation推移はepoch 5/10/15/25/30/35/40で
`0.455 / 0.476 / 0.482 / 0.491 / 0.496 / 0.498 / 0.491`だった。

validationだけでF1最大のconfidence threshold `0.40`を選び、独立testへ固定適用した。

| model | selected | threshold | mAP50:95 | AP50 | AP75 | P | R | F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTMDet-S | epoch 35 | 0.40 | **0.5366** | **0.8947** | **0.5973** | 0.9034 | **0.8549** | 0.8784 | 430 / 46 / 73 |
| YOLO11s Pose | best.pt | 0.44 | 0.5332 | 0.8850 | 0.5874 | **0.9501** | 0.8330 | **0.8877** | 419 / 22 / 84 |
| YOLO11s Detect | best.pt | 0.52 | 0.4789 | 0.8489 | 0.5215 | 0.9426 | 0.7833 | 0.8556 | 394 / 24 / 109 |
| DINOv3 + DEIMv2 unfreeze | epoch 10 | 0.51 | 0.4760 | 0.8791 | 0.4632 | 0.8957 | 0.8370 | 0.8654 | 421 / 49 / 82 |
| DINOv3 + DEIMv2 freeze | epoch 15 | 0.67 | 0.4562 | 0.8440 | 0.4222 | 0.9313 | 0.7276 | 0.8170 | 366 / 27 / 137 |

MMDetection内蔵COCO testでもmAP50:95 `0.537`、AP50 `0.895`、AP75
`0.598`となり、独自の共通evaluatorと一致した。RTMDet-Sはこの比較ではbbox
mAP、AP50、AP75、recallが最良で、YOLO11s PoseはprecisionとF1が最良だった。

## RTMPose-S

- task: top-down face 5-keypoint estimation
- keypoints: left eye、right eye、nose tip、left mouth corner、right mouth corner
- input: 256 x 256
- batch size: 64
- epochs: 100
- seed: 0
- backbone: CSPNeXt-S、ImageNet pretrained
- head: RTMCCHead / SimCC
- augmentation: bbox scale `0.75-1.25`、rotation `+-20 deg`、horizontal flip
- checkpoint selection: validation NME minimum
- selected epoch: 91、validation NME `0.038063`
- parameter count: 5,209,047

### 独立test

NMEとPCKはface bbox対角長で正規化した。`conditional`はIoU 0.5以上で
RTMDetとGTが対応した顔だけの姿勢精度、`end-to-end PCK`は未検出顔の可視点も
失敗として分母へ含めた値である。

| input bbox | evaluated faces | visible-point coverage | NME | PCK@0.03 | PCK@0.05 | PCK@0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT bbox | 503 / 503 | 1,976 / 1,976 (1.0000) | 0.0345 | 0.7839 | 0.8517 | 0.9226 |
| RTMDet bbox, conditional | 430 / 503 | 1,836 / 1,976 (0.9291) | 0.0307 | 0.8077 | 0.8791 | 0.9341 |
| RTMDet bbox, end-to-end | 430 / 503 | 1,836 / 1,976 (0.9291) | - | 0.7505 | 0.8168 | 0.8679 |
| YOLO11s Pose, conditional | 419 / 503 | 1,826 points | 0.0609 | - | 0.5728 | 0.8538 |

RTMDet bboxでconditional NMEがGT bbox入力より小さいのは、未検出73顔を除いた
比較的容易な部分集合だからであり、two-stage化による精度向上を意味しない。
MMPose内蔵testのNMEは`0.041229`だった。これはMMPoseの`bbox_size`正規化で、
上表のbbox対角正規化とは定義が異なる。

## 互換性確認

| 項目 | 結果 | 備考 |
| --- | --- | --- |
| YOLO Pose 5点からCOCO keypointsへの変換 | OK | bbox一致、split、可視性を検証 |
| MMPose dataset/model build | OK | 5点metadataとflip index `[1, 0, 2, 4, 3]` |
| RTMPose 100 epoch学習 | OK | AMP、最大GPU memory約1.6 GB |
| RTMDet学習 | OK | AMP、最大GPU memory約8.3 GB |
| RTMDet bboxからRTMPose top-down推論 | OK | 330 test imagesで実走 |
| MMDetection/MMPose内蔵test | OK | 独自evaluatorと照合 |
| PyTorch 2.12 worker終了 | 対策済み | `persistent_workers=False`で正常終了確認 |

実行環境はPyTorch `2.12.1+cu126`、MMEngine `0.10.7`、MMDetection
`3.3.0`、MMPose `1.3.2`、RTX 4090だった。

PyTorch 2.6以降の`torch.load(weights_only=True)`既定値とMMEngine
checkpoint内の`HistoryBuffer`に互換性差がある。今回生成した信頼済みcheckpointを
評価する際は次のように明示した。

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv/bin/python \
  .venv/lib/python3.12/site-packages/mmpose/.mim/tools/test.py \
  configs/rtmpose/rtmpose_s_face_pose_drive_v1_256.py \
  outputs/experiments/rtmpose_s_face_pose_drive_v1_256_20260717/best_NME_epoch_91.pth
```

この環境変数は出所を信頼できるcheckpointにだけ使用する。

## 成果物

- RTMDet config: `configs/rtmdet/rtmdet_s_face_pose_drive_v1_960.py`
- RTMDet best: `outputs/experiments/rtmdet_s_face_pose_drive_v1_960_low_lr_20260717/best_coco_bbox_mAP_epoch_35.pth`
- RTMDet test metrics: `outputs/experiments/rtmdet_s_face_pose_drive_v1_960_low_lr_20260717/eval_test_best/metrics.json`
- RTMPose config: `configs/rtmpose/rtmpose_s_face_pose_drive_v1_256.py`
- RTMPose best: `outputs/experiments/rtmpose_s_face_pose_drive_v1_256_20260717/best_NME_epoch_91.pth`
- RTMPose GT bbox metrics: `outputs/experiments/rtmpose_s_face_pose_drive_v1_256_20260717/eval_test_gt_bbox/metrics.json`
- RTMDet + RTMPose metrics: `outputs/experiments/rtmpose_s_face_pose_drive_v1_256_20260717/eval_test_rtmdet_bbox/metrics.json`
