# 基準Face Poseデータセット

新規学習の基準データセットは
`face_pose_drive_v2_balanced_811_training_export_20260724.zip`である。
データセット版は`face_pose_drive_v2_balanced_811`、分割は動画・シーンの
グループを維持した4,237 / 530 / 528画像で固定する。

## 読み取り専用ポリシー

`datasets/`内のZIPと既存データセットには書き込まない。準備スクリプトは
ZIPを読み取り、内容が同梱`SHA256SUMS`と一致するコピーを
`outputs/dataset_cache/`へ展開する。モデル固有の派生JSONと、ローカルパスを
持つYOLO設定は`outputs/dataset_adapters/`へ生成する。

```bash
.venv/bin/python scripts/prepare_default_face_pose_dataset.py
```

再検証のみを行う場合:

```bash
.venv/bin/python scripts/prepare_default_face_pose_dataset.py --verify-only
```

共通定義は`configs/datasets/default_face_pose.yaml`に置く。DEIM/DINOと
YOLO26 Poseはこの設定を継承し、RTMDetとRTMPoseも同じ
`outputs/dataset_cache/face_pose_current`を参照する。

## モデル別の入力

- YOLO Pose: ZIP同梱`images/`と`labels/`
- RTMDet: ZIP同梱`annotations/rtmdet_hf/`
- RTMPose: `outputs/dataset_adapters/face_pose_drive_v2_balanced_811/rtmpose/`
- 標準DEIMv2・楕円DEIMv2:
  `outputs/dataset_adapters/face_pose_drive_v2_balanced_811/`

DEIM用とRTMPose用のアダプターはZIP内の正規
`conversion_metadata.json`から生成する。
元ZIPの`annotations/face_dino_v1/`は頭矩形を持つため、顔矩形を前提とする
DEIMv2へ直接流用しない。
