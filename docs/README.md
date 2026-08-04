# 実験ノート

このREADMEは、主要な検証結果とその解釈を時系列で記録するための索引である。
正式な数値、checkpoint、設定、データセットのprovenanceは、各実験の
`metrics.json`と`success.json`を一次資料とする。

## 2026-08-02: YOLO26m PoseとDEIMv2-L + Query Refinementのtest比較

### 目的

YOLO26m PoseとDEIMv2-L + Query Refinementについて、valでモデルごとに
選択したFace閾値を変更せずtestへ適用し、同じデータと評価軸で比較した。
この比較ではtestの結果を使用した閾値の再選択を行っていない。

### 評価条件

- データ: Balanced-811 Face DINO V3 test
- 内訳: 6 groups / 528 images / 806 Heads
- Faceあり: 766 Heads
- Faceなし: 39 Heads
- 曖昧なため評価対象外: 1 Head
- annotation SHA-256: `bd32bd1591686947bfcc79cbbcd79e1059bba1569e8bd06127d3327f2b776487`
- YOLO26m Pose Face閾値: `0.37`（valで選択）
- DEIMv2-L Face閾値: `0.33`（valで選択）
- keypoint presence閾値: `0.70`
- keypoint誤差の正規化: `1.20 * max(GT Head width, GT Head height)`
- testでの閾値探索・再調整: なし

YOLO26mはcheckpoint
`runs/yolo/yolo26_pose/face_pose_drive_v2_balanced_811/yolo26m_pose_full/weights/best.pt`
（SHA-256: `3e9c838109ee3d326de699afba6a6bd4098ff924557a757cc78757079c65cffb`）を使用した。

DEIMv2-LはFace detector checkpoint
`outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_960_ellipse_from_coco/checkpoints/best_val_ellipse_map.pt`
（SHA-256: `ffa0ae9fe63df3f7af7ff7b772b80717243dcf544b2c9fcdf944de8899ddd8b7`）と、
Query Refinement checkpoint
`outputs/landmark_detection/dinov3_vits16_face_pose_drive_v2_balanced_811_gt_ellipse_long_axis_square_coco_visibility/query_refinement/checkpoints/best_nme.pt`
（SHA-256: `0bfb912ef395b352a8a41a21f016f1e32c399fbb22dc8247dc90192e38f6b4b2`）を使用した。

### test結果

| 指標 | YOLO26m Pose（Face 0.37） | DEIMv2-L + Query Refinement（Face 0.33） | 優位 |
|---|---:|---:|---|
| Face Macro F1 | 0.6876 | **0.7923** | DEIMv2-L |
| Face-present recall | 0.8799 | **0.9478** | DEIMv2-L |
| No-face recall | **0.9487** | 0.8974 | YOLO26m |
| False-face rate | **0.0513** | 0.1026 | YOLO26m |
| Face TP / FP / FN / TN | 674 / 2 / 92 / 37 | 726 / 4 / 40 / 35 | ― |
| Keypoint Macro F1 @ NME 0.05 | **0.8853** | 0.8722 | YOLO26m |
| Keypoint Macro F1 @ NME 0.10 | **0.9132** | 0.8981 | YOLO26m |
| Keypoint micro recall @ NME 0.05 | 0.8504 | **0.8556** | DEIMv2-L + QR |
| Keypoint micro recall @ NME 0.10 | 0.8780 | **0.8833** | DEIMv2-L + QR |
| Matched keypoint NME @ 0.10 | 0.01427 | **0.01216** | DEIMv2-L + QR |

### 解釈

val選択閾値をtestへ固定した今回の条件では、Face検出はDEIMv2-Lが優位である。
YOLO26mと比べてtrue positiveが52件多く、false negativeは92件から40件へ減った。
一方、YOLO26mはFaceなしHeadに対する誤検出抑制とkeypoint Macro F1で優位だった。
DEIMv2-L + Query Refinementはkeypointのmicro recallと、正しく対応した点の座標精度
（NME）で優位だった。

共通固定Face閾値`0.80`ではYOLO26mが優位になるが、これは信頼度校正を含む別の
運用点評価である。その結果と、モデルごとのval選択閾値を使用した本比較を混同しない。
本比較の主結論は「valで選択した運用点では、testでもFace検出はDEIMv2-Lが優位」
である。

### 成果物

- [YOLO26m test metrics](../outputs/yolo/yolo26_pose/face_pose_drive_v2_balanced_811/yolo26m_pose_full/evaluation/test/baseline_aligned/metrics.json)
- [YOLO26m test success marker](../outputs/yolo/yolo26_pose/face_pose_drive_v2_balanced_811/yolo26m_pose_full/evaluation/test/baseline_aligned/success.json)
- [DEIMv2-L + Query Refinement test metrics](../outputs/landmark_detection/dinov3_vits16_face_pose_drive_v2_balanced_811_gt_ellipse_long_axis_square_coco_visibility/query_refinement/evaluation/test/baseline_aligned/metrics.json)
- [DEIMv2-L + Query Refinement test success marker](../outputs/landmark_detection/dinov3_vits16_face_pose_drive_v2_balanced_811_gt_ellipse_long_axis_square_coco_visibility/query_refinement/evaluation/test/baseline_aligned/success.json)

両方の`success.json`は`status: success`であり、inventory、有限値、各対応指標の
assertionを通過している。
