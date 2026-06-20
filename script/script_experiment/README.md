# script_experiment

評価、比較、チューニング、可視化、手動レビューなど、実験を進めるためのスクリプトを置きます。

モデル比較や条件変更のスクリプトが増える場合は、まずここに追加します。複数の実験で共通化できる処理だけを、必要に応じて別モジュールへ切り出します。

Examples:

- `evaluate_face_detections.py`: Head/Face 検出評価
- `summarize_cvat_recall_tuning.py`: CVAT 基準のチューニング集計
- `visualize_detections.py`: 検出結果の可視化
- `rtmdet_rtmpose_experiment.py`: RTMDet + RTMPose-face のデータ変換、学習、推論、評価、可視化をまとめた実験 CLI

RTMDet + RTMPose-face 実験の出力は `output_experiment/` に保存します。

```bash
.venv/bin/python script/script_experiment/rtmdet_rtmpose_experiment.py run-all \
  --data-root . \
  --output-dir output_experiment \
  --mode sanity \
  --device cuda:0
```

本学習前の変換・annotation・評価前段の検証だけを行う場合は、次を使います。

```bash
.venv/bin/python script/script_experiment/rtmdet_rtmpose_experiment.py post-impl-validation \
  --data-root . \
  --output-dir output_experiment
```
