# script_experiment

評価、比較、チューニング、可視化、手動レビューなど、実験を進めるためのスクリプトを置きます。

モデル比較や条件変更のスクリプトが増える場合は、まずここに追加します。複数の実験で共通化できる処理だけを、必要に応じて別モジュールへ切り出します。

Examples:

- `evaluate_face_detections.py`: Head/Face 検出評価
- `summarize_cvat_recall_tuning.py`: CVAT 基準のチューニング集計
- `visualize_detections.py`: 検出結果の可視化
