# 現行baselineの構成記録

2026-10-09に顔検出成分をMAL α=2へ更新し、keypoint成分を保持しました。
2026-10-10に `baseline` ブランチを作成し、同日、現行モデルの入口と文書に整理しました。

## 定義

baselineは顔検出とkeypoint検出を合わせた一式です。
`全画像FACE/楕円（MAL α=2） → Face crop DINOv3 Query Refinement（A1/A2）`。
外部HEAD検出器は使用しません。

- 顔検出: `mal_alpha2_ellipse_from_existing_wider`、validation bbox mAPで選択したepoch45 EMA。
- Keypoint: A1/A2版 `query_refinement` の `best_nme.pt`。
- 正規の設定・重みのパスとSHA: [local_baseline.yaml](../configs/model_registry/local_baseline.yaml)。
- 顔検出score: 0.33、およびvalidationでPrecision > 0.90のRecall最大となる0.2666015625。
- Keypoint presence score: 0.70。5点はleft_eye、right_eye、nose、left_mouth、right_mouth。

顔検出部の評価値は[性能表](baseline_performance.md)を参照してください。
組み合わせ全体の推論・評価はregistry上で未実施です。checkpointの識別情報やこの状態を、
文書整理によって変更していません。過去の構成・実験資料はGit履歴に残ります。
