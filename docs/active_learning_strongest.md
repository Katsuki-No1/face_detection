# 現行最強modelを使うActive Learning

## 目的

このpipelineは、未ラベル元動画から現行のproduction最強modelが不安定なframeを抽出し、
教師データ作成の優先順位を付ける。model更新を頻繁に行う前提で、model固有処理と
選択処理を分離している。

- model registry:
  `configs/model_registry/active_learning_strongest_models.yaml`
- pipeline config: `configs/active_learning_strongest.yaml`
- CLI: `scripts/run_strongest_active_learning.py`
- 共通prediction schema: `src/active_learning/strongest_types.py`
- model adapter: `src/active_learning/strongest_adapters.py`
- model非依存の判定・選択・export:
  `src/active_learning/strongest_pipeline.py`

## 処理契約

### 1. Montage

元動画のtimelineを10等分し、各区間中央から連続60秒を取得する。10 clipを元の時間順で
連結するため、出力は厳密に600秒となる。元動画が600秒未満で、非重複の60秒clipを
10本作れない場合は停止する。

`montage_manifest.json`には各clipについて次を保存する。

- 元動画の区間開始・終了
- 実際に切り出した元動画時刻
- montage内の開始・終了時刻
- 元動画のsize、mtime、先頭・末尾を使うcontent-aware fingerprint
- montage codec条件

この対応表により、選択したmontage frameを元動画時刻へ戻し、最終JPEGは再encode済み
montageではなく元動画から原解像度でdecodeする。

### 2. 推論とmodel差し替え

既定では600秒を5fps、合計3,000 frame推論する。adapterはbatchごとに次の共通形式を返す。

```text
FaceObservation
  confidence
  bbox_xyxy = [x1, y1, x2, y2]
  ellipse = [cx, cy, semi_major, semi_minor, theta_radians]
  keypoints = [[x, y], ...]
  keypoint_scores = [p(point exists), ...]
```

座標原点は左上、xは右、yは下方向。ellipse角度はimage座標上のradian、周期はπ、
axisは半径で`semi_major >= semi_minor`とする。

新modelを追加する場合は、`StrongestModelAdapter` contractを満たすclassを実装し、
registryへ以下を記録する。

- `adapter_class`: `python.module:ClassName`
- `capabilities`: `face, ellipse, keypoints`
- checkpoint/config/threshold sourceを含む`assets`
- model固有の`options`
- validationだけで固定したface/keypoint threshold

`active_model`を変更すると推論cache signatureが変わる。旧cacheを暗黙に再利用せず、
`--stage infer --force`を要求する。

現在のQuery Refinement checkpointは構造とdata normalizationをcheckpoint内へ
埋め込んでいる。可変の外部training YAMLではなく、このimmutable configからmodelを
組み立てる。これにより、後続実験でYAMLが変わっても旧checkpointのstate contractを
維持する。

## 4条件

### 時系列的な矛盾・ellipseの不安定

face threshold以上のellipseを5fpsで追跡する。montageのclip境界とscene changeでは
trackを必ずresetするため、元動画上で連続していない時刻やshot cutを異常扱いしない。
3回以上観測したtrackに対して次を候補化する。

- 前後で同じtrackがある中間frameだけ検出がない
- ellipse中心が直径に対して急移動する
- major/minor axisが急変する
- 円に近くangleが意味を持たない場合を除き、angleが急変する

閾値はpipeline YAMLで独立に設定できる。

### 低confidence

単純に最小scoreを取らず、validation固定face thresholdに対する比が既定
`0.70〜1.30`のdecision boundary候補を対象にする。threshold比1.0に近いほど高scoreと
する。現在のface threshold `0.33`はv2 validationだけで選択され、
`precision >= 0.90`を保つ点のうちrecall最大のoperating pointである。test結果で
再選択しない。

### Faceのみ・keypointなし

face confidenceが固定threshold以上で、5点すべてのpoint-existence scoreが
keypoint threshold未満のfaceを対象にする。現在のQuery checkpointはlegacy COCO
`v > 0`のpoint-existence contractであり、visible/occludedの二値分類ではない。

### 画面端

回転ellipseから厳密なaxis-aligned extentを計算する。上下左右のいずれかが画面の
最外周1%へ到達、または画面外へ出た場合だけ対象にする。bboxだけでなくellipse形状を
用いるため、「かなり画面端」という条件を明示的に固定できる。

## 配分・重複排除

元動画ごとの上限は50 frame。比率は4条件それぞれ25%で、50件ならquotaは
`13 / 13 / 12 / 12`になる。複数条件に該当するframeは1回だけ選ぶ。希少な条件から
quotaを割り当て、不足分は他の条件を満たす未選択frameで補完する。

条件を1つも満たさないframe、弱い一様sample、同一pixel、設定した最小時間間隔内の
近接frameで水増ししない。そのため候補不足時は50件未満となり、`summary.json`へ
shortfallを明記する。

既定の最小時間間隔は元動画時刻で10秒とする。異なるfailure条件に該当する場合でも、
選択済みframeから10秒未満の候補は採用しない。最終decodeで10秒を下回らないよう、
連続値のmapping時刻ではなく`source_frame_id / source_fps`のframe格子上で判定する。

## 出力資料とprovenance

- `frames/`: アノテーション対象の原画像
- `evidence/`: ellipse、keypoint、confidenceを描いた確認画像
- `selection_manifest.csv/jsonl`: frame ID、元動画時刻、4条件の全score、主理由、
  全該当理由、標準化prediction
- `selection_report.html`: 原画像とevidenceを並べたoffline review資料
- `summary.json`: 元動画別quota、補完数、shortfall、全体件数
- `_active_learning/run_provenance.json`: resolved config、seed、座標規約、
  model assetのSHA-256、validation固定threshold

checkpoint、生成frame、推論JSONL、montageはすべて`data/`以下へ出力し、Gitには
commitしない。
# TensorRT FP16 production path

The current production pair can be prepared without occupying the training
GPU:

```bash
.venv/bin/python scripts/export_active_learning_onnx.py --component all --force
.venv/bin/python scripts/build_active_learning_tensorrt.py --prepare-only --force
```

The first command exports stable, dynamic-batch ONNX contracts:

- DEIMv2-L ellipse: `images -> pred_logits, pred_boxes, pred_ellipses`
- Query Refinement: `crops -> decoded_coordinates, keypoint_logits`

It runs ONNX checker, strict shape inference, and PyTorch CPU versus ONNX
Runtime CPU agreement at batch 1 and 2. The second command uses NVIDIA
ModelOpt AutoCast for TensorRT 11 strongly-typed mixed FP16 graphs while
retaining FP32 model I/O. The exact min/opt/max shapes are recorded in
`configs/tensorrt/active_learning_deimv2_l_query_refinement_fp16.yaml`.

After the training job releases the RTX 4090, the complete engine-build and
six-video run is:

```bash
scripts/run_active_learning_tensorrt_6videos.sh
```

The engine build is intentionally done on the target GPU. It validates
PyTorch FP32 against TensorRT FP16 at every detector and query min/opt/max
profile point before starting video inference. A failed numerical gate stops
the six-video run.

Active-learning datasets are read from `data/input/active_learning/` and
results are written to `data/output/active_learning/`. The six-video result
is one CVAT-uploadable directory:

```text
data/output/active_learning/
  active_learning(dataset=3000frame)7-17/
    frames/
    selection_manifest.csv
    video_names.csv
```

The JPEGs are lightweight quality 85 without duplicate evidence overlays.
`selection_manifest.csv` records the primary and matched extraction reasons.
Before any montage is created, source stems are Unicode-normalized and checked
for duplicates both within the current batch and against
`data/video_name_history.csv` plus past clip/active-learning manifests. A clean
completed `montage` or `all` run records its source names in both the dataset's
`video_names.csv` and the shared history. Use `--allow-known-videos` only for
an intentional rerun.
