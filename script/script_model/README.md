# script_model

モデル推論や、モデル出力の内部確認に関わるスクリプトを置きます。

Examples:

- `run_scrfd_mediapipe.py`: SCRFD/MediaPipe の推論実行
- `dump_detector_stages.py`: 検出器の各段階をデバッグ出力
- `run_dinov3.py`: DINOv3 の画像特徴量を JSONL に出力

## DINOv3

`run_dinov3.py` は DINOv3 公式 repo の PyTorch Hub entrypoint を使って、画像ごとの CLS embedding を出力します。
既定では `script/script_model/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` を `dinov3_vitb16` として読み込みます。

公式 repo をローカルに clone 済みの場合:

```bash
.venv/bin/python script/script_model/run_dinov3.py \
  --source path/to/images \
  --dinov3-repo path/to/dinov3 \
  --output output/dinov3_features.jsonl
```

`--dinov3-repo` を省略すると、PyTorch Hub が `facebookresearch/dinov3` を GitHub から取得します。
patch token の平均特徴量も保存する場合は `--include-patch-mean` を付けます。
