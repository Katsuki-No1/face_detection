# script_data

CVAT に投入するデータ、CVAT からダウンロードしたアノテーション、学習・評価用データセットの変換に関わるスクリプトを置きます。

Examples:

- `clip_frame.py`: 動画からフレーム抽出
- `export_face_keypoints_rtmpose.py`: canonical CVAT metadataからRTMPose専用COCOを直接生成
- `export_face_keypoints_deim_dino.py`: canonical CVAT metadataからDEIM/DINO専用COCOを直接生成
- `convert_face_pose_metadata_to_rtmdet_hf.py`: Head/Face metadataからRTMDet-HF用2-class COCOへの変換
- `count_head_annotations.py`: CVAT zip 内の Head アノテーション数を集計
- `convert_cvat_annotations.py`: CVAT XML を評価用 JSON に変換

## `count_head_annotations.py` と Weights & Biases

通常のローカル集計はこれまでどおり実行できます。

```bash
.venv/bin/python script/script_data/count_head_annotations.py data/annotation_data
```

W&Bへ記録する場合は、最初の1回だけログインしてから `--wandb` を付けます。APIキーはコードや
Git管理ファイルへ書かないでください。

```bash
.venv/bin/wandb login --verify
.venv/bin/python script/script_data/count_head_annotations.py data/annotation_data \
  --wandb \
  --wandb-entity sato-waseda-university \
  --wandb-project face_detection \
  --dataset-version face_dataset_v7 \
  --csv-output outputs/head_annotation_counts.csv
```

`--wandb-entity`と`--wandb-project`の既定値は、それぞれ`sato-waseda-university`と
`face_detection`です。そのため、通常は`--wandb`だけでこの接続先へ記録できます。

合計値はW&Bのスカラー、`dataset_version`はrun configとして記録されます。複数ZIPを一括処理した
場合だけ、ZIP別の結果もTableとして記録されます。画像単位の結果はW&Bへ送信せず、必要な場合は
`--csv-output`でローカルCSVへ保存します。
ネットワークを使わずローカルへ保存する場合は `--wandb-mode offline` とし、後から
`wandb sync wandb/offline-run-*` で同期できます。接続せず実装だけ確認する場合は
`--wandb-mode disabled` を使用できます。

## モデル別face keypoint export

`face_pose_drive_v2`以降、RTMPose・DEIM・DINOの教師をYOLO labelから再構成しません。
共通の意味規則だけをcanonical metadataから読み、出力schemaはモデル系統ごとに分離します。

- missing/out-of-image: `visibility=0`
- 座標付きの`occluded`: `visibility=1`
- 座標付きのvisible: `visibility=2`
- RTMPose: category id `1`、5点COCO
- DEIM/DINO: category id `2`、5点に加えて
  `[cx, cy, semi_major, semi_minor, theta_rad]` の楕円教師を保持

```bash
.venv/bin/python script/script_data/export_face_keypoints_rtmpose.py \
  datasets/face_pose_drive_v2
.venv/bin/python script/script_data/export_face_keypoints_deim_dino.py \
  datasets/face_pose_drive_v2
```

両exporterは`metadata/conversion_metadata.json`を直接読み、画像実体、split、元動画の分離を
検査します。`convert_face_pose_yolo_to_coco*.py`は旧dataset再現用のlegacy toolであり、
RTMPose/DEIM/DINOの現行学習入力には使用しません。

## Legacy: `convert_face_pose_yolo_to_coco.py`

`face_pose_drive_v1` の既存train/val/testを変更せず、DINOv3+DEIMv2用のface category ID `2`を持つ
COCO Detection JSONへ変換します。画像は複製せず、dataset rootの `train.json`、`val.json`、
`test.json` から既存の `images/<split>` を参照します。

```bash
.venv/bin/python script/script_data/convert_face_pose_yolo_to_coco.py datasets/face_pose_drive_v1
```

変換時に元動画ID、画像SHA-256、実画像パスのsplit間重複を検査します。検査に失敗した場合、生成途中の
JSONは反映されません。判断根拠と件数は `metadata/coco_conversion_summary.json` に保存されます。

## `convert_face_pose_metadata_to_rtmdet_hf.py`

`face_pose_drive_v1`の`conversion_metadata.json`に保持されているHeadとFaceを直接読み、
`head=1`、`face=2`のRTMDet-HF用COCO Detectionデータセットを生成します。Faceがない画像でもHeadが
あればHead教師を保持します。既存のtrain/val/testと元動画単位splitは変更しません。

```bash
.venv/bin/python script/script_data/convert_face_pose_metadata_to_rtmdet_hf.py \
  datasets/face_pose_drive_v1 \
  datasets/rtmdet_hf_drive_v1
```

出力先がすでに存在する場合は上書きしません。画像は既定でハードリンクされ、ハードリンクを利用できない
環境ではコピーへフォールバックします。変換件数とsplit漏洩監査は出力先の
`conversion_summary.json`へ保存されます。

## `clip_frame.py`

各動画を等間隔の100区間に分け、各区間の中央フレームを1枚ずつ抽出します。DINO特徴量、品質、
pHash、ランダム選択による候補検査は行いません。同じ動画からは常に同じsource frameを選びます。

```bash
.venv/bin/python script/script_data/clip_frame.py data/input/2026_06_24
```

選択するsource frame番号は動画の総フレーム数から計算し、整数丸めによる隣接間隔の差は最大1フレーム
です。100枚を一意に選べない100フレーム未満の動画はエラーにします。選択した100枚は動画を先頭から
1回デコードしてまとめて出力します。

最初にプロジェクト共通の `data/video_name_history.csv` に加え、過去の全出力フォルダにある
`video_names.csv` と旧 `selection_manifest.jsonl` を検索します。続いて入力動画のファイル全体から
SHA-256を計算し、次のいずれかに該当すれば、クリップを作らずに処理全体を中断します。

- 拡張子・英字大小・Unicode表記を正規化した動画名が過去履歴と一致する
- ファイル名を変更していても、動画内容のSHA-256が過去履歴と一致する
- 今回の入力フォルダ内で動画名または動画内容が重複する

重複時は `重複あり` と明示し、今回の対象動画、一致理由、履歴にある過去の動画名を表示します。
`--allow-known-videos` で意図的に再抽出する場合も、重複を表示してから続行します。

重複検索を通過した後、入力フォルダ内の全動画へ高速なパケット/NAL検査を行い、結果を動画ごとに
`Packet check OK` と表示します。1本でも破損していれば新しいフレームを出力せず、処理全体を
中断します。

検査を通過した動画は共通の一時ディレクトリへ抽出します。抽出時の完全デコードと枚数検査が全動画で
成功した場合だけ `data/output` へ反映します。途中で失敗した場合は
`Staged extraction aborted` と表示して一時結果をすべて破棄し、既存出力を変更しません。
`--full-decode-check` を指定すると、必要に応じて抽出前にも追加の完全デコード検査を実行できます。

主な既定値:

| option | default | meaning |
|---|---:|---|
| `--frame-count` | 100 | 動画ごとの等間隔抽出枚数 |
| `--quality` | 2 | ffmpegのJPEG品質（小さいほど高品質） |

枚数を変える場合は `--frame-count` を指定します。同じ動画を同じ出力先へ再実行した場合、以前の
抽出結果は全動画の成功後に置き換わります。

抽出成功時は、出力フォルダの `video_names.csv` にその回の元動画名・SHA-256・ファイルサイズを保存し、
プロジェクト共通の `data/video_name_history.csv` にも追記します。旧履歴にはSHA-256がないため、
旧動画は動画名で検査し、新しく成功した抽出から改名済みの同一ファイルも検出できるようになります。
意図した再抽出の場合だけ `--allow-known-videos` を指定します。旧オプション
`--allow-known-video-names` も互換エイリアスとして利用できます。履歴CSVを別の場所へ置く場合は
`--video-history-csv` で変更できます。

## 過去の動画名の復元

`recover_video_name_history.py` は、現存する `data/input`、抽出済みフレーム名、過去の
`selection_manifest.jsonl`、旧形式のフレームディレクトリから動画名履歴を再構築します。
Codexのローカル作業履歴も明示的に対象へ加える場合は次のように実行します。

```bash
.venv/bin/python script/script_data/recover_video_name_history.py \
  --history-root ~/.codex
```

確定できた名前は `recovery_status=exact`、フレーム接頭辞しか残っておらず全文を確定できないものは
`recovery_status=unresolved` として `data/video_name_history.csv` に保存されます。フレーム名にある
8桁値は元動画名stemのSHA-1先頭8桁なので、履歴に残る名前との照合には使えますが、その値だけから
元の名前全文を逆算することはできません。復元処理を再実行しても、今後のクリップ作成で記録された
`content_sha256` と `file_size_bytes` は維持されます。
