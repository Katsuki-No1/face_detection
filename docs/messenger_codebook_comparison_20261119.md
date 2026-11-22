# Messenger：16方向重み付け・16次元圧縮・codebook比較

状態更新（2026-11-20確認）：全6条件のR/gが検証plateauに到達。11月19日21:51 JSTに全数値評価・監査と新規26重みの削除が完了。詳細は `docs/messenger_codebook_results_20261120.md`。以下のgate開始記録は過去の実行経過。

## 目的とチャットで確定した判断

既存の特徴復元よりも、復元後のkeypoint座標を良くできるかを確かめる。特徴MSEの改善だけで採否を決めず、位置とpresenceを分離する。DINO特徴全体を復元する方式と、座標に重要な情報を重視する方式を比較し、圧縮による損失・離散化の効果・辞書更新の効果・容量の効果を段階的に調べる。

過去の感度計測でいう「16方向」は384チャンネルの線形結合であり、16チャンネルの選択ではない。共通16方向への感度集中は、小さな摂動に対する局所的な性質である。全置換時の座標維持や、16次元だけで十分という保証ではない。この理由から連続値16次元の対照を置く。

ユーザーは既存fit・動画分割、上位16方向の相対重み4倍を選択した。dataset全体への拡張、条件別辞書、32方向以上への拡張は今回行わない。ユーザーの構想はVQ-VAEに近い圧縮であり、384次元のまま重要方向を重くする方式は独立した比較条件として残す。

既存keypoint headがあるため、heatmap生成器は追加しない。DINO・headを固定し、headを通る座標損失で復元器を更新する。ORFormerの事前辞書の固定をそのまま再現する実験ではなく、計算的に初期化した同じ辞書を固定する条件と、位置損失も辞書に伝える条件を比較する。

## 7条件

| 番号 | 実装名 | 特徴経路 | 辞書 |
|---|---|---|---|
| ① | saved_control | 保存済み384次元連続復元 | なし・再学習なし |
| ② | weighted16 | 384次元連続復元、重要16方向のMSEを4倍 | なし |
| ③ | continuous16 | encoder→16次元連続値→decoder→384次元 | なし |
| ④ | vq4096_fixed | encoder→16次元→最近傍コード→decoder→384次元 | 4,096、固定 |
| ⑤ | vq4096_trainable | ④と同じ初期値・構造 | 4,096、更新 |
| ⑥ | vq8192_fixed | encoder→16次元→最近傍コード→decoder→384次元 | 8,192、固定 |
| ⑦ | vq8192_trainable | ⑥と同じ初期値・構造 | 8,192、更新 |

①は `outputs/diagnostics/messenger_position_primary_20261118/det_position/` の保存数値を使う。位置＋0.01 MSE＋0.01 cosineの回帰で、既に重みは削除済み。②との違いを方向重みに限定できるため、この対照を選んだ。cosineのみの回帰や保持候補scale Cとの同一比較ではない。①を再学習する経路は実装していない。

## モデルと損失

②は既存scale復元器を同一seedで初期化する。fit勾配Gramの上位固有ベクトルをU（384×16）とし、誤差eの特徴損失を `(||e||² + 3||Uᵀe||²) / 432` とする。16方向と残り368方向の比は4:1、384方向の平均重みは1。cosine補助は全384次元に適用する。

③〜⑦は、最終DINO mapの学習可能な384→16射影に既存3スケールAttentionからの16次元補正を加える。射影をUᵀ、補正出力をゼロで初期化する。復号は16→384線形層＋MLPで、線形層をU、MLP最終層をゼロで初期化する。全出力経路が16次元を通り、元384次元を復号結果へ直接加える経路はない。学習で射影は回転できるので、学習後も固定の感度16方向に限定する方式ではない。

辞書の中身は16次元ベクトルであり、384次元への復号は別の学習可能decoderが担当する。辞書があれば正しく安定復元できる、とは仮定しない。③との位置精度比較で離散化の効果を判断する。

初期辞書は人工fit256顔の遮蔽前最終DINO特徴（65,536パッチ）を同じUとfit平均で射影して作る。PyTorchの分割Lloyd k-means、seed42のfitサンプル初期化、最大30反復、相対変化1e-5、空クラスタは前値維持。dev/evalは初期化に使わない。同じKの固定／更新条件で辞書・encoder・decoderの初期値を共通化する。推論はhard nearest neighbor。学習はencoderへのstraight-through勾配に加え、更新条件では選択されたコード値へ位置損失を直接流す。最近傍の選択インデックス自体を微分しているわけではない。

実装上の初期設定は以下。追加係数は今回の技術的な初期値で、効果が検証済みという意味ではない。

- 復元器：遮蔽点／可視点を等重みとする座標誤差＋0.01 MSE＋0.01 cosine。VQ条件のみ `0.01 × (code loss + 0.25 × commitment)` を追加する。
- 人工学習では既知mask内だけを復元して固定headへ渡す。遮蔽前の正解特徴をheadへ渡す比較は行わない。
- 自然遮蔽の利用方法を従来と揃えるため、各条件で既存natural gate gも学習する。旧5点弱教師＋0.01 soft BCE、位置主目的、headは固定。全置換full_Rとg融合後を別々に評価し、gの効果と復元器の効果を混同しない。
- AdamW、seed42、実効batch16／microbatch4、R lr1e-4、g lr5e-4、weight decay0.01。R/gの最良重みはdev位置で選択。既存plateau判定を使用し、上限600/300 epoch到達を収束と扱わない。

重要方向は既存natural fit512顔・137動画のみのGramから定義する。人工fit256/dev128/eval128顔×3遮蔽形状、natural fit512/dev128/eval512、既存の動画分割・顔ID・画像SHAを照合する。train内小規模診断で、独立global val/testや本学習ではない。GT顔楕円cropを使用し、NMEはGT顔楕円長径で正規化する。この値を通常の検出を含むend-to-end指標と混ぜない。

## 保存・比較・制約

`run_messenger_codebook.py` が6条件を実行し、`report_messenger_codebook.py` が保存数値を監査する。位置平均・中央値・p90、presence、2/3/5/10%成功、自然遮蔽／可視、人工遮蔽／非遮蔽、5点別指標、動画対応bootstrap区間、コード使用数・perplexity・量子化誤差、実行時間・GPUメモリを保存する。固定vs更新、4,096vs8,192の対応比較を出す。raw入力の予測が保存対照と一致すること、顔／動画／点対応、最良epoch・収束・重みSHAを確認する。

16次元化してもAttentionの入力は384次元である。軽量化・高速化を保証しない。③〜⑦と①②の比較は復元構造の変更も含む。圧縮の純粋な次元効果だけとは解釈しない。④対⑤、⑥対⑦は固定／更新だけを変える。

全数値評価と監査完了後、今回のbest/last R/g（24ファイル）と初期辞書2ファイルだけを削除する。完了marker、数値成果物SHA、対象重みSHA・絶対パス範囲・利用中プロセスを検査し、削除記録を残す。未完了の学習／評価では保持する。baseline、固定親、保持候補scale C／level＋scale C、無関係な過去実験には触れない。

## 実行入口

プロジェクト直下で実行する。

```bash
.venv/bin/python scripts/run_messenger_codebook.py plan
.venv/bin/python scripts/preflight_messenger_codebook.py
.venv/bin/python scripts/run_messenger_codebook.py train --wait
# 中断後：plan/source/data identityが同一の場合のみ
.venv/bin/python scripts/run_messenger_codebook.py train --resume --wait
```

canonical leaseとGPU使用状況を確認する。再開では設定・ソース・入力・辞書初期値の一致を要求する。`report` は既存結果の全数値監査後に削除まで行い、`cleanup` は数値完了markerのある対象だけを扱う。canonical queueには6条件をdisabledの研究診断として登録し、自動起動／productionの追加はしない。

CPU preflightは実fit2顔で6条件を各1更新し、固定親の不変性、同じKの初期出力一致、位置だけからの辞書勾配を調べる。2顔の射影を反復した辞書は接続確認専用であり、本比較のk-means辞書ではない。技術確認で検証重みはディスクへ保存しない。

## 2026-11-19：gate後の比較学習開始指示

ユーザー指示：「gate通過を確認して、本学習に進むようにしてください」。既存fit・動画分割という直前の確定条件を維持する。以下は実行前に固定した品質基準で、結果を見て緩和しない。

- schema：実際の特徴・対象形状・有限値・パラメータと初期値を確認する。
- gpu_step：FP32/TF32、microbatch4／実効16の共通学習関数でRを更新する。gも更新し、位置だけから更新対象コードへの勾配、固定辞書・固定親の不変性を確認する。
- overfit：人工fit16対をRで32更新。natural fit32顔のgを32epoch。初期値より位置と目的関数が同一評価時点でともに改善することを要求する（Rは1e-5、gは1e-6を超える改善）。
- small_scale：人工fit64顔／dev32顔×3形状のR、natural fit128顔／dev64顔のg、それぞれ12epoch。初期dev位置よりRは1e-5、gは1e-6を超える改善と有限値を要求する。gateの初期辞書は本比較と同じfit256顔の65,536パッチから計算する。evalは選択・品質判定に使用しない。

各段階を同じ初期値から始め、本比較へgateの学習済み重みを持ち越さない。gateモデルはメモリ内に置き、重みファイルを生成しない。gateの報告と独立した24個のsuccess markerを保存する。学習入口でソース・固定親・入力・設定・runtime、全success markerのSHAと初期辞書の一致を認証する。1件でも不通過なら比較学習を開始しない。

```bash
.venv/bin/python -u scripts/gate_messenger_codebook.py --wait --then-train
```

出力：`outputs/training_gates/messenger_codebook_20261119/`。`run.log`はgate後も同じプロセスの比較学習ログとして継続する。比較学習のstatus・checkpointは既存指定の `outputs/diagnostics/messenger_codebook_20261119/`。canonical queueでは6比較条件が各4gateへ依存し、今回の限定runnerから実行するため一般キューによる重複起動は無効のまま。

実行証跡：
- gate結果：outputs/training_gates/messenger_codebook_20261119/gate_report.md
- 24markerと入力・ソースの認証：同ディレクトリ completion.json
- 学習開始・canonical lease・checkpoint確認：同ディレクトリ training_start_audit.json
