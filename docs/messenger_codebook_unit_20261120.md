# 教師g固定のcodebook正規化修正

選択時だけ正規化した教師g固定比較を受け、B・D・E×16／32次元×4,096／8,192コードの12条件を新規学習する。元の履歴を固定対照とし、各条件の修正前後、辞書なし16／32次元、対応する正規化なしA／Cと同じデータ上で比較する。

| 条件 | コードと潜在値 | 辞書更新 | 復号器 |
|---|---|---|---|
| B | 選択・損失・復号で単位ベクトル | 位置損失を含む勾配＋AdamW、更新後に単位長 | 位置独立 |
| D | 同上 | 割り当てられた単位方向の平均でEMA、更新後に単位長 | 位置独立 |
| E | 同上 | Dと同じEMA | D＋既存の空間Attention残差 |

初期辞書は前回と同じraw Lloyd中心を再生成してSHAを照合し、コードごとに単位ベクトル化する。正規化前の振幅はdecoderへ渡さない。旧実験のraw空間の量子化MSEと、今回の単位ベクトル間の量子化MSEは尺度が異なる。変更による位置精度・特徴MSE/cosine・コード使用偏りを評価する。

人工の生成maskと自然の5点由来の弱教師gを固定する。Rのみ学習し、DINOとhead、既存GT crop・train内動画分割・seed42・損失の係数・最良dev位置での選択を維持する。人工fit256/dev128/eval128×3、自然fit512/dev128/eval512。固定headへ遮蔽前の教師特徴は入力しない。位置主損失＋0.01MSE＋0.01cosine＋0.01(code＋0.25commitment)、EMAはcode lossなし。AdamW、microbatch4/effective16、FP32/TF32、lr1e-4、既存plateau。

各条件のschema・GPU更新・overfit・small-scaleの48 gate通過を認証してから、新規初期化で本学習する。品質閾値は前回と同じ。実際のhead座標勾配、unitノルム、学習・評価中の更新可否、教師map同一性を確認する。本学習は教師g付きのcrop内診断であり、親の運用はFACE楕円→crop keypoint。独立global val/testは含まない。

入口 `.venv/bin/python scripts/gate_messenger_codebook_unit.py --wait --then-train`。本学習再開はcanonical lease・生存PID・保存plan/source/dataを照合し、`.venv/bin/python scripts/run_messenger_codebook_unit.py train --resume --wait`。出力は `outputs/diagnostics/messenger_codebook_unit_20261120/`、gateは `outputs/training_gates/messenger_codebook_unit_20261120/`。全数値評価・監査後、今回作成したbest/last24個と初期辞書4個をSHA・パス・使用プロセス確認後に削除する。
