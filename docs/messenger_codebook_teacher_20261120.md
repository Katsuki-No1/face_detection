# 教師g固定によるcodebook比較（2026-11-20）

## ユーザー修正と目的

今回の目的は固定keypoint headに有益な復元特徴Rを生成する方法の比較。gの予測・最適化は別課題であり、この比較では教師gを入力する。ユーザーは前回から引き継ぐ比較元も含めて新規学習するよう指示した。

前回のR学習は既に生成maskを使っていたが、その後にNaturalFeatureGateを別学習し、自然遮蔽の学習済みg融合評価を主に報告していた。旧factorial gateの不合格3件もRではなくgのsmall-scaleである。旧記録を保持し、本設計ではgモデル・optimizer・BCE・g重み・g品質gateを作らない。旧失敗を通過へ書き換えたものではない。

## 23条件を新規初期化

順序：辞書なしcontinuous32、384次元回帰regression384、辞書なしcontinuous16、その後A〜E×16／32次元×4,096／8,192コードの20条件。保存済み384次元/continuous16の性能値を主比較へ流用しない。旧4倍重み付けや固定辞書は今回のA〜Eには含めない。

| 条件 | code選択 | 更新 | decoder |
|---|---|---|---|
| A | 正規化なしL2 | 勾配 | 位置独立 |
| B | 選択時のみ特徴/code正規化 | 勾配 | 位置独立 |
| C | 正規化なしL2 | EMA | 位置独立 |
| D | 選択時のみ正規化 | EMA | 位置独立 |
| E | Dと同じ | EMA | D＋空間Attention残差 |

EMAはraw値の割当encoder出力平均へdecay0.99、実効batchごとに更新。空codeは維持。正規化は距離計算だけで、decoderへ渡すcode振幅は維持。Eは量子化後mapから幅96/6heads/1層Attention＋FFN＋2D sine、zero初期化384次元残差。初期D/E出力同一、DINO迂回路なし。16/32は既存fit512顔137動画のhead感度Gram上位方向による初期化で、学習中は可変。特徴PCAではない。辞書は既存clean synthetic-fit256顔65,536patch、seed42、Lloyd最大30回。A〜Eは次元/容量ごとに同一初期辞書。

## 教師gとhead入力

`Y = (1 - g_teacher) * X + g_teacher * R`

- 人工遮蔽：既存の生成maskから作られたbinary patch maskをR学習/dev/evalで共通使用。既知mask内をR、外をXとする。旧学習のmask定義を変更しない。
- 自然遮蔽：既存 `src/data/natural_gate_teacher.py` の5点可視/遮蔽注釈＋楕円による弱教師mapを固定使用。confidenceは以前の教師損失用の重みであり、g値へ追加乗算しない。自然gは厳密な遮蔽segmentationの正解ではない。
- gはdetachした外部定数。自然のGT由来gを使う結果は教師条件下の診断であり、運用可能なg推定や独立汎化の実証ではない。
- R学習は以前と同じ人工fit、特徴MSE/cosineの教師はclean特徴。headへclean教師特徴を渡す比較は行わない。自然fitを新たなR学習へ混ぜる変更はしない。

## 学習と品質確認

既存train内動画分割、synthetic fit256/dev128/eval128×3mask、natural fit512/dev128/eval512、GT crop、seed42。DINO/head固定。位置主損失＋0.01MSE＋0.01cosine＋0.01(code loss＋0.25commitment)、EMAではcodeへの直接勾配とcode lossなし。mask内1/外0.1。FP32/TF32、AdamW lr1e-4、weight decay0.01、microbatch4/effective16、clip5。dev位置最良で選択し、位置/総合損失plateau、最低40epoch、LR縮小patience8、floor1e-6後stop15、cap600は収束扱いしない。事前学習なし。

23条件×4stageの92gateを新規確認する。schema/実GPU更新/同一データoverfit32step/small-scale人工fit64/dev32顔12epoch。R位置の改善を必要とし、overfitでは総合損失も同epochで改善を要求。自然gを学習するgateは設けない。GPU technical gateで最終評価関数もfit顔だけで実行し、保存g mapが教師に完全一致することを検査。全gate通過後にfresh学習を開始する。

親topologyは全画面FACE/ellipse→顔crop keypoint（HEADなしcascade）。この実験はcrop特徴部分の小規模診断で、全件productionや独立global val/testではない。学習プロセスはローカルJSONに履歴を保存する。19:25 JSTに別プロセスでW&B同期を開始し、開始済みの履歴と以降のエポック指標を `face_detection` のrun `rf8q3hk6` へ反映する。学習側のソースと重みは同期対象外。

## 評価・保存

両domainでraw、full_R、teacher_gを分離。人工の既知maskと自然の弱教師mapを混同しない。固定教師gでのhead位置誤差、5%座標成功/presence込み成功、5点別F1、遮蔽/可視、自然顔遮蔽flag×遮蔽点数12群、特徴MSE/cosine/局所誤差/ノルム、code利用数/perplexity/量子化誤差を保存。差分は同一動画対応bootstrap、単一seed・探索的多重比較。比較元はfresh regression384/continuous16/continuous32。

全数値評価・SHA監査が終わるまで23条件best/last R46重み＋初期辞書4個を保持し、完了後にこの50ファイルのみ安全確認して削除。baseline/固定教師/過去実験には触れない。

入口：`.venv/bin/python scripts/gate_messenger_codebook_teacher.py --wait --then-train`

再開：生存PID・canonical lease・plan/source/dataを照合し、`.venv/bin/python scripts/run_messenger_codebook_teacher.py train --resume --wait`。実行中ソース変更禁止。gateは新規開始で、品質不合格時は自動再試行しない。

出力：`outputs/diagnostics/messenger_codebook_teacher_20261120/`、gate：`outputs/training_gates/messenger_codebook_teacher_20261120/`、preflight：`outputs/diagnostics/messenger_codebook_teacher_preflight_20261120/audit.json`。

実装検証：関連17テスト成功（新規教師g契約6件を含む）。CPU実fit2顔の全23条件preflight成功。GPU gate/本比較は次に開始する。

初回GPU technical gateでbool教師gの分位点集計エラーを検出。失敗と元ソースは `outputs/training_gates/messenger_codebook_teacher_20261120_technical_failure_01/` に保存。集計を数値型へ変換し、回帰テストを追加、17テストとCPU実fit全23条件preflightを再通過。品質閾値を変更せず新gateを実行中。起動は `launch_02.json`、PID18850。
