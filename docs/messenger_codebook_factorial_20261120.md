# Codebook選択・更新・次元・decoder比較（2026-11-20）

## 意図とユーザー確認

DINO特徴の復元は既存Messengerが担い、BEiTの欠損コード予測は導入しない。コードの選択と辞書更新を切り分け、32次元化の効果も確認する。ユーザーは「辞書なし32次元を先に、A〜Eを16／32次元×4,096／8,192で比較。EはDに空間Attentionを追加」の範囲を明示了承した。

## 条件（21新規学習）

最初にcontinuous32。次に各16／32次元、各4,096／8,192コードで以下を実行する。

| 条件 | 選択距離 | 辞書更新 | decoder |
|---|---|---|---|
| A | 正規化なしL2 | 勾配 | 位置独立 |
| B | 特徴とcodeを選択時だけL2正規化 | 勾配 | 位置独立 |
| C | 正規化なしL2 | EMA | 位置独立 |
| D | 選択時のみL2正規化 | EMA | 位置独立 |
| E | Dと同じ | EMA | D＋空間Attention残差 |

重要方向は既存natural fit512顔137動画のhead感度Gram上位16／32方向で初期化する。特徴分散PCAではなく、学習中のencoderは方向を変更できる。初期辞書は既存synthetic fit256顔65,536patchのclean特徴を射影し、seed42、30回上限のLloyd k-means。次元・容量ごとにA〜Eで同じ初期辞書と共通重みを使う。正規化条件の初期化も同一にして選択方式だけを変更する。

B/D/Eの正規化は距離計算だけ。decoderへは振幅を持つ元のcodeを渡す。C/D/Eは各実効batch16の割当encoder出力の平均へdecay0.99で更新し、割当なしcodeは維持。microbatch4の統計を合算してoptimizer後に一度更新する。EMAは全train patchから計算し、dev/evalでは更新しない。codeへの直接head勾配とcode-fitting lossはEMA条件で無効、encoderへのSTE・commitmentとdecoderのhead勾配は維持する。BEiTの正規化codeまで含む完全再現ではない。

Eは量子化後のmapのみから幅96・6heads・1層self-attention＋FFN・2D sine位置を使い、384次元残差をdecoder出力へ加える。出力射影はzero初期化し、初期出力はDと同じ。入力DINOの迂回路やGT遮蔽maskによるAttention制御は追加しない。

事前学習なしを固定。これは有効性を否定した意味ではなく、今回の比較要因から外す判断。headに正解特徴を与える評価は実施しない。

## 学習・評価契約

既存synthetic fit256/dev128/eval128×3mask、natural fit512/dev128/eval512、GT crop、train内動画分割、seed42。DINOとkeypoint head固定。位置主損失＋0.01MSE＋0.01cosine＋0.01(code loss＋0.25commitment)。領域は既存mask内1・外0.1。gは別学習、旧弱教師＋0.01BCE。Rはdev位置最良、gもdev位置最良を選択し、既存plateau基準（R最低40/g最低20、LR縮小8、floor後15、上限は収束と扱わない）を維持。FP32/TF32、AdamW、lr R1e-4/g5e-4、weight decay0.01、microbatch4/effective16。

親の運用topologyはfull-image FACE/ellipse→face-crop keypointのHEADなしcascade。この実験はそのcrop特徴部分の小規模診断で、全件productionや独立global val/testではない。旧continuous16と384次元回帰は保存数値を参照。A16も新規学習し、実装変更の影響を抑えた現行対照とする。旧実験と固定baselineは変更しない。

自然遮蔽／可視でraw・full_R・learned_gを区別。人工遮蔽はmask_Rも評価。位置誤差、5%座標成功、presence込み成功、5点別F1、特徴誤差・ノルム、辞書使用数／perplexity／量子化誤差、動画対応bootstrapを保存する。比較はB−A、D−C、C−A、D−B、E−D、32−16、8192−4096、VQ−連続値。単一seed、探索的多重比較である。

## 実行と状態

入口：`.venv/bin/python scripts/gate_messenger_codebook_factorial.py --wait --then-train`

全21条件×schema/gpu_step/overfit/small_scaleの84成功markerとplan/source/data/初期辞書一致を認証してから学習。品質gate不合格を自動で設定変更・再試行しない。技術不具合は原因と証跡を保存して修正する。canonical GPU lease、既存GPUプロセス不在を確認。実行中ソース変更禁止。

実装テスト37成功、CPU実fit2顔の全21条件preflight成功。全21条件のgate確認完了。Rは全通過、gのsmall-scaleがB16/8192・E16/8192・B32/4096で不合格。81/84成功、本比較は未開始。詳細は `docs/messenger_codebook_factorial_gate_results_20261120.md`。W&B runなし（既存小規模診断と同じローカルJSON履歴）。

出力：`outputs/diagnostics/messenger_codebook_factorial_20261120/`、gate：`outputs/training_gates/messenger_codebook_factorial_20261120/`、CPU監査：`outputs/diagnostics/messenger_codebook_factorial_preflight_20261120/audit.json`。

再開はplan/source/data一致・生存PID・leaseを確認し、main runnerの`train --resume --wait`を使用。全数値評価と監査の完了までbest/last R/g84ファイル＋初期辞書4個を保持し、完了後はSHAとパス・使用中プロセスを検査してこの88ファイルのみ削除。失敗／未完了時は保持する。設定・ログ・予測・監査・削除記録を保存する。
