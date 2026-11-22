# 人工遮蔽によるMessenger復元器と位置headの交互更新（2026-11-19）

ユーザー指示：復元特徴へheadを適応させる必要性を調べる。遮蔽なし顔に多様な人工遮蔽を与え、正確な遮蔽領域をgへ渡す。ViT回帰復元器はhead位置主損失、特徴MSE・cosine距離を補助とする。自然遮蔽は評価しない。

## 比較と更新順
1. warm_R_fixed_head：baselineの5点keypoint Query headを固定してRを検証plateauまで学習。
2. fixed_R_adapt_head：1の位置最良Rを固定し、baseline headを追加学習。
3. alternating：1の位置最良Rとbaseline headから、Hのみ1epoch、Rのみ1epochを交互に更新。
4. raw_adapt_head：復元なしの人工遮蔽特徴にheadのみを適応。追加学習だけの効果を分離。
original_headは人工遮蔽特徴をbaseline headへ渡す無学習対照。各方式に遮蔽なし入力のhead精度も計測する。これはmask内を正解特徴に置き換えるoracle比較ではない。全方式の固定DINOは同一。DFは今回の対象外。

## データとg
前回の人工fit256/dev128/eval128顔を再使用。全5点可視、face_occluded=false、train内動画単位分離。既存DINOにはtrainの過去露出があり、独立global val/testや本学習ではない。GT楕円crop、256px、1.25倍、5点位置は楕円長径で正規化した誤差（従来のHead矩形正規化NMEと区別）。
各顔9パターン。顔楕円の5–20%、20–40%、40–70%を占めるmask、矩形・楕円・曲がった形、顔内3×3の9位置、単色・画素ノイズ・色勾配を組み合わせる。各顔・各パターンの連続的な位置・大きさはseedから生成し、GT keypointはmaskの生成条件に使わない。maskは楕円内へclipするので指定位置は生成中心であり、最終maskの重心ではない。
学習は1顔3パターン/epoch、3epochで9パターンを巡回。各条件で同じ順序。dev/evalは毎回全9パターン。有限の固定mask bankによる多様化であり、毎epoch無制限に新しいmaskを作る設定ではない。
gは画像上の正確な二値maskを16×16特徴格子へany-poolingした二値占有。maskと交差するpatchのみRを採用、他は遮蔽画像の元特徴X。patch境界の可視画素を含むcellも置換される。DINO attentionによりmask外のXも影響を受け得る。g推定の学習はしない。
教師Zは遮蔽なしDINO特徴で、MSE/cosine補助と特徴評価だけに用いる。自然遮蔽データ、旧弱遮蔽教師、clean特徴のmask内oracle置換は使わない。

## 損失・最良選択・停止
R：遮蔽点/非遮蔽点の平均位置誤差を等重み + 0.01 MSE + 0.01 cosine距離。特徴項はmask patch平均 + 0.1×顔内非mask patch平均。独立scale1/2/4 attention、幅96。枝復元損失は追加しない。位置head更新時は同じ位置損失のみでRをdetach。
AdamW、R lr1e-4/H lr1e-5、weight decay .01、effective16/micro4、clip5。headもeval modeでparameter更新し、dropout状態変更は交絡させない。presenceの専用損失は追加しないため、位置成功とpresence併用成功を分けて集計する。
dev位置最良のR+Hペアを保存、epoch0も対象。R更新を伴う段はdev位置と合計目的、Hのみはdev位置を監視。minimum40、8epoch改善なしでlr半減、初期の1/100を下限、下限で15epoch停滞を検証plateauとする。上限600到達は収束扱いしない。数学的収束や全母集団への一般化は主張しない。

## 評価と保存
遮蔽点/非遮蔽点/全点の平均・中央値・p90、2/5/10%位置成功、presence0.7併用成功、遮蔽率/形状/生成位置/塗り条件別集計。特徴MSE、cosine距離、ノルム比、GT点での局所特徴誤差も保存。対照との差は動画対応bootstrap2000回の95%区間。遮蔽なし入力の位置精度劣化も確認。
全段のbest/last R+Hペア（optimizer付きlast）計8重みは全評価・独立再集計・ハッシュ照合まで保持し、完了後に削除してから報告する。ユーザーの確認は待たない。固定baseline重みは削除しない。
入口：scripts/run_messenger_alternating_head.py。出力：outputs/diagnostics/messenger_alternating_head_20261119/。再開は同一plan/source/dataと残存重みのもとで--resume。canonical GPU leaseを使用し重複起動しない。

## 完了結果（2026-11-19）

全4学習段は検証plateauまで完了。warm_R_fixed_head 94epoch（選択31）、fixed_R_adapt_head 86（37）、alternating 87（32）、raw_adapt_head 113（54）。ユーザーのPC再起動後、初期Rの7epochから同じoptimizer・条件で再開し、ソース/データSHAと特徴cache一致を確認した。

人工eval128顔・35動画×9mask＝1,152対、遮蔽2,010点／非遮蔽3,750点。位置誤差は楕円長径比%、低いほど良い。特徴MSEとcosine距離はmask patchごとの顔平均（学習用の外側0.1加算を含めない）。

|条件|全点位置%|遮蔽点位置%|非遮蔽点位置%|特徴MSE|cosine距離|R/Zノルム比|
|---|---:|---:|---:|---:|---:|---:|
|復元なし・固定head|2.89051|4.64369|1.95081|0.05032|0.21364|0.96976|
|復元＋固定head|2.90000|4.27734|2.16174|0.04892|0.22041|0.88950|
|復元器固定＋head適応|2.85794|4.17192|2.15365|0.04892|0.22041|0.88950|
|復元器とheadの交互更新|2.87213|4.17941|2.17142|0.04789|0.20185|0.97156|
|復元なし＋head適応|2.58273|3.81205|1.92382|0.05032|0.21364|0.96976|

- 固定Rのままheadを適応すると、遮蔽点誤差は4.27734→4.17192%。差−0.10542pp、動画対応95%区間[−0.15418, −0.04474]pp。復元特徴へのhead適合不足が一部の要因だったという仮説を支持する。ただし全点差の区間はゼロを含む。
- 交互更新は特徴MSE・cosine距離が最良。GT遮蔽点でサンプルした局所特徴MSE＋cosine距離も固定R0.35477→0.33616へ改善。局所指標も改善したが、headのみ適応との差は遮蔽点+0.00749pp、95%区間[−0.06847, +0.07559]ppで、位置への追加利益は確認できない。
- 最良の位置精度は「復元なし＋head適応」。交互更新はこの対照より遮蔽点誤差が+0.36736pp、95%区間[+0.17538, +0.54820]pp。全点差+0.28939ppも区間はゼロを含まない。head固定だけで以前の問題を説明することはできず、特徴復元・強制置換の位置精度上の利益はなお不足している。
- 遮蔽点5%位置成功は元1289→復元固定H1395→固定R適応H1436→交互1427、復元なしH適応1513（全2010点）。presenceを混ぜない位置成功。presence併用は保存JSONで別集計。
- 遮蔽なし入力の平均位置誤差は元1.64617%、固定R/head適応1.64513%、交互1.65356%、復元なしhead適応1.63893%。交互で微小な悪化もあり、この値だけで通常顔の改善を主張しない。

新規best/last計8重み486,783,702 bytes（464.23MiB）は、全数値評価・監査・成果物保存後に自動削除済み。固定baseline・採用候補の重みは変更なし。プロセス正常終了。再実行して重みを作り直さない。numeric_completion.jsonは削除前の評価完了時点、現在はstatus.jsonとcheckpoint_deletion.jsonを参照。

証跡：summary.json、report.md、learning_curves.png、各条件の評価npz/JSON、numeric_completion.json、checkpoint_deletion.json、post_completion_verification.json。固定R条件の特徴指標不変、全評価成果物SHA、baseline/ソース/データSHA、削除後重み0を確認。自然遮蔽・独立global val/testは未評価、単一学習seedのtrain内診断である。
