# 固定baseline DEIMでの楕円形状と特徴復元

ユーザー指定：顔形状はkeypoint配置ではなく、DEIMの予測楕円とGT楕円の差。①顔crop DINOは学習しない対照、②ViT／③DFは復元器のみ学習する。主損失を楕円形状、補助を特徴MSEとcosine距離とし、収束まで比較する。全数値評価まで重み保持。

## 経路と固定対象

研究用の順序付き経路は、baseline全画像FACE検出の保存予測 → 予測bbox中心の軸平行正方形crop（長辺1.5倍、256px、回転なし）→ baseline FACE側DINO → 復元の有無 → baseline STA／encoder／DEIM楕円decoder。HEAD検出器やkeypoint headは使用しない。登録V3 FACE checkpoint SHA `5e17564ccf772a37b5c825614635ed0d93344e65fdb4757740a2ed782c9ba2d3`、EMAを固定。

前回のkeypoint用crop DINOとは重みが異なるため、この比較ではbaseline FACEのDINO特徴をキャッシュし、新しい復元器をseed42から学習する。既存のViT回帰／DFと同じscale1/2/4独立attention、幅96、最終block11の384×16×16特徴を全域置換。block5/8、RGB STA、後段の学習済み重みとBatchNorm状態は固定。DFは16step実生成全体に逆伝播。学習済みgは追加しない。

これは最終DINO層の特徴復元を検証するもので、DEIMへ入る全経路の特徴を復元する実験ではない。未復元の浅い層とRGB特徴にも遮蔽の影響が残る。各条件で同じ経路を使う。元特徴の差し替えを恒等にしたとき、通常のbaseline crop推論と一致することを実データで確認する。

## 入力と教師

GT楕円による中心・角度合わせは行わない。GT楕円はcropへ座標変換した教師としてのみ扱う。cropは既存の予測bboxから作る。小さな対応済みcropキャッシュを、元の大きな予測JSONとストリーム照合する。元JSON全体をメモリへ展開しない。

前回のtrain内動画fit/dev/eval分割を継承。人工は非遮蔽・5点注釈ありのfit256/dev128/eval128顔にline/face/handの3遮蔽、自然はfit512/dev128/eval512顔。予測bboxが対応しない顔、3遮蔽が作れない顔は記録し、同じ動画群の固定順から補充する。自然の母集団も従来に合わせて1点以上の注釈あり。keypoint座標は人工遮蔽配置にのみ使用し、損失には使わない。

人工遮蔽はcrop後に加えるため、上流の全画像検出失敗を測る実験ではない。自然でも保存済み検出との対応顔に限定される。全件本学習・global val/test・全画像検出のprecision/recall/F1/mAPではなく、条件付きcropの楕円診断とする。

## 損失と選択

予測楕円はGTを使わず、crop内のFACEクラスscoreが最大のqueryから選ぶ。主損失は既存DEIMと同じ形の中心SmoothL1×5、半軸SmoothL1×5、`1−cos(2Δθ)`×1、36点輪郭の対称Chamfer×2。ほぼ円形（GT長短軸比<1.05）の角度項は0。顔ごとの固定分母で平均し、既存criterionの「非円形だけで角度を平均」とは分母が異なることを明記する。輪郭距離は同一点での丸め誤差を抑える直接距離計算を使用する。

補助は0.01×特徴MSE＋0.01×cosine距離。人工遮蔽内の平均＋0.1×楕円内非遮蔽領域の平均。遮蔽前の教師特徴は特徴損失・特徴ノルムのみに使い、headへ教師特徴を渡す比較はしない。復元特徴はmaskで選別せず全block11 mapを置換し、推論にもGT遮蔽maskを渡さない。

両復元器はAdamW1e-4、weight decay0.01、clip5、micro4／effective16。学習生成seed42+1009×epoch、dev42、DF評価42/1042/2042/3042。最良重みは人工dev楕円主損失の最小値で選択。楕円損失と全目的の検証plateauを監視し、最低40epoch、8epoch停滞ごとにlr半減、下限1e-6で15epoch停滞まで継続。600epoch上限は収束とは扱わない。新規重みbest/last×2方式の4個を全評価まで保持。

## 評価

- 楕円IoU、中心誤差／GT長径、軸誤差／GT長径、軸方向角誤差、輪郭誤差／GT長径。
- FACE scoreと固定score0.33・楕円IoU0.5/0.75の条件付き成功数。低信頼度の予測を誤差集計から除外しない。
- 自然は輪郭遮蔽属性あり／なし、および点遮蔽も含めた群を分離。
- 人工は同一顔の3遮蔽間の中心・軸の変動と相互楕円IoU。安定して間違った形に寄る可能性があるため、GTへの精度を必ず併記。時系列の揺れを測ったとは扱わない。
- 特徴MSE・cosine・ノルム、DF4生成seed、動画単位対応bootstrap区間、再現可能な失敗例の図。

入口 `scripts/run_messenger_ellipse_comparison.py --wait`。出力 `outputs/diagnostics/messenger_ellipse_comparison_20261119/`。実データ事前確認は `scripts/preflight_messenger_ellipse.py`、独立数値集計は `scripts/report_messenger_ellipse_comparison.py`。既存実験のソース・重み・候補選定は変更しない。

## 重み削除時点の更新（2026-11-19）

全数値評価・比較監査・結果保存が完了したら、検証で生成した4重みをSHA付き削除記録を残して削除し、その後に結果を報告する。ユーザーの評価・返答は待たない。評価未完了時と固定baselineは保持する。
