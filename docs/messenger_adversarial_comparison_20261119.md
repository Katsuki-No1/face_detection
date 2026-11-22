# 位置主目的に対する敵対的損失・特徴一致の3条件比較

2026-11-19のユーザー指示。現在の位置主損失＋0.01 MSE＋0.01 cosineを基準とし、同じデータで収束まで比較する。最良重みは位置精度で選び、正解特徴をheadへ渡す比較は行わない。

入口：`.venv/bin/python scripts/run_messenger_adversarial_comparison.py --wait`。
出力：`outputs/diagnostics/messenger_adversarial_comparison_20261119/`。
別チャットのcosine比較が使用中のcanonical leaseを待ち、実行やソースを変更しない。新規本学習ではなく、過去と同じ小規模train内動画分割の比較。

## 固定条件

- 回帰型scale1/2/4、幅96、2層6head。3条件ともseed42新規初期化。DFを同時に変えず、損失だけ比較する。
- A baseline：位置＋0.01 MSE＋0.01 cosine。B adversarial：A＋0.001敵対的損失。C adversarial_fm：B＋0.01 D中間特徴一致。追加係数は事前固定の実装選択で最適値ではない。初期勾配比を保存。
- 旧scale Cの枝教師・枝間一致を追加しない。位置は人工遮蔽／非遮蔽GT点のEuclidean誤差／GT顔楕円長径を別平均して等重み。headは固定、mask内R／外X。正解Zは特徴教師・D正例だけに使用。
- 人工fit256/dev128/eval128顔×3形状、自然fit512/dev128/eval512顔。旧manifestと顔ID・動画役割・画像hashを照合。global val/test未使用。Dの正例は人工遮蔽前特徴であり、自然遮蔽裏の未知特徴を正例扱いしない。
- R AdamW lr1e-4/weight decay0.01、microbatch4/effective16、clip5。FP32/TF32。DはAdamW lr1e-4/weight decay0、1対1更新、clip5。DのlrはRの減衰に合わせる。
- D：候補384ch＋元X384ch＋mask1ch＋座標2ch。1x1/128、3x3/128、stride2の3x3/192、3x3/192、1x1/logit。LeakyReLU0.2、SpectralNorm、BatchNormなし。入力の平均・標準偏差は人工fitのclean特徴だけで固定（std下限0.01）。
- D正例はmask内Z／外X、負例はmask内R／外X。同じX・mask・座標で条件付けし、受容野がmaskに交わる出力を各例内平均、その後例平均。正例と負例の外側は一致。
- 敵対的損失はnon-saturating logistic。FMは3つの空間畳み込み層の中間特徴MSEを、同じ受容野maskで位置・channel平均して層平均。G更新ではDをeval/frozen、正例特徴はstop-gradient。D更新ではRをdetach。R1を同時追加しない。
- Rの最良はdev位置誤差最小（epoch0含む）。収束は共通の定常的なdev位置・基準損失のplateau：最低40epoch、8停滞でlr半減、下限1e-6で15停滞。相対改善0.001/絶対1e-5。敵対的目的はDとともに動くため共通停止判定に混ぜず別記録。600上限を収束扱いしない。GANの均衡収束を証明する検証ではない。
- R固定後にgを同条件で学習。自然位置＋旧教師0.01 soft BCE、dev位置選択。lr5e-4、最低20epoch、下限5e-6、8/15patience、300上限。Dはgへの追加入力にしない。
- best/last R/g計12ファイルを保持。敵対的条件のRファイル内に対応Dを保持し、lastにD optimizer・RNG・SN bufferも記録。全数値評価完了まで削除なし、自動削除なし。再開は同じplan/source/dataのみ。

## 検証と評価

CPUで領域一致・勾配の分離・FMゼロ・SN buffer固定・保存再現をテスト。GPU使用権取得後、実データ4例12更新で全3条件の有限性・基準目的改善・位置/特徴/敵対的/FM勾配・固定head不変を確認。技術確認の重みは本比較へ転用しない。

R/gの検証plateau後に人工・自然evalを評価。元特徴、全R、学習g、人工mask内Rを比較。正解特徴head入力は作らない。平均/中央値/p90、2/3/5/10%座標成功とpresence0.7込み成功を分ける。自然可視/遮蔽、人工遮蔽/非遮蔽を分け、g採用率・特徴ノルム・局所分散・D識別値を保存。3条件の動画対応bootstrap95%区間を算出。単一学習seed・再利用データ・多重比較未調整の探索的結果とする。

この比較は条件付きGT crop内の下流位置評価であり、検出器を含むend-to-end精度ではない。位置正規化は前回と同じGT顔楕円長径、従来のHead矩形正規化と混同しない。
