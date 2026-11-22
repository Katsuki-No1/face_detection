# Messengerの自然遮蔽対応と敵対的augmentationの実装計画メモ

作成日: 2026-10-15。作成時は設計メモのみ（当時の実装・学習・評価はnot run）。同日の後続指示で中心学習と第一段階のRGB補正G/D/Tを実装し、CPUの小規模動作検証を実施した。**2026-10-18の最新計画は末尾の「段階的学習への改訂（新規実装の優先契約）」を優先する。** 素材選択・顔crop全体D・4遮蔽率区間など、同節で変更しない条件は「合成方策Gの確定条件」を引き継ぐ。以下のG/D/T同時更新、旧checkpoint選択、旧実行状態は当時の履歴であり、新規段階実験へ自動継承しない。今回の更新は計画のみで、実行中・完了済みrunの変更や新規学習の開始ではない。

目的は、自然な遮蔽がある顔でも、遮蔽されたkeypointの位置を正しく推定することである。そのために、遮蔽候補を特定し、必要な特徴をMessengerで復元・融合してkeypoint headへ渡す。次の敵対的学習は、人工遮蔽を自然遮蔽へ近づけ、自然画像での位置推定へ改善をつなげる目的で設計する。

## 目的と成功の定義

最終目的は遮蔽keypointの位置推定である。遮蔽判定、特徴復元、人工遮蔽の自然さは、そのための中間能力として別々に検証する。中間指標だけの改善を最終目的の達成としない。

推論構成はHEAD-free ordered cascadeを維持する。

`full-image FACE/ellipse → Face-crop DINO → Messenger復元と事前分布による融合 → Query Refinement`

現行prior構成には固定DINO observerもある。新しい遮蔽生成器と自然さの識別器は学習専用とする。GT HeadはNMEの正規化に用いる注釈で、推論にHEAD検出器を追加しない。現行local baselineはV3 FACE/ellipse → V3 Query Refinementであり、過去のMessenger対照群とは区別する。

## ユーザーの認識を整理した現状

| 認識 | 実装と結果に合わせた記述 |
|---|---|
| 遮蔽keypointの位置を正しく推定したい | 最終目的として一致する。自然遮蔽点を出力し、正しい位置へ推定できたかで評価する。 |
| 遮蔽を検出し、特徴を復元してheadへ渡す | 現行構成の方針と一致する。遮蔽候補マスク、復元候補、融合係数、最終位置を分けて点検する。 |
| 自然遮蔽を正しく検出できない問題がある | 既存Messengerの自然遮蔽位置での融合係数gは弱い。gは遮蔽確率ではない。prior異常マスクの自然遮蔽検知精度は未確定であり、全構成について検知失敗が確定したとは言わない。 |
| 遮蔽部分の特徴復元精度の向上は確認済み | ABC系などの人工遮蔽診断で、融合後特徴が同じ顔の人工遮蔽追加前の教師特徴へ近づく改善を確認済み。自然遮蔽の正解特徴復元や、prior/GAN条件でも同じ改善が成立することまでは確認していない。 |
| 人工遮蔽を自然遮蔽と同じ品質にしたい | 次の敵対的augmentationの目的として採用する。自然らしさが改善すれば位置推定も改善する、という部分は検証すべき仮説。 |
| k-means中心へ近づけることで安定した復元を行う | codebookとして使う今後の設計意図として理解する。ただし、現在は中心を復元目標にしていない。下記に現在と拡張案を区別する。 |
| k-meansの効果は確認できている | 旧データ・旧初期重み・単一seedの条件で、k=3のprior判定と融合を組み込んだ構成の自然遮蔽点位置推定に改善がある。k-means単独、codebook復元、現行V3 baselineへの効果を実証した結果ではない。 |

## k-meansの現在の役割とcodebook案

現在の処理は以下の順である。

1. 非遮蔽資格を監査したtrain顔の固定DINO空間特徴をk-meansで分類し、クラスタごとの平均特徴マップを保存する。
2. 入力の予測楕円のスケール・長短軸比・角度から、幾何空間で最も近い参照クラスタを選ぶ。入力の欠損特徴から単純に最近傍中心を選ぶ方式ではない。
3. 固定observerの入力特徴を参照中心と比較し、位置別の正常範囲を下回る場所を異常候補マスクmにする。
4. Messengerが復元候補Rを生成し、mと採用係数gを用いてstudent特徴Xへ融合する。

融合式は `Y = X + m × g × sigmoid(a_channel) × (R − X)`。参照中心そのものはRの生成入力や正解目標として使われていない。復元教師は同じ対象顔の人工遮蔽追加前の特徴であり、クラスタ中心ではない。

codebook案は、参照中心を復元候補の条件入力や弱い構造制約として使う独立した拡張である。平均中心へ強制的に近づけると、対象顔の向き・表情・個体差を失う可能性があるため、可視部との整合と対象顔の位置教師を優先する。採用中心、検索方法、制約の強さ、復元への接続方法を設計してから専用ablationを行う。次の敵対的augmentationとの同時変更は避け、既存k=3 priorを固定して効果を切り分ける。

既存の自然遮蔽位置推定診断は次のとおり。固定FACE0.33、presence0.70、NME0.10、GT Head基準の関連付けで、FACE見逃しも失敗に含む。モデルによる遮蔽状態の正解判定は成功条件に含まない。

| 条件 | Val自然遮蔽455点 | Test自然遮蔽437点 |
|---|---:|---:|
| 同条件のMessenger対照群 | 285点 62.64% | 300点 68.65% |
| k=3 prior 敵対的学習なし | 301点 66.15% | 320点 73.23% |
| k=3 prior 既存の特徴空間敵対的学習あり | 282点 61.98% | 298点 68.19% |

k=3は先行k1〜7のValidationのみで選択されている。Test値は既存結果の確認であり、今後の設計・設定選択の根拠にはしない。

## 既存の敵対的学習と目的の違い

既存実装のGはstudent DINO・Messenger・融合側で、Dは融合特徴と固定教師特徴を識別する。人工遮蔽は固定ルールによる単色の矩形・不規則形・境界遮蔽であり、G/Dは遮蔽画像の自然さを学習していない。

| 項目 | 既存の特徴空間敵対的学習 | 次に設計する敵対的augmentation |
|---|---|---|
| Gの生成対象 | 復元・融合特徴 | 遮蔽物の見た目、必要に応じて形状・配置 |
| Dの比較対象 | 人工遮蔽追加前の教師特徴と融合特徴 | 実際の自然遮蔽画像と人工遮蔽合成画像の適切に対応した領域 |
| 直接の目的 | 復元特徴を教師らしくする | 人工遮蔽を自然遮蔽の分布へ近づける |
| 最終判定 | 自然遮蔽keypoint位置推定 | 同左。自然さだけで成功としない。 |

既存の敵対的runは14epoch・1358更新まで実行されたが、bestはepoch2である。最初の2epochは融合無効、最初の5epochはbackbone凍結のため、選択checkpointは敵対的損失がMessengerの融合経路へ作用し始める前にある。Dはその間も更新される。epoch3以降の学習はbestを上回っていない。この開始時期のずれは再設計・診断対象とし、原因の確定とはしない。

自然遮蔽Testでは、presence0.70以上の出力数が355→327、正しい位置の点が320→298へ減っている。今後は位置ずれと未出力・confidence低下を必ず分けて確認する。

## 次の敵対的augmentationの設計案

### 生成するものと固定するもの

遮蔽生成器G_occと特徴復元器Messengerを別モジュールにする。G_occはclean recipientへ貼る遮蔽物を生成・調整する。D_occは自然遮蔽と合成遮蔽の見た目を識別する。既存の特徴識別器D_featureは別の任意機能として扱い、最初の比較では無効にして目的を混ぜない。

最初の実装候補は、監査済み素材とbinary支持領域・配置を使い、支持領域内のRGBや素材変換を学習する段階である。形状・配置の学習は別段階とし、離散選択やhard maskへの勾配の扱いを実装前に決める。画像空間の自然さを学ぶ設計であり、位置だけを難しくする敵対的augmentationとは区別する。

合成は `I_syn = (1 − M) × I_clean + M × O` とし、Mはbinary mask、Oは遮蔽物。recipientのmask外の画素は変更しない。透明化・顔の消去・隠れた正解keypointの移動を自然さ獲得の手段にしない。従来の実顔width48/strength2を使うarmは、その合成契約も保持する。新policyの変更点は別の実験条件として宣言する。

### 自然遮蔽の教師と識別器

自然遮蔽realはtrain動画内から採る。FACE輪郭のoccludedフラグと内部keypoint state1を独立に扱い、state0を遮蔽に読み替えない。recipient・素材・自然遮蔽realの動画、注釈、hashを記録し、固定video splitを維持する。

既存注釈が与えるのは輪郭の遮蔽フラグと点状態であり、遮蔽物の密な領域maskではない。D_occの比較領域、real側での遮蔽領域の取得方法と監査を先に設計する。点注釈だけから正確なpixel maskがあると仮定しない。密な自然遮蔽maskを評価するなら、その評価注釈を別途用意する。

realとsyntheticで顔向き・サイズ・遮蔽点数・画質などを揃え、Dが背景・動画ID・crop境界などで容易に識別する経路を点検する。Dの条件入力は両側で同じ規則で作り、合成maskやclean正解をfake側だけへ渡して識別の手掛かりにしない。realに存在しない合成前画像はDへ渡さない。

### 学習順と勾配の分離

まずG_occ/D_occの自然さと合成不変条件を単独で検証し、生成器を凍結する。その後、固定ルールaugmentationと学習済みaugmentationを同じMessengerの初期重み・データ・予算で比較する。最初からG_occ・D_occ・Messenger・D_featureの4者を同時更新しない。

Messengerは同じ顔・同じcrop幾何の人工遮蔽前後の対から復元を学習し、最終keypoint位置・presence・state損失を受ける。clean教師には元からの自然遮蔽が残り得るため、完全な非遮蔽正解とは呼ばない。G_occが教師を変更したり、正解keypointの情報を遮蔽物へ埋め込んだりしないよう、入力と勾配を分離する。

将来joint trainingを試す場合は、各損失の勾配先、G/D更新開始、更新頻度、凍結対象、係数を別途固定する。融合無効中に敵対的損失だけを計算してGが学習したと扱わない。各moduleへの勾配量を記録する。

## 比較と受け入れの計画

1. 現行k=3の固定augmentationを再現対象として定義し、旧完了runを再実行せず、必要な新比較armだけを独立出力へ登録する。
2. 固定augmentationと学習済みaugmentationの比較では、復元器・prior・初期重み・split・seed・学習予算を一致させる。復元on/offの追加比較で、改善がMessengerの活用によるものか、augmentation一般の効果かを確認する。
3. codebook復元とD_featureは別ablationへ分ける。k=3は旧条件のValidation選択結果であり、新条件で最適と仮定しない。新しいk選択が必要ならtrainでbankを構築し、Validationだけで選択する。
4. 人工遮蔽の自然さは学習中Dの損失だけで判断しない。held-out動画の比較、独立した識別評価、目視、遮蔽物種類・形状・強度の多様性と制約遵守を組み合わせる。識別率が偶然水準でも、Dが学習できなかった可能性を除く。
5. 自然Validationでは遮蔽点の位置推定を主目的として評価し、overall・visibleの回帰も確認する。自然さ改善だけで位置推定改善がなければ、最終目的未達とする。

| 検証軸 | 保存する証拠 |
|---|---|
| 人工遮蔽の自然さ | 独立評価、real/syntheticの条件対応、合成例、多様性、適用・見送り・強度の分布 |
| 遮蔽候補の検知 | 自然state1/2位置での候補maskスコアとPR、閾値、slot別件数。密なmask評価と点位置の評価を区別する。 |
| 特徴復元 | 人工対でX/R/Yと教師のcosine距離・振幅誤差、遮蔽外の維持。自然遮蔽の真の復元正解は仮定しない。 |
| 最終位置推定 | 自然state1点の成功数・全GT分母recall、5slot Macro F1、出力数、presence分布、出力された全点の位置誤差 |
| 回帰とコスト | overall/visible、FACE検出、輪郭遮蔽×state1点数、profile、重なり・誤帰属、推論時間・VRAM |

NMEは元画像座標の誤差を `1.20 × max(GT Head width, height)` で正規化し、0.05/0.10を報告する。FACE missesを含むend-to-endと、正しく関連付いた予測FACE後のconditionalを分ける。位置成功と位置＋遮蔽状態成功も別列にする。5slot等重みMacro F1とmicro recallを混同しない。未出力を位置誤差0にせず、出力数とrecallへ反映する。

訓練checkpoint選択は既存のconditional semantic Macro F1 NME0.10・presence0.70契約を保持する。新比較の評価閾値をF1で選ぶ場合はFACE、次に共有presence閾値をValidationで選びTest前に凍結する。common0.33/0.70も別表で残す。自然さの合格値、回帰許容幅、遮蔽物内訳、G/D係数、評価用自然maskは実装前の決定事項であり、このメモでは数値を確定しない。

## 実装前に決める事項と実行条件

- trainの自然遮蔽real集合とDの比較領域。顔貼り付け・非顔物体・自己遮蔽をどこまで対象にするか。
- G_occが学習する変数。RGB、素材選択、形状、配置、照明調整の導入順と各制約。
- codebookを参照入力にするか、弱い復元制約にするか。これはaugmentation比較後の独立設計とする。
- 自然さ・位置推定改善・visible回帰の定量基準と、Validationでの設定選択方法。
- 新比較の初期checkpointとsplit。旧canonical-v2再現と現行V3 baselineへの導入検証を同じ結果として扱わない。

新実装は独立config、source/data/checkpoint hash、W&B、G/Dとoptimizer・scheduler・EMA・RNGを含むresumeを持つ。実行可能なtrainer完成後にcanonical queueへ登録し、各armの独立4/8/16 preflight、fresh schema/gpu_step/overfit/small_scaleを通してからproductionへ進む。訓練用supervisionはtrainだけから作り、Testで設定を選ばない。既存の完了実験・未完了detector/FOR計画・baseline・優先順位をこのメモで変更しない。

## 根拠となる実装と既存結果

- [現在のprior設計](messenger_geometry_prior_design.md)
- [prior参照と融合、特徴識別器と損失](../keypoint_comparison1/landmark_detection/geometry_prior_recovery.py)
- [warmupとepoch管理](../keypoint_comparison1/landmark_detection/geometry_prior_epochs.py)
- [敵対的run設定](../configs/training_plans/messenger_geometry_prior_runs/selected_k_adversarial.json)
- [自然遮蔽の既存診断](../outputs/landmark_detection/messenger_geometry_prior/selection/natural_occlusion_diagnostic_20261013.json)
- [人工特徴復元の既存診断](../outputs/landmark_detection/messenger_three_axis_report_20261013/metrics.json)
- [敵対的run完了記録](../outputs/landmark_detection/messenger_geometry_prior/selected_k_adversarial/training_success.json)

## 追記 G D Tによる敵対的学習と推論時の遮蔽判定

2026-10-15の追加合意として、自然さを評価するDと位置推定を担うTの両方からGへフィードバックする構成を、次の実装方針とする。上記の「将来joint trainingを試す」という位置付けを、段階的な比較後にG D Tを組み合わせる計画へ具体化する。今回の追記は設計の記録で、コード実装・キュー登録・学習開始はnot run。

### 各モデルの役割

| 部分 | 学習中の役割 | 推論時 |
|---|---|---|
| G_occ 遮蔽生成器 | 実顔・非顔素材を用いて合成し、Dに人工だと見抜かれにくく、Tにとって難しい遮蔽を作る。 | 使用しない。 |
| D_occ 自然さの識別器 | 自然遮蔽realと合成遮蔽fakeを識別する。遮蔽の有無や位置の判定器とは区別する。 | 使用しない。 |
| T 位置推定モデル | 遮蔽候補の判定、Messenger復元・融合、Query Refinementを通じて位置・presence・stateを推定する。 | 使用する。 |
| O 遮蔽判定部 | T内部で復元・融合する場所を決める。現在は固定observerとk-means priorによる異常判定。学習可能な判定器への拡張は別途比較する。 | 使用する。 |

ここでいうTはcrop側モデルを指す。全体の推論は上流のfull-image FACE/ellipse検出から始まる。最終headの遮蔽state出力と、特徴融合前に使うOのマスクは役割と評価を分ける。Messengerのgも復元特徴の採用率であり、遮蔽確率とは呼ばない。

### GがDとTに対抗する意味

GはDに対しては自然らしい合成を目指し、Tに対しては正解位置の推定を難しくする。Dはreal/fake識別誤差を減らし、Tはkeypoint誤差を減らす。難しさだけを最大化すると、顔を隠し過ぎる例や不自然な細工に偏り得るため、自然さ、遮蔽率、対象顔の教師保持、素材の構造、多様性を制約にする。

Gの目的の概念形は `L_G = λ_real L_realism + λ_keep L_preserve − λ_hard L_task` とする。Gはこの式を最小化する。L_taskには対象遮蔽点の位置とpresenceの扱いを明示し、座標を出さないだけで位置誤差が小さく見える指標を使わない。各項の定義と係数は実装前に固定し、Validationで比較する。これは既存runの損失変更を意味しない。

更新先は以下のように分離する。

- G更新時はDとTの重みを固定し、D/T入力からG出力への勾配は通す。hard maskや素材選択などの離散操作は、別途決めた学習方法を使う。
- D更新時はG出力をdetachし、Dの識別損失だけでDを更新する。
- T更新時はG出力をdetachし、Tの位置・presence・stateと特徴復元の損失でTを更新する。
- 合成前の教師画像・教師特徴・正解座標は固定する。Gに隠れたrecipient画素を入力しない。教師を改変したり遮蔽物へ正解情報を埋め込んだりする経路を作らない。

G/D/Tのoptimizer、更新開始時期、更新頻度、勾配量を記録する。融合warmup等で経路が無効な時期を、敵対的勾配でMessengerが学習した期間として扱わない。

### 遮蔽物の生成方法

初期段階ではtrainの別動画の実顔・非顔素材を使う。任意の背景cropは質感素材にはなるが、自然物体の輪郭を与えるとは限らない。物体輪郭を使う拡張ではmaskの取得・監査を準備する。素材内訳は勝手に変更せず、新policyとして固定する。

Gが最初に学習する対象は、素材の色・明るさと貼付側境界付近の小さなRGB補正とする。境界は常にぼかさず、実際の前景に必要な明瞭なエッジと色差を残す。binary paste、mask外画素不変、donor構造保持を守る。既存width48/strength2は固定補正の対照として保存する。

次に、素材選択・位置・大きさ・回転を学習する。面積だけでは同じ難しさにならないため、どのsemantic slotを覆うか、顔向き、遮蔽物種類による偏りも点検する。形状・配置の学習とRGB補正を一度に導入せず、各効果を切り分ける。

### 推論時の遮蔽判定

推論構成を次のように明示する。

`full-image FACE/ellipse → predicted Face crop → DINO特徴と遮蔽判定O → Messenger復元・融合 → Query Refinementによる位置・presence・state`

現行のOは固定DINO observerの特徴を、予測楕円の幾何から選んだk-means中心と比較し、低整合の場所を異常候補マスクにする。自然遮蔽だけを検知していると保証されたマスクではない。GとDを追加しても、この判定の精度が自動的に改善するわけではない。

学習可能なOを追加する場合は独立configで切り替え、無効時の既存経路・checkpoint契約を保つ。人工paste maskは追加した遮蔽の既知領域として教師にできるが、元からの自然遮蔽をすべて表すmaskではない。自然画像のstate1/state2点では点位置の検知を評価し、state0や未注釈pixelを無条件に遮蔽・非遮蔽へ割り当てない。密なmask精度を評価するには対応する自然遮蔽maskが必要である。

Oの改善は、Gによるaugmentation改善、codebook復元、D_featureの追加から分離して検証する。G D Tは今回採用する学習構成であり、学習可能なOへの置き換えやcodebook復元まで同時に決定したものではない。

### 比較を進める順序

1. 素材・mask・配置を一致させ、固定補正とGの学習補正を比較する。
2. 素材の見た目を一致させ、ランダム配置と学習配置を比較する。
3. 上記の各効果を確認後、Dの自然さとTの難しさを同時に使うG D T構成を比較する。自然さのみのG Dとの対照を残す。
4. 必要なら学習可能なOを独立比較する。人工遮蔽maskでの検知、自然点位置での検知、自然遮蔽keypoint位置推定を別々に報告する。

設定はtrainで学習しValidationで選択する。Testは凍結後の確認に限る。成功は自然遮蔽点の位置推定改善で判断し、overall・visibleの回帰と未出力の増加も確認する。各新armの実行には前節の独立preflight・fresh四gate・共通評価を適用する。

参考研究は設計の着想であり、本パイプラインの効果の証明ではない。

- [A-Fast-RCNN 難しい遮蔽を敵対的に作る研究](https://arxiv.org/abs/1704.03414)
- [SimGAN 注釈を保持して合成画像を補正する研究](https://arxiv.org/abs/1612.07828)
- [Learning to Generate Synthetic Data via Compositing 自然さと対象タスクを併用する研究](https://arxiv.org/abs/1904.05475)

## 追記 合意したGの遮蔽生成条件

2026-10-15、ユーザーは直前の条件評価と補足条件のすべてに賛同した。以下を新しいGの計画条件として採用する。合成は対象face crop内で行い、推論構成は引き続きHEAD-free FACE/ellipse → crop DINO → Oによる遮蔽候補判定 → Messenger復元・融合 → Query Refinementとする。この合意は計画書の更新であり、既存augmentation・完了runの条件変更や新規学習の起動ではない。

### 合成条件

| 項目 | 合意した条件 |
|---|---|
| 合成対象 | 対象face cropに合成する。crop側の遮蔽判定・特徴復元・keypoint位置推定の学習用とする。 |
| 最小遮蔽率 | 対象FACEの5%以上を実際に覆う合成だけを許可する。 |
| 遮蔽率の分母 | crop内の対象FACE楕円の画素数。crop全体やdonor外接矩形の面積は使わない。 |
| 遮蔽率の分子 | 最終的に貼り付けられたbinary maskと、対象FACE楕円の重複画素数。resize・切り落とし後の実支持領域で測る。 |
| 強度分布 | 20%境界で区間を分け、対象区間の目標件数を均等に割り当てる。区間と最大遮蔽率の扱いは下記。 |
| 透明度 | 変更禁止。完全不透明のbinary pasteとし、alpha blendingで隠れたrecipientを混ぜない。 |
| 素材比率 | 合成サンプルの顔crop50%・背景素材50%。各強度区間内でも半々を目標にする。 |
| 素材の出所 | 固定train split内の、recipientとは異なる動画から取得する。Val/Test素材は使用しない。 |
| 背景素材 | 顔を含まない素材として監査する。任意の背景cropに自然物体の輪郭が備わっているとは仮定しない。 |
| 色の調整 | 素材の構造を保つ範囲で、明るさ・色温度・コントラスト・滑らかな照明補正・境界RGB補正を許可する。 |
| Gの自由度 | 無制限の画素生成にはしない。素材の消去、構造の大幅な変更、微細ノイズでTだけを誤らせる細工を防ぐ制約を設ける。 |
| mask外 | recipient画素を変更しない。 |
| recipientの隠れる画素 | Gへ入力しない。正解情報が遮蔽物へ漏れる経路を作らない。 |
| 無加工入力 | 無加工cropも残す。顔/背景の50/50は合成例の内訳であり、全cropを必ず加工するという意味ではない。 |

遮蔽率は `area(M_paste ∩ E_target_in_crop) / area(E_target_in_crop)` とする。対象楕円の注釈/予測の出所、cropへの座標変換、支持領域の生成・補間規則を実装契約へ記録する。楕円が無効または分母が空の場合は合成成功とせず、見送り理由を残す。

### 強度区間と最大遮蔽率

20%境界の基本区間は `[5%,20%)`、`[20%,40%)`、`[40%,60%)`、`[60%,80%)`、`[80%,100%]` とする。ちょうど20/40/60/80%の例は上側の区間へ入れる。最初の区間だけ幅15%である。均等とは区間ごとの件数を揃えることであり、遮蔽率全体の連続一様分布と同じではない。

80〜100%の極端な遮蔽は、対象顔の情報がほぼ失われる影響を切り分けるため、通常学習へ自動的に混ぜず独立比較にする。通常学習の最大遮蔽率の具体値は今回の合意では未確定である。全5区間を対象にする独立比較では各区間20%を目標とし、上限を制限する通常armでは許可された区間を明示して、その対象区間間で均等割当する。最大値が区間途中の場合の最終区間の境界も実行前に固定する。

強度区間と素材familyは先に指定し、生成できない例を別familyや軽い区間へ黙って置き換えない。要求数、成功数、見送り数・理由、実現した遮蔽率分布を、区間×family別に保存する。要求分布が均等でも成功分布が偏る可能性があるため、実合成分布も監査する。有限件数の割当・許容差・見送り率の受け入れ値は実装前に定め、既存の別実験の閾値を自動継承しない。

合成の適用確率は、強度区間の割当と顔/背景比率から独立した設定にする。無加工cropを残すことは合意済みだが、新Gの適用確率の具体値は未確定である。

### 教師と学習対象の保持

- 既存augmentationに合わせ、元から輪郭非遮蔽かつ全5点visibleのrecipientを対象とする。crop外へ出たkeypointはmissingとして扱う。元注釈は保存し、合成例の派生教師だけを更新する。
- recipientのkeypoint座標を保持し、pasteに覆われたpresent点のみ遮蔽stateへ更新する。元のmissingを遮蔽やvisibleへ変更しない。
- donor顔のkeypointを対象recipientの教師へ混ぜない。他人の顔が重なった場合の誤帰属は、通常の位置誤差・未出力と分けて確認する。
- clean教師と合成studentは同じcrop幾何を共有する。donorの色補正によってrecipient教師や座標を変更しない。
- 顔遮蔽面積と、覆われたsemantic slot数は別々に記録する。5%以上の面積を覆っていてもkeypointを覆わない例はあり得る。
- このcrop合成はFACE検出器そのものの遮蔽耐性を学習させる条件ではない。独立した全画像detector/FORの未完了計画と置き換えない。

### Oの診断と新条件の検証順

G D T学習より先に、現在のOが何を異常と判定しているかを診断する。同じ人工遮蔽に対してOの候補maskと既知の追加paste maskを比較し、自然画像ではGT state1/state2位置で検知と誤検知を確認する。自然画像の密な遮蔽maskが存在するとは仮定しない。診断専用の既知人工maskを使った復元比較は、推論時Oを使う主評価と区別する。

Oが見逃した場所でも人工対の候補復元損失からMessengerは学習できるが、その場所では融合が無効なため、最終keypoint損失から復元候補へ至る融合経路が遮断される。O診断の目的は、この利用制限と復元そのものの失敗を分けることである。k-means中心の勾配更新案は別の比較候補であり、今回のG生成条件への賛同だけで採用確定とはしない。

新Gはまず既存固定補正と比較し、その後に配置学習・G D T構成へ進む。条件の有効性は自然Validationの遮蔽keypoint位置推定、5slot Macro F1、visible/overallの回帰、出力数・presence分布で確認する。自然に見えるだけでは最終目的の達成としない。設定はTest前に凍結し、独立preflight・fresh四gate・共通評価の条件を維持する。

## 2026-10-15追記：中心だけを学習するOの実装と小規模検証

ユーザーの新しい指示により、k-means中心を初期値とする学習可能な参照特徴を独立実装した。前節の「中心の勾配更新案は採用未確定」は、この実装指示により更新される。既存固定Oの実験・checkpoint・本学習契約は保存する。

推論topologyは **HEAD-free full-image FACE/ellipse → Face-crop DINO → Messenger geometry-prior fusion → Query Refinement**。今回の対象は既存k3 MessengerのOであり、現在のV3 local baselineへの導入・昇格ではない。

### 実装契約

- `src/models/messenger_prototype_observer.py` の `LearnablePrototypeBank.centers` だけを `nn.Parameter` にする。k3では3×384×16×16 = 294,912パラメータ。初期値は既存k-means中心と完全一致する。
- observer、DINO、Messenger、融合係数g、channel gate、Query、geometryによる中心選択、channel scale、類似度閾値を固定する。画像・特徴・geometryをdetachし、optimizerは中心テンソル1個だけを受け取る。weight decayは0。
- 類似度式は既存と同じ。遮蔽logitを `(固定閾値 − 類似度) / 0.1` とし、自然注釈state1を正例、state2を負例とする点位置のBCEを使う。0.1は固定の初期設定であり、validationで選んだ温度ではない。確率校正済みとは扱わない。
- state0、crop外、予測楕円support外を損失から除外し、未注釈patchを負例にしない。同じpatch内のstate1とstate2は別々の観測として残す。教師が0件のbatchはoptimizer更新対象にしない。
- xは右、yは下。正規化点座標を `pixel = normalized * (input_size − 1)` へ戻し、その位置を特徴gridへ対応させる。crop外座標を境界へclipしない。
- 初期中心からのchannel-scale正規化二乗距離に係数0.001を掛けたanchorを加える。クラス重みは使用するtrain集合の負例数/正例数を一度計算して固定する。validation/testから重みを作らない。
- 既存推論forwardではOが`no_grad`に含まれるため、中心学習には別の`observe()`経路を設ける。hard maskの比較演算へ擬似勾配を通さない。推論時は更新された中心による同じhard maskを既存融合経路へ渡す。
- `attach_learnable_prototypes(..., enabled=False)` は元モデルそのものを返す。enabled時は元モデルを複製して固定する。既存runのソースは変更しない。
- prototype checkpointは元bankのテンソル/provenance hashと温度を照合する。元bank・温度が異なるresumeを拒否する。optimizer状態からの再開一致もテストする。

### 実データによる実装検証

実行設定：`configs/training_plans/messenger_prototype_observer_smoke.json`。
実行コマンド：`.venv/bin/python scripts/verify_messenger_prototype_observer.py --output outputs/diagnostics/messenger_O_learnable_k3_smoke_20261015`。

既存k3 epoch19 EMA（SHA-256 `ef7dc88c1395e1871d432114003edab526f5ea61061a82d7bcad195934d81fa7`）を固定し、canonical trainの予測FACE cropからseed42のhash順で128 cropを抽出。train/val動画の非重複を確認した。自然点教師はstate1 63点、state2 476点。state0 93点とsupport外8点は除外した。合成augmentationなし、Adam lr0.001、batch64、40更新、CPU2 threadsで実行した。実装検証のための固定予算であり、全train本学習・本学習gate・W&B production runではない。

train損失は0.692325→0.572524。実際の勾配と更新は中心だけに限定され、全固定テンソルの更新前後hashは一致した。中心のL2変化量は5.907657。固定40更新後の1個だけを保存し、validationからepoch・閾値・設定を選択していない。Testは使用していない。

validationは従来診断と同じ768 predicted crop。表の分母はcrop内state1 433点、state2 2,863点（support外も含む）。これは自然遮蔽の**点位置におけるO判定**であり、密な遮蔽segmentationやkeypoint位置推定の評価ではない。

| 指標 | 固定k-means中心 | 学習可能中心・小規模検証 |
|---|---:|---:|
| TP / FP / FN / TN | 215 / 825 / 218 / 2038 | 218 / 727 / 215 / 2136 |
| 遮蔽precision | 20.67% | 23.07% |
| 遮蔽recall | 49.65% | 50.35% |
| visible誤検知率 | 28.82% | 25.39% |
| 遮蔽判定F1 | 29.19% | 31.64% |
| AP | 0.20596 | 0.22577 |
| AUROC | 0.63139 | 0.65882 |

小規模検証では、主にvisibleのFPを98点減らし、遮蔽TPを3点増やした。Oを遮蔽判定へ適応させる方向性は確認できたが、全train学習での再現性や最終keypoint改善は未検証。support外のstate1 39点は中心の学習では救えず、残りのsupport内FNも179→176にとどまる。gは前回診断でほぼ閉じており、今回も固定しているので、O改善から融合・位置推定改善を推定しない。

成果物：`outputs/diagnostics/messenger_O_learnable_k3_smoke_20261015/` の `metrics.json`、`loss_history.json`、`train_manifest.json`、`image_hashes.json`、`prototypes_last.pt`。source/annotation/parent/checkpoint hashesを保存。既存parentの後日の無関係なsource変更は明示し、過去gateを現在のgateと認定していない。

検証：`tests/test_messenger_prototype_observer.py`、既存O診断テスト、既存geometry-priorテストの計34件がpass。無効時のモデル同一性、初期O score/mask/融合特徴の完全一致、凍結・勾配・更新範囲、重複patch教師、境界/欠損除外、正負それぞれの更新方向、checkpoint拒否・optimizer再開一致を確認した。モデル凍結に伴う下流attention演算のfloat32丸め差は最大約1.2e-7で、座標出力は一致した。

次の本学習には独立したproduction launcher/W&B/resume、fresh四gateとisolated4/8/16 preflight、固定Oとの同条件比較、共通val/test評価が必要。この小規模検証ではそれらを完了扱いにせず、canonical本学習queueやbaseline registryを変更していない。

## 2026-10-15追記：第一段階の実装とF1 checkpoint選択

ユーザーの「実装してください」「checkpointは基本的にF1選択に使用」という指示を反映した。今回実装したのは、Oの中心だけを学習する経路と、固定素材・mask・配置に対するRGB補正G/D/Tの第一段階である。素材・形状・配置の学習、80〜100%遮蔽、codebook復元、D_feature、V3 baselineへの導入は、前段に記した別実験のままとする。

### F1の定義と保存・再開

| 学習対象 | validationによるcheckpoint選択 |
|---|---|
| O単独 | `occlusion_f1`：自然state1を正例とする点位置の遮蔽判定F1。crop内のstate1/state2を分母とし、support外の見逃しもFNに含める。既存位置別閾値を固定する。 |
| fixed / GD / frozen-G / GDTのT | `conditional_semantic_macro_f1_nme_0.10`：既存の予測FACE関連付け後、5semantic slot等重み、presence0.70、GT Head side = 1.20×max(width,height)。 |

Oのbinary F1とkeypointのsemantic Macro F1を同じ指標として混ぜない。AP、学習損失、Dのreal/fake識別精度はbest選択に使わない。GD/GDTでは、TのF1が最大になったepochのG・D・Tを一緒に保存する。これでGを「Dを欺いた度合い」だけで選ばない。

bestはF1が厳密に上がるたびに更新し、同点は先のepochを維持する。early stoppingのmin_delta1e-4はpatience判定だけに使い、微小なF1上昇をbest保存から除外しない。production既定は最大40epoch、minimum_epoch10、patience5、warmup5直後からcosine decay。lastは毎epoch、回復snapshotは5epoch間隔・最新1個、正常完了時はlast/bestだけ残す。

新経路はraw weight・FP32学習を明示し、EMAは無効。module、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、loader RNG、F1選択状態とsource/data/parent契約を保存・照合する。smoke checkpointをproductionのO/G初期値として受け入れず、完了runの自動再試行もしない。

### 実装した生成・学習条件

- 初期設定は、資格を満たすrecipientの50%に合成を要求する。全cropの50%という意味ではない。今回のtrain predicted crop 6,185中、資格を満たす2,546に対し1,273要求となる。残りは自然入力として保持する。
- 通常armは `[5%,20%) / [20%,40%) / [40%,60%) / [60%,80%)` の4区間。区間×顔/非顔の8層へ可能な限り同数を割り当て、端数差は最大1件。familyや区間を途中で変更しない。
- binary pasteの実支持領域と予測FACE楕円の重複面積で強度を検査する。source-video、素材、配置、実現率、覆ったslot、見送り理由を追跡する。
- 顔素材は異なるtrain動画を選び、その動画内の顔から選択する。recipient全体のRGB記述子で素材を選ばず、隠れたrecipient画素が素材選択へ入る経路も避ける。非顔素材は既存bankの注釈済みFACE除外boxに重ならない領域を使用する。未注釈顔が存在しないことまではこのbankだけで保証しない。
- Gはmasked donor・mask外のrecipient・binary maskだけを入力する。4×4の低解像度RGB補正場を滑らかに拡大し、補正量を各channel最大0.12（RGB 0〜1）へ制約する。mask外は元画素と完全一致し、alphaは常にbinary。補正場を滑らかにする処理とpaste境界をぼかす処理を区別する。
- Dは遮蔽semantic点を中心とする32×32 RGB patchとslotを、real/fake両方で同じ規則により受け取る。crop端で同じ領域を取れない例は両側から除外する。realはtrain自然state1点から、同じslot・source楕円のscale/ratio/angle bin、異なる動画で対応付ける。明るさは観測可能なraw合成画像とreal画像の平均RGB差0.2以内を使い、隠れる前のclean RGBは対応付けに使わない。これは密な自然遮蔽maskの代用評価ではない。
- G損失は自然さ1.0＋RGB保持10.0−難しさ0.1。難しさには覆った点の座標誤差とpresence損失を含め、出力を消して座標誤差だけを逃れることを防ぐ。これらの係数・上限はvalidationで選択済みの値ではなく、明示した初期設定である。
- G更新ではD/Tの重みを固定し、入力画像への微分だけを残す。D更新ではfakeをdetachし、T更新でもG出力をdetachする。OとobserverはG/D/T段階では固定する。各損失・module別勾配normを記録する。教師がないO/G/D更新ではoptimizerとschedulerを進めず、過去のmomentumだけで重みを変えない。
- Oは中心だけに勾配を通す。中心×semantic slot×空間位置ごとの正負教師数、未監督cell、同一sample内の相反ラベルcellを監査する。更新後の中心は遮蔽判別用の参照特徴であり、正常顔の平均や復元正解として転用しない。
- 独立armとして `observer / fixed / gd / frozen_g / gdt` を実装した。`fixed`は既存width48/strength2補正を使う。`gd`はGのtask難化項を無効にした自然さ目的の対照で、Tは同じ合成を使って学習する。`frozen_g`はF1選択済みbundleのGを固定して新たなTを共通初期値から学習する。G単独の自然さ改善を最終成功とはしない。
- 新policyの監査初期値はaggregate見送り10%以下、各層20%以下、各層10要求以上、合成成功例の自然real対応率50%以上。8層全ての証拠を必要とする。これは既存real-faceの7%契約を変更・継承するものではない。full-train監査の合格はまだ確認していない。

### コードと実行経路

- 設定：`configs/training_plans/messenger_natural_occlusion_v1.json`
- O指標・監査・更新・F1保存：`src/models/messenger_occlusion_training.py`
- G/Dと勾配分離：`src/models/messenger_occlusion_augmentation.py`
- T adapter・G/D/T更新・OOM fallback：`src/models/messenger_natural_task.py`
- train素材・配置・real対応・分布監査：`src/data/messenger_natural_occlusion.py`
- source/data/親checkpoint照合：`scripts/messenger_natural_occlusion_common.py`
- 学習runner：`scripts/train_messenger_natural_occlusion.py`。既定はCPU smoke。production経路はarmとoptional O/G初期値まで一致するfresh四gateとisolated4/8/16証拠を要求し、W&Bを有効にする。新gateの発行・GPU preflight実行・canonical queue登録は今回行っていない。
- 最終版の実素材更新検証：`scripts/verify_messenger_natural_occlusion_step.py`

例：`.venv/bin/python scripts/train_messenger_natural_occlusion.py --arm observer --output <新しいsmoke出力先>`。同じrunnerの`--arm gdt`でG/D/Tを確認できる。既存出力先の上書きを拒否し、再開には`--resume`と同一契約を必要とする。

### 実施した検証と結果の限界

関連テスト49件がpass。Oだけへの勾配、G/D/T別の勾配先、凍結Tを通るGの入力勾配、hidden recipient不変性、mask外完全一致、区間/family割当、実支持領域、F1同点処理・微小改善・保存個数、完全resume一致、教師なしmomentum更新の抑止、OのOOM replay、CPUのRNG保存がCUDA contextを初期化しないことを検証した。

F1保存経路のCPU smokeは、それぞれfull canonical validation 768 cropでepochごとに評価した。Oはtrain128 crop・2epoch/4更新でF1 0.291921→0.295284、bestはepoch2。G/D/Tはtrain16 crop・2epoch/2更新で、更新前Macro F1 0.887594、epoch1 0.887556、epoch2 0.885175となり、epoch1をbestとして保持した。後者は更新前より改善しておらず、品質合格とは判定しない。2epoch smokeはその実行時source snapshotとともに保存し、最終の素材選択・RGB情報漏洩対策後には別の実素材1更新検証も実施した。

最終版8素材例の検証では、G/D/Tの各勾配normが非ゼロで有限、O/observerは更新なし、mask外画素完全一致、hidden recipient画素変更に出力不変を確認した。成果物は以下に保存した。

- `outputs/diagnostics/messenger_natural_v1_O_f1_smoke_20261015/`
- `outputs/diagnostics/messenger_natural_v1_GDT_f1_smoke_20261015/`
- `outputs/diagnostics/messenger_natural_v1_final_step_v2_20261015/verification.json`

**本学習・GPU preflight・fresh四gate・新armの共通val/test評価・baseline昇格はnot run。** 小規模G/D/Tでは要求8件のため層別監査の必要件数を満たさず、監査をpassedにしていない。自然さの独立評価、反復seed、全train学習での効果、最終keypoint改善も未確認である。既存の本学習queue・完了run・現在のV3 local baselineは変更していない。

## 2026-10-18追記：合成方策Gの確定条件

この節は先行のRGB限定・32×32局所D・遮蔽率区間・presenceを含む難化項について、次の実装の設計条件を更新する。2026-10-15の実装と保存済み結果は当時の契約のまま保持する。今回は条件の文書化であり、新方策の実装・学習・評価はnot run。

### 適用範囲と入力

- 遮蔽augmentationとG/D/Tの敵対的更新はtrain時だけ実施する。recipient、素材候補、Dの自然遮蔽realは全て固定train splitに含まれるframeから取得する。val/testを素材bank、報酬計算、方策更新へ流用しない。validationは自然入力によるF1選択に使い、testは条件固定後の評価だけに使う。
- Gは素材の組み合わせ・大きさ・位置等を選ぶ合成方策とする。RGB補正だけに限定しない。recipientの資格条件、異動画donor、素材provenanceを維持する。素材数・回転・形状等の具体的な行動空間は実装時に明示する。
- 合成はbinary maskによる完全不透明な貼り付け。透明度の学習・alpha blendingを行わず、recipientのmask外画素を変更しない。隠れたrecipient画素を使う素材補正や復元正解の漏洩を避ける。
- Dの主入力は顔crop全体のRGBとする。real/fakeのcrop生成・解像度・条件入力をそろえ、fake側だけに合成maskやclean画像を渡さない。局所Dだけで全体の自然さを判断しない。Dのreal/fake分類能力と、背景・動画・撮影条件への依存を別途監査する。
- 推論はHEAD-free ordered `full-image FACE/ellipse -> Face-crop DINO -> Messenger geometry-prior fusion -> Query Refinement`。G/Dは推論に含めず、外部HEADを追加しない。現行V3 local baselineの自動変更は行わない。

### 遮蔽率と分布

ユーザー指定の4区間を `[0.05,0.20) / [0.20,0.40) / [0.60,0.80) / [0.80,1.00]` とし、各25%とする。40〜60%区間は追加しない。これは先行の5〜80%の連続4区間を置き換え、80〜100%を通常の新方策に含める指示である。

遮蔽率の測定は既存の実支持領域の定義を継承し、crop内の予測FACE楕円のpixel集合Eに対して `|M ∩ E| / |E|` とする。複数素材の場合はmaskの和集合で測り、重複を二重計上しない。素材box面積や隠れた点の割合で代用しない。

外側の割当で各区間を同数（端数差最大1）にし、Gは指定区間内で素材・幾何条件を選ぶ。Gが区間自体の比率を崩さない。要求数だけでなく成功合成の実測分布も記録し、配置失敗を別区間への変更で埋め合わせない。層別失敗・不足を明示し、成功例が偏った状態を等分布達成としない。未合成の自然train画像の比率は、この合成4区間の25%ずつという条件と区別する。

80〜100%でも、recipientの元から位置が注釈されたcrop内の点は、覆われたらstate1として位置教師を維持する。遮蔽だけを理由にstate0やno-faceへ変えない。元から位置不明のstate0やcrop外の点は位置損失から除外する。

### 自然さ・位置難化の目的関数

以下は全optimizerが損失を最小化する記法。Dのlogitをs、実画像確率を `D(x)=sigmoid(s(x))`、合成画像をx_syn、自然遮蔽train画像をx_realとする。

```
L_D       = E[softplus(-s(x_real))] + E[softplus(s(stopgrad(x_syn)))]
L_natural = E[softplus(-s(x_syn))] = -E[log D(x_syn)]
L_G       = lambda_natural * L_natural - lambda_position * L_position
L_T       = +lambda_T_position * L_position + L_T_other
```

自然さはNon-saturating GANのG損失を使う。Gで `log(1-D(x_syn))` を最小化するsaturating版や、Dの損失全体の単純な符号反転へ置き換えない。原典：[Goodfellow et al., Generative Adversarial Nets, Section 3](https://papers.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)。既存RGB版にもこの自然さ損失の形式はあるが、顔全体Dと合成方策の実装は別途必要である。

L_positionは同じ合成画像・GT位置・有効点集合・座標正規化・reductionで評価するkeypoint位置誤差項とし、T側は小さく、G側は大きくする。初期実装候補は既存のdecoded座標に対するSmooth L1、対象は合成で覆われた位置既知のcrop内recipient点。Tの他の可視点への位置教師は維持する。厳密な座標尺度とloss係数は実装設定に固定・記録する。

Gの難化項にはpresence BCE、state分類損失、Tの総損失を混ぜない。T自身のpresence/state学習はL_T_otherとして独立に維持できる。予測presenceによってL_positionの点を除外・重み付けせず、出力confidenceを下げても位置教師が消えないようにする。位置教師がない例に人工的な位置損失を作らない。

符号が逆になるのは共有する位置誤差項であり、自然さを含むG損失全体がT損失の負になるわけではない。lambdaは非負とし、損失値とGへの勾配量を項別に記録する。係数は未選択で、Testを使って調整しない。追加の正則化を採用する場合は独立項として宣言し、位置難化へ混入させない。

### 更新経路と自動化

- D更新：G/Tを更新せず、fakeをdetachしてreal/fakeを識別する。
- G更新：D/Tのparameterを固定して、自然さと位置誤差でGを更新する。微分可能なRGB変換等では画像を通る勾配を保持する。素材ID・hard maskの離散配置は通常の画像逆伝播だけでは更新できないため、同じ目的から得る報酬 `lambda_position * L_position - lambda_natural * L_natural` を使う方策勾配等を明示的に実装する。報酬をdetachしてlog-probabilityへ渡し、T/Dへ勾配を漏らさない。
- T更新：合成結果をdetachして通常の教師学習を行う。Gによる難化とTによる誤差低減を同じ符号にしない。Oの中心学習は独立し、G/D/T段階ではO/observerを固定する。
- 自然さの比較例がないという理由で自然さ項だけを無効にして難化を続けない。全体Dに対するtrain realの供給を確保し、不足時の方策更新は明示的に見送り・記録する。
- train候補取得、区間割当、G選択、合成、mask由来の教師更新、D/G/T更新を自動化する。方策と合成器のseed、候補集合、行動、実現遮蔽率、失敗、optimizer状態を再開可能に保存する。
- checkpoint選択はvalidationの条件付き5semantic-slot Macro F1 NME0.10を維持する。DスコアやG報酬の最大化だけを成功・checkpoint選択基準にしない。新実装の独立gate・preflight・生産学習順序は既存契約を維持する。

## 2026-10-18追記：素材・サイズ・配置の方策Gと顔全体Dの実装

新しい独立経路を `configs/training_plans/messenger_natural_policy_v2.json` として実装した。上記の確定条件に対する実装であり、旧RGB版の設定・コード・保存済みrunを置き換えない。以下の範囲は実装済みだが、本学習・fresh四gate・isolated GPU preflight・canonical登録はnot run。

### Gと合成器

`src/data/messenger_policy_occlusion.py` は認証済みtrain bankから異動画の素材を取得する。1recipientにつき初期設定では4素材（顔2・非顔2）のそれぞれに最大2通りのサイズ・平行移動を提案し、実支持領域の遮蔽率が指定区間に入る最大8候補を作る。Gはこの候補集合から素材と幾何変換の組を確率的に選ぶ。顔/非顔は候補の内訳であり、採用素材のfamily比を50/50へ強制しない。

Gは候補のdonor RGB preview、配置・サイズ・実測遮蔽率、指定区間、recipientの予測楕円幾何から各候補のlogitを出す。recipient RGBは入力せず、別候補の可視画素経由で選択後の隠れた画素を読むことも避ける。現在の行動空間は1素材の矩形cropの拡大縮小・移動・crop端での切り取りであり、任意形状生成・回転・複数donorの同時合成は含めない。サイズと位置は候補pool内で学習して選択する方式で、連続値の直接回帰ではない。通常のdataset利用ではepoch seedで候補を更新でき、bounded smokeでは同じ候補集合を再利用する。

5〜20/20〜40/60〜80/80〜100%を外側で同数割当する。最終区間のみ上端100%を含む。候補不足や自然real不足で偏りが出る場合、`PolicyUpdate` はeffective window内の利用可能な各区間の最小件数に合成数をそろえる。超過分は自然画像としてTを学習し、`bin_balance_deferred` として記録する。4区間のどれかがゼロなら、そのwindowの合成G/D更新は行わない。別区間へ付け替えず、要求数・実施数・見送り理由を分ける。この方策は偏りを防ぐ一方で合成実施率を下げ得るため、本学習へ接続する際は各区間をそろえたbatch samplerとfull-train実施率監査が必要である。

合成画素は `torch.where(binary_mask, donor, recipient)`。覆った元位置既知の点をstate1へ更新し、crop外はstate0へ更新する。元からstate0の点は位置教師にしない。位置座標自体は変えず、100%遮蔽でもcrop内位置教師を保持する。

### Dと目的関数の実装

`src/models/messenger_occlusion_policy.py` の `FullCropDiscriminator` は顔crop全体のRGBを受け取り、畳み込みと空間4×4のheadで1logitを出す。realは同じ予測楕円幾何bin・異なるtrain動画・state1を含む顔cropから選ぶ。全体Dでは32×32切り出し・点中心のcrop端除外は行わない。密な自然遮蔽maskや正確な遮蔽物種別を持つと仮定せず、実画像側の遮蔽面積との厳密なmatchingは未実装である。

Gの自然さコストは `softplus(-D_logit(fake))`、Dはreal/fakeのlogistic損失。難しさは覆われたcrop内位置既知点の正規化xy Smooth L1（各点でxy平均、cropごとに有効点平均）。presenceの値・閾値・損失を含めない。Gの報酬は `0.1 * position_error - 1.0 * natural_cost`、最小化する方策損失は `-stopgrad(reward - baseline) * log_probability`。係数は明示した初期値で、validationで最適化済みとは扱わない。

Tには同じ位置誤差項を係数+1で加え、既存の全点landmark損失・特徴復元損失も保持する。したがって覆った点の位置教師は既存の全点項にも含まれ、この追加項でその重みが増える。Gの難化項へTの総損失を渡さない。G/D/T間の比較でこのT損失定義をそろえる必要がある。

離散選択にはCategorical REINFORCEを使う。D/Tの評価値をdetachし、Gのlog-probabilityへだけ勾配を流す。RGB版のようなTから合成画素への直接微分ではなく、位置誤差を報酬にした方策勾配推定である。前回までのrewardのEMA（0.9）をbaselineに使い、現在の行動の勾配には含めない。baselineはGのbufferとしてcheckpointへ保存する。

### 更新・検証用runner

`PolicyUpdate` はG/D/Tのoptimizer所有権を検査する。全勾配の有限性を確認してから各optimizerを更新する。O/observerは固定。自然realのない例でGの難化だけを進めず、G/D教師なしではmomentumによる更新も止める。effective windowの行動を先にsampleし、microbatch OOM時は同じ行動・更新前buffer・RNGで再試行する。部分commit時は完全checkpointからの復元を要求する。

実行入口は `.venv/bin/python scripts/train_messenger_natural_policy.py --output <新しい出力先>`。現時点ではCPUによるbounded implementation smoke専用で、production有効化は拒否する。既存k3 best EMAと固定Oから開始し、smoke中はbackboneも固定する。学習済みOとの統合、現在のV3 baselineの比較ではない。

smokeは合成各区間2cropと自然8cropの16crop、2epoch/2更新を既定とする。feasible fixtureを探した履歴を全て保存し、full-train分布監査とは区別する。validationは既存条件の768 predicted FACE crop、conditional semantic Macro F1 NME0.10・presence0.70でbestを選択する。G/D/T、全optimizer、reward baseline、RNG、loader RNG、source/data契約の保存・再開を共通F1Checkpointへ接続した。本学習品質や収束をこの2更新で判定しない。

検証テストは `tests/test_messenger_occlusion_policy.py`。列挙可能な2行動でREINFORCEの期待勾配が期待コストの直接微分と一致すること、位置誤差増大/自然さコスト低下の選択方向、G/D/Tの勾配分離、presence独立、full-crop Dの周辺画素への感度、全遮蔽の教師、4区間・境界・実支持率、自然real欠損時の更新停止、checkpoint再開一致、OOM時の行動再利用、実Messenger headへの位置勾配を検証する。

### 実行した検証結果

新規16件と既存49件の計65テストがpass。実素材CPU smokeも正常完了し、train16 crop（合成8・自然8）、seed42、effective batch16/microbatch2、2epoch/2更新を実行した。G/DはAdam lr1e-4、TはAdamW lr1e-4/weight decay1e-4、FP32、smoke中のschedulerなし。初期重みは先行k1〜7のvalidationで選ばれたk3 best EMA（SHA `ef7dc88c1395e1871d432114003edab526f5ea61061a82d7bcad195934d81fa7`）。固定canonical-v2のtrain予測cropを使い、5train動画から選ばれた合成recipient8件は全て8候補を生成できた。fixture探索の見送りは0件。各更新の合成成功数は `[2,2,2,2]`、全て指定区間内だった。

G/D/Tのpre-clip勾配normは1更新目 `0.08650 / 0.09948 / 1.13778`、2更新目 `0.07831 / 0.09071 / 1.39053` で、全て有限かつ非ゼロ。O・observer・固定backboneの全tensorは更新前後で一致した。

validation768 cropのconditional semantic Macro F1 NME0.10は更新前 `0.8875939582`、epoch1 `0.8865177060`、epoch2 `0.8880295223`。F1選択bestはepoch2で、best checkpoint SHAは `8095f40fd29f022c2ca85b0a16b12e88a8a9a54ff0790fd22ac3ad489e20c375`。last/bestのみを保持した。初期値との差は約+0.0436 percentage pointsだが、わずか2更新・単一seed・matched control未実行の動作検証であり、敵対的方策の効果や品質gate合格を意味しない。

成果物：`outputs/diagnostics/messenger_natural_policy_v2_smoke_20261018/` の `implementation_verification.json`、`tests.txt`、`fixtures.json`、`history.jsonl`、`contract.json`、`source_snapshot/`、`checkpoints/{best,last}.pt`。新実装sourceの実行前後hash一致も確認した。W&B production run、本学習、GPU preflight、fresh四gate、test評価、baseline昇格はnot run。

## 2026-10-18追記：人工遮蔽・自然遮蔽・可視点のg推移を出力

ユーザーの追加指示により、位置誤差と併せて採用係数gの状態と変化を記録する。gは復元特徴の採用係数であり、遮蔽確率やOの異常判定mとは区別する。この追加で学習損失・Gの報酬・F1 checkpoint選択を変更しない。

`src/models/messenger_gate_metrics.py` の定数メモリ集計を導入した。比較単位は位置が既知でcrop内にあるkeypoint。人工遮蔽は既知paste maskで覆った点、自然遮蔽は追加pasteで覆っていないstate1点、遮蔽なしはstate2の可視点とする。可視点を含む顔全体が非遮蔽とは限らない。state0・crop外・不正座標を除外し、予測presenceでは除外しない。点をcrop pixel座標へ戻し、`floor(pixel * feature_grid / input_size)` のtokenからgを取得する。複数点が同じtokenに入っても、点単位の分母を維持する。

各集合について点数、gの平均・標準偏差・最小/最大、g<0.01とg<0.001の割合、O陽性割合、予測楕円support外の件数、実効融合係数 `m*g*mean(sigmoid(channel_adoption))`、位置誤差Smooth L1を出力する。位置誤差はcrop正規化xyに対するもので、F1評価のGT Head基準NMEと混ぜない。空集合は0ではなくnull、初期g平均が0のとき相対変化はnullとする。

今後のpolicy runnerは初期値と毎epochで次の2系統を記録する。

- 未加工validation：既存のF1評価forwardを再利用し、自然遮蔽点・可視点を集計する。`validate_prior` にoptional diagnostics callbackを追加した。無効時の既存経路・戻り値は変えず、validationに人工遮蔽を加えない。
- 固定train診断：準備した各recipientの候補0を固定し、同じ素材・サイズ・配置・点集合をeval modeで追跡する。合成後と合成前の同じ点のgを別々に記録し、人工点は遮蔽率4区間別にも出力する。Gの選択方策の変化とg自体の学習変化を混同しない。診断はGの乱数を消費せず、重み・requires_grad・学習モードを保持する。

初期値は `epoch0_validation.json` と `gate_reference.json`、epochごとの結果は `history.jsonl` 内の `gates`、最新値は `gate_progress.json` に保存する。標準出力にも人工遮蔽・自然遮蔽・可視点の集計を表示する。初期値に対する平均gの絶対差・相対変化、実効係数と位置誤差の差も記録する。異なる分母を黙って比較しない。

保存済みrunには `.venv/bin/python scripts/report_messenger_natural_gates.py --run <保存済みpolicy run> --output <新しい診断出力先>` を使用できる。この診断では候補poolの完全再現とcheckpoint/source/dataのprovenanceを確認し、第1更新で実際に選ばれた行動を固定して、初期と保存済みbestを比較する。過去のhistoryを書き換えず、存在しないepochのgを補間しない。人工遮蔽はtrain画像にだけ適用する。過去のF1の再現も確認し、詳細は独立した `metrics.json` / `report.md` に保存する。

### 保存済み2更新runの計測結果

新規4件を含む関連69テストがpass。集計が予測presenceに依存しないこと、同一点のtoken割当、欠損・crop外・support外の扱い、空集合のnull、初期値との差、固定診断による重み・モード・RNG不変、既存validationと同じforward回数・F1を検証した。

`messenger_natural_policy_v2_smoke_20261018` の初期重みとbest epoch2をCPUで再計測した。追加学習はnot run。合成入力は第1更新の実際の行動を固定し、人工20点、未加工validationの自然遮蔽433点・可視2863点を比較した。両checkpointのF1は保存済み値と一致した。

| 条件 | 初期g平均 | epoch2 g平均 | 相対変化 | 初期位置誤差 | epoch2位置誤差 |
|---|---:|---:|---:|---:|---:|
| 人工遮蔽・train固定20点 | 0.000407191 | 0.000391834 | -3.77% | 0.011217561 | 0.010332253 |
| 自然遮蔽・val433点 | 0.000378359 | 0.000301481 | -20.32% | 0.002550756 | 0.002519119 |
| 可視点・val2863点 | 0.000411979 | 0.000309829 | -24.80% | 0.001052355 | 0.001061899 |

位置誤差はcrop正規化xyのSmooth L1であり、集合間の平均差を因果効果として比較しない。3集合とも全点でg<0.01。実効融合係数の平均も、人工 `0.000111430 -> 0.000108331`、自然 `0.000100495 -> 0.000079839`、可視 `0.000057053 -> 0.000043381` と低下した。少なくとも今回の2更新では、遮蔽時の復元採用を増やす変化は平均値に現れていない。位置誤差やF1の微小変化を、gが開いた効果とは説明できない。

成果物は `outputs/diagnostics/messenger_natural_policy_v2_gates_20261018/{metrics.json,report.md,verification.json,initial.json,best.json,tests.txt}`。既存runのhistory・checkpointを変更していない。本学習・test評価・baseline変更はnot run。

## 2026-10-18追記：今回だけの精度改善gate例外

ユーザーは今回の合成方策G/顔全体Dの実験に限り、精度改善量の最低値を撤廃し、少しでも向上すればよいと明示した。この指示を新しい受入条件として扱う。2更新では学習不足の可能性が大きいが、g低下の原因が学習不足だけであると確定したとは扱わない。

今回のfresh small-scale gateの精度条件は、同じrunの更新前epoch0を基準に、同一validation集合・条件付き5semantic-slot Macro F1 NME0.10・presence0.70で `best subsequent F1 > epoch0 F1` とする。改善差は厳密に正ならよく、1e-4等の下限を設けず、丸めた表示値で比較しない。同値・低下・非有限値は不合格。

例外は `configs/training_plans/overrides/messenger_natural_policy_v2_small_scale_once_20261018.json` に独立保存し、実験config SHAと `messenger_natural_policy_v2_small_scale_20261018` という単一gate run IDに限定する。共通既定値、旧geometry-prior/feature-adversarial gate、他run、将来のconfigへ継承しない。base configへ自動適用設定を追加しない。

`src/models/messenger_policy_quality_override.py` は明示された例外の数値条件だけを評価し、run/stage/split/config hash/validation契約の不一致を拒否する。成功markerは発行しない。新方策のfresh small-scale実行系と本学習への接続は引き続き未完了であり、その接続時にこの明示overrideを読み込む。サンプル数・更新予算・他gate・有限性・合成分布・GPU安全性などの条件は緩めない。early stopping用min_deltaとcheckpoint best保存の契約も別用途として維持する。

既存の2更新smokeを正式gate合格へ読み替えない。なお、既存smokeのF1差約0.00043556は1e-4よりも大きく、正式gateとして扱っていなかった理由は改善幅の不足ではなく、必要なsmall-scale実行と認証を完了していないためである。今回の変更で新規学習は開始していない。

## 2026-10-18追記：正式small-scale gateと本学習の接続

今回専用の数値例外を `scripts/run_messenger_natural_policy_gate.py` のsmall_scale判定に明示接続した。固定済みpolicy-v2 configのSHAと過去smokeは変更せず、実行条件を `configs/training_plans/messenger_natural_policy_v2_execution_20261018.json` に分離した。旧configのproduction=falseは旧smoke runnerの制限として維持し、新しい実行系だけが四gate認証後に本学習を許す。

推論はHEAD-free ordered FACE/ellipse -> crop DINO -> Messenger prior fusion -> Query Refinement。G/Dは学習のみ。初期化は既存のvalidation選択k3 epoch19 EMA（ef7dc88c1395e1871d432114003edab526f5ea61061a82d7bcad195934d81fa7）。学習済みOの別実験や現行V3 complete local baselineへの自動置換は行わない。

- schema: 関連82 testsと、全train 6,185 crop中の資格を満たす2,546 cropを母集団としたepoch1/20/40の合成監査。各epoch1,273要求。成功数を4区間同数にそろえるための除外もskipへ計上し、aggregate <=10%、区間 <=20%、各区間要求 >=10、matched real >=50%を保持する。
- gpu_step: 各4/8/16を別processで測る。各4回のeffective64更新、backboneを含む全経路、全768 valを2回、最初2更新をtiming warmupとして除外。温度<83C・物理VRAM<=90%・有限性・OOMなしを満たす候補の予測所要時間を比較し、3%以内なら小さいbatchを採用する。
- overfit: 固定8入力・各区間1合成と4自然画像、合成actionも固定、24更新。固定入力をevalで再計測したT損失の厳密な低下を確認する。checkpoint選択はF1。epoch1で別processへ保存・再開し、G/D/T・optimizer・scheduler・RNGと同じonline W&B runを復帰する。
- small_scale: feasibilityで選別しないseed42 hash順の256 train crop、20 epoch、effective64で80更新。全768 validationを初期および毎epochで評価し、途中早期停止しない。best subsequent conditional semantic Macro F1 NME0.10 > 同run epoch0を唯一の改善幅条件とする。有限性、全validation、予算完了、実際の合成分布・skip条件は緩めない。
- production: 全6,185 train crop、最大40 epoch。四gateすべてのsource/data/config/証拠hashを再検証し、gate checkpointは流用せず同じk3からfresh開始。seed42、effective64、FP32更新、backbone最初5 epoch固定、5 epoch warmup後直ちにcosine減衰、backbone LR1e-5/その他T LR1e-4、G/D各1e-4。minimum epoch10/patience5/min_delta1e-4は変更しない。raw validation F1でbest選択、last毎epoch、5 epoch間隔recoveryは最新1個、成功後best/lastのみ。

small-scale/productionではepochごとに素材・位置候補を更新し、同じepoch内の選択対象を重複なく使用する。外側で4区間要求を均等化し、effective windowに各区間を分配する。素材・配置が不成立なら自然入力でTを学習し、G/Dへ空の報酬を与えない。欠損区間に合わせて成功合成も同数化し、余剰除外を記録する。runtime OOM fallbackは既存PolicyUpdateのサンプルaction/RNG/buffer replayを使用する。

固定train診断32合成（各区間8、候補0を固定）では人工遮蔽90点を追跡する。未加工valでは自然遮蔽433点・可視2863点に加え、`face_occluded=false AND original_visible.all()` の顔だけの可視1790点を別集計する。後者を「遮蔽なしの顔」と明示し、全可視点集合と混同しない。gは復元特徴の採用係数でありOの遮蔽確率ではない。平均g・実効m*g*channel・位置Smooth L1・母数を初期と毎epochに保存する。検証F1は既存validate_priorのCUDA autocast FP16、固定g診断はFP32であり、それぞれ同一経路内で比較する。

canonical queueに四gate、本学習、共通val/testの7 jobsを登録した。`scripts/run_image_policy_sequence.py` は今回の列だけを実行し、失敗gateを自動再試行せず、本学習完了で止まる。common評価は `scripts/evaluate_image_policy.py` で全FACE予測を処理する独立jobとして登録し、testはvalidation終了後にのみ適用する。現時点の実行結果は `outputs/training_gates/messenger_natural_policy_v2_20261018` とdevelopment journalを正とする。schema・GPUは通過、overfit以降を実行中であり、未完了stageを合格とは扱わない。

online W&Bは `sato-waseda-university/face-keypoint`。詳細historyは各stageの `history.jsonl`、人間向け推移図・表は `scripts/report_image_policy_learning.py` で別出力へ作る。既存gate evidenceを書き換えて報告を追加しない。

### 正式実行結果

schema・GPU・overfit・small_scaleの四gateは全て通過。small_scaleは256 crop/20 epoch/80更新、F1初期0.8875939582→best epoch3 0.8888636613（+0.0012697030）。最終epoch20は0.8783331304へ回帰した。人工/自然/遮蔽なしの平均gは最終0.0261992/0.0212386/0.0256954で、遮蔽なしにも採用量が増加した。位置誤差は人工18.30%改善・自然2.07%改善・遮蔽なし23.37%悪化。bestでの正の改善という今回の受入条件を満たしたためfresh k3から全train本学習へ移行したが、安定改善やbaseline採用を意味しない。全推移・図・解釈は `outputs/diagnostics/messenger_policy_small_scale_report_20261018/{report.md,metrics.json,learning.png}`。本学習出力は `outputs/landmark_detection/messenger_natural_policy_v2_20261018`、W&B run `bk9bt8t9`。


## 2026-10-18追記：段階的学習への改訂（新規実装の優先契約）

### 適用範囲・改訂理由

ユーザーの指示により、新規実験は **S1 生成G/D → S2 遮蔽判定O → S3a Messenger復元 → S3b 採用係数g → S3c head** の順に学習する。新しい段階実験の実装・学習・評価・キュー登録は **not run**。本節は実装仕様であり、既存policy-v2の実行を停止・再開・変更する指示ではない。旧runのgate、checkpoint、W&B、source snapshot、単発例外、履歴はそのまま保存する。旧結果を新段階の合格証拠に転用しない。

勾配診断では、人工遮蔽のvisibility/座標損失が共有gateを通して遮蔽なしのgも開く経路があり、g=0でも残るhead側の性能変化が確認された。継承configのoptimal-adoption等は旧PolicyMessengerTaskで未使用だった。詳細は [2026-10-18診断](../outputs/diagnostics/policy_adoption_gradient_20261018/report.md)。これは順序を分ける理由であり、段階学習の優位性が実証済みという意味ではない。

推論topologyはHEAD-free ordered **full-image FACE/ellipse → predicted Face crop → DINO特徴・O → Messenger復元・融合 → Query Refinement**。G/Dと独立自然さ評価器は学習・監査専用。新段階の最初の比較は旧canonical-v2/k3親の同条件再設計として扱い、現行V3 complete local baselineやglobal referenceへの昇格と混同しない。親k3 best EMA SHA-256は `ef7dc88c1395e1871d432114003edab526f5ea61061a82d7bcad195934d81fa7`。データ、detector、FACE閾値、予測crop manifest、親config/bank/source hashを新manifestで認証する。V3への移植は別実験とする。

### 用語と共通の固定条件

- `F_o`: O用の固定observer。`C`: k-means初期値の学習可能な中心。`m`: Oのhard異常mask。Oと最終headのvisibility出力は別物。
- `F`: student DINO特徴抽出器。`X=F(I)`、`R=M(X)`。`A`: local_gate、structureのgに寄与する部分、structure_gate。`g=A(X,R)` は遮蔽確率ではなく採用係数。
- `c=sigmoid(a_channel)`、`Y=X+m*g*c*(R-X)`、`H(Y)`が最終keypoint出力。Cは遮蔽判定の参照であり、復元正解・codebookとして転用しない。
- S1〜S3cの標準系列ではFACE detector、F_o、Fを全期間固定する。旧policy-v2の「epoch6からbackbone解凍」は継承しない。これによりO・復元教師・head入力の基準空間を安定させる。将来のbackbone解凍や全体joint微調整は別armであり、自動追加しない。
- 素材、recipient、自然real、合成probeは固定train frameのみ。外側のsource-video train/val/test splitは不変。validation/testへ人工遮蔽を追加しない。自然valは段階選択・移行判定用、testは全選択凍結後の評価のみ。
- train内に固定の生成fit群・自然さ評価器fit群・audit群を動画単位で分け、同じrecipient/donor/real動画が評価器fitとauditを跨がないmanifestを作る。audit群をG/D/O/M/A/Hの更新に使わない。同一動画の別frameだけのholdoutで独立評価と呼ばない。分割変更・全trainへの再fitは新契約・再監査が必要。
- binary paste、mask外不変、alpha禁止、異動画donor、recipient資格（輪郭非遮蔽かつ元5点visible）を維持する。この資格は全pixelに自然遮蔽がない証明ではない。追加前教師を自然遮蔽の真の非遮蔽正解と呼ばない。
- 実支持率区間は `[5%,20%) / [20%,40%) / [60%,80%) / [80%,100%]`、成功合成の各区間25%。40〜60%を追加しない。family顔/非顔50/50、適用確率は資格recipientの0.5を初期案として別項目で保持する。要求・成功・skipを区間×familyで保存する。
- 今回の行動空間は既存policyの1 donor・候補pool内の素材/scale/translation選択。複数donor合成・任意形状・回転・RGB補正を新たに追加しない。表現できない自然遮蔽には限界を記載し、全種類を再現できたと扱わない。
- 「固定」はoptimizerから除外、requires_grad=False、eval mode、buffer/EMA不変を含む。必要な入力微分は残す。特にS3bのHをno_gradで囲むとgへの座標勾配が消えるので禁止。段階境界で残留gradをclearし、optimizer/schedulerを新規構築する。

### 段階別の更新対象と成果物

| 段階 | optimizer所有対象 | 固定する対象 | 次段階へ渡すもの |
|---|---|---|---|
| S1 | 合成方策G、全顔crop Dを別optimizerで交互更新 | C/F_o/F/M/A/c/H、FACE detector | 自然さ監査を通過したG、候補生成契約、素材bank、監査manifest |
| S2 | `LearnablePrototypeBank.centers` の1テンソルのみ | G/D/F_o/F/M/A/c/H、routing/scale/閾値 | 学習済みCと元bank/observer identity |
| S3a | `messenger_token`, `layers`, `decoder`（Rとgate入力previousを作る部分） | G/D/C/F_o/F/A/c/H | 復元Mのcheckpoint、固定Fによる教師契約 |
| S3b | local_gate、structureのcontext/prototypes、structure_gate | G/D/C/F_o/F/M/c/H | 採用Aのcheckpointとg診断 |
| S3c | Query Refinement head Hのみ | G/D/C/F_o/F/M/A/c | 最終bundleとvalidation選択結果 |

`structure.descriptor` のみのheadは損失に使用しないため固定する。名称にprototypeがあっても、A内のstructure prototypesとOの中心Cは別パラメータである。`channel_adoption` は識別不能なg×cの自由度を増やさないため全段階で親値固定。reference adapter等が有効な親を使う場合は、この表へ所有権を追加し認証するまで拒否する。モジュール名のprefixだけで無監督パラメータまでoptimizerへ入れない。

### S1：人工遮蔽の生成を先に学習する

**主目的と損失**：自然さを主目的に、既存の位置誤差による難化も固定T_refに対して計測する。T_refは親のFACE/crop/F/O/M/A/Hを全て固定し、後続S2/S3のcheckpointへ追従させない。自然さのみの対照は `lambda_hard=0`、標準候補は従来の `lambda_hard=0.1`。Dと自然さの重みは1.0。両armを同一候補pool・予算で比較し、標準候補も自然さ基準を満たさなければ不合格とする。難化だけの改善で先へ進めない。

- `L_D = mean(softplus(-D(real))) + mean(softplus(D(stopgrad(fake))))`。
- `C_nat = softplus(-D(fake))`（non-saturating GAN）。`E_pos` は追加pasteで覆ったcrop内位置既知点のxy SmoothL1。presence・visibility・T総損失を難化報酬へ含めない。
- 離散Gの報酬 `r = 0.1*E_pos - C_nat`、最小化損失 `L_G = -mean(stopgrad(r-b)*log pi_G(action|context))`。bは前更新までの報酬EMA。自然さのみarmでは位置項を0にする。
- D更新ではfake/素材/recipientをdetach。G更新ではD/T_refの評価値をdetachし、log pi経由でGだけを更新する。合成mask/座標へのstraight-through勾配を作らない。T_refへ負の位置損失を逆伝播しない。自然real不成立例ではG/Dを更新せず、momentum/schedulerも進めない。
- 学習中のD損失、識別精度、報酬、候補選択entropy、family/素材/位置/scale/区間の分布を記録する。候補poolを固定した比較とepoch更新poolの実運用監査を分ける。

**独立自然さ評価**：Gの候補と同じpoolからの一様選択G_randomを対照に、独立評価器をtrain内評価fitで学習し、動画非重複auditで評価する。Gの更新にこの評価器の値を返さない。顔向き・size・画質・既知point遮蔽数を揃え、同一枚数・同一評価予算を使う。自然realには密maskがないため、人工面積binと自然遮蔽面積binが一致したと偽らない。全4区間の自然比較が必要ならtrain-audit自然画像へ面積maskを追加監査する。該当区間の自然realがなければその自然さはN/Aで、全区間合格を出さない。

評価器自身の機能確認には、同じ素材で明らかな不整合を作った固定負対照を用いる。独立識別が難しくなったことは自然分布との完全一致の証明ではない。固定seedのblind montageで境界・配置・対象顔の整合を確認し、判定rubricとレビュー結果を保存する。

### S2：遮蔽判定Oを中心だけで学習する

Gのcheckpointと候補生成規則を固定する。Gの重み固定は合成画像を毎回同じにする意味ではなく、seedと分布を固定して学習用サンプルを再生成できる。自然画像も混ぜ、自然遮蔽を人工遮蔽の教師だけで置き換えない。

`z_O = (threshold_fixed - similarity(F_o(I), C_selected)) / 0.1`。

`L_O = BCE_nat(z_O, state1/2) + BCE_syn(z_O, updated_state1/2) + 0.001*anchor(C,C_initial)`。

自然・人工のBCEは集合ごとに重み正規化してから加算し、単純に点数の多い人工群へ支配させない。正例重みは各train-fit集合からのみ算出・固定する。空集合はその項をskipし、両方教師なしならanchorだけで更新しない。

自然state1は1、state2は0、state0は除外。人工では元present点のうちpasteが覆った点を1に変更し、元のmissingを教師化しない。未注釈patchやpaste外の全pixelを負例にしない。密な人工paste maskは追加遮蔽の領域診断に使うが、初版の中心学習は既存の点教師に限定する。crop外/support外は学習から除外し、自然valではsupport外FNを含む全点指標とsupport内指標を併記する。

勾配は連続similarity→Cのみ。F_o/画像/geometry/routing/scale/thresholdはdetach・固定、hard mには擬似勾配を使わない。中心は元のk-meansから開始し、旧O smokeの重みを本学習初期値にしない。自然valの点位置binary occlusion F1で選択し、AP/AUROC、state2誤検知率、state1 recall、5slot/輪郭遮蔽/横顔別の分母も保存する。

### S3a：採用・headを動かさず復元候補を学習する

GとOを固定し、**Fの同一snapshot**から `X=stopgrad(F(I_syn))`, `Z=stopgrad(F(I_before_paste))` を得る。F_oはO専用であり、別世代observerを復元正解へ流用しない。crop幾何は同一。Zは追加paste前の特徴であり、元からある自然遮蔽の裏側の正解ではない。

`L_M = L_pair_covered(R,Z) + L_pair_outside(R,Z) + L_visible_preserve(R,X)` を初期案とし、各係数1.0。前2項は人工対のみ、binary pasteを固定面積集約したtoken coverageで内外を重み正規化するSmoothL1。coverageは画像上の追加pasteとの対応であり、DINOの広い受容野に対する正確な汚染maskとは呼ばない。最後の項は無加工画像のstate2点位置だけで元特徴を保存する。自然state1に `R=F_o(I)` や `R=X` を復元教師として課さない。

学習勾配はR→Mのみ。A/c/Hは固定で、keypoint/visibility/自然さ/難化/adoption lossはこの段階ではMへ流さない。g≈0でもRの直接再構成損失で学習できる。評価時は実際のm/gによる通常経路と、既知人工maskを使う候補の能力診断を明示して分ける。

S3aは復元能力の中間選別であり、gが閉じた通常F1だけで候補を順位付けすると復元改善が見えない。そのため**人工train-auditのcovered feature errorをcheckpoint選択に使う明示的な例外**とし、通常経路の自然val F1を回帰guardにする。損失だけの合格は最終モデルのF1合格ではない。

### S3b：復元候補を固定して採用係数gを学習する

F/M/O/c/Hを固定する。X/R/previousはdetachし、Aの入力にする。structure contextはAの学習対象なので、その演算全体をno_gradにしない。融合Yから固定Hを通る入力微分を残す。

人工対の追加遮蔽tokenだけに、次の採用教師を明示接続する。

`v = m*c*(R-X)`、`q = Z-X`。

`g_star = stopgrad(clamp(sum_channel(v*q) / (sum_channel(v*v)+epsilon), 0, 1))`。

`epsilon=1e-8`（FP32）を初期案とする。m=0またはsum(v*v)<=epsilonのtokenには有用な採用教師がないので除外し、除外数を報告する。必要ならscale正規化はv/q双方へ同一固定scaleを適用し、式と設定をhashに含める。cを固定するためgとchannelの積の曖昧さを増やさない。この教師は追加遮蔽への復元候補の有用性であり、自然遮蔽の正解gateではない。

`L_A = 0.01*L_adopt_syn + 0.1*L_fused_pair + 0.1*L_position + 0.1*L_visible_identity` を初期案とする。

- `L_adopt_syn` は上記対象tokenでのMSE(g,g_star)。`L_fused_pair` は人工対の追加遮蔽tokenでSmoothL1(Y,Z)。
- `L_position` は人工・自然・無加工それぞれのstate>0/crop内位置既知点での座標SmoothL1を集合ごと平均し、存在する集合を等重み平均する。presence予測で重み・分母を変えない。自然遮蔽はこの実座標教師で扱い、隠れた特徴の正解を仮定しない。
- `L_visible_identity` は無加工state2位置のSmoothL1(Y,X)。自然state1をゼロgate教師にせず、無加工画像全体を非遮蔽と決めつけない。
- **presence/visibility分類損失のAへの係数はこの段階では0**。前診断で得た分類勾配との競合を分離する設計で、座標勾配によるg上昇までなくなるという主張ではない。descriptor/consistency/gram lossも初版では0を明示し、継承configの未使用非ゼロ値はschema errorにする。

`dL/dg = m*sum_channel[dL/dY*c*(R-X)]` と、独立したadoption項を区別して検証する。g_star・教師Z・X/R・mから上流へ勾配を返さない。O陰性ではtaskの直接g勾配が0でも、共有Aの更新でraw gは変わり得る。raw g低下を合格条件にせず、m*g*c、実際の位置性能、g=0介入で採用の効果を判定する。

### S3c：復元・採用を固定したままheadを適応させる

G/O/F/M/A/cを固定し、YをdetachしてHだけを学習する。自然・人工・無加工を混ぜ、既存LandmarkLossの座標Wing2.0、presence0.1、visibility0.1、auxiliary0.5、refinement auxiliary0.75と、人工で覆った点の位置SmoothL1係数+1を使う。親で存在しないheatmap等を追加しない。全点座標項とcovered項の二重の重み付けを明記する。

全損失→Hのみ。候補/融合特徴再構成、adoption、G/D/Oの損失はこの段階では0として計算・optimizerから外す。分類損失はHを改善できるがAを押し開かない。Hの変更が人工だけに適応し自然・遮蔽なしを悪化させた場合はS3bを保持し、S3cは不合格とする。自動的に全体jointへ切り替えない。

### 次段階への判定条件とcheckpoint選択

以下は**新段階実験専用の計画上の初期基準**であり、旧runの受入値や既存一回限りの例外を書き換えない。数値・損失係数・予算・比較manifestは実装時の独立configへ固定し、schema gateで必須にする。未設定/N/A/必要件数不足はpending、明示的な基準違反はfailed。どちらも自動進行不可。結果を見てから同じrunの条件を緩めない。

共通記号：`F_all` は自然valのconditional 5slot Macro F1 NME0.10、presence0.70、GT Head side1.20。`R_nat` は同一eligible自然state1点の正しい位置出力recall。`F_clean` は輪郭非遮蔽かつ元5点state2の顔の同定義Macro F1。`FP_clean` と位置誤差も記録する。自然state1は欠損state0と分ける。比較は同一crop・分母・閾値・精度・seed・予算で行う。

- S3の機能改善幅の初期案は `delta_task_min=1e-4`、F_clean許容低下は `0`。これは新段階の基準案で、20261018一回限りの「任意の正差」overrideも、patience用min_deltaも流用しない。
- S1の独立識別基準案は `AUC_sep=max(AUC,1-AUC)` の動画bootstrap95%上限<=0.60、負対照の95%下限>=0.80。G_randomよりAUC_sepが低いことのpaired動画bootstrap95%区間上限<0も必要とする。0.5近傍を完全同一と解釈しない。
- S1/S2/S3の支持率は自然val/auditの固定manifestで数える。bootstrapは動画単位・seed42・2,000反復を計画値とし、独立動画が少なく区間を解釈できない場合はその不確実性を明示し、必要な評価群を事前に確定する。複数候補の反復audit最適化を避け、候補数・評価頻度を予算へ含める。

| 遷移 | 必須条件（全てAND） | bestの選択・不合格時 |
|---|---|---|
| S1 → S2 | binary/教師/漏洩の全不変条件、aggregate skip<=10%・区間×family skip<=20%・各層>=10要求・各層real対応>=50%。成功4区間均等、family50/50（有限端数差<=1）。独立識別の上記基準を全体・評価可能な4区間で満たす。負対照で評価器の能力確認。blind reviewで重大な合成違反0。選択素材有効種類数exp(entropy)と位置/scale占有bin数が同pool G_randomの80%以上。 | 全guard適合候補のAUC_sep最小、同値は先のcheckpoint。自然さと多様性の中間選択で、DのF1最大・固定T_refのF1最大では選ばない。必要な自然面積監査が欠ける区間はpending。 |
| S2 → S3a | 自然val occlusion F1が固定k-means Oより>=0.001改善、自然state1 recall>=対照、state2誤検知率<=対照。人工train-auditでも点F1>=固定O。C以外不変、全点/support内の両分母とsupport外FNを報告。 | 自然val binary occlusion F1最大、同値は先。閾値は親値固定。Oの改善を最終keypoint改善と読み替えない。 |
| S3a → S3b | 人工auditのcovered feature errorが段階開始時Rと未復元Xの双方より厳密に低い。4区間に有効教師があり各区間でXより悪化なし。outside errorと無加工state2の特徴変化が開始時以下。通常経路F_allとF_cleanが開始時から低下なし。 | guard内でcovered feature error最小、同値は先。自然の真の復元精度はN/A。最終F1改善が未確定の中間pass。 |
| S3b → S3c | F_allがS3b開始時より>=delta_task_min改善、R_natが開始時より厳密に改善、F_clean>=開始時。さらに同一checkpoint g=0介入よりF_allとR_natが厳密に高く、F_cleanは低下なし。固定Hの重み不変を確認。 | guard内の自然val F_all最大、同値は先。g平均が増えた・teacher MSEだけが下がった状態では進めない。 |
| S3c → 最終評価 | F_allがS3b bestより>=delta_task_min改善、R_natとF_cleanがS3b bestから低下なし。復元なしのmatched head学習対照に対してF_all/R_natが厳密に高く、F_cleanは低下なし。 | guard内の自然val F_all最大。満たさなければS3c不合格、S3bを保存するが全計画完了・baseline採用とは扱わない。testを見て救済しない。 |

S1/S3aは目的の異なる中間学習であり、固定された下流F1が学習対象を順位付けできないため選択指標を上表のように明示的に分ける。**Oと最終keypointを選べる段階はF1選択を維持する。** 中間指標による選択がユーザーの「基本的にF1選択」に対する例外であることを実行configと報告にも表示する。

各段階とも「best selected」は「transition passed」と別フィールドにする。常に無条件F1最大checkpointをquality合格とみなさず、guard適合候補の中から選ぶ。適合候補0ならbest_eligible=null。同点は先のepoch、表示丸め前の値で判定。自然valの全体・輪郭遮蔽×state1点数12条件・横顔・出力数・FP/FNを報告し、平均で隠れる回帰を点検する。定義未確定の誤帰属等はN/Aと理由を残す。

### 実装する段階管理・勾配検証・実行ゲート

新規のstage configとrunnerを独立作成する計画とし、旧policy-v2 trainerへstage分岐を後付けして実行中runを変えない。予定の状態遷移は `S1→S2→S3a→S3b→S3c→validation確定→test`。各stage checkpointに上流の選択済みhashを含め、違うG/O/F/M/Aを組み合わせたresumeを拒否する。

- configに `stage`, `trainable_names`, `frozen_names`, `active_losses`, `loss_weights`, `gradient_targets`, `teacher_identity`, `selection_metric`, `guards`, `update_budget`, `upstream_hashes` を必須化する。lossの未使用非ゼロ値やoptimizer所有権の重複/漏れを拒否する。
- S1はG/Dの選択符号と分離、S2はCのみ更新、S3aはMのみ更新、S3bはAのみ更新かつ固定H経由の入力微分、S3cはHのみ更新をactual autogradで検証する。active moduleの有限・非ゼロ勾配だけでなく、禁止moduleのgradなし・全固定tensor前後hash一致を要求する。
- S3bではoracleの端点（元特徴が正解なら0、候補が正解なら許容範囲の最適値）、m=0、R=X、coverage0、空教師、channel係数を含む融合微分式を検証する。自然state1へ人工oracleを適用していないこともテストする。
- 固定Hでも座標勾配がAへ届くこと、分類lossのA勾配0、g_star/teacherからM/Fへ勾配0、S3c分類lossからAへ勾配0を別々に確認する。共有gateによりO陰性raw gが変化し得ることを誤ってバグ判定しない。
- G/Dも含めRNG、候補生成seed、sampler、reward baseline、optimizer/scheduler、選択状態、module/bufferを完全resumeする。段階間は上流bestを新段階の固定入力にし、上流optimizerは引き継がない。同段階resumeだけoptimizerを戻す。
- 新stage/比較armごとにfresh schema/gpu_step/overfit/small_scaleを通す。四gateは実装/数値/小規模学習の確認で、上表の段階完了品質gateを代替しない。逆も同じ。2更新smoke、旧policy-v2の成功marker、単発例外は流用不可。
- GPU preflightはisolated4/8/16、反復optimizer更新と当該stageの実validation経路、同期計時、VRAM<=90%、温度<83C、OOM/非有限なし。予測所要時間で選び3%以内なら小batch、effective64を維持しruntime OOM fallbackを有効にする。
- 初期予算案は各stage max40epoch、warmup5の直後にdecay。S1/S2/S3aのsmall_scaleは固定20epochかつ最低80回の有効optimizer更新、S3b/S3cも20epoch・最低80回で途中打切りなし。有効教師なしskipを更新数へ数えない。監査対象数/独立動画不足なら予算だけ満たしても合格にしない。
- S2 Adam lr1e-3/weight_decay0、S1 G/D Adam lr1e-4、S3a/A/H AdamW lr1e-4/weight_decay1e-4を初期案とする。seed42、FP32、EMA off。S3b/S3cは毎epoch自然val、minimum_epoch10/patience5/min_delta1e-4を適用。S1/S3aは事前固定予算を完走し独立監査選択、S2も自然F1を毎epoch測り最小epoch/patienceを明示する。
- checkpointはstage別last毎epoch・選択best、recoveryは5epoch以上間隔で最新1個、正常完了後last/bestのみ。成功gateはbest1個。上流の固定bestを参照し、各段階に同じ巨大bundleの無制限コピーを作らない。独立評価器の保存数・容量もconfigに含める。

canonical queue実装時は段階ごとの独立四gateをproductionに依存付ける。下流gateには上流の選択済み成果物が必要なので、全段階gateを未学習の上流で先に「合格」にしない。S1 group終了・認証後にS2 groupを準備する等のbounded wave方式で、既存の「enabled gate jobsを未完了productionより前に置く」規則と両立させる。独立runnerと真正な依存関係が完成するまで架空の実行command/成功marker/有効jobを登録しない。今回の文書更新ではqueueを編集しない。

### 比較・報告と未実装項目

必要な比較は、同一予算のG_random対学習G、固定O対学習O、復元あり対なし、S3bのg=0/初期gate介入、S3cのhead-only matched control。全てを一度に増やさず当該段階の能力を判定する。旧同時学習結果は履歴対照であり、train内audit除外・損失・固定範囲が変わる新系列と直接の因果差にはしない。段階化そのものの効果を主張するには、別途同データ・総更新予算の同時学習対照が必要。

人工/自然/遮蔽なしのg、m陽性率、m*g*c、位置誤差、presence/visibility、TP/FP/FN、候補/融合の特徴誤差、全lossのmodule別勾配normを初期・毎epochで集計する。人工はtrain内固定probeで比較し、Gの候補選択変化とは分離する。S3bで採用teacherの有効token数・m陰性による除外数も残す。epoch中はscalar集計のみ、詳細画像・全表は段階完了時に作る。

S1の独立自然さ評価器・面積別natural監査・多様性評価、段階所有権の実装、S3a専用教師、S3bの明示adoptionと勾配制限、guard付き選択、段階間manifest/runner、fresh gate/production統合は未実装。既存部品（中心のみO、policy G/full-crop D、Messenger、F1評価、g集計）は再利用候補だが、存在だけで段階対応済みと扱わない。段階学習の各passはその能力だけの確認であり、自然遮蔽の真の非遮蔽特徴を得た証明やbaseline昇格ではない。


## 2026-10-18追記：段階実装とS1単独検証の結果

新しい独立実装は `src/staged_occlusion/`、設定は `configs/training_plans/staged_occlusion_v1.json`、bounded S1入口は `scripts/run_staged_occlusion.py`。旧policy-v2のtrainer/config、完了checkpoint、canonical queueを変更していない。以下は実装・実行済み範囲を記録するもので、前節の品質基準を変更しない。

- 実装済み：5段階のtrainable所有権、S1専用G/D optimizerと固定T報酬、S2中心のみBCE、S3a復元損失、S3b明示adoptionと固定headを通る座標勾配、S3c head-only損失。`GuardedStageUpdate` は真正な上流transitionなしに構築できない。
- S1は自然さのみ／自然さ＋位置難化0.1の2条件。family×4区間を外側で均等指定し、各候補pool内でのみGが素材/scale/translationを選ぶ。成功数を8層同数にし、余剰もskipとして数える。
- train44動画をgenerator-fit28/evaluator-fit8/audit8に固定分割。recipient/donor/自然realは各群内に限定する。独立評価器はDと異なる全顔crop CNNで、各条件のlearned/random/明瞭な不整合負対照を同じ予算で学習する。自然realの面積未注釈は既知制限のまま保持する。
- S1 runnerは検証のみ。S2/S3の損失と更新機構は実装したが、実データの本学習launcher、各段階の四gate・共通評価の接続は未完了。S1未合格のため実データで後続更新を行っていない。自然面積maskとblind reviewを認証して受理するadapterも未実装で、欠落はpendingとし、入力が来ても未対応のまま合格へ読み替えない。
- source/config/data/親identityをsnapshotとhashで保存。optimizer、reward baseline、epoch/step、global RNGを保存・再開する。段階のassessmentと4gateの証拠hash・上流bestがそろわなければtransition_successを受理しない。単発F1例外は未使用。

初回GPU preflightはT正規化bufferのCPU/CUDA不一致で失敗。taskをbufferごとdeviceへ移し回帰テストを追加した。初回証拠は `outputs/diagnostics/staged_occlusion_s1_20261018/` に保持。修正版の19テストはpass、isolated micro4/8/16も全てsafe（最大46C、peak physical VRAM比2.44%）。計測時間からmicro16/effective64を選択した。固定入力24更新でD損失1.386932→1.377210、同じ最終D/Tに対するGの期待コスト0.699542→0.699521でoverfit gateを通過した。

修正版S1は2条件とも20epoch・80更新を完了。generator-fitからhash選択した512cropに対し各epoch98合成要求。更新は4回/epochを満たすため有効balanced例を循環再利用し、ユニーク例数とは分離して記録した。全train productionではない。TはF/O/M/g/channel/Hを含め全tensor hashが不変。独立評価器はfit104対（208画像）、audit80対（160画像、8動画）、各20epoch、動画bootstrap2000反復。validation/test合成・test評価はnot run。

| 条件 | 独立AUC_sep（95%区間） | randomとの差（95%区間） | 判定 |
|---|---|---|---|
| 自然さのみ | 0.520625 [0.501538, 0.606492] | -0.079688 [-0.143270, +0.044555] | 不合格 |
| 自然さ＋位置難化 | 0.520625 [0.501538, 0.606492] | -0.079688 [-0.143270, +0.044555] | 不合格 |

AUC_sepはmax(AUC,1-AUC)。95%上限<=0.60、randomとの差の上限<0の両条件を満たさなかった。負対照は0.999531 [0.997714,1.0]で評価器の強い色不整合に対する能力は確認したが、矩形境界や自然な配置の判定能力まで保証しない。面積別natural対応、厳密な面積別自然さ、blind reviewは未評価。これらを0や成功に置換しない。

固定train88候補poolで、Gの一様確率との差（各pool内最大差の平均）は自然さのみ0.001503、難化あり0.001505、KLは約1.06e-5。選択方策は一様からほとんど離れておらず、上のlearned/random点推定差を獲得した自然さの効果とは認定できない。保存済み重みで再計測したG/D勾配normはそれぞれ約0.00192/0.25510、0.00199/0.25511。G→D、D→G、Tへの接続gradは0。計算経路は動作するが、有効な自然遮蔽生成を獲得したという品質条件は満たしていない。

**S1 assessmentはfailed、S2〜S3cは未起動。** S2起動ガードの実拒否も確認し、transition_successは発行していない。合格値を変更せず、品質失敗の自動再試行・後続本学習・baseline昇格は行わない。各armのlastと独立評価器のみ合計2.81MiBを保存し、不合格checkpointをbestと呼ばない。

結果・図・source/hash・勾配証拠は [S1検証report](../outputs/diagnostics/staged_occlusion_s1_20261018_r2/report.md)、`gradient_and_barrier_verification.json`、各armの `history.jsonl` / `audit/{statistics.json,scores.json,evaluation_population.json,audit_gallery.png}`。W&Bは `sato-waseda-university/face-keypoint/d5z2mzdt`。図は確認済み。次の診断対象は、自然さ評価器の構造判定能力、素材候補の矩形境界・表現範囲、Gの微小な選択差と学習予算であり、今回それぞれの寄与を個別に実証したわけではない。


## 2026-10-18 追記：S1a D事前学習 → S1b G/D交互更新

ユーザー指示により、ランダム合成をDが識別できるか先に確認する。今回の設定は
`configs/training_plans/staged_occlusion_d_first_v1.json`。旧同時更新の結果・設定は保存する。
本検証はtrain内のbounded verificationでありproductionではない。推論構成は引き続き
HEAD-free `full-image FACE/ellipse -> Face-crop DINO -> Messenger prior fusion -> Query Refinement`。
G/Dは学習専用で、T/O/g/Messenger/headは全期間固定。S2以降の全条件は緩和しない。

- 元train44動画をG/D側28、独立評価器fit8、最終audit8に固定分割する。
  G/D側28はさらに学習20、D checkpoint選択4、D合否確認4に分割する。
  recipient/自然real/donorの全てを各集合内に限定し、同じ素材の動画が集合をまたがないようにする。
- S1aはGを使用せず、各候補集合から一様に選ぶ。DのみAdamで30epoch×16更新=480更新。
  学習crop上限2048、effective64、最大lr1e-4、warmup3後cosine減衰、seed42/FP32。
  各epochの有効合成を循環利用するため、延べ提示数と独立例数を区別して記録する。
- Dは全顔cropを入力し、自然real=1、人工fake=0のlogistic損失。
  G/Tへは勾配を流さない。checkpointは選択用動画の二値クラスMacro F1最大、
  probability0.5固定、同値は早いepochで選ぶ。選択したDを別の合否確認4動画で一度だけ評価する。
- S1aの事前固定した操作的判定は、480更新完了、凍結確認、3動画以上かつ32組以上、
  正方向AUCのrecipient動画bootstrap95%下限>0.5、固定閾値Macro F1下限>0.5、
  顔/非顔それぞれの正方向AUC>0.5。AUCを反転して合格させない。
  これはランダム合成を識別する前提能力の判定で、自然さの意味を学んだ証明ではない。
  real/donor再利用の依存がrecipient単位bootstrapで完全に除かれない点を明記する。
- S1a不合格ならG学習を開始しない。assessment・checkpoint・reportのhashを認証したsuccessだけで進める。
  成功時はbestのみ、失敗時はbest/lastを保持する。resumeは同一source/config/親checkpointのみ。
- S1bはS1aのbest DとAdam状態を引き継ぐ。自然さのみ／自然さ＋固定T位置誤差0.1を別々に
  seed42のGから開始し、各20epoch×16更新=320更新。各更新でDを先に1回更新し、
  更新後DのNon-saturating costをdetachしてREINFORCEでGを1回更新する。D/TへのG勾配はゼロ。
  Gの入力・候補・不透明合成方法は維持し、この変更で顔RGB入力追加や合成器変更を混在させない。
- preflight4/8/16はD-onlyと交互G/D更新、および検証経路を使う独立の使い捨て試験。
  G/D overfitはS1a合格後に合格Dを初期値として行う独立の機能試験で、その更新後重みをS1bへ流用しない。
  OOM時はoptimizerを含めた更新前状態に戻してmicrobatchを縮小する。
- S1b後の独立評価とS1→S2ガードは保持する。自然面積別の評価・blind reviewの未実装部分を
  D事前学習の成功で代用しない。旧80更新との比較には、順序に加え今回の更新数・データ予算が
  増えていることを明記し、順序だけの因果効果とは解釈しない。


### D-first実行結果（2026-10-18）

`outputs/diagnostics/staged_occlusion_d_first_20261018_r2`でS1aの480更新を完了。
26 tests、fresh GPU preflight4/8/16通過、micro16を選択。
F1選択epoch17（272更新）のDを別4動画72組で評価したところ、
正方向AUC=0.467785（95%区間0.338867–0.571875）、
二値Macro F1=0.509222（0.408110–0.587363）で不合格。
自然recall=0.333333、人工recall=0.722222。顔素材AUC=0.385031、非顔素材AUC=0.537037。
予算・凍結・件数条件は通過。失敗は正方向AUC下限、固定閾値F1下限、顔/非顔双方の識別条件。
G/D overfit、S1b学習、最終独立自然さ評価、S2以降は未実行。G/T/O/g/headは更新なし。
Dのbest/lastのみ保存。学習損失は1.386869から1.313298へ下がったが、別動画での識別改善は確立しなかった。
Dを先行学習する実装は動作したが、その前提能力は現データ・モデル・予算で未達。

先行出力`staged_occlusion_d_first_20261018`は、G overfitが未学習Dを使う旧順序を残していたため、
Dの1epoch時点で中断して証跡を保持した。修正版ではG overfitもD合格後にのみ実行する。
旧試験の成功やcheckpointはr2へ流用していない。基準の緩和や品質失敗の自動再試行はしていない。
W&B: https://wandb.ai/sato-waseda-university/face-keypoint/runs/x4xn9ewn 。報告: 同出力の`report.md`、`d_pretraining_learning.png`、`barrier_verification.json`。
