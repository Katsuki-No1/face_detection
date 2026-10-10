# Messenger連続特徴復元：設計・実装記録

## 2026-10-10: Messenger評価の重点（ユーザー確認）

Messenger token modelでは、次の二点を重点として確認する。

1. **遮蔽時の特徴融合品質**：遮蔽された入力に対し、融合後の最終特徴が改善しているか。
   復元候補だけの改善や全体keypoint F1だけでは、この確認の代用にならない。
2. **自然遮蔽の検知**：人工遮蔽を追加していないvalidationで、元から存在する遮蔽を
   十分に検知できているか。人工遮蔽への反応と自然遮蔽への汎化を分けて確認する。

既存の診断設計との対応：clean参照を確保できる人工遮蔽の対では、融合前X・復元候補R・
融合後Yを比較し、最終Yの改善と非遮蔽領域の維持を調べる。この合成診断を無加工valの
結果と混同しない。自然遮蔽には遮蔽前の正解特徴がないため、直接の特徴復元正解を仮定しない。
無加工valでは自然遮蔽GTに対する検知と融合の下流寄与を分けて確認する。

検知対象の出力（内部α/g等か、最終keypoint stateか）とその意味の対応、指標・分母・
判定閾値・「十分」とする定量基準は、この確認だけでは新たに確定しない。
内部の融合係数を検証なしに遮蔽確率と呼ばず、最終state精度だけから内部検知の成功も断定しない。
顔輪郭の遮蔽と内部keypoint遮蔽は独立に扱い、missing=0を遮蔽=1に読み替えない。
既存の共通評価およびvalidation-only選択契約は維持する。

対象の推論構成はHEAD-free ordered cascade:
full-image FACE/ellipse → Face-crop DINO → Messenger recovery → Query Refinement。
今回の変更は評価方針の記録のみ。実装・推論・評価・学習はnot run。
前回報告した全体F1だけでは上記二点は未確認であり、Messengerの有効性の結論は出していない。

## 2026-10-09: 顔・非顔50/50の自然遮蔽近似をMessenger loaderへ接続

以下の「設計のみ」の非顔案を、今回の追加依頼で実装へ進めた。従来は顔pasteの独立prototypeが
存在したが、非顔素材・50/50混合・Messengerの実学習loaderへの接続は未実装だった。
今回の状態は **loader実装済み、training not run、学習plan/queue変更なし**。

### 推論構成と適用条件

- 有効側の構成はHEAD-free ordered cascade:
  **full-image FACE/ellipse → Face-crop DINO → continuous Messenger recovery → Query Refinement**。
  無効側は同じ構成からrecoveryのみを外す。外部HEAD検出器を追加しない。
- `continuous_recovery_real_mixed.yaml` / `continuous_recovery_real_mixed_control.yaml`は
  同じ初期checkpoint・train seed42・data/augmentation/train条件を継承した比較候補。
  個別outputを使い、旧実験・既存スケジュールを置き換えない。schemaは明示されたon/off armと
  50/50・clean recipient・48/2条件を照合し、専用の実データ分布監査を行う。現在の固定v2 splitを使い、
  別作業のv3 baseline再学習とは独立。新しい四ゲート・batch preflightは未実行。
- 適用元は元注釈の`face_occluded == false`かつ5点すべてvisible=2、canonicalのvalidな
  Face対応があるものだけ。該当は2,550 records。crop前の遮蔽/missingを持つ顔は除外する。
  cropにより外れた点はmissing=0とし、structure補助教師でも除外する。
- 適格な顔がある画像ごとに既存の適用確率0.30、一画像最大一顔を維持。
  **適用する人工遮蔽の内訳**を顔50%／非顔50%で選ぶ。全画像の半分ずつを加工する意味ではない。
  family・遮蔽強度枠を先に選び、失敗しても別familyや軽い枠へ置き換えない。

### 素材・配置・教師

`scripts/prepare_real_occlusion_bank.py`でtrain 4,237画像／44動画だけを索引化した。
`outputs/dataset_adapters/real_occlusion_mixed_v1/manifest.json`のSHA-256は
`587da1883206607ca2b883676195059aaa2ff227b11d038265ee260255ea6718`。
manifest、canonical/adapter注釈、素材画像のhashを検証し、別動画から採る。
顔素材は5,032候補のcontext色調・輝度記述子を用いて候補を優先する。
非顔素材は動画→画像→位置・サイズ・縦横比をランダムに選ぶ。全注釈FACE楕円の外接矩形に
長軸直径の5%余白を加えて除外し、少しでも入る素材は棄却する。遮蔽あり・missing顔も除外対象。
背景／身体などの意味カテゴリによる制限は設けない。

非顔切出しの代表辺は画像短辺の8–40%、aspectは0.4–2.5の対数一様。
貼付時の長辺は対象顔長軸直径の0.4–1.4倍、aspectを保持する。
顔・非顔とも実不透明maskと対象FACE楕円から配置後に遮蔽率を計算する。
軽度(0–10%、実keypoint被覆必須)／中度(10–20%)／強度(20–40%)を20/40/40%で要求する。
顔素材のみ、素材FACE自体も対象FACEを最低1%覆うことを要求する。
素材一つにつき配置256試行、最大8素材で成立しなければ無加工で返し、理由を記録する。
以前の「10–40%に置けず学習全体が終了する」処理へは戻さない。

合成は採用済みの**幅48px・強度2、crop側のみの不透明RGB境界補正**。
元画像のmask外を変更せず、alpha透明化なし。小さいpatchでは補正帯が全体に及ぶ。
clean teacher／occluded studentは同じcrop幾何・photometric・flipを共有する。
人工maskが点を覆う場合だけstateを2→1、missing=0は維持する。
素材側の顔を対象顔のIDやkeypoint教師に置き換えない。FOR/full-image detectorへの接続は
別作業であり、このcrop loader変更だけでその学習接続が完了したとは扱わない。

### 特徴再現の評価方法

`scripts/evaluate_real_mixed_recovery.py`は将来の学習済みon/off checkpointに対する独立診断。
固定valの元から遮蔽なしの351顔へ各familyを一回ずつ要求し、素材はtrain限定。
同じ合成入力をon/off両方へ渡したことをhashで照合する。通常のvalを改変して学習選択に使わない。
GT cropの合成診断であり、deployed full-image FACEからの主評価とは分けて報告する。

- clean参照は固定初期teacherと、各student自身のclean backboneの両方を記録する。
- face／non_face別、遮蔽token／非遮蔽token別にraw・candidate・**final**のcosineと
  channel scaleで正規化したMSE、およびfinalがrawより改善したtoken率を測る。
- candidateのみの改善を採用根拠にしない。finalの改善、非遮蔽領域の維持、同一条件on/offの
  validation semantic keypoint F1と自然遮蔽sliceの結果を合わせて判断する。
- testでaugmentation・閾値・checkpointを選ばない。自然遮蔽のclean正解特徴は存在しないため、
  合成対で特徴改善を測り、実自然遮蔽ではkeypoint/stateの改善を確認する。

実データ監査は`outputs/diagnostics/real_mixed_occlusion_20261009/`へ保存。
seed42/43・epoch1で1,223要求、1,222適用、skip1（素材/配置の試行上限）。
実適用は顔642（52.5%）／非顔580（47.5%）、軽度243（19.9%）／中度499（40.8%）／
強度480（39.3%）で、事前の分布チェックを満たした。
初回`v1`は1,223組すべての実RGB生成・assertion完了後、集計値numpy.int64のJSON保存で失敗。
型変換と逐次sample保存を修正し、同じseedの`v2_geometry_replay`で配置分布を再取得した。
replayは先頭8例のみgradient描画、残りは配置統計のみ（gradientは乱数や配置判定に影響しない）。
さらに現コードで顔/非顔×軽/中/強の6例を完全描画し、保存したmask hashとの一致とmask外不変を確認。
`balanced_full_rgb_examples.png`はこの6例であり、頻度を表す標本ではない。
production schemaはreplayを受け付けず、epoch1/12/40のfull RGB監査を別途要求する。
意味カテゴリを制限しないため、文字や透かしを含む背景patchが入る例も目視で確認した。
この監査は分布・教師整合の確認であり、モデルの有効性評価やproduction gateではない。
関連8 test modulesの拡張確認は86 passed / 1 failed。失敗は既存
`test_conditional_object_training_schedule.py::test_bounded_plan_has_only_completed_mal_and_new_messenger`が
別作業で追加されたv3優先queueを考慮しない順序期待値であり、本変更ではqueue/当該testを変更していない。
新規augmentation・strict適用条件・on/off設定・feature診断・既存recoveryのテストは通過した。
val pair作成のみCPUで2例確認（顔/非顔各1）、mask外のnormalized tensorが完全一致することを確認。
モデルを用いた702 val pairsの評価は未実施。
注釈されていない顔の非顔素材への混入は完全には防げず、矩形patchは物体の形状・奥行き・照明を
完全再現しない。「自然遮蔽の近似」であるという制限を維持する。

## 2026-10-09: 顔以外のランダム領域による遮蔽augmentation — 設計のみ

ユーザー指定: 顔cropによる遮蔽と別に、素材側の顔領域を避け、背景・身体の部位などから
対象とサイズをランダムに選んで対象顔の周辺へ重ねる。意味カテゴリを細かく限定しない。
**今回は設計のみ。実装・画像生成・学習・評価はnot run。学習plan/queueへの追加なし。**
将来のfamily名案は`non_face_region_paste`。既存の実顔pasteとは独立に有効/無効を設定する。
以下の「引継ぎ」は設計上の方針であり、実行中のaugmentation設定を変更するものではない。

### 素材選択: 顔を含まない領域のランダムcrop

1. 学習実験が指定する固定trainの中から、受け手とは別のsource videoを選ぶ。
   動画→画像→領域の順にランダムに選び、長い動画のフレーム数だけで素材が偏らないようにする。
   val/testを素材選択・色調整・サイズ決定に使わず、今回データversionやsplitは変更しない。
2. 元画像内の**全ての注釈Face**から除外領域を作る。受け手候補のような5点可視条件で
   顔を絞らず、遮蔽あり・keypoint欠損・非対象Faceも全て含める。
   GT FACE楕円/Face bboxに安全余白を加え、その領域へ少しでも入る素材cropを棄却する。
   HEAD bboxをFace bboxとして誤用しない。除外に必要なFace幾何が壊れている画像は候補から外す。
   顔の未注釈があると混入を完全には防げないため、将来の素材監査で確認する。
3. 残りの範囲から位置・面積・縦横比をランダムに選び、矩形patchを切り出す。
   背景、衣服、手、腕、肩、髪、家具などを意味カテゴリで制限せず、分類器やsegmentationを
   必須にしない。「物体一個」の切り抜きでなく、顔を除いた実画像領域を使う。
   元画像の有効画素のみを使い、画像外のpaddingを素材としない。
4. 貼付サイズも対象Faceの大きさに対してランダムに変える。切出し時に縦横比を選び、
   resize時はその比率を保持する。小さいpatchばかりにならないよう、サイズ分布の範囲と
   サンプリング方式は生成例・分布監査で確定する（今回は数値を固定しない）。

素材側に「目がある」「横顔である」「肌である」等の条件は設けない。
既存実顔pasteの`minimum_donor_face_overlap`やprofile素材比率は、このfamilyには適用しない。
色・輝度の近さを用いる場合も、カテゴリを狭める必須条件ではなく候補の優先順位付けを
初期案とする。具体的なphotometric閾値・適用率は未確定であり、既存値を無断で流用しない。

### 対象顔への配置と境界

- 適用先は従来の方針を引き継ぎ、元注釈で輪郭遮蔽なし・5点すべて可視のFaceとする。
  crop後に外れた点はmissingとして扱う。元の自然遮蔽へさらに人工遮蔽を重ねる条件は今回追加しない。
- 対象顔の周辺を基準に位置・貼付サイズをランダムに選ぶ。画像端からの進入や目/鼻/口付近も
  候補に含めるが、顔の近くにあるだけでなく、対象Faceの実画素を覆った場合に適用と数える。
- 遮蔽率は配置後の**実不透明支持領域と対象FACE楕円の共通面積**から計算する。
  素材側の矩形全体が遮蔽物であり、物体の意味領域だけをmaskにする必要はない。
  顔全体の率に加え、各keypoint近傍と点自体の被覆を記録する。
- 小面積・中程度・強い遮蔽の枠を分けて要求し、生成失敗を軽い例に黙って置き換えない。
  小面積でもkeypoint周辺を隠す例は別枠で扱う。具体的な面積範囲・比率・試行上限は
  実顔familyの値を参考にしつつ、この素材分布で確認して固定する。
- 合成は既存の**crop側のみ、幅48px・強度2、不透明RGBグラデーション**を再利用する設計。
  貼付位置の外側にある元画像の色へつなげ、元画像側は変更しない。alpha合成や隠された
  受け手画素の参照は禁止。小さい素材では48pxが領域の大半に及ぶことを記録する。
- 素材選択/配置不能時は無加工で返し、理由を記録する。別のfamilyや単色へ無断で置換しない。

### 教師・他のaugmentationとの関係

対象FaceのID・座標・楕円は保持。maskが点自体を覆った時だけstate=2→1。
近傍のみの被覆でstateを変えない。crop外の点はstate=0、座標教師対象外とし、
既にmissingの点をoccludedへ変更しない。元データの注釈ファイルは書き換えない。
Messengerで使う場合は合成前teacher/合成後studentとし、復元maskと同じ幾何変換を共有する。
顔素材とは異なり追加されたFaceの教師は作らない（顔が混入した素材は除外対象）。
FORで使う場合も合成後のfull-image queryを取得する原則を維持する。

実顔paste、非顔領域paste、従来object遮蔽は独立したfamilyとし、初期設計では一つの対象に
同時適用しない。混合する場合の適用率は未決定。既存FOR on/off ablationへ追加する際には
両条件へ同一の素材/配置を共有する必要があるが、**今回その学習予定は変更しない**。

### 将来の実装時に確認する項目（今回は未実施）

素材動画・画像・patch座標、注釈/画像/hash、除外mask、貼付変換、seed、要求/実適用経路を記録。
サイズ・縦横比・遮蔽率・隠れたkeypoint数・低コントラスト素材の率、動画ごとの使用率、
skip理由を確認し、軽い例や特定動画への偏りを調べる。意味カテゴリの正解ラベルは捏造しない。
顔の混入、素材がほぼ無地になる例、境界補正が素材を過度に変える例を目視確認する。
元画像側不変、mask/state/missing整合、train限定・別動画、無効時の記録を検証する。

これにより実顔pasteだけでは不足する**非顔の外観・テクスチャによる遮蔽**を増やす狙い。
背景patchが実際の遮蔽物の形状・奥行き・照明を再現するとは限らず、これだけで
自然遮蔽を完全再現したものとは扱わない。名称・結果報告では「非顔領域pasteによる近似」と明記する。

## 2026-10-09: 実顔遮蔽を用いた学習予定

幅48px・強度2をMessenger、現MAL α=2検出器、FOR keypointの3実験に適用する学習予定を登録。
詳細は`docs/real_face_occlusion_training_schedule.md`、機械可読planは
`configs/training_plans/real_face_occlusion_w48_s2.yaml`。loader/教師/query接続待ちで、training **not run**。
今回の追加augmentationは顔による遮蔽のみ。自然遮蔽全般の再現ではなく、今後は顔以外の
物体による遮蔽familyも必要。顔重なりには同じ合成操作を使い、検出教師とFOR合成後queryを別途接続する。

## 2026-10-09: 採用設定 — 幅48px・強度2

ユーザーがv4比較から「48px・強度2」を暫定採用した。実顔crop遮蔽の不透明RGB
グラデーションの既定値をwidth=48、strength=2.0へ更新。幅はresize後のcrop上のpx。
crop側のみ補正し、元画像側は不変、alpha合成なし、crop外keypointはmissingを維持する。
生成処理とmetadata、比較rendererの既定値を一致させた。旧条件の再現は幅/強度の明示指定で可能。
採用対象はaugmentationの境界処理設定であり、本番loader接続・新規学習は **not run**。
本番へ導入する際の独立config・分布確認・四ゲートは引き続き必要。現在の学習条件は変更しない。

## 2026-10-09: グラデーションの幅・強度を独立設定し、広く強い例を比較

ユーザーはv3に「いい感じになっている」と評価し、より広く強いグラデーションを希望。
`opaque_boundary_gradient(..., width=24, strength=1.0)`に強度を追加した。
widthは入力crop上の補正帯の幅（px）、strengthは色補正をcrop内側まで残す程度。
境界の一致を壊す単純なRGB倍率ではない。境界値1・帯外0の影響場uをRGB補正と同時に拡散し、
補正量へ`s/(1+(s-1)*u)`を掛ける。s=1は従来と同じ、s>1は内側への補正を強め、
境界の一致と帯外0は保持する。これは色補正量の重みでありalpha合成ではない。

同じtrain由来8配置を、24px/強度1、48px/強度1、48px/強度2、72px/強度2で比較。
crop側だけを補正し、元画像側は全画素不変。幅が素材サイズに近い場合は、素材の広い範囲が
補正対象となり、色や明暗の変化が強くなる。大きい設定を一律に最良とはみなさない。
training **not run**、本番設定への反映なし。HEAD-free推論topologyと教師状態・二値maskは維持。

再現: `.venv/bin/python scripts/render_opaque_gradient_strength_sweep.py --source outputs/diagnostics/cross_video_face_occlusion_20261009/opaque_gradient_v2 --output <fresh-directory>`。
成果物: `outputs/diagnostics/cross_video_face_occlusion_20261009/opaque_gradient_v4/`の
preview.png、example_01..08/comparison.png、各幅/強度の個別画像、report.json、verification.json。
19 tests passed。幅拡張、強度増加、境界固定、元画像側不変、隠れた画素への非依存を検証。

## 2026-10-09: 貼付位置の境界色へ直接接続、元画像側の変更を禁止

ユーザー指定を受け、元画像の境界色を参照するだけでなく、crop側の境界画素を
直近の元画像外側画素のRGBに直接合わせる方式へ修正。元画像側へのグラデーションは禁止。
前版も貼付位置の元画像を参照していたが、補正量48/255の上限と境界差の平滑化が残差を生んでいた。
その上限と境界値の平滑化を撤廃し、crop内周24pxでのみ補正を0へ減衰させる。
境界の角で外側隣接画素が2つある場合は両者の平均色を用いる。
二値maskを維持し、alpha合成なし。隠れた元画像の内部を参照しない。

`scripts/render_opaque_boundary_gradient_examples.py`でv2の同一8配置を再利用。
出力: `outputs/diagnostics/cross_video_face_occlusion_20261009/opaque_gradient_v3/`。
`preview.png`/各`example_*/comparison.png`は元画像、通常貼付、旧補正、新補正の順。
全8例でmask外画素は元画像と完全一致（変更0画素）。14 tests passed。
境界を挟む隣接画素のRGB絶対差平均は旧15.58–87.55から新0–0.115/255へ低下。
これは境界の色差だけの指標であり、自然さやモデル精度の指標ではない。
目視では色の段差は軽減するが、素材の形や向き、内部の色変化による不自然さは残る。
CPUプレビューのみ、training **not run**。稼働中の学習・loader・config・queueは未変更。

## 2026-10-09: 透明度を使わない境界グラデーションとcrop切り詰めの例

ユーザーの追加指定に従い、素材側の顔は切れていても使用可能とする。
受け手の元注釈は引き続き輪郭遮蔽false・5点可視を要求するが、cropの周辺が画像外へ
はみ出すことは拒否理由にしない。画像範囲で切り詰め、縦横比を保持してresizeする。
crop後に範囲外となったkeypointはstate=0（missing）、座標0（教師対象外）へ変更し、
人工遮蔽によるstate=1と区別する。元データの注釈は変更しない。
下のv1の「cropはみ出し56件を除外」は過去の試作仕様として残す。

`scripts/opaque_boundary_gradient.py`を追加。二値の貼付maskを保持し、alpha合成はしない。
貼付境界の外側に実在する受け手画素と素材境界画素のRGB差を低周波化し、
境界から内側24pxの帯で補正量を徐々に0へ減衰させる。チャンネルごとの補正上限48/255。
隠された受け手画素は参照しない。素材中央・mask外の画像は保持し、画像端そのものには
偽の背景を作らない。輪郭切り抜きや完全なシーム除去ではなく、色・輝度の段差を軽減する試作。
補正で素材の境界色が変わることはあるが、元の顔が透ける処理ではない。

HEAD-free FACE/ellipse -> Face-crop DINO -> Messenger -> Query Refinementを想定した
独立CPUプレビューのみ。本番loader・実行中の学習・ゲート・キューは未変更。training **not run**。
再現: `.venv/bin/python scripts/prototype_cross_video_face_occlusion.py --output <fresh-directory> --samples 240 --seed 42 --opaque-gradient`。
成果物は`outputs/diagnostics/cross_video_face_occlusion_20261009/opaque_gradient_v2/`。
`preview.png`と`example_01`〜`example_08`の`comparison.png`は、左から受け手、別動画素材、
通常貼付、RGBグラデーション補正の同一配置比較。個別RGB画像・二値mask・metadataも保存。
試作分布はreport.json、全240要求の記録はsamples.jsonl。先頭8適用例を表示し、見栄えで選別しない。
13 tests passed: crop切り詰め・missing保持、元注釈遮蔽条件、動画/色選択、面積と状態対応に加え、
mask内部の元画像をランダムに変更しても補正出力が変わらないことを確認した。

## 2026-10-09: 別動画の実顔cropによる遮蔽 — CPU試作と設計

状態: 独立した生成試作まで実装。本番data loaderへの接続・新規学習・モデル精度評価は **not run**。
現在実行中のconditional-object Messengerのコード/config/ゲート/キューは変更しない。
想定配備はHEAD-free ordered cascade: full-image FACE/ellipse -> Face-crop DINO ->
Messenger recovery -> Query Refinement。GT HEAD bboxはtrain素材を切り出すためだけに使い、
推論へHEAD段を追加するものではない。現local baselineはMAL alpha2 FACE/ellipse、
従来のA1/A2 Query pipelineは別の固定keypoint比較対象であり、構成を混合しない。

### ユーザー指定と適用対象

- 別動画の実顔cropを部分的に重ね、背景の色・輝度などが近い組み合わせを選ぶ。
- 貼り付け先はcanonical trainの属性有効・Face対応済み、元の輪郭遮蔽フラグが
  **明示的にfalse**、かつ5点すべてstate=2。state=0/1、フラグ不明、対応不明は除外。
  これは「注釈上の遮蔽なし」であり、顔の全画素に遮蔽がない保証ではない。
- canonical注釈には輪郭遮蔽フラグがないため、train adapterと画像パス＋楕円（小数4桁）で
  一意に照合する。画像・動画IDと画像SHA-256を確認する。
- 素材側もcanonical trainから取得し、貼り付け先とsource_video_idが異なるものだけ使用。
  素材側の5点可視は必須にしない（横顔はmissingを含み得る）。試作では素材側の輪郭遮蔽falseも要求。
- 添付ギャラリーのval/test例は外観の参考のみ。素材・受け手・photometric調整には使用しない。
- crop後の点のはみ出しやcrop自体の画像外へのはみ出しも除外。実運用ではcrop jitter後にも再検査する。

### 試作した選択・合成

`scripts/prototype_cross_video_face_occlusion.py`は学習処理から独立したCPUツール。
HEAD bboxを1.8倍に広げた周辺から全GT HEAD領域を除き、Labの平均色とLの標準偏差を取得する。
衣服・髪・未注釈物体などが残り得るため、これは背景の**近似指標**であり背景segmentationではない。
暫定条件はL差<=12、ab距離<=12、L標準偏差差<=15。条件内の各動画の最良候補を取り、
上位5動画からランダム選択する。同一動画の類似フレーム数で選択確率が増えないようにする。
適合素材がなければ明示的skip。肌や背景を無理に色変換して条件へ合わせない。

素材cropはHEAD bbox＋10% context、受け手はFACE楕円長軸直径の1.25倍の正方形を等方resize。
素材は縦横比を保存して受け手cropの端から進入させる。背景を含む矩形全体が不透明なので、
**貼った矩形全体**を遮蔽マスクとする。素材FACE楕円だけをマスク扱いしない。
受け手FACE楕円に対する実被覆率を配置後に計算し、素材FACE自体も1%以上重なることを要求。
これにより背景だけを貼って「顔による遮蔽」と記録する例を除外する。

暫定の要求分布は小面積(0–10%]20%、中程度(10–20%]40%、強度(20–40%]40%。
小面積は受け手keypointを実際に1点以上覆う場合のみ許可。素材の横顔ratio>=3を50%要求する。
これらの率・photometric閾値は試作設定であり、ユーザー承認済みの学習hyperparameterではない。
経路を選んで最大128回試し、失敗時に軽い経路へ変更しない。seed、両動画、両注釈ID、
経路、矩形、実面積率、素材顔の面積率、状態、mask hash、skip理由を記録する。

座標教師と対象Faceの対応は保持し、覆ったkeypointだけstate=2->1へ更新。state=0へは変えない。
clean teacherは合成前、studentのみ合成後を入力する設計とする。素材のkeypointを対象Faceの
正解へ混ぜない。既存object/boundary遮蔽を適用した上に重ねず、排他的なaugmentation familyとする。
最終的な適用確率・既存familyとの配分は未確定。Full-image検出学習へ転用する際は、
追加された素材Faceの検出教師まで設計し直す必要があり、このcrop段の試作を直接流用しない。

### 実測と採用前に残る点

固定train: 4,237画像/44動画/7,129 HEAD。属性有効Face対応は6,447件。
受け手のラベル条件を満たすのは2,550件。周囲の色測定に十分な画素がある候補は2,524件。
素材候補4,959件。seed42で受け手240件を試し、150件適用、90件skip:
cropはみ出し56、配置予算33、photometric適合素材なし1。

| 実際の遮蔽率 | 件数 | 適用150件中 |
|---|---:|---:|
| 0%超–10%以下（keypoint実被覆あり） | 10 | 6.7% |
| 10%超–20%以下 | 67 | 44.7% |
| 20%超–40%以下 | 73 | 48.7% |

被覆keypoint数0/1/2/3/4/5の件数は44/68/26/11/1/0。41素材動画から選ばれ、最多は13/150件。
軽い例への集中はない一方、要求した小面積枠の適用率は低い。全要求のskip率37.5%は高く、
本番分布ゲートの合格とはしない。まずcrop適格性を先に判定し、経路別・動画別・profile別の
実適用率を監査する必要がある。現在のGT固定crop監査は本番jitter分布を代表しない。

12例の目視では、色・輝度が近くても矩形の境界が目立ち、髪や顔の自然な輪郭による遮蔽と
異なる例が残った。**矩形pasteを完成した自然遮蔽として採用しない**。次の比較対象は
顔・髪を含むforeground silhouetteによるcutout。FACE楕円だけの切り抜きでは髪を失うため不十分。
soft alphaを導入する場合は境界点の状態教師を別設計し、alpha>0だけでoccludedにしない。
素材maskの品質と出所の監査、実cropでの分布確認、独立config、四つの新ゲートを経て
将来のMessenger学習条件へ接続する。現在の試作は精度改善・自然さを証明しない。

再現: `.venv/bin/python scripts/prototype_cross_video_face_occlusion.py --output <fresh-directory> --samples 240 --seed 42`。
成果物: `outputs/diagnostics/cross_video_face_occlusion_20261009/prototype_v1/`
の`report.json`（inventory/分布/hash/制約）、`samples.jsonl`（個別生成記録）、
`preview.jpg`（左=受け手、中央=素材、右=合成後）、`verification.json`（テスト記録）。
`tests/test_cross_video_face_occlusion_prototype.py`: 10 passed。
未知/既遮蔽の受け手拒否、別動画・色条件、実マスクとstate対応、teacher保持、経路面積上限を検証。

## 2026-10-09: 条件付き下限免除の実装とtrain分布確認

ユーザー指定を受け、`object_overlap_policy`を明示したobject augmentationだけに適用する
独立した`conditional_v1`を実装した。既存A/B/C configと学習キューは変更していない。
HEAD-free FACE/ellipse -> crop DINO -> Messenger -> Query Refinement。trainのみのCPU監査であり、
本学習・GPU評価・再試行は **not run**。以下の以前の再設計案は議論の履歴。

### 受入条件

- 通常は顔楕円内の実不透明面積10–40%。全経路で40%上限、ゼロ重なりの拒否を維持する。
- 素材例外: 明示した`exempt_material_ids`に限り10%下限を外す。初期対象は`glasses`。
- 部位例外: 素材を問わず、ラベルありかつcrop内のkeypoint近傍を20%以上覆えば10%未満を許可。
  近傍は半径`0.08 * (顔楕円の長軸直径)`の円と顔楕円の共通部分。
  left_eye/right_eye/nose/left_mouth/right_mouthを個別に計測する。
  state=0、非有限座標、crop外の点は例外判定に使わない。
- 近傍を覆うことと点自体を隠すことを区別し、keypoint stateを1へ変更するのは不透明画素が
  点自体を覆った場合だけ。元のmissing/natural-occludedを保持する。
- 縦横比を保持して、顔長軸直径の0.20–1.20倍の範囲で素材を配置。回転±20度、左右反転。
  alpha0/255の穴は保持する。半透明合成は今回の実装対象外。
- 例外を全サンプルへ適用しない。要求時に素材例外10%、部位例外10%、10–20%の通常例40%、
  20–40%の通常例40%から経路を選び、その経路を維持して最大128回配置する。
  強い遮蔽を配置できなかった場合に、軽い遮蔽へ黙って置き換えない。
- 配置不能/有効keypointなしは原画像へ戻し、理由・経路・素材・seed/epoch・sampleを記録。
  別素材/単色へ置換しない。素材破損・hash不一致・非有限値は引き続きエラー。
- 素材/epoch/seed別の実適用率と分布を監査。事前設定した許容値はskip<=5%、適用例中の
  軽度(<10%)<=30%、強度(20–40%)>=25%。全体と各seed/epochで確認し、素材別skipも<=5%。
  glassesだけは意図した追加の軽度枠を持つため、素材ごとの強度分布は別掲する。

### 実測結果（精度評価ではない）

固定trainの6,529 Face records全体をseed42/43、epoch1/12/40で走査（39,174 record-epoch visits）。
各source imageにつき確率30%で最大1 Faceへ要求する既存ルールを維持。
6,838回要求、6,800回適用、38回skip（0.556%、全て`no_eligible_keypoint`）。
配置試行上限への到達と生成例外はいずれも0。

| 遮蔽率 | 適用数 | 適用例中の割合 |
|---|---:|---:|
| 0%超～10%未満 | 1,367 | 20.10% |
| 10%以上～20%未満 | 2,729 | 40.13% |
| 20%以上～40%以下 | 2,704 | 39.76% |

軽度例は素材例外728、部位例外639。各seed/epochの軽度18.69–21.07%、強度38.55–41.21%。
素材別の軽度/中度/強度はglasses34.71/32.54/32.75%、leaf10.22/45.79/43.99%、
bottle10.06/44.82/45.12%。軽度のみへの集中は観測されない。
ただし有限の生成監査であり、学習精度改善・全seedでの無失敗を保証しない。

コード: `keypoint_comparison1/landmark_detection/occlusion_policy.py`。
独立config: `keypoint_comparison1/configs/occlusion_conditional_object_audit.yaml`（本学習登録なし）。
再現: `.venv/bin/python scripts/audit_occlusion_distribution.py --output <fresh-directory> --epochs 1 12 40 --seeds 42 43`。
成果物: `outputs/diagnostics/occlusion_conditional_object_20261009/v1/{distribution.json,samples.jsonl,resolved_config.json,verification.json,pytest.txt}`。
48 tests passed。既存素材処理の互換性、例外条件、ラベル、上限、失敗記録、軽度偏重の拒否を確認。
通常データローダーを使うが、crop乱数は監査用sample seedで固定しておりtraining worker順序と同一ではない。
新モジュールをproduction実装hashの対象へ追加した。旧成功markerを更新/捏造せず、
本学習へ採用する際は新条件用の独立した四ゲートとconfig/実装の再認証が必要。


## 2026-10-09: 遮蔽物の形状・透明度を保つaugmentationの再設計案

状態: 設計案。実装・config変更・学習・再試行・新規評価は **not run**。
10月4日の10–40%契約と実験結果は履歴として保持し、既存ゲートを成功へ変更しない。
配備topologyはHEAD-free ordered cascade: full-image FACE/ellipse -> Face-crop DINO -> Messenger recovery -> Query Refinement。
local baselineはA1/A2、global baselineは配布referenceのまま。外部HEADを追加しない。

### 問題の切り分け

- 現行素材のalphaは0/255だけ。透明な穴は表現するが、透けたレンズ等の半透明は表現しない。
- 全素材に実不透明面積10–40%を要求すると、細い眼鏡や穴のある形状を棄却しやすい。
  薄い遮蔽の頻度・自然遮蔽に対する精度への影響は未測定。設計上の選別バイアスとして扱う。
- 現行objectは矩形候補へ素材を縦横別にresizeし、実alpha支持領域で再判定する。
  面積を合わせるための巨大化・形状変形・不透明化は、この問題の解決方針にしない。

### 提案する生成と計測

1. 素材family、サイズ、位置、回転を先に定義した分布から生成する。
   objectは縦横比を保持して配置し、10–40%用の矩形生成器への依存を外す。
   眼鏡は目の近傍への配置と自由配置を分ける。必要な位置情報はtrainの注釈のみを使う。
2. 顔全体の遮蔽率はまず計測値とする。5%などの低面積でも、目の近傍を部分的に隠す
   サンプルを面積不足だけで棄却しない。細い物体に一律の10%下限を設けない。
   無遮蔽・軽度・中度・強度の分布を別途確認し、低面積の許容が強い遮蔽の消失を招かないようにする。
3. 顔楕円内の支持領域率、alpha平均（不透明度で重みづけした面積）、目/鼻/口近傍の
   局所的な被覆を別々に記録する。alphaは知覚的な見えづらさの正解とはみなさない。
   透明部分は数えず、全体面積だけで局所的な難しさを代表させない。
4. 半透明はbinary cutoutと分離した追加条件とする。導入時は
   I_aug=(1-alpha)*I_clean+alpha*I_materialで合成し、soft alphaをbool maskへ潰さない。
   元の自然遮蔽stateを保存し、alpha>0だけを理由にGT visibleをoccludedへ変更しない。
   特徴損失の重みとkeypoint状態教師への反映を個別に設計・検証してから導入する。
5. 同じ素材familyの中で有界の配置試行を行う。配置不能時は原画像を返し、
   applied=false・理由・素材ID・sample ID・seed/epoch・試行数を記録する方針を候補とする。
   別素材や単色への黙示的な置換は禁止。skipによる実適用率低下と素材別偏りをゲートで検査する。
   壊れた素材、hash不一致、非有限値等の不正はエラーのまま維持する。

### 実装前に確定する項目と受入検証

- family別の確率、サイズ/位置/回転分布、面積上限、無遮蔽率、許容skip率は未確定。
  trainのみのCPU生成監査で実現可能性を調べ、学習前に固定する。testで選ばない。
- 全trainサンプルと事前宣言した複数epoch/seedを対象に、生成例外、family別の
  要求/実適用数、skip理由、支持率・alpha・局所被覆の分布、形状保持を検査する。
  先頭96サンプル/epoch1の検査だけで安全を認定しない。
- レビュー画像はtrainのみ。clean/aug/alphaの対応、透明な穴、細いフレーム、半透明、
  顔端、小さい顔を確認する。固定split・clean/student座標共有・自然stateの保存を維持する。
- 新しい独立config/output/素材provenanceで四つのゲートを再実施し、コードhashを再認証する。
  bounded overfit、production augmentation small-scaleの有限な検証改善、安全条件を維持する。
- textureは同一maskで見た目を比較できるが、objectは形状・位置も変わるため、
  結果を純粋な見た目の効果と解釈しない。自然遮蔽・可視点・遮蔽強度別のvalidationを併記する。
  性能向上は現時点で未確認。実装・学習開始の指示としてこの設計案を使わない。

2026-09-30更新: 案1・案3の実装、個別ablationと案1教師を使う統合構成、MAL validation後の
スケジュール登録を実施。案2は保留。本学習はnot run。
初期値・判断根拠・検証結果は[実装記録](messenger_continuous_feature_recovery_implementation.md)を参照。
以下の「計画のみ/not run」は各設計メモ記録時点の履歴。

記録日: 2026-09-30。状態: 設計メモ作成済み、実装・学習・評価は **not run**。
既存日誌 `messenger-recurrent-recovery-design-20260930` の2層Messenger設計を拡張する。
今回の依頼は追加実装内容の記録であり、実行コード・config・queueの変更ではない。

## 合意した目的と範囲

1. 各空間パッチのDINO特徴を、人工遮蔽を加える前の特徴へ復元する。
2. パッチ同士の特徴類似度が、人工遮蔽を加える前の顔と同じ構造になるよう復元する。

ここで「部位」はDINOの空間パッチを指し、目・鼻などのsemantic slotを直接表すものではない。
関係の復元は特徴類似度の構造を対象とし、明示的な座標・距離・3D顔形状の一致を保証しない。
cleanとは人工遮蔽前の対応画像であり、元画像に存在する自然遮蔽を除去した正解画像ではない。

配備topologyは既存のHEAD-free ordered cascade:
`full-image FACE/ellipse -> Face-crop keypoint`。
復元器はFace-crop DINOとQuery Refinementの間に配置し、HEADを追加しない。
比較基準はローカルbaseline `deimv2_l_query_refinement_landmark_occlusion_a1_a2`。
無効時は既存経路・checkpoint contractを維持する。

## 記号と教師

同じcrop・座標変換を共有する人工遮蔽前画像を I、人工遮蔽後画像を I_occ とする。
N=H*W個のパッチ、C次元（現行モデルでは384）について、

\[
X=f_\theta(I_{\mathrm{occ}})\in\mathbb R^{N\times C},\qquad
Z=\operatorname{sg}[f_{\bar\theta}(I)]\in\mathbb R^{N\times C}.
\]

sgはstop-gradient。最初の比較実験では採用済みcheckpointのDINOを固定し、
theta=bar(theta)として同じ特徴座標系を使う。checkpointパス・SHA-256は実験開始時に記録する。
固定DINOで効果確認後に微調整とEMA教師を独立した実験として検討する。

人工遮蔽の画素マスクから、各パッチの遮蔽面積率 r_i in [0,1] を求める。
教師・学生は同じパッチ格子に対応させ、左右反転などの幾何変換も共有する。

## 2層Messengerとの接続

既存メモの方針を維持し、Xは全復元層で固定の共有メモリ・比較基準として使う。
Messenger状態 M^(l) のみ前層から引き継ぐ。以下は実装時に詳細を確定する数式仕様案。

\[
M^{(0)}_i=t+p_i,\qquad
M^{(l)}=M^{(l-1)}+R_l(M^{(l-1)},X;A^{(l)}),\quad l=1,2.
\]

tは学習可能token、p_iは位置表現。R_lはcross-attention/FFNを含む残差更新で、
出力側をゼロ初期化する。復元候補を直接監督し、初期のゲート抑制で学習が止まらないようにする。

\[
A^{(l)}_{ij}=\operatorname{softmax}_j
\left(\frac{q_i^{(l)\top}k_j}{\sqrt d}+b_j^{(l)}+D_{ij}\right),
\quad D_{ii}=-\infty,\ D_{ij}=0\ (i\ne j),
\]
\[
b_j^{(1)}=0,\qquad b_j^{(2)}=\log(1-\alpha_j^{(1)}+\epsilon).
\]

前層の遮蔽推定が大きいkeyの参照を単調に抑制する。対角除外は最終DINOパッチの直接参照を
除外するだけで、DINO内部ですでに混合された遮蔽情報の完全排除を保証しない。
softmaxはFP32等で安定化し、参照先が存在しない退化ケースを別途扱う。

ゲート入力には正規化特徴のコサイン非類似度を明示的に与える。

\[
d_i^{(l)}=1-\cos(X_i,M_i^{(l)}),\qquad
\alpha_i^{(l)}=\sigma\{h_l([\operatorname{LN}(X_i),
\operatorname{LN}(M_i^{(l)}),d_i^{(l)}])\}.
\]

両層のゲートをr_iで監督し、最終層を主とする。3層目はvalidationで継続的改善がある場合のみ検討。

## ① 連続DINO特徴の直接復元

\[
\hat Z_i=D_\phi(M_i^{(2)}),\qquad
Y_i=(1-g_i)X_i+g_i\hat Z_i,\quad 0\le g_i\le1.
\]

D_phiはC次元のraw DINO特徴を出す小さなMLPを第一候補とする。離散量子化・argmaxは不要。
Yを既存Query Refinementへ渡す。初期の比較では g_i=alpha_i^(2) とし、復元器を先に学習してから
混合を有効にする。遮蔽率と復元候補の採用率は異なる概念であり、別の信頼度ゲートは後続候補。

この式もX+g*(Zhat-X)と書けるため、残差形式自体の排除が目的ではない。
現行のX+alpha*W(M-X)と異なり、完全な復元候補Zhatと実際の出力Yの双方を教師へ直接整合させる。
凸結合は特徴多様体上にあることや、精度向上を保証しない。

固定教師のtrain特徴から算出するチャネル別スケールs_c>0を使い、

\[
\ell(u,z)=\lambda_{\cos}\left(1-
\frac{u^\top z}{\max(\|u\|_2,\epsilon)\max(\|z\|_2,\epsilon)}\right)
+\frac{\lambda_{\mathrm{amp}}}{C}\sum_{c=1}^C
\operatorname{Huber}_\delta\left(\frac{u_c-z_c}{s_c}\right),
\]
\[
\operatorname{Huber}_\delta(e)=
\begin{cases}e^2/2&|e|\le\delta,\\
\delta(|e|-\delta/2)&|e|>\delta.
\end{cases}
\]

方向と振幅の両方を監督する。HuberとライブラリのSmoothL1はスケールが異なる場合があるため、
実装時には上式かSmoothL1かを明記する。予測と教師の個別正規化だけで振幅誤差を消さない。
s_cはtrainのみから求める標準偏差等を下限でclampして固定し、算出方法・値・provenanceを保存する。

\[
w_i=r_i+\beta(1-r_i),\quad 0<\beta<1,
\]
\[
L_{\mathrm{candidate}}=\frac{\sum_i w_i\ell(\hat Z_i,Z_i)}{\sum_iw_i},\qquad
L_{\mathrm{output}}=\frac{\sum_i w_i\ell(Y_i,Z_i)}{\sum_iw_i}.
\]

非遮蔽位置も弱く監督する。Attentionにより遮蔽影響が他パッチへ広がる可能性があるため、
非遮蔽位置で常にY=Xを厳密強制することはしない。人工遮蔽なしのサンプルではZ=Xなので保存を促せる。

## ② パッチ間関係の復元（Gram整合性）

行ごとにL2正規化した特徴を bar(Y),bar(Z) として、

\[
\bar Y_i=Y_i/\max(\|Y_i\|_2,\epsilon),\qquad
\bar Z_i=Z_i/\max(\|Z_i\|_2,\epsilon),
\]
\[
G_Y=\bar Y\bar Y^\top,\qquad G_Z=\bar Z\bar Z^\top,
\qquad L_{\mathrm{relation}}=\frac{1}{N^2}\|G_Y-G_Z\|_F^2.
\]

画像ごとに計算してbatch平均する。これはチャネル間GramではなくN*Nのパッチ間Gram。
遮蔽パッチと可視パッチの関係も含め、顔全体の特徴類似度構造を教師へ合わせる。
Gramだけでは共通直交変換等の自由度が残り、既存headとの特徴座標系が決まらないため、①と併用する。
空間近傍の単純な平滑化を強制するものではない。計算量が問題なら固定条件のペアサンプリングを
後続候補とし、初回はfull Gramを基準にする。

## ゲートと総損失

S_occ={i:r_i>0}, S_vis={i:r_i=0} とし、存在する集合ごとの平均を等重みで平均する。

\[
L_{\mathrm{gate}}^{(l)}=
\frac{1}{|\mathcal S|}\sum_{S\in\mathcal S}
\frac{1}{|S|}\sum_{i\in S}(\alpha_i^{(l)}-r_i)^2,
\quad \mathcal S=\{S_{\mathrm{occ}},S_{\mathrm{vis}}\}\setminus\{\varnothing\},
\]
\[
L_{\mathrm{gate}}=\eta L_{\mathrm{gate}}^{(1)}+L_{\mathrm{gate}}^{(2)},\quad 0<\eta\le1,
\]
\[
L=L_{\mathrm{task}}+\lambda_pL_{\mathrm{candidate}}
+\lambda_yL_{\mathrm{output}}+\lambda_rL_{\mathrm{relation}}
+\lambda_gL_{\mathrm{gate}}.
\]

taskは既存キーポイント/state等のタスク損失。空の集合に対する平均を作らない。
係数、Huber閾値、teacherスケール下限、混合有効化の時期は未決定。validationで選択しtestでは選ばない。
2層間の誤差が必ず単調減少する保証はないため、各層のゲート・復元指標を診断する。

## 根拠と実証範囲

- [CAE](https://arxiv.org/abs/2202.03026): 可視表現と位置から欠損領域の連続表現を予測し、encoder表現に整合させる設計の根拠。
- [I-JEPA](https://arxiv.org/abs/2301.08243): 画像の文脈から別領域の表現を予測する根拠。
- [MaskFeat](https://arxiv.org/abs/2112.09133): DINOを含む連続特徴を予測対象として検証した先行例。
- [DINOv3 §4.2](https://arxiv.org/html/2508.10104v1#S4.SS2): パッチ間Gram整合性によるdense特徴の維持・改善の根拠。

これらの組み合わせによる本データセットの遮蔽顔キーポイント改善は未実証。
教師固定、混合、損失の組み合わせと係数は今回の応用設計であり、論文の忠実再現ではない。

## 比較実験と未実施事項

1. 現行構造に最終出力への直接特徴回帰を追加し、監督不足の影響を切り分ける。
2. 2層Messenger＋直接復元候補＋混合を評価する。層数と出力形式の寄与を分離する必要があれば、
   1層の直接復元候補を対照として追加する。
3. 同じ条件にGram損失を追加する。

固定source-video splitを維持。教師統計・追加学習はtrainのみ。
validationの既定conditional semantic keypoint Macro F1@NME0.10をcheckpoint選択に使い、
可視/遮蔽NME、復元誤差、特徴norm、Gram誤差、ゲート誤検出も確認する。
人工遮蔽復元と自然遮蔽のタスク精度を区別し、特徴誤差だけで採用を決めない。
共通評価はHEAD-free契約を守り、既存local baselineと同じbbox検出契約・閾値条件を使う。
seed、checkpoint/config/annotationのSHA-256、batch plan、W&B、教師統計provenanceは実験時に記録する。
今回それらの実験値は未設定で、学習・評価・テストはnot run。

## 合成方法の比較方針（2026-09-30追記）

復元器・データ・下流モデルを可能な限り共通化し、以下の3案で採用率の決め方を比較する。
本節は上記の復元器/Gramの比較とは別の、合成方式の比較軸である。

1. 基準方式: clean特徴に対する二乗誤差を最小化するパッチ別混合率を学習時の教師とする。
   Xを元特徴、Rを復元候補、Zをclean教師として、
   g_star=clip(dot(Z-X,R-X)/||R-X||^2,0,1)。分母が小さい位置の扱いと
   スケール正規化は実装時に規定する。推論時はZを使わず学習したゲートを使う。
2. **事前学習済み姿勢モデルを使う方式は後日実装。** モデル選定・重み取得は未実施。
   checkpoint/ライセンス/前処理/姿勢座標系/遮蔽下の信頼性を確認し、学習時のみの教師として
   蒸留するか推論時にも使うかを決める必要がある。現在は角度GTがない。
   姿勢を条件として合成後の画像固有の顔構造が正しくなる係数を学習する方針。
   この記録はモデルのダウンロード、コード実装、queue登録、学習開始を伴わない。
3. 潜在構造モードを使う方式は、以下の構成を案3内のbaselineとして後日実装する計画に決定。
   角度教師なしの潜在変数をyaw/pitch/rollと断定しない。詳細ハイパーパラメータは未決定。

今回の追記に伴う学習・評価はnot run。

## 案3 baseline：構造コンテキストによる採用率補正の実装計画

2026-09-30ユーザー指定: 今回は計画の記録のみ。実装は後日、メモを参照してまとめて行う。
本節のbaselineは「案3の初期構成」を意味し、プロジェクトのlocal/global baselineを置き換えない。
モデルコード・config・queueは未変更。実装・学習・評価はnot run。

### 目的と構成

顔全体の構造モードとパッチごとの採用率を分ける。既存GTの相対配置を説明できる潜在モードを
学習し、それで局所採用ゲートを補正する。独立した顔らしさ判定器や明示的姿勢角は要求しない。
既存の2層Messenger、直接復元候補R、HEAD-free topologyを維持し、初期楕円の再補正は本案の範囲外。

1. 位置情報を含む元DINOパッチから、軽量Attentionで構造コンテキストを集約する。
2. 学習可能なプロトタイプへのsoft assignmentから潜在構造表現zを作る。
3. zのみからGTキーポイントの相対配置を予測し、構造との対応を監督する。
4. 局所ゲートのlogitに、zで条件付けした構造補正を加える。
5. 合成結果の特徴・Gram・タスク損失も伝播し、構造が説明できるだけでなく合成に役立つよう学習する。

### 数式

X_iは元特徴、R_iは復元候補、alpha_iは遮蔽推定、p_iはパッチ位置表現。

\[
c=A_\psi(\{X_i+p_i\}_{i=1}^N),\qquad
\pi_k=\operatorname{softmax}_k\left(\frac{\operatorname{sim}(Wc,e_k)}{\tau}\right),
\qquad z=\sum_{k=1}^K\pi_ke_k.
\]

e_kは学習可能プロトタイプ。simは正規化内積を候補とする。tau>0。
K、潜在次元、Attentionのhead数/深さ、tauは未確定で、実装時に明示的configへ記録する。
正面/横顔等のクラス名を手で割り当てず、角度・表情・個人差を含み得る構造表現として扱う。
zの混合はゲートの条件表現の混合であり、複数の完成顔や座標を平均する処理ではない。

固定semantic slotのGT座標をq_a^*とし、

\[
s_{ab}^*=\frac{q_a^*-q_b^*}{s},\qquad
\hat s=D_{\mathrm{shape}}(z),\qquad
L_{\mathrm{descriptor}}=
\frac{1}{|\mathcal E_{\mathrm{valid}}|}
\sum_{(a,b)\in\mathcal E_{\mathrm{valid}}}\rho(\hat s_{ab}-s_{ab}^*).
\]

候補は5点の全10ペア(a<b)。両端のGT state>0かつ有効な座標のペアだけを監督し、
ペアがない画像はdescriptor lossから除外する。rhoは成分ごとHuber等を候補とする。
s=1.20*max(GT Head width,GT Head height)は学習損失の正規化にのみ使い、GT Headを推論入力にしない。
qは画像座標へ揃えるか、crop逆変換と同等の正規化を使い、軸方向・反転時のsemantic slot交換を
実装時に明記する。並進を除いた粗い2D配置の監督であり、輪郭や正確な3D姿勢の教師ではない。
可視/遮蔽stateを姿勢クラスへ変換しない。

\[
u_i=[\operatorname{LN}(X_i),\operatorname{LN}(R_i),
1-\cos(X_i,R_i),\alpha_i,p_i],
\]
\[
g_i=\sigma\big(h_{\mathrm{local}}(u_i)+h_{\mathrm{structure}}(u_i,z)\big),\qquad
Y_i=(1-g_i)X_i+g_iR_i.
\]

h_structureの最終出力層のweight/biasをゼロ初期化し、初期状態は局所ゲートと同一にする。
補助descriptor lossはこの初期状態でも構造ヘッドを学習できる。
alphaは人工遮蔽率で監督するが、採用率gを遮蔽率へ直接一致させない。
案1のoptimal gate教師は案3で必須とせず、利用/初期化する場合は追加条件として記録して切り分ける。

既存の総損失に追加する基本項は、

\[
L_{\mathrm{option3}}=L_{\mathrm{existing}}
+\lambda_dL_{\mathrm{descriptor}}
+\lambda_cL_{\mathrm{consistency}}.
\]

L_existingは候補Rと出力Yの復元、YのGram、既存キーポイント/stateタスク、alphaゲート損失を含む。
合成後の予測キーポイントにも相対配置損失を加える案は追加ablation候補とし、
descriptor監督との役割を分ける。lambda等の数値は未決定。

### 人工遮蔽前後の整合性（任意の追加項）

同じ幾何変換を共有するclean/occludedペアでは構造が変わらないことを利用する。

\[
L_{\mathrm{consistency}}=
w_{\mathrm{pair}}D_{\mathrm{KL}}
\big(\operatorname{sg}[\pi_{\mathrm{clean}}]\|\pi_{\mathrm{occ}}\big).
\]

clean側は学習時だけ使用。初期baselineはlambda_c=0を対照として保持し、この項の有無を比較する。
有効化する場合、強い遮蔽でw_pairを弱める条件、clean側の直接descriptor監督と
stop-gradient/EMAの利用方針を確定する。EMAは必須としない。
全画像で同じpiでも整合性損失は小さくなるため、単独の学習目的にしない。

### 崩壊・無視の検出と評価

- mode平均使用率、画像間piの変動、assignment entropy、prototype間類似度を確認する。
- モードを均等に使用しただけでは構造学習の成功としない。均等化正則化は初期必須項にせず、
  実際の姿勢頻度を歪める可能性を考慮し、崩壊が観測された場合の独立候補とする。
- descriptor誤差、局所gateに対する構造logit補正量、最終gの変化を診断する。
- 構造補正をゼロにする対照と、zを別画像のものに入れ替える感度診断を行う。
  入れ替え時の悪化だけでは正しい構造利用の証明にならず、補正なしに対するvalidation改善も必要。
- モードごとの画像/GT配置を確認する場合はtrain/valのみで解釈し、横顔等の意味を事前に断定しない。
- 案1、局所gateのみ、構造条件付きgateを比較し、復元器・入力・学習予算・下流構成を可能な限り揃える。
  局所gateのみと構造付きではoptimal gate教師の有無も揃え、複数要因を混同しない。
- 既定validation conditional semantic Macro F1@NME0.10で選択し、可視/遮蔽NME、state、
  人工遮蔽の強度別指標も確認。testでKや係数・checkpointを選ばない。

### 後日の実装手順と検証予定

構造ヘッド・プロトタイプ・descriptor decoder・gate補正・追加損失を独立configで追加し、
無効時の既存経路/checkpoint互換性を保つ。shape、有限値、空のvalid pair、座標/反転、
ゼロ初期化時の局所gate一致、構造ヘッドとgateへの勾配、clean教師の非使用を検証する。
推論はoccluded/通常入力のみで動き、GT座標・GT Head・clean画像を必要としないことを確認する。
実装時にcanonical queueへ登録し、schema/gpu_step/overfit/small_scaleとmicrobatch preflightを
経てからproductionを行う。active runの契約は変更しない。
今回これらは計画のみであり、コードテスト・gate・学習・評価・重み取得はいずれもnot run。

## Downstreamの決定（2026-09-30）

ユーザー合意: Query Refinementは現状の構成を維持し、設計中の合成特徴Yを入力する。
復元・合成方式の初期比較では、downstreamの構造を同時に変更せず、その効果を切り分ける。
特徴の次元・空間配置・既存headの入力契約を維持する。
初期構成には独立した品質判定器、品質によるrawへの自動切り替え、信頼度入力や
二系統メモリを追加しない。合成率gを復元品質の保証とは解釈しない。
現状構成の維持はhead重みの凍結を意味せず、凍結/共同学習の条件は実験設定時に明示する。
HEAD-free full-image FACE/ellipse -> Face-crop keypointのtopologyを維持する。
本決定は計画の記録のみ。今回の実装・学習・評価はnot run。

## キーポイント損失の勾配経路（2026-09-30確認・追加要件）

現在のコードを静的確認: model.pyのstudent forwardはDINO -> Messenger -> Query Refinementを
detachせず接続し、training.pyはタスク損失と復元損失を合計してbackwardする。
optimizerはbackbone/head/messengerのrequires_grad=Trueのパラメータを登録する。
Messenger configの継承元はmodel.backbone.freeze=false、freeze_backbone_epochs=5、
lr_backbone=1.25e-5。既定staged_unfreezeでは最初の5 epochはDINOを凍結し6 epoch目以降更新する。
clean教師forwardのno_gradは意図した教師側のみの停止。実行時の勾配測定は今回not run。
現行correction projectionがゼロ初期化のため、初回のキーポイント勾配は補正projectionには届くが、
その先のMessenger予測器・alphaゲートへは同経路の勾配がゼロになる。補助損失は別経路で監督する。

後日実装の要件: キーポイント損失からY、採用ゲート、構造補正、soft assignment、
構造Attention/プロトタイプ、復元decoder、Messenger、studentのcrop後DINOまで微分可能に接続する。
student経路でdetach/no_grad、hard argmax、オフライン特徴cacheによる切断を行わない。
GT・clean教師・oracle gate教師のstop-gradientは維持する。
descriptor専用decoderはキーポイントforwardに含まれないためdescriptor損失で学習し、
共有構造表現にはdescriptorとキーポイント双方の勾配を伝える。

上記の固定DINO案は切り分け用の初期実験に限定する。最終的な共同学習フェーズでは
student DINOとQuery Refinementも更新可能にし、凍結期間/解除時期/LRを明記する。
教師固定/EMAとstudent更新による特徴座標系の整合は別途実験条件として確定する。
復元層間でXを固定メモリとして使うことは、Xのautograd切断やDINOパラメータ凍結を意味しない。
全画像FACE検出器までの逆伝播を今回の要件に含めない。

ゼロ初期化した構造logit補正や復元残差は初回に上流勾配をゼロにし得る。
補助損失で初期学習を支え、更新後にキーポイント単独損失からも勾配が届くことを確認する。
単にtotal lossで非ゼロ勾配があるだけでは合格にしない。
後日の検証では、キーポイント単独損失のbackwardによる各対象moduleの有限gradと、
初期化直後/複数step後/凍結解除後の更新を分けて検証し、optimizer登録漏れ・重複も検査する。
構造分岐の無効時は既存経路を保持する。今回の変更はメモのみでコードテスト・学習はnot run。

## ユーザーが示した理解（原文）

> ①は「部位ごとのDINO特徴の復元」
> ②は「それらの部位同士の関係が、遮蔽なしの顔と同じ構造になっているかの復元」

## Messenger token modelの改善アイデア：multi-scale参照と動的範囲（2026-10-02）

状態: 改善候補のメモ。実装・テスト・学習・評価は **not run**。
既存の実装・実験設定・学習キューを変更する決定ではない。
配備topologyはHEAD-free ordered cascade
`full-image FACE/ellipse -> Face-crop keypoint`を維持し、Query Refinementの構成と入力契約を保つ。

### 目的と情報の流れ

局所的な遮蔽には近隣の細かい特徴、大きな遮蔽には離れた顔部位や顔全体の配置を利用する。
現行attentionも参照重みは入力依存である。追加候補は参照特徴の空間スケールと、位置に応じた参照範囲の学習。
ORFormerのMessengerは他のregular tokenをcross-attentionで参照し、自身に対応するtokenを除外する。
Messenger同士のself-attentionは導入しない。regular特徴側の処理とMessenger更新を分離する。
出典: [ORFormer §3.2](https://arxiv.org/html/2412.13174v1#S3.SS2)。以下の拡張と効果予測は本プロジェクトの未検証案。

### 初期候補と判断根拠

- 顔cropは256×256、DINOのpatch sizeは16のまま、Messengerを16×16個、出力を384次元で維持する。
  既存DINOとQuery Refinementの入力・出力契約を変えず、参照方法の効果を切り分けるため。
- regular参照メモリを16×16、8×8、4×4の3段階とする。初期候補は非重複2×2/4×4平均pooling。
  追加DINO forwardなしに局所・中域・広域の情報を用意でき、集約元を明確に追跡できるため。
  これは出力特徴のmulti-scale化であり、元の16画素patchより細かな画像情報を増やす処理ではない。
- 各Messengerは複数スケールのregularメモリだけをKey/Valueとして参照し、位置・スケール情報を付加する。
  初期候補はスケールのsoftな混合と学習可能な相対位置バイアス。遠方への参照経路を残す。
  参照範囲の予測はMessenger状態と位置に基づけ、対象位置のregular特徴を直接入力しない。
  第1層で集めた文脈を第2層の動的選択に利用する。自由なdeformable samplingは後続の独立比較候補。
- 自己参照禁止は全スケールで適用する。対象patchを集約元に含む粗いセルは丸ごとmaskする方式を初期候補とする。
  中心座標ではなく集約元の範囲で判定し、softmax前に禁止する。単純で検証しやすいため。
  対象patchのみを除いて再集約する方式は、近隣情報を残せる一方、queryごとのメモリ作成が必要なため後続候補。
- 将来convや補間を使う場合は、その演算が参照する全領域までmaskの判定対象とする。
  DINO/regular側のself-attentionによる間接的な情報混入まで完全に除く保証ではない。

### Messenger self-attentionを初期案に含めない意味的理由

Messenger M_iは「位置iの観測を直接使わず、他の場所から位置iを推測する」役割を持つ。
他のMessenger M_jはregular X_iを参照できるので、Messenger間self-attentionを加えると
`X_i -> M_j -> M_i`という追加の自己情報の戻り道ができる。Messenger間の対角maskだけでは防げない。
また、誤った補完仮説が他のMessengerへ広がり、局所観測と周囲からの推測の差を遮蔽判定に使う意味が弱まる可能性がある。
これらは構造からの懸念であり、self-attention追加が必ず精度を下げるという実証結果ではない。
部位間の整合性向上という利点も考えられ、採用可否には独立ablationが必要。
ここで制限するのは異なる位置のMessenger間の情報交換であり、同じMessengerの層間状態更新やtoken単位FFNではない。

### 後日の比較と検証候補

- 現方式、multi-scaleのみ、動的範囲のみ、両方の4条件を、合成方式・Query Refinement・学習予算を揃えて比較する。
- 可視/遮蔽、遮蔽面積、目・鼻・口別の精度に加え、選択スケール・参照距離・禁止領域へのattention massを確認する。
- 合成メモリの単体試験では、集約前のX_iだけを変えた際にM_i用の許可メモリが変化しないことを確認する。
  画像画素からDINOを通す試験とは区別し、DINO内部の大域文脈を完全排除したと解釈しない。
- キーポイント単独損失から新規参照モジュールとcrop後DINOへの勾配、無効時の既存経路、全参照禁止時の有限値を検証する。
- 選択は既定validation契約で行い、testをパラメータ選択に使わない。実験時にseed・checkpoint/hash・座標系を記録する。


## 人工遮蔽と自然遮蔽の差：教師定義と汎化の検証メモ（2026-10-04）

状態: 実装を読み取って記録した未検証の仮説。追加実装・学習・評価は **not run**。
既存の学習、config、queue、checkpoint、採用判断は変更しない。
HEAD-free ordered cascade `full-image FACE/ellipse -> Face-crop DINO -> Messenger recovery -> Query Refinement` を維持する。

### 実装から確認できる事実

- 人工遮蔽は黒固定ではなく、RGB各チャネルをランダムに選び、遮蔽領域全体をその単色で塗る。
  形状はrectangle / irregular / boundary。実画像の手・髪・マスク等のテクスチャを貼る方式ではない。
- Messengerの遮蔽推定alphaの直接教師は、人工遮蔽マスクから求めるpatchごとの面積率。
  元画像に自然遮蔽があっても、人工遮蔽を追加していないpatchの教師面積率は0になる。
- 復元候補と混合後出力の教師は、人工遮蔽前の同じcropを固定教師DINOへ通した特徴。
  この教師には元画像の自然遮蔽が残るため、自然遮蔽の奥の真の顔特徴は直接与えていない。
- キーポイント・状態の下流損失も学習に使うため、自然遮蔽への対応が全く学習できないとは限らない。
  ただし、人工遮蔽率alphaと復元採用率gは別の量であり、alphaの教師0をgの強制0と解釈しない。

根拠: `src/data/landmark_occlusion.py` の `_random_fill` / `apply_landmark_occlusion_numpy`、
`keypoint_comparison1/landmark_detection/dataset.py` のpaired clean/occluded crop生成、
`keypoint_comparison1/landmark_detection/continuous_recovery.py` の `ContinuousRecoveryLoss`。

### 仮説と限界

単色領域やその境界を人工遮蔽の手掛かりとして学習し、自然遮蔽に対してはalphaや復元採用率が
十分反応せず、復元機構が役立っていない可能性がある。教師が自然遮蔽を残していることも、
人工遮蔽除去から自然遮蔽下の顔推定への汎化を制限する可能性がある。
これは性能低下の原因を確定したものではなく、反応・復元誤差・下流精度による検証が必要。
自然遮蔽を一律に除去すべきだと仮定せず、遮蔽物による遮蔽と姿勢による自己遮蔽も区別する。

### 次の確認項目

1. 同じcheckpointで、単色人工遮蔽、テクスチャを持つ人工遮蔽、実際の自然遮蔽を比較する。
   人工遮蔽同士は可能な限り同じ画像・位置・マスク・面積を使い、塗り方の差を切り分ける。
2. 遮蔽部と可視部を分け、alphaと最終採用率g、補正量、キーポイント精度を確認する。
   高いgate値だけを成功とせず、Messenger補正あり／迂回で実際の寄与を測る。
3. 人工遮蔽では元画像の教師特徴を使い、元特徴・復元候補・混合後出力の誤差を分ける。
   自然遮蔽では遮蔽なしの正解特徴がないため、復元成功を特徴誤差で断定しない。
   自然遮蔽の正確なpixel maskも既存キーポイントstateだけから得られるとは仮定しない。
4. 単色人工遮蔽にのみ反応するなら、層数や機構の追加より先に遮蔽生成と教師定義を見直す。
   9/20の1層版と2層版を共通条件で含め、生成方法だけを変える比較を候補とする。

診断・設定選択は固定train/validation内で行い、testで選択しない。
この追記は計画の記録のみであり、新たなデータ生成、評価、学習は開始していない。


## 実装引き継ぎ：αの人工遮蔽率教師を外し、貼り付け遮蔽を段階比較する（2026-10-04）

状態: ユーザーの設計案を推敲した実装・実験仕様。今回の作業はメモの記録のみ。
コード修正・学習・評価・queue起動は **not run**。この節を次の実装依頼の仕様として渡す。
先行する設計履歴は残すが、本実験のα教師・参照方式・実験順序は本節を優先する。
末尾の「実装前レビュー」で具体化した仕様を含めて引き継ぐ。旧実装記録は現行コードの説明であり、新仕様ではない。
代表モデルは `continuous_recovery_combined_option1`（combined）に確定。先のdirect代表案を置き換える。
ユーザーの「合成後の特徴が一番正解に近いのはcombinedである」という観測に基づき、
復元能力を持つ代表で自然遮蔽への適用を調べる。今回その順位を再測定したわけではない。
初回は7構成すべてを再学習しない。有効性を確認後に別構成への展開を検討する。

### 目的と仮説

人工遮蔽では元特徴と復元特徴の合成により特徴の質が向上する、というユーザーの観測を出発点とする。
今回その観測を再測定したわけではなく、自然遮蔽でも特徴復元が成功すると拡張解釈しない。
自然遮蔽下のキーポイント精度に結び付かない原因候補として、人工遮蔽率をαの教師にする設計を調べる。
現在の教師は二値0/1ではなく人工遮蔽のpatch面積率r。人工遮蔽を追加していない場所は、
自然遮蔽があってもr=0となるため、αを0へ押す直接損失が発生する。
これは予測αが実際に0になっていると確認した結果ではなく、教師設計から導かれる仮説。

αを人工遮蔽率に合わせる制約を外し、最終推定と特徴合成に役立つ値として学習させる。
以後のαは校正済みの物理的遮蔽率ではなく、元のpatch情報の信頼性を下げる学習値として解釈する。
自然遮蔽への反応増加自体を成功条件にはせず、座標・存在・状態の精度改善を要求する。

配備topology: HEAD-free ordered cascade
`full-image FACE/ellipse -> Face-crop DINO -> Messenger recovery -> Query Refinement`。
全画像FACE検出器とQuery Refinementの構造は維持し、外部HEADを追加しない。
比較役割: 現行local baselineはA1/A2、直接の旧方式比較は9/20実装の1層Messenger v1。
これらを新方式で置換したとは扱わない。global/reference baselineは本実験の選択基準にしない。

### 学習信号の分担

X: 入力画像のstudent DINO特徴、R: Messenger復元候補、Z: 人工遮蔽を加える前の同じcropの固定教師特徴。
Y=(1-g)X+gRをQuery Refinementへ渡す。Zと教師側はstop-gradientを維持する。

| 対象 | 残す学習信号と役割 |
|---|---|
| 復元候補R | 人工遮蔽前の教師特徴Zへの復元損失。下流からの勾配も遮断しない |
| 合成出力Y | 教師特徴Zへの復元損失と、Query Refinementの座標・存在・状態等の既存タスク損失 |
| α | 参照抑制とgを通じた復元・下流損失から更新。各層αの人工遮蔽率rへの直接回帰は使わない |
| 最終採用率g | 合成後復元・下流損失に加え、combined既存のoracle採用率教師を維持する |
| 自然遮蔽下の性能 | GTキーポイント・状態に基づく下流損失で学習・評価する。自然遮蔽の奥の正解特徴があるとは仮定しない |

combinedのαとgは別の値として、用途と学習信号を維持する。
1. 1層目のαは2層目のcross-attentionで参照先の信頼性を下げる。既存のlog(clamp(1-alpha,1e-6))バイアスを維持する。
2. 2層目のαは、元特徴・復元候補・非類似度・位置とともに局所gateと構造補正の入力になり、最終gの判断材料となる。
3. 最終混合はY=(1-g)X+gRで行う。combinedでαを直接gへ置換せず、構造補正も外さない。
各層αの人工遮蔽率回帰だけを外し、αモジュール・参照抑制・gへの入力は残す。
人工遮蔽前の教師特徴から得るgのoracle教師は別の監督なので、最初の比較では残す。
αの教師除去後も自然遮蔽でgが低い場合は、oracle教師が採用を抑えていないかを追加診断し、
g教師の除去は後続の独立比較とする。α教師とg教師を同時に外さない。
Query Refinementからの勾配経路は既に存在する。本変更は経路の新設ではなく、競合し得る直接教師の除去。

重要な限界: Zには元画像の自然遮蔽が残る。人工遮蔽のない場所でもZは計算できるが、
その奥の顔特徴の正解にはならない。現行の復元損失はw=r+beta*(1-r), beta=0.10で非人工遮蔽patchにも作用する。
最初の比較ではこの重みを維持し、αの教師除去だけを切り分ける。
自然遮蔽部分の保持を促す残余制約となる可能性は別の検討項目として残し、同時には変更しない。

### 段階A：αの直接教師だけを外す

1. 主比較は既存2層combinedと、同じcombinedのα人工遮蔽率損失だけを無効にした新条件。
   継承元configは `keypoint_comparison1/configs/continuous_recovery_combined_option1.yaml`。
   continuous_recoveryのalpha_weight=0を明示し、初層・最終層の両方について最適化への寄与を除く。
   同じ量を別の正則化・補助目的として残さない。ログ用の人工遮蔽率誤差は診断と明記すれば保持できる。
2. combinedのcandidate_weight=0.10、output_weight=0.10、adoption_weight=0.01、gram_weight=0.05、
   descriptor_weight=0.10、consistency_weight=0.01と既存タスク損失を維持する。
   2層、FFN、混合warmup、構造補正、oracle採用率教師、augmentationを同時に変更しない。
3. 人工遮蔽はpaired_occlusion.probability=0.30の単色方式を維持する。ORFormerの40%へ同時変更しない。
   rectangle/irregular/boundary、顔重なり10–40%、既存の座標変換・点の状態更新を維持する。
   旧occlusion_probability=0は別経路の無効化で、paired augmentationが有効であることを実行設定に残す。
4. 新規runとして既存と同じbaseline checkpointから開始する。既存Messenger best/lastから再開して
   教師変更を混ぜない。初期checkpoint SHA256はbaf7a6d0b1df6d1089357b2f28ffc1b2da3aadbb6239bae2cb5085d36ef016e0。
   seed42、effective batch64、DINO凍結epoch1–5/解除6、既存LR・warmupを維持する。
5. 最適化前epoch0を同じ軽量validation条件で測定し、以降の改善量を残す。
   epoch0は診断のみとし、best選択やearly-stoppingのpatienceに混入させない。
   混合warmup中は訓練時g=0なので、αへの下流勾配が抑制されることを明記する。
   勾配検査は混合有効化後を含め、初期状態での勾配0だけで失敗判定しない。

実装候補箇所: `keypoint_comparison1/landmark_detection/continuous_recovery.py` の
`ContinuousRecoveryLoss`、config/schema、gradient/gate tests、train/evaluation logging。
α教師なしを独立configで表し、従来設定・disabled時の経路・checkpoint互換性を維持する。
既存ファイルのdefaultやactive/resumed runの損失契約を黙って変更しない。

### 段階B：テクスチャ・物体の貼り付け遮蔽を実験する

この段階はユーザー要求に含まれる。段階Aの実装修正と評価を終えた後に行う別実験であり、
単色方式だけで作業完了としない。段階Aで改善しなかった場合も結果を記録し、事前に決めた比較を恣意的に変更しない。

- α教師なしcombined・単色を対照に、同じcombinedでテクスチャ貼り付け、物体切り抜き貼り付けを別条件として比較する。
  対象モデル・損失・初期checkpoint・seed・学習予算を揃え、形状とテクスチャの差を区別して記録する。
- 元cropを教師用として保存した後、student入力だけへ貼り付ける。幾何変換と点の対応は共有する。
  顔の教師画像やGT座標を別画像のものに置換しない。人工遮蔽された有効GT点の状態更新は既存契約に従う。
- 確率30%、顔重なり10–40%を初期比較で維持する。テクスチャ比較は可能なら同一maskで行う。
  物体形状を使う条件ではmask形状も変わるため、「色だけの差」とは主張しない。
- 実際の貼り付け支持領域から遮蔽maskを生成し、非矩形物体の外接矩形をmaskの代用にしない。
  最初は不透明な貼り付けを基準とする。境界ぼかし・半透明化を追加する場合は、alpha matteからの
  被覆率と点の状態更新規則を事前に定義し、追加要因として記録する。
- テクスチャ/物体素材はtrain由来または利用条件を確認した独立素材を使用し、val/test由来の素材は使わない。
  素材の入手・作成元、使用条件、ID、SHA256、貼り付け位置/サイズ/変換/seedをmanifestへ記録する。
  素材庫はまだ未選定。実装時に小さな再現可能な素材集合を確定し、無断でデータセットを差し替えない。
- 成功を「見た目が自然」だけで判断しない。人工遮蔽での復元と、無加工validationの自然遮蔽指標の両方を確認する。

### 段階C：Messenger参照のmulti-scale・multi-levelを独立に実装・比較する

ユーザー追加要求（2026-10-04）: 10/02メモのmulti-scale参照を、検討候補から実装対象へ変更する。
今回の作業は仕様追記のみで、実装・学習は **not run**。過去メモの「実装する決定ではない」は当時の履歴。
動的参照範囲やdeformable samplingまで同時に追加する要求とは解釈せず、それらは別候補のまま残す。

- 代表はα直接教師なしのcombined。DINOの画像patch size=16、顔crop256×256、
  Messenger数16×16、最終特徴16×16×384とQuery Refinementの入力契約を維持する。
- multi-scale有効時に変更するのはMessengerが参照するregular DINO特徴の空間スケール。
  元の16×16特徴に、非重複2×2平均poolingの8×8、4×4平均poolingの4×4を加えた3スケールを用意する。
  追加のDINO forwardや高解像度画像patch化ではなく、元特徴を中域・広域へ集約する方式。
- multi-scale有効時、各Messengerは3スケールのregularメモリをKey/Valueとして利用し、位置・スケールを識別できるようにする。
  スケールのsoftな混合で参照させる。具体的な混合・位置表現はconfigと実装記録に明記し、
  新たな動的範囲制御・Messengerの層数変更・Messenger同士のself-attentionを同時に導入しない。
- 自己patch除外を全スケールで守る。対象patchが集約元に含まれる粗いセルは、セル全体をsoftmax前にmaskする。
  中心座標だけで判定しない。集約元領域を追跡し、対象patchの情報が粗い参照メモリから直接戻らないようにする。
  DINO内部のself-attentionによる間接混入まで除去できるとは主張しない。
- αによる参照抑制を粗いスケールにも適用する。細粒度αから粗いセルの信頼性への集約規則を実装時に明記し、
  人工遮蔽maskを推論時の信頼性に使わない。最終αからgへの入力とcombinedの既存損失は維持する。
- multi-scaleは独立設定で有効化する。無効時の経路・checkpoint契約を維持し、過去/稼働中runを変更しない。

段階A→B→Cとして比較を整理する。追加合意（2026-10-04）として、段階Cを以下の2×2比較へ拡張する。
DINO ViT-S/16の中間層は最終層と同じ16×16であり、層を取り出すだけでは8×8・4×4にならない。
空間的な集約範囲（multi-scale）と特徴の深さ（multi-level）を別要因として検証する。
現行attentionも顔crop全域を参照するため、「参照距離の拡大」の効果とは呼ばない。

| 条件 | 参照するDINO層 | 各層の空間解像度 | 目的 |
| --- | --- | --- | --- |
| C0：対照 | 最終層のみ | 16×16 | α教師なしcombinedの基準 |
| C1：multi-scaleのみ | 最終層のみ | 16×16・8×8・4×4 | 空間的な集約の効果 |
| C2：multi-levelのみ | 中間2層＋最終層 | 各層16×16 | 深さの異なる特徴の効果 |
| C3：両方 | C2と同じ3層 | 各層16×16・8×8・4×4 | 両要因の併用・相互作用 |

- 中間層は0始まりblock index [3, 7, 11]（第4・8・12層）に固定する。C0/C1は最終index11のみ。
  ローカルDINOv3 ViT-S/16は12ブロック・384次元であることを確認済み。C2/C3で同じ層を使用し、
  実装のschemaで深さ・index・重複・順序を検証する。test結果で層を選ばない。
- 1回のstudent DINO forwardで必要な層を取得する。各層に位置・level識別、必要な正規化/射影を設け、
  C3ではさらにscaleを識別する。最終層の復元対象X、教師の最終層特徴、出力16×16×384は維持し、
  中間層への新しい復元損失は同時に追加しない。変更対象はMessengerの参照メモリ。
- multi-scaleとmulti-levelを独立設定にし、両方無効なら既存の経路・checkpoint契約を保つ。
  自己patch除外は全levelと全scaleに適用する。αによる参照抑制も位置対応と集約規則を明記して適用する。
  DINO内部のattentionによる間接混入は除去できるとは主張しない。
- C0/C1/C2/C3で初期checkpoint、seed、学習予算、Messenger層数、損失、教師、Query Refinement、
  augmentation（単色30%）、有効batch、評価契約を揃える。段階Aの結果をC0へ再利用する場合も
  config・checkpoint・評価契約の一致を検証する。貼り付け遮蔽をこの4条件の一部だけに加えない。
- C1−C0/C3−C2でscaleの効果、C2−C0/C3−C1でlevelの効果、
  C3−C1−C2＋C0で相互作用を確認する。同一seedの差は初期的な結果として扱い、僅差を断定しない。
  参照token数、追加parameter数、計算時間、VRAMも併記し、性能差を情報内容だけの因果と断定しない。
- 検証には全level/scaleのshape・座標・自己patch除外、層選択の正しさ、参照経路の下流勾配、
  disabled互換性を含める。level別/scale別の参照重み、人工遮蔽の特徴復元と無加工valの自然遮蔽精度を記録する。
  学習開始は既定gateとmicrobatch probeの成功後。本追記では実装・学習はnot run。
貼り付け遮蔽との併用は追加比較として分離し、一度に両方を変えた結果をmulti-scale単独の効果と呼ばない。
全7構成への展開は要求せず、9/20の1層版と現行local baselineの既存比較は保持する。

検証: shape/座標/集約元mask、禁止領域のattention mass=0、全参照禁止時の有限値、disabled互換性、
DINO抽出後の各level特徴の対象patchだけを変更しても、許可された参照K/Vが変化しないこと、下流損失から新規参照経路と
crop DINOへの勾配を確認する。特徴誤差、可視/自然遮蔽キーポイント精度、スケール別attention、
計算時間・VRAMを記録する。既定4ゲート・microbatch probe・共通評価・保存容量規則を本段階にも適用する。

### 診断・比較・完了条件

- 9/20の1層Messenger、現行local baseline、既存combined、α教師なしcombinedを共通validation表へ含める。
  教師除去の効果は既存combined対新combinedで判定し、7構成全体の再学習は要求しない。
  1層v1と2層は層数以外にも差があるため、純粋な層数効果とは呼ばない。
  旧1層の未実施testを無理に実行せず、まず共通validationで比較する。
- 人工遮蔽/自然遮蔽/可視部別に各層α、最終g、補正量、分布・飽和率を診断する。
  自然遮蔽はGT state1の点位置でのpatch値等を使い、画素単位遮蔽maskがあるとは仮定しない。
  元の自然遮蔽stateと人工遮蔽後stateを診断用に区別し、点の重複計上を避ける。
- 人工遮蔽ではX/R/Yの教師特徴誤差を比較する。自然遮蔽では補正あり/迂回の下流精度差を調べる。
  キーポイント全体・可視・自然遮蔽、occlusion_yes slice、遮蔽点E2E F1を区別する。
  迂回比較は同一checkpointでの推論時の寄与であり、学習時の寄与や因果全体とは区別する。
- lossの単体試験では、復元損失のmask重みを固定/分離したうえで、α面積率の直接項がないことを検証する。
  maskを変えれば他の復元重みも変わるため、total loss全体のmask不変性を誤った要件にしない。
  合成後復元損失と下流損失それぞれからのα/gへの有限勾配を確認し、ゼロ初期化・warmupによる
  遮断を切り分ける。復元補助損失だけの勾配を下流勾配の証拠としない。
- 固定splitはtrain44groups/4237images/7129Heads、val6/530/882、test6/528/806。
  使用annotation SHA256を照合し、教師統計・素材はtrainのみ。testで設定・checkpointを選ばない。
- checkpoint選択とearly stoppingは既定conditional semantic MacroF1@NME0.10、point0.70、
  normalization=1.20*max(GT Head width,height)、毎epoch軽量validation、minimum_epoch10、
  patience5 completed validations、min_delta1e-4、最大40epoch、warmup=flat_epochs5を維持する。
  細かい診断は別の学習後評価に分離する。
- 新規architecture/configはcanonical queueへ独立登録し、schema/gpu_step/overfit/small_scaleの
  専用success markerと安全な4/8/16 microbatch probeを経てproductionへ進める。
  effective batch64を保ち、throughput/validation時間と安全条件でbatchを選ぶ。active runには触れない。
  W&Bに条件・履歴・選択計画を保存し、local last/bestと既存の容量制約を守る。
- 共通val/testは各モデルのbaseline_aligned契約に従う。自然遮蔽診断を主FACE bbox表の代用にせず、
  HEAD依存行はN/Aを維持する。hash、座標、cache provenance、seed、閾値、使用branchを記録する。
- 実装完了、gate成功、学習完了、性能改善は別々に判定する。段階A/B/Cで改善がなくても正直に記録し、
  test結果によるやり直しや自動baseline昇格は行わない。実行した各活動を開発日誌に記録する。

### 実装前レビュー：接続仕様・受入条件（2026-10-04）

確認結果: 設計メモとして実装へ引き継げる。以下は現行コードとの照合で判明した接続条件であり、
既に新方式が実装・試験済みであるという意味ではない。コード変更・モデル実行・学習はnot run。

#### 既存経路への接続

- `keypoint_comparison1/landmark_detection/model.py` は現在Messengerと
  `feature_pyramid` / `semantic_feature_levels` / `rgb_refinement`の併用を拒否し、
  forwardでもtuple特徴を拒否する。既存のhead用multi-level設定を有効化するだけでは実装できない。
  `model.continuous_recovery`配下にMessenger専用の参照設定を追加し、最終mapと参照mapを別引数で渡す。
  既存head用設定の意味や旧Messenger v1の制約を一括で緩めない。
- `src/models/dino_backbone.py::forward_intermediate_feature_maps`とローカルDINOの
  `get_intermediate_layers(n=[3,7,11], norm=True)`を利用できる。CLS/storage tokensを除き、
  各levelを[B,384,16,16]で取得する。最終index11をXとし、重複したDINO forwardを行わない。
  同一重み・同一入力のeval時、index11が通常の`x_norm_patchtokens`と数値的に一致することを検証する。
  fp32比較はatol=1e-6/rtol=1e-5を初期許容値とし、不一致を無断で大きな許容値に隠さない。
- `continuous_recovery.py::RecoveryLayer` は現在query数とkey数を同じnでreshapeし、正方形対角maskを使う。
  query数Nq=256と参照数Nmを分け、Qは[B,heads,Nq,head_dim]、K/Vは[B,heads,Nm,head_dim]、
  maskは[Nq,Nm]に拡張する。C0/C1/C2/C3のNmは順に256/336/768/1008。
  alphaを計算するoriginalと最終g/structureへの入力は引き続き最終層の256位置であり、Nmへ拡張しない。
- `continuous_teacher.py`の教師は固定baseline backboneの最終特徴だけを使う。
  `extract_backbone_feature_map`も最終mapを返す契約を維持し、中間層tupleを教師や既存headへ渡さない。
  clean側structure descriptorは現行どおり共有の学習可能headを通す。固定DINO教師と
  stop-gradientなKL教師を維持するが、clean descriptorの学習まで新たにdetachしない。

#### 参照メモリ・α・maskの確定仕様

- 各levelの正規化済みDINO mapに対し、factor1/2/4の非重複平均poolingを行う。pooling後に
  既存RecoveryLayerのtoken単位LayerNormと共有K/V射影を適用する。空間全体を混ぜる追加正規化は使わない。
- layer index昇順、pooling factor昇順、各map内row-majorの順に参照メモリを連結する。
  初回仕様は連結した全ての有効keyへの単一FP32 softmaxとする。scale/levelの寄与は対応する
  attention massの和で測定する。groupごとのtoken数の差による重みの偏りは記録し、
  group別softmax・均等化・新しい混合ネットワークはこの初回比較には同時導入しない。
- 位置は共通crop座標で表し、各セルの中心と集約元の矩形領域を保存する。level/scale識別を含む
  Kへの位置表現の具体的な符号化・初期化は実装担当の裁量事項だが、C1/C2/C3で同一方式を使い、
  config/記録に固定してgateで検証する。C0では追加経路を完全に迂回し既存数値とstate_dictを保つ。
  このadapter追加も新方式の一部であり、C0との差を空間情報または層情報だけの純粋な因果とは断定しない。
- 1層目はprevious_alphaなし。2層目ではα1[B,256]を16×16へ戻し、各参照セルの信頼性を
  `rho = mean(1-alpha1)`（そのセルの集約元位置について平均）として計算する。
  同じ空間セルのrhoは全levelで共有し、log(clamp(rho,1e-6))をkey logitへ加える。
  factor1では現行biasと一致する。αをdetachせず、人工遮蔽maskをrhoに使用しない。
- query位置iを集約元に含むkeyは、levelによらずsoftmax前に除外する。通常256queryでは
  各queryあたりC0/C1/C2/C3で1/3/3/9個が禁止される。alphaのsoft抑制とは別のhard mask。
  あるqueryで全keyが禁止された場合はattention/contextを0にし、NaNを作らない。
- 自己patch除外テストはDINO抽出後の特徴を対象とし、他の位置とprevious_alphaを固定して行う。
  画像画素の変更後にDINOを再実行して全参照が不変であることや、R/Yが対象patchに依存しないことは要求しない。
  元特徴Xとα/gには意図的な自己情報の経路があり、DINO自体にも全域attentionがあるため。

#### 実装の完了判定と未確定事項の扱い

- 新config名・architecture ID・output/W&B名をA/B/Cの条件ごとに独立させる。
  C0はAと厳密同条件なら再利用できる。同じseedだけで共通重みの初期値一致を仮定せず、
  共通モジュールを同じ初期stateから設定してhashを確認し、追加モジュールの乱数生成を分離する。
- disabled旧checkpointのstrict loadと数値一致、raw/recovered両経路、学習・validation・診断/evaluationの
  全経路が同じ参照設定を使用することを検証する。メモを読んで新しいconfig項目を書くだけでは完了としない。
- 単体検証は `.venv/bin/python -m pytest` を使用する。実DINO最終map一致、全level/scale mask、
  all-masked有限値、既存損失係数の維持、alpha_weight=0、mixed出力/下流損失からの勾配を受入条件とする。
  zero-initと混合warmupを越えた状態の勾配を調べ、候補損失だけの勾配で代用しない。
- 既定schema/gpu_step/overfit/small_scaleとbatch probeを通過して初めてproduction可能とする。
  canonical queueへ登録する際は既存稼働runを変更せず、ユーザーのAGENTS.mdのgate先行順序を満たす。
  A/B/Cは比較の論理段階であり、新jobをgate・production交互に配置する指示ではない。
- 貼り付け素材の具体的な集合は未選定だが、段階A/Cの実装を妨げない。
  段階Bのgate前に出所・利用条件・train/val/test分離・manifestを確定する。素材が未準備のまま
  Bを完了扱いにしたり、単色へ黙ってfallbackしたりしない。
- 今回確認したのは仕様とソースの整合性。性能向上・実装の正しさ・GPU安全性の確認は未実施。
  実装者は本節の受入条件と既定評価を実行し、その実測結果を別途記録する。

### 実装担当への依頼文

この節を仕様として、まずαの人工遮蔽率直接教師を無効化できる独立設定と検証を実装する。
代表はcombinedとし、αの参照抑制・最終gへの入力を残したまま、各層αの人工遮蔽率直接教師だけを外す。
単色遮蔽30%・combined既存の復元/Gram/構造/整合性損失・gのoracle教師を保った段階Aを、
既存combined、9/20の1層版、現行local baselineと比較する。7構成すべての再学習は行わない。
その修正・評価の後、α教師なしcombinedを固定し、テクスチャおよび物体貼り付けの段階Bを別実験として実施する。
さらに、3スケール参照（16×16/8×8/4×4）とDINO中間層参照（multi-level）を独立設定で実装し、段階CのC0/C1/C2/C3の4条件で比較する。
自己patch除外を粗いセルにも適用し、最終出力とQuery Refinementを維持する。動的参照範囲は今回の追加対象に含めない。
既存の活動中runや過去のcheckpoint契約は変更しない。素材未選定などの未確定事項は記録して具体化する。
今回のメモ作成自体では実装・学習を開始していない。
