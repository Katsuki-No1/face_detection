# 実顔crop遮蔽・顔重なりaugmentationの学習予定

2026-10-09。ユーザー指定の**幅48px・強度2、透明度なし、元画像側の補正禁止**を採用。
今回の作業は学習スケジュールと実装前提の登録。学習は **not run**。
実行可能な学習ジョブへの接続はまだ完了しておらず、日時予約や自動開始はしていない。
正本は`configs/training_plans/real_face_occlusion_w48_s2.yaml`。
canonical queueの`scheduled_training_plans`に登録し、実装待ちを明示する。
未実装の引数で既存trainerを走らせたり、previewをゲート成功と扱ったりしない。

## 対象と順序

ユーザーは顔重なりaugmentationの適用先として検出器・crop keypointの「両方」を指定。
Messengerを含め、以下の4実験を独立した出力・ゲートで用意する。
FORの「使う／使わない」を判断するため、同条件のFORなし対照を追加した。

| 順 | 学習 | 配備構成 | 目的 |
|---|---|---|---|
| 1 | Messenger実顔遮蔽 | full-image FACE/ellipse → crop DINO → Messenger → Query Refinement | 別の顔で隠れた対象顔の特徴復元 |
| 2 | MAL α=2顔重なり | non-cascade full-image FACE/ellipse | 顔が重なる場面での検出・楕円推定 |
| 3 | FORなし・実顔重なり対照 | full-image FACE/ellipse → crop DINO → Query Refinement | FORの効果を測る同条件の対照 |
| 4 | FOR v1あり・実顔重なり | full-image FACE/ellipse → crop DINO → conditioned Query Refinement | 別の顔へのkeypointの取り違えを抑える |

いずれも外部HEAD段なし。新しい検出器の結果をFORへ自動で組み込まず、FORは現行の固定
alpha2検出器を使用する。モデル選択やbaseline更新は今回の予定登録には含まない。

実行順は、既存の優先データ取得・推論を尊重した上で、
**共有処理の学習接続/分布確認 → 4モデルの全ゲート → 上表順の連続本学習 → 共通val/test評価**。
schema/gpu_step/overfit/small_scaleの4つの新しい成功markerを各本学習の必須依存にする。
未成功・古いhash・診断reportしかない場合はそのモデルの本学習を開始しない。
失敗したモデルと独立したモデルは継続可能。無関係な既存失敗モデルを自動再試行しない。

## FOR v1を使うか判断するablation

| 条件 | 顔重なりaugmentation | FOR v1の条件付け |
|---|---|---|
| 対照（FOR off） | あり、48px・強度2 | なし |
| 比較対象（FOR on） | 同一の合成例 | あり |

変更するのは、FOR v1の**対象Face query特徴＋予測bbox/楕円位置情報でkeypoint queryを条件付ける設計**の有無。
v1のIDは画像内の検出queryを対応付ける番号であり、人物ID分類や番号自体の学習embeddingではない。
offでも同じ顔のcropを選ぶための対応付けは保つが、query特徴と位置情報をheadへ渡さず、
FORの追加パラメータも作らない。学習済みFORの推論時スイッチを切るだけではなく、両条件を別々に学習する。

公平性を保つため、固定alpha2検出器とその閾値、対象Faceの対応付け、train/val/test、
合成前後画像・素材動画・配置・skip・crop・photometric乱数、共通重み初期値をそろえる。
両方とも**合成後の画像から得た同じ検出・crop manifest**を使う。offだけGT cropへ戻さない。
FOR追加投影をゼロ初期化し、更新前の出力一致と共通重みhash一致を検証する。
追加層の初期化でaugmentation乱数列が変わらないよう乱数系列を独立させる。
seed42、effective batch64、損失、LR、warmup/decay、最大epoch、early stop契約は共通。
両armの四ゲートはそれぞれ独立に必要。学習完了epochが異なる場合は実更新数も報告する。

採用判断は**validation**で行い、次をon−offの差分と対象件数付きで比較する。

- 全体の条件付きsemantic Macro F1（NME0.10、既存のcheckpoint選択指標）。
- 自然な顔重なりのある例でのkeypoint性能、別の顔への取り違え。
- 重なりのない例での性能低下の有無、eye/nose/mouth別の結果。
- 通常の位置ずれ・未検出と、別Faceへの取り違えを分けた内訳。
- 共通の遮蔽robustness表、推論時間・VRAM・追加パラメータ数。

顔重なりslice、取り違え判定、曖昧例/分母、許容する性能差の具体的定義は、既存の議論で
未確定なので**両本学習前に固定する前提**を追加した。別Faceの点に近いだけで取り違えと断定しない。
合成例の見た目やtestで採用を決めない。validationで重なり性能・通常例の退行・計算負荷を
まとめてユーザーが採用を判断できる表を出し、優劣が不明なら不明のまま報告する。
まずseed42の対応比較であり、1seedから再現性を断定しない。

この2条件は**FOR設計全体の追加効果**を測る。augmentation自体の効果、ID単体の効果、
query特徴と幾何情報それぞれの寄与を分離する実験は今回の予定には含めない。
比較先は`outputs/face_overlap_robustness/ablation_real_face_w48_s2/val/comparison_table.md`。

## 共通の合成条件

- 素材/適用先は固定canonical trainのみ。別source_video_idから、周囲のLab色・輝度が近い素材を選ぶ。
- 適用先の元ラベルは輪郭遮蔽false・5点すべてvisible。これは注釈上の無遮蔽であり全画素の保証ではない。
- 素材側の顔は切れてよい。cropを画像範囲で切り詰め、対象keypointがcrop外ならstate=0。
  実不透明領域で隠れたcrop内の点はstate=1。素材の点を対象顔の正解へ混ぜない。
- 貼付境界の元画像外側RGBにcrop側だけを合わせる。48pxは256px級crop上の座標単位。
  full-imageへ展開する場合は同じ見た目になるようスケール換算する。
- 背景を含む貼付矩形全体を二値遮蔽maskとして扱い、素材の顔も対象顔へ実際に重ねる。
  元画像側は画素不変。透明度合成や隠された元画像画素の利用は禁止。
- 暫定の初期要求率は1画像最大1対象・30%。既存の人工遮蔽とは排他的に選び、二重に重ねない。
  小面積/中/強の要求比20/40/40とphotometric条件は試作を引き継ぐ初期案で、実loaderで確認する。

## Messengerの限界と今後の作業

今回の追加augmentationが作れるのは**顔で遮蔽された状況**。
自然遮蔽を完全に再現したものではなく、自然遮蔽全般に強くなったという結論も出せない。
今後、**顔以外の物体による遮蔽**を別familyとして追加する必要がある。
既存の手続き的glasses/leaf/bottle実験は別条件として保持し、今回の実顔合成だけで
物体の形状・材質・透明性・照明などを十分再現したとは扱わない。
この限界はMessenger実験のconfig/結果/W&B条件にも引き継ぐ。

clean teacherと合成studentは同じ対象顔・同じ幾何変換を共有し、mask、点のmissing状態、
復元損失の対象範囲を一致させる。合成後の特徴を教師と取り違えない。

## 顔の重なり側への接続

顔重なりの見た目を作る**合成操作自体は共通利用する**。教師と入力の対応はモデルごとに接続する。

検出器ではfull-imageへ合成し、貼り付けた素材Faceにもbbox/楕円教師を付ける。
影響を受ける既存Faceを全て追跡し、既遮蔽の適用対象外Faceに重なる配置を避ける。
素材が部分的に切れる場合の検出/楕円教師とignore扱いをschema前に固定する。
素材Faceを未注釈のままnegativeとして学習させない。

FORは**合成 → frozen検出器のquery特徴抽出 → 対象Face対応付け → crop学習**の順。
合成前のcached queryを合成後cropへ渡すと、隠す前の情報を使えるため禁止。
full-image特徴とcropが同じ合成画像に由来することをhash/geometryで認証する。
素材のqueryが対象Faceのqueryへ取り違えられた例を無条件に教師化しない。

## 学習前に終える作業

1. 共有のdonor選択・合成を本番loaderへ接続し、CPU試作依存を整理する。
2. Messengerのteacher/mask、検出器の追加Face教師、FORの合成後queryを各々接続する。
3. 実際のcrop jitter/resize/photometric/flipでseed42/43、epoch1/12/40の分布を監査する。
   軽い例の比率、強い例の比率、素材動画/profile/経路/keypoint数別のskipを報告する。
   初期目標はskip<=5%、適用例の小面積<=30%、強度>=25%。
   現CPU試作は240要求中44skipであり、この実行条件の成功ゲートではない。
4. 実checkpointのSHA、独立config/output/W&B条件、四ゲートと共通val/testジョブを認証して実行可能にする。

keypoint系はeffective batch64、最大40epoch、毎epoch条件付きsemantic Macro F1 NME0.10で選択。
検出器はeffective batch16、MAL α=2・FACE/ellipse専用のvalidation契約を維持し、
追加fine-tuneのepoch/LR予算は実行config化の段階で固定する。keypointの選択指標を流用しない。
全モデルで4/8/16の同期時間測定・複数更新・本番validationを含むpreflightとruntime OOM fallback。
testを閾値/設定/モデル選択に使わず、各モデル共通のbaseline-aligned表を残す。

ユーザーの採用条件と将来の学習結果は区別する。現在はスケジュール登録のみで、精度結果はない。
