# Messenger連続特徴復元：実装記録

## 2026-10-04: α教師除去・貼り付け遮蔽・参照multi-scale/multi-level

最新仕様は設計文書の「実装引き継ぎ」「実装前レビュー」。以下は今回の実装と実測を区別した記録。
配備は **HEAD-free ordered cascade**:
`full-image FACE/ellipse -> Face-crop DINO -> Messenger recovery -> Query Refinement`。
外部HEADは追加しない。local baseline=A1/A2、global/reference=配布モデルの役割を維持する。

### 実験の固定条件

`keypoint_comparison1/configs/continuous_recovery_abc_{arm}.yaml`を新規登録。
全条件ともcombined継承、alpha_weight=0。candidate/output=0.10、adoption=0.01、Gram=0.05、
descriptor=0.10、consistency=0.01、beta=0.10、oracle g教師・構造補正・2層・FFN・混合warmup2を維持する。
αの面積率誤差はdetachした診断値で、最適化への寄与はない。α自体、2層目参照の信頼性抑制、gへの入力は残す。

| arm | 段階 | 素材 | 参照block index | pooling factor | K/V token数 | query毎の禁止key数 |
|---|---|---|---|---|---:|---:|
| a | A=C0 | 単色 | 11 | 1 | 256 | 1 |
| b_texture | B | テクスチャ | 11 | 1 | 256 | 1 |
| b_object | B | 物体形状切り抜き | 11 | 1 | 256 | 1 |
| c1 | C1 | 単色 | 11 | 1,2,4 | 336 | 3 |
| c2 | C2 | 単色 | 3,7,11 | 1 | 768 | 3 |
| c3 | C3 | 単色 | 3,7,11 | 1,2,4 | 1008 | 9 |

初期checkpoint SHA256は`baf7a6d0b1df6d1089357b2f28ffc1b2da3aadbb6239bae2cb5085d36ef016e0`。
seed42、effective batch64、DINO凍結epoch1–5/解除6、LR backbone=1.25e-5/head=5e-4/recovery=1e-4、
最大40epoch、毎epoch軽量validation、conditional semantic MacroF1@NME0.10最大化、point0.70、
Head正規化1.20*max(w,h)、minimum10/patience5/min_delta1e-4、warmup=flat5を維持。
epoch0は別JSONとW&B診断だけに記録し、best/early-stop履歴に入れない。各validationにepoch0との差を残す。
既存Messengerからの再開は行わず、別出力先から開始。再開時はarm・素材・recovery契約も照合する。

共通モジュールの初期状態は6条件すべて実測で
`69728f8b4669b74f490d4b37163b21996f872cf46e53269340cd16f51dc1fea6`。
追加adapterは独立したCPU RNG context/seed20261004で生成し、既存headの初期化乱数を消費しない。
本学習開始前にも共通hashを照合し、出力とW&Bに保存する。

### 参照実装

`recovery_reference.py`は正規化済みDINO mapを非重複平均poolし、block昇順→factor昇順→row-majorで連結。
token単位LayerNormと共有K/V射影の後、Kへ8次元の座標/level/scale情報の線形射影を加える。
8次元は `(center_x,center_y,x0,y0,x1,y1,block/11,log2(factor)/2)`。
座標は共通cropの[0,1]、x右/y下、矩形は半開区間。adapterの重みはnormal std0.02、bias0。
方式IDは`crop_rect_level_scale_linear_v1`。C0はadapterもメモリ再構成も完全に迂回しstate_dictを維持。

Nq=256、Nmは表の通り。1回のstudent DINO forwardから必要な正規化済み層を取得し、index11だけをXとして復元する。
教師と`extract_backbone_feature_map`は最終mapの契約を維持する。出力384×16×16とQuery Refinementは変更しない。
1層目はα biasなし。2層目は各参照セルの集約元に対して`rho=mean(1-alpha1)`、
`log(clamp(rho,1e-6))`を加える。同じセルのrhoを全levelで共有し、αをdetachしない。
自己patchを集約元に含むセルは全level/scaleでsoftmax前に禁止。全key禁止時はattention/context=0。
全有効keyに単一FP32 softmaxを使い、group別softmaxやtoken数均等化は導入しない。
groupのtoken数、attention mass、追加parameter数を記録し、情報の種類だけの純粋な因果とは解釈しない。

### 段階Bの素材と分離

素材は独自生成した4テクスチャ（checker/fabric_stripes/grain/noise）と3切り抜き形状（leaf/bottle/glasses）。
独立した手続き生成で、train/val/testの画像・アノテーションはいずれも素材作成に使わない。
**実写物体コーパスではなく、自然物体の写実性を検証したとは扱わない。**
生成seed=20261004。生成コード・利用条件・ID・各PNGのSHA256は
`outputs/dataset_adapters/recovery_abc_materials_v1/manifest.json`に固定。
manifest SHA256=`2972f0f3036761f9863a02f05485789c9a64b0568475d1a5d7217c8392daeafd`。
再生成は`.venv/bin/python scripts/prepare_recovery_occlusion_materials.py`。素材と生成物はコミットしない。

確率30%/顔重なり10–40%。clean crop保存後にstudentだけへ貼り付け、座標・共有flip・状態更新を維持。
テクスチャは単色条件とmask乱数を共有。物体はbinary alphaの実支持領域を使用し、外接矩形をmaskにしない。
物体条件には形状差も含まれる。素材ID/hash、貼り付け矩形、90度回転/反転、seed、支持領域hash、
共有後段flipをサンプルmetadataへ記録。元の自然遮蔽stateを別保持し、診断で人工遮蔽点を二重計上しない。
素材欠落/改変や配置失敗はエラーにし、単色へfallbackしない。
各schemaで実train先頭96サンプルを固定seedで検査し、各条件20適用例をmanifestへ保存した。

### 実行済み検証と残るゲート

- 関連単体・回帰123件passed。`outputs/training_gates/recovery_abc_shared/pytest.xml`。
- 実DINO6条件passed。最終map一致atol1e-6/rtol1e-5（実測最大差0）、1forward、全層shape、
  disabled baseline checkpoint strict load/raw出力完全一致、C0/旧combinedのstrict load/出力完全一致、
  共通初期hash一致、C2/C3の最終座標損失だけからα/KV/adapter/crop DINOへの有限非ゼロ勾配を確認。
  `outputs/training_gates/recovery_abc_shared/real_dino_acceptance.json`。
- B_object/C3の実データCPU smokeを各3epoch実行。train4/val4、microbatch2/accum1、
  teacher scale_samples4、backbone凍結0という隔離した検証override。混合warmup2は維持しepoch3を確認。
  lossはB_object 0.3264→0.2642、C3 0.3689→0.2746、4件のconditional F1は双方0.8457→0.9029。
  **収束や精度改善の証拠ではない。GPU/overfit/small_scale合格の代用にしない。** W&Bなし。
  各`*_cpu_smoke/query_refinement`にconfig/epoch0/metrics/teacher provenanceを保存し、best1個だけ保持。
- 診断CLIは既存combined checkpointとval2件でraw/recovered・固定教師・3人工遮蔽条件をCPU smoke確認。
  `outputs/training_gates/recovery_abc_shared/diagnostic_smoke.json`。学習後の全体診断は未実行。
- 各6条件のschemaはpassed。`outputs/training_gates/deimv2_l_query_refinement_continuous_abc_{arm}_seed42/schema/success.json`。
- **gpu_step（4/8/16 probe）、overfit、small_scale、本学習、ABC本評価はnot run／Co-DINO終了待ち。**
  Co-DINOがGPUを使用中のため同時実行しない。成功マーカーは未作成。

### 後続実行と評価

canonical queueへ24 gate、6 production、共通val/testと診断/reportを登録。
物理順序は全gate→連続production→評価。既存training_setsは維持し、新しい6条件だけ
`gate_first_production_groups`で全ゲート先行・本学習連続・評価後続にする。
全productionは4成功markerと`codino_official_sfp_ellipse_test_evaluate`に明示依存。
起動CLI単独でもgateの設定/実装hashとCo-DINO評価のidentity/inventory/成果物hashを確認する。
既存稼働runnerは変更・再起動しない。

`scripts/run_recovery_abc_after_codino.py`の独立watcherが既存runner終了を待ち、Co-DINO評価成功を照合して、
canonical queueを`--stop-after-job recovery_abc_comparison_report`まで引き継ぐ。
状態は`outputs/training_queues/recovery_abc_after_codino/status.json`、実行ログは同ディレクトリ。
失敗/未完了を成功扱いせず、依存skipと実測結果を保存する。
2026-10-04にwatcher PID1387759を起動し、既存Co-DINO runner PID1375251の終了待ちを確認した。
このチャットのheartbeat `messenger-abc-co-dino`も毎時の後続確認として登録。
既存のCo-DINO監視automationは変更しない。通常の待機/稼働中は通知・追加GPU実行を行わず、
gate不合格を条件緩和で回避しない。

各armの共通評価は`evaluation/{val,test}/baseline_aligned`。local FACE bbox主表とGT Head診断を分離し、
HEAD依存行はN/A。valで閾値固定、testは設定選択に使わない。
`evaluate_recovery_abc_diagnostics.py`は同じmodel forwardを通して各層α/g/構造補正の分布、
人工X/R/Y固定教師誤差、自然点のraw/recovered、group attention massを別診断へ出力する。
`report_recovery_abc.py`は既存combined・9/20 v1・local baseline・6条件のvalidationを比較し、
事前に定めたC1−C0/C3−C2/C2−C0/C3−C1および交互作用C3−C1−C2+C0を計算する。
単一seedの差であり、自動baseline昇格は行わない。旧v1のtestは新たに要求しない。
旧v1/local baselineの保存済み共通レポートにはFACE bbox主表がなかったため、実在する各検出cacheを
同じmatcherとcap100で再集計する。旧成果物は上書きせず、report側に出所/hashと再集計結果を保存する。
今回のCPU再集計では双方、Face0.33でTP762/FP85/FN72、validation選択閾値0.332763671875で
TP762/FP84/FN72だった。これは同じ検出器の結果で、復元枝の改善とは扱わない。
旧方式のHead基準選択閾値との混在を避け、旧/新の属性比較は共通Face0.33・point0.70を使用する。

## 2026-09-30: 初回実装・ablation登録（以下は当時の履歴）

対象は合成方法の案1と案3。案2（事前学習済み姿勢モデル）は未実装・未登録。
本学習、実データGPU preflight、overfit/small-scale gate、性能評価は **not run**。
CPU単体・回帰テストと、実DINOv3/checkpointを使う合成入力の勾配監査のみ実行した。

配備topologyはHEAD-free ordered cascade:
`full-image FACE/ellipse -> Face-crop DINO -> continuous recovery -> existing Query Refinement`。
HEADを追加せず、Query Refinementの層・次元・refinement stageは継承元と同一。
ローカルbaselineは`deimv2_l_query_refinement_landmark_occlusion_a1_a2`。
グローバルbaselineは`reference_model/face_dino_v2_distribution`の固定レポートであり、置換しない。

## 独立設定と比較順

`model.continuous_recovery.enabled=false`または設定なしでは、従来のモデル経路と
state_dictを維持する。既存1層`model.messenger_recovery`とは排他。
新規パラメータは`messenger_recovery.*`に格納し、そのprefixだけをwarm-startの不足キーとして許可。

| 順序・config suffix | 採用率 | Gram | 構造補正 | clean/occ整合性 | 主な比較目的 |
|---|---|---:|---:|---:|---|
| direct | alpha | 0 | 無 | 0 | 2層・直接候補/出力復元の基準 |
| gram | alpha | 0.05 | 無 | 0 | directとの差でGramの追加効果 |
| option1 | optimal教師付きlocal gate | 0 | 無 | 0 | 案1単独 |
| local | local gate | 0 | 無 | 0 | oracle教師なしの案3対照 |
| option3 | local gate + structure logit | 0 | 有 | 0 | localとの差で構造補正のみ |
| option3_consistency | local gate + structure logit | 0 | 有 | 0.01 | option3との差で整合性のみ |
| combined_option1 | optimal教師付きlocal gate + structure logit | 0.05 | 有 | 0.01 | 最後の統合。案1を採用率教師に使用 |

configは`keypoint_comparison1/configs/continuous_recovery_{suffix}.yaml`。
共通値は`continuous_recovery_base.yaml`。既存Messenger matched controlも比較対象として維持する。
統合では案3の構造補正も残し、最終gへの教師を案1にする。案3単独はoracle教師を使わず、
局所gate対照と教師条件を揃える。全構成の学習予算・データ・初期checkpointを揃える。

## 初期値と判断根拠

数値は未検証の初期値であり、testから選んだ値ではない。

| 項目 | 初期値 | 根拠・扱い |
|---|---|---|
| Messenger | 2層、6 heads、FFN比2、dropoutなし | 設計の2段更新。C=384に整合し、FFN比は追加メモリを抑制 |
| 位置・初期状態 | sine 2D、学習token std=0.02 | 特徴の空間格子を保ち、画像内容を直接query初期値に混ぜない |
| residual初期化 | 各層attention出力/FFN最終層をzero | 初期変動を抑え、候補損失で上流を開く |
| Attention | FP32 softmax、自己patch除外、2層目bias=log(clamp(1-alpha1,1e-6)) | 半精度安定性と前段の遮蔽推定による参照抑制 |
| 1 patch退化 | attention全0 | 空softmaxを作らない。通常cropは16×16 patches |
| 復元decoder | LN -> Linear(C,2C) -> GELU -> Linear(2C,C) | raw DINO座標系で完全候補を出す軽量MLP |
| alpha/gの初期値 | 0.10 | 最初から強い置換を避ける。gとalphaの教師は別 |
| 混合warmup | epoch 1–2は訓練時のみg=0、3以降有効 | 候補decoderを先に更新。推論・validationは常に学習gateを使用 |
| 非遮蔽の重みbeta | 0.10 | 全patchを監督しつつ人工遮蔽を重視。マスクなしsampleも除外しない |
| cosine/振幅比 | 1 / 1 | 方向だけ合わせて振幅誤差を消すことを避ける |
| Huber delta | 1.0 | 固定教師の標準偏差で規格化した誤差1を境にする。torch Huberを使用しSmoothL1と混同しない |
| teacher scale | trainからseed42で256 crop、全patchの母標準偏差、floor=0.05 | boundedな統計計算。float64集計、弱いchannelでの過大勾配を抑える |
| oracle小分母 | mean(((R-X)/s)^2) <= 1e-6ならg*=0 | 候補が原特徴とほぼ同じ時の不安定教師を除く |
| oracle座標 | X/R/Zを固定teacherのchannel scaleで割る | 高分散channelだけが採用教師を支配するのを避ける。教師全体をdetach |
| loss係数 | candidate=.10、output=.10、alpha=.01、alpha1比=.5、oracle=.01 | taskを主とし、候補/実出力と両段alphaに直接監督 |
| Gram係数 | 0または.05 | N×Nのfull patch Gramを独立ablation。行L2正規化、FP32、batch平均 |
| 構造Attention | 1層、96次元、4 heads、dropoutなし | 追加負荷を抑え、顔全体を単一queryで集約 |
| prototypes | K=8、96次元、cosine soft assignment、tau=.20 | 少数データで過細分化を避けつつ複数の構造状態を許容 |
| 構造logit補正 | 最終Linear weight/bias=0 | 初期状態をlocal gateと完全一致させる |
| descriptor | 全10ペア、Huber delta=1、係数.10 | visible/occluded両端state>0の相対配置。空ペアsampleは平均から除外 |
| 整合性 | 初期案3は0、追加/統合は.01 | collapseを単独目的で誘発しないよう独立比較 |
| 整合性の詳細 | KL(sg(pi_clean) || pi_occ)、w=(1-平均遮蔽率)^2 | 強い遮蔽ほど要求を弱める。EMAなし。共有structure headのclean側もdescriptor監督し、clean/occ損失を等平均 |
| 均等化正則化 | なし | 実際の構造頻度を強制的に均等化しない |

潜在モードはyaw/pitch/rollや手動の姿勢クラスとは呼ばず、角度・表情・個人差を含み得る表現として扱う。

## 学習・教師・座標

全構成はseed42。crop DINOはepoch1–5凍結、6以降共同学習。
LRはDINO=1.25e-5、既存Query Refinement=5e-4、新規モジュール=1e-4。
headは凍結しない。warmup=flat_epochs=5の後ただちにdecay。
最大40 epoch、毎epoch軽量validation。選択指標は既定の
conditional semantic Macro F1@NME0.10（maximize）、point=0.70、minimum_epoch=10、
patience=5 completed validations、min_delta=1e-4。testは選択に使わない。

初期checkpoint:
`outputs/landmark_detection/dinov3_vits16_face_pose_drive_v2_balanced_811_gt_ellipse_long_axis_square_coco_visibility_landmark_occlusion/query_refinement/checkpoints/best_nme.pt`

SHA-256: `baf7a6d0b1df6d1089357b2f28ffc1b2da3aadbb6239bae2cb5085d36ef016e0`。
clean教師はこのcheckpointのDINOを固定し、studentの更新に追従するEMA教師は今回採用しない。
studentは常に画像からforwardし、特徴cache/detachを使わない。
教師統計のsample IDs、channel値、annotation/checkpoint hash、seedを
`query_refinement/continuous_teacher_provenance.json`とW&B configへ保存する。
教師とstudentの特徴座標系のずれは候補/出力のcosine・振幅監督で抑える設計であり、改善保証はない。

train教師・構造教師にvalidation/testを使わない。既定train adapterとcanonical trainの
file_name・正規化ellipse（小数4桁）で同一Faceに属するGT Headを結ぶ。
現データでは6,529 Faceのうち6,447が有効なcanonical GT Headに一致、82は対応なしとして
descriptor対象外。Face bboxをHeadサイズの代用にしない。除外数・理由をprovenanceへ記録。
全10ペアはcropのx右/y下座標で、inverse affineの等方scaleにより画像pixel単位へ戻し、
`1.20*max(GT Head width,height)`で割る。反転時はx反転とslot `[1,0,2,4,3]`交換を共有する。
GT Headは損失の分母だけで、推論入力ではない。

## スケジュールと安全条件

canonical queueの実行対象に7構成を追加。MAL validationの
`validation_selection_success.json`を全新規productionの必須依存にした。
ユーザーの「MALのvalidation完了後」を反映し、MAL test/reportは最後へ明示的に延期。
物理YAMLは全gate→production→evaluation順を保ち、既存training_sets plannerが
validation依存を解決する。各構成はgate→本学習→val/test、統合構成が最後。
structure有効3構成にはvalの補正zero・別画像zへの置換診断も登録。

schema/gpu_step/overfit/small_scaleは各構成専用success.jsonを必須とし、失敗reportを成功扱いしない。
microbatch 4/8/16、effective batch64を調べる。probe専用configで凍結と混合warmupだけを0にして
共同学習のメモリ・勾配経路を実行する。本学習configは書き換えない。
128 train/64 val、2epoch・複数optimizer step、最初のepochをtiming warmupとして除外。
同じsample/augmentationで比較し、VRAM<=90%、温度<83℃、OOM/non-finiteなしを必須にする。
実際のdataset件数から投影時間を計算し、3%以内は小さいbatchを選ぶ。
選択前のbatch8/accum8は仮値。runtime OOMではmicrobatchを半分にしてeffective batchを保ち、
lastから再開する。試行・選択はJSONとW&Bに記録。
productionは毎epoch last、validation改善時bestのみ。新構成のcheckpoint_every=0を実装し、
既存runの保存間隔は変更しない。ゲート成功時は既存retentionでbestだけを残す。

## 評価と検証

共通val/test出力は各構成の`query_refinement/evaluation/{val,test}/baseline_aligned/`。
canonical annotationの44/6/6 group、4,237/530/528 image、7,129/882/806 Headと
SHA-256を固定。新構成の主FACE表は同じ`face_bbox_iou50.threshold_metrics`を使用する。
GT FACE bboxはcanonicalな有効GT ellipseのaxis-aligned外接矩形を画像境界でclipして定義する。
全未対応predictionをFPにし、重複数も残す。Face0.33と、cacheに存在するscore境界の中で
precision>0.90を満たすrecall最大のvalidation固定点を別々に報告する。
検索下限は既存full-image detectorの0.33、prediction capは100で、testで再選択しない。
7構成はfull-image detectorがbaselineと同一であることをconfig/checkpoint hashで確認し、
同一prediction cacheを同一matcherで再集計するため、FACE bboxのlocal baseline差は0になる。
この同一性は変更されるキーポイント枝には主張しない。従来Head基準表・reference差分は診断として分離し、
HEAD/matched-HEAD行は既存HEAD-free unsupported/N/A契約を維持する。

検証成果物: `outputs/implementation_checks/continuous_recovery_20260930/`。
`real_dino_gradients.json`は案1・案3・統合について、初期状態/補助4step後/凍結解除後の
**キーポイント座標損失単独**の勾配L1、有限値、DINO更新、optimizer重複/漏れなしを記録。
実DINOv3と実baseline checkpoint、CPU、seed42、合成2×3×64×64入力を使用。
descriptor専用decoderはキーポイントforwardに含まれず、descriptor損失で検査する。
初期構造logit最終層までは勾配が届き、初期prototype側は0、更新後は非ゼロになる。
teacher/oracleの勾配停止、空ペア、singleton attention、CPU AMP forward、旧checkpoint互換性、
teacher provenance・再開契約・保存retentionを単体テストする。
CPUのoneDNNはbf16 backward非対応のため、そのbackward検査はFP32。
本番CUDA AMP/VRAM/温度/収束は未実行のGPU gateで確認する。

schemaを実装検査ディレクトリで実行した。production用ゲート成功マーカーは作成していない。
キュー起動・本学習・モデル昇格は実行していない。

最終回帰: 9 suite / **131 passed**（`pytest.xml`）。その後の評価provenance追記について
評価・キュー関連の29件も再検証してpassed。実DINO勾配監査は3構成ともpassed。
queue dry planningは137 job、そのうち新規productionは7、実行はしていない。
