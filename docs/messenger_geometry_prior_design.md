# Messengerの楕円条件付き事前分布と復元特徴融合

## 簡潔な説明に使う要約

非遮蔽train顔の空間RoI特徴をk-meansでk＝1〜7に分け、向き・状態による複数の顔構造を事前知識にする。予測FACE楕円のスケール・長短軸比・角度から参照中心を選び、位置ごとの相関と特徴差から遮蔽候補を求める。連続領域の面積5%を顔単位判定の初期候補とし、局所の異常部分だけMessengerの復元候補へ学習可能な割合で融合する。まず既存Messengerとの比較を行い、その後に同じ条件で敵対的損失の有無を比較する。キーポイント損失は生成器と融合・headへ流し、識別器は別の真偽判別損失で学習する。

2026-10-10のユーザー依頼は、この設計と意図の記録、および敵対的学習までの実装。実データの学習・評価・起動・モデル昇格は今回依頼の実行範囲に含めない。以下の数値は実装上の初期値であり、実測で採用された値ではない。

## 意図と比較順

一つの平均顔では向き・状態の違いを吸収できない可能性がある。複数中心を参照すると、横顔の自然な構造を遮蔽と取り違える誤りを減らせるか検証する。k＝1は単一中心の対照、k＝2〜7は複数中心の候補とする。k-meansが自動的に姿勢だけを表すとは断定しない。人物、照明、画質による分類の可能性も点検する。

第1段階は既存Messengerの条件を揃えた対照と、k＝1〜7の事前分布＋遮蔽判定＋特徴融合を比較する。敵対的損失は無効。第2段階ではvalidationで選択した同じk・閾値・学習条件について敵対的損失の有無だけを比較する。testでk、閾値、モデル、係数を選ばない。旧checkpointの保存済みスコアだけを新しい条件の対照とは扱わない。

## 推論構成と実装範囲

HEAD-free ordered cascade:

`full-image FACE/ellipse → Face-crop DINO → Messenger復元と事前分布による融合 → Query Refinement`

遮蔽判定用に、同じcropへ固定DINOを適用する補助分岐を持つ。これはHEAD段ではない。事前分布を作った特徴座標系を保持し、student DINOの学習で中心が古くなる問題を避けるための実装判断。DINOを二度通す時間・メモリの追加コストを将来の実測に含める。識別器は学習時だけ使用する。

`geometry_prior_recovery.py`は既存コードから自動importされない独立モジュール。既存モデルを`attach_geometry_prior(..., enabled=True)`で明示的に包む。無効時は元モデルそのものを返し、旧state_dictと推論経路を変えない。現在の対応は最終DINO特徴・single-scaleのContinuous Messenger＋Query Refinement。multi-level、multi-scale、FORは明示的に拒否する。既存のキューの実行選択・ジョブ・checkpointは変更しない。

## 事前分布の作成と参照中心

- source-videoを保ったtrain内で、中心を作るfit群とスコア分布を作るcalibration群を分ける。両者の動画は重複させない。
- 非遮蔽資格は元アノテーションと監査で明示する。`face_occluded=false`だけでは内部の非遮蔽を保証しない。state0を可視と読み替えず、横顔の自己遮蔽・missingを扱う資格基準は実データcacheを作る前に確定する。
- 同じcrop幾何変換・同じ固定backboneで得た`N×C×H×W`特徴を使う。位置を潰すglobal averageは用いない。チャネル標準化後の空間特徴をflattenしL2正規化、seed42のk-means++と最大100回のLloyd更新で分類する。
- 各クラスタの位置別平均を保存する。相関の正常範囲は別train動画の位置別スコア分布から求める。少数クラスタ・空クラスタ・幾何ルーティングで参照されないクラスタは失敗として報告し、黙ってkを減らさない。コードの最小2件は計算上の下限であり、統計的な十分性の合格基準ではない。
- geometryは`[scale, ratio, theta]`。scaleは予測楕円の長軸直径を元画像の`√(width×height)`で除す。ratioはmajor/minor≧1。thetaは画像x軸から時計回り、ラジアン、π周期。crop変換・反転後の対応を揃える。
- 入力表現は`[log(scale), log(ratio), e sin(2θ), e cos(2θ)]`、`e=1−1/ratio`。円に近いときは角度の影響を弱める。trainで標準化した幾何空間で、特徴クラスタに対応する平均幾何へ最も近い中心を選ぶ。
- soft weightsは診断用に返すが、初期実装の参照は最近傍一つ。異なる向きの空間特徴を平均して不自然な中間構造を作らない。幾何情報だけでは左右向き等を区別できない場合があり、可視部で候補を絞る拡張は未採用。

固定観測器のcheckpoint SHAと実際のbackbone tensor state SHAを照合する。cacheにはannotation・detector・backboneのSHA、crop仕様、資格監査、動画・record ID、seedを記録する。teacherは同じ対象顔の人工遮蔽前cropであり、中心画像や別人を復元正解にはしない。

## 遮蔽候補の判定

位置p・参照中心kについて、チャネル方向Pearson相関を使う。

`score[k,p] = corr(X[p], center[k,p]) / (1 + mean_channel(abs(standardized_X[p] − standardized_center[k,p])))`

定数特徴の相関は根拠なしとして0。標準化は固定train特徴から得たチャネルscaleを使用し、初期floorは0.05。scoreは遮蔽確率ではなく、非遮蔽参照との整合度。

対象画像自身の平均を閾値にしない。非遮蔽calibration群でのクラスタ別・位置別下位分位点を使う。初期分位点は5%。これは**スコア分布の5%**で、以下の**面積5%**とは別のパラメータ。分位点・面積条件の採否はvalidationで決める。

予測FACE楕円内部のfeature cellを面積分母とする。4近傍で連結した低スコア領域のうち、一つでも分母の5%以上なら顔単位の遮蔽フラグを立てる。小さい眼部等を消さないため、局所融合マスクは顔単位フラグにかかわらず全低スコアcellを保持する。16×16全体なら5%は約13cellだが、実際は楕円内cell数を分母とする。

この判定はhardで、中心選択・閾値・連結判定には逆伝播しない。連結処理はCPUの検証用実装。自然遮蔽と、表情・照明・ブレ・人物差による低スコアの区別は未検証。

## 復元特徴の融合

`Y = X + m × g × sigmoid(a_channel) × (R − X)`

Xはstudentの観測特徴、Rは既存Messengerの復元候補、mは固定観測器からの遮蔽候補、gは既存の学習可能な採用率。チャネル別aは0初期化（初期係数0.5）。可視領域m＝0のYはXと厳密に一致する。全特徴への自由な変換は加えない。親モデルのmixing warmupスイッチを保持する。

`fusion_mode="messenger"`は同じwrapper内で元Messengerの融合結果を使う対照。候補復元・融合後復元・キーポイント損失の条件を揃えられる。`enabled=false`はwrapper自体を付けない完全な旧経路。

## 敵対的学習

生成器側はMessengerと学習するstudent/融合。識別器は空間Convネットで、候補特徴と同じ可視context・geometry・maskを受け取り、clean特徴か融合特徴かを判別する。real/fakeで同じ条件を使い、隠れたclean特徴をconditionへ漏らさない。

- Gの損失：既存LandmarkLoss＋0.1×（候補RとcleanのSmooth L1＋融合YとcleanのSmooth L1）＋任意の敵対的損失。敵対的係数の初期候補は0.01、第一段階では0。
- Dの損失：遮蔽候補cellでのlogistic real/fake。D更新時はfakeをdetachする。
- Gの敵対的損失：Dの重みを固定し、D入力への勾配だけGへ流す。G/Dのoptimizerは分離する。
- キーポイント誤差はhead→融合→Messenger/studentへ伝える。Dにはキーポイント誤差を直接流さない。
- マスクが空のbatchではDを更新しない。weight decayだけでDが変わることも避ける。
- 非遮蔽らしい特徴と、対象顔に正しい特徴は別。対となるclean復元と位置・state損失を残す。識別器の出力を推論時の品質保証とは呼ばない。

`research_train_step`はこの更新順のFP32研究用1stepを実装し、同じ既存Query Refinement/LandmarkLossでCPU検証する。productionのAMP・勾配蓄積・OOM fallback・epoch validation・W&B・保存ローテーション・四ゲートへの接続は未実施。現在の本番trainerへこの関数を置き換えてはいけない。

## 利用方法と準備物

`scripts/prepare_messenger_geometry_priors.py --help`で準備CLIを確認できる。入力は`torch.load(weights_only=True)`で読める監査済みtrain cache。キーは`features`, `geometry`, `records`, `metadata`。recordsの各要素には`record_id`, `source_video_id`, `split="train"`, `clean_eligible=true`が必要。metadataは上記provenanceと`backbone_state_sha256`を含む。

CLIへannotation・backbone checkpoint・detector checkpointの実ファイルとcalibration用train動画IDを渡す。hashを確認し、k＝1〜7のbankと`preparation_report.json`を新規出力先へ保存する。既存出力の上書きを拒否する。準備成功は訓練ゲート合格ではなく、`success.json`を作らない。実画像からこのcacheを抽出する作業と資格監査はまだ実行していない。

モデルの組み立ては次の形。既存baseは対応するcheckpointからロード済みとする。

```python
bank = GeometryPriorBank(torch.load(prior_path, weights_only=True))
model = attach_geometry_prior(
    base, bank, enabled=True,
    observation_backbone_sha256=authenticated_checkpoint_sha256,
)
output = model(crop_images, geometry=predicted_geometry, support=predicted_ellipse_support)
```

生成器optimizerにはmodelのrequires_gradなパラメータだけを登録し、Dは別optimizerに登録する。`research_train_step`へ同一cropのclean/occluded画像と既存targets/criterionを渡す。再開する実験ではmodel・G/D optimizer・D・bank provenance・fusion mode・係数・epoch/step・RNGを同一契約で保存する。本番resume経路は未接続。

## 今回の検証と次の判断

新規テストはk＝1〜7の再現性、train/calibration分離、geometry周期、人物差の正常範囲を扱うスコアの有限性、面積境界、可視部の完全保持、実backbone照合、D/G勾配分離、キーポイント単独勾配、複数step、空マスク、state_dict復元、準備CLIの拒否条件を検証する。合成入力とdummy DINOを用いるCPUコード検証であり、実モデルの精度改善・GPU安全性・production gatesの代用ではない。

2026-10-10の実行結果：下記4ファイルのCPU pytestは**52 passed**。既存TorchScript非推奨warningのみ。準備CLIの`--help`も動作確認した。

```text
keypoint_comparison1/tests/test_geometry_prior_recovery.py
keypoint_comparison1/tests/test_continuous_recovery.py
keypoint_comparison1/tests/test_messenger_recovery.py
tests/test_prepare_messenger_geometry_priors.py
```

今後の実データ検証では、非遮蔽の誤警報、人工遮蔽でのX/R/Yの特徴品質、無加工validationの自然遮蔽検知、最終conditional semantic Macro F1 NME0.10、非遮蔽・profile回帰、runtimeを確認する。既定の基本・遮蔽・横顔テーブルと失敗レビューを保存する。学習前には各候補のschema/gpu_step/overfit/small_scale、isolated batch4/8/16 preflightが必要。

今回の実データ学習・validation性能評価・test：**not run**。本番キューにはimplementation-only、enabled=falseで登録する。local/global baselineの昇格は行わない。


## 2026-10-11 training schedule

The user requested scheduling this implemented family after existing follow-up training.
The canonical queue now references `messenger_geometry_prior_adversarial_20261011`.
Run matched control and k1..7 without adversarial loss first, select k on validation,
then run the same k with adversarial loss. Each arm needs four independent authenticated
gates, production, and common val/test artifacts. Test never chooses k or GAN settings.
The exact dependency plan, output roots and training contract are in
`configs/training_plans/messenger_geometry_prior_v1.yaml`.
Production integration of the observer, fusion, two optimizers, AMP, resume and runtime
OOM fallback is still pending; this is a scheduled dependency plan, not a running job.

## 2026-10-12 train-only clean inventory

`scripts/prepare_messenger_prior_inventory.py` authenticates the parent canonical
train/val/test hashes and source-video separation, then selects explicit
contour-unoccluded Faces with five visible, finite, in-image keypoints. Missing
states or unknown contour labels never imply clean. This annotation eligibility
does not prove that every face pixel is visible and does not authorize GT crops.

The frozen inventory is
`outputs/landmark_detection/messenger_geometry_prior/preparation/clean_train_inventory_20261012.json`:
2,550 eligible records from 6,529 Face annotations. A seed42 SHA256 ranking of
eligible train videos assigns the first ceil(20%) to calibration, the remainder
to fitting: 2,036 fit records and 514 calibration records. The 20% split is a
deterministic preparation default, not selected using validation or test.

The prior-fitting CLI now requires `--inventory`; feature metadata must contain
its `clean_inventory_sha256`. Record IDs and source videos must be members of
that inventory, and `--calibration-video` must match its frozen partition.
Unmatched/crop-ineligible clean records may be absent from the feature cache and
are explicitly counted in the preparation report. They are never replaced with
GT features. The actual predicted-crop feature extraction, production trainer
integration and four gates remain pending. Verification: 47 CPU tests passed;
training and model evaluation: **not run**.

## 2026-10-12 predicted-crop feature preparation

`scripts/extract_messenger_geometry_features.py` is now registered as the
canonical setup job `messenger_geometry_prior_prepare_train_features`. It uses
the parent ABC-A detector (SHA `8fa78596fabb31e1daba7594cadd8eec48979f73563b51fed5921145d960e013`),
its fixed score0.33 and prediction cap100, and the pinned initial Query EMA
backbone (`baf7a6d0...`). These identities preserve this older matched-control
experiment; the later V3 baseline promotion does not change its initialization.

Preparation authenticates the full train image inventory, performs train-only
full-image FACE/ellipse inference, matches predicted crops one-to-one at bbox
IoU0.50, and extracts frozen float32 DINO features from deterministic 256px
ellipse-aligned square crops. Original clean eligibility is additionally checked
after cropping; cropped-out points and unmatched Faces are counted separately.
No GT crop fallback, HEAD inference, train threshold search or test inference is
used. The fixed train-video partition then fits/calibrates k1..7 independently.

The output is one bounded feature cache (approximately1GiB maximum), its
predictions/provenance and seven small prior banks. Completed stages are reused
only after input/code/artifact hash checks. Unverified partial outputs require
inspection. `preparation_complete.json` explicitly means preparation only, not a
model gate or production success. The nine training arms remain not runnable
until trainer integration and their independent gates are completed.
## 2026-10-12 preparation completion and update runtime

Train preparation finished at 06:20 JST: all 4,237 train images were predicted,
2,546 of 2,550 originally eligible records had a matched predicted crop, and
k1..7 were fitted/calibrated successfully. Four unmatched records remain excluded;
no GT crop substituted for them. The feature cache contains 2,032 fitting records
and 514 calibration records from the frozen disjoint train-video partition.
The preparation report and SHA-authenticated banks are under
`outputs/landmark_detection/messenger_geometry_prior/preparation/predicted_features_20261012`.
This is preparation completion, not model-gate or production completion.

`geometry_prior_training.py` now provides opt-in model construction, separate G/D
optimizers, AMP and effective-window accumulation, immutable-observer-aware EMA,
and complete checkpoint state (model, observer/prior identity, EMA, G/D optimizers,
D, scaler, epoch/step, schedulers, loader and Python/numpy/Torch/CUDA RNG).
G and D gradients accumulate against the same pre-update D; both finite-gradient
checks precede either optimizer commit. An empty anomaly window never updates D.
Pre-commit OOM replays the same already-augmented effective window at a smaller
microbatch. OOM during a possibly partial optimizer commit instead requires a
complete checkpoint restore. The lazy AMP scale initializes before first forward
so a first-forward OOM can reset scaler state safely.

The fixed observer now explicitly runs in FP32 inside student AMP, matching
preparation. Both model and EMA observers, and the prior bank, remain immutable.
The only change to the old preparation dependency is this wrapper forward block;
reconstructing the old block reproduces its recorded source SHA exactly. Separate
evidence `prior_preparation_runtime_compatibility_20261012.json` preserves this
compatibility proof without rewriting the original preparation identity.

The real-data integration probe uses 16 paired train crops, k1, G/D AMP and 4/8/16
microbatches. Its adversarial path is a software diagnostic, not the scheduled
validation-selected adversarial experiment. CUDA Query grid_sample backward has
no deterministic implementation: next-step bitwise equality is not a valid CUDA
resume requirement. The probe instead requires bitwise equality of every restored
checkpoint state and finite continuation at the same step; next-update parameter
drift is recorded separately. The initial failed equality diagnostic and the
strict-determinism rejection remain recorded. No threshold of a quality gate was
changed. CPU next-update equality is still required by unit tests.

The probe is NOT an effective64 production preflight: it does not perform full
conditional validation, isolate timings from common initial state, select a batch,
or certify any model gate. Epoch scheduling/staged unfreezing, deployed conditional
validation, W&B/retention, independent four gates and production/evaluation queue
registrations remain unfinished. All nine model arms remain not runnable; the new
35% fine-tunes remain deferred behind the older plans.

Verification at 09:22 JST: the GPU diagnostic completed. Each microbatch 4/8/16
ran four updates without natural OOM or non-finite loss; peak PyTorch reserved
memory was 2,526 MiB and sampled temperature was at most 49C. Injected first-forward
AMP OOM reduced microbatch4 to2 while preserving effective16. All serialized
runtime state restored bitwise exactly. The next CUDA update was finite with the
same step; maximum generator parameter drift was 1.1920929e-6 and D drift was0,
recorded as nondeterminism rather than exact next-step equality. Frozen model and
EMA observers stayed bitwise unchanged. 63 CPU tests passed, then the 35 affected
tests passed again after the lazy-scaler repair. Temporary probe checkpoints were
removed; no production model or gate was run. Detailed result:
`outputs/landmark_detection/messenger_geometry_prior/runtime_probe_20261012/probe_complete.json`.

## 2026-10-12 epoch and conditional-validation integration

The frozen parent detector now has an authenticated full canonical validation
cache prepared by `scripts/prepare_messenger_prior_validation.py`: 530 images,
882 native Heads, 48 invalid/ambiguous attribute records excluded, 47 all-missing
records excluded, 787 eligible Heads before matching, and 768 assigned predicted
FACE crops (19 unmatched eligible Heads). Assignments consider all GT Heads
one-to-one, including ineligible anchors, then retain the conditional population.
No HEAD detector is introduced. The fixed parent FACE threshold0.33, cap100 and
EMA checkpoint are unchanged; no validation/test threshold search was performed.

`geometry_prior_validation.py` uses the predicted ellipse for crop pixels and
geometry, and explicitly adapted GT Head bboxes only for normalization. It
accumulates five TP/prediction/GT counters, reports equal-weight semantic Macro F1
at inclusive presence0.70 and NME0.10 with side1.20*max(GT Head width,height), and
retains no per-image prediction rows during an epoch. GT crop fallback is rejected.
Face misses and all-missing Heads stay outside this conditional selection metric;
this is not a substitute for the end-to-end common performance report.

`geometry_prior_epochs.py` now integrates effective-window loaders, staged
backbone freezing, inherited Messenger mixing warmup, per-epoch validation,
validation-selected best/early stopping, G/D scheduler steps, scalar W&B logging
adapter and complete epoch-boundary resume including patience and data-order RNG.
Workers must be nonpersistent for faithful worker seeding on epoch-boundary
resume. Any error after optimizer updates, including validation/logging failure,
latches checkpoint restoration as required; the same epoch cannot silently be
replayed over advanced weights. Frozen backbone stays in eval mode even when the
runtime enters train mode. Production policy remains freeze/warmup5, inherited
Messenger mixing warmup2, effective64, max40 and minimum10/patience5/min_delta1e-4.

`last.pt` is saved after every validated epoch and `best.pt` after improvement;
one recovery snapshot is kept every5 epochs and removed after genuine completion.
CPU tests verify exact next-epoch resume, sampler/scheduler state, freeze/unfreeze,
partial effective windows, early stopping, retention and failed-epoch safeguards.

The bounded real GPU diagnostic completed at12:21 JST. It used128 actual train
samples, effective64/microbatch4, a deliberately separate two-epoch schedule with
freeze/warmup/mixing1, full768-crop conditional validation each epoch, and a replay
of epoch2 after restoring epoch1. Frozen model and EMA observers were unchanged;
GPU reserved memory peaked at1284MiB and sampled temperature reached52C.
Temporary diagnostic checkpoints were removed. The diagnostic is not production,
not a model gate, not a batch-selection preflight and not evidence of improved
model performance. Its scores and nondeterministic resumed CUDA scores remain
labeled diagnostics. Verification:71 CPU tests passed.

Evidence:
`outputs/training_queues/pending_models_tests_then_training/monitor_20261012_1210_prior_epochs.json`
and `outputs/landmark_detection/messenger_geometry_prior/epoch_probe_20261012/probe_complete.json`.
Remaining: authenticated per-arm configs/production CLI and online W&B binding,
isolated4/8/16 production preflight, independent current-source four gates,
canonical production/common evaluation registrations, and validation-only k
selection before the adversarial arm. All nine arms remain not runnable; the
later35% fine-tunes remain deferred behind the older plans.


## 2026-10-12 authenticated launcher and online resume

After real-face Messenger completed19epochs (best14) at16:44 JST, the next
integration step adds eight resolved control/k1..7 configurations under
`configs/training_plans/messenger_geometry_prior_runs`. Every configuration binds
its prior, initial checkpoint, train annotations and predicted crops, full
conditional validation preparation and hashes. No adversarial k is selected.
`geometry_prior_launch.py` binds code/config/artifacts and output identity, requires
all four exact-identity gate markers and a safe batch plan for production, and
creates/resumes one explicitly online W&B run. Incompatible output reuse and a
checkpoint whose remote run cannot be resumed fail closed.

`train_messenger_geometry_prior.py` uses the shared epoch driver, original seed42,
effective64, freeze/warmup5 and mixing2, validation every epoch, max40/min10/patience5,
and original initialization. It records epoch0 validation, preserves optimizer,
EMA, scaler, schedulers, RNG and patience in epoch checkpoints, and logs scalar
metrics after durable checkpoint saving. Training OOM replay retains the effective
window; validation OOM reduces only validation microbatch. Train structure
annotations remain train-only. The discriminator arm remains unavailable until
validation selects k and its own authenticated launch/gates are integrated.

The separate launcher-check output uses128 train samples/two diagnostic epochs,
freeze/warmup/mixing1 and full768-crop conditional validation; it is not production,
a formal gate, a batch preflight or a model-performance comparison. A canonical
bounded setup runner executes epoch1 then resumes in a separate process with the
same W&B run. Production registrations, isolated4/8/16 preflight, formal four gates,
common evaluation and validation k selection are still pending. CPU checks:19 passed.

The two-process launcher check completed at18:15:40 JST: checkpoint epoch1/step2
was restored and continued to epoch2/step4, with W&B `resumed=true` on the same run
`gp-d0b721fd85b876f23da1`. These are diagnostic updates, not production/gates.
The next bounded setup runner serially probes control and k1..7 at4/8/16; each
candidate has an isolated process, effective64, two epochs/four joint-backbone
updates, full768-crop validation, second-epoch synchronized timing and sampled
GPU temperature/device-memory guards. Results remain separate preflight evidence,
not formal GPU success markers. No additional acceptance threshold was relaxed.

## 2026-10-12 formal gate integration

All eight isolated preflights completed at18:34:33 JST. Every4/8/16 candidate
passed, with microbatch16 selected by measured projected time for every arm
(effective64). Their original config, artifact and source identities still match;
no training-path code changed. The formal GPU gate revalidates every candidate's
identity, repeated joint updates, finite metrics, full conditional validation,
safety evidence and timing selection, preserving original probe files.

`scripts/run_messenger_geometry_prior_gate.py` supplies independent schema/GPU/
overfit/small-scale authentication. Schema checks HEAD-free topology, canonical
split hashes/inventories, fixed policy and predicted-crop cohorts, plus tests.
Overfit fixes four augmented train examples for12epochs and requires a lower
total objective than the initial epoch. Small-scale uses256train samples and
production augmentation/AMP/freeze5/mixing2/warmup5 for20epochs, validates all768
conditional crops each epoch, and requires best subsequent EMA semantic Macro F1
to improve its epoch0 value by more than1e-4. Early stopping is disabled only in
these bounded gates; production remains max40/min10/patience5. One best checkpoint
is retained per learning gate. Quality rejection and execution errors are distinct.

The canonical bounded runner now contains32 formal gate jobs. No gate report is
promoted without its assertions; no preflight diagnostic was relabeled in place.
48 relevant CPU tests passed. Production/common evaluation integration and
validation-only k selection before the adversarial arm remain incomplete.
