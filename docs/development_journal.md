# 開発日誌

実装・学習・評価・可視化について、後から「何を、なぜ行い、何が分かったか」を日付順に確認するための記録です。新しい日付・項目を上に置きます。

各項目の第4節は、結果を確認したユーザー本人の言葉だけを原文のまま記録します。未確認・未回答の場合は未記入のまま残します。

更新には `.venv/bin/python scripts/update_development_journal.py` を使用します。同じ論理作業には同じentry IDを使うため、再実行しても重複しません。

<!-- development-journal-entries -->

## 2026-10-10

<!-- dev-journal-entry:start id=baseline-branch-focused-cleanup-20261010 -->
### 18:09 JST — Restrict baseline branch to model entry points and standalone performance documentation

- ID: `baseline-branch-focused-cleanup-20261010`
- 状態: `completed`
- 記録元: `User requested removing unnecessary model scripts and descriptions from baseline and presenting the model performance without comparisons.`
- Git: `ec2614d60c9f` (`dirty`)

#### 1. 設計意図・判断根拠

- Remove inherited unrelated experiments from the baseline tree with a normal deletion commit; preserve Git history. Use an isolated WSL Git worktree because the desktop worktree tool reported Git unavailable. Original occlusion checkout and uncommitted work remain untouched.
- Preserve the exact HEAD-free full-image MAL alpha2 FACE/ellipse -> Face-crop DINOv3 Query Refinement topology, frozen registered config identities and imported internal shared implementations. Keep FACE-component evidence separate from combined-pipeline not_run status.

実装:

- Deleted 436 tracked files outside the baseline dependency/configuration/test closure, including obsolete model scripts, comparison reports, experimental configurations, queues and unrelated utilities. Removed unused top-level dependency entries and the ablation-only test attached to the deleted comparison tool. Model inference/training implementation and stored evaluation artifacts were not changed.
- Rewrote README, docs/README.md and model_baseline_history.md for the current model; added docs/baseline_performance.md and a branch-specific AGENTS scope. This entry also records journal maintenance.

#### 2. 学習条件・学習結果

条件:

- not run; no training, checkpoint selection or new performance evaluation.

結果:

- not run. Preserved complete baseline combined_inference_status and combined_evaluation_status not_run.

#### 3. 出力・検証・可視化

出力:

- docs/baseline_performance.md: only candidate FACE-component values copied from hash-verified existing val/test metrics and authenticated success markers. Canonical-v2 val530/FACE840 and test528/FACE769; do not label as latest r3. Common threshold0.33 and validation-locked threshold0.2666015625. Agreed occlusion/profile tables explicitly N/A when matching slice aggregates are unavailable.
- Configs/model registry remains unchanged: FACE checkpoint SHA 3bd6e40517bc6f6e276c47e1dfdf639173e8b9c04d32486adbb6f4b407f006a0; retained keypoint checkpoint SHA baf7a6d0b1df6d1089357b2f28ffc1b2da3aadbb6239bae2cb5085d36ef016e0.

検証と意図:

- .venv/bin/python -m pytest -q in isolated baseline worktree: 49 passed. Six README-documented CLI entry points passed --help, including FACE image/video inference, detector evaluation, gated detector launcher, keypoint launcher and downstream diagnostic evaluator.
- Both registered config SHA-256 values unchanged; all retained Python source parsed; README/doc links resolved; public model docs contain no obsolete model-comparison tables. Existing performance sources and marker hashes verified before copying values.
- Authorized baseline commit and push follow this entry; verify remote SHA and unchanged occlusion SHA after push.

#### 4. 結果に対するユーザーの考え・認識

> 未記入（結果確認後、ユーザー本人の言葉を追記する）
<!-- dev-journal-entry:end id=baseline-branch-focused-cleanup-20261010 -->

<!-- dev-journal-entry:start id=baseline-github-source-snapshot-20261010 -->
### 16:14 JST — Publish current MAL alpha2 and Query Refinement baseline source snapshot

- ID: `baseline-github-source-snapshot-20261010`
- 状態: `completed`
- 記録元: `User requested commit and push to GitHub branch baseline; scope confirmed as current baseline and necessary dependencies.`
- Git: `704b3994a06d` (`dirty`)

#### 1. 設計意図・判断根拠

- Current local baseline is the HEAD-free ordered full-image MAL alpha2 FACE/ellipse -> Face-crop DINOv3 Query Refinement pipeline. Preserve registered identities and separate component evidence from combined evaluation.
- Stage only current baseline configs, source/import dependencies, baseline documentation and related test changes. Independent experimental queues, datasets, checkpoints, generated caches and media remain outside this commit. Shared modules retain their imported optional implementations; no optional model is activated.

実装:

- Created baseline from local HEAD 704b399. Selected README and AGENTS sections without modifying the remaining local changes. Updated this chronological journal and included its maintenance tool.

#### 2. 学習条件・学習結果

条件:

- not run; no new training, inference, checkpoint selection or performance evaluation.

結果:

- not run; combined baseline inference/evaluation remains not_run as recorded in the registry.

#### 3. 出力・検証・可視化

出力:

- configs/model_registry/local_baseline.yaml: authoritative complete-pipeline definition and component asset identities. docs/model_baseline_history.md: baseline history.
- Git index snapshot and temporary isolated verification directory /tmp/face-baseline-verify-x_jhd_sv; no model artifacts included.

検証と意図:

- Isolated index snapshot: .venv/bin/python -m pytest -q tests/test_face_bbox_iou50.py tests/test_baseline_aligned_evaluation.py tests/test_update_development_journal.py keypoint_comparison1/tests/test_landmark_detection.py -> 39 passed.
- Both registered config SHA-256 hashes, config inheritance, HEAD-free baseline declaration and staged Python syntax passed. Existing terminal blank lines preserved to retain config hashes; whitespace check excluding blank-at-eof passed.
- Initial check exposed an unrelated comparison-test import and a historical threshold assertion; staged only the baseline threshold update and retained other experimental test additions locally. CPU-only CUDA inventory test was rerun with normal device visibility.
- Commit and GitHub push follow this source-snapshot entry; remote branch SHA will be checked after push.

#### 4. 結果に対するユーザーの考え・認識

> 未記入（結果確認後、ユーザー本人の言葉を追記する）
<!-- dev-journal-entry:end id=baseline-github-source-snapshot-20261010 -->
