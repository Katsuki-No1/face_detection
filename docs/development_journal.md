# 開発日誌

実装・学習・評価・可視化について、後から「何を、なぜ行い、何が分かったか」を日付順に確認するための記録です。新しい日付・項目を上に置きます。

各項目の第4節は、結果を確認したユーザー本人の言葉だけを原文のまま記録します。未確認・未回答の場合は未記入のまま残します。

更新には `.venv/bin/python scripts/update_development_journal.py` を使用します。同じ論理作業には同じentry IDを使うため、再実行しても重複しません。

<!-- development-journal-entries -->

## 2026-10-10

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
