# Source-of-truth audit

## Live repository and branch selection

- Repository: `Aishwarya000777/manzil` (public).
- The repository has two branches at inspection: `feature/destination-understanding` and `feature/conditional-llm-routing`.
- **Patch base:** `feature/conditional-llm-routing`, inspected commit `a92b8e04feca2776399834038c5b21fe14efa9b3`. The partner's `readme1.md` states that this is the branch implementing the complete conditional-LLM application architecture.
- `feature/destination-understanding` (inspected commit `3ca404d10bb564b46833b1f4fed1c8536b2f3581`) is a predecessor and lacks `backend/conditional_router.py` and `backend/ml_response_formatter.py`. Do not apply this patch to that branch.
- `readme1.md` is the detailed application/architecture README. The root `README.md` is an earlier research/data summary; keep both and do not replace the architecture README.

## Architecture source verification

The selected branch includes the full frontend and backend tree, Whisper transcription endpoint, Gemini service, conditional routing logic, ML-only response formatter, response normalizer, and all protected backend files. `backend/model_service.py` exposes `intent_confidence`, `intent_probabilities` and `intent_margin` to the conditional router.

The README and checked-in router currently use confidence threshold `0.50` and margin threshold `0.20`, with REF/REL detections also triggering escalation. The user's supplied audio transcript described a `0.75` confidence threshold. This is a requirement/code discrepancy; the NER patch does not change router code or thresholds.

## Data and checkpoint provenance

The following core v2 assets were verified against GitHub blob SHAs on the shared branch and match the local Research Completion Package copies:

- `data/manzil_v2_token_level.jsonl`: `fc8ca69897f024ebad5759a8d26e155e5635bc1c`
- `data/manzil_v2_full.jsonl`: `e2aee4c64cef9f8e2bc32ecf123dc0eda3e6909f`
- `models/manzil_linear_crf.pt`: `cb622541d695dcafa6cb1ce3104f25fe744a6b98`
- `models/intent_char_tfidf_logreg.joblib`: `e5b657f2a287d6ef61aa150ba2d3daddff595dd6`

The NER patch is based on these current v2 data/model assets. It preserves the original 1,000 rows, repairs 200 malformed BIO transitions, adds 75 synthetic training/evaluation candidates, and supplies a replacement CRF checkpoint plus the original CRF backup. The intent model is unchanged.

## ZIP inventory: what the extra files are for

These extra files are ML/research reproducibility assets, not files the frontend/backend must load at runtime.

### `Manzil_All_Documents_and_Dataset.zip` — earlier pilot + Phase 2 report

Contains:
- `Manzil_Dataset_v2_Pilot/README.md` and `ANNOTATION_GUIDELINES.md`: pilot documentation and annotation conventions.
- `manzil_v2.csv`, `manzil_v2_token_dataset.jsonl`, `manzil_v2_semantic_dataset.jsonl`: earlier generated pilot in flat, token/BIO and semantic representations.
- `manually_spot_checked_sample.jsonl`: manually inspected sample (not equivalent to full independent annotation).
- `statistics.json`: dataset distribution and summary.
- `duplicate_check.py`: duplicate audit utility.
- `train_baselines.py`: initial model-training baseline.
- `predict.py`: early prediction example.
- `Phase2_Report/Group-16(2).pdf`: Phase 2 report.

### `Manzil_Complete_Improved_Project.zip` — 20-entry historical integrated package

Contains the earlier integrated `manzil_v2_integrated.csv` / `.jsonl`, audit statistics, legacy template audit, preserved original dataset, duplicate audit and baseline training script, plus the initial pilot CSV/JSONL, its token and semantic annotations, README, annotation guidelines, spot-check sample, predictor, and duplicate/training scripts. These files document prior data experiments and preserve the legacy corpus; do not mix their older training labels or splits into the current v2 CRF training job.

### `Manzil_Complete_Improved_Project_with_Diagnostics.zip`

Adds an experimental intent diagnostic training script, diagnostic report and diagnostic `.joblib` checkpoint to the historical integrated package. This is not the current intent checkpoint and is not used by the partner app.

### `Manzil_v2_Research_Completion_Package.zip` / `Manzil_v2_Final_Package.zip`

These are later research bundles. The Research Completion Package includes the 1,000-row balanced v2 pilot, current core model checkpoints, intent/NER evaluation files, original source dataset, human-review workbook and XLM-R/mBERT Colab notebook. Its four core v2 data/model assets match the GitHub branch assets listed above. The workbook and notebook are research assets, not app runtime dependencies.

### This NER improvement patch

Supersedes only the CRF training dataset/checkpoint for the next experiment. It includes clean/augmented JSONL, CRF checkpoint, baseline checkpoint backup, training/audit/smoke-test scripts, evaluation reports/confusion matrices, workbook, transformer notebook and integration instructions. It does not contain or modify the app's backend/frontend source.

## Runtime vs research assets

- App runtime: frontend/backend source, requirements, current intent checkpoint and current CRF checkpoint.
- Model training/reproduction: token-level/full/semantic JSONL, train scripts, audit scripts, evaluation scripts/results.
- Human validation: annotation workbook and guidelines.
- Optional research: XLM-R/mBERT Colab notebook, references and reports.

## Access status

The GitHub connection used for this session reported repository `pull=true`, `push=false`. No remote branch or pull request has been created. The included `apply_patch.sh` creates a local branch from `origin/feature/conditional-llm-routing`, stages only ML/research files, and leaves the protected backend/frontend files untouched.
