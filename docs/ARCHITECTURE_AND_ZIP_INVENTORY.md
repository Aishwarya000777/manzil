# Architecture and ZIP inventory notes

## Application architecture (partner's documented branch)

Source: `readme1.md` on `feature/conditional-llm-routing`.

- ML-first path: intent classifier + CRF BIO NER run for every query.
- Conditional evaluator examines intent support, confidence, margin, destination availability, REF and REL entities.
- Simple/high-confidence cases use `ML_ONLY` + `format_ml_response`.
- Complex/uncertain cases use `ML_WITH_LLM`; Gemini receives both original query and ML predictions.
- `response_normalizer.py` normalizes both outputs.
- `/api/transcribe` handles uploaded speech; `/api/ml-predict` exposes ML-only predictions; `/api/analyze` runs the conditional path.

Current checked-in code and `readme1.md` specify confidence threshold 0.50 and margin threshold 0.20. The partner's audio transcript says 0.75 for confidence. This discrepancy is recorded only; the NER patch does not alter it.

## Purpose of the ZIP files

The extra files are ML/research reproducibility assets, not files the frontend/backend must load at runtime. Below is the exact inventory of the two earlier archives.

### `Manzil_All_Documents_and_Dataset.zip` — 11 entries

`Manzil_Dataset_v2_Pilot/`:

- `README.md` — describes the pilot and how to use it.
- `ANNOTATION_GUIDELINES.md` — labeling conventions for the pilot.
- `manzil_v2.csv` — flat row-level dataset.
- `manzil_v2_token_dataset.jsonl` — token sequences and BIO entity tags for NER training/evaluation.
- `manzil_v2_semantic_dataset.jsonl` — sentence-level/semantic representations.
- `manually_spot_checked_sample.jsonl` — small manually inspected sample; not proof that the full corpus is human-validated.
- `statistics.json` — pilot counts and distribution summary.
- `duplicate_check.py` — duplicate-audit script.
- `train_baselines.py` — initial baseline model training script.
- `predict.py` — early prediction example.

`Phase2_Report/Group-16(2).pdf` — the Phase 2 report.

### `Manzil_Complete_Improved_Project.zip` — 20 entries

`Manzil_Integrated_v2/` (9 files):

- `dataset_audit_statistics.json` — integrated dataset audit/counts.
- `README.md` — notes for the integrated dataset.
- `manzil_v2_integrated.csv` and `manzil_v2_integrated.jsonl` — row-level integrated corpus in tabular and JSONL formats.
- `legacy_template_audit.csv` — audit of the old template set.
- `legacy_original_preserved.csv` — preserved original/legacy rows.
- `duplicate_audit.py` — duplicate-audit utility.
- `train_baseline.py` — baseline-training script.
- `ANNOTATION_GUIDELINES_v2.md` — guidelines for the integrated labels.

`Initial_Pilot_Artifacts/` (10 files):

- `statistics.json`, `manzil_v2.csv` — pilot statistics and flat dataset.
- `manzil_v2_token_dataset.jsonl`, `manzil_v2_semantic_dataset.jsonl` — token/BIO and semantic data.
- `manually_spot_checked_sample.jsonl` — sample checked manually.
- `README.md`, `ANNOTATION_GUIDELINES.md` — documentation and labeling rules.
- `duplicate_check.py` — duplicate checker.
- `train_baselines.py` — initial training script.
- `predict.py` — sample inference code.

`Source_Original/manzil_dataset.csv` (1 file) — source/original dataset snapshot. This archive is an intermediate integrated-data snapshot; it is useful for provenance and historical analysis but should not replace the current v2 dataset files in the live repo.

### `Manzil_Complete_Improved_Project_with_Diagnostics.zip` — 25 entries

This is the preceding 20-entry integrated package plus an experimental intent-diagnostic script (`train_intent_diagnostic.py`), its report (`intent_v2_diagnostic_report.txt`), diagnostic checkpoint (`manzil_intent_v2_diagnostic.joblib`), additional execution/audit report (`EXECUTION_AND_AUDIT_REPORT.md`) and a temporary `test.tmp`, with some folder/path names flattened. It is a historical experiment, not the app's current intent model. Do not overwrite `models/intent_char_tfidf_logreg.joblib` with the diagnostic `.joblib`.

### `Manzil_v2_Research_Completion_Package.zip` / `Manzil_v2_Final_Package.zip`

These are later research bundles. The Research Completion Package includes the 1,000-row balanced v2 pilot, current baseline checkpoints/evaluation, original source dataset, human-review workbook and XLM-R/mBERT Colab notebook. Its four core v2 data/model assets match the live GitHub branch assets listed above. The workbook and notebook are research assets, not app runtime dependencies.

### This NER improvement patch

Supersedes only the CRF training dataset/checkpoint for the next experiment. It includes clean/augmented JSONL, CRF checkpoint, baseline checkpoint backup, training/audit/smoke-test scripts, evaluation reports/confusion matrices, workbook, transformer notebook and integration instructions. It does not contain or modify the app's backend/frontend source.

## Runtime vs research assets

- App runtime: frontend/backend source, requirements, current intent checkpoint and current CRF checkpoint.
- Model training/reproduction: token-level/full/semantic JSONL, train scripts, audit scripts, evaluation scripts/results.
- Human validation: annotation workbook and guidelines.
- Optional research: XLM-R/mBERT Colab notebook, references and reports.
