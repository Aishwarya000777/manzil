# Source-of-truth and change audit

## Integration target

The target branch is `feature/conditional-llm-routing` because it contains the partner's integrated conditional-LLM backend and frontend work. The NER patch branch should be created from that branch and remain separate until reviewed.

## Files to be changed

- `models/manzil_linear_crf.pt`: replacement trained CRF checkpoint.
- `models/manzil_linear_crf_baseline_backup.pt`: original checkpoint backup.
- `data/manzil_v2_full_clean_augmented.jsonl`, `data/manzil_v2_semantic_level_clean_augmented.jsonl`, `data/manzil_v2_token_level_clean_augmented.jsonl`: cleaned/augmented synthetic research data.
- `src/train_crf_improved.py`: reproducible training script.
- `scripts/audit_ner_dataset.py` and `scripts/smoke_test_model_service.py`: validation utilities.
- `evaluation/ner_improvement/`: evaluation reports, metrics, confusion matrices and the target regression case.
- `docs/`: training and integration documentation.
- `annotation/` and `notebooks/`: human review workbook and optional future transformer benchmark.

## Files intentionally not changed

No frontend files, no backend files, no intent model checkpoint, no speech service, no Gemini service, no conditional routing or response normalization code are modified by the patch.

## Validation limits

The corpus is synthetic and requires human review. The evaluation is a held-out synthetic test, not a benchmark on a representative real-user corpus. The 300-example annotation workbook remains uncompleted and no human agreement score is claimed. The existing intent model was not retrained.
