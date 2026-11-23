# Model integration notes

## Branch

Apply the patch on `feature/ner-crf-improvement`, based on `feature/conditional-llm-routing`. Do not use the older `feature/destination-understanding` branch as the integration base.

## Protected files

The NER overlay does not modify any backend source. In particular, it leaves the following unchanged:
- `backend/ambiguity.py`
- `backend/conditional_router.py`
- `backend/ml_response_formatter.py`
- `backend/main.py`
- `backend/response_normalizer.py`
- `backend/model_service.py` (coordinate before any future edits)
- frontend, Whisper/speech, Gemini/LLM and routing code.

## Model compatibility

The replacement CRF uses feature_dim 4096, the existing 15 BIO tags, state-dict keys `em`, `trans`, `start`, `end`, and the existing hashed token feature templates. The original runtime decoder can enforce BIO legality because the saved transition and start scores assign very negative weights to illegal transitions.

The intent classifier is unchanged. The `model_service.predict(query)` interface and output fields are intended to remain unchanged.

## Validation

Run from the repository root:

```bash
python scripts/audit_ner_dataset.py --data data/manzil_v2_token_level_clean_augmented.jsonl
python scripts/smoke_test_model_service.py
```

Then test the complete FastAPI/frontend application with both ML_ONLY and ML_WITH_LLM paths. Do not change routing thresholds as part of this NER patch. The architecture README/code currently specifies confidence threshold 0.50 and margin threshold 0.20; the partner's voice message mentioned 0.75, so that discrepancy needs partner confirmation.
