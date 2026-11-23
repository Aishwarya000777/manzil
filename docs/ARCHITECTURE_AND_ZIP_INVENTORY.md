# Architecture and ZIP inventory notes

## Application architecture

Source: `readme1.md` on `feature/conditional-llm-routing`.

- Intent classifier and CRF BIO NER run first for every query.
- The conditional evaluator checks supported intent, confidence, margin, destination availability, REF and REL entities.
- Simple queries use ML_ONLY and the ML response formatter.
- Complex or uncertain queries use ML_WITH_LLM; Gemini receives the original query and ML predictions.
- Response normalization standardizes the outputs.
- `/api/transcribe` handles audio; `/api/ml-predict` exposes model predictions; `/api/analyze` runs conditional routing.

The checked-in code and README specify confidence threshold 0.50 and margin threshold 0.20. The partner's voice note mentioned 0.75. This patch records the discrepancy and does not alter the router.

## Purpose of the extra files

The extra files are ML/research reproducibility assets, not frontend/backend runtime dependencies.

- Full and token-level JSONL files support model training, BIO tagging, and evaluation.
- Semantic-level JSONL captures sentence-level semantic representations.
- Training code recreates the CRF checkpoint.
- Audit scripts validate BIO sequences, token offsets, duplicates and split isolation.
- Evaluation files contain classification reports, confusion matrices, paired results and regression examples.
- The annotation workbook supports manual review; it has not yet been completed by two independent annotators.
- The XLM-R/mBERT notebook is a proposed benchmark workflow; those transformer models have not been trained in this experiment.

## Runtime versus research

- Runtime: frontend/backend code, intent checkpoint and CRF checkpoint.
- Training/reproducibility: JSONL data, CRF training script and audit scripts.
- Evaluation: reports, confusion matrices and regression examples.
- Human validation: annotation workbook and annotation guide.
- Optional research: XLM-R/mBERT notebook.

The NER improvement patch is limited to the CRF and supporting research assets. It does not modify backend, frontend, intent classifier, Whisper, Gemini or conditional routing source.