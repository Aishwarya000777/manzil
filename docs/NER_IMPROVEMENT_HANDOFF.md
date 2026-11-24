# Manzil CRF NER improvement

This branch is based on Aishwarya Vadla's `feature/conditional-llm-routing` implementation. It is intended to contain only the CRF/BIO NER improvement, training and evaluation artefacts. The protected API, conditional routing, formatter, Whisper, Gemini and frontend files are intentionally unchanged.

## Reported experiment

On the paired 176-query synthetic test subset:

| Metric | Existing CRF | Candidate CRF |
|---|---:|---:|
| Token accuracy | 71.58% | 85.97% |
| Token macro-F1 | 22.82% | 70.46% |
| Exact-span precision | 42.29% | 70.52% |
| Exact-span recall | 28.43% | 72.05% |
| Exact-span F1 | 34.01% | 71.28% |
| Invalid BIO transitions | 43 | 0 |

The separate 177-row final synthetic test evaluation reports exact-span F1 71.75%. All examples are synthetic candidates pending human review, so these figures are not claims of real-user performance.

## Model changes

- Repair leading-whitespace BIO annotation errors.
- Add 75 synthetic candidate examples to improve rare and nested entity coverage.
- Retain the 4,096-dimensional feature-hashed linear-chain CRF and existing checkpoint parameter names.
- Enforce legal BIO start/transition constraints at training and decoding.
- Oversample utterances containing rare DIR and CONTEXT entities.
- Leave the intent classifier and protected backend integration unchanged.

## Integration

Use `feature/ner-crf-improvement` as the candidate branch, based on `feature/conditional-llm-routing`. Review the accompanying training, dataset-audit, smoke-test, evaluation and model integration files before merging. The 1,075-row candidate dataset requires human review before it is treated as gold-standard data.