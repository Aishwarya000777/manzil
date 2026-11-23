# Manzil NER improvement report

## Scope and source of truth

The NER improvement is intended for `feature/conditional-llm-routing`, the branch containing the integrated frontend, backend and conditional LLM routing implementation. The original dataset and model files were checked against the repository's v2 files before training.

## Data repair and additions

The original 1,000-row v2 token dataset contained 200 entity spans with leading whitespace included in their offsets. Rebuilding BIO tags from trimmed character offsets corrected these; the final audit reports zero malformed I transitions, zero token/tag/offset errors and zero entity-offset errors.

The cleaned dataset contains 1,075 rows: 717 train, 181 validation and 177 test. It retains the original 1,000 records and adds 75 synthetic candidate utterances to increase coverage of complex navigation and rare entity labels. All added records are synthetic and require human review. The final audit reports no normalized exact duplicates and no scenario-family overlap across splits.

## Model changes

The NER model remains a first-order linear-chain CRF with the same 4,096-dimensional hashed token features, tag list and checkpoint parameter names (em, trans, start and end) used by the partner backend. Training adds hard BIO start/transition constraints and utterance-level oversampling for examples containing DIR or CONTEXT. The selected candidate was chosen using validation performance; the test set was not used to select it.

The replacement checkpoint is intended for `models/manzil_linear_crf.pt`. No backend, router, response formatter, intent model, Whisper, Gemini or frontend source changes are required. The checkpoint preserves the backend CRF state-dict tensor shapes and feature dimension.

## Before/after NER results

The same 176 held-out test rows are used for the paired before/after comparison. The Madhapur/ATM regression query is excluded from this comparison because an identical query was present in the original baseline training data; it is evaluated separately after being held out from the final candidate's training data.

| Metric | Existing checkpoint | Improved checkpoint |
|---|---:|---:|
| Test rows | 176 | 176 |
| Token accuracy | 71.58% | 85.97% |
| Token macro-F1 | 22.82% | 70.46% |
| Exact-span precision | 42.29% | 70.52% |
| Exact-span recall | 28.43% | 72.05% |
| **Exact-span F1** | **34.01%** | **71.28%** |
| Invalid BIO transitions predicted | 43 | 0 |

The final candidate on all 177 test rows reports token accuracy 86.06%, token macro-F1 71.34%, exact-span precision 71.00%, recall 72.51%, exact-span F1 71.75%, and zero invalid BIO transitions.

## Per-entity and per-language analysis

| Entity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CONTEXT | 40.00% | 18.18% | 25.00% | 33 |
| DESC | 33.33% | 30.00% | 31.58% | 20 |
| DEST | 66.67% | 79.14% | 72.37% | 139 |
| DIR | 87.50% | 77.78% | 82.35% | 9 |
| DIST | 88.89% | 88.89% | 88.89% | 9 |
| REF | 86.36% | 79.83% | 82.97% | 119 |
| REL | 67.68% | 77.91% | 72.43% | 86 |

| Language | Precision | Recall | F1 |
|---|---:|---:|---:|
| Hindi (Devanagari) | 83.56% | 82.43% | 82.99% |
| Romanized Hindi | 59.60% | 61.22% | 60.40% |
| Hinglish | 68.50% | 72.50% | 70.45% |

The gains are substantial, but DESC, CONTEXT and Romanized Hindi remain weak spots. Small test supports for DIR and DIST mean their per-class estimates are uncertain.

## Madhapur/ATM regression case

For `Madhapur mein jo mall hai uske peeche wali building ke paas ek ATM hai, wahan jaana hai`, the candidate predicted Madhapur as CONTEXT, mall as REF, uske peeche as REL, wali building as DESC, ke paas as REL, ATM as DEST, and wahan as CONTEXT. The prediction matches the held-out test labels for this example.

## Intent classification

The existing `intent_char_tfidf_logreg.joblib` checkpoint was not retrained or modified. On the current 162-row synthetic held-out split it scores 90.74% accuracy, 91.46% macro-F1 and 91.18% weighted-F1. These metrics are not directly comparable with the separate Phase 2 supplementary development score, which used a different evaluation set.

## Limitations

All 1,075 dataset rows are synthetic candidates and have not been fully human validated. The annotation workbook is provided for subsequent review; no inter-annotator agreement is claimed. Romanized Hindi, DESC and CONTEXT remain weaker categories. XLM-R and mBERT are proposed benchmarks only and have not been trained as part of this CRF improvement.
