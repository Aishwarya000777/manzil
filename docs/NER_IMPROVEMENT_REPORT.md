# Manzil CRF NER improvement report

## Scope

This patch addresses the requested ML improvement only: CRF-based NER and BIO sequence quality. The intent classifier, frontend, Whisper transcription, Gemini/LLM integration, conditional router, response formatter, response normalizer and all protected backend files are left unchanged.

## Source branch and project context

The patch is prepared for `Aishwarya000777/manzil` on **`feature/conditional-llm-routing`**, commit inspected: `a92b8e04feca2776399834038c5b21fe14efa9b3`. The partner's `readme1.md` explicitly documents this architecture branch. It contains the complete frontend/backend application, Whisper endpoint, Gemini service, conditional router, ML formatter and protected files. The older `feature/destination-understanding` branch is a predecessor and does not contain the conditional-router files; do not base this patch on that branch.

## Dataset repair and augmentation

The current 1,000-row token-level dataset had **200 malformed BIO `I-` transitions**, caused by spatial-relation entity offsets beginning with leading whitespace. BIO labels were rebuilt from the trimmed character spans. The 1,000 original records are preserved in the clean augmented dataset, and 75 additional synthetic examples were added to give training coverage for rare `DIR` and `CONTEXT` labels and complex destination/reference constructions.

Final dataset: **1,075 records** — 717 train, 181 validation, 177 test. The 75 added examples are 100% synthetic candidates, not real user utterances. The exact partner-supplied Madhapur/ATM regression query is held out from training. The latest recomputed data audit reports 0 invalid BIO transitions, 0 token/tag/offset mismatches, 0 entity-offset mismatches, 0 normalized exact-duplicate groups and 0 scenario-family split overlaps.

See `data/ner_dataset_audit.json`, `data/dataset_statistics_ner_improved.json` and `evaluation/ner_improvement/final_dataset_audit.json`.

## Model changes

The model remains the same first-order linear-chain CRF architecture and the same feature hashing expected by the partner backend:

- `feature_dim = 4096`;
- the original 15 BIO tags in the original sorted order;
- state-dict keys remain `em`, `trans`, `start`, `end` with shapes `(4096, 15)`, `(15, 15)`, `(15,)`, `(15,)`;
- token feature extraction retains the same word, shape, prefix/suffix, BOS/EOS and neighbouring-token templates.

Training changes are limited to BIO-legal start/transition constraints and utterance-level oversampling of examples containing rare `DIR`/`CONTEXT` entities. The selected checkpoint records epoch 3 and validation macro entity-span F1 0.75598 (the training script selects on macro span F1). The separate validation report gives exact-span micro F1 0.83369 and token macro-F1 0.79570. Illegal `I-` transitions are stored as large negative start/transition scores, so the partner's existing Viterbi decoder enforces BIO legality without editing its code.

## Before/after results

For a paired comparison, the same **176 held-out test rows** are evaluated by both checkpoints. The exact Madhapur/ATM regression row is excluded from this paired subset because an exact copy had appeared in the original baseline's training data; it is evaluated separately against the new checkpoint after the copy was removed from new training data.

| Metric | Existing CRF | Improved CRF |
|---|---:|---:|
| Token accuracy | 71.58% | **85.97%** |
| Token macro-F1 | 22.82% | **70.46%** |
| Exact-span precision | 42.29% | **70.52%** |
| Exact-span recall | 28.43% | **72.05%** |
| **Exact-span F1** | **34.01%** | **71.28%** |
| Invalid BIO predictions | 43 | **0** |

On all 177 held-out rows, including the separate regression row, the final checkpoint scores token accuracy 86.06%, token macro-F1 71.34%, exact-span precision 71.00%, recall 72.51%, and exact-span F1 71.75%, with 0 invalid BIO predictions.

## Entity-wise exact-span metrics (paired 176-row test)

| Entity | Precision | Recall | F1 | Gold support |
|---|---:|---:|---:|---:|
| CONTEXT | 40.00% | 18.18% | 25.00% | 33 |
| DESC | 33.33% | 30.00% | 31.58% | 20 |
| DEST | 66.67% | 79.14% | 72.37% | 139 |
| DIR | 87.50% | 77.78% | 82.35% | 9 |
| DIST | 88.89% | 88.89% | 88.89% | 9 |
| REF | 86.36% | 79.83% | 82.97% | 119 |
| REL | 67.68% | 77.91% | 72.43% | 86 |

## Language-wise exact-span metrics (paired 176-row test)

| Language | Precision | Recall | F1 |
|---|---:|---:|---:|
| Hindi (Devanagari) | 83.56% | 82.43% | 82.99% |
| Romanized Hindi | 59.60% | 61.22% | 60.40% |
| Hinglish | 68.50% | 72.50% | 70.45% |

**Interpretation:** overall NER and BIO validity improved substantially on this synthetic evaluation, but `DESC`, `CONTEXT` and Romanized Hindi remain clear weaknesses. Per-label supports for `DIR` and `DIST` are small, so their scores are uncertain. This is not yet evidence of production/general-user performance.

## Partner's Madhapur/ATM regression case

Input:

`Madhapur mein jo mall hai uske peeche wali building ke paas ek ATM hai, wahan jaana hai.`

The updated checkpoint extracts all seven annotated spans correctly:

- `Madhapur` → `CONTEXT`
- `mall` → `REF`
- `uske peeche` → `REL`
- `wali building` → `DESC`
- `ke paas` → `REL`
- `ATM` → `DEST`
- `wahan` → `CONTEXT`

The saved regression test is `evaluation/ner_improvement/target_regression_example.json`. I also ran the same feature extraction and checkpoint load path from the existing predictor against this query; it returned the seven spans above and `NAVIGATE` intent.

## Intent classification

The existing intent checkpoint is unchanged byte-for-byte and was not retrained. On its existing synthetic test split (162 rows), it reports accuracy **90.74%**, macro-F1 **91.46%**, weighted-F1 **91.18%**. Per-class precision/recall/F1 and the test confusion matrix are provided. This does not reproduce the ~98% Phase 2 figure in the voice note; that earlier result belongs to a different evaluation/checkpoint/split and must not be compared without its original test artefact.

## Backend and routing integration

No backend file is included in this overlay. In the selected `feature/conditional-llm-routing` branch, the existing `backend/model_service.py` returns `intent_confidence`, `intent_probabilities` and `intent_margin`; the router uses these values. The CRF checkpoint is compatible with its current class and feature templates. The intent classifier is unchanged. The overlay does not touch `backend/model_service.py`, `backend/ambiguity.py`, `backend/conditional_router.py`, `backend/ml_response_formatter.py`, `backend/main.py` or `backend/response_normalizer.py`.

**Routing-threshold discrepancy to confirm separately:** `readme1.md` and checked-in `backend/ambiguity.py` specify intent-confidence threshold `0.50` and intent-margin threshold `0.20`, while the supplied audio transcript describes a threshold of `0.75`. Because this task is limited to NER and explicitly protects routing/backend files, I have not changed either threshold. The partner should confirm the intended threshold before changing the architecture configuration.

## Remaining work

- Human review: the annotation workbook is prepared but has not been completed by two independent annotators; no inter-annotator agreement is claimed.
- Real-user data: the current corpus is synthetic, requiring human validation and a representative test set.
- Transformer baselines: the XLM-R/mBERT notebook is included but neither model has been fine-tuned here.
- Runtime integration: apply the patch on `feature/conditional-llm-routing`, then run the included `scripts/smoke_test_model_service.py` and a full local FastAPI/frontend test. The connected GitHub permission available in this session is read-only, so the remote branch has not been pushed.
