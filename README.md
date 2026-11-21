# Manzil: Hindi / Romanized Hindi / Hinglish Navigation Query Understanding

Manzil is an NLP research prototype for understanding navigation-related utterances in Hindi, Romanized Hindi and Hinglish. It includes intent classification, BIO-style entity extraction, sentence-level semantic annotations and an experimental multilingual-transformer benchmark workflow.

> **Research integrity:** The 1,000-row Manzil v2 pilot is a synthetic candidate dataset. Its annotations have not been certified by human annotators. The reported baseline metrics are preliminary results on the pilot's held-out synthetic split, not real-world performance. The annotation workbook is a review tool, not proof of completed human annotation.

## Model status

| Component | Status |
|---|---|
| Character n-gram TF-IDF + Logistic Regression intent baseline | Trained and evaluated |
| Feature-hashed first-order linear-chain CRF NER baseline | Trained and evaluated |
| XLM-R fine-tuning | Colab benchmark workflow prepared; not yet trained |
| mBERT fine-tuning | Colab benchmark workflow prepared; not yet trained |
| Double human annotation / adjudication | Pending; not completed |

### Reported baseline results

**Intent classifier** (162 held-out synthetic pilot records):
- Accuracy: **90.74%**
- Macro precision: **92.25%**
- Macro recall: **92.75%**
- Macro F1: **91.46%**
- Weighted F1: **91.18%**

**CRF entity extraction** (held-out synthetic pilot split):
- Exact-span precision: **73.15%**
- Exact-span recall: **71.17%**
- Exact-span F1: **72.15%**
- Token accuracy: **88.73%**

These figures are reproduced in `evaluation/`. They must be labelled synthetic-pilot results.

## Repository layout

- `src/`: CRF training, inference and inter-annotator agreement scripts.
- `notebooks/`: Colab workflow for XLM-R and mBERT intent/NER comparison.
- `docs/`: annotation guidelines, research workflow and method references.
- `evaluation/`: classification reports, exact-span metrics and confusion matrices.
- `annotation/`: instructions for independent annotators and adjudication.
- `data/dataset_statistics.json`: pilot distribution, split and limitation summary.
- `examples/`: manual smoke-test queries (not an evaluation set).

The full 1,000-row pilot, original 3,024-row source dataset, 300-query Excel annotation workbook and trained checkpoint binaries are in the companion project ZIP shared with this repository. They are not duplicated in this lightweight source repository.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Extract the companion project ZIP into the repository root to create the `data/` and `models/` files required by the scripts.

## Run inference

```bash
python src/predict.py "रेलवे स्टेशन के सामने वाला अस्पताल कहाँ है?"
```

The predictor returns an intent, token sequence, BIO tags and extracted entity spans. Predictions are baseline outputs and should be manually inspected.

## Retrain the CRF

```bash
python src/train_crf.py --data data/manzil_v2_token_level.jsonl --epochs 15 --out models/manzil_linear_crf.pt
```

Training and inference use the same feature templates and checkpoint parameter names.

## Human annotation

The companion ZIP contains `Manzil_Human_Annotation_Workbook_300.xlsx`, with independent sheets for two annotators and a separate adjudication sheet. The workbook is not completed. Two people must independently review the selected examples, disagreements must be adjudicated, and agreement must be calculated before the examples can be described as human-validated. See `docs/ANNOTATION_GUIDELINES.md`.

## XLM-R / mBERT comparison

Open `notebooks/Manzil_XLMR_mBERT_Colab_Benchmark.ipynb` in Google Colab, select a GPU runtime, upload the companion project ZIP and run the cells. It fine-tunes intent and NER heads for XLM-R and mBERT and reports intent and entity metrics. These transformer experiments have not yet been run; do not report scores until actual execution.

## Dataset caveats

The pilot is generated from scenario/syntactic frames and must not be described as collected real-user traffic. A family-aware split reduces direct scenario-family overlap, but does not establish generalization to natural user queries. All generated candidate annotations require human review.

## References

See `docs/REFERENCES.md`. These publications support the methods, not the origin of the generated Manzil utterances.

## License

No open-source license is currently specified.