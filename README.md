# Manzil — Hindi/Hinglish Navigation Query Understanding

Manzil is an NLP prototype for understanding navigation-related queries in Hindi, Romanized Hindi, and Hinglish. It explores intent classification, BIO-based entity extraction, and sentence-level semantic representations of destinations and spatial relations.

## Project status and research integrity

This repository contains a **synthetic-data research pilot and baseline implementation**. The 1,000-row Manzil v2 pilot contains generated candidate utterances. The original 3,024-row dataset is preserved separately under `source_original/`. Combined index files are provided for convenience.

The included intent classifier and CRF baseline were trained and evaluated on the pilot's predefined train/validation/test split. Reported scores are preliminary results on synthetic candidate data and should not be interpreted as real-world performance.

Human validation has **not** been completed. The annotation workbook is a review tool, not evidence of completed annotation. XLM-R and mBERT are **not yet trained**; a Google Colab notebook is included to run those experiments.

## Repository contents

- `manzil_v2_full.jsonl/.csv`: full pilot records.
- `manzil_v2_token_level.jsonl`: token-level BIO NER data.
- `manzil_v2_semantic_level.jsonl`: sentence-level semantic representations.
- `MANZIL_COMBINED_INDEX_4024.csv`: combined index of legacy and pilot rows.
- `source_original/`: preserved original dataset and legacy audit files.
- `models/`: trained intent and CRF baseline models.
- `evaluation/`: validation/test reports and confusion matrices.
- `train_crf.py`, `predict.py`: baseline training and inference.
- `Manzil_Human_Annotation_Workbook_300.xlsx`: independent double-annotation workbook (pending human review).
- `annotation_agreement.py`: agreement calculation after annotator review.
- `Manzil_XLMR_mBERT_Colab_Benchmark.ipynb`: Colab benchmark notebook.
- `HUMAN_REVIEW_AND_TRANSFORMER_GUIDE.md`: detailed next steps.

## Quick start

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the included inference script according to its CLI help / README instructions. The model artifact is a project-specific baseline; use the supplied training script and evaluation files to inspect the implementation and outputs.

## Human annotation

Use the two independent annotator sheets in the workbook. Keep annotators blind to the other person's labels, resolve disagreements in the adjudication sheet, and report agreement statistics. Do not mark examples as human-validated until this process has actually taken place.

## Transformer comparison

Open the Colab notebook, select a GPU runtime, upload the project ZIP/datasets, and run the cells. Record actual outputs; do not report scores before training and evaluation have completed.

## Citations

The project should cite the methods it uses, including:
- Lafferty, McCallum & Pereira (2001), *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*.
- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R).

## License

No license is specified yet. Do not assume the project is open for reuse until a license is added.