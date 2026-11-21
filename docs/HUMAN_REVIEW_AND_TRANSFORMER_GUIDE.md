# Manzil v2 — human review and transformer benchmark

## Human annotation
The companion project ZIP contains `Manzil_Human_Annotation_Workbook_300.xlsx`, with 300 candidate queries across Devanagari Hindi, Romanized Hindi and Hinglish and ten complexity groups. Its two annotator sheets and adjudication sheet are blank. **No human annotation has been completed.**

1. Give Annotator A and Annotator B separate copies of the workbook. They must work independently and must not see each other's answers.
2. They should label intent, entity spans with exact character offsets, ambiguity, resolution and spatial edges.
3. Export the completed sheets as CSV, retaining the column headers.
4. Run `python src/annotation_agreement.py --a Annotator_A.csv --b Annotator_B.csv`.
5. Adjudicate disagreements and retain raw annotations plus final decisions.

Agreement measures consistency, not correctness. Do not describe the set as human-validated until the actual review and adjudication are complete.

## Transformer comparison
Open `notebooks/Manzil_XLMR_mBERT_Colab_Benchmark.ipynb` in Google Colab, select a GPU runtime, upload the companion full project ZIP and run all cells. The notebook is designed to fine-tune XLM-R and mBERT for intent and token-level NER, evaluate on the existing pilot splits, and report per-language metrics.

The transformer experiments have not yet been run. Do not invent or pre-fill results. Results on the synthetic pilot must be kept separate from later evaluation on a human-adjudicated test set.

## Caveat
The v2 pilot utterances are synthetic candidates, not naturally collected user traffic. The evaluation figures in this repository are preliminary synthetic-split baselines and do not establish real-world generalization.
