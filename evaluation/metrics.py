
import json
import csv
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# ============================================================
# BASIC METRICS
# ============================================================

def normalize(value):
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip().lower()

    return str(value).strip().lower()


def classification_metrics(ground_truth, predictions):
    """
    Accuracy, macro precision, macro recall and macro F1.
    """

    if not ground_truth:
        return None

    labels = sorted(
        set(ground_truth) | set(predictions),
        key=str
    )

    accuracy = sum(
        normalize(g) == normalize(p)
        for g, p in zip(ground_truth, predictions)
    ) / len(ground_truth)

    precisions = []
    recalls = []
    f1_scores = []

    for label in labels:
        tp = sum(
            normalize(g) == normalize(label)
            and normalize(p) == normalize(label)
            for g, p in zip(ground_truth, predictions)
        )

        fp = sum(
            normalize(g) != normalize(label)
            and normalize(p) == normalize(label)
            for g, p in zip(ground_truth, predictions)
        )

        fn = sum(
            normalize(g) == normalize(label)
            and normalize(p) != normalize(label)
            for g, p in zip(ground_truth, predictions)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(
            sum(precisions) / len(precisions), 4
        ),
        "macro_recall": round(
            sum(recalls) / len(recalls), 4
        ),
        "macro_f1": round(
            sum(f1_scores) / len(f1_scores), 4
        ),
        "sample_count": len(ground_truth)
    }


def exact_match_metrics(ground_truth, predictions):
    if not ground_truth:
        return None

    correct = sum(
        normalize(g) == normalize(p)
        for g, p in zip(ground_truth, predictions)
    )

    return {
        "exact_match_accuracy": round(
            correct / len(ground_truth), 4
        ),
        "correct": correct,
        "total": len(ground_truth)
    }


# ============================================================
# SPATIAL RELATIONSHIP METRICS
# ============================================================

def relationship_set(relationships):
    result = set()

    for relation in relationships or []:
        if not isinstance(relation, dict):
            continue

        relation_type = normalize(
            relation.get("relationship_type")
        )

        target = normalize(relation.get("target"))
        reference = normalize(relation.get("reference"))

        if relation_type and target and reference:
            result.add((
                relation_type,
                target,
                reference
            ))

    return result


def relationship_metrics(ground_truth, predictions):

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gt, pred in zip(ground_truth, predictions):

        expected = relationship_set(gt)
        actual = relationship_set(pred)

        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative
    }


# ============================================================
# ENTITY EXTRACTION METRICS
# ============================================================

def entity_set(entities):
    result = set()

    for entity in entities or []:
        if not isinstance(entity, dict):
            continue

        text = normalize(entity.get("text"))
        label = normalize(entity.get("label"))

        if text and label:
            result.add((text, label))

    return result


def entity_metrics(ground_truth, predictions):

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gt, pred in zip(ground_truth, predictions):

        expected = entity_set(gt)
        actual = entity_set(pred)

        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "note": "Exact entity text and label match; not token-level BIO span evaluation."
    }


# ============================================================
# LOAD EXPERIMENT OUTPUTS
# ============================================================

def load_jsonl(filename):

    path = RESULTS_DIR / filename
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


# ============================================================
# EVALUATE ONE APPROACH
# ============================================================

def evaluate_approach(filename, approach):

    records = load_jsonl(filename)

    gt_intent = []
    pred_intent = []

    gt_destination = []
    pred_destination = []

    gt_reference = []
    pred_reference = []

    gt_ambiguity = []
    pred_ambiguity = []

    gt_clarification = []
    pred_clarification = []

    gt_relationships = []
    pred_relationships = []

    gt_entities = []
    pred_entities = []

    for record in records:

        gt = record["ground_truth"]
        pred = record["prediction"]

        if gt.get("intent") is not None:
            gt_intent.append(gt["intent"])
            pred_intent.append(pred.get("intent"))

        if gt.get("destination") is not None:
            gt_destination.append(gt["destination"])
            pred_destination.append(pred.get("destination"))

        if gt.get("reference_location") is not None:
            gt_reference.append(gt["reference_location"])
            pred_reference.append(
                pred.get("reference_location")
            )

        if gt.get("ambiguity") is not None:
            gt_ambiguity.append(gt["ambiguity"])
            pred_ambiguity.append(pred.get("ambiguity"))

        if gt.get("clarification_required") is not None:
            gt_clarification.append(
                gt["clarification_required"]
            )
            pred_clarification.append(
                pred.get("clarification_required")
            )

        gt_relationships.append(
            gt.get("spatial_relationships", [])
        )

        pred_relationships.append(
            pred.get("spatial_relationships", [])
        )

        if approach == "ML-only":
            gt_entities.append(gt.get("entities", []))
            pred_entities.append(
                pred.get("ml_entities", [])
            )

    report = {
        "approach": approach,
        "query_count": len(records),

        "intent": classification_metrics(
            gt_intent, pred_intent
        ),

        "destination": exact_match_metrics(
            gt_destination, pred_destination
        ),

        "reference_location": exact_match_metrics(
            gt_reference, pred_reference
        ),

        "spatial_relationships": relationship_metrics(
            gt_relationships, pred_relationships
        ),

        "ambiguity": (
            classification_metrics(
                gt_ambiguity, pred_ambiguity
            )
            if approach != "ML-only" else None
        ),

        "clarification": (
            exact_match_metrics(
                gt_clarification, pred_clarification
            )
            if approach != "ML-only" else None
        ),

        "entity_extraction": (
            entity_metrics(gt_entities, pred_entities)
            if approach == "ML-only" else None
        )
    }

    return report


# ============================================================
# LATENCY SUMMARY
# ============================================================

def calculate_latency():

    path = RESULTS_DIR / "latency_comparison.csv"

    if not path.exists():
        return {}

    values = {}

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            model = row["model"]
            latency = float(row["latency_seconds"])

            values.setdefault(model, []).append(latency)

    return {
        model: {
            "average_seconds": round(
                sum(times) / len(times), 4
            ),
            "total_seconds": round(sum(times), 4),
            "query_count": len(times)
        }
        for model, times in values.items()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    approaches = [
        ("ml_only.jsonl", "ML-only"),
        ("llm_only.jsonl", "LLM-only"),
        ("hybrid.jsonl", "Hybrid")
    ]

    reports = {}

    for filename, approach in approaches:

        print(f"\nEvaluating {approach}...")

        reports[approach] = evaluate_approach(
            filename,
            approach
        )

    final_report = {
        "evaluation": reports,
        "latency": calculate_latency()
    }

    output_path = RESULTS_DIR / "comparison_metrics.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")
    print("       METRICS GENERATED")
    print("======================================")

    for approach, report in reports.items():

        print(f"\n{approach}")

        for metric in (
            "intent",
            "destination",
            "reference_location",
            "spatial_relationships",
            "ambiguity",
            "clarification",
            "entity_extraction"
        ):
            print(f"{metric}: {report[metric]}")

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()

