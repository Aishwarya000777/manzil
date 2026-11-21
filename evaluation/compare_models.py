
import json
import time
import csv
from pathlib import Path
from typing import Any

from backend.model_service import model_service
from backend.llm_service import (
    analyze_navigation_query,
    analyze_navigation_query_llm_only,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "evaluation" / "test_dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    records = []

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                )

            if "query" not in record or "ground_truth" not in record:
                raise ValueError(
                    f"Line {line_number} must contain query and ground_truth."
                )

            records.append(record)

    if not records:
        raise ValueError("The evaluation dataset is empty.")

    return records


# ============================================================
# OUTPUT NORMALIZATION
# ============================================================

def extract_text(value: Any):
    """
    Convert destination/reference objects or strings
    into comparable text values.
    """

    if value is None:
        return None

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        for key in ("category", "name", "text", "value"):
            if value.get(key):
                return str(value[key]).strip()

        return None

    return str(value).strip() or None


def normalize_relationships(relationships):
    normalized = []

    if not isinstance(relationships, list):
        return normalized

    for relation in relationships:

        if isinstance(relation, str):
            normalized.append({
                "relationship_type": relation.upper(),
                "target": None,
                "reference": None
            })

        elif isinstance(relation, dict):
            normalized.append({
                "relationship_type": (
                    relation.get("relationship_type")
                    or relation.get("type")
                    or "UNKNOWN"
                ).upper(),

                "target": extract_text(
                    relation.get("target")
                ),

                "reference": extract_text(
                    relation.get("reference")
                )
            })

    return normalized


def canonicalize(result):
    """
    Convert model output into a common evaluation structure.
    """

    if not isinstance(result, dict):
        raise ValueError("Model returned an invalid output.")

    return {
        "intent": result.get("intent"),

        "destination": extract_text(
            result.get("destination")
        ),

        "reference_location": extract_text(
            result.get("reference_location")
        ),

        "spatial_relationships": normalize_relationships(
            result.get("spatial_relationships", [])
        ),

        "ambiguity": result.get("ambiguity"),

        "clarification_required": result.get(
            "clarification_required"
        )
    }


# ============================================================
# 1. ML-ONLY PREDICTIONS
# ============================================================

def run_ml_only(query: str):
    raw = model_service.predict(query)

    entities = raw.get("entities", [])

    destinations = [
        entity["text"]
        for entity in entities
        if entity.get("label") == "DEST"
    ]

    references = [
        entity["text"]
        for entity in entities
        if entity.get("label") == "REF"
    ]

    # ML currently extracts entity spans but does not
    # independently resolve spatial relationships or ambiguity.

    result = {
        "intent": raw.get("intent"),

        "destination": (
            " ".join(destinations)
            if destinations else None
        ),

        "reference_location": (
            " ".join(references)
            if references else None
        ),

        "spatial_relationships": [],

        "ambiguity": None,

        "clarification_required": None,

        "ml_entities": entities,

        "bio_tags": raw.get("bio_tags"),

        "tokens": raw.get("tokens"),

        "intent_confidence": raw.get("intent_confidence")
    }

    return result


# ============================================================
# 2. LLM-ONLY PREDICTIONS
# ============================================================

def run_llm_only(query: str):
    raw = analyze_navigation_query_llm_only(query)
    return canonicalize(raw)


# ============================================================
# 3. HYBRID ML + LLM PREDICTIONS
# ============================================================

def run_hybrid(query: str):
    ml_predictions = model_service.predict(query)

    raw = analyze_navigation_query(
        query,
        ml_predictions
    )

    result = canonicalize(raw)

    # Preserve ML validation details for analysis.
    result["ml_validation"] = raw.get("ml_validation")

    return result


# ============================================================
# SAVE JSONL PREDICTIONS
# ============================================================

def save_jsonl(filename, records):
    output_path = RESULTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Saved: {output_path}")


# ============================================================
# SAVE LATENCY REPORT
# ============================================================

def save_latency_report(latency_records):
    output_path = RESULTS_DIR / "latency_comparison.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "query_number",
                "latency_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(latency_records)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n======================================")
    print("       MANZIL MODEL EVALUATION")
    print("======================================\n")

    dataset = load_dataset()

    print(f"Dataset size: {len(dataset)} queries")

    print("\nLoading trained ML models...")

    model_service.load_models()

    if not model_service.ready:
        raise RuntimeError("ML models failed to load.")

    print("ML models loaded successfully.")

    ml_results = []
    llm_results = []
    hybrid_results = []
    latency_records = []

    for index, record in enumerate(dataset, start=1):

        query = record["query"]
        ground_truth = record["ground_truth"]

        print(f"\n[{index}/{len(dataset)}] {query}")

        # --------------------------------------------
        # ML ONLY
        # --------------------------------------------

        start = time.perf_counter()

        ml_prediction = run_ml_only(query)

        ml_latency = time.perf_counter() - start

        ml_results.append({
            "query": query,
            "ground_truth": ground_truth,
            "prediction": ml_prediction
        })

        latency_records.append({
            "model": "ML-only",
            "query_number": index,
            "latency_seconds": round(ml_latency, 4)
        })

        print(f"ML-only: {ml_latency:.3f}s")

        # --------------------------------------------
        # LLM ONLY
        # --------------------------------------------

        start = time.perf_counter()

        llm_prediction = run_llm_only(query)

        llm_latency = time.perf_counter() - start

        llm_results.append({
            "query": query,
            "ground_truth": ground_truth,
            "prediction": llm_prediction
        })

        latency_records.append({
            "model": "LLM-only",
            "query_number": index,
            "latency_seconds": round(llm_latency, 4)
        })

        print(f"LLM-only: {llm_latency:.3f}s")

        # --------------------------------------------
        # HYBRID
        # --------------------------------------------

        start = time.perf_counter()

        hybrid_prediction = run_hybrid(query)

        hybrid_latency = time.perf_counter() - start

        hybrid_results.append({
            "query": query,
            "ground_truth": ground_truth,
            "prediction": hybrid_prediction
        })

        latency_records.append({
            "model": "Hybrid",
            "query_number": index,
            "latency_seconds": round(hybrid_latency, 4)
        })

        print(f"Hybrid: {hybrid_latency:.3f}s")

    # --------------------------------------------
    # SAVE ALL PREDICTIONS
    # --------------------------------------------

    save_jsonl("ml_only.jsonl", ml_results)
    save_jsonl("llm_only.jsonl", llm_results)
    save_jsonl("hybrid.jsonl", hybrid_results)

    save_latency_report(latency_records)

    print("\n======================================")
    print("       EVALUATION COMPLETED")
    print("======================================")

    print(f"Queries evaluated: {len(dataset)}")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
