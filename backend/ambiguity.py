
def detect_ambiguity(result):
    """
    Detect destination ambiguity after semantic analysis.
    """

    destination = result.get("destination") or {}

    category = destination.get("category")
    description = destination.get("description")

    relationships = result.get("spatial_relationships") or []

    reasons = []

    if not category and not description:
        reasons.append("Destination is missing")
        status = "UNRESOLVED"

    elif not category:
        reasons.append("Destination category is unclear")
        status = "PARTIALLY_IDENTIFIED"

    else:
        status = "DESCRIBED"

    clarification_required = bool(reasons)

    return {
        "ambiguity": "POSSIBLE" if reasons else "NONE",
        "resolution_status": status,
        "clarification_required": clarification_required,
        "clarification_question": (
            result.get("clarification_question")
            if clarification_required else None
        ),
        "ambiguity_reasons": reasons,
        "relationship_count": len(relationships)
    }


def evaluate_ml_predictions(ml_predictions):
    """
    Decide whether ML predictions can be handled directly
    or require Gemini contextual reasoning.

    Initial heuristic thresholds must be validated experimentally.
    """

    reasons = []

    intent = ml_predictions.get("intent")
    confidence = ml_predictions.get("intent_confidence")
    margin = ml_predictions.get("intent_margin")

    entities = ml_predictions.get("entities") or []

    labels = {
        entity.get("label")
        for entity in entities
        if entity.get("label")
    }

    supported_intents = {
        "NAVIGATE",
        "SEARCH",
        "ETA_QUERY"
    }

    # 1. Check supported intent
    if intent not in supported_intents:
        reasons.append("Unsupported or uncertain intent")

    # 2. Check probability confidence
    if confidence is None:
        reasons.append("Intent confidence unavailable")
    elif confidence < 0.50:
        reasons.append("Low intent confidence")

    # 3. Check probability margin
    if margin is None:
        reasons.append("Intent probability margin unavailable")
    elif margin < 0.20:
        reasons.append("Competing intent predictions")

    # 4. Check destination extraction
    has_destination = "DEST" in labels
    has_description = "DESC" in labels

    if not has_destination and not has_description:
        reasons.append("Destination entity missing")

    # 5. Check contextual references
    has_reference = "REF" in labels
    has_relationship = "REL" in labels

    if has_reference or has_relationship:
        reasons.append("Reference-based or spatial query")

    invoke_llm = bool(reasons)

    return {
        "invoke_llm": invoke_llm,
        "processing_path": (
            "ML_WITH_LLM" if invoke_llm else "ML_ONLY"
        ),
        "routing_reasons": reasons,
        "intent_confidence": confidence,
        "intent_margin": margin,
        "detected_entities": sorted(labels)
    }
