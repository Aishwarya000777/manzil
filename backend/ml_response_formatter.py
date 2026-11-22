
def format_ml_response(ml_predictions):
    """
    Convert ML predictions into a structured Manzil response.

    This formatter is intended for simple queries that have
    already passed the conditional routing evaluator.
    """

    entities = ml_predictions.get("entities") or []

    def get_entities(label):
        return [
            entity.get("text", "").strip()
            for entity in entities
            if entity.get("label") == label
            and entity.get("text")
        ]

    destinations = get_entities("DEST")
    descriptions = get_entities("DESC")
    references = get_entities("REF")
    directions = get_entities("DIR")
    distances = get_entities("DIST")

    # Build destination
    destination = None

    if destinations:
        destination = {
            "category": destinations[-1],
            "description": descriptions[-1] if descriptions else None
        }

    elif descriptions:
        destination = {
            "category": None,
            "description": descriptions[-1]
        }

    # Build reference location
    reference_location = (
        references[-1] if references else None
    )

    # Build spatial relationships conservatively.
    # CRF entity extraction alone does not establish which
    # reference belongs to which destination.
    spatial_relationships = []

    result = {
        "query": ml_predictions.get("query"),
        "intent": ml_predictions.get("intent"),
        "destination": destination,
        "reference_location": reference_location,
        "spatial_relationships": spatial_relationships,
        "ambiguity": "NONE",
        "resolution_status": (
            "EXPLICIT" if destinations else "PARTIALLY_IDENTIFIED"
        ),
        "clarification_required": False,
        "clarification_question": None,
        "ambiguity_reasons": [],
        "ml_predictions": ml_predictions,
        "ml_validation": {
            "source": "ML_ONLY",
            "directions": directions,
            "distances": distances
        }
    }

    return result
