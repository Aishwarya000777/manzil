
def detect_ambiguity(result):
    destination = result.get("destination") or {}

    category = destination.get("category")
    description = destination.get("description")

    relationships = result.get("spatial_relationships") or []

    reasons = []

    # No destination information
    if not category and not description:
        reasons.append("Destination is missing")
        status = "UNRESOLVED"

    # Some destination information exists, but category is missing
    elif not category:
        reasons.append("Destination category is unclear")
        status = "PARTIALLY_IDENTIFIED"

    # A destination category is available
    else:
        status = "DESCRIBED"

    # Only accept clarification requests when destination
    # information is genuinely insufficient.
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

