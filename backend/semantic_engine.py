
SUPPORTED_RELATIONS = {
    "NEAR",
    "OPPOSITE",
    "IN_FRONT_OF",
    "BEHIND",
    "LEFT_OF",
    "RIGHT_OF",
    "NEXT_TO",
    "BETWEEN",
    "INSIDE",
    "BEYOND",
    "AFTER",
    "BEFORE"
}


def normalize_relationships(relationships):
    normalized = []

    for relationship in relationships or []:
        relation = relationship.get("relation", "").upper().strip()

        if relation not in SUPPORTED_RELATIONS:
            continue

        normalized.append({
            "relation": relation,
            "target": relationship.get("target"),
            "direction": relationship.get("direction"),
            "distance": relationship.get("distance")
        })

    return normalized
