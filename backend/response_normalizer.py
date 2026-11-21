
from typing import Any


def normalize_destination(destination: Any) -> dict:
    """Convert destination into a consistent dictionary."""

    if isinstance(destination, str):
        return {
            "category": None,
            "name": None,
            "description": destination
        }

    if isinstance(destination, dict):
        return {
            "category": destination.get("category"),
            "name": destination.get("name"),
            "description": destination.get("description")
        }

    return {
        "category": None,
        "name": None,
        "description": None
    }


def normalize_reference(reference: Any) -> dict | None:
    """Convert reference location into a consistent dictionary."""

    if reference is None:
        return None

    if isinstance(reference, str):
        return {
            "category": None,
            "name": reference,
            "description": None
        }

    if isinstance(reference, dict):
        return {
            "category": reference.get("category"),
            "name": reference.get("name"),
            "description": reference.get("description")
        }

    return None


def normalize_relationships(relationships: Any) -> list:
    """Standardize spatial relationship formats."""

    normalized = []

    if not isinstance(relationships, list):
        return normalized

    relation_keywords = {
        "NEAR": ["near", "close to"],
        "BEHIND": ["behind"],
        "OPPOSITE": ["opposite", "across from"],
        "IN_FRONT_OF": ["in front of"],
        "LEFT_OF": ["left of"],
        "RIGHT_OF": ["right of"],
        "NEXT_TO": ["next to", "beside"],
        "INSIDE": ["inside", "within"],
        "BEYOND": ["beyond"]
    }

    for item in relationships:

        # Handle relationships returned as plain text
        if isinstance(item, str):
            text = item.lower()
            relation = "UNSPECIFIED"

            for relation_name, keywords in relation_keywords.items():
                if any(keyword in text for keyword in keywords):
                    relation = relation_name
                    break

            normalized.append({
                "relationship_type": relation,
                "target": None,
                "reference": None,
                "description": item
            })
            continue

        # Ignore unsupported formats
        if not isinstance(item, dict):
            continue

        # Handle different relationship field names
        relation = (
            item.get("relationship_type")
            or item.get("type")
            or item.get("relation")
            or "UNSPECIFIED"
        )

        target = (
            item.get("target_object")
            or item.get("target_entity")
            or item.get("entity1_description")
            or item.get("target")
        )

        reference = (
            item.get("reference_object")
            or item.get("anchor_entity")
            or item.get("entity2_description")
            or item.get("reference")
        )

        normalized.append({
            "relationship_type": str(relation).upper(),
            "target": target,
            "reference": reference,
            "description": item.get("description")
        })

    return normalized


def normalize_ml_validation(validation: Any) -> dict:
    """Preserve Gemini's ML validation output."""

    if not isinstance(validation, dict):
        validation = {}

    return {
        "intent_retained": validation.get("intent_retained", False),
        "entities_retained": validation.get("entities_retained", []),
        "entities_corrected": validation.get("entities_corrected", []),
        "entities_added": validation.get("entities_added", []),
        "entities_rejected": validation.get("entities_rejected", []),
        "reasoning": validation.get("reasoning", "")
    }


def normalize_response(result: dict, original_query: str) -> dict:
    """Normalize the complete Gemini hybrid response."""

    return {
        "query": original_query,

        "intent": result.get("intent", "UNKNOWN"),

        "destination": normalize_destination(
            result.get("destination")
        ),

        "reference_location": normalize_reference(
            result.get("reference_location")
        ),

        "spatial_relationships": normalize_relationships(
            result.get("spatial_relationships")
        ),

        "ambiguity": result.get("ambiguity", "UNKNOWN"),

        "clarification_required": bool(
            result.get("clarification_required", False)
        ),

        "clarification_question": result.get(
            "clarification_question"
        ),

        "resolution_status": result.get(
            "resolution_status", "PENDING"
        ),

        "ambiguity_reasons": result.get(
            "ambiguity_reasons", []
        ),

        "ml_predictions": result.get("ml_predictions"),

        "ml_validation": normalize_ml_validation(
            result.get("ml_validation")
        )
    }
