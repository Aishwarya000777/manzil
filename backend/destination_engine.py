
def normalize_destination(destination):
    """
    Normalize destination information returned by Gemini.
    """

    if not destination:
        return None

    category = destination.get("category")
    description = destination.get("description")

    return {
        "category": category.strip().lower() if category else None,
        "description": description.strip() if description else None
    }


def extract_destination_context(result):
    """
    Extract destination and reference-location context.
    """

    destination = normalize_destination(
        result.get("destination")
    )

    reference_location = result.get("reference_location")

    if isinstance(reference_location, str):
        reference_location = reference_location.strip()

    return {
        "destination": destination,
        "reference_location": reference_location
    }
