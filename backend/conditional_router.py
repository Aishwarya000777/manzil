
from backend.ambiguity import evaluate_ml_predictions
from backend.ml_response_formatter import format_ml_response
from backend.llm_service import analyze_navigation_query


def process_navigation_query(query, ml_predictions):
    """
    Route a navigation query through either:

    1. ML-only processing for sufficiently clear predictions.
    2. ML + Gemini processing for uncertain or complex predictions.
    """

    # Step 1: Evaluate ML predictions
    routing = evaluate_ml_predictions(ml_predictions)

    # Step 2: Select processing path
    if routing["processing_path"] == "ML_ONLY":

        result = format_ml_response(ml_predictions)

        result["processing_path"] = "ML_ONLY"
        result["llm_invoked"] = False
        result["routing_reasons"] = routing["routing_reasons"]

    else:

        # Pass original query AND ML predictions to Gemini
        result = analyze_navigation_query(
            query,
            ml_predictions
        )

        result["processing_path"] = "ML_WITH_LLM"
        result["llm_invoked"] = True
        result["routing_reasons"] = routing["routing_reasons"]

    # Preserve ML predictions for evaluation and debugging
    result["ml_predictions"] = ml_predictions

    return result
