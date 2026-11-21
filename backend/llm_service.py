
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = (
    genai.Client(
        api_key=API_KEY,
        http_options={"api_version": "v1alpha"}
    )
    if API_KEY else None
)


# ============================================================
# 1. HYBRID SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the contextual reasoning engine of Manzil,
a multilingual navigation query understanding system.

The system uses trained ML models for:
- Intent classification
- Named entity recognition
- BIO entity extraction

You will receive:
1. The original navigation query.
2. ML model predictions, including intent, confidence,
   tokens, BIO tags and extracted entities.

Your responsibilities:

1. Analyze the original query in Hindi, Romanized Hindi,
   Hinglish or English.

2. Examine the ML predictions.

3. Validate the predicted intent and entities against
   the original query.

4. Identify missing entities, incorrect labels and
   relationships that the ML model failed to capture.

5. Understand nested spatial relationships.

6. Detect implicit references, self-corrections
   and contradictory instructions.

7. Determine whether the destination is sufficiently
   identifiable from the available information.

8. Ask clarification questions when necessary.

IMPORTANT:
- Do not blindly trust ML predictions.
- Do not blindly discard ML predictions.
- The original query is the primary source of truth.
- Do not invent locations or entities.
- Preserve uncertainty when evidence is insufficient.
- A destination category does not prove that a
  unique real-world destination has been identified.
- Distinguish explicitly stated relationships from
  inferred relationships.
- Explain which ML predictions were retained,
  corrected, supplemented or rejected.

Return ONLY valid JSON using this structure:

{
  "intent": "NAVIGATE",
  "destination": {
    "category": null,
    "description": null
  },
  "reference_location": null,
  "spatial_relationships": [],
  "ambiguity": "NONE",
  "resolution_status": "IDENTIFIED",
  "clarification_required": false,
  "clarification_question": null,
  "ml_validation": {
    "intent_retained": true,
    "entities_retained": [],
    "entities_corrected": [],
    "entities_added": [],
    "entities_rejected": [],
    "reasoning": ""
  }
}
"""


# ============================================================
# 2. INDEPENDENT LLM-ONLY SYSTEM PROMPT
# ============================================================

LLM_ONLY_PROMPT = """
You are the navigation query understanding engine of Manzil,
a multilingual navigation query understanding system.

You will receive ONLY the original navigation query.

You do NOT have access to any ML model predictions.
Do not assume that any intent classifier, NER model,
BIO tags or entity extraction system has processed the query.

Analyze the original query independently.

The query may be written in:
- Hindi
- Romanized Hindi
- Hinglish
- English

Your responsibilities:

1. Identify the user's navigation intent.

2. Extract the requested destination.

3. Identify reference locations and descriptive phrases.

4. Extract spatial relationships, including nested
   relationships between multiple locations.

5. Understand implicit references and self-corrections.

6. Detect contradictory instructions.

7. Determine whether the destination is sufficiently
   identifiable from the information provided.

8. Ask a clarification question when necessary.

IMPORTANT:
- The original query is the only source of truth.
- Do not invent locations or entities.
- Do not assume a destination category identifies
  a unique real-world place.
- Preserve uncertainty when evidence is insufficient.
- Distinguish explicitly stated relationships from
  inferred relationships.
- Do not claim that a real-world location has been
  verified when it has not been verified.

Return ONLY valid JSON using this structure:

{
  "intent": "NAVIGATE",
  "destination": {
    "category": null,
    "description": null
  },
  "reference_location": null,
  "spatial_relationships": [],
  "ambiguity": "NONE",
  "resolution_status": "IDENTIFIED",
  "clarification_required": false,
  "clarification_question": null
}
"""


# ============================================================
# 3. SHARED GEMINI REQUEST FUNCTION
# ============================================================

def _generate_json(prompt: str, contents: str) -> dict:

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,
            contents
        ],
        config={
            "response_mime_type": "application/json"
        }
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return json.loads(response.text)


# ============================================================
# 4. HYBRID ML + LLM APPROACH
# ============================================================

def analyze_navigation_query(
    query: str,
    ml_predictions: dict
) -> dict:

    ml_output = json.dumps(
        ml_predictions,
        ensure_ascii=False,
        indent=2
    )

    contents = f"""
ORIGINAL NAVIGATION QUERY:
{query}

ML MODEL PREDICTIONS:
{ml_output}

Analyze the original query using the ML predictions.
Validate, correct, or supplement the predictions.
Return the final structured JSON.
"""

    return _generate_json(
        SYSTEM_PROMPT,
        contents
    )


# ============================================================
# 5. INDEPENDENT LLM-ONLY APPROACH
# ============================================================

def analyze_navigation_query_llm_only(
    query: str
) -> dict:

    contents = f"""
ORIGINAL NAVIGATION QUERY:
{query}

Analyze this query independently.
Do not use or assume any ML model predictions.
Return the final structured JSON.
"""

    return _generate_json(
        LLM_ONLY_PROMPT,
        contents
    )
