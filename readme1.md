
# Manzil — Conditional LLM Routing Architecture

## 1. Project Overview

Manzil is a Hindi, Romanized Hindi and Hinglish navigation-query understanding system.

It understands conversational navigation queries and extracts structured information such as:

- Navigation intent
- Destination
- Reference locations
- Spatial relationships
- Direction and distance
- Ambiguity and clarification requirements

Manzil focuses on understanding navigation language. It is not a map or route-planning application.

This README documents the architecture implementation contributed on the `feature/conditional-llm-routing` branch.

**Scope:** My contribution focuses on conditional LLM routing and backend architecture. ML model training and dataset enhancement are handled separately by my teammate.

---

## 2. Problem Statement

Previously, every navigation query was processed by the ML models and then sent to Gemini for contextual reasoning.

This meant even simple queries required an LLM API call.

This resulted in:

- Unnecessary LLM API requests.
- Increased response latency.
- Higher API usage.
- Dependency on Gemini availability and quota, even for straightforward queries.

To address this, I implemented a **Conditional LLM Invocation Architecture**.

The new architecture uses ML as the first processing layer and invokes Gemini only when the ML predictions indicate uncertainty or query complexity.

---

## 3. Previous Architecture

### ML + LLM for Every Query

```text
             User Query
                  |
                  v
          Intent Classifier
                  |
                  v
              CRF NER
                  |
                  v
            ML Predictions
                  |
                  v
                Gemini
          (Every Query)
                  |
                  v
       Response Normalization
                  |
                  v
         Structured JSON Output
```

### Limitations

- Gemini was called for every query.
- Simple queries unnecessarily consumed LLM requests.
- The system depended heavily on external API availability.
- Latency and API consumption increased.

---

## 4. New Architecture

### ML-First Conditional LLM Routing

```text
                     User Query
                          |
                          v
                  Existing ML Models
               (Intent Classifier + CRF)
                          |
                          v
                    ML Predictions
                          |
                          v
              Conditional Routing Evaluator
                          |
                +---------+---------+
                |                   |
                v                   v
          ML_ONLY Path          ML_WITH_LLM Path
                |                   |
                v                   v
          ML Formatter       Original Query +
                |            ML Predictions
                |                   |
                |                   v
                |                 Gemini
                |                   |
                |                   v
                |           Contextual Reasoning
                |                   |
                +---------+---------+
                          |
                          v
                Response Normalization
                          |
                          v
                Structured JSON Output
```

### How the New Architecture Works

1. Every incoming query is processed by the existing ML models.
2. The intent classifier predicts the navigation intent.
3. The CRF model extracts entities using BIO tagging.
4. The routing evaluator examines the ML predictions.
5. Based on confidence, entity completeness and query complexity, it selects a processing path.
6. Simple and sufficiently confident queries are handled through the ML-only path.
7. Uncertain or spatially complex queries are sent to Gemini along with the original query and ML predictions.
8. Both paths produce a normalized structured response.

The key difference is that **Gemini is no longer called unconditionally.**

---

## 5. Conditional Routing Logic

The routing evaluator is implemented in:

`backend/ambiguity.py`

It evaluates multiple signals instead of relying only on intent confidence.

### Signals Used

| Signal | Routing condition |
|---|---|
| Intent | Unsupported or uncertain intent triggers escalation |
| Intent confidence | Below 0.50 triggers escalation |
| Intent margin | Below 0.20 triggers escalation |
| Destination entities | Missing both DEST and DESC triggers escalation |
| Reference entities | REF detection triggers escalation |
| Spatial relationships | REL detection triggers escalation |

### Intent Confidence

The intent classifier produces probabilities for each supported intent.

For example:

```json
{
  "ETA_QUERY": 0.18,
  "NAVIGATE": 0.56,
  "OTHER": 0.09,
  "SEARCH": 0.16
}
```

The highest probability represents the predicted intent confidence.

### Intent Probability Margin

The probability margin measures the difference between the highest and second-highest intent probabilities.

```text
intent_margin =
    highest_probability - second_highest_probability
```

A smaller margin indicates that competing intent predictions are closer.

### Current Experimental Thresholds

```text
Intent confidence threshold = 0.50
Intent margin threshold     = 0.20
```

If any escalation condition is met:

```text
processing_path = ML_WITH_LLM
```

Otherwise:

```text
processing_path = ML_ONLY
```

**Note:** These are initial experimental thresholds, not calibrated or scientifically optimized values. They should be evaluated and tuned using a representative validation dataset.

The CRF model does not currently expose calibrated entity-level confidence, so the router does not invent entity confidence scores.

---

## 6. Files Implemented or Modified

| File | Responsibility |
|---|---|
| `backend/model_service.py` | Exposes intent probabilities and probability margin along with existing ML predictions. |
| `backend/ambiguity.py` | Implements the conditional routing evaluator. |
| `backend/conditional_router.py` | Selects ML-only or ML-plus-LLM processing. |
| `backend/ml_response_formatter.py` | Converts ML predictions into a structured response for the ML-only path. |
| `backend/main.py` | Integrates conditional routing into the `/api/analyze` endpoint. |
| `backend/response_normalizer.py` | Normalizes results from both paths into a common response structure. |

### Important Functions

**1. `model_service.predict(query)`**

- Runs the intent classifier.
- Runs CRF-based entity extraction.
- Returns intent confidence.
- Returns intent probability distribution.
- Calculates the probability margin.
- Returns tokens, BIO tags and extracted entities.

**2. `evaluate_ml_predictions(ml_predictions)`**

- Evaluates ML predictions.
- Checks intent confidence and margin.
- Checks destination information.
- Checks reference and spatial entities.
- Determines whether Gemini should be invoked.

**3. `format_ml_response(ml_predictions)`**

- Converts ML predictions into a structured navigation response.
- Used when the query is handled through the ML-only path.

**4. `process_navigation_query(query, ml_predictions)`**

- Executes the conditional routing decision.
- Returns the ML-only response when escalation is unnecessary.
- Invokes Gemini when the evaluator identifies uncertainty or complexity.

**5. `normalize_response(result, query)`**

- Normalizes the output from either processing path.
- Preserves routing metadata and ML predictions.
- Maintains a consistent API response structure.

---

## 7. Three Approaches for Evaluation

We consider three processing strategies to compare the effectiveness of the architecture.

### A. ML-Only Approach

```text
Query
  |
  v
Intent Classifier + CRF NER
  |
  v
ML Predictions
  |
  v
ML Response Formatter
  |
  v
Structured Output
```

Characteristics:

- Uses only the trained ML models.
- Does not invoke Gemini.
- Provides a baseline for evaluating the ML models.
- Has no LLM API dependency during inference.
- May struggle with implicit references, nested descriptions, self-corrections and complex spatial expressions.

### B. Hybrid / Conditional ML + LLM Approach

```text
Query
  |
  v
ML Models
  |
  v
Routing Evaluator
  |
  +-----------------------+
  |                       |
  v                       v
Simple Query         Complex Query
  |                       |
  v                       v
ML Formatter       Gemini + ML Predictions
  |                       |
  +-----------+-----------+
              |
              v
       Response Normalizer
              |
              v
       Structured Output
```

Characteristics:

- ML processes every query first.
- Simple queries are handled without Gemini.
- Complex or uncertain queries are escalated to Gemini.
- Gemini receives both the original query and ML predictions.
- Aims to balance response quality, latency and API usage.

**This is the architecture implemented in my contribution.**

### C. LLM-Only Approach

```text
Query
  |
  v
Gemini
  |
  v
Response Normalization
  |
  v
Structured Output
```

Characteristics:

- Sends the original query directly to Gemini.
- Does not use ML predictions as input.
- Provides a baseline for comparing contextual language understanding.
- Requires an LLM request for every query.

**Note:** ML-only and LLM-only are comparison strategies for evaluation. The implementation delivered in this branch is the conditional ML-first architecture, not a separate production LLM-only endpoint.

---

## 8. Evaluation Methodology

To compare the three approaches fairly, use the same held-out test dataset for all three.

The dataset should include:

- Simple navigation queries.
- Complex navigation queries.
- Hindi queries.
- Romanized Hindi queries.
- Hinglish queries.
- Queries containing spatial relationships.
- Queries containing multiple reference locations.
- Ambiguous queries requiring clarification.

### Evaluation Parameters

| Parameter | What We Evaluate |
|---|---|
| Intent Accuracy | Whether the predicted intent is correct. |
| Entity Extraction | Whether destinations and references are extracted correctly. |
| Entity Precision, Recall and F1 | Quality of entity extraction. |
| Spatial Relationship Accuracy | Whether relationships such as NEAR, BEHIND and OPPOSITE are correct. |
| Clarification Accuracy | Whether clarification is requested when necessary. |
| Structured Output | Whether the final JSON represents the expected information. |
| Latency | Time taken to process each query. |
| LLM Usage | Number of queries requiring Gemini. |

### Results Comparison Table

Fill this table with actual experimental measurements.

| Metric | ML-Only | Hybrid | LLM-Only |
|---|---|---|---|
| Intent Accuracy | To measure | To measure | To measure |
| Entity Precision | To measure | To measure | To measure |
| Entity Recall | To measure | To measure | To measure |
| Entity F1 | To measure | To measure | To measure |
| Spatial Relationship Accuracy | To measure | To measure | To measure |
| Clarification Accuracy | To measure | To measure | To measure |
| Average Latency | To measure | To measure | To measure |
| LLM Calls | 0 | Actual routed calls | One per query |

For the hybrid architecture, we should additionally measure the percentage of queries handled entirely by ML.

This helps establish whether conditional routing reduces unnecessary LLM requests without compromising navigation understanding.

**Important:** No accuracy or latency improvement should be claimed until the three approaches have been evaluated on the same test set.

---

## 9. Local Setup and Execution

### Requirements

- Python 3.11
- Existing project dependencies
- Trained model files in the expected `models/` directory
- Gemini API key for the LLM fallback path

### Step 1: Navigate to the Project

Windows PowerShell:

```powershell
cd "C:\Users\<YOUR_WINDOWS_USERNAME>\manzil-nlp-navigation"
```

### Step 2: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Configure Gemini API Key

Add the Gemini API key to your local `.env` file using the environment variable expected by `backend/llm_service.py`.

Do not commit `.env` or expose API keys in source code.

### Step 5: Start the Backend

```powershell
uvicorn backend.main:app --reload
```

Backend runs at:

```text
http://localhost:8000
```

---

## 10. Localhost URLs

| URL | Purpose |
|---|---|
| http://localhost:8000/ | Backend root endpoint |
| http://localhost:8000/health | Backend health check |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/openapi.json | OpenAPI specification |

### Frontend (Optional)

From the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL displayed in the terminal.

During development, the frontend ran on:

```text
http://localhost:5174
```

Port 5173 was already occupied.

---

## 11. How to Test the Implementation

The easiest way to test the backend is through Swagger.

Open:

```text
http://localhost:8000/docs
```

Expand the required endpoint, select **Try it out**, enter the request body and click **Execute**.

### A. Health Check

Endpoint:

```text
GET /health
```

Checks whether the backend is running.

### B. ML Prediction Endpoint

Endpoint:

```text
POST /api/ml-predict
```

Example request:

```json
{
  "query": "Mujhe airport jaana hai"
}
```

Inspect:

- `intent`
- `intent_confidence`
- `intent_probabilities`
- `intent_margin`
- `tokens`
- `bio_tags`
- `entities`

This endpoint lets us inspect the raw ML predictions before conditional routing.

### C. Conditional Analysis Endpoint

Endpoint:

```text
POST /api/analyze
```

**Test 1: Simple Navigation Query**

```json
{
  "query": "Mujhe airport jaana hai"
}
```

Expected routing for this example, if it satisfies the current evaluator:

```json
{
  "processing_path": "ML_ONLY",
  "llm_invoked": false
}
```

**Test 2: Complex Spatial Query**

```json
{
  "query": "Madhapur mein jo mall hai uske peeche wali building ke paas ek ATM hai, wahan jaana hai"
}
```

This query contains nested reference and spatial information.

It should trigger the LLM fallback when the evaluator detects the relevant escalation conditions.

Illustrative successful hybrid response metadata:

```json
{
  "processing_path": "ML_WITH_LLM",
  "llm_invoked": true
}
```

These JSON snippets show the routing fields, not the complete API response.

### Important Response Fields

| Field | Meaning |
|---|---|
| `processing_path` | Selected processing path: ML_ONLY or ML_WITH_LLM. |
| `llm_invoked` | Whether the LLM was invoked for the successful response. |
| `routing_reasons` | Reasons for selecting the LLM fallback. |
| `intent` | Predicted navigation intent. |
| `destination` | Extracted destination. |
| `reference_location` | Extracted reference location. |
| `spatial_relationships` | Extracted spatial relationships. |
| `clarification_required` | Whether clarification is needed. |
| `ml_predictions` | Original ML predictions retained in the response. |

### D. Speech Transcription Endpoint

Endpoint:

```text
POST /api/transcribe
```

Upload an audio file through Swagger.

The endpoint transcribes the audio, and the resulting text can be passed into the navigation analysis pipeline.

---

## 12. Understanding API Errors

During development, the complex query reached the Gemini fallback path but returned:

```text
429 RESOURCE_EXHAUSTED
```

The reason was that the Gemini 2.5 Flash free-tier daily request quota had been exhausted.

This indicates:

- ML prediction completed.
- The routing evaluator selected the LLM path.
- The backend attempted to invoke Gemini.
- Gemini rejected the request because of the quota limit.

It does not mean Gemini successfully generated a response.

A successful hybrid response should be tested again when API quota is available.

---

## 13. Important Implementation Considerations

1. The confidence and margin thresholds are experimental and should be tuned using validation data.
2. CRF entity predictions do not currently provide calibrated entity-level confidence.
3. Entity detection does not guarantee correct semantic-role assignment. For example, a reference phrase could be incorrectly tagged as a destination by the current ML model.
4. The ML-only formatter structures available ML predictions but does not perform the same contextual reasoning as Gemini.
5. The fallback path is intended to handle uncertain and complex queries.
6. Model enhancement and dataset improvement remain separate from this architecture contribution.
7. Final accuracy and efficiency claims require evaluation on a representative held-out dataset.

---

## 14. Git Information

Implementation branch:

```text
feature/conditional-llm-routing
```

Commit message:

```text
Implement conditional LLM routing for navigation queries
```

---

## 15. Final Testing Checklist

- [ ] Backend starts successfully.
- [ ] `/health` returns a successful response.
- [ ] `/api/ml-predict` returns intent probabilities and margin.
- [ ] Simple queries are routed to ML_ONLY.
- [ ] ML-only responses have `llm_invoked: false`.
- [ ] Complex queries are routed to ML_WITH_LLM.
- [ ] Routing reasons are returned correctly.
- [ ] Successful Gemini fallback response is validated when quota is available.
- [ ] Responses from both paths follow the common normalized schema.
- [ ] ML-only, Hybrid and LLM-only are evaluated on the same test dataset.
- [ ] Accuracy, latency and LLM usage measurements are recorded.

---

## 16. Summary of My Contribution

My contribution was implementing the **Conditional LLM Invocation Architecture** for Manzil.

Previously, every query went through ML prediction followed by Gemini reasoning.

I changed this into an ML-first architecture where:

- Every query is processed by the existing ML models.
- Intent confidence and probability margin are calculated.
- A routing evaluator checks uncertainty, entity completeness and spatial complexity.
- Straightforward queries are handled using the ML-only path.
- Complex or uncertain queries are escalated to Gemini.
- Gemini receives both the original query and ML predictions.
- Both paths return a common normalized response structure.

The main objective is to reduce unnecessary LLM calls while preserving contextual reasoning for difficult navigation queries.

The architecture provides three evaluation strategies: ML-only, Hybrid and LLM-only. Their relative accuracy, latency and efficiency should be established through controlled evaluation on the same test dataset.
