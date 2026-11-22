import tempfile
import os

from fastapi import UploadFile, File, HTTPException
from backend.speech_service import transcribe_audio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.model_service import model_service
from backend.schemas import NavigationRequest, NavigationResponse
from backend.conditional_router import process_navigation_query
from backend.response_normalizer import normalize_response
from backend.destination_engine import extract_destination_context
from backend.semantic_engine import normalize_relationships
from backend.ambiguity import detect_ambiguity

app = FastAPI(
    title="Manzil API",
    description="Hindi and Hinglish Navigation Query Understanding System",
    version="1.0.0"
)

@app.on_event("startup")
def load_ml_models():
    model_service.load_models()

@app.post("/api/ml-predict")
def ml_predict(request: NavigationRequest):
    try:
        predictions = model_service.predict(request.query)
        return predictions

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ML prediction failed: {str(e)}"
        )    

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Manzil API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Manzil Backend"
    }



@app.post("/api/analyze", response_model=NavigationResponse)
def analyze_query(request: NavigationRequest):
    try:
        # 1. Run ML
        ml_predictions = model_service.predict(request.query)

        # 2. Conditional ML / Gemini routing
        result = process_navigation_query(
            request.query,
            ml_predictions
        )

        # 3. Normalize both processing paths
        normalized_result = normalize_response(
            result,
            request.query
        )

        return normalized_result

    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exceeded. Please retry after the quota resets."
            )

        if "503" in error_message or "UNAVAILABLE" in error_message:
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily unavailable. Please retry shortly."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Navigation analysis failed: {error_message}"
        )

@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid audio file."
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )

    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Audio file is too large. Maximum size is 20 MB."
        )

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        result = transcribe_audio(temp_path)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)