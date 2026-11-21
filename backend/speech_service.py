
import os
import tempfile
from faster_whisper import WhisperModel

# Load the model only when transcription is first requested
_model = None


def get_model():
    global _model

    if _model is None:
        print("Loading Whisper model...")
        _model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper model loaded successfully!")

    return _model


def transcribe_audio(audio_path: str):
    model = get_model()

    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        beam_size=5,
        vad_filter=True
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return {
        "transcription": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 4)
    }
