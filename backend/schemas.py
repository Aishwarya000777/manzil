
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class NavigationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    language: Optional[str] = "auto"


class Destination(BaseModel):
    category: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class NavigationResponse(BaseModel):
    query: str
    intent: str

    destination: Optional[Any] = None
    reference_location: Optional[Any] = None

    spatial_relationships: List[Any] = Field(
        default_factory=list
    )

    ambiguity: str = "UNKNOWN"
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    resolution_status: str = "PENDING"
    ambiguity_reasons: List[str] = Field(default_factory=list)

    # Preserve original ML predictions
    ml_predictions: Optional[Dict[str, Any]] = None

    # Preserve Gemini's ML validation output
    ml_validation: Optional[Dict[str, Any]] = None

