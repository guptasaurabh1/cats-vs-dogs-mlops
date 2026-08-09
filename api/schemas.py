"""Pydantic schemas for FastAPI request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Inference device")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class label: 'cat' or 'dog'")
    class_id: int = Field(..., description="Numeric class ID (0=cat, 1=dog)")
    confidence: float = Field(..., description="Confidence score of the prediction")
    probabilities: dict = Field(..., description="Per-class probabilities")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
