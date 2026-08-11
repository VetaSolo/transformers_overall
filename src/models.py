"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment: positive or negative")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
