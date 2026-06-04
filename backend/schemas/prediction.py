"""
Prediction Schemas

Pydantic models for prediction API requests and responses.
Validates: Requirements 4.3, 15.6
"""

from typing import List
from datetime import datetime
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    """Single prediction alternative with confidence score."""
    
    species: str = Field(
        ..., 
        min_length=1,
        description="Species common name"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    scientific_name: str | None = Field(
        None, 
        description="Scientific name if available"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "species": "Ladybug",
                "confidence": 0.94,
                "scientific_name": "Coccinellidae"
            }
        }


class PredictionResponse(BaseModel):
    """Primary prediction response with species identification results."""
    
    species: str = Field(
        ..., 
        min_length=1,
        description="Predicted species name"
    )
    scientific_name: str | None = Field(
        None, 
        description="Scientific name of the species"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the primary prediction"
    )
    top_predictions: List[TopPrediction] = Field(
        ..., 
        min_length=1,
        description="Top 5 alternative predictions with confidence scores"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when prediction was generated"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "species": "Ladybug",
                "scientific_name": "Coccinellidae",
                "confidence": 0.94,
                "top_predictions": [
                    {
                        "species": "Ladybug",
                        "confidence": 0.94,
                        "scientific_name": "Coccinellidae"
                    },
                    {
                        "species": "Asian Lady Beetle",
                        "confidence": 0.03,
                        "scientific_name": "Harmonia axyridis"
                    },
                    {
                        "species": "Convergent Lady Beetle",
                        "confidence": 0.02,
                        "scientific_name": "Hippodamia convergens"
                    },
                    {
                        "species": "Seven-spotted Ladybug",
                        "confidence": 0.005,
                        "scientific_name": "Coccinella septempunctata"
                    },
                    {
                        "species": "Two-spotted Lady Beetle",
                        "confidence": 0.003,
                        "scientific_name": "Adalia bipunctata"
                    }
                ],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
