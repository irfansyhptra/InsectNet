"""
History Schemas

Pydantic models for prediction history storage and retrieval.
Validates: Requirements 7.4, 15.6
"""

from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
from .prediction import TopPrediction


class HistoryRecord(BaseModel):
    """
    Stored prediction record with image reference.
    
    Represents a single historical prediction including the uploaded image,
    prediction results, and metadata.
    """
    
    id: str = Field(
        ..., 
        description="Unique record identifier (UUID)"
    )
    image_url: str = Field(
        ..., 
        description="Relative URL path to stored image file"
    )
    species: str = Field(
        ..., 
        min_length=1,
        description="Predicted species name"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score of the prediction"
    )
    top_predictions: List[TopPrediction] = Field(
        ...,
        min_length=1,
        description="Top alternative predictions"
    )
    timestamp: datetime = Field(
        ...,
        description="When the prediction was made"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "image_url": "/storage/images/550e8400-e29b-41d4-a716-446655440000.jpg",
                "species": "Ladybug",
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
                    }
                ],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HistoryListResponse(BaseModel):
    """
    List of history records with metadata.
    
    Includes pagination and filtering information alongside the actual records.
    """
    
    records: List[HistoryRecord] = Field(
        ...,
        description="List of history records sorted by timestamp (newest first)"
    )
    total: int = Field(
        ..., 
        ge=0,
        description="Total number of records in the database"
    )
    filtered: int = Field(
        ..., 
        ge=0,
        description="Number of records after applying filters"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "records": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "image_url": "/storage/images/550e8400-e29b-41d4-a716-446655440000.jpg",
                        "species": "Ladybug",
                        "confidence": 0.94,
                        "top_predictions": [
                            {
                                "species": "Ladybug",
                                "confidence": 0.94,
                                "scientific_name": "Coccinellidae"
                            }
                        ],
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "total": 150,
                "filtered": 1
            }
        }
