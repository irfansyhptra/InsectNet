"""
Error Schemas

Pydantic models for standardized API error responses.
Validates: Requirements 15.6
"""

from datetime import datetime
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standardized error response for all API endpoints.
    
    Provides consistent error structure with error codes, human-readable
    messages, and additional context for debugging.
    """
    
    error: str = Field(
        ..., 
        min_length=1,
        description="Machine-readable error code/type (e.g., 'INVALID_IMAGE_FORMAT')"
    )
    message: str = Field(
        ..., 
        min_length=1,
        description="Human-readable error message for end users"
    )
    detail: str | None = Field(
        None, 
        description="Additional technical details or context about the error"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the error occurred"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "error": "INVALID_IMAGE_FORMAT",
                    "message": "Unsupported file format. Please upload PNG, JPG, JPEG, or WEBP.",
                    "detail": "Received file type: application/pdf",
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                {
                    "error": "FILE_TOO_LARGE",
                    "message": "Image file size exceeds the 10MB limit.",
                    "detail": "Received file size: 12.5 MB",
                    "timestamp": "2024-01-15T10:31:00Z"
                },
                {
                    "error": "MODEL_NOT_LOADED",
                    "message": "ML model is not available. Please contact support.",
                    "detail": "Model file not found at backend/artifacts/model.pth",
                    "timestamp": "2024-01-15T10:32:00Z"
                },
                {
                    "error": "INFERENCE_FAILED",
                    "message": "Failed to analyze the image. Please try again.",
                    "detail": "RuntimeError: CUDA out of memory",
                    "timestamp": "2024-01-15T10:33:00Z"
                },
                {
                    "error": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "detail": "Field 'species' is required",
                    "timestamp": "2024-01-15T10:34:00Z"
                }
            ]
        }
