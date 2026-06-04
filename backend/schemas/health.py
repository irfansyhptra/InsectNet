"""
Health Schemas

Pydantic models for system health check responses.
Validates: Requirements 8.4, 15.6
"""

from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    System health status response.
    
    Provides comprehensive health check information for monitoring
    system availability and diagnosing issues.
    """
    
    status: str = Field(
        ..., 
        description="Overall system status: 'healthy' or 'degraded'"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether the ML model is loaded in memory"
    )
    gemini_available: bool = Field(
        ..., 
        description="Whether Gemini API is reachable (optional service)"
    )
    database_connected: bool = Field(
        ..., 
        description="Whether storage/database is accessible"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when health check was performed"
    )
    uptime_seconds: float = Field(
        ..., 
        ge=0.0,
        description="Number of seconds since system startup"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "gemini_available": True,
                    "database_connected": True,
                    "timestamp": "2024-01-15T10:30:00Z",
                    "uptime_seconds": 3600.5
                },
                {
                    "status": "degraded",
                    "model_loaded": True,
                    "gemini_available": False,
                    "database_connected": True,
                    "timestamp": "2024-01-15T10:35:00Z",
                    "uptime_seconds": 3900.2
                }
            ]
        }
