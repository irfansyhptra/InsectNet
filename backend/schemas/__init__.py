"""
Schemas Package

Pydantic models for API request and response validation.
Provides structured data validation with field constraints and OpenAPI documentation.
"""

from .prediction import TopPrediction, PredictionResponse
from .insights import InsightsRequest, InsightsResponse
from .history import HistoryRecord, HistoryListResponse
from .health import HealthResponse
from .errors import ErrorResponse

__all__ = [
    # Prediction schemas
    "TopPrediction",
    "PredictionResponse",
    
    # Insights schemas
    "InsightsRequest",
    "InsightsResponse",
    
    # History schemas
    "HistoryRecord",
    "HistoryListResponse",
    
    # Health schemas
    "HealthResponse",
    
    # Error schemas
    "ErrorResponse",
]
