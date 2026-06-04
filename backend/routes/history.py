from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException, status
from schemas.history import HistoryRecord
from services.history_service import HistoryService
from routes.dependencies import get_history_service
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/history", response_model=List[HistoryRecord])
async def get_history(
    limit: int = Query(100, ge=1, le=500, description="Maximum records to return"),
    species: Optional[str] = Query(None, description="Filter by species name"),
    date_from: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Retrieve prediction history with optional filters.
    
    Query Parameters:
        - limit: Maximum records to return (1-500, default: 100)
        - species: Filter by species name (case-insensitive)
        - date_from: Start date (ISO 8601)
        - date_to: End date (ISO 8601)
    
    Response:
        List of history records sorted by timestamp DESC
    """
    try:
        records = await history_service.get_history(
            limit=limit,
            species_filter=species,
            date_from=date_from,
            date_to=date_to
        )
        return records
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history records"
        )
