import time
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from schemas.health import HealthResponse
from services.ml_service import MLService
from services.gemini_service import GeminiService
from services.history_service import HistoryService
from routes.dependencies import get_ml_service, get_gemini_service, get_history_service

router = APIRouter()
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    ml_service: MLService = Depends(get_ml_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
    history_service: HistoryService = Depends(get_history_service)
):
    """
    System health check endpoint.
    """
    model_loaded = ml_service.is_loaded()
    
    # Non-blocking check for Gemini: is api key set?
    gemini_available = bool(gemini_service.api_key)
    
    # Check if database is accessible
    database_connected = history_service.db_path.exists()
    
    uptime = time.time() - START_TIME
    
    if not model_loaded:
        sys_status = "degraded"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif not gemini_available or not database_connected:
        sys_status = "degraded"
        status_code = status.HTTP_200_OK
    else:
        sys_status = "healthy"
        status_code = status.HTTP_200_OK
        
    response_data = HealthResponse(
        status=sys_status,
        model_loaded=model_loaded,
        gemini_available=gemini_available,
        database_connected=database_connected,
        uptime_seconds=uptime
    )
    
    if status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        # Return 503 if critical service (model) is down
        return JSONResponse(status_code=status_code, content=response_data.model_dump(mode='json'))
        
    return response_data
