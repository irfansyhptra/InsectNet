from fastapi import APIRouter, Depends
from schemas.insights import InsightsRequest, InsightsResponse
from services.gemini_service import GeminiService
from routes.dependencies import get_gemini_service
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/insights", response_model=InsightsResponse)
async def get_insights(
    request: InsightsRequest,
    gemini_service: GeminiService = Depends(get_gemini_service)
):
    """
    Generate AI insights for identified species.
    
    Returns graceful degradation response if Gemini fails.
    """
    # 1. Validate species name is already done by Pydantic (min_length=1)
    
    # 2. Request insights from Gemini service
    # The service already handles retry logic, timeout, and returns None on failure
    insights = await gemini_service.generate_insights(request.species, language=request.language)
    
    # 3. Handle failures gracefully
    if insights is None:
        logger.warning(f"Insights unavailable for {request.species}, returning graceful degradation response")
        return InsightsResponse(
            insights=None,
            available=False,
            error="AI insights temporarily unavailable."
        )
        
    # 4. Return response with insights
    return InsightsResponse(
        insights=insights,
        available=True,
        error=None
    )
