from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from schemas.prediction import PredictionResponse
from services.ml_service import MLService
from services.history_service import HistoryService
from routes.dependencies import get_ml_service, get_history_service
from config import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    ml_service: MLService = Depends(get_ml_service),
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Predict insect species from uploaded image.
    
    Request:
        - file: Image file (PNG/JPG/JPEG/WEBP, max 10MB)
    """
    settings = get_settings()
    
    # Check if ML Service is ready
    if not ml_service.is_loaded():
        logger.error("Prediction failed: Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded"
        )
    
    # 1. Validate file format
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid file format uploaded: {ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
        
    # Read bytes and validate size
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE:
        logger.warning(f"File too large: {len(image_bytes)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )
        
    try:
        # Preprocess and predict
        tensor = ml_service.preprocess_image(image_bytes)
        prediction_result = ml_service.predict(tensor, top_k=5)
        
        # prediction_result format:
        # { "species": ..., "confidence": ..., "scientific_name": ..., "top_predictions": [...] }
        
        # Save to history (non-blocking in terms of waiting for other long operations, but we await it here)
        history_record = await history_service.save_prediction(
            image_bytes=image_bytes,
            species=prediction_result["species"],
            confidence=prediction_result["confidence"],
            top_predictions=prediction_result["top_predictions"]
        )
        
        return PredictionResponse(
            species=prediction_result["species"],
            scientific_name=prediction_result.get("scientific_name"),
            confidence=prediction_result["confidence"],
            top_predictions=prediction_result["top_predictions"],
            timestamp=history_record["timestamp"]
        )
        
    except ValueError as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during inference"
        )
