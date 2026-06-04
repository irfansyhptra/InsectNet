from pathlib import Path
from fastapi import Request
from config import get_settings
from services.ml_service import MLService
from services.gemini_service import GeminiService
from services.history_service import HistoryService

def get_ml_service() -> MLService:
    """Dependency injector for ML Service"""
    return MLService.get_instance()

def get_gemini_service() -> GeminiService:
    """Dependency injector for Gemini Service"""
    settings = get_settings()
    # Gemini service can handle empty key (it checks inside or we can check)
    return GeminiService(
        api_key=settings.GEMINI_API_KEY, 
        timeout=settings.GEMINI_TIMEOUT
    )

def get_history_service() -> HistoryService:
    """Dependency injector for History Service"""
    settings = get_settings()
    return HistoryService(storage_path=Path(settings.STORAGE_PATH))
