"""
Configuration Management System
Loads and validates environment variables
"""
import os
from pathlib import Path
from typing import List
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden by creating a .env file in the backend directory.
    See .env.example for a template with all available configuration options.
    
    Configuration categories:
    - API Configuration: Server host, port, and environment mode
    - CORS Configuration: Allowed origins for frontend integration
    - Gemini API: Google Gemini API credentials and behavior
    - Storage: File system paths for data persistence
    - Model: PyTorch model and metadata file locations
    - Logging: Log level and output format configuration
    - Upload Limits: File size and type restrictions
    
    Example:
        # Load settings (cached singleton)
        settings = get_settings()
        
        # Access configuration
        print(f"API running on {settings.API_HOST}:{settings.API_PORT}")
    """
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_TIMEOUT: int = 10
    GEMINI_MAX_RETRIES: int = 3
    
    # Storage Configuration
    STORAGE_PATH: str = "backend/storage"
    
    # Model Configuration
    MODEL_PATH: str = "backend/artifacts/model.pth"
    LABELS_PATH: str = "backend/artifacts/labels.json"
    METADATA_PATH: str = "backend/artifacts/metadata.json"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    
    # File Upload Limits
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: List[str] = ["png", "jpg", "jpeg", "webp"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure singleton pattern.
    
    Returns:
        Settings: Validated configuration object
    
    Raises:
        FileNotFoundError: If required model files are missing
    
    Note:
        GEMINI_API_KEY is optional. System will gracefully degrade
        without it (AI insights unavailable).
    """
    settings = Settings()
    
    # Validate required settings (with graceful degradation for optional ones)
    if not settings.GEMINI_API_KEY:
        import logging
        logging.warning(
            "GEMINI_API_KEY not set - AI insights will be unavailable. "
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
    
    # Validate critical paths exist (model files are required)
    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {settings.MODEL_PATH}\n"
            f"Please ensure the pre-trained model is available at this path."
        )
    
    labels_path = Path(settings.LABELS_PATH)
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {settings.LABELS_PATH}\n"
            f"Please ensure the class labels JSON file is available at this path."
        )
    
    metadata_path = Path(settings.METADATA_PATH)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {settings.METADATA_PATH}\n"
            f"Please ensure the model metadata JSON file is available at this path."
        )
    
    # Create storage directory if it doesn't exist
    storage_path = Path(settings.STORAGE_PATH)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    return settings
