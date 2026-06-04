"""
Smart Insect Identifier Backend - Main Application Entry Point
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config import get_settings
from utils.logger import setup_logging, get_logger
from services.ml_service import MLService

# Set up logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup: Load ML model
    settings = get_settings()
    logger.info("Starting Smart Insect Identifier Backend")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    try:
        # Initialize ML Service (loads model)
        ml_service = MLService.get_instance()
        logger.info("ML Service initialized successfully")
        
        # Verify model is loaded
        if not ml_service.is_loaded():
            logger.error("Model failed to load")
            raise RuntimeError("ML model not loaded")
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutdown initiated")


# Create FastAPI app
app = FastAPI(
    title="Smart Insect Identifier API",
    description="AI-powered insect species identification with educational insights",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """
    Middleware to log all incoming requests and their responses.
    Satisfies Requirement 20.1: Log all incoming API requests with timestamp and endpoint.
    """
    import time
    
    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else "unknown"
        }
    )
    
    # Process request and measure duration
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Log response
    logger.info(
        f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 3)
        }
    )
    
    return response

# Mount static files for image storage
storage_path = Path(settings.STORAGE_PATH)
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")

# Import and include routers
from routes import predict, insights, history, health

app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(insights.router, prefix="/api", tags=["Insights"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(health.router, prefix="/api", tags=["Health"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Insect Identifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == "development"
    )
