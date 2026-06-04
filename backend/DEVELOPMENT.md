# Development Guide

This guide covers development workflows, best practices, and common tasks for the Smart Insect Identifier backend.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [API Development](#api-development)
7. [Debugging](#debugging)
8. [Common Tasks](#common-tasks)

## Getting Started

### Initial Setup

```bash
# Clone repository (if not already cloned)
git clone <repository-url>
cd backend

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Copy environment template
cp .env.example .env

# Edit .env and add your Gemini API key
nano .env
```

### Verify Setup

```bash
# Run infrastructure tests
python test_infrastructure.py

# Start development server
python main.py
```

Visit http://localhost:8000/docs to see the API documentation.

## Development Workflow

### Daily Development

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Pull latest changes**:
   ```bash
   git pull origin main
   ```

3. **Install/update dependencies** (if requirements.txt changed):
   ```bash
   pip install -r requirements.txt
   ```

4. **Start development server with auto-reload**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Make changes** and see them automatically reload

6. **Test your changes**:
   ```bash
   pytest
   ```

7. **Commit and push**:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-branch
   ```

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes

## Project Structure

```
backend/
├── routes/              # API endpoints (Controllers)
│   ├── health.py       # Health check endpoint
│   ├── predict.py      # Prediction endpoints
│   ├── insights.py     # AI insights endpoints
│   └── history.py      # History endpoints
│
├── services/           # Business logic (Services)
│   ├── ml_service.py   # ML inference service
│   ├── gemini_service.py  # Gemini API integration
│   └── history_service.py # History management
│
├── schemas/            # Data models (DTOs)
│   └── prediction.py   # Prediction schemas
│
├── utils/              # Shared utilities
│   └── logger.py       # Logging configuration
│
├── artifacts/          # Model files
│   ├── model.pth      # PyTorch model
│   ├── labels.json    # Class labels
│   └── metadata.json  # Model metadata
│
├── storage/            # Runtime data (git-ignored)
│   ├── history.json   # Prediction history
│   └── images/        # Uploaded images
│
├── config.py           # Configuration management
├── main.py             # Application entry point
└── test_*.py          # Test files
```

### Architecture Layers

1. **Routes Layer** (`routes/`)
   - Handle HTTP requests/responses
   - Validate input data
   - Call service layer
   - Format responses

2. **Service Layer** (`services/`)
   - Business logic
   - External API integration
   - Data processing
   - No HTTP knowledge

3. **Schema Layer** (`schemas/`)
   - Pydantic models
   - Request/response validation
   - Type safety

4. **Utils Layer** (`utils/`)
   - Cross-cutting concerns
   - Logging, helpers, etc.

## Configuration

### Environment Variables

All configuration is managed through environment variables. See `.env.example` for available options.

**Development Configuration** (`.env`):
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
LOG_FORMAT=text
GEMINI_API_KEY=your_dev_api_key
```

**Production Configuration**:
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
LOG_FORMAT=json
GEMINI_API_KEY=your_prod_api_key
```

### Configuration Best Practices

1. ✅ **Never commit `.env` files**
2. ✅ **Use `.env.example` as template**
3. ✅ **Document all environment variables**
4. ✅ **Provide sensible defaults**
5. ✅ **Validate required variables at startup**

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_infrastructure.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Structure

```
test_infrastructure.py   # Project setup tests
test_routes.py          # API endpoint tests
test_services.py        # Service layer tests
test_integration.py     # End-to-end tests
```

### Writing Tests

Example test structure:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check returns 200"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_ml_service():
    """Test ML service inference"""
    from services.ml_service import MLService
    ml_service = MLService.get_instance()
    assert ml_service.is_loaded()
```

## API Development

### Adding New Endpoints

1. **Create route file** (`routes/new_feature.py`):

```python
from fastapi import APIRouter, Depends, HTTPException
from schemas.new_feature import NewFeatureRequest, NewFeatureResponse

router = APIRouter()

@router.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature(request: NewFeatureRequest):
    """
    New feature endpoint.
    
    Args:
        request: Request data
        
    Returns:
        Response data
    """
    # Implementation
    return NewFeatureResponse(...)
```

2. **Create schemas** (`schemas/new_feature.py`):

```python
from pydantic import BaseModel, Field

class NewFeatureRequest(BaseModel):
    """Request schema for new feature"""
    field: str = Field(..., description="Description")

class NewFeatureResponse(BaseModel):
    """Response schema for new feature"""
    result: str
```

3. **Register router** in `main.py`:

```python
from routes import new_feature
app.include_router(new_feature.router, prefix="/api", tags=["New Feature"])
```

4. **Add tests**:

```python
def test_new_feature():
    response = client.post("/api/new-feature", json={"field": "value"})
    assert response.status_code == 200
```

### API Best Practices

1. ✅ **Use proper HTTP methods** (GET, POST, PUT, DELETE)
2. ✅ **Return appropriate status codes**
3. ✅ **Validate input with Pydantic**
4. ✅ **Document with docstrings**
5. ✅ **Handle errors gracefully**
6. ✅ **Use dependency injection**
7. ✅ **Log important operations**

## Debugging

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG
LOG_FORMAT=text  # More readable than JSON for debugging
```

### Interactive Debugging

Add breakpoints in code:

```python
import pdb; pdb.set_trace()  # Python debugger
```

Or use IDE debugger (VS Code, PyCharm).

### Common Debug Scenarios

**Issue: Model not loading**
```bash
# Check model file exists
ls -lh artifacts/model.pth

# Check logs for errors
python main.py 2>&1 | grep -i error
```

**Issue: Gemini API not working**
```bash
# Test API key
echo $GEMINI_API_KEY

# Check network connectivity
curl -I https://generativelanguage.googleapis.com

# Review logs
grep -i gemini storage/app.log
```

**Issue: CORS errors**
```bash
# Verify CORS origins in .env
cat .env | grep CORS

# Check browser console for specific error
```

### Log Analysis

**JSON Format** (production):
```bash
# Parse JSON logs
cat app.log | jq '.level, .message'

# Filter by level
cat app.log | jq 'select(.level=="ERROR")'

# Find specific errors
cat app.log | jq 'select(.message | contains("Gemini"))'
```

**Text Format** (development):
```bash
# Tail logs
tail -f app.log

# Filter errors
grep ERROR app.log

# Search for patterns
grep -i "inference" app.log
```

## Common Tasks

### Update Dependencies

```bash
# Install new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt

# Or add manually to requirements.txt
echo "package-name==1.0.0" >> requirements.txt
pip install -r requirements.txt
```

### Database Migration (if using database)

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Reset Development Environment

```bash
# Remove virtual environment
rm -rf venv/

# Remove storage data
rm -rf storage/

# Re-run setup
./setup.sh
```

### Performance Profiling

```python
# Add to endpoint
import time

@router.post("/predict")
async def predict(...):
    start = time.time()
    # ... operation
    logger.info(f"Inference took {time.time() - start:.3f}s")
```

Or use profiling tools:

```bash
# Install profiler
pip install py-spy

# Profile running server
py-spy top --pid <pid>

# Generate flamegraph
py-spy record -o profile.svg --pid <pid>
```

### Memory Debugging

```python
# Check memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
logger.info(f"Memory usage: {memory_mb:.2f} MB")
```

## Code Style Guidelines

### Python Style

Follow PEP 8:

```bash
# Install linters
pip install flake8 black isort

# Format code
black .
isort .

# Check style
flake8 .
```

### Type Hints

Use type hints for better IDE support:

```python
from typing import List, Optional, Dict

def process_image(
    image_bytes: bytes,
    options: Optional[Dict[str, any]] = None
) -> List[str]:
    """Process image with type hints"""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description.
    
    Longer description explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When invalid input provided
        RuntimeError: When operation fails
    
    Example:
        >>> function_name("test", 42)
        True
    """
    ...
```

## Performance Considerations

### Model Loading
- ✅ Load once at startup (singleton pattern)
- ✅ Keep in memory (don't reload per request)
- ⚠️ Watch memory usage for large models

### Request Processing
- ✅ Use async/await for I/O operations
- ✅ Implement request timeouts
- ✅ Add response caching if appropriate

### Database Operations
- ✅ Use connection pooling
- ✅ Add indexes for common queries
- ✅ Implement pagination for large results

## Security Checklist

- [ ] API keys in environment variables, not code
- [ ] Input validation on all endpoints
- [ ] File upload size limits enforced
- [ ] File type validation
- [ ] CORS configured correctly
- [ ] HTTPS in production
- [ ] Rate limiting (if public API)
- [ ] Authentication (if required)
- [ ] SQL injection prevention (if using SQL)
- [ ] XSS prevention in responses

## Deployment Checklist

Before deploying to production:

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] LOG_FORMAT=json
- [ ] LOG_LEVEL=INFO or WARNING
- [ ] ENVIRONMENT=production
- [ ] Model files present and validated
- [ ] CORS origins set to production domains
- [ ] API keys are production keys
- [ ] Health check endpoint working
- [ ] Monitoring/alerting configured
- [ ] Backup strategy in place

## Getting Help

- **API Documentation**: http://localhost:8000/docs
- **Architecture**: See `README.md`
- **Configuration**: See `.env.example`
- **Model Setup**: See `artifacts/README.md`
- **Logs**: Check application logs with appropriate log level

## Contributing

1. Create feature branch from `develop`
2. Make changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with description
6. Wait for code review
7. Address feedback
8. Merge when approved

## Useful Commands Cheat Sheet

```bash
# Activate environment
source venv/bin/activate

# Run dev server
uvicorn main:app --reload

# Run tests
pytest -v

# Check code style
black . && isort . && flake8 .

# Update dependencies
pip freeze > requirements.txt

# View logs (JSON)
tail -f app.log | jq

# View logs (text)
tail -f app.log

# Check API docs
open http://localhost:8000/docs

# Profile code
py-spy top --pid $(pgrep -f main:app)
```
