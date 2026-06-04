# Quick Start Guide

Get the Smart Insect Identifier backend running in 5 minutes.

## Prerequisites Check

```bash
# Check Python version (need 3.8+)
python3 --version

# Check pip is available
python3 -m pip --version
```

If Python is not installed:
- **Ubuntu/Debian**: `sudo apt install python3 python3-pip python3-venv`
- **macOS**: `brew install python3`
- **Arch Linux**: `sudo pacman -S python python-pip`

## Step 1: Setup Environment (2 minutes)

```bash
# Navigate to backend directory
cd backend

# Run automated setup
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

**Expected Output**: 
```
✅ Setup complete!
```

## Step 2: Configure Environment (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

**Minimum Configuration**:
Just press Ctrl+X to save (defaults work for development)

**Optional - Add Gemini API Key** (for AI insights):
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

Get your key from: https://makersuite.google.com/app/apikey

## Step 3: Verify Setup (30 seconds)

```bash
# Run infrastructure tests
python test_infrastructure.py
```

**Expected Output**: 
```
✅ All tests passed
```

## Step 4: Start Server (30 seconds)

```bash
# Start development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Step 5: Test API (1 minute)

Open your browser and visit:

1. **API Root**: http://localhost:8000
   - Should see welcome message

2. **API Documentation**: http://localhost:8000/docs
   - Interactive API documentation (Swagger UI)

3. **Health Check**: http://localhost:8000/api/health
   - Should return health status

Or use curl:
```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Troubleshooting

### Issue: Python not found
```bash
# Try python instead of python3
python --version
```

### Issue: Permission denied on setup.sh
```bash
chmod +x setup.sh
```

### Issue: Port 8000 already in use
```bash
# Use different port
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Issue: Model files not found
**Expected for fresh setup** - Template files are in place.

For production use, you need:
1. Trained PyTorch model (`artifacts/model.pth`)
2. Class labels (`artifacts/labels.json`)
3. Model metadata (`artifacts/metadata.json`)

See `artifacts/README.md` for details.

### Issue: Gemini API not working
**This is OK** - System works without it (graceful degradation).

AI insights will show "temporarily unavailable" but core prediction functionality works.

## What's Next?

1. **Explore API Documentation**: http://localhost:8000/docs
2. **Read Full Documentation**: `README.md`
3. **Development Guide**: `DEVELOPMENT.md`
4. **Configure Gemini API**: Add API key to `.env`
5. **Update Model Files**: See `artifacts/README.md`

## Daily Usage

```bash
# Activate environment (do this each time)
source venv/bin/activate

# Start development server
uvicorn main:app --reload

# In another terminal - run tests
pytest

# Deactivate environment when done
deactivate
```

## Production Deployment

For production deployment, see:
- `README.md` - Production deployment section
- `DEVELOPMENT.md` - Deployment checklist

## Getting Help

- **Documentation**: `README.md`, `DEVELOPMENT.md`
- **API Docs**: http://localhost:8000/docs (when running)
- **Configuration**: `.env.example`
- **Model Setup**: `artifacts/README.md`

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `.env` file created
- [ ] Infrastructure tests pass
- [ ] Server starts without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Health endpoint returns 200 OK

If all items checked, you're ready to develop! 🚀
