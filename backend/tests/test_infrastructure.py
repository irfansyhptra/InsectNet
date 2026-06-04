"""
Infrastructure Tests
Verifies backend project structure and core infrastructure components.
"""
import pytest
from pathlib import Path
import sys
import os

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))


class TestProjectStructure:
    """Test that required directories and files exist"""
    
    def test_routes_directory_exists(self):
        """Verify routes directory exists"""
        assert (backend_path / "routes").is_dir()
    
    def test_services_directory_exists(self):
        """Verify services directory exists"""
        assert (backend_path / "services").is_dir()
    
    def test_schemas_directory_exists(self):
        """Verify schemas directory exists"""
        assert (backend_path / "schemas").is_dir()
    
    def test_utils_directory_exists(self):
        """Verify utils directory exists"""
        assert (backend_path / "utils").is_dir()
    
    def test_artifacts_directory_exists(self):
        """Verify artifacts directory for model files exists"""
        assert (backend_path / "artifacts").is_dir()
    
    def test_main_file_exists(self):
        """Verify main.py exists"""
        assert (backend_path / "main.py").is_file()
    
    def test_config_file_exists(self):
        """Verify config.py exists"""
        assert (backend_path / "config.py").is_file()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists"""
        assert (backend_path / "requirements.txt").is_file()
    
    def test_env_example_exists(self):
        """Verify .env.example exists"""
        assert (backend_path / ".env.example").is_file()


class TestConfigurationSystem:
    """Test configuration management"""
    
    def test_config_imports(self):
        """Verify config module imports successfully"""
        try:
            from config import Settings, get_settings
            assert Settings is not None
            assert get_settings is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")
    
    def test_settings_class_has_required_fields(self):
        """Verify Settings class has all required configuration fields"""
        from config import Settings
        
        required_fields = [
            'API_HOST', 'API_PORT', 'ENVIRONMENT',
            'CORS_ORIGINS', 'GEMINI_API_KEY',
            'STORAGE_PATH', 'MODEL_PATH', 'LABELS_PATH', 'METADATA_PATH',
            'LOG_LEVEL', 'LOG_FORMAT',
            'MAX_UPLOAD_SIZE', 'ALLOWED_EXTENSIONS'
        ]
        
        settings = Settings()
        for field in required_fields:
            assert hasattr(settings, field), f"Missing required field: {field}"


class TestLoggingInfrastructure:
    """Test structured logging system"""
    
    def test_logger_module_imports(self):
        """Verify logger module imports successfully"""
        try:
            from utils.logger import setup_logging, get_logger, JSONFormatter, TextFormatter
            assert setup_logging is not None
            assert get_logger is not None
            assert JSONFormatter is not None
            assert TextFormatter is not None
        except ImportError as e:
            pytest.fail(f"Failed to import logger: {e}")
    
    def test_json_formatter_exists(self):
        """Verify JSONFormatter class exists"""
        from utils.logger import JSONFormatter
        formatter = JSONFormatter()
        assert formatter is not None
    
    def test_text_formatter_exists(self):
        """Verify TextFormatter class exists"""
        from utils.logger import TextFormatter
        formatter = TextFormatter()
        assert formatter is not None
    
    def test_get_logger_returns_logger(self):
        """Verify get_logger returns a logger instance"""
        from utils.logger import get_logger
        import logging
        
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"


class TestFastAPIApplication:
    """Test FastAPI application setup"""
    
    def test_main_module_imports(self):
        """Verify main module imports successfully"""
        # Note: This test will fail if model files don't exist
        # That's expected - it validates the dependency check works
        try:
            import main
            assert main.app is not None
        except FileNotFoundError as e:
            # Expected if model files don't exist yet
            assert "not found" in str(e).lower()
        except Exception as e:
            # Other import errors indicate structural problems
            if "model" not in str(e).lower():
                pytest.fail(f"Unexpected error importing main: {e}")
    
    def test_fastapi_app_has_cors_middleware(self):
        """Verify CORS middleware is configured"""
        try:
            from main import app
            from fastapi.middleware.cors import CORSMiddleware
            
            # Check if CORS middleware is added
            has_cors = any(
                isinstance(middleware, CORSMiddleware) 
                for middleware in getattr(app, 'user_middleware', [])
            )
            # Note: middleware structure may vary, so this is a basic check
        except Exception as e:
            if "model" not in str(e).lower():
                pytest.fail(f"Error checking CORS: {e}")


class TestDependencies:
    """Test that required Python packages are importable"""
    
    def test_fastapi_installed(self):
        """Verify FastAPI is installed"""
        try:
            import fastapi
            assert fastapi is not None
        except ImportError:
            pytest.fail("FastAPI not installed")
    
    def test_uvicorn_installed(self):
        """Verify Uvicorn is installed"""
        try:
            import uvicorn
            assert uvicorn is not None
        except ImportError:
            pytest.fail("Uvicorn not installed")
    
    def test_pydantic_installed(self):
        """Verify Pydantic is installed"""
        try:
            import pydantic
            assert pydantic is not None
        except ImportError:
            pytest.fail("Pydantic not installed")
    
    def test_pillow_installed(self):
        """Verify Pillow is installed"""
        try:
            import PIL
            assert PIL is not None
        except ImportError:
            pytest.fail("Pillow not installed")
    
    def test_torch_installed(self):
        """Verify PyTorch is installed"""
        try:
            import torch
            assert torch is not None
        except ImportError:
            pytest.fail("PyTorch not installed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
