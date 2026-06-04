"""
Unit tests for ML Service singleton

Tests cover:
- Singleton pattern enforcement
- Model loading at initialization
- File validation and error handling
- Device selection (CPU/GPU)
- Health check method
- Property accessors
- Image preprocessing pipeline
"""

import pytest
import json
import torch
import io
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from PIL import Image
from services.ml_service import MLService


class TestMLServiceSingleton:
    """Test singleton pattern implementation"""
    
    def test_singleton_instance_creation(self):
        """Test that get_instance() creates and returns singleton"""
        # Reset singleton for test
        MLService._instance = None
        
        with patch.object(MLService, '_load_model'):
            instance1 = MLService.get_instance()
            instance2 = MLService.get_instance()
            
            assert instance1 is instance2
            assert isinstance(instance1, MLService)
    
    def test_direct_instantiation_raises_error(self):
        """Test that direct __init__() call raises RuntimeError"""
        # Create instance first to set _instance
        MLService._instance = MLService.__new__(MLService)
        
        with pytest.raises(RuntimeError, match="singleton"):
            MLService()
    
    def test_get_instance_calls_load_model_once(self):
        """Test that _load_model is called only once"""
        MLService._instance = None
        
        with patch.object(MLService, '_load_model') as mock_load:
            MLService.get_instance()
            MLService.get_instance()
            
            # Should only be called once despite multiple get_instance calls
            assert mock_load.call_count == 1


class TestModelLoading:
    """Test model loading functionality"""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings with test paths"""
        settings = MagicMock()
        settings.MODEL_PATH = "backend/artifacts/model.pth"
        settings.LABELS_PATH = "backend/artifacts/labels.json"
        settings.METADATA_PATH = "backend/artifacts/metadata.json"
        return settings
    
    @pytest.fixture
    def mock_labels(self):
        """Mock labels.json content"""
        return {
            "0": "Monarch Butterfly",
            "1": "Honey Bee",
            "2": "Ladybug"
        }
    
    @pytest.fixture
    def mock_metadata(self):
        """Mock metadata.json content"""
        return {
            "input_size": 224,
            "num_classes": 3,
            "model_architecture": "resnet18",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    
    def test_load_model_success(self, mock_settings, mock_labels, mock_metadata):
        """Test successful model loading"""
        MLService._instance = None
        
        # Mock file existence checks
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_labels))), \
             patch('json.load') as mock_json_load, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            # Setup mock returns
            mock_json_load.side_effect = [mock_labels, mock_metadata]
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_torch_load.return_value = mock_model
            
            service = MLService.get_instance()
            
            assert service.is_loaded()
            assert service._labels == mock_labels
            assert service._metadata == mock_metadata
            assert service._model is not None
            assert service._device == torch.device('cpu')
    
    def test_load_model_with_cuda(self, mock_settings, mock_labels, mock_metadata):
        """Test model loading with CUDA device"""
        MLService._instance = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('json.load') as mock_json_load, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            mock_json_load.side_effect = [mock_labels, mock_metadata]
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_torch_load.return_value = mock_model
            
            service = MLService.get_instance()
            
            assert service._device == torch.device('cuda')
    
    def test_missing_model_file_raises_error(self, mock_settings):
        """Test that missing model.pth raises FileNotFoundError"""
        MLService._instance = None
        
        exists_side_effect = [False, True, True]
        
        with patch('pathlib.Path.exists', side_effect=exists_side_effect), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            with pytest.raises(FileNotFoundError, match="Model file"):
                MLService.get_instance()
    
    def test_missing_labels_file_raises_error(self, mock_settings):
        """Test that missing labels.json raises FileNotFoundError"""
        MLService._instance = None
        
        exists_side_effect = [True, False, True]
        
        with patch('pathlib.Path.exists', side_effect=exists_side_effect), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            with pytest.raises(FileNotFoundError, match="Labels file"):
                MLService.get_instance()
    
    def test_missing_metadata_file_raises_error(self, mock_settings):
        """Test that missing metadata.json raises FileNotFoundError"""
        MLService._instance = None
        
        exists_side_effect = [True, True, False]
        
        with patch('pathlib.Path.exists', side_effect=exists_side_effect), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            with pytest.raises(FileNotFoundError, match="Metadata file"):
                MLService.get_instance()
    
    def test_malformed_json_raises_error(self, mock_settings):
        """Test that malformed JSON files raise RuntimeError"""
        MLService._instance = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")), \
             patch('json.load', side_effect=json.JSONDecodeError("test", "test", 0)), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            with pytest.raises(RuntimeError, match="Failed to parse JSON"):
                MLService.get_instance()
    
    def test_torch_load_failure_raises_error(self, mock_settings, mock_labels, mock_metadata):
        """Test that PyTorch loading failure raises RuntimeError"""
        MLService._instance = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('json.load') as mock_json_load, \
             patch('torch.load', side_effect=Exception("PyTorch error")), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            mock_json_load.side_effect = [mock_labels, mock_metadata]
            
            with pytest.raises(RuntimeError, match="Failed to load model"):
                MLService.get_instance()


class TestHealthCheck:
    """Test is_loaded() health check method"""
    
    def test_is_loaded_returns_true_when_ready(self):
        """Test is_loaded() returns True when model is loaded"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = {"0": "Test"}
        service._metadata = {"test": "data"}
        service._device = torch.device('cpu')
        
        assert service.is_loaded() is True
    
    def test_is_loaded_returns_false_when_model_missing(self):
        """Test is_loaded() returns False when model is None"""
        service = MLService.__new__(MLService)
        service._model = None
        service._labels = {"0": "Test"}
        service._metadata = {"test": "data"}
        service._device = torch.device('cpu')
        
        assert service.is_loaded() is False
    
    def test_is_loaded_returns_false_when_labels_missing(self):
        """Test is_loaded() returns False when labels is None"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = None
        service._metadata = {"test": "data"}
        service._device = torch.device('cpu')
        
        assert service.is_loaded() is False
    
    def test_is_loaded_returns_false_when_metadata_missing(self):
        """Test is_loaded() returns False when metadata is None"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = {"0": "Test"}
        service._metadata = None
        service._device = torch.device('cpu')
        
        assert service.is_loaded() is False
    
    def test_is_loaded_returns_false_when_device_missing(self):
        """Test is_loaded() returns False when device is None"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = {"0": "Test"}
        service._metadata = {"test": "data"}
        service._device = None
        
        assert service.is_loaded() is False


class TestPropertyAccessors:
    """Test property accessor methods"""
    
    def test_device_property_returns_device(self):
        """Test device property returns correct device"""
        service = MLService.__new__(MLService)
        service._device = torch.device('cpu')
        
        assert service.device == torch.device('cpu')
    
    def test_device_property_returns_none_when_not_loaded(self):
        """Test device property returns None when not loaded"""
        service = MLService.__new__(MLService)
        service._device = None
        
        assert service.device is None
    
    def test_labels_property_returns_labels(self):
        """Test labels property returns correct labels"""
        service = MLService.__new__(MLService)
        labels = {"0": "Test Species"}
        service._labels = labels
        
        assert service.labels == labels
    
    def test_labels_property_returns_none_when_not_loaded(self):
        """Test labels property returns None when not loaded"""
        service = MLService.__new__(MLService)
        service._labels = None
        
        assert service.labels is None
    
    def test_metadata_property_returns_metadata(self):
        """Test metadata property returns correct metadata"""
        service = MLService.__new__(MLService)
        metadata = {"input_size": 224}
        service._metadata = metadata
        
        assert service.metadata == metadata
    
    def test_metadata_property_returns_none_when_not_loaded(self):
        """Test metadata property returns None when not loaded"""
        service = MLService.__new__(MLService)
        service._metadata = None
        
        assert service.metadata is None
    
    def test_model_property_returns_model(self):
        """Test model property returns correct model"""
        service = MLService.__new__(MLService)
        model = MagicMock()
        service._model = model
        
        assert service.model is model
    
    def test_model_property_returns_none_when_not_loaded(self):
        """Test model property returns None when not loaded"""
        service = MLService.__new__(MLService)
        service._model = None
        
        assert service.model is None


class TestModelEvaluationMode:
    """Test that model is set to evaluation mode"""
    
    def test_model_set_to_eval_mode(self, mock_settings, mock_labels, mock_metadata):
        """Test that model.eval() is called during loading"""
        MLService._instance = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('json.load') as mock_json_load, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            mock_json_load.side_effect = [mock_labels, mock_metadata]
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_torch_load.return_value = mock_model
            
            service = MLService.get_instance()
            
            # Verify eval() was called
            mock_model.eval.assert_called_once()
            # Verify model was moved to device
            mock_model.to.assert_called_once()


# Fixture setup
@pytest.fixture
def mock_settings():
    """Mock settings fixture"""
    settings = MagicMock()
    settings.MODEL_PATH = "backend/artifacts/model.pth"
    settings.LABELS_PATH = "backend/artifacts/labels.json"
    settings.METADATA_PATH = "backend/artifacts/metadata.json"
    return settings


@pytest.fixture
def mock_labels():
    """Mock labels fixture"""
    return {
        "0": "Monarch Butterfly",
        "1": "Honey Bee",
        "2": "Ladybug"
    }


@pytest.fixture
def mock_metadata():
    """Mock metadata fixture"""
    return {
        "input_size": 224,
        "num_classes": 3,
        "model_architecture": "resnet18",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }



class TestImagePreprocessing:
    """Test image preprocessing pipeline"""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock ML service with loaded model"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = {"0": "Test Species"}
        service._metadata = {
            "input_size": 224,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
        service._device = torch.device('cpu')
        return service
    
    @pytest.fixture
    def create_test_image(self):
        """Factory to create test image bytes"""
        def _create_image(mode='RGB', size=(512, 512), format='PNG'):
            """Create a test image in specified format"""
            img = Image.new(mode, size, color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=format)
            img_bytes.seek(0)
            return img_bytes.read()
        return _create_image
    
    def test_preprocess_image_success_png(self, mock_service, create_test_image):
        """Test successful preprocessing of PNG image"""
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # Verify tensor shape: (batch, channels, height, width)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(tensor, torch.Tensor)
    
    def test_preprocess_image_success_jpeg(self, mock_service, create_test_image):
        """Test successful preprocessing of JPEG image"""
        image_bytes = create_test_image(mode='RGB', size=(640, 480), format='JPEG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(tensor, torch.Tensor)
    
    def test_preprocess_image_converts_grayscale_to_rgb(self, mock_service, create_test_image):
        """Test that grayscale images are converted to RGB"""
        image_bytes = create_test_image(mode='L', size=(512, 512), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # Should have 3 channels after RGB conversion
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_converts_rgba_to_rgb(self, mock_service, create_test_image):
        """Test that RGBA images are converted to RGB"""
        image_bytes = create_test_image(mode='RGBA', size=(512, 512), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # Should have 3 channels after RGB conversion
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_resizes_to_model_input_size(self, mock_service, create_test_image):
        """Test that images are resized to model input dimensions"""
        # Create various sized images
        sizes = [(100, 100), (512, 512), (1024, 768), (300, 400)]
        
        for size in sizes:
            image_bytes = create_test_image(mode='RGB', size=size, format='PNG')
            tensor = mock_service.preprocess_image(image_bytes)
            
            # All should be resized to 224x224
            assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_applies_normalization(self, mock_service, create_test_image):
        """Test that normalization is applied to tensor values"""
        image_bytes = create_test_image(mode='RGB', size=(224, 224), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # After normalization, values should not be in [0, 1] range
        # (they'll be centered around 0 with the ImageNet mean/std)
        assert tensor.min() < 0  # Some values should be negative after normalization
        assert tensor.max() > 1  # Some values should exceed 1 after normalization
    
    def test_preprocess_image_adds_batch_dimension(self, mock_service, create_test_image):
        """Test that batch dimension is added to tensor"""
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # First dimension should be batch size of 1
        assert tensor.shape[0] == 1
        assert len(tensor.shape) == 4  # (B, C, H, W)
    
    def test_preprocess_image_invalid_bytes_raises_valueerror(self, mock_service):
        """Test that invalid image bytes raise ValueError"""
        invalid_bytes = b'not an image'
        
        with pytest.raises(ValueError, match="Failed to decode image bytes"):
            mock_service.preprocess_image(invalid_bytes)
    
    def test_preprocess_image_corrupted_data_raises_valueerror(self, mock_service):
        """Test that corrupted image data raises ValueError"""
        # Create partially corrupted PNG data
        corrupted_bytes = b'\x89PNG\r\n\x1a\n\x00\x00corrupted'
        
        with pytest.raises(ValueError, match="Failed to decode image bytes"):
            mock_service.preprocess_image(corrupted_bytes)
    
    def test_preprocess_image_empty_bytes_raises_valueerror(self, mock_service):
        """Test that empty bytes raise ValueError"""
        empty_bytes = b''
        
        with pytest.raises(ValueError, match="Failed to decode image bytes"):
            mock_service.preprocess_image(empty_bytes)
    
    def test_preprocess_image_model_not_loaded_raises_runtimeerror(self, create_test_image):
        """Test that preprocessing fails if model is not loaded"""
        service = MLService.__new__(MLService)
        service._model = None
        service._labels = None
        service._metadata = None
        service._device = None
        
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            service.preprocess_image(image_bytes)
    
    def test_preprocess_image_missing_metadata_field_raises_runtimeerror(self, create_test_image):
        """Test that missing metadata fields raise RuntimeError"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = {"0": "Test"}
        service._metadata = {
            # Missing 'input_size' field
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
        service._device = torch.device('cpu')
        
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        with pytest.raises(RuntimeError, match="Missing required metadata field"):
            service.preprocess_image(image_bytes)
    
    def test_preprocess_image_missing_normalization_raises_runtimeerror(self, create_test_image):
        """Test that missing normalization parameters raise RuntimeError"""
        service = MLService.__new__(MLService)
        service._model = MagicMock()
        service._labels = {"0": "Test"}
        service._metadata = {
            "input_size": 224
            # Missing 'normalization' field
        }
        service._device = torch.device('cpu')
        
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        with pytest.raises(RuntimeError, match="Missing required metadata field"):
            service.preprocess_image(image_bytes)
    
    def test_preprocess_image_tensor_dtype(self, mock_service, create_test_image):
        """Test that output tensor has correct dtype (float32)"""
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        assert tensor.dtype == torch.float32
    
    def test_preprocess_image_different_input_sizes(self, create_test_image):
        """Test preprocessing with different model input sizes"""
        input_sizes = [224, 299, 512]
        
        for input_size in input_sizes:
            service = MLService.__new__(MLService)
            service._model = MagicMock()
            service._labels = {"0": "Test"}
            service._metadata = {
                "input_size": input_size,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
            service._device = torch.device('cpu')
            
            image_bytes = create_test_image(mode='RGB', size=(640, 480), format='PNG')
            tensor = service.preprocess_image(image_bytes)
            
            # Should match the specified input size
            assert tensor.shape == (1, 3, input_size, input_size)
    
    def test_preprocess_image_webp_format(self, mock_service):
        """Test preprocessing of WEBP format images"""
        # Create WEBP image
        img = Image.new('RGB', (512, 512), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='WEBP')
        img_bytes.seek(0)
        webp_bytes = img_bytes.read()
        
        tensor = mock_service.preprocess_image(webp_bytes)
        
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_very_small_image(self, mock_service, create_test_image):
        """Test preprocessing of very small images"""
        image_bytes = create_test_image(mode='RGB', size=(16, 16), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # Should be upscaled to model input size
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_very_large_image(self, mock_service, create_test_image):
        """Test preprocessing of very large images"""
        image_bytes = create_test_image(mode='RGB', size=(4096, 4096), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # Should be downscaled to model input size
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_non_square_aspect_ratio(self, mock_service, create_test_image):
        """Test preprocessing maintains correct dimensions for non-square images"""
        # Wide image
        image_bytes = create_test_image(mode='RGB', size=(1920, 1080), format='PNG')
        tensor = mock_service.preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)
        
        # Tall image
        image_bytes = create_test_image(mode='RGB', size=(1080, 1920), format='PNG')
        tensor = mock_service.preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_tensor_values_in_expected_range(self, mock_service, create_test_image):
        """Test that tensor values are in reasonable range after normalization"""
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        
        tensor = mock_service.preprocess_image(image_bytes)
        
        # With ImageNet normalization, values typically range from about -2 to 2.5
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0

    def test_predict_format(self, mock_settings, mock_labels, mock_metadata):
        MLService._instance = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('json.load') as mock_json_load, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('services.ml_service.get_settings', return_value=mock_settings):
            
            mock_json_load.side_effect = [mock_labels, mock_metadata]
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # Setup mock inference return
            mock_out = torch.tensor([[5.0, 1.0, -1.0]])
            mock_model.return_value = mock_out
            
            mock_torch_load.return_value = mock_model
            
            service = MLService.get_instance()
            
            tensor = torch.randn(1, 3, 224, 224)
            result = service.predict(tensor, top_k=2)
            
            assert "species" in result
            assert "confidence" in result
            assert "top_predictions" in result
            assert len(result["top_predictions"]) == 2
            
            # Since mock_labels is 0: string, we check what it mocked
            assert result["confidence"] > 0
