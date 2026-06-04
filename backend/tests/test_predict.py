import pytest
import torch
from unittest.mock import patch, MagicMock
from services.ml_service import MLService

def test_predict_format():
    MLService._instance = None
    service = MLService.get_instance()
    
    # Mock model and labels
    service._model = MagicMock()
    service._device = torch.device('cpu')
    service._labels = {"0": "Cat", "1": "Dog", "2": "Bird"}
    
    # Create mock tensor
    tensor = torch.randn(1, 3, 224, 224)
    
    # Mock model output
    mock_out = torch.tensor([[5.0, 1.0, -1.0]])
    service._model.return_value = mock_out
    
    result = service.predict(tensor, top_k=2)
    
    assert "species" in result
    assert "confidence" in result
    assert "top_predictions" in result
    assert len(result["top_predictions"]) == 2
    
    assert result["species"] == "Cat"
    assert result["top_predictions"][0]["species"] == "Cat"

def test_predict_model_not_loaded():
    MLService._instance = None
    service = MLService._instance
    service = MLService.__new__(MLService)
    
    tensor = torch.randn(1, 3, 224, 224)
    with pytest.raises(RuntimeError, match="Model is not loaded"):
        service.predict(tensor)
