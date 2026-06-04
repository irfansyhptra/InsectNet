"""
ML Service - Model loading and inference

This module provides a singleton service for PyTorch model loading and inference.
The service loads the model once at application startup and maintains it in memory
for efficient inference across multiple requests.

Key Features:
- Singleton pattern ensures model loads only once
- Automatic device selection (CPU/GPU)
- Loads model, class labels, and metadata from artifacts
- Image preprocessing pipeline matching training conditions
- Comprehensive error handling for missing files
- Health check support via is_loaded() method
"""

import io
import json
import torch
import torchvision.models as models
from pathlib import Path
from typing import Optional, Dict, List, Any
from PIL import Image
from torchvision import transforms
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MLService:
    """
    Singleton service for ML model loading and inference.
    
    This service implements the singleton pattern to ensure the PyTorch model
    is loaded exactly once at application startup and retained in memory for
    all subsequent inference requests.
    
    Attributes:
        _instance: Class-level singleton instance
        _model: Loaded PyTorch model in evaluation mode
        _labels: Dictionary mapping class indices to species names
        _metadata: Model metadata including input size and normalization params
        _device: PyTorch device (CPU or CUDA) for inference
    
    Example:
        # Get singleton instance
        ml_service = MLService.get_instance()
        
        # Check if model is loaded
        if ml_service.is_loaded():
            print("Model ready for inference")
    """
    
    _instance: Optional['MLService'] = None
    _model: Optional[torch.nn.Module] = None
    _labels: Optional[Dict[str, str]] = None
    _metadata: Optional[Dict[str, Any]] = None
    _device: Optional[torch.device] = None
    
    def __init__(self):
        """
        Private constructor - use get_instance() instead.
        
        Raises:
            RuntimeError: If called directly instead of through get_instance()
        """
        if MLService._instance is not None:
            raise RuntimeError(
                "MLService is a singleton. Use MLService.get_instance() instead of direct instantiation."
            )
    
    @classmethod
    def get_instance(cls) -> 'MLService':
        """
        Get singleton instance of ML service.
        
        Creates the instance and loads the model on first call.
        Subsequent calls return the cached instance without reloading.
        
        Returns:
            MLService: Singleton instance with loaded model
            
        Raises:
            FileNotFoundError: If model, labels, or metadata files are missing
            RuntimeError: If model loading fails
        """
        if cls._instance is None:
            logger.info("Creating MLService singleton instance")
            cls._instance = cls.__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self) -> None:
        """
        Load model, labels, and metadata from artifacts directory.
        
        This method:
        1. Loads PyTorch model from model.pth
        2. Loads class labels from labels.json
        3. Loads metadata (input size, normalization) from metadata.json
        4. Sets model to evaluation mode
        5. Moves model to appropriate device (CPU/GPU)
        
        Raises:
            FileNotFoundError: If any required file is missing
            RuntimeError: If model loading fails
            json.JSONDecodeError: If JSON files are malformed
        """
        settings = get_settings()
        
        # Define paths
        model_path = Path(settings.MODEL_PATH)
        labels_path = Path(settings.LABELS_PATH)
        metadata_path = Path(settings.METADATA_PATH)
        
        logger.info("Starting model loading process")
        
        # Validate all required files exist
        self._validate_artifact_files(model_path, labels_path, metadata_path)
        
        try:
            # Load class labels
            logger.info(f"Loading class labels from {labels_path}")
            with open(labels_path, 'r') as f:
                self._labels = json.load(f)
            logger.info(f"Loaded {len(self._labels)} class labels")
            
            # Load metadata
            logger.info(f"Loading model metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
            logger.info(
                f"Loaded metadata: {self._metadata.get('model_architecture', 'unknown')} "
                f"with {self._metadata.get('num_classes', 0)} classes"
            )
            
            # Determine device (GPU if available, otherwise CPU)
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self._device}")
            
            # Load PyTorch model
            logger.info(f"Loading PyTorch model from {model_path}")
            self._model = torch.load(
                model_path,
                map_location=self._device,
                weights_only=False
            )
            
            # Handle checkpoint dict or raw model
            if isinstance(self._model, dict):
                logger.info("Loaded model is a checkpoint dict")
                state_dict = self._model.get('model', self._model)
                
                # Determine architecture from metadata
                arch = self._metadata.get('model_architecture', 'regnet_y_32gf')
                logger.info(f"Instantiating architecture: {arch}")
                
                # Build the correct model architecture
                if hasattr(models, arch):
                    model_arch = getattr(models, arch)(weights=None)
                else:
                    logger.warning(f"Architecture '{arch}' not found in torchvision, falling back to regnet_y_32gf")
                    model_arch = models.regnet_y_32gf(weights=None)
                
                # Modify final layer to match number of classes
                num_classes = len(self._labels)
                num_ftrs = model_arch.fc.in_features
                model_arch.fc = torch.nn.Linear(num_ftrs, num_classes)
                logger.info(f"Modified fc layer: {num_ftrs} -> {num_classes}")
                
                # Load state dict
                try:
                    model_arch.load_state_dict(state_dict, strict=True)
                    logger.info("State dict loaded successfully (strict mode)")
                except RuntimeError as e:
                    logger.warning(f"Strict loading failed, trying non-strict: {e}")
                    missing, unexpected = model_arch.load_state_dict(state_dict, strict=False)
                    if missing:
                        logger.warning(f"Missing keys: {missing[:5]}...")
                    if unexpected:
                        logger.warning(f"Unexpected keys: {unexpected[:5]}...")
                
                self._model = model_arch
            
            # Set model to evaluation mode (disables dropout, batch norm, etc.)
            self._model.eval()
            # Move model to device
            self._model = self._model.to(self._device)
                
            logger.info("Model set to evaluation mode")
            logger.info(f"Model moved to {self._device}")
            logger.info("Model loading completed successfully")
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON file: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _validate_artifact_files(
        self,
        model_path: Path,
        labels_path: Path,
        metadata_path: Path
    ) -> None:
        """
        Validate that all required artifact files exist.
        
        Args:
            model_path: Path to model.pth file
            labels_path: Path to labels.json file
            metadata_path: Path to metadata.json file
            
        Raises:
            FileNotFoundError: If any required file is missing
        """
        missing_files = []
        
        if not model_path.exists():
            missing_files.append(f"Model file: {model_path}")
        
        if not labels_path.exists():
            missing_files.append(f"Labels file: {labels_path}")
        
        if not metadata_path.exists():
            missing_files.append(f"Metadata file: {metadata_path}")
        
        if missing_files:
            error_msg = (
                "Required artifact files are missing:\n"
                + "\n".join(f"  - {file}" for file in missing_files)
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded in memory.
        
        This method is used by the health check endpoint to verify
        the ML service is operational.
        
        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        is_ready = (
            self._model is not None
            and self._labels is not None
            and self._metadata is not None
            and self._device is not None
        )
        
        if is_ready:
            logger.debug("Model is loaded and ready for inference")
        else:
            logger.warning("Model is not fully loaded")
        
        return is_ready
    
    @property
    def device(self) -> Optional[torch.device]:
        """
        Get the device (CPU/GPU) used for inference.
        
        Returns:
            Optional[torch.device]: Current device or None if not loaded
        """
        return self._device
    
    @property
    def labels(self) -> Optional[Dict[str, str]]:
        """
        Get class labels dictionary.
        
        Returns:
            Optional[Dict[str, str]]: Labels mapping or None if not loaded
        """
        return self._labels
    
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary or None if not loaded
        """
        return self._metadata
    
    @property
    def model(self) -> Optional[torch.nn.Module]:
        """
        Get the loaded PyTorch model.
        
        Returns:
            Optional[torch.nn.Module]: Model or None if not loaded
        """
        return self._model
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image to match model input requirements.
        
        This method implements the complete preprocessing pipeline:
        1. Decode bytes to PIL Image
        2. Convert to RGB format
        3. Resize to model input dimensions
        4. Apply normalization with training parameters
        5. Convert to PyTorch tensor
        6. Add batch dimension
        
        Args:
            image_bytes: Raw image bytes from upload
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for inference with shape (1, 3, H, W)
            
        Raises:
            ValueError: If image format is invalid or cannot be decoded
            RuntimeError: If model is not loaded or preprocessing fails
            
        Example:
            >>> ml_service = MLService.get_instance()
            >>> with open('insect.jpg', 'rb') as f:
            ...     image_bytes = f.read()
            >>> tensor = ml_service.preprocess_image(image_bytes)
            >>> tensor.shape
            torch.Size([1, 3, 224, 224])
        """
        # Validate model is loaded
        if not self.is_loaded():
            error_msg = "Model is not loaded. Cannot preprocess image."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # Step 1: Decode bytes to PIL Image
            logger.debug("Decoding image bytes to PIL Image")
            image = Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            error_msg = f"Failed to decode image bytes. Invalid image format or corrupted data: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        try:
            # Step 2: Convert to RGB (handles grayscale, RGBA, etc.)
            logger.debug(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
            
            # Step 3: Get input dimensions from metadata
            input_size = self._metadata['input_size']
            logger.debug(f"Target input size: {input_size}x{input_size}")
            
            # Step 4: Get normalization parameters from metadata
            norm_mean = self._metadata['normalization']['mean']
            norm_std = self._metadata['normalization']['std']
            logger.debug(f"Normalization - mean: {norm_mean}, std: {norm_std}")
            
            # Step 5: Build transformation pipeline
            # Use validation transforms: resize to 256, center crop to input_size (224)
            # This matches the training validation pipeline (val_resize_size=256, val_crop_size=224)
            resize_size = self._metadata.get('val_resize_size', 256)
            transform = transforms.Compose([
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            
            # Step 6: Apply transformations
            logger.debug("Applying preprocessing transformations")
            tensor = transform(image)
            
            # Step 7: Add batch dimension (B, C, H, W)
            tensor = tensor.unsqueeze(0)
            
            logger.debug(f"Preprocessing complete. Tensor shape: {tensor.shape}")
            return tensor
            
        except KeyError as e:
            error_msg = f"Missing required metadata field: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Image preprocessing failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict(self, tensor: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform inference and return predictions.
        
        Args:
            tensor: Preprocessed image tensor
            top_k: Number of predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Cannot run inference.")
            
        try:
            with torch.no_grad():
                # Forward pass
                tensor = tensor.to(self._device)
                outputs = self._model(tensor)
                
                # Calculate probabilities using softmax
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Ensure top_k is not strictly larger than the number of classes available
                actual_top_k = min(top_k, probabilities.size(0))
                top_prob, top_catid = torch.topk(probabilities, actual_top_k)
                
                top_predictions = []
                for i in range(top_prob.size(0)):
                    prob = float(top_prob[i].item())
                    idx = str(top_catid[i].item())
                    species_name = self._labels.get(idx, f"Unknown ID {idx}")
                    
                    top_predictions.append({
                        "species": species_name,
                        "confidence": prob,
                        "scientific_name": None
                    })
                
                # Primary prediction is the first item
                primary = top_predictions[0] if top_predictions else {
                    "species": "Unknown",
                    "confidence": 0.0,
                    "scientific_name": None
                }
                
                return {
                    "species": primary["species"],
                    "confidence": primary["confidence"],
                    "scientific_name": primary.get("scientific_name"),
                    "top_predictions": top_predictions
                }
                
        except Exception as e:
            error_msg = f"Inference failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

