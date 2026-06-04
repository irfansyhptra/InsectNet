#!/usr/bin/env python3
"""
Test script to verify image preprocessing pipeline
"""

import io
import sys
from PIL import Image
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.ml_service import MLService


def create_test_image(mode='RGB', size=(512, 512), format='PNG'):
    """Create a test image in specified format"""
    img = Image.new(mode, size, color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes.read()


def test_preprocess_image():
    """Test the preprocessing pipeline"""
    print("Testing image preprocessing pipeline...")
    
    # Get ML service instance
    try:
        ml_service = MLService.get_instance()
        print("✓ ML Service loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load ML Service: {e}")
        return False
    
    # Test 1: RGB PNG image
    print("\nTest 1: RGB PNG image")
    try:
        image_bytes = create_test_image(mode='RGB', size=(512, 512), format='PNG')
        tensor = ml_service.preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {tensor.shape}"
        print(f"✓ PNG preprocessing successful. Tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"✗ PNG preprocessing failed: {e}")
        return False
    
    # Test 2: JPEG image
    print("\nTest 2: JPEG image")
    try:
        image_bytes = create_test_image(mode='RGB', size=(640, 480), format='JPEG')
        tensor = ml_service.preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)
        print(f"✓ JPEG preprocessing successful. Tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"✗ JPEG preprocessing failed: {e}")
        return False
    
    # Test 3: Grayscale to RGB conversion
    print("\nTest 3: Grayscale to RGB conversion")
    try:
        image_bytes = create_test_image(mode='L', size=(512, 512), format='PNG')
        tensor = ml_service.preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224), "Grayscale should be converted to 3 channels"
        print(f"✓ Grayscale conversion successful. Tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"✗ Grayscale conversion failed: {e}")
        return False
    
    # Test 4: RGBA to RGB conversion
    print("\nTest 4: RGBA to RGB conversion")
    try:
        image_bytes = create_test_image(mode='RGBA', size=(512, 512), format='PNG')
        tensor = ml_service.preprocess_image(image_bytes)
        assert tensor.shape == (1, 3, 224, 224), "RGBA should be converted to 3 channels"
        print(f"✓ RGBA conversion successful. Tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"✗ RGBA conversion failed: {e}")
        return False
    
    # Test 5: Different sizes (resize test)
    print("\nTest 5: Image resize (various sizes)")
    sizes = [(100, 100), (1024, 768), (300, 400)]
    for size in sizes:
        try:
            image_bytes = create_test_image(mode='RGB', size=size, format='PNG')
            tensor = ml_service.preprocess_image(image_bytes)
            assert tensor.shape == (1, 3, 224, 224)
            print(f"✓ Size {size} resized correctly to {tensor.shape}")
        except Exception as e:
            print(f"✗ Size {size} resize failed: {e}")
            return False
    
    # Test 6: Normalization applied
    print("\nTest 6: Normalization check")
    try:
        image_bytes = create_test_image(mode='RGB', size=(224, 224), format='PNG')
        tensor = ml_service.preprocess_image(image_bytes)
        # After normalization, values should not be in [0, 1] range
        assert tensor.min() < 0, "Normalization should produce negative values"
        assert tensor.max() > 1, "Normalization should produce values > 1"
        print(f"✓ Normalization applied. Min: {tensor.min():.3f}, Max: {tensor.max():.3f}")
    except Exception as e:
        print(f"✗ Normalization check failed: {e}")
        return False
    
    # Test 7: Invalid image bytes (error handling)
    print("\nTest 7: Invalid image bytes (error handling)")
    try:
        invalid_bytes = b'not an image'
        tensor = ml_service.preprocess_image(invalid_bytes)
        print("✗ Should have raised ValueError for invalid bytes")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"✗ Unexpected exception type: {e}")
        return False
    
    # Test 8: Empty bytes (error handling)
    print("\nTest 8: Empty bytes (error handling)")
    try:
        empty_bytes = b''
        tensor = ml_service.preprocess_image(empty_bytes)
        print("✗ Should have raised ValueError for empty bytes")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"✗ Unexpected exception type: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All preprocessing tests passed successfully!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_preprocess_image()
    sys.exit(0 if success else 1)
