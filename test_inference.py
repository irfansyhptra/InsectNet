import asyncio
from backend.services.ml_service import MLService
from PIL import Image
import io

def test_inference():
    ml = MLService.get_instance()
    # create a mock image
    img = Image.new('RGB', (224, 224), color = 'red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    image_bytes = buf.getvalue()
    
    tensor = ml.preprocess_image(image_bytes)
    res = ml.predict(tensor)
    print("Success:", res)

test_inference()
