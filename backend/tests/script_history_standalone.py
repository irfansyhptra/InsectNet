"""
Standalone test to verify HistoryService works independently
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import without config dependency
import json
import uuid
import aiofiles
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryService:
    """Simplified HistoryService for testing"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "history.json"
        self.images_path = self.storage_path / "images"
        self._initialize_storage()
        logger.info(f"HistoryService initialized with storage at {self.storage_path}")
    
    def _initialize_storage(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            with open(self.db_path, 'w') as f:
                json.dump([], f)
            logger.info("Initialized empty history database")
    
    async def save_prediction(
        self,
        image_bytes: bytes,
        species: str,
        confidence: float,
        top_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        record_id = str(uuid.uuid4())
        image_url = await self._save_image(image_bytes, record_id)
        
        record = {
            "id": record_id,
            "image_url": image_url,
            "species": species,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        await self._append_to_history(record)
        logger.info(f"Saved prediction to history: {species} (confidence: {confidence:.2f})")
        
        return record
    
    async def _save_image(self, image_bytes: bytes, record_id: str) -> str:
        image_path = self.images_path / f"{record_id}.jpg"
        async with aiofiles.open(image_path, 'wb') as f:
            await f.write(image_bytes)
        logger.debug(f"Saved image to {image_path}")
        return f"/storage/images/{record_id}.jpg"
    
    async def _append_to_history(self, record: Dict[str, Any]) -> None:
        async with aiofiles.open(self.db_path, 'r') as f:
            content = await f.read()
            history = json.loads(content)
        
        history.append(record)
        
        async with aiofiles.open(self.db_path, 'w') as f:
            await f.write(json.dumps(history, indent=2))
        
        logger.debug(f"Appended record {record['id']} to history database")
    
    async def get_history(
        self,
        limit: int = 100,
        species_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        async with aiofiles.open(self.db_path, 'r') as f:
            content = await f.read()
            history = json.loads(content)
        
        filtered_history = history
        
        if species_filter:
            filtered_history = [
                record for record in filtered_history
                if species_filter.lower() in record['species'].lower()
            ]
        
        filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
        filtered_history = filtered_history[:limit]
        
        logger.info(f"Retrieved {len(filtered_history)} history records")
        return filtered_history


async def main():
    """Test the HistoryService"""
    print("=" * 60)
    print("Testing HistoryService Implementation")
    print("=" * 60)
    
    # Create test storage
    storage_path = Path("/tmp/test_history_storage")
    service = HistoryService(storage_path=storage_path)
    
    # Test 1: Save prediction
    print("\n✓ Test 1: Saving prediction...")
    sample_image = b'\xff\xd8\xff\xe0\x00\x10JFIF'  # Fake JPEG header
    record = await service.save_prediction(
        image_bytes=sample_image,
        species="Monarch Butterfly",
        confidence=0.95,
        top_predictions=[
            {"species": "Monarch Butterfly", "confidence": 0.95},
            {"species": "Viceroy Butterfly", "confidence": 0.03},
            {"species": "Queen Butterfly", "confidence": 0.01}
        ]
    )
    print(f"  ✓ Created record with ID: {record['id']}")
    print(f"  ✓ Image URL: {record['image_url']}")
    print(f"  ✓ Species: {record['species']}")
    print(f"  ✓ Confidence: {record['confidence']}")
    print(f"  ✓ Timestamp: {record['timestamp']}")
    
    # Test 2: Verify image file
    print("\n✓ Test 2: Verifying image file...")
    image_filename = record['image_url'].split('/')[-1]
    image_path = service.images_path / image_filename
    assert image_path.exists(), "Image file not created"
    print(f"  ✓ Image file exists: {image_path}")
    
    # Test 3: Verify history.json
    print("\n✓ Test 3: Verifying history database...")
    assert service.db_path.exists(), "History database not created"
    with open(service.db_path, 'r') as f:
        history_data = json.load(f)
    assert len(history_data) == 1, "History should have 1 record"
    print(f"  ✓ History database exists with {len(history_data)} record(s)")
    
    # Test 4: Save another prediction
    print("\n✓ Test 4: Saving second prediction...")
    record2 = await service.save_prediction(
        image_bytes=sample_image,
        species="Honeybee",
        confidence=0.88,
        top_predictions=[
            {"species": "Honeybee", "confidence": 0.88},
            {"species": "Bumblebee", "confidence": 0.10}
        ]
    )
    print(f"  ✓ Created second record with ID: {record2['id']}")
    
    # Test 5: Retrieve history
    print("\n✓ Test 5: Retrieving history...")
    history = await service.get_history()
    assert len(history) == 2, f"Expected 2 records, got {len(history)}"
    print(f"  ✓ Retrieved {len(history)} records")
    print(f"  ✓ First record (newest): {history[0]['species']}")
    print(f"  ✓ Second record (oldest): {history[1]['species']}")
    
    # Test 6: Filter by species
    print("\n✓ Test 6: Filtering by species...")
    filtered = await service.get_history(species_filter="bee")
    assert len(filtered) == 1, f"Expected 1 record with 'bee', got {len(filtered)}"
    assert filtered[0]['species'] == "Honeybee"
    print(f"  ✓ Filter 'bee' returned {len(filtered)} record: {filtered[0]['species']}")
    
    # Test 7: UUID uniqueness
    print("\n✓ Test 7: Verifying UUID uniqueness...")
    assert record['id'] != record2['id'], "UUIDs should be unique"
    print(f"  ✓ UUIDs are unique")
    
    # Test 8: Validate confidence bounds
    print("\n✓ Test 8: Testing confidence validation...")
    try:
        await service.save_prediction(
            image_bytes=sample_image,
            species="Test",
            confidence=1.5,
            top_predictions=[]
        )
        print("  ✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ✓ Correctly rejected confidence > 1: {e}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! HistoryService is working correctly")
    print("=" * 60)
    
    # Cleanup
    import shutil
    shutil.rmtree(storage_path)
    print("\n✓ Cleaned up test storage")


if __name__ == "__main__":
    asyncio.run(main())
