"""
History Service - Prediction history management

This service manages the storage and retrieval of prediction history records.
Each prediction is stored with its image, species information, confidence score,
and timestamp in a file-based storage system.

Storage structure:
    storage/
        ├── history.json          # Metadata database
        └── images/
            ├── {uuid1}.jpg
            ├── {uuid2}.jpg
            └── ...

Requirements: 7.1, 7.4
"""
import json
import uuid
import aiofiles
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class HistoryService:
    """
    Service for prediction history management with file-based storage.
    
    This service provides functionality to:
    - Save prediction results with images
    - Retrieve prediction history with filtering
    - Manage image storage on disk
    - Maintain a JSON-based metadata database
    
    The service uses a simple file-based approach suitable for single-instance
    deployments. For production multi-instance deployments, consider migrating
    to a proper database system.
    
    Attributes:
        storage_path (Path): Root directory for storage
        db_path (Path): Path to history.json metadata file
        images_path (Path): Directory for storing images
    
    Example:
        service = HistoryService(storage_path=Path("backend/storage"))
        record = await service.save_prediction(
            image_bytes=image_data,
            species="Monarch Butterfly",
            confidence=0.95,
            top_predictions=[...]
        )
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize history service with storage configuration.
        
        Creates necessary directory structure if it doesn't exist:
        - storage_path/history.json
        - storage_path/images/
        
        Args:
            storage_path: Root directory for storing history data
        
        Raises:
            OSError: If directory creation fails due to permissions
        """
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "history.json"
        self.images_path = self.storage_path / "images"
        
        # Create storage directories if they don't exist
        self._initialize_storage()
        
        logger.info(f"HistoryService initialized with storage at {self.storage_path}")
    
    def _initialize_storage(self) -> None:
        """
        Create storage directory structure if it doesn't exist.
        
        Creates:
        - storage_path directory
        - storage_path/images directory
        - storage_path/history.json file (empty array if not exists)
        
        This method is idempotent - safe to call multiple times.
        """
        try:
            # Create directories
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize history.json if it doesn't exist
            if not self.db_path.exists():
                with open(self.db_path, 'w') as f:
                    json.dump([], f)
                logger.info("Initialized empty history database")
        except OSError as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise
    
    async def save_prediction(
        self,
        image_bytes: bytes,
        species: str,
        confidence: float,
        top_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Save prediction to history with image storage.
        
        This method:
        1. Generates a unique UUID for the record
        2. Saves the image to storage/images/{uuid}.jpg
        3. Creates a metadata record with all prediction info
        4. Appends the record to history.json
        5. Returns the created record
        
        Args:
            image_bytes: Raw image bytes from upload
            species: Predicted species name
            confidence: Confidence score (0.0 to 1.0)
            top_predictions: List of alternative predictions with format:
                [{"species": str, "confidence": float}, ...]
        
        Returns:
            Created history record containing:
                - id: Unique identifier (UUID)
                - image_url: Relative URL to stored image
                - species: Species name
                - confidence: Confidence score
                - top_predictions: Alternative predictions
                - timestamp: ISO 8601 timestamp
        
        Raises:
            IOError: If file writing fails
            ValueError: If confidence is not between 0 and 1
        
        Example:
            record = await service.save_prediction(
                image_bytes=b'...',
                species="Monarch Butterfly",
                confidence=0.95,
                top_predictions=[
                    {"species": "Viceroy Butterfly", "confidence": 0.03},
                    {"species": "Queen Butterfly", "confidence": 0.01}
                ]
            )
            # Returns: {
            #     "id": "550e8400-e29b-41d4-a716-446655440000",
            #     "image_url": "/storage/images/550e8400-e29b-41d4-a716-446655440000.jpg",
            #     "species": "Monarch Butterfly",
            #     "confidence": 0.95,
            #     "top_predictions": [...],
            #     "timestamp": "2024-01-15T10:30:00.000Z"
            # }
        """
        # Validate confidence score
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        # Generate unique ID
        record_id = str(uuid.uuid4())
        
        # Save image to storage
        image_url = await self._save_image(image_bytes, record_id)
        
        # Create history record
        record = {
            "id": record_id,
            "image_url": image_url,
            "species": species,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Append to history database
        await self._append_to_history(record)
        
        logger.info(f"Saved prediction to history: {species} (confidence: {confidence:.2f})")
        
        return record
    
    async def _save_image(self, image_bytes: bytes, record_id: str) -> str:
        """
        Write image bytes to storage and return relative URL.
        
        Images are saved as JPEG files in the images directory with
        the record UUID as the filename. This ensures unique filenames
        and easy correlation with history records.
        
        Args:
            image_bytes: Raw image bytes to save
            record_id: UUID of the history record
        
        Returns:
            Relative URL path to the saved image (e.g., "/storage/images/{uuid}.jpg")
        
        Raises:
            IOError: If file writing fails
        
        Note:
            The returned URL is relative and assumes the backend serves
            static files from the storage directory.
        """
        image_path = self.images_path / f"{record_id}.jpg"
        
        try:
            # Write image bytes asynchronously
            async with aiofiles.open(image_path, 'wb') as f:
                await f.write(image_bytes)
            
            logger.debug(f"Saved image to {image_path}")
        except IOError as e:
            logger.error(f"Failed to save image: {e}")
            raise
        
        # Return relative URL
        return f"/storage/images/{record_id}.jpg"
    
    async def _append_to_history(self, record: Dict[str, Any]) -> None:
        """
        Append a record to the history.json database.
        
        This method loads the existing history, appends the new record,
        and writes back to disk. For high-volume deployments, consider
        using a proper database instead of this file-based approach.
        
        Args:
            record: History record to append
        
        Raises:
            IOError: If file reading or writing fails
            json.JSONDecodeError: If history.json is corrupted
        """
        try:
            # Load existing history
            async with aiofiles.open(self.db_path, 'r') as f:
                content = await f.read()
                history = json.loads(content)
            
            # Append new record
            history.append(record)
            
            # Write back to disk
            async with aiofiles.open(self.db_path, 'w') as f:
                await f.write(json.dumps(history, indent=2))
            
            logger.debug(f"Appended record {record['id']} to history database")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to update history database: {e}")
            raise
    
    async def get_history(
        self,
        limit: int = 100,
        species_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve prediction history with optional filters.
        
        Returns records in reverse chronological order (newest first).
        Supports filtering by species name and date range.
        
        Args:
            limit: Maximum number of records to return (default: 100)
            species_filter: Filter by species name (case-insensitive substring match)
            date_from: Start of date range (inclusive)
            date_to: End of date range (inclusive)
        
        Returns:
            List of history records sorted by timestamp DESC
        
        Raises:
            IOError: If history.json cannot be read
            json.JSONDecodeError: If history.json is corrupted
        
        Example:
            # Get all history
            records = await service.get_history()
            
            # Get Monarch Butterfly predictions from last 7 days
            from datetime import timedelta
            records = await service.get_history(
                species_filter="Monarch",
                date_from=datetime.utcnow() - timedelta(days=7)
            )
        """
        try:
            # Load history database
            async with aiofiles.open(self.db_path, 'r') as f:
                content = await f.read()
                history = json.loads(content)
            
            # Apply filters
            filtered_history = history
            
            # Filter by species (case-insensitive substring match)
            if species_filter:
                filtered_history = [
                    record for record in filtered_history
                    if species_filter.lower() in record['species'].lower()
                ]
            
            # Filter by date range
            if date_from or date_to:
                filtered_history = self._filter_by_date(
                    filtered_history,
                    date_from,
                    date_to
                )
            
            # Sort by timestamp descending (newest first)
            filtered_history.sort(
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            # Apply limit
            filtered_history = filtered_history[:limit]
            
            logger.info(f"Retrieved {len(filtered_history)} history records")
            
            return filtered_history
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read history database: {e}")
            raise
    
    def _filter_by_date(
        self,
        records: List[Dict[str, Any]],
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """
        Filter records by date range.
        
        Args:
            records: List of history records
            date_from: Start of date range (inclusive)
            date_to: End of date range (inclusive)
        
        Returns:
            Filtered list of records
        """
        filtered = []
        
        for record in records:
            # Parse timestamp from ISO 8601 format
            timestamp_str = record['timestamp'].rstrip('Z')
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Check date range
            if date_from and timestamp < date_from:
                continue
            if date_to and timestamp > date_to:
                continue
            
            filtered.append(record)
        
        return filtered
    
    async def delete_record(self, record_id: str) -> bool:
        """
        Delete a history record and its associated image.
        
        This method:
        1. Loads the history database
        2. Finds and removes the record with matching ID
        3. Deletes the associated image file
        4. Saves the updated history database
        
        Args:
            record_id: UUID of the record to delete
        
        Returns:
            True if record was deleted, False if not found
        
        Raises:
            IOError: If file operations fail
        
        Example:
            deleted = await service.delete_record("550e8400-e29b-41d4-a716-446655440000")
            if deleted:
                print("Record deleted successfully")
        """
        try:
            # Load existing history
            async with aiofiles.open(self.db_path, 'r') as f:
                content = await f.read()
                history = json.loads(content)
            
            # Find and remove record
            original_length = len(history)
            history = [r for r in history if r['id'] != record_id]
            
            if len(history) == original_length:
                logger.warning(f"Record {record_id} not found")
                return False
            
            # Delete associated image
            image_path = self.images_path / f"{record_id}.jpg"
            if image_path.exists():
                image_path.unlink()
                logger.debug(f"Deleted image {image_path}")
            
            # Write updated history back to disk
            async with aiofiles.open(self.db_path, 'w') as f:
                await f.write(json.dumps(history, indent=2))
            
            logger.info(f"Deleted record {record_id}")
            return True
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to delete record: {e}")
            raise
