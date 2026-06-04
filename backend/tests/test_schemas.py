"""
Test script to verify Pydantic schemas are correctly defined.
"""

from datetime import datetime
from schemas import (
    TopPrediction,
    PredictionResponse,
    InsightsRequest,
    InsightsResponse,
    HistoryRecord,
    HistoryListResponse,
    HealthResponse,
    ErrorResponse,
)


def test_top_prediction():
    """Test TopPrediction schema validation."""
    print("Testing TopPrediction...")
    
    # Valid data
    valid_pred = TopPrediction(
        species="Ladybug",
        confidence=0.94,
        scientific_name="Coccinellidae"
    )
    assert valid_pred.species == "Ladybug"
    assert valid_pred.confidence == 0.94
    print("✓ Valid TopPrediction created")
    
    # Test confidence bounds
    try:
        invalid_pred = TopPrediction(species="Test", confidence=1.5)
        print("✗ Should have failed: confidence > 1.0")
    except ValueError:
        print("✓ Confidence validation works (> 1.0)")
    
    try:
        invalid_pred = TopPrediction(species="Test", confidence=-0.1)
        print("✗ Should have failed: confidence < 0.0")
    except ValueError:
        print("✓ Confidence validation works (< 0.0)")
    
    # Test empty species
    try:
        invalid_pred = TopPrediction(species="", confidence=0.5)
        print("✗ Should have failed: empty species")
    except ValueError:
        print("✓ Non-empty species validation works")


def test_prediction_response():
    """Test PredictionResponse schema validation."""
    print("\nTesting PredictionResponse...")
    
    valid_response = PredictionResponse(
        species="Ladybug",
        scientific_name="Coccinellidae",
        confidence=0.94,
        top_predictions=[
            TopPrediction(species="Ladybug", confidence=0.94, scientific_name="Coccinellidae"),
            TopPrediction(species="Asian Lady Beetle", confidence=0.03),
        ]
    )
    assert len(valid_response.top_predictions) == 2
    assert valid_response.timestamp is not None
    print("✓ Valid PredictionResponse created")
    
    # Test empty top_predictions
    try:
        invalid_response = PredictionResponse(
            species="Test",
            confidence=0.5,
            top_predictions=[]
        )
        print("✗ Should have failed: empty top_predictions")
    except ValueError:
        print("✓ Non-empty top_predictions validation works")


def test_insights_schemas():
    """Test Insights request and response schemas."""
    print("\nTesting Insights schemas...")
    
    # Test request
    valid_request = InsightsRequest(species="Ladybug")
    assert valid_request.species == "Ladybug"
    print("✓ Valid InsightsRequest created")
    
    try:
        invalid_request = InsightsRequest(species="")
        print("✗ Should have failed: empty species")
    except ValueError:
        print("✓ Non-empty species validation works")
    
    # Test response (success case)
    success_response = InsightsResponse(
        insights="## Taxonomy\n\nKingdom: Animalia",
        available=True,
        error=None
    )
    assert success_response.available is True
    print("✓ Valid InsightsResponse (success) created")
    
    # Test response (failure case)
    failure_response = InsightsResponse(
        insights=None,
        available=False,
        error="Service unavailable"
    )
    assert failure_response.available is False
    print("✓ Valid InsightsResponse (failure) created")


def test_history_schemas():
    """Test History schemas."""
    print("\nTesting History schemas...")
    
    valid_record = HistoryRecord(
        id="550e8400-e29b-41d4-a716-446655440000",
        image_url="/storage/images/test.jpg",
        species="Ladybug",
        confidence=0.94,
        top_predictions=[
            TopPrediction(species="Ladybug", confidence=0.94)
        ],
        timestamp=datetime.utcnow()
    )
    assert valid_record.id == "550e8400-e29b-41d4-a716-446655440000"
    print("✓ Valid HistoryRecord created")
    
    valid_list = HistoryListResponse(
        records=[valid_record],
        total=150,
        filtered=1
    )
    assert len(valid_list.records) == 1
    assert valid_list.total == 150
    print("✓ Valid HistoryListResponse created")
    
    # Test negative total
    try:
        invalid_list = HistoryListResponse(
            records=[],
            total=-1,
            filtered=0
        )
        print("✗ Should have failed: negative total")
    except ValueError:
        print("✓ Non-negative total validation works")


def test_health_schema():
    """Test HealthResponse schema."""
    print("\nTesting HealthResponse...")
    
    valid_health = HealthResponse(
        status="healthy",
        model_loaded=True,
        gemini_available=True,
        database_connected=True,
        uptime_seconds=3600.5
    )
    assert valid_health.status == "healthy"
    assert valid_health.uptime_seconds == 3600.5
    print("✓ Valid HealthResponse created")
    
    # Test negative uptime
    try:
        invalid_health = HealthResponse(
            status="healthy",
            model_loaded=True,
            gemini_available=True,
            database_connected=True,
            uptime_seconds=-10.0
        )
        print("✗ Should have failed: negative uptime")
    except ValueError:
        print("✓ Non-negative uptime validation works")


def test_error_schema():
    """Test ErrorResponse schema."""
    print("\nTesting ErrorResponse...")
    
    valid_error = ErrorResponse(
        error="INVALID_IMAGE_FORMAT",
        message="Unsupported file format",
        detail="Received: application/pdf"
    )
    assert valid_error.error == "INVALID_IMAGE_FORMAT"
    assert valid_error.timestamp is not None
    print("✓ Valid ErrorResponse created")
    
    # Test empty error code
    try:
        invalid_error = ErrorResponse(
            error="",
            message="Test message"
        )
        print("✗ Should have failed: empty error code")
    except ValueError:
        print("✓ Non-empty error code validation works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Pydantic Schema Validation Tests")
    print("=" * 60)
    
    test_top_prediction()
    test_prediction_response()
    test_insights_schemas()
    test_history_schemas()
    test_health_schema()
    test_error_schema()
    
    print("\n" + "=" * 60)
    print("All schema tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
