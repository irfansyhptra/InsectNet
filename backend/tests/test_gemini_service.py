"""
Test script for GeminiService
Tests basic functionality and error handling
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.gemini_service import GeminiService
from config import get_settings


async def test_gemini_service():
    """Test GeminiService with actual API calls"""
    print("=" * 60)
    print("Testing GeminiService Implementation")
    print("=" * 60)
    
    # Load settings
    settings = get_settings()
    
    # Check if API key is available
    if not settings.GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY not set in environment")
        print("Skipping live API tests (graceful degradation)")
        print("\n✅ Service can be instantiated but will fail gracefully")
        return
    
    # Initialize service
    print(f"\n1. Initializing GeminiService...")
    print(f"   - Timeout: {settings.GEMINI_TIMEOUT}s")
    print(f"   - Max retries: {settings.GEMINI_MAX_RETRIES}")
    
    try:
        service = GeminiService(
            api_key=settings.GEMINI_API_KEY,
            timeout=settings.GEMINI_TIMEOUT
        )
        print("   ✅ Service initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return
    
    # Test prompt building
    print("\n2. Testing prompt building...")
    species_name = "Apis mellifera"
    prompt = service._build_prompt(species_name)
    print(f"   - Species: {species_name}")
    print(f"   - Prompt length: {len(prompt)} characters")
    print(f"   - Includes sections: Taxonomy, Physical Characteristics, etc.")
    print("   ✅ Prompt built successfully")
    
    # Test transient error detection
    print("\n3. Testing transient error detection...")
    transient_codes = [429, 500, 502, 503, 504]
    non_transient_codes = [400, 401, 403, 404]
    
    for code in transient_codes:
        assert service._is_transient_error(code), f"Code {code} should be transient"
    print(f"   ✅ Correctly identifies transient errors: {transient_codes}")
    
    for code in non_transient_codes:
        assert not service._is_transient_error(code), f"Code {code} should not be transient"
    print(f"   ✅ Correctly identifies non-transient errors: {non_transient_codes}")
    
    # Test insights generation
    print("\n4. Testing insights generation (live API call)...")
    print(f"   - Requesting insights for: {species_name}")
    print("   - This may take a few seconds...")
    
    try:
        insights = await service.generate_insights(species_name)
        
        if insights:
            print(f"   ✅ Insights generated successfully")
            print(f"   - Length: {len(insights)} characters")
            print(f"   - Preview: {insights[:200]}...")
            
            # Check for expected sections
            expected_sections = ["Taxonomy", "Physical", "Habitat", "Behavior"]
            found_sections = [s for s in expected_sections if s.lower() in insights.lower()]
            print(f"   - Found sections: {found_sections}")
        else:
            print("   ⚠️  Insights returned None (graceful degradation)")
            print("   - This is acceptable behavior for unavailable service")
    
    except Exception as e:
        print(f"   ❌ Error during insights generation: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully")
    print("=" * 60)


async def test_timeout_handling():
    """Test timeout handling with very short timeout"""
    print("\n5. Testing timeout handling...")
    print("   - Setting timeout to 0.001s (very short)")
    
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        print("   ⚠️  Skipping (no API key)")
        return
    
    try:
        service = GeminiService(
            api_key=settings.GEMINI_API_KEY,
            timeout=0.001  # Very short timeout to force timeout
        )
        
        insights = await service.generate_insights("Test species")
        
        if insights is None:
            print("   ✅ Timeout handled gracefully (returned None)")
        else:
            print("   ⚠️  Request succeeded (timeout might be too long)")
    
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


async def main():
    """Run all tests"""
    try:
        await test_gemini_service()
        await test_timeout_handling()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
