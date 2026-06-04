#!/usr/bin/env python3
"""Simple test to verify _build_prompt method"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from services.gemini_service import GeminiService
    
    # Test the _build_prompt method
    service = GeminiService(api_key="test_key", timeout=10)
    prompt = service._build_prompt("Apis mellifera")
    
    print("✓ _build_prompt method exists and executes")
    print(f"✓ Prompt length: {len(prompt)} characters")
    print()
    print("Generated Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print()
    
    # Simple checks
    checks = [
        ("Species name included", "Apis mellifera" in prompt),
        ("Taxonomy section", "Taxonomy" in prompt),
        ("Physical Characteristics", "Physical" in prompt),
        ("Habitat", "Habitat" in prompt),
        ("Behavior", "Behavior" in prompt),
        ("Identification", "Identification" in prompt),
        ("Conservation", "Conservation" in prompt),
        ("Markdown requested", "markdown" in prompt.lower()),
    ]
    
    print("Checks:")
    all_pass = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("\n✓ All checks passed!")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
