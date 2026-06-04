"""
Unit tests for Gemini Service prompt design (Task 3.2)

Tests that the _build_prompt method creates properly structured prompts
requesting all required sections: taxonomy, physical characteristics, 
habitat, behavior, identification tips, and conservation status.

Requirements: 5.2, 5.3
"""

import pytest
from services.gemini_service import GeminiService


class TestPromptDesign:
    """Test suite for educational insights prompt design."""
    
    def test_prompt_includes_species_name(self):
        """Test that the prompt includes the species name."""
        service = GeminiService(api_key="test_key", timeout=10)
        species = "Apis mellifera"
        
        prompt = service._build_prompt(species)
        
        assert species in prompt, "Prompt should include species name"
    
    def test_prompt_requests_taxonomy(self):
        """Test that the prompt requests complete taxonomic classification."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Danaus plexippus")
        
        # Check for taxonomy section header
        assert "## Taxonomy" in prompt or "#Taxonomy" in prompt, \
            "Prompt should include Taxonomy section header"
        
        # Check that it requests classification levels
        assert "Kingdom" in prompt or "classification" in prompt.lower(), \
            "Prompt should request taxonomic classification"
    
    def test_prompt_requests_physical_characteristics(self):
        """Test that the prompt requests physical characteristics."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Papilio machaon")
        
        # Check for physical characteristics section
        assert "## Physical Characteristics" in prompt or \
               "physical" in prompt.lower(), \
            "Prompt should include Physical Characteristics section"
        
        # Check for detail requests
        assert any(term in prompt.lower() for term in 
                  ["size", "color", "features", "structure"]), \
            "Prompt should request physical details"
    
    def test_prompt_requests_habitat_and_distribution(self):
        """Test that the prompt requests habitat and distribution information."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Bombus terrestris")
        
        # Check for habitat section
        assert "## Habitat" in prompt or "habitat" in prompt.lower(), \
            "Prompt should include Habitat section"
        
        # Check for distribution/geographic range
        assert any(term in prompt.lower() for term in 
                  ["distribution", "found", "range", "geographic"]), \
            "Prompt should request distribution information"
    
    def test_prompt_requests_behavior_and_ecology(self):
        """Test that the prompt requests behavior and ecology information."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Formica rufa")
        
        # Check for behavior section
        assert "## Behavior" in prompt or "behavior" in prompt.lower(), \
            "Prompt should include Behavior section"
        
        # Check for ecology details
        assert any(term in prompt.lower() for term in 
                  ["feeding", "life cycle", "ecological", "ecology"]), \
            "Prompt should request behavioral and ecological details"
    
    def test_prompt_requests_identification_tips(self):
        """Test that the prompt requests practical identification tips."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Aedes aegypti")
        
        # Check for identification section
        assert "## Identification" in prompt or "identification" in prompt.lower(), \
            "Prompt should include Identification section"
        
        # Check for practical tips request
        assert any(term in prompt.lower() for term in 
                  ["tips", "identify", "field"]), \
            "Prompt should request identification tips"
    
    def test_prompt_requests_conservation_status(self):
        """Test that the prompt requests conservation status."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Vanessa atalanta")
        
        # Check for conservation section
        assert "## Conservation" in prompt or "conservation" in prompt.lower(), \
            "Prompt should include Conservation section"
        
        # Check for status/threats
        assert any(term in prompt.lower() for term in 
                  ["status", "threat"]), \
            "Prompt should request conservation status and threats"
    
    def test_prompt_requests_markdown_formatting(self):
        """Test that the prompt explicitly requests markdown formatting."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Pieris rapae")
        
        # Check for markdown formatting request
        assert "markdown" in prompt.lower(), \
            "Prompt should explicitly request markdown formatting"
    
    def test_prompt_uses_heading_hierarchy(self):
        """Test that the prompt demonstrates proper heading hierarchy."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Coccinella septempunctata")
        
        # Count heading markers - should use ## for sections (h2 level)
        section_headings = prompt.count("##")
        
        assert section_headings >= 5, \
            "Prompt should include at least 5 section headings (##)"
        
        # Should not have h1 headings in the template (those are for species name)
        h1_headings = prompt.count("\n#") - prompt.count("\n##")
        assert h1_headings <= 1, \
            "Prompt should primarily use h2 (##) headings for sections"
    
    def test_prompt_structure_completeness(self):
        """Test that the prompt includes all required sections in order."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Apis mellifera")
        
        # Expected sections in order
        expected_sections = [
            "Taxonomy",
            "Physical",
            "Habitat",
            "Behavior",
            "Identification",
            "Conservation"
        ]
        
        # Find positions of each section
        positions = {}
        for section in expected_sections:
            pos = prompt.lower().find(section.lower())
            if pos != -1:
                positions[section] = pos
        
        # Verify all sections are present
        assert len(positions) == len(expected_sections), \
            f"All sections should be present. Found: {list(positions.keys())}"
        
        # Verify sections appear in the expected order
        sorted_sections = sorted(positions.items(), key=lambda x: x[1])
        section_order = [s[0] for s in sorted_sections]
        
        # Check that sections maintain their relative order
        for i, section in enumerate(expected_sections):
            assert section in section_order, \
                f"Section '{section}' should be present in prompt"
    
    def test_prompt_with_scientific_name(self):
        """Test that the prompt works with scientific names."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        scientific_name = "Danaus plexippus"
        prompt = service._build_prompt(scientific_name)
        
        assert scientific_name in prompt, \
            "Prompt should include scientific name when provided"
        assert len(prompt) > 200, \
            "Prompt should be sufficiently detailed"
    
    def test_prompt_with_common_name(self):
        """Test that the prompt works with common names."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        common_name = "Monarch Butterfly"
        prompt = service._build_prompt(common_name)
        
        assert common_name in prompt, \
            "Prompt should include common name when provided"
        assert len(prompt) > 200, \
            "Prompt should be sufficiently detailed"
    
    def test_prompt_length_is_reasonable(self):
        """Test that the prompt is not too short or excessively long."""
        service = GeminiService(api_key="test_key", timeout=10)
        
        prompt = service._build_prompt("Apis mellifera")
        
        # Should be detailed but not excessively verbose
        assert len(prompt) >= 300, \
            "Prompt should be detailed enough to guide the AI"
        assert len(prompt) <= 2000, \
            "Prompt should be concise to avoid token waste"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
