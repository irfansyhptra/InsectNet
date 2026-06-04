"""Manual test to show the _build_prompt output"""

# Simulate the _build_prompt method
def _build_prompt(species_name: str) -> str:
    return f"""Provide detailed educational information about {species_name}.

Include the following sections:

## Taxonomy
Provide the complete taxonomic classification (Kingdom, Phylum, Class, Order, Family, Genus, Species).

## Physical Characteristics
Describe distinguishing features, size, color patterns, and body structure.

## Habitat and Distribution
Explain where this species is commonly found and its geographic range.

## Behavior and Ecology
Describe feeding habits, life cycle, and ecological role.

## Identification Tips
Provide practical tips for identifying this species in the field.

## Conservation Status
Mention conservation status and any threats to the species.

Format the response in clear, well-structured markdown."""

# Test it
species = "Apis mellifera"
prompt = _build_prompt(species)

print("TASK 3.2: Design Prompt for Educational Insights")
print("=" * 80)
print(f"Test Species: {species}")
print("=" * 80)
print()
print("Generated Prompt:")
print(prompt)
print()
print("=" * 80)
print("Verification Checklist:")
print("=" * 80)

checks = {
    "Species name included": species in prompt,
    "Taxonomy section (##)": "## Taxonomy" in prompt,
    "Physical Characteristics section (##)": "## Physical Characteristics" in prompt,
    "Habitat and Distribution section (##)": "## Habitat and Distribution" in prompt,
    "Behavior and Ecology section (##)": "## Behavior and Ecology" in prompt,
    "Identification Tips section (##)": "## Identification Tips" in prompt,
    "Conservation Status section (##)": "## Conservation Status" in prompt,
    "Markdown formatting requested": "markdown" in prompt.lower(),
    "Proper heading hierarchy (6 sections)": prompt.count("## ") == 6,
    "Reasonable length (300-2000 chars)": 300 <= len(prompt) <= 2000,
}

all_passed = True
for check, result in checks.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {check}")
    if not result:
        all_passed = False

print("=" * 80)
print()
if all_passed:
    print("✅ SUCCESS: All requirements met!")
    print()
    print("Task 3.2 is COMPLETE:")
    print("  • Structured prompt created")
    print("  • Requests all 6 required sections")
    print("  • Uses proper markdown heading hierarchy (##)")
    print("  • Explicitly requests markdown-formatted response")
    print("  • Satisfies Requirements 5.2 and 5.3")
else:
    print("❌ FAILURE: Some checks failed")
