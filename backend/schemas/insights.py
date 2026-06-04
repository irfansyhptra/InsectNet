"""
Insights Schemas

Pydantic models for AI insights API requests and responses.
Validates: Requirements 5.7, 15.6
"""

from pydantic import BaseModel, Field


class InsightsRequest(BaseModel):
    """Request for AI-generated educational insights about a species."""
    
    species: str = Field(
        ..., 
        min_length=1,
        description="Species name (common or scientific) to generate insights for"
    )
    language: str = Field(
        "id",
        description="Language code for insights output: 'id' for Indonesian (default), 'en' for English"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "species": "Ladybug",
                "language": "id"
            }
        }


class InsightsResponse(BaseModel):
    """
    AI insights response with graceful degradation support.
    
    Returns insights when Gemini API is available, or null with error details
    when the service is unavailable. Always returns 200 OK for graceful degradation.
    """
    
    insights: str | None = Field(
        None, 
        description="Markdown-formatted educational content about the species, or null if unavailable"
    )
    available: bool = Field(
        ..., 
        description="Whether insights were successfully generated"
    )
    error: str | None = Field(
        None, 
        description="Error message explaining why insights are unavailable"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "insights": "## Taxonomy\n\nKingdom: Animalia\nPhylum: Arthropoda\nClass: Insecta\nOrder: Coleoptera\nFamily: Coccinellidae\n\n## Physical Characteristics\n\nLadybugs are small beetles, typically 0.3 to 0.4 inches (8-10 mm) in length. They have dome-shaped bodies with distinctive red or orange wing covers (elytra) adorned with black spots...",
                    "available": True,
                    "error": None
                },
                {
                    "insights": None,
                    "available": False,
                    "error": "AI insights service temporarily unavailable. Please try again later."
                }
            ]
        }
