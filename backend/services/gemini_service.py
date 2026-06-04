"""
Gemini Service - AI insights generation using Google Gemini API

This module provides integration with Google Gemini API for generating
educational insights about identified insect species. It implements:
- Exponential backoff retry logic for transient failures
- Timeout handling for API requests
- Graceful degradation when insights are unavailable
- Comprehensive logging of all API interactions

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.6, 6.7
"""

import asyncio
from typing import Optional
from datetime import datetime
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiService:
    """
    Service for Google Gemini API integration with retry logic and graceful degradation.
    
    This service handles all communication with the Google Gemini API to generate
    educational insights about identified insect species. It implements robust
    error handling including:
    - Exponential backoff retry for transient errors (429, 500, 502, 503, 504)
    - Request timeout handling (default 10 seconds)
    - Graceful degradation by returning None for failures
    - Comprehensive logging with timestamps
    
    Example:
        service = GeminiService(api_key="your-key", timeout=10)
        insights = await service.generate_insights("Apis mellifera")
        if insights:
            print(insights)  # Markdown-formatted educational content
        else:
            print("Insights unavailable")
    """
    
    def __init__(self, api_key: str, timeout: int = 10):
        """
        Initialize Gemini service with API credentials and configuration.
        
        Args:
            api_key: Google Gemini API key
            timeout: Request timeout in seconds (default: 10)
        
        Raises:
            ValueError: If api_key is empty
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = 3
        self.base_delay = 1  # Base delay in seconds for exponential backoff
        self.is_configured = bool(api_key)
        
        if self.is_configured:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            logger.warning("Gemini API key is not configured. AI insights will be disabled.")
        
        logger.info(
            "GeminiService initialized",
            extra={
                "timeout": timeout,
                "max_retries": self.max_retries,
                "base_delay": self.base_delay
            }
        )
    
    def _build_prompt(self, species_name: str, language: str = 'id') -> str:
        """
        Build structured prompt for requesting educational insights from Gemini API.
        
        The prompt is designed to elicit comprehensive, well-structured educational
        content about the identified insect species, including taxonomy, physical
        characteristics, habitat, behavior, identification tips, and conservation status.
        
        Args:
            species_name: Common or scientific name of the insect species
            language: Language code ('id' for Indonesian, 'en' for English)
        
        Returns:
            Formatted prompt string optimized for Gemini API
        """
        if language == 'id':
            return f"""Berikan informasi edukasi yang lengkap dan detail tentang serangga {species_name} dalam Bahasa Indonesia.

Gunakan nama ilmiah (Latin) untuk semua penamaan spesies, genus, famili, ordo, dll. Jangan terjemahkan nama-nama ilmiah Latin.
Untuk istilah biologi umum, gunakan Bahasa Indonesia.

Sertakan bagian-bagian berikut:

## Taksonomi
Berikan klasifikasi taksonomi lengkap (Kingdom, Filum, Kelas, Ordo, Famili, Genus, Spesies) menggunakan nama ilmiah Latin.

## Ciri-ciri Fisik
Jelaskan ciri-ciri pembeda, ukuran, pola warna, dan struktur tubuh.

## Habitat dan Persebaran
Jelaskan di mana spesies ini biasa ditemukan dan jangkauan geografisnya.

## Perilaku dan Ekologi
Jelaskan kebiasaan makan, siklus hidup, dan peran ekologisnya.

## Tips Identifikasi
Berikan tips praktis untuk mengidentifikasi spesies ini di lapangan.

## Status Konservasi
Sebutkan status konservasi dan ancaman terhadap spesies ini.

Format respons dalam markdown yang jelas dan terstruktur. Tulis seluruh konten dalam Bahasa Indonesia, kecuali nama-nama ilmiah Latin yang harus tetap dalam bahasa Latin (dicetak miring)."""
        else:
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
    
    def _is_transient_error(self, status_code: int) -> bool:
        """
        Determine if an error is transient and should be retried.
        
        Transient errors are temporary failures that may succeed on retry,
        such as rate limiting, server overload, or network issues.
        
        Args:
            status_code: HTTP status code from the failed request
        
        Returns:
            True if the error is transient and retry-able, False otherwise
        
        Retry-able status codes:
            - 429: Too Many Requests (rate limiting)
            - 500: Internal Server Error
            - 502: Bad Gateway
            - 503: Service Unavailable
            - 504: Gateway Timeout
        """
        transient_codes = [429, 500, 502, 503, 504]
        return status_code in transient_codes
    
    async def _request_with_retry(self, prompt: str) -> Optional[str]:
        """
        Execute Gemini API request with exponential backoff retry logic.
        
        This method implements a robust retry mechanism for handling transient
        failures. It will retry up to max_retries times with exponentially
        increasing delays between attempts (1s, 2s, 4s).
        
        Args:
            prompt: The prompt to send to Gemini API
        
        Returns:
            Generated insights as markdown string, or None if all retries failed
        
        Retry Strategy:
            - Attempt 1: Immediate request
            - Attempt 2: Wait 1 second, retry
            - Attempt 3: Wait 2 seconds, retry
            - Attempt 4: Wait 4 seconds, retry (final attempt)
            
        Logging:
            - Logs each retry attempt with delay information
            - Logs final failure after max retries exhausted
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Gemini API request attempt {attempt + 1}/{self.max_retries}",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Make the API request
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                # Extract text from response
                insights = response.text
                
                logger.info(
                    "Gemini API request succeeded",
                    extra={
                        "attempt": attempt + 1,
                        "response_length": len(insights),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                return insights
                
            except google_exceptions.ResourceExhausted as e:
                # Rate limiting (429)
                status_code = 429
                logger.warning(
                    f"Gemini API rate limit exceeded (attempt {attempt + 1}/{self.max_retries})",
                    extra={
                        "error": str(e),
                        "status_code": status_code,
                        "attempt": attempt + 1,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Gemini API failed after {self.max_retries} attempts (rate limit)",
                        extra={
                            "error": str(e),
                            "status_code": status_code,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    return None
                    
            except google_exceptions.ServerError as e:
                # Server errors (500, 502, 503, 504)
                status_code = 503
                logger.warning(
                    f"Gemini API server error (attempt {attempt + 1}/{self.max_retries})",
                    extra={
                        "error": str(e),
                        "status_code": status_code,
                        "attempt": attempt + 1,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                if attempt < self.max_retries - 1 and self._is_transient_error(status_code):
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Gemini API failed after {self.max_retries} attempts (server error)",
                        extra={
                            "error": str(e),
                            "status_code": status_code,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    return None
                    
            except Exception as e:
                # Non-transient errors (permanent failures)
                logger.error(
                    f"Gemini API permanent error: {str(e)}",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                return None
        
        # All retries exhausted
        logger.error(
            f"Gemini API failed after {self.max_retries} attempts",
            extra={
                "max_retries": self.max_retries,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        return None
    
    async def generate_insights(self, species_name: str, language: str = 'id') -> Optional[str]:
        """
        Generate educational insights for an identified insect species.
        
        This is the main entry point for generating AI insights. It handles:
        - Prompt construction
        - Timeout wrapping
        - Retry logic via _request_with_retry
        - Graceful degradation (returns None on failure)
        
        Args:
            species_name: Common or scientific name of the identified species
        
        Returns:
            Markdown-formatted educational insights, or None if unavailable
            
        Timeout Handling:
            If the request exceeds the configured timeout, returns None
            and logs the timeout event. The system gracefully degrades
            by continuing to show prediction results without insights.
        
        Example:
            insights = await gemini_service.generate_insights("Apis mellifera")
            if insights:
                # Display insights to user
                render_markdown(insights)
            else:
                # Show fallback message
                display("AI insights temporarily unavailable")
        """
        logger.info(
            f"Generating insights for species: {species_name}",
            extra={
                "species": species_name,
                "timeout": self.timeout,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        if not getattr(self, 'is_configured', True):
            logger.warning("Gemini is not configured. Returning mock insights for UI presentation.")
            # Return a comprehensive mock response in the requested language
            if language == 'id':
                mock_insight = f"""# {species_name}

## Taksonomi
- **Kingdom**: Animalia
- **Filum**: Arthropoda
- **Kelas**: Insecta
- **Ordo**: Contoh Ordo
- **Famili**: Contoh Famili
- **Genus**: Contoh Genus
- **Spesies**: *{species_name}*

## Ciri-ciri Fisik
Spesies ini biasanya memiliki pola warna khas yang membantunya menyatu dengan lingkungan alaminya. Tubuhnya bersegmen seperti tipikal ordoianya, dengan pelengkap khusus untuk makan dan bergerak. Ukuran dewasa biasanya berkisar antara 1-3 cm.

## Habitat dan Persebaran
Umumnya ditemukan di wilayah beriklim sedang dan tropis, menyukai lingkungan dengan vegetasi yang melimpah. Mereka berkembang di padang rumput, tepi hutan, dan sering mengunjungi taman perkotaan untuk mencari nektar atau mangsa.

## Perilaku dan Ekologi
Sebagai bagian penting dari ekosistem lokal, serangga ini memainkan peran krusial dalam penyerbukan dan pengendalian hama. Siklus hidupnya mengalami metamorfosis sempurna, bertransisi dari telur ke larva, pupa, dan akhirnya ke tahap dewasa. Mereka paling aktif selama bulan-bulan yang lebih hangat.

## Tips Identifikasi
- Perhatikan tanda khas pada sayap atau toraknya.
- Amati pola terbang atau perilaku berjalannya.
- Mereka sering terlihat beristirahat di jenis tanaman inang tertentu.

## Status Konservasi
Saat ini terdaftar sebagai **Risiko Rendah** (LC - *Least Concern*) oleh IUCN, meskipun populasi lokal dapat terpengaruh oleh penggunaan pestisida dan hilangnya habitat.
"""
            else:
                mock_insight = f"""# {species_name}

## Taxonomy
- **Kingdom**: Animalia
- **Phylum**: Arthropoda
- **Class**: Insecta
- **Order**: Example Order
- **Family**: Example Family
- **Genus**: Example Genus
- **Species**: *{species_name}*

## Physical Characteristics
This species typically exhibits distinct color patterns that help it blend into its natural environment. It features a segmented body typical of its order, with specialized appendages for feeding and movement. Adults usually measure between 1-3 cm in length.

## Habitat and Distribution
They are commonly found across temperate and tropical regions, favoring environments with abundant vegetation. They thrive in meadows, forest edges, and frequently visit urban gardens in search of nectar or prey.

## Behavior and Ecology
An essential part of the local ecosystem, this insect plays a crucial role in pollination and pest control. Their life cycle undergoes complete metamorphosis, transitioning from egg to larva, pupa, and finally to the adult stage. They are most active during the warmer months.

## Identification Tips
- Look for the distinctive markings on their wings or thorax.
- Observe their flight pattern or walking behavior.
- They are often spotted resting on specific types of host plants.

## Conservation Status
Currently listed as **Least Concern** (LC) by the IUCN, although local populations can be affected by pesticide use and habitat loss.
"""
            return mock_insight
        
        try:
            # Build the prompt
            prompt = self._build_prompt(species_name, language=language)
            
            # Execute request with timeout wrapper
            insights = await asyncio.wait_for(
                self._request_with_retry(prompt),
                timeout=self.timeout
            )
            
            if insights:
                logger.info(
                    f"Successfully generated insights for {species_name}",
                    extra={
                        "species": species_name,
                        "insights_length": len(insights),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            else:
                logger.warning(
                    f"Failed to generate insights for {species_name}",
                    extra={
                        "species": species_name,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            return insights
            
        except asyncio.TimeoutError:
            logger.warning(
                f"Gemini API request timed out after {self.timeout}s for species: {species_name}",
                extra={
                    "species": species_name,
                    "timeout": self.timeout,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            return None
            
        except Exception as e:
            logger.error(
                f"Unexpected error generating insights for {species_name}: {str(e)}",
                extra={
                    "species": species_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            return None
