"""OCR API Module - Cloud API-based OCR backends (Gemma 3, Gemini, etc.)

Usage:
    from workflow.ocr_api import GeminiOCR, check_gemini_available

Uses Google AI Studio API with Gemma 3 models (or Gemini as fallback).
These are cloud-based alternatives to local VLM OCR that require API keys
but don't need local GPU/llama-server.

Default model: gemma-3-27b-it (Gemma 3 27B instruction-tuned)
"""

from .gemini_ocr import (
    GeminiOCR,
    check_gemini_available,
    get_gemini_ocr,
    reset_gemini_ocr,
    GEMMA_MODELS,
    GEMINI_MODELS,
    ALL_MODELS,
    DEFAULT_MODEL as GEMINI_DEFAULT_MODEL,
)

__all__ = [
    'GeminiOCR',
    'check_gemini_available',
    'get_gemini_ocr',
    'reset_gemini_ocr',
    'GEMMA_MODELS',
    'GEMINI_MODELS',
    'ALL_MODELS',
    'GEMINI_DEFAULT_MODEL',
]
