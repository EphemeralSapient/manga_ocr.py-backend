"""VLM OCR & LLM Translation - llama.cpp based OCR/translation for non-Windows"""

from .vlm_ocr import VlmOCR, check_vlm_available, create_ocr_grid, VLM_MODEL
from .llm_translate import LlmTranslator, check_translate_server, TRANSLATE_MODEL

# Backwards compatibility aliases
LfmOCR = VlmOCR
check_lfm_available = check_vlm_available

__all__ = [
    'VlmOCR', 'check_vlm_available', 'create_ocr_grid', 'VLM_MODEL',
    'LlmTranslator', 'check_translate_server', 'TRANSLATE_MODEL',
    # Backwards compat
    'LfmOCR', 'check_lfm_available',
]
