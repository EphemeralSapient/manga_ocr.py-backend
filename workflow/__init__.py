"""
Workflow Module - Detection, OCR, Inpainting, and ONNX Runtime management

Usage:
    from workflow import detect_mode, create_session, detect_all
    from workflow import grid_bubbles, run_ocr, map_ocr
    from workflow import create_inpainter
    from workflow.ort import get_provider_info
"""

# Detection
from .detect import (
    detect_mode,
    create_session,
    detect_all,
    CROP_PADDING,
    THRESHOLD,
)

# OCR
from .ocr import (
    grid_bubbles,
    run_ocr,
    run_ocr_on_bubbles,
    map_ocr,
    reset_vlm_ocr,
    reset_api_ocr,
    OCR_URL,
    HAS_LFM_OCR,  # Backwards compat
    HAS_VLM_OCR,
    HAS_GEMINI_OCR,
)

# Inpainting
from .inpaint import (
    create_inpainter,
    Inpainter,
)

# Text Segmentation
from .text_seg import (
    create_text_segmenter,
    get_text_segmenter,
    reset_text_segmenter,
    TextSegmenter,
)

# Rendering
from .render import (
    render_text_on_image,
    fit_text,
    get_font,
)

# Translation
from .translate import (
    translate_texts,
    BATCH_SIZE as TRANSLATION_BATCH_SIZE,
)

# Gemini Translation
try:
    from .gemini_translate import (
        translate_texts_gemini,
        check_gemini_available as check_gemini_translate_available,
        BATCH_SIZE as GEMINI_TRANSLATION_BATCH_SIZE,
    )
    HAS_GEMINI_TRANSLATE = True
except ImportError:
    translate_texts_gemini = None
    check_gemini_translate_available = lambda: False
    GEMINI_TRANSLATION_BATCH_SIZE = 33
    HAS_GEMINI_TRANSLATE = False

# ONNX Runtime
from .ort import (
    get_best_provider,
    get_provider_info,
    create_session as create_ort_session,
    create_session_with_info,
    is_gpu_available,
    is_cuda_available,
    is_tensorrt_available,
    is_coreml_available,
)

__all__ = [
    # Detection
    'detect_mode', 'create_session', 'detect_all', 'CROP_PADDING', 'THRESHOLD',
    # OCR
    'grid_bubbles', 'run_ocr', 'run_ocr_on_bubbles', 'map_ocr', 'reset_vlm_ocr', 'reset_api_ocr',
    'OCR_URL', 'HAS_LFM_OCR', 'HAS_VLM_OCR', 'HAS_GEMINI_OCR',
    # Inpainting
    'create_inpainter', 'Inpainter',
    # Text Segmentation
    'create_text_segmenter', 'get_text_segmenter', 'reset_text_segmenter', 'TextSegmenter',
    # Rendering
    'render_text_on_image', 'fit_text', 'get_font',
    # Translation
    'translate_texts', 'TRANSLATION_BATCH_SIZE',
    'translate_texts_gemini', 'check_gemini_translate_available',
    'GEMINI_TRANSLATION_BATCH_SIZE', 'HAS_GEMINI_TRANSLATE',
    # ORT
    'get_best_provider', 'get_provider_info', 'create_ort_session',
    'create_session_with_info', 'is_gpu_available', 'is_cuda_available',
    'is_tensorrt_available', 'is_coreml_available',
]
