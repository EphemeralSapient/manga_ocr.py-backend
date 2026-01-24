#!/usr/bin/env python3
"""
Manga Processing Server
Full pipeline: Detection -> OCR -> Translation -> Inpainting -> Rendering

Endpoints:
  POST /api/v1/process        - Main pipeline: detect -> OCR -> translate -> AOT inpaint -> render
  POST /api/v1/process/stream - SSE streaming version
  POST /api/v1/process/legacy - Legacy sequential mode

Request: multipart/form-data with 'images' field (multiple files)
Response: JSON with base64 encoded output images
"""

import os
import sys
import site
import ctypes

# Preload TensorRT libraries (must be before importing ORT)
def _preload_tensorrt():
    try:
        site_pkgs = site.getsitepackages()[0]
        trt_libs = os.path.join(site_pkgs, 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            # Load in dependency order
            for lib in ['libnvinfer.so.10', 'libnvinfer_plugin.so.10', 'libnvonnxparser.so.10']:
                lib_path = os.path.join(trt_libs, lib)
                if os.path.exists(lib_path):
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        pass  # TensorRT optional, will fallback to CUDA/CPU

_preload_tensorrt()

# Performance tuning environment variables (set before importing ORT)
os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_ENABLE', '1')
os.environ.setdefault('ORT_TENSORRT_FP16_ENABLE', '1')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import io
import json
import base64
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Response
import queue
import threading

# Import workflow components
from workflow import detect_mode, create_session, detect_all, CROP_PADDING
from workflow import grid_bubbles, run_ocr, run_ocr_on_bubbles, map_ocr, reset_vlm_ocr, reset_api_ocr, HAS_LFM_OCR, HAS_VLM_OCR, HAS_GEMINI_OCR
from workflow import create_inpainter, Inpainter
from workflow import create_text_segmenter, TextSegmenter
from workflow import render_text_on_image
from workflow import translate_texts
from workflow import translate_texts_gemini, HAS_GEMINI_TRANSLATE


import re

# CJK Unicode ranges for detecting text that needs translation
_CJK_RANGES = (
    '\u4e00-\u9fff'    # CJK Unified Ideographs (Chinese)
    '\u3040-\u309f'    # Hiragana (Japanese)
    '\u30a0-\u30ff'    # Katakana (Japanese)
    '\uac00-\ud7af'    # Hangul Syllables (Korean)
    '\u1100-\u11ff'    # Hangul Jamo (Korean)
    '\u3130-\u318f'    # Hangul Compatibility Jamo
    '\uff66-\uff9f'    # Halfwidth Katakana
)
_CJK_PATTERN = re.compile(f'[{_CJK_RANGES}]')


def should_translate(text: str, min_cjk_ratio: float = 0.2) -> bool:
    """
    Check if text should be translated based on CJK content ratio.

    Returns True if text has enough CJK characters to warrant translation.
    Skips text that's mostly English/numbers (like stats: LV:100, HP:2000, MAX:999).

    Args:
        text: Text to check
        min_cjk_ratio: Minimum ratio of CJK characters (default 0.2 = 20%)
    """
    if not text:
        return False

    # Count CJK characters
    cjk_chars = len(_CJK_PATTERN.findall(text))

    # Count meaningful characters (exclude spaces, punctuation)
    meaningful = sum(1 for c in text if c.isalnum() or ord(c) > 127)

    if meaningful == 0:
        return False

    # If mostly CJK (>30%), translate it
    # This skips stats panels like "LV:100 HP:2000 ポチ" where CJK is minority
    return cjk_chars / meaningful >= min_cjk_ratio


def contains_cjk(text: str) -> bool:
    """Check if text contains any CJK - use should_translate() for smarter check."""
    return bool(_CJK_PATTERN.search(text))


def merge_ocr_texts_for_translation(texts: list, bubble_box: list = None) -> str:
    """
    Merge OCR text items for translation.

    For VLM OCR: returns concatenated text (VLM understands reading order)
    For OneOCR (multiple line-level bboxes): returns bracketed format [line1,line2,...]
    to indicate uncertain reading order (let LLM determine correct order from context)

    Detection: OneOCR has multiple texts with small bboxes (< 70% of bubble area)
    """
    if not texts:
        return ""

    text_strings = [t.get('text', '').strip() for t in texts if t.get('text', '').strip()]

    if len(text_strings) == 0:
        return ""
    elif len(text_strings) == 1:
        # Single text - return as-is (both VLM and single-line OneOCR)
        return text_strings[0]
    else:
        # Multiple texts - check if OneOCR (small bboxes) or VLM (cell-level)
        is_oneocr = False

        if bubble_box and len(bubble_box) == 4:
            bx1, by1, bx2, by2 = bubble_box
            bubble_area = (bx2 - bx1) * (by2 - by1)

            # Calculate total text bbox area
            text_area = 0
            for t in texts:
                bbox = t.get('bbox')
                if bbox and len(bbox) == 4:
                    tx1, ty1, tx2, ty2 = bbox
                    if tx2 > tx1 and ty2 > ty1:
                        text_area += (tx2 - tx1) * (ty2 - ty1)

            # OneOCR: text bboxes cover < 70% of bubble (precise line-level)
            # VLM: text bboxes cover ~100% of bubble (cell-level)
            is_oneocr = bubble_area > 0 and text_area < bubble_area * 0.7
        else:
            # No bubble_box provided - assume OneOCR if multiple texts
            is_oneocr = True

        if is_oneocr:
            # OneOCR - newline-separated for LLM to understand context and order
            return "\n".join(text_strings)
        else:
            # VLM - concatenate (VLM understands reading order)
            return "".join(text_strings)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Loading - Use centralized config module
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    load_config, CONFIG_FILE,
    get_ocr_method, get_translate_method, get_target_language,
    get_ocr_model, get_ocr_mmproj, get_translate_model, get_cerebras_api_key,
    get_ocr_server_url, get_translate_server_url, get_auto_start_servers,
    get_llama_cli_path, get_llama_context_size, get_llama_gpu_layers,
    get_server_port, needs_llama, get_sequential_model_loading, get_streaming_enabled,
    get_ocr_grid_max_cells, get_translation_batch_size,
    find_llama, download_mmproj, build_llama_command,
    get_text_seg_enabled, get_text_seg_model, get_text_seg_input_size,
    is_api_ocr_method, get_gemini_api_key, get_gemini_model, get_gemini_translate_model,
    get_debug_l1_boxes, get_debug_l2_boxes, get_debug_ocr_text, get_debug_save_overlay,
)

# Load configuration
CONFIG = load_config()

# Apply config to environment/globals
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", get_cerebras_api_key())
OCR_METHOD = get_ocr_method()
TRANSLATE_METHOD = get_translate_method()
AUTO_START_SERVERS = get_auto_start_servers()
SEQUENTIAL_MODEL_LOADING = get_sequential_model_loading()
STREAMING_ENABLED = get_streaming_enabled()
OCR_BATCH_SIZE = get_ocr_grid_max_cells()  # Bubbles per OCR grid (default: 9)
TRANSLATE_BATCH_SIZE = get_translation_batch_size()  # Texts per translation batch (default: 25)

# Determine default translate mode based on config
# NOTE: VLM translation disabled - Qwen 3 VL hallucinates instead of reading text
# Always use separate OCR (VLM) + Translation (HunyuanMT or API) flow
if TRANSLATE_METHOD in ("qwen_vlm", "hunyuan_mt"):
    DEFAULT_TRANSLATE_MODE = "local"  # Use HunyuanMT for translation
elif TRANSLATE_METHOD == "gemini_translate":
    DEFAULT_TRANSLATE_MODE = "gemini"  # Use Gemini API for translation
else:
    DEFAULT_TRANSLATE_MODE = "api"  # Use Cerebras API

# Override VLM/LLM URLs from config
os.environ.setdefault('LLAMA_SERVER_URL', get_ocr_server_url())
os.environ.setdefault('TRANSLATE_SERVER_URL', get_translate_server_url())
os.environ.setdefault('VLM_MODEL', get_ocr_model())
os.environ.setdefault('TRANSLATE_MODEL', get_translate_model())

# Import VLM/LLM components for auto-start
try:
    from workflow.ocr_vlm import (
        VLM_MODEL, TRANSLATE_MODEL,
        check_vlm_available as check_vlm_server,
        check_translate_server,
        LlmTranslator
    )
    HAS_LOCAL_TRANSLATE = True
except ImportError:
    VLM_MODEL, TRANSLATE_MODEL = None, None
    check_vlm_server = check_translate_server = lambda: False
    LlmTranslator = None
    HAS_LOCAL_TRANSLATE = False


def _kill_existing_llama_servers():
    """Kill any existing llama-server processes."""
    import subprocess
    import platform

    try:
        if platform.system() == 'Windows':
            subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'],
                          capture_output=True, timeout=5)
        else:
            # Kill all llama-server processes
            result = subprocess.run(['pkill', '-9', 'llama-server'],
                                   capture_output=True, timeout=5)
            if result.returncode == 0:
                print("  Killed existing llama-server processes")
                import time
                time.sleep(1)  # Wait for ports to be released
    except Exception:
        pass  # Ignore errors (no processes to kill)


def _start_llama_servers():
    """Auto-start llama servers for OCR and translation."""
    import subprocess

    _kill_existing_llama_servers()

    llama_server = find_llama()
    if not llama_server:
        print("  llama-server not found, skipping auto-start")
        print("  Install llama.cpp or set llama_cli_path in config.json")
        return

    print(f"  Using llama-server: {llama_server}")

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

    log_dir = os.path.join(os.path.dirname(__file__), '.llama_logs')
    os.makedirs(log_dir, exist_ok=True)

    ocr_model = get_ocr_model()
    translate_model = get_translate_model()
    mmproj_path = download_mmproj(get_ocr_mmproj())

    # Start OCR server (port 8080)
    if ocr_model:
        print(f"  Starting OCR server ({ocr_model})...")
        cmd = build_llama_command(llama_server, ocr_model, '8080', mmproj_path)
        with open(os.path.join(log_dir, 'ocr.log'), 'w') as log:
            subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    # Start Translation server (port 8081) - only if using local translation
    if TRANSLATE_METHOD in ("qwen_vlm", "hunyuan_mt") and translate_model:
        if not (OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8") and TRANSLATE_METHOD == "qwen_vlm"):
            print(f"  Starting Translation server ({translate_model})...")
            # Translation doesn't need mmproj
            ctx = str(get_llama_context_size())
            ngl = str(get_llama_gpu_layers())
            with open(os.path.join(log_dir, 'translate.log'), 'w') as log:
                subprocess.Popen(
                    [llama_server, '-hf', translate_model, '--port', '8081', '-c', ctx, '-ngl', ngl],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )

    # Wait for servers to be ready (up to 300 seconds for model download)
    if needs_llama():
        import time
        import requests

        def _check_server_health(url):
            """Strict health check - only returns True if server responds OK."""
            try:
                r = requests.get(f"{url}/health", timeout=2)
                return r.status_code == 200
            except:
                return False

        timeout_seconds = 300  # 5 minutes for model downloads
        print(f"  Waiting for servers to start (up to {timeout_seconds}s, may download models)...")

        check_ocr = OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8")
        check_trans = TRANSLATE_METHOD in ("qwen_vlm", "hunyuan_mt") and not (OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8") and TRANSLATE_METHOD == "qwen_vlm")

        ocr_url = get_ocr_server_url()
        trans_url = get_translate_server_url()

        ocr_ok = not check_ocr  # True if not checking
        trans_ok = not check_trans  # True if not checking

        for i in range(timeout_seconds):
            if check_ocr and not ocr_ok:
                ocr_ok = _check_server_health(ocr_url)
                if ocr_ok:
                    print(f"    OCR server ready ({i}s)")

            if check_trans and not trans_ok:
                trans_ok = _check_server_health(trans_url)
                if trans_ok:
                    print(f"    Translation server ready ({i}s)")

            if ocr_ok and trans_ok:
                print(f"  All servers ready in {i}s")
                break

            time.sleep(1)
            if i > 0 and i % 15 == 0:
                status = []
                if check_ocr:
                    status.append(f"OCR={'ready' if ocr_ok else 'loading'}")
                if check_trans:
                    status.append(f"Translate={'ready' if trans_ok else 'loading'}")
                print(f"    Still waiting... ({i}s) [{', '.join(status)}]")

        # Final status check after timeout
        if not (ocr_ok and trans_ok):
            print(f"  Warning: Timeout after {timeout_seconds}s, but servers may still be starting...")
            print(f"    Check logs in .llama_logs/ for details")


# ─────────────────────────────────────────────────────────────────────────────
# Sequential Model Manager - For low VRAM GPUs
# Loads only one model at a time: OCR -> kill -> Translate -> kill
# ─────────────────────────────────────────────────────────────────────────────

class SequentialModelManager:
    """
    Manages llama-server instances for sequential model loading.
    Loads one model at a time to minimize VRAM usage.
    Meant for budget GPUs (4-6GB VRAM).
    """

    def __init__(self):
        self._ocr_process = None
        self._translate_process = None
        self._lock = None  # Lazy init threading lock

    def _get_lock(self):
        """Lazy init lock to avoid issues at import time."""
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock

    def _wait_for_server(self, url, timeout=120, check_fn=None):
        """Wait for server to be ready."""
        import time
        check_fn = check_fn or (lambda: False)
        for i in range(timeout):
            try:
                import requests
                r = requests.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except:
                pass

            if check_fn():
                return True

            time.sleep(1)
            if i > 0 and i % 15 == 0:
                print(f"    Still waiting for server... ({i}s)")
        return False

    def _kill_process(self, process):
        """Kill a subprocess and wait for it to terminate."""
        if process is None:
            return

        try:
            import signal
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                process.kill()
                process.wait(timeout=2)
        except Exception as e:
            print(f"  Warning: Error killing process: {e}")

    def start_ocr_server(self):
        """Start OCR llama-server (kills translate server if running)."""
        with self._get_lock():
            if self._translate_process is not None:
                print("  [Sequential] Stopping translation server to free VRAM...")
                self._kill_process(self._translate_process)
                self._translate_process = None
                time.sleep(1)

            if check_vlm_server():
                return True

            llama_server = find_llama()
            if not llama_server:
                print("  [Sequential] llama-server not found")
                return False

            ocr_model = get_ocr_model()
            mmproj_path = download_mmproj(get_ocr_mmproj())

            print(f"  [Sequential] Starting OCR server ({ocr_model})...")
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

            log_dir = os.path.join(os.path.dirname(__file__), '.llama_logs')
            os.makedirs(log_dir, exist_ok=True)

            cmd = build_llama_command(llama_server, ocr_model, '8080', mmproj_path)

            import subprocess
            with open(os.path.join(log_dir, 'ocr_sequential.log'), 'w') as log:
                self._ocr_process = subprocess.Popen(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )

            if self._wait_for_server(get_ocr_server_url(), timeout=120, check_fn=check_vlm_server):
                print(f"  [Sequential] OCR server ready")
                return True
            else:
                print(f"  [Sequential] OCR server failed to start")
                return False

    def stop_ocr_server(self):
        """Stop OCR llama-server to free VRAM."""
        with self._get_lock():
            if self._ocr_process is not None:
                print("  [Sequential] Stopping OCR server...")
                self._kill_process(self._ocr_process)
                self._ocr_process = None
                time.sleep(1)  # Wait for VRAM to be released

            # Also kill any orphaned processes on port 8080
            import subprocess
            import platform
            try:
                if platform.system() != 'Windows':
                    subprocess.run(['pkill', '-f', 'llama-server.*--port.*8080'],
                                  capture_output=True, timeout=5)
            except:
                pass

    def start_translate_server(self):
        """Start translation llama-server (kills OCR server if running)."""
        with self._get_lock():
            if self._ocr_process is not None:
                print("  [Sequential] Stopping OCR server to free VRAM...")
                self._kill_process(self._ocr_process)
                self._ocr_process = None
                time.sleep(1)

            # Also kill any orphaned OCR server
            import subprocess
            import platform
            try:
                if platform.system() != 'Windows':
                    subprocess.run(['pkill', '-f', 'llama-server.*--port.*8080'],
                                  capture_output=True, timeout=5)
                    time.sleep(0.5)
            except:
                pass

            if check_translate_server():
                return True

            llama_server = find_llama()
            if not llama_server:
                print("  [Sequential] llama-server not found")
                return False

            translate_model = get_translate_model()
            ctx = str(get_llama_context_size())
            ngl = str(get_llama_gpu_layers())

            print(f"  [Sequential] Starting translation server ({translate_model})...")
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

            log_dir = os.path.join(os.path.dirname(__file__), '.llama_logs')
            os.makedirs(log_dir, exist_ok=True)

            import subprocess
            with open(os.path.join(log_dir, 'translate_sequential.log'), 'w') as log:
                self._translate_process = subprocess.Popen(
                    [llama_server, '-hf', translate_model, '--port', '8081', '-c', ctx, '-ngl', ngl],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )

            # Wait for server to be ready
            if self._wait_for_server(get_translate_server_url(), timeout=120, check_fn=check_translate_server):
                print(f"  [Sequential] Translation server ready")
                return True
            else:
                print(f"  [Sequential] Translation server failed to start")
                return False

    def stop_translate_server(self):
        """Stop translation llama-server to free VRAM."""
        with self._get_lock():
            if self._translate_process is not None:
                print("  [Sequential] Stopping translation server...")
                self._kill_process(self._translate_process)
                self._translate_process = None
                time.sleep(1)  # Wait for VRAM to be released

            # Also kill any orphaned processes on port 8081
            import subprocess
            import platform
            try:
                if platform.system() != 'Windows':
                    subprocess.run(['pkill', '-f', 'llama-server.*--port.*8081'],
                                  capture_output=True, timeout=5)
            except:
                pass

    def stop_all(self):
        """Stop all llama-server instances."""
        self.stop_ocr_server()
        self.stop_translate_server()


# Global sequential model manager instance
_sequential_manager = None

def get_sequential_manager():
    """Get or create the sequential model manager."""
    global _sequential_manager
    if _sequential_manager is None:
        _sequential_manager = SequentialModelManager()
    return _sequential_manager


app = Flask(__name__)

# Global model instances (lazy loaded)
_detector_session = None
_detector_mode = None
_inpainter = None
_text_segmenter = None

# Text segmentation config
TEXT_SEG_ENABLED = get_text_seg_enabled()
# Force text_seg mask for text clearing instead of OneOCR char bboxes (slower but more precise)
# Can be set via config.json or environment variable: FORCE_TEXT_SEG_MASK=1
from config import get_force_text_seg_mask, get_aot_inpaint_enabled

def get_force_text_seg_mask_runtime() -> bool:
    """Get force_text_seg_mask setting (reads from config each time for dynamic updates)."""
    if os.environ.get('FORCE_TEXT_SEG_MASK', '').lower() in ('1', 'true', 'yes'):
        return True
    return get_force_text_seg_mask()

def get_aot_inpaint_enabled_runtime() -> bool:
    """Get aot_inpaint_enabled setting (reads from config each time for dynamic updates)."""
    if os.environ.get('AOT_INPAINT_ENABLED', '').lower() in ('0', 'false', 'no'):
        return False
    return get_aot_inpaint_enabled()


def get_detector():
    """Lazy load detector model."""
    global _detector_session, _detector_mode
    if _detector_session is None:
        _detector_mode = detect_mode()
        _detector_session, _detector_mode = create_session(_detector_mode)
    return _detector_session, _detector_mode


def get_inpainter() -> Inpainter:
    """Lazy load inpainter model."""
    global _inpainter
    if _inpainter is None:
        _inpainter = create_inpainter()
    return _inpainter


def get_text_segmenter_instance() -> TextSegmenter:
    """Lazy load text segmenter model."""
    global _text_segmenter
    if _text_segmenter is None:
        model_path = get_text_seg_model()
        input_size = get_text_seg_input_size()
        _text_segmenter = create_text_segmenter(model_path, input_size)
    return _text_segmenter


def draw_debug_overlay(images: List[Image.Image], bubbles_l1: List[Dict], bubbles_l2: List[Dict],
                       ocr_data: Dict = None, output_dir: str = None, tab_title: str = None) -> List[Image.Image]:
    """Draw debug visualization on images.

    Args:
        images: Output images to draw on
        bubbles_l1: L1 bubbles (speech bubbles) with 'page_idx', 'box'
        bubbles_l2: L2 bubbles (text-free regions) with 'page_idx', 'box'
        ocr_data: OCR data per page for text overlay
        output_dir: If set, save debug images to this directory
        tab_title: Tab title for saving debug images

    Returns:
        Images with debug overlays drawn
    """
    from PIL import ImageDraw, ImageFont

    debug_l1 = get_debug_l1_boxes()
    debug_l2 = get_debug_l2_boxes()
    debug_ocr = get_debug_ocr_text()
    debug_save = get_debug_save_overlay()

    if not (debug_l1 or debug_l2 or debug_ocr):
        return images

    print(f"  [Debug] Drawing overlays: L1={debug_l1}, L2={debug_l2}, OCR={debug_ocr}")
    if debug_ocr and ocr_data:
        print(f"  [Debug] OCR data keys: {list(ocr_data.keys())}")
        for page_idx, page_data in ocr_data.items():
            print(f"  [Debug] Page {page_idx}: {len(page_data)} bubbles")
            for bd in page_data[:3]:  # Show first 3
                texts = bd.get('texts', [])
                text_preview = ''.join([t.get('text', '')[:20] for t in texts])
                print(f"    box={bd.get('bubble_box')}, text='{text_preview}'")

    # Colors for different labels
    L1_COLOR = (0, 255, 0)         # Green for speech bubbles
    L2_COLOR = (255, 0, 0)         # Red for text-free regions
    OCR_BG_COLOR = (0, 0, 0, 220)  # Black background for OCR text
    OCR_TEXT_COLOR = (255, 255, 0) # Yellow for OCR text

    debug_images = []

    # Try to get a font - use a larger size for visibility
    font = None
    font_size = 14
    try:
        # Try common font paths that support CJK characters
        font_paths = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # macOS CJK
            "/System/Library/Fonts/STHeiti Light.ttc",     # macOS CJK
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS CJK
            "/Library/Fonts/Arial Unicode.ttf",            # macOS Unicode
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux CJK
            "C:/Windows/Fonts/msgothic.ttc",  # Windows CJK
            "C:/Windows/Fonts/arial.ttf",     # Windows
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                print(f"  [Debug] Using font: {fp}")
                break
        if font is None:
            font = ImageFont.load_default()
            print(f"  [Debug] Using default font")
    except Exception as e:
        print(f"  [Debug] Font loading failed: {e}, using default")
        try:
            font = ImageFont.load_default()
        except:
            font = None

    for page_idx, img in enumerate(images):
        # Convert to RGBA for transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Create overlay layer
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw L1 boxes (speech bubbles)
        if debug_l1:
            for b in bubbles_l1:
                if b['page_idx'] == page_idx:
                    box = b['box']
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    draw.rectangle([x1, y1, x2, y2], outline=L1_COLOR, width=3)
                    # Draw label with background
                    label = f"L1:{b['bubble_idx']}"
                    label_x, label_y = x1 + 4, y1 + 4
                    if font:
                        try:
                            lbox = draw.textbbox((label_x, label_y), label, font=font)
                            draw.rectangle([lbox[0]-1, lbox[1]-1, lbox[2]+1, lbox[3]+1], fill=(0,0,0,200))
                        except:
                            pass
                    draw.text((label_x, label_y), label, fill=L1_COLOR, font=font)

        # Draw L2 boxes (text-free regions)
        if debug_l2:
            for b in bubbles_l2:
                if b['page_idx'] == page_idx:
                    box = b['box']
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    draw.rectangle([x1, y1, x2, y2], outline=L2_COLOR, width=3)
                    # Draw label with background
                    label = f"L2:{b['bubble_idx']}"
                    label_x, label_y = x1 + 4, y1 + 4
                    if font:
                        try:
                            lbox = draw.textbbox((label_x, label_y), label, font=font)
                            draw.rectangle([lbox[0]-1, lbox[1]-1, lbox[2]+1, lbox[3]+1], fill=(0,0,0,200))
                        except:
                            pass
                    draw.text((label_x, label_y), label, fill=L2_COLOR, font=font)

        # Draw OCR text overlay
        if debug_ocr and ocr_data and page_idx in ocr_data:
            for bubble_data in ocr_data[page_idx]:
                box = bubble_data.get('bubble_box', [])
                texts = bubble_data.get('texts', [])
                if box and texts:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    # Combine OCR texts
                    ocr_text = ''.join([t.get('text', '') for t in texts])[:60]  # Limit length
                    if ocr_text:
                        # Calculate text position - inside the box at top if box is near top of image
                        text_y = y1 - font_size - 4 if y1 > font_size + 10 else y1 + 2
                        text_x = x1 + 2

                        # Get text bounding box
                        if font:
                            try:
                                text_bbox = draw.textbbox((text_x, text_y), ocr_text, font=font)
                            except:
                                text_bbox = (text_x, text_y, text_x + len(ocr_text) * 8, text_y + font_size)
                        else:
                            text_bbox = (text_x, text_y, text_x + len(ocr_text) * 8, text_y + font_size)

                        # Draw background rectangle
                        draw.rectangle(
                            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                            fill=OCR_BG_COLOR
                        )
                        # Draw text
                        draw.text((text_x, text_y), ocr_text, fill=OCR_TEXT_COLOR, font=font)

        # Composite overlay onto image
        result = Image.alpha_composite(img, overlay)
        debug_images.append(result.convert('RGB'))

    # Save debug images if requested
    if debug_save and output_dir and tab_title:
        debug_dir = os.path.join(output_dir, 'debug', tab_title)
        os.makedirs(debug_dir, exist_ok=True)
        for i, img in enumerate(debug_images):
            debug_path = os.path.join(debug_dir, f'debug_{i:03d}.png')
            img.save(debug_path)
        print(f"  [Debug] Saved {len(debug_images)} debug images to {debug_dir}")

    return debug_images




def inpaint_image(img_array: np.ndarray, bubbles: List[Dict], inpainter: Inpainter,
                  text_segmenter: TextSegmenter = None, char_bboxes_map: Dict = None,
                  verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """Inpaint L2 regions using AOT-GAN.

    Mask source priority: char_bboxes > text_segmenter > whole bbox

    Args:
        img_array: Input image as numpy array
        bubbles: List of bubble dicts with 'bubble_box' and 'bubble_idx'
        inpainter: Inpainter instance for AOT-GAN inpainting
        text_segmenter: If provided, use text_seg mask to only inpaint text pixels
        char_bboxes_map: Dict mapping bubble_idx to list of char bboxes from OneOCR
        verbose: If True, log each operation

    Returns:
        Tuple of (inpainted image array, stats dict)
    """
    inpainter.reset_stats()

    mode = "char_bbox" if char_bboxes_map else ("text_seg" if text_segmenter else "bbox")
    if verbose:
        print(f"  [Inpaint L2] Processing {len(bubbles)} regions ({mode})...")

    for i, bubble in enumerate(bubbles):
        bbox = bubble["bubble_box"]
        bubble_idx = bubble.get("bubble_idx", i)
        x1, y1, x2, y2 = bbox

        # Priority 1: Use char bboxes from OneOCR
        if char_bboxes_map and bubble_idx in char_bboxes_map:
            char_bboxes = char_bboxes_map[bubble_idx]
            if char_bboxes:
                # Create mask from char bboxes
                region_h, region_w = y2 - y1, x2 - x1
                char_mask = np.zeros((region_h, region_w), dtype=np.uint8)
                for cb in char_bboxes:
                    cx1, cy1, cx2, cy2 = cb
                    # Convert to region coordinates
                    cx1_rel = max(0, cx1 - x1)
                    cy1_rel = max(0, cy1 - y1)
                    cx2_rel = min(region_w, cx2 - x1)
                    cy2_rel = min(region_h, cy2 - y1)
                    if cx2_rel > cx1_rel and cy2_rel > cy1_rel:
                        char_mask[cy1_rel:cy2_rel, cx1_rel:cx2_rel] = 255

                if np.any(char_mask > 0):
                    coords, result = inpainter.inpaint_with_mask(img_array, bbox, char_mask, verbose=verbose)
                    img_array[coords[0]:coords[1], coords[2]:coords[3]] = result
                    continue

        # Priority 2: Use text_segmenter
        if text_segmenter:
            region = img_array[y1:y2, x1:x2]
            if region.size == 0:
                continue

            text_mask = text_segmenter(region, verbose=False)
            if text_mask is not None and np.any(text_mask > 127):
                coords, result = inpainter.inpaint_with_mask(img_array, bbox, text_mask, verbose=verbose)
                img_array[coords[0]:coords[1], coords[2]:coords[3]] = result
                continue

        # Priority 3: Fallback to whole bbox
        coords, result = inpainter(img_array, bbox, verbose=verbose)
        img_array[coords[0]:coords[1], coords[2]:coords[3]] = result

    stats = inpainter.get_stats()

    if verbose:
        print(f"  [Inpaint L2] Done: {stats['count']} regions, {stats['total_ms']}ms total")

    return img_array, stats


def process_images(images: List[Image.Image], api_key: str = None, output_type: str = "full_page",
                   ocr_translate: bool = False, translate_local: bool = False,
                   inpaint_background: bool = True, tab_title: str = None) -> Tuple[List[Dict], Any, Dict]:
    """Main processing pipeline: detect -> OCR -> translate -> AOT inpaint -> render

    Args:
        images: Input images
        api_key: Optional API key for translation
        output_type: One of "full_page", "speech_image_only", "text_only"
        ocr_translate: If True, use VLM to OCR and translate in one step
        translate_local: If True, use local LLM (HY-MT) for translation instead of Cerebras API
        inpaint_background: If True, detect text-free regions and inpaint them in parallel (AOT inpainting)

    Returns:
        Tuple of (ocr_data, output, stats) where output varies by output_type
    """
    stats = {"pages": len(images), "output_type": output_type, "ocr_translate": ocr_translate, "translate_local": translate_local, "inpaint_background": inpaint_background}
    session, mode = get_detector()

    # Detect bubbles (label1) and text-free regions (label2) if inpainting enabled
    t0 = time.time()
    bubbles, detect_time = detect_all(session, images, mode, target_label=1)

    bubbles_l2 = []
    inpaint_future = None
    text_seg_future = None
    inpainted_images = None

    # Track L2 data for later rendering (kept separate from L1)
    l2_data = {}
    l2_ocr_data = {}  # Separate OCR data for L2
    l2_skipped_regions = []  # Non-CJK L2 regions to restore from original
    # Cache L2 OCR results (to avoid running twice when using OneOCR)
    l2_ocr_cached = None
    l2_positions_cached = None
    l2_ocr_time_cached = 0

    # Parallel executor for background tasks
    _parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    # Text segmentation instance (will run on grid after OCR creates it)
    text_seg = get_text_segmenter_instance() if (TEXT_SEG_ENABLED and output_type != "text_only") else None
    text_seg_future = None  # Will be started after we have the grid

    if inpaint_background and output_type != "text_only":
        # Detect label2 regions for inpainting + OCR/translate
        bubbles_l2, detect_time_l2 = detect_all(session, images, mode, target_label=2)
        stats["label2_detected"] = len(bubbles_l2)
        detect_time += detect_time_l2
        print(f"  [Detect] {len(bubbles)} bubbles (L1) + {len(bubbles_l2)} text-free regions (L2) in {detect_time:.0f}ms")

        # Build l2_data for inpainting (DO NOT mix with L1 bubbles)
        for b in bubbles_l2:
            page_idx = b['page_idx']
            if page_idx not in l2_data:
                l2_data[page_idx] = []
            l2_data[page_idx].append({
                'bubble_box': list(b['box']),
                'bubble_idx': b['bubble_idx']
            })

        # Determine OCR method for L2 inpainting strategy
        uses_oneocr = OCR_METHOD in ("oneocr", "oneocr_remote")

        # For OneOCR: Run L2 OCR FIRST to get char bboxes, then inpaint with char bboxes
        # For VLM: Use text_seg for inpainting (runs in parallel with OCR)
        l2_char_bboxes = {}  # bubble_idx -> list of char bboxes

        if bubbles_l2 and uses_oneocr:
            # Run L2 OCR first to get char bboxes for inpainting mask
            t_l2_ocr_start = time.time()
            print(f"  [OCR L2] Running OneOCR on {len(bubbles_l2)} regions for char bboxes...")
            l2_ocr_cached, l2_positions_cached, _ = run_ocr_on_bubbles(bubbles_l2, translate=False)
            l2_ocr_time_cached = int((time.time() - t_l2_ocr_start) * 1000)
            print(f"  [OCR L2] Got {l2_ocr_cached.get('line_count', 0)} lines in {l2_ocr_time_cached}ms")
            stats["ocr_l2_ms"] = l2_ocr_time_cached
            stats["ocr_l2_lines"] = l2_ocr_cached.get('line_count', 0)

            # Extract char bboxes from OCR result
            l2_bubble_texts_for_mask = map_ocr(l2_ocr_cached, l2_positions_cached)
            for (page_idx, bubble_idx), texts in l2_bubble_texts_for_mask.items():
                all_chars = []
                for text_item in texts:
                    chars = text_item.get('chars', [])
                    for c in chars:
                        cb = c.get('bbox')
                        if cb and len(cb) == 4:
                            all_chars.append(cb)
                if all_chars:
                    l2_char_bboxes[bubble_idx] = all_chars

            print(f"  [OCR L2] Extracted char bboxes for {len(l2_char_bboxes)}/{len(bubbles_l2)} regions")

        # Start inpainting (or white fill if AOT disabled)
        aot_enabled = get_aot_inpaint_enabled_runtime()
        if bubbles_l2 and aot_enabled:
            inpainter = get_inpainter()
            # For OneOCR: use char bboxes; For VLM: use text_seg
            l2_text_seg = None if uses_oneocr else (get_text_segmenter_instance() if TEXT_SEG_ENABLED else None)

            def do_inpainting():
                t_start = time.time()
                inpainted = []
                total_regions = 0
                for page_idx, img in enumerate(images):
                    img_array = np.array(img)
                    page_l2 = l2_data.get(page_idx, [])
                    if page_l2:
                        img_array, page_stats = inpaint_image(
                            img_array, page_l2, inpainter,
                            text_segmenter=l2_text_seg,
                            char_bboxes_map=l2_char_bboxes if uses_oneocr else None,
                            verbose=False
                        )
                        total_regions += page_stats.get('count', 0)
                    inpainted.append(Image.fromarray(img_array))
                elapsed_ms = int((time.time() - t_start) * 1000)
                mode_str = " (char_bbox)" if uses_oneocr else (" (text_seg)" if l2_text_seg else "")
                print(f"  [Inpaint L2] {total_regions} regions in {elapsed_ms}ms{mode_str}")
                return inpainted, elapsed_ms

            inpaint_future = _parallel_executor.submit(do_inpainting)
        elif bubbles_l2 and not aot_enabled:
            # AOT disabled - fill L2 regions with white instead
            def do_white_fill():
                t_start = time.time()
                filled = []
                total_regions = 0
                for page_idx, img in enumerate(images):
                    img_array = np.array(img)
                    page_l2 = l2_data.get(page_idx, [])
                    for bubble in page_l2:
                        bbox = bubble["bubble_box"]
                        x1, y1, x2, y2 = bbox
                        img_array[y1:y2, x1:x2] = 255  # Fill with white
                        total_regions += 1
                    filled.append(Image.fromarray(img_array))
                elapsed_ms = int((time.time() - t_start) * 1000)
                print(f"  [L2 White Fill] {total_regions} regions in {elapsed_ms}ms (AOT disabled)")
                return filled, elapsed_ms

            inpaint_future = _parallel_executor.submit(do_white_fill)
    else:
        print(f"  [Detect] {len(bubbles)} bubbles in {detect_time:.0f}ms")
        uses_oneocr = OCR_METHOD in ("oneocr", "oneocr_remote")

    # Start text segmentation in PARALLEL with OCR (right after detection)
    # Skip text_seg when using OneOCR (unless force_text_seg_mask is set)
    force_text_seg = get_force_text_seg_mask_runtime()
    pages_with_bubbles = set(b['page_idx'] for b in bubbles) | set(b['page_idx'] for b in bubbles_l2)
    skip_text_seg = uses_oneocr and not force_text_seg
    if text_seg is not None and pages_with_bubbles and not skip_text_seg:
        def do_text_segmentation_on_pages():
            t_start = time.time()
            page_masks = {}
            for page_idx in pages_with_bubbles:
                if page_idx < len(images):
                    page_array = np.array(images[page_idx])
                    page_masks[page_idx] = text_seg(page_array, verbose=False)
            elapsed_ms = int((time.time() - t_start) * 1000)
            mode_note = " (force_text_seg_mask)" if force_text_seg else ""
            print(f"  [TextSeg] {len(page_masks)}/{len(images)} pages (with bubbles) in {elapsed_ms}ms{mode_note}")
            return page_masks, elapsed_ms
        text_seg_future = _parallel_executor.submit(do_text_segmentation_on_pages)
    elif skip_text_seg:
        print(f"  [TextSeg] Skipped - using OneOCR char bboxes")

    # Debug: Log detected bubbles count (details only if few bubbles)
    if len(bubbles) <= 10:
        print(f"  [Detect] L1 bubbles: {[(b['page_idx'], b['bubble_idx']) for b in bubbles]}")

    stats["detect_ms"] = int(detect_time)
    stats["bubbles_detected"] = len(bubbles)

    # Sequential model loading: start OCR server on demand
    uses_vlm_ocr = OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8")
    if SEQUENTIAL_MODEL_LOADING and uses_vlm_ocr and HAS_VLM_OCR:
        seq_mgr = get_sequential_manager()
        seq_mgr.start_ocr_server()
        stats["sequential_mode"] = True

    # === Run L1 OCR and L2 OCR in PARALLEL ===
    _ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def do_ocr_l1():
        """OCR for L1 (text bubbles)."""
        from workflow.ocr import _get_configured_ocr_method
        t0 = time.time()
        ocr_method, is_api, is_remote = _get_configured_ocr_method()

        if ocr_translate and HAS_VLM_OCR:
            print(f"  [OCR L1] Using VLM for combined OCR and translation...")
            result, pos, grid = run_ocr_on_bubbles(bubbles, translate=True)
            mode = "ocr_translate"
        elif HAS_VLM_OCR:
            # Show actual OCR method being used
            method_name = ocr_method.replace("_", " ").title() if ocr_method else "VLM"
            print(f"  [OCR L1] Using {method_name} for OCR ({len(bubbles)} bubbles)...")
            result, pos, grid = run_ocr_on_bubbles(bubbles, translate=False)
            mode = ocr_method or "vlm"
        else:
            grid, pos, _ = grid_bubbles(bubbles)
            result = run_ocr(grid)
            mode = "http"
        elapsed = int((time.time() - t0) * 1000)
        print(f"  [OCR L1] {result.get('line_count', 0)} lines in {elapsed}ms (method: {mode})")
        return result, pos, grid, elapsed, mode

    def do_ocr_l2():
        """OCR for L2 (text-free regions)."""
        nonlocal l2_ocr_cached, l2_positions_cached, l2_ocr_time_cached
        if not bubbles_l2:
            return None, None, 0
        # If we already ran L2 OCR for mask extraction (OneOCR), reuse that result
        if l2_ocr_cached is not None:
            print(f"  [OCR L2] Reusing cached result: {l2_ocr_cached.get('line_count', 0)} lines (from mask extraction)")
            return l2_ocr_cached, l2_positions_cached, l2_ocr_time_cached
        t0 = time.time()
        if HAS_VLM_OCR:
            print(f"  [OCR L2] Using VLM for {len(bubbles_l2)} text-free regions...")
            result, pos, _ = run_ocr_on_bubbles(bubbles_l2, translate=False)
        else:
            l2_grid, pos, _ = grid_bubbles(bubbles_l2)
            result = run_ocr(l2_grid)
        elapsed = int((time.time() - t0) * 1000)
        print(f"  [OCR L2] {result.get('line_count', 0)} lines in {elapsed}ms")
        return result, pos, elapsed

    # Start both OCR tasks in parallel
    ocr_l1_future = _ocr_executor.submit(do_ocr_l1)
    ocr_l2_future = _ocr_executor.submit(do_ocr_l2) if (bubbles_l2 and inpaint_background) else None

    # Wait for L1 OCR
    ocr_result, positions, grid, ocr_time, ocr_mode = ocr_l1_future.result()
    stats["ocr_translate_mode"] = (ocr_mode == "ocr_translate")
    stats["grid_size"] = f"{grid.width}x{grid.height}" if grid else "batched"
    stats["ocr_ms"] = ocr_time
    stats["ocr_lines"] = ocr_result.get('line_count', 0)
    # Track failed/incomplete OCR batches
    if ocr_result.get('failed_batches'):
        stats["ocr_failed_batches"] = ocr_result['failed_batches']
    is_translated = ocr_result.get('translated', False)

    # Map OCR to bubbles
    bubble_texts = map_ocr(ocr_result, positions, verbose=False)

    # OCR mapping summary (compact)
    all_keys = set(pos['key'] for pos in positions)
    mapped_keys = set(bubble_texts.keys())
    unmapped_keys = all_keys - mapped_keys
    unmapped_count = len(unmapped_keys)

    print(f"  [OCR Map] {len(mapped_keys)}/{len(all_keys)} bubbles mapped ({100*len(mapped_keys)/max(1,len(all_keys)):.0f}%){f', {unmapped_count} unmapped' if unmapped_count else ''}")

    # Build OCR data structure per page (skip non-CJK text - English/numbers only)
    ocr_data = {}
    bubble_map = {(b['page_idx'], b['bubble_idx']): b for b in bubbles}
    skipped_non_cjk = 0

    for key, text_items in bubble_texts.items():
        page_idx, bubble_idx = key
        b = bubble_map.get(key)
        if b:
            # Get combined text to check for CJK
            combined_text = "".join(t.get('text', '') for t in text_items)
            if not should_translate(combined_text):
                # Skip non-CJK text entirely (English/numbers - leave as-is)
                skipped_non_cjk += 1
                continue

            if page_idx not in ocr_data:
                ocr_data[page_idx] = []
            ocr_data[page_idx].append({
                'idx': bubble_idx,
                'bubble_box': list(b['box']),
                'texts': text_items
            })

    # Sort bubbles by idx
    for page_idx in ocr_data:
        ocr_data[page_idx].sort(key=lambda x: x['idx'])

    total_l1_bubbles = sum(len(v) for v in ocr_data.values())
    if skipped_non_cjk > 0:
        print(f"  [OCR] Skipped {skipped_non_cjk} non-CJK bubbles (English/numbers)")

    # Wait for L2 OCR (if running)
    l2_ocr_result = None
    l2_positions = None
    if ocr_l2_future is not None:
        l2_ocr_result, l2_positions, l2_ocr_time = ocr_l2_future.result()
        if l2_ocr_result:
            stats["ocr_l2_ms"] = l2_ocr_time
            stats["ocr_l2_lines"] = l2_ocr_result.get('line_count', 0)
            # Track failed/incomplete L2 OCR batches
            if l2_ocr_result.get('failed_batches'):
                stats["ocr_l2_failed_batches"] = l2_ocr_result['failed_batches']

            # Map L2 OCR to bubbles
            l2_bubble_texts = map_ocr(l2_ocr_result, l2_positions)
            l2_bubble_map = {(b['page_idx'], b['bubble_idx']): b for b in bubbles_l2}

            # Track ALL L2 regions - will restore those without CJK OCR results
            # Start with all L2 regions as "to restore", then remove ones with CJK text
            all_l2_regions = {(b['page_idx'], b['bubble_idx']): {'page_idx': b['page_idx'], 'box': list(b['box'])}
                             for b in bubbles_l2}
            l2_with_cjk = set()  # Keys of L2 regions with CJK text (will be processed)

            for key, text_items in l2_bubble_texts.items():
                page_idx, bubble_idx = key
                b = l2_bubble_map.get(key)
                if b:
                    combined_text = "".join(t.get('text', '') for t in text_items)
                    if should_translate(combined_text):
                        # Has CJK text - will be translated, add to ocr_data
                        l2_with_cjk.add(key)
                        if page_idx not in l2_ocr_data:
                            l2_ocr_data[page_idx] = []
                        l2_ocr_data[page_idx].append({
                            'idx': bubble_idx,
                            'bubble_box': list(b['box']),
                            'texts': text_items
                        })

            # L2 regions to restore = all L2 regions MINUS those with CJK text
            l2_skipped_regions = [info for key, info in all_l2_regions.items() if key not in l2_with_cjk]

            # L2 OCR mapping summary
            l2_total = sum(len(v) for v in l2_ocr_data.values())
            restore_count = len(l2_skipped_regions)
            skip_msg = f", restoring {restore_count} non-CJK/empty" if restore_count else ""
            print(f"  [OCR Map L2] {l2_total} regions with CJK text{skip_msg}")

    # Sequential model loading: stop OCR server after ALL OCR is done (L1 + L2)
    if SEQUENTIAL_MODEL_LOADING and uses_vlm_ocr and HAS_VLM_OCR:
        seq_mgr = get_sequential_manager()
        seq_mgr.stop_ocr_server()

    # Extract texts for translation (non-CJK already filtered out in ocr_data building)
    all_texts = []
    text_mapping = []  # (page_idx, bubble_idx, is_l2)
    is_already_translated = ocr_result.get('translated', False)

    # L1 texts
    for page_idx, page_bubbles in sorted(ocr_data.items()):
        for bubble in page_bubbles:
            merged = merge_ocr_texts_for_translation(bubble['texts'], bubble.get('bubble_box'))
            if merged.strip():
                all_texts.append(merged.strip())
                text_mapping.append((page_idx, bubble['idx'], False))  # is_l2=False

    l1_text_count = len(all_texts)

    # L2 texts
    for page_idx, page_bubbles in sorted(l2_ocr_data.items()):
        for bubble in page_bubbles:
            merged = merge_ocr_texts_for_translation(bubble['texts'], bubble.get('bubble_box'))
            if merged.strip():
                all_texts.append(merged.strip())
                text_mapping.append((page_idx, bubble['idx'], True))  # is_l2=True

    stats["texts_to_translate"] = len(all_texts)
    stats["texts_l1"] = l1_text_count
    stats["texts_l2"] = len(all_texts) - l1_text_count

    # Sequential model loading: start translate server if using local LLM
    uses_local_translate = translate_local and HAS_LOCAL_TRANSLATE and TRANSLATE_METHOD in ("qwen_vlm", "hunyuan_mt")
    if SEQUENTIAL_MODEL_LOADING and uses_local_translate and not is_already_translated:
        seq_mgr = get_sequential_manager()
        seq_mgr.start_translate_server()

    # Translate (skip if VLM already provided translations)
    t0 = time.time()
    if is_already_translated:
        print(f"  [Translate] Skipped - VLM provided {len(all_texts)} translations directly")
        translations = all_texts  # VLM output is already English
        stats["translate_skipped"] = True
        stats["translation_ms"] = 0  # Included in OCR time
        stats["translation_batches"] = 0  # VLM does it in one pass
        stats["translations_success"] = len(translations)
        stats["translations_failed"] = 0
        stats["translate_method"] = "vlm"
    elif translate_local and HAS_LOCAL_TRANSLATE:
        # Use local LLM for translation
        translator = LlmTranslator()
        batch_count = (len(all_texts) + translator.batch_size - 1) // translator.batch_size
        print(f"  [Translate] {len(all_texts)} texts in {batch_count} batches (batch_size={translator.batch_size})...")
        result = translator.translate(all_texts)
        translations = result.get('translations', [''] * len(all_texts))
        trans_time = int((time.time() - t0) * 1000)
        stats["translation_ms"] = trans_time
        stats["translation_batches"] = result.get('batches', batch_count)
        stats["translation_batch_size"] = translator.batch_size
        success_count = sum(1 for t in translations if t and t.strip())
        stats["translations_success"] = success_count
        stats["translations_failed"] = len(translations) - success_count
        stats["translate_method"] = "local_llm"
        if result.get('errors'):
            print(f"  [Translate] Errors: {len(result['errors'])} batch(es) failed")
            stats["translate_errors"] = result['errors']
        print(f"  [Translate] {success_count} translations in {trans_time}ms (local LLM)")
    elif translate_local and not HAS_LOCAL_TRANSLATE:
        print(f"  [Warning] translate_local requested but local LLM not available")
        translations = ['[LOCAL_TRANSLATE_UNAVAILABLE]'] * len(all_texts)
        stats["translation_ms"] = 0
        stats["translations_success"] = 0
        stats["translations_failed"] = len(all_texts)
        stats["translate_method"] = "none"
    elif TRANSLATE_METHOD == "gemini_translate" and HAS_GEMINI_TRANSLATE:
        # Use Gemini API for translation
        translations = translate_texts_gemini(all_texts, stats=stats)
        # Count successful translations
        success_count = sum(1 for t in translations if t and not t.startswith("["))
        no_text_count = sum(1 for t in translations if t == "[NO TEXT]")
        failed_count = sum(1 for t in translations if t == "[TRANSLATION FAILED]")
        empty_count = sum(1 for t in translations if not t)
        stats["translations_success"] = success_count
        stats["translations_failed"] = len(translations) - success_count
        stats["translations_no_text"] = no_text_count
        stats["translate_method"] = "gemini_api"
        print(f"  [Translate] Results: {success_count} success, {no_text_count} [NO TEXT], {failed_count} failed, {empty_count} empty")
    else:
        # Use Cerebras API for translation
        print(f"  [Translate] {len(all_texts)} texts using Cerebras API...")
        translations = translate_texts(all_texts, api_key=api_key, stats=stats)
        # Count successful translations
        success_count = sum(1 for t in translations if t and not t.startswith("["))
        no_text_count = sum(1 for t in translations if t == "[NO TEXT]")
        failed_count = sum(1 for t in translations if t == "[TRANSLATION FAILED]")
        empty_count = sum(1 for t in translations if not t)
        stats["translations_success"] = success_count
        stats["translations_failed"] = len(translations) - success_count
        stats["translations_no_text"] = no_text_count
        stats["translate_method"] = "cerebras_api"
        print(f"  [Translate] Results: {success_count} success, {no_text_count} [NO TEXT], {failed_count} failed, {empty_count} empty")

    # Sequential model loading: stop translate server after translation is done
    if SEQUENTIAL_MODEL_LOADING and uses_local_translate and not is_already_translated:
        seq_mgr = get_sequential_manager()
        seq_mgr.stop_translate_server()

    # Build translation lookup per page (separate L1 and L2)
    translation_data = {}  # L1 translations
    l2_translation_data = {}  # L2 translations
    for i, (page_idx, bubble_idx, is_l2) in enumerate(text_mapping):
        trans_text = translations[i] if i < len(translations) else "[MISSING]"
        if is_l2:
            if page_idx not in l2_translation_data:
                l2_translation_data[page_idx] = {}
            l2_translation_data[page_idx][bubble_idx] = trans_text
        else:
            if page_idx not in translation_data:
                translation_data[page_idx] = {}
            translation_data[page_idx][bubble_idx] = trans_text

    # Translation mapping summary (compact)
    l1_count = len(translation_data)
    l2_count = len(l2_translation_data)
    print(f"  [Translate Map] L1: {sum(len(v) for v in translation_data.values())} bubbles, L2: {sum(len(v) for v in l2_translation_data.values())} regions")

    # Handle different output types
    if output_type == "text_only":
        # Return just the data - no image rendering
        text_output = {}
        for page_idx, page_bubbles in sorted(ocr_data.items()):
            text_output[page_idx] = []
            page_translations = translation_data.get(page_idx, {})
            for bubble in page_bubbles:
                bubble_data = {
                    'bubble_idx': bubble['idx'],
                    'bubble_box': bubble['bubble_box'],
                    'original_texts': bubble['texts'],
                    'original_text': "".join([t['text'] for t in bubble['texts']]),
                    'translated_text': page_translations.get(bubble['idx'], "[MISSING]")
                }
                text_output[page_idx].append(bubble_data)
        print(f"  [Text Only] Returning text data for {len(text_output)} pages")
        return ocr_data, text_output, stats

    elif output_type == "speech_image_only":
        # Wait for label2 inpainting to complete if running
        use_inpaint = False
        if inpaint_future is not None:
            try:
                inpainted_images, inpaint_ms = inpaint_future.result(timeout=120)
                stats["inpaint_l2_ms"] = inpaint_ms
                use_inpaint = True
                print(f"  [Inpaint L2] Completed {len(bubbles_l2)} regions in {inpaint_ms}ms")
            except Exception as e:
                print(f"  [Inpaint L2] Failed: {e}")
                inpainted_images = None

        # Wait for text segmentation to complete if running
        text_seg_masks = None
        if text_seg_future is not None:
            try:
                text_seg_masks, text_seg_ms = text_seg_future.result(timeout=120)
                stats["text_seg_ms"] = text_seg_ms
            except Exception as e:
                print(f"  [TextSeg] Failed: {e}")
                text_seg_masks = None

        # Use L2-inpainted images (or original if no L2 inpainting)
        source_images = inpainted_images if use_inpaint else images

        # Extract and return cropped bubble images with positions
        # Note: Text clearing now samples background color for colored backgrounds (no L1 AOT needed)
        t0 = time.time()
        bubble_images_output = {}
        for page_idx, page_bubbles in sorted(ocr_data.items()):
            bubble_images_output[page_idx] = []
            page_translations = translation_data.get(page_idx, {})
            img = source_images[page_idx]
            page_mask = text_seg_masks.get(page_idx) if text_seg_masks else None

            for bubble in page_bubbles:
                # Create a copy of just the bubble area
                bbox = bubble['bubble_box']
                x1, y1, x2, y2 = bbox
                bubble_img = img.crop((x1, y1, x2, y2))

                # Extract mask region for this bubble if available
                bubble_mask = page_mask[y1:y2, x1:x2] if page_mask is not None else None

                # Render text on the bubble image
                bubble_copy = bubble_img.copy()
                # Create a temporary structure for rendering
                temp_bubble = {
                    'idx': bubble['idx'],
                    'bubble_box': [0, 0, x2-x1, y2-y1],  # Relative to cropped image
                    'texts': bubble['texts']
                }
                temp_translations = {bubble['idx']: page_translations.get(bubble['idx'], "[MISSING]")}

                # Use precomputed mask region for text cleaning (background color sampled automatically)
                rendered_bubble = render_text_on_image(bubble_copy, [temp_bubble], temp_translations, use_inpaint=use_inpaint, precomputed_mask=bubble_mask)

                # Convert to base64
                buf = io.BytesIO()
                rendered_bubble.save(buf, format='JPEG', quality=95)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                bubble_data = {
                    'bubble_idx': bubble['idx'],
                    'bubble_box': bbox,  # Position on original page
                    'image_base64': b64,
                    'original_text': "".join([t['text'] for t in bubble['texts']]),
                    'translated_text': page_translations.get(bubble['idx'], "[MISSING]"),
                    'inpainted_background': use_inpaint
                }
                bubble_images_output[page_idx].append(bubble_data)

        render_time = int((time.time() - t0) * 1000)
        stats["render_ms"] = render_time
        stats["bubble_count"] = sum(len(bubbles) for bubbles in bubble_images_output.values())
        print(f"  [Speech Image Only] Extracted {stats['bubble_count']} bubble images in {render_time}ms" + (" (inpainted)" if use_inpaint else ""))
        return ocr_data, bubble_images_output, stats

    else:  # full_page (default)
        # Wait for label2 inpainting to complete if running
        use_inpaint = False
        if inpaint_future is not None:
            try:
                inpainted_images, inpaint_ms = inpaint_future.result(timeout=120)
                stats["inpaint_l2_ms"] = inpaint_ms
                use_inpaint = True
                print(f"  [Inpaint L2] Completed {len(bubbles_l2)} regions in {inpaint_ms}ms")
            except Exception as e:
                print(f"  [Inpaint L2] Failed: {e}")
                inpainted_images = None

        # Wait for text segmentation to complete if running
        text_seg_masks = None
        if text_seg_future is not None:
            try:
                text_seg_masks, text_seg_ms = text_seg_future.result(timeout=120)
                stats["text_seg_ms"] = text_seg_ms
            except Exception as e:
                print(f"  [TextSeg] Failed: {e}")
                text_seg_masks = None

        # Use L2-inpainted images (or original if no L2 inpainting)
        source_images = inpainted_images if use_inpaint else images

        # Restore original regions for non-CJK L2 bubbles (they got inpainted but shouldn't have)
        if use_inpaint and l2_skipped_regions:
            for skip_info in l2_skipped_regions:
                page_idx = skip_info['page_idx']
                x1, y1, x2, y2 = skip_info['box']
                if page_idx < len(source_images) and page_idx < len(images):
                    # Copy original region back to inpainted image
                    orig_region = images[page_idx].crop((x1, y1, x2, y2))
                    source_images[page_idx].paste(orig_region, (x1, y1))
            print(f"  [Restore] {len(l2_skipped_regions)} non-CJK L2 regions restored from original")

        # Render full pages
        # Note: Text clearing now samples background color for colored backgrounds (no L1 AOT needed)
        t0 = time.time()
        output_images = []
        render_timing = {}  # Collect detailed timing
        for page_idx, img in enumerate(source_images):
            img_copy = img.copy()

            # Step 1: Render L1 bubbles (use pre-computed text_seg mask from grid)
            page_bubbles = ocr_data.get(page_idx, [])
            page_translations = translation_data.get(page_idx, {})
            page_mask = text_seg_masks.get(page_idx) if text_seg_masks else None

            rendered = render_text_on_image(img_copy, page_bubbles, page_translations, use_inpaint=use_inpaint, precomputed_mask=page_mask, timing=render_timing, prefer_mask=force_text_seg)

            # Step 2: Render L2 translations (use same text_seg mask for any residual text after AOT)
            page_l2_bubbles = l2_ocr_data.get(page_idx, [])
            page_l2_translations = l2_translation_data.get(page_idx, {})
            if page_l2_bubbles and page_l2_translations:
                # Use page_mask for L2 too - handles imperfect AOT inpainting
                # If AOT disabled, use white text on white background
                aot_enabled_for_render = get_aot_inpaint_enabled_runtime()
                rendered = render_text_on_image(rendered, page_l2_bubbles, page_l2_translations, use_inpaint=aot_enabled_for_render, precomputed_mask=page_mask, timing=render_timing, prefer_mask=force_text_seg, force_white_text=not aot_enabled_for_render)

            output_images.append(rendered)

        render_time = int((time.time() - t0) * 1000)
        stats["render_ms"] = render_time
        stats["render_detail"] = render_timing
        # Print detailed render timing
        detail_str = f"text_seg={render_timing.get('text_seg_ms', 0)}ms, mask={render_timing.get('mask_apply_ms', 0)}ms, text={render_timing.get('text_render_ms', 0)}ms"
        print(f"  [Full Page] Rendered {len(output_images)} pages in {render_time}ms ({detail_str})" + (" (inpainted)" if use_inpaint else ""))

        # Apply debug overlay if enabled
        if get_debug_l1_boxes() or get_debug_l2_boxes() or get_debug_ocr_text():
            from config import get_output_dir
            output_images = draw_debug_overlay(
                output_images, bubbles, bubbles_l2, ocr_data,
                output_dir=get_output_dir(), tab_title=tab_title
            )

        return ocr_data, output_images, stats


def process_images_legacy(images: List[Image.Image], api_key: str = None, output_type: str = "full_page") -> Tuple[List[Dict], Any, Dict]:
    """Legacy processing pipeline: detect -> {inpaint, translate} sequential -> render

    Args:
        images: Input images
        api_key: Optional API key for translation
        output_type: One of "full_page", "speech_image_only", "text_only"

    Returns:
        Tuple of (ocr_data, output, stats) where output varies by output_type
    """
    stats = {"pages": len(images), "output_type": output_type}
    session, mode = get_detector()
    inpainter = get_inpainter() if output_type != "text_only" else None
    text_seg = get_text_segmenter_instance() if (TEXT_SEG_ENABLED and output_type != "text_only") else None

    # Detect for both labels
    bubbles_l1, _ = detect_all(session, images, mode, target_label=1)
    bubbles_l2, detect_time = detect_all(session, images, mode, target_label=2)
    print(f"  Detected {len(bubbles_l1)} text bubbles, {len(bubbles_l2)} text-free regions in {detect_time:.0f}ms")

    # Sequential model loading: start OCR server on demand
    uses_vlm_ocr = OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8")
    if SEQUENTIAL_MODEL_LOADING and uses_vlm_ocr and HAS_VLM_OCR:
        seq_mgr = get_sequential_manager()
        seq_mgr.start_ocr_server()
        stats["sequential_mode"] = True

    # Grid and OCR for label 1 (text bubbles)
    grid, positions, _ = grid_bubbles(bubbles_l1)
    ocr_result = run_ocr(grid)
    bubble_texts = map_ocr(ocr_result, positions)

    # Sequential model loading: stop OCR server after OCR is done
    if SEQUENTIAL_MODEL_LOADING and uses_vlm_ocr and HAS_VLM_OCR:
        seq_mgr = get_sequential_manager()
        seq_mgr.stop_ocr_server()

    # Build OCR data (skip non-CJK text)
    ocr_data = {}
    bubble_map = {(b['page_idx'], b['bubble_idx']): b for b in bubbles_l1}

    for key, text_items in bubble_texts.items():
        page_idx, bubble_idx = key
        b = bubble_map.get(key)
        if b:
            # Skip non-CJK text entirely
            combined_text = "".join(t.get('text', '') for t in text_items)
            if not should_translate(combined_text):
                continue

            if page_idx not in ocr_data:
                ocr_data[page_idx] = []
            ocr_data[page_idx].append({
                'idx': bubble_idx,
                'bubble_box': list(b['box']),
                'texts': text_items
            })

    for page_idx in ocr_data:
        ocr_data[page_idx].sort(key=lambda x: x['idx'])

    # Build label 2 data
    l2_data = {}
    for b in bubbles_l2:
        page_idx = b['page_idx']
        if page_idx not in l2_data:
            l2_data[page_idx] = []
        l2_data[page_idx].append({
            'idx': b['bubble_idx'],
            'bubble_box': list(b['box'])
        })

    # Extract texts for translation
    all_texts = []
    text_mapping = []

    for page_idx, page_bubbles in sorted(ocr_data.items()):
        for bubble in page_bubbles:
            merged = merge_ocr_texts_for_translation(bubble['texts'], bubble.get('bubble_box'))
            if merged.strip():
                all_texts.append(merged.strip())
                text_mapping.append((page_idx, bubble['idx']))

    # Handle different output types
    if output_type == "text_only":
        # Skip inpainting for text_only mode
        if TRANSLATE_METHOD == "gemini_translate" and HAS_GEMINI_TRANSLATE:
            translations = translate_texts_gemini(all_texts, stats=stats)
        else:
            translations = translate_texts(all_texts, api_key=api_key, stats=stats)

        # Build translation lookup
        translation_data = {}
        for i, (page_idx, bubble_idx) in enumerate(text_mapping):
            if page_idx not in translation_data:
                translation_data[page_idx] = {}
            translation_data[page_idx][bubble_idx] = translations[i] if i < len(translations) else "[MISSING]"

        # Return text data only
        text_output = {}
        for page_idx, page_bubbles in sorted(ocr_data.items()):
            text_output[page_idx] = []
            page_translations = translation_data.get(page_idx, {})
            for bubble in page_bubbles:
                bubble_data = {
                    'bubble_idx': bubble['idx'],
                    'bubble_box': bubble['bubble_box'],
                    'original_texts': bubble['texts'],
                    'original_text': "".join([t['text'] for t in bubble['texts']]),
                    'translated_text': page_translations.get(bubble['idx'], "[MISSING]")
                }
                text_output[page_idx].append(bubble_data)

        # Add label 2 regions info (text-free regions)
        for page_idx, page_l2 in l2_data.items():
            if page_idx not in text_output:
                text_output[page_idx] = []
            for region in page_l2:
                text_output[page_idx].append({
                    'bubble_idx': region['idx'],
                    'bubble_box': region['bubble_box'],
                    'is_text_free': True,  # Mark as text-free region
                    'inpainted': False  # Not inpainted in text_only mode
                })

        stats["text_bubbles"] = len(bubbles_l1)
        stats["text_free_regions"] = len(bubbles_l2)
        print(f"  [Text Only] Returning text data for {len(text_output)} pages")
        return ocr_data, text_output, stats

    # For modes that need images, run inpainting and translation
    def do_inpainting():
        inpainted = []
        total_regions = 0
        t_start = time.time()
        for page_idx, img in enumerate(images):
            img_array = np.array(img)
            page_l2 = l2_data.get(page_idx, [])
            if page_l2:
                img_array, page_stats = inpaint_image(img_array, page_l2, inpainter, text_seg, verbose=False)
                total_regions += page_stats.get('count', 0)
            inpainted.append(Image.fromarray(img_array))
        elapsed_ms = int((time.time() - t_start) * 1000)
        mode_str = " (text_seg)" if text_seg else ""
        print(f"  [Inpaint L2] {total_regions} regions in {elapsed_ms}ms{mode_str}")
        return inpainted

    def do_translation():
        if TRANSLATE_METHOD == "gemini_translate" and HAS_GEMINI_TRANSLATE:
            return translate_texts_gemini(all_texts, stats=stats)
        return translate_texts(all_texts, api_key=api_key, stats=stats)

    # Note: Label2 uses Cerebras API for translation by default (not local LLM)
    # Inpainting uses PyTorch (not llama server), so no VRAM conflict with sequential mode
    if SEQUENTIAL_MODEL_LOADING:
        print(f"  [Sequential] Running inpainting first, then translation (API)...")
        inpainted_images = do_inpainting()
        translations = do_translation()
    else:
        print(f"  Running inpainting and translation in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            inpaint_future = executor.submit(do_inpainting)
            translate_future = executor.submit(do_translation)
            inpainted_images = inpaint_future.result()
            translations = translate_future.result()

    # Build translation lookup
    translation_data = {}
    for i, (page_idx, bubble_idx) in enumerate(text_mapping):
        if page_idx not in translation_data:
            translation_data[page_idx] = {}
        translation_data[page_idx][bubble_idx] = translations[i] if i < len(translations) else "[MISSING]"

    stats["text_bubbles"] = len(bubbles_l1)
    stats["text_free_regions"] = len(bubbles_l2)

    if output_type == "speech_image_only":
        # Extract bubble images from inpainted pages
        t0 = time.time()
        bubble_images_output = {}
        for page_idx, page_bubbles in sorted(ocr_data.items()):
            bubble_images_output[page_idx] = []
            page_translations = translation_data.get(page_idx, {})
            img = inpainted_images[page_idx]  # Use inpainted image

            for bubble in page_bubbles:
                # Create a copy of just the bubble area
                bbox = bubble['bubble_box']
                x1, y1, x2, y2 = bbox
                bubble_img = img.crop((x1, y1, x2, y2))

                # Render text on the bubble image
                bubble_copy = bubble_img.copy()
                temp_bubble = {
                    'idx': bubble['idx'],
                    'bubble_box': [0, 0, x2-x1, y2-y1],  # Relative to cropped image
                    'texts': bubble['texts']
                }
                temp_translations = {bubble['idx']: page_translations.get(bubble['idx'], "[MISSING]")}
                # Use text_seg for pixel-level text cleaning (L1 text bubbles inside L2 inpainted image)
                rendered_bubble = render_text_on_image(bubble_copy, [temp_bubble], temp_translations, use_inpaint=True, text_segmenter=text_seg)

                # Convert to base64
                buf = io.BytesIO()
                rendered_bubble.save(buf, format='JPEG', quality=95)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                bubble_data = {
                    'bubble_idx': bubble['idx'],
                    'bubble_box': bbox,
                    'image_base64': b64,
                    'original_text': "".join([t['text'] for t in bubble['texts']]),
                    'translated_text': page_translations.get(bubble['idx'], "[MISSING]"),
                    'inpainted_background': True  # Extracted from inpainted image
                }
                bubble_images_output[page_idx].append(bubble_data)

        render_time = int((time.time() - t0) * 1000)
        stats["render_ms"] = render_time
        stats["bubble_count"] = sum(len(bubbles) for bubbles in bubble_images_output.values())
        print(f"  [Speech Image Only] Extracted {stats['bubble_count']} bubble images in {render_time}ms (inpainted)")
        return ocr_data, bubble_images_output, stats

    else:  # full_page (default)
        # Render on inpainted images
        t0 = time.time()
        output_images = []
        for page_idx, img in enumerate(inpainted_images):
            page_bubbles = ocr_data.get(page_idx, [])
            page_translations = translation_data.get(page_idx, {})
            # Use text_seg for pixel-level text cleaning (L1 text bubbles inside L2 inpainted image)
            rendered = render_text_on_image(img, page_bubbles, page_translations, use_inpaint=True, text_segmenter=text_seg)
            output_images.append(rendered)

        render_time = int((time.time() - t0) * 1000)
        stats["render_ms"] = render_time
        print(f"  [Full Page] Rendered {len(output_images)} pages with inpainting in {render_time}ms")
        return ocr_data, output_images, stats


def images_to_base64(images: List[Image.Image]) -> List[str]:
    """Convert PIL images to base64 strings."""
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        result.append(b64)
    return result


@app.route('/api/v1/process', methods=['POST'])
def process():
    """Main pipeline: detect -> OCR -> translate -> AOT inpaint -> render"""
    start_time = time.time()

    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400

    # Get optional parameters from form data
    api_key = request.form.get('api_key', None)
    output_type = request.form.get('output_type', 'full_page')

    # Save options - for saving images before/after processing
    # Use request value if provided, otherwise fall back to config defaults
    from config import get_save_before, get_save_after
    save_before_param = request.form.get('save_before', '')
    save_after_param = request.form.get('save_after', '')

    if save_before_param:
        save_before = save_before_param.lower() in ('true', '1', 'yes')
    else:
        save_before = get_save_before()

    if save_after_param:
        save_after = save_after_param.lower() in ('true', '1', 'yes')
    else:
        save_after = get_save_after()

    tab_title_raw = request.form.get('tab_title', 'untitled')
    # Sanitize tab title for use as folder name (replace spaces with underscores)
    tab_title = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in tab_title_raw).strip('_')[:50] or 'untitled'

    # Debug save options
    print(f"  Save options: before={save_before}, after={save_after}, tab_title='{tab_title}'")

    # Parse translation mode flags - apply defaults from config if not explicitly set
    ocr_translate_param = request.form.get('ocr_translate', None)
    translate_local_param = request.form.get('translate_local', None)
    inpaint_background_param = request.form.get('inpaint_background', None)

    # If neither flag is explicitly set, use default from config
    if ocr_translate_param is None and translate_local_param is None:
        ocr_translate = (DEFAULT_TRANSLATE_MODE == 'vlm')
        translate_local = (DEFAULT_TRANSLATE_MODE == 'local')
    else:
        ocr_translate = (ocr_translate_param or '').lower() in ('true', '1', 'yes')
        translate_local = (translate_local_param or '').lower() in ('true', '1', 'yes')

    # AOT inpainting: default True (inpaint label2 regions in parallel with OCR/translate)
    if inpaint_background_param is None:
        inpaint_background = True
    else:
        inpaint_background = (inpaint_background_param or '').lower() in ('true', '1', 'yes')

    # Validate output_type
    if output_type not in ['full_page', 'speech_image_only', 'text_only']:
        return jsonify({'error': f'Invalid output_type: {output_type}. Must be one of: full_page, speech_image_only, text_only'}), 400

    # Validate API key for Cerebras mode
    effective_api_key = api_key or CEREBRAS_API_KEY
    if not ocr_translate and not translate_local and not effective_api_key:
        return jsonify({
            'error': 'No Cerebras API key provided. Set CEREBRAS_API_KEY env var, pass api_key parameter, use ocr_translate=true / translate_local=true, or run: python setup.py --configure'
        }), 400

    try:
        # Load images
        images = []
        for f in files:
            img = Image.open(f.stream).convert('RGB')
            images.append(img)

        # Save "before" images if requested
        if save_before:
            before_dir = os.path.join('output', 'before', tab_title)
            os.makedirs(before_dir, exist_ok=True)
            for idx, img in enumerate(images):
                img.save(os.path.join(before_dir, f'image_{idx:03d}.png'))
            print(f"  Saved {len(images)} 'before' images to {before_dir}/")

        # Build mode string for logging
        if ocr_translate:
            mode_str = "ocr_translate (VLM)"
        elif translate_local:
            mode_str = "ocr_vlm + translate_local (LLM)"
        else:
            mode_str = "ocr_vlm + translate_api (Cerebras)"
        inpaint_str = " + AOT inpaint" if inpaint_background else ""
        print(f"Processing {len(images)} images ({mode_str}{inpaint_str}, output_type={output_type})...")
        if api_key:
            print(f"  Using custom API key: {api_key[:10]}...")
        if ocr_translate:
            print(f"  OCR+Translate mode: {'VLM available' if HAS_VLM_OCR else 'VLM not available (will use standard flow)'}")
        if translate_local and not ocr_translate:
            print(f"  Local translate: {'Available' if HAS_LOCAL_TRANSLATE else 'Not available'}")
        if inpaint_background:
            print(f"  AOT Inpainting: Enabled (text-free regions inpainted in parallel)")

        # Process
        ocr_data, output, stats = process_images(images, api_key, output_type, ocr_translate, translate_local, inpaint_background, tab_title=tab_title)

        # Format response based on output_type
        elapsed = time.time() - start_time
        stats["total_ms"] = int(elapsed * 1000)

        if output_type == "text_only":
            # Return text data only
            response = {
                'status': 'success',
                'output_type': 'text_only',
                'page_count': len(output),
                'processing_time_ms': int(elapsed * 1000),
                'stats': stats,
                'pages': output  # Dictionary of page_idx -> bubble data
            }

        elif output_type == "speech_image_only":
            # Return bubble images with positions
            response = {
                'status': 'success',
                'output_type': 'speech_image_only',
                'page_count': len(output),
                'bubble_count': stats.get('bubble_count', 0),
                'processing_time_ms': int(elapsed * 1000),
                'stats': stats,
                'pages': output  # Dictionary of page_idx -> list of bubbles with base64 images
            }

        else:  # full_page (default)
            # Save "after" images if requested
            if save_after:
                after_dir = os.path.join('output', 'after', tab_title)
                os.makedirs(after_dir, exist_ok=True)
                for idx, img in enumerate(output):
                    img.save(os.path.join(after_dir, f'image_{idx:03d}.png'))
                print(f"  Saved {len(output)} 'after' images to {after_dir}/")

            # Encode output images
            t0 = time.time()
            output_b64 = images_to_base64(output)
            stats["encode_ms"] = int((time.time() - t0) * 1000)

            # Format results as expected by frontend (array of {success, data_url})
            # Add data URL prefix for browser compatibility
            results = [{'success': True, 'data_url': f'data:image/jpeg;base64,{img_b64}'} for img_b64 in output_b64]

            response = {
                'status': 'success',
                'output_type': 'full_page',
                'page_count': len(output),
                'processing_time_ms': int(elapsed * 1000),
                'stats': stats,
                'results': results
            }

        print(f"  [Total] {elapsed:.2f}s")
        print(f"  Stats: {json.dumps({k: v for k, v in stats.items() if 'details' not in k}, indent=2)}")

        # Clear GPU cache to free VRAM
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/process/legacy', methods=['POST'])
def process_legacy():
    """Legacy sequential mode: detect -> {inpaint, translate} -> render"""
    start_time = time.time()

    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400

    # Get optional parameters from form data
    api_key = request.form.get('api_key', None)
    output_type = request.form.get('output_type', 'full_page')

    # Validate output_type
    if output_type not in ['full_page', 'speech_image_only', 'text_only']:
        return jsonify({'error': f'Invalid output_type: {output_type}. Must be one of: full_page, speech_image_only, text_only'}), 400

    try:
        # Load images
        images = []
        for f in files:
            img = Image.open(f.stream).convert('RGB')
            images.append(img)

        print(f"Processing {len(images)} images (legacy mode with sequential inpainting, output_type={output_type})...")
        if api_key:
            print(f"  Using custom API key: {api_key[:10]}...")

        # Process
        ocr_data, output, stats = process_images_legacy(images, api_key, output_type)

        # Format response based on output_type
        elapsed = time.time() - start_time
        stats["total_ms"] = int(elapsed * 1000)

        if output_type == "text_only":
            # Return text data only
            response = {
                'status': 'success',
                'output_type': 'text_only',
                'page_count': len(output),
                'processing_time_ms': int(elapsed * 1000),
                'stats': stats,
                'pages': output  # Dictionary of page_idx -> bubble/region data
            }

        elif output_type == "speech_image_only":
            # Return bubble images with positions
            response = {
                'status': 'success',
                'output_type': 'speech_image_only',
                'page_count': len(output),
                'bubble_count': stats.get('bubble_count', 0),
                'processing_time_ms': int(elapsed * 1000),
                'stats': stats,
                'pages': output  # Dictionary of page_idx -> list of bubbles with base64 images
            }

        else:  # full_page (default)
            # Encode output images
            t0 = time.time()
            output_b64 = images_to_base64(output)
            stats["encode_ms"] = int((time.time() - t0) * 1000)

            # Format results as expected by frontend (array of {success, data_url})
            # Add data URL prefix for browser compatibility
            results = [{'success': True, 'data_url': f'data:image/jpeg;base64,{img_b64}'} for img_b64 in output_b64]

            response = {
                'status': 'success',
                'output_type': 'full_page',
                'page_count': len(output),
                'processing_time_ms': int(elapsed * 1000),
                'stats': stats,
                'results': results
            }

        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Stats: {json.dumps({k: v for k, v in stats.items() if 'details' not in k}, indent=2)}")

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


# ─────────────────────────────────────────────────────────────────────────────
# Configuration API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    from config import load_config
    cfg = load_config()
    return jsonify(cfg)


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration. Partial updates supported."""
    from config import load_config, save_config

    try:
        updates = request.get_json()
        if not updates:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Load current config and merge updates
        cfg = load_config()
        for key, value in updates.items():
            if key in cfg:
                cfg[key] = value

        # Save updated config
        save_config(cfg)

        return jsonify(cfg)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/defaults', methods=['GET'])
def get_config_defaults():
    """Get default configuration values."""
    from config import DEFAULT_CONFIG
    return jsonify(DEFAULT_CONFIG)


@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get list of supported target languages."""
    from config import LANGUAGES
    languages = [
        {'code': code, 'name': name}
        for code, name in LANGUAGES.items()
    ]
    return jsonify(languages)


@app.route('/api/ocr-methods', methods=['GET'])
def get_ocr_methods():
    """Get list of available OCR methods with details."""
    import platform
    is_windows = platform.system() == 'Windows'

    methods = [
        {
            'id': 'qwen_vlm',
            'name': 'Qwen 3 VL 2B',
            'description': 'Recommended - Good balance of speed and quality',
            'vram': '~2.3GB',
            'requires_api_key': False,
            'requires_remote_url': False,
        },
        {
            'id': 'lfm_vlm',
            'name': 'LFM 2.5 VL 1.6B',
            'description': 'Smallest and fastest local model',
            'vram': '~1.7GB',
            'requires_api_key': False,
            'requires_remote_url': False,
        },
        {
            'id': 'ministral_vlm_q8',
            'name': 'Ministral 3B Q8',
            'description': 'Best quality local model',
            'vram': '~4.5GB',
            'requires_api_key': False,
            'requires_remote_url': False,
        },
        {
            'id': 'oneocr_remote',
            'name': 'OneOCR (Remote)',
            'description': 'Network-based Windows OCR server',
            'vram': '0GB',
            'requires_api_key': False,
            'requires_remote_url': True,
        },
        {
            'id': 'gemini_api',
            'name': 'Gemini API',
            'description': 'Cloud-based OCR using Google Gemini',
            'vram': '0GB',
            'requires_api_key': True,
            'api_key_type': 'gemini',
            'requires_remote_url': False,
        },
    ]

    # Add local OneOCR only on Windows
    if is_windows:
        methods.insert(0, {
            'id': 'oneocr',
            'name': 'OneOCR (Local)',
            'description': 'Windows-only, uses no VRAM',
            'vram': '0GB',
            'requires_api_key': False,
            'requires_remote_url': False,
            'platform': 'windows',
        })

    return jsonify(methods)


@app.route('/api/translation-methods', methods=['GET'])
def get_translation_methods():
    """Get list of available translation methods with details."""
    methods = [
        {
            'id': 'hunyuan_mt',
            'name': 'Hunyuan MT (Local)',
            'description': 'Fast local translation model',
            'vram': '~1.1GB',
            'requires_api_key': False,
        },
        {
            'id': 'cerebras_api',
            'name': 'Cerebras API',
            'description': 'Cloud-based, 1M tokens/day free tier',
            'vram': '0GB',
            'requires_api_key': True,
            'api_key_type': 'cerebras',
        },
        {
            'id': 'gemini_translate',
            'name': 'Gemini API',
            'description': 'Cloud-based translation using Google Gemini',
            'vram': '0GB',
            'requires_api_key': True,
            'api_key_type': 'gemini',
        },
    ]
    return jsonify(methods)


@app.route('/api/verify/gemini-api-key', methods=['POST'])
def verify_gemini_key():
    """Verify a Gemini API key is valid."""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '')

        if not api_key:
            return jsonify({'valid': False, 'error': 'No API key provided'})

        # Test the API key by making a simple request
        import requests
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(test_url, timeout=10)

        if response.status_code == 200:
            return jsonify({'valid': True, 'message': 'API key is valid'})
        elif response.status_code == 400:
            return jsonify({'valid': False, 'error': 'Invalid API key format'})
        elif response.status_code == 403:
            return jsonify({'valid': False, 'error': 'API key is invalid or expired'})
        else:
            return jsonify({'valid': False, 'error': f'Unexpected response: {response.status_code}'})

    except requests.exceptions.Timeout:
        return jsonify({'valid': False, 'error': 'Request timed out'})
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})


@app.route('/api/verify/cerebras-api-key', methods=['POST'])
def verify_cerebras_key():
    """Verify a Cerebras API key is valid."""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '')

        if not api_key:
            return jsonify({'valid': False, 'error': 'No API key provided'})

        # Test the API key by making a simple request
        import requests
        test_url = "https://api.cerebras.ai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(test_url, headers=headers, timeout=10)

        if response.status_code == 200:
            return jsonify({'valid': True, 'message': 'API key is valid'})
        elif response.status_code == 401:
            return jsonify({'valid': False, 'error': 'Invalid API key'})
        else:
            return jsonify({'valid': False, 'error': f'Unexpected response: {response.status_code}'})

    except requests.exceptions.Timeout:
        return jsonify({'valid': False, 'error': 'Request timed out'})
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})


@app.route('/api/verify/oneocr-server', methods=['POST'])
def verify_oneocr_server():
    """Verify a OneOCR server is reachable."""
    try:
        data = request.get_json()
        url = data.get('url', '')

        if not url:
            return jsonify({'valid': False, 'error': 'No URL provided'})

        from config import verify_oneocr_server as check_oneocr
        success, message, details = check_oneocr(url, timeout=5.0)

        return jsonify({
            'valid': success,
            'message': message,
            'details': details
        })

    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)})


@app.route('/reset-vlm', methods=['POST'])
def reset_vlm():
    """Reset VLM OCR instance to allow model switching.

    Call this after changing ocr_method in config.json to reload the model.
    """
    try:
        reset_vlm_ocr()
        reset_api_ocr()
        # Reload config to get new model info
        from config import load_config, get_ocr_method, get_ocr_model, is_api_ocr_method
        cfg = load_config()
        ocr_method = get_ocr_method()
        response = {
            'status': 'ok',
            'message': 'OCR instances reset. Next OCR request will use new model.',
            'ocr_method': ocr_method,
        }
        if is_api_ocr_method(ocr_method):
            response['ocr_type'] = 'api'
        else:
            response['ocr_model'] = get_ocr_model()
            response['ocr_type'] = 'local'
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# SSE Streaming Endpoint - Parallel OCR→Translate Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _create_sse_event(event_type: str, data: dict) -> str:
    """Create an SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _run_ocr_batch(batch_bubbles: list):
    """
    Run OCR on a single batch of bubbles (OCR_BATCH_SIZE bubbles per grid).
    Returns list of bubble_info dicts with original_text.
    """
    from workflow.ocr_vlm import VlmOCR, create_ocr_grid
    vlm = VlmOCR()

    # Create grid and run OCR
    grid_img, positions, grid_info = create_ocr_grid(batch_bubbles)
    ocr_result = vlm.run(grid_img, positions, grid_info, translate=False)

    # Map OCR results to bubbles
    bubble_texts = map_ocr(ocr_result, positions)

    # Extract bubble info with original text
    results = []
    for pos in positions:
        key = pos['key']
        page_idx, bubble_idx = key
        texts = bubble_texts.get(key, [])
        merged = "".join([t['text'] for t in texts])
        results.append({
            'page_idx': page_idx,
            'bubble_idx': bubble_idx,
            'bubble_box': list(pos['bubble_box']),
            'original_text': merged.strip() if merged.strip() else '',
            'global_idx': None  # Will be set by caller
        })

    return results


def _translate_texts_batch(texts: list, translate_local: bool, api_key: str) -> list:
    """
    Translate a batch of texts (TRANSLATE_BATCH_SIZE texts).
    Returns list of translations.
    Dispatches to local, Gemini, or Cerebras based on TRANSLATE_METHOD config.
    """
    if not texts:
        return []

    if translate_local and HAS_LOCAL_TRANSLATE:
        translator = LlmTranslator()
        result = translator.translate(texts)
        return result.get('translations', [''] * len(texts))
    elif TRANSLATE_METHOD == "gemini_translate" and HAS_GEMINI_TRANSLATE:
        # Use Gemini API
        return translate_texts_gemini(texts)
    else:
        # Use Cerebras API
        return translate_texts(texts, api_key=api_key)


def _translate_batch_async(translate_batch_idx: int, bubble_infos: list, texts: list,
                           translate_local: bool, api_key: str, result_queue: queue.Queue):
    """
    Translate a batch and put results in queue.
    bubble_infos: list of bubble info dicts (with global_idx for ordering)
    texts: list of texts to translate (aligned with bubble_infos that have text)
    """
    try:
        # Translate
        translations = _translate_texts_batch(texts, translate_local, api_key)

        # Map translations back to bubble_infos
        # Only bubbles with original_text get translations
        text_idx = 0
        batch_results = []
        for info in bubble_infos:
            result_info = dict(info)
            if info['original_text']:
                result_info['translated_text'] = translations[text_idx] if text_idx < len(translations) else '[MISSING]'
                text_idx += 1
            else:
                result_info['translated_text'] = ''
            batch_results.append(result_info)

        result_queue.put({
            'translate_batch_idx': translate_batch_idx,
            'status': 'success',
            'results': batch_results,
            'translate_count': len([t for t in translations if t and t.strip()])
        })

    except Exception as e:
        # On error, return bubble_infos with error translations
        batch_results = []
        for info in bubble_infos:
            result_info = dict(info)
            result_info['translated_text'] = '[TRANSLATE_ERROR]'
            batch_results.append(result_info)

        result_queue.put({
            'translate_batch_idx': translate_batch_idx,
            'status': 'error',
            'error': str(e),
            'results': batch_results
        })


@app.route('/api/v1/process/stream', methods=['POST'])
def process_stream():
    """
    SSE Streaming endpoint: detect -> parallel OCR+translate pipeline -> stream results

    Two-stage pipeline with different batch sizes:
    - OCR: processes OCR_BATCH_SIZE (9) bubbles per grid
    - Translation: processes TRANSLATE_BATCH_SIZE (25) texts per batch

    OCR runs sequentially, but translation runs in parallel as texts accumulate.
    Results stream back in bubble order (page 0 bubble 0, page 0 bubble 1, etc.)

    Response format (SSE events):
      event: start
      data: {"total_batches": N, "total_bubbles": M, "pages": P, "detect_ms": T}

      event: result  (one per bubble, in order)
      data: {"global_idx": 0, "page_idx": 0, "bubble_idx": 0, "bubble_box": [...],
             "original_text": "...", "translated_text": "..."}

      event: complete
      data: {"total_time_ms": ..., "stats": {...}}

      event: error
      data: {"error": "..."}
    """
    if not STREAMING_ENABLED:
        return jsonify({'error': 'Streaming not enabled. Set streaming_enabled: true in config.json'}), 400

    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400

    # Get parameters
    api_key = request.form.get('api_key', None)
    translate_local_param = request.form.get('translate_local', None)
    translate_local = (translate_local_param or '').lower() in ('true', '1', 'yes')
    if translate_local_param is None:
        translate_local = (DEFAULT_TRANSLATE_MODE == 'local')

    # Validate API key for Cerebras mode
    effective_api_key = api_key or CEREBRAS_API_KEY
    if not translate_local and not effective_api_key:
        return jsonify({
            'error': 'No Cerebras API key provided. Set CEREBRAS_API_KEY env var, pass api_key parameter, or use translate_local=true'
        }), 400

    # Load images
    try:
        images = []
        for f in files:
            img = Image.open(f.stream).convert('RGB')
            images.append(img)
    except Exception as e:
        return jsonify({'error': f'Failed to load images: {e}'}), 400

    def generate_events():
        """Generator for SSE events."""
        start_time = time.time()
        stats = {"pages": len(images), "streaming": True}

        try:
            # Detection
            session, mode = get_detector()
            bubbles, detect_time = detect_all(session, images, mode, target_label=1)
            stats["detect_ms"] = int(detect_time)
            stats["bubbles_detected"] = len(bubbles)

            # Sequential mode: start OCR server
            uses_vlm_ocr = OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8")
            if SEQUENTIAL_MODEL_LOADING and uses_vlm_ocr and HAS_VLM_OCR:
                seq_mgr = get_sequential_manager()
                seq_mgr.start_ocr_server()

            # Create batches (ordered by page, then bubble_idx within page)
            # Bubbles are already sorted in detect_all
            batch_size = OCR_BATCH_SIZE
            batches = []
            for i in range(0, len(bubbles), batch_size):
                batches.append(bubbles[i:i + batch_size])

            stats["total_batches"] = len(batches)

            # Send start event
            yield _create_sse_event("start", {
                "total_batches": len(batches),
                "total_bubbles": len(bubbles),
                "pages": len(images),
                "detect_ms": int(detect_time)
            })

            if not bubbles:
                yield _create_sse_event("complete", {
                    "total_time_ms": int((time.time() - start_time) * 1000),
                    "stats": stats
                })
                return

            # Two-stage pipeline with different batch sizes:
            # Stage 1: OCR batches (OCR_BATCH_SIZE bubbles, e.g., 9)
            # Stage 2: Translation batches (TRANSLATE_BATCH_SIZE texts, e.g., 25)
            #
            # Flow:
            # - OCR runs sequentially (llama-server limitation)
            # - Accumulate OCR results until we have enough for a translation batch
            # - Translation runs in parallel threads (different server/API)
            # - Stream results as translation batches complete, in bubble order

            result_queue = queue.Queue()
            translate_threads = []

            # Accumulator for OCR results pending translation
            pending_bubble_infos = []  # List of bubble_info dicts
            pending_texts = []  # Texts to translate (only non-empty)
            global_bubble_idx = 0
            translate_batch_idx = 0

            # Track which translate batches contain which global indices for ordering
            translate_batch_ranges = {}  # translate_batch_idx -> (start_global_idx, end_global_idx)
            completed_translate_batches = {}
            next_global_idx_to_send = 0
            all_results = {}  # global_idx -> result

            def flush_to_translation():
                """Send accumulated texts to translation in a separate thread."""
                nonlocal translate_batch_idx, pending_bubble_infos, pending_texts

                if not pending_bubble_infos:
                    return

                # Record range for this translate batch
                start_idx = pending_bubble_infos[0]['global_idx']
                end_idx = pending_bubble_infos[-1]['global_idx']
                translate_batch_ranges[translate_batch_idx] = (start_idx, end_idx)

                # Start translation thread
                t = threading.Thread(
                    target=_translate_batch_async,
                    args=(translate_batch_idx, list(pending_bubble_infos), list(pending_texts),
                          translate_local, effective_api_key, result_queue)
                )
                t.start()
                translate_threads.append(t)

                translate_batch_idx += 1
                pending_bubble_infos = []
                pending_texts = []

            # Process OCR batches
            for ocr_batch_idx, batch_bubbles in enumerate(batches):
                try:
                    # Run OCR for this batch (sequential)
                    ocr_results = _run_ocr_batch(batch_bubbles)

                    # Add global indices and accumulate
                    for info in ocr_results:
                        info['global_idx'] = global_bubble_idx
                        pending_bubble_infos.append(info)
                        if info['original_text']:
                            pending_texts.append(info['original_text'])
                        global_bubble_idx += 1

                    # Check if we have enough texts for a translation batch
                    while len(pending_texts) >= TRANSLATE_BATCH_SIZE:
                        # Find cutoff point: take TRANSLATE_BATCH_SIZE texts worth of bubbles
                        text_count = 0
                        cutoff_idx = 0
                        for i, info in enumerate(pending_bubble_infos):
                            if info['original_text']:
                                text_count += 1
                            if text_count >= TRANSLATE_BATCH_SIZE:
                                cutoff_idx = i + 1
                                break

                        # Split at cutoff
                        batch_infos = pending_bubble_infos[:cutoff_idx]
                        batch_texts = pending_texts[:TRANSLATE_BATCH_SIZE]
                        pending_bubble_infos = pending_bubble_infos[cutoff_idx:]
                        pending_texts = pending_texts[TRANSLATE_BATCH_SIZE:]

                        # Record range and start translation
                        start_idx = batch_infos[0]['global_idx']
                        end_idx = batch_infos[-1]['global_idx']
                        translate_batch_ranges[translate_batch_idx] = (start_idx, end_idx)

                        t = threading.Thread(
                            target=_translate_batch_async,
                            args=(translate_batch_idx, batch_infos, batch_texts,
                                  translate_local, effective_api_key, result_queue)
                        )
                        t.start()
                        translate_threads.append(t)
                        translate_batch_idx += 1

                except Exception as e:
                    # OCR failed, add error results
                    for b in batch_bubbles:
                        all_results[global_bubble_idx] = {
                            'page_idx': b['page_idx'],
                            'bubble_idx': b['bubble_idx'],
                            'bubble_box': list(b['box']),
                            'original_text': '[OCR_ERROR]',
                            'translated_text': '[OCR_ERROR]',
                            'global_idx': global_bubble_idx
                        }
                        global_bubble_idx += 1

                # Check for completed translations and stream results
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()
                        batch_idx = result['translate_batch_idx']
                        for r in result['results']:
                            all_results[r['global_idx']] = r

                        # Stream results in order
                        while next_global_idx_to_send in all_results:
                            r = all_results[next_global_idx_to_send]
                            yield _create_sse_event("result", {
                                'global_idx': r['global_idx'],
                                'page_idx': r['page_idx'],
                                'bubble_idx': r['bubble_idx'],
                                'bubble_box': r['bubble_box'],
                                'original_text': r['original_text'],
                                'translated_text': r['translated_text']
                            })
                            del all_results[next_global_idx_to_send]
                            next_global_idx_to_send += 1
                    except queue.Empty:
                        break

            # Flush remaining pending texts to translation
            if pending_bubble_infos:
                flush_to_translation()

            # Sequential mode: stop OCR server (done with OCR)
            if SEQUENTIAL_MODEL_LOADING and uses_vlm_ocr and HAS_VLM_OCR:
                seq_mgr = get_sequential_manager()
                seq_mgr.stop_ocr_server()

            # Wait for all translation threads
            for t in translate_threads:
                t.join(timeout=120)

            # Collect remaining results
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    for r in result['results']:
                        all_results[r['global_idx']] = r
                except queue.Empty:
                    break

            # Stream any remaining results in order
            total_bubbles = global_bubble_idx
            while next_global_idx_to_send < total_bubbles:
                if next_global_idx_to_send in all_results:
                    r = all_results[next_global_idx_to_send]
                    yield _create_sse_event("result", {
                        'global_idx': r['global_idx'],
                        'page_idx': r['page_idx'],
                        'bubble_idx': r['bubble_idx'],
                        'bubble_box': r['bubble_box'],
                        'original_text': r['original_text'],
                        'translated_text': r['translated_text']
                    })
                    del all_results[next_global_idx_to_send]
                    next_global_idx_to_send += 1
                else:
                    # Missing result, wait a bit
                    time.sleep(0.1)
                    # Timeout check
                    if time.time() - start_time > 300:  # 5 min timeout
                        yield _create_sse_event("error", {"error": f"Timeout waiting for result {next_global_idx_to_send}"})
                        break

            # Send complete event
            total_time = int((time.time() - start_time) * 1000)
            stats["total_time_ms"] = total_time
            yield _create_sse_event("complete", {
                "total_time_ms": total_time,
                "stats": stats
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _create_sse_event("error", {"error": str(e)})

    return Response(
        generate_events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering
        }
    )


if __name__ == '__main__':
    # Rich UI imports
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        console = Console()
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False
        console = None

    def _get_ocr_display():
        """Get OCR method display string."""
        if OCR_METHOD == "gemini_api":
            model = get_gemini_model()
            return f"Gemini API ({model})"
        elif OCR_METHOD == "oneocr_remote":
            return "OneOCR Remote"
        return OCR_METHOD

    def _get_translate_display():
        """Get translate method display string."""
        if TRANSLATE_METHOD == "cerebras_api":
            return "Cerebras API"
        elif TRANSLATE_METHOD == "gemini_translate":
            model = get_gemini_translate_model()
            return f"Gemini API ({model})"
        return TRANSLATE_METHOD

    def _get_gemini_keys_info():
        """Get Gemini API keys info."""
        from config import get_gemini_api_keys
        keys = get_gemini_api_keys()
        if not keys:
            return "[red]NOT SET[/red]", 0
        return f"[green]{len(keys)} key(s)[/green]", len(keys)

    if HAS_RICH:
        # Rich UI banner
        console.print(Panel.fit(
            "[bold cyan]Manga Translation Server[/bold cyan]",
            border_style="cyan"
        ))

        # Configuration table
        config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value")

        config_file_exists = os.path.exists(CONFIG_FILE)
        config_table.add_row("Config", f"{CONFIG_FILE} [dim]({'loaded' if config_file_exists else 'defaults'})[/dim]")
        config_table.add_row("OCR", f"[cyan]{_get_ocr_display()}[/cyan]")
        config_table.add_row("Translate", f"[cyan]{_get_translate_display()}[/cyan]")
        config_table.add_row("Target", f"[green]{get_target_language()}[/green]")

        # API keys status
        if OCR_METHOD == "gemini_api" or TRANSLATE_METHOD == "gemini_translate":
            gemini_status, key_count = _get_gemini_keys_info()
            rate_info = f" [dim]({key_count * 30} req/min)[/dim]" if key_count > 0 else ""
            config_table.add_row("Gemini Keys", f"{gemini_status}{rate_info}")

        if TRANSLATE_METHOD == "cerebras_api":
            cerebras_status = "[green]configured[/green]" if CEREBRAS_API_KEY else "[red]NOT SET[/red]"
            config_table.add_row("Cerebras Key", cerebras_status)

        console.print(config_table)

        # Endpoints
        console.print("\n[bold]Endpoints:[/bold]")
        console.print("  [cyan]POST[/cyan] /api/v1/process        [dim]- Main pipeline[/dim]")
        console.print("  [cyan]POST[/cyan] /api/v1/process/stream [dim]- SSE streaming[/dim]")
        console.print("  [cyan]POST[/cyan] /api/v1/process/legacy [dim]- Legacy sequential[/dim]")
        console.print("  [cyan]GET[/cyan]  /health                [dim]- Health check[/dim]")

        # Status
        console.print("")
        if is_api_ocr_method(OCR_METHOD) and OCR_METHOD == "gemini_api":
            gemini_key = get_gemini_api_key() or os.environ.get("GEMINI_API_KEY", "")
            if HAS_GEMINI_OCR and gemini_key:
                console.print(f"[green]✓[/green] Gemini OCR ready ({get_gemini_model()})")
            elif not HAS_GEMINI_OCR:
                console.print("[red]✗[/red] Gemini SDK not installed [dim](pip install google-genai)[/dim]")
            else:
                console.print("[red]✗[/red] Gemini API key not set")

        if TRANSLATE_METHOD == "gemini_translate":
            if HAS_GEMINI_TRANSLATE:
                console.print(f"[green]✓[/green] Gemini Translate ready ({get_gemini_translate_model()})")
        elif TRANSLATE_METHOD == "cerebras_api":
            if CEREBRAS_API_KEY:
                console.print("[green]✓[/green] Cerebras Translate ready")
            else:
                console.print("[red]✗[/red] Cerebras API key not set")

        console.print("")
    else:
        # Fallback plain text
        print("Manga Translation Server")
        print("=" * 40)
        print(f"OCR: {_get_ocr_display()}")
        print(f"Translate: {_get_translate_display()}")
        print(f"Target: {get_target_language()}")
        print("=" * 40)

    # Auto-start llama servers if configured and needed
    if needs_llama() and AUTO_START_SERVERS and not SEQUENTIAL_MODEL_LOADING:
        # Standard mode: start both servers at startup
        print("\nStarting llama servers...")
        _start_llama_servers()
        # _start_llama_servers already reports final status with proper health checks
    elif needs_llama() and SEQUENTIAL_MODEL_LOADING:
        # Sequential mode: servers started on demand per request
        print("\n[Sequential Mode] llama servers will start/stop on demand per request")
        print("  This is SLOW but uses less VRAM (only one model loaded at a time)")
        print("  Ideal for GPUs with 4-6GB VRAM")
    elif needs_llama() and not AUTO_START_SERVERS:
        print("\nAuto-start disabled. Start llama servers manually:")
        llama_server = llama_server_path or "llama-server"
        ctx = get_llama_context_size()
        ngl = get_llama_gpu_layers()
        mmproj = get_ocr_mmproj()
        # mmproj needs to be downloaded first, then use local path with --mmproj
        mmproj_note = f"\n  # Download mmproj first: curl -L -o mmproj.gguf https://huggingface.co/{mmproj.replace(':', '/resolve/main/')}" if mmproj and ':' in mmproj else ""
        mmproj_flag = " --mmproj mmproj.gguf" if mmproj else ""
        print(f"  {llama_server} -hf {get_ocr_model()}{mmproj_flag} --port 8080 -c {ctx} -ngl {ngl}{mmproj_note}")
        if TRANSLATE_METHOD in ("qwen_vlm", "hunyuan_mt") and not (OCR_METHOD in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8") and TRANSLATE_METHOD == "qwen_vlm"):
            print(f"  {llama_server} -hf {get_translate_model()} --port 8081 -c {ctx} -ngl {ngl}")

    # Pre-load detector only (inpainter loaded on-demand for legacy/AOT inpaint)
    print("\nLoading models...")
    get_detector()
    # get_inpainter()  # Loaded on-demand to save VRAM
    print("Detector loaded. Inpainter will load on first request requiring it.\n")

    port = int(os.environ.get('PORT', get_server_port()))
    app.run(host='0.0.0.0', port=port, debug=False)
