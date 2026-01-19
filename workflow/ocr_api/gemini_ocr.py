"""Gemma 3 API OCR - Uses Google AI Studio with Gemma 3 for OCR via API

Gemma 3 is Google's open multimodal model, available via the Gemini API.
Uses grid images (same as VLM OCR) - one grid per API call, batches run in parallel.

Requires: pip install google-genai
API Key: Set GEMINI_API_KEY environment variable or configure in config.json
         Get your key at: https://aistudio.google.com/apikey
"""

import io
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    Image = None

# Import from centralized config
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    get_target_language_name,
    get_ocr_grid_max_cells,
)

# Import grid creation from VLM OCR module
try:
    from ..ocr_vlm import create_ocr_grid
    _HAS_GRID = True
except ImportError:
    create_ocr_grid = None
    _HAS_GRID = False

# Lazy import google-genai to avoid import errors if not installed
_genai_client = None
_genai_types = None


def _get_genai():
    """Lazy load google-genai module."""
    global _genai_client, _genai_types
    if _genai_client is not None:
        return _genai_client, _genai_types

    try:
        from google import genai
        from google.genai import types

        # Get API key from environment or config
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            from config import get_gemini_api_key
            api_key = get_gemini_api_key()

        if not api_key:
            return None, None

        _genai_client = genai.Client(api_key=api_key)
        _genai_types = types
        return _genai_client, _genai_types
    except ImportError:
        return None, None
    except Exception as e:
        print(f"[Gemini OCR] Failed to initialize: {e}")
        return None, None


def check_gemini_available() -> bool:
    """Check if Gemini API is available (SDK installed + API key configured)."""
    client, _ = _get_genai()
    return client is not None


# Available Gemma 3 models for vision/OCR tasks (via Google AI Studio)
# Gemma 3 is multimodal and supports text + image input
# NOTE: Only 27B IT model is recommended for OCR quality
GEMMA_MODELS = {
    "gemma3_27b": "gemma-3-27b-it",      # Best quality, instruction-tuned 27B (RECOMMENDED)
}

# Gemini models - cloud API options
GEMINI_MODELS = {
    "gemini_flash_lite_2.5": "gemini-2.5-flash-lite", # Flash Lite 2.5
    "gemini_flash_3": "gemini-3-flash-preview",       # Flash 3 preview (newest)
}

# All supported models
ALL_MODELS = {**GEMMA_MODELS, **GEMINI_MODELS}

DEFAULT_MODEL = "gemma-3-27b-it"  # Gemma 3 27B for best OCR quality

# Max cells per grid batch (same as VLM OCR - uses config value)


def _image_to_bytes(image: Image.Image) -> Tuple[bytes, str]:
    """Convert PIL Image to bytes and mime type."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=95)
    return buf.getvalue(), 'image/jpeg'


def _build_ocr_prompt(grid_info: Dict, translate: bool = False) -> str:
    """Build the OCR prompt for Gemini (grid-based, same as VLM)."""
    n = grid_info['total_cells']
    target_lang = get_target_language_name()

    if n == 1:
        if translate:
            return f"OCR: Read the text in this image (Japanese/Korean/Chinese) and translate to {target_lang}. Output ONLY the translation, no preamble."
        return "OCR: Read the text in this image exactly as written. Output ONLY the text, no preamble or explanation."
    else:
        if translate:
            return f"""OCR: {n} manga speech bubbles in a grid (BLUE=column separator, RED=row separator).
Output exactly {n} entries: [0]: translation [1]: translation ... [{n-1}]: translation
Translate Japanese/Korean/Chinese to {target_lang}. Output ONLY the numbered entries, nothing else."""
        return f"""OCR: {n} manga speech bubbles in a grid (BLUE=column separator, RED=row separator).
Output exactly {n} entries: [0]: text [1]: text ... [{n-1}]: text
Read each bubble's text exactly as written. Output ONLY the numbered entries, nothing else."""


def _parse_gemini_output(output: str, num_expected: int) -> Dict[int, str]:
    """Parse Gemini output to extract cell texts."""
    cell_texts = {}

    # Strip common preambles that Gemini adds
    output = output.strip()
    preamble_patterns = [
        r"^Here'?s the OCR (?:result|output)[^:]*:\s*",
        r"^OCR (?:result|output)[^:]*:\s*",
        r"^Following your (?:instructions|format)[^:]*:\s*",
    ]
    for pattern in preamble_patterns:
        output = re.sub(pattern, '', output, flags=re.IGNORECASE)
    output = output.strip()

    if num_expected == 1:
        text = output
        if text.startswith('[0]:'):
            text = text[4:].strip()
        if text:
            cell_texts[0] = text
        return cell_texts

    # Multi-cell: parse [N]: format - handles both newline and space separated
    # Pattern matches [N]: followed by text until next [N]: or end
    # Use non-greedy match and lookahead for next bracket
    pattern = r'\[(\d+)\]:\s*(.+?)(?=\s*\[\d+\]:|$)'
    for match in re.findall(pattern, output, re.DOTALL):
        try:
            idx, text = int(match[0]), match[1].strip()
            if text and idx < num_expected:
                cell_texts[idx] = text
        except (ValueError, IndexError):
            continue

    return cell_texts


class GeminiOCR:
    """Gemma 3 / Gemini API-based OCR for manga/manhwa bubble text extraction.

    Uses Google AI Studio API with Gemma 3 (or Gemini) models.
    Uses grid images (same as VLM OCR) - one grid per API call, batches run in parallel.

    Default model: gemma-3-27b-it (Gemma 3 27B instruction-tuned)
    """

    def __init__(self, model: str = None, max_cells_per_batch: int = None):
        """Initialize Gemma/Gemini OCR.

        Args:
            model: Model to use (default: gemma-3-27b-it)
            max_cells_per_batch: Max cells per grid batch (default from config)
        """
        self.model = model or os.environ.get('GEMINI_MODEL', DEFAULT_MODEL)
        self.max_cells_per_batch = max_cells_per_batch or get_ocr_grid_max_cells()
        self._client = None
        self._types = None

    def _ensure_client(self) -> bool:
        """Ensure Gemini client is initialized."""
        if self._client is not None:
            return True

        self._client, self._types = _get_genai()
        return self._client is not None

    def run(self, image: Image.Image, positions: List[Dict] = None,
            grid_info: Dict = None, translate: bool = False, max_retries: int = 4) -> Dict[str, Any]:
        """Run OCR on a single grid image.

        Args:
            image: PIL Image (grid of bubbles)
            positions: Position info for each cell in the grid
            grid_info: Grid layout info from create_ocr_grid
            translate: If True, translate to target language
            max_retries: Number of retries on failure

        Returns:
            OCR result dict with 'lines', 'line_count', 'translated', etc.
        """
        t0 = time.time()

        if image is None:
            return {'lines': [], 'line_count': 0, 'processing_time_ms': 0,
                    'translated': translate}

        if not self._ensure_client():
            return {'lines': [], 'line_count': 0, 'processing_time_ms': 0,
                    'error': 'Gemini API not available (check API key)', 'translated': translate}

        # Build prompt based on grid info
        if grid_info is None:
            grid_info = {'total_cells': len(positions) if positions else 1}

        prompt = _build_ocr_prompt(grid_info, translate=translate)
        expected_cells = grid_info.get('total_cells', len(positions) if positions else 1)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Build content with prompt and single grid image
                img_bytes, mime_type = _image_to_bytes(image)
                content = [
                    prompt,
                    self._types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                ]

                # Make API request
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=content,
                    config={
                        "temperature": 0.3 + (attempt * 0.05),  # Increase temp on retry
                        "max_output_tokens": 4096,
                    }
                )

                output = response.text if hasattr(response, 'text') else str(response)

                # Parse output
                cell_texts = _parse_gemini_output(output, expected_cells)

                # Check for missing cells
                missing = set(range(expected_cells)) - set(cell_texts.keys())
                if missing and attempt < max_retries:
                    missing_str = ','.join(str(i) for i in sorted(missing)[:5])
                    if len(missing) > 5:
                        missing_str += f"...+{len(missing)-5}more"
                    print(f"[Gemini OCR] Missing cells [{missing_str}], retry {attempt + 1}/{max_retries}")
                    last_error = f"missing_cells:[{missing_str}]"
                    time.sleep(0.5)
                    continue

                # Log raw output
                if output:
                    attempt_str = f" (attempt {attempt + 1}/{max_retries + 1})" if attempt > 0 else ""
                    print(f"[Gemini OCR] Output ({len(output)} chars){attempt_str}: {output[:500]}{'...' if len(output) > 500 else ''}")

                # Build lines result
                lines = []
                for cell_idx, text in cell_texts.items():
                    line_data = {
                        'text': text,
                        'cell_idx': cell_idx,
                        'bbox': {}
                    }
                    if positions and cell_idx < len(positions):
                        pos = positions[cell_idx]
                        if 'grid_box' in pos:
                            gx1, gy1, gx2, gy2 = pos['grid_box']
                            line_data['bbox'] = {
                                'x': gx1, 'y': gy1,
                                'width': gx2 - gx1, 'height': gy2 - gy1
                            }
                        if 'grid_row' in pos:
                            line_data['grid_pos'] = (pos['grid_row'], pos['grid_col'])
                    if translate:
                        line_data['translated'] = True
                    lines.append(line_data)

                print(f"[Gemini OCR] {len(lines)}/{expected_cells} cells parsed")

                return {
                    'lines': lines,
                    'line_count': len(lines),
                    'processing_time_ms': (time.time() - t0) * 1000,
                    'translated': translate,
                    'raw_output': output,
                    'retries': attempt
                }

            except Exception as e:
                last_error = str(e)
                # Print detailed error info
                error_msg = f"[Gemini OCR] Error (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}"
                print(error_msg)
                # Try to get more details from the exception
                if hasattr(e, 'response'):
                    print(f"[Gemini OCR] Response: {e.response}")
                if hasattr(e, 'status_code'):
                    print(f"[Gemini OCR] Status code: {e.status_code}")
                if hasattr(e, 'message'):
                    print(f"[Gemini OCR] Message: {e.message}")
                if attempt < max_retries:
                    # Exponential backoff for rate limiting
                    time.sleep(1 * (attempt + 1))

        print(f"[Gemini OCR] All retries failed. Last error: {last_error}")
        return {'lines': [], 'line_count': 0, 'processing_time_ms': (time.time() - t0) * 1000,
                'translated': translate, 'error': last_error}

    def run_grid(self, bubbles: List[Dict], row_width: int = 1600,
                 padding: int = 10, translate: bool = False) -> Tuple[Dict, List[Dict], Image.Image]:
        """Create grid and run OCR in one call (matches VlmOCR.run_grid).

        Args:
            bubbles: List of bubble dicts with 'image', 'page_idx', 'bubble_idx', 'box'
            row_width: Max width for grid rows
            padding: Padding between cells
            translate: If True, translate to target language

        Returns:
            Tuple of (ocr_result, positions, grid_image)
        """
        if not bubbles:
            return {'lines': [], 'line_count': 0, 'translated': translate}, [], None

        if not _HAS_GRID or create_ocr_grid is None:
            return {'lines': [], 'line_count': 0, 'translated': translate,
                    'error': 'Grid creation not available'}, [], None

        # Create grid image
        grid_img, positions, grid_info = create_ocr_grid(bubbles, row_width, padding)

        # Run OCR on grid
        ocr_result = self.run(grid_img, positions, grid_info, translate=translate)
        return ocr_result, positions, grid_img

    def run_batched(self, bubbles: List[Dict], row_width: int = 1600,
                    padding: int = 10, translate: bool = False) -> Tuple[Dict, List[Dict], None]:
        """Process bubbles in batches with parallel API calls (matches VlmOCR.run_batched).

        Each batch creates a grid image, and all batches are processed in parallel.

        Args:
            bubbles: List of bubble dicts
            row_width: Max width for grid rows
            padding: Padding between cells
            translate: If True, translate to target language

        Returns:
            Tuple of (ocr_result, positions, None)
        """
        if not bubbles:
            return {'lines': [], 'line_count': 0, 'translated': translate}, [], None

        # For small batches, run directly
        if len(bubbles) <= self.max_cells_per_batch:
            return self.run_grid(bubbles, row_width, padding, translate)

        # Split into batches
        t0 = time.time()
        batches = []
        for i in range(0, len(bubbles), self.max_cells_per_batch):
            batch_bubbles = bubbles[i:i + self.max_cells_per_batch]
            batches.append((i, batch_bubbles))

        print(f"[Gemini OCR] Processing {len(bubbles)} bubbles in {len(batches)} parallel batches")

        # Process all batches in parallel
        def process_batch(batch_data):
            batch_idx, (offset, batch_bubbles) = batch_data
            print(f"[Gemini OCR] Batch {batch_idx + 1}/{len(batches)} starting ({len(batch_bubbles)} bubbles)")
            result, positions, _ = self.run_grid(batch_bubbles, row_width, padding, translate)
            return offset, result, positions, batch_idx

        results = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=len(batches)) as executor:
            futures = {executor.submit(process_batch, (idx, batch)): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                future_idx = futures[future]
                try:
                    offset, result, positions, batch_idx = future.result()
                    results[batch_idx] = (offset, result, positions)
                    print(f"[Gemini OCR] Batch {batch_idx + 1}/{len(batches)} completed ({result.get('line_count', 0)} lines)")
                except Exception as e:
                    print(f"[Gemini OCR] Batch {future_idx + 1}/{len(batches)} failed: {type(e).__name__}: {e}")
                    results[future_idx] = (batches[future_idx][0], {'lines': [], 'error': str(e)}, [])

        # Merge results
        all_lines = []
        all_positions = []
        for offset, result, positions in results:
            for line in result.get('lines', []):
                line['cell_idx'] = line.get('cell_idx', 0) + offset
                all_lines.append(line)
            for pos in positions:
                pos['cell_idx'] = pos.get('cell_idx', 0) + offset
            all_positions.extend(positions)

        return {
            'lines': all_lines,
            'line_count': len(all_lines),
            'processing_time_ms': (time.time() - t0) * 1000,
            'translated': translate,
            'batch_count': len(batches)
        }, all_positions, None


# Module-level convenience functions
_gemini_ocr_instance = None


def get_gemini_ocr() -> Optional[GeminiOCR]:
    """Get singleton GeminiOCR instance."""
    global _gemini_ocr_instance
    if _gemini_ocr_instance is None and check_gemini_available():
        _gemini_ocr_instance = GeminiOCR()
    return _gemini_ocr_instance


def reset_gemini_ocr():
    """Reset Gemini OCR instance (for model switching)."""
    global _gemini_ocr_instance, _genai_client, _genai_types
    _gemini_ocr_instance = None
    _genai_client = None
    _genai_types = None
