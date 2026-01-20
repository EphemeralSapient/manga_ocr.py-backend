"""OCR Module - Grid creation, Windows OneOCR, VLM OCR (Qwen/etc), result mapping"""

import io
import os
import platform
import importlib.util
import requests
from PIL import Image
from collections import defaultdict

# Load Windows OneOCR module (importlib avoids name conflict with this file)
try:
    _spec = importlib.util.spec_from_file_location('oneocr', os.path.join(os.path.dirname(__file__), 'ocr', 'oneocr.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    OneOCR, check_ocr_available = _mod.OneOCR, _mod.check_ocr_available
    HAS_WINDOWS_OCR = check_ocr_available()
except:
    OneOCR, HAS_WINDOWS_OCR = None, False

# Load VLM OCR (Qwen, etc) via llama.cpp - for non-Windows or as alternative
try:
    from .ocr_vlm import VlmOCR, check_vlm_available, create_ocr_grid
    _HAS_VLM_MODULE = True
except:
    VlmOCR, check_vlm_available, create_ocr_grid = None, None, None
    _HAS_VLM_MODULE = False

# Load API-based OCR (Gemini, etc) - cloud alternatives
try:
    from .ocr_api import GeminiOCR, check_gemini_available, get_gemini_ocr, reset_gemini_ocr
    _HAS_GEMINI_MODULE = True
except:
    GeminiOCR, check_gemini_available, get_gemini_ocr, reset_gemini_ocr = None, None, None, None
    _HAS_GEMINI_MODULE = False

_local_ocr = None
_vlm_ocr = None
_gemini_ocr = None
_remote_ocr = None
_vlm_checked = False  # Track if we've done runtime check
_gemini_checked = False
_remote_checked = False
OCR_URL = "http://localhost:1377/ocr"


# ─────────────────────────────────────────────────────────────────────────────
# OneOCR Remote Client
# ─────────────────────────────────────────────────────────────────────────────

class OneOCRRemote:
    """Client for remote OneOCR server."""

    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        self._available = None

    def check_available(self):
        """Check if remote server is reachable."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            self._available = resp.status_code == 200
        except:
            self._available = False
        return self._available

    def run(self, image, quality=95):
        """Run OCR on a single image.

        Args:
            image: PIL Image
            quality: JPEG quality (1-100). Higher = better quality, larger file.
                     95 is a good balance for OCR (sharp text, ~3-5x smaller than PNG)
        """
        import base64
        buf = io.BytesIO()

        # Store image dimensions for coordinate normalization
        img_width, img_height = image.size

        # Convert to RGB if needed (JPEG doesn't support RGBA)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        # Use JPEG for smaller payload (much faster over network)
        # Quality 95 preserves text sharpness while being ~3-5x smaller than PNG
        image.save(buf, format='JPEG', quality=quality, optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = requests.post(
            f"{self.server_url}/ocr",
            json={"image": img_b64},
            timeout=60  # 60s for large grids with many bubbles
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get('success'):
            raise Exception(data.get('error', 'Unknown error'))

        # Convert remote format to local format
        lines = []
        for result in data.get('results', []):
            bbox = result.get('bbox', [0, 0, 0, 0])
            lines.append({
                'text': result.get('text', ''),
                'bbox': {
                    'x': bbox[0],
                    'y': bbox[1],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1],
                },
                'confidence': result.get('confidence', 1.0)
            })

        # Fix Windows DPI scaling issue: normalize coordinates if they exceed image bounds
        if lines:
            max_x = max(l['bbox']['x'] + l['bbox']['width'] for l in lines)
            max_y = max(l['bbox']['y'] + l['bbox']['height'] for l in lines)

            # Check if coordinates are scaled beyond image dimensions
            scale_x = max_x / img_width if max_x > img_width else 1.0
            scale_y = max_y / img_height if max_y > img_height else 1.0

            if scale_x > 1.01 or scale_y > 1.01:  # More than 1% over = DPI scaling
                print(f"[OCR] Fixing DPI scaling: coords scaled by {scale_x:.2f}x/{scale_y:.2f}x, normalizing...")
                for line in lines:
                    line['bbox']['x'] /= scale_x
                    line['bbox']['y'] /= scale_y
                    line['bbox']['width'] /= scale_x
                    line['bbox']['height'] /= scale_y

        return {'lines': lines, 'line_count': len(lines)}

    def run_batched(self, bubbles, row_width=600, padding=30, translate=False, max_height=600):
        """Run OCR on multiple bubbles using grid (like Windows OneOCR).

        Args:
            bubbles: List of bubble dicts with 'image' key
            row_width: Max width for grid rows (default 600)
            padding: Padding between cells (default 30)
            translate: Not supported for OneOCR (ignored)
            max_height: Max height for each grid batch (default 600)

        Returns:
            (ocr_result, positions, grid_image)
            Note: grid_image is the first batch only (for debugging)
        """
        if not bubbles:
            return {'lines': [], 'line_count': 0}, [], None

        all_lines = []
        all_positions = []
        first_grid_img = None
        batch_num = 0
        remaining = bubbles

        while remaining:
            batch_num += 1
            # Create grid from remaining bubbles (respecting max_height)
            grid_img, positions, remaining = grid_bubbles(remaining, row_width, padding, max_height)

            # Debug: save grid image for inspection
            try:
                debug_path = os.path.join(os.path.dirname(__file__), '..', 'output', f'debug_ocr_grid_batch{batch_num}.png')
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                grid_img.save(debug_path)
                print(f"[OCR] Debug: Batch {batch_num} grid ({grid_img.width}x{grid_img.height}), {len(positions)} bubbles, {len(remaining)} remaining")
            except Exception as e:
                print(f"[OCR] Debug: Could not save grid image: {e}")

            if first_grid_img is None:
                first_grid_img = grid_img

            # Run OCR on this batch's grid
            ocr_result = self.run(grid_img)

            # Debug: Check for out-of-bounds OCR coordinates
            out_of_bounds = []
            for line in ocr_result.get('lines', []):
                bbox = line.get('bbox', {})
                x2 = bbox.get('x', 0) + bbox.get('width', 0)
                y2 = bbox.get('y', 0) + bbox.get('height', 0)
                if x2 > grid_img.width or y2 > grid_img.height:
                    out_of_bounds.append({
                        'text': line.get('text', '')[:20],
                        'x2': x2, 'y2': y2
                    })
            if out_of_bounds:
                print(f"[OCR] WARNING: Batch {batch_num} has {len(out_of_bounds)} OCR lines outside grid ({grid_img.width}x{grid_img.height})")

            # Add batch_idx to lines and positions for correct matching
            for line in ocr_result.get('lines', []):
                line['batch_idx'] = batch_num
            for pos in positions:
                pos['batch_idx'] = batch_num

            # Accumulate results
            all_lines.extend(ocr_result.get('lines', []))
            all_positions.extend(positions)

        # Combine all batch results
        combined_result = {
            'lines': all_lines,
            'line_count': len(all_lines),
            'batches': batch_num
        }

        if batch_num > 1:
            print(f"[OCR] Processed {len(bubbles)} bubbles in {batch_num} batches")

        return combined_result, all_positions, first_grid_img


def _get_remote_ocr_url():
    """Get configured remote OneOCR server URL."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from config import get_oneocr_server_url
        return get_oneocr_server_url()
    except:
        return ""


def _check_remote_ocr_available():
    """Check if remote OneOCR is available. Raises error if configured but not working."""
    global _remote_checked, _remote_ocr

    if _remote_checked:
        return _remote_ocr is not None and _remote_ocr.check_available()

    url = _get_remote_ocr_url()
    if not url:
        return False

    # Verify server is reachable using config helper
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from config import verify_oneocr_server

        success, message, details = verify_oneocr_server(url)
        if not success:
            # Server is configured but not working - raise error
            raise RuntimeError(f"OneOCR server not available: {message}")

        print(f"[OCR] OneOCR Remote: {message}")
        _remote_ocr = OneOCRRemote(url)
        _remote_checked = True
        return True

    except ImportError:
        # Fallback to simple check if config module not available
        try:
            _remote_ocr = OneOCRRemote(url)
            available = _remote_ocr.check_available()
            if not available:
                raise RuntimeError(f"OneOCR server not available at {url}")
            _remote_checked = True
            return True
        except Exception as e:
            raise RuntimeError(f"OneOCR server failed: {e}")


def _get_remote_ocr():
    """Get remote OneOCR instance. Raises error if configured but not working."""
    global _remote_ocr
    if not _check_remote_ocr_available():
        return None
    return _remote_ocr


def reset_remote_ocr():
    """Reset remote OneOCR client."""
    global _remote_ocr, _remote_checked
    _remote_ocr = None
    _remote_checked = False


def reset_vlm_ocr():
    """Reset VLM OCR instance to allow model switching.

    Call this when changing VLM models in config to force re-initialization.
    """
    global _vlm_ocr, _vlm_checked
    if _vlm_ocr is not None:
        # Stop any running server process
        try:
            _vlm_ocr.stop_server()
        except Exception:
            pass
    _vlm_ocr = None
    _vlm_checked = False


def reset_api_ocr():
    """Reset API-based OCR instances (Gemini, etc)."""
    global _gemini_ocr, _gemini_checked
    _gemini_ocr = None
    _gemini_checked = False
    if _HAS_GEMINI_MODULE and reset_gemini_ocr:
        reset_gemini_ocr()


SKIP_CHARS = {'ノ', 'ー', '一', '|'}

# Determine which OCR backend to use
IS_WINDOWS = platform.system() == 'Windows'


def _check_vlm_available():
    """Check VLM OCR availability at runtime (server may start after import)."""
    global _vlm_checked
    if not _HAS_VLM_MODULE or check_vlm_available is None:
        return False
    # Re-check each time until server is available, then cache
    if not _vlm_checked:
        available = check_vlm_available()
        if available:
            _vlm_checked = True
        return available
    return True


# For backwards compatibility exports
HAS_LFM_OCR = _HAS_VLM_MODULE  # Module available (runtime check done in _get_vlm_ocr)
HAS_VLM_OCR = _HAS_VLM_MODULE
HAS_GEMINI_OCR = _HAS_GEMINI_MODULE
HAS_REMOTE_OCR = True  # Always available (just needs server URL configured)
HAS_LOCAL_OCR = HAS_WINDOWS_OCR if IS_WINDOWS else HAS_VLM_OCR


def _get_local_ocr():
    """Get Windows OneOCR instance."""
    global _local_ocr
    if not HAS_WINDOWS_OCR or OneOCR is None:
        return None
    if _local_ocr is None:
        try:
            _local_ocr = OneOCR()
        except:
            return None
    return _local_ocr


def _get_vlm_ocr():
    """Get VLM OCR instance (Qwen, etc via llama.cpp)."""
    global _vlm_ocr
    if not _HAS_VLM_MODULE or VlmOCR is None:
        return None
    # Check availability at runtime (server may have started after import)
    if not _check_vlm_available():
        return None
    if _vlm_ocr is None:
        try:
            _vlm_ocr = VlmOCR()
        except:
            return None
    return _vlm_ocr


def _check_gemini_available():
    """Check Gemini OCR availability at runtime."""
    global _gemini_checked
    if not _HAS_GEMINI_MODULE or check_gemini_available is None:
        return False
    if not _gemini_checked:
        available = check_gemini_available()
        if available:
            _gemini_checked = True
        return available
    return True


def _get_gemini_ocr():
    """Get Gemini OCR instance (Google AI Studio API)."""
    global _gemini_ocr
    if not _HAS_GEMINI_MODULE or GeminiOCR is None:
        return None
    if not _check_gemini_available():
        return None
    if _gemini_ocr is None:
        try:
            _gemini_ocr = GeminiOCR()
        except:
            return None
    return _gemini_ocr


def grid_bubbles(bubbles, row_width=600, padding=30, max_height=600):
    """Arrange bubble crops into a grid for batch OCR.

    Args:
        bubbles: List of bubble dicts with 'image' key
        row_width: Max width for grid (default 600)
        padding: Padding between cells (default 30)
        max_height: Max height for grid (default 600). If bubbles don't fit,
                    returns remaining bubbles for next batch.

    Returns:
        (grid_image, positions, remaining_bubbles)
        remaining_bubbles will be empty if all fit, otherwise contains bubbles for next batch
    """
    if not bubbles:
        return Image.new('RGB', (100, 100), (255, 255, 255)), [], []

    # First pass: organize into rows
    rows, row, row_w = [], [], padding
    for b in bubbles:
        w = b['image'].width
        if row_w + w + padding > row_width and row:
            rows.append(row)
            row, row_w = [], padding
        row.append(b)
        row_w += w + padding
    if row:
        rows.append(row)

    # Second pass: determine which rows fit within max_height
    rows_to_include = []
    current_h = padding
    remaining_rows = []

    for row in rows:
        row_h = max(b['image'].height for b in row) + padding
        if current_h + row_h <= max_height or not rows_to_include:
            # Always include at least one row, even if it exceeds max_height
            rows_to_include.append(row)
            current_h += row_h
        else:
            remaining_rows.append(row)

    # Flatten remaining rows back to bubbles
    remaining_bubbles = [b for row in remaining_rows for b in row]

    total_h = sum(max(b['image'].height for b in r) + padding for r in rows_to_include) + padding
    grid = Image.new('RGB', (row_width, total_h), (255, 255, 255))
    positions = []

    y = padding
    for row in rows_to_include:
        row_h = max(b['image'].height for b in row)
        x = padding
        for b in row:
            y_off = y + (row_h - b['image'].height) // 2
            grid.paste(b['image'], (x, y_off))
            positions.append({
                'key': (b['page_idx'], b['bubble_idx']),
                'grid_box': (x, y_off, x + b['image'].width, y_off + b['image'].height),
                'bubble_box': b['box'],
                'grid_offset': (x, y_off),
                'crop_offset': b.get('crop_offset', (0, 0))
            })
            x += b['image'].width + padding
        y += row_h + padding

    return grid.crop((0, 0, row_width, y)), positions, remaining_bubbles


def run_ocr(image, ocr_url=OCR_URL, positions=None, grid_info=None, translate=False):
    """Run OCR on image. Uses local OCR if available, falls back to HTTP.

    For Windows: Uses OneOCR DLL
    For other platforms: Uses VLM OCR via llama.cpp (Qwen, etc)
    Fallback: HTTP OCR service

    Args:
        image: PIL Image to OCR
        ocr_url: URL for HTTP OCR fallback
        positions: Position info from grid creation
        grid_info: Grid layout info
        translate: If True and using VLM, output English translations directly
    """
    # Try Windows OneOCR (doesn't support translate mode)
    if IS_WINDOWS and not translate:
        local = _get_local_ocr()
        if local:
            try:
                return local.run(image)
            except:
                pass

    # Try VLM OCR (Qwen, etc via llama.cpp)
    vlm = _get_vlm_ocr()
    if vlm:
        try:
            return vlm.run(image, positions=positions, grid_info=grid_info, translate=translate)
        except:
            pass

    # Fallback to HTTP OCR (doesn't support translate)
    if translate:
        return {'lines': [], 'line_count': 0, 'error': 'Translate mode requires VLM OCR', 'translated': False}

    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=95)
    return requests.post(ocr_url, data=buf.getvalue(),
                         headers={"Content-Type": "application/octet-stream"}).json()


def _get_configured_ocr_method():
    """Get the configured OCR method from config."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from config import get_ocr_method, is_api_ocr_method, is_remote_ocr_method
        method = get_ocr_method()
        return method, is_api_ocr_method(method), is_remote_ocr_method(method)
    except:
        return "qwen_vlm", False, False


def run_ocr_on_bubbles(bubbles, row_width=600, padding=30, ocr_url=OCR_URL, translate=False):
    """
    High-level OCR: create grid and run OCR in one call.

    For OneOCR Remote: Sends grid to remote Windows server.
    For Gemini API: Sends individual bubble images directly (no grid needed).
    For VLM OCR: Uses colored separator grid with batched processing.
    For Windows/HTTP: Uses standard grid.

    Args:
        bubbles: List of bubble dicts
        row_width: Max width for grid rows
        padding: Padding between cells
        ocr_url: URL for HTTP OCR fallback
        translate: If True and using VLM/API, output English translations directly

    Returns: (ocr_result, positions, grid_image)
    """
    ocr_method, is_api, is_remote = _get_configured_ocr_method()

    # Try OneOCR Remote if configured (doesn't support translate mode)
    if is_remote and ocr_method == "oneocr_remote" and not translate:
        remote = _get_remote_ocr()
        if remote:
            try:
                ocr_result, positions, grid_img = remote.run_batched(
                    bubbles, row_width=row_width, padding=padding, translate=translate
                )
                return ocr_result, positions, grid_img
            except Exception as e:
                # OneOCR Remote failed - raise error instead of falling back
                error_msg = f"[OCR] OneOCR Remote error: {e}"
                print(error_msg)
                raise RuntimeError(f"OneOCR Remote server failed: {e}. Check server logs for details. URL: {remote.server_url}")

    # Try Gemini API OCR if configured
    if is_api and ocr_method == "gemini_api":
        gemini = _get_gemini_ocr()
        if gemini:
            try:
                # Gemini sends individual images, no grid needed
                ocr_result, positions, _ = gemini.run_batched(
                    bubbles, row_width=row_width, padding=padding, translate=translate
                )
                return ocr_result, positions, None
            except Exception as e:
                print(f"[OCR] Gemini API error: {e}")
                # Fall through to try VLM or other backends

    # Use VLM OCR with batched processing (faster, handles many bubbles)
    vlm = _get_vlm_ocr()
    if vlm and create_ocr_grid is not None:
        try:
            # Use batched processing for better performance
            ocr_result, positions, grid_img = vlm.run_batched(
                bubbles, row_width=row_width, padding=padding, translate=translate
            )
            return ocr_result, positions, grid_img
        except Exception as e:
            print(f"[OCR] VLM error: {e}")
            pass

    # Translate mode requires VLM or API OCR
    if translate:
        grid_img, positions, _ = grid_bubbles(bubbles, row_width, padding)
        return {'lines': [], 'line_count': 0, 'error': 'Translate mode requires VLM or API OCR', 'translated': False}, positions, grid_img

    # Standard grid for Windows OCR or HTTP fallback
    grid_img, positions, _ = grid_bubbles(bubbles, row_width, padding)
    ocr_result = run_ocr(grid_img, ocr_url)
    return ocr_result, positions, grid_img


def map_ocr(ocr_result, positions, is_translated=None, verbose=False):
    """Map OCR text lines to bubbles with translated bboxes.

    Supports both:
    - VLM OCR output with cell_idx for direct mapping
    - Standard OCR output with bbox center-point matching

    Args:
        ocr_result: OCR result dict with 'lines'
        positions: Position info from grid creation
        is_translated: If True, skip Japanese character filtering (output is English).
                      If None, auto-detect from ocr_result['translated']
        verbose: If True, print detailed debug info about mapping failures
    """
    results = defaultdict(list)

    # Auto-detect if output is translated
    if is_translated is None:
        is_translated = ocr_result.get('translated', False)

    # Build cell_idx lookup for VLM output
    cell_to_pos = {i: pos for i, pos in enumerate(positions) if 'grid_box' in pos}

    # Debug counters
    stats = {
        'total_lines': 0,
        'skipped_empty': 0,
        'skipped_no_japanese': 0,
        'skipped_no_bbox': 0,
        'matched_cell_idx': 0,
        'matched_bbox': 0,
        'unmatched': 0
    }

    unmatched_lines = []

    for line in ocr_result.get('lines', []):
        stats['total_lines'] += 1
        text = line.get('text', '').strip()
        bbox = line.get('bbox', {})

        if not text or text in SKIP_CHARS:
            stats['skipped_empty'] += 1
            continue

        # Skip Japanese character filter if output is already translated to English
        if not is_translated:
            # Include: hiragana, katakana, CJK, and half-width katakana (U+FF65-U+FF9F)
            if not any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' or '\uff65' <= c <= '\uff9f' for c in text):
                stats['skipped_no_japanese'] += 1
                continue

        # Try direct cell_idx mapping (VLM OCR output)
        cell_idx = line.get('cell_idx')
        if cell_idx is not None and cell_idx in cell_to_pos:
            pos = cell_to_pos[cell_idx]
            bx, by = pos['bubble_box'][:2]
            gx1, gy1, gx2, gy2 = pos['grid_box']

            result_item = {
                'text': text,
                'bbox': [bx, by, bx + (gx2 - gx1), by + (gy2 - gy1)]
            }
            if is_translated:
                result_item['translated'] = True

            results[pos['key']].append(result_item)
            stats['matched_cell_idx'] += 1
            continue

        # Fallback: bbox matching with tolerance (standard OCR like OneOCR)
        if not bbox:
            stats['skipped_no_bbox'] += 1
            continue

        # Get OCR bbox bounds
        ocr_x1 = bbox.get('x', 0)
        ocr_y1 = bbox.get('y', 0)
        ocr_x2 = ocr_x1 + bbox.get('width', 0)
        ocr_y2 = ocr_y1 + bbox.get('height', 0)
        cx = (ocr_x1 + ocr_x2) / 2
        cy = (ocr_y1 + ocr_y2) / 2

        # Matching with small tolerance (DPI normalization isn't always perfect)
        # With 30px padding, 15px tolerance is safe and won't cross-match
        TOLERANCE = 15
        matched = False
        matched_pos = None

        # Get batch_idx from line (if batched OCR)
        line_batch = line.get('batch_idx')

        for pos in positions:
            # Only match within same batch (if batch info available)
            pos_batch = pos.get('batch_idx')
            if line_batch is not None and pos_batch is not None and line_batch != pos_batch:
                continue

            gx1, gy1, gx2, gy2 = pos['grid_box']

            # Center point inside grid_box (with small tolerance) = match
            if (gx1 - TOLERANCE) <= cx <= (gx2 + TOLERANCE) and (gy1 - TOLERANCE) <= cy <= (gy2 + TOLERANCE):
                matched_pos = pos
                break

        if matched_pos:
            pos = matched_pos
            ox, oy = pos['grid_offset']
            bx, by = pos['bubble_box'][:2]
            px, py = pos.get('crop_offset', (0, 0))
            results[pos['key']].append({
                'text': text,
                'bbox': [int(bbox['x'] - ox + bx - px), int(bbox['y'] - oy + by - py),
                         int(bbox['x'] - ox + bx - px + bbox['width']),
                         int(bbox['y'] - oy + by - py + bbox['height'])]
            })
            matched = True
            stats['matched_bbox'] += 1

        if not matched:
            stats['unmatched'] += 1
            # Find the closest grid_box for debugging
            closest_pos = None
            closest_dist = float('inf')
            for pos in positions:
                gx1, gy1, gx2, gy2 = pos['grid_box']
                # Distance from center to grid_box (0 if inside)
                x_dist = max(gx1 - cx, 0, cx - gx2)
                y_dist = max(gy1 - cy, 0, cy - gy2)
                dist = (x_dist ** 2 + y_dist ** 2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = pos

            unmatched_lines.append({
                'text': text[:30],
                'cx': cx, 'cy': cy,
                'bbox': bbox,
                'closest_key': closest_pos['key'] if closest_pos else None,
                'closest_dist': closest_dist,
                'closest_gbox': closest_pos['grid_box'] if closest_pos else None
            })

    if verbose:
        print(f"  [map_ocr] Stats: {stats}")
        if unmatched_lines:
            print(f"  [map_ocr] Unmatched Japanese lines ({len(unmatched_lines)}):")
            for ul in unmatched_lines[:10]:
                gbox = ul.get('closest_gbox', [])
                print(f"    text='{ul['text']}' center=({ul['cx']:.0f}, {ul['cy']:.0f}) closest={ul['closest_key']} dist={ul['closest_dist']:.0f}px gbox={gbox}")
            if len(unmatched_lines) > 10:
                print(f"    ... and {len(unmatched_lines) - 10} more")

    return dict(results)
