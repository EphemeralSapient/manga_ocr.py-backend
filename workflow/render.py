#!/usr/bin/env python3
"""
Text Rendering Module - Renders translated text onto manga images using Skia

Features:
- Skia Paragraph API for fast auto-wrapping text layout
- Binary search for optimal font fit (using Skia's layout engine)
- Hardware-accelerated rendering
- Text segmentation for pixel-level text cleaning (L1 bubbles)
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image

import skia

# Font search paths
FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/System/Library/Fonts/Arial.ttf",       # macOS fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",            # Windows
]

# Cached font manager, font collection, and unicode handler
_font_mgr: Optional[skia.FontMgr] = None
_font_collection = None  # skia.textlayout_FontCollection
_unicode_handler = None  # skia.Unicode
_font_families = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
_skia_is_bgra: Optional[bool] = None  # Detected at runtime


def _detect_skia_color_format():
    """Detect if Skia N32 format is BGRA or RGBA on this platform."""
    global _skia_is_bgra

    if _skia_is_bgra is not None:
        return _skia_is_bgra

    # Create tiny test surface and draw pure red
    surface = skia.Surface.MakeRasterN32Premul(1, 1)
    canvas = surface.getCanvas()
    canvas.drawColor(skia.ColorRED)  # RGB: (255, 0, 0)

    arr = np.array(surface.makeImageSnapshot())
    # If BGRA: red = [0, 0, 255, 255] (B=0, G=0, R=255)
    # If RGBA: red = [255, 0, 0, 255] (R=255, G=0, B=0)
    _skia_is_bgra = arr[0, 0, 0] == 0 and arr[0, 0, 2] == 255

    return _skia_is_bgra


def _init_skia_fonts():
    """Initialize Skia font manager and unicode handler."""
    global _font_mgr, _font_collection, _unicode_handler

    if _font_mgr is not None:
        return

    _font_mgr = skia.FontMgr()
    _unicode_handler = skia.Unicode.ICU_Make()

    # Detect color format on first init
    _detect_skia_color_format()

    # Create font collection for paragraph layout
    _font_collection = skia.textlayout_FontCollection()
    _font_collection.setDefaultFontManager(_font_mgr)


def _measure_text_width(text: str, font_size: float) -> float:
    """Measure text width using Skia font."""
    _init_skia_fonts()

    # Use Skia Font for measurement
    typeface = None
    for family in _font_families:
        typeface = _font_mgr.matchFamilyStyle(family, skia.FontStyle())
        if typeface:
            break
    if not typeface:
        typeface = skia.Typeface.MakeDefault()

    font = skia.Font(typeface, font_size)
    return font.measureText(text)


def _check_all_words_fit(text: str, font_size: float, max_width: float) -> bool:
    """
    Check if all individual words fit within max_width at the given font size.

    Returns False if any single word is too wide (would cause mid-word breaks).
    Uses 95% of max_width as safety margin to account for measurement discrepancies.
    """
    # Use same 95% margin as _make_paragraph's wrapping
    safe_width = max_width * 0.95
    words = text.split()
    for word in words:
        if _measure_text_width(word, font_size) > safe_width:
            return False
    return True


def _wrap_text_by_words(text: str, font_size: float, max_width: float) -> str:
    """
    Wrap text by words (not characters) to fit within max_width.

    Returns text with newlines inserted at word boundaries.
    Long words that don't fit are kept intact on their own line.
    """
    words = text.split()
    if not words:
        return text.strip()

    lines = []
    current_line = words[0].strip()

    for word in words[1:]:
        word = word.strip()
        if not word:
            continue
        test_line = current_line + ' ' + word
        width = _measure_text_width(test_line, font_size)

        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word

    if current_line.strip():
        lines.append(current_line.strip())
    return '\n'.join(lines)


def detect_text_color(img_array: np.ndarray, bbox: List[int], text_mask: np.ndarray = None, threshold: int = 128) -> bool:
    """
    Detect if we should use white text based on background brightness.

    Uses perceived luminance formula for accurate brightness detection.
    White text on dark backgrounds, black text on light/medium backgrounds.

    Args:
        img_array: Image as numpy array (H, W, C)
        bbox: [x1, y1, x2, y2] bounding box
        text_mask: Text segmentation mask (high values = text pixels)
        threshold: Brightness threshold (default 128) - below this uses white text

    Returns:
        True if background is dark (use white text), False if background is light (use black text)
    """
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False  # Default to black text

    region = img_array[y1:y2, x1:x2]

    if text_mask is not None:
        mask_region = text_mask[y1:y2, x1:x2]
        # Sample BACKGROUND pixels (where mask is LOW - not text)
        background_mask = mask_region <= 5  # Non-text pixels

        if not np.any(background_mask):
            return False  # No background pixels found, default to black text

        background_pixels = region[background_mask]
    else:
        # No mask - sample edge pixels as background approximation
        top = region[0, :]
        bottom = region[-1, :]
        left = region[:, 0]
        right = region[:, -1]
        background_pixels = np.concatenate([
            top.reshape(-1, region.shape[-1]) if len(region.shape) == 3 else top.flatten(),
            bottom.reshape(-1, region.shape[-1]) if len(region.shape) == 3 else bottom.flatten(),
            left.reshape(-1, region.shape[-1]) if len(region.shape) == 3 else left.flatten(),
            right.reshape(-1, region.shape[-1]) if len(region.shape) == 3 else right.flatten()
        ])

    if background_pixels.size == 0:
        return False

    # Calculate perceived brightness
    if len(background_pixels.shape) == 1:
        # Grayscale
        avg_brightness = np.mean(background_pixels)
    else:
        # RGB - use perceived luminance formula: 0.299*R + 0.587*G + 0.114*B
        if background_pixels.shape[-1] >= 3:
            luminance = (0.299 * background_pixels[..., 0] +
                        0.587 * background_pixels[..., 1] +
                        0.114 * background_pixels[..., 2])
            avg_brightness = np.mean(luminance)
        else:
            avg_brightness = np.mean(background_pixels)

    # Use white text only if background is truly dark (below threshold)
    # Default threshold 128 = middle gray
    return avg_brightness < threshold


def _get_longest_word_width(text: str, font_size: float) -> float:
    """Get the width of the longest word in the text."""
    words = text.split()
    if not words:
        return 0
    return max(_measure_text_width(word, font_size) for word in words)


def _make_paragraph(text: str, font_size: float, max_width: float, center: bool = True, use_white: bool = False):
    """Create a Skia paragraph with word-level wrapping."""
    _init_skia_fonts()

    # Ensure max_width is at least as wide as the longest word (prevents Skia from breaking words)
    longest_word = _get_longest_word_width(text, font_size)
    effective_width = max(max_width, longest_word + 2)  # +2 for safety margin

    # Pre-wrap text by words
    wrap_width = effective_width * 0.95
    wrapped_text = _wrap_text_by_words(text, font_size, wrap_width)

    # Text style (using textlayout module)
    text_style = skia.textlayout_TextStyle()
    text_color = skia.ColorWHITE if use_white else skia.ColorBLACK
    text_style.setForegroundColor(skia.Paint(text_color))
    text_style.setFontSize(font_size)
    text_style.setFontFamilies(_font_families)

    # Paragraph style
    para_style = skia.textlayout_ParagraphStyle()
    if center:
        para_style.setTextAlign(skia.textlayout_TextAlign.kCenter)
    para_style.setTextStyle(text_style)

    # Build paragraph (requires unicode handler)
    builder = skia.textlayout_ParagraphBuilder.make(para_style, _font_collection, _unicode_handler)
    builder.pushStyle(text_style)
    builder.addText(wrapped_text)
    builder.pop()

    para = builder.Build()
    para.layout(effective_width)  # Use effective_width to prevent word breaking
    return para


def check_bbox_edges_white(img_array: np.ndarray, bbox: List[int], threshold: int = 240) -> bool:
    """
    Check if all edges of a bbox are white (for skipping text_seg).

    Args:
        img_array: Image as numpy array (H, W, C)
        bbox: [x1, y1, x2, y2] bounding box
        threshold: Minimum value to consider "white" (default 240)

    Returns:
        True if all edge pixels are white, False otherwise
    """
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    # Sample edge pixels (top, bottom, left, right edges)
    edges = []

    # Top edge
    if y1 < h:
        edges.append(img_array[y1, x1:x2])
    # Bottom edge
    if y2 - 1 >= 0 and y2 - 1 < h:
        edges.append(img_array[y2 - 1, x1:x2])
    # Left edge
    if x1 < w:
        edges.append(img_array[y1:y2, x1])
    # Right edge
    if x2 - 1 >= 0 and x2 - 1 < w:
        edges.append(img_array[y1:y2, x2 - 1])

    if not edges:
        return False

    # Check if all edge pixels are white (all channels >= threshold)
    for edge in edges:
        if edge.size == 0:
            continue
        # Handle both RGB and grayscale
        if len(edge.shape) == 1:
            if np.any(edge < threshold):
                return False
        else:
            # For RGB, check if all channels are white
            if np.any(np.min(edge, axis=-1) < threshold):
                return False

    return True


def check_bubble_background_white(img_array: np.ndarray, bbox: List[int],
                                   text_mask: np.ndarray = None, threshold: int = 240) -> bool:
    """
    Check if the background (non-text) pixels in a bubble region are white.

    Args:
        img_array: Image as numpy array (H, W, C)
        bbox: [x1, y1, x2, y2] bounding box
        text_mask: Text segmentation mask for the region (text pixels have high values)
        threshold: Minimum value to consider "white" (default 240)

    Returns:
        True if background is white, False if colored background detected
    """
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return True

    region = img_array[y1:y2, x1:x2]

    if text_mask is not None:
        # Background = pixels where text_mask is low (not text)
        mask_region = text_mask[y1:y2, x1:x2]
        background_mask = mask_region <= 5  # Non-text pixels

        if not np.any(background_mask):
            return True  # No background pixels found

        # Check background pixels
        background_pixels = region[background_mask]
    else:
        # No mask, check entire region
        background_pixels = region.reshape(-1, region.shape[-1]) if len(region.shape) == 3 else region.flatten()

    if background_pixels.size == 0:
        return True

    # Check if background is white (all channels >= threshold)
    if len(background_pixels.shape) == 1:
        return np.all(background_pixels >= threshold)
    else:
        return np.all(np.min(background_pixels, axis=-1) >= threshold)


def fit_text_skia(text: str, box: List[int], use_white: bool = False) -> Tuple[any, int, int, int]:
    """
    Fit text in bubble using Skia's paragraph layout with binary search.

    Args:
        text: Text to fit
        box: Bounding box [x1, y1, x2, y2]
        use_white: If True, render text in white (for dark backgrounds)

    Returns:
        (paragraph, font_size, start_x, start_y)
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    # Padding (5% of smaller dimension, min 5px)
    padding = max(5, int(min(box_width, box_height) * 0.05))
    max_width = box_width - (padding * 2)
    max_height = box_height - (padding * 2)

    # Font size range
    min_font = 10  # Minimum readable font
    max_font = max(min_font, min(72, int(box_height * 0.4)))

    # Binary search for largest font that fits
    best_para = None
    best_size = min_font

    lo, hi = min_font, max_font
    while lo <= hi:
        mid = (lo + hi) // 2

        # Check if all words fit at this font size (prevents mid-word breaks)
        if not _check_all_words_fit(text, mid, max_width):
            hi = mid - 1
            continue

        para = _make_paragraph(text, mid, max_width, use_white=use_white)

        # Check if it fits
        if para.Height <= max_height and para.LongestLine <= max_width:
            best_para = para
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Fallback: if no font fits with word constraints, use min_font anyway
    # Words stay intact, may overflow slightly - readability > perfect fit
    if best_para is None:
        best_para = _make_paragraph(text, min_font, max_width, use_white=use_white)
        best_size = min_font

    # Calculate position (centered vertically)
    para_height = best_para.Height
    start_x = x1 + padding
    start_y = y1 + padding + (max_height - para_height) / 2

    return best_para, best_size, int(start_x), int(start_y)


def render_text_on_image(
    img: Image.Image,
    bubbles: List[Dict],
    translations: Dict[int, str],
    use_inpaint: bool = False,
    text_segmenter=None,
    precomputed_mask: np.ndarray = None,
    timing: Dict = None,
    skip_clear_indices: set = None,
    prefer_mask: bool = False,
    force_white_text: bool = False
) -> Image.Image:
    """
    Render translated text onto image using Skia.

    Args:
        img: PIL Image to render on
        bubbles: List of bubble dicts with 'idx', 'bubble_box', 'texts'
        translations: Dict mapping bubble idx to translated text
        use_inpaint: If True, L2 background was inpainted (informational only)
        text_segmenter: Optional TextSegmenter for pixel-level text cleaning - deprecated, use precomputed_mask
        precomputed_mask: Pre-computed text segmentation mask for this page (faster)
        timing: Optional dict to accumulate timing stats
        skip_clear_indices: Set of bubble indices to skip text clearing (already AOT inpainted)
        prefer_mask: If True, prefer text_seg mask over char bboxes even when char bboxes available
        force_white_text: If True, always render text in white (for white-filled L2 regions)

    Returns:
        Modified PIL Image
    """
    import time
    t_start = time.time()

    img_array = np.array(img)
    img_h, img_w = img_array.shape[:2]
    full_page_mask = precomputed_mask

    # Save original for text color detection (before mask application)
    original_img_array = img_array.copy()

    # Fallback: Run text segmentation if no precomputed mask (legacy path)
    t_seg_start = time.time()
    if full_page_mask is None and text_segmenter is not None and bubbles:
        full_page_mask = text_segmenter(img_array, verbose=False)
    t_seg = time.time() - t_seg_start

    if timing is not None:
        timing['text_seg_ms'] = timing.get('text_seg_ms', 0) + int(t_seg * 1000)
        timing['text_seg_calls'] = timing.get('text_seg_calls', 0) + (1 if (text_segmenter and bubbles and precomputed_mask is None) else 0)

    # Detect text color for each bubble (before clearing)
    bubble_use_white = {}  # idx -> bool (True if should use white text)
    for bubble in bubbles:
        idx = bubble["idx"]
        bubble_box = bubble["bubble_box"]
        # Detect if original text is white (light on dark background)
        use_white = detect_text_color(original_img_array, bubble_box, full_page_mask)
        bubble_use_white[idx] = use_white

    # Track white text detection stats (no per-page logging)

    # Apply text masks (clear original text)
    t_mask_start = time.time()
    text_seg_skipped = []  # Track bubble indices where text_seg was skipped
    text_seg_used = []  # Track bubble indices where text_seg was used
    total_line_bboxes = 0  # Total line-level bboxes processed

    for bubble in bubbles:
        idx = bubble["idx"]
        bubble_box = bubble["bubble_box"]
        texts = bubble.get("texts", [])

        translated = translations.get(idx, "")
        # Skip text clearing if no translation or failed - leave original untouched
        if not translated or translated.startswith("["):
            continue

        bx1, by1, bx2, by2 = bubble_box
        from scipy import ndimage

        # Collect line-level and char-level bboxes from OCR results
        bubble_area = (bx2 - bx1) * (by2 - by1)
        text_bbox_area = 0
        valid_text_bboxes = []  # (line_bbox, char_bboxes or None)
        has_any_char_bboxes = False

        for text_item in texts:
            text_bbox = text_item.get("bbox")
            if text_bbox and len(text_bbox) == 4:
                tx1, ty1, tx2, ty2 = text_bbox
                if tx2 > tx1 and ty2 > ty1:
                    text_bbox_area += (tx2 - tx1) * (ty2 - ty1)
                    # Get char bboxes if available (from OneOCR)
                    chars = text_item.get("chars", [])
                    char_bboxes = [c.get("bbox") for c in chars if c.get("bbox") and len(c.get("bbox")) == 4]
                    if char_bboxes:
                        has_any_char_bboxes = True
                    valid_text_bboxes.append((text_bbox, char_bboxes if char_bboxes else None))

        use_line_level = bubble_area > 0 and text_bbox_area < bubble_area * 0.7 and valid_text_bboxes

        # OneOCR with char bboxes: use char bboxes directly (skip text_seg)
        # VLM or no char bboxes: use text_seg mask as fallback
        # If prefer_mask is set, use text_seg mask even when char bboxes available
        if use_line_level and has_any_char_bboxes and not prefer_mask:
            # OneOCR mode: use character bboxes for precise text clearing
            bubble_skipped = False
            for text_bbox, char_bboxes in valid_text_bboxes:
                tx1, ty1, tx2, ty2 = text_bbox
                tx1 = max(0, min(tx1, img_w - 1))
                ty1 = max(0, min(ty1, img_h - 1))
                tx2 = max(0, min(tx2, img_w))
                ty2 = max(0, min(ty2, img_h))

                if tx2 <= tx1 or ty2 <= ty1:
                    continue

                total_line_bboxes += 1

                if char_bboxes:
                    # Have char bboxes: fill ONLY char bbox areas (not entire line)
                    is_white_bg = check_bbox_edges_white(img_array, [tx1, ty1, tx2, ty2])

                    for cb in char_bboxes:
                        cx1, cy1, cx2, cy2 = cb
                        # Clamp to image bounds
                        cx1 = max(0, min(cx1, img_w - 1))
                        cy1 = max(0, min(cy1, img_h - 1))
                        cx2 = max(0, min(cx2, img_w))
                        cy2 = max(0, min(cy2, img_h))

                        if cx2 <= cx1 or cy2 <= cy1:
                            continue

                        if is_white_bg:
                            # White background: fill char bbox with white
                            img_array[cy1:cy2, cx1:cx2] = 255
                        else:
                            # Colored background: sample bg from char bbox edges and fill
                            char_region = img_array[cy1:cy2, cx1:cx2]
                            ch, cw = char_region.shape[:2]
                            if ch > 2 and cw > 2:
                                edge_pixels = np.concatenate([
                                    char_region[0, :].reshape(-1, 3),
                                    char_region[-1, :].reshape(-1, 3),
                                    char_region[1:-1, 0].reshape(-1, 3),
                                    char_region[1:-1, -1].reshape(-1, 3),
                                ])
                                bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)
                            else:
                                bg_color = np.array([255, 255, 255], dtype=np.uint8)
                            img_array[cy1:cy2, cx1:cx2] = bg_color

                    bubble_skipped = is_white_bg
                else:
                    # No char bboxes for this line - fall back to filling entire line bbox
                    if check_bbox_edges_white(img_array, [tx1, ty1, tx2, ty2]):
                        img_array[ty1:ty2, tx1:tx2] = 255
                        bubble_skipped = True
                    else:
                        region = img_array[ty1:ty2, tx1:tx2]
                        h, w = region.shape[:2]
                        if h > 2 and w > 2:
                            edge_pixels = np.concatenate([
                                region[0, :].reshape(-1, 3),
                                region[-1, :].reshape(-1, 3),
                                region[1:-1, 0].reshape(-1, 3),
                                region[1:-1, -1].reshape(-1, 3),
                            ])
                            bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)
                            img_array[ty1:ty2, tx1:tx2] = bg_color

            if bubble_skipped:
                text_seg_skipped.append(idx)

        elif full_page_mask is not None:
            # VLM or no char bboxes: use text_seg mask
            if use_line_level:
                bubble_skipped = False
                bubble_used_seg = False
                for text_bbox, _ in valid_text_bboxes:
                    tx1, ty1, tx2, ty2 = text_bbox
                    tx1 = max(0, min(tx1, img_w - 1))
                    ty1 = max(0, min(ty1, img_h - 1))
                    tx2 = max(0, min(tx2, img_w))
                    ty2 = max(0, min(ty2, img_h))

                    if tx2 <= tx1 or ty2 <= ty1:
                        continue

                    total_line_bboxes += 1

                    if check_bbox_edges_white(img_array, [tx1, ty1, tx2, ty2]):
                        img_array[ty1:ty2, tx1:tx2] = 255
                        bubble_skipped = True
                    else:
                        region = img_array[ty1:ty2, tx1:tx2]
                        mask_region = full_page_mask[ty1:ty2, tx1:tx2]
                        if np.any(mask_region > 5):
                            text_mask = mask_region > 5
                            bg_mask = ~text_mask
                            if np.any(bg_mask):
                                bg_color = np.median(region[bg_mask], axis=0).astype(np.uint8)
                            else:
                                bg_color = np.array([255, 255, 255], dtype=np.uint8)
                            final_mask = ndimage.binary_dilation(text_mask, iterations=2)
                            img_array[ty1:ty2, tx1:tx2][final_mask] = bg_color
                        bubble_used_seg = True

                if bubble_skipped and not bubble_used_seg:
                    text_seg_skipped.append(idx)
                elif bubble_used_seg:
                    text_seg_used.append(idx)
            else:
                # Cell-level bbox (VLM): use text_seg on whole bubble
                mask_region = full_page_mask[by1:by2, bx1:bx2]
                if np.any(mask_region > 5):
                    text_mask = mask_region > 5
                    bg_mask = ~text_mask
                    region = img_array[by1:by2, bx1:bx2]
                    if np.any(bg_mask):
                        bg_color = np.median(region[bg_mask], axis=0).astype(np.uint8)
                    else:
                        bg_color = np.array([255, 255, 255], dtype=np.uint8)
                    final_mask = ndimage.binary_dilation(text_mask, iterations=2)
                    img_array[by1:by2, bx1:bx2][final_mask] = bg_color

        elif valid_text_bboxes:
            # No text_seg mask and no char bboxes - use line bboxes directly
            for text_bbox, _ in valid_text_bboxes:
                tx1, ty1, tx2, ty2 = text_bbox
                tx1 = max(0, min(tx1, img_w - 1))
                ty1 = max(0, min(ty1, img_h - 1))
                tx2 = max(0, min(tx2, img_w))
                ty2 = max(0, min(ty2, img_h))

                if tx2 <= tx1 or ty2 <= ty1:
                    continue

                total_line_bboxes += 1

                if check_bbox_edges_white(img_array, [tx1, ty1, tx2, ty2]):
                    img_array[ty1:ty2, tx1:tx2] = 255
                else:
                    region = img_array[ty1:ty2, tx1:tx2]
                    h, w = region.shape[:2]
                    if h > 2 and w > 2:
                        edge_pixels = np.concatenate([
                            region[0, :].reshape(-1, 3),
                            region[-1, :].reshape(-1, 3),
                            region[1:-1, 0].reshape(-1, 3),
                            region[1:-1, -1].reshape(-1, 3),
                        ])
                        bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)
                        img_array[ty1:ty2, tx1:tx2] = bg_color

    t_mask = time.time() - t_mask_start
    if timing is not None:
        timing['mask_apply_ms'] = timing.get('mask_apply_ms', 0) + int(t_mask * 1000)

    # text_seg stats are tracked in timing dict (no per-page logging)

    # Create Skia surface from image array
    t_text_start = time.time()

    # Ensure RGBA for Skia
    if img_array.shape[2] == 3:
        img_rgba = np.concatenate([img_array, np.full((*img_array.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
    else:
        img_rgba = img_array

    # Create Skia surface
    surface = skia.Surface.MakeRasterN32Premul(img_w, img_h)
    canvas = surface.getCanvas()

    # Draw image onto canvas
    skia_image = skia.Image.fromarray(img_rgba)
    canvas.drawImage(skia_image, 0, 0)

    # Render translated text
    for bubble in bubbles:
        idx = bubble["idx"]
        bubble_box = bubble["bubble_box"]

        translated = translations.get(idx, "")
        # Skip rendering if no translation or any failure marker
        if not translated or translated.startswith("["):
            continue

        # Use white text if original text was white (dark background) or if forced
        use_white = force_white_text or bubble_use_white.get(idx, False)
        para, font_size, start_x, start_y = fit_text_skia(translated, bubble_box, use_white=use_white)
        para.paint(canvas, start_x, start_y)

    # Convert back to PIL
    skia_image_out = surface.makeImageSnapshot()
    result_array = np.array(skia_image_out)

    # Convert to RGB (handle platform-specific BGRA vs RGBA)
    if result_array.shape[2] == 4:
        if _skia_is_bgra:
            # BGRA -> RGB: swap B and R, drop alpha
            result_array = result_array[:, :, [2, 1, 0]]
        else:
            # RGBA -> RGB: just drop alpha
            result_array = result_array[:, :, :3]

    img = Image.fromarray(result_array)

    t_text = time.time() - t_text_start
    t_total = time.time() - t_start

    if timing is not None:
        timing['text_render_ms'] = timing.get('text_render_ms', 0) + int(t_text * 1000)
        timing['render_total_ms'] = timing.get('render_total_ms', 0) + int(t_total * 1000)
        timing['render_calls'] = timing.get('render_calls', 0) + 1

    return img


# Legacy PIL-based function for compatibility
def fit_text(text: str, box: List[int]) -> Tuple[List[str], any, int, int]:
    """Legacy wrapper - converts Skia result to PIL-compatible format."""
    para, font_size, start_x, start_y = fit_text_skia(text, box)
    # Extract lines from paragraph (approximation)
    lines = text.split('\n') if '\n' in text else [text]
    return lines, None, start_x, start_y


def get_font(size: int):
    """Legacy function - returns None (Skia handles fonts internally)."""
    return None


def find_l1_bubbles_needing_aot(
    img_array: np.ndarray,
    bubbles: List[Dict],
    text_seg_mask: np.ndarray = None,
    threshold: int = 240
) -> List[Dict]:
    """
    Find L1 bubbles that need AOT inpainting due to non-white background.

    Args:
        img_array: Image as numpy array (H, W, C)
        bubbles: List of bubble dicts with 'bubble_box'
        text_seg_mask: Text segmentation mask (high values = text pixels)
        threshold: Minimum value to consider "white" (default 240)

    Returns:
        List of bubbles that need AOT inpainting (colored background detected)
    """
    need_aot = []

    for bubble in bubbles:
        bubble_box = bubble.get("bubble_box")
        if not bubble_box:
            continue

        x1, y1, x2, y2 = bubble_box

        # Check if background is white
        if not check_bubble_background_white(img_array, [x1, y1, x2, y2], text_seg_mask, threshold):
            need_aot.append(bubble)

    return need_aot


__all__ = [
    'render_text_on_image', 'fit_text', 'fit_text_skia', 'get_font',
    'check_bbox_edges_white', 'check_bubble_background_white', 'find_l1_bubbles_needing_aot',
    'detect_text_color'
]
