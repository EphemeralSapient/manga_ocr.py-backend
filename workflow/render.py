#!/usr/bin/env python3
"""
Text Rendering Module - Renders translated text onto manga images

Features:
- Dynamic font sizing based on bubble dimensions
- Binary search for optimal font fit
- Automatic text wrapping
- Skip [NO TEXT] markers (sound effects, etc.)
- Text segmentation for pixel-level text cleaning (L1 bubbles)
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont

# Font search paths
FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/System/Library/Fonts/Arial.ttf",       # macOS fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",            # Windows
]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font at given size, with fallbacks."""
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                continue
    return ImageFont.load_default()


def fit_text(text: str, box: List[int]) -> Tuple[List[str], ImageFont.FreeTypeFont, int, int]:
    """
    Fit text in bubble using binary search for optimal font size.

    Args:
        text: Text to fit
        box: Bounding box [x1, y1, x2, y2]

    Returns:
        (lines, font, start_x, start_y)
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    # Use percentage-based padding (5% of smaller dimension)
    padding = max(5, int(min(box_width, box_height) * 0.05))
    max_width = box_width - (padding * 2)
    max_height = box_height - (padding * 2)

    # Scale max font based on bubble size - larger bubbles get larger fonts
    # Use ~40% of bubble height as max font, capped between 12 and 72
    min_font = 10
    max_font = max(min_font, min(72, int(box_height * 0.4)))

    def wrap_and_measure(font_size: int) -> Tuple[List[str], int, int]:
        font = get_font(font_size)
        words = text.split()
        lines = []
        while words:
            line = words.pop(0)
            while words:
                test = f"{line} {words[0]}"
                bbox = font.getbbox(test)
                if bbox[2] - bbox[0] <= max_width:
                    line = test
                    words.pop(0)
                else:
                    break
            lines.append(line)
        line_height = font_size + 4
        total_height = len(lines) * line_height
        max_line_width = max((font.getbbox(l)[2] - font.getbbox(l)[0]) for l in lines) if lines else 0
        return lines, max_line_width, total_height

    # Binary search for largest font that fits
    best_lines, best_size = [text], min_font
    lo, hi = min_font, max_font
    while lo <= hi:
        mid = (lo + hi) // 2
        lines, w, h = wrap_and_measure(mid)
        if w <= max_width and h <= max_height:
            best_lines, best_size = lines, mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size == min_font:
        best_lines, _, _ = wrap_and_measure(min_font)

    font = get_font(best_size)
    line_height = best_size + 4
    total_height = len(best_lines) * line_height
    start_y = y1 + padding + (max_height - total_height) // 2
    start_x = x1 + padding
    return best_lines, font, start_x, start_y


def render_text_on_image(
    img: Image.Image,
    bubbles: List[Dict],
    translations: Dict[int, str],
    use_inpaint: bool = False,
    text_segmenter = None,
    precomputed_mask: np.ndarray = None,
    timing: Dict = None
) -> Image.Image:
    """
    Render translated text onto image.

    Args:
        img: PIL Image to render on
        bubbles: List of bubble dicts with 'idx', 'bubble_box', 'texts'
        translations: Dict mapping bubble idx to translated text
        use_inpaint: If True, L2 background was inpainted (informational only)
        text_segmenter: Optional TextSegmenter for pixel-level text cleaning (L1 bubbles) - deprecated, use precomputed_mask
        precomputed_mask: Pre-computed text segmentation mask for this page (faster)
        timing: Optional dict to accumulate timing stats

    Returns:
        Modified PIL Image
    """
    import time
    t_start = time.time()

    draw = ImageDraw.Draw(img)
    img_array = np.array(img)
    full_page_mask = precomputed_mask  # Use precomputed if available

    # Fallback: Run text segmentation if no precomputed mask (legacy path)
    t_seg_start = time.time()
    if full_page_mask is None and text_segmenter is not None and bubbles:
        full_page_mask = text_segmenter(img_array, verbose=False)
    t_seg = time.time() - t_seg_start

    if timing is not None:
        timing['text_seg_ms'] = timing.get('text_seg_ms', 0) + int(t_seg * 1000)
        timing['text_seg_calls'] = timing.get('text_seg_calls', 0) + (1 if (text_segmenter and bubbles and precomputed_mask is None) else 0)

    t_mask_start = time.time()
    for bubble in bubbles:
        idx = bubble["idx"]
        bubble_box = bubble["bubble_box"]
        texts = bubble.get("texts", [])

        # Skip bubbles with [NO TEXT] - don't white out or render
        translated = translations.get(idx, "")
        if translated == "[NO TEXT]":
            continue

        # Clean text in bubbles - use text_seg for pixel-level or bbox fallback
        # Note: L2 bubbles are rendered in a separate call with text_segmenter=None (skip clearing)
        bx1, by1, bx2, by2 = bubble_box
        img_h, img_w = img_array.shape[:2]

        if full_page_mask is not None:
            from scipy import ndimage

            # Check if we have precise line-level bboxes (OneOCR) vs cell-level (VLM)
            # OneOCR: multiple small bboxes per bubble, total area << bubble area
            # VLM: bbox roughly equals bubble area
            bubble_area = (bx2 - bx1) * (by2 - by1)
            text_bbox_area = 0
            valid_text_bboxes = []
            for text_item in texts:
                text_bbox = text_item.get("bbox")
                if text_bbox and len(text_bbox) == 4:
                    tx1, ty1, tx2, ty2 = text_bbox
                    if tx2 > tx1 and ty2 > ty1:
                        text_bbox_area += (tx2 - tx1) * (ty2 - ty1)
                        valid_text_bboxes.append(text_bbox)

            # Use line-level masking if text bboxes cover < 70% of bubble area (OneOCR precision)
            use_line_level = bubble_area > 0 and text_bbox_area < bubble_area * 0.7 and valid_text_bboxes

            if use_line_level:
                # OneOCR: Use precise line-level bboxes for masking
                for text_bbox in valid_text_bboxes:
                    tx1, ty1, tx2, ty2 = text_bbox
                    # Clamp to image bounds
                    tx1 = max(0, min(tx1, img_w - 1))
                    ty1 = max(0, min(ty1, img_h - 1))
                    tx2 = max(0, min(tx2, img_w))
                    ty2 = max(0, min(ty2, img_h))

                    if tx2 <= tx1 or ty2 <= ty1:
                        continue

                    # Extract mask region for this OCR line (NOT the full bubble)
                    mask_region = full_page_mask[ty1:ty2, tx1:tx2]

                    if np.any(mask_region > 5):
                        text_mask = mask_region > 5
                        final_mask = ndimage.binary_dilation(text_mask, iterations=2)
                        img_array[ty1:ty2, tx1:tx2][final_mask] = 255
            else:
                # VLM/other: Use bubble_box for masking (original behavior)
                mask_region = full_page_mask[by1:by2, bx1:bx2]

                if np.any(mask_region > 5):
                    text_mask = mask_region > 5
                    final_mask = ndimage.binary_dilation(text_mask, iterations=2)
                    img_array[by1:by2, bx1:bx2][final_mask] = 255

        elif text_segmenter is not None:
            # Fallback: bbox-based white fill (only if text_segmenter was provided but mask failed)
            margin = 3
            safe_bx1, safe_by1 = bx1 + margin, by1 + margin
            safe_bx2, safe_by2 = bx2 - margin, by2 - margin

            for text_item in texts:
                tx1, ty1, tx2, ty2 = text_item["bbox"]
                clipped = [max(tx1, safe_bx1), max(ty1, safe_by1),
                           min(tx2, safe_bx2), min(ty2, safe_by2)]
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    draw.rectangle(clipped, fill="white")
        # else: text_segmenter is None = L2 regions already inpainted, skip clearing

    t_mask = time.time() - t_mask_start
    if timing is not None:
        timing['mask_apply_ms'] = timing.get('mask_apply_ms', 0) + int(t_mask * 1000)

    # Convert back from array
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Render translated text
    t_text_start = time.time()
    for bubble in bubbles:
        idx = bubble["idx"]
        bubble_box = bubble["bubble_box"]

        translated = translations.get(idx, "")
        if translated == "[NO TEXT]" or not translated:
            continue

        lines, font, start_x, start_y = fit_text(translated, bubble_box)
        line_height = font.size + 4 if hasattr(font, 'size') else 12

        box_width = bubble_box[2] - bubble_box[0]
        for i, line in enumerate(lines):
            bbox = font.getbbox(line)
            text_width = bbox[2] - bbox[0]
            text_x = bubble_box[0] + (box_width - text_width) // 2
            text_y = start_y + i * line_height
            draw.text((text_x, text_y), line, fill="black", font=font)

    t_text = time.time() - t_text_start
    t_total = time.time() - t_start

    if timing is not None:
        timing['text_render_ms'] = timing.get('text_render_ms', 0) + int(t_text * 1000)
        timing['render_total_ms'] = timing.get('render_total_ms', 0) + int(t_total * 1000)
        timing['render_calls'] = timing.get('render_calls', 0) + 1

    return img


__all__ = ['render_text_on_image', 'fit_text', 'get_font']
