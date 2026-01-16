#!/usr/bin/env python3
"""
Text Rendering Module - Renders translated text onto manga images

Features:
- Dynamic font sizing based on bubble dimensions
- Binary search for optimal font fit
- Automatic text wrapping
- Skip [NO TEXT] markers (sound effects, etc.)
"""

import os
from typing import List, Dict, Tuple, Any
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
    translations: Dict[int, str]
) -> Image.Image:
    """
    Render translated text onto image.

    Args:
        img: PIL Image to render on
        bubbles: List of bubble dicts with 'idx', 'bubble_box', 'texts'
        translations: Dict mapping bubble idx to translated text

    Returns:
        Modified PIL Image
    """
    draw = ImageDraw.Draw(img)

    for bubble in bubbles:
        idx = bubble["idx"]
        bubble_box = bubble["bubble_box"]
        texts = bubble.get("texts", [])

        # Skip bubbles with [NO TEXT] - don't white out or render
        translated = translations.get(idx, "")
        if translated == "[NO TEXT]":
            continue

        # White out OCR text bboxes (clipped to bubble bounds)
        bx1, by1, bx2, by2 = bubble_box
        margin = 3
        safe_bx1, safe_by1 = bx1 + margin, by1 + margin
        safe_bx2, safe_by2 = bx2 - margin, by2 - margin

        for text_item in texts:
            tx1, ty1, tx2, ty2 = text_item["bbox"]
            clipped = [max(tx1, safe_bx1), max(ty1, safe_by1),
                       min(tx2, safe_bx2), min(ty2, safe_by2)]
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                draw.rectangle(clipped, fill="white")

        # Render translated text
        if translated:
            lines, font, start_x, start_y = fit_text(translated, bubble_box)
            line_height = font.size + 4 if hasattr(font, 'size') else 12

            box_width = bubble_box[2] - bubble_box[0]
            for i, line in enumerate(lines):
                bbox = font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                text_x = bubble_box[0] + (box_width - text_width) // 2
                text_y = start_y + i * line_height
                draw.text((text_x, text_y), line, fill="black", font=font)

    return img


__all__ = ['render_text_on_image', 'fit_text', 'get_font']
