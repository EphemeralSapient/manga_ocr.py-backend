#!/usr/bin/env python3
"""
Full server flow debug for page 6 - uses actual OCR to trace the complete pipeline.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from workflow import detect_mode, create_session, detect_all
from workflow import grid_bubbles, run_ocr_on_bubbles, map_ocr
from workflow import create_text_segmenter

INPUT_PATH = "input/page_6.jpg"
OUTPUT_DIR = "output/debug_page6_full"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load image
    print(f"Loading {INPUT_PATH}...")
    img = Image.open(INPUT_PATH).convert("RGB")
    print(f"  Image size: {img.width}x{img.height}")

    # Initialize detector
    print("\nInitializing detector...")
    mode = detect_mode()
    session, mode = create_session(mode)

    # Detect L1 bubbles
    print("\nDetecting L1 bubbles...")
    bubbles, detect_time = detect_all(session, [img], mode, target_label=1)
    print(f"  Found {len(bubbles)} L1 bubbles in {detect_time:.0f}ms")

    print("\n" + "="*60)
    print("DETECTED BUBBLES:")
    print("="*60)
    for i, b in enumerate(bubbles):
        box = b['box']
        crop_offset = b.get('crop_offset', (0,0))
        crop_img = b.get('image')
        crop_size = f"{crop_img.width}x{crop_img.height}" if crop_img else "N/A"
        print(f"  [{i}] page={b['page_idx']} idx={b['bubble_idx']} box={list(box)} crop_offset={crop_offset} crop_size={crop_size}")

    # Save bubble crops for inspection
    print("\n  Saving bubble crops...")
    for i, b in enumerate(bubbles):
        if b.get('image'):
            crop_path = os.path.join(OUTPUT_DIR, f"bubble_{i}_crop.jpg")
            b['image'].save(crop_path, quality=95)
            print(f"    Saved bubble {i} crop to {crop_path}")

    # Run OCR on bubbles
    print("\n" + "="*60)
    print("RUNNING OCR...")
    print("="*60)
    ocr_result, positions, grid_img = run_ocr_on_bubbles(bubbles, translate=False)

    print(f"\n  OCR returned {ocr_result.get('line_count', 0)} lines")
    print(f"  Positions: {len(positions)} entries")

    # Save grid if available
    if grid_img:
        grid_path = os.path.join(OUTPUT_DIR, "ocr_grid.jpg")
        grid_img.save(grid_path, quality=95)
        print(f"  Saved OCR grid to {grid_path}")

    # Show OCR lines
    print("\n  OCR Lines:")
    for i, line in enumerate(ocr_result.get('lines', [])):
        text = line.get('text', '')[:50]
        cell_idx = line.get('cell_idx', 'N/A')
        print(f"    [{i}] cell_idx={cell_idx}: {text}{'...' if len(line.get('text', '')) > 50 else ''}")

    # Map OCR to bubbles
    print("\n" + "="*60)
    print("MAP_OCR RESULTS:")
    print("="*60)
    bubble_texts = map_ocr(ocr_result, positions)

    print(f"  Mapped to {len(bubble_texts)} bubbles")
    for key, texts in sorted(bubble_texts.items()):
        page_idx, bubble_idx = key
        merged = "".join([t['text'] for t in texts])[:60]
        print(f"    page={page_idx} bubble={bubble_idx}: {merged}{'...' if len(merged) >= 60 else ''}")

    # Check which bubbles are MISSING from map_ocr
    print("\n" + "="*60)
    print("MISSING BUBBLES (detected but no OCR match):")
    print("="*60)
    bubble_keys = set(bubble_texts.keys())
    all_bubble_keys = {(b['page_idx'], b['bubble_idx']) for b in bubbles}
    missing = all_bubble_keys - bubble_keys

    if missing:
        for key in sorted(missing):
            page_idx, bubble_idx = key
            b = next(b for b in bubbles if b['page_idx'] == page_idx and b['bubble_idx'] == bubble_idx)
            box = b['box']
            print(f"  MISSING: page={page_idx} idx={bubble_idx} box={list(box)}")
    else:
        print("  None - all bubbles have OCR matches")

    # Save bubble visualization
    from PIL import ImageDraw, ImageFont
    viz_img = img.copy()
    draw = ImageDraw.Draw(viz_img)

    for b in bubbles:
        box = b['box']
        key = (b['page_idx'], b['bubble_idx'])
        color = "green" if key in bubble_texts else "red"
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]-15), f"L1:{b['bubble_idx']}", fill=color)

    viz_path = os.path.join(OUTPUT_DIR, "bubbles_ocr_status.jpg")
    viz_img.save(viz_path, quality=95)
    print(f"\n  Saved visualization to {viz_path}")
    print("  GREEN = has OCR match, RED = no OCR match (will be lost)")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
