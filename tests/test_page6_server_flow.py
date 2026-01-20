#!/usr/bin/env python3
"""
Debug page 6 following the EXACT server.py flow:
1. Detect L1 and L2
2. Grid bubbles
3. OCR on grid
4. Text segmentation on full page (parallel)
5. Translation
6. Rendering with page mask
"""

import os
import sys
import numpy as np
from PIL import Image
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from workflow import detect_mode, create_session, detect_all
from workflow import grid_bubbles, run_ocr_on_bubbles, map_ocr
from workflow import create_text_segmenter
from workflow import render_text_on_image
from workflow.ocr_vlm import create_ocr_grid

INPUT_PATH = "input/page_6.jpg"
OUTPUT_DIR = "output/debug_page6_server"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load image
    print(f"Loading {INPUT_PATH}...")
    img = Image.open(INPUT_PATH).convert("RGB")
    images = [img]  # Server processes list of images
    print(f"  Image size: {img.width}x{img.height}")

    # Initialize models
    print("\n" + "="*60)
    print("STEP 1: INITIALIZE MODELS")
    print("="*60)
    mode = detect_mode()
    session, mode = create_session(mode)
    text_seg = create_text_segmenter()
    print(f"  Detector mode: {mode}")

    # Detect L1 bubbles
    print("\n" + "="*60)
    print("STEP 2: DETECT BUBBLES")
    print("="*60)
    bubbles_l1, detect_time = detect_all(session, images, mode, target_label=1)
    bubbles_l2, detect_time_l2 = detect_all(session, images, mode, target_label=2)
    print(f"  L1 bubbles: {len(bubbles_l1)} in {detect_time:.0f}ms")
    print(f"  L2 regions: {len(bubbles_l2)} in {detect_time_l2:.0f}ms")

    for i, b in enumerate(bubbles_l1):
        print(f"    L1[{i}] page={b['page_idx']} idx={b['bubble_idx']} box={list(b['box'])}")

    # Create OCR grid (following server.py flow)
    print("\n" + "="*60)
    print("STEP 3: CREATE OCR GRID")
    print("="*60)

    # Use VLM OCR flow which creates grid
    grid_img, positions, grid_info = create_ocr_grid(bubbles_l1)
    print(f"  Grid size: {grid_img.width}x{grid_img.height}")
    print(f"  Grid info: {grid_info}")
    print(f"  Positions: {len(positions)} entries")

    # Save grid image
    grid_path = os.path.join(OUTPUT_DIR, "ocr_grid.jpg")
    grid_img.save(grid_path, quality=95)
    print(f"  Saved grid to {grid_path}")

    # Show position mapping
    print("\n  Position mapping (cell -> original):")
    for i, pos in enumerate(positions):
        print(f"    [{i}] key={pos.get('key')} bubble_box={pos.get('bubble_box')} grid_box={pos.get('grid_box')}")

    # Run OCR on bubbles (uses grid internally)
    print("\n" + "="*60)
    print("STEP 4: RUN OCR")
    print("="*60)
    ocr_result, ocr_positions, _ = run_ocr_on_bubbles(bubbles_l1, translate=False)
    print(f"  OCR returned {ocr_result.get('line_count', 0)} lines")

    print("\n  OCR lines:")
    for i, line in enumerate(ocr_result.get('lines', [])):
        text = line.get('text', '')[:60]
        cell_idx = line.get('cell_idx', 'N/A')
        print(f"    [{i}] cell_idx={cell_idx}: {text}")

    # Map OCR to bubbles
    print("\n" + "="*60)
    print("STEP 5: MAP OCR TO BUBBLES")
    print("="*60)
    bubble_texts = map_ocr(ocr_result, ocr_positions)
    print(f"  Mapped to {len(bubble_texts)} bubbles")

    for key, texts in sorted(bubble_texts.items()):
        page_idx, bubble_idx = key
        merged = "".join([t['text'] for t in texts])[:60]
        print(f"    page={page_idx} bubble={bubble_idx}: {merged}")

    # Build ocr_data structure (as server does)
    ocr_data = {}
    bubble_map = {(b['page_idx'], b['bubble_idx']): b for b in bubbles_l1}
    for key, text_items in bubble_texts.items():
        page_idx, bubble_idx = key
        b = bubble_map.get(key)
        if b:
            if page_idx not in ocr_data:
                ocr_data[page_idx] = []
            ocr_data[page_idx].append({
                'idx': bubble_idx,
                'bubble_box': list(b['box']),
                'texts': text_items
            })

    # Run text segmentation on FULL PAGE (as server does)
    print("\n" + "="*60)
    print("STEP 6: TEXT SEGMENTATION (on full page)")
    print("="*60)
    img_array = np.array(img)
    t0 = time.time()
    page_mask = text_seg(img_array, verbose=True)
    text_seg_ms = int((time.time() - t0) * 1000)
    print(f"  Mask shape: {page_mask.shape}")
    print(f"  Mask stats: min={page_mask.min()}, max={page_mask.max()}, mean={page_mask.mean():.2f}")
    print(f"  Time: {text_seg_ms}ms")

    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, "text_seg_mask.png")
    Image.fromarray(page_mask).save(mask_path)
    print(f"  Saved mask to {mask_path}")

    # Create fake translations for testing
    print("\n" + "="*60)
    print("STEP 7: CREATE TEST TRANSLATIONS")
    print("="*60)
    translation_data = {}
    for page_idx, page_bubbles in ocr_data.items():
        translation_data[page_idx] = {}
        for bubble in page_bubbles:
            idx = bubble['idx']
            translation_data[page_idx][idx] = f"Translation_{idx}"
            print(f"    page={page_idx} bubble={idx}: '{translation_data[page_idx][idx]}'")

    # Render (as server does)
    print("\n" + "="*60)
    print("STEP 8: RENDER")
    print("="*60)
    page_bubbles = ocr_data.get(0, [])
    page_translations = translation_data.get(0, {})

    print(f"  Rendering {len(page_bubbles)} bubbles with precomputed_mask")
    print(f"  Bubbles being rendered:")
    for bubble in page_bubbles:
        print(f"    idx={bubble['idx']} box={bubble['bubble_box']} texts={len(bubble['texts'])}")

    img_copy = img.copy()
    rendered = render_text_on_image(
        img_copy,
        page_bubbles,
        page_translations,
        use_inpaint=False,
        precomputed_mask=page_mask
    )

    # Save rendered output
    render_path = os.path.join(OUTPUT_DIR, "rendered.jpg")
    rendered.save(render_path, quality=95)
    print(f"  Saved rendered image to {render_path}")

    # Also create visualization showing bubble boxes on original
    from PIL import ImageDraw
    viz_img = img.copy()
    draw = ImageDraw.Draw(viz_img)
    for b in bubbles_l1:
        box = b['box']
        draw.rectangle(box, outline="green", width=2)
        draw.text((box[0], box[1]-15), f"L1:{b['bubble_idx']}", fill="green")
    for b in bubbles_l2:
        box = b['box']
        draw.rectangle(box, outline="blue", width=2)
        draw.text((box[0], box[1]-15), f"L2:{b['bubble_idx']}", fill="blue")

    viz_path = os.path.join(OUTPUT_DIR, "detection_viz.jpg")
    viz_img.save(viz_path, quality=95)
    print(f"  Saved detection visualization to {viz_path}")

    print("\n" + "="*60)
    print("DONE! Check output in:", OUTPUT_DIR)
    print("="*60)


if __name__ == '__main__':
    main()
