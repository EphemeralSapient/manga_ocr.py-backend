#!/usr/bin/env python3
"""
Test the new grid-based text_seg workflow on page 6.
"""

import os
import sys
import numpy as np
from PIL import Image
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from workflow import detect_mode, create_session, detect_all
from workflow import run_ocr_on_bubbles, map_ocr
from workflow import create_text_segmenter
from workflow import render_text_on_image

INPUT_PATH = "input/page_6.jpg"
OUTPUT_DIR = "output/debug_page6_grid_textseg"


def map_grid_mask_to_pages(grid_mask, positions, page_images):
    """Map a grid text_seg mask back to per-page masks."""
    page_masks = {}

    for pos in positions:
        page_idx, bubble_idx = pos['key']
        grid_box = pos['grid_box']
        bubble_box = pos['bubble_box']

        # Extract mask region from grid
        gx1, gy1, gx2, gy2 = grid_box
        mask_region = grid_mask[gy1:gy2, gx1:gx2]

        # Initialize page mask if not exists
        if page_idx not in page_masks:
            img = page_images[page_idx]
            page_masks[page_idx] = np.zeros((img.height, img.width), dtype=np.uint8)

        # Place mask region at bubble_box position
        bx1, by1, bx2, by2 = bubble_box
        mask_h, mask_w = mask_region.shape
        target_h, target_w = by2 - by1, bx2 - bx1

        # Resize mask if needed
        if mask_h != target_h or mask_w != target_w:
            mask_pil = Image.fromarray(mask_region)
            mask_pil = mask_pil.resize((target_w, target_h), Image.Resampling.NEAREST)
            mask_region = np.array(mask_pil)

        # Paste mask region onto page mask
        page_masks[page_idx][by1:by2, bx1:bx2] = np.maximum(
            page_masks[page_idx][by1:by2, bx1:bx2],
            mask_region
        )

    return page_masks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load image
    print(f"Loading {INPUT_PATH}...")
    img = Image.open(INPUT_PATH).convert("RGB")
    images = [img]
    print(f"  Image size: {img.width}x{img.height}")

    # Initialize
    print("\n" + "="*60)
    print("STEP 1: INITIALIZE")
    print("="*60)
    mode = detect_mode()
    session, mode = create_session(mode)
    text_seg = create_text_segmenter()

    # Detect
    print("\n" + "="*60)
    print("STEP 2: DETECT BUBBLES")
    print("="*60)
    bubbles, detect_time = detect_all(session, images, mode, target_label=1)
    print(f"  Found {len(bubbles)} L1 bubbles in {detect_time:.0f}ms")

    for i, b in enumerate(bubbles):
        print(f"    [{i}] box={list(b['box'])}")

    # Run text_seg on FULL PAGE for comparison
    print("\n" + "="*60)
    print("STEP 2.5: TEXT_SEG ON FULL PAGE (for comparison)")
    print("="*60)
    page_array = np.array(images[0])

    # Save full page BEFORE
    fullpage_before_path = os.path.join(OUTPUT_DIR, "fullpage_BEFORE_textseg.jpg")
    images[0].save(fullpage_before_path, quality=95)
    print(f"  Saved full page BEFORE to {fullpage_before_path}")

    # Run text_seg on full page
    fullpage_mask = text_seg(page_array, verbose=True)
    print(f"  Full page mask shape: {fullpage_mask.shape}")
    print(f"  Full page mask stats: min={fullpage_mask.min()}, max={fullpage_mask.max()}, mean={fullpage_mask.mean():.2f}")

    # Save full page mask
    fullpage_mask_path = os.path.join(OUTPUT_DIR, "fullpage_text_seg_mask.png")
    Image.fromarray(fullpage_mask).save(fullpage_mask_path)
    print(f"  Saved full page mask to {fullpage_mask_path}")

    # Save full page overlay (red = detected at threshold 5)
    fullpage_overlay = page_array.copy()
    fullpage_mask_bool = fullpage_mask > 5
    fullpage_overlay[fullpage_mask_bool] = [255, 0, 0]
    fullpage_overlay_path = os.path.join(OUTPUT_DIR, "fullpage_mask_OVERLAY.jpg")
    Image.fromarray(fullpage_overlay).save(fullpage_overlay_path, quality=95)
    print(f"  Saved full page overlay to {fullpage_overlay_path}")

    # OCR (may or may not create grid depending on backend)
    print("\n" + "="*60)
    print("STEP 3: OCR")
    print("="*60)
    ocr_result, ocr_positions, grid = run_ocr_on_bubbles(bubbles, translate=False)
    print(f"  OCR lines: {ocr_result.get('line_count', 0)}")
    print(f"  OCR positions from backend: {len(ocr_positions) if ocr_positions else 0} entries")
    if ocr_positions:
        for i, pos in enumerate(ocr_positions):
            print(f"    ocr_pos[{i}]: key={pos.get('key')} bubble_box={pos.get('bubble_box')}")

    # If no grid returned (e.g., Gemini API), create one for text_seg
    # IMPORTANT: Keep ocr_positions separate for OCR mapping, use textseg_positions for mask mapping
    if grid is None:
        from workflow.ocr_vlm import create_ocr_grid
        grid, textseg_positions, grid_info = create_ocr_grid(bubbles)
        print(f"  Created grid for text_seg: {grid.width}x{grid.height}")
    else:
        textseg_positions = ocr_positions  # Same positions if grid was returned
        print(f"  Grid size: {grid.width}x{grid.height}")

    # Use textseg_positions for grid operations, ocr_positions for OCR mapping
    positions = textseg_positions

    # Save grid
    grid_path = os.path.join(OUTPUT_DIR, "ocr_grid.jpg")
    grid.save(grid_path, quality=95)
    print(f"  Saved grid to {grid_path}")

    # Show positions
    print("\n  Position mapping:")
    for i, pos in enumerate(positions):
        print(f"    [{i}] key={pos['key']} grid_box={pos['grid_box']} bubble_box={pos['bubble_box']}")

    # Use FULL PAGE text_seg (not grid) - gives cleaner results without bubble edge detection
    print("\n" + "="*60)
    print("STEP 4: TEXT_SEG ON FULL PAGE")
    print("="*60)

    # Use the fullpage_mask we already computed in step 2.5
    page_mask = fullpage_mask
    print(f"  Using full page mask (already computed)")
    print(f"  Page mask shape: {page_mask.shape}")
    print(f"  Page mask stats: min={page_mask.min()}, max={page_mask.max()}, mean={page_mask.mean():.2f}")

    # Save page mask
    page_mask_path = os.path.join(OUTPUT_DIR, "page_mask_fullpage.png")
    Image.fromarray(page_mask).save(page_mask_path)
    print(f"  Saved page mask to {page_mask_path}")

    # Create page_masks dict for render
    page_masks = {0: page_mask}

    # Build OCR data and translations
    print("\n" + "="*60)
    print("STEP 5: BUILD OCR DATA & VERIFY MAPPING")
    print("="*60)

    # Show OCR result structure
    print(f"\n  OCR lines ({len(ocr_result.get('lines', []))} total):")
    for i, line in enumerate(ocr_result.get('lines', [])):
        cell_idx = line.get('cell_idx', '?')
        text = line.get('text', '')[:30]
        print(f"    line[{i}]: cell_idx={cell_idx}, text='{text}'")

    # Show OCR positions mapping
    print(f"\n  OCR positions ({len(ocr_positions)} total):")
    for i, pos in enumerate(ocr_positions):
        key = pos.get('key', '?')
        bbox = pos.get('bubble_box', [])
        print(f"    pos[{i}]: key={key}, bubble_box={bbox}")

    # Use ocr_positions (from OCR result) for mapping OCR to bubbles
    bubble_texts = map_ocr(ocr_result, ocr_positions)

    print(f"\n  Mapped bubble_texts:")
    for key, text_items in sorted(bubble_texts.items()):
        texts = [t['text'][:20] for t in text_items]
        print(f"    {key}: {texts}")

    bubble_map = {(b['page_idx'], b['bubble_idx']): b for b in bubbles}

    ocr_data = {}
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

    # Create fake translations
    print(f"\n  Final OCR data for rendering:")
    translations = {0: {}}
    for bubble in sorted(ocr_data.get(0, []), key=lambda x: x['idx']):
        ocr_text = "".join([t['text'] for t in bubble['texts']])[:30]
        translations[0][bubble['idx']] = f"Translation_{bubble['idx']}"
        print(f"    Bubble {bubble['idx']}: box={bubble['bubble_box']}, ocr='{ocr_text}'")

    # Render
    print("\n" + "="*60)
    print("STEP 7: RENDER")
    print("="*60)
    page_bubbles = ocr_data.get(0, [])
    page_translations = translations.get(0, {})

    print(f"  Rendering {len(page_bubbles)} bubbles")
    img_copy = img.copy()
    rendered = render_text_on_image(
        img_copy,
        page_bubbles,
        page_translations,
        use_inpaint=False,
        precomputed_mask=page_mask
    )

    # Save rendered
    render_path = os.path.join(OUTPUT_DIR, "rendered.jpg")
    rendered.save(render_path, quality=95)
    print(f"  Saved rendered to {render_path}")

    # Also save comparison visualization
    from PIL import ImageDraw
    viz_img = img.copy()
    draw = ImageDraw.Draw(viz_img)
    for b in bubbles:
        box = b['box']
        draw.rectangle(box, outline="green", width=2)
        draw.text((box[0], box[1]-15), f"L1:{b['bubble_idx']}", fill="green")
    viz_path = os.path.join(OUTPUT_DIR, "detection_viz.jpg")
    viz_img.save(viz_path, quality=95)
    print(f"  Saved detection viz to {viz_path}")

    print("\n" + "="*60)
    print("DONE! Check output in:", OUTPUT_DIR)
    print("="*60)


if __name__ == '__main__':
    main()
