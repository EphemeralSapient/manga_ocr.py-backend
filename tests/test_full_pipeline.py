#!/usr/bin/env python3
"""Test full pipeline: Detection -> OCR/Translate on pages 1-4"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
from workflow import detect_mode, create_session, detect_all, CROP_PADDING
from workflow.ocr_vlm import check_vlm_available, VlmOCR, create_ocr_grid
from workflow.ocr import map_ocr

def main():
    print("=" * 60)
    print("Full Pipeline Test: Detection -> OCR -> Translate")
    print("=" * 60)

    # Check VLM
    if not check_vlm_available():
        print("\n[ERROR] llama-cli not found!")
        return 1
    print("\n[OK] llama-cli available")

    # Load pages 1-4
    pages = []
    for i in [1, 2, 3, 4]:
        path = f"input/page_{i}.jpg"
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            pages.append((i, img))
            print(f"  Loaded: {path} ({img.width}x{img.height})")

    if not pages:
        print("[ERROR] No pages found")
        return 1

    # Load detector
    print("\n[1/4] Loading detector...")
    t0 = time.time()
    mode = detect_mode()
    session, mode = create_session(mode)
    print(f"  Detector loaded in {time.time()-t0:.1f}s")

    # Detect bubbles
    print("\n[2/4] Detecting speech bubbles...")
    t0 = time.time()
    images = [p[1] for p in pages]
    bubbles, detect_time = detect_all(session, images, mode, target_label=1)
    print(f"  Found {len(bubbles)} bubbles in {detect_time:.0f}ms")

    # Show per-page breakdown
    by_page = {}
    for b in bubbles:
        pid = b['page_idx']
        if pid not in by_page:
            by_page[pid] = []
        by_page[pid].append(b)

    for pid in sorted(by_page.keys()):
        page_num = pages[pid][0]
        print(f"    Page {page_num}: {len(by_page[pid])} bubbles")

    if not bubbles:
        print("[WARNING] No bubbles detected")
        return 0

    # Create grid
    print("\n[3/4] Creating OCR grid...")
    t0 = time.time()
    grid_img, positions, grid_info = create_ocr_grid(bubbles)
    print(f"  Grid: {grid_img.width}x{grid_img.height}, {grid_info['total_cells']} cells")
    print(f"  Created in {(time.time()-t0)*1000:.0f}ms")

    # Save grid
    os.makedirs("output", exist_ok=True)
    grid_img.save("output/test_grid_pages1-4.jpg")
    print(f"  Saved: output/test_grid_pages1-4.jpg")

    # Initialize OCR
    ocr = VlmOCR(max_cells_per_batch=8)  # Process 8 cells at a time

    # Run OCR (Japanese) - using batched processing
    print("\n[4a/4] Running OCR (Japanese) - batched...")
    t0 = time.time()
    ocr_result, ocr_positions, _ = ocr.run_batched(bubbles, translate=False)
    ocr_time = time.time() - t0
    batch_info = f" ({ocr_result.get('batch_count', 1)} batches)" if ocr_result.get('batch_count', 1) > 1 else ""
    print(f"  {ocr_result.get('line_count', 0)} lines in {ocr_time:.1f}s{batch_info}")

    if ocr_result.get('error'):
        print(f"  Error: {ocr_result['error']}")
    else:
        # Map to bubbles
        mapped = map_ocr(ocr_result, ocr_positions)
        print(f"\n  OCR Results by page:")
        for (page_idx, bubble_idx), texts in sorted(mapped.items()):
            page_num = pages[page_idx][0]
            for t in texts:
                text_preview = t['text'][:40] + "..." if len(t['text']) > 40 else t['text']
                print(f"    [Page {page_num}, Bubble {bubble_idx}]: {text_preview}")

    # Run OCR+Translate (English) - using batched processing
    print("\n[4b/4] Running OCR+Translate (English) - batched...")
    t0 = time.time()
    translate_result, trans_positions, _ = ocr.run_batched(bubbles, translate=True)
    translate_time = time.time() - t0
    batch_info = f" ({translate_result.get('batch_count', 1)} batches)" if translate_result.get('batch_count', 1) > 1 else ""
    print(f"  {translate_result.get('line_count', 0)} lines in {translate_time:.1f}s{batch_info}")

    if translate_result.get('error'):
        print(f"  Error: {translate_result['error']}")
    else:
        # Map to bubbles
        mapped = map_ocr(translate_result, trans_positions)
        print(f"\n  Translation Results by page:")
        for (page_idx, bubble_idx), texts in sorted(mapped.items()):
            page_num = pages[page_idx][0]
            for t in texts:
                text_preview = t['text'][:50] + "..." if len(t['text']) > 50 else t['text']
                print(f"    [Page {page_num}, Bubble {bubble_idx}]: {text_preview}")

    print("\n" + "=" * 60)
    print("Pipeline Test Complete!")
    print(f"  Detection: {detect_time:.0f}ms")
    print(f"  OCR: {ocr_time:.1f}s")
    print(f"  Translate: {translate_time:.1f}s")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
