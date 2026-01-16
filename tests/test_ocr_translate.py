#!/usr/bin/env python3
"""Quick test for OCR+Translate feature using VLM."""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
from workflow.ocr_vlm import check_vlm_available, VlmOCR, create_ocr_grid
from workflow import detect_mode, create_session, detect_all, CROP_PADDING

def main():
    print("=" * 50)
    print("OCR+Translate Test")
    print("=" * 50)

    # Check VLM availability
    if not check_vlm_available():
        print("\n[ERROR] llama-cli not found!")
        print("Install llama.cpp to use OCR+Translate feature.")
        return 1

    print("\n[OK] llama-cli found")

    # Find test image
    test_images = [
        "input/page_1.jpg",
        "input/page_2.jpg",
    ]

    img_path = None
    for p in test_images:
        if os.path.exists(p):
            img_path = p
            break

    if not img_path:
        print("\n[ERROR] No test images found in input/")
        return 1

    print(f"\n[Test] Using image: {img_path}")

    # Load image
    img = Image.open(img_path).convert('RGB')
    print(f"  Image size: {img.width}x{img.height}")

    # Detect bubbles
    print("\n[Detect] Loading detector...")
    mode = detect_mode()
    session, mode = create_session(mode)

    print("[Detect] Running detection...")
    bubbles, detect_time = detect_all(session, [img], mode, target_label=1)
    print(f"  Found {len(bubbles)} bubbles in {detect_time:.0f}ms")

    if not bubbles:
        print("\n[WARNING] No bubbles detected, skipping OCR test")
        return 0

    # Create grid
    print("\n[Grid] Creating grid with colored separators...")
    grid_img, positions, grid_info = create_ocr_grid(bubbles[:5])  # Test with first 5 bubbles
    print(f"  Grid size: {grid_img.width}x{grid_img.height}")
    print(f"  Cells: {grid_info['total_cells']}")

    # Save grid for inspection
    grid_img.save("output/test_grid.jpg")
    print("  Saved grid to: output/test_grid.jpg")

    # Test OCR only
    print("\n[OCR] Testing OCR only mode...")
    ocr = VlmOCR()
    result = ocr.run(grid_img, positions, grid_info, translate=False)
    print(f"  Lines: {result.get('line_count', 0)}")
    print(f"  Time: {result.get('processing_time_ms', 0):.0f}ms")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    else:
        for line in result.get('lines', [])[:3]:
            print(f"    [{line.get('cell_idx')}]: {line.get('text', '')[:50]}")

    # Test OCR+Translate
    print("\n[OCR+Translate] Testing combined mode...")
    result = ocr.run(grid_img, positions, grid_info, translate=True)
    print(f"  Lines: {result.get('line_count', 0)}")
    print(f"  Time: {result.get('processing_time_ms', 0):.0f}ms")
    print(f"  Translated: {result.get('translated', False)}")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    else:
        for line in result.get('lines', [])[:3]:
            print(f"    [{line.get('cell_idx')}]: {line.get('text', '')[:50]}")

    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    return 0


if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    sys.exit(main())
