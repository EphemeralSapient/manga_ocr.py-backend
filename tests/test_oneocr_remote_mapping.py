#!/usr/bin/env python3
"""
Test OneOCR Remote server and verify bbox mapping.

This script tests:
1. Direct OneOCR remote call with a single image
2. Grid-based OCR and bbox matching
3. Debug why some bubbles don't get text mapped
"""

import os
import sys
import json
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import get_oneocr_server_url, verify_oneocr_server
from workflow.ocr import grid_bubbles, map_ocr, OneOCRRemote

INPUT_DIR = "input"
OUTPUT_DIR = "output/debug_ocr_mapping"


def test_direct_ocr(remote, image_path):
    """Test direct OCR on a single image."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct OCR on single image")
    print("=" * 60)

    img = Image.open(image_path).convert('RGB')
    print(f"  Image: {image_path}")
    print(f"  Size: {img.width}x{img.height}")

    # Run OCR
    result = remote.run(img)

    print(f"\n  OCR returned {result['line_count']} lines:")
    for i, line in enumerate(result['lines'][:20]):  # First 20 lines
        text = line['text'][:30].replace('\n', '\\n')
        bbox = line['bbox']
        cx = bbox['x'] + bbox['width'] / 2
        cy = bbox['y'] + bbox['height'] / 2
        print(f"    [{i}] text='{text}' bbox=({bbox['x']:.0f},{bbox['y']:.0f},{bbox['width']:.0f},{bbox['height']:.0f}) center=({cx:.0f},{cy:.0f})")

    if result['line_count'] > 20:
        print(f"    ... and {result['line_count'] - 20} more lines")

    return result


def test_grid_ocr(remote, image_paths, max_pages=3):
    """Test grid-based OCR with multiple pages."""
    print("\n" + "=" * 60)
    print("TEST 2: Grid OCR with bubble detection")
    print("=" * 60)

    # Load images
    images = []
    for path in image_paths[:max_pages]:
        img = Image.open(path).convert('RGB')
        images.append(img)
    print(f"  Loaded {len(images)} pages")

    # Detect bubbles
    from workflow import detect_mode, create_session, detect_all
    mode = detect_mode()
    session, mode = create_session(mode)
    bubbles, detect_time = detect_all(session, images, mode, target_label=1)
    print(f"  Detected {len(bubbles)} bubbles in {detect_time:.0f}ms")

    # Print bubble details
    print(f"\n  Bubble details:")
    for b in bubbles[:20]:
        box = b['box']
        print(f"    page={b['page_idx']}, idx={b['bubble_idx']}, box=[{box[0]}, {box[1]}, {box[2]}, {box[3]}], size={box[2]-box[0]}x{box[3]-box[1]}")
    if len(bubbles) > 20:
        print(f"    ... and {len(bubbles) - 20} more bubbles")

    # Create grid
    grid_img, positions, _ = grid_bubbles(bubbles)
    print(f"\n  Grid created: {grid_img.width}x{grid_img.height}")

    # Save grid for inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(OUTPUT_DIR, "ocr_grid.png")
    grid_img.save(grid_path)
    print(f"  Saved grid to: {grid_path}")

    # Print position details (grid_box)
    print(f"\n  Position details (showing grid_box):")
    for i, pos in enumerate(positions[:20]):
        key = pos['key']
        gbox = pos['grid_box']
        bbox = pos['bubble_box']
        print(f"    pos[{i}]: key={key}, grid_box=({gbox[0]}, {gbox[1]}, {gbox[2]}, {gbox[3]}), bubble_box=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    if len(positions) > 20:
        print(f"    ... and {len(positions) - 20} more positions")

    # Run OCR on grid
    print(f"\n  Running OneOCR on grid...")
    ocr_result = remote.run(grid_img)
    print(f"  OCR returned {ocr_result['line_count']} lines")

    # Show OCR line details with bbox
    print(f"\n  OCR line bboxes:")
    for i, line in enumerate(ocr_result['lines'][:30]):
        text = line['text'][:20].replace('\n', '\\n')
        bbox = line['bbox']
        cx = bbox['x'] + bbox['width'] / 2
        cy = bbox['y'] + bbox['height'] / 2
        print(f"    line[{i}]: text='{text}' center=({cx:.0f},{cy:.0f}) bbox=({bbox['x']:.0f},{bbox['y']:.0f},{bbox['width']:.0f},{bbox['height']:.0f})")
    if len(ocr_result['lines']) > 30:
        print(f"    ... and {len(ocr_result['lines']) - 30} more lines")

    # Map OCR to bubbles
    bubble_texts = map_ocr(ocr_result, positions)

    # Count mapped vs unmapped
    mapped_keys = set(bubble_texts.keys())
    all_keys = set(pos['key'] for pos in positions)
    unmapped_keys = all_keys - mapped_keys

    print(f"\n  Mapping results:")
    print(f"    Total bubbles: {len(all_keys)}")
    print(f"    Mapped: {len(mapped_keys)} ({100*len(mapped_keys)/len(all_keys):.1f}%)")
    print(f"    Unmapped: {len(unmapped_keys)}")

    # Show unmapped bubbles
    if unmapped_keys:
        print(f"\n  Unmapped bubble positions:")
        for key in sorted(unmapped_keys)[:20]:
            pos = next(p for p in positions if p['key'] == key)
            gbox = pos['grid_box']
            print(f"    {key}: grid_box=({gbox[0]}, {gbox[1]}, {gbox[2]}, {gbox[3]})")
        if len(unmapped_keys) > 20:
            print(f"    ... and {len(unmapped_keys) - 20} more")

    # Debug: Try to find OCR lines that should match unmapped bubbles
    print(f"\n  Debugging unmapped bubbles:")
    for key in sorted(unmapped_keys)[:10]:
        pos = next(p for p in positions if p['key'] == key)
        gx1, gy1, gx2, gy2 = pos['grid_box']

        # Find OCR lines that might match
        candidates = []
        for line in ocr_result['lines']:
            bbox = line['bbox']
            cx = bbox['x'] + bbox['width'] / 2
            cy = bbox['y'] + bbox['height'] / 2

            # Check if center is in or near this grid box
            x_dist = max(gx1 - cx, 0, cx - gx2)
            y_dist = max(gy1 - cy, 0, cy - gy2)

            if x_dist < 50 and y_dist < 50:  # Within 50 pixels
                candidates.append({
                    'text': line['text'][:20],
                    'cx': cx, 'cy': cy,
                    'in_box': gx1 <= cx <= gx2 and gy1 <= cy <= gy2,
                    'x_dist': x_dist, 'y_dist': y_dist
                })

        print(f"\n    Bubble {key}: grid_box=({gx1}, {gy1}, {gx2}, {gy2})")
        if candidates:
            for c in candidates[:3]:
                status = "IN BOX" if c['in_box'] else f"out by ({c['x_dist']:.0f}, {c['y_dist']:.0f})"
                print(f"      Candidate: '{c['text']}' center=({c['cx']:.0f},{c['cy']:.0f}) {status}")
        else:
            print(f"      No candidates within 50px")

    return ocr_result, positions, bubble_texts


def visualize_grid_mapping(grid_img, positions, ocr_result):
    """Create visualization of OCR bboxes vs grid positions."""
    viz = grid_img.copy()
    draw = ImageDraw.Draw(viz)

    # Draw grid positions in blue
    for pos in positions:
        gbox = pos['grid_box']
        draw.rectangle(gbox, outline="blue", width=2)

    # Draw OCR bbox centers in red
    for line in ocr_result['lines']:
        bbox = line['bbox']
        cx = bbox['x'] + bbox['width'] / 2
        cy = bbox['y'] + bbox['height'] / 2
        # Draw cross at center
        draw.line([(cx-5, cy), (cx+5, cy)], fill="red", width=2)
        draw.line([(cx, cy-5), (cx, cy+5)], fill="red", width=2)

    viz_path = os.path.join(OUTPUT_DIR, "grid_mapping_viz.png")
    viz.save(viz_path)
    print(f"\n  Saved visualization to: {viz_path}")
    print(f"    Blue boxes = grid positions")
    print(f"    Red crosses = OCR line centers")

    return viz_path


def main():
    # Check OneOCR server
    url = get_oneocr_server_url()
    if not url:
        print("ERROR: OneOCR server not configured")
        print("Set 'oneocr_server_url' in config.json")
        return

    success, message, details = verify_oneocr_server(url)
    if not success:
        print(f"ERROR: {message}")
        return

    print(f"OneOCR server: {url}")
    remote = OneOCRRemote(url)

    # Get input images
    import glob
    patterns = [f"{INPUT_DIR}/page_*.jpg", f"{INPUT_DIR}/page_*.png"]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    # Filter and sort
    exclude = ['viz', 'debug', 'detected', 'fixed', 'compare', 'test']
    image_paths = [f for f in sorted(set(all_files))
                   if not any(kw in os.path.basename(f).lower() for kw in exclude)]

    if not image_paths:
        print(f"No input images found in {INPUT_DIR}/")
        return

    print(f"Found {len(image_paths)} input images")

    # Test 1: Direct OCR
    test_direct_ocr(remote, image_paths[0])

    # Test 2: Grid OCR with mapping
    ocr_result, positions, bubble_texts = test_grid_ocr(remote, image_paths, max_pages=3)

    # Test 3: Visualize
    from PIL import Image
    grid_path = os.path.join(OUTPUT_DIR, "ocr_grid.png")
    if os.path.exists(grid_path):
        grid_img = Image.open(grid_path)
        visualize_grid_mapping(grid_img, positions, ocr_result)

    print("\n" + "=" * 60)
    print("DONE - Check output in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
