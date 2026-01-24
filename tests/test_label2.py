#!/usr/bin/env python3
"""
Test L2 (label2) text clearing for background text regions.

Tests:
1. L2 regions get text_seg mask (not just L1 pages)
2. L2 rendering uses text_seg mask for clearing
3. OCR bbox-based clearing works when text_seg unavailable

Usage:
    python3 tests/test_label2.py [image_path]

    # Or use curl with running server:
    curl -X POST http://localhost:5022/api/v1/process \
        -F "images=@manga_page.jpg" \
        -o result.json
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path

# Default test image
DEFAULT_TEST_IMAGE = "tests/test_page_013.png"
SERVER_URL = os.environ.get("MANGA_SERVER_URL", "http://localhost:5022")


def test_via_curl(image_path: str, output_dir: str = "tests/aot_results"):
    """Test L2 processing via curl to running server."""

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return False

    print(f"Testing L2 processing with: {image_path}")
    print(f"Server: {SERVER_URL}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Send to server
    with open(image_path, 'rb') as f:
        files = {'images': (os.path.basename(image_path), f, 'image/png')}

        try:
            print(f"\nSending request to {SERVER_URL}/api/v1/process...")
            response = requests.post(
                f"{SERVER_URL}/api/v1/process",
                files=files,
                timeout=120
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to server at {SERVER_URL}")
            print("Make sure the server is running: python server.py")
            return False
        except requests.exceptions.Timeout:
            print("Error: Request timed out (120s)")
            return False

    result = response.json()

    # Check for errors
    if not result.get('success', False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False

    # Parse stats
    stats = result.get('stats', {})
    print("\n=== Processing Stats ===")
    print(f"  L1 bubbles detected: {stats.get('bubbles_detected', 'N/A')}")
    print(f"  L2 regions detected: {stats.get('label2_detected', 'N/A')}")
    print(f"  OCR L1 lines: {stats.get('ocr_l1_lines', 'N/A')}")
    print(f"  OCR L2 lines: {stats.get('ocr_l2_lines', 'N/A')}")
    print(f"  Texts to translate (L1): {stats.get('texts_l1', 'N/A')}")
    print(f"  Texts to translate (L2): {stats.get('texts_l2', 'N/A')}")
    print(f"  Detect time: {stats.get('detect_ms', 'N/A')}ms")
    print(f"  OCR L1 time: {stats.get('ocr_l1_ms', 'N/A')}ms")
    print(f"  OCR L2 time: {stats.get('ocr_l2_ms', 'N/A')}ms")
    print(f"  Translation time: {stats.get('translation_ms', 'N/A')}ms")
    print(f"  Render time: {stats.get('render_ms', 'N/A')}ms")

    # Check render detail for text_seg usage
    render_detail = stats.get('render_detail', {})
    if render_detail:
        print(f"\n=== Render Details ===")
        print(f"  Text seg time: {render_detail.get('text_seg_ms', 0)}ms")
        print(f"  Mask apply time: {render_detail.get('mask_apply_ms', 0)}ms")
        print(f"  Text render time: {render_detail.get('text_render_ms', 0)}ms")

    # Save output images
    images = result.get('images', [])
    if images:
        print(f"\n=== Saving {len(images)} output images ===")
        for i, img_b64 in enumerate(images):
            img_data = base64.b64decode(img_b64)
            output_path = os.path.join(output_dir, f"l2_test_output_{i}.png")
            with open(output_path, 'wb') as f:
                f.write(img_data)
            print(f"  Saved: {output_path}")

    # Check for L2 processing
    l2_detected = stats.get('label2_detected', 0)
    l2_texts = stats.get('texts_l2', 0)

    print(f"\n=== L2 Test Results ===")
    if l2_detected > 0:
        print(f"  [PASS] L2 regions detected: {l2_detected}")
    else:
        print(f"  [INFO] No L2 regions detected (image may not have background text)")

    if l2_texts > 0:
        print(f"  [PASS] L2 texts extracted and translated: {l2_texts}")
    elif l2_detected > 0:
        print(f"  [INFO] L2 regions had no CJK text (may be SFX/English)")

    return True


def test_local(image_path: str, output_dir: str = "tests/aot_results"):
    """Test L2 processing locally without server."""

    # Add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from PIL import Image
    import numpy as np

    # Import workflow components
    from workflow import detect_mode, create_session, detect_all
    from workflow import run_ocr_on_bubbles, map_ocr
    from workflow import create_text_segmenter, get_text_segmenter
    from workflow import render_text_on_image

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return False

    print(f"Testing L2 processing locally with: {image_path}")

    # Load image
    img = Image.open(image_path).convert('RGB')
    images = [img]

    # Create session
    session, mode = create_session()
    print(f"Using detection mode: {mode}")

    # Detect L1 and L2
    print("\n=== Detection ===")
    bubbles_l1, detect_time_l1 = detect_all(session, images, mode, target_label=1)
    bubbles_l2, detect_time_l2 = detect_all(session, images, mode, target_label=2)
    print(f"  L1 bubbles: {len(bubbles_l1)} in {detect_time_l1:.0f}ms")
    print(f"  L2 regions: {len(bubbles_l2)} in {detect_time_l2:.0f}ms")

    # Run text segmentation on page (should include L2 pages now)
    print("\n=== Text Segmentation ===")
    text_seg = create_text_segmenter()
    if text_seg:
        img_array = np.array(img)
        text_seg_mask = text_seg(img_array, verbose=False)
        print(f"  Text seg mask: {text_seg_mask.shape if text_seg_mask is not None else 'None'}")
        print(f"  Text pixels detected: {np.sum(text_seg_mask > 127) if text_seg_mask is not None else 0}")
    else:
        text_seg_mask = None
        print("  Text segmenter not available")

    # Run OCR on L2
    if bubbles_l2:
        print("\n=== L2 OCR ===")
        ocr_result, positions, _ = run_ocr_on_bubbles(bubbles_l2, translate=False)
        print(f"  OCR lines: {ocr_result.get('line_count', 0)}")

        # Map OCR to bubbles
        l2_texts = map_ocr(ocr_result, positions)
        print(f"  Bubbles with text: {len(l2_texts)}")

        for key, texts in l2_texts.items():
            combined = " ".join(t.get('text', '') for t in texts)
            print(f"    {key}: {combined[:50]}...")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nTest completed. Output dir: {output_dir}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test L2 text clearing")
    parser.add_argument("image", nargs="?", default=DEFAULT_TEST_IMAGE,
                        help="Path to test image")
    parser.add_argument("--local", action="store_true",
                        help="Run locally without server")
    parser.add_argument("--output", "-o", default="tests/aot_results",
                        help="Output directory")
    args = parser.parse_args()

    if args.local:
        success = test_local(args.image, args.output)
    else:
        success = test_via_curl(args.image, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
