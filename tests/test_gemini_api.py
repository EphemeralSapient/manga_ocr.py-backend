#!/usr/bin/env python3
"""Test Gemini API OCR integration.

Tests the Gemini API OCR workflow with different models:
- gemma-3-27b-it (Gemma 3 27B IT - RECOMMENDED)
- gemini-2.0-flash-lite (Flash Lite 2.0)
- gemini-2.5-flash-lite-preview-06-17 (Flash Lite 2.5)
- gemini-2.5-flash-preview-05-20 (Flash 2.5 preview)

Usage:
    python tests/test_gemini_api.py [--model MODEL] [--image PATH]
"""

import sys
import os
import argparse
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image


def test_gemini_availability():
    """Test if Gemini API is available."""
    print("\n[1] Testing Gemini API availability...")

    from workflow.ocr_api import check_gemini_available, GEMMA_MODELS, GEMINI_MODELS, ALL_MODELS

    print(f"    Available Gemma models: {list(GEMMA_MODELS.values())}")
    print(f"    Available Gemini models: {list(GEMINI_MODELS.values())}")

    available = check_gemini_available()
    if available:
        print("    [OK] Gemini API is available")
        return True
    else:
        print("    [ERROR] Gemini API not available!")
        print("    - Check if google-genai is installed: pip install google-genai")
        print("    - Check if API key is set in config.json or GEMINI_API_KEY env var")
        return False


def test_simple_ocr(model: str = None):
    """Test simple OCR with a generated test image."""
    print(f"\n[2] Testing simple OCR{f' with model: {model}' if model else ''}...")

    from workflow.ocr_api import GeminiOCR

    # Create a simple test image with text
    img = Image.new('RGB', (200, 100), color='white')

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        # Draw some text
        draw.text((20, 30), "Hello\nWorld", fill='black')
    except Exception as e:
        print(f"    [WARN] Could not draw text: {e}")

    # Run OCR
    ocr = GeminiOCR(model=model) if model else GeminiOCR()
    print(f"    Using model: {ocr.model}")

    t0 = time.time()
    result = ocr.run([img], translate=False)
    elapsed = (time.time() - t0) * 1000

    print(f"    Processing time: {elapsed:.0f}ms")
    print(f"    Lines found: {result.get('line_count', 0)}")

    if result.get('error'):
        print(f"    [ERROR] {result['error']}")
        return False

    for line in result.get('lines', []):
        print(f"    [{line.get('cell_idx', 0)}]: {line.get('text', '')}")

    print("    [OK] Simple OCR test passed")
    return True


def test_ocr_with_image(image_path: str, model: str = None):
    """Test OCR with a real image."""
    print(f"\n[3] Testing OCR with image: {image_path}")

    if not os.path.exists(image_path):
        print(f"    [ERROR] Image not found: {image_path}")
        return False

    from workflow.ocr_api import GeminiOCR

    img = Image.open(image_path).convert('RGB')
    print(f"    Image size: {img.width}x{img.height}")

    ocr = GeminiOCR(model=model) if model else GeminiOCR()
    print(f"    Using model: {ocr.model}")

    t0 = time.time()
    result = ocr.run([img], translate=False)
    elapsed = (time.time() - t0) * 1000

    print(f"    Processing time: {elapsed:.0f}ms")
    print(f"    Lines found: {result.get('line_count', 0)}")

    if result.get('error'):
        print(f"    [ERROR] {result['error']}")
        return False

    for line in result.get('lines', []):
        text = line.get('text', '')[:100]
        print(f"    [{line.get('cell_idx', 0)}]: {text}")

    print("    [OK] Image OCR test passed")
    return True


def test_ocr_with_translation(image_path: str, model: str = None):
    """Test OCR + Translation with a real image."""
    print(f"\n[4] Testing OCR + Translation with image: {image_path}")

    if not os.path.exists(image_path):
        print(f"    [ERROR] Image not found: {image_path}")
        return False

    from workflow.ocr_api import GeminiOCR
    from config import get_target_language_name

    img = Image.open(image_path).convert('RGB')
    target_lang = get_target_language_name()
    print(f"    Image size: {img.width}x{img.height}")
    print(f"    Target language: {target_lang}")

    ocr = GeminiOCR(model=model) if model else GeminiOCR()
    print(f"    Using model: {ocr.model}")

    t0 = time.time()
    result = ocr.run([img], translate=True)
    elapsed = (time.time() - t0) * 1000

    print(f"    Processing time: {elapsed:.0f}ms")
    print(f"    Lines found: {result.get('line_count', 0)}")
    print(f"    Translated: {result.get('translated', False)}")

    if result.get('error'):
        print(f"    [ERROR] {result['error']}")
        return False

    for line in result.get('lines', []):
        text = line.get('text', '')[:100]
        print(f"    [{line.get('cell_idx', 0)}]: {text}")

    print("    [OK] OCR + Translation test passed")
    return True


def test_full_workflow(image_path: str, model: str = None):
    """Test full workflow: detect bubbles -> OCR with Gemini API."""
    print(f"\n[5] Testing full workflow with image: {image_path}")

    if not os.path.exists(image_path):
        print(f"    [ERROR] Image not found: {image_path}")
        return False

    try:
        # Import workflow components
        from workflow import detect_mode, create_session, detect_all, CROP_PADDING
        from workflow.ocr import run_ocr_on_bubbles
        from config import get_ocr_method, get_gemini_model

        ocr_method = get_ocr_method()
        gemini_model = get_gemini_model()
        print(f"    OCR method: {ocr_method}")
        print(f"    Gemini model: {gemini_model}")

        # Load image
        img = Image.open(image_path).convert('RGB')
        print(f"    Image size: {img.width}x{img.height}")

        # Detect bubbles
        print("    Detecting bubbles...")
        mode = detect_mode()
        session, mode = create_session(mode)
        bubbles, detect_time = detect_all(session, [img], mode, target_label=1)
        print(f"    Found {len(bubbles)} bubbles in {detect_time:.0f}ms")

        if not bubbles:
            print("    [WARN] No bubbles detected - nothing to OCR")
            return True

        # Run OCR
        print("    Running OCR via workflow...")
        t0 = time.time()
        ocr_result, positions, _ = run_ocr_on_bubbles(bubbles[:5], translate=False)  # Test with first 5
        elapsed = (time.time() - t0) * 1000

        print(f"    Processing time: {elapsed:.0f}ms")
        print(f"    Lines found: {ocr_result.get('line_count', 0)}")

        if ocr_result.get('error'):
            print(f"    [ERROR] {ocr_result['error']}")
            return False

        for line in ocr_result.get('lines', [])[:3]:
            text = line.get('text', '')[:80]
            print(f"    [{line.get('cell_idx', 0)}]: {text}")

        print("    [OK] Full workflow test passed")
        return True

    except FileNotFoundError as e:
        print(f"    [SKIP] Detection model not found: {e}")
        print("    Run setup.py to download detection models")
        return None  # Skipped, not failed


def test_model_selection():
    """Test different model options."""
    print("\n[6] Testing model selection...")

    from workflow.ocr_api import GeminiOCR, ALL_MODELS

    print(f"    All supported models: {list(ALL_MODELS.values())}")

    # Test instantiating with different models
    for name, model_id in ALL_MODELS.items():
        ocr = GeminiOCR(model=model_id)
        print(f"    {name}: {ocr.model} - OK")

    print("    [OK] Model selection test passed")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test Gemini API OCR')
    parser.add_argument('--model', type=str, help='Model to use (e.g., gemma-3-27b-it, gemini-2.0-flash-lite)')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--quick', action='store_true', help='Run quick availability test only')
    args = parser.parse_args()

    print("=" * 60)
    print("Gemini API OCR Test Suite")
    print("=" * 60)

    # Test 1: Availability
    if not test_gemini_availability():
        print("\n[FAIL] Gemini API not available - aborting tests")
        return 1

    if args.quick:
        print("\n[OK] Quick test passed - Gemini API is available")
        return 0

    # Test 2: Simple OCR
    test_simple_ocr(args.model)

    # Test 3-5: With image (if provided or found)
    image_path = args.image
    if not image_path:
        # Try to find a test image
        for p in ['input/page_1.jpg', 'input/page_2.jpg', 'input/test.jpg', 'input/test.png']:
            if os.path.exists(p):
                image_path = p
                break

    if image_path and os.path.exists(image_path):
        test_ocr_with_image(image_path, args.model)
        test_ocr_with_translation(image_path, args.model)
        test_full_workflow(image_path, args.model)
    else:
        print("\n[SKIP] No test image found - skipping image tests")
        print("    Provide --image PATH or put images in input/ folder")

    # Test 6: Model selection
    test_model_selection()

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
