#!/usr/bin/env python3
"""Compare different Gemini/Gemma API models for OCR.

Tests all available models and compares results:
- gemma-3-27b-it (Gemma 3 27B IT - recommended for quality)
- gemini-2.0-flash-lite (Flash Lite 2.0 - faster)
- gemini-2.5-flash-lite-preview-06-17 (Flash Lite 2.5)
- gemini-2.5-flash-preview-05-20 (Flash 2.5 preview)

Usage:
    python tests/test_gemini_models.py [--image PATH]
"""

import sys
import os
import argparse
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image


def compare_models(image_path: str, translate: bool = False):
    """Compare OCR results across all models."""
    from workflow.ocr_api import GeminiOCR, ALL_MODELS, check_gemini_available

    if not check_gemini_available():
        print("[ERROR] Gemini API not available!")
        return None

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None

    img = Image.open(image_path).convert('RGB')
    print(f"Image: {image_path} ({img.width}x{img.height})")
    print(f"Translate: {translate}")
    print("=" * 70)

    results = {}

    for name, model_id in ALL_MODELS.items():
        print(f"\n[{name}] Testing {model_id}...")

        ocr = GeminiOCR(model=model_id)

        try:
            t0 = time.time()
            result = ocr.run([img], translate=translate)
            elapsed = (time.time() - t0) * 1000

            results[name] = {
                'model': model_id,
                'time_ms': elapsed,
                'lines': result.get('lines', []),
                'line_count': result.get('line_count', 0),
                'error': result.get('error'),
            }

            if result.get('error'):
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    Time: {elapsed:.0f}ms")
                print(f"    Lines: {result.get('line_count', 0)}")
                for line in result.get('lines', [])[:2]:
                    text = line.get('text', '')[:60]
                    print(f"    > {text}")

        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results[name] = {
                'model': model_id,
                'error': str(e),
            }

    return results


def print_comparison_table(results: dict):
    """Print comparison results as a table."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Time (ms)':<12} {'Lines':<8} {'Status':<10}")
    print("-" * 70)

    for name, data in results.items():
        model = data.get('model', 'unknown')[:30]
        time_ms = f"{data.get('time_ms', 0):.0f}" if 'time_ms' in data else '-'
        lines = str(data.get('line_count', '-'))
        status = 'ERROR' if data.get('error') else 'OK'
        print(f"{model:<35} {time_ms:<12} {lines:<8} {status:<10}")


def main():
    parser = argparse.ArgumentParser(description='Compare Gemini API models for OCR')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--translate', action='store_true', help='Also test translation')
    args = parser.parse_args()

    print("=" * 70)
    print("Gemini/Gemma Model Comparison Test")
    print("=" * 70)

    # Find test image
    image_path = args.image
    if not image_path:
        for p in ['input/page_1.jpg', 'input/page_2.jpg', 'input/test.jpg', 'input/test.png']:
            if os.path.exists(p):
                image_path = p
                break

    if not image_path or not os.path.exists(image_path):
        print("\n[ERROR] No test image found!")
        print("Provide --image PATH or put images in input/ folder")
        return 1

    # Compare OCR only
    print("\n--- OCR Only ---")
    results_ocr = compare_models(image_path, translate=False)
    print_comparison_table(results_ocr)

    # Compare OCR + Translation (if requested)
    if args.translate:
        print("\n--- OCR + Translation ---")
        results_translate = compare_models(image_path, translate=True)
        print_comparison_table(results_translate)

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
