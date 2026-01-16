#!/usr/bin/env python3
"""Full pipeline test with detailed timing measurements."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
from workflow import detect_mode, create_session, detect_all, CROP_PADDING
from workflow.ocr_vlm import (
    VlmOCR, create_ocr_grid, check_vlm_available, VLM_MODEL,
    LlmTranslator, check_translate_server, TRANSLATE_MODEL
)
from workflow.ocr import map_ocr


class Timer:
    """Simple timing helper."""
    def __init__(self):
        self.times = {}
        self._start = None
        self._name = None

    def start(self, name):
        self._name = name
        self._start = time.time()

    def stop(self):
        if self._start and self._name:
            elapsed = (time.time() - self._start) * 1000
            self.times[self._name] = elapsed
            self._start = None
            return elapsed
        return 0

    def report(self):
        print("\n" + "=" * 60)
        print("TIMING BREAKDOWN")
        print("=" * 60)
        total = 0
        for name, ms in self.times.items():
            print(f"  {name:25s}: {ms:8.1f} ms")
            total += ms
        print("  " + "-" * 40)
        print(f"  {'TOTAL':25s}: {total:8.1f} ms ({total/1000:.2f}s)")
        print("=" * 60)


def main():
    timer = Timer()

    print("=" * 60)
    print("Full Pipeline Test with Timing")
    print("=" * 60)

    # Check servers
    print("\n[Setup] Checking servers...")
    ocr_ok = check_vlm_available()
    translate_ok = check_translate_server()
    print(f"  OCR Server (8080):       {'OK' if ocr_ok else 'NOT RUNNING'}")
    print(f"  Translate Server (8081): {'OK' if translate_ok else 'NOT RUNNING'}")

    if not ocr_ok:
        print(f"\n[ERROR] Start OCR server: llama-server -hf {VLM_MODEL} --port 8080")
        return 1

    # Load pages
    print("\n[1] Loading images...")
    timer.start("Image Loading")
    pages = []
    for i in [1, 2, 3, 4]:
        path = f"input/page_{i}.jpg"
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            pages.append((i, img))
            print(f"     Page {i}: {img.width}x{img.height}")
    timer.stop()

    if not pages:
        print("[ERROR] No pages found in input/")
        return 1

    # Load detector
    print("\n[2] Loading detector...")
    timer.start("Detector Load")
    mode = detect_mode()
    session, mode = create_session(mode)
    timer.stop()

    # Detect bubbles
    print("\n[3] Detecting speech bubbles...")
    timer.start("Detection")
    images = [p[1] for p in pages]
    bubbles, detect_time = detect_all(session, images, mode, target_label=1)
    timer.stop()
    print(f"     Found {len(bubbles)} bubbles")

    by_page = {}
    for b in bubbles:
        pid = b['page_idx']
        by_page.setdefault(pid, []).append(b)
    for pid in sorted(by_page.keys()):
        print(f"       Page {pages[pid][0]}: {len(by_page[pid])} bubbles")

    if not bubbles:
        print("[WARNING] No bubbles detected")
        timer.report()
        return 0

    # Create grid
    print("\n[4] Creating OCR grid...")
    timer.start("Grid Creation")
    grid_img, positions, grid_info = create_ocr_grid(bubbles)
    timer.stop()
    print(f"     Grid: {grid_img.width}x{grid_img.height}, {grid_info['total_cells']} cells")

    os.makedirs("output", exist_ok=True)
    grid_img.save("output/test_grid_timed.jpg")

    # OCR
    print("\n[5] Running OCR (Japanese)...")
    ocr = VlmOCR(max_cells_per_batch=8)
    timer.start("OCR")
    ocr_result, ocr_positions, _ = ocr.run_batched(bubbles, translate=False)
    timer.stop()

    ocr_lines = ocr_result.get('line_count', 0)
    ocr_batches = ocr_result.get('batch_count', 1)
    print(f"     {ocr_lines} lines extracted ({ocr_batches} batches)")

    if ocr_result.get('error'):
        print(f"     Error: {ocr_result['error']}")

    # Show OCR results
    mapped = map_ocr(ocr_result, ocr_positions)
    print("\n     OCR Results (first 10):")
    count = 0
    for (page_idx, bubble_idx), texts in sorted(mapped.items()):
        if count >= 10:
            print(f"       ... and {len(mapped) - 10} more")
            break
        page_num = pages[page_idx][0]
        for t in texts:
            text = t['text'].replace('\n', ' ')[:50]
            print(f"       [P{page_num} B{bubble_idx}]: {text}")
        count += 1

    # Translation
    if translate_ok:
        print("\n[6] Running Translation (Japanese → English)...")
        translator = LlmTranslator()

        # Extract all OCR texts
        all_texts = []
        text_keys = []
        for (page_idx, bubble_idx), texts in sorted(mapped.items()):
            for t in texts:
                all_texts.append(t['text'].replace('\n', ' '))
                text_keys.append((page_idx, bubble_idx))

        timer.start("Translation")
        trans_result = translator.translate(all_texts)
        timer.stop()

        trans_count = trans_result.get('count', 0)
        print(f"     {trans_count}/{len(all_texts)} texts translated")

        if trans_result.get('errors'):
            print(f"     Errors: {trans_result['errors']}")

        # Show translations
        translations = trans_result.get('translations', [])
        print("\n     Translations (first 10):")
        for i, (key, trans) in enumerate(zip(text_keys[:10], translations[:10])):
            page_num = pages[key[0]][0]
            print(f"       [P{page_num} B{key[1]}]: {trans[:50]}")
        if len(translations) > 10:
            print(f"       ... and {len(translations) - 10} more")
    else:
        print("\n[6] Skipping translation (server not running)")
        print(f"    Start with: llama-server -hf {TRANSLATE_MODEL} --port 8081")

    # Report timing
    timer.report()

    return 0


if __name__ == '__main__':
    sys.exit(main())
