#!/usr/bin/env python3
"""
Test script for AOT inpainting with text segmentation masks.

Tests different approaches:
1. Whole bbox inpainting (current default without text_seg)
2. Text pixel mask inpainting (current with text_seg)
3. Improved: bubble-sized mask with proper spatial mapping

Usage:
    python3.12 tests/test_aot_inpaint.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import time

from workflow.inpaint import create_inpainter, get_crop, get_crop_with_mask, prep, composite
from workflow.text_seg import create_text_segmenter

# Test image and bbox (stats panel from page 13)
TEST_IMAGE = "tests/test_page_013.png"
# Approximate bbox for the stats panel (adjust if needed)
STATS_BBOX = [354, 719, 573, 863]  # [x1, y1, x2, y2] - stats panel area

# Output directory
OUTPUT_DIR = "tests/aot_results"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_image():
    """Load test image as numpy array."""
    img = Image.open(TEST_IMAGE).convert("RGB")
    return np.array(img)


def save_result(img_array, name):
    """Save result image."""
    Image.fromarray(img_array).save(f"{OUTPUT_DIR}/{name}.png")
    print(f"  Saved: {OUTPUT_DIR}/{name}.png")


def test_bbox_inpaint(img_array, bbox, inpainter):
    """Test 1: Inpaint entire bbox (no text_seg mask)."""
    print("\n[Test 1] Whole bbox inpainting...")

    img_copy = img_array.copy()
    t0 = time.time()

    coords, result = inpainter(img_copy, bbox, verbose=False)
    img_copy[coords[0]:coords[1], coords[2]:coords[3]] = result

    elapsed = (time.time() - t0) * 1000
    print(f"  Time: {elapsed:.0f}ms")

    save_result(img_copy, "test1_bbox_inpaint")

    # Also save just the bbox region
    x1, y1, x2, y2 = bbox
    save_result(img_copy[y1:y2, x1:x2], "test1_bbox_region")

    return img_copy


def test_textseg_mask_inpaint(img_array, bbox, inpainter, text_seg):
    """Test 2: Inpaint only text pixels using text_seg mask."""
    print("\n[Test 2] Text_seg mask inpainting...")

    img_copy = img_array.copy()
    x1, y1, x2, y2 = bbox

    # Extract region and get text mask
    region = img_copy[y1:y2, x1:x2]
    t0 = time.time()
    text_mask = text_seg(region, verbose=False)
    seg_time = (time.time() - t0) * 1000
    print(f"  Text_seg time: {seg_time:.0f}ms")

    # Save the mask for inspection
    if text_mask is not None:
        save_result(text_mask, "test2_text_mask")
        mask_pixels = np.sum(text_mask > 127)
        total_pixels = text_mask.size
        print(f"  Mask: {mask_pixels} text pixels / {total_pixels} total ({100*mask_pixels/total_pixels:.1f}%)")

    # Inpaint with mask
    if text_mask is not None and np.any(text_mask > 127):
        t0 = time.time()
        coords, result = inpainter.inpaint_with_mask(img_copy, bbox, text_mask, verbose=False)
        img_copy[coords[0]:coords[1], coords[2]:coords[3]] = result
        inpaint_time = (time.time() - t0) * 1000
        print(f"  Inpaint time: {inpaint_time:.0f}ms")
    else:
        print(f"  No text detected - skipping inpaint")

    save_result(img_copy, "test2_textseg_inpaint")
    save_result(img_copy[y1:y2, x1:x2], "test2_textseg_region")

    return img_copy


def test_improved_mask_inpaint(img_array, bbox, inpainter, text_seg):
    """Test 3: Improved approach with proper spatial mapping.

    - Input image: bbox region with context padding
    - Input mask: same size as padded region, text pixels marked
    - Proper offset handling for spatial mapping
    """
    print("\n[Test 3] Improved mask inpainting with spatial mapping...")

    img_copy = img_array.copy()
    x1, y1, x2, y2 = bbox
    h, w = img_copy.shape[:2]

    # Context padding (same as AOT uses internally)
    context_pad = 64
    pad = 5

    # Calculate padded bbox
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(w, x2 + pad)
    py2 = min(h, y2 + pad)

    # Calculate context region
    cx1 = max(0, px1 - context_pad)
    cy1 = max(0, py1 - context_pad)
    cx2 = min(w, px2 + context_pad)
    cy2 = min(h, py2 + context_pad)

    print(f"  Original bbox: [{x1}, {y1}, {x2}, {y2}]")
    print(f"  Padded bbox: [{px1}, {py1}, {px2}, {py2}]")
    print(f"  Context region: [{cx1}, {cy1}, {cx2}, {cy2}]")

    # Extract the original bbox region for text_seg
    region = img_copy[y1:y2, x1:x2]

    t0 = time.time()
    text_mask_region = text_seg(region, verbose=False)
    seg_time = (time.time() - t0) * 1000
    print(f"  Text_seg time: {seg_time:.0f}ms")

    if text_mask_region is None or not np.any(text_mask_region > 127):
        print(f"  No text detected - skipping inpaint")
        return img_copy

    # Save region mask
    save_result(text_mask_region, "test3_region_mask")

    # Create full-size mask for the context region
    context_h = cy2 - cy1
    context_w = cx2 - cx1
    full_mask = np.zeros((context_h, context_w), dtype=np.uint8)

    # Place the text mask at the correct offset within the context region
    # The bbox region starts at (y1-cy1, x1-cx1) within the context crop
    offset_y = y1 - cy1
    offset_x = x1 - cx1
    mask_h, mask_w = text_mask_region.shape[:2]

    print(f"  Mask offset in context: ({offset_x}, {offset_y})")
    print(f"  Mask size: {mask_w}x{mask_h}")

    # Place mask at correct position
    full_mask[offset_y:offset_y+mask_h, offset_x:offset_x+mask_w] = text_mask_region

    # Save full context mask
    save_result(full_mask, "test3_context_mask")

    # Extract context crop for AOT
    context_crop = img_copy[cy1:cy2, cx1:cx2].astype(np.float32)

    # Prepare for AOT (pad to multiple of 8)
    crop_h, crop_w = context_crop.shape[:2]
    nh = (crop_h + 7) // 8 * 8
    nw = (crop_w + 7) // 8 * 8

    padded_img = np.zeros((nh, nw, 3), np.float32)
    padded_mask = np.zeros((nh, nw), np.float32)
    padded_img[:crop_h, :crop_w] = context_crop
    padded_mask[:crop_h, :crop_w] = full_mask.astype(np.float32) / 255.0

    # Zero out masked region in input
    padded_img[padded_mask > 0.5] = 0

    print(f"  Padded size: {nw}x{nh}")

    # Run AOT
    t0 = time.time()
    out = inpainter.forward(padded_img, padded_mask)
    inpaint_time = (time.time() - t0) * 1000
    print(f"  AOT forward time: {inpaint_time:.0f}ms")

    # Composite result
    out_clipped = np.clip((out[:crop_h, :crop_w] + 1) * 127.5, 0, 255)
    mask_3ch = np.stack([full_mask.astype(np.float32) / 255.0] * 3, axis=-1)
    result = (context_crop * (1 - mask_3ch) + out_clipped * mask_3ch).astype(np.uint8)

    # Place result back
    img_copy[cy1:cy2, cx1:cx2] = result

    save_result(img_copy, "test3_improved_inpaint")
    save_result(img_copy[y1:y2, x1:x2], "test3_improved_region")

    return img_copy


def test_line_by_line_inpaint(img_array, bbox, inpainter, text_seg):
    """Test 4: Inpaint each text line separately.

    Find connected components in text mask and inpaint each separately.
    """
    print("\n[Test 4] Line-by-line inpainting...")

    from scipy import ndimage

    img_copy = img_array.copy()
    x1, y1, x2, y2 = bbox

    # Get text mask
    region = img_copy[y1:y2, x1:x2]
    text_mask = text_seg(region, verbose=False)

    if text_mask is None or not np.any(text_mask > 127):
        print(f"  No text detected - skipping")
        return img_copy

    # Find connected components (text lines/groups)
    binary_mask = text_mask > 127
    labeled, num_features = ndimage.label(binary_mask)
    print(f"  Found {num_features} text components")

    # Process each component
    for i in range(1, num_features + 1):
        component_mask = (labeled == i).astype(np.uint8) * 255

        # Get bounding box of this component
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue

        comp_y1, comp_y2 = ys.min(), ys.max() + 1
        comp_x1, comp_x2 = xs.min(), xs.max() + 1

        # Global coordinates
        global_bbox = [x1 + comp_x1, y1 + comp_y1, x1 + comp_x2, y1 + comp_y2]
        comp_mask = component_mask[comp_y1:comp_y2, comp_x1:comp_x2]

        print(f"    Component {i}: bbox={global_bbox}, pixels={np.sum(comp_mask > 127)}")

        # Inpaint this component
        if np.any(comp_mask > 127):
            coords, result = inpainter.inpaint_with_mask(img_copy, global_bbox, comp_mask, verbose=False)
            img_copy[coords[0]:coords[1], coords[2]:coords[3]] = result

    save_result(img_copy, "test4_linewise_inpaint")
    save_result(img_copy[y1:y2, x1:x2], "test4_linewise_region")

    return img_copy


def main():
    print("=" * 60)
    print("AOT Inpainting Test Script")
    print("=" * 60)

    ensure_output_dir()

    # Load test image
    print(f"\nLoading test image: {TEST_IMAGE}")
    img_array = load_test_image()
    print(f"  Image size: {img_array.shape[1]}x{img_array.shape[0]}")

    # Save original bbox region for comparison
    x1, y1, x2, y2 = STATS_BBOX
    save_result(img_array[y1:y2, x1:x2], "original_region")
    print(f"  Test bbox: {STATS_BBOX}")

    # Initialize models
    print("\nInitializing models...")
    inpainter = create_inpainter()
    text_seg = create_text_segmenter()
    print("  Models loaded")

    # Run tests
    test_bbox_inpaint(img_array.copy(), STATS_BBOX, inpainter)
    test_textseg_mask_inpaint(img_array.copy(), STATS_BBOX, inpainter, text_seg)
    test_improved_mask_inpaint(img_array.copy(), STATS_BBOX, inpainter, text_seg)
    test_line_by_line_inpaint(img_array.copy(), STATS_BBOX, inpainter, text_seg)

    print("\n" + "=" * 60)
    print(f"Results saved to {OUTPUT_DIR}/")
    print("Compare the *_region.png files to see differences")
    print("=" * 60)


if __name__ == "__main__":
    main()
