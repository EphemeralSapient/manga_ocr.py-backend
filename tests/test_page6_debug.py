#!/usr/bin/env python3
"""
Debug test for page 6 - runs server functions directly to diagnose text clearing issues.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from workflow import detect_mode, create_session, detect_all
from workflow import create_text_segmenter
from workflow import render_text_on_image
from workflow.ocr import HAS_GEMINI_OCR

INPUT_PATH = "input/page_6.jpg"
OUTPUT_DIR = "output/debug_page6"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load image
    print(f"Loading {INPUT_PATH}...")
    img = Image.open(INPUT_PATH).convert("RGB")
    img_array = np.array(img)
    print(f"  Image size: {img.width}x{img.height}")

    # Initialize models
    print("\nInitializing models...")
    mode = detect_mode()
    session, mode = create_session(mode)
    text_seg = create_text_segmenter()
    print(f"  Detector mode: {mode}")

    # Detect bubbles (label1)
    print("\nDetecting bubbles (label1)...")
    bubbles_l1, detect_time = detect_all(session, [img], mode, target_label=1)
    print(f"  Found {len(bubbles_l1)} L1 bubbles in {detect_time:.0f}ms")

    # Detect label2 regions
    print("\nDetecting label2 regions...")
    bubbles_l2, detect_time_l2 = detect_all(session, [img], mode, target_label=2)
    print(f"  Found {len(bubbles_l2)} L2 regions in {detect_time_l2:.0f}ms")

    # Print bubble details
    print("\n" + "="*60)
    print("L1 BUBBLES (text bubbles):")
    print("="*60)
    for i, b in enumerate(bubbles_l1):
        box = b['box']
        print(f"  [{i}] idx={b['bubble_idx']} box=[{box[0]}, {box[1]}, {box[2]}, {box[3]}] size={box[2]-box[0]}x{box[3]-box[1]}")

    print("\n" + "="*60)
    print("L2 REGIONS (text-free/background):")
    print("="*60)
    for i, b in enumerate(bubbles_l2):
        box = b['box']
        print(f"  [{i}] idx={b['bubble_idx']} box=[{box[0]}, {box[1]}, {box[2]}, {box[3]}] size={box[2]-box[0]}x{box[3]-box[1]}")

    # Run text segmentation
    print("\n" + "="*60)
    print("TEXT SEGMENTATION:")
    print("="*60)
    mask = text_seg(img_array, verbose=True)
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Mask min/max: {mask.min()}/{mask.max()}")
    print(f"  Mask mean: {mask.mean():.2f}")
    print(f"  Pixels > 10: {np.sum(mask > 10)} ({100*np.sum(mask > 10)/mask.size:.2f}%)")
    print(f"  Pixels > 30: {np.sum(mask > 30)} ({100*np.sum(mask > 30)/mask.size:.2f}%)")
    print(f"  Pixels > 127: {np.sum(mask > 127)} ({100*np.sum(mask > 127)/mask.size:.2f}%)")

    # Save mask visualization
    mask_img = Image.fromarray(mask)
    mask_path = os.path.join(OUTPUT_DIR, "text_seg_mask.png")
    mask_img.save(mask_path)
    print(f"  Saved mask to {mask_path}")

    # Check mask coverage for each L1 bubble
    print("\n" + "="*60)
    print("MASK COVERAGE PER L1 BUBBLE:")
    print("="*60)
    for i, b in enumerate(bubbles_l1):
        box = b['box']
        bx1, by1, bx2, by2 = box
        mask_region = mask[by1:by2, bx1:bx2]
        coverage_10 = np.sum(mask_region > 10) / max(1, mask_region.size) * 100
        coverage_30 = np.sum(mask_region > 30) / max(1, mask_region.size) * 100
        max_val = mask_region.max()
        print(f"  [{i}] idx={b['bubble_idx']}: coverage(>10)={coverage_10:.1f}%, coverage(>30)={coverage_30:.1f}%, max={max_val}")

    # Find where text actually is in the mask (for debugging offset issues)
    print("\n" + "="*60)
    print("TEXT LOCATIONS IN MASK (where mask > 100):")
    print("="*60)
    text_pixels = np.where(mask > 100)
    if len(text_pixels[0]) > 0:
        y_coords, x_coords = text_pixels
        print(f"  X range: {x_coords.min()} - {x_coords.max()}")
        print(f"  Y range: {y_coords.min()} - {y_coords.max()}")
        # Find clusters
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask > 100)
        print(f"  Found {num_features} text regions")
        for region_id in range(1, min(num_features + 1, 20)):  # Show first 20
            region_coords = np.where(labeled == region_id)
            if len(region_coords[0]) > 30:  # Show smaller regions too
                y_min, y_max = region_coords[0].min(), region_coords[0].max()
                x_min, x_max = region_coords[1].min(), region_coords[1].max()
                print(f"    Region {region_id}: x=[{x_min}, {x_max}] y=[{y_min}, {y_max}] size={len(region_coords[0])}px")

    # Specifically check for "取り分けるか" area - use actual L1:1 bubble box
    # L1:1 box=[144, 300, 208, 446]
    print("\n  Checking L1:1 bubble region x=144-208, y=300-446:")
    check_region = mask[300:446, 144:208]
    print(f"    Max value in region: {check_region.max()}")
    print(f"    Pixels > 10: {np.sum(check_region > 10)}")
    print(f"    Pixels > 50: {np.sum(check_region > 50)}")
    print(f"    Pixels > 5: {np.sum(check_region > 5)}")

    # Check with padding (what render uses with pad=150)
    print("\n  Checking L1:1 with 150px padding (x=0-358, y=150-596):")
    check_region_padded = mask[max(0,300-150):min(1280,446+150), max(0,144-150):min(900,208+150)]
    print(f"    Region shape: {check_region_padded.shape}")
    print(f"    Max value: {check_region_padded.max()}")
    print(f"    Pixels > 10: {np.sum(check_region_padded > 10)}")
    print(f"    Pixels > 5: {np.sum(check_region_padded > 5)}")

    # Check specifically the LEFT area where 取り分けるか should be (x=0-144)
    print("\n  Checking LEFT side of padded region (x=0-144, y=150-596) - where 取り分けるか text is:")
    left_region = mask[150:596, 0:144]
    print(f"    Region shape: {left_region.shape}")
    print(f"    Max value: {left_region.max()}")
    print(f"    Pixels > 10: {np.sum(left_region > 10)}")
    print(f"    Pixels > 5: {np.sum(left_region > 5)}")

    # Save the left region crop for visual inspection
    left_region_img = Image.fromarray(left_region)
    left_region_path = os.path.join(OUTPUT_DIR, "mask_left_region.png")
    left_region_img.save(left_region_path)
    print(f"    Saved left region mask to {left_region_path}")

    # Also check where 取り分けるか actually appears in mask (look for bright pixels)
    print("\n  Finding brightest regions in x=0-200, y=200-500:")
    search_region = mask[200:500, 0:200]
    print(f"    Max value: {search_region.max()}")
    # Find location of max
    if search_region.max() > 50:
        max_loc = np.unravel_index(search_region.argmax(), search_region.shape)
        print(f"    Max location (relative): y={max_loc[0]}, x={max_loc[1]}")
        print(f"    Max location (absolute): y={200+max_loc[0]}, x={max_loc[1]}")

    # Create test render with fake translations
    print("\n" + "="*60)
    print("TEST RENDER:")
    print("="*60)

    # Build bubble data structure for render
    ocr_data = []
    translations = {}
    for i, b in enumerate(bubbles_l1):
        box = b['box']
        bubble_data = {
            'idx': b['bubble_idx'],
            'bubble_box': list(box),
            'texts': [{'bbox': list(box), 'text': f'TEXT_{i}'}]  # Use bubble box as text bbox
        }
        ocr_data.append(bubble_data)
        translations[b['bubble_idx']] = f"Translation {i}"

    print(f"  Rendering {len(ocr_data)} bubbles with test translations...")

    # Render
    img_copy = img.copy()
    rendered = render_text_on_image(img_copy, ocr_data, translations, use_inpaint=False, precomputed_mask=mask)

    # Save rendered output
    render_path = os.path.join(OUTPUT_DIR, "rendered_test.jpg")
    rendered.save(render_path, quality=95)
    print(f"  Saved rendered image to {render_path}")

    # Also save original with bubble boxes drawn
    from PIL import ImageDraw
    viz_img = img.copy()
    draw = ImageDraw.Draw(viz_img)
    for b in bubbles_l1:
        box = b['box']
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]-15), f"L1:{b['bubble_idx']}", fill="red")
    for b in bubbles_l2:
        box = b['box']
        draw.rectangle(box, outline="blue", width=2)
        draw.text((box[0], box[1]-15), f"L2:{b['bubble_idx']}", fill="blue")

    viz_path = os.path.join(OUTPUT_DIR, "bubbles_viz.jpg")
    viz_img.save(viz_path, quality=95)
    print(f"  Saved bubble visualization to {viz_path}")

    print("\n" + "="*60)
    print("DONE! Check output in:", OUTPUT_DIR)
    print("="*60)


if __name__ == '__main__':
    main()
