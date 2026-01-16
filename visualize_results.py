#!/usr/bin/env python3
"""
Visualize OCR results on original pages
- Draws bubble boxes (green) and OCR text boxes (red) on each page
- Creates a combined grid of all pages
"""

import json
from PIL import Image, ImageDraw, ImageFont
import math
import sys

def load_font(size=14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        return ImageFont.load_default()

def visualize_page(image_path, bubbles, font):
    """Draw boxes on a single page"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    for b in bubbles:
        # Bubble box (green)
        bx1, by1, bx2, by2 = b['bubble_box']
        draw.rectangle([bx1, by1, bx2, by2], outline=(0, 200, 0), width=2)
        draw.text((bx1 + 2, by1 + 2), f"B{b['idx']}", fill=(0, 200, 0), font=font)

        # OCR text boxes (red)
        for t in b.get('texts', []):
            tx1, ty1, tx2, ty2 = t['bbox']
            draw.rectangle([tx1, ty1, tx2, ty2], outline=(255, 0, 0), width=1)
            # Draw text (truncated)
            text_preview = t['text'][:8]
            draw.text((tx1, ty1 - 12), text_preview, fill=(255, 0, 0), font=font)

    return img

def create_grid(images, cols=4, thumb_width=400):
    """Create grid of images"""
    if not images:
        return Image.new('RGB', (100, 100), (255, 255, 255))

    # Resize to thumbnails
    thumbs = []
    for img in images:
        ratio = thumb_width / img.width
        thumb_h = int(img.height * ratio)
        thumbs.append(img.resize((thumb_width, thumb_h), Image.Resampling.LANCZOS))

    rows = math.ceil(len(thumbs) / cols)
    max_h = max(t.height for t in thumbs)

    grid = Image.new('RGB', (cols * thumb_width, rows * max_h), (40, 40, 40))

    for i, thumb in enumerate(thumbs):
        r, c = i // cols, i % cols
        x = c * thumb_width
        y = r * max_h + (max_h - thumb.height) // 2
        grid.paste(thumb, (x, y))

    return grid

def main(json_path='ocr_result.json', output_path='visualized_output.jpg'):
    print(f"Loading results from {json_path}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    font = load_font(14)
    visualized = []

    # Sort pages numerically
    sorted_pages = sorted(results.keys(),
                         key=lambda x: int(x.replace('page_', '').replace('.jpg', '')))

    print(f"Visualizing {len(sorted_pages)} pages...")

    for page_path in sorted_pages:
        bubbles = results[page_path]
        img = visualize_page(page_path, bubbles, font)
        visualized.append(img)

        n_texts = sum(len(b.get('texts', [])) for b in bubbles)
        print(f"  {page_path}: {len(bubbles)} bubbles, {n_texts} text boxes")

    # Create grid
    print(f"\nCreating grid...")
    grid = create_grid(visualized, cols=5, thumb_width=350)
    grid.save(output_path, quality=95)
    print(f"Saved: {output_path} ({grid.width}x{grid.height})")

    # Also save individual pages
    for page_path, img in zip(sorted_pages, visualized):
        out_name = page_path.replace('.jpg', '_viz.jpg')
        img.save(out_name, quality=95)
    print(f"Saved individual pages: *_viz.jpg")

if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'ocr_result.json'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'visualized_output.jpg'
    main(json_path, output_path)
