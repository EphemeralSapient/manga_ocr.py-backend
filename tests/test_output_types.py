#!/usr/bin/env python3
"""
Test script for different output types in the Manga Translation Server

Tests the three output modes:
1. full_page (default) - Returns full page images with rendered text
2. speech_image_only - Returns cropped speech bubble images with their positions
3. text_only - Returns only text data without any images
"""

import requests
import base64
import json
import io
from PIL import Image
import os
import sys


def test_full_page_mode(image_path, server_url="http://localhost:1389"):
    """Test full_page output mode (default behavior)"""
    print("\n" + "="*60)
    print("Testing FULL_PAGE mode (default)")
    print("="*60)

    url = f"{server_url}/api/v1/process"

    with open(image_path, 'rb') as f:
        files = {'images': ('test.jpg', f, 'image/jpeg')}
        data = {'output_type': 'full_page'}

        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Output Type: {result.get('output_type', 'full_page')}")
        print(f"Pages: {result['page_count']}")
        print(f"Processing Time: {result['processing_time_ms']}ms")
        print(f"Stats: {json.dumps(result['stats'], indent=2)}")

        # Save output images
        if 'images' in result:
            for i, img_b64 in enumerate(result['images']):
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data))
                output_path = f"output_full_page_{i}.jpg"
                img.save(output_path)
                print(f"Saved full page output: {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def test_speech_image_only_mode(image_path, server_url="http://localhost:1389"):
    """Test speech_image_only output mode"""
    print("\n" + "="*60)
    print("Testing SPEECH_IMAGE_ONLY mode")
    print("="*60)

    url = f"{server_url}/api/v1/process"

    with open(image_path, 'rb') as f:
        files = {'images': ('test.jpg', f, 'image/jpeg')}
        data = {'output_type': 'speech_image_only'}

        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Output Type: {result['output_type']}")
        print(f"Pages: {result['page_count']}")
        print(f"Total Bubbles: {result.get('bubble_count', 0)}")
        print(f"Processing Time: {result['processing_time_ms']}ms")

        # Process bubble images
        if 'pages' in result:
            for page_idx, bubbles in result['pages'].items():
                print(f"\nPage {page_idx}: {len(bubbles)} bubbles")
                for bubble in bubbles:
                    print(f"  Bubble {bubble['bubble_idx']}:")
                    print(f"    Position: {bubble['bubble_box']}")
                    print(f"    Original: {bubble.get('original_text', '')[:50]}...")
                    print(f"    Translated: {bubble.get('translated_text', '')[:50]}...")

                    # Save bubble image
                    if 'image_base64' in bubble:
                        img_data = base64.b64decode(bubble['image_base64'])
                        img = Image.open(io.BytesIO(img_data))
                        output_path = f"output_bubble_p{page_idx}_b{bubble['bubble_idx']}.jpg"
                        img.save(output_path)
                        print(f"    Saved bubble image: {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def test_text_only_mode(image_path, server_url="http://localhost:1389"):
    """Test text_only output mode"""
    print("\n" + "="*60)
    print("Testing TEXT_ONLY mode")
    print("="*60)

    url = f"{server_url}/api/v1/process"

    with open(image_path, 'rb') as f:
        files = {'images': ('test.jpg', f, 'image/jpeg')}
        data = {'output_type': 'text_only'}

        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Output Type: {result['output_type']}")
        print(f"Pages: {result['page_count']}")
        print(f"Processing Time: {result['processing_time_ms']}ms")

        # Process text data
        if 'pages' in result:
            for page_idx, bubbles in result['pages'].items():
                print(f"\nPage {page_idx}: {len(bubbles)} items")
                for bubble in bubbles:
                    print(f"\n  Bubble {bubble['bubble_idx']}:")
                    print(f"    Bounding Box: {bubble['bubble_box']}")
                    print(f"    Original Text: {bubble.get('original_text', '')}")
                    print(f"    Translated: {bubble.get('translated_text', '')}")
                    if 'original_texts' in bubble:
                        print(f"    Text Items: {len(bubble['original_texts'])}")
                    if bubble.get('is_text_free'):
                        print(f"    Type: Text-free region (for inpainting)")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def test_legacy_modes(image_path, server_url="http://localhost:1389"):
    """Test different modes for legacy endpoint (with sequential inpainting)"""
    print("\n" + "="*60)
    print("Testing legacy endpoint with different output modes")
    print("="*60)

    url = f"{server_url}/api/v1/process/legacy"

    for output_type in ['full_page', 'speech_image_only', 'text_only']:
        print(f"\n--- Testing {output_type.upper()} mode for legacy ---")

        with open(image_path, 'rb') as f:
            files = {'images': ('test.jpg', f, 'image/jpeg')}
            data = {'output_type': output_type}

            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Output type: {result.get('output_type')}")
            print(f"Processing time: {result['processing_time_ms']}ms")

            if 'stats' in result:
                print(f"Text bubbles: {result['stats'].get('text_bubbles', 0)}")
                print(f"Text-free regions: {result['stats'].get('text_free_regions', 0)}")
        else:
            print(f"Error: {response.status_code}")


def main():
    # Check for test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for any image in current directory
        for ext in ['.jpg', '.jpeg', '.png']:
            files = [f for f in os.listdir('.') if f.lower().endswith(ext)]
            if files:
                image_path = files[0]
                print(f"Using test image: {image_path}")
                break
        else:
            print("Please provide an image file as argument")
            print("Usage: python test_output_types.py <image_file>")
            return

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Run tests
    server_url = "http://localhost:1389"

    # Check if server is running
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code != 200:
            print("Server health check failed")
            return
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to server at {server_url}")
        print("Please ensure the server is running: python run.py")
        return

    print(f"Testing with image: {image_path}")

    # Test all three modes for main endpoint
    test_full_page_mode(image_path, server_url)
    test_speech_image_only_mode(image_path, server_url)
    test_text_only_mode(image_path, server_url)

    # Test legacy endpoint with different modes
    test_legacy_modes(image_path, server_url)

    print("\n" + "="*60)
    print("All tests completed!")
    print("Check the output files in the current directory.")
    print("="*60)


if __name__ == "__main__":
    main()