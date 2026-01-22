#!/usr/bin/env python3
"""
Test client for Manga Translation Server
Sends all images from input folder and saves results to output folder.
"""

import os
import sys
import json
import glob
import base64
import time
import requests
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load config from centralized config module
try:
    from config import get_server_port, load_config
    load_config()
    SERVER_PORT = get_server_port()
except ImportError:
    # Fallback to direct file read
    def _get_port():
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    return json.load(f).get('server_port', 5000)
            except:
                pass
        return 5000
    SERVER_PORT = _get_port()

SERVER_URL = f"http://localhost:{SERVER_PORT}"
INPUT_DIR = "input"
OUTPUT_DIR = "output/translated"


def get_input_images():
    """Get all input images, excluding viz/debug files."""
    patterns = [
        f"{INPUT_DIR}/page_*.jpg",
        f"{INPUT_DIR}/page_*.png",
        f"{INPUT_DIR}/*.jpg",
        f"{INPUT_DIR}/*.png",
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    # Filter out viz, debug, detected files
    exclude_keywords = ['viz', 'debug', 'detected', 'fixed', 'compare', 'test']
    filtered = []
    for f in all_files:
        basename = os.path.basename(f).lower()
        if not any(kw in basename for kw in exclude_keywords):
            filtered.append(f)

    # Remove duplicates and sort
    filtered = sorted(set(filtered), key=lambda x: (
        int(''.join(filter(str.isdigit, os.path.basename(x))) or '0'),
        x
    ))

    return filtered


def send_request(endpoint: str, image_paths: list, ocr_translate: bool = False,
                 translate_local: bool = False, inpaint_background: bool = True) -> dict:
    """Send images to server and return response."""
    url = f"{SERVER_URL}{endpoint}"

    files = []
    for path in image_paths:
        files.append(('images', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')))

    # Form data for options
    data = {}
    if ocr_translate:
        data['ocr_translate'] = 'true'
    if translate_local:
        data['translate_local'] = 'true'
    # AOT inpainting (default True)
    data['inpaint_background'] = 'true' if inpaint_background else 'false'

    try:
        mode_parts = []
        if ocr_translate:
            mode_parts.append("ocr_translate")
        elif translate_local:
            mode_parts.append("translate_local")
        if inpaint_background:
            mode_parts.append("AOT inpaint")
        mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
        print(f"Sending {len(image_paths)} images to {endpoint}{mode_str}...")
        start = time.time()
        response = requests.post(url, files=files, data=data, timeout=300)
        elapsed = time.time() - start

        # Close file handles
        for _, (_, f, _) in files:
            f.close()

        if response.status_code == 200:
            data = response.json()
            print(f"  Response received in {elapsed:.2f}s (server: {data.get('processing_time_ms', 0)}ms)")
            return data
        else:
            print(f"  Error: {response.status_code} - {response.text}")
            return {'error': response.text}

    except requests.exceptions.ConnectionError:
        print(f"  Error: Cannot connect to server at {SERVER_URL}")
        print(f"  Make sure server is running: python3.12 server.py")
        return {'error': 'Connection failed'}
    except Exception as e:
        print(f"  Error: {e}")
        return {'error': str(e)}


def print_stats(response: dict):
    """Print detailed processing stats."""
    stats = response.get('stats', {})
    if not stats:
        return

    print("\n" + "=" * 50)
    print("PROCESSING STATS")
    print("=" * 50)

    # Main timings
    print(f"\n  Pages:           {stats.get('pages', 'N/A')}")
    print(f"  Bubbles:         {stats.get('bubbles_detected', 'N/A')}")
    print(f"  Texts:           {stats.get('texts_to_translate', 'N/A')}")
    print(f"  OCR Lines:       {stats.get('ocr_lines', 'N/A')}")

    print(f"\n  TIMINGS:")
    print(f"    Detection:     {stats.get('detect_ms', 'N/A')}ms")
    if stats.get('label2_detected'):
        print(f"    Text-free Regions:{stats.get('label2_detected', 0)}")
    if stats.get('inpaint_l2_ms'):
        print(f"    Inpaint L2:    {stats.get('inpaint_l2_ms', 'N/A')}ms (parallel with OCR)")
    if stats.get('inpaint_l1_ms'):
        print(f"    Inpaint L1:    {stats.get('inpaint_l1_ms', 'N/A')}ms (text areas)")
    print(f"    OCR:           {stats.get('ocr_ms', 'N/A')}ms")
    print(f"    Translation:   {stats.get('translation_ms', 'N/A')}ms")
    print(f"    Render:        {stats.get('render_ms', 'N/A')}ms")
    print(f"    Encode:        {stats.get('encode_ms', 'N/A')}ms")
    print(f"    ─────────────────────")
    print(f"    TOTAL:         {stats.get('total_ms', 'N/A')}ms")

    # Translation details
    print(f"\n  TRANSLATION:")
    print(f"    Method:        {stats.get('translate_method', 'N/A')}")
    print(f"    Success:       {stats.get('translations_success', 'N/A')}")
    print(f"    Failed:        {stats.get('translations_failed', 'N/A')}")
    print(f"    Batches:       {stats.get('translation_batches', 'N/A')}")

    # Batch details
    batch_details = stats.get('translation_batch_details', [])
    if batch_details:
        print(f"\n  BATCH DETAILS:")
        for batch in batch_details:
            status = batch.get('status', 'unknown')
            time_ms = batch.get('time_ms', 0)
            count = batch.get('count', 0)
            print(f"    Batch {batch.get('batch_idx', '?')}: {count} texts, {time_ms}ms - {status}")

    print("=" * 50)


def save_images(response: dict, image_paths: list, output_dir: str):
    """Save base64 images from response."""
    if 'error' in response:
        return

    results = response.get('results', [])
    if not results:
        print("  No images in response")
        return

    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for i, result in enumerate(results):
        if not result.get('success') or not result.get('data_url'):
            continue

        # Extract base64 data from data URL (remove "data:image/png;base64," prefix)
        data_url = result['data_url']
        if ',' in data_url:
            b64_data = data_url.split(',', 1)[1]
        else:
            b64_data = data_url

        if i < len(image_paths):
            basename = os.path.basename(image_paths[i])
            name, ext = os.path.splitext(basename)
            output_name = f"{name}_translated{ext}"
        else:
            output_name = f"page_{i+1}_translated.jpg"

        output_path = os.path.join(output_dir, output_name)

        img_data = base64.b64decode(b64_data)
        with open(output_path, 'wb') as f:
            f.write(img_data)
        saved_count += 1

    print(f"\n  Saved {saved_count} images to {output_dir}/")


def main():
    # Parse arguments
    endpoint = "/api/v1/process"
    ocr_translate = False
    translate_local = False
    inpaint_background = True  # AOT inpainting enabled by default

    args = sys.argv[1:]
    for arg in args:
        if arg in ['--legacy', '-2', 'legacy']:
            endpoint = "/api/v1/process/legacy"
        elif arg in ['--ocr-translate', '--vlm', '-v']:
            ocr_translate = True
        elif arg in ['--translate-local', '--local', '-l']:
            translate_local = True
        elif arg in ['--no-inpaint', '--no-aot', '-n']:
            inpaint_background = False
        elif arg in ['--help', '-h']:
            print("Usage: python3.12 test_client.py [options]")
            print("")
            print("Options:")
            print("  --legacy           Legacy sequential inpainting mode")
            print("  --no-inpaint       Disable AOT inpainting (white background only)")
            print("  --ocr-translate    Use VLM for combined OCR+translate (requires llama.cpp)")
            print("  --translate-local  Use local LLM for translation (HY-MT via llama.cpp)")
            print("")
            print("Modes:")
            print("  Default:           VLM OCR -> API translate + AOT inpaint (parallel)")
            print("  --ocr-translate:   VLM OCR+translate in one step + AOT inpaint")
            print("  --translate-local: VLM OCR -> local LLM translate + AOT inpaint")
            print("  --legacy:          Sequential processing mode")
            print("  --no-inpaint:      Disable background inpainting (white out text only)")
            print("")
            print(f"Input:  {INPUT_DIR}/")
            print(f"Output: {OUTPUT_DIR}/")
            return

    # Get input images
    image_paths = get_input_images()
    if not image_paths:
        print(f"No images found in {INPUT_DIR}/")
        return

    print(f"Found {len(image_paths)} images:")
    for p in image_paths[:5]:
        print(f"  {p}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")
    print()

    # Check server health
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"Server not healthy: {health.status_code}")
            return
    except:
        print(f"Cannot connect to server at {SERVER_URL}")
        print(f"Start the server first: python3.12 server.py")
        return

    print(f"Server is running at {SERVER_URL}")
    print(f"Using endpoint: {endpoint}")
    if ocr_translate:
        print(f"Using OCR+Translate mode (VLM)")
    elif translate_local:
        print(f"Using VLM OCR + Local LLM Translation")
    else:
        print(f"Using VLM OCR + API Translation (see server config)")
    if inpaint_background:
        print(f"AOT Inpainting: Enabled (parallel with OCR/translate)")
    else:
        print(f"AOT Inpainting: Disabled (white background only)")
    print()

    # Send request
    response = send_request(endpoint, image_paths, ocr_translate=ocr_translate,
                           translate_local=translate_local, inpaint_background=inpaint_background)

    # Print stats
    if 'stats' in response:
        print_stats(response)

    # Save results
    if 'results' in response:
        if ocr_translate:
            suffix = "_vlm"
        elif translate_local:
            suffix = "_local"
        else:
            suffix = ""
        if inpaint_background:
            suffix += "_inpaint"
        output_subdir = OUTPUT_DIR + ("_legacy" if "legacy" in endpoint else "") + suffix
        save_images(response, image_paths, output_subdir)

        print()
        print(f"Done! {response.get('page_count', 0)} pages processed")
        print(f"Output saved to: {output_subdir}/")


if __name__ == '__main__':
    main()
