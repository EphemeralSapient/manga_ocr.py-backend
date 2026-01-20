#!/usr/bin/env python3
"""
OneOCR Remote Server - Expose Windows OneOCR over HTTP

Run this on a Windows machine (local or Azure VM) to provide OCR services
to other machines on the network.

Usage:
    python server.py [--port PORT] [--host HOST]

The server will:
1. Check/setup OneOCR files (requires admin on first run)
2. Display IP address and port for clients to connect
3. Expose /ocr endpoint for image processing
"""

import os
import sys
import socket
import argparse
import base64
import json
import platform
import ctypes
import shutil
from io import BytesIO

# Check Windows
if platform.system() != 'Windows':
    print("ERROR: OneOCR server only runs on Windows")
    print("This machine is running:", platform.system())
    sys.exit(1)

# Flask for HTTP server
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Installing Flask...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', '-q'])
    from flask import Flask, request, jsonify

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'Pillow', '-q'])
    from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# OneOCR Setup
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_DIR = os.path.join(SCRIPT_DIR, 'ocr')


def is_admin():
    """Check if running as administrator."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def setup_oneocr():
    """Copy OneOCR files from Windows Snipping Tool."""
    dll_path = os.path.join(OCR_DIR, 'oneocr.dll')

    if os.path.exists(dll_path):
        print(f"[OK] OneOCR files found in {OCR_DIR}")
        return True

    print("\n" + "=" * 60)
    print("OneOCR Setup Required")
    print("=" * 60)

    if not is_admin():
        print("\nERROR: Administrator access required to copy OneOCR files.")
        print("\nPlease run this script as Administrator:")
        print("  1. Right-click Command Prompt or PowerShell")
        print("  2. Select 'Run as administrator'")
        print(f"  3. Run: python {os.path.basename(__file__)}")
        return False

    # Find Snipping Tool
    base = "C:\\Program Files\\WindowsApps"
    src = None

    try:
        for folder in os.listdir(base):
            if folder.startswith("Microsoft.ScreenSketch") and "x64" in folder:
                snip = os.path.join(base, folder, "SnippingTool")
                if os.path.exists(snip):
                    src = snip
                    break
    except PermissionError:
        print("ERROR: Cannot access WindowsApps folder")
        return False

    if not src:
        print("ERROR: Windows Snipping Tool not found")
        print("Please ensure Snipping Tool is installed from Microsoft Store")
        return False

    print(f"Found Snipping Tool: {src}")

    # Copy files
    os.makedirs(OCR_DIR, exist_ok=True)
    copied = 0

    for filename in os.listdir(src):
        if filename.endswith('.dll') or filename.endswith('.onemodel'):
            src_path = os.path.join(src, filename)
            dst_path = os.path.join(OCR_DIR, filename)
            shutil.copy2(src_path, dst_path)
            copied += 1
            print(f"  Copied: {filename}")

    print(f"\n[OK] Copied {copied} OneOCR files to {OCR_DIR}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# OneOCR Engine
# ─────────────────────────────────────────────────────────────────────────────

_ocr_engine = None


def get_ocr_engine():
    """Get or create OneOCR engine instance."""
    global _ocr_engine

    if _ocr_engine is not None:
        return _ocr_engine

    # Add OCR directory to DLL search path
    os.add_dll_directory(OCR_DIR)

    # Import and initialize OneOCR
    try:
        # Try to import the oneocr module
        sys.path.insert(0, SCRIPT_DIR)
        from oneocr_wrapper import OneOCR
        _ocr_engine = OneOCR(OCR_DIR)
        print("[OK] OneOCR engine initialized")
        return _ocr_engine
    except Exception as e:
        print(f"ERROR: Failed to initialize OneOCR: {e}")
        raise


class OneOCREngine:
    """Simple OneOCR wrapper using ctypes."""

    def __init__(self, ocr_dir):
        self.ocr_dir = ocr_dir
        os.add_dll_directory(ocr_dir)

        # Load the DLL
        dll_path = os.path.join(ocr_dir, 'oneocr.dll')
        self.dll = ctypes.CDLL(dll_path)

        # Find model file
        self.model_path = None
        for f in os.listdir(ocr_dir):
            if f.endswith('.onemodel'):
                self.model_path = os.path.join(ocr_dir, f)
                break

        if not self.model_path:
            raise FileNotFoundError("No .onemodel file found")

        self._initialized = False

    def recognize(self, image):
        """
        Recognize text in image.

        Args:
            image: PIL Image or path to image file

        Returns:
            List of dicts with 'text', 'confidence', 'bbox'
        """
        # This is a placeholder - actual implementation depends on OneOCR API
        # The real implementation would call the DLL functions
        raise NotImplementedError("Use oneocr_wrapper.py for actual OCR")


# ─────────────────────────────────────────────────────────────────────────────
# Flask Server
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'oneocr',
        'platform': platform.system(),
        'version': '1.0.0'
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    OCR endpoint - process image and return text.

    Accepts:
        - JSON with base64 encoded image: {"image": "base64..."}
        - Form data with image file: files['image']

    Returns:
        JSON with OCR results
    """
    try:
        image = None

        # Handle JSON request with base64 image
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                image_data = base64.b64decode(data['image'])
                image = Image.open(BytesIO(image_data))

        # Handle form data with file upload
        elif 'image' in request.files:
            image = Image.open(request.files['image'])

        # Handle raw image data
        elif request.data:
            image = Image.open(BytesIO(request.data))

        if image is None:
            return jsonify({'error': 'No image provided'}), 400

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run OCR using the workflow module
        try:
            # Import the actual OneOCR implementation from workflow
            sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
            from workflow.ocr_oneocr import recognize_text

            # Save temp image for OCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                results = recognize_text(tmp.name, ocr_dir=OCR_DIR)
                os.unlink(tmp.name)

            return jsonify({
                'success': True,
                'results': results
            })

        except ImportError:
            # Fallback: use simple OCR wrapper
            return jsonify({
                'error': 'OneOCR wrapper not available',
                'hint': 'Ensure oneocr_wrapper.py is present'
            }), 500

    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/ocr/batch', methods=['POST'])
def ocr_batch():
    """
    Batch OCR endpoint - process multiple images.

    Accepts JSON:
        {"images": ["base64...", "base64...", ...]}

    Returns:
        JSON with list of OCR results
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON required'}), 400

        data = request.get_json()
        images = data.get('images', [])

        if not images:
            return jsonify({'error': 'No images provided'}), 400

        results = []

        for i, img_b64 in enumerate(images):
            try:
                image_data = base64.b64decode(img_b64)
                image = Image.open(BytesIO(image_data))

                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Import OCR module
                sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
                from workflow.ocr_oneocr import recognize_text

                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name)
                    ocr_result = recognize_text(tmp.name, ocr_dir=OCR_DIR)
                    os.unlink(tmp.name)

                results.append({
                    'index': i,
                    'success': True,
                    'results': ocr_result
                })

            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


def get_local_ip():
    """Get the local IP address."""
    try:
        # Connect to external address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description='OneOCR Remote Server')
    parser.add_argument('--port', type=int, default=5050, help='Port to run server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--setup-only', action='store_true', help='Only setup OneOCR files, then exit')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("OneOCR Remote Server")
    print("=" * 60)

    # Setup OneOCR files
    if not setup_oneocr():
        sys.exit(1)

    if args.setup_only:
        print("\nSetup complete. Run without --setup-only to start server.")
        sys.exit(0)

    # Get IP address
    local_ip = get_local_ip()

    print("\n" + "-" * 60)
    print("Server Configuration")
    print("-" * 60)
    print(f"  Local URL:    http://127.0.0.1:{args.port}")
    print(f"  Network URL:  http://{local_ip}:{args.port}")
    print("-" * 60)
    print("\nUse this URL in your config.json:")
    print(f'  "oneocr_server_url": "http://{local_ip}:{args.port}"')
    print("-" * 60)
    print("\nEndpoints:")
    print("  GET  /health     - Health check")
    print("  POST /ocr        - OCR single image")
    print("  POST /ocr/batch  - OCR multiple images")
    print("-" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    # Run server
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
