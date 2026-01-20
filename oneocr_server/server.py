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

# Set DPI awareness to avoid coordinate scaling issues
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    print("[DPI] Set per-monitor DPI awareness")
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # Fallback for older Windows
        print("[DPI] Set process DPI aware (fallback)")
    except Exception:
        print("[DPI] Warning: Could not set DPI awareness")

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
# OneOCR Engine (embedded, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

import struct
import time

MODEL_KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'


class OcrImage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("col", ctypes.c_int32),
        ("row", ctypes.c_int32),
        ("_unk", ctypes.c_int32),
        ("step", ctypes.c_int64),
        ("data_ptr", ctypes.c_int64)
    ]


class BBox(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float)
    ]


def _check_dll_arch(dll_path):
    """Check if DLL matches Python architecture."""
    py_bits = struct.calcsize('P') * 8
    with open(dll_path, 'rb') as f:
        f.seek(0x3C)
        f.seek(struct.unpack('<I', f.read(4))[0] + 4)
        machine = struct.unpack('<H', f.read(2))[0]
    dll_bits = {0x14c: 32, 0x8664: 64, 0xaa64: 64}.get(machine, 0)
    return py_bits == dll_bits


class OneOCR:
    """Embedded OneOCR wrapper - no external dependencies."""

    def __init__(self, ocr_dir, max_lines=1000):
        self.ocr_dir = ocr_dir
        self.dll_path = os.path.join(ocr_dir, 'oneocr.dll')
        self.model_path = None

        # Find model file
        for f in os.listdir(ocr_dir):
            if f.endswith('.onemodel'):
                self.model_path = os.path.join(ocr_dir, f)
                break

        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"DLL not found: {self.dll_path}")
        if not self.model_path:
            raise FileNotFoundError(f"No .onemodel file found in {ocr_dir}")
        if not _check_dll_arch(self.dll_path):
            raise RuntimeError("DLL/Python architecture mismatch (need 64-bit Python for 64-bit DLL)")

        # Load DLL
        os.add_dll_directory(ocr_dir)
        self._dll = ctypes.WinDLL(self.dll_path)
        self._setup_dll()
        self._init(max_lines)
        print(f"[OK] OneOCR initialized: {self.dll_path}")

    def _setup_dll(self):
        d = self._dll
        i64 = ctypes.c_int64
        pi64 = ctypes.POINTER(ctypes.c_int64)

        d.CreateOcrInitOptions.argtypes = [pi64]
        d.CreateOcrInitOptions.restype = i64
        d.OcrInitOptionsSetUseModelDelayLoad.argtypes = [i64, ctypes.c_char]
        d.OcrInitOptionsSetUseModelDelayLoad.restype = i64
        d.CreateOcrPipeline.argtypes = [ctypes.c_char_p, ctypes.c_char_p, i64, pi64]
        d.CreateOcrPipeline.restype = i64
        d.CreateOcrProcessOptions.argtypes = [pi64]
        d.CreateOcrProcessOptions.restype = i64
        d.OcrProcessOptionsSetMaxRecognitionLineCount.argtypes = [i64, i64]
        d.OcrProcessOptionsSetMaxRecognitionLineCount.restype = i64
        d.RunOcrPipeline.argtypes = [i64, ctypes.POINTER(OcrImage), i64, pi64]
        d.RunOcrPipeline.restype = i64
        d.GetOcrLineCount.argtypes = [i64, pi64]
        d.GetOcrLineCount.restype = i64
        d.GetOcrLine.argtypes = [i64, i64, pi64]
        d.GetOcrLine.restype = i64
        d.GetOcrLineContent.argtypes = [i64, pi64]
        d.GetOcrLineContent.restype = i64
        d.GetOcrLineBoundingBox.argtypes = [i64, pi64]
        d.GetOcrLineBoundingBox.restype = i64

    def _init(self, max_lines):
        self._ctx = ctypes.c_int64()
        self._pipeline = ctypes.c_int64()
        self._opt = ctypes.c_int64()

        self._dll.CreateOcrInitOptions(ctypes.byref(self._ctx))
        self._dll.OcrInitOptionsSetUseModelDelayLoad(self._ctx.value, ctypes.c_char(0))
        self._dll.CreateOcrPipeline(
            self.model_path.encode(),
            MODEL_KEY,
            self._ctx.value,
            ctypes.byref(self._pipeline)
        )
        self._dll.CreateOcrProcessOptions(ctypes.byref(self._opt))
        self._dll.OcrProcessOptionsSetMaxRecognitionLineCount(self._opt.value, max_lines)

    def _prepare_image(self, image_path):
        """Prepare image for OCR."""
        pil = Image.open(image_path)
        if pil.mode != 'RGBA':
            pil = pil.convert('RGBA')

        # Convert RGB to BGR for Windows OCR
        r, g, b, a = pil.split()
        bgra = Image.merge('RGBA', (b, g, r, a))

        # Create buffer
        buf = ctypes.create_string_buffer(bgra.tobytes())
        ocr_img = OcrImage(3, bgra.width, bgra.height, 0, bgra.width * 4, ctypes.addressof(buf))
        return ocr_img, buf

    def recognize(self, image_path):
        """Run OCR on image file.

        Args:
            image_path: Path to image file

        Returns:
            List of dicts with 'text', 'bbox', 'confidence'
        """
        t0 = time.time()
        ocr_img, data = self._prepare_image(image_path)

        result = ctypes.c_int64()
        self._dll.RunOcrPipeline(
            self._pipeline.value,
            ctypes.byref(ocr_img),
            self._opt.value,
            ctypes.byref(result)
        )

        count = ctypes.c_int64()
        self._dll.GetOcrLineCount(result.value, ctypes.byref(count))

        lines = []
        for i in range(count.value):
            line = ctypes.c_int64()
            if self._dll.GetOcrLine(result.value, i, ctypes.byref(line)) != 0:
                continue

            # Get text
            txt_ptr = ctypes.c_int64()
            self._dll.GetOcrLineContent(line.value, ctypes.byref(txt_ptr))
            try:
                text = ctypes.cast(txt_ptr.value, ctypes.c_char_p).value.decode('utf-8')
            except:
                continue

            # Get bounding box
            box_ptr = ctypes.c_int64()
            self._dll.GetOcrLineBoundingBox(line.value, ctypes.byref(box_ptr))
            if box_ptr.value:
                b = ctypes.cast(box_ptr.value, ctypes.POINTER(BBox)).contents
                bbox = [b.x, b.y, b.x + b.width, b.y + b.height]
            else:
                bbox = [0, 0, 0, 0]

            lines.append({
                'text': text,
                'bbox': bbox,
                'confidence': 1.0
            })

        elapsed = (time.time() - t0) * 1000
        print(f"[OCR] Recognized {len(lines)} lines in {elapsed:.0f}ms")

        # Debug: Check for out-of-bounds coordinates
        pil = Image.open(image_path)
        img_w, img_h = pil.size
        out_of_bounds = [l for l in lines if l['bbox'][2] > img_w or l['bbox'][3] > img_h]
        if out_of_bounds:
            print(f"[OCR] WARNING: {len(out_of_bounds)} lines have bbox outside image ({img_w}x{img_h}):")
            for l in out_of_bounds[:5]:
                print(f"  text='{l['text'][:20]}' bbox={l['bbox']}")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# OneOCR Instance
# ─────────────────────────────────────────────────────────────────────────────

_ocr_engine = None


def _get_ocr_engine():
    """Get or create OneOCR engine instance."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    _ocr_engine = OneOCR(OCR_DIR)
    return _ocr_engine


def _run_oneocr(image_path: str) -> list:
    """Run OneOCR on an image file.

    Args:
        image_path: Path to image file

    Returns:
        List of dicts with 'text', 'confidence', 'bbox'
    """
    ocr = _get_ocr_engine()
    return ocr.recognize(image_path)


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
    import traceback

    try:
        image = None
        print(f"[OCR] Received request, content_type={request.content_type}")

        # Handle JSON request with base64 image
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                print(f"[OCR] Decoding base64 image ({len(data['image'])} chars)")
                image_data = base64.b64decode(data['image'])
                image = Image.open(BytesIO(image_data))
                print(f"[OCR] Image loaded: {image.size}, mode={image.mode}")

        # Handle form data with file upload
        elif 'image' in request.files:
            print("[OCR] Loading from form file upload")
            image = Image.open(request.files['image'])

        # Handle raw image data
        elif request.data:
            print(f"[OCR] Loading from raw data ({len(request.data)} bytes)")
            image = Image.open(BytesIO(request.data))

        if image is None:
            print("[OCR] ERROR: No image provided in request")
            return jsonify({'error': 'No image provided'}), 400

        # Convert to RGB if needed
        if image.mode != 'RGB':
            print(f"[OCR] Converting from {image.mode} to RGB")
            image = image.convert('RGB')

        # Run OCR using local OneOCR wrapper (no external deps needed)
        try:
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp_path)
                print(f"[OCR] Saved temp image: {tmp_path}")
                print(f"[OCR] Running OneOCR...")
                results = _run_oneocr(tmp_path)
                print(f"[OCR] Got {len(results) if results else 0} results")
            finally:
                # Clean up temp file (ignore errors - Windows may hold the file)
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass  # File will be cleaned up by OS later

            return jsonify({
                'success': True,
                'results': results
            })

        except Exception as e:
            print(f"[OCR] ERROR during OCR: {type(e).__name__}: {e}")
            print(f"[OCR] Traceback:\n{traceback.format_exc()}")
            return jsonify({
                'error': str(e),
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            }), 500

    except Exception as e:
        print(f"[OCR] ERROR in request handling: {type(e).__name__}: {e}")
        print(f"[OCR] Traceback:\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
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

        import tempfile

        for i, img_b64 in enumerate(images):
            tmp_path = None
            try:
                image_data = base64.b64decode(img_b64)
                image = Image.open(BytesIO(image_data))

                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Use local OCR wrapper (no external deps)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp_path)
                ocr_result = _run_oneocr(tmp_path)

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
            finally:
                # Clean up temp file (ignore errors)
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

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
