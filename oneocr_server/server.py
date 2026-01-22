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


class BBoxQuad(ctypes.Structure):
    """Bounding box as a quad (4 corner points = 8 floats).

    The DLL returns bbox as 4 corner points in order:
    - (x1, y1): top-left
    - (x2, y2): top-right
    - (x3, y3): bottom-right
    - (x4, y4): bottom-left
    """
    _fields_ = [
        ("x1", ctypes.c_float), ("y1", ctypes.c_float),  # top-left
        ("x2", ctypes.c_float), ("y2", ctypes.c_float),  # top-right
        ("x3", ctypes.c_float), ("y3", ctypes.c_float),  # bottom-right
        ("x4", ctypes.c_float), ("y4", ctypes.c_float),  # bottom-left
    ]

    def to_rect(self):
        """Convert quad to axis-aligned bounding rectangle [x1, y1, x2, y2]."""
        xs = [self.x1, self.x2, self.x3, self.x4]
        ys = [self.y1, self.y2, self.y3, self.y4]
        return [min(xs), min(ys), max(xs), max(ys)]


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

        # Line style (returns 2 int32 values - may indicate text orientation)
        d.GetOcrLineStyle.argtypes = [i64, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        d.GetOcrLineStyle.restype = i64

        # Word-level functions
        d.GetOcrLineWordCount.argtypes = [i64, pi64]
        d.GetOcrLineWordCount.restype = i64

        d.GetOcrWord.argtypes = [i64, i64, pi64]
        d.GetOcrWord.restype = i64

        d.GetOcrWordBoundingBox.argtypes = [i64, pi64]
        d.GetOcrWordBoundingBox.restype = i64

        d.GetOcrWordContent.argtypes = [i64, pi64]
        d.GetOcrWordContent.restype = i64

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

        # Get image dimensions BEFORE processing for DPI scaling fix
        pil = Image.open(image_path)
        img_w, img_h = pil.size

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
        max_x = 0
        max_y = 0

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

            # Get line style (may indicate vertical/horizontal orientation)
            style1 = ctypes.c_int32(0)
            style2 = ctypes.c_int32(0)
            style_res = self._dll.GetOcrLineStyle(line.value, ctypes.byref(style1), ctypes.byref(style2))
            is_vertical = False
            if style_res == 0:
                # Check if style indicates vertical text
                # style1 might be: 0=horizontal, 1=vertical (need to verify)
                is_vertical = (style1.value == 1)
                print(f"[OCR STYLE] '{text[:15]}': style1={style1.value}, style2={style2.value}, vertical={is_vertical}")

            # Get line bounding box (quad format: 8 floats = 4 corner points)
            box_ptr = ctypes.c_int64()
            self._dll.GetOcrLineBoundingBox(line.value, ctypes.byref(box_ptr))
            bbox = [0, 0, 0, 0]
            if box_ptr.value:
                quad = ctypes.cast(box_ptr.value, ctypes.POINTER(BBoxQuad)).contents
                bbox = quad.to_rect()
                print(f"[OCR BBOX] '{text[:15]}': quad corners=({quad.x1:.0f},{quad.y1:.0f}),({quad.x2:.0f},{quad.y2:.0f}),({quad.x3:.0f},{quad.y3:.0f}),({quad.x4:.0f},{quad.y4:.0f}) -> rect=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")

            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

            lines.append({
                'text': text,
                'bbox': bbox,
                'confidence': 1.0,
            })

        # Clamp bboxes to image dimensions
        print(f"[OCR DEBUG] Image: {img_w}x{img_h}, max_x={max_x:.1f}, max_y={max_y:.1f}")
        for line in lines:
            bbox = line['bbox']
            line['bbox'] = [
                max(0, min(bbox[0], img_w)),
                max(0, min(bbox[1], img_h)),
                max(0, min(bbox[2], img_w)),
                max(0, min(bbox[3], img_h)),
            ]

        elapsed = (time.time() - t0) * 1000
        print(f"[OCR] Recognized {len(lines)} lines in {elapsed:.0f}ms")

        # Final check - any boxes still outside bounds?
        for line in lines:
            bbox = line['bbox']
            if bbox[2] > img_w or bbox[3] > img_h:
                print(f"[OCR WARNING] Box still outside: {bbox} for '{line['text'][:20]}'")

        # Debug: Check for remaining out-of-bounds coordinates
        out_of_bounds = [l for l in lines if l['bbox'][2] > img_w + 1 or l['bbox'][3] > img_h + 1]
        if out_of_bounds:
            print(f"[OCR] WARNING: {len(out_of_bounds)} lines still have bbox outside image ({img_w}x{img_h}):")
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


def _save_debug_image(image, results, debug_dir, prefix="ocr_debug"):
    """Save debug image with OCR bounding boxes drawn."""
    from PIL import ImageDraw, ImageFont
    import datetime

    os.makedirs(debug_dir, exist_ok=True)

    # Create a copy to draw on
    debug_img = image.copy()
    if debug_img.mode != 'RGB':
        debug_img = debug_img.convert('RGB')

    draw = ImageDraw.Draw(debug_img)

    # Try to load a font
    font = None
    font_size = 10
    try:
        font_paths = [
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
    except:
        pass

    # Draw each OCR result with current interpretation
    for i, r in enumerate(results):
        bbox = r.get('bbox', [0, 0, 0, 0])
        text = r.get('text', '')[:20]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw bbox rectangle (green = current)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"[{i}]", fill=(0, 255, 0), font=font)

    # Save with timestamp
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    debug_path = os.path.join(debug_dir, f"{prefix}_{timestamp}_{image.width}x{image.height}.png")
    debug_img.save(debug_path)
    print(f"[OCR Debug] Saved: {debug_path}")
    return debug_path


def _save_debug_image_all_variants(image, raw_results, debug_dir, prefix="ocr_variants"):
    """Save debug image showing ALL possible bbox interpretations."""
    from PIL import ImageDraw, ImageFont
    import datetime

    os.makedirs(debug_dir, exist_ok=True)

    # Create a copy to draw on
    debug_img = image.copy()
    if debug_img.mode != 'RGB':
        debug_img = debug_img.convert('RGB')

    draw = ImageDraw.Draw(debug_img)

    # Try to load a font
    font = None
    font_size = 9
    try:
        font_paths = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msgothic.ttc"]
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
    except:
        pass

    # Color scheme for different interpretations
    COLORS = {
        'A': (255, 0, 0),     # Red: x,y,x+w,y+h (normal)
        'B': (0, 255, 0),     # Green: x,y,x+h,y+w (swapped)
        'C': (0, 0, 255),     # Blue: x,y,w,h (direct/no addition)
        'D': (255, 255, 0),   # Yellow: y,x,h+y,w+x (x/y swapped)
    }

    for i, r in enumerate(raw_results):
        x = r['raw_x']
        y = r['raw_y']
        w = r['raw_w']
        h = r['raw_h']
        text = r.get('text', '')[:15]

        # Interpretation A: x, y, x+w, y+h (standard)
        try:
            ax1, ay1, ax2, ay2 = int(x), int(y), int(x+w), int(y+h)
            if 0 <= ax1 < image.width and 0 <= ay1 < image.height:
                draw.rectangle([ax1, ay1, min(ax2, image.width-1), min(ay2, image.height-1)],
                              outline=COLORS['A'], width=2)
                draw.text((ax1+2, ay1+2), f"A{i}", fill=COLORS['A'], font=font)
        except: pass

        # Interpretation B: x, y, x+h, y+w (swapped w/h)
        try:
            bx1, by1, bx2, by2 = int(x), int(y), int(x+h), int(y+w)
            if 0 <= bx1 < image.width and 0 <= by1 < image.height:
                draw.rectangle([bx1, by1, min(bx2, image.width-1), min(by2, image.height-1)],
                              outline=COLORS['B'], width=2)
                draw.text((bx1+2, by1+15), f"B{i}", fill=COLORS['B'], font=font)
        except: pass

        # Interpretation C: x, y, w, h as x1,y1,x2,y2 directly (no addition)
        try:
            cx1, cy1, cx2, cy2 = int(x), int(y), int(w), int(h)
            if 0 <= cx1 < image.width and 0 <= cy1 < image.height and cx2 > cx1 and cy2 > cy1:
                draw.rectangle([cx1, cy1, min(cx2, image.width-1), min(cy2, image.height-1)],
                              outline=COLORS['C'], width=2)
                draw.text((cx1+2, cy1+28), f"C{i}", fill=COLORS['C'], font=font)
        except: pass

        # Interpretation D: treat as center + half dimensions? x-w/2, y-h/2, x+w/2, y+h/2
        try:
            dx1, dy1, dx2, dy2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            if 0 <= dx1 < image.width and 0 <= dy1 < image.height:
                draw.rectangle([max(0,dx1), max(0,dy1), min(dx2, image.width-1), min(dy2, image.height-1)],
                              outline=COLORS['D'], width=1)
                draw.text((max(0,dx1)+2, max(0,dy1)+41), f"D{i}", fill=COLORS['D'], font=font)
        except: pass

    # Add legend
    legend_y = 5
    draw.text((5, legend_y), "A=x,y,x+w,y+h (RED)", fill=COLORS['A'], font=font)
    draw.text((5, legend_y+12), "B=x,y,x+h,y+w (GREEN/swap)", fill=COLORS['B'], font=font)
    draw.text((5, legend_y+24), "C=x,y,w,h direct (BLUE)", fill=COLORS['C'], font=font)
    draw.text((5, legend_y+36), "D=center+half (YELLOW)", fill=COLORS['D'], font=font)

    # Save
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    debug_path = os.path.join(debug_dir, f"{prefix}_{timestamp}_{image.width}x{image.height}.png")
    debug_img.save(debug_path)
    print(f"[OCR Variants] Saved: {debug_path}")
    return debug_path


# Global debug flag - can be toggled via /debug endpoint
_ocr_debug_enabled = False


@app.route('/debug', methods=['GET', 'POST'])
def toggle_debug():
    """Toggle or check debug mode."""
    global _ocr_debug_enabled
    if request.method == 'POST':
        data = request.get_json() if request.is_json else {}
        if 'enabled' in data:
            _ocr_debug_enabled = bool(data['enabled'])
        else:
            _ocr_debug_enabled = not _ocr_debug_enabled
        print(f"[OCR Debug] Debug mode: {'ENABLED' if _ocr_debug_enabled else 'DISABLED'}")
    return jsonify({
        'debug_enabled': _ocr_debug_enabled,
        'debug_dir': os.path.join(SCRIPT_DIR, 'debug_output')
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    OCR endpoint - process image and return text.

    Accepts:
        - JSON with base64 encoded image: {"image": "base64...", "debug": true}
        - Form data with image file: files['image']
        - Query param: ?debug=1

    Returns:
        JSON with OCR results
    """
    import traceback

    try:
        image = None
        debug_this_request = _ocr_debug_enabled
        print(f"[OCR] Received request, content_type={request.content_type}")

        # Check for debug flag in query string
        if request.args.get('debug', '').lower() in ('1', 'true', 'yes'):
            debug_this_request = True

        # Handle JSON request with base64 image
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                print(f"[OCR] Decoding base64 image ({len(data['image'])} chars)")
                image_data = base64.b64decode(data['image'])
                image = Image.open(BytesIO(image_data))
                print(f"[OCR] Image loaded: {image.size}, mode={image.mode}")
            # Check debug flag in JSON
            if data.get('debug'):
                debug_this_request = True

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

            # Save debug image if enabled
            debug_path = None
            if debug_this_request and results:
                debug_dir = os.path.join(SCRIPT_DIR, 'debug_output')
                os.makedirs(debug_dir, exist_ok=True)

                import datetime
                import json as json_module
                timestamp = datetime.datetime.now().strftime("%H%M%S")

                # Save the original "before" image (raw input)
                before_path = os.path.join(debug_dir, f"before_{timestamp}_{image.width}x{image.height}.png")
                image.save(before_path)
                print(f"[OCR Debug] Saved before: {before_path}")

                # Save INCOMING request details as JSON (exclude image data)
                incoming_request = {
                    'timestamp': timestamp,
                    'method': request.method,
                    'content_type': request.content_type,
                    'content_length': request.content_length,
                    'args': dict(request.args),
                    'headers': {k: v for k, v in request.headers if k.lower() not in ['cookie', 'authorization']},
                    'json_keys': list(request.get_json(silent=True).keys()) if request.is_json else None,
                    'debug_flag': debug_this_request,
                    'image_width': image.width,
                    'image_height': image.height,
                    'image_mode': image.mode,
                }
                request_json_path = os.path.join(debug_dir, f"incoming_{timestamp}_{image.width}x{image.height}.json")
                with open(request_json_path, 'w', encoding='utf-8') as f:
                    json_module.dump(incoming_request, f, indent=2, ensure_ascii=False)
                print(f"[OCR Debug] Saved incoming request: {request_json_path}")

                debug_path = _save_debug_image(image, results, debug_dir)

            response = {
                'success': True,
                'results': results
            }
            if debug_path:
                response['debug_image'] = debug_path

            return jsonify(response)

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
    print("  GET  /health       - Health check")
    print("  POST /ocr          - OCR single image (?debug=1 for debug output)")
    print("  POST /ocr/batch    - OCR multiple images")
    print("  GET  /debug        - Check debug status")
    print("  POST /debug        - Toggle debug mode (saves bbox images)")
    print("-" * 60)
    print(f"\nDebug output: {os.path.join(SCRIPT_DIR, 'debug_output')}")
    print("-" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    # Run server
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
