"""OneOCR - ctypes wrapper for oneocr.dll"""

import ctypes
import os
import struct
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

MODEL_KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'


class OcrImage(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("col", ctypes.c_int32), ("row", ctypes.c_int32),
                ("_unk", ctypes.c_int32), ("step", ctypes.c_int64), ("data_ptr", ctypes.c_int64)]


class BBox(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),
                ("width", ctypes.c_float), ("height", ctypes.c_float)]


def _check_arch(dll_path):
    """Check if DLL matches Python architecture."""
    py_bits = struct.calcsize('P') * 8
    with open(dll_path, 'rb') as f:
        f.seek(0x3C)
        f.seek(struct.unpack('<I', f.read(4))[0] + 4)
        machine = struct.unpack('<H', f.read(2))[0]
    dll_bits = {0x14c: 32, 0x8664: 64, 0xaa64: 64}.get(machine, 0)
    return py_bits == dll_bits


class OneOCR:
    def __init__(self, dll_path=None, model_path=None, max_lines=1000):
        base = Path(__file__).parent
        self.dll_path = Path(dll_path) if dll_path else base / "oneocr.dll"
        self.model_path = Path(model_path) if model_path else base / "oneocr.onemodel"

        if not self.dll_path.exists():
            raise FileNotFoundError(f"DLL not found: {self.dll_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not _check_arch(self.dll_path):
            raise RuntimeError("DLL/Python architecture mismatch")

        os.add_dll_directory(str(self.dll_path.parent))
        self._dll = ctypes.WinDLL(str(self.dll_path))
        self._setup_dll()
        self._init(max_lines)

    def _setup_dll(self):
        d = self._dll
        i64, pi64, pf = ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float)
        d.CreateOcrInitOptions.argtypes, d.CreateOcrInitOptions.restype = [pi64], i64
        d.OcrInitOptionsSetUseModelDelayLoad.argtypes, d.OcrInitOptionsSetUseModelDelayLoad.restype = [i64, ctypes.c_char], i64
        d.CreateOcrPipeline.argtypes, d.CreateOcrPipeline.restype = [ctypes.c_char_p, ctypes.c_char_p, i64, pi64], i64
        d.CreateOcrProcessOptions.argtypes, d.CreateOcrProcessOptions.restype = [pi64], i64
        d.OcrProcessOptionsSetMaxRecognitionLineCount.argtypes, d.OcrProcessOptionsSetMaxRecognitionLineCount.restype = [i64, i64], i64
        d.RunOcrPipeline.argtypes, d.RunOcrPipeline.restype = [i64, ctypes.POINTER(OcrImage), i64, pi64], i64
        d.GetOcrLineCount.argtypes, d.GetOcrLineCount.restype = [i64, pi64], i64
        d.GetOcrLine.argtypes, d.GetOcrLine.restype = [i64, i64, pi64], i64
        d.GetOcrLineContent.argtypes, d.GetOcrLineContent.restype = [i64, pi64], i64
        d.GetOcrLineBoundingBox.argtypes, d.GetOcrLineBoundingBox.restype = [i64, pi64], i64

    def _init(self, max_lines):
        self._ctx = ctypes.c_int64()
        self._pipeline = ctypes.c_int64()
        self._opt = ctypes.c_int64()

        self._dll.CreateOcrInitOptions(ctypes.byref(self._ctx))
        self._dll.OcrInitOptionsSetUseModelDelayLoad(self._ctx.value, ctypes.c_char(0))
        self._dll.CreateOcrPipeline(str(self.model_path).encode(), MODEL_KEY, self._ctx.value, ctypes.byref(self._pipeline))
        self._dll.CreateOcrProcessOptions(ctypes.byref(self._opt))
        self._dll.OcrProcessOptionsSetMaxRecognitionLineCount(self._opt.value, max_lines)

    def _prepare(self, image):
        if cv2 is not None:
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            elif isinstance(image, np.ndarray):
                img = image
            else:
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img = np.ascontiguousarray(img)
            ocr_img = OcrImage(3, img.shape[1], img.shape[0], 0, img.strides[0], img.ctypes.data)
            return ocr_img, img
        else:
            pil = Image.open(str(image)) if isinstance(image, (str, Path)) else image
            if pil.mode != 'RGBA':
                pil = pil.convert('RGBA')
            r, g, b, a = pil.split()
            bgra = Image.merge('RGBA', (b, g, r, a))
            buf = ctypes.create_string_buffer(bgra.tobytes())
            ocr_img = OcrImage(3, bgra.width, bgra.height, 0, bgra.width * 4, ctypes.addressof(buf))
            return ocr_img, buf

    def run(self, image):
        t0 = time.time()
        ocr_img, data = self._prepare(image)

        result = ctypes.c_int64()
        self._dll.RunOcrPipeline(self._pipeline.value, ctypes.byref(ocr_img), self._opt.value, ctypes.byref(result))

        count = ctypes.c_int64()
        self._dll.GetOcrLineCount(result.value, ctypes.byref(count))

        lines = []
        for i in range(count.value):
            line = ctypes.c_int64()
            if self._dll.GetOcrLine(result.value, i, ctypes.byref(line)) != 0:
                continue

            txt_ptr = ctypes.c_int64()
            self._dll.GetOcrLineContent(line.value, ctypes.byref(txt_ptr))
            try:
                text = ctypes.cast(txt_ptr.value, ctypes.c_char_p).value.decode('utf-8')
            except:
                continue

            box_ptr = ctypes.c_int64()
            self._dll.GetOcrLineBoundingBox(line.value, ctypes.byref(box_ptr))
            if box_ptr.value:
                b = ctypes.cast(box_ptr.value, ctypes.POINTER(BBox)).contents
                bbox = {'x': b.x, 'y': b.y, 'width': b.width, 'height': b.height}
            else:
                bbox = {'x': 0, 'y': 0, 'width': 0, 'height': 0}

            lines.append({'text': text, 'bbox': bbox})

        return {'lines': lines, 'line_count': len(lines), 'processing_time_ms': (time.time() - t0) * 1000}


def check_ocr_available():
    try:
        base = Path(__file__).parent
        return (base / "oneocr.dll").exists() and (base / "oneocr.onemodel").exists() and _check_arch(base / "oneocr.dll")
    except:
        return False
