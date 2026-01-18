#!/usr/bin/env python3
"""
Text segmentation using comic-text-detector with ONNX Runtime (TensorRT/CUDA optimized).

Provides pixel-level text masks for better inpainting quality compared to bbox-only.
"""
import os
import time
import numpy as np
import cv2
from typing import Tuple, List, Optional

# Default config
CONFIG = {
    "model_path": "../text_seg/comic-text-detector/data/comictextdetector.pt.onnx",
    "input_size": 1024,
    "mask_thresh": 0.3,
    "conf_thresh": 0.4,
    "nms_thresh": 0.35,
}


def letterbox(img, new_shape=(640, 640), auto=False, stride=32):
    """Resize and pad image to new_shape with stride-multiple padding."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, r, (int(dw * 2), int(dh * 2))


def preprocess(img: np.ndarray, input_size: int = 1024) -> Tuple[np.ndarray, float, int, int]:
    """Preprocess image for text segmentation model."""
    # Convert BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Letterbox resize
    img_in, ratio, (dw, dh) = letterbox(img_rgb, new_shape=(input_size, input_size), auto=False, stride=64)

    # HWC to CHW, normalize
    img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_in = np.ascontiguousarray(img_in).astype(np.float32) / 255.0
    img_in = img_in[np.newaxis, ...]  # Add batch dimension

    return img_in, ratio, dw, dh


def postprocess_mask(mask: np.ndarray, dw: int, dh: int, orig_size: Tuple[int, int]) -> np.ndarray:
    """Postprocess mask output to original image size."""
    mask = mask.squeeze()

    # Remove padding
    if dh > 0:
        mask = mask[:-dh, :]
    if dw > 0:
        mask = mask[:, :-dw]

    # Resize to original
    im_h, im_w = orig_size
    mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

    # Threshold and convert to uint8
    mask = (mask * 255).astype(np.uint8)
    return mask


class TextSegmenter:
    """Text segmentation using ONNX Runtime with TensorRT/CUDA support."""

    def __init__(self, model_path: str = None, input_size: int = None):
        self.input_size = input_size or CONFIG["input_size"]
        self.model_path = model_path or CONFIG["model_path"]

        # Resolve relative path
        if not os.path.isabs(self.model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Text segmentation model not found: {self.model_path}")

        # Load with ONNX Runtime (TensorRT/CUDA optimized)
        from .ort import create_session_with_info
        self.sess, info = create_session_with_info(self.model_path, verbose=False)
        self.provider = info['provider']

        # Get output names
        self.output_names = [o.name for o in self.sess.get_outputs()]

        # Stats tracking
        self._call_count = 0
        self._total_time = 0.0
        self.verbose = False

        print(f"[TextSeg] Using ONNX ({info['provider']})")

    def __call__(self, img: np.ndarray, verbose: bool = None) -> np.ndarray:
        """
        Run text segmentation on image.

        Args:
            img: Input image (BGR, HWC format)
            verbose: Override instance verbose setting

        Returns:
            mask: Text segmentation mask (uint8, same size as input)
        """
        t0 = time.time()
        im_h, im_w = img.shape[:2]

        # Preprocess
        img_in, ratio, dw, dh = preprocess(img, self.input_size)

        # Run inference
        outputs = self.sess.run(self.output_names, {'images': img_in})

        # Extract mask (second output typically)
        # Output order: blks, mask, lines_map
        mask = outputs[1] if len(outputs) > 1 else outputs[0]

        # Handle potential reversed outputs (some ONNX versions)
        if mask.shape[1] == 2:
            mask = outputs[2]  # swap with lines_map

        # Postprocess
        result = postprocess_mask(mask, dw, dh, (im_h, im_w))

        # Stats
        elapsed = time.time() - t0
        self._call_count += 1
        self._total_time += elapsed

        if verbose or (verbose is None and self.verbose):
            print(f"    [TextSeg] call {self._call_count}: {im_w}x{im_h}px in {elapsed*1000:.0f}ms")

        return result

    def get_text_mask_for_bbox(self, img: np.ndarray, bbox: List[int],
                                padding: int = 5, verbose: bool = None) -> np.ndarray:
        """
        Get text mask for a specific bbox region.

        Args:
            img: Full image (BGR, HWC)
            bbox: [x1, y1, x2, y2] bounding box
            padding: Padding around bbox for context
            verbose: Override instance verbose setting

        Returns:
            mask: Binary mask for the bbox region (uint8)
        """
        t0 = time.time()
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]

        # Add padding
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)

        # Extract region
        region = img[py1:py2, px1:px2]

        # Run segmentation on region
        mask = self(region, verbose=False)

        # Threshold to binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        elapsed = time.time() - t0
        if verbose or (verbose is None and self.verbose):
            region_size = f"{px2-px1}x{py2-py1}"
            print(f"    [TextSeg] bbox region {region_size}px in {elapsed*1000:.0f}ms")

        return binary_mask

    def reset_stats(self):
        """Reset call counter and timing stats."""
        self._call_count = 0
        self._total_time = 0.0

    def get_stats(self):
        """Get segmentation statistics."""
        return {
            'count': self._call_count,
            'total_ms': int(self._total_time * 1000),
            'avg_ms': int(self._total_time * 1000 / self._call_count) if self._call_count > 0 else 0
        }


# Global instance (lazy loaded)
_text_segmenter = None


def get_text_segmenter(model_path: str = None, input_size: int = None) -> TextSegmenter:
    """Get or create global text segmenter instance."""
    global _text_segmenter
    if _text_segmenter is None:
        _text_segmenter = TextSegmenter(model_path, input_size)
    return _text_segmenter


def reset_text_segmenter():
    """Reset global text segmenter instance."""
    global _text_segmenter
    _text_segmenter = None


def create_text_segmenter(model_path: str = None, input_size: int = None) -> TextSegmenter:
    """Create a new text segmenter instance (for explicit control)."""
    return TextSegmenter(model_path, input_size)
