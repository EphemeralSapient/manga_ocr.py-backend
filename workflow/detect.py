#!/usr/bin/env python3
"""
Detection Module - Bubble/text region detection for manga pages
Supports: CoreML (native), ONNX+CoreML EP, ONNX CPU
"""

import os
import time
import platform
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional

# Config
COREML_MODEL = "detector.mlpackage"
ONNX_STATIC = "detector_static.onnx"
ONNX_DYNAMIC = "detector.onnx"
THRESHOLD = 0.5
CROP_PADDING = 5

# GPU providers that benefit from static shapes
GPU_PROVIDERS = {
    'NvTensorRTRTXExecutionProvider', 'TensorrtExecutionProvider',
    'CUDAExecutionProvider', 'DmlExecutionProvider',
    'OpenVINOExecutionProvider', 'CoreMLExecutionProvider'
}


def detect_mode() -> str:
    """Auto-detect best inference mode."""
    # Prefer native CoreML on macOS (fastest)
    if platform.system() == 'Darwin' and os.path.exists(COREML_MODEL):
        return 'coreml'

    # Check ONNX Runtime providers
    from .ort import get_best_provider
    provider, _ = get_best_provider()

    # GPU providers work better with static shapes
    if provider in GPU_PROVIDERS and os.path.exists(ONNX_STATIC):
        return 'static'
    if os.path.exists(ONNX_DYNAMIC):
        return 'dynamic'
    if os.path.exists(ONNX_STATIC):
        return 'static'

    raise FileNotFoundError(f"No model found: {COREML_MODEL}, {ONNX_STATIC}, or {ONNX_DYNAMIC}")


def create_session(mode: str) -> Tuple[Any, str]:
    """Create inference session for given mode."""
    if mode == 'coreml':
        import coremltools as ct
        model = ct.models.MLModel(COREML_MODEL)
        print(f"  Mode: coreml | Model: {COREML_MODEL} | Compute: {model.compute_unit}")
        return model, mode

    # Use unified ORT manager
    from .ort import create_session_with_info
    model_path = ONNX_STATIC if mode == 'static' else ONNX_DYNAMIC
    session, info = create_session_with_info(model_path, verbose=False)

    print(f"  Mode: {mode} | Model: {model_path} | Provider: {info['provider']} ({info['provider_name']})")
    return session, mode


def _preprocess(img: Image.Image) -> np.ndarray:
    """Preprocess image for inference."""
    return np.array(img.resize((640, 640))).astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0


def _detect_coreml(model, img: Image.Image, target_label: int) -> List[Dict]:
    """CoreML inference."""
    w, h = img.size
    out = model.predict({'pixel_values': _preprocess(img)})

    logits = out.get('logits', list(out.values())[0])[0]
    boxes = out.get('pred_boxes', list(out.values())[1])[0]
    scores = 1 / (1 + np.exp(-logits[:, target_label]))

    results = []
    for (cx, cy, bw, bh), score in zip(boxes, scores):
        if score > THRESHOLD:
            results.append({
                'box': (int((cx - bw/2) * w), int((cy - bh/2) * h),
                        int((cx + bw/2) * w), int((cy + bh/2) * h)),
                'score': float(score)
            })
    return results


def _detect_onnx(session, img: Image.Image, target_label: int) -> List[Dict]:
    """ONNX single-image inference."""
    w, h = img.size
    labels, boxes, scores = session.run(None, {
        'images': _preprocess(img),
        'orig_target_sizes': np.array([[w, h]], dtype=np.int64)
    })

    return [{'box': tuple(map(int, box)), 'score': float(score)}
            for label, box, score in zip(labels[0], boxes[0], scores[0])
            if score > THRESHOLD and int(label) == target_label]


def _detect_batch(session, images: List[Image.Image], target_label: int) -> List[List[Dict]]:
    """ONNX batch inference."""
    batch = np.stack([_preprocess(img)[0] for img in images])
    sizes = np.array([[img.width, img.height] for img in images], dtype=np.int64)
    labels, boxes, scores = session.run(None, {'images': batch, 'orig_target_sizes': sizes})

    return [[{'box': tuple(map(int, box)), 'score': float(score)}
             for label, box, score in zip(labels[i], boxes[i], scores[i])
             if score > THRESHOLD and int(label) == target_label]
            for i in range(len(images))]


def _crop_padded(img: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int]]:
    """Crop with padding for OCR edge text capture."""
    x1, y1, x2, y2 = box
    pad_l, pad_t = min(CROP_PADDING, x1), min(CROP_PADDING, y1)
    cropped = img.crop((x1 - pad_l, y1 - pad_t, min(img.width, x2 + CROP_PADDING), min(img.height, y2 + CROP_PADDING)))
    return cropped, (pad_l, pad_t)


def detect_all(session, images: List[Image.Image], mode: str, target_label: int = 1) -> Tuple[List[Dict], float]:
    """
    Detect bubbles in all images.

    Returns:
        bubbles: List of {image, page_idx, bubble_idx, box, score, crop_offset}
        detect_time_ms: Detection time in milliseconds
    """
    t0 = time.time()
    bubbles = []

    # Get detections based on mode
    if mode == 'coreml':
        all_dets = [_detect_coreml(session, img, target_label) for img in images]
    elif mode == 'static':
        all_dets = [_detect_onnx(session, img, target_label) for img in images]
    else:
        # Dynamic mode - check if CoreML (doesn't support batch well)
        provider = session.get_providers()[0] if hasattr(session, 'get_providers') else ''
        if 'CoreML' in provider:
            # CoreML doesn't handle batch, use single-image detection
            all_dets = [_detect_onnx(session, img, target_label) for img in images]
        else:
            all_dets = _detect_batch(session, images, target_label)

    # Process detections
    for page_idx, (img, dets) in enumerate(zip(images, all_dets)):
        # Sort: top-to-bottom rows, right-to-left within rows (manga reading order)
        dets.sort(key=lambda d: (d['box'][1] // 100, -d['box'][0]))

        for bubble_idx, det in enumerate(dets):
            crop_img, crop_offset = _crop_padded(img, det['box'])
            bubbles.append({
                'image': crop_img,
                'page_idx': page_idx,
                'bubble_idx': bubble_idx,
                'box': det['box'],
                'score': det['score'],
                'crop_offset': crop_offset
            })

    return bubbles, (time.time() - t0) * 1000


# Convenience exports
__all__ = ['detect_mode', 'create_session', 'detect_all', 'CROP_PADDING', 'THRESHOLD']
