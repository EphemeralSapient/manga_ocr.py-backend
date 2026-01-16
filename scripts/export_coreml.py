#!/usr/bin/env python3
"""
Export RT-DETR-v2 to native CoreML (.mlpackage) for maximum Apple Silicon performance.

Requires: pip install transformers torch coremltools

Performance (FP32):
- Native CoreML: ~60ms/page (99.3% GPU/ANE utilization)
- ONNX + CoreML EP: ~400ms/page (77% GPU utilization)
- ONNX CPU: ~580ms/page
- Speedup: ~9.5x faster than CPU with native CoreML

Usage:
    # Export model (removes ONNX files after)
    python export_coreml.py

    # Export but keep ONNX files
    python export_coreml.py --keep-onnx

    # Use for detection
    from export_coreml import detect
    import coremltools as ct
    from PIL import Image

    model = ct.models.MLModel('detector.mlpackage')
    image = Image.open('page.jpg')
    boxes = detect(model, image, class_id=1)  # 1=text_bubble
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

REPO_ID = "ogkalu/comic-text-and-bubble-detector"
OUTPUT = "detector.mlpackage"


def detect(model, image, class_id=1, threshold=0.5):
    """
    Run detection on an image using CoreML model.

    Args:
        model: CoreML MLModel (loaded with ct.models.MLModel)
        image: PIL Image
        class_id: 0=bubble, 1=text_bubble, 2=text_free
        threshold: detection threshold (default 0.5)

    Returns:
        list of (box, score) where box is [x1, y1, x2, y2]
    """
    orig_w, orig_h = image.size
    arr = np.array(image.resize((640, 640))).astype(np.float32)
    arr = arr.transpose(2, 0, 1) / 255.0
    arr = arr[np.newaxis, ...]

    out = model.predict({'pixel_values': arr})

    # Get outputs by name or fall back to positional
    if 'logits' in out:
        logits = out['logits'][0]          # [300, 3]
        boxes_norm = out['pred_boxes'][0]  # [300, 4] cx,cy,w,h normalized
    else:
        values = list(out.values())
        logits = values[0][0] if values[0].shape[-1] == 3 else values[1][0]
        boxes_norm = values[1][0] if values[1].shape[-1] == 4 else values[0][0]

    # Sigmoid scoring (matches ONNX output exactly)
    scores = 1 / (1 + np.exp(-logits[:, class_id]))

    mask = scores > threshold
    results = []
    for box, score in zip(boxes_norm[mask], scores[mask]):
        cx, cy, w, h = box
        x1 = int((cx - w/2) * orig_w)
        y1 = int((cy - h/2) * orig_h)
        x2 = int((cx + w/2) * orig_w)
        y2 = int((cy + h/2) * orig_h)
        results.append(([x1, y1, x2, y2], float(score)))

    return sorted(results, key=lambda x: -x[1])

class RTDetrWrapper(nn.Module):
    """Wrapper to output simple tensors instead of complex dataclass"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        # Return raw logits and normalized boxes
        return outputs.logits, outputs.pred_boxes

def export():
    from transformers import RTDetrV2ForObjectDetection
    import coremltools as ct

    print(f"Loading model from {REPO_ID}...")
    base_model = RTDetrV2ForObjectDetection.from_pretrained(REPO_ID)
    base_model.eval()

    model = RTDetrWrapper(base_model)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.1f}M")

    print("Exporting with torch.export...")
    dummy = torch.randn(1, 3, 640, 640)
    exported = torch.export.export(model, (dummy,))
    decomposed = exported.run_decompositions({})
    print("  Done")

    print("Converting to CoreML (FP32)...")
    mlmodel = ct.convert(
        decomposed,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, 640, 640))],
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="pred_boxes")],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13
    )

    print(f"Saving {OUTPUT}...")
    mlmodel.save(OUTPUT)

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, dn, fn in os.walk(OUTPUT) for f in fn) / 1e6
    print(f"Done: {OUTPUT} ({size:.1f} MB)")
    print()
    print("Usage:")
    print("  from export_coreml import detect")
    print("  import coremltools as ct")
    print("  from PIL import Image")
    print()
    print("  model = ct.models.MLModel('detector.mlpackage')")
    print("  boxes = detect(model, Image.open('page.jpg'), class_id=1)")
    print()
    print("Classes: 0=bubble, 1=text_bubble, 2=text_free")

def cleanup_onnx():
    """Remove ONNX files after CoreML export (no longer needed on macOS)."""
    onnx_files = ['detector.onnx', 'detector_static.onnx']
    removed = []
    for f in onnx_files:
        if os.path.exists(f):
            os.remove(f)
            removed.append(f)
    if removed:
        print(f"Cleaned up: {', '.join(removed)}")
    return removed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Export RT-DETR-v2 to CoreML')
    parser.add_argument('--keep-onnx', action='store_true',
                        help='Keep ONNX files after export')
    args = parser.parse_args()

    export()

    if not args.keep_onnx:
        cleanup_onnx()
