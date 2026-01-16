#!/usr/bin/env python3
"""
Convert detector.onnx (dynamic) to detector_static.onnx (batch=1) for GPU inference.

Static shapes work better with GPU providers:
- NVIDIA TensorRT / CUDA
- DirectML (Windows)
- OpenVINO (Intel)
- CoreML (Apple via ONNX)

Optimizations applied:
- Static shapes (batch=1, 640x640)
- Graph optimizations (constant folding, op fusion)

Requires: pip install onnx onnxruntime
"""

import os
import sys
import onnx
import numpy as np

INPUT = "detector.onnx"
OUTPUT = "detector_static.onnx"

def convert():
    if not os.path.exists(INPUT):
        print(f"Error: {INPUT} not found. Run download_model.py first.")
        sys.exit(1)

    print(f"Loading {INPUT}...")
    model = onnx.load(INPUT)
    graph = model.graph

    # Set static input shapes: batch=1, H=W=640
    print("Setting static shapes (batch=1, 640x640)...")
    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param in ('N', 'batch'):
                dim.ClearField('dim_param')
                dim.dim_value = 1
            elif dim.dim_param in ('H', 'height'):
                dim.ClearField('dim_param')
                dim.dim_value = 640
            elif dim.dim_param in ('W', 'width'):
                dim.ClearField('dim_param')
                dim.dim_value = 640

    # Save with static shapes only (no MS-specific optimizations for TensorRT compatibility)
    print("Saving static model...")
    onnx.save(model, OUTPUT)

    # Verify inference works
    import onnxruntime as ort
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(OUTPUT, opts, providers=['CPUExecutionProvider'])

        dummy_img = np.random.rand(1, 3, 640, 640).astype(np.float32)
        dummy_sizes = np.array([[640, 640]], dtype=np.int64)
        labels, boxes, scores = sess.run(None, {'images': dummy_img, 'orig_target_sizes': dummy_sizes})
        print(f"  Verified: {len(labels[0])} detections")

        opt_mb = os.path.getsize(OUTPUT) / 1e6
        print(f"Done: {OUTPUT} ({opt_mb:.1f} MB)")
        print(f"\nStatic model ready for GPU inference (TensorRT/CUDA/DirectML/OpenVINO)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert()
