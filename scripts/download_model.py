#!/usr/bin/env python3
"""Download models from HuggingFace

Downloads:
  - detector.onnx: Comic text/bubble detector (RT-DETR-v2)
  - aot_traced.pt: AOT inpainting model (macOS/CPU - PyTorch)
  - aot.onnx: AOT inpainting model (GPU - TensorRT/CUDA/DirectML)
"""

import urllib.request
import os
import sys
import platform
import shutil

MODELS = {
    "detector.onnx": "https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector.onnx",
}

AOT_PT_URL = "https://huggingface.co/ogkalu/aot-inpainting/resolve/main/aot_traced.pt"
AOT_ONNX_URL = "https://huggingface.co/ogkalu/aot-inpainting/resolve/main/aot.onnx"


def download_file(url, output):
    """Download a file with progress bar"""
    if os.path.exists(output):
        print(f"{output} already exists ({os.path.getsize(output) / 1e6:.1f} MB)")
        return False

    print(f"Downloading {output}...")

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        mb = count * block_size / 1e6
        total_mb = total_size / 1e6
        sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output, reporthook=progress)
    print(f"\nSaved: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
    return True


def detect_inpaint_backend():
    """Detect best inpainting backend: pt for macOS/CPU, onnx for GPU."""
    # macOS uses PyTorch MPS (fastest)
    if platform.system() == 'Darwin':
        return 'pt'

    # NVIDIA GPU or Windows (DirectML) → use ONNX for GPU acceleration
    if shutil.which('nvidia-smi') or platform.system() == 'Windows':
        return 'onnx'

    # CPU fallback → PyTorch
    return 'pt'


def download():
    # Download detector model
    for output, url in MODELS.items():
        download_file(url, output)

    # Download AOT inpainting model based on platform
    backend = detect_inpaint_backend()

    if backend == 'onnx':
        print(f"\nUsing ONNX inpainting (GPU accelerated)")
        download_file(AOT_ONNX_URL, "aot.onnx")
    else:
        print(f"\nUsing PyTorch inpainting (MPS/CPU)")
        download_file(AOT_PT_URL, "aot_traced.pt")


if __name__ == "__main__":
    download()
