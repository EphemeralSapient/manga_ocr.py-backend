#!/usr/bin/env python3
"""AOT-GAN manga inpainting. Auto-selects: MPS > CUDA > CPU > ONNX"""
import json, time
import numpy as np
from PIL import Image
from pathlib import Path

CONFIG = {
    "model_pt": "aot_traced.pt",
    "model_onnx": "aot.onnx",
    "ocr_json": "output/ocr_result_label2.json",
    "output_dir": "output/inpainted",
    "quality": 95,
    "context_pad": 64,
}

def get_crop(img, bbox, pad=5):
    """Extract crop region around bbox with context padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    cp = CONFIG["context_pad"]
    cx1, cy1 = max(0, x1-cp), max(0, y1-cp)
    cx2, cy2 = min(w, x2+cp), min(h, y2+cp)
    crop = img[cy1:cy2, cx1:cx2].astype(np.float32)
    mask = np.zeros(crop.shape[:2], np.float32)
    mask[y1-cy1:y2-cy1, x1-cx1:x2-cx1] = 1.0
    return crop, mask, (cy1, cy2, cx1, cx2)

def prep(crop, mask):
    """Prepare input tensors (pad to mult of 8, zero mask region)."""
    h, w = crop.shape[:2]
    nh, nw = (h+7)//8*8, (w+7)//8*8
    img = np.zeros((nh, nw, 3), np.float32)
    msk = np.zeros((nh, nw), np.float32)
    img[:h,:w], msk[:h,:w] = crop, mask
    img[msk > 0.5] = 0
    return img, msk, h, w

def composite(crop, mask, out, h, w):
    """Blend model output with original."""
    out = np.clip((out[:h,:w] + 1) * 127.5, 0, 255)
    m3 = np.stack([mask]*3, -1)
    return (crop * (1-m3) + out * m3).astype(np.uint8)

class Inpainter:
    """Base inpainter with common logic."""
    def __call__(self, img, bbox):
        crop, mask, coords = get_crop(img, bbox)
        padded, msk, h, w = prep(crop, mask)
        out = self.forward(padded, msk)
        return coords, composite(crop, mask, out, h, w)

class PyTorchInpainter(Inpainter):
    def __init__(self, path, device):
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        self.model = torch.jit.load(path, map_location='cpu').float().to(self.dev).eval()
        if device == 'mps':
            with torch.no_grad():  # warmup (needs 256+ for padding layers)
                self.model(torch.zeros(1,3,256,256).to(self.dev), torch.zeros(1,1,256,256).to(self.dev))
            torch.mps.synchronize()
        print(f"Using PyTorch {device.upper()}")

    def forward(self, img, mask):
        t = self.torch
        img_t = t.from_numpy((img/255).transpose(2,0,1)[None]).float().to(self.dev)
        msk_t = t.from_numpy(mask[None,None]).float().to(self.dev)
        with t.no_grad():
            out = self.model(img_t, msk_t)
        if self.dev.type == 'mps':
            t.mps.synchronize()
        return out[0].cpu().numpy().transpose(1,2,0)

class ONNXInpainter(Inpainter):
    def __init__(self, path):
        from .ort import create_session_with_info
        self.sess, info = create_session_with_info(path, verbose=False)
        print(f"Using ONNX ({info['provider']})")

    def forward(self, img, mask):
        img_t = (img/255).transpose(2,0,1)[None].astype(np.float32)
        msk_t = mask[None,None].astype(np.float32)
        return self.sess.run(['inpainted'], {'image': img_t, 'mask': msk_t})[0][0].transpose(1,2,0)

def create_inpainter():
    """Auto-select best backend: MPS > ONNX(TensorRT/CUDA) > PyTorch"""
    import os

    # Check for ONNX model (preferred for GPU - TensorRT/CUDA/DirectML)
    has_onnx = os.path.exists(CONFIG["model_onnx"])
    has_pt = os.path.exists(CONFIG["model_pt"])

    try:
        import torch
        # macOS MPS - use PyTorch (fastest)
        if torch.backends.mps.is_available() and has_pt:
            return PyTorchInpainter(CONFIG["model_pt"], 'mps')

        # NVIDIA GPU - prefer ONNX for TensorRT acceleration
        if torch.cuda.is_available():
            if has_onnx:
                return ONNXInpainter(CONFIG["model_onnx"])
            elif has_pt:
                return PyTorchInpainter(CONFIG["model_pt"], 'cuda')

        # CPU fallback - PyTorch
        if has_pt:
            return PyTorchInpainter(CONFIG["model_pt"], 'cpu')
    except ImportError:
        pass

    # ONNX fallback (DirectML/OpenVINO or no PyTorch)
    if has_onnx:
        return ONNXInpainter(CONFIG["model_onnx"])

    raise FileNotFoundError(f"No inpainting model found: {CONFIG['model_pt']} or {CONFIG['model_onnx']}")

def main():
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inpaint = create_inpainter()

    with open(CONFIG["ocr_json"]) as f:
        ocr = json.load(f)

    total = 0
    for path, bubbles in ocr.items():
        img = np.array(Image.open(path).convert("RGB"))
        t0 = time.time()
        for b in bubbles:
            coords, result = inpaint(img, b["bubble_box"])
            img[coords[0]:coords[1], coords[2]:coords[3]] = result
        elapsed = time.time() - t0
        total += elapsed

        out_path = out_dir / f"{Path(path).stem}_inpainted.jpg"
        Image.fromarray(img).save(out_path, quality=CONFIG["quality"])
        print(f"{path}: {len(bubbles)} regions in {elapsed:.2f}s")

    print(f"Total: {total:.2f}s")

if __name__ == "__main__":
    main()
