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


def get_crop_with_mask(img, bbox, pixel_mask, pad=5):
    """Extract crop region with a custom pixel-level mask.

    Args:
        img: Full image array (H, W, 3)
        bbox: [x1, y1, x2, y2] bounding box
        pixel_mask: Pixel-level mask for the bbox region (uint8, 0-255)
        pad: Padding around bbox

    Returns:
        crop: Cropped image region with context
        mask: Float mask (0-1) positioned in the crop
        coords: (cy1, cy2, cx1, cx2) coordinates for placing result back
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    cp = CONFIG["context_pad"]
    cx1, cy1 = max(0, x1-cp), max(0, y1-cp)
    cx2, cy2 = min(w, x2+cp), min(h, y2+cp)

    crop = img[cy1:cy2, cx1:cx2].astype(np.float32)
    mask = np.zeros(crop.shape[:2], np.float32)

    # Place the pixel mask into the crop region
    # pixel_mask is for the bbox region, we need to position it correctly
    bbox_h, bbox_w = y2 - y1, x2 - x1
    mask_h, mask_w = pixel_mask.shape[:2]

    # Resize pixel_mask if needed to match bbox size (using PIL)
    if mask_h != bbox_h or mask_w != bbox_w:
        mask_pil = Image.fromarray(pixel_mask)
        mask_pil = mask_pil.resize((bbox_w, bbox_h), Image.BILINEAR)
        pixel_mask = np.array(mask_pil)

    # Convert to float and place in correct position
    mask_float = pixel_mask.astype(np.float32) / 255.0
    mask[y1-cy1:y2-cy1, x1-cx1:x2-cx1] = mask_float

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
    def __init__(self):
        self._call_count = 0
        self._total_time = 0.0
        self.verbose = False

    def __call__(self, img, bbox, verbose=None):
        t0 = time.time()
        crop, mask, coords = get_crop(img, bbox)
        padded, msk, h, w = prep(crop, mask)
        out = self.forward(padded, msk)
        result = composite(crop, mask, out, h, w)

        elapsed = time.time() - t0
        self._call_count += 1
        self._total_time += elapsed

        # Log if verbose (instance or call-level)
        if verbose or (verbose is None and self.verbose):
            bbox_size = f"{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}"
            print(f"    [Inpaint] region {self._call_count}: {bbox_size}px in {elapsed*1000:.0f}ms")

        return coords, result

    def inpaint_with_mask(self, img, bbox, pixel_mask, verbose=None):
        """Inpaint using a custom pixel-level mask instead of rectangular bbox.

        Args:
            img: Full image array (H, W, 3)
            bbox: [x1, y1, x2, y2] bounding box for context cropping
            pixel_mask: Pixel-level mask (uint8, 0-255) - same size as bbox or will be resized
            verbose: If True, log timing info

        Returns:
            coords: (cy1, cy2, cx1, cx2) for placing result
            result: Inpainted region
        """
        t0 = time.time()
        crop, mask, coords = get_crop_with_mask(img, bbox, pixel_mask)
        padded, msk, h, w = prep(crop, mask)
        out = self.forward(padded, msk)
        result = composite(crop, mask, out, h, w)

        elapsed = time.time() - t0
        self._call_count += 1
        self._total_time += elapsed

        if verbose or (verbose is None and self.verbose):
            bbox_size = f"{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}"
            mask_pixels = int(np.sum(pixel_mask > 127))
            print(f"    [Inpaint] region {self._call_count}: {bbox_size}px ({mask_pixels} mask px) in {elapsed*1000:.0f}ms")

        return coords, result

    def reset_stats(self):
        """Reset call counter and timing stats."""
        self._call_count = 0
        self._total_time = 0.0

    def get_stats(self):
        """Get inpainting statistics."""
        return {
            'count': self._call_count,
            'total_ms': int(self._total_time * 1000),
            'avg_ms': int(self._total_time * 1000 / self._call_count) if self._call_count > 0 else 0
        }

class PyTorchInpainter(Inpainter):
    def __init__(self, path, device):
        super().__init__()
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        self.model = torch.jit.load(path, map_location='cpu').float().to(self.dev).eval()
        if device == 'mps':
            with torch.no_grad():  # warmup (needs 256+ for padding layers)
                self.model(torch.zeros(1,3,256,256).to(self.dev), torch.zeros(1,1,256,256).to(self.dev))
            torch.mps.synchronize()
        print(f"[Inpaint] Using PyTorch {device.upper()}")

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
        super().__init__()
        from .ort import create_session_with_info
        self.sess, info = create_session_with_info(path, verbose=False)
        self.provider = info['provider']
        print(f"[Inpaint] Using ONNX ({info['provider']})")

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
