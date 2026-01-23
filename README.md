# Manga Translation Server

Automatic manga/comic translation pipeline: detect speech bubbles → OCR → translate → render.

## Features

- **Detection**: ONNX-based bubble/text detection with TensorRT/CoreML acceleration
- **OCR**: Local VLM (Qwen, LFM, Ministral), Gemini API, or OneOCR (Windows)
- **Translation**: Local LLM (HunyuanMT), Cerebras API, or Gemini API
- **Rendering**: Inpaints original text, renders translation with proper formatting

## Quick Start

```bash
# Clone and run setup (creates venv, installs deps, downloads models)
python run.py

# Or run setup separately
python setup.py
python server.py
```

The setup wizard will guide you through configuration options.

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (for local OCR/translation)
- ~4GB disk space for models

### Platform-specific

| Platform | Acceleration | Notes |
|----------|--------------|-------|
| macOS | CoreML + Metal | Auto-detected |
| Linux/Windows + NVIDIA | CUDA + TensorRT | Auto-detected |
| Windows | DirectML | Fallback for AMD/Intel GPUs |
| Windows | OneOCR | Uses Windows Snipping Tool OCR (fastest) |

## API

### Process Images

```bash
curl -X POST http://localhost:5000/api/v1/process \
  -F "images=@manga_page.jpg"
```

### Options

| Parameter | Values | Description |
|-----------|--------|-------------|
| `output_type` | `full_page`, `speech_image_only`, `text_only` | Output format |
| `target_language` | `en`, `zh`, `ko`, `es`, etc. | Translation target |
| `ocr_translate` | `true` | Use VLM for both OCR and translation |
| `translate_local` | `true` | Use local LLM for translation |

### Response

```json
{
  "status": "success",
  "images": ["base64_encoded_image"],
  "stats": {
    "detection_ms": 150,
    "ocr_ms": 800,
    "translate_ms": 200,
    "render_ms": 100
  }
}
```

## Configuration

Run `python config.py` to reconfigure, or edit `config.json` directly:

```json
{
  "ocr_method": "qwen_vlm",
  "translate_method": "hunyuan_mt",
  "target_language": "en",
  "server_port": 5000
}
```

### OCR Methods

| Method | Speed | Quality | Requirements |
|--------|-------|---------|--------------|
| `oneocr` | Fastest | Good | Windows only |
| `gemini_api` | Fast | Best | API key |
| `qwen_vlm` | Medium | Good | llama.cpp + ~4GB VRAM |

### Translation Methods

| Method | Speed | Quality | Requirements |
|--------|-------|---------|--------------|
| `cerebras_api` | Fast | Best | API key |
| `gemini_translate` | Fast | Great | API key |
| `hunyuan_mt` | Medium | Good | llama.cpp |

## Testing

```bash
# Put test images in input/
python tests/test_client.py

# Output saved to output/translated/
```

## Project Structure

```
├── server.py          # Main Flask server
├── config.py          # Configuration management
├── setup.py           # Installation script
├── run.py             # Entry point (setup + run)
├── workflow/          # Processing modules
│   ├── detect.py      # Bubble detection
│   ├── ocr.py         # OCR routing
│   ├── translate.py   # Translation
│   ├── inpaint.py     # Text removal
│   └── render.py      # Text rendering
└── tests/             # Test suite
```

## License

MIT
