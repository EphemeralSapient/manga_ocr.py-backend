# Translation Mode Options

The Manga Translation Server supports three translation modes for the `/translate/label1` endpoint.

## Translation Modes

### 1. VLM OCR + Translate (Combined)

**Flag:** `ocr_translate=true`

The VLM (Vision Language Model) performs both OCR and translation in a single inference step.

| Pros | Cons |
|------|------|
| Single model inference | Translation quality depends on VLM |
| No separate translation step | Less control over translation |
| Works fully offline | May produce repetitive translations |

**Request:**
```python
data = {'ocr_translate': 'true'}
```

**CLI:**
```bash
python tests/test_client.py --ocr-translate
```

---

### 2. VLM OCR + Local LLM Translate

**Flag:** `translate_local=true`

VLM performs OCR (extracts Japanese text), then a local LLM (HunyuanMT) translates to English.

| Pros | Cons |
|------|------|
| Better translation quality | Requires two llama-server instances |
| Works fully offline | Slightly slower than combined mode |
| Dedicated translation model | More GPU memory usage |

**Request:**
```python
data = {'translate_local': 'true'}
```

**CLI:**
```bash
python tests/test_client.py --translate-local
```

**Requires:**
- OCR server: `llama-server -hf Qwen/Qwen3-VL-2B-Instruct-GGUF --port 8080`
- Translate server: `llama-server -hf tencent/HY-MT1.5-1.8B-GGUF --port 8081`

---

### 3. VLM OCR + Cerebras API Translate (Default)

**Flag:** None (default behavior)

VLM performs OCR, then Cerebras cloud API translates to English.

| Pros | Cons |
|------|------|
| Best translation quality | Requires API key |
| Fast cloud inference | Requires internet connection |
| No local translation model | API costs may apply |

**Request:**
```python
data = {}  # default, or pass api_key
data = {'api_key': 'your-cerebras-api-key'}
```

**CLI:**
```bash
python tests/test_client.py
```

**Requires:**
- Environment variable: `CEREBRAS_API_KEY=your-key`
- Or pass `api_key` in request

**Note:** Server returns 400 error if no API key is available and neither `ocr_translate` nor `translate_local` is set.

---

## Quick Reference

| Mode | Flags | OCR Model | Translate Model | Offline |
|------|-------|-----------|-----------------|---------|
| Combined | `ocr_translate=true` | Qwen3-VL-2B | (same) | Yes |
| Local | `translate_local=true` | Qwen3-VL-2B | HY-MT1.5-1.8B | Yes |
| API | (default) | Qwen3-VL-2B | Cerebras API | No |

## API Usage Examples

### Python

```python
import requests

url = "http://localhost:1389/translate/label1"

with open('manga_page.jpg', 'rb') as f:
    files = {'images': ('page.jpg', f, 'image/jpeg')}

    # Mode 1: VLM OCR+Translate
    data = {'ocr_translate': 'true'}
    response = requests.post(url, files=files, data=data)

    # Mode 2: VLM OCR + Local LLM Translate
    data = {'translate_local': 'true'}
    response = requests.post(url, files=files, data=data)

    # Mode 3: VLM OCR + Cerebras API (default)
    data = {'api_key': 'your-key'}  # or set CEREBRAS_API_KEY env var
    response = requests.post(url, files=files, data=data)
```

### cURL

```bash
# Mode 1: VLM OCR+Translate
curl -X POST http://localhost:1389/translate/label1 \
  -F "images=@manga_page.jpg" \
  -F "ocr_translate=true"

# Mode 2: VLM OCR + Local LLM Translate
curl -X POST http://localhost:1389/translate/label1 \
  -F "images=@manga_page.jpg" \
  -F "translate_local=true"

# Mode 3: VLM OCR + Cerebras API
curl -X POST http://localhost:1389/translate/label1 \
  -F "images=@manga_page.jpg" \
  -F "api_key=your-cerebras-key"
```

## Response Stats

The response includes a `translate_method` field indicating which mode was used:

```json
{
  "stats": {
    "translate_method": "vlm",        // ocr_translate mode
    "translate_method": "local_llm",  // translate_local mode
    "translate_method": "cerebras_api" // default mode
  }
}
```

## Configuration

### Interactive Setup

Run the configuration wizard:

```bash
python setup.py --configure
```

This creates a `config.json` file with your settings:

```json
{
  "translate_mode": "local",
  "ocr_server_url": "http://localhost:8080",
  "translate_server_url": "http://localhost:8081",
  "vlm_model": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
  "translate_model": "tencent/HY-MT1.5-1.8B-GGUF",
  "cerebras_api_key": "",
  "auto_start_servers": true
}
```

### Config Options

| Option | Description | Values |
|--------|-------------|--------|
| `translate_mode` | Default translation mode | `vlm`, `local`, `api` |
| `ocr_server_url` | VLM OCR server URL | URL string |
| `translate_server_url` | Local LLM translate server | URL string |
| `vlm_model` | HuggingFace model for OCR | Model ID |
| `translate_model` | HuggingFace model for translation | Model ID |
| `cerebras_api_key` | API key for Cerebras | Key string |
| `auto_start_servers` | Auto-start llama servers | `true`/`false` |

## Environment Variables

Environment variables override config file settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `CEREBRAS_API_KEY` | API key for Cerebras translation | (from config) |
| `LLAMA_SERVER_URL` | OCR VLM server URL | `http://localhost:8080` |
| `TRANSLATE_SERVER_URL` | Local translate server URL | `http://localhost:8081` |
| `VLM_MODEL` | OCR model HuggingFace ID | `Qwen/Qwen3-VL-2B-Instruct-GGUF` |
| `TRANSLATE_MODEL` | Translate model HuggingFace ID | `tencent/HY-MT1.5-1.8B-GGUF` |
