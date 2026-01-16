# API Output Types Documentation

The Manga Translation Server now supports three different output types for both `/translate/label1` and `/translate/label2` endpoints.

## Output Types

### 1. `full_page` (Default)
Returns the complete page images with translated text rendered on them.

**Use Case:** When you want the final translated manga pages ready for viewing.

**Request:**
```python
data = {'output_type': 'full_page'}  # or omit for default
```

**Response Structure:**
```json
{
  "status": "success",
  "output_type": "full_page",
  "page_count": 1,
  "processing_time_ms": 2500,
  "stats": {...},
  "images": ["base64_encoded_full_page_1", "base64_encoded_full_page_2", ...]
}
```

### 2. `speech_image_only`
Returns individual speech bubble images with their positions on the original page.

**Use Case:** When you need individual bubble images for further processing or custom rendering.

**Request:**
```python
data = {'output_type': 'speech_image_only'}
```

**Response Structure:**
```json
{
  "status": "success",
  "output_type": "speech_image_only",
  "page_count": 1,
  "bubble_count": 5,
  "processing_time_ms": 2000,
  "stats": {...},
  "pages": {
    "0": [
      {
        "bubble_idx": 0,
        "bubble_box": [x1, y1, x2, y2],  // Position on original page
        "image_base64": "base64_encoded_bubble_image",
        "original_text": "Original Japanese text",
        "translated_text": "Translated English text"
      },
      ...
    ]
  }
}
```

### 3. `text_only`
Returns only the text data without any images. Skips rendering for maximum speed.

**Use Case:** When you only need the OCR and translation results for analysis or custom rendering.

**Request:**
```python
data = {'output_type': 'text_only'}
```

**Response Structure:**
```json
{
  "status": "success",
  "output_type": "text_only",
  "page_count": 1,
  "processing_time_ms": 1500,
  "stats": {...},
  "pages": {
    "0": [
      {
        "bubble_idx": 0,
        "bubble_box": [x1, y1, x2, y2],
        "original_texts": [
          {"text": "こんにちは", "confidence": 0.95, ...}
        ],
        "original_text": "こんにちは",
        "translated_text": "Hello"
      },
      ...
    ]
  }
}
```

## API Usage Examples

### Python Example

```python
import requests
import base64
from PIL import Image
import io

# Server URL
url = "http://localhost:1389/translate/label1"

# Read image
with open('manga_page.jpg', 'rb') as f:
    files = {'images': ('page.jpg', f, 'image/jpeg')}

    # Example 1: Full page output (default)
    response = requests.post(url, files=files)

    # Example 2: Speech bubbles only
    data = {'output_type': 'speech_image_only'}
    response = requests.post(url, files=files, data=data)

    # Example 3: Text only (fastest)
    data = {'output_type': 'text_only'}
    response = requests.post(url, files=files, data=data)

# Process response
result = response.json()
if result['status'] == 'success':
    if result['output_type'] == 'full_page':
        # Save full page images
        for i, img_b64 in enumerate(result['images']):
            img_data = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_data))
            img.save(f'output_page_{i}.jpg')

    elif result['output_type'] == 'speech_image_only':
        # Process individual bubbles
        for page_idx, bubbles in result['pages'].items():
            for bubble in bubbles:
                print(f"Bubble at {bubble['bubble_box']}: {bubble['translated_text']}")
                # Save bubble image if needed
                img_data = base64.b64decode(bubble['image_base64'])
                img = Image.open(io.BytesIO(img_data))
                img.save(f'bubble_{page_idx}_{bubble["bubble_idx"]}.jpg')

    elif result['output_type'] == 'text_only':
        # Process text data
        for page_idx, bubbles in result['pages'].items():
            for bubble in bubbles:
                print(f"Original: {bubble['original_text']}")
                print(f"Translated: {bubble['translated_text']}")
                print(f"Position: {bubble['bubble_box']}")
```

### cURL Example

```bash
# Full page (default)
curl -X POST http://localhost:1389/translate/label1 \
  -F "images=@manga_page.jpg"

# Speech bubbles only
curl -X POST http://localhost:1389/translate/label1 \
  -F "images=@manga_page.jpg" \
  -F "output_type=speech_image_only"

# Text only
curl -X POST http://localhost:1389/translate/label1 \
  -F "images=@manga_page.jpg" \
  -F "output_type=text_only"
```

## Performance Comparison

| Output Type | Processing Components | Relative Speed | Use Case |
|------------|----------------------|----------------|-----------|
| `full_page` | Detect → OCR → Translate → Render Full Page | Baseline | Ready-to-view translated pages |
| `speech_image_only` | Detect → OCR → Translate → Crop & Render Bubbles | ~90% of baseline | Custom rendering or bubble analysis |
| `text_only` | Detect → OCR → Translate | ~60% of baseline (fastest) | Data analysis or custom rendering |

## Label2 Endpoint (With Inpainting)

The `/translate/label2` endpoint supports the same output types but includes inpainting:

- **`full_page`**: Returns inpainted pages with translated text
- **`speech_image_only`**: Returns bubble images extracted from inpainted pages
- **`text_only`**: Returns text data only (skips inpainting for speed)

## Notes

1. **Memory Usage**: `speech_image_only` mode may use more memory when processing many bubbles
2. **API Key**: All modes support optional `api_key` parameter for translation service
3. **Multiple Pages**: All modes support processing multiple pages in a single request
4. **Error Handling**: Invalid output_type values will return a 400 error with details