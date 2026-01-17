"""VLM OCR - Uses llama-server with VLM (Qwen, etc) for OCR on non-Windows"""

import io
import os
import re
import json
import time
import base64
import shutil
import requests
import subprocess
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

# ─────────────────────────────────────────────────────────────────────────────
# Configuration - Load from config.json
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')

def _load_config():
    """Load config from JSON file."""
    defaults = {
        "ocr_server_url": "http://localhost:8080",
        "ocr_quantization": "Q8",
        "llama_cli_path": "",
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
                defaults.update(cfg)
        except Exception:
            pass
    return defaults

_config = _load_config()

# VLM OCR models - model path, optional mmproj, and temperature
VLM_MODELS = {
    "qwen_vlm": {
        "model": "Qwen/Qwen3-VL-2B-Instruct-GGUF:Qwen3VL-2B-Instruct-Q8_0.gguf",
        "mmproj": "Qwen/Qwen3-VL-2B-Instruct-GGUF:mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
        "temperature": 0.7,
    },
    "lfm_vlm": {
        "model": "LiquidAI/LFM2.5-VL-1.6B-GGUF:LFM2.5-VL-1.6B-Q8_0.gguf",
        "mmproj": "LiquidAI/LFM2.5-VL-1.6B-GGUF:mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        "temperature": 0.1,
        "min_p": 0.15,
        "repetition_penalty": 1.05,
    },
    "ministral_vlm_q4": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        "mmproj": "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
        "temperature": 0.1,  # Low temp for production
    },
    "ministral_vlm_q8": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q8_0.gguf",
        "mmproj": "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
        "temperature": 0.1,  # Low temp for production
    },
}

def _get_ocr_model():
    """Get OCR model based on ocr_method setting."""
    cfg = _load_config()
    # Check for custom model override
    custom = cfg.get('ocr_model', '')
    if custom:
        return custom
    # Auto-select based on OCR method
    ocr_method = cfg.get('ocr_method', 'qwen_vlm')
    if ocr_method in VLM_MODELS:
        return VLM_MODELS[ocr_method]["model"]
    return VLM_MODELS["qwen_vlm"]["model"]

def _get_ocr_mmproj():
    """Get OCR vision encoder path (if needed for the model)."""
    cfg = _load_config()
    custom = cfg.get('ocr_mmproj', '')
    if custom:
        return custom
    # Auto-select based on OCR method
    ocr_method = cfg.get('ocr_method', 'qwen_vlm')
    if ocr_method in VLM_MODELS:
        return VLM_MODELS[ocr_method].get("mmproj") or ""
    return VLM_MODELS["qwen_vlm"]["mmproj"]

def _get_ocr_temperature():
    """Get OCR temperature based on model."""
    cfg = _load_config()
    ocr_method = cfg.get('ocr_method', 'qwen_vlm')
    if ocr_method in VLM_MODELS:
        return VLM_MODELS[ocr_method].get("temperature", 0.7)
    return 0.7

def _get_ocr_gen_params():
    """Get all generation params for current OCR model."""
    cfg = _load_config()
    ocr_method = cfg.get('ocr_method', 'qwen_vlm')
    if ocr_method in VLM_MODELS:
        model_cfg = VLM_MODELS[ocr_method]
        return {
            "temperature": model_cfg.get("temperature", 0.7),
            "min_p": model_cfg.get("min_p"),
            "top_p": model_cfg.get("top_p", 0.8),
            "top_k": model_cfg.get("top_k", 20),
            "repetition_penalty": model_cfg.get("repetition_penalty", 1.0),
            "presence_penalty": model_cfg.get("presence_penalty", 1.5),
        }
    # Default Qwen params
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "repetition_penalty": 1.0}

# Apply config (env vars override config.json)
LLAMA_SERVER_URL = os.environ.get('LLAMA_SERVER_URL', _config.get('ocr_server_url', 'http://localhost:8080'))
VLM_MODEL = os.environ.get('VLM_MODEL', _get_ocr_model())
VLM_MMPROJ = _get_ocr_mmproj()
LLAMA_CLI_PATH = _config.get('llama_cli_path', '')
OCR_GRID_MAX_CELLS = _config.get('ocr_grid_max_cells', 9)
TARGET_LANGUAGE = _config.get('target_language', 'en')

# Language code to full name mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
}

def get_target_language_name():
    """Get the full name of the target language."""
    # Reload config to get latest value
    cfg = _load_config()
    lang_code = cfg.get('target_language', 'en')
    return LANGUAGE_NAMES.get(lang_code, lang_code.capitalize())

# Grid separator colors (RGB)
COL_SEP_COLOR = (0, 100, 255)   # Blue - vertical lines between columns
ROW_SEP_COLOR = (255, 50, 50)   # Red - horizontal lines between rows
SEP_WIDTH = 6
MIN_GRID_HEIGHT = 240  # VLM servers often need minimum image dimensions


def find_llama_server():
    """Find llama-server executable, checking config.json llama_cli_path first."""
    # 1. Check config.json llama_cli_path
    if LLAMA_CLI_PATH:
        # If it's a directory, look for llama-server inside it
        if os.path.isdir(LLAMA_CLI_PATH):
            for name in ['llama-server', 'llama-server.exe']:
                path = os.path.join(LLAMA_CLI_PATH, name)
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return path
        # If it's a file, check if it's executable
        elif os.path.isfile(LLAMA_CLI_PATH) and os.access(LLAMA_CLI_PATH, os.X_OK):
            return LLAMA_CLI_PATH
        # Maybe it's a path to llama-cli, check for llama-server in same dir
        elif os.path.isfile(LLAMA_CLI_PATH):
            parent = os.path.dirname(LLAMA_CLI_PATH)
            for name in ['llama-server', 'llama-server.exe']:
                path = os.path.join(parent, name)
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return path

    # 2. Check PATH
    if shutil.which('llama-server'):
        return shutil.which('llama-server')

    # 3. Check common locations (including build directories)
    # Go up from workflow/ocr_vlm/ to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    paths = [
        # Local project llama.cpp build (most common)
        os.path.join(project_root, 'llama.cpp', 'build', 'bin', 'llama-server'),
        os.path.join(project_root, 'llama.cpp', 'llama-server'),
        # Relative paths
        './llama.cpp/build/bin/llama-server',
        './llama.cpp/llama-server',
        './llama-server',
        # Home directory
        os.path.expanduser('~/llama.cpp/build/bin/llama-server'),
        os.path.expanduser('~/llama.cpp/llama-server'),
        # System paths
        '/usr/local/bin/llama-server',
        '/opt/homebrew/bin/llama-server',
    ]
    for path in paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    return None


def check_vlm_available():
    """Check if llama-server is running or llama-server binary exists."""
    # Check if server is already running
    try:
        r = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        if r.status_code == 200:
            return True
    except:
        pass

    # Check if binary exists (using find_llama_server which reads config.json)
    return find_llama_server() is not None


def create_ocr_grid(bubbles, row_width=1600, padding=10):
    """
    Arrange bubble crops into a grid with colored separators for VLM OCR.

    Blue vertical lines separate columns (X axis).
    Red horizontal lines separate rows (Y axis).

    Returns: (grid_image, positions, grid_info)
    """
    if not bubbles or Image is None:
        return Image.new('RGB', (100, 100), (255, 255, 255)), [], {'rows': 0, 'cols': 0, 'total_cells': 0}

    # Calculate row layout
    rows, row, row_w = [], [], padding
    for b in bubbles:
        w = b['image'].width
        if row_w + w + padding + SEP_WIDTH > row_width and row:
            rows.append(row)
            row, row_w = [], padding
        row.append(b)
        row_w += w + padding
    if row:
        rows.append(row)

    max_cols = max(len(r) for r in rows)

    # Calculate total dimensions
    total_h = padding
    for row in rows:
        total_h += max(b['image'].height for b in row) + padding
    total_h += SEP_WIDTH * (len(rows) - 1) if len(rows) > 1 else 0

    # Create grid image
    grid = Image.new('RGB', (row_width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    positions = []

    y = padding
    for row_idx, row in enumerate(rows):
        row_h = max(b['image'].height for b in row)
        x = padding

        for col_idx, b in enumerate(row):
            y_off = y + (row_h - b['image'].height) // 2
            grid.paste(b['image'], (x, y_off))

            positions.append({
                'key': (b['page_idx'], b['bubble_idx']),
                'grid_box': (x, y_off, x + b['image'].width, y_off + b['image'].height),
                'bubble_box': b['box'],
                'grid_offset': (x, y_off),
                'crop_offset': b.get('crop_offset', (0, 0)),
                'grid_row': row_idx,
                'grid_col': col_idx,
                'cell_idx': len(positions)
            })

            if col_idx < len(row) - 1:
                sep_x = x + b['image'].width + padding // 2
                draw.rectangle([sep_x, y - padding // 2, sep_x + SEP_WIDTH, y + row_h + padding // 2], fill=COL_SEP_COLOR)

            x += b['image'].width + padding + (SEP_WIDTH if col_idx < len(row) - 1 else 0)

        if row_idx < len(rows) - 1:
            sep_y = y + row_h + padding // 2
            draw.rectangle([0, sep_y, row_width, sep_y + SEP_WIDTH], fill=ROW_SEP_COLOR)
            y += row_h + padding + SEP_WIDTH
        else:
            y += row_h + padding

    grid_info = {
        'rows': len(rows),
        'cols': max_cols,
        'total_cells': len(positions),
        'layout': [[pos['cell_idx'] for pos in positions if pos['grid_row'] == r] for r in range(len(rows))]
    }

    # Crop and ensure minimum height for VLM compatibility
    cropped = grid.crop((0, 0, row_width, y))
    if cropped.height < MIN_GRID_HEIGHT:
        padded = Image.new('RGB', (row_width, MIN_GRID_HEIGHT), (255, 255, 255))
        padded.paste(cropped, (0, 0))
        cropped = padded

    return cropped, positions, grid_info


def build_ocr_prompt(grid_info, translate=False):
    """Build the prompt for VLM OCR optimized for Qwen3-VL."""
    n = grid_info['total_cells']
    target_lang = get_target_language_name()

    if n == 1:
        if translate:
            return f"OCR: Read the text in this image (Japanese/Korean/Chinese) and translate to {target_lang}. Output only the translation."
        return "OCR: Read the text in this image exactly as written. Output only the text, nothing else."
    else:
        if translate:
            return f"""OCR Task: This image shows {n} manga/manhwa/manhua speech bubbles arranged in a grid.
- Bubbles are separated by BLUE vertical lines (columns) and RED horizontal lines (rows)
- Read each bubble's text (Japanese/Korean/Chinese) and translate to {target_lang}
- Output exactly {n} entries, one per bubble, numbered [0] to [{n-1}]
- Format: [0]: translation [1]: translation ... [{n-1}]: translation
- Do NOT repeat entries or skip numbers"""
        return f"""OCR Task: This image shows {n} manga/manhwa/manhua speech bubbles arranged in a grid.
- Bubbles are separated by BLUE vertical lines (columns) and RED horizontal lines (rows)
- Read each bubble's text exactly as written (Japanese/Korean/Chinese)
- Output exactly {n} entries, one per bubble, numbered [0] to [{n-1}]
- Format: [0]: text [1]: text ... [{n-1}]: text
- Do NOT repeat entries or skip numbers"""


def parse_ocr_output(output, positions, grid_info, is_translated=False):
    """Parse VLM output back to OCR result format."""
    cell_texts = {}

    # Single cell: use raw output
    if len(positions) == 1:
        text = output.strip()
        # Remove any [0]: prefix if present
        if text.startswith('[0]:'):
            text = text[4:].strip()
        if text:
            cell_texts[0] = text
    else:
        # Multi-cell: parse [N]: format
        pattern = r'\[(\d+)\]:\s*(.+?)(?=\[\d+\]:|$)'
        for match in re.findall(pattern, output, re.DOTALL):
            try:
                idx, text = int(match[0]), match[1].strip()
                if text and idx < len(positions):
                    cell_texts[idx] = text
            except (ValueError, IndexError):
                continue

    lines = []
    for cell_idx, text in cell_texts.items():
        pos = positions[cell_idx]
        gx1, gy1, gx2, gy2 = pos['grid_box']
        line_data = {
            'text': text,
            'bbox': {'x': gx1, 'y': gy1, 'width': gx2 - gx1, 'height': gy2 - gy1},
            'cell_idx': cell_idx,
            'grid_pos': (pos['grid_row'], pos['grid_col'])
        }
        if is_translated:
            line_data['translated'] = True
        lines.append(line_data)
    return lines


def is_bad_ocr_output(output, cell_texts, expected_cells=None):
    """
    Detect bad OCR output that should trigger a retry.

    Returns: (is_bad, reason)
    """
    if not output or not output.strip():
        return True, "empty_output"

    # Check if more indices returned than expected cells
    if expected_cells and cell_texts:
        max_idx = max(cell_texts.keys()) if cell_texts else 0
        if max_idx >= expected_cells:
            return True, f"index_overflow:{max_idx + 1}_vs_{expected_cells}"

    # Check for repetitive patterns (same word/phrase repeated 3+ times)
    words = output.split()
    if len(words) >= 6:
        # Check if same word repeats 3+ times consecutively
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2] and len(words[i]) > 1:
                return True, f"word_repetition:{words[i]}"

        # Check for alternating patterns like "A B A B A B"
        if len(words) >= 6:
            for i in range(len(words) - 5):
                if (words[i] == words[i+2] == words[i+4] and
                    words[i+1] == words[i+3] == words[i+5]):
                    return True, f"alternating_repetition:{words[i]}_{words[i+1]}"

    # Check for character-level repetition (e.g., "かんたんに、かんたんに、かんたんに")
    for sep in ['、', ',', '。', ' ']:
        parts = output.split(sep)
        if len(parts) >= 3:
            for i in range(len(parts) - 2):
                p1, p2, p3 = parts[i].strip(), parts[i+1].strip(), parts[i+2].strip()
                if p1 and p1 == p2 == p3 and len(p1) > 1:
                    return True, f"phrase_repetition:{p1[:20]}"

    # Check if output is excessively long (likely looping)
    if len(output) > 500:
        unique_ratio = len(set(output)) / len(output)
        if unique_ratio < 0.08:  # Less than 8% unique characters
            return True, "low_uniqueness"

    # Check if same text appears in multiple cells (duplicate detection)
    if cell_texts and len(cell_texts) >= 3:
        from collections import Counter
        counts = Counter(cell_texts.values())
        max_count = max(counts.values())
        # If any text appears 3+ times, it's likely hallucination
        if max_count >= 3:
            bad_text = [t for t, c in counts.items() if c == max_count][0]
            return True, f"duplicate_cells:{max_count}x'{bad_text[:10]}'"

    return False, None


def image_to_base64(image):
    """Convert PIL Image to base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


class VlmOCR:
    """VLM-based OCR using llama-server HTTP API (Qwen, etc)."""

    def __init__(self, server_url=None, model=None, max_cells_per_batch=None):
        """
        Initialize VLM OCR.

        Args:
            server_url: URL of running llama-server (default: http://localhost:8080)
            model: Model identifier for starting server if needed
            max_cells_per_batch: Max cells to process at once (default from config)
        """
        # Read config dynamically to support model switching
        cfg = _load_config()
        self.server_url = server_url or os.environ.get('LLAMA_SERVER_URL', cfg.get('ocr_server_url', 'http://localhost:8080'))
        self.model = model or os.environ.get('VLM_MODEL', _get_ocr_model())
        self.mmproj = _get_ocr_mmproj()
        self.max_cells_per_batch = max_cells_per_batch or cfg.get('ocr_grid_max_cells', 9)
        self._server_process = None

    def _ensure_server(self):
        """Ensure llama-server is running, auto-start if needed."""
        # Check if already running
        try:
            r = requests.get(f"{self.server_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass

        # Server not running - try to auto-start it
        llama_server = find_llama_server()
        if not llama_server:
            print(f"[VLM OCR] Server not running and llama-server not found")
            print(f"[VLM OCR] Install llama.cpp or set llama_cli_path in config.json")
            return False

        # Get config values
        cfg = _load_config()
        context_size = str(cfg.get('llama_context_size', 2048))
        gpu_layers = str(cfg.get('llama_gpu_layers', 99))
        mmproj = self.mmproj  # Use instance mmproj set at init

        # Parse port from server_url
        import re
        port_match = re.search(r':(\d+)$', self.server_url)
        port = port_match.group(1) if port_match else '8080'

        print(f"[VLM OCR] Auto-starting llama-server...")
        print(f"  Model: {self.model}")
        print(f"  mmproj: {mmproj}")
        print(f"  Port: {port}")

        # Create log directory
        log_dir = os.path.join(os.path.dirname(CONFIG_FILE), '.llama_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'ocr_autostart.log')

        # Download mmproj if needed (llama-server doesn't support -hf for mmproj)
        mmproj_path = None
        if mmproj:
            # Parse HF format: "owner/repo:filename"
            if ':' in mmproj:
                repo, filename = mmproj.rsplit(':', 1)
                cache_dir = os.path.expanduser('~/.cache/llama.cpp')
                os.makedirs(cache_dir, exist_ok=True)
                mmproj_path = os.path.join(cache_dir, filename)

                if not os.path.exists(mmproj_path):
                    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
                    print(f"[VLM OCR] Downloading mmproj: {filename}")
                    try:
                        import urllib.request
                        urllib.request.urlretrieve(url, mmproj_path)
                        print(f"[VLM OCR] Downloaded mmproj to {mmproj_path}")
                    except Exception as e:
                        print(f"[VLM OCR] Failed to download mmproj: {e}")
                        mmproj_path = None
                else:
                    print(f"[VLM OCR] Using cached mmproj: {mmproj_path}")

        # Start the server with live output streaming
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

        try:
            log_handle = open(log_file, 'w')
            # Build command with mmproj for VLM models
            cmd = [llama_server, '-hf', self.model, '--port', port, '-c', context_size, '-ngl', gpu_layers]
            if mmproj_path:
                cmd.extend(['--mmproj', mmproj_path])
            # Required for Qwen VL models to work correctly with images
            cmd.extend(['--image-min-tokens', '1024'])
            self._server_process = subprocess.Popen(
                cmd,
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, bufsize=1, universal_newlines=True
            )
        except Exception as e:
            print(f"[VLM OCR] Failed to start server: {e}")
            return False

        # Background thread to stream output
        import threading
        output_lines = []
        stop_streaming = threading.Event()

        def stream_output():
            try:
                for line in iter(self._server_process.stdout.readline, ''):
                    if stop_streaming.is_set():
                        break
                    line = line.rstrip()
                    if line:
                        output_lines.append(line)
                        log_handle.write(line + '\n')
                        log_handle.flush()
                        # Show download progress and errors
                        if 'download' in line.lower() or 'error' in line.lower() or '%' in line:
                            print(f"  [llama] {line}")

            except:
                pass

        stream_thread = threading.Thread(target=stream_output, daemon=True)
        stream_thread.start()

        # Wait for server to be ready (up to 600s for model download)
        print(f"[VLM OCR] Waiting for server (downloading model)...")
        for i in range(600):
            # Check if process died
            if self._server_process.poll() is not None:
                stop_streaming.set()
                stream_thread.join(timeout=1)
                log_handle.close()
                print(f"[VLM OCR] Server process exited with code {self._server_process.returncode}")
                if output_lines:
                    print("[VLM OCR] Last output:")
                    for line in output_lines[-10:]:
                        print(f"  {line}")
                return False

            try:
                r = requests.get(f"{self.server_url}/health", timeout=2)
                if r.status_code == 200:
                    print(f"[VLM OCR] Server ready!")
                    return True
            except:
                pass
            time.sleep(1)
            # Show progress every 30 seconds
            if i > 0 and i % 30 == 0:
                print(f"[VLM OCR] Still waiting... ({i}s elapsed)")

        stop_streaming.set()
        print(f"[VLM OCR] Server failed to start within 600s")
        log_path = os.path.abspath(log_file)
        print(f"[VLM OCR] Check logs at: {log_path}")
        return False

    def stop_server(self):
        """Stop the llama-server process if we started it."""
        if self._server_process is not None:
            try:
                # Try graceful termination first
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    self._server_process.kill()
                    self._server_process.wait()
                print(f"[VLM OCR] Server stopped")
            except Exception as e:
                print(f"[VLM OCR] Error stopping server: {e}")
            finally:
                self._server_process = None

    def _stream_with_early_abort(self, response, expected_cells=None):
        """
        Stream response and detect bad patterns early to abort.

        Returns:
            (output_text, early_abort, abort_reason)
        """
        import json as json_module
        output_chunks = []
        seen_indices = set()
        last_texts = []  # Track last few texts for repetition detection

        try:
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                # SSE format: "data: {...}"
                if not line_str.startswith('data: '):
                    continue

                data_str = line_str[6:]  # Remove "data: " prefix
                if data_str == '[DONE]':
                    break

                try:
                    data = json_module.loads(data_str)
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        output_chunks.append(content)

                        # Check for bad patterns in accumulated output
                        full_output = ''.join(output_chunks)

                        # 1. Check for index overflow
                        if expected_cells:
                            for match in re.finditer(r'\[(\d+)\]:', full_output):
                                idx = int(match.group(1))
                                seen_indices.add(idx)
                                if idx >= expected_cells:
                                    response.close()
                                    return full_output, True, f"index_overflow:{idx}_vs_{expected_cells}"

                        # 2. Check for too many indices (looping)
                        if expected_cells and len(seen_indices) > expected_cells * 2:
                            response.close()
                            return full_output, True, f"too_many_indices:{len(seen_indices)}"

                        # 3. Check for repetition in recent text
                        # Extract recent [N]: text entries
                        recent_matches = list(re.finditer(r'\[(\d+)\]:\s*([^\[]+)', full_output))
                        if len(recent_matches) >= 4:
                            recent_texts = [m.group(2).strip()[:50] for m in recent_matches[-6:]]
                            # Check for same text repeated 3+ times
                            from collections import Counter
                            counts = Counter(recent_texts)
                            for text, count in counts.items():
                                if count >= 3 and len(text) > 2:
                                    response.close()
                                    return full_output, True, f"repetition:{count}x'{text[:20]}'"

                        # 4. Check for single character spam (like い い い い)
                        if len(full_output) > 100:
                            # Check last 100 chars for single char repetition
                            last_chunk = full_output[-100:]
                            words = last_chunk.split()
                            if len(words) >= 10:
                                single_chars = [w for w in words if len(w) == 1]
                                if len(single_chars) >= 8:
                                    response.close()
                                    return full_output, True, "single_char_spam"

                except json_module.JSONDecodeError:
                    continue

        except Exception as e:
            # Return what we have on error
            return ''.join(output_chunks), False, None

        return ''.join(output_chunks), False, None

    def run(self, image, positions=None, grid_info=None, translate=False, max_retries=2):
        """
        Run OCR (and optionally translate) on image via llama-server API.

        Args:
            image: PIL Image
            positions: Position info from create_ocr_grid
            grid_info: Grid layout info
            translate: If True, output English translations
            max_retries: Number of retries on bad output (default: 2)

        Returns:
            Dict with 'lines', 'line_count', 'processing_time_ms', 'translated'
        """
        t0 = time.time()

        if Image is None:
            return {'lines': [], 'line_count': 0, 'processing_time_ms': 0, 'error': 'PIL not available'}

        if not self._ensure_server():
            return {'lines': [], 'line_count': 0, 'processing_time_ms': 0,
                    'error': f'llama-server not running at {self.server_url}', 'translated': translate}

        # Build prompt
        if grid_info:
            prompt = build_ocr_prompt(grid_info, translate=translate)
        else:
            target_lang = get_target_language_name()
            if translate:
                prompt = f"OCR: Read the text in this image (Japanese/Korean/Chinese) and translate to {target_lang}. Format: [N]: translation"
            else:
                prompt = "OCR: Read the text in this image exactly as written. Format: [N]: text"

        # Convert image to base64
        img_b64 = image_to_base64(image)

        last_error = None
        gen_params = _get_ocr_gen_params()
        base_temp = gen_params["temperature"]
        for attempt in range(max_retries + 1):
            # Use model-specific temp, vary slightly on retries
            temperature = base_temp + (attempt * 0.05)

            # Build request with model-specific parameters
            payload = {
                "model": "gpt-4-vision-preview",  # Ignored by llama-server but required
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_b64}}
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": min(temperature, 0.85),
                "top_p": gen_params.get("top_p", 0.8),
                "top_k": gen_params.get("top_k", 20),
                "presence_penalty": gen_params.get("presence_penalty", 1.5),
                "repetition_penalty": gen_params.get("repetition_penalty", 1.0),
                "stream": True
            }
            # Add min_p if specified (LFM uses this)
            if gen_params.get("min_p"):
                payload["min_p"] = gen_params["min_p"]

            try:
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                    stream=True
                )

                if response.status_code != 200:
                    last_error = f'Server error: {response.status_code} - {response.text[:200]}'
                    continue

                # Stream and check for bad output early
                expected_cells = len(positions) if positions else None
                output, early_abort, abort_reason = self._stream_with_early_abort(
                    response, expected_cells
                )

                if early_abort:
                    print(f"[VLM OCR] Early abort: {abort_reason}, attempt {attempt + 1}/{max_retries + 1}")
                    last_error = f"early_abort:{abort_reason}"
                    if attempt < max_retries:
                        time.sleep(0.3)
                        continue
                    # Max retries reached with early abort - use what we have
                    print(f"[VLM OCR] Max retries reached, using partial output")

                # Parse ALL indices from output (for validation - don't filter yet)
                all_cell_texts = {}
                if positions and grid_info:
                    if len(positions) == 1:
                        text = output.strip()
                        if text.startswith('[0]:'):
                            text = text[4:].strip()
                        if text:
                            all_cell_texts[0] = text
                    else:
                        pattern = r'\[(\d+)\]:\s*(.+?)(?=\[\d+\]:|$)'
                        for match in re.findall(pattern, output, re.DOTALL):
                            try:
                                idx, text = int(match[0]), match[1].strip()
                                if text:
                                    all_cell_texts[idx] = text  # Don't filter by index yet
                            except (ValueError, IndexError):
                                continue

                # Check for bad output (using all parsed cells for validation)
                expected_cells = len(positions) if positions else None
                is_bad, reason = is_bad_ocr_output(output, all_cell_texts, expected_cells)
                if is_bad:
                    print(f"[VLM OCR] Bad output detected ({reason}), attempt {attempt + 1}/{max_retries + 1}")
                    last_error = f"bad_output:{reason}"
                    if attempt < max_retries:
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    else:
                        print(f"[VLM OCR] Max retries reached, using last output")

                # Debug: log raw VLM output (first 500 chars)
                if output:
                    print(f"[VLM OCR] Raw output ({len(output)} chars): {output[:500]}{'...' if len(output) > 500 else ''}")
                else:
                    print(f"[VLM OCR] Warning: Empty response from VLM")

                # Parse output into lines
                if positions and grid_info:
                    lines = parse_ocr_output(output, positions, grid_info, is_translated=translate)
                else:
                    lines = []
                    for match in re.finditer(r'\[(\d+)\]:\s*(.+)', output):
                        line_data = {'text': match.group(2).strip(), 'bbox': {'x': 0, 'y': 0, 'width': 0, 'height': 0}}
                        if translate:
                            line_data['translated'] = True
                        lines.append(line_data)

                return {
                    'lines': lines,
                    'line_count': len(lines),
                    'processing_time_ms': (time.time() - t0) * 1000,
                    'translated': translate,
                    'raw_output': output,
                    'retries': attempt
                }

            except requests.exceptions.Timeout:
                last_error = 'Request timed out'
                if attempt < max_retries:
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    continue

        # All retries failed
        return {'lines': [], 'line_count': 0, 'processing_time_ms': (time.time() - t0) * 1000,
                'translated': translate, 'error': last_error}

    def run_grid(self, bubbles, row_width=1600, padding=10, translate=False):
        """Create grid and run OCR in one call."""
        grid_img, positions, grid_info = create_ocr_grid(bubbles, row_width, padding)
        ocr_result = self.run(grid_img, positions, grid_info, translate=translate)
        return ocr_result, positions, grid_img

    def run_batched(self, bubbles, row_width=1600, padding=10, translate=False, parallel=False):
        """Process bubbles in batches. Note: parallel=False by default as VLM servers typically can't handle concurrent requests."""
        if len(bubbles) <= self.max_cells_per_batch:
            return self.run_grid(bubbles, row_width, padding, translate)

        # Prepare batches (just bubble slices, not pre-rendered grids)
        batches = []
        for i in range(0, len(bubbles), self.max_cells_per_batch):
            batch = bubbles[i:i + self.max_cells_per_batch]
            batches.append((i, batch))

        t0 = time.time()

        if parallel and len(batches) > 1:
            # Parallel execution
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = [None] * len(batches)

            def process_batch(idx, batch_bubbles):
                result, positions, _ = self.run_grid(batch_bubbles, row_width, padding, translate)
                return idx, result, positions

            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
                futures = [executor.submit(process_batch, i, b) for i, b in batches]
                for future in as_completed(futures):
                    idx, result, positions = future.result()
                    results[idx // self.max_cells_per_batch] = (idx, result, positions)
        else:
            # Sequential execution
            results = []
            for i, batch_bubbles in batches:
                result, positions, _ = self.run_grid(batch_bubbles, row_width, padding, translate)
                results.append((i, result, positions))

        # Collect results
        all_lines = []
        all_positions = []
        for offset, result, positions in results:
            for line in result.get('lines', []):
                line['cell_idx'] = line.get('cell_idx', 0) + offset
                all_lines.append(line)
            all_positions.extend(positions)

        return {
            'lines': all_lines,
            'line_count': len(all_lines),
            'processing_time_ms': (time.time() - t0) * 1000,
            'translated': translate,
            'batch_count': len(batches)
        }, all_positions, None
