"""VLM OCR - Uses llama-server with VLM (Qwen, etc) for OCR on non-Windows"""

import io
import os
import re
import time
import base64
import requests
import subprocess

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

# Import from centralized config
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    get_ocr_server_url, get_ocr_model, get_ocr_mmproj, get_ocr_gen_params,
    get_ocr_grid_max_cells, get_target_language_name, find_llama,
    download_mmproj, build_llama_command, get_llama_context_size, get_llama_gpu_layers
)

# Module-level model name (for export)
VLM_MODEL = os.environ.get('VLM_MODEL', get_ocr_model())

# Grid separator colors (RGB)
COL_SEP_COLOR = (0, 100, 255)   # Blue - vertical lines between columns
ROW_SEP_COLOR = (255, 50, 50)   # Red - horizontal lines between rows
SEP_WIDTH = 6
MIN_GRID_HEIGHT = 240  # VLM servers often need minimum image dimensions


def check_vlm_available():
    """Check if llama-server is running or llama-server binary exists."""
    server_url = get_ocr_server_url()
    try:
        r = requests.get(f"{server_url}/health", timeout=2)
        if r.status_code == 200:
            return True
    except:
        pass
    return find_llama() is not None


def create_ocr_grid(bubbles, row_width=600, padding=30):
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
    """Detect bad OCR output that should trigger a retry. Returns: (is_bad, reason)"""
    if not output or not output.strip():
        return True, "empty_output"

    # Validate cell indices if we know expected count
    if expected_cells and expected_cells > 1:
        if not cell_texts:
            return True, "no_cells_parsed"

        max_idx = max(cell_texts.keys()) if cell_texts else 0
        if max_idx >= expected_cells:
            return True, f"index_overflow:{max_idx + 1}_vs_{expected_cells}"

        # Check for missing indices - all [0] to [n-1] should be present
        missing = set(range(expected_cells)) - set(cell_texts.keys())
        if missing:
            missing_str = ','.join(str(i) for i in sorted(missing)[:5])
            if len(missing) > 5:
                missing_str += f"...+{len(missing)-5}more"
            return True, f"missing_indices:[{missing_str}]_of_{expected_cells}"

    # Check for repetitive patterns
    words = output.split()
    if len(words) >= 6:
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2] and len(words[i]) > 1:
                return True, f"word_repetition:{words[i]}"
        for i in range(len(words) - 5):
            if words[i] == words[i+2] == words[i+4] and words[i+1] == words[i+3] == words[i+5]:
                return True, f"alternating_repetition:{words[i]}_{words[i+1]}"

    # Check for phrase repetition
    for sep in ['、', ',', '。', ' ']:
        parts = output.split(sep)
        if len(parts) >= 3:
            for i in range(len(parts) - 2):
                p1, p2, p3 = parts[i].strip(), parts[i+1].strip(), parts[i+2].strip()
                if p1 and p1 == p2 == p3 and len(p1) > 1:
                    return True, f"phrase_repetition:{p1[:20]}"

    # Check for low uniqueness (looping)
    if len(output) > 500 and len(set(output)) / len(output) < 0.08:
        return True, "low_uniqueness"

    # Check for duplicate cells
    if cell_texts and len(cell_texts) >= 3:
        from collections import Counter
        counts = Counter(cell_texts.values())
        max_count = max(counts.values())
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
        self.server_url = server_url or os.environ.get('LLAMA_SERVER_URL', get_ocr_server_url())
        self.model = model or os.environ.get('VLM_MODEL', get_ocr_model())
        self.mmproj = get_ocr_mmproj()
        self.max_cells_per_batch = max_cells_per_batch or get_ocr_grid_max_cells()
        self._server_process = None

    def _ensure_server(self):
        """Ensure llama-server is running, auto-start if needed."""
        try:
            r = requests.get(f"{self.server_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass

        llama_server = find_llama()
        if not llama_server:
            print("[VLM OCR] Server not running and llama-server not found")
            return False

        port_match = re.search(r':(\d+)$', self.server_url)
        port = port_match.group(1) if port_match else '8080'

        print(f"[VLM OCR] Auto-starting llama-server on port {port}...")

        # Download mmproj if needed
        mmproj_path = download_mmproj(self.mmproj)
        if self.mmproj and not mmproj_path:
            print(f"[VLM OCR] Warning: Failed to download mmproj")

        # Build and start server
        cmd = build_llama_command(llama_server, self.model, port, mmproj_path)
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

        try:
            self._server_process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, bufsize=1, universal_newlines=True
            )
        except Exception as e:
            print(f"[VLM OCR] Failed to start server: {e}")
            return False

        # Wait for server
        print("[VLM OCR] Waiting for server...")
        for i in range(600):
            if self._server_process.poll() is not None:
                print(f"[VLM OCR] Server exited with code {self._server_process.returncode}")
                return False
            try:
                r = requests.get(f"{self.server_url}/health", timeout=2)
                if r.status_code == 200:
                    print("[VLM OCR] Server ready!")
                    return True
            except:
                pass
            time.sleep(1)
            if i > 0 and i % 30 == 0:
                print(f"[VLM OCR] Still waiting... ({i}s)")

        print("[VLM OCR] Server failed to start within 600s")
        return False

    def stop_server(self):
        """Stop the llama-server process if we started it."""
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server_process.kill()
                    self._server_process.wait()
                print("[VLM OCR] Server stopped")
            except Exception as e:
                print(f"[VLM OCR] Error stopping server: {e}")
            finally:
                self._server_process = None

    def _stream_with_early_abort(self, response, expected_cells=None):
        """Stream response and detect bad patterns early to abort."""
        import json as json_module
        output_chunks = []
        seen_indices = set()

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                if not line_str.startswith('data: '):
                    continue

                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break

                try:
                    data = json_module.loads(data_str)
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        output_chunks.append(content)
                        full_output = ''.join(output_chunks)

                        # Check for index overflow
                        if expected_cells:
                            for match in re.finditer(r'\[(\d+)\]:', full_output):
                                idx = int(match.group(1))
                                seen_indices.add(idx)
                                if idx >= expected_cells:
                                    response.close()
                                    return full_output, True, f"index_overflow:{idx}_vs_{expected_cells}"

                        # Check for too many indices
                        if expected_cells and len(seen_indices) > expected_cells * 2:
                            response.close()
                            return full_output, True, f"too_many_indices:{len(seen_indices)}"

                        # Check for repetition
                        recent_matches = list(re.finditer(r'\[(\d+)\]:\s*([^\[]+)', full_output))
                        if len(recent_matches) >= 4:
                            from collections import Counter
                            recent_texts = [m.group(2).strip()[:50] for m in recent_matches[-6:]]
                            for text, count in Counter(recent_texts).items():
                                if count >= 3 and len(text) > 2:
                                    response.close()
                                    return full_output, True, f"repetition:{count}x'{text[:20]}'"

                        # Check for single char spam
                        if len(full_output) > 100:
                            words = full_output[-100:].split()
                            if len(words) >= 10 and len([w for w in words if len(w) == 1]) >= 8:
                                response.close()
                                return full_output, True, "single_char_spam"

                except json_module.JSONDecodeError:
                    continue
        except Exception:
            pass

        return ''.join(output_chunks), False, None

    def run(self, image, positions=None, grid_info=None, translate=False, max_retries=4):
        """Run OCR on image via llama-server API."""
        t0 = time.time()

        if Image is None:
            return {'lines': [], 'line_count': 0, 'processing_time_ms': 0, 'error': 'PIL not available'}

        if not self._ensure_server():
            return {'lines': [], 'line_count': 0, 'processing_time_ms': 0,
                    'error': f'llama-server not running at {self.server_url}', 'translated': translate}

        prompt = build_ocr_prompt(grid_info, translate=translate) if grid_info else (
            f"OCR: Read the text in this image and translate to {get_target_language_name()}. Format: [N]: translation"
            if translate else "OCR: Read the text in this image exactly as written. Format: [N]: text"
        )

        img_b64 = image_to_base64(image)
        gen_params = get_ocr_gen_params()
        base_temp = gen_params["temperature"]
        last_error = None

        for attempt in range(max_retries + 1):
            temperature = base_temp + (attempt * 0.05)

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_b64}}
                ]}],
                "max_tokens": 2048,
                "temperature": min(temperature, 0.85),
                "top_p": gen_params.get("top_p", 0.8),
                "top_k": gen_params.get("top_k", 20),
                "presence_penalty": gen_params.get("presence_penalty", 1.5),
                "repetition_penalty": gen_params.get("repetition_penalty", 1.0),
                "stream": True
            }
            if gen_params.get("min_p"):
                payload["min_p"] = gen_params["min_p"]

            try:
                response = requests.post(f"{self.server_url}/v1/chat/completions", json=payload, timeout=120, stream=True)
                if response.status_code != 200:
                    last_error = f'Server error: {response.status_code}'
                    continue

                expected_cells = len(positions) if positions else None
                output, early_abort, abort_reason = self._stream_with_early_abort(response, expected_cells)

                if early_abort:
                    print(f"[VLM OCR] Early abort: {abort_reason}, attempt {attempt + 1}/{max_retries + 1}")
                    last_error = f"early_abort:{abort_reason}"
                    if attempt < max_retries:
                        time.sleep(0.3)
                        continue
                    print("[VLM OCR] Max retries reached, using partial output")

                # Parse output
                all_cell_texts = {}
                if positions and grid_info:
                    if len(positions) == 1:
                        text = output.strip()
                        if text.startswith('[0]:'):
                            text = text[4:].strip()
                        if text:
                            all_cell_texts[0] = text
                    else:
                        for match in re.findall(r'\[(\d+)\]:\s*(.+?)(?=\[\d+\]:|$)', output, re.DOTALL):
                            try:
                                idx, text = int(match[0]), match[1].strip()
                                if text:
                                    all_cell_texts[idx] = text
                            except (ValueError, IndexError):
                                continue

                is_bad, reason = is_bad_ocr_output(output, all_cell_texts, expected_cells)
                if is_bad:
                    print(f"[VLM OCR] Bad output ({reason}), attempt {attempt + 1}/{max_retries + 1}")
                    last_error = f"bad_output:{reason}"
                    if attempt < max_retries:
                        time.sleep(0.5)
                        continue

                if output:
                    attempt_str = f" (attempt {attempt + 1}/{max_retries + 1})" if attempt > 0 else ""
                    print(f"[VLM OCR] Output ({len(output)} chars){attempt_str}: {output[:500]}{'...' if len(output) > 500 else ''}")

                lines = parse_ocr_output(output, positions, grid_info, is_translated=translate) if positions and grid_info else []

                return {
                    'lines': lines, 'line_count': len(lines),
                    'processing_time_ms': (time.time() - t0) * 1000,
                    'translated': translate, 'raw_output': output, 'retries': attempt
                }

            except requests.exceptions.Timeout:
                last_error = 'Request timed out'
                print(f"[VLM OCR] Timeout, attempt {attempt + 1}/{max_retries + 1}")
            except Exception as e:
                last_error = str(e)
                print(f"[VLM OCR] Error: {str(e)[:100]}, attempt {attempt + 1}/{max_retries + 1}")

        return {'lines': [], 'line_count': 0, 'processing_time_ms': (time.time() - t0) * 1000,
                'translated': translate, 'error': last_error}

    def run_grid(self, bubbles, row_width=600, padding=30, translate=False):
        """Create grid and run OCR in one call."""
        grid_img, positions, grid_info = create_ocr_grid(bubbles, row_width, padding)
        ocr_result = self.run(grid_img, positions, grid_info, translate=translate)
        return ocr_result, positions, grid_img

    def run_batched(self, bubbles, row_width=600, padding=30, translate=False, parallel=False):
        """Process bubbles in batches."""
        if len(bubbles) <= self.max_cells_per_batch:
            return self.run_grid(bubbles, row_width, padding, translate)

        batches = [(i, bubbles[i:i + self.max_cells_per_batch]) for i in range(0, len(bubbles), self.max_cells_per_batch)]
        t0 = time.time()

        if parallel and len(batches) > 1:
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
            results = [(i, *self.run_grid(b, row_width, padding, translate)[:2]) for i, b in batches]

        all_lines, all_positions = [], []
        for offset, result, positions in results:
            for line in result.get('lines', []):
                line['cell_idx'] = line.get('cell_idx', 0) + offset
                all_lines.append(line)
            all_positions.extend(positions)

        return {
            'lines': all_lines, 'line_count': len(all_lines),
            'processing_time_ms': (time.time() - t0) * 1000,
            'translated': translate, 'batch_count': len(batches)
        }, all_positions, None
