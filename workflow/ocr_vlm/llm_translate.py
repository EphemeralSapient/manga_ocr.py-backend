"""LLM Translation using llama-server with HunyuanMT"""

import os
import json
import time
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration - Load from config.json
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')

def _load_config():
    """Load config from JSON file."""
    defaults = {
        "translate_server_url": "http://localhost:8081",
        "translate_model": "tencent/HY-MT1.5-1.8B-GGUF:HY-MT1.5-1.8B-Q4_K_M.gguf",
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

# Apply config (env vars override config.json)
TRANSLATE_SERVER_URL = os.environ.get('TRANSLATE_SERVER_URL', _config.get('translate_server_url', 'http://localhost:8081'))
TRANSLATE_MODEL = os.environ.get('TRANSLATE_MODEL', _config.get('translate_model', 'tencent/HY-MT1.5-1.8B-GGUF:HY-MT1.5-1.8B-Q4_K_M.gguf'))

# HunyuanMT supported languages (code -> English name, Chinese name)
LANGUAGE_NAMES = {
    'zh': ('Chinese', '中文'),
    'zh-Hant': ('Traditional Chinese', '繁体中文'),
    'en': ('English', '英语'),
    'fr': ('French', '法语'),
    'pt': ('Portuguese', '葡萄牙语'),
    'es': ('Spanish', '西班牙语'),
    'ja': ('Japanese', '日语'),
    'tr': ('Turkish', '土耳其语'),
    'ru': ('Russian', '俄语'),
    'ar': ('Arabic', '阿拉伯语'),
    'ko': ('Korean', '韩语'),
    'th': ('Thai', '泰语'),
    'it': ('Italian', '意大利语'),
    'de': ('German', '德语'),
    'vi': ('Vietnamese', '越南语'),
    'ms': ('Malay', '马来语'),
    'id': ('Indonesian', '印尼语'),
    'tl': ('Filipino', '菲律宾语'),
    'hi': ('Hindi', '印地语'),
    'pl': ('Polish', '波兰语'),
    'cs': ('Czech', '捷克语'),
    'nl': ('Dutch', '荷兰语'),
    'km': ('Khmer', '高棉语'),
    'my': ('Burmese', '缅甸语'),
    'fa': ('Persian', '波斯语'),
    'gu': ('Gujarati', '古吉拉特语'),
    'ur': ('Urdu', '乌尔都语'),
    'te': ('Telugu', '泰卢固语'),
    'mr': ('Marathi', '马拉地语'),
    'he': ('Hebrew', '希伯来语'),
    'bn': ('Bengali', '孟加拉语'),
    'ta': ('Tamil', '泰米尔语'),
    'uk': ('Ukrainian', '乌克兰语'),
    'bo': ('Tibetan', '藏语'),
    'kk': ('Kazakh', '哈萨克语'),
    'mn': ('Mongolian', '蒙古语'),
    'ug': ('Uyghur', '维吾尔语'),
    'yue': ('Cantonese', '粤语'),
}

def get_target_language_name():
    """Get the full name of the target language from config."""
    cfg = _load_config()
    lang_code = cfg.get('target_language', 'en')
    names = LANGUAGE_NAMES.get(lang_code)
    if names:
        return names[0]  # Return English name
    return lang_code.capitalize()

def get_target_language_chinese_name():
    """Get the Chinese name of the target language from config."""
    cfg = _load_config()
    lang_code = cfg.get('target_language', 'en')
    names = LANGUAGE_NAMES.get(lang_code)
    if names:
        return names[1]  # Return Chinese name
    return lang_code.capitalize()


def check_translate_server(url=None):
    """Check if translation server is running."""
    url = url or TRANSLATE_SERVER_URL
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


class LlmTranslator:
    """LLM-based Japanese to English translator using llama-server."""

    def __init__(self, server_url=None, batch_size=None):
        self.server_url = server_url or TRANSLATE_SERVER_URL
        # Default batch_size from config
        cfg_batch = _config.get('translation_batch_size', 25)
        self.batch_size = batch_size or cfg_batch

    def _ensure_server(self):
        """Check if server is running."""
        try:
            r = requests.get(f"{self.server_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        print(f"[Translate] Server not running at {self.server_url}")
        print(f"[Translate] Start with: llama-server -hf {TRANSLATE_MODEL} --port 8081")
        return False

    def translate(self, texts, source_lang=None, target_lang=None, parallel=True):
        """
        Translate texts from source to target language.

        Args:
            texts: List of strings to translate
            source_lang: Source language name (default: auto-detect Japanese/Korean/Chinese)
            target_lang: Target language name (default: from config.json target_language)
            parallel: Process batches in parallel

        Returns:
            Dict with 'translations', 'count', 'time_ms', 'error'
        """
        t0 = time.time()

        # Use defaults from config if not specified
        if source_lang is None:
            source_lang = "Japanese/Korean/Chinese"
        if target_lang is None:
            target_lang = get_target_language_name()

        if not texts:
            return {'translations': [], 'count': 0, 'time_ms': 0}

        if not self._ensure_server():
            return {'translations': [], 'count': 0, 'time_ms': 0,
                    'error': f'Server not running at {self.server_url}'}

        # Prepare batches
        batches = [(i, texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]

        if parallel and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = [None] * len(batches)
            errors = []

            def process_batch(batch_idx, batch_texts):
                return batch_idx, self._translate_batch(batch_texts, source_lang, target_lang)

            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
                futures = [executor.submit(process_batch, i, b) for i, b in batches]
                for future in as_completed(futures):
                    batch_idx, batch_result = future.result()
                    idx = batch_idx // self.batch_size
                    if 'error' in batch_result:
                        errors.append(batch_result['error'])
                        results[idx] = [''] * len(batches[idx][1])
                    else:
                        results[idx] = batch_result.get('translations', [])

            translations = []
            for r in results:
                translations.extend(r if r else [])
        else:
            translations = []
            errors = []
            for batch_num, (i, batch) in enumerate(batches):
                print(f"[Translate] Batch {batch_num + 1}/{len(batches)} ({len(batch)} texts)...")
                batch_results = self._translate_batch(batch, source_lang, target_lang)
                if 'error' in batch_results:
                    errors.append(batch_results['error'])
                    translations.extend([''] * len(batch))
                else:
                    batch_trans = batch_results.get('translations', [''] * len(batch))
                    success = sum(1 for t in batch_trans if t and t.strip())
                    print(f"[Translate] Batch {batch_num + 1}: {success}/{len(batch)} translated")
                    translations.extend(batch_trans)

        result = {
            'translations': translations,
            'count': len([t for t in translations if t]),
            'time_ms': (time.time() - t0) * 1000,
            'batches': len(batches),
            'batch_size': self.batch_size
        }
        if errors:
            result['errors'] = errors
        return result

    def _stream_with_early_abort(self, response, expected_count, target_lang):
        """
        Stream translation response and detect bad patterns early to abort.

        Returns:
            (output_text, early_abort, abort_reason)
        """
        output_chunks = []
        import re

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
                    data = json.loads(data_str)
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        output_chunks.append(content)

                        # Check for bad patterns periodically
                        full_output = ''.join(output_chunks)

                        # 1. Check for too many line numbers (looping)
                        line_nums = re.findall(r'^\d+[\.\)]', full_output, re.MULTILINE)
                        if expected_count and len(line_nums) > expected_count * 2:
                            response.close()
                            return full_output, True, f"too_many_lines:{len(line_nums)}"

                        # 2. Check for repetition in output
                        if len(full_output) > 200:
                            lines = [l.strip() for l in full_output.split('\n') if l.strip()]
                            if len(lines) >= 4:
                                from collections import Counter
                                counts = Counter(lines[-8:])
                                for text, count in counts.items():
                                    if count >= 3 and len(text) > 5:
                                        response.close()
                                        return full_output, True, f"repetition:{count}x"

                        # 3. For non-CJK targets, check if still mostly CJK
                        non_cjk_targets = ['english', 'spanish', 'french', 'german', 'portuguese',
                                          'italian', 'russian', 'arabic', 'hindi', 'thai',
                                          'vietnamese', 'indonesian']
                        if target_lang.lower() in non_cjk_targets and len(full_output) > 150:
                            # CJK pattern includes Japanese, Chinese, Korean
                            cjk_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]')
                            cjk_chars = len(cjk_pattern.findall(full_output))
                            total_chars = len(full_output)
                            if cjk_chars > total_chars * 0.5:
                                response.close()
                                return full_output, True, f"still_cjk:{cjk_chars}/{total_chars}"

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            return ''.join(output_chunks), False, None

        return ''.join(output_chunks), False, None

    def _translate_batch(self, texts, source_lang, target_lang, max_retries=2):
        """Translate a batch of texts with retry logic using HunyuanMT prompt format."""
        if not texts:
            return {'translations': []}

        # Get target language code from config
        cfg = _load_config()
        target_code = cfg.get('target_language', 'en')

        # Check if Chinese is involved (use Chinese prompt template)
        chinese_codes = ['zh', 'zh-Hant', 'yue']
        use_chinese_prompt = target_code in chinese_codes

        # Build prompt using HunyuanMT format
        # For batch: use [N]: format consistent with OCR output
        source_text = "\n".join(f"[{i+1}]: {text}" for i, text in enumerate(texts))

        if use_chinese_prompt:
            # ZH<=>XX prompt (Chinese template)
            target_name = get_target_language_chinese_name()
            prompt = f"将以下文本翻译为{target_name}，注意只需要输出翻译后的结果，保持[N]:格式，不要额外解释：\n\n{source_text}"
        else:
            # XX<=>XX prompt (English template)
            prompt = f"Translate the following into {target_lang}. Keep [N]: format, output only translations:\n\n{source_text}"

        last_error = None
        last_translations = None

        for attempt in range(max_retries + 1):
            # HunyuanMT recommended: temp=0.7, vary slightly on retries
            temperature = 0.7 + (attempt * 0.05)

            # HunyuanMT recommended parameters
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": min(temperature, 0.85),
                "top_k": 20,
                "top_p": 0.6,
                "repetition_penalty": 1.05,
                "stream": True
            }

            try:
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=60,
                    stream=True
                )

                if response.status_code != 200:
                    print(f"[Translate] Server error {response.status_code}: {response.text[:200]}")
                    last_error = f'Server error: {response.status_code}'
                    if attempt < max_retries:
                        print(f"[Translate] Retrying ({attempt + 1}/{max_retries + 1})...")
                        continue
                    # Max retries reached
                    print(f"[Translate] Max retries reached")

                # Stream with early abort detection
                output, early_abort, abort_reason = self._stream_with_early_abort(
                    response, len(texts), target_lang
                )

                if early_abort:
                    print(f"[Translate] Early abort: {abort_reason}, attempt {attempt + 1}/{max_retries + 1}")
                    last_error = f"early_abort:{abort_reason}"
                    if attempt < max_retries:
                        continue
                    # Max retries reached - use partial output
                    print(f"[Translate] Max retries reached, using partial output")

                # Debug: log translation output
                if output:
                    print(f"[Translate] Raw output ({len(output)} chars): {output[:300]}{'...' if len(output) > 300 else ''}")
                else:
                    print(f"[Translate] Warning: Empty response from LLM")

                # Parse output - expect numbered list or one per line
                translations = self._parse_translations(output, len(texts))
                last_translations = translations

                # Check for bad translation output
                is_bad, reason = self._is_bad_translation(translations, texts, target_lang)
                if is_bad:
                    print(f"[Translate] Bad output detected ({reason}), attempt {attempt + 1}/{max_retries + 1}")
                    if attempt < max_retries:
                        continue
                    # Max retries reached - use last output
                    print(f"[Translate] Max retries reached, using last output")

                return {'translations': translations}

            except Exception as e:
                print(f"[Translate] Error: {e}")
                last_error = str(e)
                if attempt < max_retries:
                    print(f"[Translate] Retrying ({attempt + 1}/{max_retries + 1})...")
                    continue

        # All retries failed - return last translations or empty
        if last_translations:
            return {'translations': last_translations}
        return {'translations': [''] * len(texts), 'error': last_error or 'Unknown error'}

    def _is_bad_translation(self, translations, source_texts, target_lang):
        """Check if translation output is bad and should be retried."""
        if not translations:
            return True, "empty_translations"

        # Count empty translations
        empty_count = sum(1 for t in translations if not t or not t.strip())
        if empty_count > len(translations) * 0.5:
            return True, f"too_many_empty:{empty_count}/{len(translations)}"

        # Check if output is still in source language (CJK characters in non-CJK target)
        # CJK = Chinese/Japanese/Korean
        non_cjk_targets = ['english', 'spanish', 'french', 'german', 'portuguese',
                          'italian', 'russian', 'arabic', 'hindi', 'thai',
                          'vietnamese', 'indonesian']
        if target_lang.lower() in non_cjk_targets:
            import re
            # Pattern for CJK characters (Japanese hiragana, katakana, and CJK unified ideographs)
            cjk_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]')
            cjk_count = 0
            for t in translations:
                if t and cjk_pattern.search(t):
                    cjk_count += 1
            if cjk_count > len(translations) * 0.3:
                return True, f"untranslated:{cjk_count}/{len(translations)}"

        return False, None

    def _parse_translations(self, output, expected_count):
        """Parse translation output in [N]: format."""
        import re
        translations = [''] * expected_count

        # Try to parse [N]: format first
        pattern = r'\[(\d+)\]:\s*(.+?)(?=\[\d+\]:|$)'
        matches = re.findall(pattern, output, re.DOTALL)

        if matches:
            for idx_str, text in matches:
                idx = int(idx_str) - 1  # Convert 1-based to 0-based
                if 0 <= idx < expected_count:
                    translations[idx] = text.strip()
        else:
            # Fallback: parse line by line, removing any numbering
            lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
            for i, line in enumerate(lines):
                if i >= expected_count:
                    break
                # Remove numbering if present (1. text, 1) text, [1] text)
                cleaned = re.sub(r'^(\d+[\.\)]|\[\d+\]:?)\s*', '', line)
                if cleaned:
                    translations[i] = cleaned

        return translations

    def translate_single(self, text, source_lang=None, target_lang=None):
        """Translate a single text."""
        result = self.translate([text], source_lang, target_lang)
        if result.get('translations'):
            return result['translations'][0]
        return ''
