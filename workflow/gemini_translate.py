#!/usr/bin/env python3
"""
Gemini Translation Module - Japanese to English manga translation via Google AI Studio API

Uses Gemma 3 / Gemini models for translation with:
- Batch processing (33 texts per batch - 1/3 of Cerebras)
- Parallel execution
- [NO TEXT] markers for sound effects/meaningless text

Requires: pip install google-genai
API Key: Set GEMINI_API_KEY environment variable or configure in config.json
"""

import os
import json
import time
import concurrent.futures
from typing import List, Dict, Tuple, Optional

# Import from centralized config
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    get_gemini_api_key,
    get_next_gemini_api_key,
    get_gemini_translate_model,
    get_target_language_name,
)

# Lazy import google-genai
_genai_module = None
_genai_types = None


def _get_genai_module():
    """Lazy load google-genai module (without creating client)."""
    global _genai_module, _genai_types
    if _genai_module is not None:
        return _genai_module, _genai_types

    try:
        from google import genai
        from google.genai import types
        _genai_module = genai
        _genai_types = types
        return _genai_module, _genai_types
    except ImportError:
        return None, None


def _get_client_with_key(api_key: str = None):
    """Create a Gemini client with the specified (or rotated) API key."""
    genai, types = _get_genai_module()
    if genai is None:
        return None, None

    # Get API key - use rotated key if not specified
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        api_key = get_next_gemini_api_key()

    if not api_key:
        return None, None

    try:
        client = genai.Client(api_key=api_key)
        return client, types
    except Exception as e:
        print(f"[Gemini Translate] Failed to create client: {e}")
        return None, None


def _get_genai():
    """Legacy function - creates client with first available key."""
    return _get_client_with_key()


def check_gemini_available() -> bool:
    """Check if Gemini API is available (SDK installed + API key configured)."""
    client, _ = _get_genai()
    return client is not None


# Translation config - 1/3 of Cerebras batch size for rate limiting
BATCH_SIZE = 33
MAX_WORKERS = 5

# Available models
DEFAULT_MODEL = "gemma-3-27b-it"


def _build_system_prompt(target_lang: str) -> str:
    """Build the translation system prompt."""
    return f"""You are a Japanese to English translator for manga. Translate each numbered Japanese text to {target_lang} accurately. Keep the same order. Return translations in a JSON object with "translated" array WITHOUT numbers. If the input is meaningless (single random characters, sound effects like 'ぎゅっ', symbols, or gibberish that doesn't form proper text), return [NO TEXT] for that entry.

Output format: {{"translated": ["translation1", "translation2", ...]}}"""


def translate_batch_gemini(
    client,
    batch_idx: int,
    batch: List[str],
    model: str,
    target_lang: str
) -> Tuple[int, List[str], Dict]:
    """Translate a single batch of texts using Gemini API."""
    batch_start = time.time()
    batch_info = {"batch_idx": batch_idx, "count": len(batch), "status": "success"}

    try:
        numbered = [f"{i+1}. {t}" for i, t in enumerate(batch)]
        # Log what we're sending
        input_preview = json.dumps(numbered[:5], ensure_ascii=False)
        if len(numbered) > 5:
            input_preview = input_preview[:-1] + f', ... +{len(numbered)-5} more]'
        print(f"    [Gemini Translate] Batch {batch_idx} sending {len(batch)} texts: {input_preview[:300]}")

        system_prompt = _build_system_prompt(target_lang)
        user_prompt = f"Translate these {len(batch)} Japanese texts to {target_lang}:\n{json.dumps(numbered, ensure_ascii=False)}"

        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
            config={
                "temperature": 0.3,
                "max_output_tokens": 8192,
            }
        )

        content = response.text if hasattr(response, 'text') else str(response)
        batch_info["time_ms"] = int((time.time() - batch_start) * 1000)

        if content:
            # Log raw response
            print(f"    [Gemini Translate] Batch {batch_idx} response ({len(content)} chars): {content[:300]}{'...' if len(content) > 300 else ''}")

            # Parse JSON response - try to extract the translated array
            try:
                # Try to find JSON in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    translations = result.get("translated", [])
                else:
                    # Fallback: try parsing the whole thing
                    result = json.loads(content)
                    translations = result.get("translated", [])
            except json.JSONDecodeError:
                # Try line-by-line parsing as fallback
                translations = []
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('{') and not line.startswith('}'):
                        # Remove numbering if present
                        import re
                        line = re.sub(r'^\d+\.\s*', '', line)
                        if line:
                            translations.append(line)

            # Log translation breakdown
            no_text = sum(1 for t in translations if t == "[NO TEXT]")
            empty = sum(1 for t in translations if not t)
            actual = len(translations) - no_text - empty
            print(f"    [Gemini Translate] Batch {batch_idx} parsed: {actual} translations, {no_text} [NO TEXT], {empty} empty")

            # Pad if needed
            if len(translations) < len(batch):
                batch_info["status"] = f"partial ({len(translations)}/{len(batch)})"
                print(f"    [Gemini Translate] Batch {batch_idx} WARNING: got {len(translations)} translations for {len(batch)} inputs")
            while len(translations) < len(batch):
                translations.append("[TRANSLATION FAILED]")
            return batch_idx, translations[:len(batch)], batch_info
        else:
            batch_info["status"] = "empty_response"
            print(f"    [Gemini Translate] Batch {batch_idx} empty response")

    except Exception as e:
        batch_info["time_ms"] = int((time.time() - batch_start) * 1000)
        batch_info["status"] = f"error: {type(e).__name__}"
        batch_info["error"] = str(e)

        # Print detailed error info
        print(f"    [Gemini Translate] Batch {batch_idx} error: {type(e).__name__}: {e}")
        if hasattr(e, 'response'):
            print(f"    [Gemini Translate] Response: {e.response}")
        if hasattr(e, 'status_code'):
            print(f"    [Gemini Translate] Status code: {e.status_code}")
        if hasattr(e, 'message'):
            print(f"    [Gemini Translate] Message: {e.message}")

    return batch_idx, ["[TRANSLATION FAILED]"] * len(batch), batch_info


def translate_texts_gemini(
    texts: List[str],
    stats: Optional[Dict] = None,
    verbose: bool = True,
    model: str = None,
    target_lang: str = None
) -> List[str]:
    """
    Translate Japanese texts to target language using Gemini API.

    Args:
        texts: List of Japanese texts to translate
        stats: Optional dict to populate with timing stats
        verbose: Print progress info
        model: Model to use (default from config)
        target_lang: Target language (default from config)

    Returns:
        List of translated texts
    """
    if not texts:
        return []

    # Check if API is available (don't create client yet - each batch gets rotated key)
    genai_module, _ = _get_genai_module()
    if genai_module is None:
        print("[Gemini Translate] Error: Gemini API not available (SDK not installed)")
        return ["[TRANSLATION FAILED]"] * len(texts)

    t0 = time.time()
    model = model or get_gemini_translate_model() or DEFAULT_MODEL
    target_lang = target_lang or get_target_language_name()

    if verbose:
        print(f"  [Translate] {len(texts)} texts using Gemini API ({model})...")

    # Split into batches (1/3 of Cerebras batch size)
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_translations = []
    batch_stats = []

    def translate_batch_with_rotated_key(batch_idx, batch):
        """Wrapper that creates a client with rotated key for each batch."""
        client, _ = _get_client_with_key()  # Gets next key in rotation
        if client is None:
            print(f"[Gemini Translate] Error: No API key available for batch {batch_idx}")
            return batch_idx, ["[TRANSLATION FAILED]"] * len(batch), {"error": "no_api_key"}
        return translate_batch_gemini(client, batch_idx, batch, model, target_lang)

    # Run batches in parallel (each batch gets rotated API key)
    batch_results = [None] * len(batches)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batches), MAX_WORKERS)) as executor:
        futures = {
            executor.submit(translate_batch_with_rotated_key, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(futures):
            batch_idx, translations, info = future.result()
            batch_results[batch_idx] = translations
            batch_stats.append(info)
            if verbose:
                print(f"    Batch {info['batch_idx']}: {info['count']} texts, {info.get('time_ms', 0)}ms, {info['status']}")

    for result in batch_results:
        all_translations.extend(result)

    total_time = int((time.time() - t0) * 1000)

    if stats is not None:
        stats["translation_ms"] = total_time
        stats["translation_batches"] = len(batches)
        stats["translation_batch_details"] = sorted(batch_stats, key=lambda x: x["batch_idx"])

        # Count successes/failures
        success = sum(1 for t in all_translations if t not in ["[TRANSLATION FAILED]", "[NO TEXT]"])
        failed = sum(1 for t in all_translations if t == "[TRANSLATION FAILED]")
        no_text = sum(1 for t in all_translations if t == "[NO TEXT]")
        stats["translation_success"] = success
        stats["translation_failed"] = failed + no_text

    return all_translations


__all__ = ['translate_texts_gemini', 'translate_batch_gemini', 'BATCH_SIZE', 'check_gemini_available']
