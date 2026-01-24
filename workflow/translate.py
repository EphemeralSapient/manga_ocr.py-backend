#!/usr/bin/env python3
"""
Translation Module - Comic/manga translation to English

Uses Cerebras API for translation with:
- Batch processing for efficiency
- Parallel execution
- [NO TEXT] markers for sound effects/meaningless text
- Language agnostic (Japanese, Korean, Chinese, etc.)
"""

import os
import json
import time
import concurrent.futures
from typing import List, Dict, Tuple, Optional

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

# Import config to get API key from config.json
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import get_cerebras_api_key
    _CONFIG_API_KEY = get_cerebras_api_key()
except ImportError:
    _CONFIG_API_KEY = None

# Default API key: config.json first, then environment variable
DEFAULT_API_KEY = _CONFIG_API_KEY or os.environ.get("CEREBRAS_API_KEY", "")

# Translation config
BATCH_SIZE = 100
MAX_WORKERS = 5
MODEL = "gpt-oss-120b"

SYSTEM_PROMPT = """You are a comic/manga/webtoon translator. Translate each numbered text to natural English. Return translations in the translated array WITHOUT numbers.

CRITICAL: You MUST translate ALL dialogue text. Only use [NO TEXT] for:
- Pure sound effects with NO meaning (ドドド, ゴゴゴ, 쾅쾅)
- Random symbols or completely illegible text
- Single characters that are just decorative

RULES:
1. TRANSLATE dialogue, thoughts, narration, and meaningful exclamations - even short ones like あ! (Ah!), え? (Huh?), うん (Yeah)
2. NEVER add brackets [] to translations. Only keep brackets if the ORIGINAL text literally starts with [
3. Multi-line inputs are from the same speech bubble - combine into one natural sentence
4. Consider story context - maintain consistent tone and flow

For Japanese: Use grammar (particles, verb endings) to determine correct phrase order.
For Korean: Text order is usually correct. Focus on natural, contextual translation.

Output clean, natural English. When in doubt, TRANSLATE - don't mark as [NO TEXT]."""


def translate_batch(
    client: Cerebras,
    batch_idx: int,
    batch: List[str],
    max_retries: int = 3
) -> Tuple[int, List[str], Dict]:
    """Translate a single batch of texts with retry logic."""
    batch_start = time.time()
    batch_info = {"batch_idx": batch_idx, "count": len(batch), "status": "success"}

    numbered = [f"{i+1}. {t}" for i, t in enumerate(batch)]

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                print(f"    [Translate] Batch {batch_idx} retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Translate these {len(batch)} texts to English:\n{json.dumps(numbered, ensure_ascii=False)}"
                    }
                ],
                model=MODEL,
                stream=False,
                max_completion_tokens=8192,
                temperature=1,
                top_p=1,
                reasoning_effort="low",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_schema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "input_text": {"type": "array", "items": {"type": "string"}},
                                "translated": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["input_text", "translated"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            content = completion.choices[0].message.content
            batch_info["time_ms"] = int((time.time() - batch_start) * 1000)
            batch_info["finish_reason"] = completion.choices[0].finish_reason

            if content:
                result = json.loads(content)
                translations = result.get("translated", [])

                # Pad if needed
                if len(translations) < len(batch):
                    batch_info["status"] = f"partial ({len(translations)}/{len(batch)})"
                while len(translations) < len(batch):
                    translations.append("[TRANSLATION FAILED]")
                return batch_idx, translations[:len(batch)], batch_info
            else:
                batch_info["status"] = "empty_response"

        except Exception as e:
            last_error = e
            batch_info["time_ms"] = int((time.time() - batch_start) * 1000)
            batch_info["status"] = f"error: {type(e).__name__}"
            batch_info["error"] = str(e)
            print(f"    [Translate] Batch {batch_idx} error: {type(e).__name__}: {str(e)[:100]}")

            # Check if we should retry (rate limit or server error)
            is_retryable = (
                'RateLimitError' in type(e).__name__ or
                '429' in str(e) or
                '503' in str(e) or
                '502' in str(e) or
                'timeout' in str(e).lower()
            )
            if is_retryable and attempt < max_retries - 1:
                continue  # Retry
            else:
                break  # Give up

    # All retries exhausted
    batch_info["retries"] = attempt + 1 if last_error else 0
    return batch_idx, ["[TRANSLATION FAILED]"] * len(batch), batch_info


def translate_texts(
    texts: List[str],
    api_key: Optional[str] = None,
    stats: Optional[Dict] = None,
    verbose: bool = True
) -> List[str]:
    """
    Translate Japanese texts to English.

    Args:
        texts: List of Japanese texts to translate
        api_key: Cerebras API key (uses env var if not provided)
        stats: Optional dict to populate with timing stats
        verbose: Print progress info

    Returns:
        List of translated English texts
    """
    if not texts:
        return []

    t0 = time.time()
    client = Cerebras(api_key=api_key or DEFAULT_API_KEY)

    # Split into batches
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_translations = []
    batch_stats = []

    # Run batches in parallel
    batch_results = [None] * len(batches)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batches), MAX_WORKERS)) as executor:
        futures = {
            executor.submit(translate_batch, client, i, batch): i
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

    # Retry failed translations (collect indices, retry in smaller batches)
    failed_indices = [i for i, t in enumerate(all_translations) if t == "[TRANSLATION FAILED]"]
    if failed_indices and len(failed_indices) <= len(texts) * 0.5:  # Only retry if <50% failed
        if verbose:
            print(f"    [Retry] {len(failed_indices)} failed translations, retrying...")

        # Retry failed texts in smaller batches of 20
        retry_batch_size = 20
        failed_texts = [texts[i] for i in failed_indices]
        retry_batches = [failed_texts[i:i + retry_batch_size] for i in range(0, len(failed_texts), retry_batch_size)]

        retry_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(retry_batches), MAX_WORKERS)) as executor:
            futures = {
                executor.submit(translate_batch, client, i, batch): i
                for i, batch in enumerate(retry_batches)
            }
            for future in concurrent.futures.as_completed(futures):
                batch_idx, translations, info = future.result()
                retry_results.append((batch_idx, translations))

        # Sort by batch_idx and flatten
        retry_results.sort(key=lambda x: x[0])
        retry_translations = []
        for _, trans in retry_results:
            retry_translations.extend(trans)

        # Update original translations with retry results
        retry_success = 0
        for i, orig_idx in enumerate(failed_indices):
            if i < len(retry_translations) and retry_translations[i] != "[TRANSLATION FAILED]":
                all_translations[orig_idx] = retry_translations[i]
                retry_success += 1

        if verbose:
            print(f"    [Retry] Recovered {retry_success}/{len(failed_indices)} translations")

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


__all__ = ['translate_texts', 'translate_batch', 'BATCH_SIZE', 'MODEL']
