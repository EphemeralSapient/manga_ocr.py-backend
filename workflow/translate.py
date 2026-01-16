#!/usr/bin/env python3
"""
Translation Module - Japanese to English manga translation

Uses Cerebras API for translation with:
- Batch processing for efficiency
- Parallel execution
- [NO TEXT] markers for sound effects/meaningless text
"""

import os
import json
import time
import concurrent.futures
from typing import List, Dict, Tuple, Optional

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

# Default API key from environment
DEFAULT_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# Translation config
BATCH_SIZE = 100
MAX_WORKERS = 5
MODEL = "gpt-oss-120b"

SYSTEM_PROMPT = """You are a Japanese to English translator for manga. Translate each numbered Japanese text to English accurately. Keep the same order. Return translations in the translated array WITHOUT numbers. If the input is meaningless (single random characters, sound effects like 'ぎゅっ', symbols, or gibberish that doesn't form proper text), return [NO TEXT] for that entry."""


def translate_batch(
    client: Cerebras,
    batch_idx: int,
    batch: List[str]
) -> Tuple[int, List[str], Dict]:
    """Translate a single batch of texts."""
    batch_start = time.time()
    batch_info = {"batch_idx": batch_idx, "count": len(batch), "status": "success"}

    try:
        numbered = [f"{i+1}. {t}" for i, t in enumerate(batch)]
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Translate these {len(batch)} Japanese texts to English:\n{json.dumps(numbered, ensure_ascii=False)}"
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
            batch_info["finish_reason"] = completion.choices[0].finish_reason

    except Exception as e:
        batch_info["time_ms"] = int((time.time() - batch_start) * 1000)
        batch_info["status"] = f"error: {str(e)[:100]}"
        print(f"    Translation batch {batch_idx} error: {e}")

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
