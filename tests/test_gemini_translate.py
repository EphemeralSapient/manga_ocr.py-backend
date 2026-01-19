#!/usr/bin/env python3
"""Test Gemini API connection and translation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

# Import config to get API key from config.json
from config import get_gemini_api_key, get_gemini_translate_model, load_config


def test_api_key():
    """Check if API key is set."""
    # Try config.json first, then env var
    config_key = get_gemini_api_key()
    env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = config_key or env_key

    print("=" * 50)
    print("Gemini API Key Check")
    print("=" * 50)

    print(f"Source: {'config.json' if config_key else 'environment' if env_key else 'NOT FOUND'}")

    if config_key:
        print(f"  config.json: {config_key[:8]}...{config_key[-4:] if len(config_key) > 12 else ''} ({len(config_key)} chars)")
    else:
        print(f"  config.json: NOT SET")

    if env_key:
        print(f"  GEMINI_API_KEY env: {env_key[:8]}...{env_key[-4:] if len(env_key) > 12 else ''} ({len(env_key)} chars)")
    else:
        print(f"  GEMINI_API_KEY env: NOT SET")

    if not api_key:
        print("\n❌ No API key found!")
        print("\nTo fix:")
        print("  1. Get API key from: https://aistudio.google.com/apikey")
        print("  2. Add to config.json: \"gemini_api_key\": \"your_key_here\"")
        print("  3. Or add to .env file: GEMINI_API_KEY=your_key_here")
        return False, None

    if len(api_key) < 10:
        print(f"\n❌ API key seems too short: '{api_key}'")
        return False, None

    # Check for common issues
    if api_key.startswith('"') or api_key.endswith('"'):
        print(f"\n❌ API key has quotes around it - remove them")
        print(f"   Current: {api_key}")
        return False, None

    if ' ' in api_key or '\n' in api_key or '\t' in api_key:
        print(f"\n❌ API key contains whitespace - check for extra spaces/newlines")
        repr_key = repr(api_key)
        print(f"   Current: {repr_key}")
        return False, None

    print(f"\n✓ API key looks valid: {api_key[:8]}...{api_key[-4:]}")
    return True, api_key


def test_connection(api_key):
    """Test basic API connection."""
    print("\n" + "=" * 50)
    print("Gemini API Connection Test")
    print("=" * 50)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("❌ google-genai not installed")
        print("   Run: pip install google-genai")
        return False

    if not api_key:
        print("❌ No API key - skipping connection test")
        return False

    try:
        client = genai.Client(api_key=api_key)
        print("✓ Client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False

    # Test simple completion
    model = get_gemini_translate_model()
    print(f"\nUsing model: {model}")

    try:
        print("Testing simple completion...")
        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": "Say 'hello' in Japanese. Reply with just the word."}]}
            ],
            config={
                "temperature": 0.3,
                "max_output_tokens": 50,
            }
        )
        text = response.text if hasattr(response, 'text') else str(response)
        print(f"✓ API responded: {text}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {type(e).__name__}: {e}")
        if hasattr(e, 'status_code'):
            print(f"   Status code: {e.status_code}")
        if hasattr(e, 'message'):
            print(f"   Message: {e.message}")
        return False


def test_translation(api_key):
    """Test actual translation workflow."""
    print("\n" + "=" * 50)
    print("Gemini Translation Test")
    print("=" * 50)

    if not api_key:
        print("❌ No API key - skipping translation test")
        return False

    try:
        from workflow.gemini_translate import translate_texts_gemini, BATCH_SIZE
        print(f"✓ Module imported (batch_size={BATCH_SIZE})")
    except ImportError as e:
        print(f"❌ Failed to import gemini_translate module: {e}")
        return False

    # Test with sample Japanese texts
    test_texts = [
        "こんにちは",
        "ありがとう",
        "お兄ちゃん大好き！",
    ]

    print(f"\nTesting translation of {len(test_texts)} texts...")
    print(f"Input: {test_texts}")

    stats = {}
    try:
        translations = translate_texts_gemini(test_texts, stats=stats, verbose=True)
        print(f"\n✓ Translation completed!")
        print(f"Output: {translations}")
        print(f"Stats: {stats}")

        # Check for failures
        failures = [t for t in translations if t.startswith("[")]
        if failures:
            print(f"⚠ {len(failures)} translations failed: {failures}")
            return False
        return True
    except Exception as e:
        print(f"❌ Translation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_batching(api_key):
    """Test parallel batch processing with larger input."""
    print("\n" + "=" * 50)
    print("Parallel Batching Test")
    print("=" * 50)

    if not api_key:
        print("❌ No API key - skipping batching test")
        return False

    try:
        from workflow.gemini_translate import translate_texts_gemini, BATCH_SIZE
    except ImportError as e:
        print(f"❌ Failed to import gemini_translate module: {e}")
        return False

    # Create enough texts to trigger multiple batches (BATCH_SIZE = 33)
    # Use 40 texts to ensure at least 2 batches
    test_texts = [
        "こんにちは",
        "ありがとう",
        "お兄ちゃん大好き！",
        "今日はいい天気ですね",
        "何をしていますか？",
        "お腹が空いた",
        "眠いです",
        "頑張って！",
        "大丈夫ですか？",
        "すごい！",
    ] * 4  # 40 texts total

    expected_batches = (len(test_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Testing {len(test_texts)} texts (expecting {expected_batches} batches, batch_size={BATCH_SIZE})...")

    stats = {}
    try:
        import time
        t0 = time.time()
        translations = translate_texts_gemini(test_texts, stats=stats, verbose=True)
        elapsed = time.time() - t0

        print(f"\n✓ Parallel batching completed in {elapsed:.2f}s")
        print(f"Batches: {stats.get('translation_batches', 'N/A')}")
        print(f"Success: {stats.get('translation_success', 'N/A')}")
        print(f"Failed: {stats.get('translation_failed', 'N/A')}")

        # Verify we got all translations back
        if len(translations) != len(test_texts):
            print(f"❌ Got {len(translations)} translations for {len(test_texts)} inputs")
            return False

        # Check failure rate
        failures = [t for t in translations if t.startswith("[TRANSLATION")]
        if len(failures) > len(test_texts) * 0.1:  # Allow up to 10% failures
            print(f"❌ Too many failures: {len(failures)}/{len(test_texts)}")
            return False

        print(f"✓ All {len(translations)} translations received")
        return True
    except Exception as e:
        print(f"❌ Parallel batching failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🧪 Gemini Translation API Test Suite\n")

    results = []

    # Test 1: API Key
    key_ok, api_key = test_api_key()
    results.append(("API Key Check", key_ok))

    # Test 2: Connection
    results.append(("API Connection", test_connection(api_key)))

    # Test 3: Basic Translation
    results.append(("Basic Translation", test_translation(api_key)))

    # Test 4: Parallel Batching
    results.append(("Parallel Batching", test_parallel_batching(api_key)))

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed - check output above")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
