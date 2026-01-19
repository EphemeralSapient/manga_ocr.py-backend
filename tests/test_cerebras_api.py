#!/usr/bin/env python3
"""Test Cerebras API connection and translation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

# Import config to get API key from config.json
from config import get_cerebras_api_key, load_config

def test_api_key():
    """Check if API key is set."""
    # Try config.json first, then env var
    config_key = get_cerebras_api_key()
    env_key = os.environ.get("CEREBRAS_API_KEY", "")
    api_key = config_key or env_key

    print("=" * 50)
    print("Cerebras API Key Check")
    print("=" * 50)

    print(f"Source: {'config.json' if config_key else 'environment' if env_key else 'NOT FOUND'}")

    if config_key:
        print(f"  config.json: {config_key[:8]}...{config_key[-4:] if len(config_key) > 12 else ''} ({len(config_key)} chars)")
    else:
        print(f"  config.json: NOT SET")

    if env_key:
        print(f"  CEREBRAS_API_KEY env: {env_key[:8]}...{env_key[-4:] if len(env_key) > 12 else ''} ({len(env_key)} chars)")
    else:
        print(f"  CEREBRAS_API_KEY env: NOT SET")

    if not api_key:
        print("\n❌ No API key found!")
        print("\nTo fix:")
        print("  1. Get API key from: https://cloud.cerebras.ai/")
        print("  2. Add to config.json: \"cerebras_api_key\": \"your_key_here\"")
        print("  3. Or add to .env file: CEREBRAS_API_KEY=your_key_here")
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
    print("Cerebras API Connection Test")
    print("=" * 50)

    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError:
        print("❌ cerebras-cloud-sdk not installed")
        print("   Run: pip install cerebras-cloud-sdk")
        return False

    if not api_key:
        print("❌ No API key - skipping connection test")
        return False

    try:
        client = Cerebras(api_key=api_key)
        print("✓ Client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False

    # Test simple completion
    try:
        print("\nTesting simple completion...")
        completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'hello' in Japanese"}
            ],
            model="llama-3.3-70b",
            max_completion_tokens=50,
        )
        response = completion.choices[0].message.content
        print(f"✓ API responded: {response}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {type(e).__name__}: {e}")
        if hasattr(e, 'status_code'):
            print(f"   Status code: {e.status_code}")
        if hasattr(e, 'body'):
            print(f"   Body: {e.body}")
        return False


def test_translation(api_key):
    """Test actual translation workflow."""
    print("\n" + "=" * 50)
    print("Translation Test")
    print("=" * 50)

    if not api_key:
        print("❌ No API key - skipping translation test")
        return False

    try:
        from workflow.translate import translate_texts
    except ImportError as e:
        print(f"❌ Failed to import translate module: {e}")
        return False

    # Test with sample Japanese texts
    test_texts = [
        "こんにちは",
        "ありがとう",
        "お兄ちゃん大好き！",
    ]

    print(f"Testing translation of {len(test_texts)} texts...")
    print(f"Input: {test_texts}")

    stats = {}
    try:
        translations = translate_texts(test_texts, api_key=api_key, stats=stats, verbose=True)
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
        return False


def main():
    print("\n🧪 Cerebras API Test Suite\n")

    results = []

    # Test 1: API Key
    key_ok, api_key = test_api_key()
    results.append(("API Key Check", key_ok))

    # Test 2: Connection
    results.append(("API Connection", test_connection(api_key)))

    # Test 3: Translation
    results.append(("Translation", test_translation(api_key)))

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
