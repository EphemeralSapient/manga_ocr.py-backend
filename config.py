#!/usr/bin/env python3
"""Manga Translation Server - Configuration Management

All settings are stored in config.json. Other scripts should import from here.
"""

import os
import json
import subprocess
import sys
import platform

PY = platform.system()

# Bootstrap rich
def _ensure_rich():
    try:
        import rich
        return True
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'rich', '-q'], capture_output=True)
        return True

_ensure_rich()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_FILE = "config.json"

LANGUAGES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ko": "Korean",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}

DEFAULT_CONFIG = {
    # ══════════════════════════════════════════════════════════════════════════
    # User Settings (shown in wizard)
    # ══════════════════════════════════════════════════════════════════════════

    # OCR method: "qwen_vlm" | "lfm_vlm" | "ministral_vlm_q8" | "gemini_api" | "oneocr" | "oneocr_remote"
    "ocr_method": "qwen_vlm",

    # OneOCR remote server URL (for oneocr_remote method)
    # Run oneocr_server/server.py on a Windows machine to get the URL
    "oneocr_server_url": "",

    # Translation method: "hunyuan_mt" | "cerebras_api"
    # Note: VLM translation (qwen_vlm) disabled due to hallucination issues
    "translate_method": "hunyuan_mt",

    # Target language for translation
    "target_language": "en",

    # Server port
    "server_port": 5000,

    # Output settings
    "output_dir": "output",
    "jpeg_quality": 95,

    # ══════════════════════════════════════════════════════════════════════════
    # Model Settings (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    # OCR quantization: only "Q8" works reliably (Q4 produces garbage with VLM)
    "ocr_quantization": "Q8",

    # Model paths (auto-selected based on ocr_quantization, or override manually)
    "ocr_model": "",  # Leave empty to auto-select based on ocr_quantization
    "ocr_mmproj": "", # Leave empty to auto-select (always Q8 for vision encoder)
    "translate_model": "tencent/HY-MT1.5-1.8B-GGUF:HY-MT1.5-1.8B-Q4_K_M.gguf",
    "cerebras_api_key": "",

    # Gemma/Gemini API settings (for gemini_api OCR and gemini_translate methods)
    # Uses Google AI Studio API - get key at https://aistudio.google.com/apikey
    "gemini_api_key": "",
    "gemini_model": "gemma-3-27b-it",           # OCR model options:
                                                 # - gemma-3-27b-it (14.4k req/day - RECOMMENDED)
                                                 # - gemini-2.5-flash-lite (20 req/day - limited)
                                                 # - gemini-3-flash-preview (20 req/day - limited)
    "gemini_translate_model": "gemma-3-27b-it", # Translation model (same options as above)

    # ══════════════════════════════════════════════════════════════════════════
    # Server Settings (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    "ocr_server_url": "http://localhost:8080",
    "translate_server_url": "http://localhost:8081",
    "auto_start_servers": True,

    # llama-server parameters
    "llama_context_size": 2048,
    "llama_gpu_layers": 99,

    # Sequential model loading - load only one model at a time to save VRAM
    # When enabled: slower (models load/unload per request) but uses less VRAM
    # When disabled: faster (both models stay loaded) but uses more VRAM
    # Meant for budget GPUs with limited VRAM (e.g., 4-6GB)
    "sequential_model_loading": False,

    # Streaming mode - enable SSE streaming endpoints for real-time results
    # When enabled: /translate/label1/stream endpoint becomes available
    # Results stream back as OCR→translate batches complete (lower latency)
    "streaming_enabled": False,

    # OCR grid settings (for VLM batch processing)
    "ocr_grid_max_cells": 9,  # Max bubbles per grid (e.g., 9 = 3x3)

    # ══════════════════════════════════════════════════════════════════════════
    # Detection Settings (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    "detection_threshold": 0.5,
    "detection_padding": 5,

    # ══════════════════════════════════════════════════════════════════════════
    # Text Segmentation Settings (config.json only)
    # Uses comic-text-detector for pixel-level text masks (better inpainting)
    # ══════════════════════════════════════════════════════════════════════════

    # Enable text segmentation masks for inpainting (instead of bbox-only)
    "text_seg_enabled": True,

    # Text segmentation model path (relative to project root)
    "text_seg_model": "text_seg/comic-text-detector/data/comictextdetector.pt.onnx",

    # Text segmentation input size (1024 recommended)
    "text_seg_input_size": 1024,

    # ══════════════════════════════════════════════════════════════════════════
    # Translation API Settings (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    "translation_temperature": 0.3,
    "translation_max_tokens": 8192,
    "translation_batch_size": 25,

    # ══════════════════════════════════════════════════════════════════════════
    # Paths (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    "venv_path": ".venv",
    "python_version": "3.12",
    "llama_cli_path": "",

    "scripts": {
        "install_runtime": "scripts/install_runtime.py",
        "download_model": "scripts/download_model.py",
        "export_coreml": "scripts/export_coreml.py",
        "export_static": "scripts/export_static.py",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ok(msg): console.print(f"[green]✓[/] {msg}")
def warn(msg): console.print(f"[yellow]⚠[/] {msg}")
def err(msg): console.print(f"[red]✗[/] {msg}")
def info(msg): console.print(f"[dim]→[/] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Config I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config from JSON file, merging with defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            for key, value in saved.items():
                if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
        except Exception as e:
            warn(f"Could not load {CONFIG_FILE}: {e}")
    return cfg


def save_config(cfg: dict):
    """Save config to JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
    ok(f"Saved {CONFIG_FILE}")


def config_exists() -> bool:
    return os.path.exists(CONFIG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Config Accessors
# ─────────────────────────────────────────────────────────────────────────────

def get(key: str, default=None):
    cfg = load_config()
    return cfg.get(key, default)

# User settings
def get_ocr_method() -> str:
    return get("ocr_method", "qwen_vlm")

def get_translate_method() -> str:
    return get("translate_method", "qwen_vlm")

def get_target_language() -> str:
    return get("target_language", "en")

def get_server_port() -> int:
    return get("server_port", 5000)

def get_output_dir() -> str:
    return get("output_dir", "output")

def get_jpeg_quality() -> int:
    return get("jpeg_quality", 95)

# Model settings
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
    "ministral_vlm_q8": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q8_0.gguf",
        "mmproj": "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
        "temperature": 0.1,  # Low temp for production
    },
}

def get_ocr_model() -> str:
    """Get OCR model path based on ocr_method."""
    # Check if user has manually specified a model
    custom = get("ocr_model", "")
    if custom:
        return custom
    # Auto-select based on OCR method
    ocr_method = get("ocr_method", "qwen_vlm")
    if ocr_method in VLM_MODELS:
        return VLM_MODELS[ocr_method]["model"]
    return VLM_MODELS["qwen_vlm"]["model"]

def get_ocr_mmproj() -> str:
    """Get OCR vision encoder path (if needed for the model)."""
    custom = get("ocr_mmproj", "")
    if custom:
        return custom
    # Auto-select based on OCR method
    ocr_method = get("ocr_method", "qwen_vlm")
    if ocr_method in VLM_MODELS:
        return VLM_MODELS[ocr_method].get("mmproj") or ""
    return VLM_MODELS["qwen_vlm"]["mmproj"]

def get_ocr_temperature() -> float:
    """Get OCR temperature based on model."""
    ocr_method = get("ocr_method", "qwen_vlm")
    if ocr_method in VLM_MODELS:
        return VLM_MODELS[ocr_method].get("temperature", 0.7)
    return 0.7

def get_translate_model() -> str:
    return get("translate_model", DEFAULT_CONFIG["translate_model"])

def get_cerebras_api_key() -> str:
    return get("cerebras_api_key", "")

def get_gemini_api_key() -> str:
    return get("gemini_api_key", "")

def get_gemini_model() -> str:
    return get("gemini_model", "gemma-3-27b-it")

def get_gemini_translate_model() -> str:
    return get("gemini_translate_model", "gemma-3-27b-it")

# Server settings
def get_ocr_server_url() -> str:
    return get("ocr_server_url", "http://localhost:8080")

def get_translate_server_url() -> str:
    return get("translate_server_url", "http://localhost:8081")

def get_auto_start_servers() -> bool:
    return get("auto_start_servers", True)

def get_llama_context_size() -> int:
    return get("llama_context_size", 2048)

def get_llama_gpu_layers() -> int:
    return get("llama_gpu_layers", 99)

def get_ocr_grid_max_cells() -> int:
    return get("ocr_grid_max_cells", 9)

def get_sequential_model_loading() -> bool:
    """Check if sequential model loading is enabled (low VRAM mode)."""
    return get("sequential_model_loading", False)

def get_streaming_enabled() -> bool:
    """Check if streaming endpoints are enabled."""
    return get("streaming_enabled", False)

# Detection settings
def get_detection_threshold() -> float:
    return get("detection_threshold", 0.5)

def get_detection_padding() -> int:
    return get("detection_padding", 5)

# Text segmentation settings
def get_text_seg_enabled() -> bool:
    return get("text_seg_enabled", True)

def get_text_seg_model() -> str:
    return get("text_seg_model", "text_seg/comic-text-detector/data/comictextdetector.pt.onnx")

def get_text_seg_input_size() -> int:
    return get("text_seg_input_size", 1024)

# Translation settings
def get_translation_temperature() -> float:
    return get("translation_temperature", 0.3)

def get_translation_max_tokens() -> int:
    return get("translation_max_tokens", 8192)

def get_translation_batch_size() -> int:
    return get("translation_batch_size", 50)

# Paths
def get_venv_path() -> str:
    return get("venv_path", ".venv")

def get_python_version() -> str:
    return get("python_version", "3.12")

def get_llama_cli_path() -> str:
    return get("llama_cli_path", "")

def get_script_path(name: str) -> str:
    cfg = load_config()
    return cfg.get("scripts", {}).get(name, f"scripts/{name}.py")

# Convenience
def needs_llama() -> bool:
    """Check if current config needs llama.cpp."""
    ocr = get_ocr_method()
    translate = get_translate_method()
    # API-based and remote OCR don't need llama.cpp
    if ocr in API_OCR_METHODS or ocr in REMOTE_OCR_METHODS or ocr == "oneocr":
        ocr_needs_llama = False
    else:
        ocr_needs_llama = ocr in VLM_METHODS
    translate_needs_llama = translate in ("qwen_vlm", "hunyuan_mt")
    return ocr_needs_llama or translate_needs_llama

def uses_same_model_for_ocr_and_translate() -> bool:
    """Check if OCR and translation use the same model."""
    return get_ocr_method() == "qwen_vlm" and get_translate_method() == "qwen_vlm"

# VLM method constants (local llama.cpp based)
VLM_METHODS = ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8")

# API-based OCR methods (cloud services, no local GPU needed)
API_OCR_METHODS = ("gemini_api",)

# Remote OCR methods (network services)
REMOTE_OCR_METHODS = ("oneocr_remote",)

def is_vlm_method(method: str) -> bool:
    """Check if method is a local VLM-based method (needs llama.cpp)."""
    return method in VLM_METHODS

def is_api_ocr_method(method: str) -> bool:
    """Check if method is an API-based OCR method (cloud service)."""
    return method in API_OCR_METHODS


def is_remote_ocr_method(method: str) -> bool:
    """Check if method is a remote OCR method (network service)."""
    return method in REMOTE_OCR_METHODS


def get_oneocr_server_url() -> str:
    """Get OneOCR remote server URL."""
    return get("oneocr_server_url", "")

def get_target_language_name() -> str:
    """Get the full name of the target language."""
    lang_code = get_target_language()
    return LANGUAGES.get(lang_code, lang_code.capitalize())

def get_ocr_gen_params() -> dict:
    """Get all generation params for current OCR model."""
    ocr_method = get_ocr_method()
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
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "repetition_penalty": 1.0}

# ─────────────────────────────────────────────────────────────────────────────
# Llama.cpp Utilities
# ─────────────────────────────────────────────────────────────────────────────

import shutil

def find_llama() -> str | None:
    """Find llama-server executable. Checks config, PATH, and common locations."""
    cfg_path = get_llama_cli_path()

    # Check config.json llama_cli_path first
    if cfg_path:
        if os.path.isdir(cfg_path):
            for name in ['llama-server', 'llama-cli', 'llama-server.exe', 'llama-cli.exe']:
                path = os.path.join(cfg_path, name)
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return path
        elif os.path.isfile(cfg_path) and os.access(cfg_path, os.X_OK):
            return cfg_path
        elif os.path.isfile(cfg_path):
            parent = os.path.dirname(cfg_path)
            for name in ['llama-server', 'llama-cli']:
                path = os.path.join(parent, name)
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return path

    # Check PATH and common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ['llama-server', 'llama-cli']:
        paths = [
            shutil.which(name) or '',
            os.path.join(script_dir, 'llama.cpp', 'build', 'bin', name),
            os.path.join(script_dir, 'llama.cpp', name),
            f'./llama.cpp/build/bin/{name}',
            f'./llama.cpp/{name}',
            f'./{name}',
            os.path.expanduser(f'~/llama.cpp/build/bin/{name}'),
            os.path.expanduser(f'~/llama.cpp/{name}'),
            f'/usr/local/bin/{name}',
            f'/opt/homebrew/bin/{name}',
        ]
        for p in paths:
            if p and os.path.exists(p) and os.access(p, os.X_OK):
                return p
    return None

def download_mmproj(mmproj: str) -> str | None:
    """Download mmproj file if needed. Returns local path or None."""
    if not mmproj or ':' not in mmproj:
        return None

    repo, filename = mmproj.rsplit(':', 1)
    cache_dir = os.path.expanduser('~/.cache/llama.cpp')
    os.makedirs(cache_dir, exist_ok=True)
    mmproj_path = os.path.join(cache_dir, filename)

    if os.path.exists(mmproj_path):
        return mmproj_path

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, mmproj_path)
        return mmproj_path
    except Exception:
        return None

def build_llama_command(llama_path: str, model: str, port: str, mmproj_path: str = None) -> list:
    """Build llama-server command with standard options."""
    ctx = str(get_llama_context_size())
    ngl = str(get_llama_gpu_layers())
    cmd = [llama_path, '-hf', model, '--port', port, '-c', ctx, '-ngl', ngl]
    if mmproj_path:
        cmd.extend(['--mmproj', mmproj_path])
    cmd.extend(['--image-min-tokens', '1024'])
    return cmd

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Prompts
# ─────────────────────────────────────────────────────────────────────────────

def prompt_choice(prompt: str, options: list, default: int = 0):
    """Prompt user to choose from options using arrow keys. Returns the key."""
    import sys
    import tty
    import termios

    def get_key():
        """Read a single keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Escape sequence
                ch2 = sys.stdin.read(2)
                if ch2 == '[A':
                    return 'up'
                elif ch2 == '[B':
                    return 'down'
            elif ch in ('\r', '\n'):
                return 'enter'
            elif ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def render(selected):
        """Render the options list."""
        # Move cursor up and clear lines if re-rendering
        if hasattr(render, 'rendered'):
            sys.stdout.write(f"\x1b[{len(options)}A")  # Move up
            sys.stdout.write("\x1b[J")  # Clear to end
        render.rendered = True

        for i, (key, desc) in enumerate(options):
            if i == selected:
                console.print(f"  [green]>[/] [bold]{desc}[/]")
            else:
                console.print(f"    [dim]{desc}[/]")
        sys.stdout.flush()

    # Check if terminal supports interactive mode
    try:
        if not sys.stdin.isatty():
            raise OSError("Not a TTY")

        console.print(f"\n[bold]{prompt}[/]")
        console.print("[dim]↑/↓ to select, Enter to confirm[/]\n")

        selected = default
        render(selected)

        while True:
            key = get_key()
            if key == 'up':
                selected = (selected - 1) % len(options)
                render(selected)
            elif key == 'down':
                selected = (selected + 1) % len(options)
                render(selected)
            elif key == 'enter':
                console.print()  # New line after selection
                return options[selected][0]

    except (OSError, termios.error, KeyboardInterrupt, EOFError):
        # Fallback to number-based selection
        console.print(f"\n[bold]{prompt}[/]")
        for i, (key, desc) in enumerate(options):
            marker = "[green]>[/]" if i == default else " "
            console.print(f"  {marker} [{i+1}] {desc}")

        while True:
            try:
                choice = console.input(f"\n[dim]Enter choice [1-{len(options)}] (default: {default+1}):[/] ").strip()
                if not choice:
                    return options[default][0]
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
            except (ValueError, KeyboardInterrupt, EOFError):
                return options[default][0]
            console.print("[red]Invalid choice[/]")


def prompt_input(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt user for text input."""
    try:
        if secret:
            display = "***" if default else "[dim]not set[/]"
        else:
            display = default if default else "[dim]not set[/]"
        value = console.input(f"[bold]{prompt}[/] [dim]({display}):[/] ", password=secret).strip()
        return value if value else default
    except (KeyboardInterrupt, EOFError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Wizard
# ─────────────────────────────────────────────────────────────────────────────

def run_wizard() -> dict:
    """Run the full configuration wizard."""
    console.print(Panel.fit(
        "[bold]Manga Translation Server Setup[/]\n\n"
        "Configure OCR and translation for your system.\n"
        "Press [cyan]Enter[/] to accept defaults shown in parentheses.\n\n"
        "[dim]Press Ctrl+D to exit at any time.[/]",
        border_style="blue"
    ))

    cfg = load_config()

    try:
        # ══════════════════════════════════════════════════════════════════════
        # Step 1: Setup Type (Local / API / Mixed)
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 1: Setup Type[/]")
        console.print("[dim]Choose how OCR and Translation will be processed[/]")

        console.print(Panel(
            "[bold]Local[/]\n"
            "  [green]+[/] No API keys needed, fully offline\n"
            "  [green]+[/] No rate limits or usage costs\n"
            "  [green]+[/] Privacy - data stays on your machine\n"
            "  [red]-[/] Requires [yellow]llama.cpp[/] compilation (time & resource intensive)\n"
            "  [red]-[/] Needs GPU with sufficient VRAM (1.7-5.6GB)\n"
            "  [red]-[/] Slower than cloud APIs on weak hardware\n\n"
            "[bold]API[/]\n"
            "  [green]+[/] No local GPU required\n"
            "  [green]+[/] Fast processing (cloud infrastructure)\n"
            "  [green]+[/] No llama.cpp compilation needed\n"
            "  [red]-[/] Requires API keys (free tiers available)\n"
            "  [red]-[/] Rate limits apply (varies by provider)\n"
            "  [red]-[/] Data sent to cloud servers\n\n"
            "[bold]Mixed[/]\n"
            "  [green]+[/] Best of both worlds - flexibility\n"
            "  [green]+[/] Use local for one task, API for another\n"
            "  [red]-[/] May still need llama.cpp for local parts",
            title="[bold]Pros & Cons[/]",
            border_style="dim"
        ))

        setup_options = [
            ("local", "Local - Run everything on your machine [cyan](needs llama.cpp + GPU)[/]"),
            ("api", "API - Use cloud services [yellow](needs API keys, no GPU required)[/]"),
            ("mixed", "Mixed - Combine local and cloud [dim](choose per task)[/]"),
        ]

        # Detect current setup type from config
        current_ocr = cfg.get("ocr_method", "qwen_vlm")
        current_translate = cfg.get("translate_method", "hunyuan_mt")
        ocr_is_api = current_ocr in API_OCR_METHODS
        translate_is_api = current_translate in ("cerebras_api", "gemini_translate")

        if ocr_is_api and translate_is_api:
            current_setup = "api"
        elif not ocr_is_api and not translate_is_api:
            current_setup = "local"
        else:
            current_setup = "mixed"

        default_idx = next((i for i, (k, _) in enumerate(setup_options) if k == current_setup), 0)
        setup_type = prompt_choice("Select setup type:", setup_options, default=default_idx)

        # ══════════════════════════════════════════════════════════════════════
        # Step 2: OCR Method (filtered by setup type)
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 2: OCR Method[/]")
        console.print("[dim]Choose how to extract text from manga/comic images[/]")

        # Build OCR options based on setup type
        local_ocr_options = [
            ("qwen_vlm", "Qwen 3 VL 2B [cyan](~2.3GB VRAM)[/] - Good balance"),
            ("lfm_vlm", "LFM 2.5 VL 1.6B [cyan](~1.7GB VRAM)[/] - Smallest, fastest"),
            ("ministral_vlm_q8", "Ministral 3B Q8 [cyan](~4.5GB VRAM)[/] - Best local quality"),
        ]

        api_ocr_options = [
            ("oneocr_remote", "OneOCR Remote [cyan](Windows server on network)[/] - Fastest, most accurate"),
            ("gemini_api:gemma-3-27b-it", "Gemma 3 27B IT [green](14.4k req/day)[/] - Best for free tier"),
            ("gemini_api:gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite [red](20 req/day)[/] - Fast but limited"),
            ("gemini_api:gemini-3-flash-preview", "Gemini 3 Flash Preview [red](20 req/day)[/] - Newest but limited"),
        ]

        if PY == 'Windows':
            # Check if OneOCR is available (files exist or can get admin)
            oneocr_available = False
            oneocr_files_exist = os.path.exists(os.path.join(os.path.dirname(__file__), 'workflow', 'ocr', 'oneocr.dll'))

            if oneocr_files_exist:
                oneocr_available = True
            else:
                # Only show OneOCR if currently running as admin (can set up now)
                try:
                    import ctypes
                    oneocr_available = bool(ctypes.windll.shell32.IsUserAnAdmin())
                except:
                    oneocr_available = False

            if oneocr_available:
                local_ocr_options.insert(0, ("oneocr", "OneOCR [green](no VRAM, Windows only)[/] - Fastest & most accurate"))
                if not oneocr_files_exist:
                    console.print(Panel(
                        "[bold green]Windows Detected - OneOCR Recommended[/]\n\n"
                        "[bold]OneOCR[/] uses Windows' built-in Snipping Tool OCR engine:\n"
                        "  [green]+[/] [bold]10x faster[/] than local VLM models\n"
                        "  [green]+[/] [bold]Most accurate[/] for manga/comic text\n"
                        "  [green]+[/] [bold]No GPU/VRAM required[/]\n"
                        "  [green]+[/] Works offline\n\n"
                        "[yellow]⚠ Requires one-time Admin access[/]\n"
                        "  Setup will request Administrator privileges to copy OCR files\n"
                        "  from Windows Snipping Tool. You can Accept or Decline.\n\n"
                        "[dim]If declined, you'll need to use Local VLM or API options,\n"
                        "which are significantly slower (~10x) for OCR.[/]",
                        title="[bold cyan]OneOCR Setup[/]",
                        border_style="cyan"
                    ))
            else:
                console.print(Panel(
                    "[bold yellow]OneOCR Not Available[/]\n\n"
                    "OneOCR requires Administrator access to set up, but admin\n"
                    "privileges are not available in this session.\n\n"
                    "[dim]To use OneOCR, run the config wizard as Administrator.\n"
                    "For now, please choose from Local VLM or API options below.[/]",
                    title="[yellow]Windows OCR[/]",
                    border_style="yellow"
                ))

        if setup_type == "local":
            ocr_options = local_ocr_options
            console.print("[dim]Local models require llama.cpp and GPU VRAM[/]")
        elif setup_type == "api":
            ocr_options = api_ocr_options
            console.print(Panel(
                "[bold cyan]OneOCR Remote[/] (Recommended if you have Windows access)\n"
                "  [green]+[/] Fastest & most accurate OCR\n"
                "  [green]+[/] No rate limits - runs on your own Windows machine\n"
                "  [green]+[/] Works with Azure free tier Windows VM\n\n"
                "[bold]API Rate Limits (Free Tier):[/]\n"
                "  [green]gemma-3-27b-it[/]: 14,400 requests/day - Good for free use\n"
                "  [red]gemini-2.5-flash-lite[/]: 20 requests/day - Too limited\n"
                "  [red]gemini-3-flash-preview[/]: 20 requests/day - Too limited\n\n"
                "[dim]Gemini key: https://aistudio.google.com/apikey[/]",
                title="[yellow]OCR Options[/]",
                border_style="yellow"
            ))
        else:  # mixed
            ocr_options = local_ocr_options + api_ocr_options

        # Find current selection in options
        current_ocr_key = current_ocr
        if current_ocr == "gemini_api":
            current_model = cfg.get("gemini_model", "gemma-3-27b-it")
            current_ocr_key = f"gemini_api:{current_model}"

        default_idx = 0
        for i, (k, _) in enumerate(ocr_options):
            if k == current_ocr_key or k == current_ocr:
                default_idx = i
                break

        ocr_choice = prompt_choice("Select OCR method:", ocr_options, default=default_idx)

        # Parse OCR choice (handle gemini_api:model format)
        if ocr_choice.startswith("gemini_api:"):
            cfg["ocr_method"] = "gemini_api"
            cfg["gemini_model"] = ocr_choice.split(":", 1)[1]
        elif ocr_choice == "oneocr_remote":
            cfg["ocr_method"] = "oneocr_remote"
        else:
            cfg["ocr_method"] = ocr_choice

        # OneOCR Remote server URL
        if cfg["ocr_method"] == "oneocr_remote":
            console.print(Panel(
                "[bold cyan]OneOCR Remote Server Setup[/]\n\n"
                "Run OneOCR on a Windows machine and connect to it over the network.\n"
                "This gives you the fastest, most accurate OCR without needing Windows locally.\n\n"
                "[bold]Step 1: Copy files to Windows machine[/]\n"
                "  Copy the entire [cyan]oneocr_server/[/] folder to your Windows machine.\n"
                "  Location: [dim]" + os.path.join(os.path.dirname(__file__), 'oneocr_server') + "[/]\n\n"
                "[bold]Step 2: Run server (first time needs Administrator)[/]\n"
                "  [cyan]cd oneocr_server[/]\n"
                "  [cyan]python server.py[/]\n\n"
                "  First run will:\n"
                "  • Request Administrator access (to copy Windows OCR files)\n"
                "  • Install Flask if needed\n"
                "  • Display the Network URL\n\n"
                "[bold]Step 3: Note the Network URL[/]\n"
                "  Server will display something like:\n"
                "  [green]Network URL:  http://192.168.1.50:5050[/]\n\n"
                "[bold]Step 4: Enter URL below[/]\n"
                "  Paste the Network URL when prompted.\n\n"
                "[bold yellow]Azure Free Tier Option:[/]\n"
                "  Create a free Windows VM (B1s tier) on Azure and run the server there.\n"
                "  Open port 5050 in the VM's Network Security Group.",
                title="[bold]OneOCR Remote Setup Guide[/]",
                border_style="cyan"
            ))
            cfg["oneocr_server_url"] = prompt_input(
                "OneOCR Server URL",
                cfg.get("oneocr_server_url", "http://192.168.1.x:5050")
            )
            if not cfg["oneocr_server_url"] or cfg["oneocr_server_url"] == "http://192.168.1.x:5050":
                cfg["oneocr_server_url"] = ""
                warn("No server URL set - add 'oneocr_server_url' to config.json after setting up the server")

        # Gemini API key if using gemini_api OCR
        if cfg["ocr_method"] == "gemini_api":
            console.print("[dim]Input is hidden for security - just type and press Enter[/]")
            cfg["gemini_api_key"] = prompt_input(
                "Gemini API Key (from aistudio.google.com/apikey)",
                cfg.get("gemini_api_key", ""),
                secret=True
            )
            if not cfg["gemini_api_key"]:
                warn("No Gemini API key set - add it to config.json or set GEMINI_API_KEY env var")

        # ══════════════════════════════════════════════════════════════════════
        # Step 3: Translation Method (filtered by setup type)
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 3: Translation Method[/]")
        console.print("[dim]Choose how to translate extracted text[/]")

        local_translate_options = [
            ("hunyuan_mt", "HunyuanMT [cyan](~1.1GB VRAM)[/] - Dedicated MT model, good quality"),
        ]

        api_translate_options = [
            ("cerebras_api", "Cerebras API [green](1M tokens/day)[/] - Fastest (1-2s), best quality"),
            ("gemini_translate:gemma-3-27b-it", "Gemma 3 27B IT [green](14.4k req/day)[/] - Good for free tier"),
            ("gemini_translate:gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite [red](20 req/day)[/] - Limited"),
            ("gemini_translate:gemini-3-flash-preview", "Gemini 3 Flash Preview [red](20 req/day)[/] - Limited"),
        ]

        if setup_type == "local":
            translate_options = local_translate_options
            console.print("[dim]Local translation requires llama.cpp and GPU VRAM[/]")
        elif setup_type == "api":
            translate_options = api_translate_options
            console.print(Panel(
                "[bold]API Rate Limits (Free Tier):[/]\n"
                "  [green]Cerebras[/]: 1 million tokens/day, [bold]1-2 second response[/] - [bold]RECOMMENDED[/]\n"
                "  [green]gemma-3-27b-it[/]: 14,400 requests/day - Good alternative\n"
                "  [red]gemini-2.5/3-flash[/]: 20 requests/day - Too limited\n\n"
                "[dim]Cerebras key: https://cloud.cerebras.ai/[/]\n"
                "[dim]Gemini key: https://aistudio.google.com/apikey[/]",
                title="[yellow]Rate Limits[/]",
                border_style="yellow"
            ))
        else:  # mixed
            translate_options = local_translate_options + api_translate_options

        # Find current selection
        current_translate_key = current_translate
        default_idx = 0
        for i, (k, _) in enumerate(translate_options):
            if k == current_translate_key or k == current_translate:
                default_idx = i
                break

        translate_choice = prompt_choice("Select translation method:", translate_options, default=default_idx)

        # Parse translation choice
        if translate_choice.startswith("gemini_translate:"):
            cfg["translate_method"] = "gemini_translate"
            cfg["gemini_translate_model"] = translate_choice.split(":", 1)[1]
            # Ensure we have API key for Gemini translation
            if not cfg.get("gemini_api_key"):
                console.print("[dim]Input is hidden for security - just type and press Enter[/]")
                cfg["gemini_api_key"] = prompt_input(
                    "Gemini API Key (from aistudio.google.com/apikey)",
                    cfg.get("gemini_api_key", ""),
                    secret=True
                )
        else:
            cfg["translate_method"] = translate_choice

        # API key for Cerebras
        if cfg["translate_method"] == "cerebras_api":
            console.print("[dim]Input is hidden for security - just type and press Enter[/]")
            cfg["cerebras_api_key"] = prompt_input(
                "Cerebras API Key (from cloud.cerebras.ai)",
                cfg.get("cerebras_api_key", ""),
                secret=True
            )
            if not cfg["cerebras_api_key"]:
                warn("No API key set - add it to config.json later")

        # ══════════════════════════════════════════════════════════════════════
        # Step 4: Target Language
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 4: Target Language[/]")
        console.print("[dim]Language to translate manga into[/]")

        lang_options = [(code, name) for code, name in LANGUAGES.items()]
        current_lang = cfg.get("target_language", "en")
        default_idx = next((i for i, (k, _) in enumerate(lang_options) if k == current_lang), 0)

        target_lang = prompt_choice("Select target language:", lang_options, default=default_idx)
        cfg["target_language"] = target_lang

        # ══════════════════════════════════════════════════════════════════════
        # Step 5: Server Settings
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 5: Server Settings[/]")

        cfg["server_port"] = int(prompt_input(
            "Server port",
            str(cfg.get("server_port", 5000))
        ) or 5000)

        cfg["output_dir"] = prompt_input(
            "Output directory",
            cfg.get("output_dir", "output")
        )

        # ══════════════════════════════════════════════════════════════════════
        # Summary & Save
        # ══════════════════════════════════════════════════════════════════════
        show_summary(cfg)
        show_vram_estimate(cfg)

        confirm = prompt_choice("Save this configuration?", [("yes", "Yes"), ("no", "No")], default=0)
        if confirm == "yes":
            save_config(cfg)
            return cfg
        else:
            info("Configuration not saved")
            return None

    except EOFError:
        console.print("\n[yellow]Exited[/]")
        return None


def show_vram_estimate(cfg: dict):
    """Show estimated VRAM usage."""
    ocr = cfg.get("ocr_method", "qwen_vlm")
    translate = cfg.get("translate_method", "qwen_vlm")
    sequential = cfg.get("sequential_model_loading", False)

    ocr_vram = 0
    translate_vram = 0
    components = []

    if ocr == "qwen_vlm":
        ocr_vram = 2.3  # Q8_0 ~1.8GB + mmproj Q8_0 ~445MB
        components.append("Qwen VL ~2.3GB (OCR)")
    elif ocr == "lfm_vlm":
        ocr_vram = 1.7  # Q8_0 ~1.25GB + mmproj ~450MB
        components.append("LFM VL ~1.7GB (OCR)")
    elif ocr == "ministral_vlm_q8":
        ocr_vram = 4.5  # Q8_0 ~3.65GB + mmproj ~0.85GB
        components.append("Ministral Q8 ~4.5GB (OCR)")
    elif ocr == "gemini_api":
        model = cfg.get("gemini_model", "gemma-3-27b-it")
        components.append(f"Gemini API: {model} (cloud, no VRAM)")
    elif ocr == "oneocr":
        components.append("OneOCR (no VRAM)")
    elif ocr == "oneocr_remote":
        server_url = cfg.get("oneocr_server_url", "not set")
        components.append(f"OneOCR Remote: {server_url} (network, no VRAM)")

    if translate == "hunyuan_mt":
        translate_vram = 1.1
        components.append("HunyuanMT ~1.1GB (Translation)")
    elif translate == "cerebras_api":
        components.append("Cerebras API (cloud, 1M tokens/day)")
    elif translate == "gemini_translate":
        model = cfg.get("gemini_translate_model", "gemma-3-27b-it")
        components.append(f"Gemini Translate: {model} (cloud, no VRAM)")

    # Calculate total based on sequential mode
    if sequential and ocr_vram > 0 and translate_vram > 0:
        # Sequential: only one model loaded at a time
        total_vram = max(ocr_vram, translate_vram)
        console.print(f"\n[bold]Estimated VRAM:[/] [cyan]~{total_vram}GB[/] [dim](sequential mode - one model at a time)[/]")
        console.print(f"  [dim]• Peak usage: ~{total_vram}GB (larger of OCR/translate)[/]")
    elif ocr_vram > 0 or translate_vram > 0:
        # Parallel: both models loaded
        total_vram = ocr_vram + translate_vram
        console.print(f"\n[bold]Estimated VRAM:[/] [cyan]~{total_vram}GB[/]")
    else:
        console.print(f"\n[bold]Estimated VRAM:[/] [green]0GB (all cloud APIs)[/]")

    for c in components:
        console.print(f"  [dim]• {c}[/]")

    if sequential:
        console.print(f"  [yellow]• Sequential mode: SLOW but low VRAM[/]")


def show_summary(cfg: dict):
    """Display configuration summary."""
    console.print("\n[bold]Configuration Summary:[/]")
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    ocr = cfg.get("ocr_method", "qwen_vlm")
    translate = cfg.get("translate_method", "qwen_vlm")
    lang = cfg.get("target_language", "en")
    sequential = cfg.get("sequential_model_loading", False)

    ocr_names = {
        "qwen_vlm": "Qwen 3 VL",
        "lfm_vlm": "LFM 2.5 VL",
        "ministral_vlm_q8": "Ministral 3B Q8",
        "gemini_api": "Gemini API",
        "oneocr": "OneOCR",
        "oneocr_remote": "OneOCR Remote"
    }
    translate_names = {
        "qwen_vlm": "Qwen 3 VL",
        "hunyuan_mt": "HunyuanMT",
        "cerebras_api": "Cerebras API",
        "gemini_translate": "Gemini API"
    }

    # OCR row
    ocr_display = ocr_names.get(ocr, ocr)
    if ocr == "gemini_api":
        model = cfg.get("gemini_model", "gemma-3-27b-it")
        ocr_display = f"Gemini API ({model})"
    table.add_row("OCR", f"[bold]{ocr_display}[/]")
    if ocr == "gemini_api":
        table.add_row("  API Key", "[green]Set[/]" if cfg.get("gemini_api_key") else "[yellow]Not set[/]")
    elif ocr == "oneocr_remote":
        server_url = cfg.get("oneocr_server_url", "")
        table.add_row("  Server URL", f"[green]{server_url}[/]" if server_url else "[yellow]Not set[/]")

    # Translation row
    translate_display = translate_names.get(translate, translate)
    if translate == "gemini_translate":
        model = cfg.get("gemini_translate_model", "gemma-3-27b-it")
        translate_display = f"Gemini API ({model})"
    table.add_row("Translation", f"[bold]{translate_display}[/]")
    if translate == "cerebras_api":
        table.add_row("  API Key", "[green]Set[/]" if cfg.get("cerebras_api_key") else "[yellow]Not set[/]")
    elif translate == "gemini_translate":
        table.add_row("  API Key", "[green]Set[/]" if cfg.get("gemini_api_key") else "[yellow]Not set[/]")

    table.add_row("", "")
    table.add_row("Target Language", f"[bold]{LANGUAGES.get(lang, lang)}[/]")
    table.add_row("Server Port", str(cfg.get("server_port", 5000)))
    table.add_row("Output Dir", cfg.get("output_dir", "output"))

    # Show sequential mode if enabled
    if sequential:
        table.add_row("", "")
        table.add_row("Sequential Loading", "[yellow]Enabled[/] [dim](low VRAM mode)[/]")

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='Configure Manga Translation Server')
    p.add_argument('--show', action='store_true', help='Show current configuration')
    p.add_argument('--reset', action='store_true', help='Reset to defaults')
    p.add_argument('--json', action='store_true', help='Output config as JSON')
    args = p.parse_args()

    if args.json:
        cfg = load_config()
        print(json.dumps(cfg, indent=2))
        return 0

    if args.show:
        if not config_exists():
            warn(f"{CONFIG_FILE} not found - showing defaults")
        cfg = load_config()
        show_summary(cfg)
        show_vram_estimate(cfg)
        return 0

    if args.reset:
        confirm = prompt_choice("Reset to default configuration?", [("no", "No"), ("yes", "Yes")], default=0)
        if confirm == "yes":
            save_config(DEFAULT_CONFIG.copy())
        return 0

    run_wizard()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/]")
        sys.exit(1)
    except EOFError:
        console.print("\n[yellow]Exited[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)
