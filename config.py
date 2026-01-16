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

    # OCR method: "qwen_vlm" | "lfm_vlm" | "ministral_vlm_q4" | "ministral_vlm_q8" | "oneocr"
    "ocr_method": "qwen_vlm",

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

    # ══════════════════════════════════════════════════════════════════════════
    # Server Settings (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    "ocr_server_url": "http://localhost:8080",
    "translate_server_url": "http://localhost:8081",
    "auto_start_servers": True,

    # llama-server parameters
    "llama_context_size": 2048,
    "llama_gpu_layers": 99,

    # OCR grid settings (for VLM batch processing)
    "ocr_grid_max_cells": 9,  # Max bubbles per grid (e.g., 9 = 3x3)

    # ══════════════════════════════════════════════════════════════════════════
    # Detection Settings (config.json only)
    # ══════════════════════════════════════════════════════════════════════════

    "detection_threshold": 0.5,
    "detection_padding": 5,

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

# Detection settings
def get_detection_threshold() -> float:
    return get("detection_threshold", 0.5)

def get_detection_padding() -> int:
    return get("detection_padding", 5)

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
    vlm_methods = ("qwen_vlm", "lfm_vlm", "ministral_vlm_q4", "ministral_vlm_q8")
    return ocr in vlm_methods or translate in ("qwen_vlm", "hunyuan_mt")

def uses_same_model_for_ocr_and_translate() -> bool:
    """Check if OCR and translation use the same model."""
    return get_ocr_method() == "qwen_vlm" and get_translate_method() == "qwen_vlm"

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
        # Step 1: OCR Method
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 1: OCR Method[/]")
        console.print("[dim]Choose how to extract text from images[/]")

        ocr_options = [
            ("qwen_vlm", f"Qwen 3 VL 2B - Vision model for OCR [cyan](~2.3GB VRAM)[/]"),
            ("lfm_vlm", f"LFM 2.5 VL 1.6B - Smaller & faster [cyan](~1.7GB VRAM)[/]"),
            ("ministral_vlm_q4", f"Ministral 3B Q4 - Good quality [cyan](~3.0GB VRAM)[/]"),
            ("ministral_vlm_q8", f"Ministral 3B Q8 - Best quality [cyan](~4.5GB VRAM)[/]"),
        ]

        if PY == 'Windows':
            ocr_options.insert(0, ("oneocr", "OneOCR - Fastest & most accurate [green](no VRAM, Windows only)[/]"))

        current_ocr = cfg.get("ocr_method", "qwen_vlm")
        default_idx = next((i for i, (k, _) in enumerate(ocr_options) if k == current_ocr), 0)

        ocr_method = prompt_choice("Select OCR method:", ocr_options, default=default_idx)
        cfg["ocr_method"] = ocr_method

        # ══════════════════════════════════════════════════════════════════════
        # Step 2: Translation Method
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 2: Translation Method[/]")
        console.print("[dim]Choose how to translate extracted text[/]")

        translate_options = []

        # Note: VLM translation disabled due to Qwen 3 VL hallucination issues
        # Always use separate translation model

        translate_options.extend([
            ("hunyuan_mt", f"HunyuanMT - Dedicated translator [cyan](~1.1GB VRAM)[/]"),
            ("cerebras_api", f"Cerebras API - Cloud LLM, best quality [yellow](needs API key)[/]"),
        ])

        current_translate = cfg.get("translate_method", "hunyuan_mt")
        default_idx = next((i for i, (k, _) in enumerate(translate_options) if k == current_translate), 0)

        translate_method = prompt_choice("Select translation method:", translate_options, default=default_idx)
        cfg["translate_method"] = translate_method

        # API key if needed
        if translate_method == "cerebras_api":
            cfg["cerebras_api_key"] = prompt_input(
                "Cerebras API Key",
                cfg.get("cerebras_api_key", ""),
                secret=True
            )
            if not cfg["cerebras_api_key"]:
                warn("No API key set - add it to config.json later")

        # ══════════════════════════════════════════════════════════════════════
        # Step 3: Target Language
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 3: Target Language[/]")
        console.print("[dim]Language to translate manga into[/]")

        lang_options = [(code, name) for code, name in LANGUAGES.items()]
        current_lang = cfg.get("target_language", "en")
        default_idx = next((i for i, (k, _) in enumerate(lang_options) if k == current_lang), 0)

        target_lang = prompt_choice("Select target language:", lang_options, default=default_idx)
        cfg["target_language"] = target_lang

        # ══════════════════════════════════════════════════════════════════════
        # Step 4: Server Settings
        # ══════════════════════════════════════════════════════════════════════
        console.print("\n[bold cyan]Step 4: Server Settings[/]")

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

    total_vram = 0
    components = []

    if ocr == "qwen_vlm":
        total_vram += 2.3  # Q8_0 ~1.8GB + mmproj Q8_0 ~445MB
        components.append("Qwen VL ~2.3GB (OCR)")
    elif ocr == "lfm_vlm":
        total_vram += 1.7  # Q8_0 ~1.25GB + mmproj ~450MB
        components.append("LFM VL ~1.7GB (OCR)")
    elif ocr == "ministral_vlm_q4":
        total_vram += 3.0  # Q4_K_M ~2.15GB + mmproj ~0.85GB
        components.append("Ministral Q4 ~3.0GB (OCR)")
    elif ocr == "ministral_vlm_q8":
        total_vram += 4.5  # Q8_0 ~3.65GB + mmproj ~0.85GB
        components.append("Ministral Q8 ~4.5GB (OCR)")
    elif ocr == "oneocr":
        components.append("OneOCR (no VRAM)")

    if translate == "hunyuan_mt":
        total_vram += 1.1
        components.append("HunyuanMT ~1.1GB")
    elif translate == "cerebras_api":
        components.append("Cerebras API (cloud)")

    console.print(f"\n[bold]Estimated VRAM:[/] [cyan]~{total_vram}GB[/]")
    for c in components:
        console.print(f"  [dim]• {c}[/]")


def show_summary(cfg: dict):
    """Display configuration summary."""
    console.print("\n[bold]Configuration Summary:[/]")
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    ocr = cfg.get("ocr_method", "qwen_vlm")
    translate = cfg.get("translate_method", "qwen_vlm")
    lang = cfg.get("target_language", "en")

    ocr_names = {"qwen_vlm": "Qwen 3 VL", "lfm_vlm": "LFM 2.5 VL", "ministral_vlm_q4": "Ministral 3B Q4", "ministral_vlm_q8": "Ministral 3B Q8", "oneocr": "OneOCR"}
    translate_names = {"qwen_vlm": "Qwen 3 VL", "hunyuan_mt": "HunyuanMT", "cerebras_api": "Cerebras API"}

    table.add_row("OCR", f"[bold]{ocr_names.get(ocr, ocr)}[/]")
    table.add_row("Translation", f"[bold]{translate_names.get(translate, translate)}[/]")
    if translate == "cerebras_api":
        table.add_row("  API Key", "[green]Set[/]" if cfg.get("cerebras_api_key") else "[yellow]Not set[/]")

    table.add_row("", "")
    table.add_row("Target Language", f"[bold]{LANGUAGES.get(lang, lang)}[/]")
    table.add_row("Server Port", str(cfg.get("server_port", 5000)))
    table.add_row("Output Dir", cfg.get("output_dir", "output"))

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
