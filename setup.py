#!/usr/bin/env python3
"""Manga Translation Server - Setup

Reads configuration from config.json and installs dependencies based on selected mode.
If config.json doesn't exist, runs the configuration wizard first.
"""

import subprocess, sys, os, platform, shutil, re

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
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

# Import config management
from config import (
    load_config, config_exists, run_wizard, show_summary, show_vram_estimate,
    get_venv_path, get_python_version, get_server_port, get_target_language,
    get_ocr_method, get_ocr_model, get_ocr_mmproj, get_translate_method, get_translate_model,
    get_script_path, get_llama_cli_path, get_cerebras_api_key, get_output_dir,
    get_ocr_server_url, get_translate_server_url, get_auto_start_servers,
    get_llama_context_size, get_llama_gpu_layers,
    needs_llama, find_llama, download_mmproj, build_llama_command, LANGUAGES,
    ok, warn, err, info, CONFIG_FILE
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, desc=None, env=None, capture=False, timeout=300):
    """Run command with optional progress spinner."""
    if desc:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as p:
            p.add_task(desc, total=None)
            r = subprocess.run(cmd if isinstance(cmd, list) else cmd, shell=isinstance(cmd, str),
                              env=env, capture_output=capture, text=capture, timeout=timeout)
    else:
        r = subprocess.run(cmd if isinstance(cmd, list) else cmd, shell=isinstance(cmd, str),
                          env=env, capture_output=capture, text=capture, timeout=timeout)
    return r

def venv_python():
    venv = get_venv_path()
    return os.path.join(venv, 'Scripts' if PY == 'Windows' else 'bin', 'python' + ('.exe' if PY == 'Windows' else ''))

def venv_env():
    venv = get_venv_path()
    env = os.environ.copy()
    bin_dir = os.path.join(os.getcwd(), venv, 'Scripts' if PY == 'Windows' else 'bin')
    env['PATH'] = bin_dir + os.pathsep + env.get('PATH', '')
    env['VIRTUAL_ENV'] = os.path.join(os.getcwd(), venv)
    # Add tensorrt_libs to library path
    try:
        r = subprocess.run([venv_python(), '-c', 'import site; print(site.getsitepackages()[0])'],
                          capture_output=True, text=True)
        if r.returncode == 0:
            trt = os.path.join(r.stdout.strip(), 'tensorrt_libs')
            if PY == 'Linux':
                env['LD_LIBRARY_PATH'] = trt + ':' + env.get('LD_LIBRARY_PATH', '')
            elif PY == 'Windows':
                env['PATH'] = trt + os.pathsep + env['PATH']
    except: pass
    return env

# ─────────────────────────────────────────────────────────────────────────────
# Core Setup Steps
# ─────────────────────────────────────────────────────────────────────────────

def install_uv():
    if shutil.which('uv'):
        ok("uv installed")
        return True
    info("Installing uv package manager...")
    for cmd in [
        [sys.executable, '-m', 'pip', 'install', 'uv'],
        [sys.executable, '-m', 'pip', 'install', 'uv', '--break-system-packages'],
        'curl -LsSf https://astral.sh/uv/install.sh | sh' if PY != 'Windows' else None
    ]:
        if cmd and run(cmd, capture=True).returncode == 0:
            ok("uv installed")
            return True
    err("Failed to install uv - run: pip install uv")
    return False

def create_venv():
    venv = get_venv_path()
    py_ver = get_python_version()

    if os.path.exists(venv):
        r = run([venv_python(), '--version'], capture=True)
        ver = r.stdout.strip() if r.returncode == 0 else ''
        if '3.14' not in ver and ver:
            ok(f"Venv exists ({ver})")
            return True
        info("Recreating venv...")
        shutil.rmtree(venv)

    info(f"Creating venv with Python {py_ver}...")
    if run(['uv', 'venv', venv, '--python', py_ver], capture=True).returncode == 0:
        ok("Venv created")
        return True
    err("Failed to create venv")
    return False

def install_deps():
    console.print("\n[bold cyan]Installing Dependencies[/]")
    if run(['uv', 'pip', 'install', '-r', 'requirements.txt'], "Installing packages...", env=venv_env()).returncode != 0:
        err("Failed to install requirements")
        return False
    ok("Dependencies installed")
    return True

def install_runtime():
    console.print("\n[bold cyan]Installing ONNX Runtime[/]")
    script = get_script_path("install_runtime")
    if run([venv_python(), script], "Detecting hardware...", env=venv_env()).returncode != 0:
        err("Failed to install runtime")
        return False
    ok("ONNX Runtime installed")
    return True

def get_cuda_version():
    """Detect CUDA major.minor version from nvidia-smi."""
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', r.stdout)
        if m:
            return int(m.group(1)), int(m.group(2))
    except: pass
    return None, None

def install_tensorrt():
    if not shutil.which('nvidia-smi'):
        return True
    console.print("\n[bold cyan]Installing TensorRT (NVIDIA GPU Optimization)[/]")

    cuda_major, cuda_minor = get_cuda_version()
    if not cuda_major:
        warn("Could not detect CUDA version")
        return True

    info(f"CUDA {cuda_major}.{cuda_minor} detected")

    if cuda_major >= 13:
        pkg = 'tensorrt'
    else:
        pkg = 'tensorrt-cu12'

    if run(['uv', 'pip', 'install', pkg], f"Installing {pkg}...", env=venv_env()).returncode == 0:
        ok(f"{pkg} installed")

        # Show TensorRT caching info
        console.print()
        console.print(Panel.fit(
            "[bold]TensorRT Engine Caching[/]\n\n"
            "TensorRT builds optimized inference engines for your GPU.\n"
            "This happens [bold]once[/] on first run and is cached for faster startup.\n\n"
            "[dim]Environment variables (already set by server.py):[/]\n"
            "  • ORT_TENSORRT_ENGINE_CACHE_ENABLE=1\n"
            "  • ORT_TENSORRT_FP16_ENABLE=1\n\n"
            "[bold yellow]First inference will be slower[/] while TensorRT builds\n"
            "the optimized engine. Subsequent runs will be much faster.",
            title="[bold green]TensorRT Installed[/]",
            border_style="green",
            box=box.ROUNDED
        ))
    else:
        warn(f"{pkg} failed (optional)")
        console.print("[dim]  TensorRT is optional but recommended for NVIDIA GPUs[/]")
        console.print("[dim]  You can install it manually later: pip install tensorrt[/]")
    return True

def download_models():
    console.print("\n[bold cyan]Downloading Models[/]")
    script = get_script_path("download_model")
    if run([venv_python(), script], "Downloading...", env=venv_env()).returncode != 0:
        err("Failed to download models")
        return False
    ok("Models downloaded")

    # Platform-specific exports
    env = venv_env()
    if PY == 'Darwin':
        info("Exporting CoreML model...")
        run([venv_python(), get_script_path("export_coreml")], env=env, capture=True)
    elif PY == 'Windows' or shutil.which('nvidia-smi'):
        info("Exporting static ONNX...")
        run([venv_python(), get_script_path("export_static")], env=env, capture=True)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Component Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_llama():
    """Setup llama.cpp for local inference."""
    console.print("\n[bold cyan]Setting up llama.cpp[/]")
    llama = find_llama()

    if llama:
        ok(f"llama found: [dim]{llama}[/]")
        return True

    # Check if llama_cli_path is set but invalid
    cfg_path = get_llama_cli_path()
    config_info = ""
    if cfg_path:
        if os.path.isdir(cfg_path):
            config_info = (
                f"[bold yellow]Config Issue:[/]\n"
                f"  llama_cli_path: [cyan]{cfg_path}[/]\n"
                f"  Directory exists, but [bold]llama-server[/] not found inside.\n"
                f"  Expected: [cyan]{os.path.join(cfg_path, 'llama-server')}[/]\n\n"
            )
        elif not os.path.exists(cfg_path):
            config_info = (
                f"[bold yellow]Config Issue:[/]\n"
                f"  llama_cli_path: [cyan]{cfg_path}[/]\n"
                f"  [red]Path does not exist[/]\n\n"
            )
        else:
            config_info = (
                f"[bold yellow]Config Issue:[/]\n"
                f"  llama_cli_path: [cyan]{cfg_path}[/]\n"
                f"  File exists but is not executable or not llama-server\n\n"
            )

    console.print()

    # Build platform-specific instructions
    has_nvidia = shutil.which('nvidia-smi') is not None

    if PY == 'Darwin':
        # macOS instructions - Metal GPU is auto-enabled
        install_instructions = (
            "[bold]Option 1: Homebrew (recommended)[/]\n"
            "  [cyan]brew install llama.cpp[/]\n\n"
            "[bold]Option 2: Build from source[/]\n"
            "  [cyan]git clone https://github.com/ggerganov/llama.cpp[/]\n"
            "  [cyan]cd llama.cpp[/]\n"
            "  [cyan]cmake -B build -DGGML_METAL=ON[/]\n"
            "  [cyan]cmake --build build --config Release -j[/]\n\n"
            "[bold yellow]Note:[/] Metal GPU acceleration is enabled by default on macOS.\n\n"
            "[bold]Option 3: Pre-built binaries[/]\n"
            "  [dim]https://github.com/ggerganov/llama.cpp/releases[/]"
        )
    elif has_nvidia:
        # Linux/Windows with NVIDIA GPU - need CUDA and dev libraries
        install_instructions = (
            "[bold]1. Install dependencies:[/]\n"
            "  [cyan]sudo apt install libcurl4-openssl-dev libssl-dev ccache[/]\n\n"
            "[bold]2. Build from source (with CUDA):[/]\n"
            "  [cyan]git clone https://github.com/ggerganov/llama.cpp[/]\n"
            "  [cyan]cd llama.cpp[/]\n"
            "  [cyan]cmake -B build -DGGML_CUDA=ON[/]\n"
            "  [cyan]cmake --build build --config Release -j[/]\n\n"
            "[bold yellow]Notes:[/]\n"
            "  • [bold]-DGGML_CUDA=ON[/] enables GPU acceleration\n"
            "  • curl/SSL auto-detected if dev libs installed\n\n"
            "[bold]Pre-built binaries:[/]\n"
            "  [dim]https://github.com/ggerganov/llama.cpp/releases[/]\n"
            "  [dim](Choose the CUDA version matching your GPU)[/]"
        )
    else:
        # Linux/Windows without NVIDIA - CPU only
        install_instructions = (
            "[bold]1. Install dependencies:[/]\n"
            "  [cyan]sudo apt install libcurl4-openssl-dev libssl-dev ccache[/]\n\n"
            "[bold]2. Build from source:[/]\n"
            "  [cyan]git clone https://github.com/ggerganov/llama.cpp[/]\n"
            "  [cyan]cd llama.cpp[/]\n"
            "  [cyan]cmake -B build[/]\n"
            "  [cyan]cmake --build build --config Release -j[/]\n\n"
            "[bold]Pre-built binaries:[/]\n"
            "  [dim]https://github.com/ggerganov/llama.cpp/releases[/]"
        )

    console.print(Panel.fit(
        "[bold yellow]llama.cpp not found[/]\n\n"
        f"{config_info}"
        f"Install llama.cpp for local inference:\n\n"
        f"{install_instructions}\n\n"
        "[bold]After installing:[/]\n"
        "  • Add to PATH, or\n"
        "  • Set [cyan]llama_cli_path[/] in config.json to the [bold]bin directory[/]\n"
        "    containing llama-server (e.g., [dim]/path/to/llama.cpp/build/bin[/])",
        title="[bold red]llama.cpp Required[/]",
        border_style="yellow",
        box=box.ROUNDED
    ))
    console.print()
    warn("Local models require llama.cpp")
    return False

def setup_windows_ocr():
    """Copy OCR files from Windows Snipping Tool (OneOCR)."""
    if PY != 'Windows':
        return True

    dst = os.path.join(os.path.dirname(__file__), 'workflow', 'ocr')
    if os.path.exists(os.path.join(dst, 'oneocr.dll')):
        ok("OneOCR files present")
        return True

    console.print("\n[bold cyan]Setting up OneOCR[/]")

    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            warn("Admin required - relaunching...")
            script = os.path.abspath(sys.argv[0])
            args = ' '.join(f'"{a}"' for a in sys.argv[1:])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {args}', None, 1)
            sys.exit(0)
    except: pass

    base = "C:\\Program Files\\WindowsApps"
    src = None
    try:
        for f in os.listdir(base):
            if f.startswith("Microsoft.ScreenSketch") and "x64" in f:
                snip = os.path.join(base, f, "SnippingTool")
                if os.path.exists(snip):
                    src = snip
                    break
    except:
        err("Cannot access WindowsApps")
        return False

    if not src:
        err("Snipping Tool not found")
        return False

    os.makedirs(dst, exist_ok=True)
    copied = 0
    for f in os.listdir(src):
        if f.endswith('.dll') or f.endswith('.onemodel'):
            shutil.copy2(os.path.join(src, f), dst)
            copied += 1

    ok(f"Copied {copied} OneOCR files")
    return True

def setup_ocr():
    """Setup OCR based on config."""
    cfg = load_config()
    ocr_method = cfg.get("ocr_method", "qwen_vlm")

    console.print(f"\n[bold cyan]Setting up OCR ({ocr_method})[/]")

    if ocr_method == "oneocr":
        if PY != 'Windows':
            err("OneOCR is only available on Windows")
            return False
        return setup_windows_ocr()

    elif ocr_method in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8"):
        info(f"Model: {cfg.get('ocr_model')}")
        return setup_llama()

    return True

def setup_translation():
    """Setup translation based on config."""
    cfg = load_config()
    translate_method = cfg.get("translate_method", "qwen_vlm")

    console.print(f"\n[bold cyan]Setting up Translation ({translate_method})[/]")

    if translate_method == "cerebras_api":
        if cfg.get("cerebras_api_key"):
            ok("Cerebras API key configured")
        else:
            warn("No API key set in config.json")
            info("Add 'cerebras_api_key' to config.json or pass in requests")
        return True

    elif translate_method == "qwen_vlm":
        if cfg.get("ocr_method") in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8"):
            info("Using same VLM for OCR and translation")
            return True  # Already set up in OCR step
        else:
            info(f"Model: {cfg.get('translate_model')}")
            return setup_llama()

    elif translate_method == "hunyuan_mt":
        info(f"Model: {cfg.get('translate_model')}")
        return setup_llama()

    return True

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify():
    console.print("\n[bold cyan]Verification[/]")
    env = venv_env()
    cfg = load_config()

    ocr_method = cfg.get("ocr_method", "qwen_vlm")
    translate_method = cfg.get("translate_method", "qwen_vlm")
    target_lang = cfg.get("target_language", "en")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Component", style="dim")
    table.add_column("Status")

    # Target language
    table.add_row("Target", f"[bold]{LANGUAGES.get(target_lang, target_lang)}[/]")
    table.add_row("", "")

    # OCR
    ocr_names = {"qwen_vlm": "Qwen VLM", "lfm_vlm": "LFM VL", "ministral_vlm_q8": "Ministral Q8", "oneocr": "OneOCR"}
    table.add_row("OCR", f"[bold]{ocr_names.get(ocr_method, ocr_method)}[/]")

    if ocr_method == "oneocr":
        ocr_ok = os.path.exists(os.path.join('workflow', 'ocr', 'oneocr.dll'))
        table.add_row("  Status", "[green]✓ Ready[/]" if ocr_ok else "[red]✗ Not setup[/]")
    elif ocr_method in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8"):
        table.add_row("  Model", cfg.get("ocr_model", ""))

    # Translation
    translate_names = {"qwen_vlm": "Qwen VLM", "hunyuan_mt": "HunyuanMT", "cerebras_api": "Cerebras API"}
    table.add_row("", "")
    table.add_row("Translation", f"[bold]{translate_names.get(translate_method, translate_method)}[/]")

    if translate_method == "cerebras_api":
        table.add_row("  API Key", "[green]✓ Set[/]" if cfg.get("cerebras_api_key") else "[yellow]⚠ Not set[/]")
    elif translate_method == "qwen_vlm" and ocr_method in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8"):
        table.add_row("  Model", "[dim]Same as OCR[/]")
    elif translate_method in ("qwen_vlm", "hunyuan_mt"):
        table.add_row("  Model", cfg.get("translate_model", ""))

    # llama.cpp (if needed)
    if needs_llama():
        table.add_row("", "")
        llama = find_llama()
        llama_cfg = get_llama_cli_path()
        if llama:
            table.add_row("llama.cpp", f"[green]✓[/] {llama}")
        elif llama_cfg:
            table.add_row("llama.cpp", f"[red]✗ Not found[/]")
            table.add_row("  llama_cli_path", f"[yellow]{llama_cfg}[/]")
            if os.path.isdir(llama_cfg):
                table.add_row("  Status", "[yellow]Directory exists, llama-server not found inside[/]")
            elif not os.path.exists(llama_cfg):
                table.add_row("  Status", "[red]Path does not exist[/]")
        else:
            table.add_row("llama.cpp", "[red]✗ Not found (set llama_cli_path in config.json)[/]")

    # ONNX Runtime
    table.add_row("", "")
    r = run([venv_python(), '-c',
             'import onnxruntime as ort; print(f"{ort.__version__}|{",".join(ort.get_available_providers())}")'],
            capture=True, env=env)
    if r.returncode == 0:
        ver, provs = r.stdout.strip().split('|')
        table.add_row("ONNX Runtime", f"[green]✓[/] {ver}")
        table.add_row("  Providers", provs)
    else:
        table.add_row("ONNX Runtime", "[red]✗ Not installed[/]")

    # Detector model
    models = [('detector.mlpackage', 'CoreML'), ('detector_static.onnx', 'Static'),
              ('detector.onnx', 'Dynamic')]
    for path, name in models:
        if os.path.exists(path):
            sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(path) for f in fn) if os.path.isdir(path) else os.path.getsize(path)
            table.add_row("Detector", f"[green]✓[/] {name} ({sz/1e6:.1f}MB)")
            break
    else:
        table.add_row("Detector", "[red]✗ Not found[/]")

    # CUDA & TensorRT
    if shutil.which('nvidia-smi'):
        cuda_major, cuda_minor = get_cuda_version()
        if cuda_major:
            table.add_row("CUDA", f"[green]✓[/] {cuda_major}.{cuda_minor}")

        r = run([venv_python(), '-c', 'import tensorrt; print(tensorrt.__version__)'], capture=True, env=env)
        if r.returncode == 0:
            table.add_row("TensorRT", f"[green]✓[/] {r.stdout.strip()}")

    console.print(table)


def start_llama_server():
    """Start llama-server for VLM OCR if needed."""
    ocr_method = get_ocr_method()
    if ocr_method not in ("qwen_vlm", "lfm_vlm", "ministral_vlm_q8"):
        return True  # Not needed

    llama = find_llama()
    if not llama:
        warn("llama-server not found - skipping auto-start")
        return False

    # Check if already running
    import requests
    server_url = get_ocr_server_url()
    try:
        r = requests.get(f"{server_url}/health", timeout=2)
        if r.status_code == 200:
            ok(f"llama-server already running at {server_url}")
            return True
    except:
        pass

    console.print("\n[bold cyan]Starting llama-server[/]")

    model = get_ocr_model()
    mmproj = get_ocr_mmproj()

    import re
    port_match = re.search(r':(\d+)$', server_url)
    port = port_match.group(1) if port_match else '8080'

    info(f"Model: {model}")
    info(f"Port: {port}")

    # Download mmproj and build command using centralized functions
    mmproj_path = download_mmproj(mmproj)
    if mmproj and not mmproj_path:
        warn("Failed to download mmproj")

    log_dir = os.path.join(os.path.dirname(__file__), '.llama_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'setup_start.log')

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

    try:
        log_handle = open(log_file, 'w')
        cmd = build_llama_command(llama, model, port, mmproj_path)
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, bufsize=1, universal_newlines=True
        )
    except Exception as e:
        err(f"Failed to start: {e}")
        return False

    # Background thread to stream output
    import threading
    output_lines = []
    stop_streaming = threading.Event()

    def stream_output():
        try:
            for line in iter(proc.stdout.readline, ''):
                if stop_streaming.is_set():
                    break
                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    log_handle.write(line + '\n')
                    log_handle.flush()
                    # Show download progress and errors
                    if 'download' in line.lower() or 'error' in line.lower() or '%' in line:
                        console.print(f"  [dim]{line}[/]")
        except:
            pass

    stream_thread = threading.Thread(target=stream_output, daemon=True)
    stream_thread.start()

    # Wait for server to be ready (up to 600s for model download)
    import time
    console.print("[bold green]Waiting for server (downloading model)...[/]")
    for i in range(600):
        # Check if process died
        if proc.poll() is not None:
            stop_streaming.set()
            stream_thread.join(timeout=1)
            log_handle.close()
            err(f"Server process exited with code {proc.returncode}")
            # Show last few lines
            if output_lines:
                console.print("[dim]Last output:[/]")
                for line in output_lines[-10:]:
                    console.print(f"  [dim]{line}[/]")
            return False

        try:
            r = requests.get(f"{server_url}/health", timeout=2)
            if r.status_code == 200:
                break
        except:
            pass
        # Show progress every 30 seconds
        if i > 0 and i % 30 == 0:
            console.print(f"  [dim]Still waiting... ({i}s elapsed)[/]")
        time.sleep(1)
    else:
        stop_streaming.set()
        warn(f"Server didn't start in 600s - check {log_file}")
        return False

    ok("llama-server running")
    return True


def warmup_models():
    """Run ONNX models once to build and cache TensorRT engines."""
    if not shutil.which('nvidia-smi'):
        return True  # Skip on non-NVIDIA systems

    console.print("\n[bold cyan]Warming up models (TensorRT engine caching)[/]")
    info("First run builds optimized engines - this may take a minute...")

    env = venv_env()

    # Warmup script that imports workflow and runs detector once
    warmup_code = '''
import os
os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_ENABLE', '1')
os.environ.setdefault('ORT_TENSORRT_FP16_ENABLE', '1')

from PIL import Image

# Create demo image (640x640 manga-like)
demo = Image.new('RGB', (640, 640), (255, 255, 255))

# Import and run detector with TensorRT (static model preferred for GPU)
try:
    from workflow import detect_mode, create_session, detect_all
    mode = detect_mode()
    print(f"WARMUP:mode:{mode}")
    session, _ = create_session(mode)
    print("WARMUP:detector_loaded")

    # Run detection on demo image to build TensorRT engine
    results = detect_all([demo], session)
    print("WARMUP:detector_done")
except Exception as e:
    print(f"WARMUP:error:{e}")

# Try inpainter warmup too (optional, may use significant VRAM)
try:
    import numpy as np
    from workflow import create_inpainter
    inpainter = create_inpainter()
    print("WARMUP:inpainter_loaded")

    # Run inpaint on small demo (inpainter takes img array and bbox)
    small = np.ones((256, 256, 3), dtype=np.uint8) * 255
    bbox = [50, 50, 150, 150]  # x1, y1, x2, y2
    inpainter(small, bbox)
    print("WARMUP:inpainter_done")
except Exception as e:
    print(f"WARMUP:inpainter_skip:{e}")

# Try text segmentation warmup (TensorRT cached)
try:
    import numpy as np
    from workflow import create_text_segmenter
    text_seg = create_text_segmenter()
    print("WARMUP:textseg_loaded")

    # Run segmentation on small demo
    demo_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    text_seg(demo_img)
    print("WARMUP:textseg_done")
except Exception as e:
    print(f"WARMUP:textseg_skip:{e}")

print("WARMUP:complete")
'''

    with console.status("[bold green]Building TensorRT engines...", spinner="dots"):
        result = run([venv_python(), '-c', warmup_code], capture=True, env=env, timeout=300000)

    if result.returncode == 0:
        output = result.stdout
        # Extract and show mode
        for line in output.splitlines():
            if line.startswith('WARMUP:mode:'):
                mode = line.split(':')[-1]
                info(f"Detector mode: {mode}")
        if 'WARMUP:detector_done' in output:
            ok("Detector engine cached")
        if 'WARMUP:inpainter_done' in output:
            ok("Inpainter engine cached")
        elif 'WARMUP:inpainter_skip' in output:
            info("Inpainter will cache on first use")
        if 'WARMUP:textseg_done' in output:
            ok("Text segmentation engine cached")
        elif 'WARMUP:textseg_skip' in output:
            info("Text segmentation will cache on first use")
        if 'WARMUP:complete' in output:
            ok("TensorRT engines ready - server will start faster")
    else:
        warn("Warmup failed - engines will cache on first inference")
        if result.stderr:
            console.print(f"[dim]{result.stderr[:200]}[/]")

    return True


def show_next_steps():
    venv = get_venv_path()
    py = venv_python()
    cfg = load_config()
    port = cfg.get("server_port", 5000)

    activate = f"source {venv}/bin/activate" if PY != 'Windows' else f"{venv}\\Scripts\\activate"

    steps = f"[bold]Activate environment:[/]\n  [cyan]{activate}[/]\n\n"

    if needs_llama() and not find_llama():
        if PY == 'Darwin':
            steps += "[bold]Install llama.cpp:[/]\n  [cyan]brew install llama.cpp[/]\n  [dim](Metal GPU auto-enabled)[/]\n\n"
        elif shutil.which('nvidia-smi'):
            steps += (
                "[bold]Install llama.cpp (with CUDA):[/]\n"
                "  [cyan]sudo apt install libcurl4-openssl-dev libssl-dev ccache[/]\n"
                "  [cyan]git clone https://github.com/ggerganov/llama.cpp[/]\n"
                "  [cyan]cd llama.cpp && cmake -B build -DGGML_CUDA=ON[/]\n"
                "  [cyan]cmake --build build --config Release -j[/]\n\n"
            )
        else:
            steps += (
                "[bold]Install llama.cpp:[/]\n"
                "  [cyan]sudo apt install libcurl4-openssl-dev libssl-dev ccache[/]\n"
                "  [cyan]git clone https://github.com/ggerganov/llama.cpp[/]\n"
                "  [cyan]cd llama.cpp && cmake -B build[/]\n"
                "  [cyan]cmake --build build --config Release -j[/]\n\n"
            )

    # Check if llama-server is running
    llama_running = False
    if needs_llama():
        try:
            import requests
            server_url = cfg.get("ocr_server_url", "http://localhost:8080")
            r = requests.get(f"{server_url}/health", timeout=1)
            llama_running = r.status_code == 200
        except:
            pass

    if llama_running:
        steps += "[bold]llama-server:[/] [green]Running[/] (auto-started)\n\n"

    steps += f"[bold]Start server:[/]\n  [cyan]{py} server.py[/]  (port {port})\n\n"
    steps += f"[bold]Test:[/]\n  [cyan]{py} tests/test_client.py[/]\n\n"
    steps += f"[bold]Reconfigure:[/]\n  [cyan]{py} config.py[/]"

    console.print()
    console.print(Panel.fit(steps, title="[bold green]Setup Complete[/]", border_style="green"))

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='Setup Manga Translation Server')
    p.add_argument('--deps-only', action='store_true', help='Only install dependencies')
    p.add_argument('--models-only', action='store_true', help='Only download models')
    p.add_argument('--verify', action='store_true', help='Verify setup')
    p.add_argument('--warmup', action='store_true', help='Run model warmup (TensorRT caching)')
    p.add_argument('--start-server', action='store_true', help='Start llama-server for VLM OCR')
    p.add_argument('--no-config', action='store_true', help='Skip config wizard')
    args = p.parse_args()

    console.print(Panel.fit(
        f"[bold]Platform:[/] {PY} {platform.machine()}\n"
        f"[bold]Python:[/] {sys.version.split()[0]}",
        title="[bold blue]Manga Translation Server[/]",
        border_style="blue"
    ))

    # ── Step 0: Configuration ──
    if not config_exists() and not args.no_config:
        console.print("\n[yellow]No config.json found - starting configuration wizard...[/]")
        cfg = run_wizard()
        if cfg is None:
            err("Configuration required to continue")
            return 1
    else:
        cfg = load_config()
        if not args.verify:
            console.print(f"\n[dim]Using config from {CONFIG_FILE}[/]")
            show_summary(cfg)
            show_vram_estimate(cfg)

    # Quick actions
    if args.verify:
        verify()
        return 0

    if args.start_server:
        start_llama_server()
        return 0

    if args.warmup:
        start_llama_server()  # Ensure server is running for warmup
        warmup_models()
        return 0

    # ── Step 1: Core setup ──
    if not install_uv() or not create_venv():
        return 1

    if args.models_only:
        return 0 if download_models() else 1

    if args.deps_only:
        ok_deps = install_deps() and install_runtime()
        install_tensorrt()
        return 0 if ok_deps else 1

    # ── Step 2: Install dependencies ──
    if not install_deps() or not install_runtime():
        return 1
    install_tensorrt()

    # ── Step 3: Download detector models ──
    if not download_models():
        return 1

    # ── Step 4: Setup OCR and Translation ──
    setup_ocr()
    setup_translation()

    # ── Step 5: Start llama-server (for VLM OCR) ──
    start_llama_server()

    # ── Step 6: Warmup models (TensorRT caching) ──
    warmup_models()

    # ── Step 7: Verify and show next steps ──
    verify()
    show_next_steps()
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)
