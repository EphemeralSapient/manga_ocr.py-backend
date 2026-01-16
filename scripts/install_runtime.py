#!/usr/bin/env python3
"""
Auto-detect hardware and install optimal ONNX Runtime variant.

Usage: python install_runtime.py [--force VARIANT]

Variants:
  gpu      - NVIDIA CUDA 12.x (PyPI stable)
  gpu-cu13 - NVIDIA CUDA 13.x (nightly builds)
  directml - DirectML (Windows)
  openvino - Intel OpenVINO
  cpu      - CPU only (fallback)
  auto     - Auto-detect (default)

CUDA Notes:
  - onnxruntime-gpu on PyPI only supports CUDA 12.x
  - CUDA 13.x requires nightly builds from Azure DevOps feed
  - Auto-detect will choose the right variant based on nvidia-smi
"""

import subprocess
import platform
import sys
import shutil

VARIANTS = {
    'gpu': 'onnxruntime-gpu',          # CUDA 12.x (PyPI default)
    'gpu-cu13': 'onnxruntime-gpu',     # CUDA 13.x (nightly index)
    'directml': 'onnxruntime-directml',
    'openvino': 'onnxruntime-openvino',
    'cpu': 'onnxruntime',
}

# CUDA 13 requires nightly builds from special index
CUDA13_INDEX = 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/'


def get_cuda_version():
    """Get CUDA major version from nvidia-smi."""
    if not shutil.which('nvidia-smi'):
        return None
    try:
        import re
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None


def check_nvidia_cuda():
    """Check for NVIDIA GPU with CUDA. Returns (available, cuda_major)."""
    cuda_ver = get_cuda_version()
    if cuda_ver:
        return True, cuda_ver[0]

    # Fallback: check nvidia-smi exists
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return True, 12  # Assume CUDA 12 if can't detect
        except:
            pass

    # Check for CUDA via torch
    try:
        import torch
        if torch.cuda.is_available():
            return True, 12  # Assume CUDA 12
    except:
        pass

    return False, None


def check_directml():
    """Check for DirectML support (Windows with DirectX 12 GPU)."""
    if platform.system() != 'Windows':
        return False

    # Check for DirectX 12 capable GPU via dxdiag or WMI
    try:
        import subprocess
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Has a GPU - DirectML should work
            return True
    except:
        pass

    return False


def check_openvino():
    """Check for Intel hardware suitable for OpenVINO."""
    try:
        # Check CPU vendor
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'GenuineIntel' in cpuinfo:
                    return True
        elif platform.system() == 'Windows':
            import subprocess
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'manufacturer'],
                capture_output=True, text=True, timeout=5
            )
            if 'Intel' in result.stdout:
                return True
    except:
        pass

    return False


def detect_best_variant():
    """Detect the best ONNX Runtime variant for this system.

    Returns: (variant, cuda_major) - cuda_major is None for non-CUDA variants
    """
    system = platform.system()
    machine = platform.machine().lower()

    print(f"System: {system} {machine}")

    # macOS - use standard onnxruntime (includes CoreML EP)
    if system == 'Darwin':
        print("  -> macOS detected, using onnxruntime (CoreML EP included)")
        return 'cpu', None

    # Check NVIDIA CUDA first (best performance for NVIDIA GPUs)
    has_cuda, cuda_major = check_nvidia_cuda()
    if has_cuda:
        cuda_ver = get_cuda_version()
        ver_str = f"{cuda_ver[0]}.{cuda_ver[1]}" if cuda_ver else str(cuda_major)
        print(f"  -> NVIDIA CUDA {ver_str} detected")

        if cuda_major and cuda_major >= 13:
            print("     Note: CUDA 13 uses nightly onnxruntime-gpu builds")
            return 'gpu-cu13', cuda_major
        return 'gpu', cuda_major

    # Windows - check DirectML
    if system == 'Windows' and check_directml():
        print("  -> DirectML supported (Windows GPU)")
        return 'directml', None

    # Intel - check OpenVINO
    if check_openvino():
        print("  -> Intel CPU detected, OpenVINO available")
        # OpenVINO is optional - only suggest if user wants it
        print("     (using CPU by default, use --force openvino for OpenVINO)")
        return 'cpu', None

    print("  -> Using CPU runtime")
    return 'cpu', None


def uninstall_existing():
    """Uninstall any existing onnxruntime variants."""
    variants = ['onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-openvino']

    # Try uv first, fall back to pip
    use_uv = shutil.which('uv') is not None

    for variant in variants:
        if use_uv:
            subprocess.run(['uv', 'pip', 'uninstall', variant], capture_output=True)
        else:
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', variant], capture_output=True)


def install_variant(variant, break_system=False):
    """Install the specified ONNX Runtime variant."""
    package = VARIANTS.get(variant, 'onnxruntime')
    use_nightly = variant == 'gpu-cu13'

    if use_nightly:
        print(f"\nInstalling {package} (CUDA 13 nightly)...")
        print(f"  Index: {CUDA13_INDEX}")
    else:
        print(f"\nInstalling {package}...")

    # Try uv first, fall back to pip
    if shutil.which('uv'):
        if use_nightly:
            # uv with custom index
            cmd = ['uv', 'pip', 'install', '--index-url', CUDA13_INDEX, '--pre', package]
        else:
            cmd = ['uv', 'pip', 'install', package]
    else:
        if use_nightly:
            # pip with custom index and --pre for nightly
            cmd = [sys.executable, '-m', 'pip', 'install', '--pre',
                   '--index-url', CUDA13_INDEX, package]
        else:
            cmd = [sys.executable, '-m', 'pip', 'install', package]
        if break_system:
            cmd.append('--break-system-packages')

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n✓ Successfully installed {package}" + (" (nightly)" if use_nightly else ""))
        return True
    else:
        # Fallback for CUDA 13: try stable CUDA 12 build
        if use_nightly:
            print(f"\n⚠ CUDA 13 nightly failed, trying CUDA 12 stable build...")
            return install_variant('gpu', break_system)
        print(f"\n✗ Failed to install {package}")
        return False


def verify_installation():
    """Verify ONNX Runtime installation and show available providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"\nONNX Runtime {ort.__version__}")
        print(f"Available providers: {', '.join(providers)}")
        return True
    except ImportError:
        print("\n✗ ONNX Runtime not found")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Install optimal ONNX Runtime')
    parser.add_argument('--force', choices=['gpu', 'gpu-cu13', 'directml', 'openvino', 'cpu'],
                        help='Force specific variant (gpu=CUDA12, gpu-cu13=CUDA13 nightly)')
    parser.add_argument('--no-uninstall', action='store_true',
                        help='Skip uninstalling existing variants')
    parser.add_argument('--break-system-packages', action='store_true',
                        help='Allow install on externally-managed Python (Homebrew)')
    parser.add_argument('--detect-only', action='store_true',
                        help='Only detect hardware, do not install')
    args = parser.parse_args()

    print("ONNX Runtime Installer")
    print("=" * 40)

    # Detect or use forced variant
    if args.force:
        variant = args.force
        print(f"Forcing variant: {variant}")
    else:
        variant, cuda_major = detect_best_variant()

    if args.detect_only:
        pkg = VARIANTS.get(variant, 'onnxruntime')
        extra = " (nightly)" if variant == 'gpu-cu13' else ""
        print(f"\nRecommended: {pkg}{extra}")
        return 0

    # Uninstall existing
    if not args.no_uninstall:
        print("\nRemoving existing ONNX Runtime installations...")
        uninstall_existing()

    # Install
    success = install_variant(variant, break_system=args.break_system_packages)

    # Verify
    if success:
        verify_installation()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
