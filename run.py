#!/usr/bin/env python3
"""
Run the Manga Translation Server

Automatically sets up the environment if needed and starts the server.

Usage:
  python run.py              # Start server (runs setup if needed)
  python run.py --setup      # Force re-run setup
  python run.py --port 8080  # Custom port
"""

import subprocess
import sys
import os
import platform

VENV_DIR = ".venv"


def get_venv_python():
    """Get path to Python in venv."""
    if platform.system() == 'Windows':
        return os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(VENV_DIR, 'bin', 'python')


def get_venv_env():
    """Get environment variables for venv."""
    env = os.environ.copy()

    if platform.system() == 'Windows':
        bin_dir = os.path.join(os.getcwd(), VENV_DIR, 'Scripts')
    else:
        bin_dir = os.path.join(os.getcwd(), VENV_DIR, 'bin')

    env['PATH'] = bin_dir + os.pathsep + env.get('PATH', '')
    env['VIRTUAL_ENV'] = os.path.join(os.getcwd(), VENV_DIR)

    # For CUDA/TensorRT
    venv_python = get_venv_python()
    if os.path.exists(venv_python):
        result = subprocess.run(
            [venv_python, '-c', 'import site; print(site.getsitepackages()[0])'],
            capture_output=True, text=True
        )
        site_packages = result.stdout.strip()
        if site_packages:
            tensorrt_libs = os.path.join(site_packages, 'tensorrt_libs')
            if platform.system() == 'Linux':
                ld_path = env.get('LD_LIBRARY_PATH', '')
                env['LD_LIBRARY_PATH'] = tensorrt_libs + ':' + ld_path if ld_path else tensorrt_libs

    # Performance tuning environment variables
    env['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
    env['ORT_TENSORRT_FP16_ENABLE'] = '1'
    env['CUDA_MODULE_LOADING'] = 'LAZY'  # Faster CUDA startup
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging noise

    return env


def run_setup():
    """Run setup.py to create venv and install dependencies."""
    print("Setting up environment...")
    result = subprocess.run([sys.executable, 'setup.py'])
    return result.returncode == 0


def check_setup():
    """Check if setup is complete."""
    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        return False

    # Check if onnxruntime is installed
    env = get_venv_env()
    result = subprocess.run(
        [venv_python, '-c', 'import onnxruntime'],
        capture_output=True, env=env
    )
    if result.returncode != 0:
        return False

    # Check for at least one detector model
    models = ['detector.mlpackage', 'detector_static.onnx', 'detector.onnx']
    if not any(os.path.exists(m) for m in models):
        return False

    return True


def run_server(port=None):
    """Run the server."""
    venv_python = get_venv_python()
    env = get_venv_env()

    cmd = [venv_python, 'server.py']

    print(f"\nStarting server...")
    print(f"Python: {venv_python}")
    if port:
        print(f"Port: {port}")
        # Modify server.py port via environment variable
        env['PORT'] = str(port)

    print("-" * 40)

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Manga Translation Server')
    parser.add_argument('--setup', action='store_true',
                        help='Force re-run setup')
    parser.add_argument('--port', type=int,
                        help='Server port (default: 1389)')
    args = parser.parse_args()

    # Check if setup needed
    if args.setup or not check_setup():
        if not run_setup():
            print("\n✗ Setup failed")
            return 1
        print()

    # Run server
    run_server(port=args.port)
    return 0


if __name__ == '__main__':
    sys.exit(main())
