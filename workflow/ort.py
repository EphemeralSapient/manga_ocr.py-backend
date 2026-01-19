#!/usr/bin/env python3
"""
ONNX Runtime Manager - Unified provider detection and session management

Handles:
- Auto-detection of best execution provider
- Session creation with optimal settings
- TensorRT/CUDA/CoreML/DirectML/OpenVINO/CPU support

Usage:
    from ort import create_session, get_provider_info

    # Auto-detect best provider and create session
    session = create_session("model.onnx")

    # Get provider info
    info = get_provider_info()
    print(info)  # {'provider': 'CUDAExecutionProvider', 'name': 'CUDA', 'device': 'NVIDIA RTX 4090'}
"""

import os
import site
import ctypes
import platform
from typing import Tuple, Dict, Any, Optional, List

# Preload TensorRT libraries before ORT import
def _preload_tensorrt():
    try:
        for sp in site.getsitepackages():
            trt_libs = os.path.join(sp, 'tensorrt_libs')
            if os.path.isdir(trt_libs):
                for lib in ['libnvinfer.so.10', 'libnvinfer_plugin.so.10']:
                    lib_path = os.path.join(trt_libs, lib)
                    if os.path.exists(lib_path):
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                break
    except:
        pass

_preload_tensorrt()

# Provider priority order
PROVIDER_PRIORITY = [
    ('NvTensorRTRTXExecutionProvider', 'TensorRT-RTX'),  # RTX optimized
    ('TensorrtExecutionProvider', 'TensorRT'),
    ('CUDAExecutionProvider', 'CUDA'),
    ('DmlExecutionProvider', 'DirectML'),
    ('OpenVINOExecutionProvider', 'OpenVINO'),
    ('CoreMLExecutionProvider', 'CoreML'),
    ('CPUExecutionProvider', 'CPU'),
]

# Cache directories (for pre-compiled engines)
TRT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.trt_cache')
COREML_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.coreml_cache')

# Provider-specific configurations
PROVIDER_CONFIGS = {
    'NvTensorRTRTXExecutionProvider': {
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': TRT_CACHE_DIR,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': TRT_CACHE_DIR,
        'trt_builder_optimization_level': 3,  # Max optimization
    },
    'TensorrtExecutionProvider': {
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': TRT_CACHE_DIR,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': TRT_CACHE_DIR,
        'trt_builder_optimization_level': 3,  # Max optimization
    },
    'CUDAExecutionProvider': {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    },
    'DmlExecutionProvider': {
        'device_id': 0,
    },
    'OpenVINOExecutionProvider': {
        'device_type': 'GPU_FP16',
    },
    'CoreMLExecutionProvider': {
        'ModelFormat': 'MLProgram',   # MLProgram (iOS 15+/macOS 12+) or NeuralNetwork (iOS 13+/macOS 10.15+)
        'MLComputeUnits': 'ALL',      # ALL, CPUOnly, CPUAndGPU, or CPUAndNeuralEngine
        'EnableOnSubgraphs': '1',     # Enable on subgraphs in control flow ops for better performance
        'ModelCacheDirectory': COREML_CACHE_DIR,
    },
}


def get_available_providers() -> List[str]:
    """Get list of available ONNX Runtime providers."""
    import onnxruntime as ort
    return ort.get_available_providers()


def get_best_provider() -> Tuple[str, str]:
    """
    Get the best available provider based on priority.

    Returns:
        (provider_name, friendly_name)
    """
    available = get_available_providers()

    for provider, name in PROVIDER_PRIORITY:
        if provider in available:
            return provider, name

    return 'CPUExecutionProvider', 'CPU'


def get_provider_info() -> Dict[str, Any]:
    """
    Get detailed info about the best available provider.

    Returns:
        Dict with 'provider', 'name', 'device', 'available_providers'
    """
    import onnxruntime as ort

    provider, name = get_best_provider()
    available = get_available_providers()

    info = {
        'provider': provider,
        'name': name,
        'available_providers': available,
        'onnxruntime_version': ort.__version__,
    }

    # Try to get device info
    if provider == 'CUDAExecutionProvider':
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info['device'] = result.stdout.strip().split('\n')[0]
        except:
            pass
    elif provider == 'CoreMLExecutionProvider':
        info['device'] = 'Apple Silicon'
    elif provider == 'DmlExecutionProvider':
        info['device'] = 'DirectX 12 GPU'

    return info


def create_session(
    model_path: str,
    provider: Optional[str] = None,
    **kwargs
) -> 'ort.InferenceSession':
    """
    Create an ONNX Runtime inference session with optimal settings.

    Args:
        model_path: Path to ONNX model
        provider: Force specific provider (auto-detect if None)
        **kwargs: Additional provider options

    Returns:
        ort.InferenceSession
    """
    import onnxruntime as ort

    # Session options
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Determine provider
    if provider is None:
        provider, name = get_best_provider()
    else:
        name = dict(PROVIDER_PRIORITY).get(provider, provider)

    # Build provider list with configs
    if provider in PROVIDER_CONFIGS:
        config = {**PROVIDER_CONFIGS[provider], **kwargs}
        # Ensure cache directories exist
        if 'trt_engine_cache_path' in config:
            os.makedirs(config['trt_engine_cache_path'], exist_ok=True)
        if 'ModelCacheDirectory' in config:
            os.makedirs(config['ModelCacheDirectory'], exist_ok=True)
        providers = [(provider, config)]
    else:
        providers = [provider]

    # Add CPU fallback
    if provider != 'CPUExecutionProvider':
        providers.append('CPUExecutionProvider')

    # CPU-specific optimizations
    if provider == 'CPUExecutionProvider':
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 2

    # Create session
    session = ort.InferenceSession(model_path, opts, providers=providers)

    # Verify provider was used
    active_provider = session.get_providers()[0]

    return session


def create_session_with_info(
    model_path: str,
    provider: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple['ort.InferenceSession', Dict[str, Any]]:
    """
    Create session and return info about the configuration.

    Args:
        model_path: Path to ONNX model
        provider: Force specific provider (auto-detect if None)
        verbose: Print info to console
        **kwargs: Additional provider options

    Returns:
        (session, info_dict)
    """
    session = create_session(model_path, provider, **kwargs)

    active_provider = session.get_providers()[0]
    provider_name = dict(PROVIDER_PRIORITY).get(active_provider, active_provider)

    info = {
        'model': os.path.basename(model_path),
        'provider': active_provider,
        'provider_name': provider_name,
        'all_providers': session.get_providers(),
    }

    if verbose:
        print(f"  Model: {info['model']} | Provider: {active_provider} ({provider_name})")

    return session, info


# Convenience function for quick check
def is_gpu_available() -> bool:
    """Check if any GPU provider is available."""
    provider, _ = get_best_provider()
    return provider != 'CPUExecutionProvider'


def is_cuda_available() -> bool:
    """Check if CUDA provider is available."""
    return 'CUDAExecutionProvider' in get_available_providers()


def is_tensorrt_available() -> bool:
    """Check if TensorRT provider is available."""
    return 'TensorrtExecutionProvider' in get_available_providers()


def is_coreml_available() -> bool:
    """Check if CoreML provider is available."""
    return 'CoreMLExecutionProvider' in get_available_providers()


# CLI for testing
if __name__ == '__main__':
    print("ONNX Runtime Provider Info")
    print("=" * 40)

    info = get_provider_info()
    print(f"Version: {info['onnxruntime_version']}")
    print(f"Best Provider: {info['provider']} ({info['name']})")
    if 'device' in info:
        print(f"Device: {info['device']}")
    print(f"\nAvailable Providers:")
    for p in info['available_providers']:
        marker = "→" if p == info['provider'] else " "
        print(f"  {marker} {p}")
