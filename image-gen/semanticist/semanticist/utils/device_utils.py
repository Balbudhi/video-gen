# semanticist/utils/device_utils.py

import torch

def configure_compute_backend():
    """
    Backend settings safe across CUDA/MPS/CPU.
    This function isolates hardware-specific optimizations to prevent
    crashes on non-NVIDIA hardware.
    """
    try:
        # Optimizes matrix multiplication precision for potential speedups.
        # On CUDA, this controls TensorFloat32.
        # On MPS/CPU, this sets the preferred kernel precision where applicable.
        # We wrap this in a try-except block because some PyTorch versions
        # on specific OS/hardware combos may not expose this API.
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.is_available():
        # Retain original CUDA optimizations for backward compatibility
        # if the user migrates this code back to a cluster.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_device():
    """
    Robust device selector implementing the priority queue:
    CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU (Fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for Metal Performance Shaders (MPS) availability on macOS.
    # We check both the module existence and the is_available() flag
    # to support a wide range of PyTorch versions (1.12+).
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

