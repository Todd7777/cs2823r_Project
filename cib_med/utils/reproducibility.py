"""Reproducibility utilities for CIB-Med-1."""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available compute device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device for computation
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_system_info() -> dict:
    """Get system information for reproducibility logging."""
    info = {
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    return info
