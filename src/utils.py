"""
Shared utilities: seed setting, logging, I/O helpers.
"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_step(step_name: str, detail: str = "") -> None:
    """Simple console logger for pipeline steps."""
    msg = f"[STEP] {step_name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
