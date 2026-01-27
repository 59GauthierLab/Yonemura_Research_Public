from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

# force PYTHONHASHSEED to a fixed value
# if not set or set to a different value, raise a RuntimeError
PYTHONHASHSEED = 42


def fix_seed(seed: Optional[int] = None, strict: bool = False) -> int:
    """
    Fix random seeds for reproducibility across Python, NumPy, and PyTorch.

    CRITICAL WARNING:
    This function verifies that required environment variables (PYTHONHASHSEED, and CUBLAS_WORKSPACE_CONFIG for strict mode)
    are already set before process startup.
    Before running your script, ensure that the environment variables are set correctly to guarantee reproducibility.

    Args:
        seed (Optional[int]): The seed value. If None, a seed is generated
            using the operating system's random number generator (os.urandom).
        strict (bool): If True, enables deterministic algorithms in PyTorch.
            This ensures bit-level reproducibility at the cost of performance
            (approx. 10-30% slowdown) and potentially higher memory usage.

    Returns:
        int: The seed value used for initialization.
    """

    # pythonのハッシュシードの設定
    if os.environ.get("PYTHONHASHSEED") != str(PYTHONHASHSEED):
        raise RuntimeError(
            f"PYTHONHASHSEED is not set to '{PYTHONHASHSEED}'. "
            f"Please set it as follows:\n"
            f"Linux/Mac: export PYTHONHASHSEED={PYTHONHASHSEED}\n"
            f"Windows: set PYTHONHASHSEED={PYTHONHASHSEED}"
        )

    # cuBLASの決定論的挙動を制御する設定（特定の演算で必要）
    # strictモードの場合に強制
    if strict and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG is not set to ':4096:8'. "
            "Please set it as follows before running your script:\n"
            "Linux/Mac: export CUBLAS_WORKSPACE_CONFIG=':4096:8'\n"
            "Windows: set CUBLAS_WORKSPACE_CONFIG=':4096:8'"
        )

    # seed値がNoneの場合: デバイスの乱数生成器を使用してシードを生成
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")  # 32bit int

    # seed値設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 厳密な再現性設定(パフォーマンス低下の可能性あり)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 一部の演算で非決定的なアルゴリズムを禁止
        # 非決定的な演算を使用した場合は例外
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    return seed


def get_fixed_pythonhashseed() -> int:
    """
    Get a fixed seed value for reproducibility.

    Returns:
        int: The fixed seed value.
    """
    return PYTHONHASHSEED
