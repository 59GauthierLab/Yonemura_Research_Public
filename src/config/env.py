from __future__ import annotations

import torch


class EnvConfig:
    """Configuration for the environment."""

    def __init__(self) -> None:
        # computation device: 'auto', 'cpu', 'cuda'
        self._device: str = "auto"
        self.seed: int = 42

    @property
    def cuda_available(self) -> bool:
        return torch.cuda.is_available()

    @property
    def device(self) -> torch.device:
        if self._device == "auto":
            return torch.device("cuda" if self.cuda_available else "cpu")
        else:
            return torch.device(self._device)
