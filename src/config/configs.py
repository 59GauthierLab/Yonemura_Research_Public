"""Main configuration class that combines all configuration modules."""

from __future__ import annotations

from .env import EnvConfig
from .path import PathConfig


class Config:
    """Main configuration class."""

    def __init__(self) -> None:
        self.env = EnvConfig()
        self.path = PathConfig()

    def __repr__(self) -> str:
        return f"Config(env={self.env}, path={self.path})"
