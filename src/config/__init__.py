"""Configuration package"""

from __future__ import annotations

from .configs import Config
from .env import EnvConfig
from .path import PathConfig

__all__ = ["EnvConfig", "PathConfig", "Config"]

# default instance
config = Config()
