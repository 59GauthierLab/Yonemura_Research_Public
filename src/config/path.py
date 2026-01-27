from __future__ import annotations

from pathlib import Path


class PathConfig:
    """Configuration for the pathfinding module."""

    def __init__(self) -> None:
        # root directory of the project
        self.root: Path = Path(__file__).parent.parent.parent
        # sub directories
        self.data: Path = self.root / "data"
        self.src: Path = self.root / "src"

    def mkdir(self, exist_ok: bool = False) -> None:
        self.data.mkdir(parents=True, exist_ok=exist_ok)
        self.src.mkdir(parents=True, exist_ok=exist_ok)

    def __repr__(self) -> str:
        return f"Path(root={self.root}, data={self.data}, src={self.src})"
