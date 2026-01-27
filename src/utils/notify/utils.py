from __future__ import annotations

import platform
from datetime import datetime
from typing import Sequence

from ..model_manager import ModelManager


def timestamp() -> str:
    """
    Generate a timestamp string in the format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y/%m/%d-%H:%M:%S")


def osname() -> str:
    """
    Return the name of the operating system.
    """
    return platform.system()


def hostinfo() -> str:
    """
    Return a string describing the host OS and runtime information.
    """
    uname = platform.uname()
    return (
        f"Host: {uname.node}\n"
        f"OS: {uname.system} {uname.release}\n"
        f"Arch: {uname.machine}\n"
        f"Python: {platform.python_version()}"
    )


def separator() -> str:
    """
    Generate a separator line for notifications.
    """
    return "-" * 30


def epoch_message(
    model_managers: Sequence[ModelManager], epoch: int, total_epochs: int
) -> str:
    return (
        (
            f"# Training Update\n"
            f"Epoch {epoch}/{total_epochs}({epoch / total_epochs * 100:.1f}%) completed.\n"
        )
        + "\n".join(
            (
                f"## Model {i + 1}: \n"
                f"- Train Loss: {mm.get_train_data_safety().history[epoch - 1].train_loss:.4f}\n"
                f"- Val Loss: {mm.get_train_data_safety().history[epoch - 1].valid_loss:.4f}\n"
                f"- Train Acc: {mm.get_train_data_safety().history[epoch - 1].train_accuracy:.2f}%\n"
                f"- Val Acc: {mm.get_train_data_safety().history[epoch - 1].valid_accuracy:.2f}%\n"
            )
            for i, mm in enumerate(model_managers)
        )
        + (f"Timestamp: {timestamp()}")
    )
