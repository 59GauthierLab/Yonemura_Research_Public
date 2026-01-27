from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

script_content = """\
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, cast

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from utils import ModelManager
from utils.metrics import MultilabelBinaryAccuracy
from utils.random import fix_seed, gen_worker_init_fn

# -------------------------------
# 乱数設定
# -------------------------------
seed = fix_seed(seed=None, strict=False)

g = torch.Generator()
g.manual_seed(seed)

"""


def initialize_experiment(base_dir: Path, experiment_name: str) -> Path:
    experiment_dir = base_dir / experiment_name
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory '{experiment_dir}' already exists."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create python script
    (experiment_dir / "main.py").write_text(script_content, encoding="utf-8")

    # Create project config
    (experiment_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "description": "",
                "created_at": datetime.now().isoformat(),
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    return experiment_dir
