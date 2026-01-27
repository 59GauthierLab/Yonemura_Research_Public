from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


def gen_worker_init_fn(seed: int) -> Callable[[int], None]:
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn
