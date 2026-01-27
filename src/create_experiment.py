from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from utils import initialize_experiment

arg_parser = ArgumentParser(
    description="Initialize a new experiment directory."
)
arg_parser.add_argument(
    "experiment_name", type=str, help="Name of the experiment."
)

args = arg_parser.parse_args()
experiment_name: str = args.experiment_name
base_dir = Path.cwd()

experiment_dir = initialize_experiment(base_dir, experiment_name)
print(f"Initialized new experiment at: {experiment_dir}")
