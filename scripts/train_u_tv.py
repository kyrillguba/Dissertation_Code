"""Train U-TV with the shared denoising experiment pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.u_tv import UTVModel
from src.training.denoising import (
    add_common_training_arguments,
    apply_common_defaults,
    train_denoising_model,
)


def train_u_tv(args: argparse.Namespace, train_loader=None, val_loader=None):
    """Train U-TV and return ``(model, history_dataframe)``."""
    apply_common_defaults(args, checkpoint_dir="checkpoints/u_tv")
    for name, value in {
        "num_pdhg_iters": 64,
        "map_scale": 0.1,
        "pdhg_tau": None,
        "pdhg_sigma": None,
        "pdhg_theta": None,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, value)

    # Delegate all data, optimization, validation, and checkpoint handling to the shared pipeline.
    return train_denoising_model(
        args,
        model_name="u_tv",
        train_loader=train_loader,
        val_loader=val_loader,
        model_factory=lambda: UTVModel(
            base_channels=args.base_channels,
            depth=args.depth,
            num_pdhg_iters=args.num_pdhg_iters,
            map_scale=args.map_scale,
            pdhg_tau=args.pdhg_tau,
            pdhg_sigma=args.pdhg_sigma,
            pdhg_theta=args.pdhg_theta,
        ),
    )


# Compatibility with the original notebook import: ``from ... import train``.
train = train_u_tv


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a U-TV training run."""
    parser = argparse.ArgumentParser(description="Train U-TV on SeaTurtleID2022.")
    add_common_training_arguments(parser, checkpoint_dir="checkpoints/u_tv")
    parser.add_argument("--num-pdhg-iters", type=int, default=64)
    parser.add_argument("--map-scale", type=float, default=0.1)
    parser.add_argument("--pdhg-tau", type=float, default=None)
    parser.add_argument("--pdhg-sigma", type=float, default=None)
    parser.add_argument("--pdhg-theta", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train_u_tv(parse_args())

