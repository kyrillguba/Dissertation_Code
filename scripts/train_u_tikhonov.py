"""Train U-Tikhonov with the shared denoising experiment pipeline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cg_tikhonov import UTikhonovModel
from src.training.denoising import (
    add_common_training_arguments,
    apply_common_defaults,
    train_denoising_model,
)


def load_reference_lambda(summary_path: str | Path) -> float:
    """Read ``best_scalar_lambda`` from a scalar-baseline summary CSV."""
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Scalar Tikhonov summary not found: {path}")
    summary = pd.read_csv(path)
    if summary.empty or "best_scalar_lambda" not in summary:
        raise ValueError(
            f"{path} must contain a non-empty 'best_scalar_lambda' column."
        )
    value = float(summary.iloc[0]["best_scalar_lambda"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid reference lambda: {value}")
    return value


def calibrated_map_scale(reference_lambda: float) -> float:
    """Choose scale ``s`` so ``s * softplus(0) = reference_lambda``."""
    if reference_lambda <= 0:
        raise ValueError("reference_lambda must be strictly positive.")
    return float(reference_lambda) / math.log(2.0)


def initialize_lambda_output(
    model: UTikhonovModel,
    reference_lambda: float,
) -> None:
    """Initialize the predicted map to the scalar-reference solution.

    Zeroing the final convolutional weights makes the initial map exactly
    constant instead of merely constant in expectation.
    """
    if model.parameterization == "bounded_sigmoid":
        initial_fraction = (
            (reference_lambda - model.lambda_min)
            / (model.lambda_max - model.lambda_min)
        )
        if not 0.0 < initial_fraction < 1.0:
            raise ValueError(
                "reference_lambda must lie strictly between lambda_min "
                "and lambda_max."
            )
        initial_bias = math.log(initial_fraction / (1.0 - initial_fraction))
    else:
        initial_bias = math.log(
            math.expm1(reference_lambda / model.map_scale)
        )

    with torch.no_grad():
        model.net.final.weight.zero_()
        model.net.final.bias.fill_(initial_bias)


def train_u_tikhonov(args: argparse.Namespace, train_loader=None, val_loader=None):
    """Train U-Tikhonov and return ``(model, history_dataframe)``."""
    apply_common_defaults(args, checkpoint_dir="checkpoints/u_tikhonov")
    for name, value in {
        "cg_iters": 32,
        "cg_relative_tol": 1e-7,
        "reference_lambda": 2.49805,
        "scalar_baseline_summary": None,
        "map_scale": None,
        "lambda_parameterization": "scaled_softplus",
        "lambda_min": 1e-4,
        "lambda_max": None,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, value)

    # Prefer the calibrated CSV value when supplied; otherwise retain the configured reference.
    reference_lambda = (
        load_reference_lambda(args.scalar_baseline_summary)
        if args.scalar_baseline_summary
        else float(args.reference_lambda)
    )
    map_scale = (
        calibrated_map_scale(reference_lambda)
        if args.map_scale is None
        else float(args.map_scale)
    )
    args.reference_lambda = reference_lambda
    args.map_scale = map_scale

    # Create a new identically initialised model for the shared training pipeline.
    def model_factory() -> UTikhonovModel:
        model = UTikhonovModel(
            base_channels=args.base_channels,
            depth=args.depth,
            cg_iters=args.cg_iters,
            map_scale=args.map_scale,
            cg_relative_tol=args.cg_relative_tol,
            parameterization=args.lambda_parameterization,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            check_spd=True,
        )
        initialize_lambda_output(model, reference_lambda)
        return model

    return train_denoising_model(
        args,
        model_name="u_tikhonov",
        train_loader=train_loader,
        val_loader=val_loader,
        model_factory=model_factory,
    )


train = train_u_tikhonov


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a U-Tikhonov training run."""
    parser = argparse.ArgumentParser(
        description="Train learned spatially weighted Tikhonov denoising."
    )
    add_common_training_arguments(parser, checkpoint_dir="checkpoints/u_tikhonov")
    parser.add_argument("--cg-iters", type=int, default=32)
    parser.add_argument("--cg-relative-tol", type=float, default=1e-7)
    parser.add_argument("--reference-lambda", type=float, default=2.49805)
    parser.add_argument("--map-scale", type=float, default=None)
    parser.add_argument("--scalar-baseline-summary", type=str, default=None)
    parser.add_argument(
        "--lambda-parameterization",
        choices=("scaled_softplus", "bounded_sigmoid"),
        default="scaled_softplus",
    )
    parser.add_argument("--lambda-min", type=float, default=1e-4)
    parser.add_argument("--lambda-max", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train_u_tikhonov(parse_args())
