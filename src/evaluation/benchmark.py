"""Shared loading, timing, and paired statistics for the denoising benchmark."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.denoising_eval import (
    call_denoising_model,
    unpack_model_output,
)
from src.evaluation.tikhonov_eval import scalar_tikhonov_denoise
from src.models.cg_tikhonov import UTikhonovModel
from src.models.u_tgv import UTGVModel
from src.models.u_tv import UTVModel
from src.training.denoising import describe_device, synchronize_device
from src.utils.metrics import batch_mse, batch_psnr, batch_ssim


def load_checkpoint(path: str | Path, device: str | torch.device) -> dict:
    """Load a project checkpoint across supported PyTorch versions."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    try:
        return torch.load(
            resolved,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(resolved, map_location=device)


def _optional_float(value: Any) -> float | None:
    """Convert an optional checkpoint value to float while preserving None."""
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def load_project_model(
    path: str | Path,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Reconstruct U-TV, U-TGV, or U-Tikhonov from its checkpoint arguments."""
    checkpoint = load_checkpoint(path, device)
    args = checkpoint["args"]
    model_name = str(checkpoint.get("model_name", "")).lower()

    if model_name == "u_tv":
        model = UTVModel(
            base_channels=int(args["base_channels"]),
            depth=int(args["depth"]),
            num_pdhg_iters=int(args["num_pdhg_iters"]),
            map_scale=float(args["map_scale"]),
            pdhg_tau=_optional_float(args.get("pdhg_tau")),
            pdhg_sigma=_optional_float(args.get("pdhg_sigma")),
            pdhg_theta=_optional_float(args.get("pdhg_theta")),
        )
    elif model_name == "u_tgv":
        model = UTGVModel(
            base_channels=int(args["base_channels"]),
            depth=int(args["depth"]),
            num_pdhg_iters=int(args["num_pdhg_iters"]),
            map_scale=float(args["map_scale"]),
            pdhg_tau=_optional_float(args.get("pdhg_tau")),
            pdhg_sigma=_optional_float(args.get("pdhg_sigma")),
            pdhg_theta=_optional_float(args.get("pdhg_theta")),
        )
    elif model_name in {"u_tikhonov", "utikhonov"} or "cg_iters" in args:
        map_scale = args.get("map_scale", 0.1)
        if map_scale is None:
            map_scale = 0.1
        model = UTikhonovModel(
            base_channels=int(args["base_channels"]),
            depth=int(args["depth"]),
            cg_iters=int(args["cg_iters"]),
            map_scale=float(map_scale),
            cg_relative_tol=float(args.get("cg_relative_tol", 1e-7)),
            parameterization=str(
                args.get("lambda_parameterization", "scaled_softplus")
            ),
            lambda_min=float(args.get("lambda_min", 1e-4)),
            lambda_max=_optional_float(args.get("lambda_max")),
            check_spd=True,
        )
    else:
        raise ValueError(
            f"Unsupported checkpoint model_name={model_name!r} at {path}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def validate_paired_results(
    method_results: Mapping[str, pd.DataFrame],
) -> None:
    """Require identical sample IDs, targets, noise, and ordering for all methods."""
    if not method_results:
        raise ValueError("No method results were supplied.")
    reference_name, reference = next(iter(method_results.items()))
    identity_columns = (
        "sample_id",
        "dataset",
        "image_name",
        "sigma",
        "clean_sha256",
        "noisy_sha256",
    )
    missing = [column for column in identity_columns if column not in reference]
    if missing:
        raise ValueError(f"{reference_name} is missing identity columns: {missing}")

    # Treat the first method's ordered sample identity table as the pairing contract.
    expected = reference.loc[:, identity_columns].reset_index(drop=True)
    for method, frame in method_results.items():
        actual = frame.loc[:, identity_columns].reset_index(drop=True)
        if not actual.equals(expected):
            raise RuntimeError(
                f"{method} did not receive the exact samples used by "
                f"{reference_name}."
            )


def method_summary(
    long_results: pd.DataFrame,
    group_columns: Sequence[str] = ("dataset", "sigma", "method"),
) -> pd.DataFrame:
    """Mean, standard deviation, median, and sample count for each method."""
    required = {"recon_mse", "recon_psnr", "recon_ssim", *group_columns}
    missing = required - set(long_results)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    return (
        long_results.groupby(list(group_columns), as_index=False)
        .agg(
            images=("sample_id", "nunique"),
            mse_mean=("recon_mse", "mean"),
            mse_std=("recon_mse", "std"),
            mse_median=("recon_mse", "median"),
            psnr_mean=("recon_psnr", "mean"),
            psnr_std=("recon_psnr", "std"),
            psnr_median=("recon_psnr", "median"),
            ssim_mean=("recon_ssim", "mean"),
            ssim_std=("recon_ssim", "std"),
            ssim_median=("recon_ssim", "median"),
        )
        .sort_values(list(group_columns))
        .reset_index(drop=True)
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap interval for the sample mean."""
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    # Resample image-level differences with replacement for the percentile interval.
    indices = generator.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    bootstrap_means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return float(lower), float(upper)


def paired_difference_summary(
    long_results: pd.DataFrame,
    method_pairs: Sequence[tuple[str, str]],
    metrics: Sequence[str] = ("recon_psnr", "recon_ssim"),
    *,
    tie_tolerance: float = 1e-12,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-image differences and paired summaries.

    Differences are always ``first method - second method``. Positive PSNR or
    SSIM values therefore favour the first method.
    """
    per_image_rows = []
    summary_rows = []
    for first, second in method_pairs:
        first_frame = long_results.loc[
            long_results["method"] == first,
            ["sample_id", "dataset", "image_name", "sigma", *metrics],
        ]
        second_frame = long_results.loc[
            long_results["method"] == second,
            ["sample_id", *metrics],
        ]
        paired = first_frame.merge(
            second_frame,
            on="sample_id",
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )
        if len(paired) != len(first_frame) or len(paired) != len(second_frame):
            raise RuntimeError(f"Unpaired samples for {first} versus {second}.")

        for metric in metrics:
            values = (
                paired[f"{metric}_first"] - paired[f"{metric}_second"]
            ).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            wins = int(np.sum(values > tie_tolerance))
            ties = int(np.sum(np.abs(values) <= tie_tolerance))
            losses = int(np.sum(values < -tie_tolerance))
            summary_rows.append(
                {
                    "first_method": first,
                    "second_method": second,
                    "difference": f"{first} - {second}",
                    "metric": metric.removeprefix("recon_").upper(),
                    "images": values.size,
                    "mean_difference": float(values.mean()),
                    "std_difference": float(values.std(ddof=1)),
                    "median_difference": float(np.median(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_percentage": 100.0 * wins / values.size,
                }
            )
            for row, difference in zip(paired.to_dict("records"), values):
                per_image_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "dataset": row["dataset"],
                        "image_name": row["image_name"],
                        "sigma": row["sigma"],
                        "first_method": first,
                        "second_method": second,
                        "metric": metric.removeprefix("recon_").upper(),
                        "difference": float(difference),
                    }
                )
    return pd.DataFrame(per_image_rows), pd.DataFrame(summary_rows)


def best_method_counts(
    long_results: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    *,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Count exclusive and tied per-image best results."""
    subset = long_results.loc[
        long_results["method"].isin(methods),
        ["sample_id", "method", metric],
    ]
    table = subset.pivot(index="sample_id", columns="method", values=metric)
    table = table.loc[:, list(methods)]
    if table.isna().any().any():
        raise RuntimeError("Every method must have one result for every sample.")

    higher_is_better = metric not in {"recon_mse", "mse"}
    best_values = table.max(axis=1) if higher_is_better else table.min(axis=1)
    distance = table.sub(best_values, axis=0).abs()
    tied_for_best = distance <= tie_tolerance
    number_best = tied_for_best.sum(axis=1)

    rows = []
    for method in methods:
        best_mask = tied_for_best[method]
        exclusive = best_mask & (number_best == 1)
        tied = best_mask & (number_best > 1)
        rows.append(
            {
                "metric": metric.removeprefix("recon_").upper(),
                "method": method,
                "images": len(table),
                "exclusive_best_count": int(exclusive.sum()),
                "tied_best_count": int(tied.sum()),
                "best_or_tied_count": int(best_mask.sum()),
                "exclusive_best_percentage": 100.0 * float(exclusive.mean()),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def benchmark_inference_runtime(
    model: torch.nn.Module,
    loader: Iterable,
    device: str | torch.device,
    *,
    max_batches: int = 10,
    warmup_runs: int = 3,
    repeats: int = 5,
) -> pd.DataFrame:
    """Time fixed batch-size inference after accelerator synchronization."""
    batches = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batches.append(
            (
                batch["noisy"].to(device),
                batch.get("sigma"),
            )
        )
    if not batches:
        raise ValueError("Timing loader is empty.")

    model.eval()
    warmup_noisy, warmup_sigma = batches[0]
    for _ in range(warmup_runs):
        unpack_model_output(
            call_denoising_model(model, warmup_noisy, warmup_sigma)
        )
    synchronize_device(device)

    rows = []
    for repeat in range(repeats):
        for batch_index, (noisy, sigma) in enumerate(batches):
            synchronize_device(device)
            start = time.perf_counter()
            unpack_model_output(call_denoising_model(model, noisy, sigma))
            synchronize_device(device)
            elapsed = time.perf_counter() - start
            row = {
                "repeat": repeat,
                "batch_index": batch_index,
                "batch_size": int(noisy.shape[0]),
                "runtime_sec": elapsed,
                "runtime_ms_per_image": 1000.0 * elapsed / noisy.shape[0],
            }
            for attribute, column in (
                ("last_outer_iterations", "outer_iterations"),
                ("last_inner_iterations", "last_inner_iterations"),
                ("last_total_inner_iterations", "total_inner_iterations"),
            ):
                if hasattr(model, attribute):
                    row[column] = int(getattr(model, attribute))
            rows.append(row)
    return pd.DataFrame(rows)


def runtime_summary(
    timing_rows: pd.DataFrame,
    *,
    method: str,
    model: torch.nn.Module,
    device: str | torch.device,
) -> dict[str, Any]:
    """Aggregate synchronized timing rows and solver metadata for one method."""
    values = timing_rows["runtime_ms_per_image"]
    summary = {
        "method": method,
        "device": str(torch.device(device)),
        "device_description": describe_device(device),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "timed_measurements": len(timing_rows),
        "timed_images_per_repeat": int(timing_rows["batch_index"].nunique()),
        "mean_runtime_ms_per_image": float(values.mean()),
        "std_runtime_ms_per_image": float(values.std()),
        "median_runtime_ms_per_image": float(values.median()),
        "images_per_second": float(1000.0 / values.mean()),
        "solver_iterations": getattr(
            model,
            "cg_iters",
            getattr(model, "num_pdhg_iters", None),
        ),
    }
    if hasattr(model, "protocol"):
        summary.update(
            {
                "solver_protocol": getattr(model, "protocol"),
                "outer_iteration_cap": getattr(
                    model, "max_outer_iterations", None
                ),
                "inner_iteration_cap": getattr(
                    model, "max_inner_iterations", None
                ),
                "inner_tolerance": getattr(model, "inner_tolerance", None),
                "outer_tolerance": getattr(model, "outer_tolerance", None),
            }
        )
    for column in (
        "outer_iterations",
        "last_inner_iterations",
        "total_inner_iterations",
    ):
        if column in timing_rows:
            summary[f"mean_{column}"] = float(timing_rows[column].mean())
    return summary


@torch.no_grad()
def evaluate_scalar_tikhonov_per_image(
    loader: Iterable,
    lam: float,
    device: str | torch.device,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
    clamp_for_metrics: bool = True,
) -> pd.DataFrame:
    """Evaluate the scalar reference without losing paired image identities."""
    rows = []
    for batch in loader:
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        reconstruction, info = scalar_tikhonov_denoise(
            noisy,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
        )
        scored = (
            reconstruction.clamp(0.0, 1.0)
            if clamp_for_metrics
            else reconstruction
        )
        mse_values = batch_mse(scored, clean)
        psnr_values = batch_psnr(scored, clean)
        ssim_values = batch_ssim(scored, clean)
        for index in range(clean.shape[0]):
            rows.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "dataset": batch["dataset"][index],
                    "image_name": batch["image_name"][index],
                    "path": batch["path"][index],
                    "base_idx": int(batch["base_idx"][index]),
                    "sigma": float(batch["sigma"][index].reshape(-1)[0]),
                    "noise_seed": int(batch["noise_seed"][index]),
                    "clean_sha256": batch["clean_sha256"][index],
                    "noisy_sha256": batch["noisy_sha256"][index],
                    "recon_mse": float(mse_values[index]),
                    "recon_psnr": float(psnr_values[index]),
                    "recon_ssim": float(ssim_values[index]),
                    "cg_iterations": int(info["per_sample_iterations"][index]),
                    "cg_final_relative_residual": float(
                        info["per_sample_final_rel_residual"][index]
                    ),
                }
            )
    return pd.DataFrame(rows)
