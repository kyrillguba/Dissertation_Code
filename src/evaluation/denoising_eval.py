"""Common per-image evaluation for U-TV, U-TGV and U-Tikhonov."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pandas as pd
import torch

from src.utils.metrics import batch_mse, batch_psnr, batch_ssim

Tensor = torch.Tensor


def call_denoising_model(
    model: torch.nn.Module,
    noisy: Tensor,
    sigma: Any = None,
) -> Any:
    """Call ordinary project models or noise-aware external denoisers.

    Existing U-TV, U-TGV, and U-Tikhonov models still receive only the noisy
    tensor.  Models that explicitly declare ``requires_noise_level=True`` also
    receive the batch's known normalized Gaussian noise level.
    """
    if bool(getattr(model, "requires_noise_level", False)):
        if sigma is None:
            raise ValueError(
                f"{type(model).__name__} requires a known noise level."
            )
        return model(noisy, sigma)
    return model(noisy)


def unpack_model_output(output: Any) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Return ``(reconstruction, parameter_maps)`` from any project model."""
    if torch.is_tensor(output):
        return output, ()
    if isinstance(output, Sequence) and output and torch.is_tensor(output[0]):
        maps = tuple(value for value in output[1:] if torch.is_tensor(value))
        return output[0], maps
    raise TypeError("A model must return a tensor or a tuple beginning with one.")


def _parameter_names(
    model: torch.nn.Module,
    model_type: Optional[str],
    count: int,
) -> tuple[str, ...]:
    """Resolve stable names for the parameter maps returned by a model."""
    names = tuple(getattr(model, "parameter_names", ()))
    if len(names) == count:
        return names

    legacy = {
        "utv": ("lambda",),
        "utgv": ("lambda0", "lambda1"),
        "utikhonov": ("lambda",),
        "u_tikhonov": ("lambda",),
    }
    if model_type is not None and len(legacy.get(model_type.lower(), ())) == count:
        return legacy[model_type.lower()]
    return tuple(f"parameter_{index}" for index in range(count))


def _metadata_value(batch: Mapping[str, Any], key: str, index: int, default: Any) -> Any:
    """Extract one sample's metadata from tensor, sequence, or scalar batches."""
    if key not in batch:
        return default
    value = batch[key]
    if torch.is_tensor(value):
        flattened = value.reshape(value.shape[0], -1) if value.ndim > 0 else value.reshape(1, 1)
        selected = flattened[min(index, flattened.shape[0] - 1)]
        return selected[0].item() if selected.numel() else default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[min(index, len(value) - 1)] if value else default
    return value


@torch.no_grad()
def evaluate_denoising_model(
    model: torch.nn.Module,
    loader,
    device: str | torch.device,
    model_type: Optional[str] = None,
    *,
    clamp_for_metrics: bool = True,
) -> pd.DataFrame:
    """Evaluate any learned denoiser with identical noise and metric handling."""
    model.eval()
    rows = []
    global_index = 0

    for batch in loader:
        if not isinstance(batch, Mapping) or "noisy" not in batch or "clean" not in batch:
            raise TypeError("Evaluation batches must contain 'noisy' and 'clean'.")
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        reconstruction, maps = unpack_model_output(
            call_denoising_model(model, noisy, batch.get("sigma"))
        )
        scored = reconstruction.clamp(0.0, 1.0) if clamp_for_metrics else reconstruction

        noisy_mse = batch_mse(noisy, clean)
        noisy_psnr = batch_psnr(noisy, clean)
        noisy_ssim = batch_ssim(noisy, clean)
        recon_mse = batch_mse(scored, clean)
        recon_psnr = batch_psnr(scored, clean)
        recon_ssim = batch_ssim(scored, clean)
        names = _parameter_names(model, model_type, len(maps))

        # Store one row per image so later comparisons remain paired by sample ID.
        for sample_index in range(clean.shape[0]):
            row = {
                "idx": global_index,
                "base_idx": _metadata_value(batch, "base_idx", sample_index, global_index),
                "crop_id": _metadata_value(batch, "crop_id", sample_index, 0),
                "path": _metadata_value(batch, "path", sample_index, ""),
                "sample_id": _metadata_value(
                    batch, "sample_id", sample_index, f"sample_{global_index:05d}"
                ),
                "dataset": _metadata_value(batch, "dataset", sample_index, ""),
                "image_name": _metadata_value(batch, "image_name", sample_index, ""),
                "noise_seed": _metadata_value(
                    batch, "noise_seed", sample_index, float("nan")
                ),
                "clean_sha256": _metadata_value(
                    batch, "clean_sha256", sample_index, ""
                ),
                "noisy_sha256": _metadata_value(
                    batch, "noisy_sha256", sample_index, ""
                ),
                "sigma": _metadata_value(batch, "sigma", sample_index, float("nan")),
                "noisy_mse": float(noisy_mse[sample_index].item()),
                "noisy_psnr": float(noisy_psnr[sample_index].item()),
                "noisy_ssim": float(noisy_ssim[sample_index].item()),
                "recon_mse": float(recon_mse[sample_index].item()),
                "recon_psnr": float(recon_psnr[sample_index].item()),
                "recon_ssim": float(recon_ssim[sample_index].item()),
            }
            for attribute, column in (
                ("last_outer_iterations", "solver_outer_iterations"),
                ("last_inner_iterations", "solver_last_inner_iterations"),
                (
                    "last_total_inner_iterations",
                    "solver_total_inner_iterations",
                ),
            ):
                if hasattr(model, attribute):
                    row[column] = int(getattr(model, attribute))
            if hasattr(model, "last_outer_relative_residual"):
                residual = torch.as_tensor(
                    getattr(model, "last_outer_relative_residual")
                ).reshape(-1)
                if residual.numel():
                    row["solver_outer_relative_residual"] = float(
                        residual[min(sample_index, residual.numel() - 1)]
                    )
            if hasattr(model, "last_lambda"):
                lambda_values = torch.as_tensor(
                    getattr(model, "last_lambda")
                ).reshape(-1)
                if lambda_values.numel():
                    row["deal_lambda"] = float(
                        lambda_values[
                            min(sample_index, lambda_values.numel() - 1)
                        ]
                    )
            for name, parameter_map in zip(names, maps):
                values = (
                    parameter_map[sample_index]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                flattened = values.reshape(-1)
                # Summarize each learned map without retaining full-resolution tensors in the CSV.
                quantiles = torch.quantile(
                    flattened,
                    torch.tensor(
                        [0.01, 0.50, 0.99],
                        dtype=flattened.dtype,
                    ),
                )
                row[f"{name}_mean"] = float(values.mean().item())
                row[f"{name}_std"] = float(values.std(unbiased=False).item())
                row[f"{name}_min"] = float(values.amin().item())
                row[f"{name}_max"] = float(values.amax().item())
                row[f"{name}_p01"] = float(quantiles[0].item())
                row[f"{name}_median"] = float(quantiles[1].item())
                row[f"{name}_p99"] = float(quantiles[2].item())

                lower_bound = getattr(model, f"{name}_min", None)
                upper_bound = getattr(model, f"{name}_max", None)
                if name == "lambda":
                    lower_bound = getattr(model, "lambda_min", lower_bound)
                    upper_bound = getattr(model, "lambda_max", upper_bound)
                if lower_bound is not None and upper_bound is not None:
                    value_range = float(upper_bound) - float(lower_bound)
                    tolerance = max(0.01 * value_range, 1e-12)
                    row[f"{name}_lower_saturation_fraction"] = float(
                        (values <= float(lower_bound) + tolerance)
                        .to(torch.float32)
                        .mean()
                        .item()
                    )
                    row[f"{name}_upper_saturation_fraction"] = float(
                        (values >= float(upper_bound) - tolerance)
                        .to(torch.float32)
                        .mean()
                        .item()
                    )
            rows.append(row)
            global_index += 1

    if not rows:
        raise ValueError("Evaluation loader is empty.")
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> dict[str, float]:
    """Average per-image metrics and aggregate parameter-map statistics."""
    if results.empty:
        raise ValueError("Cannot summarize an empty result table.")

    summary = {
        column: float(results[column].mean())
        for column in (
            "noisy_mse",
            "noisy_psnr",
            "noisy_ssim",
            "recon_mse",
            "recon_psnr",
            "recon_ssim",
        )
    }
    parameter_suffixes = (
        "_mean",
        "_std",
        "_min",
        "_max",
        "_p01",
        "_median",
        "_p99",
        "_saturation_fraction",
    )
    for column in results.columns:
        if column in summary or not column.endswith(parameter_suffixes):
            continue
        if column.endswith("_min"):
            summary[column] = float(results[column].min())
        elif column.endswith("_max"):
            summary[column] = float(results[column].max())
        else:
            summary[column] = float(results[column].mean())
    return summary
