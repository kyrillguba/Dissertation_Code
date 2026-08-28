"""Scalar Tikhonov baselines and compatibility wrappers for learned evaluation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.evaluation.denoising_eval import evaluate_denoising_model, summarize_results
from src.models.cg_tikhonov import scalar_tikhonov_operator
from src.solvers.conjugate_gradient import cg_solve_until_converged
from src.utils.metrics import batch_mse, batch_psnr, batch_ssim

Tensor = torch.Tensor
Batch = Union[Tensor, Sequence[object], Mapping[str, object]]


def extract_clean_image(batch: Batch) -> Tensor:
    """Extract a clean target from dictionary, tuple/list, or tensor batches."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("clean", "target", "ground_truth", "image"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
        raise KeyError("Could not find a clean image tensor in the batch.")
    if isinstance(batch, Sequence):
        if len(batch) >= 2 and torch.is_tensor(batch[1]):
            return batch[1]
        if batch and torch.is_tensor(batch[0]):
            return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}.")


def extract_noisy_image(batch: Batch) -> Optional[Tensor]:
    """Extract a noisy input when the loader supplies one."""
    if isinstance(batch, Mapping):
        value = batch.get("noisy")
        return value if torch.is_tensor(value) else None
    if isinstance(batch, Sequence) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    return None


def make_deterministic_noisy(
    clean: Tensor,
    sigma: float,
    batch_idx: int,
    base_seed: int = 10_000,
) -> Tensor:
    """Create deterministic Gaussian noise without changing global RNG state."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    noisy_samples = []
    for sample_index, clean_sample in enumerate(clean):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(base_seed) + int(batch_idx) * clean.shape[0] + sample_index
        )
        noise = torch.randn(
            clean_sample.shape,
            generator=generator,
            dtype=clean.dtype,
            device="cpu",
        ).to(clean.device)
        noisy_samples.append(
            (clean_sample + float(sigma) * noise).clamp(0.0, 1.0)
        )
    return torch.stack(noisy_samples, dim=0)


@torch.no_grad()
def scalar_tikhonov_denoise(
    noisy: Tensor,
    lam: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[Tensor, Dict[str, object]]:
    """Solve the scalar quadratic Tikhonov system to a residual tolerance."""

    def operator(u: Tensor) -> Tensor:
        """Apply the scalar Tikhonov system matrix for conjugate gradient."""
        return scalar_tikhonov_operator(u, lam)

    return cg_solve_until_converged(
        A=operator,
        b=noisy,
        x0=noisy,
        max_iter=max_iter,
        tol=tol,
        eps=eps,
        check_spd=True,
    )


def _batch_with_noise(batch: Batch, noisy: Tensor, clean: Tensor, sigma: float) -> dict:
    """Return a dictionary batch containing the supplied noisy and clean tensors."""
    result = dict(batch) if isinstance(batch, Mapping) else {}
    result.update({"noisy": noisy, "clean": clean, "sigma": torch.full((clean.shape[0], 1), sigma)})
    return result


@torch.no_grad()
def evaluate_learned_tikhonov(
    model: torch.nn.Module,
    loader: Iterable[Batch],
    sigma: Optional[float],
    device: str | torch.device,
    base_seed: int = 10_000,
    clamp_for_metrics: bool = True,
) -> Dict[str, float]:
    """Compatibility wrapper using the common learned-model evaluator.

    Pass ``sigma=None`` to use the loader's noisy images. Supplying a value is
    retained for the earlier notebook API and creates deterministic noise.
    """
    evaluation_loader = loader
    if sigma is not None:
        evaluation_loader = (
            _batch_with_noise(
                batch,
                make_deterministic_noisy(
                    extract_clean_image(batch).to(device),
                    sigma,
                    batch_idx,
                    base_seed,
                ),
                extract_clean_image(batch).to(device),
                sigma,
            )
            for batch_idx, batch in enumerate(loader)
        )

    results = evaluate_denoising_model(
        model,
        evaluation_loader,
        device,
        model_type="utikhonov",
        clamp_for_metrics=clamp_for_metrics,
    )
    common = summarize_results(results)
    return {
        "mse": common["recon_mse"],
        "psnr": common["recon_psnr"],
        "ssim": common["recon_ssim"],
        "noisy_mse": common["noisy_mse"],
        "noisy_psnr": common["noisy_psnr"],
        "noisy_ssim": common["noisy_ssim"],
        "lambda_mean": common.get("lambda_mean", float("nan")),
        "lambda_min": common.get("lambda_min", float("nan")),
        "lambda_max": common.get("lambda_max", float("nan")),
    }


def _synchronize(device: str | torch.device) -> None:
    """Synchronize CUDA or MPS work before or after wall-clock timing."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def evaluate_scalar_tikhonov_grid(
    loader: Iterable[Batch],
    lambdas: Sequence[float],
    sigma: Optional[float],
    device: str | torch.device,
    max_iter: int = 100,
    tol: float = 1e-6,
    base_seed: int = 10_000,
    clamp_for_metrics: bool = True,
) -> pd.DataFrame:
    """Grid-search scalar lambda using the same loader noise and metrics."""
    rows = []
    # Evaluate every scalar candidate over the identical loader and noise convention.
    for lam in lambdas:
        metric_values = {name: [] for name in ("mse", "psnr", "ssim")}
        iteration_counts, residuals, runtimes = [], [], []

        for batch_idx, batch in enumerate(loader):
            clean = extract_clean_image(batch).to(device)
            noisy = extract_noisy_image(batch)
            if sigma is not None:
                noisy = make_deterministic_noisy(clean, sigma, batch_idx, base_seed)
            elif noisy is None:
                raise ValueError("The loader has no noisy image and sigma=None.")
            else:
                noisy = noisy.to(device)

            _synchronize(device)
            start = time.perf_counter()
            reconstruction, info = scalar_tikhonov_denoise(
                noisy,
                lam=float(lam),
                max_iter=max_iter,
                tol=tol,
            )
            _synchronize(device)
            runtimes.append(time.perf_counter() - start)

            scored = reconstruction.clamp(0.0, 1.0) if clamp_for_metrics else reconstruction
            metric_values["mse"].extend(batch_mse(scored, clean).cpu().tolist())
            metric_values["psnr"].extend(batch_psnr(scored, clean).cpu().tolist())
            metric_values["ssim"].extend(batch_ssim(scored, clean).cpu().tolist())
            iteration_counts.extend(info["per_sample_iterations"])
            residuals.extend(info["per_sample_final_rel_residual"])

        if not metric_values["mse"]:
            raise ValueError("Evaluation loader is empty.")
        rows.append(
            {
                "lambda": float(lam),
                "mse": float(np.mean(metric_values["mse"])),
                "psnr": float(np.mean(metric_values["psnr"])),
                "ssim": float(np.mean(metric_values["ssim"])),
                "avg_cg_iters": float(np.mean(iteration_counts)),
                "min_cg_iters": int(np.min(iteration_counts)),
                "max_cg_iters": int(np.max(iteration_counts)),
                "mean_final_rel_residual": float(np.mean(residuals)),
                "avg_runtime_sec": float(np.mean(runtimes)),
            }
        )

    return pd.DataFrame(rows).sort_values(["mse", "lambda"]).reset_index(drop=True)

