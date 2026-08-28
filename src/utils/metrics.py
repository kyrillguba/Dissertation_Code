"""MSE, PSNR and SSIM with a single convention for every model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _validate_pair(pred: Tensor, target: Tensor) -> None:
    """Require matching non-empty batched image tensors."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
    if pred.ndim != 4:
        raise ValueError("Expected tensors with shape (B, C, H, W).")
    if pred.shape[0] == 0:
        raise ValueError("Metric batches must not be empty.")


def batch_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Return one mean-squared error per image."""
    _validate_pair(pred, target)
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over the complete batch."""
    return batch_mse(pred, target).mean()


def batch_psnr(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """Return one peak signal-to-noise ratio per image."""
    if data_range <= 0 or eps <= 0:
        raise ValueError("data_range and eps must be strictly positive.")
    values = batch_mse(pred, target)
    numerator = torch.as_tensor(data_range**2, dtype=values.dtype, device=values.device)
    return 10.0 * torch.log10(numerator / values.clamp_min(eps))


def psnr(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """Average per-image PSNR.

    Averaging per-image values matches the evaluation pipeline and avoids a
    batch-size-dependent change in meaning.
    """
    return batch_psnr(pred, target, data_range=data_range, eps=eps).mean()


def _ssim_per_image(
    pred: Tensor,
    target: Tensor,
    data_range: float,
    window_size: int,
    eps: float,
) -> Tensor:
    """Compute local-window SSIM independently for each image."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")
    if min(pred.shape[-2:]) < window_size:
        raise ValueError(
            f"Images must be at least {window_size}x{window_size} for SSIM."
        )

    padding = window_size // 2
    # Compute local first and second moments with the shared uniform window.
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(
        pred.square(), window_size, stride=1, padding=padding
    ) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(
        target.square(), window_size, stride=1, padding=padding
    ) - mu_y_sq
    sigma_xy = F.avg_pool2d(
        pred * target, window_size, stride=1, padding=padding
    ) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (
        sigma_x_sq + sigma_y_sq + c2
    )
    ssim_map = numerator / (denominator + eps)
    return ssim_map.mean(dim=(1, 2, 3))


def batch_ssim(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    eps: float = 1e-12,
) -> Tensor:
    """Return one local-window SSIM value per image."""
    _validate_pair(pred, target)
    if data_range <= 0 or eps <= 0:
        raise ValueError("data_range and eps must be strictly positive.")
    return _ssim_per_image(pred, target, data_range, window_size, eps)


def ssim(
    pred: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    eps: float = 1e-12,
) -> Tensor:
    """Average per-image SSIM."""
    return batch_ssim(
        pred,
        target,
        data_range=data_range,
        window_size=window_size,
        eps=eps,
    ).mean()

