"""PDHG solvers for scalar and spatially weighted anisotropic TV."""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.operators.finite_differences import divergence, gradient

Tensor = torch.Tensor
TV_OPERATOR_NORM_SQUARED_BOUND = 8.0
DEFAULT_TV_STEP = 0.99 / math.sqrt(TV_OPERATOR_NORM_SQUARED_BOUND)


def _validate_image(f: Tensor) -> None:
    """Require a batched image tensor for the TV solver."""
    if f.ndim != 4:
        raise ValueError(f"Expected f with shape (B, C, H, W), got {f.shape}.")


def _steps(
    tau: Optional[float],
    sigma: Optional[float],
    theta: Optional[float],
) -> tuple[float, float, float]:
    """Resolve and validate stable PDHG step and extrapolation parameters."""
    tau = DEFAULT_TV_STEP if tau is None else float(tau)
    sigma = DEFAULT_TV_STEP if sigma is None else float(sigma)
    theta = 1.0 if theta is None else float(theta)
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be strictly positive.")
    if not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1].")
    if tau * sigma * TV_OPERATOR_NORM_SQUARED_BOUND >= 1:
        raise ValueError("TV PDHG requires tau * sigma * 8 < 1.")
    return tau, sigma, theta


def project_box(p: Tensor, radius: Tensor) -> Tensor:
    """Project onto the anisotropic box ``[-radius, radius]``."""
    return torch.maximum(torch.minimum(p, radius), -radius)


def weighted_tv_denoise_pdhg(
    f: Tensor,
    Lambda: Tensor,
    num_iters: int = 256,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    theta: Optional[float] = None,
) -> Tensor:
    r"""Approximately solve

    ``min_u 0.5 ||u-f||^2 + sum_x Lambda(x) |D u(x)|_1``.

    The unconstrained reconstruction is returned. Clamp only for image display
    or metric computation so all three learned models use the same convention.
    """
    _validate_image(f)
    if num_iters < 0:
        raise ValueError("num_iters must be non-negative.")
    if Lambda.ndim != 4 or Lambda.shape[0] != f.shape[0] or Lambda.shape[-2:] != f.shape[-2:]:
        raise ValueError(f"Incompatible f/Lambda shapes: {f.shape} and {Lambda.shape}.")
    if torch.any(Lambda < 0):
        raise ValueError("Lambda must be non-negative.")

    tau, sigma, theta = _steps(tau, sigma, theta)
    gradient_channels = 2 * f.shape[1]
    if Lambda.shape[1] == 1:
        radius = Lambda.expand(-1, gradient_channels, -1, -1)
    elif Lambda.shape[1] == gradient_channels:
        radius = Lambda
    else:
        raise ValueError("Lambda must have 1 or 2*C channels.")

    u = f.clone()
    u_bar = u.clone()
    p = torch.zeros_like(gradient(u))
    # Apply the fixed-depth dual, primal, and extrapolation updates.
    for _ in range(num_iters):
        p = project_box(p + sigma * gradient(u_bar), radius)
        u_old = u
        u = (u + tau * divergence(p) + tau * f) / (1.0 + tau)
        u_bar = u + theta * (u - u_old)
    return u


def tv_denoise_pdhg(
    f: Tensor,
    lam: float = 0.05,
    num_iters: int = 256,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    theta: Optional[float] = None,
) -> Tensor:
    """Scalar anisotropic TV denoising using the weighted implementation."""
    if lam < 0:
        raise ValueError("lam must be non-negative.")
    Lambda = torch.full_like(f[:, :1], float(lam))
    return weighted_tv_denoise_pdhg(
        f,
        Lambda,
        num_iters=num_iters,
        tau=tau,
        sigma=sigma,
        theta=theta,
    )
