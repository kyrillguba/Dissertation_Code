"""PDHG solvers for scalar and spatially weighted anisotropic TGV."""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.operators.finite_differences import (
    divergence,
    gradient,
    sym_divergence,
    sym_gradient,
)

Tensor = torch.Tensor
TGV_OPERATOR_NORM_SQUARED_BOUND = 13.0
DEFAULT_TGV_STEP = 0.99 / math.sqrt(TGV_OPERATOR_NORM_SQUARED_BOUND)


def _steps(
    tau: Optional[float],
    sigma: Optional[float],
    theta: Optional[float],
) -> tuple[float, float, float]:
    """Resolve and validate stable PDHG step and extrapolation parameters."""
    tau = DEFAULT_TGV_STEP if tau is None else float(tau)
    sigma = DEFAULT_TGV_STEP if sigma is None else float(sigma)
    theta = 1.0 if theta is None else float(theta)
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be strictly positive.")
    if not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1].")
    if tau * sigma * TGV_OPERATOR_NORM_SQUARED_BOUND >= 1:
        raise ValueError("TGV PDHG requires tau * sigma * 13 < 1.")
    return tau, sigma, theta


def project_box(value: Tensor, radius: Tensor) -> Tensor:
    """Project a tensor elementwise onto a spatially varying symmetric box."""
    return torch.maximum(torch.minimum(value, radius), -radius)


def weighted_tgv_denoise_pdhg(
    f: Tensor,
    Lambda0: Tensor,
    Lambda1: Tensor,
    num_iters: int = 256,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    theta: Optional[float] = None,
) -> Tensor:
    r"""Approximately solve spatially weighted second-order TGV denoising.

    ``Lambda1`` weights ``|D u-w|`` and ``Lambda0`` weights ``|E w|``.
    The unconstrained reconstruction is returned.
    """
    if f.ndim != 4:
        raise ValueError(f"Expected f with shape (B, C, H, W), got {f.shape}.")
    if num_iters < 0:
        raise ValueError("num_iters must be non-negative.")
    for name, parameter in (("Lambda0", Lambda0), ("Lambda1", Lambda1)):
        if (
            parameter.ndim != 4
            or parameter.shape[0] != f.shape[0]
            or parameter.shape[-2:] != f.shape[-2:]
        ):
            raise ValueError(f"Incompatible f/{name} shapes: {f.shape} and {parameter.shape}.")
        if torch.any(parameter < 0):
            raise ValueError(f"{name} must be non-negative.")

    tau, sigma, theta = _steps(tau, sigma, theta)
    channels = f.shape[1]
    first_order_channels = 2 * channels
    second_order_channels = 3 * channels

    if Lambda1.shape[1] == 1:
        radius1 = Lambda1.expand(-1, first_order_channels, -1, -1)
    elif Lambda1.shape[1] == first_order_channels:
        radius1 = Lambda1
    else:
        raise ValueError("Lambda1 must have 1 or 2*C channels.")

    if Lambda0.shape[1] == 1:
        radius0 = Lambda0.expand(-1, second_order_channels, -1, -1)
    elif Lambda0.shape[1] == second_order_channels:
        radius0 = Lambda0
    else:
        raise ValueError("Lambda0 must have 1 or 3*C channels.")

    batch, _, height, width = f.shape
    u = f.clone()
    w = torch.zeros(
        batch,
        first_order_channels,
        height,
        width,
        dtype=f.dtype,
        device=f.device,
    )
    u_bar = u.clone()
    w_bar = w.clone()
    p = torch.zeros_like(w)
    q = torch.zeros(
        batch,
        second_order_channels,
        height,
        width,
        dtype=f.dtype,
        device=f.device,
    )

    # Alternate dual projections, primal proximal updates, and extrapolation.
    for _ in range(num_iters):
        p = project_box(p + sigma * (gradient(u_bar) - w_bar), radius1)
        q = project_box(q + sigma * sym_gradient(w_bar), radius0)

        u_old, w_old = u, w
        u = (u + tau * divergence(p) + tau * f) / (1.0 + tau)
        w = w + tau * p + tau * sym_divergence(q)
        u_bar = u + theta * (u - u_old)
        w_bar = w + theta * (w - w_old)

    return u


def tgv_denoise_pdhg(
    f: Tensor,
    alpha0: float = 0.08,
    alpha1: float = 0.04,
    num_iters: int = 256,
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    theta: Optional[float] = None,
) -> Tensor:
    """Scalar TGV denoising using the weighted implementation."""
    if alpha0 < 0 or alpha1 < 0:
        raise ValueError("alpha0 and alpha1 must be non-negative.")
    Lambda0 = torch.full_like(f[:, :1], float(alpha0))
    Lambda1 = torch.full_like(f[:, :1], float(alpha1))
    return weighted_tgv_denoise_pdhg(
        f,
        Lambda0,
        Lambda1,
        num_iters=num_iters,
        tau=tau,
        sigma=sigma,
        theta=theta,
    )
