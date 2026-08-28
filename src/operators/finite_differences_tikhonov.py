"""Backward-compatible Tikhonov wrappers around the shared operators.

New code should import :func:`gradient` and :func:`gradient_adjoint` from
``finite_differences`` directly. These wrappers retain the earlier two-tensor
interface used by the notebooks.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .finite_differences import gradient, gradient_adjoint as _gradient_adjoint

Tensor = torch.Tensor


def forward_gradient(u: Tensor) -> Tuple[Tensor, Tensor]:
    """Return horizontal and vertical forward differences separately."""
    stacked = gradient(u)
    channels = u.shape[1]
    return stacked[:, :channels], stacked[:, channels:]


def gradient_adjoint(gx: Tensor, gy: Tensor) -> Tensor:
    """Apply ``D^T`` to separately supplied gradient components."""
    if gx.shape != gy.shape:
        raise ValueError(
            f"gx and gy must have the same shape, got {gx.shape} and {gy.shape}."
        )
    return _gradient_adjoint(torch.cat((gx, gy), dim=1))

