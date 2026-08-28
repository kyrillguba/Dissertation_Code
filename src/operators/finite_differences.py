"""Shared forward finite differences and their exact Euclidean adjoints.

For an image ``u`` with ``C`` channels, :func:`gradient` returns ``2*C``
channels ordered as all horizontal components followed by all vertical
components. Boundary differences on the final column/row are zero.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _require_image_tensor(tensor: Tensor, name: str) -> None:
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected {name} with shape (B, C, H, W), got {tuple(tensor.shape)}."
        )


def gradient(u: Tensor) -> Tensor:
    """Apply the 2-D forward finite-difference operator ``D``."""
    _require_image_tensor(u, "u")
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    gy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return torch.cat((gx, gy), dim=1)


def gradient_adjoint(p: Tensor) -> Tensor:
    """Apply the exact adjoint ``D^T`` of :func:`gradient`."""
    _require_image_tensor(p, "p")
    if p.shape[1] % 2 != 0:
        raise ValueError("The gradient field must have 2*C channels.")

    channels = p.shape[1] // 2
    px = p[:, :channels]
    py = p[:, channels:]
    out = torch.zeros_like(px)

    out[..., :, :-1] -= px[..., :, :-1]
    out[..., :, 1:] += px[..., :, :-1]
    out[..., :-1, :] -= py[..., :-1, :]
    out[..., 1:, :] += py[..., :-1, :]
    return out


def divergence(p: Tensor) -> Tensor:
    """Apply ``div = -D^T`` under the project's finite-difference convention."""
    return -gradient_adjoint(p)


def sym_gradient(w: Tensor) -> Tensor:
    """Apply the symmetrised gradient ``E`` to a 2-D vector field.

    ``w`` has ``2*C`` channels, ordered ``(w_x, w_y)``. The output has
    ``3*C`` channels ordered ``(e_xx, e_yy, e_xy)`` with
    ``e_xy = 0.5 * (d_y w_x + d_x w_y)``.
    """
    _require_image_tensor(w, "w")
    if w.shape[1] % 2 != 0:
        raise ValueError("The vector field must have 2*C channels.")

    channels = w.shape[1] // 2
    wx = w[:, :channels]
    wy = w[:, channels:]
    grad_wx = gradient(wx)
    grad_wy = gradient(wy)

    dx_wx, dy_wx = grad_wx[:, :channels], grad_wx[:, channels:]
    dx_wy, dy_wy = grad_wy[:, :channels], grad_wy[:, channels:]
    return torch.cat(
        (dx_wx, dy_wy, 0.5 * (dy_wx + dx_wy)),
        dim=1,
    )


def sym_gradient_adjoint(q: Tensor) -> Tensor:
    """Apply the exact adjoint ``E^T`` of :func:`sym_gradient`."""
    _require_image_tensor(q, "q")
    if q.shape[1] % 3 != 0:
        raise ValueError("The symmetric tensor field must have 3*C channels.")

    channels = q.shape[1] // 3
    q_xx = q[:, :channels]
    q_yy = q[:, channels : 2 * channels]
    q_xy = q[:, 2 * channels :]

    adjoint_x = gradient_adjoint(torch.cat((q_xx, 0.5 * q_xy), dim=1))
    adjoint_y = gradient_adjoint(torch.cat((0.5 * q_xy, q_yy), dim=1))
    return torch.cat((adjoint_x, adjoint_y), dim=1)


def sym_divergence(q: Tensor) -> Tensor:
    """Apply ``-E^T`` under the project's symmetrised-gradient convention."""
    return -sym_gradient_adjoint(q)


def gradient_adjoint_error(
    *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64
) -> float:
    """Return the relative inner-product error for ``D`` and ``D^T``."""
    u = torch.randn(2, 1, 17, 19, device=device, dtype=dtype)
    p = torch.randn(2, 2, 17, 19, device=device, dtype=dtype)
    lhs = torch.sum(gradient(u) * p)
    rhs = torch.sum(u * gradient_adjoint(p))
    scale = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(dtype).eps)
    return float(((lhs - rhs).abs() / scale).item())


def sym_gradient_adjoint_error(
    *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64
) -> float:
    """Return the relative inner-product error for ``E`` and ``E^T``."""
    w = torch.randn(2, 2, 17, 19, device=device, dtype=dtype)
    q = torch.randn(2, 3, 17, 19, device=device, dtype=dtype)
    lhs = torch.sum(sym_gradient(w) * q)
    rhs = torch.sum(w * sym_gradient_adjoint(q))
    scale = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(dtype).eps)
    return float(((lhs - rhs).abs() / scale).item())

