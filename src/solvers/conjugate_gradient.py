"""Batched conjugate-gradient solvers for SPD linear systems."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

Tensor = torch.Tensor
LinearOperator = Callable[[Tensor], Tensor]


def batched_inner(x: Tensor, y: Tensor) -> Tensor:
    """Return one inner product per batch element with broadcastable shape."""
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}.")
    if x.ndim < 2:
        raise ValueError("Batched tensors must have at least two dimensions.")
    return torch.sum(x * y, dim=tuple(range(1, x.ndim)), keepdim=True)


def _batch_mask(mask: Tensor, reference: Tensor) -> Tensor:
    """Reshape a per-sample Boolean mask for tensor-wide broadcasting."""
    return mask.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def _validate_solver_arguments(iterations: int, tolerance: float, eps: float) -> None:
    """Validate iteration, tolerance, and numerical-safeguard settings."""
    if iterations < 0:
        raise ValueError("The iteration count must be non-negative.")
    if tolerance < 0:
        raise ValueError("The relative tolerance must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")


def _safe_denominator(
    denominator: Tensor,
    active: Tensor,
    eps: float,
    check_spd: bool,
    context: str,
) -> Tensor:
    """Validate or stabilize a CG denominator for active batch elements."""
    if check_spd and torch.any(denominator[active] <= 0):
        bad = denominator[active][denominator[active] <= 0]
        raise RuntimeError(
            f"{context} encountered p^T A p <= 0. The operator is not SPD; "
            f"first offending values: {bad.detach().cpu().tolist()[:5]}"
        )

    safe = torch.where(active, denominator, torch.ones_like(denominator))
    if not check_spd:
        sign = torch.where(safe < 0, -torch.ones_like(safe), torch.ones_like(safe))
        safe = torch.where(
            active & (safe.abs() < eps),
            sign * eps,
            safe,
        )
    return safe


@torch.no_grad()
def cg_solve_until_converged(
    A: LinearOperator,
    b: Tensor,
    x0: Optional[Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
    check_spd: bool = True,
) -> Tuple[Tensor, Dict[str, object]]:
    """Solve ``A x = b`` and report per-sample convergence diagnostics."""
    _validate_solver_arguments(max_iter, tol, eps)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    if x.shape != b.shape:
        raise ValueError("x0 and b must have identical shapes.")

    r = b - A(x)
    p = r.clone()
    rs_old = batched_inner(r, r)
    b_norm_sq = batched_inner(b, b).clamp_min(eps)

    # Track convergence independently because batch samples can stop at different iterations.
    initial_rel = torch.sqrt(rs_old / b_norm_sq).reshape(b.shape[0])
    converged = initial_rel <= tol
    iteration_counts = torch.zeros(b.shape[0], dtype=torch.long, device=b.device)
    mean_history = [float(initial_rel.mean().item())]
    max_history = [float(initial_rel.max().item())]

    iteration = 0
    for next_iteration in range(1, max_iter + 1):
        active = ~converged
        if not torch.any(active):
            break
        iteration = next_iteration

        Ap = A(p)
        pAp = batched_inner(p, Ap).reshape(b.shape[0])
        safe_pAp = _safe_denominator(pAp, active, eps, check_spd, "CG")

        alpha_flat = torch.where(
            active,
            rs_old.reshape(b.shape[0]) / safe_pAp,
            torch.zeros_like(safe_pAp),
        )
        alpha = _batch_mask(alpha_flat, b)
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = batched_inner(r, r)
        rel_res = torch.sqrt(rs_new / b_norm_sq).reshape(b.shape[0])
        mean_history.append(float(rel_res.mean().item()))
        max_history.append(float(rel_res.max().item()))

        newly_converged = active & (rel_res <= tol)
        iteration_counts[newly_converged] = iteration
        converged = converged | newly_converged

        if torch.all(converged):
            rs_old = rs_new
            break

        old_flat = rs_old.reshape(b.shape[0])
        new_flat = rs_new.reshape(b.shape[0])
        safe_old = torch.where(
            active & (old_flat > eps),
            old_flat,
            torch.ones_like(old_flat),
        )
        beta_flat = torch.where(
            ~converged,
            new_flat / safe_old,
            torch.zeros_like(new_flat),
        )
        p = r + _batch_mask(beta_flat, b) * p
        p = torch.where(_batch_mask(~converged, b), p, torch.zeros_like(p))
        rs_old = rs_new

    unfinished = (~converged) & (iteration_counts == 0)
    iteration_counts[unfinished] = min(iteration, max_iter)

    true_residual = b - A(x)
    final_rel = torch.sqrt(
        batched_inner(true_residual, true_residual) / b_norm_sq
    ).reshape(b.shape[0])
    final_converged = final_rel <= tol

    return x, {
        "num_iter": int(min(iteration, max_iter)),
        "converged": bool(torch.all(final_converged).item()),
        "per_sample_converged": final_converged.cpu().tolist(),
        "per_sample_iterations": iteration_counts.cpu().tolist(),
        "final_rel_residual": float(final_rel.mean().item()),
        "max_final_rel_residual": float(final_rel.max().item()),
        "per_sample_final_rel_residual": final_rel.cpu().tolist(),
        "residual_history": mean_history,
        "max_residual_history": max_history,
    }


def cg_solve_fixed_iters(
    A: LinearOperator,
    b: Tensor,
    x0: Optional[Tensor] = None,
    num_iters: int = 32,
    relative_tol: float = 1e-7,
    eps: float = 1e-12,
    check_spd: bool = True,
) -> Tensor:
    """Differentiable fixed-depth CG for the unrolled learned model.

    Samples are frozen after satisfying ``relative_tol``. The maximum unrolled
    depth remains fixed by ``num_iters`` and gradients flow through every active
    arithmetic update.
    """
    _validate_solver_arguments(num_iters, relative_tol, eps)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    if x.shape != b.shape:
        raise ValueError("x0 and b must have identical shapes.")

    r = b - A(x)
    p = r.clone()
    rs_old = batched_inner(r, r)
    b_norm_sq = batched_inner(b, b).clamp_min(eps)

    # Freeze converged samples while preserving differentiability for active samples.
    for _ in range(num_iters):
        rel_res = torch.sqrt(rs_old / b_norm_sq).reshape(b.shape[0])
        active = rel_res > relative_tol
        if not torch.any(active):
            break

        Ap = A(p)
        pAp = batched_inner(p, Ap).reshape(b.shape[0])
        safe_pAp = _safe_denominator(
            pAp,
            active,
            eps,
            check_spd,
            "Differentiable CG",
        )
        alpha_flat = torch.where(
            active,
            rs_old.reshape(b.shape[0]) / safe_pAp,
            torch.zeros_like(safe_pAp),
        )
        alpha = _batch_mask(alpha_flat, b)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = batched_inner(r_new, r_new)

        old_flat = rs_old.reshape(b.shape[0])
        new_flat = rs_new.reshape(b.shape[0])
        safe_old = torch.where(
            active & (old_flat > eps),
            old_flat,
            torch.ones_like(old_flat),
        )
        beta_flat = torch.where(
            active,
            new_flat / safe_old,
            torch.zeros_like(new_flat),
        )
        p_new = r_new + _batch_mask(beta_flat, b) * p

        active_mask = _batch_mask(active, b)
        x = torch.where(active_mask, x_new, x)
        r = torch.where(active_mask, r_new, r)
        p = torch.where(active_mask, p_new, torch.zeros_like(p))
        rs_old = torch.where(active_mask, rs_new, rs_old)

    return x
