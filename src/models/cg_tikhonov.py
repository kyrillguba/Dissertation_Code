"""U-Tikhonov: learned quadratic regularisation map and unrolled CG."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import UNet
from src.operators.finite_differences import gradient, gradient_adjoint
from src.solvers.conjugate_gradient import cg_solve_fixed_iters

Tensor = torch.Tensor


def scalar_tikhonov_operator(u: Tensor, lam: float) -> Tensor:
    r"""Apply ``A u = u + lambda D^T D u``."""
    if lam < 0:
        raise ValueError("lam must be non-negative for an SPD system.")
    return u + float(lam) * gradient_adjoint(gradient(u))


def weighted_tikhonov_operator(u: Tensor, lambda_map: Tensor) -> Tensor:
    r"""Apply ``A_Lambda u = u + D^T Lambda D u``."""
    if u.ndim != 4 or lambda_map.ndim != 4:
        raise ValueError("u and lambda_map must have shape (B, C, H, W).")
    if lambda_map.shape[0] != u.shape[0] or lambda_map.shape[-2:] != u.shape[-2:]:
        raise ValueError(
            f"Incompatible u/lambda_map shapes: {u.shape} and {lambda_map.shape}."
        )
    if lambda_map.shape[1] not in (1, u.shape[1], 2 * u.shape[1]):
        raise ValueError("lambda_map must have 1, C, or 2*C channels.")

    grad_u = gradient(u)
    if lambda_map.shape[1] == u.shape[1]:
        lambda_map = torch.cat((lambda_map, lambda_map), dim=1)
    return u + gradient_adjoint(lambda_map * grad_u)


class UTikhonovModel(nn.Module):
    """Predict ``Lambda`` and solve the weighted quadratic system with CG."""

    parameter_names = ("lambda",)

    def __init__(
        self,
        base_channels: int = 32,
        depth: int = 3,
        cg_iters: int = 32,
        map_scale: float = 0.1,
        cg_relative_tol: float = 1e-7,
        cg_eps: float = 1e-12,
        check_spd: bool = True,
        parameterization: str = "scaled_softplus",
        lambda_min: float = 1e-4,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if cg_iters < 0:
            raise ValueError("cg_iters must be non-negative.")
        if map_scale <= 0:
            raise ValueError("map_scale must be strictly positive.")
        if parameterization not in {"scaled_softplus", "bounded_sigmoid"}:
            raise ValueError(
                "parameterization must be 'scaled_softplus' or 'bounded_sigmoid'."
            )
        if lambda_min < 0:
            raise ValueError("lambda_min must be non-negative.")
        if (
            parameterization == "bounded_sigmoid"
            and (lambda_max is None or lambda_max <= lambda_min)
        ):
            raise ValueError(
                "bounded_sigmoid requires a finite lambda_max > lambda_min."
            )

        self.net = UNet(1, 1, base_channels=base_channels, depth=depth)
        self.cg_iters = int(cg_iters)
        self.map_scale = float(map_scale)
        self.cg_relative_tol = float(cg_relative_tol)
        self.cg_eps = float(cg_eps)
        self.check_spd = bool(check_spd)
        self.parameterization = str(parameterization)
        self.lambda_min = float(lambda_min)
        self.lambda_max = None if lambda_max is None else float(lambda_max)

    @property
    def lambda_net(self) -> UNet:
        """Compatibility alias for earlier notebooks."""
        return self.net

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Migrate checkpoints saved before ``lambda_net`` became ``net``."""
        old_prefix = f"{prefix}lambda_net."
        new_prefix = f"{prefix}net."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                replacement = new_prefix + key[len(old_prefix) :]
                if replacement not in state_dict:
                    state_dict[replacement] = state_dict[key]
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def predict_parameters(self, noisy: Tensor) -> Tuple[Tensor]:
        raw = self.net(noisy)
        if self.parameterization == "bounded_sigmoid":
            lambda_map = self.lambda_min + (
                self.lambda_max - self.lambda_min
            ) * torch.sigmoid(raw)
        else:
            lambda_map = self.map_scale * F.softplus(raw)
        return (lambda_map,)

    def predict_lambda(self, noisy: Tensor) -> Tensor:
        """Compatibility wrapper returning the single predicted map."""
        return self.predict_parameters(noisy)[0]

    def forward(self, noisy: Tensor) -> tuple[Tensor, Tensor]:
        (lambda_map,) = self.predict_parameters(noisy)

        def operator(u: Tensor) -> Tensor:
            return weighted_tikhonov_operator(u, lambda_map)

        reconstruction = cg_solve_fixed_iters(
            A=operator,
            b=noisy,
            x0=noisy,
            num_iters=self.cg_iters,
            relative_tol=self.cg_relative_tol,
            eps=self.cg_eps,
            check_spd=self.check_spd,
        )
        return reconstruction, lambda_map


# Preserve the original public class name while making the three model names
# symmetric in new code.
WeightedTikhonovCGDenoiser = UTikhonovModel
