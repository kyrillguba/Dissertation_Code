"""U-TV: learned spatial TV map followed by unrolled PDHG."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import UNet
from src.solvers.pdhg_tv import weighted_tv_denoise_pdhg

Tensor = torch.Tensor


class UTVModel(nn.Module):
    """Predict ``Lambda`` and solve the weighted TV reconstruction problem."""

    parameter_names = ("lambda",)

    def __init__(
        self,
        base_channels: int = 32,
        depth: int = 3,
        num_pdhg_iters: int = 64,
        map_scale: float = 0.1,
        pdhg_tau: Optional[float] = None,
        pdhg_sigma: Optional[float] = None,
        pdhg_theta: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_pdhg_iters < 0:
            raise ValueError("num_pdhg_iters must be non-negative.")
        if map_scale <= 0:
            raise ValueError("map_scale must be strictly positive.")

        self.net = UNet(1, 1, base_channels=base_channels, depth=depth)
        self.num_pdhg_iters = int(num_pdhg_iters)
        self.map_scale = float(map_scale)
        self.pdhg_tau = pdhg_tau
        self.pdhg_sigma = pdhg_sigma
        self.pdhg_theta = pdhg_theta

    def predict_parameters(self, noisy: Tensor) -> Tuple[Tensor]:
        return (self.map_scale * F.softplus(self.net(noisy)),)

    def forward(self, noisy: Tensor) -> tuple[Tensor, Tensor]:
        (lambda_map,) = self.predict_parameters(noisy)
        reconstruction = weighted_tv_denoise_pdhg(
            noisy,
            Lambda=lambda_map,
            num_iters=self.num_pdhg_iters,
            tau=self.pdhg_tau,
            sigma=self.pdhg_sigma,
            theta=self.pdhg_theta,
        )
        return reconstruction, lambda_map

