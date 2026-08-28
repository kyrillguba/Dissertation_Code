"""Adapter for the authors' official pretrained grayscale DEAL checkpoint.

The learned architecture and checkpoint are unchanged.  This module only
bridges the official implementation to the project's normalized tensors,
noise-aware evaluator, device handling, and diagnostic conventions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.third_party.deal_official import DEAL

Tensor = torch.Tensor

OFFICIAL_REPOSITORY = "https://github.com/mehrsapo/DEAL"
OFFICIAL_COMMIT = "554820ca356ff8a78bc49097e3ffcba3875a3ac2"
OFFICIAL_GRAY_SHA256 = (
    "ed3fc0b3284b4951dafb810e8a383e26d543ad1cb8cbfc41484e4f16b847d385"
)
OFFICIAL_GRAY_PARAMETER_COUNT = 468_570

_PROTOCOLS = {
    # This is the protocol in the authors' test_gray_denoising.ipynb.
    "official_notebook": {
        "max_outer_iterations": 1_000,
        "max_inner_iterations": 1_000,
        "inner_tolerance": 1e-6,
        "outer_tolerance": 1e-5,
    },
    # Section 3.3 of the paper uses the stricter inner tolerance below for
    # general inverse-problem inference.
    "paper_conservative": {
        "max_outer_iterations": 1_000,
        "max_inner_iterations": 1_000,
        "inner_tolerance": 1e-8,
        "outer_tolerance": 1e-5,
    },
    # The authors' DEAL.denoise() evaluation branch uses these smaller caps.
    "released_fast": {
        "max_outer_iterations": 60,
        "max_inner_iterations": 200,
        "inner_tolerance": 1e-6,
        "outer_tolerance": 1e-5,
    },
}


def _load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    """Load the explicitly vendored, hash-verified authors' checkpoint."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_GRAY_SHA256:
        raise RuntimeError(
            "The DEAL checkpoint hash does not match the official grayscale "
            f"checkpoint. Expected {OFFICIAL_GRAY_SHA256}, received {digest}."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Official DEAL checkpoint must contain 'state_dict'.")
    return checkpoint


class OfficialDEALDenoiser(nn.Module):
    """Pipeline-native wrapper around the official grayscale DEAL model.

    Parameters are loaded from the authors' ``deal_gray.pth`` checkpoint.  A
    normalized project sigma is converted to the checkpoint's 8-bit convention
    by ``sigma_255 = 255 * sigma``.  The default solver settings reproduce the
    authors' grayscale denoising notebook rather than retraining or tuning.
    """

    requires_noise_level = True
    parameter_names = ("attention_mean", "effective_weight")
    model_name = "deal_official"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        protocol: str = "official_notebook",
        max_outer_iterations: int | None = None,
        max_inner_iterations: int | None = None,
        inner_tolerance: float | None = None,
        outer_tolerance: float | None = None,
        verify_parameter_count: bool = True,
    ) -> None:
        super().__init__()
        if protocol not in _PROTOCOLS:
            raise ValueError(
                f"Unknown DEAL protocol {protocol!r}; choose from "
                f"{sorted(_PROTOCOLS)}."
            )
        defaults = _PROTOCOLS[protocol]
        self.protocol = protocol
        self.max_outer_iterations = int(
            defaults["max_outer_iterations"]
            if max_outer_iterations is None
            else max_outer_iterations
        )
        self.max_inner_iterations = int(
            defaults["max_inner_iterations"]
            if max_inner_iterations is None
            else max_inner_iterations
        )
        self.inner_tolerance = float(
            defaults["inner_tolerance"]
            if inner_tolerance is None
            else inner_tolerance
        )
        self.outer_tolerance = float(
            defaults["outer_tolerance"]
            if outer_tolerance is None
            else outer_tolerance
        )
        if self.max_outer_iterations < 1 or self.max_inner_iterations < 1:
            raise ValueError("DEAL iteration caps must be positive.")
        if self.inner_tolerance <= 0 or self.outer_tolerance <= 0:
            raise ValueError("DEAL stopping tolerances must be positive.")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Official DEAL checkpoint not found: {self.checkpoint_path}"
            )
        checkpoint = _load_trusted_checkpoint(self.checkpoint_path)
        self.checkpoint_epoch = checkpoint.get("epoch")

        self.official_model = DEAL(color=False)
        self.official_model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.official_model.eval()

        parameter_count = sum(
            parameter.numel() for parameter in self.official_model.parameters()
        )
        if verify_parameter_count and parameter_count != OFFICIAL_GRAY_PARAMETER_COUNT:
            raise RuntimeError(
                "Unexpected DEAL parameter count: "
                f"{parameter_count:,} != {OFFICIAL_GRAY_PARAMETER_COUNT:,}."
            )

        # The official spectral-normalization routine uses a complex FFT.
        # Compute it once on CPU so inference also works on Apple MPS, where
        # complex FFT support varies by PyTorch/macOS version.  The exact
        # official routine and learned weights are used; only its device is
        # changed.  DEAL.solve_inverse_problem is therefore reproduced below
        # without its redundant per-image W1.spectral_norm() call.
        with torch.no_grad():
            spectral_norm = self.official_model.W1.spectral_norm(mode="Fourier")
        self.register_buffer(
            "_official_spectral_norm",
            spectral_norm.detach().reshape(()).to(dtype=torch.float32),
        )

        self.last_outer_iterations = 0
        self.last_inner_iterations = 0
        self.last_total_inner_iterations = 0
        self.last_outer_relative_residual = torch.empty(0)
        self.last_sigma_255 = torch.empty(0)
        self.last_lambda = torch.empty(0)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def normalized_sigma_to_255(
        sigma: float | Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        values = torch.as_tensor(sigma, device=device, dtype=dtype)
        if values.numel() == 1:
            values = values.reshape(1, 1).expand(batch_size, 1)
        elif values.shape[0] == batch_size:
            values = values.reshape(batch_size, -1)[:, :1]
        else:
            raise ValueError(
                "Sigma must be scalar or provide one value per batch sample."
            )
        if not torch.isfinite(values).all():
            raise ValueError("Sigma contains a non-finite value.")
        if bool((values < 0).any()) or bool((values > 0.2 + 1e-7).any()):
            raise ValueError(
                "Official DEAL expects this adapter's normalized sigma in "
                "[0, 0.2], corresponding to [0, 51] in 8-bit units."
            )
        return 255.0 * values

    def _sync_spectral_norm(self, reference: Tensor) -> None:
        self.official_model.W1.L = self._official_spectral_norm.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        # M1 is not spectrally normalized in the authors' model, but its
        # implementation stores the constant L=1 as a plain tensor rather than
        # a registered buffer. Move that scalar explicitly for MPS portability.
        self.official_model.M1.L = torch.as_tensor(
            self.official_model.M1.L,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _solve(self, noisy: Tensor, sigma_255: Tensor) -> Tensor:
        model = self.official_model
        self._sync_spectral_norm(noisy)
        with torch.no_grad():
            model.cal_lambda(sigma_255)
            model.cal_scaling(sigma_255)

        current = torch.zeros_like(noisy)
        previous = current.clone()
        total_inner_iterations = 0
        relative_residual = torch.full(
            (noisy.shape[0],),
            float("inf"),
            device=noisy.device,
            dtype=noisy.dtype,
        )

        with torch.no_grad():
            for outer_index in range(self.max_outer_iterations):
                model.cal_mask(current)
                current, inner_index = model.cg(
                    noisy,
                    previous,
                    self.max_inner_iterations,
                    eps=self.inner_tolerance,
                )
                if not torch.isfinite(current).all():
                    raise FloatingPointError(
                        "Official DEAL produced a non-finite reconstruction."
                    )

                difference = torch.linalg.vector_norm(
                    (current - previous).reshape(noisy.shape[0], -1),
                    dim=1,
                )
                denominator = torch.linalg.vector_norm(
                    previous.reshape(noisy.shape[0], -1),
                    dim=1,
                )
                relative_residual = difference / denominator.clamp_min(
                    torch.finfo(noisy.dtype).eps
                )
                previous = current.clone()

                inner_iterations = int(inner_index) + 1
                total_inner_iterations += inner_iterations
                if bool((relative_residual <= self.outer_tolerance).all()):
                    break

            # Store a mask evaluated at the returned reconstruction for
            # interpretable summaries.  This does not alter the reconstruction.
            model.cal_mask(current)

        self.last_outer_iterations = outer_index + 1
        self.last_inner_iterations = inner_iterations
        self.last_total_inner_iterations = total_inner_iterations
        self.last_outer_relative_residual = relative_residual.detach().cpu()
        self.last_sigma_255 = sigma_255.detach().cpu()
        self.last_lambda = model.lmbda.detach().cpu()
        model.number_of_cgs = self.last_outer_iterations
        model.last_cg_iter = inner_iterations - 1
        return current.clamp(0.0, 1.0)

    def attention_summaries(self) -> tuple[Tensor, Tensor]:
        """Return channel-mean attention and a scalar weight summary.

        DEAL has 128 full-resolution masks, not one lambda map.  The second
        output is ``lambda * mean_c(mask_c**2)`` and must be described as an
        effective-weight summary rather than as DEAL's lambda map.
        """
        if not hasattr(self.official_model, "mask"):
            raise RuntimeError("Run DEAL once before requesting its masks.")
        mask = self.official_model.mask
        attention_mean = mask.mean(dim=1, keepdim=True)
        lambda_values = self.official_model.lmbda.reshape(
            mask.shape[0], 1, 1, 1
        )
        effective_weight = lambda_values * mask.square().mean(
            dim=1, keepdim=True
        )
        return attention_mean, effective_weight

    def forward(
        self,
        noisy: Tensor,
        sigma: float | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if noisy.ndim != 4 or noisy.shape[1] != 1:
            raise ValueError(
                "Official grayscale DEAL expects input shaped [B, 1, H, W]."
            )
        if noisy.shape[0] != 1:
            raise ValueError(
                "The authors' released denoising inference is evaluated with "
                "batch size 1; keep the benchmark loader at BATCH_SIZE=1."
            )
        sigma_255 = self.normalized_sigma_to_255(
            sigma,
            batch_size=noisy.shape[0],
            device=noisy.device,
            dtype=noisy.dtype,
        )
        reconstruction = self._solve(noisy, sigma_255)
        attention_mean, effective_weight = self.attention_summaries()
        return reconstruction, attention_mean, effective_weight

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "epoch": self.checkpoint_epoch,
            "source_repository": OFFICIAL_REPOSITORY,
            "source_commit": OFFICIAL_COMMIT,
            "checkpoint_sha256": OFFICIAL_GRAY_SHA256,
            "pretrained_external_reference": True,
            "protocol": self.protocol,
            "args": {
                "max_outer_iterations": self.max_outer_iterations,
                "max_inner_iterations": self.max_inner_iterations,
                "inner_tolerance": self.inner_tolerance,
                "outer_tolerance": self.outer_tolerance,
                "sigma_units": "normalized input converted with sigma_255=255*sigma",
            },
        }


def load_official_deal_model(
    checkpoint_path: str | Path,
    device: str | torch.device,
    *,
    protocol: str = "official_notebook",
    **overrides: Any,
) -> tuple[OfficialDEALDenoiser, dict[str, Any]]:
    """Load the official grayscale checkpoint and move it to ``device``."""
    model = OfficialDEALDenoiser(
        checkpoint_path,
        protocol=protocol,
        **overrides,
    )
    model.to(device).eval()
    return model, model.checkpoint_metadata()


__all__ = [
    "OFFICIAL_COMMIT",
    "OFFICIAL_GRAY_PARAMETER_COUNT",
    "OFFICIAL_GRAY_SHA256",
    "OFFICIAL_REPOSITORY",
    "OfficialDEALDenoiser",
    "load_official_deal_model",
]
