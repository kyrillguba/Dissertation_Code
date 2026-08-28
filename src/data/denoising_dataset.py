"""Shared image loading and Gaussian-noise datasets.

Training samples use fresh Gaussian noise every time they are read. Validation
samples are wrapped in :class:`FixedNoiseDataset`, which generates noise from a
per-image local random generator and therefore does not perturb the training
random-number stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

Tensor = torch.Tensor
Sample = Dict[str, Any]

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _validate_noise_range(sigma_min: float, sigma_max: float) -> None:
    """Validate an inclusive non-negative Gaussian-noise interval."""
    if not 0.0 <= sigma_min <= sigma_max:
        raise ValueError(
            "Noise levels must satisfy 0 <= sigma_min <= sigma_max, "
            f"got {sigma_min} and {sigma_max}."
        )


def add_gaussian_noise(
    clean: Tensor,
    sigma_min: float,
    sigma_max: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Add clipped Gaussian noise with one sampled standard deviation.

    This function operates on one unbatched image of shape ``(C, H, W)`` and
    returns ``sigma`` with shape ``(1,)`` so the default DataLoader collator
    produces a batch tensor of shape ``(B, 1)``.
    """
    _validate_noise_range(sigma_min, sigma_max)
    if clean.ndim != 3:
        raise ValueError(
            f"Expected an image with shape (C, H, W), got {tuple(clean.shape)}."
        )

    sigma = torch.empty(1, dtype=clean.dtype, device="cpu").uniform_(
        sigma_min,
        sigma_max,
        generator=generator,
    )
    noise = torch.randn(
        clean.shape,
        dtype=clean.dtype,
        device="cpu",
        generator=generator,
    )
    noisy = clean + sigma.to(clean.device) * noise.to(clean.device)
    return noisy.clamp(0.0, 1.0), sigma.to(clean.device)


class ImageDenoisingDataset(Dataset):
    """Load grayscale images and add stochastic Gaussian training noise.

    All image datasets in this project use this class so they expose the same
    sample contract: ``clean``, ``noisy``, ``sigma``, ``path`` and ``base_idx``.
    Images are resized to a square without changing the historic preprocessing
    convention used by the existing experiments.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        sigma_min: float = 0.0,
        sigma_max: float = 0.2,
        max_images: Optional[int] = None,
        seed: int = 42,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        """Discover images and configure reproducible selection and noise settings."""
        super().__init__()
        _validate_noise_range(sigma_min, sigma_max)
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if max_images is not None and max_images <= 0:
            raise ValueError("max_images must be positive when provided.")

        self.root_dir = Path(root_dir).expanduser()
        self.image_size = int(image_size)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.seed = int(seed)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        allowed = {
            suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
            for suffix in extensions
        }
        image_paths = sorted(
            path
            for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )

        if not image_paths:
            raise RuntimeError(f"No supported images found in {self.root_dir}")

        if max_images is not None and max_images < len(image_paths):
            generator = torch.Generator().manual_seed(self.seed)
            # Select a fixed random subset without disturbing the global random state.
            selected = torch.randperm(len(image_paths), generator=generator)[:max_images]
            image_paths = [image_paths[index] for index in selected.tolist()]

        self.image_paths = image_paths

    def __len__(self) -> int:
        """Return the number of selected source images."""
        return len(self.image_paths)

    def get_clean_item(self, idx: int) -> Sample:
        """Load a clean image without sampling noise or consuming global RNG."""
        path = self.image_paths[idx]
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.image_size, self.image_size),
                resample=resampling,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0

        clean = torch.from_numpy(array.copy()).unsqueeze(0)
        return {
            "clean": clean,
            "path": str(path),
            "base_idx": int(idx),
        }

    def __getitem__(self, idx: int) -> Sample:
        """Load a clean image and add freshly sampled Gaussian noise."""
        sample = self.get_clean_item(idx)
        noisy, sigma = add_gaussian_noise(
            sample["clean"],
            self.sigma_min,
            self.sigma_max,
        )
        return {**sample, "noisy": noisy, "sigma": sigma}


def _get_clean_item(dataset: Dataset, idx: int) -> Sample:
    """Resolve nested Subsets without asking the base dataset to sample noise."""
    if isinstance(dataset, Subset):
        return _get_clean_item(dataset.dataset, int(dataset.indices[idx]))

    get_clean_item = getattr(dataset, "get_clean_item", None)
    if callable(get_clean_item):
        return get_clean_item(idx)

    sample = dataset[idx]
    if not isinstance(sample, dict) or "clean" not in sample:
        raise TypeError(
            "FixedNoiseDataset requires dictionary samples containing 'clean'."
        )
    return {key: value for key, value in sample.items() if key != "noisy"}


class FixedNoiseDataset(Dataset):
    """Apply a fixed noise level and pattern to every validation image."""

    def __init__(
        self,
        dataset: Dataset,
        sigma: float = 0.1,
        seed: int = 10_000,
    ) -> None:
        """Configure a fixed noise level and per-image validation seed."""
        super().__init__()
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        self.dataset = dataset
        self.sigma = float(sigma)
        self.seed = int(seed)

    def __len__(self) -> int:
        """Return the number of wrapped validation samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Sample:
        """Apply the deterministic validation-noise realisation for one image."""
        sample = _get_clean_item(self.dataset, idx)
        clean = sample["clean"]
        base_idx = int(sample.get("base_idx", idx))

        generator = torch.Generator(device="cpu")
        # Use a local per-image generator so validation noise is repeatable.
        generator.manual_seed(self.seed + base_idx)
        noisy, sigma = add_gaussian_noise(
            clean,
            self.sigma,
            self.sigma,
            generator=generator,
        )
        return {**sample, "noisy": noisy, "sigma": sigma}

