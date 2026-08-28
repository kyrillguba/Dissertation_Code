"""Deterministic datasets for fair denoising comparisons.

The benchmark deliberately separates clean-image selection from noise
generation. Every learned method receives samples with identical tensor
checksums, and the same standard-normal realization is scaled across the
requested Gaussian noise levels for a paired noise-severity experiment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.data.denoising_dataset import (
    DEFAULT_EXTENSIONS,
    _get_clean_item,
    add_gaussian_noise,
)

Tensor = torch.Tensor
Sample = Dict[str, Any]


def tensor_sha256(tensor: Tensor) -> str:
    """Hash a tensor's contiguous CPU values and shape."""
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


class MultiNoiseBenchmarkDataset(Dataset):
    """Apply several deterministic Gaussian noise levels to one clean dataset."""

    def __init__(
        self,
        dataset: Dataset,
        sigmas: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
        seed: int = 20_000,
        dataset_name: str = "benchmark",
        paired_noise_across_sigmas: bool = True,
    ) -> None:
        """Configure deterministic multi-sigma noise generation around a clean dataset.
        """
        super().__init__()
        if not sigmas:
            raise ValueError("sigmas must contain at least one noise level.")
        if any(float(sigma) < 0 for sigma in sigmas):
            raise ValueError("All noise levels must be non-negative.")
        self.dataset = dataset
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.seed = int(seed)
        self.dataset_name = str(dataset_name)
        self.paired_noise_across_sigmas = bool(paired_noise_across_sigmas)

    def __len__(self) -> int:
        """Return the number of clean-image and noise-level combinations."""
        return len(self.dataset) * len(self.sigmas)

    def _indices(self, idx: int) -> tuple[int, int]:
        """Map a flat benchmark index to clean-image and sigma indices."""
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        base_count = len(self.dataset)
        return idx % base_count, idx // base_count

    def __getitem__(self, idx: int) -> Sample:
        """Build one traceable clean/noisy benchmark sample."""
        base_idx, sigma_idx = self._indices(idx)
        sample = _get_clean_item(self.dataset, base_idx)
        clean = sample["clean"]
        sigma = self.sigmas[sigma_idx]
        stable_base_idx = int(sample.get("base_idx", base_idx))

        # Reuse a clean image's noise seed across sigmas for paired severity comparisons.
        noise_seed = self.seed + stable_base_idx
        if not self.paired_noise_across_sigmas:
            noise_seed += sigma_idx * 1_000_000

        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        noisy, sigma_tensor = add_gaussian_noise(
            clean,
            sigma,
            sigma,
            generator=generator,
        )

        path = str(sample.get("path", f"sample_{stable_base_idx:05d}"))
        image_name = str(sample.get("image_name", Path(path).name))
        sigma_label = f"{sigma:.3f}"
        sample_id = f"{self.dataset_name}/{image_name}/sigma_{sigma_label}"

        return {
            **sample,
            "base_idx": stable_base_idx,
            "dataset": self.dataset_name,
            "image_name": image_name,
            "sample_id": sample_id,
            "sigma": sigma_tensor,
            "noise_seed": noise_seed,
            "clean_sha256": tensor_sha256(clean),
            "noisy_sha256": tensor_sha256(noisy),
            "noisy": noisy,
        }


def _pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a grayscale PIL image to a single-channel float tensor in [0, 1]."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _square_gradient(size: int) -> Tensor:
    """Create the square-with-reversed-gradient diagnostic phantom."""
    horizontal = np.linspace(0.05, 0.95, size, dtype=np.float32)
    array = np.repeat(horizontal[None, :], size, axis=0)
    start, stop = size // 4, 3 * size // 4
    reversed_gradient = np.linspace(0.95, 0.05, stop - start, dtype=np.float32)
    array[start:stop, start:stop] = reversed_gradient[None, :]
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _step_edge(size: int) -> Tensor:
    """Create a two-level vertical step-edge phantom."""
    array = np.full((size, size), 0.18, dtype=np.float32)
    array[:, size // 2 :] = 0.82
    return torch.from_numpy(array).unsqueeze(0)


def _draw_shape(size: int, kind: str) -> Tensor:
    """Rasterise one named geometric phantom as a grayscale tensor."""
    image = Image.new("L", (size, size), color=45)
    draw = ImageDraw.Draw(image)
    pad = size // 5
    if kind == "disk":
        draw.ellipse((pad, pad, size - pad, size - pad), fill=210)
    elif kind == "triangle":
        draw.polygon(
            (
                (size // 2, pad),
                (size - pad, size - pad),
                (pad, size - pad),
            ),
            fill=210,
        )
    elif kind == "nested_rectangles":
        width = max(2, size // 64)
        draw.rectangle(
            (pad, pad, size - pad, size - pad),
            outline=220,
            width=width,
        )
        inner = 2 * pad
        draw.rectangle(
            (inner, inner, size - inner, size - inner),
            outline=140,
            width=width,
        )
    else:
        raise ValueError(f"Unknown geometric phantom: {kind}")
    return _pil_to_tensor(image)


class GeometricShapesDataset(Dataset):
    """Small clean phantom set for inspecting edge-dependent parameter maps."""

    names = (
        "square_gradient",
        "step_edge",
        "disk",
        "triangle",
        "nested_rectangles",
    )

    def __init__(self, image_size: int = 256) -> None:
        """Precompute the fixed geometric phantoms at the requested resolution."""
        super().__init__()
        if image_size < 32:
            raise ValueError("image_size must be at least 32.")
        self.image_size = int(image_size)
        self._images = {
            "square_gradient": _square_gradient(self.image_size),
            "step_edge": _step_edge(self.image_size),
            "disk": _draw_shape(self.image_size, "disk"),
            "triangle": _draw_shape(self.image_size, "triangle"),
            "nested_rectangles": _draw_shape(
                self.image_size, "nested_rectangles"
            ),
        }

    def __len__(self) -> int:
        """Return the number of available geometric phantoms."""
        return len(self.names)

    def get_clean_item(self, idx: int) -> Sample:
        """Return a cloned clean phantom with stable identifying metadata."""
        name = self.names[idx]
        return {
            "clean": self._images[name].clone(),
            "path": f"synthetic://{name}",
            "image_name": f"{name}.png",
            "base_idx": int(idx),
        }

    def __getitem__(self, idx: int) -> Sample:
        """Return the indexed clean geometric phantom."""
        return self.get_clean_item(idx)


class UnreferencedImageDataset(Dataset):
    """Load real noisy images when no registered clean target is available."""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        """Discover supported real-noise images and configure square resizing."""
        super().__init__()
        self.root_dir = Path(root_dir).expanduser()
        self.image_size = int(image_size)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Real-noise directory not found: {self.root_dir}")

        # Normalize extension spellings before recursively discovering image files.
        allowed = {
            suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
            for suffix in extensions
        }
        self.image_paths = sorted(
            path
            for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found in {self.root_dir}")

    def __len__(self) -> int:
        """Return the number of discovered real-noise images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        """Load one real-noise image and attach traceability metadata."""
        path = self.image_paths[idx]
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        with Image.open(path) as image:
            tensor = _pil_to_tensor(
                image.convert("L").resize(
                    (self.image_size, self.image_size),
                    resample=resampling,
                )
            )
        return {
            "noisy": tensor,
            "path": str(path),
            "image_name": path.name,
            "sample_id": f"real_noise/{path.name}",
            "dataset": "real_noise",
            "base_idx": int(idx),
            "noisy_sha256": tensor_sha256(tensor),
        }


def benchmark_manifest(dataset: Dataset) -> pd.DataFrame:
    """Materialize the exact sample contract used by every compared method."""
    rows = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        rows.append(
            {
                "benchmark_index": idx,
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "image_name": sample["image_name"],
                "path": sample["path"],
                "base_idx": int(sample["base_idx"]),
                "sigma": float(sample["sigma"].reshape(-1)[0].item()),
                "noise_seed": int(sample["noise_seed"]),
                "clean_sha256": sample["clean_sha256"],
                "noisy_sha256": sample["noisy_sha256"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"]
        raise RuntimeError(f"Duplicate benchmark sample IDs: {duplicates.tolist()}")
    return frame
