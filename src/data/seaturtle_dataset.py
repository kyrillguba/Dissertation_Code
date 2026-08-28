"""SeaTurtleID2022 denoising dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .denoising_dataset import ImageDenoisingDataset


class SeaTurtleDenoisingDataset(ImageDenoisingDataset):
    """SeaTurtle images with the shared grayscale/noise preprocessing."""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        sigma_min: float = 0.0,
        sigma_max: float = 0.2,
        max_images: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """Initialise SeaTurtleID2022 with the shared denoising pipeline."""
        super().__init__(
            root_dir=root_dir,
            image_size=image_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            max_images=max_images,
            seed=seed,
        )

