"""Configurable 2-D U-Net used by all three learned denoisers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with Leaky-ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net whose width and number of downsampling levels are configurable."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels, depth) < 1:
            raise ValueError(
                "in_channels, out_channels, base_channels and depth must be positive."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.enc_blocks = nn.ModuleList()
        current_channels = self.in_channels
        for level in range(self.depth):
            next_channels = self.base_channels * 2**level
            self.enc_blocks.append(ConvBlock(current_channels, next_channels))
            current_channels = next_channels

        bottleneck_channels = self.base_channels * 2**self.depth
        self.bottleneck = ConvBlock(current_channels, bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for level in reversed(range(self.depth)):
            skip_channels = self.base_channels * 2**level
            self.upconvs.append(
                nn.ConvTranspose2d(
                    current_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.dec_blocks.append(ConvBlock(2 * skip_channels, skip_channels))
            current_channels = skip_channels

        self.final = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected (B, {self.in_channels}, H, W), got {tuple(x.shape)}."
            )
        minimum_size = 2**self.depth
        if min(x.shape[-2:]) < minimum_size:
            raise ValueError(
                f"Spatial dimensions must be at least {minimum_size} for depth={self.depth}."
            )

        skips = []
        for encoder in self.enc_blocks:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(
            self.upconvs,
            self.dec_blocks,
            reversed(skips),
        ):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = decoder(torch.cat((x, skip), dim=1))
        return self.final(x)


class UNetSmall(UNet):
    """Backward-compatible name for the former fixed-depth small U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=3,
        )

