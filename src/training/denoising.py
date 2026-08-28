"""One training, validation, history and checkpoint pipeline for every model."""

from __future__ import annotations

import argparse
import platform
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.data.denoising_dataset import FixedNoiseDataset
from src.data.seaturtle_dataset import SeaTurtleDenoisingDataset
from src.evaluation.denoising_eval import (
    evaluate_denoising_model,
    summarize_results,
    unpack_model_output,
)

ModelFactory = Callable[[], torch.nn.Module]

COMMON_DEFAULTS = {
    "checkpoint_dir": "checkpoints",
    "image_size": 256,
    "max_images": 550,
    "val_images": 50,
    "sigma_min": 0.0,
    "sigma_max": 0.2,
    "val_sigma": 0.1,
    "val_noise_seed": 10_000,
    "base_channels": 32,
    "depth": 3,
    "epochs": 20,
    "batch_size": 1,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "num_workers": 0,
    "print_every": 50,
    "seed": 42,
    "device": None,
    "clamp_for_metrics": True,
}


def apply_common_defaults(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Optional[str] = None,
) -> argparse.Namespace:
    """Fill fields omitted by an older notebook ``Namespace``."""
    defaults = dict(COMMON_DEFAULTS)
    if checkpoint_dir is not None:
        defaults["checkpoint_dir"] = checkpoint_dir
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA and MPS-compatible PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs from a DataLoader worker's PyTorch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(requested: Optional[str | torch.device] = None) -> torch.device:
    """Resolve an explicit device or choose CUDA, then MPS, then CPU."""
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: str | torch.device) -> None:
    """Wait for queued accelerator work before reading a wall-clock timer."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def describe_device(device: str | torch.device) -> str:
    """Return a stable, human-readable device description for run tables."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        return torch.cuda.get_device_name(resolved)
    if resolved.type == "mps":
        return "Apple Metal Performance Shaders (MPS)"
    return platform.processor() or platform.machine() or "CPU"


def _validate_args(args: argparse.Namespace) -> None:
    """Validate the common denoising experiment arguments."""
    if not hasattr(args, "data_root"):
        raise ValueError("args.data_root is required.")
    if not 0 <= args.sigma_min <= args.sigma_max:
        raise ValueError("Require 0 <= sigma_min <= sigma_max.")
    if args.val_sigma < 0:
        raise ValueError("val_sigma must be non-negative.")
    for name in ("image_size", "val_images", "base_channels", "depth", "epochs", "batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.max_images is not None and int(args.max_images) <= 0:
        raise ValueError("max_images must be positive or None.")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay non-negative.")


def build_seaturtle_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    """Build the shared split, stochastic training data and fixed validation data."""
    _validate_args(args)
    dataset = SeaTurtleDenoisingDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        max_images=args.max_images,
        seed=args.seed,
    )
    if args.val_images >= len(dataset):
        raise ValueError(
            f"val_images={args.val_images} must be smaller than dataset size {len(dataset)}."
        )

    train_size = len(dataset) - args.val_images
    # Use the same seeded split for every model configuration.
    train_subset, val_subset = random_split(
        dataset,
        (train_size, args.val_images),
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_dataset = FixedNoiseDataset(
        val_subset,
        sigma=args.val_sigma,
        seed=args.val_noise_seed,
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": _seed_worker if args.num_workers > 0 else None,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_dir: str,
) -> argparse.ArgumentParser:
    """Add the exactly shared command-line interface to a model script."""
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=checkpoint_dir)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=550)
    parser.add_argument("--val-images", type=int, default=50)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--sigma-max", type=float, default=0.2)
    parser.add_argument("--val-sigma", type=float, default=0.1)
    parser.add_argument("--val-noise-seed", type=int, default=10_000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-clamp-for-metrics",
        dest="clamp_for_metrics",
        action="store_false",
        help="Score raw reconstructions instead of clipping to [0, 1].",
    )
    parser.set_defaults(clamp_for_metrics=True)
    return parser


def _plain_args(args: argparse.Namespace) -> dict:
    """Convert path and device arguments to checkpoint-serializable values."""
    return {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
    }


def train_denoising_model(
    args: argparse.Namespace,
    *,
    model_factory: ModelFactory,
    model_name: str,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Train any project model through one controlled experiment pipeline."""
    apply_common_defaults(args)
    _validate_args(args)
    if (train_loader is None) != (val_loader is None):
        raise ValueError("Provide both train_loader and val_loader, or neither.")

    seed_everything(args.seed)
    device = get_device(args.device)
    if train_loader is None:
        train_loader, val_loader = build_seaturtle_loaders(args)

    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Keep latest and best checkpoints separate from the append-only epoch history.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.csv"
    latest_path = checkpoint_dir / f"{model_name}_latest.pth"
    best_path = checkpoint_dir / f"{model_name}_best.pth"

    history: list[dict] = []
    best_val_psnr = -float("inf")
    best_epoch = 0
    synchronize_device(device)
    training_start = time.perf_counter()
    parameter_names = tuple(getattr(model, "parameter_names", ()))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Using device: {device}")
    print(
        f"Model: {model_name} | train images: {len(train_loader.dataset)} | "
        f"validation images: {len(val_loader.dataset)}"
    )

    # Time training and validation separately after synchronizing accelerator work.
    for epoch in range(1, args.epochs + 1):
        synchronize_device(device)
        epoch_start = time.perf_counter()
        train_phase_start = epoch_start
        model.train()
        loss_sum = 0.0
        sample_count = 0

        for step, batch in enumerate(train_loader, start=1):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            reconstruction, _ = unpack_model_output(model(noisy))
            loss = F.mse_loss(reconstruction, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = clean.shape[0]
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            if args.print_every > 0 and step % args.print_every == 0:
                print(
                    f"Epoch {epoch:03d} | batch {step:04d}/{len(train_loader):04d} "
                    f"| train MSE {loss.item():.6e}"
                )

        synchronize_device(device)
        train_phase_time_sec = time.perf_counter() - train_phase_start
        if sample_count == 0:
            raise ValueError("Training loader is empty.")
        train_mse = loss_sum / sample_count
        synchronize_device(device)
        validation_start = time.perf_counter()
        val_table = evaluate_denoising_model(
            model,
            val_loader,
            device,
            model_type=model_name,
            clamp_for_metrics=args.clamp_for_metrics,
        )
        synchronize_device(device)
        validation_time_sec = time.perf_counter() - validation_start
        val = summarize_results(val_table)
        synchronize_device(device)
        epoch_time_sec = time.perf_counter() - epoch_start
        total_time_sec = time.perf_counter() - training_start

        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val["recon_mse"],
            "val_psnr": val["recon_psnr"],
            "val_ssim": val["recon_ssim"],
            "noisy_mse": val["noisy_mse"],
            "noisy_psnr": val["noisy_psnr"],
            "noisy_ssim": val["noisy_ssim"],
            "train_phase_time_sec": train_phase_time_sec,
            "validation_time_sec": validation_time_sec,
            "epoch_time_sec": epoch_time_sec,
            "total_time_sec": total_time_sec,
            "train_images_per_sec": sample_count / train_phase_time_sec,
            "validation_images_per_sec": (
                len(val_loader.dataset) / validation_time_sec
            ),
        }
        for key, value in val.items():
            if key not in row and key not in {"recon_mse", "recon_psnr", "recon_ssim"}:
                row[f"val_{key}"] = value
        history.append(row)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(history_path, index=False)

        payload = {
            "epoch": epoch,
            "model_name": model_name,
            "resolved_device": str(device),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": _plain_args(args),
            "val_metrics": val,
            "parameter_names": parameter_names,
            "history": history,
        }
        torch.save(payload, latest_path)
        if val["recon_psnr"] > best_val_psnr:
            best_val_psnr = val["recon_psnr"]
            best_epoch = epoch
            torch.save(payload, best_path)
            best_marker = " | saved best"
        else:
            best_marker = ""

        print(
            f"Epoch {epoch:03d} | train MSE {train_mse:.6e} | "
            f"val MSE {val['recon_mse']:.6e} | val PSNR {val['recon_psnr']:.3f} | "
            f"val SSIM {val['recon_ssim']:.4f} | noisy PSNR {val['noisy_psnr']:.3f}"
            f"{best_marker}"
        )

    history_frame = pd.DataFrame(history)
    timing_summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "device": str(device),
                "device_description": describe_device(device),
                "torch_version": torch.__version__,
                "parameter_count": parameter_count,
                "epochs": len(history_frame),
                "training_images": len(train_loader.dataset),
                "validation_images": len(val_loader.dataset),
                "best_epoch": best_epoch,
                "best_val_psnr": best_val_psnr,
                "total_time_sec": float(history_frame["total_time_sec"].iloc[-1]),
                "mean_epoch_time_sec": float(
                    history_frame["epoch_time_sec"].mean()
                ),
                "mean_train_phase_time_sec": float(
                    history_frame["train_phase_time_sec"].mean()
                ),
                "mean_validation_time_sec": float(
                    history_frame["validation_time_sec"].mean()
                ),
                "mean_train_images_per_sec": float(
                    history_frame["train_images_per_sec"].mean()
                ),
                "solver_iterations": getattr(
                    args,
                    "cg_iters",
                    getattr(args, "num_pdhg_iters", None),
                ),
            }
        ]
    )
    timing_summary.to_csv(
        checkpoint_dir / "training_summary.csv",
        index=False,
    )
    return model, history_frame
