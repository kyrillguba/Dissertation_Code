"""Numerical and interface checks for the cleaned denoising pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Subset

from scripts.train_u_tikhonov import initialize_lambda_output
from src.data.benchmark_dataset import (
    GeometricShapesDataset,
    MultiNoiseBenchmarkDataset,
    benchmark_manifest,
)
from src.data.denoising_dataset import FixedNoiseDataset
from src.evaluation.benchmark import (
    best_method_counts,
    paired_difference_summary,
)
from src.evaluation.denoising_eval import call_denoising_model
from src.data.seaturtle_dataset import SeaTurtleDenoisingDataset
from src.models.cg_tikhonov import (
    UTikhonovModel,
    scalar_tikhonov_operator,
    weighted_tikhonov_operator,
)
from src.models.deal_official import (
    OFFICIAL_GRAY_PARAMETER_COUNT,
    OfficialDEALDenoiser,
)
from src.models.u_tgv import UTGVModel
from src.models.u_tv import UTVModel
from src.operators.finite_differences import (
    gradient,
    gradient_adjoint,
    sym_gradient,
    sym_gradient_adjoint,
)
from src.solvers.conjugate_gradient import cg_solve_until_converged


class OperatorTests(unittest.TestCase):
    def test_gradient_adjoint(self) -> None:
        torch.manual_seed(0)
        u = torch.randn(2, 1, 13, 17, dtype=torch.float64)
        p = torch.randn(2, 2, 13, 17, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(
                torch.sum(gradient(u) * p),
                torch.sum(u * gradient_adjoint(p)),
                atol=1e-10,
                rtol=1e-10,
            )
        )

    def test_symmetric_gradient_adjoint(self) -> None:
        torch.manual_seed(1)
        w = torch.randn(2, 2, 13, 17, dtype=torch.float64)
        q = torch.randn(2, 3, 13, 17, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(
                torch.sum(sym_gradient(w) * q),
                torch.sum(w * sym_gradient_adjoint(q)),
                atol=1e-10,
                rtol=1e-10,
            )
        )

    def test_weighted_tikhonov_is_positive_definite(self) -> None:
        torch.manual_seed(2)
        u = torch.randn(2, 1, 13, 17, dtype=torch.float64)
        lam = torch.rand(2, 1, 13, 17, dtype=torch.float64)
        quadratic_form = torch.sum(u * weighted_tikhonov_operator(u, lam))
        self.assertGreater(float(quadratic_form), 0.0)


class ConjugateGradientTests(unittest.TestCase):
    def test_scalar_tikhonov_residual(self) -> None:
        torch.manual_seed(3)
        b = torch.randn(2, 1, 16, 16, dtype=torch.float64)
        operator = lambda value: scalar_tikhonov_operator(value, 0.5)
        solution, info = cg_solve_until_converged(
            operator,
            b,
            x0=b,
            max_iter=100,
            tol=1e-10,
        )
        residual = torch.linalg.vector_norm((operator(solution) - b).reshape(2, -1), dim=1)
        relative = residual / torch.linalg.vector_norm(b.reshape(2, -1), dim=1)
        self.assertTrue(torch.all(relative < 1e-9))
        self.assertTrue(info["converged"])


class DatasetTests(unittest.TestCase):
    def test_stochastic_train_and_fixed_validation_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                array = np.full((20, 24), 80 + 20 * index, dtype=np.uint8)
                Image.fromarray(array).save(root / f"image_{index}.png")

            dataset = SeaTurtleDenoisingDataset(
                root,
                image_size=16,
                sigma_min=0.1,
                sigma_max=0.1,
                seed=42,
            )
            first = dataset[0]["noisy"]
            second = dataset[0]["noisy"]
            self.assertFalse(torch.equal(first, second))

            fixed = FixedNoiseDataset(Subset(dataset, [0, 1]), sigma=0.1, seed=10_000)
            fixed_first = fixed[0]
            fixed_second = fixed[0]
            self.assertTrue(torch.equal(fixed_first["noisy"], fixed_second["noisy"]))
            self.assertAlmostEqual(float(fixed_first["sigma"]), 0.1, places=6)

    def test_multi_noise_benchmark_is_deterministic_and_identified(self) -> None:
        geometry = GeometricShapesDataset(image_size=32)
        benchmark = MultiNoiseBenchmarkDataset(
            geometry,
            sigmas=(0.05, 0.10),
            seed=20_000,
            dataset_name="geometry",
        )
        first = benchmark[0]
        repeated = benchmark[0]
        second_sigma = benchmark[len(geometry)]
        self.assertTrue(torch.equal(first["noisy"], repeated["noisy"]))
        self.assertEqual(first["noise_seed"], second_sigma["noise_seed"])
        self.assertNotEqual(first["sample_id"], second_sigma["sample_id"])

        manifest = benchmark_manifest(benchmark)
        self.assertEqual(len(manifest), 2 * len(geometry))
        self.assertFalse(manifest["sample_id"].duplicated().any())
        self.assertFalse(manifest["noisy_sha256"].isna().any())


class ModelInterfaceTests(unittest.TestCase):
    def test_noise_aware_dispatch_preserves_normalized_sigma(self) -> None:
        class NoiseAwareEcho(torch.nn.Module):
            requires_noise_level = True

            def forward(self, noisy, sigma):
                self.received_sigma = torch.as_tensor(sigma).clone()
                return noisy

        model = NoiseAwareEcho()
        noisy = torch.rand(1, 1, 16, 16)
        output = call_denoising_model(model, noisy, torch.tensor([0.1]))
        self.assertTrue(torch.equal(output, noisy))
        self.assertAlmostEqual(
            float(model.received_sigma.reshape(-1)[0]),
            0.1,
            places=6,
        )

    def test_official_deal_checkpoint_and_small_forward(self) -> None:
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "checkpoints"
            / "deal_official"
            / "deal_gray.pth"
        )
        model = OfficialDEALDenoiser(
            checkpoint,
            protocol="official_notebook",
            max_outer_iterations=2,
            max_inner_iterations=3,
        )
        self.assertEqual(model.parameter_count, OFFICIAL_GRAY_PARAMETER_COUNT)

        noisy = torch.rand(1, 1, 16, 16)
        reconstruction, attention, effective = model(
            noisy,
            torch.tensor([0.1]),
        )
        self.assertEqual(reconstruction.shape, noisy.shape)
        self.assertEqual(attention.shape, noisy.shape)
        self.assertEqual(effective.shape, noisy.shape)
        self.assertAlmostEqual(float(model.last_sigma_255[0, 0]), 25.5)
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertTrue(torch.all((attention >= 0.01) & (attention <= 1.0)))

    def test_bounded_tikhonov_initialization_and_bounds(self) -> None:
        reference = 2.0
        model = UTikhonovModel(
            base_channels=2,
            depth=1,
            cg_iters=2,
            parameterization="bounded_sigmoid",
            lambda_min=1e-4,
            lambda_max=16.0,
        )
        initialize_lambda_output(model, reference)
        noisy = torch.rand(1, 1, 16, 16)
        lambda_map = model.predict_lambda(noisy)
        self.assertTrue(
            torch.allclose(
                lambda_map,
                torch.full_like(lambda_map, reference),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertGreaterEqual(float(lambda_map.min()), model.lambda_min)
        self.assertLessEqual(float(lambda_map.max()), model.lambda_max)

    def test_all_models_backpropagate(self) -> None:
        torch.manual_seed(4)
        noisy = torch.rand(1, 1, 16, 16)
        models = (
            (UTVModel(base_channels=2, depth=1, num_pdhg_iters=2), 2),
            (UTGVModel(base_channels=2, depth=1, num_pdhg_iters=2), 3),
            (
                UTikhonovModel(
                    base_channels=2,
                    depth=1,
                    cg_iters=2,
                    map_scale=1.0,
                ),
                2,
            ),
        )
        for model, output_count in models:
            output = model(noisy)
            self.assertEqual(len(output), output_count)
            self.assertEqual(output[0].shape, noisy.shape)
            output[0].square().mean().backward()
            gradients = [parameter.grad for parameter in model.parameters()]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(
                all(
                    gradient is None or torch.isfinite(gradient).all()
                    for gradient in gradients
                )
            )


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_paired_differences_and_win_counts(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sample_id": sample,
                    "dataset": "toy",
                    "image_name": f"{sample}.png",
                    "sigma": 0.1,
                    "method": method,
                    "recon_mse": mse,
                    "recon_psnr": psnr,
                    "recon_ssim": ssim,
                }
                for sample, values in {
                    "a": {"A": (0.1, 20.0, 0.7), "B": (0.2, 19.0, 0.6)},
                    "b": {"A": (0.3, 18.0, 0.5), "B": (0.2, 19.0, 0.6)},
                }.items()
                for method, (mse, psnr, ssim) in values.items()
            ]
        )
        _, summary = paired_difference_summary(
            frame,
            [("A", "B")],
            bootstrap_samples=100,
        )
        psnr = summary.loc[summary["metric"] == "PSNR"].iloc[0]
        self.assertAlmostEqual(float(psnr["mean_difference"]), 0.0)
        self.assertEqual(int(psnr["wins"]), 1)
        self.assertEqual(int(psnr["losses"]), 1)

        wins = best_method_counts(frame, ("A", "B"), "recon_psnr")
        self.assertEqual(int(wins["exclusive_best_count"].sum()), 2)


if __name__ == "__main__":
    unittest.main()
