# Cleaned U-TV / U-TGV / U-Tikhonov pipeline with official DEAL reference

This version uses one shared experiment pipeline for the three project-trained
variational denoisers and optionally evaluates the authors' official pretrained
grayscale DEAL checkpoint as an external state-of-the-art reference.

## Shared experiment contract

- SeaTurtleID2022 is selected and split once with `seed=42`.
- Training noise is sampled on the fly with one
  `sigma ~ Uniform(sigma_min, sigma_max)` per image.
- Validation uses a fixed `sigma=0.1` and a deterministic per-image noise seed.
- All models use the same configurable U-Net. U-TV/U-TGV use scaled softplus;
  U-Tikhonov supports both the historical scaled-softplus map and a bounded
  sigmoid map initialized exactly at the scalar-reference lambda.
- Training uses raw solver reconstructions and MSE loss.
- Reconstructions are clipped to `[0, 1]` only for metrics and display.
- MSE, PSNR and SSIM are computed per image and then averaged.
- All scripts use AdamW, the same history schema, PSNR model selection, and
  `{model}_latest.pth` / `{model}_best.pth` checkpoint payloads.

## Structure

- `src/data/denoising_dataset.py`: shared loading, stochastic training noise,
  and fixed validation noise.
- `src/training/denoising.py`: shared split, loaders, training, validation,
  history, and checkpoints.
- `src/evaluation/denoising_eval.py`: common learned-model evaluation.
- `src/data/benchmark_dataset.py`: deterministic multi-noise DIV2K/geometric
  benchmark samples, checksums, and optional unreferenced real-noise loading.
- `src/evaluation/benchmark.py`: checkpoint reconstruction, paired statistics,
  win counts, bootstrap confidence intervals, and synchronized runtime timing.
- `src/models/deal_official.py`: hash-verified adapter for the official DEAL
  checkpoint, exact noise-unit conversion, MPS-safe spectral normalization,
  nested solver diagnostics, and attention summaries.
- `src/third_party/deal_official/`: minimally packaged official inference code
  under the authors' MIT licence.
- `src/operators/finite_differences.py`: one finite-difference convention used
  by PDHG and CG.
- `scripts/train_u_*.py`: thin model-specific entry points.
- `notebooks/01_train_u_tv_u_tgv.ipynb`: train U-TV and U-TGV.
- `notebooks/02_test_u_tv_u_tgv_div2k.ipynb`: common DIV2K evaluation and
  qualitative parameter-map figures.
- `notebooks/03_scalar_tikhonov_baseline.ipynb`: scalar-lambda selection and
  fixed-CG iteration study.
- `notebooks/04_train_u_tikhonov.ipynb`: train U-Tikhonov from the selected
  scalar reference.
- `notebooks/05_test_u_tikhonov_div2k.ipynb`: DIV2K evaluation of scalar and
  learned Tikhonov.
- `notebooks/06_train_u_tikhonov_highbound.ipynb`: second bounded
  U-Tikhonov run with `lambda_max=16*reference_lambda`, 64 CG iterations,
  standardized time logging, and a post-training converged-CG audit.
- `notebooks/07_unified_denoising_benchmark.ipynb`: canonical, checksum-verified
  comparison of every learned method on all 100 DIV2K validation images at
  sigma 0.05/0.10/0.15/0.20, geometric phantoms, and optional real-noise
  images.
- `notebooks/07_unified_denoising_benchmark_with_official_deal.ipynb`: separate
  complete benchmark variant that adds the official checkpoint without
  overwriting or removing sections from the existing Notebook 07.

Read `OFFICIAL_DEAL_INFERENCE_GUIDE.md` before running the DEAL variant. Run its
single-image timing pilot first, then continue with the full benchmark if the
projected runtime is practical on the selected device.

Run the notebooks in numerical order. They locate `cleaned_pipeline`
automatically whether Jupyter starts in `Thesis/Python`, `cleaned_pipeline`, or
`cleaned_pipeline/notebooks`. New checkpoints and results stay inside the
cleaned project. DIV2K may remain under the sibling `First Try` folder; the test
notebooks locate `DIV2K_valid_HR` there without importing any old code.

Notebook 07 is the source of truth for thesis comparison tables. Notebooks 02
and 05 remain useful for historical single-run figures, but separate notebooks
should not be used to construct the final cross-method statistics.

The official DEAL result controls only the evaluation inputs and metrics. Its
training data, training budget, optimizer schedule, and capacity are not
matched to U-TV, U-TGV, or U-Tikhonov; label it "official pretrained external
reference", not a controlled baseline.

## Example commands

Run these from the project root:

```bash
python scripts/train_u_tv.py --data-root /path/to/seaturtleid2022
python scripts/train_u_tgv.py --data-root /path/to/seaturtleid2022
python scripts/train_u_tikhonov.py --data-root /path/to/seaturtleid2022
```

For a quick local smoke run, add:

```text
--image-size 128 --max-images 30 --val-images 5 --base-channels 16
--epochs 2
```

and use `--num-pdhg-iters 10` for U-TV/U-TGV or `--cg-iters 10` for
U-Tikhonov.

## Compatibility notes

- U-TV and U-TGV still expose `train(args)` aliases.
- U-Tikhonov still exposes `WeightedTikhonovCGDenoiser`, `lambda_net`, and the
  earlier deterministic-noise evaluation helper.
- Training functions now consistently return `(model, history_dataframe)`.
- `UNetSmall` remains as a lightweight alias rather than a duplicated network.
- `finite_differences_tikhonov.py` remains as a compatibility wrapper; all new
  solver code uses the shared operator module.

Run the numerical checks locally with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Training now writes both `history.csv` and `training_summary.csv`. Timers are
synchronized on CUDA/MPS and separate optimization, validation, whole-epoch,
and cumulative time. Notebook 07 adds a same-device inference benchmark with
fixed batch size, warmups, repeats, and solver-iteration metadata.
