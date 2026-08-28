# Official pretrained DEAL comparison

This integration evaluates the authors' released grayscale DEAL model as an
external state-of-the-art reference. It does not retrain DEAL and must not be
described as a training-controlled fourth baseline.

## Primary experiment

Run `notebooks/07_unified_denoising_benchmark_with_official_deal.ipynb`.
It keeps the complete 39-cell project benchmark, adds a DEAL runtime-pilot
section, and then evaluates every loaded method on the same deterministic
DIV2K and geometric tensors.

The relevant configuration is:

```python
DEAL_METHOD = "DEAL (official pretrained)"
DEAL_PROTOCOL = "official_notebook"
RUN_DEAL_PILOT = True
```

Run the pilot cell before the full benchmark. DEAL performs nested,
convergence-controlled solves and can be much slower than the fixed 64-step
PDHG or fixed 32/64-step CG project models. The pilot prints a crude projected
runtime based on one 256 x 256 image. It does not update any weights.

## Checkpoint and source provenance

- Repository: <https://github.com/mehrsapo/DEAL>
- Vendored commit: `554820ca356ff8a78bc49097e3ffcba3875a3ac2`
- Checkpoint: `checkpoints/deal_official/deal_gray.pth`
- Checkpoint SHA-256:
  `ed3fc0b3284b4951dafb810e8a383e26d543ad1cb8cbfc41484e4f16b847d385`
- Expected grayscale parameter count: 468,570

The adapter verifies the checkpoint hash and parameter count before inference.
It loads the authors' `state_dict` without changing any learned value.

## Which official inference route is used?

The primary `official_notebook` protocol reproduces the procedure in the
authors' `test_gray_denoising.ipynb`:

1. Convert the project's normalized noise level to 8-bit units:

   `sigma_255 = 255 * sigma`.

2. Evaluate the checkpoint's learned noise-to-lambda spline
   `kappa(sigma_255)`; lambda is not tuned on DIV2K.
3. Set `H = Ht = identity` for denoising.
4. Run at most 1,000 outer iterations and 1,000 inner CG iterations.
5. Use the notebook's `eps_in = 1e-6` and `eps_out = 1e-5` stopping values.
6. Clip the returned reconstruction to `[0, 1]`, as the authors' general
   inference routine does.

Two explicitly labelled alternatives are available in the adapter:

| Protocol | Outer cap | Inner cap | Inner tolerance | Use |
| --- | ---: | ---: | ---: | --- |
| `official_notebook` | 1000 | 1000 | `1e-6` | Primary comparison |
| `paper_conservative` | 1000 | 1000 | `1e-8` | Section 3.3 conservative setting |
| `released_fast` | 60 | 200 | `1e-6` | Runtime sensitivity only |

Do not silently report `released_fast` results as the primary official result.

## Pipeline adaptations

The learned DEAL method is unchanged. The integration makes only these
engineering adaptations:

- The official Python files use package-relative imports so they can live under
  `src/third_party/deal_official/` without colliding with the project's modules.
- The official spectral-norm calculation is evaluated once on CPU, using the
  authors' unchanged Fourier routine and checkpoint weights. This avoids
  unsupported complex FFT operations on some Apple MPS versions. The resulting
  scalar is moved with the model and reused for every image.
- The authors' outer loop is reproduced in the adapter instead of calling
  `solve_inverse_problem()` directly. This avoids that method's redundant
  per-image spectral-norm FFT and records outer/inner solver diagnostics. The
  official `cal_mask()` and `cg()` implementations still perform every model
  and linear-solver update.
- After reconstruction, the mask is evaluated once at the returned image for
  visualization. This extra mask evaluation does not alter the reconstruction
  or its metrics.
- The shared evaluator passes sigma only to models declaring
  `requires_noise_level=True`; U-TV, U-TGV, and U-Tikhonov retain their original
  call signatures.

## What the DEAL maps mean

DEAL produces 128 spatial attention masks, one for each learned filter. It does
not produce a single lambda map. The notebook displays:

- mean attention: `mean_c m_c(x)`;
- effective-weight summary: `lambda * mean_c m_c(x)^2`.

The latter is useful for visualization but is not the full regularizer and
must not be called a DEAL lambda map, because each channel also multiplies a
different learned filter response.

## Interpretation and fairness

The test side is controlled: every method receives the same normalized clean
image, deterministic Gaussian tensor, sigma, grayscale preprocessing, metric
implementation, and metric clipping convention.

The training side is not controlled. The project models were trained together
on the 500-image SeaTurtle split, whereas the official DEAL checkpoint used the
authors' substantially larger training corpus and schedule. Moreover, the
paper cites a corpus containing DIV2K images, while the exact public training
manifest is unavailable in this integration. Possible overlap with
`DIV2K_valid_HR` should therefore be disclosed. The defensible thesis wording
is "external pretrained state-of-the-art reference", not "fair baseline".

## Verification

From the cleaned-pipeline root, run:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

The DEAL tensor test loads the hash-verified official checkpoint and runs a
small two-outer-step, three-inner-step smoke solve. The full solver protocol is
then exercised by the notebook pilot.
