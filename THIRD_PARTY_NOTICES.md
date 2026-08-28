# Third-party notices

## DEAL: Deep Attentive Least Squares

This bundle vendors the inference files and grayscale checkpoint from:

- Mehrsa Pourya, Erich Kobler, Michael Unser, and Sebastian Neumayer,
  "DEALing with Image Reconstruction: Deep Attentive Least Squares", ICML 2025.
- <https://github.com/mehrsapo/DEAL>
- Commit `554820ca356ff8a78bc49097e3ffcba3875a3ac2`

The repository is distributed under the MIT License. A copy is included at
`src/third_party/deal_official/LICENSE`.

The vendored `deal.py`, `model/linearspline.py`, `model/multi_conv.py`, and
`model/spline_autograd_func.py` retain the authors' implementation. Import
statements were made package-relative so they can be loaded from this project.
Pipeline-specific checkpoint loading, normalized noise conversion, device
handling, diagnostics, and visualization summaries are implemented separately
in `src/models/deal_official.py`.
