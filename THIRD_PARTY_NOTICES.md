# Third-party notices and research-software acknowledgements

This document distinguishes source code that is directly included in this
repository from papers and research repositories that influenced the project's
methodology or software design.

## 1. DEAL: vendored source code and pretrained checkpoint

This repository includes inference files and the official grayscale checkpoint
from:

- Mehrsa Pourya, Erich Kobler, Michael Unser, and Sebastian Neumayer,
  “DEALing with Image Reconstruction: Deep Attentive Least Squares”, 2025.
- Repository: <https://github.com/mehrsapo/DEAL>
- Vendored commit: `554820ca356ff8a78bc49097e3ffcba3875a3ac2`

The upstream DEAL repository is distributed under the MIT License. A copy of
that licence is retained at:

```text
src/third_party/deal_official/LICENSE

```

The directly incorporated implementation files are contained under
`src/third_party/deal_official/`. These include the official `deal.py`,
`model/linearspline.py`, `model/multi_conv.py`, and
`model/spline_autograd_func.py` implementations.

The local packaging changes make the upstream imports package-relative so that
the files can be loaded without colliding with this project's modules. These
changes do not alter the learned checkpoint parameters.

The official checkpoint is stored at
`checkpoints/deal_official/deal_gray.pth`. Its expected SHA-256 hash is
`ed3fc0b3284b4951dafb810e8a383e26d543ad1cb8cbfc41484e4f16b847d385`,
and its expected grayscale parameter count is 468,570.

The adapter in `src/models/deal_official.py` is project-specific integration
code. It performs checkpoint verification, noise-unit conversion, device
handling, solver diagnostics, and attention-map summaries while reproducing
the released DEAL denoising procedure.

DEAL is evaluated as an official pretrained external reference and is not
presented as a training-controlled baseline.

## 2. Deep unrolling for TGV: methodological and software influence

The design of the U-TGV model was informed by:

- Thanh Trung Vu, Andreas Kofler, and Kostas Papafitsoros,
  “Deep Unrolling for Learning Optimal Spatially Varying Regularisation
  Parameters for Total Generalised Variation”, 2025.
- arXiv: https://arxiv.org/abs/2502.16532
- DOI: https://doi.org/10.1007/978-3-031-92366-1_22
- Repository:
  https://github.com/trung-vt/LearningRegularizationParametersForTGV

The upstream repository is licensed under the Apache License 2.0.

This work provided the principal methodological reference for combining a
neural parameter-map predictor with an unrolled TGV reconstruction solver and
for interpreting the two learned TGV parameter maps.

The local U-TGV implementation forms part of this dissertation's shared
PyTorch pipeline. No upstream source files from the Trung-Vu repository are
included in the vendored `src/third_party/` directory. The paper and repository
are therefore acknowledged as methodological and software-design influences
rather than as bundled third-party components.

If any direct source adaptation is identified in a later revision, the
affected local file must be identified explicitly and the relevant Apache-2.0
copyright and licence notices retained.

## 3. Learning regularisation-parameter maps: methodological predecessor

The shared parameter-map and algorithm-unrolling design was also informed by:

- Andreas Kofler, Fabian Altekrüger, Fatima Antarou Ba, Christoph Kolbitsch,
  Evangelos Papoutsellis, David Schote, Clemens Sirotenko,
  Felix Frederik Zimmermann, and Kostas Papafitsoros,
  “Learning Regularization Parameter-Maps for Variational Image Reconstruction
  Using Deep Neural Networks and Algorithm Unrolling”,
  SIAM Journal on Imaging Sciences, 16(4), 2202–2246, 2023.
- DOI: https://doi.org/10.1137/23M1552486
- arXiv: https://arxiv.org/abs/2301.05888
- Repository:
  https://github.com/koflera/LearningRegularizationParameterMaps

The upstream repository is licensed under the Apache License 2.0.

This work is acknowledged as a methodological and software-design predecessor
for predicting spatially adaptive regularisation maps and incorporating them
into differentiable unrolled reconstruction algorithms.

No source files from this repository are included in the vendored
`src/third_party/` directory.

## 4. Datasets

The repository does not redistribute the training or evaluation datasets.

SeaTurtleID2022 is used for training and validation:

- Lukáš Adam, Vojtěch Čermák, Kostas Papafitsoros, and Lukáš Picek,
  “SeaTurtleID2022: A Long-Span Dataset for Reliable Sea Turtle
  Re-Identification”, WACV 2024.
- DOI: https://doi.org/10.1109/WACV57701.2024.00699

DIV2K validation images are used for held-out natural-image evaluation:

- Eirikur Agustsson and Radu Timofte,
  “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study”,
  CVPR Workshops 2017.
- DOI: https://doi.org/10.1109/CVPRW.2017.150

Users must obtain these datasets separately and comply with their respective
terms of use.

## 5. Licence boundaries

The MIT licence retained under `src/third_party/deal_official/` applies to the
vendored DEAL implementation, not automatically to the dissertation's original
code.

The Apache-2.0 licences of the Trung-Vu and Kofler repositories govern their
upstream source repositories. Those repositories are acknowledged here as
methodological and software influences; their licences do not replace the
licence status of independently implemented local files.

No general open-source licence is currently granted for the original
dissertation code.
```

