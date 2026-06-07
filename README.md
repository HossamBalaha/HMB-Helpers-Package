# HMB Helpers Package

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/hmb-helpers?label=PyPI)](https://pypi.org/project/hmb-helpers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/tests-530%2B%20passing-brightgreen.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/tests/)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/tests/)
[![Examples](https://img.shields.io/badge/examples-ready-brightgreen.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/tree/main/HMB/Examples)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Docs Status](https://img.shields.io/badge/docs-built-green.svg)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Read the Docs](https://readthedocs.org/projects/hmb-helpers-package/badge/?version=latest)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Modules](https://img.shields.io/badge/modules-40%2B-orange.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/tree/main/HMB)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CONTRIBUTING.md)
[![Maintenance](https://img.shields.io/badge/maintained-yes-green.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package)
[![Release](https://img.shields.io/badge/release-v0.2.0-blue.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/releases)
[![DOI](https://zenodo.org/badge/1029954303.svg)](https://doi.org/10.5281/zenodo.20577120)

A comprehensive collection of helper modules for image processing, segmentation, deep learning workflows, text/PDF
utilities, and scientific computing in PyTorch, TensorFlow, and beyond.

Documentation: https://hmb-helpers-package.readthedocs.io/en/latest/

---

## Table of Contents

- [Motivation](#motivation)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Features & Modules](#features--modules)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation & License](#citation--license)
- [Support & Contact](#support--contact)

---

## Motivation

HMB Helpers Package aims to accelerate research and development in computer vision, deep learning, and text analytics by
providing ready-to-use, well-tested utility modules that simplify common tasks, reduce boilerplate code, and promote
reproducibility in scientific projects.

## Installation

### Minimal Install (Recommended)

Install only core dependencies (`numpy`, `pillow`):

```bash
pip install hmb-helpers
```

Package on PyPI: https://pypi.org/project/hmb-helpers/

### Install with Optional Features

Add only the features you need:

```bash
# Computer vision & PyTorch.
pip install "hmb-helpers[cv,pytorch]"

# NLP & text processing.
pip install "hmb-helpers[nlp]"

# PDF handling.
pip install "hmb-helpers[pdf]"

# Full installation (most optional dependencies)
# Note: The `all` extra installs most optional dependencies but intentionally
# excludes large, platform- and device-specific frameworks (PyTorch, TensorFlow,
# Keras and related runtime packages). Install those frameworks separately
# using the dedicated extras below or via the framework's official installer.
pip install "hmb-helpers[all]"
```

### Development Install

For modifying the package source:

```bash
git clone https://github.com/HossamBalaha/HMB-Helpers-Package.git
cd HMB-Helpers-Package
pip install -e ".[dev]"
```

### GPU Support (PyTorch CUDA)

The `pytorch` extra installs CPU wheels by default. For CUDA:

```bash
pip install "hmb-helpers[pytorch]"
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Or use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your platform.

## Dependencies

### Core Dependencies (Always Installed)

- `numpy>=1.26.4,<2`: Numerical computing
- `pillow>=12.2.0`: Image I/O and basic processing
- `pyyaml>=6.0.3`: YAML parsing for configuration and dataset helper utilities
- `pandas>=3.0.2`: Tabular data handling used across multiple helpers
- `matplotlib>=3.9`: Basic plotting utilities used by helpers and examples
- `tqdm>=4.67.3`: Progress bars used in many processing functions
- `scikit-learn>=1.8.0`: Common ML utilities (encoders, imputers, scalers) used by preprocessors

### Optional Feature Groups

Install only what you need via extras:

| Feature              | Command        | Key Packages                              |
|----------------------|----------------|-------------------------------------------|
| **Scientific Stack** | `[scientific]` | scipy, pandas, scikit-learn, scikit-image |
| **Computer Vision**  | `[cv]`         | opencv-python, imagehash, pyvips          |
| **PyTorch**          | `[pytorch]`    | torch, torchvision, torchaudio            |
| **TensorFlow**       | `[tensorflow]` | tensorflow, keras, tf-keras               |
| **NLP**              | `[nlp]`        | nltk, spacy, transformers, gensim         |
| **PDF**              | `[pdf]`        | PyMuPDF, PyPDF2, tabula-py                |
| **Audio**            | `[audio]`      | librosa, spafe, praat-parselmouth         |
| **Medical Imaging**  | `[medical]`    | pydicom, nibabel, openslide-python        |
| **Classical ML**     | `[ml]`         | xgboost, catboost, lightgbm, optuna       |
| **Visualization**    | `[plotting]`   | matplotlib, seaborn, plotly               |
| **Utilities**        | `[utils]`      | tqdm, albumentations, shap, trimesh       |

See `requirements.txt` for exact version pins used in development. Note that
`pip install "hmb-helpers[all]"` purposefully omits PyTorch/TensorFlow/Keras
so that users can install the appropriate platform-specific (CPU/CUDA) wheels
via the framework vendor instructions or by selecting the framework extras
explicitly (for example `pip install "hmb-helpers[pytorch]"`).

## Features & Modules

### Core Modules

[//]: # (Make a numbered list of the modules with brief descriptions. Use the actual module names as they appear in the codebase.)

1. **AgentsHelper**: AI agent orchestration and interaction utilities.
1. **ArabicTextHelper**: Specialized tools for Arabic text processing and analysis.
1. **AttentionMapsHelper**: Tools for generating and visualizing attention maps in deep learning models.
1. **AudioHelper**: Audio processing, feature extraction, and manipulation utilities.
1. **CompressionsHelper**: Data compression and decompression utilities.
1. **DataAugmentationHelper**: Image and data augmentation pipelines.
1. **DatasetsHelper**: Utilities to detect, prepare, and validate image classification datasets (train/val/test
   layouts).
1. **EmbeddingsToTextHelper**: Convert between embeddings and text representations for NLP tasks.
1. **ExplainabilityHelper**: Model explainability and interpretability (e.g., SHAP analysis).
1. **HandCraftedFeatures**: Feature extraction utilities for images and tabular data.
1. **ImagesComparisonMetrics**: Image comparison metrics (SSIM, PSNR, MSE, etc.).
1. **ImageSegmentationMetrics**: Segmentation evaluation metrics (IoU, Dice, pixel accuracy).
1. **ImagesHelper**: Comprehensive image loading, saving, resizing, cropping, and manipulation.
1. **ImagesNormalization**: Image normalization, standardization, and color space conversion.
1. **ImagesToEmbeddings**: Extract embeddings from images using timm and transformers models.
1. **Initializations**: Model and layer initialization helpers for deep learning frameworks.
1. **MachineLearningHelper**: ML workflow helpers (data splitting, cross-validation, model selection).
1. **MetaheuristicsHelper**: Metaheuristic optimization algorithms (e.g., MRFO).
1. **PDFHelper**: PDF reading, extraction, manipulation, and annotation.
1. **PerformanceMetrics**: Comprehensive performance metrics for classification and regression.
1. **PlotsHelper**: Plotting and visualization helpers (wrappers around matplotlib/seaborn utilities).
1. **PyTorchClassificationLosses**: Custom classification loss functions for PyTorch.
1. **PyTorchHelper**: PyTorch utilities for models, tensors, device management, and checkpointing.
1. **PyTorchModelMemoryProfiler**: Utilities to profile PyTorch model memory usage.
1. **PyTorchSegmentationLosses**: Custom segmentation losses (Dice, BCE, DiceBCE, Focal, Tversky, IoU).
1. **PyTorchTabularModelsZoo**: Collection of tabular model utilities and reference architectures.
1. **PyTorchTrainingPipeline**: Training pipeline helpers for PyTorch experiments (data loaders, trainers, schedulers).
1. **PyTorchUNetModelsZoo**: UNet architecture implementations and utilities for PyTorch.
1. **StatisticalAnalysisHelper**: Statistical analysis and data exploration tools.
1. **StringsHelper**: String manipulation and text processing utilities.
1. **TextGenerationMetrics**: Metrics for text generation models (ROUGE, BLEU, METEOR).
1. **TextHelper**: Text normalization, cleaning, tokenization, and NLP utilities.
1. **TFAttentionBlocks**: TensorFlow/Keras attention mechanism implementations.
1. **TFHelper**: TensorFlow/Keras utilities and helpers (Grad-CAM, etc.).
1. **TFSegmentationLosses**: TensorFlow/Keras segmentation loss implementations.
1. **TFUNetHelper**: TensorFlow/Keras UNet-related helpers and utilities.
1. **Utils**: Miscellaneous utilities for file I/O, configuration, and data handling.
1. **VectorsHelper**: Vector operations and geometric computations.
1. **VideosHelper**: Video processing and frame extraction utilities.
1. **VotingHelper**: Ensemble voting methods for machine learning.
1. **WSIHelper**: Whole Slide Image (WSI) processing for digital pathology.
1. **YOLOHelper**: YOLO model training and inference utilities.

## Documentation

Full documentation is available in the `build/html/` directory after building with Sphinx.

On POSIX systems with make available (Linux, macOS):

```bash
cd source
make html
```

## Examples

All example scripts, detailed usage, CLI tables and troubleshooting tips live in the examples README:

- `HMB/Examples/README.md` — beginner-friendly guide with per-script descriptions, example run commands, and
  compact CLI tables (recommended starting point).

Quick pointers:

- Examples and helper scripts are located under `HMB/Examples`.
- Platform wrappers (Windows): `HMB/Examples/BAT Files/` — run with `call` from the repository root.
- Platform wrappers (POSIX): `HMB/Examples/SH Files/` — run with `bash`.

If you want a short list of commonly used examples, open `HMB/Examples/README.md` — it contains a curated list and
copyable run commands for each script.

Or run the equivalent POSIX wrapper (bash):

```bash
bash "HMB/Examples/SH Files/Timm_Statistics_Analysis_Ablations.sh"
```

To run an example Python file directly, quote the path if it contains spaces. Examples:

```bat
python "HMB\Examples\Timm_Statistics_Analysis_Ablations.py"
```

```bash
python3 "HMB/Examples/Timm_Statistics_Analysis_Ablations.py"
```

## Testing

The package includes comprehensive unit tests for all modules. Run tests using
the provided test runner or with pytest directly.

Run the bundled test runner:

```bash
# Run all tests.
python tests/run_tests.py

# Run a specific test file.
python tests/run_tests.py Test_ImagesHelper.py
```

Run tests with pytest (recommended if you have pytest installed):

```bash
pytest -q
# or run a specific test file
pytest -q tests/Test_ImagesHelper.py
```

Test coverage includes:

- Unit tests for all core modules
- Edge case validation
- Integration tests for complex workflows
- Performance and regression tests

Current test status: **530+ tests passing**

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` in the repository root for full contributor
guidelines (development setup, testing, linting, PR checklist):

- https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CONTRIBUTING.md

Quick summary:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -am 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines (short)

- Write comprehensive docstrings for public functions and classes
- Add unit tests for new functionality
- Follow PEP 8 style guidelines and run `black`/`flake8`/`mypy`
- Update documentation when adding or changing features
- Ensure all tests pass before submitting a PR

### Changelog

All notable changes are recorded in `CHANGELOG.md` in the repository root. See:

- https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CHANGELOG.md

## Citation & License

This project is licensed under the MIT License. See the LICENSE file for details.

Full license text is available in the repository: https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/LICENSE

If you use this package in your research, please cite the relevant modules as described in their headers.

### Attribution Requirement

If you use this package in your work, please:

1. Include a copy of the LICENSE file with any distribution
2. Credit the author in documentation or publications:
   ```bibtex
   @software{balaha_hmb_helpers_2026_020,
     author    = {Balaha, Hossam Magdy},
     title     = {HMB-Helpers-Package: HMB-Helpers-Package v0.2.0},
     year      = {2026},
     publisher = {GitHub},
     month     = apr,
     version   = {v0.2.0},
     url       = {https://github.com/HossamBalaha/HMB-Helpers-Package}
   }
 
   @software{hossam_magdy_balaha_2026_20577120,
     author    = {Hossam Magdy Balaha},
     title     = {HossamBalaha/HMB-Helpers-Package: HMB-Helpers-Package v0.2.0},
     month     = june,
     year      = 2026,
     publisher = {Zenodo},
     version   = {v0.2.0},
     doi       = {10.5281/zenodo.20577120},
     url       = {https://doi.org/10.5281/zenodo.20577120},
   }
    ```

## Support & Contact

For questions, bug reports, or contributions, please contact the author:

- Hossam Magdy Balaha
- Email: h3ossam@gmail.com
- Or open an issue on GitHub

---

Happy coding!
