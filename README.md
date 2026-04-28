# HMB Helpers Package

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/tests-530%2B%20passing-brightgreen.svg)](tests/)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Read the Docs](https://readthedocs.org/projects/hmb-helpers-package/badge/?version=latest)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Modules](https://img.shields.io/badge/modules-40%2B-orange.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/tree/main/HMB)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CONTRIBUTING.md)
[![Maintenance](https://img.shields.io/badge/maintained-yes-green.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/releases)
[![DOI](https://zenodo.org/badge/1029954303.svg)](https://doi.org/10.5281/zenodo.19860289)

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

### Install with Optional Features

Add only the features you need:

```bash
# Computer vision & PyTorch
pip install "hmb-helpers[cv,pytorch]"

# NLP & text processing
pip install "hmb-helpers[nlp]"

# PDF handling
pip install "hmb-helpers[pdf]"

# Full installation (all optional dependencies)
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

- `numpy<2`: Numerical computing
- `pillow>=9.0.0`: Image I/O and basic processing

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

See `requirements.txt` for exact version pins used in development.

## Features & Modules

### Core Modules

- **AgentsHelper**: AI agent orchestration and interaction utilities.
- **ArabicTextHelper**: Specialized tools for Arabic text processing and analysis.
- **AttentionMapsHelper**: Tools for generating and visualizing attention maps in deep learning models.
- **AudioHelper**: Audio processing, feature extraction, and manipulation utilities.
- **CompressionsHelper**: Data compression and decompression utilities.
- **DataAugmentationHelper**: Image and data augmentation pipelines.
- **DatasetsHelper**: Utilities to detect, prepare, and validate image classification datasets (train/val/test layouts).
- **EmbeddingsToTextHelper**: Convert between embeddings and text representations for NLP tasks.
- **ExplainabilityHelper**: Model explainability and interpretability (e.g., SHAP analysis).
- **HandCraftedFeatures**: Feature extraction utilities for images and tabular data.
- **ImagesComparisonMetrics**: Image comparison metrics (SSIM, PSNR, MSE, etc.).
- **ImageSegmentationMetrics**: Segmentation evaluation metrics (IoU, Dice, pixel accuracy).
- **ImagesHelper**: Comprehensive image loading, saving, resizing, cropping, and manipulation.
- **ImagesNormalization**: Image normalization, standardization, and color space conversion.
- **ImagesToEmbeddings**: Extract embeddings from images using timm and transformers models.
- **Initializations**: Model and layer initialization helpers for deep learning frameworks.
- **MachineLearningHelper**: ML workflow helpers (data splitting, cross-validation, model selection).
- **MetaheuristicsHelper**: Metaheuristic optimization algorithms (e.g., MRFO).
- **PDFHelper**: PDF reading, extraction, manipulation, and annotation.
- **PerformanceMetrics**: Comprehensive performance metrics for classification and regression.
- **PlotsHelper**: Plotting and visualization helpers (wrappers around matplotlib/seaborn utilities).
- **PyTorchClassificationLosses**: Custom classification loss functions for PyTorch.
- **PyTorchHelper**: PyTorch utilities for models, tensors, device management, and checkpointing.
- **PyTorchModelMemoryProfiler**: Utilities to profile PyTorch model memory usage.
- **PyTorchSegmentationLosses**: Custom segmentation losses (Dice, BCE, DiceBCE, Focal, Tversky, IoU).
- **PyTorchTabularModelsZoo**: Collection of tabular model utilities and reference architectures.
- **PyTorchTrainingPipeline**: Training pipeline helpers for PyTorch experiments (data loaders, trainers, schedulers).
- **PyTorchUNetModelsZoo**: UNet architecture implementations and utilities for PyTorch.
- **StatisticalAnalysisHelper**: Statistical analysis and data exploration tools.
- **StringsHelper**: String manipulation and text processing utilities.
- **TextGenerationMetrics**: Metrics for text generation models (ROUGE, BLEU, METEOR).
- **TextHelper**: Text normalization, cleaning, tokenization, and NLP utilities.
- **TFAttentionBlocks**: TensorFlow/Keras attention mechanism implementations.
- **TFHelper**: TensorFlow/Keras utilities and helpers (Grad-CAM, etc.).
- **TFSegmentationLosses**: TensorFlow/Keras segmentation loss implementations.
- **TFUNetHelper**: TensorFlow/Keras UNet-related helpers and utilities.
- **Utils**: Miscellaneous utilities for file I/O, configuration, and data handling.
- **VectorsHelper**: Vector operations and geometric computations.
- **VideosHelper**: Video processing and frame extraction utilities.
- **VotingHelper**: Ensemble voting methods for machine learning.
- **WSIHelper**: Whole Slide Image (WSI) processing for digital pathology.
- **YOLOHelper**: YOLO model training and inference utilities.

## Documentation

Full documentation is available in the `build/html/` directory after building with Sphinx.

On POSIX systems with make available (Linux, macOS):

```bash
cd source
make html
```

On Windows (cmd.exe) there is a helper `make.bat` at the project root which wraps `sphinx-build`:

```bat
call make.bat html
```

Or view the source documentation files in the `source/` directory:

- Each module has a corresponding `.rst` file with detailed API documentation
- Examples and usage patterns are included in module docstrings
- Complete API reference available at `build/html/index.html`

Examples
--------

The repository includes several example Python scripts under `HMB/Examples` and
platform-specific wrapper scripts grouped under `HMB/Examples/BAT Files` and
`HMB/Examples/SH Files`. There is no single top-level `run_examples` script; use
the per-example wrapper that matches your platform.

Common examples (see `HMB/Examples` for the full list):

- `PyTorch_UNet_Segmentation.py`
- `TF_UNet_Training.py`
- `TF_UNet_EvalPredict.py`
- `Timm_FineTune_Classification.py`
- `Timm_Statistics_Analysis_Ablations.py`
- `Train TF Pretrained Attention Model from DataFrame.py`
- `Explain TF Pretrained Attention Model from DataFrame.py`
- `Machine_Learning_Pipeline.py`

Run a Windows wrapper from the repository root (cmd.exe), for example:

```bat
call "HMB\Examples\BAT Files\Timm_Statistics_Analysis_Ablations.bat"
```

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

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -am 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write comprehensive docstrings for all functions and classes
- Add unit tests for new functionality
- Follow PEP 8 style guidelines
- Update documentation when adding new features
- Ensure all tests pass before submitting PR

See `CONTRIBUTING.md` in the repository root for more detailed contribution
guidelines (if present): https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CONTRIBUTING.md

## Citation & License

This project is licensed under the MIT License. See the LICENSE file for details.

Full license text is available in the repository: https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/LICENSE

If you use this package in your research, please cite the relevant modules as described in their headers.

### Attribution Requirement

If you use this package in your work, please:

1. Include a copy of the LICENSE file with any distribution
2. Credit the author in documentation or publications:
   ```bibtex
   @software{balaha_hmb_helpers_2026_010,
     author = {Balaha, Hossam Magdy},
     title = {HMB-Helpers-Package},
     year = {2026},
     version = {0.1.0},
     url = {https://github.com/HossamBalaha/HMB-Helpers-Package},
     doi = {10.5281/zenodo.19860290}
   }

## Support & Contact

For questions, bug reports, or contributions, please contact the author:

- Hossam Magdy Balaha
- Email: h3ossam@gmail.com
- Or open an issue on GitHub

---

Happy coding!
