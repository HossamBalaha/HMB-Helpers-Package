# HMB Helpers Package

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/tests-530%2B%20passing-brightgreen.svg)](tests/)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://www.sphinx-doc.org/)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Modules](https://img.shields.io/badge/modules-33-orange.svg)](HMB/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Maintenance](https://img.shields.io/badge/maintained-yes-green.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/releases)

A comprehensive collection of helper modules for image processing, segmentation, deep learning workflows, text/PDF
utilities, and scientific computing in PyTorch, TensorFlow, and beyond.

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

Install the package directly from source:

```bash
git clone https://github.com/yourusername/HMB-Helpers-Package.git
cd HMB-Helpers-Package
pip install -e .
```

Or install dependencies separately:

```bash
pip install -r requirements.txt
```

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- **torch, torchvision, torchaudio**: Deep learning with PyTorch
- **tensorflow, keras**: Deep learning (alternative to PyTorch)
- **opencv-python, opencv-contrib-python**: Image processing
- **openslide-python**: Whole Slide Image (WSI) support
- **numpy, pandas, scipy, scikit-learn, scikit-image**: Scientific computing and ML
- **matplotlib, seaborn**: Visualization
- **nltk, rouge, textstat, contractions**: NLP and text metrics
- **PyMuPDF (fitz)**: PDF reading and manipulation
- **tqdm**: Progress bars
- **albumentations**: Image augmentation
- **shap**: Model explainability
- **catboost, xgboost, lightgbm**: ML models
- **imblearn**: Imbalanced dataset handling
- **timm, transformers**: Pre-trained models
- **ultralytics**: YOLO models

## Features & Modules

### Core Modules

- **AgentsHelper**: AI agent orchestration and interaction utilities
- **ArabicTextHelper**: Specialized tools for Arabic text processing and analysis
- **AttentionMapsHelper**: Tools for generating and visualizing attention maps in deep learning models
- **AudioHelper**: Audio processing, feature extraction, and manipulation utilities
- **CompressionsHelper**: Data compression and decompression utilities
- **DataAugmentationHelper**: Image and data augmentation pipelines
- **EmbeddingsToTextHelper**: Convert between embeddings and text representations for NLP tasks
- **ExplainabilityHelper**: Model explainability and interpretability (e.g., SHAP analysis)
- **HandCraftedFeatures**: Feature extraction utilities for images and tabular data
- **ImagesComparisonMetrics**: Image comparison metrics (SSIM, PSNR, MSE, etc.)
- **ImageSegmentationMetrics**: Segmentation evaluation metrics (IoU, Dice, pixel accuracy)
- **ImagesHelper**: Comprehensive image loading, saving, resizing, cropping, and manipulation
- **ImagesNormalization**: Image normalization, standardization, and color space conversion
- **ImagesToEmbeddings**: Extract embeddings from images using timm and transformers models
- **Initializations**: Model and layer initialization helpers for deep learning frameworks
- **MachineLearningHelper**: ML workflow helpers (data splitting, cross-validation, model selection)
- **MetaheuristicsHelper**: Metaheuristic optimization algorithms (e.g., MRFO)
- **PDFHelper**: PDF reading, extraction, manipulation, and annotation
- **PerformanceMetrics**: Comprehensive performance metrics for classification and regression
- **PyTorchClassificationLosses**: Custom classification loss functions for PyTorch
- **PyTorchHelper**: PyTorch utilities for models, tensors, device management, and checkpointing
- **PyTorchSegmentationLosses**: Custom segmentation losses (Dice, BCE, DiceBCE, Focal, Tversky, IoU)
- **StatisticalAnalysisHelper**: Statistical analysis and data exploration tools
- **StringsHelper**: String manipulation and text processing utilities
- **TextGenerationMetrics**: Metrics for text generation models (ROUGE, BLEU, METEOR)
- **TextHelper**: Text normalization, cleaning, tokenization, and NLP utilities
- **TFAttentionBlocks**: TensorFlow/Keras attention mechanism implementations
- **TFHelper**: TensorFlow/Keras utilities and helpers (Grad-CAM, etc.)
- **Utils**: Miscellaneous utilities for file I/O, configuration, and data handling
- **VectorsHelper**: Vector operations and geometric computations
- **VideosHelper**: Video processing and frame extraction utilities
- **VotingHelper**: Ensemble voting methods for machine learning
- **WSIHelper**: Whole Slide Image (WSI) processing for digital pathology
- **YOLOHelper**: YOLO model training and inference utilities

## Documentation

Full documentation is available in the `build/html/` directory after building with Sphinx:

```bash
cd source
make html
```

Or view the source documentation files in `source/` directory:

- Each module has a corresponding `.rst` file with detailed API documentation
- Examples and usage patterns are included in module docstrings
- Complete API reference available at `build/html/index.html`

## Testing

The package includes comprehensive unit tests for all modules. Run tests using:

```bash
# Run all tests.
python tests/run_tests.py

# Run specific test file.
python tests/run_tests.py Test_ImagesHelper.py

# Run multiple test files.
python tests/run_tests.py Test_ImagesHelper.py Test_PDFHelper.py Test_TextHelper.py
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

## Citation & License

This project is licensed under the MIT License. See the LICENSE file for details.

If you use this package in your research, please cite the relevant modules as described in their headers.

## Support & Contact

For questions, bug reports, or contributions, please contact the author:

- Hossam Magdy Balaha
- Email: h3ossam@gmail.com
- Or open an issue on GitHub

---

Happy coding!
