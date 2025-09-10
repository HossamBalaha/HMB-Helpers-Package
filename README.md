# HMB Helpers Package

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive collection of helper modules for image processing, segmentation, deep learning workflows, and text/PDF
utilities in PyTorch and beyond.

---

## Table of Contents

- [Motivation](#motivation)
- [Dependencies](#dependencies)
- [Features & Modules](#features--modules)
- [Module Usage Table](#module-usage-table)
- [Quick Start](#quick-start)
- [More Usage Examples](#more-usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Changelog / Release Notes](#changelog--release-notes)
- [Citation & License](#citation--license)
- [Support & Contact](#support--contact)

---

## Motivation

HMB Helpers Package aims to accelerate research and development in computer vision, deep learning, and text analytics by
providing ready-to-use, well-tested utility modules that simplify common tasks, reduce boilerplate code, and promote
reproducibility in scientific projects.

## Dependencies

Key dependencies (see `requirements.txt` for full list):

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

- **torch, torchvision, torchaudio**: Deep learning with PyTorch
- **tensorflow, keras**: Deep learning (alternative to PyTorch)
- **opencv-python, opencv-contrib-python**: Image processing
- **openslide-python**: Whole Slide Image (WSI) support
- **numpy, pandas, scipy, scikit-learn, scikit-image**: Scientific computing and ML
- **matplotlib, seaborn**: Visualization
- **nltk, rouge, textstat**: NLP and text metrics
- **PyMuPDF**: PDF reading
- **tqdm**: Progress bars
- **albumentations**: Image augmentation
- **shap**: Model explainability
- **catboost, xgboost**: ML models
- **spams-bin**: Sparse modeling

## Features & Modules

- **AttentionMapsHelper**: Tools for generating and visualizing attention maps in deep learning models.
- **EmbeddingsToTextHelper**: Convert between embeddings and text representations for NLP tasks.
- **ExplainabilityHelper**: Utilities for model explainability and interpretability (e.g., SHAP analysis).
- **HandCraftedFeatures**: Feature extraction utilities for images and tabular data.
- **ImagesComparisonMetrics**: Functions to compare images using metrics such as SSIM, PSNR, and MSE.
- **ImageSegmentationMetrics**: Evaluation metrics for segmentation tasks, including IoU, Dice, and pixel accuracy.
- **ImagesHelper**: Utilities for loading, saving, resizing, cropping, and manipulating images.
- **ImagesNormalization**: Tools for image normalization, standardization, and preprocessing.
- **Initializations**: Model and layer initialization helpers for deep learning frameworks.
- **MachineLearningHelper**: General machine learning workflow helpers (data splitting, cross-validation, etc.).
- **MetaheuristicsHelper**: Utilities for metaheuristic optimization algorithms (e.g., MRFO).
- **PDFHelper**: Functions for reading, extracting, and processing text from PDF files.
- **PerformanceMetrics**: General performance metrics for machine learning workflows (accuracy, precision, recall, F1,
  etc.).
- **PyTorchHelper**: PyTorch model and tensor utilities, including device management and checkpointing.
- **PyTorchSegmentationLosses**: Custom loss functions for segmentation tasks (Dice, BCE, DiceBCE, Focal, Tversky, IoU
  losses) for PyTorch models.
- **StatisticalAnalysisHelper**: Statistical analysis and data exploration tools (summary stats, hypothesis testing,
  etc.).
- **TextGenerationMetrics**: Metrics for evaluating text generation models (e.g., ROUGE, BLEU, METEOR).
- **TextHelper**: Text normalization, cleaning, and tokenization utilities.
- **Utils**: Miscellaneous utilities for file I/O, logging, and more.
- **WSIHelper**: Whole Slide Image (WSI) processing tools for digital pathology.

## Quick Start

Import and use any helper module as needed. Example for segmentation loss:

```python
import torch
from HMB.PyTorchSegmentationLosses import DiceLoss, DiceBCELoss

predictions = torch.randn(1, 1, 256, 256).float()
targets = torch.randint(0, 2, (1, 1, 256, 256)).float()

criterion = DiceBCELoss()
loss = criterion(predictions, targets)
print(f"DiceBCE Loss: {loss.item()}")
```

### Example: 3D Volume Loading and LAB Normalization

```python
from HMB.ImagesHelper import ReadVolume
from HMB.ImagesNormalization import RGB2LAB

# Example: Load a 3D volume from lists of image and mask paths
img_paths = ["slice1.png", "slice2.png", "slice3.png"]
mask_paths = ["mask1.png", "mask2.png", "mask3.png"]
volume = ReadVolume(img_paths, mask_paths)

# Convert a loaded RGB image to LAB color space
import cv2

img_rgb = cv2.imread("sample.png")
l, a, b = RGB2LAB(img_rgb)
```

### Example: PDF Text Extraction

```python
from HMB.PDFHelper import ReadFullPDF

text = ReadFullPDF('document.pdf')
print(text[:500])
```

### Example: Text Generation Metrics

```python
from HMB.TextGenerationMetrics import TextGenerationMetrics

metrics = TextGenerationMetrics()
reference = ["the cat is on the mat"]
hypothesis = "the cat sat on the mat"
bleu_score = metrics.CalculateBLEU(hypothesis, reference)
print(f"BLEU score: {bleu_score}")
```

## More Usage Examples

### ExplainabilityHelper: SHAPExplainer

```python
from HMB.ExplainabilityHelper import SHAPExplainer

explainer = SHAPExplainer(
  baseDir="/path/to/baseDir",
  experimentFolderName="Experiment1",
  testFilename="test_data.csv",
  targetColumn="target",
  pickleFilePath=None,
  shapStorageKeyword="SHAP_Results"
)
explainer.run()  # See module docs for full pipeline
```

### WSIHelper: ReadWSIViaOpenSlide

```python
from HMB.WSIHelper import ReadWSIViaOpenSlide

slide = ReadWSIViaOpenSlide("/path/to/slide.svs")
print(slide.dimensions)
```

### MetaheuristicsHelper: MantaRayForagingOptimizer

```python
import numpy as np
from HMB.MetaheuristicsHelper import MantaRayForagingOptimizer

# Example: Run one iteration of MRFO
X = np.random.rand(10, 5)  # 10 candidates, 5 dimensions
Fs = np.random.rand(10)  # Fitness values
newX, bestSol, bestFit = MantaRayForagingOptimizer(
  X, Fs, Ps=10, D=5, lb=np.zeros(5), ub=np.ones(5), t=1, T=100
)
print("Best fitness:", bestFit)
```

## Documentation

- [HTML docs](build/html/index.html)
- [Source docs](source/)

Each module is documented with usage examples and API references. See the `source/` folder for detailed documentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please discuss them first
by opening an issue.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Open a pull request

## Changelog / Release Notes

### v1.0.0 (2025-09-08)

- Initial public release
- All core modules and documentation included
- Extensive examples and API docs

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
