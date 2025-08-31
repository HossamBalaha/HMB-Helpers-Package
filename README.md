# HMB Helpers Package

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive collection of helper modules for image processing, segmentation, deep learning workflows, and text/PDF
utilities in PyTorch and beyond.

## Motivation

HMB Helpers Package aims to accelerate research and development in computer vision, deep learning, and text analytics by
providing ready-to-use, well-tested utility modules.

## Features & Modules

- **PyTorchSegmentationLosses**: Custom loss functions for segmentation tasks (Dice, BCE, etc.).
- **ImagesHelper**: Image loading, saving, and manipulation utilities.
- **ImagesComparisonMetrics**: Metrics for comparing images (SSIM, PSNR, etc.).
- **ImageSegmentationMetrics**: Segmentation evaluation metrics (IoU, Dice, etc.).
- **ImagesNormalization**: Image normalization and preprocessing tools.
- **PDFHelper**: PDF reading and text extraction utilities.
- **TextHelper**: Text processing and normalization functions.
- **TextGenerationMetrics**: Metrics for evaluating text generation (ROUGE, BLEU, etc.).
- **PerformanceMetrics**: General performance metrics for ML workflows.
- **TorchHelper**: PyTorch model and tensor utilities.
- **EmbeddingsToTextHelper**: Convert embeddings to text and vice versa.
- **Initializations**: Model and layer initialization helpers.
- **Utils**: Miscellaneous utilities.
- **WSIHelper**: Whole Slide Image (WSI) processing tools.
- **HandCraftedFeatures**: Feature extraction utilities for images and data.
- **MachineLearningHelper**: General machine learning workflow helpers.
- **StatisticalAnalysisHelper**: Statistical analysis and data exploration tools.

## Installation

```bash
pip install -e .
```

Or, to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage Example

### PyTorch Segmentation Losses

```python
import torch
from HMB.PyTorchSegmentationLosses import DiceLoss, DiceBCELoss

predictions = torch.randn(1, 1, 256, 256).float()
targets = torch.randint(0, 2, (1, 1, 256, 256)).float()

criterion = DiceBCELoss()
loss = criterion(predictions, targets)
print(f"DiceBCE Loss: {loss.item()}")
```

### Other Modules

Refer to the documentation in the `source/` folder for details on other modules and their usage.

## Documentation

- [HTML docs](build/html/index.html)
- [Source docs](source/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Citation & License

Refer to the header in each module and the LICENSE file for citation and usage permissions.

---
For questions or contributions, please contact the author (Hossam Magdy Balaha, h3ossam@gmail.com) or open an
issue.
