.. image:: _static/Logo.jpg
   :alt: HMB Logo
   :align: center
   :width: 200px

.. |build| image:: https://img.shields.io/github/last-commit/HossamBalaha/HMB-Helpers-Package.png
   :alt: Last Commit
   :target: https://github.com/HossamBalaha/HMB-Helpers-Package/commits/main
.. |license| image:: https://img.shields.io/github/license/HossamBalaha/HMB-Helpers-Package.png
   :alt: MIT License
   :target: https://opensource.org/licenses/MIT
.. |python| image:: https://img.shields.io/pypi/pyversions/hmb-helpers.png
   :alt: Python versions
   :target: https://pypi.org/project/hmb-helpers/
.. |pypi| image:: https://img.shields.io/pypi/v/hmb-helpers.png
   :alt: PyPI
   :target: https://pypi.org/project/hmb-helpers/

|build| |license| |python| |pypi|

HMB Documentation
=================

**Version:** 0.1.0 ([Changelog](https://github.com/HossamBalaha/HMB-Helpers-Package/releases))

Welcome to the **HMB Helpers Package** documentation!

The HMB package provides a comprehensive suite of helper modules for image processing, text generation, PDF handling, performance metrics, and more. It is designed to accelerate research and development workflows in machine learning, computer vision, and natural language processing.

.. note::
   This documentation covers all modules and utilities included in the HMB package.

.. warning::
   This package is under active development. APIs, example scripts and behaviors
   may change between releases. Example scripts included in the repository are
   provided for demonstration purposes and may require additional datasets,
   environment configuration, or optional dependencies to run successfully.
   Use them as reference; exercise caution before running any example in
   production environments.

Features
--------
- Image comparison, normalization, and segmentation metrics.
- Text generation and embedding utilities.
- PDF processing helpers.
- PyTorch segmentation losses.
- Performance metrics and initializations.
- Utilities for working with whole slide images (WSI).

Installation
------------

Install the package as described in the :doc:`Installation <Installation>` guide.

Quick pip install:

.. code-block:: bash

   pip install hmb-helpers

Quickstart
----------

Prerequisites
~~~~~~~~~~~~~
Before running examples, ensure you have:

- Python 3.8+ installed
- Required dependencies: ``pip install -r requirements.txt``
- Optional: GPU support for PyTorch/TensorFlow examples (see `Installation <Installation.html>`_)

.. tip::
   For CPU-only environments, install PyTorch with:

   .. code-block:: bash

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Basic Examples
~~~~~~~~~~~~~~

**Image Helper: Calculate empty region percentage**

.. code-block:: python

   from PIL import Image
   import numpy as np
   from HMB.ImagesHelper import GetEmptyPercentage

   # Load and preprocess image
   img = np.array(Image.open("path/to/image.png").convert("RGB"))

   # Calculate empty region ratio (default shape: 256x256).
   emptyRatio = GetEmptyPercentage(img, shape=(256, 256))
   print(f"Empty region: {emptyRatio:.2%}")

.. code-block:: text

   # Expected output:
   # Empty region: 12.34%

**PDF Helper: Extract full text**

.. code-block:: python

   from HMB.PDFHelper import ReadFullPDF

   # Extract all text from PDF.
   text = ReadFullPDF("path/to/document.pdf")
   print(text[:200])  # Preview first 200 characters.

.. code-block:: text

   # Expected output:
   # This is the beginning of the extracted PDF content...

**Text Helper: Clean and normalize text**

.. code-block:: python

   from HMB.TextHelper import CleanText

   raw = "  Hello!!!  This is a SAMPLE text...  "
   cleaned = CleanText(
     raw,
     lowercase=True,
     removeSpecialChars=True,
     normalizeWhitespace=True
   )
   print(f"Original: '{raw}'")
   print(f"Cleaned:  '{cleaned}'")

.. code-block:: text

   # Expected output:
   # Original: '  Hello!!!  This is a SAMPLE text...  '
   # Cleaned:  'hello this is a sample text'

Advanced Examples
~~~~~~~~~~~~~~~~~

**PyTorch Model: Quick inference with device auto-detection**

.. code-block:: python

   import torch
   from HMB.PyTorchHelper import CreateTimmModel, LoadModel

   # Auto-detect device.
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

   # Create and load model.
   model = CreateTimmModel("resnet18", numClasses=10, pretrained=True)
   model = LoadModel(model, filename="model.pth", device=device)
   model.eval()

   # Run inference on dummy input.
   dummyInput = torch.randn(1, 3, 224, 224).to(device)
   with torch.no_grad():
     output = model(dummyInput)
     predictions = torch.softmax(output, dim=1)
     print(f"Top-3 classes: {torch.topk(predictions, 3)[1].cpu().numpy()}")

**Segmentation Metrics: Evaluate predictions**

.. code-block:: python

   import numpy as np
   from HMB.ImageSegmentationMetrics import ComputeIoU, ComputeDice

   # Dummy binary masks (batch=1, channels=1, H=256, W=256).
   preds = np.random.rand(1, 1, 256, 256) > 0.5
   targets = np.random.randint(0, 2, size=(1, 1, 256, 256))

   # Compute metrics.
   iou = ComputeIoU(preds.astype(float), targets.astype(float))
   dice = ComputeDice(preds.astype(float), targets.astype(float))

   print(f"IoU: {iou:.4f}, Dice: {dice:.4f}")

.. code-block:: text

   # Expected output (values will vary):
   # IoU: 0.5123, Dice: 0.6789

Common Pitfalls & Tips
~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   **File paths**: Always use absolute paths or ``pathlib.Path`` to avoid working-directory issues.

   .. code-block:: python

      from pathlib import Path
      img_path = Path("data") / "images" / "sample.png"

.. note::
   **Memory management**: For large images or WSIs, process in chunks or use ``ExtractRandomTilesFromImages`` to avoid OOM errors.

.. tip::
   **Reproducibility**: Seed all random sources for consistent results:

   .. code-block:: python

      from HMB.Initializations import SeedEverything
      SeedEverything(seed=42)

Next Steps
~~~~~~~~~~
- Explore more examples in ``HMB/Examples/``
- Read module-specific guides: `AgentsHelper <AgentsHelper.html>`_, `PyTorchHelper <PyTorchHelper.html>`_
- Run the test suite: ``python tests/run_tests.py``
- View the full API reference: `modules.html <modules.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation
   AgentsHelper
   ArabicTextHelper
   AttentionMapsHelper
   AudioHelper
   CompressionsHelper
   DataAugmentationHelper
   DatasetsHelper
   EmbeddingsToTextHelper
   ExplainabilityHelper
   HandCraftedFeatures
   ImageSegmentationMetrics
   ImagesComparisonMetrics
   ImagesHelper
   ImagesNormalization
   ImagesToEmbeddings
   Initializations
   MachineLearningHelper
   MetaheuristicsHelper
   PDFHelper
   PerformanceMetrics
   PlotsHelper
   PyTorchClassificationLosses
   PyTorchHelper
   PyTorchModelMemoryProfiler
   PyTorchSegmentationLosses
   PyTorchTabularModelsZoo
   PyTorchTrainingPipeline
   PyTorchUNetModelsZoo
   StatisticalAnalysisHelper
   StringsHelper
   TFAttentionBlocks
   TFHelper
   TFSegmentationLosses
   TFUNetHelper
   TextGenerationMetrics
   TextHelper
   Utils
   VectorsHelper
   VideosHelper
   VotingHelper
   WSIHelper
   YOLOHelper
   Examples
   modules
   FAQ

Getting Help
------------
- For questions, open an issue on `GitHub <https://github.com/HossamBalaha/HMB-Helpers-Package/issues>`_
- Email: h3ossam@gmail.com

Contributing
------------
We welcome contributions! See `CONTRIBUTING.md <https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CONTRIBUTING.md>`_ for guidelines.

FAQ
---
See the `FAQ <FAQ.html>`_ for common questions and troubleshooting tips.

Useful Links
------------
- `GitHub Repository <https://github.com/HossamBalaha/HMB-Helpers-Package.git>`_
- `API Reference <modules.html>`_
- `Contact & Support <mailto:h3ossam@gmail.com>`_

License
-------

This project is licensed under the MIT License. See the `LICENSE file <https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/LICENSE>`_ or the official MIT terms at `Open Source Initiative <https://opensource.org/licenses/MIT>`_.
