.. image:: _static/Logo.jpg
   :alt: HMB Logo
   :align: center
   :width: 200px

.. |build| image:: _static/build.png
   :alt: Build Status
.. |license| image:: _static/license.png
   :alt: License
.. |python| image:: _static/python.png
   :alt: Python Version

|build| |license| |python|

HMB Documentation
=================

**Version:** 0.1.0 ([Changelog](https://github.com/HossamBalaha/HMB-Helpers-Package/releases))

Welcome to the **HMB Helpers Package** documentation!

The HMB package provides a comprehensive suite of helper modules for image processing, text generation, PDF handling, performance metrics, and more. It is designed to accelerate research and development workflows in machine learning, computer vision, and natural language processing.

.. note::
   This documentation covers all modules and utilities included in the HMB package.

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
.. code-block:: bash

   pip install git+https://github.com/HossamBalaha/HMB-Helpers-Package.git

Quickstart
----------
**Image Helper Example:**

.. code-block:: python

   from PIL import Image
   import numpy as np
   from HMB.ImagesHelper import GetEmptyPercentage

   img = Image.open('path/to/image.png').convert('RGB')
   emptyRatio = GetEmptyPercentage(img, shape=(256, 256))
   print(f"Empty region percentage: {emptyRatio:.2f}%")

**PDF Helper Example:**

.. code-block:: python

   from HMB.PDFHelper import ReadFullPDF
   text = ReadFullPDF('path/to/file.pdf')
   print(text)

**Text Helper Example:**

.. code-block:: python

   from HMB.TextHelper import CleanText
   raw = "This is a sample!"
   cleaned = CleanText(raw)
   print(cleaned)

.. note::
   Requires a set of dependencies. See the file: `requirements.txt <https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/requirements.txt>`_

.. toctree::
   :maxdepth: 1
   :caption: Contents:

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
   PyTorchClassificationLosses
   PyTorchHelper
   PyTorchSegmentationLosses
   StatisticalAnalysisHelper
   StringsHelper
   TFAttentionBlocks
   TFHelper
   TextGenerationMetrics
   TextHelper
   Utils
   VectorsHelper
   VideosHelper
   VotingHelper
   WSIHelper
   YOLOHelper

Getting Help
------------
- For questions, open an issue on `GitHub <https://github.com/HossamBalaha/HMB-Helpers-Package/issues>`_
- Email: h3ossam@gmail.com

Contributing
------------
We welcome contributions! See `CONTRIBUTING.md <https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/CONTRIBUTING.md>`_ for guidelines.

FAQ
---
**Q:** What image formats are supported?
**A:** PNG, JPEG, TIFF, and more via Pillow.

**Q:** How do I report a bug?
**A:** Open an issue on GitHub or email support.

Useful Links
------------
- `GitHub Repository <https://github.com/HossamBalaha/HMB-Helpers-Package.git>`_
- `API Reference <modules.html>`_
- `Contact & Support <mailto:h3ossam@gmail.com>`_

LaTeX Math Example
------------------

Here is an example of a matrix using LaTeX math in Sphinx:

.. math::
  H = \begin{bmatrix}
    h_{11} & h_{12} & h_{13} \\
    h_{21} & h_{22} & h_{23} \\
    h_{31} & h_{32} & h_{33}
  \end{bmatrix}
