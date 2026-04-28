FAQ
===

This file collects frequently asked questions (FAQ) about the **HMB Helpers Package** and its example scripts.

Quick Pointers
--------------

- **Project root**: Top-level Python package is ``HMB``.
- **Example scripts**: ``HMB/Examples``.
- **Documentation source**: ``source/`` (build with Sphinx into ``build/html``).
- **Tests**: Run via ``tests/run_tests.py`` (see README for examples).
- **Package version**: 0.1.0 (`Changelog <https://github.com/HossamBalaha/HMB-Helpers-Package/releases>`_).

General
-------

**Q:** What is the HMB Helpers Package?

**A:** The HMB package provides a comprehensive suite of helper modules for image processing, text generation, PDF handling, performance metrics, reinforcement learning agents, audio processing, data augmentation, and more. It is designed to accelerate research and development workflows in machine learning, computer vision, and natural language processing.

**Q:** Is HMB stable and production-ready?

**A:** The package is under active development. Many modules are well-tested and stable for research and prototyping, but APIs and example scripts may change between releases. Check the release notes on GitHub for breaking changes and versioned behavior.

**Q:** Where can I report bugs or request features?

**A:** Open an issue on GitHub: https://github.com/HossamBalaha/HMB-Helpers-Package/issues

**Q:** Where are the project requirements listed?

**A:** See ``requirements.txt`` at the repository root. Some optional examples may require additional packages (e.g., ``timm``, ``torchvision``, ``seaborn``, ``scikit-learn``, ``nltk``, ``spacy``, ``transformers``, ``openslide``, ``tabula-py``).

**Q:** What image and data formats are supported?

**A:** See the respective helper modules for details, but generally:
  - **Images**: PNG, JPEG, TIFF, BMP, GIF, WebP, DICOM (via Pillow and OpenCV).
  - **Videos**: Common formats via OpenCV (``cv2.VideoCapture``).
  - **PDF**: Full text extraction, metadata, tables (via PyMuPDF, PyPDF2, tabula-py).
  - **Tabular data**: CSV, pandas DataFrames.
  - **Whole Slide Images (WSI)**: Via OpenSlide.
  - **Audio**: WAV, MP3, etc. via librosa and spafe.

Installation
------------

For comprehensive installation instructions, including platform-specific guidance, CUDA/PyTorch selection, and troubleshooting, see the :doc:`Installation Guide <Installation>`.

**Q:** How do I install the package?

**A:** You can install the latest release from GitHub via pip:

.. code-block:: bash

   pip install git+https://github.com/HossamBalaha/HMB-Helpers-Package.git

For development (editable install):

.. code-block:: bash

   git clone https://github.com/HossamBalaha/HMB-Helpers-Package.git
   cd HMB-Helpers-Package
   pip install -e .

**Q:** How do I choose CPU vs GPU (CUDA) builds of PyTorch?

**A:** The repository's ``requirements.txt`` may include an extra index URL for specific CUDA wheels and pins ``torch``/``torchvision``/``torchaudio`` to CUDA builds (e.g., ``+cu128``). If you don't have matching CUDA drivers or prefer CPU-only:

  - Install CPU wheels explicitly: ``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu``
  - Or remove the extra index from ``requirements.txt`` and install packages compatible with your platform.
  - Using a virtual environment (venv/conda) helps isolate these choices.
  - **Reference**: For step-by-step guidance tailored to your operating system, package manager, and CUDA version, visit the official PyTorch installation selector: https://pytorch.org/get-started/locally/

**Q:** Where can I find step-by-step installation instructions for PyTorch?

**A:** Visit the official PyTorch installation selector for interactive, platform-specific guidance: https://pytorch.org/get-started/locally/

This tool lets you select:
  - Your operating system (Windows, Linux, macOS)
  - Package manager (pip, conda)
  - Python version
  - CUDA version (or CPU-only)

It then generates the exact install command for your environment.

**Q:** I get dependency conflicts or installation failures. Any tips?

**A:** This project depends on many scientific packages; pinning can cause conflicts. Recommendations:

  - Use a fresh virtualenv or conda environment.
  - Install large binary packages (``torch``, ``tensorflow``, ``opencv``, ``openslide``) first using appropriate channels or extra-index URLs for CUDA.
  - When a package requires system libraries (e.g., ``openslide``, Java for ``tabula-py``), install the OS-level dependency first.
  - For minimal installs, install only the subset of dependencies you need for your use case.

**Q:** How do I install optional NLP dependencies?

**A:** For Arabic text processing and advanced NLP features:

.. code-block:: bash

   pip install emoji nltk qalsadi spacy langdetect textblob gtts wordcloud gensim

Then download required NLTK/spaCy models:

.. code-block:: python

   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   nltk.download("wordnet")
   # For spaCy:
   # python -m spacy download en_core_web_sm

.. note::
   For more detailed installation guidance, including platform-specific instructions and troubleshooting, see the :doc:`Installation Guide <Installation>`.

Windows-Specific Notes
----------------------

**Q:** I am on Windows. Are there any special instructions?

**A:** Yes:

  - Use ``make.bat`` included in the repo to build docs on Windows, or run Sphinx directly: ``python -m sphinx -b html source build/html``.
  - Some dependencies (e.g., ``openslide``, ``openslide-bin``, Java for ``tabula-py``) require additional native binaries. Follow upstream instructions for installing OS-level libraries on Windows before installing Python wrappers.
  - Some examples relying on POSIX shell scripts have Windows batch equivalents in ``HMB/Examples/BAT Files``.
  - Ensure your terminal supports UTF-8 encoding for proper handling of Arabic text and special characters.

Documentation
-------------

**Q:** How do I build the documentation locally?

**A:** From the repository root:

.. code-block:: bash

   # Using make (Linux/macOS)
   cd source && make html

   # Or directly with Python
   python -m sphinx -b html source build/html

   # Windows batch helper
   cd source && make.bat html

**Q:** Sphinx reports "duplicate object description" or warns about math in docstrings; what should I do?

**A:**
  - **Duplicate descriptions**: Often come from documenting the same object twice (e.g., a manual "Methods:" list in a class docstring that reproduces method signatures). Prefer documenting functions/classes in their own docstrings and avoid duplicating signatures in text.
  - **LaTeX/math warnings**: When docstrings contain LaTeX with backslashes, use raw string literals (``r'''...'''``) or escape backslashes appropriately. Ensure math blocks are separated from section headers by a blank line.

Examples
--------

**Q:** Where are example scripts and how do I run them safely?

**A:** Example scripts live under ``HMB/Examples``. There are convenience wrapper scripts (batch and shell) in ``HMB/Examples/BAT Files`` and ``HMB/Examples/SH Files``, and top-level wrappers such as ``run_examples.bat`` / ``run_examples.sh``.

**Always inspect the header comments in each example** to learn:

  - Required data paths and environment variables
  - Optional dependencies
  - GPU/CPU requirements
  - Expected input/output formats

**Q:** Some examples download pretrained weights (e.g., ``timm``, ``transformers``). Will they download large files?

**A:** Yes. Examples using ``timm``, ``transformers``, Hugging Face models, or other model libraries may download pretrained weights on first run. Ensure you have:

  - Network access
  - Sufficient disk space (some models are several GB)
  - Consider sharing a central cache for model weights to avoid repeated downloads across examples

**Q:** How do I run the Timm fine-tuning example?

**A:** The ``Timm_FineTune_Classification.py`` example requires ``timm`` and supports on-the-fly data splitting. Example invocation:

.. code-block:: bash

   python HMB/Examples/Timm_FineTune_Classification.py \
       --dataDir /path/to/dataset \
       --numClasses 3 \
       --modelName eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \
       --device cuda

**Q:** How do I use the PyTorch UNet segmentation example?

**A:** The ``PyTorch_UNet_Segmentation.py`` example checks for required modules and provides a training pipeline. Default device is ``cuda`` if available. Use ``--Device`` to override:

.. code-block:: bash

   python HMB/Examples/PyTorch_UNet_Segmentation.py \
       --dataDir /path/to/segmentation_data \
       --numClasses 2 \
       --Device cpu

Module-Specific Questions
-------------------------

Image Processing & Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Q:** What segmentation metrics are available?

**A:** The ``ImageSegmentationMetrics`` module provides:

  - IoU, Dice, Pixel Accuracy, Precision, Recall, F1-Score
  - Specificity, FPR, FNR, Balanced Accuracy
  - Hausdorff Distance, Boundary F1-Score
  - Matthews Correlation Coefficient, Cohen's Kappa
  - Surface distances (MSD, ASSD), Volumetric Overlap Error
  - Loss functions: Focal, Tversky, Combo, Tanimoto, HMB Loss

**Q:** How do I compare two images?

**A:** Use ``ImagesComparisonMetrics`` for:

  - Structural Similarity (SSIM), PSNR, MSE
  - Mutual Information, Normalized Cross-Correlation
  - Feature-based similarity (SIFT), Perceptual Hash
  - And 20+ additional metrics via ``SummaryTable()``

**Q:** How do I normalize histopathology images?

**A:** Use ``ImagesNormalization`` for:

  - Reinhard color normalization
  - Macenko stain separation
  - LAB/OD space conversions
  - Brightness standardization

Text & NLP
~~~~~~~~~~

**Q:** What Arabic text processing features are available?

**A:** The ``ArabicTextHelper`` module provides:

  - Regex-based normalization and cleaning
  - ISRI stemming (via NLTK)
  - Qalsadi lemmatization
  - Stopword removal, diacritics stripping
  - Character/word tokenization, n-gram extraction
  - Arabic-to-English digit conversion

**Q:** How do I summarize long documents?

**A:** Use the ``Summarizer`` class in ``TextHelper``:

.. code-block:: python

   from HMB.TextHelper import Summarizer
   summarizer = Summarizer(modelName="facebook/bart-large-cnn")
   summary = summarizer.Summarize("Your long text here...")

The class automatically:

  - Uses GPU if available
  - Chunks long texts (>1024 chars)
  - Adjusts max/min length dynamically

Reinforcement Learning Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Q:** What RL agents are implemented?

**A:** The ``AgentsHelper`` module includes tabular implementations of:

  - Random, Greedy, Epsilon-Greedy policies
  - Q-Learning, SARSA, Expected SARSA
  - Double Q-Learning, Q(λ), SARSA(λ)
  - Monte Carlo, Dyna-Q, Prioritized Sweeping
  - UCB1 (bandits), Count-based exploration, n-step TD
  - Softmax/Boltzmann policy agent

All agents follow a consistent API: ``ChooseAction(state)``, ``UpdateParameters(...)``.

Audio Processing
~~~~~~~~~~~~~~~~

**Q:** What audio features can I extract?

**A:** The ``AudioHelper`` module supports:

  - **Librosa-based**: MFCC (Slaney/HTK), Chroma, Mel-spectrogram, Spectral contrast, Tonnetz, RMS, ZCR
  - **Spafe-based**: BFCC, GFCC, LFCC, LPC, LPCC, PNCC, PLP, and variants
  - **Parselmouth**: Pitch, jitter, shimmer, HNR, voice quality metrics
  - Utility functions: Duration, STFT, harmonic/percussive separation

PDF & Document Handling
~~~~~~~~~~~~~~~~~~~~~~~

**Q:** What PDF operations are supported?

**A:** The ``PDFHelper`` module provides:

  - Full text extraction, page-wise reading
  - Regex-based text search and extraction
  - Metadata, annotations, hyperlinks extraction
  - Image and table extraction (via tabula-py)
  - Split, merge, rotate, encrypt/decrypt PDFs
  - Bookmark addition, text highlighting

**Note**: Table extraction requires Java and ``jpype1``: ``pip install tabula-py jpype1``

Whole Slide Images (WSI)
~~~~~~~~~~~~~~~~~~~~~~~~

**Q:** How do I work with WSIs?

**A:** The ``WSIHelper`` module supports:

  - Reading slides via OpenSlide
  - Extracting patches within annotated regions
  - Aligning HE/MT slide pairs via SIFT/ORB + homography
  - Free-form deformation (B-spline FFD) for registration
  - Background tile filtering using entropy/variance heuristics
  - Pyramidal tile extraction across resolution levels

**Prerequisite**: Install native OpenSlide binaries before ``openslide-python``.

Runtime Issues & Tips
---------------------

**Q:** I run out of GPU memory when training examples. Any mitigations?

**A:** Common mitigations include:

  - Reduce batch size or input resolution
  - Use mixed precision (AMP): ``useAmp=True`` in training pipelines
  - Enable gradient accumulation (``gradAccumSteps>1``)
  - Move to CPU for debugging smaller cases
  - Use model pruning or lighter architectures (e.g., MobileUNet)

**Q:** I have trouble opening large TIFF / WSI files.

**A:** WSI handling uses ``openslide``. Ensure:

  - Native OpenSlide binaries are installed for your OS
  - ``openslide-python`` matches the native version
  - Use streaming or region reads (avoid loading whole slides at full resolution)
  - For patch extraction, use ``ExtractRandomTilesFromImages`` with background filtering

**Q:** How do I ensure reproducibility?

**A:** Use the ``Initializations`` module utilities:

.. code-block:: python

   from HMB.Initializations import SeedEverything
   SeedEverything(seed=42, deterministic=True, benchmark=False)

This seeds NumPy, PyTorch, TensorFlow, and Python random modules, and configures cuDNN for deterministic behavior.

**Q:** How do I profile model memory usage?

**A:** Use ``PyTorchModelMemoryProfiler``:

.. code-block:: python

   from HMB.PyTorchModelMemoryProfiler import PyTorchModelMemoryProfiler
   profiler = PyTorchModelMemoryProfiler(model, inputShape=(3, 224, 224))
   profile = profiler.ProfileModelMemory(optimizerType="Adam")
   profiler.PrintMemoryReport(profile)

Reports parameter counts, activation memory, optimizer state, and FLOPs estimates.

Contributing
------------

**Q:** How do I contribute?

**A:** Follow the CONTRIBUTING guidelines in the repository:

  1. Fork the repository and create a feature branch
  2. Add unit tests for new functionality
  3. Update docstrings and documentation
  4. Ensure tests pass locally
  5. Submit a pull request with a clear description of changes

**Q:** What coding style should I follow?

**A:** The package uses:

  - **camelCase** for variables and function parameters
  - **CamelCase** for classes, file names, and dict keys for fixed strings
  - Parentheses with ``if`` statements: ``if (condition):``
  - Comments above each statement, ending with a period
  - Double quotes for strings (except docstrings)

Licensing & Citation
--------------------

**Q:** What is the license?

**A:** This project is licensed under the **MIT License**. See the ``LICENSE`` file in the repository root for details.

**Q:** How do I cite this package?

**A:** If you use the HMB Helpers Package in academic work, please cite:

.. code-block:: bibtex

   @software{balaha_hmb_helpers_2026,
     author = {Balaha, Hossam Magdy},
     title = {HMB-Helpers-Package},
     year = {2026},
     version = {0.1.0},
     url = {https://github.com/HossamBalaha/HMB-Helpers-Package}
   }

Contact & Support
-----------------

**Q:** Who maintains this project?

**A:** The package is maintained by Hossam Magdy Balaha. For most issues, open a GitHub issue in the repository so maintainers and the community can discuss and track the problem.

**Q:** How do I get urgent support?

**A:** For urgent questions, contact the maintainer at ``h3ossam@gmail.com``. Please include:

  - Package version (``HMB.__version__`` or check ``releases``)
  - Python/environment details (``python --version``, ``pip list``)
  - Minimal reproducible example or error traceback

*This FAQ is a living document. If you have improvements or new questions, please open a pull request or file an issue.*
