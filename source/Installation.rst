Installation
============

This section describes how to install the HMB Helpers Package and its dependencies.

Prerequisites
-------------

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-compatible GPU for GPU acceleration

Installation Methods
--------------------

From GitHub (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

Install the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/HossamBalaha/HMB-Helpers-Package.git

Editable Install (For Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan to modify the package source code:

.. code-block:: bash

   git clone https://github.com/HossamBalaha/HMB-Helpers-Package.git
   cd HMB-Helpers-Package
   pip install -e .

Installing Dependencies
-----------------------

Core dependencies are listed in ``requirements.txt``. Install them with:

.. code-block:: bash

   pip install -r requirements.txt

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Install optional feature groups via extras:

.. code-block:: bash

   # Scientific computing stack
   pip install "hmb-helpers[scientific]"

   # Computer vision & image processing
   pip install "hmb-helpers[cv]"

   # PyTorch deep learning (CPU wheels)
   pip install "hmb-helpers[pytorch]"

   # TensorFlow deep learning
   pip install "hmb-helpers[tensorflow]"

   # NLP & text processing (includes Arabic support)
   pip install "hmb-helpers[nlp]"

   # PDF handling with table extraction
   pip install "hmb-helpers[pdf]"

   # Audio processing & feature extraction
   pip install "hmb-helpers[audio]"

   # Medical imaging & whole slide images
   pip install "hmb-helpers[medical]"

   # Classical ML models & optimization
   pip install "hmb-helpers[ml]"

   # Visualization & plotting
   pip install "hmb-helpers[plotting]"

   # All optional dependencies (large install)
   pip install "hmb-helpers[all]"

Download required NLTK data after installing the ``nlp`` extra:

.. code-block:: bash

   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

For spaCy models:

.. code-block:: bash

   python -m spacy download en_core_web_sm
   # For Arabic: python -m spacy download xx_ent_wiki_sm

CUDA/PyTorch GPU Support
------------------------

The ``pytorch`` extra installs CPU-compatible PyTorch wheels by default. For GPU acceleration:

1. Install the package with PyTorch support:

   .. code-block:: bash

      pip install "hmb-helpers[pytorch]"

2. Replace with CUDA-specific wheels matching your system:

   .. code-block:: bash

      pip uninstall torch torchvision torchaudio -y
      pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

.. tip::
   For step-by-step guidance tailored to your operating system, package manager, and CUDA version, visit the official PyTorch installation selector: https://pytorch.org/get-started/locally/

.. note::
   TensorFlow GPU support requires separate installation. See https://www.tensorflow.org/install for platform-specific instructions.Verifying Installation

Verifying Installation
----------------------

After installation, verify the package and optional features:

.. code-block:: python

   import HMB
   print(f"Package version: {HMB.__version__}")  # Should print: 0.1.0

   # Test core functionality.
   from HMB.ImagesHelper import GetEmptyPercentage
   print("✓ Core images module loaded")

   # Test optional feature (will raise ImportError if not installed).
   try:
     from HMB.PDFHelper import ReadFullPDF
     print("✓ PDF module available")
   except ImportError:
     print("⚠ PDF module: install with pip install 'hmb-helpers[pdf]'")

Troubleshooting
---------------

- **Import errors**: Ensure you installed in the same virtual environment you are using.
- **Missing native libraries**: For ``openslide`` or ``tabula-py``, install system-level dependencies first (see upstream docs).
- **Dependency conflicts**: Use a fresh virtualenv or conda environment.
- **NLTK/spaCy models**: After installing the ``nlp`` extra, download required models as shown above.

.. tip::
   For step-by-step guidance tailored to your operating system and CUDA version, visit the official PyTorch installation selector: https://pytorch.org/get-started/locally/
