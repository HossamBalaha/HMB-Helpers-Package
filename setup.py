# Import setuptools helper for packaging functions.
from setuptools import setup, find_packages
# Import Path helper for reading long description file.
from pathlib import Path

# Compute the project root directory for reading files.
this_directory = Path(__file__).parent
# Read the long description from README.md to be used on PyPI.
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Invoke setup() to declare package metadata and installation options.
setup(
  # Distribution name used on PyPI.
  name="hmb-helpers",
  # Current package version.
  version="0.2.0",
  # Author's display name.
  author="Hossam Magdy Balaha",
  # Author contact email.
  author_email="h3ossam@gmail.com",
  # Short description of the package.
  description="A comprehensive suite of helper modules for image processing, text generation, PDF handling, and ML workflows.",
  # Long description content loaded from the README file.
  long_description=long_description,
  # MIME type of the long description.
  long_description_content_type="text/markdown",
  # Project homepage URL.
  url="https://github.com/HossamBalaha/HMB-Helpers-Package",
  # Additional project-related URLs.
  project_urls={
    "Homepage"     : "https://github.com/HossamBalaha/HMB-Helpers-Package",
    "Bug Tracker"  : "https://github.com/HossamBalaha/HMB-Helpers-Package/issues",
    "Documentation": "https://hmb-helpers-package.readthedocs.io/en/latest/",
    "Source Code"  : "https://github.com/HossamBalaha/HMB-Helpers-Package",
  },
  # License identifier for the package.
  license="MIT",
  # Files that contain license text to be included in the distribution.
  license_files=["LICENSE"],
  # Packages to include in the distribution.
  # Include all packages under the project. The examples folder is distributed as
  # package data (so users installing from PyPI will get the example scripts).
  packages=find_packages(exclude=["tests", "tests.*"]),
  # Additional package data to include. We explicitly add several glob patterns
  # under the `HMB` package so example scripts, wrappers and README appear in
  # built distributions (wheels and sdists).
  package_data={
    "HMB": [
      "py.typed",
      "Examples/*.md",
      "Examples/*.py",
      "Examples/*.json",
      "Examples/*/*",
      "Examples/*/*/*",
    ]
  },
  # Include package data defined in MANIFEST.in as well.
  include_package_data=True,
  # Do not install the package as a zipped egg.
  zip_safe=False,
  # Minimum supported Python version.
  python_requires=">=3.8",

  # ✅ Minimal core: only what's needed for basic imports.
  install_requires=[
    # Updated minima to match the pinned development environment in requirements.txt.
    "numpy>=1.26.4,<2",
    "pillow>=12.2.0",
    "pyyaml>=6.0.3",
    "pandas>=2.3.3",
    "matplotlib>=3.9",
    "tqdm>=4.67.3",
    "scikit-learn>=1.7.2",
  ],

  # ✅ Optional feature groups
  extras_require={
    # Core scientific stack.
    "scientific"  : [
      "scipy>=1.17.1",
      "scikit-image>=0.26.0",
    ],

    # Image processing & computer vision.
    "cv"          : [
      "opencv-python>=4.13.0.92",
      "opencv-python-headless>=4.13.0.92",
      "opencv-contrib-python>=4.13.0.92",
      "imagehash>=4.3.2",
      "brisque>=0.2.0",
      "simpleitk>=2.5.4",
      "PyWavelets>=1.9.0",
      "pyvips>=3.1.1",
      "patchify>=0.2.3",
      "av>=17.0.1",
    ],

    # Deep learning frameworks (CPU defaults; users install CUDA separately).
    "pytorch"     : [
      # Use non-platform-suffixed minima; CUDA-specific wheels should be installed
      # by users from the appropriate extra index (requirements.txt keeps pinned dev pins).
      "torch>=2.7.1",
      "torchvision>=0.22.1",
      "torchaudio>=2.7.1",
    ],
    "tensorflow"  : [
      "tensorflow>=2.12.0",
      "tensorboard>=2.12.0",
      "keras>=2.12.0",
      "tf-keras>=2.21.0",
    ],

    # Vision models & pretrained weights.
    "timm"        : ["timm>=1.0.26"],
    "transformers": ["transformers>=5.8.0", "sentence-transformers>=5.4.1"],
    "ultralytics" : ["ultralytics>=8.0.0"],

    # Medical imaging & WSI.
    "medical"     : [
      "pydicom>=3.0.2",
      "nibabel>=5.4.2",
      "openslide-python>=1.4.3",
      "openslide-bin>=4.0.0.13",
    ],

    # PDF handling.
    "pdf"         : [
      "PyMuPDF>=1.27.2.3",
      "PyPDF2>=3.0.1",
      "tabula-py>=2.10.0",
      "jpype1>=1.7.1",
    ],

    # NLP & text processing.
    "nlp"         : [
      "nltk>=3.9.4",
      "spacy>=3.8.14",
      "textblob>=0.20.0",
      "gensim>=4.4.0",
      "qalsadi>=0.5.1",
      "langdetect>=1.0.9",
      "gtts>=2.5.4",
      "wordcloud>=1.9.6",
      "emoji>=2.15.0",
      "contractions>=0.1.73",
      "textstat>=0.7.13",
      "rouge>=1.0.1",
    ],

    # Audio processing.
    "audio"       : [
      "librosa>=0.11.0",
      "spafe>=0.3.3",
      "praat-parselmouth>=0.4.7",
    ],

    # Classical ML & optimization.
    "ml"          : [
      "xgboost>=3.2.0",
      "catboost>=1.2.10",
      "lightgbm>=4.0.0",
      "imbalanced-learn>=0.14.1",
      "optuna>=4.8.0",
      "keras-tuner>=1.4.8",
      "libsvm-official>=3.37.0",
      "spams-bin>=2.6.13",
      "skfeature-chappers>=1.1.0",
    ],

    # Visualization & plotting.
    "plotting"    : [
      "seaborn>=0.13.2",
      "plotly>=6.7.0",
      "kaleido>=1.3.0",
      "ptitprince>=0.3.1",
      "squarify>=0.4.4",
    ],

    # Utilities & compression.
    "utils"       : [
      "sympy>=1.14.0",
      "shapely>=2.1.2",
      "trimesh>=4.12.2",
      "pyglet>=2.1.14",
      "split-folders>=0.6.1",
      "gputil>=1.4.0",
      "shutup>=0.3.0",
      "py7zr>=1.1.0",
      "rarfile>=4.2",
      "albumentations>=2.0.8",
      "shap>=0.51.0",
      "grad-cam>=1.5.5",
      "statsmodels>=0.14.6",
      "medmnist>=3.0.2",
      "huggingface-hub>=1.14.0",
      "codecarbon>=3.2.6",
    ],

    # Development & testing.
    "dev"         : [
      "pytest>=7.0.0",
      "sphinx>=4.0.0",
      "black>=22.0.0",
      "flake8>=4.0.0",
      "mypy>=1.0.0",
    ],

    # Documentation building.
    "docs"        : [
      "sphinx>=4.0.0",
      "sphinx-rtd-theme>=1.0.0",
      "myst-parser>=0.18.0",
    ],

    # ✅ Install everything (for full environment setup).
    "all"         : [
      # Note: heavy framework packages (torch, tensorflow, keras, tensorboard, etc.)
      # are intentionally NOT included here to avoid large, platform-specific
      # installs when users request the `all` extra. Install those frameworks
      # using the dedicated extras or the official installers (see README).
      "scipy>=1.17.1",
      "pandas>=2.3.3",
      "scikit-learn>=1.7.2",
      "scikit-image>=0.26.0",
      "opencv-python>=4.13.0.92",
      "opencv-contrib-python>=4.13.0.92",
      "imagehash>=4.3.2",
      "brisque>=0.2.0",
      "simpleitk>=2.5.4",
      "PyWavelets>=1.9.0",
      "pyvips>=3.1.1",
      "patchify>=0.2.3",
      "timm>=1.0.26",
      "transformers>=5.8.0",
      "sentence-transformers>=5.4.1",
      "ultralytics>=8.4.47",
      "pydicom>=3.0.2",
      "nibabel>=5.4.2",
      "openslide-python>=1.4.3",
      "PyMuPDF>=1.27.2.3",
      "PyPDF2>=3.0.1",
      "tabula-py>=2.10.0",
      "jpype1>=1.7.1",
      "nltk>=3.9.4",
      "spacy>=3.8.14",
      "textblob>=0.20.0",
      "gensim>=4.4.0",
      "qalsadi>=0.5.1",
      "langdetect>=1.0.9",
      "gtts>=2.5.4",
      "wordcloud>=1.9.6",
      "emoji>=2.15.0",
      "contractions>=0.1.73",
      "textstat>=0.7.13",
      "rouge>=1.0.1",
      "librosa>=0.11.0",
      "spafe>=0.3.3",
      "praat-parselmouth>=0.4.7",
      "xgboost>=3.2.0",
      "catboost>=1.2.10",
      "lightgbm>=4.0.0",
      "imbalanced-learn>=0.14.1",
      "optuna>=4.8.0",
      "keras-tuner>=1.4.8",
      "libsvm-official>=3.37.0",
      "spams-bin>=2.6.13",
      "skfeature-chappers>=1.1.0",
      "matplotlib>=3.9",
      "seaborn>=0.13.2",
      "plotly>=6.7.0",
      "kaleido>=1.3.0",
      "ptitprince>=0.3.1",
      "squarify>=0.4.4",
      "tqdm>=4.67.3",
      "sympy>=1.14.0",
      "shapely>=2.1.2",
      "trimesh>=4.12.2",
      "pyglet>=2.1.14",
      "split-folders>=0.6.1",
      "gputil>=1.4.0",
      "shutup>=0.3.0",
      "py7zr>=1.1.0",
      "rarfile>=4.2",
      "albumentations>=2.0.8",
      "shap>=0.51.0",
      "grad-cam>=1.5.5",
      "statsmodels>=0.14.6",
      "medmnist>=3.0.2",
      "huggingface-hub>=1.14.0",
      "codecarbon>=3.2.6",
      "av>=17.0.1",
    ],
  },

  classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Software Development :: Libraries :: Python Modules",
  ],
  keywords=[
    "image-processing", "segmentation", "deep-learning", "pytorch",
    "tensorflow", "nlp", "pdf", "utilities", "computer-vision",
    "medical-imaging", "whole-slide-images", "yolo", "transformers",
    "timm", "ultralytics", "pydicom", "nibabel", "openslide",
  ],
)
