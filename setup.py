from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
  name="hmb-helpers",
  version="0.1.0",
  author="Hossam Magdy Balaha",
  author_email="h3ossam@gmail.com",
  description="A comprehensive suite of helper modules for image processing, text generation, PDF handling, and ML workflows.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/HossamBalaha/HMB-Helpers-Package",
  project_urls={
    "Homepage"     : "https://github.com/HossamBalaha/HMB-Helpers-Package",
    "Bug Tracker"  : "https://github.com/HossamBalaha/HMB-Helpers-Package/issues",
    "Documentation": "https://hmb-helpers-package.readthedocs.io/en/latest/",
    "Source Code"  : "https://github.com/HossamBalaha/HMB-Helpers-Package",
  },
  license="MIT",
  license_files=["LICENSE"],
  packages=find_packages(exclude=["tests", "tests.*", "HMB.Examples", "HMB.Examples.*"]),
  package_data={"HMB": ["py.typed"]},
  include_package_data=True,
  zip_safe=False,
  python_requires=">=3.8",

  # ✅ Minimal core: only what's needed for basic imports
  install_requires=[
    "numpy<2",
    "pillow>=9.0.0",
  ],

  # ✅ Optional feature groups
  extras_require={
    # Core scientific stack
    "scientific"  : [
      "scipy>=1.7.0",
      "pandas>=1.3.0",
      "scikit-learn>=1.0.0",
      "scikit-image>=0.19.0",
    ],

    # Image processing & computer vision
    "cv"          : [
      "opencv-python>=4.5.0",
      "opencv-contrib-python>=4.5.0",
      "imagehash>=4.2.0",
      "brisque>=0.0.17",
      "simpleitk>=2.0.0",
      "PyWavelets>=1.1.0",
      "pyvips>=2.1.0",
      "patchify>=0.2.0",
    ],

    # Deep learning frameworks (CPU defaults; users install CUDA separately)
    "pytorch"     : [
      "torch>=2.0.0",
      "torchvision>=0.15.0",
      "torchaudio>=2.0.0",
    ],
    "tensorflow"  : [
      "tensorflow>=2.12.0",
      "keras>=2.12.0",
      "tf-keras>=2.12.0",
    ],

    # Vision models & pretrained weights
    "timm"        : ["timm>=0.9.0"],
    "transformers": ["transformers>=4.30.0", "sentence-transformers>=2.2.0"],
    "ultralytics" : ["ultralytics>=8.0.0"],

    # Medical imaging & WSI
    "medical"     : [
      "pydicom>=2.3.0",
      "nibabel>=5.0.0",
      "openslide-python>=1.1.2",
    ],

    # PDF handling
    "pdf"         : [
      "PyMuPDF>=1.22.0",
      "PyPDF2>=3.0.0",
      "tabula-py>=2.7.0",
      "jpype1>=1.4.0",
    ],

    # NLP & text processing
    "nlp"         : [
      "nltk>=3.8.0",
      "spacy>=3.5.0",
      "textblob>=0.17.0",
      "gensim>=4.3.0",
      "qalsadi>=0.4.0",
      "langdetect>=1.0.9",
      "gtts>=2.3.0",
      "wordcloud>=1.9.0",
      "emoji>=2.0.0",
      "contractions>=0.1.0",
      "textstat>=0.7.0",
      "rouge>=1.0.0",
    ],

    # Audio processing
    "audio"       : [
      "librosa>=0.10.0",
      "spafe>=0.3.0",
      "praat-parselmouth>=0.4.0",
    ],

    # Classical ML & optimization
    "ml"          : [
      "xgboost>=1.7.0",
      "catboost>=1.2.0",
      "lightgbm>=4.0.0",
      "imbalanced-learn>=0.11.0",
      "optuna>=3.0.0",
      "keras-tuner>=1.3.0",
      "libsvm-official>=3.30.0",
      "spams-bin>=2.6.0",
      "skfeature-chappers>=1.0.0",
    ],

    # Visualization & plotting
    "plotting"    : [
      "matplotlib>=3.7.0",
      "seaborn>=0.12.0",
      "plotly>=5.15.0",
      "kaleido>=0.2.0",
      "ptitprince>=0.2.0",
      "squarify>=0.4.0",
    ],

    # Utilities & compression
    "utils"       : [
      "tqdm>=4.65.0",
      "sympy>=1.11.0",
      "shapely>=2.0.0",
      "trimesh>=3.0.0",
      "pyglet>=1.5.0",
      "split-folders>=0.5.0",
      "gputil>=1.4.0",
      "shutup>=0.2.0",
      "py7zr>=0.20.0",
      "rarfile>=4.0.0",
      "albumentations>=1.3.0",
      "shap>=0.42.0",
      "grad-cam>=1.4.0",
      "statsmodels>=0.14.0",
      "medmnist>=3.0.0",
      "huggingface-hub>=0.16.0",
    ],

    # Development & testing
    "dev"         : [
      "pytest>=7.0.0",
      "sphinx>=4.0.0",
      "black>=22.0.0",
      "flake8>=4.0.0",
      "mypy>=1.0.0",
    ],

    # Documentation building
    "docs"        : [
      "sphinx>=4.0.0",
      "sphinx-rtd-theme>=1.0.0",
      "myst-parser>=0.18.0",
    ],

    # ✅ Install everything (for full environment setup)
    "all"         : [
      "scipy>=1.7.0",
      "pandas>=1.3.0",
      "scikit-learn>=1.0.0",
      "scikit-image>=0.19.0",
      "opencv-python>=4.5.0",
      "opencv-contrib-python>=4.5.0",
      "imagehash>=4.2.0",
      "brisque>=0.0.17",
      "simpleitk>=2.0.0",
      "PyWavelets>=1.1.0",
      "pyvips>=2.1.0",
      "patchify>=0.2.0",
      "torch>=2.0.0",
      "torchvision>=0.15.0",
      "torchaudio>=2.0.0",
      "tensorflow>=2.12.0",
      "keras>=2.12.0",
      "tf-keras>=2.12.0",
      "timm>=0.9.0",
      "transformers>=4.30.0",
      "sentence-transformers>=2.2.0",
      "ultralytics>=8.0.0",
      "pydicom>=2.3.0",
      "nibabel>=5.0.0",
      "openslide-python>=1.1.2",
      "PyMuPDF>=1.22.0",
      "PyPDF2>=3.0.0",
      "tabula-py>=2.7.0",
      "jpype1>=1.4.0",
      "nltk>=3.8.0",
      "spacy>=3.5.0",
      "textblob>=0.17.0",
      "gensim>=4.3.0",
      "qalsadi>=0.4.0",
      "langdetect>=1.0.9",
      "gtts>=2.3.0",
      "wordcloud>=1.9.0",
      "emoji>=2.0.0",
      "contractions>=0.1.0",
      "textstat>=0.7.0",
      "rouge>=1.0.0",
      "librosa>=0.10.0",
      "spafe>=0.3.0",
      "praat-parselmouth>=0.4.0",
      "xgboost>=1.7.0",
      "catboost>=1.2.0",
      "lightgbm>=4.0.0",
      "imbalanced-learn>=0.11.0",
      "optuna>=3.0.0",
      "keras-tuner>=1.3.0",
      "libsvm-official>=3.30.0",
      "spams-bin>=2.6.0",
      "skfeature-chappers>=1.0.0",
      "matplotlib>=3.7.0",
      "seaborn>=0.12.0",
      "plotly>=5.15.0",
      "kaleido>=0.2.0",
      "ptitprince>=0.2.0",
      "squarify>=0.4.0",
      "tqdm>=4.65.0",
      "sympy>=1.11.0",
      "shapely>=2.0.0",
      "trimesh>=3.0.0",
      "pyglet>=1.5.0",
      "split-folders>=0.5.0",
      "gputil>=1.4.0",
      "shutup>=0.2.0",
      "py7zr>=0.20.0",
      "rarfile>=4.0.0",
      "albumentations>=1.3.0",
      "shap>=0.42.0",
      "grad-cam>=1.4.0",
      "statsmodels>=0.14.0",
      "medmnist>=3.0.0",
      "huggingface-hub>=0.16.0",
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
