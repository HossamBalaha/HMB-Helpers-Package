import time, signal, multiprocessing
import numpy as np
from threading import Thread
from PIL import PngImagePlugin

IMAGE_SUFFIXES = {
  ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
  ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".TIF", ".GIF", ".WEBP"
}


def UpdateMatplotlibSettings():
  r'''
  Update Matplotlib settings for better visualization.
  This function sets various parameters in Matplotlib to improve the appearance of plots.
  It configures figure resolution, font styles, grid settings, and line properties.
  '''

  import matplotlib.pyplot as plt

  # Update Matplotlib settings for better visualization.
  params = {
    "figure.dpi"        : 720,  # Set figure DPI for better resolution.
    "figure.facecolor"  : "white",  # Set figure background color.

    "savefig.dpi"       : 720,  # Set savefig DPI for better resolution.
    "savefig.bbox"      : "tight",  # Save figures with tight bounding box.
    "savefig.pad_inches": 0.05,  # Set padding around saved figures.

    "font.family"       : "serif",  # Use serif font family.
    "font.serif"        : ["Times New Roman"],  # Specify serif font.
    "font.size"         : 12,  # Set default font size.

    "axes.titlesize"    : 16,  # Set title font size.
    "axes.labelsize"    : 14,  # Set axis label font size.
    "axes.grid"         : True,  # Enable grid by default.
    "axes.facecolor"    : "white",  # Set axes background color.

    "xtick.labelsize"   : 12,  # Set x-tick label font size.
    "ytick.labelsize"   : 12,  # Set y-tick label font size.

    "legend.fontsize"   : 12,  # Set legend font size.
    "legend.frameon"    : True,  # Enable legend frame.
    "legend.framealpha" : 0.9,  # Set legend frame transparency.

    "grid.alpha"        : 0.2,  # Set grid transparency.
    "grid.linestyle"    : "-",  # Set grid line style.
    "grid.linewidth"    : 0.5,  # Set grid line width.
    "grid.color"        : "gray",  # Set grid color.
    "grid.which"        : "both",  # Show grid on both major and minor ticks.
    "grid.axis"         : "both",  # Show grid on both x and y axes.
    "grid.zorder"       : 0,  # Set grid z-order to be behind other plot elements.

    "lines.linewidth"   : 2.0,  # Set default line width.
    "lines.markersize"  : 6,  # Set default marker size.
  }
  plt.rcParams.update(params)  # Update Matplotlib rcParams with the defined settings.

  print("Matplotlib settings updated for better visualization.")
  for key, value in params.items():
    print(f"{key}: {value}")


# -------------------------------------------------- #
def SetMaxTextChunkSize(maxChunkSize=100 * (1024 ** 2)):
  r'''
  Set the maximum size for text chunks in PNG images.
  This is useful for controlling the size of metadata stored in PNG files.
  '''

  # Set the maximum size for text chunks in PNG images to 100 MB.
  # This is useful for controlling the size of metadata stored in PNG files.
  # The default value is set to 100 MB (100 * 1024 * 1024 bytes).
  # Adjust this value as needed based on your requirements.
  PngImagePlugin.MAX_TEXT_CHUNK = maxChunkSize


# -------------------------------------------------- #

# -------------------------------------------------- #
def SetEnvironmentVariables(defaultThreads=6):
  r'''
  Set environment variables to control the number of threads used by libraries like OpenBLAS, MKL, and OMP.
  This is useful for optimizing performance and avoiding excessive CPU usage.
  '''

  import os

  # Set the number of threads for OpenBLAS, MKL, and OMP to a default value.
  # This helps in controlling the parallelism and resource usage of these libraries.
  # Adjust the value as needed based on your system's capabilities.
  os.environ["OPENBLAS_NUM_THREADS"] = f"{defaultThreads}"
  os.environ["MKL_NUM_THREADS"] = f"{defaultThreads}"
  os.environ["OMP_NUM_THREADS"] = f"{defaultThreads}"
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  # Maximize the number of threads used by PyTorch.
  # This is useful for optimizing performance on multi-core CPUs.
  os.environ["PYTORCH_MAX_NUM_THREADS"] = f"{defaultThreads}"
  # Set the number of threads for NumPy to a default value.
  # This helps in controlling the parallelism and resource usage of NumPy.
  os.environ["NUMEXPR_NUM_THREADS"] = f"{defaultThreads}"


def MaximizeThreads():
  r'''
  Maximize the number of threads used by libraries like OpenBLAS, MKL, and OMP.
  This is useful for optimizing performance on multi-core CPUs.
  '''

  # Get the maximum number of threads available on the system.
  # This is useful for optimizing performance on multi-core CPUs.
  maxThreads = multiprocessing.cpu_count() - 1
  # Set the number of threads for OpenBLAS, MKL, and OMP to the maximum value.
  SetEnvironmentVariables(defaultThreads=maxThreads)


# -------------------------------------------------- #

# -------------------------------------------------- #
def IgnoreWarnings():
  r'''
  Suppress all warnings globally.
  This function uses the `shutup` library to suppress warnings from various libraries.
  '''

  import warnings, shutup, os

  # Suppress all warnings using the `shutup` library.
  shutup.please()

  # Suppress all warnings globally.
  warnings.filterwarnings("ignore")

  # Patch warnings.warn to ignore 'skip_file_prefixes' kwarg if present.
  _orig_warn = warnings.warn

  def warn(*args, **kwargs):
    kwargs.pop("skip_file_prefixes", None)
    return _orig_warn(*args, **kwargs)

  warnings.warn = warn

  # Disable the TF warnings.
  # This is useful for TensorFlow users to suppress warnings related to TensorFlow.
  # Set the logging level to ERROR to suppress all warnings.
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings.
  os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # Suppress TensorFlow verbose logging.

  # Suppress all warnings using the `shutup` library.
  shutup.please()


# -------------------------------------------------- #

# -------------------------------------------------- #
def SeedEverything(seed=42, deterministic=True, benchmark=True):
  r'''
  Seed everything for reproducibility.
  This function sets the random seed for various libraries to ensure consistent results across runs.

  Parameters:
    seed (int): The random seed to use for seeding.
    deterministic (bool): If True, sets cuDNN to be deterministic.
    benchmark (bool): If True, enables cuDNN benchmark mode.
  '''

  import torch
  import numpy as np
  import random, os, torch

  # Set the random seed for reproducibility.
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)  # Set the random seed for Python's built-in random module to ensure reproducibility.
  np.random.seed(seed)  # Set the random seed for NumPy to ensure reproducibility.
  torch.manual_seed(seed)  # Set the random seed for PyTorch to ensure reproducibility.
  torch.cuda.manual_seed(seed)  # Set the random seed for CUDA to ensure reproducibility on GPU.
  torch.cuda.manual_seed_all(seed)  # Set the random seed for all GPUs to ensure reproducibility across multiple GPUs.

  # If cuDNN is available, set the random seed for cuDNN to ensure reproducibility.
  # This is useful for models that use cuDNN for optimized performance on NVIDIA GPUs.
  if (torch.backends.cudnn.is_available()):
    # Set the random seed for cuDNN to ensure reproducibility.
    # This is useful for models that use cuDNN for optimized performance on NVIDIA GPUs.
    # Setting this to True ensures that the same operations will produce the same results,
    # even if the hardware or software configuration changes.
    # This is useful for debugging and testing purposes.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    # However, it may lead to slower performance in some cases.
    # If you want to disable this behavior, set it to False.
    # torch.backends.cudnn.deterministic = False
    # PyTorch uses cuDNN for optimized performance on NVIDIA GPUs.
    # Enabling torch.backends.cudnn.benchmark allows cuDNN to find the optimal algorithms
    # for your specific model and hardware configuration.
    # This allows cuDNN to select the best algorithms for the hardware and input sizes,
    # which can improve performance.
    torch.backends.cudnn.benchmark = benchmark
    # However, setting this to True may lead to non-deterministic results,
    # especially if the input sizes change frequently.
    # If you want to ensure reproducibility, set it to False.
    # torch.backends.cudnn.benchmark = False


def DoRandomSeeding():
  r'''
  Perform random seeding for reproducibility.
  This function sets the random seed for various libraries to ensure consistent results across runs.
  '''

  # Set the random seed for reproducibility.
  maxInt = np.iinfo(np.int32).max
  rndNumber = np.random.randint(0, maxInt)
  SeedEverything(seed=rndNumber)
  print(f"Random seed set to: {rndNumber}")


# -------------------------------------------------- #


# -------------------------------------------------- #
def ShowGPUUtilization():
  r'''
  Show GPU utilization using GPUtil. This function displays the current GPU usage statistics.
  '''

  import GPUtil
  GPUtil.showUtilization(all=False, attrList=None, useOldCode=False)


class ShowGPUUtilizationThread(Thread):
  def __init__(self, interval=120):
    super().__init__()
    self.interval = interval
    # Set the thread as a daemon thread.
    # Daemon threads are terminated when the main program exits.
    # This is useful for background tasks that should not block the program from exiting.
    self.daemon = True
    self.isRunning = True

    signal.signal(signal.SIGINT, self.stop)  # Handle Ctrl+C gracefully.
    signal.signal(signal.SIGTERM, self.stop)  # Handle termination signals gracefully.

  def __del__(self):
    self.stop()  # Ensure the thread stops when deleted.

  def run(self):
    while (True):
      ShowGPUUtilization()
      time.sleep(self.interval)
      if (not self.isRunning):
        break

  def stop(self):
    self.isRunning = False


def PrintGPUSpecs():
  import GPUtil, torch

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device:", DEVICE)
  print("CUDA version:", torch.version.cuda)  # Current CUDA version.
  # Get current GPU usage.
  print(f"Found {len(GPUtil.getGPUs())} GPU(s) available.")
  for gpu in GPUtil.getGPUs():
    print(f"GPU [{gpu.id}]:")
    print(f"- Name: {gpu.name}")
    print(f"- Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB.")
    print(f"- Utilization: {gpu.load * 100:.2f}%.")
    print(f"- Temperature: {gpu.temperature}°C.")
    print(f"- Free memory: {gpu.memoryFree}MB.")
    print(f"- Total memory: {gpu.memoryTotal}MB.")
    print(f"- GPU Driver Version: {gpu.driver}.")
    print(f"- GPU Serial: {gpu.serial}.")
    print(f"- GPU UUID: {gpu.uuid}.")


def EnsureCUDAAvailable(strict=True, which="pytorch"):
  r'''
  Ensure that CUDA is available for GPU computations.
  This function checks if CUDA is available and exits the program if not.

  Parameters:
    strict (bool): If True, the program will exit if CUDA is not available.

  Returns:
    bool: True if CUDA is available, False otherwise.
  '''

  import sys

  if (which.lower() == "pytorch"):
    import torch

    try:
      if (not torch.cuda.is_available()):
        print("ERROR: CUDA is not available. A GPU is required for training.", flush=True)
        if (strict):
          sys.exit(1)
        else:
          return False
    except Exception:
      print("ERROR: PyTorch is not installed or failed to import.", flush=True)
      if (strict):
        sys.exit(1)
      else:
        return False

    print("Current device:", torch.cuda.get_device_name(0))
    return True
  elif (which.lower() == "tensorflow"):
    import tensorflow as tf

    try:
      if (not tf.config.list_physical_devices("GPU")):
        print("ERROR: CUDA is not available. A GPU is required for training.", flush=True)
        if (strict):
          sys.exit(1)
        else:
          return False
    except Exception:
      print("ERROR: TensorFlow is not installed or failed to import.", flush=True)
      if (strict):
        sys.exit(1)
      else:
        return False

    print("Current device:", tf.config.list_physical_devices("GPU"))
    return True
  else:
    print(f"ERROR: Unsupported library '{which}'. Supported libraries are 'pytorch' and 'tensorflow'.", flush=True)
    if (strict):
      sys.exit(1)
    else:
      return False


# -------------------------------------------------- #

# -------------------------------------------------- #
def NLTKFindDownload(resource):
  r'''
  Find and download the specified NLTK resource if it is not already available.

  Parameters:
    resource (str): The name of the NLTK resource to find and download (e.g., "punkt", "stopwords", "wordnet").
  '''

  import nltk

  mapping = {
    "punkt"                         : "tokenizers/punkt",
    "stopwords"                     : "corpora/stopwords",
    "wordnet"                       : "corpora/wordnet",
    "averaged_perceptron_tagger"    : "taggers/averaged_perceptron_tagger",
    "omw-1.4"                       : "corpora/omw-1.4",
    "punkt_tab"                     : "tokenizers/punkt_tab",
    "words"                         : "corpora/words",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
  }

  if (resource not in mapping):
    print(f"Resource '{resource}' is not recognized. Please check the resource name.")
    return

  try:
    nltk.data.find(mapping[resource])
    print(f"NLTK resource '{resource}' is already available.")
  except LookupError:
    print(f"NLTK resource '{resource}' not found. Downloading...")
    nltk.download(resource)
    print(f"NLTK resource '{resource}' downloaded successfully.")


def DownloadNLTKPackages():
  r'''
  Download necessary NLTK packages for text processing.
  This function ensures that the required NLTK resources are available for natural language processing tasks.
  '''

  import nltk

  # Download necessary NLTK resources for tokenization and POS tagging.
  # This is useful for natural language processing tasks.
  NLTKFindDownload("punkt")  # Tokenizer for splitting text into sentences and words.
  NLTKFindDownload("stopwords")  # List of common stop words in various languages.
  NLTKFindDownload("wordnet")  # WordNet lexical database for English.
  NLTKFindDownload("averaged_perceptron_tagger")  # Part-of-speech tagger.
  NLTKFindDownload("omw-1.4")  # Open Multilingual WordNet.
  NLTKFindDownload("punkt_tab")  # For tokenization.
  NLTKFindDownload("words")  # For tokenization.
  NLTKFindDownload("averaged_perceptron_tagger_eng")  # For POS tagging.
  print("NLTK packages downloaded successfully.")


def IncreaseSysRecursionLimit(limit=10000):
  r'''
  Increase the system recursion limit to allow deeper recursive calls.
  This is useful for algorithms that require deep recursion, such as certain tree or graph algorithms.

  Parameters:
    limit (int): The new recursion limit to set.
  '''

  import sys

  # Set the maximum recursion depth to a higher value.
  # This allows for deeper recursive calls without hitting the recursion limit.
  sys.setrecursionlimit(int(limit))
  print(f"System recursion limit increased to {limit}.")


# -------------------------------------------------- #


# -------------------------------------------------- #
def LoadEnglishModelSpacy(modelName="en_core_web_sm"):
  r'''
  Load the specified English model for spaCy.
  This function loads the specified spaCy model for natural language processing tasks.

  Parameters:
    modelName (str): The name of the spaCy model to load (e.g., "en_core_web_sm", "en_core_web_md", "en_core_web_lg").
  '''

  import spacy

  try:
    nlp = spacy.load(modelName)
    print(f"spaCy model '{modelName}' loaded successfully.")
    return nlp
  except OSError:
    print(f"spaCy model '{modelName}' not found. Please download it using 'python -m spacy download {modelName}'.")
    return None

# -------------------------------------------------- #


# -------------------------------------------------- #


# To initialize the environment, call the functions:
# IgnoreWarnings()  # Suppress all warnings globally.
# DownloadNLTKPackages()  # Download necessary NLTK packages for text processing.
# SetMaxTextChunkSize(maxChunkSize=100 * (1024 ** 2))  # Set the maximum size for text chunks in PNG images.
# MaximizeThreads()  # Set the maximum number of threads available on the system.
# DoRandomSeeding()  # Set the random seed for reproducibility.
# ShowGPUUtilization()  # Show GPU utilization at the start.

# # -------------------------------------------------- #
# # Suppress all warnings globally.
# IgnoreWarnings()
# print("All warnings should be suppressed.")
# print("Downloading NLTK packages...")
# # Download necessary NLTK packages for text processing.
# DownloadNLTKPackages()
# # Maximum integer value for 32-bit integers.
# maxInt = np.iinfo(np.int32).max
# # Set the random seed for reproducibility.
# rndNumber = np.random.randint(0, maxInt)
# SeedEverything(seed=rndNumber)
# print("Random seed set to:", rndNumber)
# # Set the maximum size for text chunks in PNG images.
# SetMaxTextChunkSize(maxChunkSize=100 * (1024 ** 2))
# print("Maximum text chunk size set to 100 MB.")
# # Get the maximum number of threads available on the system.
# # Leave one thread free for the OS.
# defaultThreads = multiprocessing.cpu_count() - 1
# print("Setting default threads to:", defaultThreads)
# # Set environment variables for thread control.
# SetEnvironmentVariables(defaultThreads=defaultThreads)
# # Show GPU utilization at the start.
# # ShowGPUUtilization()
# # -------------------------------------------------- #
