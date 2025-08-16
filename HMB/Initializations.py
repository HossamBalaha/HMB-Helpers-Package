'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Sep 2024
# Last Modification Date: Aug 16th, 2025
# Permissions and Citation: Refer to the README file.
'''

import os, time, signal, multiprocessing
import numpy as np
from threading import Thread
from PIL import PngImagePlugin


# -------------------------------------------------- #
def SetMaxTextChunkSize(maxChunkSize=100 * (1024 ** 2)):
  '''
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
  '''
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


# -------------------------------------------------- #

# -------------------------------------------------- #
def IgnoreWarnings():
  import warnings, shutup

  # Suppress all warnings globally.
  warnings.filterwarnings("ignore")

  def warn(*args, **kwargs):
    '''
    Custom function to suppress warnings.
    This function overrides the default warning behavior to suppress all warnings.
    '''
    pass

  # Override the default warning function with the custom function.
  warnings.warn = warn
  # Suppress all warnings using the `shutup` library.
  shutup.please()


# -------------------------------------------------- #

# -------------------------------------------------- #
def SeedEverything(seed=42, deterministic=True, benchmark=True):
  import random, os, torch
  import numpy as np
  import torch

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


# -------------------------------------------------- #


# -------------------------------------------------- #
def ShowGPUUtilization():
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


# -------------------------------------------------- #


# -------------------------------------------------- #
# Suppress all warnings globally.
IgnoreWarnings()
print("All warnings should be suppressed.")
# Maximum integer value for 32-bit integers.
maxInt = np.iinfo(np.int32).max
# Set the random seed for reproducibility.
rndNumber = np.random.randint(0, maxInt)
SeedEverything(seed=rndNumber)
print("Random seed set to:", rndNumber)
# Set the maximum size for text chunks in PNG images.
SetMaxTextChunkSize(maxChunkSize=100 * (1024 ** 2))
print("Maximum text chunk size set to 100 MB.")
# Get the maximum number of threads available on the system.
# Leave one thread free for the OS.
defaultThreads = multiprocessing.cpu_count() - 1
print("Setting default threads to:", defaultThreads)
# Set environment variables for thread control.
SetEnvironmentVariables(defaultThreads=defaultThreads)
# Show GPU utilization at the start.
# ShowGPUUtilization()
# -------------------------------------------------- #
