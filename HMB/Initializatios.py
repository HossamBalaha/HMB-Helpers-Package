'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Sep 2024
# Last Modification Date: Aug 2nd, 2025
# Permissions and Citation: Refer to the README file.
'''

# -------------------------------------------------- #
import shutup

# Suppress warnings and output from libraries.
shutup.please()
# -------------------------------------------------- #

# -------------------------------------------------- #
from PIL import PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
# -------------------------------------------------- #

# -------------------------------------------------- #
import os

defaultThreads = 6
os.environ["OPENBLAS_NUM_THREADS"] = f"{defaultThreads}"
os.environ["MKL_NUM_THREADS"] = f"{defaultThreads}"
os.environ["OMP_NUM_THREADS"] = f"{defaultThreads}"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -------------------------------------------------- #

# -------------------------------------------------- #
import random, torch, os, matplotlib
import numpy as np
import torch.backends.cudnn as cudnn


def SeedEverything(seed=42):
  # Set the random seed for reproducibility.
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  # PyTorch uses cuDNN for optimized performance on NVIDIA GPUs.
  # Enabling torch.backends.cudnn.benchmark allows cuDNN to find the optimal algorithms
  # for your specific model and hardware configuration.
  torch.backends.cudnn.benchmark = True


SeedEverything(seed=np.random.randint(0, 10000))
# matplotlib.use("Agg")
# -------------------------------------------------- #

# -------------------------------------------------- #
import GPUtil, time, signal
from threading import Thread


def ShowGPUUtilization():
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

# ShowGPUUtilization()  # Show GPU utilization at the start.
# -------------------------------------------------- #

# RuntimeError: Input type (unsigned char) and bias type (c10::Half) should be the same
# Solution: Convert the input to float32 before passing it to the model.

# print("Initialization complete. All settings are configured.")
