'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Sep 2024
# Last Modification Date: Jul 31th, 2025
# Permissions and Citation: Refer to the README file.
'''

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
import shutup

# Suppress warnings and output from libraries.
shutup.please()
# -------------------------------------------------- #


# print("Initialization complete. All settings are configured.")
