import os, timm, torch, tqdm, copy, time, cv2, json, hashlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torch.nn.functional as F
from HMB.Initializations import IMAGE_SUFFIXES
from HMB.PerformanceMetrics import (
  CalculatePerformanceMetrics, PlotConfusionMatrix, HistoryPlotter,
  PlotROCAUCCurve, PlotPRCCurve, ComputeECE, ComputeBrierScore
)
from HMB.Utils import DumpJsonFile, AppendOrCreateNewCSV
from HMB.PlotsHelper import PlotHeatmap, PlotBarChart, COLORS
from HMB.ImagesHelper import *


def GetParamCount(model):
  r'''
  Get the total number of trainable parameters in a PyTorch model.

  Parameters:
    model (torch.nn.Module): The model to count parameters for.

  Returns:
    int: Total number of trainable parameters.
  '''

  # Sum the number of elements for all trainable parameters.
  return sum(p.numel() for p in model.parameters() if (p.requires_grad))


# Function to save a PyTorch model's state dictionary to a file.
def SaveModel(model, filename="model.pth"):
  r'''
  Save the model state to a file for later use. You can load it later
  using LoadModel() function from this module.

  Parameters:
    model (torch.nn.Module): The model to save.
    filename (str): The name of the file to save the model to.
  '''

  # Save the model's state dictionary to the specified file.
  torch.save(model.state_dict(), filename)

  # Print confirmation message with filename.
  print(f"Model saved to {filename}. You can load it later using LoadModel().")


def SavePyTorchDict(modelDict, filename="model.pth"):
  r'''
  Save a PyTorch state dictionary to a file for later use. You can load it later
  using LoadPyTorchDict function.

  Parameters:
    modelDict (dict): The state dictionary to save.
    filename (str): The name of the file to save the state dictionary to.
  '''

  # Save the state dictionary to the specified file.
  torch.save(modelDict, filename)

  # Print confirmation message with filename.
  print(f"State dictionary saved to {filename}. You can load it later using torch.load().")


# Function to load a PyTorch model's state dictionary from a file and move it to a device.
def LoadModel(model, filename="model.pth", device="cuda", weightsOnly=False):
  r'''
  Load the model state from a file and move it to the specified device.

  Parameters:
    model (torch.nn.Module): The model to load the state into.
    filename (str): The name of the file to load the model from.
    device (str): The device to load the model onto (e.g., "cpu" or "cuda").
    weightsOnly (bool): If True, only load the weights without strict key matching.

  Returns:
    torch.nn.Module: The model with loaded state.
  '''

  # Check if the model file exists before loading.
  if (not os.path.exists(filename)):
    print(f"Model file not found: {filename}")
    return

  # Load the state dictionary from a file and map it to the specified device.
  stateDict = LoadPyTorchDict(filename, device=device, weightsOnly=weightsOnly)

  if ((stateDict) and (isinstance(stateDict, dict)) and ("model_state_dict" in stateDict)):
    model.load_state_dict(stateDict["model_state_dict"])
  else:
    model.load_state_dict(stateDict)

  # Move the model to the specified device.
  model.to(device)

  # Print confirmation message with filename and device.
  print(f"Model loaded from {filename} and moved to {device}.")

  return model


def LoadPyTorchDict(filename="model.pth", device="cuda", weightsOnly=False):
  r'''
  Load a PyTorch state dictionary from a file and map it to the specified device.

  Parameters:
    filename (str): The name of the file to load the state dictionary from.
    device (str): The device to map the state dictionary onto (e.g., "cpu" or "cuda").
    weightsOnly (bool): If True, only load the weights without strict key matching.

  Returns:
    dict: The loaded state dictionary.
  '''

  # Check if the state dictionary file exists before loading.
  if (not os.path.exists(filename)):
    print(f"State dictionary file not found: {filename}")
    return None

  # Load the state dictionary from file and map to the specified device.
  # Set weights_only=False to allow loading full objects (required for PyTorch >=2.6).
  stateDict = torch.load(filename, map_location=device, weights_only=weightsOnly)

  # Print confirmation message with filename and device.
  print(f"State dictionary loaded from {filename} and mapped to {device}.")

  return stateDict


def SaveCheckpoint(model, optimizer, filename="chk.pth.tar", epoch=None, hparams=None):
  r'''
  Save model and optimizer state to a checkpoint file.
  Useful for resuming training or inference later.
  This function saves the model's state dictionary and the optimizer's state dictionary
  to a specified file. You can load it later using `LoadCheckpoint` function from this module.

  Parameters:
    model (torch.nn.Module): The model to save.
    optimizer (torch.optim.Optimizer): The optimizer to save.
    filename (str): The name of the file to save the checkpoint to.
    epoch (int, optional): The current epoch number to save in the checkpoint.
    hparams (dict, optional): Hyperparameters to save in the checkpoint.
  '''

  # Create a dictionary containing model and optimizer state.
  checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer" : optimizer.state_dict(),
  }
  if (epoch is not None):
    checkpoint["epoch"] = epoch
  if (hparams is not None):
    checkpoint["hparams"] = hparams
  # Save the checkpoint dictionary to the specified file.
  torch.save(checkpoint, filename)

  # Print confirmation message with filename.
  print(f"Checkpoint saved to {filename}. You can load it later using `LoadCheckpoint`.")


def LoadCheckpoint(checkpointFile, model, optimizer, lr, device, strict=True):
  r'''
  Load model and optimizer state from a checkpoint file.
  Updates the learning rate of the optimizer if provided.
  This function loads the model's state dictionary and the optimizer's state dictionary
  from a specified checkpoint file. It also updates the learning rate of the optimizer.

  Parameters:
    checkpointFile (str): The path to the checkpoint file.
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    lr (float): The learning rate to set for the optimizer.
    device (torch.device): The device to load the model onto (e.g., "cpu" or "cuda").
    strict (bool): Whether to strictly enforce that the keys in the state dictionary match the model's keys when loading.

  Returns:
    dict: The loaded checkpoint dictionary.
  '''

  # Check if the checkpoint file exists before loading.
  if (not os.path.exists(checkpointFile)):
    print(f"Checkpoint file not found: {checkpointFile}")
    return

  # Load the checkpoint dictionary from file and map to the specified device.
  checkpoint = torch.load(checkpointFile, map_location=device)
  # Load the model state from the checkpoint.
  model.load_state_dict(checkpoint["state_dict"], strict=strict)

  # If optimizer is provided, load its state and update learning rate.
  if (optimizer is not None):
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update learning rate for all parameter groups in the optimizer.
    for paramGroup in optimizer.param_groups:
      paramGroup["lr"] = lr

  # Print confirmation message with checkpoint file and device.
  print(f"Checkpoint loaded from {checkpointFile} and model moved to {device}.")

  return checkpoint


def GetOptimizer(model, optimizerType="adamw", learningRate=1e-4, weightDecay=1e-4):
  r'''
  Create and return a PyTorch optimizer for the given model.

  Parameters:
    model (torch.nn.Module): The model whose parameters to optimize.
    optimizerType (str): The type of optimizer ("adamw", "adam", "sgd", "rmsprop", "adadelta").
    learningRate (float): Learning rate for the optimizer.
    weightDecay (float): Weight decay (L2 penalty).

  Returns:
    torch.optim.Optimizer: The created optimizer.
  '''

  import torch.optim as optim

  # Create an optimizer for the model parameters.
  if (optimizerType.lower() == "adamw"):
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weightDecay)
  elif (optimizerType.lower() == "adam"):
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
  elif (optimizerType.lower() == "sgd"):
    optimizer = optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
  elif (optimizerType.lower() == "rmsprop"):
    optimizer = optim.RMSprop(model.parameters(), lr=learningRate, weight_decay=weightDecay)
  elif (optimizerType.lower() == "adadelta"):
    optimizer = optim.Adadelta(model.parameters(), lr=learningRate, weight_decay=weightDecay)
  else:
    raise ValueError(f"Unsupported optimizer type: {optimizerType}")
  return optimizer


class CustomDataset(torch.utils.data.Dataset):
  r'''
  PyTorch dataset for image classification tasks, loading images from a directory
  structure where each class has its own subfolder.

  Parameters:
    dataDir (str): Path to the root directory containing class subfolders with images.
    transform (callable, optional): Optional transform to be applied on a sample.
    allowedExtensions (tuple, optional): Tuple of allowed image file extensions. Defaults to (".png", ".jpg", ".jpeg").
  '''

  def __init__(
    self,
    dataDir,
    transform=None,
    allowedExtensions=tuple(IMAGE_SUFFIXES)
  ):
    r'''
    Initialize the custom dataset for image classification tasks.

    Parameters:
      dataDir (str): Path to the root directory containing class subfolders with images.
      transform (callable, optional): Optional transform to be applied on a sample.
      allowedExtensions (tuple, optional): Tuple of allowed image file extensions. Defaults to (".png", ".jpg", ".jpeg").
    '''

    # Setting data directory.
    self.dataDir = dataDir
    # Setting transform.
    self.transform = transform
    # Getting classes.
    self.classes = sorted(os.listdir(dataDir))
    # Creating class to index mapping.
    self.classToIdx = {}
    # Initializing samples list.
    self.samples = []
    for idx, cls in enumerate(self.classes):
      self.classToIdx[cls] = idx
      clsDir = os.path.join(dataDir, cls)
      if (not os.path.isdir(clsDir)):
        continue
      for fname in os.listdir(clsDir):
        if (fname.lower().endswith(allowedExtensions)):
          # Getting image path.
          path = os.path.join(clsDir, fname)
          self.samples.append((path, idx))

  def __len__(self):
    r'''
    Get the total number of samples in the dataset.

    Returns:
      int: Number of samples in the dataset.
    '''

    return len(self.samples)

  def __getitem__(self, idx):
    r'''
    Retrieve an image and its label by index.

    Parameters:
      idx (int): Index of the sample to retrieve.

    Returns:
      tuple: (image, label) where image is a PIL Image or transformed tensor, and label is an int class index.
    '''

    path, label = self.samples[idx]
    img = Image.open(path).convert("RGB")
    if (self.transform):
      img = self.transform(img)
    return img, label


def CreateTimmModel(modelName, numClasses, pretrained=True):
  r'''
  Create a classification model using the timm library.

  Parameters:
    modelName (str): Name of the model architecture to create (e.g., 'resnet18', 'efficientnet_b0').
    numClasses (int): Number of output classes for the classification task.
    pretrained (bool): If True, loads pretrained weights. Defaults to True.

  Returns:
    torch.nn.Module: The created timm model instance.
  '''

  # Creating model.
  model = timm.create_model(
    modelName,  # Model architecture name.
    pretrained=pretrained,  # Whether to load pretrained weights.
    num_classes=numClasses,  # Number of output classes.
  )
  # Returning model.
  return model


def MixupFn(inputs, targets, alpha=0.4, numClasses=None):
  r'''
  Apply MixUp data augmentation to inputs and targets.
  This function mixes pairs of samples in the batch using a Beta distribution to create new training examples.
  Useful for improving model generalization and robustness.

  Parameters:
    inputs (torch.Tensor): Input data of shape (batch_size, ...).
    targets (torch.Tensor): Target labels of shape (batch_size,) or (batch_size, num_classes).
    alpha (float): MixUp alpha parameter for Beta distribution. Default is 0.4. If alpha > 0.5, stronger mixing is applied.
    numClasses (int, optional): Number of classes for one-hot encoding if targets are indices.

  Returns:
    tuple: Mixed inputs and mixed targets.
      - mixedInputs (torch.Tensor): Mixed input data.
      - mixedTargets (torch.Tensor): Mixed target labels.
  '''

  if (alpha <= 0):
    # If no mixup, ensure targets are one-hot floats if possible.
    if (targets.dim() == 1):
      if (numClasses is None):
        numClasses = int(targets.max().item()) + 1
      return inputs, F.one_hot(targets, num_classes=numClasses).float().to(inputs.device)
    else:
      return inputs, targets.float().to(inputs.device)

  lam = float(np.random.beta(alpha, alpha))
  batchSize = inputs.size(0)
  index = torch.randperm(batchSize, device=inputs.device)

  mixedInputs = inputs * lam + inputs[index] * (1.0 - lam)

  # Convert index targets to one-hot if needed, else assume targets already soft/one-hot.
  if (targets.dim() == 1):
    if (numClasses is None):
      numClasses = int(targets.max().item()) + 1
    tA = F.one_hot(targets, num_classes=numClasses).float().to(inputs.device)
    tB = F.one_hot(targets[index], num_classes=numClasses).float().to(inputs.device)
  else:
    # Targets already one-hot / soft.
    tA = targets.float().to(inputs.device)
    tB = targets[index].float().to(inputs.device)

  mixedTargets = tA * lam + tB * (1.0 - lam)
  return mixedInputs, mixedTargets


def MixupCriterion(logits, softTargets):
  r'''
  Compute the MixUp loss given logits and soft targets.

  Parameters:
    logits (torch.Tensor): Model output logits of shape (batch_size, num_classes).
    softTargets (torch.Tensor): Soft target labels of shape (batch_size, num_classes).

  Returns:
    torch.Tensor: Computed MixUp loss.
  '''

  logProbs = F.log_softmax(logits, dim=1)
  loss = - (softTargets * logProbs).sum(dim=1).mean()
  return loss


class ExponentialMovingAverage(object):
  r'''
  Implements Exponential Moving Average (EMA) for model parameters.

  Parameters:
    model (torch.nn.Module, optional): Model to initialize EMA with. If None, EMA will be initialized on first update.
    decay (float): Decay rate for EMA. Default is 0.9999.
    device (str or torch.device, optional): Device to store EMA weights. If None, uses the same device as the model.
  '''

  def __init__(self, model=None, decay=0.9999, device=None):
    self.decay = float(decay)
    self.numUpdates = 0
    self.device = torch.device(device) if device is not None else None
    self.module = None
    self._backup = None

    if (model is not None):
      # Keep a CPU/GPU copy of the model to expose EMA weights easily.
      self.module = copy.deepcopy(model)
      self.module.eval()
      for p in self.module.parameters():
        p.requires_grad = False
      if (self.device is not None):
        self.module.to(self.device)

  def update(self, model):
    r'''
    Update EMA weights using the current model parameters.

    Parameters:
      model (torch.nn.Module): Model with current parameters to update EMA from.
    '''

    if (self.module is None):
      self.module = copy.deepcopy(model)
      self.module.eval()
      for p in self.module.parameters():
        p.requires_grad = False
      if self.device is not None:
        self.module.to(self.device)

    self.numUpdates += 1
    decay = self.decay

    # Update each parameter: ema = decay * ema + (1 - decay) * param.
    with torch.no_grad():
      for emaP, modelData in zip(self.module.parameters(), model.parameters()):
        modelData = modelData.detach().to(emaP.device)
        emaP.data.mul_(decay).add_(modelData, alpha=(1.0 - decay))

  def to(self, device):
    r'''
    Move EMA weights to the specified device.

    Parameters:
      device (str or torch.device): Device to move EMA weights to.
    '''

    self.device = torch.device(device)
    if (self.module is not None):
      self.module.to(self.device)

  def state_dict(self):
    r'''
    Get the state dictionary for EMA.

    Returns:
      dict: State dictionary containing decay, num_updates, and module_state_dict.
    '''

    sd = {
      "decay"            : self.decay,
      "num_updates"      : self.numUpdates,
      "module_state_dict": self.module.state_dict() if (self.module is not None) else None,
    }
    return sd

  def load_state_dict(self, sd):
    r'''
    Load the state dictionary for EMA.

    Parameters:
      sd (dict): State dictionary to load.
    '''

    self.decay = float(sd.get("decay", self.decay))
    self.numUpdates = int(sd.get("num_updates", self.numUpdates))
    mstate = sd.get("module_state_dict", None)

    if (mstate is not None):
      if (self.module is None):
        # lazy: create a module by deep-copying nothing isn't possible; caller should initialize EMA with a model first.
        raise RuntimeError(
          "EMA module is not initialized. "
          "Initialize `ExponentialMovingAverage` with a model before loading module state."
        )
      self.module.load_state_dict(mstate)

  def apply_shadow(self, model):
    r'''
    Apply EMA weights to the given model, backing up original weights.

    Parameters:
      model (torch.nn.Module): Model to apply EMA weights to.
    '''

    if (self.module is None):
      raise RuntimeError("No EMA weights available. Call update() at least once or initialize EMA with a model.")
    self._backup = {}
    for (name, param), ema_p in zip(model.named_parameters(), self.module.parameters()):
      self._backup[name] = param.detach().clone()
      param.data.copy_(ema_p.data.to(param.device))

  def restore(self, model):
    r'''
    Restore original model weights from backup.

    Parameters:
      model (torch.nn.Module): Model to restore original weights to.
    '''

    if (self._backup is None):
      return
    nameToParam = {n: p for n, p in model.named_parameters()}
    for name, orig in self._backup.items():
      if (name in nameToParam):
        nameToParam[name].data.copy_(orig.to(nameToParam[name].device))
    self._backup = None

  # Convenience alias.
  def update_from(self, model):
    r'''
    Alias for update() method.

    Parameters:
      model (torch.nn.Module): Model with current parameters to update EMA from.
    '''

    return self.update(model)


def TrainEvaluateModel(
  model,  # Model to train and evaluate.
  criterion,  # Loss function.
  device,  # Device to run training and evaluation on (CPU or GPU).
  bestModelStoragePath,  # Path to save the best model.
  noOfClasses,  # Number of classes in the classification task.
  numEpochs,  # Total number of epochs for training.
  optimizer,  # Optimizer for updating model parameters.
  scaler,  # Gradient scaler for mixed precision training.
  scheduler,  # Learning rate scheduler.
  trainLoader,  # DataLoader for training data.
  valLoader,  # DataLoader for validation data.
  resumeFromCheckpoint=False,  # Whether to resume training from a checkpoint.
  finalModelStoragePath=None,  # Path to save the final model after training.
  judgeBy="both",  # Criterion to judge the best model ("val_loss", "val_accuracy", or "both").
  earlyStoppingPatience=None,  # Patience for early stopping.
  verbose=True,  # Verbosity flag to control logging.
  gradAccumSteps=1,  # Number of gradient accumulation steps.
  maxGradNorm=None,  # Maximum gradient norm for clipping.
  useAmp=True,  # Whether to use automatic mixed precision.
  useMixupFn=False,  # Whether to use MixUp data augmentation.
  mixUpAlpha=0.5,  # Value of alpha for MixUp data augmentation.
  useEma=False,  # Whether to use Exponential Moving Average for model parameters.
  saveEvery=None,  # Save model every N epochs.
):
  r'''
  Train and evaluate a classification model for a specified number of epochs.

  Parameters:
    model (torch.nn.Module): Model to train and evaluate.
    criterion (callable): Loss function.
    device (torch.device): Device to run training and evaluation on (CPU or GPU).
    bestModelStoragePath (str): Path to save the best model.
    noOfClasses (int): Number of classes in the classification task.
    numEpochs (int): Total number of epochs for training.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    scaler (torch.amp.GradScaler): Gradient scaler for mixed precision training.
    scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    trainLoader (torch.utils.data.DataLoader): DataLoader for training data.
    valLoader (torch.utils.data.DataLoader): DataLoader for validation data.
    resumeFromCheckpoint (bool, optional): Flag to indicate if training should resume from a checkpoint. Defaults to False.
    finalModelStoragePath (str, optional): Path to save the final model after training. Defaults to None.
    judgeBy (str, optional): Criterion to judge the best model ("val_loss", "val_accuracy", or "both"). Defaults to "both".
    earlyStoppingPatience (int, optional): Patience for early stopping. Defaults to None.
    verbose (bool, optional): Verbosity flag to control logging. Defaults to True.
    gradAccumSteps (int, optional): Number of gradient accumulation steps. Defaults to 1.
      When >1, gradients are accumulated over multiple batches before performing an optimizer step.
      When using gradient accumulation, ensure that the effective batch size (batchSize * gradAccumSteps) fits in memory.
    maxGradNorm (float, optional): Maximum gradient norm for clipping. Defaults to None.
    useAmp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
    useMixupFn (bool, optional): Whether to use MixUp data augmentation. Defaults to False.
    mixUpAlpha (float, optional): Value of alpha for MixUp data augmentation. Defaults to 0.5.
    useEma (object, optional): Whether to use Exponential Moving Average for model parameters. Defaults to False.
    saveEvery (int, optional): Save model every N epochs. Defaults to None.

  Returns:
    dict: History dictionary containing training and validation metrics.

  Examples
  --------
  .. code-block:: python

    import torch
    from torch import nn, optim
    from torch.amp import GradScaler
    from torch.utils.data import DataLoader
    from HMB.PyTorchHelper import TrainEvaluateModel

    # Prepare the data loaders.
    # Replace with actual dataset and DataLoader code.
    trainLoader = DataLoader(...)
    valLoader = DataLoader(...)

    # Create the model.
    # Replace with actual model creation code.
    model = pth.CreateTimmModel("resnet18", numClasses=10, pretrained=True)

    # Define loss function, optimizer, scaler, and scheduler.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Define device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train and evaluate the model.
    history = TrainEvaluateModel(
      model,
      criterion,
      device,
      bestModelStoragePath="best_model.pth",
      noOfClasses=10,
      numEpochs=25,
      optimizer=optimizer,
      scaler=scaler,
      scheduler=scheduler,
      trainLoader=trainLoader,
      valLoader=valLoader,
      resumeFromCheckpoint=False,
      finalModelStoragePath="final_model.pth",
      judgeBy="both",
      earlyStoppingPatience=None,
      verbose=True,
      gradAccumSteps=1,
      maxGradNorm=None,
      useAmp=True,
      useMixupFn=False,
      mixUpAlpha=0.5,
      useEma=False,
      saveEvery=None
    )
  '''

  # Initialize history dictionary to store training and validation metrics.
  history = {
    "train_accuracy": [],  # List to store training accuracy for each epoch.
    "val_accuracy"  : [],  # List to store validation accuracy for each epoch.
    "train_loss"    : [],  # List to store training loss for each epoch.
    "val_loss"      : [],  # List to store validation loss for each epoch.
  }

  # Variables to track the best validation loss and accuracy.
  bestValLoss = float("inf")
  bestValAccuracy = 0.0
  startEpoch = 0

  if (useEma):
    ema = ExponentialMovingAverage(model, decay=0.9999, device=device)
  else:
    ema = None

  # If resuming from checkpoint, load the model and optimizer state.
  if (resumeFromCheckpoint and os.path.exists(bestModelStoragePath)):
    print(f"Resuming from checkpoint: {resumeFromCheckpoint}")
    stateDict = LoadPyTorchDict(bestModelStoragePath, device=device)
    if (stateDict):
      model.load_state_dict(stateDict["model_state_dict"])
      optimizer.load_state_dict(stateDict["optimizer_state_dict"])
      scaler.load_state_dict(stateDict["scaler_state_dict"])
      startEpoch = stateDict.get("epoch", 0)
      bestValLoss = stateDict.get("best_val_loss", float("inf"))
      bestValAccuracy = stateDict.get("best_val_accuracy", 0.0)
    else:
      if (verbose):
        print("Failed to load checkpoint. Starting training from scratch.")

    if (verbose):
      print(f"Resumed training from epoch {startEpoch}.")
      if (startEpoch >= numEpochs):
        print("Warning: `startEpoch` is greater than or equal to `numEpochs`. No training will be performed.")
  else:
    if (verbose):
      print("Starting training from scratch.")

  currentPatience = 0  # Initialize patience counter for early stopping.

  # Training loop for the specified number of epochs.
  for epoch in range(startEpoch, numEpochs):
    if (verbose):
      print(f"Starting epoch {epoch + 1}/{numEpochs}")

    # Train for one epoch.
    avgTrainEpochLoss, avgTrainEpochTrain = TrainOneEpoch(
      model,  # Model to train.
      trainLoader,  # DataLoader for training data.
      criterion,  # Loss function.
      device,  # Device to run training on (CPU or GPU).
      epoch,  # Current epoch number.
      noOfClasses,  # Number of classes in the classification task.
      numEpochs,  # Total number of epochs for training.
      optimizer,  # Optimizer for updating model parameters.
      scaler,  # Gradient scaler for mixed precision training.
      gradAccumSteps=gradAccumSteps,  # Number of gradient accumulation steps.
      maxGradNorm=maxGradNorm,  # Maximum gradient norm for clipping.
      useAmp=useAmp,  # Whether to use automatic mixed precision.
      useMixupFn=useMixupFn,  # Whether to use MixUp data augmentation.
      mixUpAlpha=mixUpAlpha,  # Value of alpha for MixUp data augmentation.
      ema=ema,  # Exponential Moving Average object.
      verbose=verbose,  # Verbosity flag to control logging.
    )

    avgValEpochLoss, avgValEpochAccuracy = EvaluateOneEpoch(
      model,  # Model to evaluate.
      valLoader,  # DataLoader for evaluation data.
      criterion,  # Loss function.
      device,  # Device to run evaluation on (CPU or GPU).
      noOfClasses  # Number of classes in the classification task.
    )

    if (verbose):
      print(
        f"Epoch {epoch + 1}/{numEpochs} - "
        f"Train Loss: {avgTrainEpochLoss:.4f}, Val Loss: {avgValEpochLoss:.4f}, "
        f"Train Accuracy: {avgTrainEpochTrain:.4f}, Val Accuracy: {avgValEpochAccuracy:.4f}"
      )

    # Update history.
    history["train_loss"].append(avgTrainEpochLoss)
    history["val_loss"].append(avgValEpochLoss)
    history["train_accuracy"].append(avgTrainEpochTrain)
    history["val_accuracy"].append(avgValEpochAccuracy)

    # Save the model if validation loss improves.
    lossCondition = avgValEpochLoss < bestValLoss
    accuracyCondition = avgValEpochAccuracy > bestValAccuracy
    bothCondition = lossCondition and accuracyCondition
    if (judgeBy == "val_loss"):
      conditionToSave = lossCondition
    elif (judgeBy == "val_accuracy"):
      conditionToSave = accuracyCondition
    elif (judgeBy == "both"):
      conditionToSave = bothCondition
    else:
      raise ValueError(f"Invalid judgeBy value: {judgeBy}. Must be 'val_loss', 'val_accuracy', or 'both'.")

    if (conditionToSave):
      bestValLoss = avgValEpochLoss
      bestValAccuracy = avgValEpochAccuracy
      try:
        if (useEma and ema is not None):
          try:
            modelState = ema.module.state_dict()
          except Exception:
            try:
              modelState = ema.state_dict()
            except Exception:
              modelState = model.state_dict()
        else:
          modelState = model.state_dict()
      except Exception:
        modelState = model.state_dict()

      # SaveModel(model, bestModelStoragePath)
      SavePyTorchDict({
        "model_state_dict"    : modelState,
        "optimizer_state_dict": optimizer.state_dict() if (optimizer is not None) else None,
        "epoch"               : epoch + 1,
        "scaler_state_dict"   : scaler.state_dict() if (scaler is not None) else None,
        "best_val_loss"       : bestValLoss,
        "best_val_accuracy"   : bestValAccuracy,
      }, filename=bestModelStoragePath)
      if (verbose):
        print(
          f"Saved new best model with val loss: {bestValLoss:.4f} "
          f"and val accuracy: {bestValAccuracy:.4f}"
        )
      currentPatience = 0  # Reset patience counter on improvement.
    else:
      if (earlyStoppingPatience is not None):
        currentPatience += 1
        if (currentPatience >= earlyStoppingPatience):
          if (verbose):
            print(
              f"Early stopping triggered after {earlyStoppingPatience} epochs "
              f"without improvement."
            )
          break

    # For ReduceLROnPlateau we need to step with the validation loss after evaluation.
    try:
      if (isinstance(scheduler, (ReduceLROnPlateau,))):
        scheduler.step(avgValEpochLoss)
      else:
        scheduler.step()
    except Exception:
      pass

    if (saveEvery is not None and (epoch + 1) % saveEvery == 0):
      epochPath = os.path.join(
        os.path.dirname(bestModelStoragePath),
        f"ModelEpoch.epoch{epoch + 1}.pth"
      )
      SavePyTorchDict({
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch"               : epoch + 1,
        "scaler_state_dict"   : scaler.state_dict(),
        "best_val_loss"       : bestValLoss,
        "best_val_accuracy"   : bestValAccuracy,
      }, filename=epochPath)
      if (verbose):
        print(f"Saved model at epoch {epoch + 1} to {epochPath}")

  # Save the final model after training if a path is provided.
  if (finalModelStoragePath):
    SavePyTorchDict({
      "model_state_dict"    : model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "epoch"               : numEpochs,
      "scaler_state_dict"   : scaler.state_dict(),
      "best_val_loss"       : bestValLoss,
      "best_val_accuracy"   : bestValAccuracy,
    }, filename=finalModelStoragePath)
    if (verbose):
      print(f"Saved final model after {numEpochs} epochs to {finalModelStoragePath}")

  if (verbose):
    print("Training complete. Plotting training history...")
  HistoryPlotter(
    history,  # Dictionary containing training history.
    title="Training and Validation History",  # Title of the plot.
    metrics=("loss", "accuracy"),  # Tuple or list of metrics to plot.
    xLabel="Epochs",  # Label for x-axis.
    fontSize=14,  # Font size for labels and title.
    save=True,  # Whether to save the plot.
    savePath=os.path.join(os.path.dirname(bestModelStoragePath), "TrainingHistory.pdf"),  # Path to save the plot.
    dpi=720,  # DPI for saving the figure.
    colors=None,  # Optional dict of colors for each metric.
    labels=None,  # Optional dict of labels for each metric.
    display=False,  # Whether to display the plot.
    figSize=(10, 5),  # Figure size.
    returnFig=False,  # Whether to return the figure object.
    smooth=True,  # Whether to apply smoothing to the curves.
    smoothFactor=0.6,  # Smoothing factor for curves (0 to 1).
  )
  if (verbose):
    print("Training history plot saved.")
    print("Saving training history to CSV file...")
  # Saving the training history to a CSV file for future reference.
  historyCsvPath = os.path.join(os.path.dirname(bestModelStoragePath), "TrainingHistory.csv")
  df = pd.DataFrame(history)
  df.to_csv(historyCsvPath, index=False)
  if (verbose):
    print(f"Training history saved to {historyCsvPath}")
    print("Training and evaluation process completed.")

  return history


def TrainOneEpoch(
  model,  # Model to train.
  dataLoader,  # DataLoader for training data.
  criterion,  # Loss function.
  device,  # Device to run training on (CPU or GPU).
  epoch,  # Current epoch number.
  noOfClasses,  # Number of classes in the classification task.
  numEpochs,  # Total number of epochs for training.
  optimizer,  # Optimizer for updating model parameters.
  scaler,  # Gradient scaler for mixed precision training.
  gradAccumSteps=1,  # Number of gradient accumulation steps.
  maxGradNorm=None,  # Maximum gradient norm for clipping.
  useAmp=True,  # Whether to use automatic mixed precision.
  useMixupFn=False,  # Whether to use MixUp data augmentation.
  mixUpAlpha=0.5,  # Value of alpha for MixUp data augmentation.
  ema=None,  # Exponential Moving Average object.
  verbose=True,  # Verbosity flag to control logging.
):
  r'''
  Train the model for one epoch.

  Parameters:
    model (torch.nn.Module): Model to train.
    dataLoader (torch.utils.data.DataLoader): DataLoader for training data.
    criterion (callable): Loss function.
    device (torch.device): Device to run training on (CPU or GPU).
    epoch (int): Current epoch number.
    noOfClasses (int): Number of classes in the classification task.
    numEpochs (int): Total number of epochs for training.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    scaler (torch.amp.GradScaler): Gradient scaler for mixed precision training.
    gradAccumSteps (int, optional): Number of gradient accumulation steps. Defaults to 1.
    maxGradNorm (float, optional): Maximum gradient norm for clipping. Defaults to None.
    useAmp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
    useMixupFn (bool, optional): Whether to use MixUp data augmentation. Defaults to False.
    mixUpAlpha (float, optional): Value of alpha for MixUp data augmentation. Defaults to 0.5.
    ema (object, optional): Exponential Moving Average object. Defaults to None.
    verbose (bool, optional): Verbosity flag to control logging. Defaults to True.

  Returns:
    tuple: (avgTrainLoss, avgTrainAccuracy) for the epoch.
      - avgTrainLoss (float): Average training loss for the epoch.
      - avgTrainAccuracy (float): Average training accuracy for the epoch.
  '''

  # Set the model to training mode.
  model.train()

  # Initialize total loss and accuracy for the epoch.
  totalEpochLoss = 0.0
  totalEpochAccuracy = 0.0

  # Zero the gradients of the optimizer.
  optimizer.zero_grad()

  # Determine device type for autocast.
  deviceType = "cuda" if (
    (hasattr(device, "type") and device.type == "cuda") or
    ("cuda" in str(device))
  ) else "cpu"

  # Iterate over the training data loader with a progress bar.
  for batchIdx, batch in tqdm.tqdm(
    enumerate(dataLoader),  # Enumerate over batches.
    total=len(dataLoader),  # Total number of batches.
    desc=f"Epoch {epoch + 1}/{numEpochs}",  # Description for the progress bar.
  ):
    # Get data and labels from the batch and move them to the specified device.
    data, labels = batch
    data = data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if (useMixupFn):
      # Apply MixUp data augmentation.
      data, labels = MixupFn(data, labels, alpha=mixUpAlpha, numClasses=noOfClasses)

    # Use automatic mixed precision for the forward pass.
    with autocast(enabled=useAmp, device_type=deviceType):
      # Forward pass through the model to get outputs.
      outputs = model(data)
      # Check if labels are soft/one-hot for MixUp.
      if (useMixupFn and isinstance(labels, torch.Tensor) and labels.dim() > 1):
        # Compute the MixUp loss.
        loss = MixupCriterion(outputs, labels)
      else:
        # Compute the loss using the specified criterion.
        loss = criterion(outputs, labels)

    if (outputs.dim() == 1 or outputs.size(1) == 1):
      # Binary classification case.
      outputIdx = (outputs.cpu() > 0.5).long().squeeze()
    else:
      # Get the predicted class indices.
      outputIdx = outputs.cpu().argmax(dim=1)

    if (labels.dim() > 1):
      # If labels are soft/one-hot, convert to hard labels for accuracy calculation.
      hardLabels = labels.cpu().argmax(dim=1)
    else:
      hardLabels = labels.cpu()

    # If loss is a tensor, convert it to a scalar value.
    lossScalar = loss.item() if (isinstance(loss, torch.Tensor)) else loss
    # Accumulate the total loss for the epoch.
    totalEpochLoss += lossScalar

    # Compute the confusion matrix and accuracy.
    cm = confusion_matrix(
      hardLabels,  # True labels.
      outputIdx,  # Predicted labels.
      labels=list(range(noOfClasses)),  # List of class labels.
    )
    # Calculate performance metrics from the confusion matrix.
    metrics = CalculatePerformanceMetrics(cm)
    accuracy = metrics["Weighted Accuracy"]
    # Accumulate the total accuracy for the epoch.
    totalEpochAccuracy += accuracy

    loss = loss / gradAccumSteps  # Normalize loss for gradient accumulation.
    # Backward pass and optimization step with mixed precision.
    scaler.scale(loss).backward()

    if ((batchIdx + 1) % gradAccumSteps == 0 or (batchIdx + 1) == len(dataLoader)):
      # Clip gradients if maxGradNorm is specified.
      if (maxGradNorm is not None):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)
      # Perform optimizer step.
      scaler.step(optimizer)
      scaler.update()
      # Zero the gradients for the next iteration.
      optimizer.zero_grad()

    # Update exponential moving average of model parameters if provided.
    if (ema is not None):
      try:
        ema.update(model)
      except Exception as e:
        if (verbose):
          print(f"EMA update failed: {e}")
        try:
          ema.update(model.module)
        except Exception as e2:
          if (verbose):
            print(f"EMA update on model.module also failed: {e2}")

  # Calculate average loss and accuracy for the epoch.
  avgTrainLoss = totalEpochLoss / max(1, len(dataLoader))
  avgTrainAccuracy = totalEpochAccuracy / max(1, len(dataLoader))

  return avgTrainLoss, avgTrainAccuracy


def EvaluateOneEpoch(
  model,  # Model to evaluate.
  dataLoader,  # DataLoader for evaluation data.
  criterion,  # Loss function.
  device,  # Device to run evaluation on (CPU or GPU).
  noOfClasses,  # Number of classes in the classification task.
):
  r'''
  Evaluate the model for one epoch.

  Parameters:
    model (torch.nn.Module): Model to evaluate.
    dataLoader (torch.utils.data.DataLoader): DataLoader for evaluation data.
    criterion (callable): Loss function.
    device (torch.device): Device to run evaluation on (CPU or GPU).
    noOfClasses (int): Number of classes in the classification task.

  Returns:
    tuple: (avgValLoss, avgValAccuracy) for the epoch.
      - avgValLoss (float): Average validation loss for the epoch.
      - avgValAccuracy (float): Average validation accuracy for the epoch.
  '''

  # Set the model to evaluation mode.
  model.eval()

  # Initialize total loss and accuracy.
  totalLoss = 0.0
  totalAccuracy = 0.0

  # with torch.no_grad():
  with torch.inference_mode():
    # Iterate over the evaluation data loader with a progress bar.
    for batchIdx, batch in tqdm.tqdm(
      enumerate(dataLoader),  # Enumerate over batches.
      total=len(dataLoader),  # Total number of batches.
      desc="Evaluating",  # Description for the progress bar.
    ):
      # Get data and labels from the batch and move them to the specified device.
      data, labels = batch
      data = data.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)

      # Forward pass through the model to get outputs.
      outputs = model(data)

      if (outputs.dim() == 1 or outputs.size(1) == 1):
        # Binary classification case.
        outputIdx = (outputs.cpu() > 0.5).long().squeeze()
      else:
        # Get the predicted class indices.
        outputIdx = outputs.cpu().argmax(dim=1)

      if (labels.dim() > 1):
        # If labels are soft/one-hot, convert to hard labels for accuracy calculation.
        hardLabels = labels.cpu().argmax(dim=1)
      else:
        hardLabels = labels.cpu()

      # Compute the loss using the specified criterion.
      loss = criterion(outputs, labels)
      # If loss is a tensor, convert it to a scalar value.
      loss = loss.item() if isinstance(loss, torch.Tensor) else loss
      # Accumulate the total loss for the validation epoch.
      totalLoss += loss

      # Compute the confusion matrix and accuracy using hard labels.
      cm = confusion_matrix(
        hardLabels,  # True labels.
        outputIdx,  # Predicted labels.
        labels=list(range(noOfClasses)),  # List of class labels.
      )
      # Calculate performance metrics from the confusion matrix.
      metrics = CalculatePerformanceMetrics(cm)
      accuracy = metrics["Weighted Accuracy"]

      # Accumulate the total accuracy for the validation epoch.
      totalAccuracy += accuracy

  # Calculate average loss and accuracy for the validation epoch.
  avgValLoss = totalLoss / max(1, len(dataLoader))
  avgValAccuracy = totalAccuracy / max(1, len(dataLoader))

  return avgValLoss, avgValAccuracy


def InferenceWithPlots(
  dataDir,  # Directory containing dataset.
  model,  # Model architecture.
  modelCheckpointName=None,  # Path to model checkpoint.
  transform=None,  # Image transform to apply.
  useDefaultTransform=False,  # Whether to use default image transform if none provided.
  device=None,  # Device to run inference on.
  batchSize=1,  # Batch size for inference.
  imageSize=448,  # Image size for transforms.
  expDirs=[],  # List of experiment directories.
  overallResultsPath="Overall_Results.csv",  # Output CSV path for overall results.
  appendResults=True,  # Whether to append to existing overall results CSV.
  plotFontSize=16,  # Font size for plots.
  plotFigSize=(8, 8),  # Figure size for confusion matrix.
  rocFigSize=(5, 5),  # Figure size for ROC/PRC curves.
  dpi=720,  # DPI for saving plots.
  verbose=True,  # Whether to print progress.
):
  r'''
  Perform inference on all experiment directories and generate performance plots.

  Parameters:
    dataDir (str): Directory containing dataset.
    model (torch.nn.Module): Model architecture.
    modelCheckpointName (str, optional): Name of the model checkpoint.
    transform (callable, optional): Image transform to apply.
    useDefaultTransform (bool, optional): Whether to use default image transform if none provided.
    device (str or torch.device, optional): Device to run inference on.
    batchSize (int, optional): Batch size for inference.
    imageSize (int, optional): Image size for transforms.
    expDirs (list, optional): List of experiment directories.
    overallResultsPath (str, optional): Output CSV file for overall results.
    appendResults (bool, optional): Whether to append to existing overall results CSV.
    plotFontSize (int, optional): Font size for plots.
    plotFigSize (tuple, optional): Figure size for confusion matrix.
    rocFigSize (tuple, optional): Figure size for ROC/PRC curves.
    dpi (int, optional): DPI for saving plots.
    verbose (bool, optional): Whether to print progress.
  '''

  if (len(expDirs) == 0):
    if (verbose):
      print("No experiment directories provided.")
    return

  # Set device.
  if (device is None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if (not transform):
    if (verbose):
      print("No transform provided. Using default transform.")

    if (useDefaultTransform):
      # Prepare image transform.
      transform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),  # Resize images.
        transforms.ToTensor(),  # Convert to tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize.
      ])

  # Create dataset and dataloader.
  dataset = CustomDataset(dataDir, transform=transform)
  dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

  if (verbose):
    print(f"Dataset contains {len(dataset)} images across {len(dataset.classToIdx)} classes.")

  # Initialize overall history.
  overallHistory = []

  # Loop through each experiment directory.
  for expDirPath in expDirs:
    if (not os.path.exists(expDirPath)):
      if (verbose):
        print(f"Experiment directory not found: {expDirPath}")
      continue

    if (verbose):
      print(f"Processing directory: {expDirPath}")

    if (modelCheckpointName):
      modelPath = os.path.join(expDirPath, modelCheckpointName)
      # Load model checkpoint name from experiment directory.
      # stateDict = LoadPyTorchDict(modelPath, device=device)
      # if (stateDict and isinstance(stateDict, dict) and "model_state_dict" in stateDict):
      #   model.load_state_dict(stateDict["model_state_dict"])
      # else:
      #   model.load_state_dict(stateDict)
      #  You can use LoadModel function if preferred.
      model = LoadModel(model, modelPath, device=device)

    cmFilePath = os.path.join(expDirPath, "Inference CM.pdf")
    rocFilePath = os.path.join(expDirPath, "Inference ROC.pdf")
    rocpFilePath = os.path.join(expDirPath, "Inference ROCP.pdf")
    prcFilePath = os.path.join(expDirPath, "Inference PRC.pdf")
    prcpFilePath = os.path.join(expDirPath, "Inference PRCP.pdf")

    # Move the model to the selected device (CPU or GPU).
    model = model.to(device)

    # Set the model to evaluation mode.
    model.eval()

    # Lists to store true labels, predicted labels, and probabilities.
    yTrue, yPred, yProbs = [], [], []

    # Disable gradient calculation for evaluation. Use inference mode for better performance if available.
    with torch.inference_mode():
      # Iterate over the dataloader with a progress bar.
      for imgs, labels in tqdm.tqdm(dataloader, desc="Inference", unit="Image"):
        # Move images and labels to the selected device.
        imgs = imgs.to(device)
        labels = labels.to(device)
        # Get model outputs (logits).
        outputs = model(imgs)
        # Apply softmax to get probabilities.
        probs = torch.softmax(outputs, dim=1)
        # Get predicted class indices.
        preds = torch.argmax(outputs, dim=1)
        # Store true labels, predicted labels, and probabilities.
        yTrue.extend(labels.cpu().numpy())
        yPred.extend(preds.cpu().numpy())
        yProbs.extend(probs.cpu().numpy())

    # Convert lists to numpy arrays for easier manipulation.
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    yProbs = np.array(yProbs)

    # Compute the confusion matrix.
    cm = confusion_matrix(yTrue, yPred)
    # Calculate performance metrics using the confusion matrix.
    metrics = CalculatePerformanceMetrics(cm, addWeightedAverage=True, eps=1e-8)
    metrics["Path"] = expDirPath
    metrics["File"] = os.path.basename(expDirPath)
    overallHistory.append(metrics)

    # Plot and save the confusion matrix.
    PlotConfusionMatrix(
      cm,  # Confusion matrix (2D list or numpy array).
      classes=list(dataset.classToIdx.keys()),  # List of class names.
      normalize=False,  # Whether to normalize the confusion matrix.
      roundDigits=3,  # Number of decimal places to round normalized values.
      title="Confusion Matrix",  # Title of the plot.
      cmap=plt.cm.Blues,  # Colormap for the plot.
      display=False,  # Whether to display the plot.
      save=True,  # Whether to save the plot.
      fileName=cmFilePath,  # File path to save the plot.
      fontSize=plotFontSize,  # Font size for labels and annotations.
      annotate=True,  # Whether to annotate cells with values.
      figSize=plotFigSize,  # Figure size in inches.
      colorbar=True,  # Whether to show colorbar.
      returnFig=False,  # Whether to return the figure object.
    )

    # Plot and save the ROC curve and AUC (predicted labels).
    PlotROCAUCCurve(
      yTrue,  # True labels.
      yPred,  # Predicted labels.
      classes=list(dataset.classToIdx.keys()),  # List of class names.
      areProbabilities=False,  # Whether yPred are probabilities.
      title="ROC Curve & AUC",  # Plot title.
      figSize=rocFigSize,  # Figure size.
      cmap=None,  # Colormap for ROC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=rocFilePath,  # File path to save the plot.
      fontSize=plotFontSize,  # Font size.
      plotDiagonal=True,  # Plot diagonal reference line.
      annotateAUC=True,  # Annotate AUC value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    # Plot and save the ROC curve and AUC (probabilities).
    PlotROCAUCCurve(
      yTrue,  # True labels.
      yProbs,  # Predicted probabilities.
      classes=list(dataset.classToIdx.keys()),  # List of class names.
      areProbabilities=True,  # Whether yPred are probabilities.
      title="ROC Curve & AUC (Probabilities)",  # Plot title.
      figSize=rocFigSize,  # Figure size.
      cmap=None,  # Colormap for ROC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=rocpFilePath,  # File path to save the plot.
      fontSize=plotFontSize,  # Font size.
      plotDiagonal=True,  # Plot diagonal reference line.
      annotateAUC=True,  # Annotate AUC value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    # Plot and save the PRC curve (predicted labels).
    PlotPRCCurve(
      yTrue,  # True labels.
      yPred,  # Predicted labels.
      classes=list(dataset.classToIdx.keys()),  # List of class names.
      areProbabilities=False,  # Whether yPred are probabilities.
      title="PRC Curve",  # Plot title.
      figSize=rocFigSize,  # Figure size.
      cmap=None,  # Colormap for PRC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=prcFilePath,  # File path to save the plot.
      fontSize=plotFontSize,  # Font size.
      annotateAvg=True,  # Annotate average precision value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    # Plot and save the PRC curve (probabilities).
    PlotPRCCurve(
      yTrue,  # True labels.
      yProbs,  # Predicted probabilities.
      classes=list(dataset.classToIdx.keys()),  # List of class names.
      areProbabilities=True,  # Whether yPred are probabilities.
      title="PRC Curve (Probabilities)",  # Plot title.
      figSize=rocFigSize,  # Figure size.
      cmap=None,  # Colormap for PRC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=prcpFilePath,  # File path to save the plot.
      fontSize=plotFontSize,  # Font size.
      annotateAvg=True,  # Annotate average precision value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

  if ((not appendResults) or (not os.path.exists(overallResultsPath))):
    # Save overall metrics to CSV.
    df = pd.DataFrame(overallHistory)
    df = df[["File"] + [col for col in df.columns if col != "File"]]
    df.to_csv(overallResultsPath, index=False)
    if (verbose):
      print(f"Overall results saved to {overallResultsPath}.")
  else:
    # Append overall metrics to existing CSV.
    if (os.path.exists(overallResultsPath)):
      dfExisting = pd.read_csv(overallResultsPath)
      dfNew = pd.DataFrame(overallHistory)
      dfNew = dfNew[["File"] + [col for col in dfNew.columns if col != "File"]]
      dfCombined = pd.concat([dfExisting, dfNew], ignore_index=True)
      dfCombined.to_csv(overallResultsPath, index=False)
      if (verbose):
        print(f"Overall results appended to {overallResultsPath}.")


def ApplyDynamicQuantizationTorch(modelPath: str, outputPath: str, exampleInput=None, inputShape=None):
  r'''
  Apply dynamic quantization to a PyTorch model checkpoint and save a portable TorchScript file.

  This helper attempts to load a checkpoint from `modelPath`. It supports either:
    - a dict containing a key like "model" that is a torch.nn.Module or contains a state_dict
    - a plain torch.nn.Module object saved directly
    - a state_dict (mapping) paired with a provided model object is NOT supported here

  The function will try to obtain a live nn.Module to quantize. If only a state_dict is found
  it will return None because the architecture is not known. When successful the function:
    - applies torch.quantization.quantize_dynamic to quantize Linear and Embedding modules to qint8
    - scripts the quantized model with torch.jit.script for portability (or traces using provided exampleInput/inputShape)
    - saves the scripted module to `outputPath`

  Parameters:
    modelPath (str): Path to the saved checkpoint or model file.
    outputPath (str): Path where the scripted quantized model will be saved.
    example_input (torch.Tensor, optional): Example input tensor to use for tracing when scripting fails.
    inputShape (tuple, optional): Alternative to exampleInput; will create a random tensor with this shape for tracing.

  Returns:
    str|None: Returns the outputPath on success, None on failure.
  '''

  if (not os.path.exists(modelPath)):
    print(f"Model file not found: {modelPath}")
    return None

  ckpt = None
  try:
    # Load checkpoint with CPU mapping for safe handling.
    ckpt = torch.load(modelPath, map_location="cpu")
  except Exception as e:
    # Some PyTorch installations restrict unpickling of arbitrary globals by default (WeightsUnpickler).
    # Try to allowlist common nn containers (Sequential, ModuleList, ModuleDict) and retry loading.
    errstr = str(e)
    print(f"Initial model loading failed: {errstr}")
    try:
      from torch.serialization import WeightsUnpickler

      class CustomUnpickler(WeightsUnpickler):
        def find_class(self, module, name):
          if (
            module == "torch.nn.modules.container" and
            name in ["Sequential", "ModuleList", "ModuleDict"]
          ):
            return super().find_class(module, name)
          raise pickle.UnpicklingError(f"Global '{module}.{name}' is not allowed for unpickling.")

      with open(modelPath, "rb") as f:
        unpickler = CustomUnpickler(f)
        ckpt = unpickler.load()
    except Exception as e2:
      print(f"Model loading with custom unpickler also failed: {e2}")
      return None

  if (ckpt is None):
    print("Failed to load model checkpoint.")
    return None

  modelObj = None

  # If checkpoint is a mapping and contains a model/module inside
  if (isinstance(ckpt, dict)):
    # Common keys that may contain a module or state_dict
    # Prefer an actual nn.Module object if saved directly.
    if ("model" in ckpt and hasattr(ckpt["model"], "state_dict")):
      modelObj = ckpt["model"]
    elif ("model_state_dict" in ckpt and hasattr(ckpt["model_state_dict"], "keys")):
      # Only state_dict present - cannot reconstruct architecture here.
      print(
        "Checkpoint contains only state_dict. "
        "Quantization requires the model architecture object; "
        "please provide a full model object in the checkpoint."
      )
      return None
    elif (hasattr(ckpt, "state_dict")):
      # Edge-case: ckpt is a module-like object.
      modelObj = ckpt
  else:
    # ckpt may be a Module saved directly.
    if (hasattr(ckpt, "state_dict")):
      modelObj = ckpt

  if (modelObj is None):
    print(
      "No torch.nn.Module instance found in the checkpoint. "
      "Cannot apply dynamic quantization without the model architecture."
    )
    return None

  print("Applying dynamic quantization to the model...")

  try:
    # Only quantize common heavy-weight layers; include Embedding as optional.
    modulesToQuantize = {torch.nn.Linear, torch.nn.Embedding}
    qModel = torch.quantization.quantize_dynamic(modelObj, modulesToQuantize, dtype=torch.qint8)

    # Convert to scripted module for portability. Use try/except as some models may not script cleanly.
    try:
      ts = torch.jit.script(qModel)
    except Exception:
      # Fallback to tracing with a user-supplied exampleInput or inputShape if scripting fails.
      try:
        traceInput = None
        if (exampleInput is not None):
          # If user passed a numpy array or list, allow it by converting to tensor.
          if (not hasattr(exampleInput, "shape") or isinstance(exampleInput, (list, tuple))):
            try:
              traceInput = torch.tensor(exampleInput)
            except Exception:
              traceInput = exampleInput
          else:
            traceInput = exampleInput
        elif (inputShape is not None):
          try:
            traceInput = torch.randn(*inputShape)
          except Exception:
            traceInput = None
        else:
          # As a last resort, attempt a common image input shape used by models such as timm and torchvision.
          try:
            traceInput = torch.randn(1, 3, 224, 224)
          except Exception:
            traceInput = None

        if (traceInput is None):
          print(
            "No valid exampleInput or inputShape available for tracing; "
            "scripting failed and tracing cannot proceed."
          )
          return None

        ts = torch.jit.trace(qModel, traceInput)
      except Exception as e:
        print(f"Failed to convert quantized model to TorchScript via tracing: {e}")
        return None

    # Ensure output directory exists.
    outDir = os.path.dirname(outputPath)
    if outDir and (not os.path.exists(outDir)):
      os.makedirs(outDir, exist_ok=True)

    ts.save(outputPath)
    print(f"Quantized scripted model saved to {outputPath}")
    return outputPath
  except Exception as e:
    print(f"Dynamic quantization failed: {e}")
    return None


def GenericEvaluatePredictPlotSubset(
  datasetDir: str,
  model,
  subset: str = "test",
  prefix: str = "",
  storageDir: Optional[str] = None,
  heavy: bool = True,
  computeECE: bool = True,
  exportFailureCases: bool = True,
  eps: float = 1e-10,
  saveArtifacts: bool = True,
  maxSamples: Optional[int] = None,
  preprocessFn=None,
  dpi: int = 720,
) -> Tuple[
  Optional[str],
  Dict[str, float],
  List[int],
  List[int],
  List[List[float]],
  List[Optional[float]],
  List[Dict[str, Any]],
  List[str],
  Optional[np.ndarray],
]:
  r'''
  Evaluate a trained classification model on a specified dataset subset
  (train/val/test/all), collect predictions, compute confusion matrix and performance metrics,
  and optionally save predictions to a CSV file. It also generates and saves confusion matrix,
  ROC AUC, and PRC plots.

  Parameters:
    datasetDir (str): Path to the dataset directory containing train/val/test splits.
    model (callable): A callable that takes a NumPy array (HWC, uint8 or float32) and returns a 1D array of class probabilities.
    subset (str | None): Dataset subset to evaluate ("train", "val", "test", "all", or None). Defaults to "test".
    prefix (str): Prefix for saved figure filenames. Defaults to "".
    storageDir (str | None): Directory to save predictions CSV and figures. If None, uses current directory. Defaults to None.
    heavy (bool): Whether to compute heavy metrics and plot ROC/PRC curves. Defaults to True.
    computeECE (bool): Whether to compute Expected Calibration Error (ECE). Defaults to True.
    exportFailureCases (bool): Whether to export misclassified samples to CSV. Defaults to True.
    eps (float): Small epsilon value for numerical stability in metric calculations. Defaults to 1e-10.
    saveArtifacts (bool): Whether to save figures and artifacts. Defaults to True.
    maxSamples (int | None): Maximum number of samples to evaluate. If None, evaluates all samples. Defaults to None.
    preprocessFn (callable | None): Optional preprocessing function to apply to each PIL image before prediction. Defaults to None.
    dpi (int): DPI for saved figures. Defaults to 720.

  Returns:
    tuple: A tuple containing:
      - str|None: Path to the saved predictions CSV file (or None when not saved).
      - dict: Computed weighted performance metrics.
      - List[int]: List of predicted class indices.
      - List[int]: List of ground truth class indices.
      - List[List[float]]: List of predicted class probabilities for each sample.
      - List[Optional[float]]: List of predicted confidences for each sample.
      - List[Dict[str, Any]]: List of prediction records for each sample.
      - List[str]: List of class names.
      - numpy.ndarray|None: Confusion matrix as a 2D numpy array, or None if not computable.
  '''

  # Record the start time for total evaluation duration.
  startAll = time.perf_counter()

  # Validate the requested dataset subset.
  if (subset not in ("train", "val", "test", "all", None)):
    # Raise an error if the subset is not one of the allowed values.
    raise ValueError(f"Invalid subset name: {subset}")

  # Determine which split directories to process based on the subset.
  if (subset == "all"):
    # Include all standard splits.
    splitDirs = [Path(datasetDir) / split for split in ("train", "val", "test")]
  elif (subset in ("train", "val", "test")):
    # Use only the specified split.
    splitDirs = [Path(datasetDir) / subset]
  else:
    # Fallback to the root dataset directory.
    splitDirs = [Path(datasetDir)]

  # Initialize containers for collected data.
  allPredsIndices: List[int] = []
  allGtsIndices: List[int] = []
  allPredsProbs: List[List[float]] = []
  allPredsNames: List[str] = []
  allGtsNames: List[str] = []
  allPredsConfidences: List[Optional[float]] = []
  predictionsRecords: List[Dict[str, Any]] = []
  classNames: List[str] = []

  try:
    # Iterate over each split directory (e.g., train, val, test).
    for splitDir in splitDirs:
      # Skip if the split directory does not exist.
      if (not splitDir.exists()):
        print(f"Warning: Split directory does not exist, skipping: {splitDir}")
        continue

      # Get sorted list of class subdirectories.
      classDirs = sorted([
        directory
        for directory in splitDir.iterdir()
        if (directory.is_dir())
      ])
      if (len(classDirs) == 0):
        print(f"Warning: No class subdirectories found in split: {splitDir}")
        continue

      # Set class names from the first valid split encountered.
      if (len(classNames) == 0):
        classNames = [directory.name for directory in classDirs]
      numClasses = len(classDirs)
      print(f"Processing split: {splitDir.name} with {numClasses} classes.")
      print(f"Class names: {classNames}")

      # Process each class directory.
      for trueClassIndex, classDir in enumerate(classDirs):
        # Collect all valid image files in the class directory.
        imageFiles = [
          p
          for p in classDir.iterdir()
          if (p.is_file() and (p.suffix.lower() in IMAGE_SUFFIXES))
        ]
        if (len(imageFiles) == 0):
          print(f"Warning: No image files found in class directory, skipping: {classDir}")
          continue

        # Apply per-class sampling limit if maxSamples is set.
        if (maxSamples is not None):
          currentMax = max(1, maxSamples // max(1, numClasses))
          imageFiles = imageFiles[:currentMax]
          print(f"Limiting to {len(imageFiles)} samples from class {classDir.name}")

        # Process each image in the class.
        for imagePath in imageFiles:
          # Load image using OpenCV (BGR format).
          img = cv2.imread(str(imagePath))
          if (img is None):
            print(f"Warning: could not read image, skipping: {imagePath}")
            continue

          # Convert BGR to RGB.
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          # Convert to PIL Image for optional preprocessing.
          imgPIL = Image.fromarray(img)
          if (preprocessFn is not None):
            imgPIL = preprocessFn(imgPIL)
          img = np.array(imgPIL)

          # Make prediction using the generic model callable.
          try:
            # Should return 1D array of shape [numClasses].
            trueClassName = classDir.name
            probs = model(img, str(imagePath), trueClassName)
            probs = np.asarray(probs, dtype=np.float32)

            if (probs.ndim != 1):
              raise ValueError(f"Expected 1D probability vector, got shape {probs.shape}")
            if (len(probs) != numClasses):
              raise ValueError(
                f"Number of classes mismatch: expected {numClasses}, got {len(probs)}\n"
                f"Image path: {imagePath}"
                f", probs: {probs}"
              )

            predictedClassIndex = int(np.argmax(probs))
            predictedConfidence = float(probs[predictedClassIndex])
            probList = probs.tolist()

          except Exception as predErr:
            print(f"Prediction failed for {imagePath}: {predErr}")
            predictedClassIndex = -1
            predictedConfidence = None
            probList = []

          # Append prediction results.
          allPredsIndices.append(predictedClassIndex)
          allGtsIndices.append(trueClassIndex)
          allPredsConfidences.append(predictedConfidence)
          allPredsProbs.append(probList)

          # Resolve class names for display.
          predName = (
            classNames[predictedClassIndex]
            if (0 <= predictedClassIndex < len(classNames))
            else "Unknown"
          )
          allPredsNames.append(predName)
          allGtsNames.append(classDir.name)

          # Compute per-sample ECE if requested.
          eceValue = None
          if (computeECE and probList):
            try:
              eceValue = ComputeECE([probList], [trueClassIndex])
            except Exception:
              eceValue = None

          # Determine if prediction is correct.
          correctness = (predictedClassIndex == trueClassIndex)

          # Record full prediction metadata.
          predictionsRecords.append({
            "image"              : str(imagePath),
            "split"              : splitDir.name,
            "trueClassIndex"     : int(trueClassIndex),
            "trueClassName"      : classDir.name,
            "predictedClassIndex": predictedClassIndex,
            "predictedClassName" : predName,
            "predictedConfidence": predictedConfidence,
            "probabilities"      : (json.dumps(probList) if probList else None),
            "ece"                : (float(eceValue) if eceValue is not None else None),
            "correctness"        : correctness,
          })

      # Log progress after each split.
      print(f"Prediction collection completed for split: {splitDir.name}")
      print(f"Collected predictions for {len(allGtsIndices)} samples across {numClasses} classes.")
      print(f"Total samples collected for confusion matrix: {len(allGtsIndices)}")
      print(f"{'-' * 60}")

    # Finalize collection.
    print("Finished collecting predictions for all specified splits.")

    # Perform basic consistency checks on collected data.
    assert len(allPredsIndices) == len(allGtsIndices), "Mismatch in predictions and ground truths count."
    assert len(allPredsIndices) == len(allPredsProbs), "Mismatch in predictions and probabilities count."
    assert len(allPredsIndices) == len(allPredsNames), "Mismatch in predictions and names count."
    assert len(allGtsIndices) == len(allGtsNames), "Mismatch in ground truths and names count."
    assert len(allPredsIndices) == len(allPredsConfidences), "Mismatch in predictions and confidences count."
    assert len(predictionsRecords) == len(allGtsIndices), "Mismatch in prediction records and ground truths count."
    print(f"Total samples collected: {len(allGtsIndices)}")
    print(f"{'-' * 60}")

    # Compute confusion matrix and metrics if data exists.
    if ((len(allPredsIndices) > 0) and (len(allGtsIndices) > 0)):
      cm = confusion_matrix(allGtsIndices, allPredsIndices)
      metricResults = CalculatePerformanceMetrics(
        cm,
        eps=eps,
        addWeightedAverage=True,
        addPerClass=False
      )
      weightedMetrics = {key: value for key, value in metricResults.items() if key.startswith("Weighted")}
      print(f"Computed weighted metrics from confusion matrix on {len(allGtsIndices)} samples.")
    else:
      weightedMetrics = {}
      print("Warning: No predictions collected for confusion matrix computation.")

  except Exception as ex:
    # Handle any unexpected errors during evaluation.
    weightedMetrics = {}
    print(f"Error during prediction collection or metric computation: {ex}")

  # Apply prefix to metric keys if provided.
  if (prefix):
    weightedMetrics = {f"{prefix}{key}": value for key, value in weightedMetrics.items()}

  # Resolve storage directory.
  if (storageDir is None):
    storageDir = Path(".")
  else:
    storageDir = Path(storageDir)

  # Create storage directory if saving artifacts.
  if (saveArtifacts):
    storageDir.mkdir(parents=True, exist_ok=True)
    print(f"Using storage directory: {storageDir}")

  # Save full predictions to CSV if enabled.
  storageFilePath = None
  if (saveArtifacts):
    storageFileName = f"{prefix}_Predictions_{subset}.csv" if (prefix) else f"Predictions_{subset}.csv"
    storageFilePath = storageDir / storageFileName
    try:
      dfPreds = pd.DataFrame(predictionsRecords)
      dfPreds.to_csv(storageFilePath, index=False)
      print(f"Predictions for subset '{subset}' saved to: {storageFilePath}")
    except Exception as saveErr:
      print(f"Warning: Could not save predictions CSV: {saveErr}")

  # Export misclassified samples if requested.
  if (exportFailureCases):
    try:
      failureRecords = [
        rec for i, rec in enumerate(predictionsRecords)
        if allGtsIndices[i] != allPredsIndices[i]
      ]
      if (saveArtifacts and failureRecords):
        dfFailures = pd.DataFrame(failureRecords)
        failureFileName = f"{prefix}_Misclassified_Samples.csv" if (prefix) else "Misclassified_Samples.csv"
        failureFilePath = storageDir / failureFileName
        dfFailures.to_csv(failureFilePath, index=False)
        print(f"Misclassified samples exported to: {failureFilePath}")
      else:
        print("No misclassified samples to export.")
    except Exception as failErr:
      print(f"Warning: Could not export misclassified samples: {failErr}")

  # Recompute final confusion matrix.
  try:
    cm = confusion_matrix(allGtsIndices, allPredsIndices) if (
      len(allGtsIndices) > 0 and len(allPredsIndices) > 0) else None
    print("Confusion matrix computed.")
    print(cm)
  except Exception as cmErr:
    print(f"Warning: could not compute final confusion matrix: {cmErr}")
    cm = None

  # Print diagnostic summaries.
  print("Class names:")
  print(classNames)
  print("All collected ground truth indices (first 10):")
  print(allGtsIndices[:10], "..." if len(allGtsIndices) > 10 else "")
  print("All collected predicted indices (first 10):")
  print(allPredsIndices[:10], "..." if len(allPredsIndices) > 10 else "")
  print("All collected predicted probabilities (first 3 samples):")
  for probs in allPredsProbs[:3]:
    print(probs)
  print(f"{'-' * 60}")

  # Save confusion matrix plot.
  if (saveArtifacts):
    try:
      filename = f"{prefix}_CM.pdf" if (prefix) else "CM.pdf"
      PlotConfusionMatrix(
        cm,
        classNames,
        normalize=False,
        roundDigits=3,
        title="Confusion Matrix",
        cmap=plt.cm.Blues,
        display=False,
        save=True,
        fileName=str(storageDir / filename),
        fontSize=15,
        annotate=True,
        figSize=(8, 8),
        colorbar=True,
        returnFig=False,
        dpi=dpi,
      )
      print(f"Confusion matrix figure saved to: {storageDir / filename}")
    except Exception as figErr:
      print(f"Warning: Could not generate confusion matrix figure: {figErr}")

    # Store the metrics as CSV file.
    try:
      filename = f"{prefix}_Metrics.csv" if (prefix) else "Metrics.csv"
      metricsFilePath = storageDir / filename
      dfMetrics = pd.DataFrame([weightedMetrics])
      dfMetrics.to_csv(metricsFilePath, index=False)
      print(f"Metrics saved to: {metricsFilePath}")
    except Exception as metricsErr:
      print(f"Warning: Could not save metrics CSV: {metricsErr}")

  # Save ROC and PRC curves if heavy metrics are enabled.
  if (saveArtifacts and heavy):
    try:
      filename = f"{prefix}_ROC_AUC.pdf" if (prefix) else "ROC_AUC.pdf"
      PlotROCAUCCurve(
        np.array(allGtsIndices),
        np.array(allPredsProbs),
        classNames,
        areProbabilities=True,
        title="ROC Curve & AUC",
        figSize=(5, 5),
        cmap=None,
        display=False,
        save=True,
        fileName=str(storageDir / filename),
        fontSize=15,
        plotDiagonal=True,
        annotateAUC=True,
        showLegend=True,
        returnFig=False,
        dpi=dpi,
      )
      print(f"ROC AUC figure saved to: {storageDir / filename}")
    except Exception as rocErr:
      print(f"Warning: Could not generate ROC AUC figure: {rocErr}")

    try:
      filename = f"{prefix}_PRC.pdf" if (prefix) else "PRC.pdf"
      PlotPRCCurve(
        allGtsIndices,
        allPredsProbs,
        classNames,
        areProbabilities=True,
        title="PRC Curve",
        figSize=(5, 5),
        cmap=None,
        display=False,
        save=True,
        fileName=str(storageDir / filename),
        fontSize=15,
        annotateAvg=True,
        showLegend=True,
        returnFig=False,
        dpi=dpi,
      )
      print(f"PRC figure saved to: {storageDir / filename}")
    except Exception as prcErr:
      print(f"Warning: Could not generate PRC figure: {prcErr}")
  else:
    print("Heavy metrics and plots skipped as per configuration.")

  # Compute overall ECE if requested.
  if (computeECE):
    try:
      ece = ComputeECE(allPredsProbs, allGtsIndices)
      weightedMetrics["ECE"] = ece
      print("Expected Calibration Error (ECE):", ece)
    except Exception as eceErr:
      print(f"Warning: Could not compute ECE: {eceErr}")

  # Record total evaluation time.
  endAll = time.perf_counter()
  duration = endAll - startAll
  print(f"Total evaluation and prediction time: {duration:.2f} seconds.")
  weightedMetrics["Total Evaluation Time (s)"] = duration

  # Return all collected results.
  return (
    str(storageFilePath) if storageFilePath else None,
    weightedMetrics,
    allPredsIndices,
    allGtsIndices,
    allPredsProbs,
    allPredsConfidences,
    predictionsRecords,
    classNames,
    cm,
  )


def EvaluateModelOnPerturbations(
  model,
  run,
  datasetDir,
  storeDir,
  perturbations: List[str],
  levels: List[float],
  maxSamples: Optional[int] = 200,
  preprocessFn=None,
  subset: Optional[str] = "test",
  eps: float = 1e-10,
  dpi: int = 300,
):
  r'''
  Evaluate a classification model under a set of input perturbations and severity levels.

  This routine runs the provided model over a dataset subset (train/val/test/all) while applying
  controlled corruptions (noise, brightness, jpeg, occlusion, etc.) at several severity levels.
  It collects per-sample predictions, computes accuracy, calibration (ECE), Brier score and other
  auxiliary metrics, aggregates results per-perturbation and per-level, produces plots and CSV
  summaries, and writes a human-readable interpretation report.

  Parameters:
    model (callable | torch.nn.Module): A model or a prediction callable that accepts an image
      (PIL or NumPy HWC) and returns a 1D array of class probabilities.
    run (Any): Identifier for the current model run used in report headers (may be a Path-like
      or an object with a `.name` attribute).
    datasetDir (str | Path): Path to the dataset root containing splits (train/val/test).
    storeDir (str | Path): Directory where results, CSVs, plots and interpretation files will be
      written. The directory will be created if it does not exist.
    perturbations (List[str]): List of perturbation names to evaluate. If empty, a default set of
      available perturbations will be used.
    levels (List[float]): List of severity levels to apply for each perturbation. If empty,
      default robustness levels will be used.
    maxSamples (int | None): Maximum number of samples to evaluate per run/level. If None,
      evaluates all available samples. Default: 200.
    preprocessFn (callable | None): Optional preprocessing function applied to each image before
      prediction. When provided it should accept a PIL image and return a processed image.
    subset (str): Dataset subset to evaluate: one of ("train", "val", "test", "all").
      Defaults to "test".
    eps (float): Small epsilon for numerical stability in metric computations. Default: 1e-10.
    dpi (int): DPI used when saving figures. Default: 300.

  Returns:
    dict: A dictionary containing the robustness evaluation results. The dictionary contains keys
      such as "Dataset", "ClassNames", "Perturbations", "AverageAccuracy", "WorstAccuracy",
      and an inner "RobustnessMetrics" structure with aggregated measures. In addition to the
      return value, the function writes these artifacts to `storeDir` (e.g. "RobustnessReport.json",
      "RobustnessResults.csv", "RobustnessSummary.csv", and "Interpretation.txt").

  Example
  -------
  .. code-block:: python

    EvaluateModelOnPerturbations(
      model=myModel,
      run=myRun,
      datasetDir="./data/cifar10",
      storeDir="./out/robustness",
      perturbations=["gaussian","jpeg"],
      levels=[0.1, 0.2, 0.3],
      maxSamples=500,
      subset="test"
    )
  '''

  # Map perturbation name to handler function.
  PERTURBATION_MAP = {
    "baseline"   : lambda im, lv, p: im,
    "gaussian"   : lambda im, lv, p: AddGaussianNoise(im, sigma=lv, seed=p),
    "brightness" : lambda im, lv, p: ChangeBrightness(im, factor=lv),
    "jpeg"       : lambda im, lv, p: ApplyJPEGCompression(im, quality=int(max(1, min(95, int(lv))))),
    "speckle"    : lambda im, lv, p: AddSpeckleNoise(im, var=lv, seed=p),
    "saltPepper" : lambda im, lv, p: AddSaltPepperNoise(im, amount=lv, seed=p),
    "contrast"   : lambda im, lv, p: ChangeContrast(im, factor=lv),
    "shotNoise"  : lambda im, lv, p: AddShotNoise(im, scale=lv, seed=p),
    "motionBlur" : lambda im, lv, p: im.filter(ImageFilter.GaussianBlur(radius=max(1, int(lv)))),
    "downscale"  : lambda im, lv, p: DownscaleImage(im, lv),
    "occlusion"  : lambda im, lv, p: OccludeImage(im, lv),
    "colorJitter": lambda im, lv, p: ColorJitter(im, lv),
    "fog"        : lambda im, lv, p: FogImage(im, lv),
    "snow"       : lambda im, lv, p: AddSaltPepperNoise(im, amount=min(0.5, lv), seed=p),
    "spatter"    : lambda im, lv, p: AddSaltPepperNoise(im, amount=min(0.3, lv), seed=p),
    "glassBlur"  : lambda im, lv, p: PixelateImage(im, max(1, int(lv))).filter(ImageFilter.GaussianBlur(radius=1)),
    "pixelate"   : lambda im, lv, p: PixelateImage(im, lv),
    "saturate"   : lambda im, lv, p: SaturateImage(im, lv),
    "zoomBlur"   : lambda im, lv, p: im.filter(ImageFilter.GaussianBlur(radius=max(1, int(lv)))),
    "shadow"     : lambda im, lv, p: Image.blend(im, Image.new("RGB", im.size, (0, 0, 0)), min(0.7, lv * 0.02)),
  }

  run = Path(run)
  storeDir = Path(storeDir)
  storeDir.mkdir(parents=True, exist_ok=True)

  reportPath = storeDir / "RobustnessReport.json"
  csvPath = storeDir / "RobustnessResults.csv"
  summaryCsv = storeDir / "RobustnessSummary.csv"
  interpTxt = storeDir / "Interpretation.txt"

  AVAILABLE_PERTURBATIONS = [
    "gaussian", "brightness", "jpeg", "speckle", "saltPepper", "contrast", "shotNoise",
    "motionBlur", "downscale", "occlusion", "colorJitter", "fog", "snow", "spatter",
    "glassBlur", "pixelate", "saturate", "zoomBlur", "shadow",
  ]
  DEFAULT_ROBUSTNESS_LEVELS = [0.1, 0.2, 0.3, 10, 20, 30, 40, 50]
  if (len(perturbations) <= 0):
    perturbations = AVAILABLE_PERTURBATIONS
  if (len(levels) <= 0):
    levels = DEFAULT_ROBUSTNESS_LEVELS

  results = {
    "Dataset"      : str(datasetDir),
    "ClassNames"   : [],
    "Perturbations": []
  }

  (
    predsCsvPath, weightedMetrics, allPredsIndices, allGtsIndices,
    allPredsProbs, allPredsConfs, predictionsRecords, classNames, cm
  ) = GenericEvaluatePredictPlotSubset(
    datasetDir=str(datasetDir),
    model=model,
    subset=subset,
    prefix="",
    storageDir=str(storeDir),
    heavy=False,
    computeECE=True,
    saveArtifacts=False,
    exportFailureCases=False,
    eps=eps,
    maxSamples=maxSamples,
    preprocessFn=preprocessFn,
  )

  results["ClassNames"] = classNames
  baselineAcc = weightedMetrics["Weighted Accuracy"]
  baselineEce = weightedMetrics["ECE"]
  duration = weightedMetrics["Total Evaluation Time (s)"]
  confmat = cm.tolist() if (isinstance(cm, np.ndarray)) else cm

  baselineEntry = {
    "Perturbation": "baseline",
    "Levels"      : [
      {
        "Level"          : "baseline",
        "Accuracy"       : baselineAcc,
        "Samples"        : len(allGtsIndices),
        "DurationSeconds": duration,
        "ECE"            : baselineEce,
        "ConfusionMatrix": confmat,
        "Confidences"    : allPredsConfs,
        "Correctness"    : [el["correctness"] for el in predictionsRecords],
        "MeanConfidence" : float(np.mean(allPredsConfs)) if (allPredsConfs) else None,
        "Brier"          : ComputeBrierScore(allPredsConfs, [el["correctness"] for el in predictionsRecords]),
        "ConfAccGap"     : (float(np.mean(allPredsConfs)) - baselineAcc) if allPredsConfs else None,
        "ClasswiseAcc"   : {},  # Will be filled below if needed.
        "PredsCsvPath"   : str(predsCsvPath),
        "WeightedMetrics": weightedMetrics,
      }
    ]
  }
  results["Perturbations"].insert(0, baselineEntry)

  # Fill baseline ClasswiseAcc.
  if (allGtsIndices and allPredsIndices):
    correctByClass = Counter()
    totalByClass = Counter()
    for gt, pred in zip(allGtsIndices, allPredsIndices):
      totalByClass[gt] += 1
      if (gt == pred):
        correctByClass[gt] += 1
    baselineClasswise = {
      str(cls): correctByClass[cls] / totalByClass[cls]
      for cls in totalByClass
    }
    results["Perturbations"][0]["Levels"][0]["ClasswiseAcc"] = baselineClasswise
  print(f"Baseline accuracy: {baselineAcc:.4f} over {len(allGtsIndices)} samples.")

  for perturb in perturbations:
    print(f"Evaluating perturbation: {perturb}")
    perturbResults = {"Perturbation": perturb, "Levels": []}
    for level in levels:
      print(f"    Level: {level}")
      handler = PERTURBATION_MAP.get(perturb)
      seed = int(hashlib.sha1(str(perturb).encode("utf-8")).hexdigest()[:8], 16)
      print(f"    Using seed: {seed}")
      preprocessFn = lambda img, h=handler, l=level, s=seed: (h(img, l, s) if (h is not None) else img)
      (
        _predsCsvPath, _weightedMetrics, _allPredsIndices, _allGtsIndices,
        _allPredsProbs, _allPredsConfs, _predictionsRecords, _classNames, _cm
      ) = GenericEvaluatePredictPlotSubset(
        datasetDir=str(datasetDir),
        model=model,
        subset=subset,
        prefix="",
        storageDir=str(storeDir),
        heavy=False,
        computeECE=True,
        saveArtifacts=False,
        exportFailureCases=False,
        eps=eps,
        maxSamples=maxSamples,
        preprocessFn=preprocessFn,
      )

      accuracy = _weightedMetrics["Weighted Accuracy"]
      duration = _weightedMetrics["Total Evaluation Time (s)"]
      ece = _weightedMetrics["ECE"]
      confmat = _cm.tolist() if isinstance(_cm, np.ndarray) else _cm

      # Compute per-level auxiliary metrics.
      confidencesLevel = _allPredsConfs
      correctnessLevel = [el["correctness"] for el in _predictionsRecords]
      meanConfLevel = float(np.mean(confidencesLevel)) if (confidencesLevel) else None
      brierLevel = ComputeBrierScore(confidencesLevel, correctnessLevel)
      confAccGap = (meanConfLevel - accuracy) if (meanConfLevel is not None) else None

      # Classwise accuracy (optional, lightweight).
      classwiseAcc = {}
      if (_allGtsIndices and _allPredsIndices):
        try:
          totalByClass = Counter()
          correctByClass = Counter()
          for gt, pred in zip(_allGtsIndices, _allPredsIndices):
            totalByClass[gt] += 1
            if (gt == pred):
              correctByClass[gt] += 1
          classwiseAcc = {
            str(cls): correctByClass[cls] / totalByClass[cls]
            for cls in totalByClass
          }
        except Exception:
          classwiseAcc = {}

      perturbResults["Levels"].append({
        "Level"          : level,
        "Accuracy"       : accuracy,
        "Samples"        : len(_allGtsIndices),
        "DurationSeconds": duration,
        "ECE"            : ece,
        "ConfusionMatrix": confmat,
        "Confidences"    : confidencesLevel,
        "Correctness"    : correctnessLevel,
        "MeanConfidence" : meanConfLevel,
        "Brier"          : brierLevel,
        "ConfAccGap"     : confAccGap,
        "ClasswiseAcc"   : classwiseAcc,
        "PredsCsvPath"   : str(_predsCsvPath),
        "WeightedMetrics": _weightedMetrics,
      })
    results["Perturbations"].append(perturbResults)

  perturbList = results["Perturbations"]
  overallAccs, perturbSummaries = [], []

  for perturbEntry in perturbList:
    perturbName = perturbEntry["Perturbation"]
    levelAccs, levelDetails = [], []
    levelsReported = perturbEntry["Levels"]
    for levelEntry in levelsReported:
      acc = levelEntry["Accuracy"] if (levelEntry["Accuracy"] is not None) else 0.0
      samples = levelEntry["Samples"] if (levelEntry["Samples"] is not None) else 0
      ece = levelEntry["ECE"] if (levelEntry["ECE"] is not None) else 0.0
      levelAccs.append(float(acc))
      levelDetails.append({
        "Level"          : levelEntry["Level"],
        "Accuracy"       : float(acc),
        "Samples"        : int(samples),
        "ECE"            : ece,
        "Confidences"    : levelEntry["Confidences"],
        "MeanConfidence" : levelEntry["MeanConfidence"],
        "DurationSeconds": levelEntry["DurationSeconds"],
        "ConfAccGap"     : levelEntry["ConfAccGap"],
        "ClasswiseAcc"   : levelEntry["ClasswiseAcc"],
        "PredsCsvPath"   : levelEntry["PredsCsvPath"],
        "WeightedMetrics": levelEntry["WeightedMetrics"],
      })
    if (levelAccs):
      meanAcc = sum(levelAccs) / len(levelAccs)
      bestAcc = max(levelAccs)
      worstAcc = min(levelAccs)
      perturbSummaries.append((perturbName, meanAcc, bestAcc, worstAcc, levelDetails))
      overallAccs.extend(levelAccs)
  # Compute baseline, average, worst.
  baseline = avg = worst = 0.0
  if (overallAccs):
    baseline = None
    try:
      for p in perturbList:
        if (str(p["Perturbation"]).lower() == "baseline"):
          lvl0 = p["Levels"][0]
          bacc = lvl0["Accuracy"]
          if (bacc is not None):
            baseline = float(bacc)
            break
    except Exception:
      pass
    if (baseline is None):
      baseline = max(overallAccs)
    avg = sum(overallAccs) / len(overallAccs)
    worst = min(overallAccs)

  extendedMetrics = {}
  perPerturbMetrics = []
  baselineConfidences = []

  for (pname, meanAcc, bestAcc, worstAcc, details) in perturbSummaries:
    sevAcc = [
      (float(d["Level"]) if ((d["Level"] is not None) and (d["Level"] != "baseline")) else i, float(d["Accuracy"]))
      for i, d in enumerate(details)
    ]
    sevAcc = sorted(sevAcc, key=lambda x: x[0])
    sev = [s for s, _ in sevAcc]
    accs = [a for _, a in sevAcc]

    # Aggregate confidence-accuracy gap.
    gaps = [d["ConfAccGap"] for d in details if (d["ConfAccGap"] is not None)]
    meanConfAccGap = float(np.mean(gaps)) if gaps else None

    # Compute classwise drops vs baseline
    classwiseDrops = {}
    if ("ClasswiseAcc" in details[0] and baselineClasswise):
      for cls in baselineClasswise:
        drops = []
        for d in details:
          accDict = d.get("ClasswiseAcc", {})
          if (cls in accDict):
            drop = baselineClasswise[cls] - accDict[cls]
            drops.append(drop)
        if (drops):
          classwiseDrops[cls] = float(np.mean(drops))

    if (len(sev) >= 2):
      auc = float(np.trapz(accs, x=sev))
      maxArea = (max(sev) - min(sev)) * (baseline if (baseline > 0) else 1.0)
      normAuc = auc / maxArea if maxArea > 0 else 0.0
    else:
      auc = accs[0] if (accs) else 0.0
      normAuc = (auc / (baseline if (baseline > 0) else 1.0)) if (baseline > 0) else 0.0

    mceP = 0.0
    if (baseline > 0):
      relErrs = [max(0.0, (baseline - a) / baseline) for a in accs]
      mceP = float(np.mean(relErrs)) if (relErrs) else 0.0

    meanConfs = [d["MeanConfidence"] for d in details if (d["MeanConfidence"] is not None)]
    meanConf = float(np.mean(meanConfs)) if (meanConfs) else None
    if (meanConf is not None):
      baselineConfidences.append(meanConf)

    eces = [d["ECE"] for d in details if d["ECE"] is not None]
    meanEce = float(np.mean([float(e) for e in eces])) if eces else None

    latencies = [d["DurationSeconds"] for d in details if (d["DurationSeconds"] is not None)]
    meanLatency = float(np.mean(latencies)) if (latencies) else None

    perPerturbMetrics.append({
      "Perturbation"       : pname,
      "MeanAcc"            : meanAcc,
      "BestAcc"            : bestAcc,
      "WorstAcc"           : worstAcc,
      "MceLike"            : mceP,
      "AUC"                : auc,
      "NormAUC"            : normAuc,
      "MeanECE"            : meanEce,
      "MeanConfidence"     : meanConf,
      "MeanDurationSeconds": meanLatency,
      "MeanConfAccGap"     : meanConfAccGap,
      "ClasswiseDrops"     : classwiseDrops,
    })

  mceVals = [p["MceLike"] for p in perPerturbMetrics if (p["MceLike"] is not None)]
  overallMce = float(np.mean(mceVals)) if mceVals else None
  baselineConf = max(baselineConfidences) if baselineConfidences else None

  extendedMetrics.update({
    "PerPerturbMetrics"  : perPerturbMetrics,
    "OverallMce"         : overallMce,
    "BaselineConfidence" : baselineConf,
    "RelativeDropOverall": {
      "BaselineMinusWorst": float(baseline - worst),
      "BaselineMinusMean" : float(baseline - avg)
    }
  })
  results["RobustnessMetrics"] = extendedMetrics

  perturbSummariesDicts = []
  for (pname, meanAcc, bestAcc, worstAcc, details) in perturbSummaries:
    perturbSummariesDicts.append({
      "Perturbation": pname,
      "MeanAcc"     : float(meanAcc) if (meanAcc is not None) else None,
      "BestAcc"     : float(bestAcc) if (bestAcc is not None) else None,
      "WorstAcc"    : float(worstAcc) if (worstAcc is not None) else None,
      "Details"     : details,
    })
  results.update({
    "BaselineAccuracy": float(baseline) if (baseline is not None) else None,
    "AverageAccuracy" : float(avg) if (avg is not None) else None,
    "WorstAccuracy"   : float(worst) if (worst is not None) else None,
    "PerturbSummaries": perturbSummariesDicts,
  })
  DumpJsonFile(str(reportPath), results)
  print(f"Robustness report saved to: {reportPath}")

  perLevelRows = []
  perLevelHeader = ["Perturbation", "Level", "Accuracy", "Samples", "ECE"]
  for pEntry in perturbList:
    pName = pEntry["Perturbation"]
    levelsReported = pEntry["Levels"]
    for lvl in levelsReported:
      lvlLabel = lvl["Level"]
      accVal = lvl["Accuracy"]
      samplesVal = lvl["Samples"]
      eceVal = lvl["ECE"]
      perLevelRows.append([pName, lvlLabel, accVal, samplesVal, eceVal])

  if (perLevelRows):
    AppendOrCreateNewCSV(str(csvPath), perLevelRows, header=perLevelHeader, mode="w")
    print(f"Per-level robustness results CSV saved to: {csvPath}")

  summaryRows = []
  summaryHeader = [
    "Perturbation", "MeanAcc", "BestAcc", "WorstAcc", "MceLike", "AUC",
    "NormAUC", "MeanECE", "MeanConfidence", "MeanDurationSeconds",
    "MeanConfAccGap", "ClasswiseDrops"
  ]
  for rec in extendedMetrics["PerPerturbMetrics"]:
    summaryRows.append([
      rec["Perturbation"], rec["MeanAcc"], rec["BestAcc"], rec["WorstAcc"],
      rec["MceLike"], rec["AUC"], rec["NormAUC"], rec["MeanECE"],
      rec["MeanConfidence"], rec["MeanDurationSeconds"],
      rec["MeanConfAccGap"], json.dumps(rec["ClasswiseDrops"])
    ])
  if (summaryRows):
    AppendOrCreateNewCSV(str(summaryCsv), summaryRows, header=summaryHeader, mode="w")
    print(f"Per-perturbation summary CSV saved to: {summaryCsv}")

  perturbNames = [p[0] for p in perturbSummaries] if (perturbSummaries) else [p["Perturbation"] for p in perturbList]
  orderedLevelLabels = []
  if (perLevelRows):
    orderedLevelLabels = sorted({str(r[1]) for r in perLevelRows}, key=lambda x: (x != "baseline", x))

  heatMat = np.full((len(perturbNames), len(orderedLevelLabels)), np.nan, dtype=float)
  for row in perLevelRows:
    pName, lvlLabel, accVal, _, _ = row
    if (pName in perturbNames and str(lvlLabel) in orderedLevelLabels):
      i = perturbNames.index(pName)
      j = orderedLevelLabels.index(str(lvlLabel))
      heatMat[i, j] = float(accVal) if (accVal is not None) else np.nan

  PlotHeatmap(
    data=heatMat,
    rowLabels=perturbNames,
    colLabels=orderedLevelLabels,
    title="Accuracy Heatmap (Top-1) by Perturbation and Level",
    xlabel="Level",
    ylabel="Perturbation",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    valueFormat="{:.2f}",
    savePath=storeDir / "AccuracyHeatmap.png",
    dpi=dpi,
    save=True,
    display=False,
  )

  # mCE Bar Chart.
  mceVals = [rec["MceLike"] for rec in extendedMetrics["PerPerturbMetrics"]]
  names = [rec["Perturbation"] for rec in extendedMetrics["PerPerturbMetrics"]]
  if (names and any(v is not None for v in mceVals)):
    mcePath = storeDir / "McePerPerturbation.png"
    PlotBarChart(
      values=mceVals,
      labels=names,
      title="mCE-like Metric by Perturbation",
      ylabel="Mean relative error (mCE-like)",
      savePath=mcePath,
      color="tab:orange",
      alpha=0.9,
      dpi=dpi,
      display=False,
      save=True,
      annotate=True,
    )

  # ECE Heatmap.
  if (perLevelRows and perturbNames and orderedLevelLabels):
    eceMat = np.full((len(perturbNames), len(orderedLevelLabels)), np.nan, dtype=float)
    for row in perLevelRows:
      pName, lvlLabel, _, _, eceVal = row
      if (pName in perturbNames and str(lvlLabel) in orderedLevelLabels):
        i = perturbNames.index(pName)
        j = orderedLevelLabels.index(str(lvlLabel))
        eceMat[i, j] = float(eceVal) if eceVal is not None else np.nan

    figEce, axEce = plt.subplots(
      figsize=(max(6, len(orderedLevelLabels) * 0.6), max(4, len(perturbNames) * 0.4))
    )
    cax2 = axEce.imshow(eceMat, aspect="auto", interpolation="nearest", cmap="magma")
    axEce.set_yticks(range(len(perturbNames)))
    axEce.set_yticklabels(perturbNames)
    axEce.set_xticks(range(len(orderedLevelLabels)))
    axEce.set_xticklabels(orderedLevelLabels, rotation=45, ha="right")
    axEce.set_xlabel("Level")
    axEce.set_ylabel("Perturbation")
    axEce.set_title("ECE Heatmap by Perturbation and Level")
    figEce.colorbar(cax2, ax=axEce, label="ECE")
    for i in range(eceMat.shape[0]):
      for j in range(eceMat.shape[1]):
        val = eceMat[i, j]
        if (not np.isnan(val)):
          axEce.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=7)
    ecePath = storeDir / "EceHeatmap.png"
    figEce.tight_layout()
    figEce.savefig(str(ecePath), dpi=dpi, bbox_inches="tight")
    plt.close(figEce)
    print(f"ECE heatmap saved to: {ecePath}")

  # Per-level diagnostics.
  reportClassNames = results["ClassNames"]
  for pEntry in perturbList:
    pName = pEntry["Perturbation"]
    levelsReported = pEntry["Levels"]
    for lvl in levelsReported:
      lvlLabel = lvl["Level"]
      cmRaw = lvl["ConfusionMatrix"]
      if (cmRaw is not None):
        cmArr = np.array(cmRaw)
        if reportClassNames and len(reportClassNames) == cmArr.shape[0]:
          classNamesForCm = list(reportClassNames)
        else:
          classNamesForCm = [str(i) for i in range(cmArr.shape[0])]
        (storeDir / "CMs").mkdir(parents=True, exist_ok=True)
        PlotConfusionMatrix(
          cmArr,
          classNamesForCm,
          normalize=False,
          roundDigits=3,
          title=f"Confusion Matrix - {pName} - Level {lvlLabel}",
          cmap=plt.cm.Blues,
          display=False,
          save=True,
          fileName=str(storeDir / "CMs" / f"{pName}_{lvlLabel}_ConfusionMatrix.pdf"),
          fontSize=10,
          annotate=True,
          figSize=(6, 6),
          colorbar=True,
          returnFig=False,
          dpi=dpi,
        )
        print(f"Confusion matrix saved for {pName} level {lvlLabel}")

      confidences = lvl["Confidences"]
      correctness = lvl["Correctness"]
      (storeDir / "Reliability").mkdir(parents=True, exist_ok=True)
      if (confidences is not None and correctness is not None and len(confidences) == len(correctness)):
        try:
          confArr = np.array(confidences, dtype=float)
          corrArr = np.array(correctness, dtype=float)
          if (corrArr.ndim == 1 and not set(np.unique(corrArr)).issubset({0.0, 1.0})):
            pass
          else:
            bins = np.linspace(0, 1, 11)
            binAcc, binConf = [], []
            for i in range(len(bins) - 1):
              lo, hi = bins[i], bins[i + 1]
              mask = (confArr >= lo) & (confArr <= hi) if i == 0 else (confArr > lo) & (confArr <= hi)
              if (np.sum(mask) == 0):
                binAcc.append(np.nan)
                binConf.append(np.nan)
              else:
                binAcc.append(float(np.mean(corrArr[mask])))
                binConf.append(float(np.mean(confArr[mask])))
            x = [b for b in binConf if not np.isnan(b)]
            y = [a for a in binAcc if not np.isnan(a)]
            if (x):
              figRel, axRel = plt.subplots(figsize=(6, 6))
              axRel.plot([0, 1], [0, 1], "--", color="gray")
              axRel.plot(x, y, "o-", color="blue", label="Model")
              axRel.set_xlim(0, 1)
              axRel.set_ylim(0, 1)
              axRel.set_xlabel("Mean Confidence")
              axRel.set_ylabel("Accuracy")
              axRel.set_title(f"Reliability Diagram - {pName} - Level {lvlLabel}")
              relPath = storeDir / "Reliability" / f"{pName}_{lvlLabel}_Reliability.png"
              figRel.tight_layout()
              figRel.savefig(str(relPath), dpi=dpi, bbox_inches="tight")
              plt.close(figRel)
              print(f"Reliability diagram saved for {pName} level {lvlLabel}")
        except Exception as relEx:
          print(f"Warning: Failed to generate reliability diagram for {pName} level {lvlLabel}: {relEx}")
      else:
        print(f"Skipping reliability diagram for {pName} level {lvlLabel} due to missing data.")

  for perturbEntry in perturbList:
    # Robustly extract perturbation name and levels (support multiple key variants).
    pname = perturbEntry["Perturbation"]
    levelsReported = perturbEntry["Levels"]
    if (not levelsReported):
      continue

    # Build tuples (numericCandidate, labelStr, accuracy) to allow numeric sorting when possible.
    levelTuples = []
    for lvl in levelsReported:
      rawLabel = lvl["Level"]
      accVal = lvl["Accuracy"]
      try:
        num = float(rawLabel)
      except:
        num = 0
      levelTuples.append((num, str(rawLabel), float(accVal)))
    # Sort by numeric severity when available, otherwise by label.
    levelTuples = sorted(levelTuples, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0.0, x[1]))

    xLabels = [t[1] for t in levelTuples]
    yAccs = [t[2] for t in levelTuples]

    # Filename-safe perturbation name.
    pertName = (pname[0].upper() + pname[1:]) if (isinstance(pname, str) and pname) else "Perturbation"
    safeName = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in pertName)
    figPath = storeDir / "BarCharts" / f"{safeName}_Accuracy.png"

    # Call the project's `PlotBarChart` with the same style/params as the existing call.
    PlotBarChart(
      values=yAccs,
      labels=xLabels,
      title=f"Robustness Accuracy by Level - {pertName}",
      ylabel="Top-1 Accuracy",
      savePath=figPath,
      colors=COLORS,
      alpha=0.85,
      dpi=dpi,
      display=False,
      save=True,
      annotate=True,
    )
    print(f"Interpretation figure written to: {figPath}")

  perturbations = results["Perturbations"]
  baselineRecord = [p for p in perturbations if (p["Perturbation"] == "baseline")]
  if (not baselineRecord):
    raise ValueError("Baseline record not found in robustness report.")
  baselineRecord = baselineRecord[0]["Levels"][0]
  baselineAcc = baselineRecord["Accuracy"]
  robustnessMetrics = results["RobustnessMetrics"]
  overallMCE = robustnessMetrics["OverallMce"]
  relativeDropOverall = robustnessMetrics["RelativeDropOverall"]
  perPerturbMetrics = robustnessMetrics["PerPerturbMetrics"]
  baselineMinusWorst = relativeDropOverall["BaselineMinusWorst"]
  baselineMinusMean = relativeDropOverall["BaselineMinusMean"]
  avg = results["AverageAccuracy"]
  worst = results["WorstAccuracy"]
  perturbSummaries = results["PerturbSummaries"]

  # Build a detailed interpretation report with per-perturbation analysis and actionable guidance.
  lines = [
    "Robustness Interpretation",
    f"Generated: {datetime.now(timezone.utc).isoformat()}",  # UTC timestamp.
    f"Dataset: {results['Dataset']}",  # Dataset name.
    f"Model run: {run.name}",  # Model run.
    "",
    "This report summarizes the robustness evaluation results of the model "
    "across various perturbations and severity levels. It highlights key metrics, "
    "identifies potential weaknesses, and provides actionable recommendations "
    "to improve model robustness.",
    "",
    "Overall summary:",
    f"- Best observed accuracy (baseline): {baselineAcc:.3f}",
    f"- Average accuracy across all perturbations/levels: {avg:.3f}",
    f"- Worst observed accuracy: {worst:.3f}",
    f"- mean Corruption Error (mCE-like): {overallMCE} (fraction of baseline)",
    "- Lower mCE values indicate better robustness; aim for mCE < 0.5 for good robustness.",
    f"- Relative drops: baseline-worst={baselineMinusWorst}, baseline-mean={baselineMinusMean}",
    "",
    "Per-perturbation details:"
  ]

  # Iterate through the earlier-built perturbation summaries.
  for dictRecord in perturbSummaries:
    # Extract perturbation name and level summary details.
    pname = dictRecord["Perturbation"]
    meanAcc = dictRecord["MeanAcc"]
    bestAcc = dictRecord["BestAcc"]
    worstAcc = dictRecord["WorstAcc"]
    details = dictRecord["Details"]
    # Compute drop from baseline for this perturbation's worst level.
    dropFromBaseline = baselineAcc - worstAcc
    # Append header for this perturbation.
    lines.extend([
      f"\n{pname}:",
      f"  - mean accuracy: {meanAcc:.3f} | best: {bestAcc:.3f} | worst: {worstAcc:.3f}",
      f"  - max drop from baseline: {dropFromBaseline * 100:.1f}%"
    ])
    # Level-by-level breakdown when details are present.
    for detail in details:
      # Extract level, accuracy, samples and ECE from the level detail dictionary.
      levelVal = detail["Level"]
      acc = detail["Accuracy"]
      samples = detail["Samples"]
      ece = detail["ECE"]
      # Decide whether to add a low-sample warning.
      sampleNote = "" if (samples >= 30) else " (low sample count - interpret with caution)"
      # Build an ECE note string based on available types.
      eceNote = (
        f" | ECE={ece:.3f}"
        if (isinstance(ece, (int, float)))
        else (f" | ECE={ece}" if (ece is not None) else "")
      )
      # Append a line describing this severity level's accuracy and sample count.
      lines.append(f"    - Level {levelVal}: acc={acc:.3f}, samples={samples}{sampleNote}{eceNote}")

    # Add compact per-perturbation metrics if available from extended metrics.
    perpm = next((p for p in perPerturbMetrics if (p["Perturbation"] == pname)), None)
    # Append computed metrics to the interpretation when available.
    if (perpm is not None):
      try:
        # Append AUC and normalized AUC lines.
        lines.append(
          f"  - AUC (accuracy vs severity): {perpm['AUC']:.4f} | normalized AUC: {perpm['NormAUC']:.4f}"
        )
        # Append mean ECE when present.
        # if (perpm.get("MeanECE") is not None):
        lines.append(f"  - mean ECE across levels: {perpm['MeanEce']:.4f}")
        # Append mean confidence when present.
        #             if (perpm["MeanConfidence"] is not None):
        lines.append(f"  - mean confidence: {perpm['MeanConfidence']:.4f}")
        # Append mean latency when present.
        #             if (perpm.get("MeanDurationSeconds") is not None):
        latencyMs = perpm["MeanDurationSeconds"] * 1000
        lines.append(f"  - mean latency: {latencyMs:.2f} ms")
      except Exception:
        # Ignore errors while formatting per-perturbation metrics.
        pass

    # Actionable recommendation for this perturbation based on drop magnitude.
    if (dropFromBaseline > 0.15):
      lines.append(
        "  => Recommendation: Large drop observed. Strongly consider adding this perturbation "
        "to training augmentation, use adversarial/robust training, monitor calibration, or ensemble models."
      )
    elif (dropFromBaseline > 0.05):
      lines.append(
        "  => Recommendation: Moderate degradation. Try targeted augmentation, "
        "temperature scaling for calibration, or collect more data at problematic levels."
      )
    else:
      lines.append(
        "  => Recommendation: Model performs reasonably; validate on larger sample sizes "
        "and consider expanding perturbation types."
      )

  # High-level recommendations and next steps.
  lines.append("")
  lines.append("High-level recommendations:")
  # Add tiered advice based on the worst-case drop across perturbations.
  if ((baselineAcc - worst) > 0.15):
    lines.append(
      "- Significant robustness issues detected across tested perturbations (>15% drop). "
      "Prioritize: augmentation with specific perturbations, adversarial/robust training, "
      "evaluation on holdout sets, and model ensembling where appropriate."
    )
  elif ((baselineAcc - worst) > 0.05):
    lines.append(
      "- Moderate robustness variation detected. Consider targeted augmentations, "
      "calibration (e.g., temperature scaling or label smoothing), and gathering "
      "more evaluation samples for weak conditions."
    )
  else:
    lines.append(
      "- Minimal robustness degradation observed. Confirm findings by increasing "
      "sample sizes and testing additional perturbation types (e.g., occlusion, blur, domain shifts)."
    )

  # Confidence Calibration Insights.
  lines.append("")
  lines.append("Confidence Calibration Insights:")
  # Compute average confidence-accuracy gap across non-baseline perturbations.
  nonBaselineMetrics = [p for p in perPerturbMetrics if p["Perturbation"] != "baseline"]
  gaps = [p["MeanConfAccGap"] for p in nonBaselineMetrics if (p["MeanConfAccGap"] is not None)]
  if (gaps):
    avgGap = float(np.mean(gaps))
    if (avgGap > 0.05):
      lines.append(
        f"- Model is overconfident under perturbation (mean confidence exceeds accuracy by {avgGap:.3f}).")
      lines.append("  => Consider temperature scaling or recalibration to align confidence with accuracy.")
    elif (avgGap < -0.05):
      lines.append(
        f"- Model is underconfident under perturbation (accuracy exceeds mean confidence by {-avgGap:.3f})."
      )
      lines.append(
        "  => May indicate conservative predictions; consider confidence recalibration or threshold tuning."
      )
    else:
      lines.append(f"- Confidence is well-aligned with accuracy (mean gap: {avgGap:.3f}).")
  else:
    lines.append("- Confidence–accuracy gap data not available.")

  # Brier score insight (if available).
  brierScores = []
  for p in perturbations:
    if (p["Perturbation"] == "baseline"):
      continue
    for lvl in p["Levels"]:
      if "Brier" in lvl and lvl["Brier"] is not None:
        brierScores.append(lvl["Brier"])
  if (brierScores):
    avgBrier = np.mean(brierScores)
    lines.append(f"- Average Brier score across perturbed levels: {avgBrier:.4f} (lower is better).")

  # Classwise Robustness Analysis.
  lines.append("")
  lines.append("Classwise Robustness Analysis:")
  allClassDrops = {}
  for p in nonBaselineMetrics:
    drops = p["ClasswiseDrops"]
    for cls, drop_val in drops.items():
      if (cls not in allClassDrops):
        allClassDrops[cls] = []
      allClassDrops[cls].append(drop_val)

  if (allClassDrops):
    # Compute mean drop per class
    meanDrops = {cls: np.mean(drops) for cls, drops in allClassDrops.items()}
    # Sort by drop (largest first)
    topFragile = sorted(meanDrops.items(), key=lambda x: x[1], reverse=True)[:5]
    lines.append("- Top fragile classes (mean accuracy drop under perturbation):")
    for cls, drop in topFragile:
      lines.append(f"  - Class '{cls}': -{drop:.1%}")
    # Suggest action
    lines.append(
      "  => Recommendation: Inspect misclassifications for these classes, augment training data with "
      "perturbed examples of these classes, or use class-balanced robust training."
    )
  else:
    lines.append("- Classwise robustness data not available (requires per-class predictions).")

  # Notes and cautions to help interpret the results.
  lines.extend([
    # Notes and cautions to help interpret the results.
    "",
    "Notes and cautions:",
    "- ECE (expected calibration error) values, when present, "
    "indicate how well model confidences match accuracy; high ECE suggests recalibration is needed.",
    "- Small sample sizes per level reduce confidence in the measured accuracy; ",
    "aim for >=30 samples per level when possible.",
    "- If making deployment decisions, always re-evaluate accuracy on ",
    "your target environment and downstream tasks.",
    # Add an explanatory section describing the step and benefits.
    "",
    "Interpretation Help:",
    "- What this step does: Evaluates how model accuracy and calibration change under controlled data ",
    "perturbations (for example: noise, brightness shifts, JPEG compression). This helps quantify model ",
    "sensitivity to input degradations.",
    "- Why it matters: Understanding robustness highlights failure modes that may occur in deployment ",
    "(poor lighting, compression artifacts, sensor noise). Addressing these can reduce unexpected model failures.",
    "- Practical next steps: If large drops are observed for a perturbation, augment training data with ",
    "that perturbation, consider robust/adversarial training, or apply calibration techniques like ",
    "temperature scaling. Validate any mitigation on a held-out test set before deployment.",
    "- How to use results: Use the per-perturbation and per-level breakdown to prioritize which corruptions ",
    "to simulate during training and which to collect more real-world data for.",
    # Additional decision-oriented guidance and ranking.
    "",
    "Decision guide and ranking:",
    "- Quick decision thresholds (example): <5% worst-drop=OK for deployment, "
    "5-15%=Moderate (fix before high-risk ",
    "deployments), >15%=High risk (mitigate before deployment). ",
    "Adjust thresholds to your business risk.",
  ])

  # Build a simple ranking of perturbations by max drop from baseline for human prioritization.
  # ranked = sorted(perturbSummaries, key=lambda x: (baselineAcc - x[3]), reverse=True)
  # ranked = sorted(perturbSummaries, key=lambda x: (baselineAcc - (x["WorstAcc"] or 0.0)), reverse=True)
  # Build a simple ranking of perturbations by max drop from baseline.
  ranked = sorted(
    perturbSummaries,
    key=lambda x: baselineAcc - (x["WorstAcc"] if (x["WorstAcc"] is not None) else 0.0),
    reverse=True
  )
  # Append ranking header.
  lines.append("- Ranked perturbations by worst-case impact (largest drop first):")
  # Append each perturbation's worst-case information to the interpretation.
  for dictRecord in ranked:
    # Extract perturbation name and worst accuracy.
    pname = dictRecord["Perturbation"]
    worstAcc = dictRecord["WorstAcc"]
    # Compute the drop from baseline for reporting.
    drop = baselineAcc - worstAcc
    # Append a line describing the worst-case impact for this perturbation.
    lines.append(f"  - {pname}: worst acc={worstAcc:.3f}, dropFromBaseline={drop * 100:.1f}%")
  # Add a short recommendation on how to prioritize fixes using this ranking.
  lines.append("")
  lines.append("- Prioritization advice:")
  lines.append(
    " 1) Focus on perturbations at the top of the ranking with large drops and reasonable sample counts. "
    " 2) For moderate drops, prototype lightweight augmentations and recalibrate confidence. "
    " 3) For large drops, prefer stronger interventions (robust training or collecting targeted data)."
  )

  # Save interpretation file.
  with open(interpTxt, "w", encoding="utf-8") as f:
    # Write the joined interpretation lines to disk.
    f.write("\n".join(lines))
  # Announce that the interpretation file was written.
  # Show where the interpretation file was written.
  print(f"Interpretation written to: {interpTxt}")
