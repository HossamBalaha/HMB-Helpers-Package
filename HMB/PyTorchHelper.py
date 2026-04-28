import os, timm, torch, copy, time, json, hashlib, csv
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from HMB.PerformanceMetrics import PlotConfusionMatrix, ComputeBrierScore
from HMB.Utils import DumpJsonFile, AppendOrCreateNewCSV
from HMB.PlotsHelper import PlotHeatmap, PlotBarChart, COLORS
from HMB.ImagesHelper import *


def GetParamCount(model, restrictGrad=True, formatMillions=True):
  r'''
  Get the total number of parameters in a PyTorch model, optionally restricting to only those that require gradients (trainable parameters).

  Parameters:
    model (torch.nn.Module): The model to count parameters for.
    restrictGrad (bool): If True, only count parameters that require gradients (trainable). If False, count all parameters.
    formatMillions (bool): If True, format the result in millions with one decimal place. If False, return the raw count of parameters.

  Returns:
    int: Total number of parameters in the model, optionally restricted to those that require gradients.
  '''

  if (restrictGrad):
    # Sum the number of elements for all trainable parameters.
    result = sum(p.numel() for p in model.parameters() if (p.requires_grad))
  else:
    # Sum the number of elements for all parameters regardless of requires_grad.
    result = sum(p.numel() for p in model.parameters())

  if (formatMillions):
    # Format the result in millions with one decimal place.
    result = result / float(1e6)

  return result


def GetPyTorchDeviceName(device):
  r'''
  Get a human-readable name for a PyTorch device.

  Parameters:
    device (torch.device): The device to get the name for.

  Returns:
    str: A human-readable name for the device (e.g., "CPU", "NVIDIA GeForce RTX 3080").
  '''

  # Check if the device type is CUDA to include specific device name in title.
  if (device.type == "cuda"):
    # Attempt to retrieve CUDA device information.
    try:
      # Get the current CUDA device index.
      devIdx = torch.cuda.current_device()
      # Get the name of the current CUDA device.
      devName = torch.cuda.get_device_name(devIdx)
    # Catch any exception during CUDA device retrieval.
    except Exception:
      # Set a default device name if retrieval fails.
      devName = "CUDA"
  else:
    # For non-CUDA devices, use the device type as the name.
    devName = device.type.upper()

  return devName


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


def LoadCheckpoint(checkpointFile, model, optimizer, lr, device, strict=True, verbose=False):
  r'''
  Load model and optimizer states from a specified checkpoint file with automatic key reconciliation.

  This function initializes the model and optimizer using weights and states stored in a checkpoint file.
  It incorporates a robust mechanism to handle naming convention mismatches (e.g., camelCase versus snake_case)
  between the checkpoint keys and the model definition. If key mismatches are detected, the function employs
  fuzzy string matching to map unexpected keys to missing keys, validated by tensor shape compatibility to
  ensure weight integrity. Additionally, it updates the learning rate for the optimizer if specified.

  Parameters:
    checkpointFile (str): The absolute or relative path to the checkpoint file.
    model (torch.nn.Module): The neural network model into which the state dictionary will be loaded.
    optimizer (torch.optim.Optimizer): The optimizer into which the state dictionary will be loaded. If None, optimizer state loading is skipped.
    lr (float): The learning rate to enforce for all parameter groups in the optimizer after loading.
    device (torch.device): The target device for loading tensors (e.g., torch.device("cpu") or "cuda").
    strict (bool): If True, raises an error if any keys remain unmatched after reconciliation. If False, allows partial loading.
    verbose (bool): If True, it will print the handling messages.

  Note:
    Key reconciliation uses a similarity threshold of 0.85. Weights are only mapped if the source 
    and target tensor shapes match exactly to prevent parameter corruption.
    
  Returns:
    dict: The loaded checkpoint dictionary containing state_dict, optimizer, and other metadata. Returns None if the checkpoint file does not exist.
  '''

  import difflib

  # Check if the checkpoint file exists before loading.
  if (not os.path.exists(checkpointFile)):
    if (verbose):
      print(f"Checkpoint file not found: {checkpointFile}")
    return None

  # Load the checkpoint dictionary from file and map to the specified device.
  checkpoint = torch.load(checkpointFile, map_location=device)

  # Extract the state dict from the checkpoint.
  stateDict = checkpoint["state_dict"]

  # Initial load with strict=False to identify mismatches without raising an error.
  loadReport = model.load_state_dict(stateDict, strict=False)

  # Proceed with key reconciliation if mismatches exist.
  if (len(loadReport.missing_keys) > 0 or len(loadReport.unexpected_keys) > 0):
    if (verbose):
      print(
        f"Key mismatch detected. Missing: {len(loadReport.missing_keys)}, "
        f"Unexpected: {len(loadReport.unexpected_keys)}"
      )

    modelState = model.state_dict()
    newStateDict = {}

    # Track used unexpected keys to ensure one-to-one mapping.
    usedUnexpectedKeys = set()

    # Iterate over missing keys to find the best match from unexpected keys.
    for missingKey in loadReport.missing_keys:
      bestMatch = None
      bestRatio = 0.0

      # Ensure the missing key exists in the current model structure.
      if (missingKey not in modelState):
        continue

      targetShape = modelState[missingKey].shape

      for unexpectedKey in loadReport.unexpected_keys:
        if (unexpectedKey in usedUnexpectedKeys):
          continue

        # Calculate similarity ratio.
        ratio = difflib.SequenceMatcher(None, missingKey, unexpectedKey).ratio()

        # Validate tensor shape to prevent weight corruption.
        if (unexpectedKey in stateDict):
          sourceShape = stateDict[unexpectedKey].shape
          shapeMatch = (targetShape == sourceShape)
        else:
          shapeMatch = False

        # Update best match if ratio is higher and shapes match.
        if (ratio > bestRatio and shapeMatch):
          bestRatio = ratio
          bestMatch = unexpectedKey

      # Apply threshold for confidence (e.g., 0.85).
      if (bestMatch and bestRatio >= 0.85):
        newStateDict[missingKey] = stateDict[bestMatch]
        usedUnexpectedKeys.add(bestMatch)
        if (verbose):
          print(f"  Mapped: '{bestMatch}' -> '{missingKey}' (Similarity: {bestRatio:.2f})")
      else:
        # Retain original key if no safe match is found (will remain missing).
        if (missingKey in stateDict):
          newStateDict[missingKey] = stateDict[missingKey]

    # Add remaining keys that were not missing (already matched or not involved in mismatch).
    for key, value in stateDict.items():
      if (key not in usedUnexpectedKeys and key not in newStateDict):
        # Check if this key is actually expected by the model to avoid adding unexpected keys back.
        if (key in modelState):
          newStateDict[key] = value

    # Attempt to load the reconciled state dictionary.
    try:
      loadReport = model.load_state_dict(newStateDict, strict=strict)
      if (len(loadReport.missing_keys) == 0 and len(loadReport.unexpected_keys) == 0):
        if (verbose):
          print("Key mismatch resolved successfully via fuzzy matching.")
      else:
        if (verbose):
          print(f"Warning: Some keys remain unresolved after fuzzy matching.")
          print(f"  Remaining Missing: {len(loadReport.missing_keys)}")
    except RuntimeError as e:
      if (verbose):
        print(f"Error loading reconciled state dict: {e}")

  # If optimizer is provided, load its state and update learning rate.
  if (optimizer is not None):
    if ("optimizer" in checkpoint):
      optimizer.load_state_dict(checkpoint["optimizer"])

      # Update learning rate for all parameter groups in the optimizer.
      for paramGroup in optimizer.param_groups:
        paramGroup["lr"] = lr
    else:
      if (verbose):
        print("Warning: Optimizer state not found in checkpoint.")

  # Print confirmation message with checkpoint file and device.
  if (verbose):
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


class PyTorchCrossAttentionHead(nn.Module):
  r'''
  A PyTorch module implementing a cross-attention head for transformer architectures.
  This module computes cross-attention between input embeddings and optional metadata embeddings.
  '''

  def __init__(self, hiddenSize, attentionHeadSize, dropout, bias=True, metadataDim=None):
    r'''
    Initialize the `CrossAttentionHead` module.

    Parameters:
      hiddenSize (int): The dimensionality of the input embeddings.
      attentionHeadSize (int): The dimensionality of the attention head output.
      dropout (float): Dropout rate to apply to attention probabilities.
      bias (bool): Whether to include bias terms in linear projections (default: True).
      metadataDim (int): Dimensionality of the metadata input. If None, metadata projections are not created.
    '''

    # Call the parent class constructor.
    super().__init__()
    # Store the attention head size.
    self.attentionHeadSize = attentionHeadSize
    # Create linear projection for queries.
    self.query = nn.Linear(hiddenSize, attentionHeadSize, bias=bias)
    # Create linear projection for keys.
    self.key = nn.Linear(hiddenSize, attentionHeadSize, bias=bias)
    # Create linear projection for values.
    self.value = nn.Linear(hiddenSize, attentionHeadSize, bias=bias)
    # Create metadata projections to produce extra key/value.
    self.metadataKey = nn.Linear(metadataDim, attentionHeadSize, bias=bias)
    # Create metadata projection for value token.
    self.metadataValue = nn.Linear(metadataDim, attentionHeadSize, bias=bias)
    # Create dropout module for attention probabilities.
    self.dropout = nn.Dropout(dropout)
    # Set head type description.
    self.headType = "Cross-Attention"

  # Define the forward pass for the CrossAttentionHead.
  def forward(self, x, metadata=None):
    r'''
    Compute cross-attention between input embeddings and metadata.

    Parameters:
      x (torch.Tensor): Input embeddings of shape (batchSize, seqLen, hiddenSize).
      metadata (torch.Tensor): Metadata embeddings of shape (batchSize, metadataDim).

    Returns:
      tuple: A tuple containing:
        - context (torch.Tensor): The output of the cross-attention mechanism, of shape (batchSize, seqLen, attentionHeadSize).
        - probs (torch.Tensor): The attention probabilities, of shape (batchSize, seqLen + 1) where the last token corresponds to metadata.
    '''

    # Project queries from input embeddings.
    query = self.query(x)
    # Project keys from input embeddings.
    key = self.key(x)
    # Project values from input embeddings.
    value = self.value(x)
    # Project metadata to a single token and unsqueeze to sequence dimension.
    metaK = self.metadataKey(metadata).unsqueeze(1)
    # Project metadata to a single value token and unsqueeze to sequence dimension.
    metaV = self.metadataValue(metadata).unsqueeze(1)
    # Concatenate image keys with metadata key along the sequence dimension.
    keyTotal = torch.cat([key, metaK], dim=1)
    # Concatenate image values with metadata value along the sequence dimension.
    valueTotal = torch.cat([value, metaV], dim=1)
    # Compute scaled dot-product attention scores.
    scores = torch.matmul(query, keyTotal.transpose(-1, -2)) / (self.attentionHeadSize ** 0.5)
    # Compute attention probabilities.
    probs = torch.softmax(scores, dim=-1)
    # Apply dropout to attention probabilities.
    probs = self.dropout(probs)
    # Compute context as weighted sum of values.
    context = torch.matmul(probs, valueTotal)
    # Return context and attention probabilities.
    return (context, probs)


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
  Implements Exponential Moving Average (EMA) for model parameters and buffers.

  This class maintains a shadow copy of a model, updating its parameters via an
  exponential moving average of the main model's parameters. It ensures that
  persistent buffers (e.g., BatchNorm running statistics) are synchronized
  correctly to prevent performance degradation during evaluation.

  Parameters:
    model (torch.nn.Module, optional): Model to initialize EMA with. If None,
      EMA will be initialized on the first update.
    decay (float): Decay rate for EMA. Default is 0.9999.
    device (str or torch.device, optional): Device to store EMA weights. If None,
      uses the same device as the model provided during update.
  '''

  def __init__(
    self,
    model: Optional[torch.nn.Module] = None,
    decay: float = 0.9999,
    device: Optional[Union[str, torch.device]] = None
  ):
    self.decay = float(decay)
    self.num_updates: int = 0
    self.device: Optional[torch.device] = torch.device(device) if (device is not None) else None
    self.module: Optional[torch.nn.Module] = None
    self._backup: Optional[Dict[str, torch.Tensor]] = None
    self._buffer_backup: Optional[Dict[str, torch.Tensor]] = None

    if (model is not None):
      self._initialize_module(model)

  def _initialize_module(self, model: torch.nn.Module) -> None:
    r'''
    Internal method to initialize the EMA module as a deep copy of the provided model.
    '''

    self.module = copy.deepcopy(model)
    self.module.eval()
    for p in self.module.parameters():
      p.requires_grad = False

    if self.device is not None:
      self.module.to(self.device)

  def update(self, model: torch.nn.Module) -> None:
    r'''
    Update EMA weights and buffers using the current model parameters.

    Parameters are updated using the EMA formula. Buffers are copied directly
    from the current model to ensure statistics (e.g., BatchNorm) remain valid.

    Parameters:
      model (torch.nn.Module): Model with current parameters to update EMA from.
    '''

    if (self.module is None):
      self._initialize_module(model)

    # Ensure EMA module resides on the correct device.
    if (self.device is None):
      self.module.to(next(model.parameters()).device)

    self.num_updates += 1
    decay = self.decay

    with torch.no_grad():
      # Create dictionaries for name-based alignment.
      emaParams = dict(self.module.named_parameters())
      emaBuffers = dict(self.module.named_buffers())

      # Update Parameters.
      for name, modelParam in model.named_parameters():
        if (name in emaParams):
          emaParam = emaParams[name]
          modelData = modelParam.detach().to(emaParam.device)
          emaParam.data.mul_(decay).add_(modelData, alpha=(1.0 - decay))

      # Update Buffers (Direct Copy).
      for name, modelBuffer in model.named_buffers():
        if (name in emaBuffers):
          emaBuffer = emaBuffers[name]
          emaBuffer.copy_(modelBuffer.detach().to(emaBuffer.device))

  def to(self, device: Union[str, torch.device]) -> None:
    r'''
    Move EMA weights to the specified device.

    Parameters:
      device (str or torch.device): Device to move EMA weights to.
    '''

    self.device = torch.device(device)
    if (self.module is not None):
      self.module.to(self.device)

  def state_dict(self) -> Dict[str, Any]:
    r'''
    Get the state dictionary for EMA.

    Returns:
      dict: State dictionary containing decay, num_updates, and module_state_dict.
    '''

    sd = {
      "decay"            : self.decay,
      "num_updates"      : self.num_updates,
      "module_state_dict": self.module.state_dict() if (self.module is not None) else None,
    }
    return sd

  def load_state_dict(self, sd: Dict[str, Any]) -> None:
    r'''
    Load the state dictionary for EMA.

    Parameters:
      sd (dict): State dictionary to load.
    '''

    self.decay = float(sd.get("decay", self.decay))
    self.num_updates = int(sd.get("num_updates", self.num_updates))
    mstate = sd.get("module_state_dict", None)

    if (mstate is not None):
      if (self.module is None):
        raise RuntimeError(
          "EMA module is not initialized. "
          "Initialize `ExponentialMovingAverage` with a model before loading module state."
        )
      self.module.load_state_dict(mstate)

  def apply_shadow(self, model: torch.nn.Module) -> None:
    r'''
    Apply EMA weights and buffers to the given model, backing up original weights.

    This method allows for evaluation using EMA weights without permanently
    altering the training model.

    Parameters:
      model (torch.nn.Module): Model to apply EMA weights to.
    '''

    if (self.module is None):
      raise RuntimeError("No EMA weights available. Call update() at least once or initialize EMA with a model.")

    self._backup = {}
    self._buffer_backup = {}

    # Backup and Apply Parameters.
    for name, param in model.named_parameters():
      self._backup[name] = param.detach().clone()
      if (name in dict(self.module.named_parameters())):
        emaP = dict(self.module.named_parameters())[name]
        param.data.copy_(emaP.data.to(param.device))

    # Backup and Apply Buffers.
    for name, buffer in model.named_buffers():
      self._buffer_backup[name] = buffer.detach().clone()
      if (name in dict(self.module.named_buffers())):
        emaB = dict(self.module.named_buffers())[name]
        buffer.copy_(emaB.to(buffer.device))

  def restore(self, model: torch.nn.Module) -> None:
    r'''
    Restore original model weights and buffers from backup.

    Parameters:
      model (torch.nn.Module): Model to restore original weights to.
    '''

    if (self._backup is None and self._buffer_backup is None):
      return

    # Restore Parameters.
    if (self._backup is not None):
      for name, orig in self._backup.items():
        param = dict(model.named_parameters()).get(name)
        if (param is not None):
          param.data.copy_(orig.to(param.device))
      self._backup = None

    # Restore Buffers
    if (self._buffer_backup is not None):
      for name, orig in self._buffer_backup.items():
        buffer = dict(model.named_buffers()).get(name)
        if (buffer is not None):
          buffer.copy_(orig.to(buffer.device))
      self._buffer_backup = None

  def update_from(self, model: torch.nn.Module) -> None:
    r'''
    Alias for update() method.

    Parameters:
      model (torch.nn.Module): Model with current parameters to update EMA from.
    '''

    return self.update(model)


def ApplyDynamicQuantizationTorch(modelPath: str, outputPath: str, exampleInput=None, inputShape=None):
  r'''
  Apply dynamic quantization to a PyTorch model checkpoint and save a portable TorchScript file.

  This helper attempts to load a checkpoint from ``modelPath``. It supports the following
  common checkpoint formats:

  - a dict containing a key like ``"model"`` that is a ``torch.nn.Module`` or contains a ``state_dict``
  - a plain ``torch.nn.Module`` object saved directly

  If the checkpoint contains only a ``state_dict`` (mapping) the architecture is not known
  and quantization cannot be applied by this helper; in that case the function returns ``None``.

  When successful the function will:

  - apply ``torch.quantization.quantize_dynamic`` to quantize Linear and Embedding modules to ``qint8``
  - convert the quantized model to TorchScript (prefer ``torch.jit.script``; fallback to tracing)
  - save the scripted module to ``outputPath``

  Parameters:
    modelPath (str): Path to the saved checkpoint or model file.
    outputPath (str): Path where the scripted quantized model will be saved.
    exampleInput (torch.Tensor, optional): Example input tensor to use for tracing when scripting fails.
    inputShape (tuple, optional): Alternative to ``exampleInput``; will create a random tensor with this shape for tracing.

  Returns:
    str | None: Returns the ``outputPath`` on success, ``None`` on failure.
  '''

  # TODO: PyTorchHelper.ApplyDynamicQuantizationTorch TO BE CHECKED.

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

  from HMB.PyTorchTrainingPipeline import GenericImageryEvaluatePredictPlotSubset

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
  ) = GenericImageryEvaluatePredictPlotSubset(
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
      ) = GenericImageryEvaluatePredictPlotSubset(
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


def MeasureLatency(model, device, inputsList, runs=100, warmup=10, useCudaEvents=True):
  r'''
  Measures the latency of a PyTorch model's forward pass on a specified device, with options for warmup and precise timing.

  Parameters:
    model (torch.nn.Module): The PyTorch model to evaluate.
    device (torch.device): The device on which to run the model (e.g., CPU or CUDA).
    inputsList (list): A list of input argument tuples to pass to the model's forward method. Each tuple should contain the positional arguments for one forward pass.
    runs (int): The number of timed runs to execute for latency measurement (default: 100).
    warmup (int): The number of warmup runs to perform before timing (default: 10).
    useCudaEvents (bool): Whether to use CUDA events for timing when running on a CUDA device (default: True).

  Returns:
    tuple: A tuple containing the mean latency (in milliseconds), standard deviation of latency, and a dictionary of additional statistics (min, max, percentiles).
  '''

  # Ensure the model is set to evaluation mode.
  model.eval()
  # Move the model to the specified device.
  model.to(device)
  # Prepare a list to store the timing results.
  timings = []
  # Move all tensor arguments to the correct device.
  inputs = tuple(el.to(device) if isinstance(el, torch.Tensor) else el for el in inputsList)
  # Disable gradient computation for the measurement block.
  with torch.no_grad():
    # Execute warmup passes to stabilize the hardware performance.
    for _ in range(warmup):
      # Run the model forward pass with the prepared inputs.
      _ = model(*inputs)
      # Synchronize CUDA streams if the device is a GPU.
      if (device.type == "cuda"):
        # Force synchronization to ensure warmup completion.
        torch.cuda.synchronize()
    # Check if CUDA events should be used for precise timing.
    if (device.type == "cuda" and useCudaEvents):
      # Create a start event for recording the beginning of execution.
      startEvent = torch.cuda.Event(enable_timing=True)
      # Create an end event for recording the completion of execution.
      endEvent = torch.cuda.Event(enable_timing=True)
      # Iterate through the specified number of timed runs.
      for _ in range(runs):
        # Record the start timestamp on the GPU stream.
        startEvent.record()
        # Execute the model forward pass.
        _ = model(*inputs)
        # Record the end timestamp on the GPU stream.
        endEvent.record()
        # Synchronize to ensure the events are fully processed.
        torch.cuda.synchronize()
        # Calculate the elapsed time in milliseconds and append to the list.
        timings.append(startEvent.elapsed_time(endEvent))
    # Otherwise use the CPU high-resolution timer for measurement.
    else:
      # Iterate through the specified number of timed runs.
      for _ in range(runs):
        # Capture the start time using the performance counter.
        start = time.perf_counter()
        # Execute the model forward pass.
        _ = model(*inputs)
        # Synchronize CUDA streams if the device is a GPU.
        if (device.type == "cuda"):
          # Force synchronization to measure full kernel execution time.
          torch.cuda.synchronize()
        # Capture the end time using the performance counter.
        end = time.perf_counter()
        # Calculate duration in milliseconds and add to the timings list.
        timings.append((end - start) * 1000.0)
  # Convert the list of timings into a NumPy array for statistical analysis.
  timingsArray = np.array(timings)
  # Calculate the mean latency value.
  meanLatency = float(timingsArray.mean())
  # Calculate the standard deviation of the latency values.
  stdLatency = float(timingsArray.std())
  # Initialize a dictionary to hold additional statistical metrics.
  additionalStats = {
    "Min": float(timingsArray.min()),
    "Max": float(timingsArray.max()),
    "P50": float(np.percentile(timingsArray, 50)),
    "P95": float(np.percentile(timingsArray, 95)),
    "P99": float(np.percentile(timingsArray, 99)),
  }
  # Return the calculated mean, standard deviation, and the statistics dictionary.
  return (meanLatency, stdLatency, additionalStats)


def ComputeProfileFLOPs(model, inputsList, doCPU=True, formatGigas=True):
  r'''
  Attempts to profile the FLOPs of a PyTorch model using the thop library, with an option to run on CPU for compatibility.

  Parameters:
    model (torch.nn.Module): The PyTorch model to profile.
    inputsList (list): A list of input argument tuples to pass to the model's forward method. Each tuple should
      contain the positional arguments for one forward pass.
    doCPU (bool): Whether to move the model and inputs to CPU for profiling (default: True). This can improve
      compatibility with thop if GPU profiling causes issues, but may be slower for large models.
    formatGigas (bool): If True, format the result in Gigas with one decimal place. If False, return the raw count of FLOPs.

  Returns:
    float or None: The calculated FLOPs for the model if profiling is successful, or None if thop is not available or if profiling fails for any reason.
  '''

  # Attempt to import the profile function from the thop library.
  try:
    # Import the profile module locally to avoid errors if missing.
    from thop import profile
  # Catch any exception during the import process.
  except Exception:
    # Return None if thop is unavailable.
    return None

  # Attempt to execute the profiling logic.
  try:
    # Check if CPU profiling is requested.
    if (doCPU):
      # Move the model to the CPU device.
      model = model.cpu()
      # Move the metadata tensor to the CPU device.
      inputs = tuple(el.cpu() if isinstance(el, torch.Tensor) else el for el in inputsList)
    else:
      inputs = tuple(el for el in inputsList)
    # Run the profiling call and retrieve flops and params.
    flops, params = profile(model, inputs=inputs, verbose=False)
    if (formatGigas):
      flops = flops / float(1e9)
    # Return the calculated flops value.
    return flops
  # Catch any exception during the profiling process.
  except Exception:
    # Return None if profiling fails for any reason.
    return None


def EnableMixedPrecision(model: nn.Module) -> torch.amp.GradScaler:
  r'''
  Enable automatic mixed precision training for GPU acceleration.
  Wraps model with AMP support and returns a gradient scaler
  for stable training with float16/float32 mixed precision.

  Parameters:
    model (nn.Module): PyTorch model to wrap with AMP support.

  Returns:
    torch.amp.GradScaler: Scaler for gradient scaling during backward pass.
  '''

  # Verify CUDA availability for mixed precision training.
  if (not torch.cuda.is_available()):
    import warnings
    warnings.warn("EnableMixedPrecision: CUDA not available, returning dummy scaler.")
    return torch.amp.GradScaler("cuda", enabled=False)  # ✅ Updated to new API
  # Return gradient scaler configured for mixed precision.
  return torch.amp.GradScaler("cuda")  # ✅ Updated to new API


class EarlyStopping:
  r'''
  Early stopping callback to halt training when validation metric plateaus.

  Monitors a specified metric and stops training if no improvement
  is observed for a defined number of epochs.

  Parameters:
    patience (int): Number of epochs to wait before stopping after no improvement.
    minDelta (float): Minimum change in monitored metric to qualify as improvement.
    mode (str): Optimization direction: "min" for loss, "max" for accuracy, "auto" to infer.
    verbose (bool): Print status messages when stopping criteria are met.
  '''

  def __init__(self, patience: int = 10, minDelta: float = 0.0, mode: str = "auto", verbose: bool = True):
    # Validate callback parameters.
    if (patience < 1):
      raise ValueError(f"`patience` must be at least 1, got {patience}")
    if (minDelta < 0):
      raise ValueError(f"`minDelta` must be non-negative, got {minDelta}")
    if (mode not in ["min", "max", "auto"]):
      raise ValueError(f"`mode` must be 'min', 'max', or 'auto', got {mode}")
    # Store early stopping configuration.
    self.patience = patience
    self.minDelta = minDelta
    self.mode = mode
    self.verbose = verbose
    # Initialize internal state.
    self.counter = 0
    self.bestScore = None
    self.earlyStop = False

  def __call__(self, currentScore: float) -> bool:
    # Evaluate current metric against stopping criteria.
    # Determine optimization direction if in auto mode.
    if (self.mode == "auto"):
      self.mode = "min" if ("loss" in str(currentScore).lower()) else "max"
    # Check if current score represents improvement.
    if (self.bestScore is None):
      self.bestScore = currentScore
      return False
    # Compute score change based on optimization direction.
    if (self.mode == "min"):
      scoreChange = self.bestScore - currentScore
    else:
      scoreChange = currentScore - self.bestScore
    # Update best score and reset counter if improvement observed.
    if (scoreChange > self.minDelta):
      self.bestScore = currentScore
      self.counter = 0
      if (self.verbose):
        print(f"EarlyStopping: Metric improved to {currentScore:.4f}, resetting counter.")
      return False
    # Increment counter and check patience threshold.
    self.counter += 1
    if (self.counter >= self.patience):
      self.earlyStop = True
      if (self.verbose):
        print(f"EarlyStopping: No improvement for {self.patience} epochs, stopping training.")
      return True
    # Report current counter status if verbose.
    if (self.verbose):
      print(f"EarlyStopping: Counter {self.counter}/{self.patience}, best score: {self.bestScore:.4f}")
    return False


class CheckpointSaver:
  r'''
  Model checkpointing utility for saving best/latest weights.

  Saves model state dictionaries to disk based on monitored metric
  and optional saving strategy (best-only or periodic).

  Parameters:
    savePath (str): Directory path for checkpoint file storage.
    saveBestOnly (bool): Save only when monitored metric improves.
    monitor (str): Metric name to monitor for saving decisions.
    mode (str): Optimization direction: "min" for loss, "max" for accuracy.
    verbose (bool): Print messages when checkpoints are saved.
  '''

  def __init__(
    self, savePath: str, saveBestOnly: bool = True,
    monitor: str = "val_loss", mode: str = "min", verbose: bool = True
  ):
    # Validate checkpoint parameters.
    if (not savePath):
      raise ValueError("savePath must be a non-empty string")
    if (mode not in ["min", "max"]):
      raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    # Store checkpoint configuration.
    self.savePath = savePath
    self.saveBestOnly = saveBestOnly
    self.monitor = monitor
    self.mode = mode
    self.verbose = verbose
    # Initialize best metric tracking.
    self.bestMetric = None

  def __call__(self, model: nn.Module, currentMetric: float, epoch: int) -> str:
    # Evaluate whether to save checkpoint based on metric.
    # Determine if current metric represents improvement.
    shouldSave = False
    if (self.bestMetric is None):
      shouldSave = True
    elif (self.mode == "min" and currentMetric < self.bestMetric - 1e-6):
      shouldSave = True
    elif (self.mode == "max" and currentMetric > self.bestMetric + 1e-6):
      shouldSave = True
    # Save checkpoint if criteria met or not best-only mode.
    if (shouldSave or not self.saveBestOnly):
      import os
      # Ensure save directory exists.
      os.makedirs(self.savePath, exist_ok=True)
      # Construct checkpoint filename with epoch and metric.
      metricStr = f"{currentMetric:.4f}".replace(".", "_")
      filename = f"CheckpointEpoch{epoch}_Metric_{metricStr}.pt"
      filepath = os.path.join(self.savePath, filename)
      # Save model state dictionary with metadata.
      checkpoint = {
        "epoch"           : epoch,
        "metric"          : currentMetric,
        "model_state_dict": model.state_dict(),
      }
      torch.save(checkpoint, filepath)
      # Update best metric tracking if improvement.
      if (shouldSave):
        self.bestMetric = currentMetric
        if (self.verbose):
          print(f"CheckpointSaver: Saved best model to {filepath}")
      elif (self.verbose):
        print(f"CheckpointSaver: Saved checkpoint to {filepath}")
      return filepath
    # Return None if no checkpoint was saved.
    return None


def PreparePredTensorToNumpy(
  predTensor: torch.Tensor,
  doScale2Image: bool = False,
) -> np.ndarray:
  r'''
  Utility to convert model output tensor after the sigmoid/softmax activation to a numpy array of class indices.
  It can be used also with the original mask tensor if it is already in the correct format,
  as it handles squeezing and type conversion.

  Short summary:
    Takes the raw output tensor from the model (after activation) and processes it to produce
    a 2D numpy array of class indices. This involves squeezing unnecessary dimensions,
    converting boolean masks to integers if needed, and ensuring the final output is in the
    correct format for evaluation or visualization.

  Parameters:
    predTensor (torch.Tensor): The raw output tensor from the model after activation, expected to be of shape [B, C, H, W] or [B, 1, H, W].
    doScale2Image (bool): If True, applies a threshold to convert probabilities to binary mask. Default False.

  Returns:
    numpy.ndarray: Numpy array of shape [B, H, W] containing class indices.
  '''

  # Convert the prediction tensor to a numpy array.
  predNp = predTensor.cpu().numpy()

  # If prediction has a leading channel dimension of size 1, squeeze it away.
  if (predNp.ndim == 3 and predNp.shape[0] == 1):
    # Squeeze away the channel dimension when it is singleton.
    predNp = np.squeeze(predNp, axis=0)

  # Ensure prediction mask is 2D (H,W). If it's boolean/0-1, keep as ints.
  if (predNp.dtype == np.bool_):
    # Convert boolean mask to uint8 for compatibility.
    predMask = predNp.astype(np.uint8)
  else:
    # Convert prediction mask to integer labels.
    predMask = predNp.astype(np.int64)

  if (doScale2Image):
    # If the prediction is a probability map, apply a threshold to convert to binary mask.
    predMask = (predMask >= 0.5).astype(np.uint8)
    predMask *= 255  # Scale binary mask to 0 and 255 for visualization.
    predMask = predMask.astype(np.uint8)

  return predMask
