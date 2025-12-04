import os, timm, torch, tqdm, copy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import HMB.PerformanceMetrics as pm


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
def LoadModel(model, filename="model.pth", device="cuda"):
  r'''
  Load the model state from a file and move it to the specified device.

  Parameters:
    model (torch.nn.Module): The model to load the state into.
    filename (str): The name of the file to load the model from.
    device (str): The device to load the model onto (e.g., "cpu" or "cuda").
  '''

  # Check if the model file exists before loading.
  if (not os.path.exists(filename)):
    print(f"Model file not found: {filename}")
    return

  # Load the state dictionary from file and map to the specified device.
  model.load_state_dict(torch.load(filename, map_location=device))
  # Move the model to the specified device.
  model.to(device)

  # Print confirmation message with filename and device.
  print(f"Model loaded from {filename} and moved to {device}.")


def LoadPyTorchDict(filename="model.pth", device="cuda"):
  r'''
  Load a PyTorch state dictionary from a file and map it to the specified device.

  Parameters:
    filename (str): The name of the file to load the state dictionary from.
    device (str): The device to map the state dictionary onto (e.g., "cpu" or "cuda").

  Returns:
    dict: The loaded state dictionary.
  '''

  # Check if the state dictionary file exists before loading.
  if (not os.path.exists(filename)):
    print(f"State dictionary file not found: {filename}")
    return None

  # Load the state dictionary from file and map to the specified device.
  # Set weights_only=False to allow loading full objects (required for PyTorch >=2.6).
  stateDict = torch.load(filename, map_location=device, weights_only=False)

  # Print confirmation message with filename and device.
  print(f"State dictionary loaded from {filename} and mapped to {device}.")

  return stateDict


def SaveCheckpoint(model, optimizer, filename="chk.pth.tar"):
  r'''
  Save model and optimizer state to a checkpoint file.
  Useful for resuming training or inference later.
  This function saves the model's state dictionary and the optimizer's state dictionary
  to a specified file. You can load it later using LoadCheckpoint() function from this module.

  Parameters:
    model (torch.nn.Module): The model to save.
    optimizer (torch.optim.Optimizer): The optimizer to save.
    filename (str): The name of the file to save the checkpoint to.
  '''

  # Create a dictionary containing model and optimizer state.
  checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer" : optimizer.state_dict(),
  }
  # Save the checkpoint dictionary to the specified file.
  torch.save(checkpoint, filename)

  # Print confirmation message with filename.
  print(f"Checkpoint saved to {filename}. You can load it later using LoadCheckpoint().")


def LoadCheckpoint(checkpointFile, model, optimizer, lr, device):
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
  '''

  # Check if the checkpoint file exists before loading.
  if (not os.path.exists(checkpointFile)):
    print(f"Checkpoint file not found: {checkpointFile}")
    return

  # Load the checkpoint dictionary from file and map to the specified device.
  checkpoint = torch.load(checkpointFile, map_location=device)
  # Load the model state from the checkpoint.
  model.load_state_dict(checkpoint["state_dict"])

  # If optimizer is provided, load its state and update learning rate.
  if (optimizer is not None):
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update learning rate for all parameter groups in the optimizer.
    for paramGroup in optimizer.param_groups:
      paramGroup["lr"] = lr

  # Print confirmation message with checkpoint file and device.
  print(f"Checkpoint loaded from {checkpointFile} and model moved to {device}.")


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
    allowedExtensions=(".png", ".jpg", ".jpeg")
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

  Parameters:
    inputs (torch.Tensor): Input data of shape (batch_size, ...).
    targets (torch.Tensor): Target labels of shape (batch_size,) or (batch_size, num_classes).
    alpha (float): MixUp alpha parameter for Beta distribution. Default is 0.4.
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
    maxGradNorm (float, optional): Maximum gradient norm for clipping. Defaults to None.
    useAmp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
    useMixupFn (bool, optional): Whether to use MixUp data augmentation. Defaults to False.
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

    try:
      if (isinstance(scheduler, ReduceLROnPlateau)):
        scheduler.step(avgValEpochLoss)
      else:
        scheduler.step()
    except Exception:
      try:
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
  totalEpochSamples = 0

  # Zero the gradients of the optimizer.
  optimizer.zero_grad()

  # Determine device type for autocast.
  deviceType = "cuda" if ((hasattr(device, "type") and device.type == "cuda") or ("cuda" in str(device))) else "cpu"

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
      data, labels = MixupFn(data, labels, alpha=0.4, numClasses=noOfClasses)

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

    # Get the batch size.
    batchSize = data.size(0)
    totalEpochSamples += batchSize

    # Get the predicted class indices.
    outputIdx = outputs.argmax(dim=1)

    # If loss is a tensor, convert it to a scalar value.
    lossScalar = loss.item() if (isinstance(loss, torch.Tensor)) else loss
    # Accumulate the total loss for the epoch.
    totalEpochLoss += lossScalar * batchSize

    # Compute the confusion matrix and accuracy.
    cm = confusion_matrix(
      labels.cpu(),  # True labels.
      outputIdx.cpu(),  # Predicted labels.
      labels=list(range(noOfClasses)),  # List of class labels.
    )
    # Calculate performance metrics from the confusion matrix.
    metrics = pm.CalculatePerformanceMetrics(cm)
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
  avgTrainLoss = totalEpochLoss / max(1, totalEpochSamples)
  avgTrainAccuracy = totalEpochAccuracy / max(1, totalEpochSamples)

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

  with torch.no_grad():
    # Iterate over the evaluation data loader with a progress bar.
    for batchIdx, batch in tqdm.tqdm(
      enumerate(dataLoader),  # Enumerate over batches.
      total=len(dataLoader),  # Total number of batches.
      desc="Evaluating",  # Description for the progress bar.
    ):
      # Get data and labels from the batch and move them to the specified device.
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)

      # Forward pass through the model to get outputs.
      outputs = model(data)

      # Get the predicted class indices.
      outputIdx = outputs.argmax(dim=1)

      # Compute the loss using the specified criterion.
      loss = criterion(outputs, labels)
      # If loss is a tensor, convert it to a scalar value.
      loss = loss.item() if isinstance(loss, torch.Tensor) else loss
      # Accumulate the total loss for the validation epoch.
      totalLoss += loss

      # Compute the confusion matrix and accuracy.
      cm = confusion_matrix(
        labels.cpu(),  # True labels.
        outputIdx.cpu(),  # Predicted labels.
        labels=list(range(noOfClasses)),  # List of class labels.
      )
      # Calculate performance metrics from the confusion matrix.
      metrics = pm.CalculatePerformanceMetrics(cm)
      accuracy = metrics["Weighted Accuracy"]

      # Accumulate the total accuracy for the validation epoch.
      totalAccuracy += accuracy

  # Calculate average loss and accuracy for the validation epoch.
  avgValLoss = totalLoss / len(dataLoader)
  avgValAccuracy = totalAccuracy / len(dataLoader)

  return avgValLoss, avgValAccuracy


def InferenceWithPlots(
  dataDir,  # Directory containing dataset.
  model,  # Model architecture.
  modelCheckpointName=None,  # Path to model checkpoint.
  transform=None,  # Image transform to apply.
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
    # Prepare image transform.
    transform = transforms.Compose([
      transforms.Resize((imageSize, imageSize)),  # Resize images.
      transforms.ToTensor(),  # Convert to tensor.
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize.
    ])

  # Create dataset and dataloader.
  dataset = CustomDataset(dataDir, transform=transform)
  dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

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
      stateDict = LoadPyTorchDict(modelPath, device=device)
      if (stateDict and isinstance(stateDict, dict) and "model_state_dict" in stateDict):
        model.load_state_dict(stateDict["model_state_dict"])
      else:
        model.load_state_dict(stateDict)

    cmFilePath = os.path.join(expDirPath, "CM.pdf")
    rocFilePath = os.path.join(expDirPath, "ROC.pdf")
    rocpFilePath = os.path.join(expDirPath, "ROCP.pdf")
    prcFilePath = os.path.join(expDirPath, "PRC.pdf")
    prcpFilePath = os.path.join(expDirPath, "PRCP.pdf")

    # Move the model to the selected device (CPU or GPU).
    model = model.to(device)

    # Set the model to evaluation mode.
    model.eval()

    # Lists to store true labels, predicted labels, and probabilities.
    yTrue, yPred, yProbs = [], [], []

    # Disable gradient calculation for evaluation.
    with torch.no_grad():
      # Iterate over the dataloader with a progress bar.
      for imgs, labels in tqdm.tqdm(dataloader, desc="Evaluating", unit="Image"):
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
    metrics = pm.CalculatePerformanceMetrics(cm, addWeightedAverage=True)
    metrics["Path"] = expDirPath
    metrics["File"] = os.path.basename(expDirPath)
    overallHistory.append(metrics)

    # Plot and save the confusion matrix.
    pm.PlotConfusionMatrix(
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
    pm.PlotROCAUCCurve(
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
    pm.PlotROCAUCCurve(
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
    pm.PlotPRCCurve(
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
    pm.PlotPRCCurve(
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
