import os, tqdm, torch, torchvision, cv2, time, copy, math, shutil, csv, json
import numpy as np
import pandas as pd
from typing import *
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from HMB.Initializations import IMAGE_SUFFIXES
from HMB.PerformanceMetrics import (
  CalculatePerformanceMetrics, PlotConfusionMatrix,
  PlotROCAUCCurve, PlotPRCCurve, ComputeECE
)
from HMB.DatasetsHelper import CustomDataset
from HMB.PyTorchHelper import (
  SavePyTorchDict, LoadPyTorchDict, MixupFn, MixupCriterion, LoadModel,
  EnableMixedPrecision, EarlyStopping, CheckpointSaver, ExponentialMovingAverage,
  PreparePredTensorToNumpy
)
from HMB.PerformanceMetrics import HistoryPlotter
from HMB import ImageSegmentationMetrics as ISM
from HMB.Utils import DumpJsonFile


def TrainEvaluateClassificationModel(
  model,  # Model to train and evaluate.
  criterion,  # Loss function.
  device,  # Device to run training and evaluation on (CPU or GPU).
  bestModelStoragePath,  # Path to save the best model.
  noOfClasses,  # Number of classes in the classification task.
  numEpochs,  # Total number of epochs for training.
  optimizer,  # Optimizer for updating model parameters.
  scaler=None,  # Gradient scaler for mixed precision training. If None, created via EnableMixedPrecision.
  scheduler=None,  # Learning rate scheduler.
  trainLoader=None,  # DataLoader for training data.
  valLoader=None,  # DataLoader for validation data.
  resumeFromCheckpoint=False,  # Whether to resume training from a checkpoint.
  finalModelStoragePath=None,  # Path to save the final model after training.
  judgeBy="both",  # Criterion to judge the best model ("val_loss", "val_accuracy", or "both").
  earlyStopping=None,  # EarlyStopping callback instance. If None, uses earlyStoppingPatience fallback.
  earlyStoppingPatience=None,  # Patience for early stopping (deprecated: use earlyStopping callback).
  checkpointSaver=None,  # CheckpointSaver callback instance for saving best/latest checkpoints.
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
    scheduler (torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
    trainLoader (torch.utils.data.DataLoader): DataLoader for training data.
    valLoader (torch.utils.data.DataLoader): DataLoader for validation data.
    resumeFromCheckpoint (bool, optional): Flag to indicate if training should resume from a checkpoint. Defaults to False.
    finalModelStoragePath (str, optional): Path to save the final model after training. Defaults to None.
    judgeBy (str, optional): Criterion to judge the best model ("val_loss", "val_accuracy", or "both"). Defaults to "both".
    earlyStopping (object, optional): EarlyStopping callback instance. If None, uses earlyStoppingPatience fallback. Defaults to None.
    earlyStoppingPatience (int, optional): Patience for early stopping. Defaults to None.
    checkpointSaver (object, optional): CheckpointSaver callback instance for saving best/latest checkpoints. Defaults to None.
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

  # Validate required arguments.
  if (trainLoader is None):
    raise ValueError("`trainLoader` is required for training.")
  if (valLoader is None):
    raise ValueError("`valLoader` is required for validation.")
  if (optimizer is None):
    raise ValueError("`optimizer` is required for training.")
  if (criterion is None):
    raise ValueError("`criterion` is required for computing loss.")

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

  # Create gradient scaler for mixed precision if not provided.
  if (scaler is None):
    if (useAmp):
      scaler = EnableMixedPrecision(model)
    else:
      # Create a dummy scaler with AMP disabled to avoid None checks in TrainOneEpoch.
      scaler = torch.cuda.amp.GradScaler(enabled=False)

  # Initialize EarlyStopping callback if not provided but patience is specified.
  if (earlyStopping is None and earlyStoppingPatience is not None):
    # Infer mode based on judgeBy parameter.
    if (judgeBy == "val_loss"):
      mode = "min"
    elif (judgeBy == "val_accuracy"):
      mode = "max"
    else:  # judgeBy == "both"
      # Default to monitoring loss for early stopping when both criteria are used.
      mode = "min"
    earlyStopping = EarlyStopping(
      patience=earlyStoppingPatience,
      mode=mode,
      verbose=verbose
    )

  # Initialize CheckpointSaver callback if not provided.
  if (checkpointSaver is None and bestModelStoragePath):
    checkpointSaver = CheckpointSaver(
      savePath=os.path.dirname(bestModelStoragePath),
      saveBestOnly=True,
      monitor="val_loss" if (judgeBy == "val_loss") else "val_accuracy",
      mode="min" if (judgeBy == "val_loss") else "max",
      verbose=verbose
    )

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
      if (ema is not None):
        ema.load_state_dict(stateDict["ema_state_dict"])
      optimizer.load_state_dict(stateDict["optimizer_state_dict"])
      scaler.load_state_dict(stateDict["scaler_state_dict"])
      # Start from the next epoch after the one saved in the checkpoint.
      startEpoch = stateDict.get("epoch", 0) + 1
      bestValLoss = stateDict.get("best_val_loss", float("inf"))
      bestValAccuracy = stateDict.get("best_val_accuracy", 0.0)

      if (verbose):
        print(
          f"Loaded checkpoint from {bestModelStoragePath} with epoch {stateDict.get('epoch', 'N/A')}, "
          f"best val loss {bestValLoss:.4f}, and best val accuracy {bestValAccuracy:.4f}."
        )
        print("Training will resume from the next epoch.")
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

    # Use CheckpointSaver callback if available, otherwise fall back to inline saving.
    if (checkpointSaver is not None):
      # Determine which metric to monitor based on judgeBy parameter.
      monitorMetric = avgValEpochLoss if (judgeBy == "val_loss") else avgValEpochAccuracy
      checkpointSaver(model=model, currentMetric=monitorMetric, epoch=epoch + 1)
    elif (conditionToSave):
      # SaveModel(model, bestModelStoragePath)
      SavePyTorchDict({
        "model_state_dict"     : model.state_dict(),
        "ema_state_dict"       : ema.state_dict() if (useEma and ema is not None) else None,
        "ema_module_state_dict": ema.module.state_dict() if (useEma and ema is not None) else None,
        "optimizer_state_dict" : optimizer.state_dict() if (optimizer is not None) else None,
        "epoch"                : epoch + 1,
        "scaler_state_dict"    : scaler.state_dict() if (scaler is not None) else None,
        "best_val_loss"        : bestValLoss,
        "best_val_accuracy"    : bestValAccuracy,
      }, filename=bestModelStoragePath)
      if (verbose):
        print(
          f"Saved new best model with val loss: {bestValLoss:.4f} "
          f"and val accuracy: {bestValAccuracy:.4f} "
          f"at epoch {epoch + 1} to {bestModelStoragePath}"
        )

    # Use EarlyStopping callback if available, otherwise fall back to inline logic.
    if (earlyStopping is not None):
      # Determine which metric to monitor based on judgeBy parameter.
      monitorMetric = avgValEpochLoss if (judgeBy == "val_loss") else avgValEpochAccuracy
      if (earlyStopping(monitorMetric)):
        if (verbose):
          print(f"Training stopped early at epoch {epoch + 1}.")
        break
    else:
      # Fallback to original inline early stopping logic.
      if (not conditionToSave and earlyStoppingPatience is not None):
        currentPatience += 1
        if (currentPatience >= earlyStoppingPatience):
          if (verbose):
            print(
              f"Early stopping triggered after {earlyStoppingPatience} epochs "
              f"without improvement."
            )
          break
      else:
        currentPatience = 0  # Reset patience counter on improvement.

    # For `ReduceLROnPlateau` we need to step with the validation loss after evaluation.
    if (scheduler is not None):
      try:
        if (isinstance(scheduler, (ReduceLROnPlateau,))):
          scheduler.step(avgValEpochLoss)
        else:
          scheduler.step()
      except Exception as e:
        if (verbose):
          print(f"Warning: Failed to step the scheduler. Error: {e}. Continuing without stepping.")

    if (saveEvery is not None and (epoch + 1) % saveEvery == 0):
      epochPath = os.path.join(
        os.path.dirname(bestModelStoragePath),
        f"ModelEpoch.epoch{epoch + 1}.pth"
      )
      SavePyTorchDict({
        "model_state_dict"     : model.state_dict(),
        "ema_state_dict"       : ema.state_dict() if (useEma and ema is not None) else None,
        "ema_module_state_dict": ema.module.state_dict() if (useEma and ema is not None) else None,
        "optimizer_state_dict" : optimizer.state_dict(),
        "epoch"                : epoch + 1,
        "scaler_state_dict"    : scaler.state_dict(),
        "best_val_loss"        : bestValLoss,
        "best_val_accuracy"    : bestValAccuracy,
      }, filename=epochPath)
      if (verbose):
        print(f"Saved model at epoch {epoch + 1} to {epochPath}")

  # Save the final model after training if a path is provided.
  if (finalModelStoragePath):
    # Use CheckpointSaver if available for consistent metadata handling.
    if (checkpointSaver is not None):
      # Save final checkpoint and capture the returned filepath.
      finalCheckpointPath = checkpointSaver(
        model=model,
        currentMetric=bestValLoss if (judgeBy == "val_loss") else bestValAccuracy,
        epoch=numEpochs
      )
      # Copy the saved checkpoint to the final path if available.
      if (finalCheckpointPath is not None and os.path.exists(finalCheckpointPath)):
        shutil.copy(finalCheckpointPath, finalModelStoragePath)
        if (verbose):
          print(f"Copied final checkpoint from {finalCheckpointPath} to {finalModelStoragePath}")
    else:
      # Fallback to inline saving.
      SavePyTorchDict({
        "model_state_dict"     : model.state_dict(),
        "ema_state_dict"       : ema.state_dict() if (useEma and ema is not None) else None,
        "ema_module_state_dict": ema.module.state_dict() if (useEma and ema is not None) else None,
        "optimizer_state_dict" : optimizer.state_dict(),
        "epoch"                : numEpochs,
        "scaler_state_dict"    : scaler.state_dict(),
        "best_val_loss"        : bestValLoss,
        "best_val_accuracy"    : bestValAccuracy,
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

  loop = tqdm.tqdm(
    enumerate(dataLoader),  # Enumerate over batches.
    total=len(dataLoader),  # Total number of batches.
    desc=f"Epoch {epoch + 1}/{numEpochs}",  # Description for the progress bar.
  )

  # Iterate over the training data loader with a progress bar.
  for batchIdx, batch in loop:
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

    # Add accuracy as a metric to the progress bar description.
    loop.set_description(
      f"Epoch {epoch + 1}/{numEpochs} - Loss: {lossScalar:.4f}, Acc: {accuracy:.4f}"
    )

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
    loop = tqdm.tqdm(
      enumerate(dataLoader),  # Enumerate over batches.
      total=len(dataLoader),  # Total number of batches.
      desc="Evaluating",  # Description for the progress bar.
    )
    # Iterate over the evaluation data loader with a progress bar.
    for batchIdx, batch in loop:
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
      loss = loss.item() if (isinstance(loss, torch.Tensor)) else loss
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

      loop.set_description(
        f"Evaluating - Loss: {loss:.4f}, Acc: {accuracy:.4f}"
      )

  # Calculate average loss and accuracy for the validation epoch.
  avgValLoss = totalLoss / float(max(1, len(dataLoader)))
  avgValAccuracy = totalAccuracy / float(max(1, len(dataLoader)))

  return avgValLoss, avgValAccuracy


def ImageryInferenceWithPlots(
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


def GenericImageryEvaluatePredictPlotSubset(
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
                f"Image path: {imagePath}, "
                f"probs: {probs}"
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
    str(storageFilePath) if (storageFilePath) else None,
    weightedMetrics,
    allPredsIndices,
    allGtsIndices,
    allPredsProbs,
    allPredsConfidences,
    predictionsRecords,
    classNames,
    cm,
  )


def GenericTabularEvaluatePredictPlotSubset(
  dataPath: str,
  model,
  targetColumn: str = "Label",
  featureColumns: Optional[List[str]] = None,
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
  device=None,
  figSize=(8, 8),
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
  Evaluate a trained classification model on a tabular dataset subset,
  collect predictions, compute confusion matrix and performance metrics,
  and optionally save predictions to a CSV file. It also generates and saves
  confusion matrix, ROC AUC, and PRC plots.

  Parameters:
    dataPath (str): Path to the CSV file containing the tabular dataset.
    model (callable): A callable that takes a NumPy array (N, D) of features
      and returns a 1D array of class probabilities of shape (N, numClasses).
    targetColumn (str): Name of the column containing the target labels. Defaults to "Label".
    featureColumns (List[str] | None): List of column names to use as features.
      If None, uses all columns except targetColumn. Defaults to None.
    subset (str | None): Dataset subset to evaluate ("train", "val", "test", "all", or None).
      If the CSV has a "split" column, filters by that value. Defaults to "test".
    prefix (str): Prefix for saved figure filenames. Defaults to "".
    storageDir (str | None): Directory to save predictions CSV and figures.
      If None, uses current directory. Defaults to None.
    heavy (bool): Whether to compute heavy metrics and plot ROC/PRC curves. Defaults to True.
    computeECE (bool): Whether to compute Expected Calibration Error (ECE). Defaults to True.
    exportFailureCases (bool): Whether to export misclassified samples to CSV. Defaults to True.
    eps (float): Small epsilon value for numerical stability in metric calculations. Defaults to 1e-10.
    saveArtifacts (bool): Whether to save figures and artifacts. Defaults to True.
    maxSamples (int | None): Maximum number of samples to evaluate. If None, evaluates all samples. Defaults to None.
    preprocessFn (callable | None): Optional preprocessing function to apply to each feature row
      before prediction. Should accept a 1D NumPy array and return a processed 1D array. Defaults to None.
    dpi (int): DPI for saved figures. Defaults to 720.
    device: Optional device to run model on (e.g., "cpu" or "cuda"). If None, uses CPU. Defaults to None.
    figSize: Tuple[int, int]: Figure size for plots. Defaults to (8, 8).

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

  # Load the tabular dataset from CSV.
  try:
    df = pd.read_csv(dataPath, low_memory=False)
  except Exception as loadErr:
    print(f"Error loading dataset from {dataPath}: {loadErr}")
    return (None, {}, [], [], [], [], [], [], None)

  # Filter by subset if a "split" column exists.
  if ("split" in df.columns and subset not in ("all", None)):
    df = df[df["split"] == subset].copy()

  # Determine feature columns.
  if (featureColumns is None):
    featureColumns = [col for col in df.columns if (col != targetColumn)]

  # Extract features and labels.
  X = df[featureColumns].values
  y = df[targetColumn].values

  # Map class labels to indices if they are not already integers.
  if (not np.issubdtype(y.dtype, np.integer)):
    # Create a mapping from class names to indices.
    classNames = sorted(df[targetColumn].unique().tolist())
    classToIdx = {name: idx for idx, name in enumerate(classNames)}
    # Convert labels to indices.
    yIndices = np.array([classToIdx[label] for label in y], dtype=int)
  else:
    # Labels are already integer indices.
    classNames = [str(i) for i in range(int(np.max(y)) + 1)]
    yIndices = y.astype(int)

  # Determine number of classes.
  numClasses = len(classNames)

  # Apply sampling limit if maxSamples is set.
  if (maxSamples is not None and len(X) > maxSamples):
    # Randomly sample without replacement.
    sampleIdx = np.random.choice(len(X), size=maxSamples, replace=False)
    X = X[sampleIdx]
    yIndices = yIndices[sampleIdx]

  # Initialize containers for collected data.
  allPredsIndices: List[int] = []
  allGtsIndices: List[int] = []
  allPredsProbs: List[List[float]] = []
  allPredsNames: List[str] = []
  allGtsNames: List[str] = []
  allPredsConfidences: List[Optional[float]] = []
  predictionsRecords: List[Dict[str, Any]] = []

  try:
    # Iterate over samples and make predictions.
    loader = tqdm.tqdm(range(len(X)), desc="Evaluating samples", unit="sample")
    for i in loader:
      # Extract feature row.
      features = X[i]

      # Apply preprocessing if provided.
      if (preprocessFn is not None):
        try:
          features = preprocessFn(features)
        except Exception as prepErr:
          print(f"Preprocessing failed for sample {i}: {prepErr}")
          continue

      # Make prediction using the model callable.
      try:
        # Should return 1D array of shape [numClasses].
        try:
          probs = model(features)
        except:
          # Apply to device and convert to NumPy if it's a tensor.
          featuresTensor = torch.from_numpy(features).float()
          # Expected 2D input (batchSize, inputSize), got torch.Size([42]).
          if (featuresTensor.ndim == 1):
            featuresTensor = featuresTensor.unsqueeze(0)
          if (device is not None):
            featuresTensor = featuresTensor.to(device)
          probsTensor = model(featuresTensor)
          probs = probsTensor.detach().cpu().numpy()

        probs = np.asarray(probs, dtype=np.float32)
        if (probs.ndim == 2 and probs.shape[0] == 1):
          probs = probs[0]

        if (probs.ndim != 1):
          raise ValueError(f"Expected 1D probability vector, got shape {probs.shape}")
        if (len(probs) != numClasses):
          raise ValueError(
            f"Number of classes mismatch: expected {numClasses}, got {len(probs)}\n"
            f"Sample index: {i}, "
            f"probs: {probs}"
          )

        predictedClassIndex = int(np.argmax(probs))
        predictedConfidence = float(probs[predictedClassIndex])
        probList = probs.tolist()

      except Exception as predErr:
        print(f"Prediction failed for sample {i}: {predErr}")
        predictedClassIndex = -1
        predictedConfidence = None
        probList = []

      # Append prediction results.
      allPredsIndices.append(predictedClassIndex)
      allGtsIndices.append(int(yIndices[i]))
      allPredsConfidences.append(predictedConfidence)
      allPredsProbs.append(probList)

      # Resolve class names for display.
      predName = (
        classNames[predictedClassIndex]
        if (0 <= predictedClassIndex < len(classNames))
        else "Unknown"
      )
      allPredsNames.append(predName)
      allGtsNames.append(classNames[int(yIndices[i])])

      # Compute per-sample ECE if requested.
      eceValue = None
      if (computeECE and probList):
        try:
          eceValue = ComputeECE([probList], [int(yIndices[i])])
        except Exception:
          eceValue = None

      # Determine if prediction is correct.
      correctness = (predictedClassIndex == int(yIndices[i]))

      # Record full prediction metadata.
      predictionsRecords.append({
        "sampleIdx"          : i,
        "trueClassIndex"     : int(yIndices[i]),
        "trueClassName"      : classNames[int(yIndices[i])],
        "predictedClassIndex": predictedClassIndex,
        "predictedClassName" : predName,
        "predictedConfidence": predictedConfidence,
        "probabilities"      : (json.dumps(probList) if probList else None),
        "ece"                : (float(eceValue) if eceValue is not None else None),
        "correctness"        : correctness,
      })

    # Log progress.
    print(f"Prediction collection completed for {len(allGtsIndices)} samples across {numClasses} classes.")
    print(f"Total samples collected for confusion matrix: {len(allGtsIndices)}")
    print(f"{'-' * 60}")

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
        figSize=figSize,
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
        figSize=figSize,
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
        figSize=figSize,
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
    str(storageFilePath) if (storageFilePath) else None,
    weightedMetrics,
    allPredsIndices,
    allGtsIndices,
    allPredsProbs,
    allPredsConfidences,
    predictionsRecords,
    classNames,
    cm,
  )


class PyTorchClassificationTrainingPipeline:
  r'''
  Trainer helper for training, validating and evaluating a classification model using PyTorch.

  This class wraps a PyTorch classification model and provides convenience wiring for the
  training loop, validation, checkpointing, metric computation and saving
  prediction artifacts. The interface mirrors other helpers in this
  module and centralizes common I/O paths (logs, checkpoints, predictions,
  and metrics) so experiments are reproducible and easy to inspect.

  Parameters:
    model (torch.nn.Module): PyTorch classification model instance used for training and inference.
    trainLoader (DataLoader): Training DataLoader yielding (input, label) pairs.
    valLoader (DataLoader): Validation DataLoader used for periodic evaluation.
    allLoader (DataLoader): DataLoader covering the whole dataset (used for full predictions/exports).
    optimizer (torch.optim.Optimizer): Optimizer instance used to update model weights.
    scheduler (object | None): Learning-rate scheduler (optional) applied after each epoch/step.
    lossFn (callable): Loss function used for training (e.g., CrossEntropyLoss).
    learningRate (float): Base learning rate for bookkeeping and checkpointing (default: 1e-4).
    device (str): Device to run computations on ("cuda" or "cpu").
    outputDir (str): Directory where logs, checkpoints and outputs will be saved.
    dpi (int): DPI used when saving figures (default: 720).

  Attributes (selected):
    model, device, trainLoader, valLoader, allLoader, optimizer, scheduler,
    lossFn, writer (SummaryWriter), checkpointDir, predsDir, metricsDir, bestMetric

  Notes
  -----
    - TensorBoard summaries are written to outputDir/Logs.
    - Checkpoints are written to outputDir/Checkpoints via SaveCheckpoint wrapper.
    - Metric functions used for evaluation are mapped from HMB.PerformanceMetrics.
  '''

  # Initialize the trainer with model, dataloaders, optimizer, scheduler, and hparams.
  def __init__(
    self,
    model: nn.Module,
    trainLoader,
    valLoader,
    allLoader,
    optimizer: optim.Optimizer,
    scheduler,
    lossFn,
    noOfClasses: int,
    learningRate=1e-4,
    device: str = "cuda",
    outputDir: str = "Output",
    dpi: int = 720,
    logEveryNSteps: int = 100,
    useAmp: bool = True,  # Whether to use automatic mixed precision.
    earlyStoppingPatience: Optional[int] = None,  # Patience for early stopping.
    judgeBy: str = "val_loss",  # Metric to monitor: "val_loss" or "val_accuracy".
    checkpointSaver: Optional[CheckpointSaver] = None,  # Optional CheckpointSaver callback.
    verbose: bool = True,  # Verbosity flag to control logging.
    saveDataFrames: bool = True,  # Whether to save train/val/test/all DataFrames to outputDir.
    configs: Optional[Dict[str, Any]] = None,  # Optional config dictionary to save.
  ):
    r'''
    Initialize the PyTorch Classification Trainer.

    Parameters:
      model (torch.nn.Module): The classification model to train and evaluate.
      trainLoader (DataLoader): DataLoader for training data.
      valLoader (DataLoader): DataLoader for validation data.
      allLoader (DataLoader): DataLoader covering the entire dataset for inference/export.
      optimizer (torch.optim.Optimizer): Optimizer for training the model.
      scheduler: Learning rate scheduler (optional) to adjust learning rate during training.
      lossFn: Loss function used for training (e.g., CrossEntropyLoss).
      noOfClasses (int): Number of output classes for the classification task.
      learningRate (float): Base learning rate for bookkeeping and checkpointing (default: 1e-4).
      device (str): Device to run computations on ("cuda" or "cpu").
      outputDir (str): Directory where logs, checkpoints and outputs will be saved.
      dpi (int): DPI used when saving figures (default: 720).
      logEveryNSteps (int): Frequency of logging batch-level metrics to TensorBoard (default: 100).
      useAmp (bool): Whether to use automatic mixed precision for training (default: True).
      earlyStoppingPatience (int | None): Number of epochs with no improvement after which training will be stopped (default: None, meaning no early stopping).
      judgeBy (str): Metric to monitor for checkpointing and early stopping ("val_loss" or "val_accuracy", default: "val_loss").
      checkpointSaver (CheckpointSaver | None): Optional callback for saving checkpoints with custom logic (default: None).
      verbose (bool): Whether to print verbose logs during training and validation (default: True).
      saveDataFrames (bool): Whether to save train/val/test/all DataFrames to outputDir for reference (default: True).
      configs (dict | None): Optional configuration dictionary to save to outputDir/ConfigsUsed.json (default: None).
    '''

    # Store references to the model and device.
    self.model = model
    self.device = device
    self.noOfClasses = noOfClasses
    # Store dataloaders.
    self.trainLoader = trainLoader
    self.valLoader = valLoader
    self.allLoader = allLoader
    # Store optimizer, scheduler, and loss function.
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.lossFn = lossFn
    self.outputDir = outputDir
    self.learningRate = learningRate
    self.dpi = dpi
    self.logEveryNSteps = logEveryNSteps
    self.saveDataFrames = saveDataFrames
    self.configs = configs
    # Initialize TensorBoard writer in the output directory.
    self.writer = SummaryWriter(log_dir=os.path.join(self.outputDir, "Logs"))
    # Prepare the model on the target device.
    self.model.to(self.device)
    # Initialize the best metric for checkpointing.
    self.bestMetric = float("inf") if (judgeBy == "val_loss") else -1.0
    # Prepare checkpoint directory.
    self.checkpointDir = os.path.join(self.outputDir, "Checkpoints")
    os.makedirs(self.checkpointDir, exist_ok=True)
    # Prepare collectors for metrics computation.
    self.metricsDir = os.path.join(self.outputDir, "Metrics")
    # Ensure the metrics directory exists.
    os.makedirs(self.metricsDir, exist_ok=True)
    # Create predictions output directory inside the output directory.
    # self.predsDir = os.path.join(self.outputDir, "Preds")
    # Ensure the predictions directory exists.
    # os.makedirs(self.predsDir, exist_ok=True)
    # Create data output directory for saving DataFrames.
    self.dataDir = os.path.join(self.outputDir, "Data")
    if (self.saveDataFrames):
      os.makedirs(self.dataDir, exist_ok=True)

    # Store callback configuration.
    self.useAmp = useAmp
    self.judgeBy = judgeBy
    self.verbose = verbose

    # Create gradient scaler for mixed precision if not provided.
    if (useAmp):
      self.scaler = EnableMixedPrecision(model)
    else:
      # Create a dummy scaler with AMP disabled to avoid None checks.
      self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    # Initialize EarlyStopping callback if patience is specified.
    if (earlyStoppingPatience is not None):
      # Infer mode based on judgeBy parameter.
      mode = "min" if (judgeBy == "val_loss") else "max"
      self.earlyStopping = EarlyStopping(
        patience=earlyStoppingPatience,
        mode=mode,
        verbose=verbose
      )
    else:
      self.earlyStopping = None

    # Initialize CheckpointSaver callback if not provided.
    if (checkpointSaver is None):
      checkpointDir = os.path.join(outputDir, "Checkpoints")
      monitor = "val_loss" if (judgeBy == "val_loss") else "val_accuracy"
      mode = "min" if (judgeBy == "val_loss") else "max"
      self.checkpointSaver = CheckpointSaver(
        savePath=checkpointDir,
        saveBestOnly=True,
        monitor=monitor,
        mode=mode,
        verbose=verbose
      )
    else:
      self.checkpointSaver = checkpointSaver

    # Save configs if provided.
    if (self.configs is not None):
      self._SaveConfigs()

  def _SaveConfigs(self):
    r'''
    Save the configuration dictionary to outputDir/ConfigsUsed.json.
    '''
    if (self.configs is None):
      return
    # Create a copy to avoid modifying the original.
    configsCopy = dict(self.configs)
    # Remove large or redundant keys if needed.
    configsCopy.pop("Models", None)
    configsCopy["ModelUsed"] = getattr(self.model, "__class__", type(self.model)).__name__
    configsCopy["NumClasses"] = self.noOfClasses
    # Save to JSON file.
    configPath = os.path.join(self.outputDir, "ConfigsUsed.json")
    try:
      with open(configPath, "w") as f:
        json.dump(configsCopy, f, indent=4)
      if (self.verbose):
        print(f"Saved configs to {configPath}")
    except Exception as e:
      if (self.verbose):
        print(f"Warning: Failed to save configs: {e}")

  def SaveDataFrames(self, trainDF, valDF, testDF, allDF=None):
    r'''
    Save train, validation, test, and all DataFrames to outputDir/Data/.

    Parameters:
      trainDF (pd.DataFrame): Training DataFrame with features and labels.
      valDF (pd.DataFrame): Validation DataFrame with features and labels.
      testDF (pd.DataFrame): Test DataFrame with features and labels.
      allDF (pd.DataFrame | None): Optional combined DataFrame of all splits.
    '''
    if (not self.saveDataFrames):
      return
    try:
      trainDF.to_csv(os.path.join(self.dataDir, "TrainData.csv"), index=False)
      valDF.to_csv(os.path.join(self.dataDir, "ValData.csv"), index=False)
      testDF.to_csv(os.path.join(self.dataDir, "TestData.csv"), index=False)
      if (allDF is not None):
        allDF.to_csv(os.path.join(self.dataDir, "AllData.csv"), index=False)
      if (self.verbose):
        print(f"Saved DataFrames to {self.dataDir}")
    except Exception as e:
      if (self.verbose):
        print(f"Warning: Failed to save DataFrames: {e}")

  # Save a checkpoint to disk with a given tag.
  def SaveCheckpoint(self, epoch: int, tag: str = "latest"):
    filePath = os.path.join(self.checkpointDir, f"Checkpoint{tag.lower().capitalize()}.pth")
    SaveCheckpoint(self.model, self.optimizer, filePath, epoch=epoch, hparams=None)
    return filePath

  # Load checkpoint from disk and restore model and optimizer states.
  def LoadCheckpoint(self, filePath: str, strict: bool = True) -> int:
    checkpoint = LoadCheckpoint(
      filePath,
      self.model,
      self.optimizer,
      lr=self.learningRate,
      device=self.device,
      strict=strict
    )
    epoch = checkpoint.get("epoch", 0) + 1
    return epoch

  # Run the full training loop for a given number of epochs.
  def Train(self, numEpochs: int):
    # Loop over epochs from 1 to numEpochs inclusive.
    loader = tqdm.tqdm(range(1, numEpochs + 1), desc="Training epochs", unit="epoch")
    for epoch in loader:
      if (self.verbose):
        print(f"Starting epoch {epoch}/{numEpochs}")

      # Run a training epoch and obtain average loss and accuracy.
      trainLoss, trainAcc = self.TrainEpoch(epoch)
      # Run validation and obtain metrics dictionary.
      valLoss, valAcc = self.Validate(epoch)

      # Log scalars to TensorBoard for the epoch.
      self.writer.add_scalar("Loss/Train", trainLoss, epoch)
      self.writer.add_scalar("Loss/Val", valLoss, epoch)
      self.writer.add_scalar("Accuracy/Train", trainAcc, epoch)
      self.writer.add_scalar("Accuracy/Val", valAcc, epoch)

      # Update learning rate scheduler if present.
      if (self.scheduler is not None):
        try:
          if ("ReduceLROnPlateau" in type(self.scheduler).__name__):
            self.scheduler.step(valLoss)
          else:
            self.scheduler.step()
        except Exception as e:
          if (self.verbose):
            print(f"Warning: Failed to step scheduler. Error: {e}")

      # Determine metric to monitor based on judgeBy parameter.
      monitorMetric = valLoss if (self.judgeBy == "val_loss") else valAcc

      # Use CheckpointSaver callback for saving checkpoints.
      if (self.checkpointSaver is not None):
        self.checkpointSaver(model=self.model, currentMetric=monitorMetric, epoch=epoch)

      # Use EarlyStopping callback if available.
      if (self.earlyStopping is not None):
        if (self.earlyStopping(monitorMetric)):
          if (self.verbose):
            print(f"Early stopping triggered at epoch {epoch}.")
          break

      # Print epoch summary.
      if (self.verbose):
        print(
          f"Epoch {epoch}/{numEpochs} - Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f} - "
          f"Train Acc: {trainAcc:.4f}, Val Acc: {valAcc:.4f}"
        )

  # Run a single training epoch and return average loss and accuracy.
  def TrainEpoch(self, epoch: int) -> Tuple[float, float]:
    r'''
    Run a single training epoch over the training dataset and return the average loss and accuracy.

    Parameters:
      epoch (int): The current epoch number (used for logging and checkpointing).

    Returns:
      tuple: (avgTrainLoss, avgTrainAccuracy) for the epoch.
    '''

    # Set model to training mode.
    self.model.train()
    # Initialize running loss and accuracy.
    runningLoss = 0.0
    runningAcc = 0.0
    count = 0

    # Determine device type for autocast.
    deviceType = "cuda" if (self.device == "cuda" or "cuda" in str(self.device)) else "cpu"

    # Iterate over batches from the train loader.
    for batchIdx, (images, labels) in tqdm.tqdm(
      enumerate(self.trainLoader),
      total=len(self.trainLoader),
      desc=f"Training Epoch {epoch}"
    ):
      # Move images and labels to the configured device.
      images = images.to(self.device)
      labels = labels.to(self.device)
      # Zero gradients on optimizer.
      self.optimizer.zero_grad()

      # Use automatic mixed precision for the forward pass if enabled.
      if (self.useAmp):
        with autocast(enabled=True, device_type=deviceType):
          # Forward pass to obtain logits.
          logits = self.model(images)
          # Compute loss using the provided loss function.
          loss = self.lossFn(logits, labels)
        # Scale loss and backward pass.
        self.scaler.scale(loss).backward()
      else:
        # Forward pass without AMP.
        logits = self.model(images)
        loss = self.lossFn(logits, labels)
        loss.backward()

      # Optimizer step with scaler if AMP enabled.
      if (self.useAmp):
        self.scaler.step(self.optimizer)
        self.scaler.update()
      else:
        self.optimizer.step()

      # Compute accuracy.
      preds = torch.argmax(logits, dim=1)
      acc = (preds == labels).float().mean().item()

      # Accumulate loss and accuracy.
      runningLoss += loss.item()
      runningAcc += acc
      count += 1

      # Optionally log batch-level metrics to TensorBoard.
      if ((batchIdx + 1) % self.logEveryNSteps == 0):
        self.writer.add_scalar(
          "Loss/TrainBatch",
          runningLoss / float(count),
          epoch * len(self.trainLoader) + batchIdx
        )
        self.writer.add_scalar(
          "Accuracy/TrainBatch",
          runningAcc / float(count),
          epoch * len(self.trainLoader) + batchIdx
        )

    # Return average training loss and accuracy for the epoch.
    return runningLoss / float(max(1, count)), runningAcc / float(max(1, count))

  # Run validation over the validation set and return loss and accuracy.
  def Validate(self, epoch: int) -> Tuple[float, float]:
    r'''
    Run validation over the validation set and compute loss and accuracy.

    Parameters:
      epoch (int): The current epoch number (used for logging and checkpointing).

    Returns:
      tuple: (avgValLoss, avgValAccuracy) for the epoch.
    '''

    # Set model to evaluation mode.
    self.model.eval()

    # Initialize accumulators for loss and accuracy.
    runningLoss = 0.0
    runningAcc = 0.0
    count = 0

    # Disable gradient computation during validation.
    with torch.no_grad():
      # Iterate over validation batches.
      for (images, labels) in self.valLoader:
        # Move to device.
        images = images.to(self.device)
        labels = labels.to(self.device)
        # Forward pass to obtain logits.
        logits = self.model(images)
        # Compute loss.
        loss = self.lossFn(logits, labels)
        # Compute accuracy.
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        # Accumulate loss and accuracy.
        runningLoss += loss.item()
        runningAcc += acc
        count += 1

    # Return average validation loss and accuracy.
    return runningLoss / float(max(1, count)), runningAcc / float(max(1, count))

  def Inference(self):
    r'''
    Run inference on the entire dataset using the allLoader and save predictions and metrics to disk.
    '''

    # Set the model to evaluation mode to disable training-specific layers.
    self.model.eval()
    # Initialize a global counter for saved prediction files.
    globalIdx = 0

    # Initialize a list to store per-sample metrics.
    perSampleMetrics = []
    # Initialize a list to store any failures encountered.
    failedSamples = []
    # Disable gradient computation for inference to save memory and compute.
    with torch.inference_mode():
      # Iterate over the dataloader batches.
      for batchIdx, (images, labels) in enumerate(self.allLoader):
        # Move images to the selected device.
        images = images.to(self.device)
        # Forward pass through the model to obtain logits.
        logits = self.model(images)
        # Compute predicted class indices.
        preds = torch.argmax(logits, dim=1)

        # Iterate over items in the batch and save each prediction and compute metrics.
        for i in range(preds.shape[0]):
          # Compute the absolute sample index within the dataset.
          sampleIdx = batchIdx * (
            self.allLoader.batch_size
            if ((hasattr(self.allLoader, "batch_size") and self.allLoader.batch_size is not None)) else 1
          )
          # Add the intra-batch index to obtain the final sample index.
          sampleIdx += i
          # Extract the original label and prediction.
          trueLabel = labels[i].item()
          predLabel = preds[i].item()
          # Compute correctness.
          isCorrect = (trueLabel == predLabel)

          # Record per-sample metric.
          row = {
            "sampleIdx": sampleIdx,
            "trueLabel": trueLabel,
            "predLabel": predLabel,
            "isCorrect": isCorrect,
          }
          perSampleMetrics.append(row)

          # Increment the global index for bookkeeping.
          globalIdx += 1

    # Build paths for primary and additional metrics files.
    metricsCsvPath = os.path.join(self.metricsDir, "PerSampleMetrics.csv")
    metricsSummaryPath = os.path.join(self.metricsDir, "MetricsSummary.json")

    # Save the primary per-sample CSV.
    if (len(perSampleMetrics) > 0):
      # CSV fieldnames.
      fieldnames = ["sampleIdx", "trueLabel", "predLabel", "isCorrect"]
      try:
        with open(metricsCsvPath, "w", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for r in perSampleMetrics:
            writer.writerow(r)
      except Exception as e:
        print(f"Failed to write metrics CSV: {e}")

    # Compute the overall summary statistics for all samples.
    totalSamples = len(perSampleMetrics)
    correctSamples = sum(1 for r in perSampleMetrics if r["isCorrect"])
    overallAccuracy = correctSamples / float(max(1, totalSamples))
    overallSummary = {
      "NSamples"       : totalSamples,
      "NFailed"        : len(failedSamples),
      "OverallAccuracy": overallAccuracy,
    }

    # Write the overall summary JSON.
    DumpJsonFile(metricsSummaryPath, overallSummary)

    # Print final status messages about saved files.
    # print(f"Saved {globalIdx} predictions to: {self.predsDir}.")
    print(f"Per-sample metrics: {metricsCsvPath}")
    print(f"Metrics summary: {metricsSummaryPath}")
    print(f"Overall accuracy: {overallAccuracy:.4f}")


class PyTorchUNetSegmentationTrainingPipeline:
  r'''
  Trainer helper for training, validating and evaluating a U-Net segmentation model using PyTorch.

  This class wraps a PyTorch U-Net model and provides convenience wiring for the
  training loop, validation, checkpointing, metric computation and saving
  prediction/overlay artifacts. The interface mirrors other helpers in this
  module and centralizes common I/O paths (logs, checkpoints, overlays,
  predictions, and metrics) so experiments are reproducible and easy to inspect.

  Parameters:
    model (torch.nn.Module): PyTorch U-Net model instance used for training and inference.
    trainLoader (DataLoader): Training DataLoader yielding (input, target) pairs.
    valLoader (DataLoader): Validation DataLoader used for periodic evaluation.
    allLoader (DataLoader): DataLoader covering the whole dataset (used for full predictions/exports).
    optimizer (torch.optim.Optimizer): Optimizer instance used to update model weights.
    scheduler (object | None): Learning-rate scheduler (optional) applied after each epoch/step.
    lossFn (callable): Loss function used for training (e.g., BCEWithLogitsLoss, DiceLoss).
    learningRate (float): Base learning rate for bookkeeping and checkpointing (default: 1e-4).
    device (str): Device to run computations on ("cuda" or "cpu").
    outputDir (str): Directory where logs, checkpoints and outputs will be saved.
    dpi (int): DPI used when saving figures (default: 720).

  Attributes (selected):
    model, device, trainLoader, valLoader, allLoader, optimizer, scheduler,
    lossFn, writer (SummaryWriter), checkpointDir, overlaysDir, predsDir,
    actualDir, metricsDir, bestMetric

  Notes
  -----
    - TensorBoard summaries are written to outputDir/Logs.
    - Checkpoints are written to outputDir/Checkpoints via SaveCheckpoint wrapper.
    - Metric functions used for evaluation are mapped from HMB.ImageSegmentationMetrics.
  '''

  # Initialize the trainer with model, dataloaders, optimizer, scheduler, and hparams.
  def __init__(
    self,
    model: nn.Module,
    trainLoader,
    valLoader,
    allLoader,
    optimizer: optim.Optimizer,
    scheduler,
    lossFn,
    learningRate=1e-4,
    device: str = "cuda",
    outputDir: str = "Output",
    dpi: int = 720,
    logEveryNSteps: int = 100,
    useAmp: bool = True,  # Whether to use automatic mixed precision.
    earlyStoppingPatience: Optional[int] = None,  # Patience for early stopping.
    judgeBy: str = "val_loss",  # Metric to monitor: "val_loss" or "val_dice".
    checkpointSaver: Optional[CheckpointSaver] = None,  # Optional CheckpointSaver callback.
    verbose: bool = True,  # Verbosity flag to control logging.
  ):
    r'''
    Initialize the PyTorch UNet Segmentation Trainer.

    Parameters:
      model (torch.nn.Module): The U-Net model to train and evaluate.
      trainLoader (DataLoader): DataLoader for training data.
      valLoader (DataLoader): DataLoader for validation data.
      allLoader (DataLoader): DataLoader covering the entire dataset for inference/export.
      optimizer (torch.optim.Optimizer): Optimizer for training the model.
      scheduler: Learning rate scheduler (optional) to adjust learning rate during training.
      lossFn: Loss function used for training (e.g., BCEWithLogitsLoss, DiceLoss).
      learningRate (float): Base learning rate for bookkeeping and checkpointing (default: 1e-4).
      device (str): Device to run computations on ("cuda" or "cpu").
      outputDir (str): Directory where logs, checkpoints and outputs will be saved.
      dpi (int): DPI used when saving figures (default: 720).
      logEveryNSteps (int): Frequency of logging batch-level metrics to TensorBoard (default: 100).
      useAmp (bool): Whether to use automatic mixed precision for training (default: True).
      earlyStoppingPatience (int | None): Number of epochs with no improvement after which training will be stopped (default: None, meaning no early stopping).
      judgeBy (str): Metric to monitor for checkpointing and early stopping ("val_loss" or "val_dice", default: "val_loss").
      checkpointSaver (CheckpointSaver | None): Optional callback for saving checkpoints with custom logic (default: None).
      verbose (bool): Whether to print verbose logs during training and validation (default: True).
    '''

    # Store references to the model and device.
    self.model = model
    self.device = device
    # Store dataloaders.
    self.trainLoader = trainLoader
    self.valLoader = valLoader
    self.allLoader = allLoader
    # Store optimizer, scheduler, and loss function.
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.lossFn = lossFn
    self.outputDir = outputDir
    self.learningRate = learningRate
    self.dpi = dpi
    self.logEveryNSteps = logEveryNSteps
    # Initialize TensorBoard writer in the output directory.
    self.writer = SummaryWriter(log_dir=os.path.join(self.outputDir, "Logs"))
    # Prepare the model on the target device.
    self.model.to(self.device)
    # Initialize the best metric for checkpointing.
    self.bestMetric = -1.0
    # Prepare checkpoint directory.
    self.checkpointDir = os.path.join(self.outputDir, "Checkpoints")
    os.makedirs(self.checkpointDir, exist_ok=True)
    # Compute the output directory path for saving combined PNG files.
    self.trainSamplesDir = os.path.join(self.outputDir, "TrainSamples")
    # Ensure the output directory exists on disk.
    os.makedirs(self.trainSamplesDir, exist_ok=True)
    # Prepare collectors for metrics computation.
    self.metricsDir = os.path.join(self.outputDir, "Metrics")
    # Ensure the metrics directory exists.
    os.makedirs(self.metricsDir, exist_ok=True)
    # Prepare and save overlay images showing Original | PredOverlay (red) | TargetOverlay (green).
    # Build overlays output directory inside the experiment output dir.
    self.overlaysDir = os.path.join(self.outputDir, "Overlays")
    # Ensure the overlays directory exists.
    os.makedirs(self.overlaysDir, exist_ok=True)
    # Create predictions output directory inside the output directory.
    self.predsDir = os.path.join(self.outputDir, "Preds")
    self.actualDir = os.path.join(self.outputDir, "Actuals")
    # Ensure the predictions directory exists.
    os.makedirs(self.predsDir, exist_ok=True)
    os.makedirs(self.actualDir, exist_ok=True)

    # Map metric display names to functions in HMB.ImageSegmentationMetrics.
    self.metricsFns = {
      "IoU"          : ISM.ComputeIoU,
      "Dice"         : ISM.ComputeDice,
      "PixelAccuracy": ISM.ComputePixelAccuracy,
      "Precision"    : ISM.ComputePrecision,
      "Recall"       : ISM.ComputeRecall,
      "Specificity"  : ISM.ComputeSpecificity,
      "FPR"          : ISM.ComputeFPR,
      "FNR"          : ISM.ComputeFNR,
      "F1Score"      : ISM.ComputeF1Score,
      "mAP"          : ISM.ComputeMeanAveragePrecision,
      "Hausdorff"    : ISM.ComputeHausdorffDistance,
      "BoundaryF1"   : ISM.ComputeBoundaryF1Score,
      "MCC"          : ISM.ComputeMatthewsCorrelationCoefficient,
      "CohensKappa"  : ISM.ComputeCohensKappa,
    }

    # Store callback configuration.
    self.useAmp = useAmp
    self.judgeBy = judgeBy
    self.verbose = verbose

    # Create gradient scaler for mixed precision if not provided.
    if (useAmp):
      self.scaler = EnableMixedPrecision(model)
    else:
      # Create a dummy scaler with AMP disabled to avoid None checks.
      self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    # Initialize EarlyStopping callback if patience is specified.
    if (earlyStoppingPatience is not None):
      # Infer mode based on judgeBy parameter.
      mode = "min" if (judgeBy == "val_loss") else "max"
      self.earlyStopping = EarlyStopping(
        patience=earlyStoppingPatience,
        mode=mode,
        verbose=verbose
      )
    else:
      self.earlyStopping = None

    # Initialize CheckpointSaver callback if not provided.
    if (checkpointSaver is None):
      checkpointDir = os.path.join(outputDir, "Checkpoints")
      monitor = "val_loss" if (judgeBy == "val_loss") else "val_dice"
      mode = "min" if (judgeBy == "val_loss") else "max"
      self.checkpointSaver = CheckpointSaver(
        savePath=checkpointDir,
        saveBestOnly=True,
        monitor=monitor,
        mode=mode,
        verbose=verbose
      )
    else:
      self.checkpointSaver = checkpointSaver

    # Track best metric for early stopping and checkpointing.
    self.bestMetric = float("inf") if (judgeBy == "val_loss") else -1.0

  # Save a checkpoint to disk with a given tag.
  def SaveCheckpoint(self, epoch: int, tag: str = "latest"):
    filePath = os.path.join(self.checkpointDir, f"Checkpoint{tag.lower().capitalize()}.pth")
    SaveCheckpoint(self.model, self.optimizer, filePath, epoch=epoch, hparams=None)
    return filePath

  # Load checkpoint from disk and restore model and optimizer states.
  def LoadCheckpoint(self, filePath: str, strict: bool = True) -> int:
    checkpoint = LoadCheckpoint(
      filePath,
      self.model,
      self.optimizer,
      lr=self.learningRate,
      device=self.device,
      strict=strict
    )
    epoch = checkpoint.get("epoch", 0) + 1
    return epoch

  # Run the full training loop for a given number of epochs.
  def Train(self, numEpochs: int):
    # Loop over epochs from 1 to numEpochs inclusive.
    for epoch in range(1, numEpochs + 1):
      if (self.verbose):
        print(f"Starting epoch {epoch}/{numEpochs}")

      # Run a training epoch and obtain average loss.
      trainLoss = self.TrainEpoch(epoch)
      # Run validation and obtain metrics dictionary.
      valMetrics = self.Validate(epoch)

      # Log scalars to TensorBoard for the epoch.
      self.writer.add_scalar("Loss/Train", trainLoss, epoch)
      self.writer.add_scalar("Loss/Val", valMetrics.get("Loss", 0.0), epoch)
      self.writer.add_scalar("Metrics/ValDice", valMetrics.get("MeanDice", 0.0), epoch)

      # Update learning rate scheduler if present.
      if (self.scheduler is not None):
        try:
          if ("ReduceLROnPlateau" in type(self.scheduler).__name__):
            self.scheduler.step(valMetrics.get("Loss", 0.0))
          else:
            self.scheduler.step()
        except Exception as e:
          if (self.verbose):
            print(f"Warning: Failed to step scheduler. Error: {e}")

      # Determine metric to monitor based on judgeBy parameter.
      monitorMetric = (
        valMetrics.get("Loss", float("inf"))
        if (self.judgeBy == "val_loss") else valMetrics.get("MeanDice", -1.0)
      )

      # Use CheckpointSaver callback for saving checkpoints.
      if (self.checkpointSaver is not None):
        self.checkpointSaver(model=self.model, currentMetric=monitorMetric, epoch=epoch)

      # Use EarlyStopping callback if available.
      if (self.earlyStopping is not None):
        if (self.earlyStopping(monitorMetric)):
          if (self.verbose):
            print(f"Early stopping triggered at epoch {epoch}.")
          break

      # Print epoch summary.
      if (self.verbose):
        print(
          f"Epoch {epoch}/{numEpochs} - Train Loss: {trainLoss:.4f} - "
          f"Val Loss: {valMetrics.get('Loss', 0.0):.4f} - "
          f"Val Dice: {valMetrics.get('MeanDice', 0.0):.4f} - "
          f"Val IoU: {valMetrics.get('MeanIoU', 0.0):.4f} - "
          f"Val Pixel Acc: {valMetrics.get('PixelAccuracy', 0.0):.4f}"
        )

  def _VisualizeImage(
    self,
    image,
    mask,
    pred,
    logits,
    probPath,
    combinedPath,
  ):
    r'''
    Visualize the input image, ground truth mask, and predicted mask side by side.

    This function prepares the input image, ground truth mask, and predicted mask for visualization.
    It handles both single-channel and multi-channel images, normalizes them for display,
    and saves a combined figure showing the image, mask, and prediction. If the model outputs binary
    logits, it also saves the probability map as a separate image for inspection.

    Parameters:
      image (torch.Tensor): The input image tensor to visualize.
      mask (torch.Tensor): The ground truth mask tensor to visualize.
      pred (torch.Tensor): The predicted mask tensor to visualize.
      logits (torch.Tensor): The raw output logits from the model (used to determine if binary).
      probPath (str): The file path to save the probability map image if applicable.
      combinedPath (str): The file path to save the combined visualization image.

    Notes:
      - The function handles both [C, H, W] and [H, W] image tensors, normalizing them for display.
      - The ground truth mask and predicted mask are also prepared for visualization, squeezing channel dimensions if necessary.
      - If the model outputs binary logits (single channel), the function saves the probability map as a separate image using a perceptually-uniform colormap.
      - The combined figure shows the input image, ground truth mask, and predicted mask side by side with appropriate titles and no axis decorations.
    '''

    # Prepare image, mask and prediction numpy arrays for plotting.
    # images: [B, C, H, W] or [B, H, W].
    imgTensor = image.detach().cpu()
    # Handle both [C,H,W] and [H,W] image tensors.
    try:
      imgNp = imgTensor.numpy()
    except Exception:
      imgNp = np.array(imgTensor)

    if (imgNp.ndim == 3):
      # Convert from [C, H, W] to [H, W, C].
      imgVis = np.transpose(imgNp, (1, 2, 0))
      # If single-channel, squeeze to [H, W].
      if (imgVis.shape[2] == 1):
        imgVis = imgVis.squeeze(2)
    elif (imgNp.ndim == 2):
      imgVis = imgNp
    else:
      # Fallback: try to squeeze to 2D.
      imgVis = np.squeeze(imgNp)

    # Normalize image for display (if values appear in 0-255 range).
    try:
      if ((imgVis.dtype == np.uint8) or (imgVis.max() > 1.0)):
        imgVis = imgVis.astype(np.float32) / 255.0
    except Exception:
      pass

    # Prepare mask numpy.
    maskTensor = mask.detach().cpu()
    try:
      maskNp = maskTensor.numpy()
    except Exception:
      maskNp = np.array(maskTensor)
    # If mask has channel dim [1, H, W], squeeze it.
    if (maskNp.ndim == 3 and maskNp.shape[0] == 1):
      maskNp = maskNp.squeeze(0)
    # If mask is [H, W, 1], squeeze last dim.
    if (maskNp.ndim == 3 and maskNp.shape[-1] == 1):
      maskNp = maskNp.squeeze(-1)

    # Prepare prediction numpy.
    try:
      predTensor = pred.detach().cpu()
      predNp = predTensor.numpy()
    except Exception:
      predNp = np.array(preds[i].cpu())

    # Ensure prediction is 2D.
    if (predNp.ndim == 3 and predNp.shape[0] == 1):
      predNp = predNp.squeeze(0)

    # If logits are binary, save the probability map for inspection.
    if (logits.shape[1] == 1):
      # Convert the per-sample probability map to numpy.
      probNp = probs[i].cpu().numpy()
      # Squeeze a leading channel dimension if present.
      if (probNp.ndim == 3 and probNp.shape[0] == 1):
        probNp = probNp.squeeze(0)
      # Create a small figure and save the probability heatmap.
      figProb, axProb = plt.subplots(1, 1, figsize=(4, 4))
      # Display the probability map with a perceptually-uniform colormap.
      axProb.imshow(probNp, cmap="viridis", vmin=0.0, vmax=1.0)
      # Turn off axis decorations.
      axProb.axis("off")
      # Save the probability map to disk.
      figProb.savefig(probPath, dpi=self.dpi, bbox_inches="tight")
      # Close the small figure to free memory.
      plt.close(figProb)

    # Create a matplotlib figure with three horizontal subplots.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Display the input image on the first axis.
    if (imgVis.ndim == 2):
      axes[0].imshow(imgVis, cmap="gray")
    # Display the RGB image when it has multiple channels.
    else:
      axes[0].imshow(np.clip(imgVis, 0, 1))
    # Set the title for the input image subplot.
    axes[0].set_title("Image")
    # Turn off axis ticks and labels for the image subplot.
    axes[0].axis("off")
    # Display the mask on the second axis using a grayscale colormap.
    try:
      axes[1].imshow(maskNp, cmap="gray")
    except Exception:
      axes[1].imshow(np.squeeze(maskNp), cmap="gray")
    # Set the title for the mask subplot.
    axes[1].set_title("Mask")
    # Turn off axis ticks and labels for the mask subplot.
    axes[1].axis("off")
    # Display the prediction on the third axis using a grayscale colormap.
    axes[2].imshow(predNp, cmap="gray")
    # Set the title for the prediction subplot.
    axes[2].set_title("Pred")
    # Turn off axis ticks and labels for the prediction subplot.
    axes[2].axis("off")
    # Adjust subplot spacing to prevent overlap.
    plt.tight_layout()
    # Save the figure to disk as a PNG file with a moderate DPI.
    fig.savefig(combinedPath, dpi=self.dpi, bbox_inches="tight")
    # Close the figure to release memory resources.
    plt.close(fig)

  def PredictImage(self, image: torch.Tensor) -> torch.Tensor:
    r'''
    Run inference on a single input image tensor and return the predicted mask tensor.

    Parameters:
      image (torch.Tensor): The input image tensor to run inference on. Expected shape is [C, H, W] or [H, W].

    Returns:
      torch.Tensor: The predicted mask tensor. Shape will be [H, W] for single-channel output or [C, H, W] for multi-channel output.
    '''

    # Set model to evaluation mode.
    self.model.eval()
    # Disable gradient calculation for inference.
    with torch.no_grad():
      # Move the input image to the configured device.
      image = image.to(self.device)
      # Forward pass through the model to obtain logits.
      logits = self.model(image.unsqueeze(0))  # Add batch dimension.
      # Check if the model outputs binary or multi-class logits and convert to predictions.
      if (logits.shape[1] == 1):
        # Binary case: apply sigmoid and threshold at 0.5.
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(0).squeeze(0)  # Remove batch and channel dims.
      else:
        # Multi-class case: take argmax along channel dimension.
        preds = torch.argmax(logits, dim=1).squeeze(0)  # Remove batch dim.
    return preds

  # Run a single training epoch and return average loss.
  def TrainEpoch(self, epoch: int) -> float:
    r'''
    Run a single training epoch over the training dataset and return the average loss.

    Parameters:
      epoch (int): The current epoch number (used for logging and checkpointing).

    Returns:
      float: The average training loss for the epoch.
    '''

    # Set model to training mode.
    self.model.train()
    # Initialize running loss and sample count.
    runningLoss = 0.0
    count = 0

    # Determine device type for autocast.
    deviceType = "cuda" if (self.device == "cuda" or "cuda" in str(self.device)) else "cpu"

    # Iterate over batches from the train loader.
    for batchIdx, (images, masks) in tqdm.tqdm(
      enumerate(self.trainLoader),
      total=len(self.trainLoader),
      desc=f"Training Epoch {epoch}"
    ):
      # Move images and masks to the configured device.
      images = images.to(self.device)
      masks = masks.to(self.device)
      # Zero gradients on optimizer.
      self.optimizer.zero_grad()

      # Use automatic mixed precision for the forward pass if enabled.
      if (self.useAmp):
        with autocast(enabled=True, device_type=deviceType):
          # Forward pass to obtain logits.
          logits = self.model(images)
          # Compute loss using the provided loss function.
          loss = self.lossFn(logits, masks)
        # Scale loss and backward pass.
        self.scaler.scale(loss).backward()
      else:
        # Forward pass without AMP.
        logits = self.model(images)
        loss = self.lossFn(logits, masks)
        loss.backward()

      # ... optimizer step with scaler if AMP enabled ...
      if (self.useAmp):
        self.scaler.step(self.optimizer)
        self.scaler.update()
      else:
        self.optimizer.step()

      # Accumulate loss and increment count.
      runningLoss += loss.item()
      count += 1
      # Optionally log batch-level loss to TensorBoard.
      if ((batchIdx + 1) % self.logEveryNSteps == 0):
        self.writer.add_scalar(
          "Loss/TrainBatch",
          runningLoss / float(count),
          epoch * len(self.trainLoader) + batchIdx
        )
        # Predict on a few samples and log images to `TensorBoard`.
        # Disable gradient calculation for prediction logging.
        with torch.no_grad():
          # Check whether the logits have a single channel indicating binary segmentation.
          if (logits.shape[1] == 1):
            # Compute probabilities from logits using the sigmoid function.
            probs = torch.sigmoid(logits)
            # Threshold probabilities at 0.5 to obtain binary predictions.
            preds = (probs > 0.5).long().squeeze(1)
          # Handle the multi-class logits case when channels > 1.
          if (logits.shape[1] != 1):
            # Compute argmax across the channel dimension for multi-class predictions.
            preds = torch.argmax(logits, dim=1)

          # Loop over up to two samples in the current batch for visualization.
          for i in range(min(2, int(images.size(0)))):
            # Compose the filename for the saved combined image.
            combinedPath = os.path.join(self.trainSamplesDir, f"Epoch{epoch}_Batch{batchIdx}_Sample{i}.png")
            # Build a filename for the probability map image.
            probPath = os.path.join(self.trainSamplesDir, f"Epoch{epoch}_Batch{batchIdx}_Sample{i}_Prob.png")
            # Call the visualization function to save the combined image and probability map.
            self._VisualizeImage(
              image=images[i],
              mask=masks[i],
              pred=preds[i],
              logits=logits[i],
              probPath=probPath,
              combinedPath=combinedPath,
            )

    # Return average training loss for the epoch.
    return runningLoss / float(max(1, count))

  # Run validation over the validation set and return metrics dictionary.
  def Validate(self, epoch: int) -> Dict:
    r'''
    Run validation over the validation set and compute metrics.

    Parameters:
      epoch (int): The current epoch number (used for logging and checkpointing).

    Returns:
      dict: A dictionary containing average loss and computed metrics (e.g., MeanDice, Mean IoU, PixelAccuracy) for the validation set.
    '''

    # Set model to evaluation mode.
    self.model.eval()

    # Initialize accumulators for loss and metrics.
    runningLoss = 0.0
    count = 0
    diceScores = []
    iouScores = []
    pixelAccs = []

    # Disable gradient computation during validation.
    with torch.no_grad():
      # Iterate over validation batches.
      for (images, masks) in self.valLoader:
        # Move to device.
        images = images.to(self.device)
        masks = masks.to(self.device)
        # Forward pass to obtain logits.
        logits = self.model(images)
        # Compute loss.
        loss = self.lossFn(logits, masks)
        # Convert logits to predicted masks depending on number of classes.
        if (logits.shape[1] == 1):
          # Binary case: apply sigmoid and threshold at 0.5.
          probs = torch.sigmoid(logits)
          preds = (probs > 0.5).long().squeeze(1)
          # Ensure masks are also [B, H, W].
          if (masks.dim() == 4 and masks.shape[1] == 1):
            masks = masks.squeeze(1)  # [B, 1, H, W] → [B, H, W].
        else:
          # Multi-class case: take argmax along channel dimension.
          preds = torch.argmax(logits, dim=1)
        # Compute metrics per batch by converting to appropriate tensors.
        for i in range(preds.shape[0]):
          pred = preds[i].float().cpu().numpy()
          tgt = masks[i].float().cpu().numpy()
          # Compute Dice, IoU, and pixel accuracy for the instance.
          diceScores.append(self.metricsFns["Dice"](pred, tgt))
          iouScores.append(self.metricsFns["IoU"](pred, tgt))
          pixelAccs.append(self.metricsFns["PixelAccuracy"](pred, tgt))
        # Accumulate loss and increment count.
        runningLoss += loss.item()
        count += 1
    # Compute average metrics across validation.
    avgLoss = runningLoss / float(max(1, count))
    # Compute mean values using numpy for stability.
    meanDice = float(np.mean(diceScores) if (len(diceScores) > 0) else 0.0)
    meanIoU = float(np.mean(iouScores) if (len(iouScores) > 0) else 0.0)
    meanAcc = float(np.mean(pixelAccs) if (len(pixelAccs) > 0) else 0.0)
    # Return a dictionary containing computed metrics and average loss.
    return {"Loss": avgLoss, "MeanDice": meanDice, "MeanIoU": meanIoU, "PixelAccuracy": meanAcc}

  def Inference(self):
    r'''
    Run inference on the entire dataset using the allLoader and save predicted masks, actual masks, and overlays to disk.
    '''

    # Set the model to evaluation mode to disable training-specific layers.
    self.model.eval()
    # Initialize a global counter for saved prediction files.
    globalIdx = 0

    # Initialize a list to store per-image metrics.
    perImageMetrics = []
    # Initialize a list to store any failures encountered.
    failedImages = []
    # Disable gradient computation for inference to save memory and compute.
    with torch.inference_mode():
      # Iterate over the validation dataloader batches.
      for batchIdx, (images, masks) in enumerate(self.allLoader):
        # Move images to the selected device.
        images = images.to(self.device)
        # Forward pass through the model to obtain logits.
        logits = self.model(images)

        # Check whether the logits have a single channel indicating binary segmentation.
        if (logits.shape[1] == 1):
          # Compute probabilities from logits using the sigmoid function.
          probs = torch.sigmoid(logits)
          # Threshold probabilities at 0.5 to obtain binary predictions.
          preds = (probs > 0.5).long().squeeze(1)
        # Handle the multi-class logits case when channels > 1.
        else:
          # Compute argmax across the channel dimension for multi-class predictions.
          preds = torch.argmax(logits, dim=1)

        # Iterate over items in the batch and save each predicted mask and compute metrics.
        for i in range(preds.shape[0]):
          # Convert the predicted mask and target mask tensors to numpy arrays for saving and metric computation.
          predMask = PreparePredTensorToNumpy(preds[i], doScale2Image=True)
          targetMask = PreparePredTensorToNumpy(masks[i], doScale2Image=True)

          # Compute the absolute sample index within the dataset.
          sampleIdx = batchIdx * (
            self.allLoader.batch_size
            if ((hasattr(self.allLoader, "batch_size") and self.allLoader.batch_size is not None)) else 1
          )
          # Add the intra-batch index to obtain the final sample index.
          sampleIdx += i
          origImagePath = (
            self.allLoader.dataset.samples[sampleIdx][0]
            if (hasattr(self.allLoader.dataset, "samples") and len(self.allLoader.dataset.samples) > sampleIdx)
            else f"sample_{sampleIdx}.png"
          )
          # Extract the basename of the original image to use as the output filename.
          origBaseName = os.path.basename(origImagePath)

          # Build the output path using the original filename.
          predOutputPath = os.path.join(self.predsDir, origBaseName)
          actualOutputPath = os.path.join(self.actualDir, origBaseName)
          # Write the mask image to disk using OpenCV at the built path.
          cv2.imwrite(predOutputPath, predMask)
          cv2.imwrite(actualOutputPath, targetMask)
          # Increment the global index for bookkeeping.
          globalIdx += 1

          # Ensure shapes match: if not, try to resize prediction to match target using nearest-neighbor.
          if (predMask.shape != targetMask.shape):
            try:
              # Use simple nearest-neighbor resizing via OpenCV to preserve integer labels.
              # Get target height and width for resizing.
              targetH, targetW = targetMask.shape[-2:]
              # Convert prediction to uint8 before resizing.
              _pm = predMask.astype(np.uint8)
              # Resize prediction using nearest neighbor interpolation.
              resized = cv2.resize(_pm, (targetW, targetH), interpolation=cv2.INTER_NEAREST)
              # Convert resized prediction back to integer labels.
              predMask = resized.astype(np.uint8)
            except Exception as e:
              # Record a failure when resize fails.
              failedImages.append({"image": origBaseName, "reason": f"shape mismatch and resize failed: {e}"})
              # Skip metric computation for this image.
              continue

          # Now call each metric function with (preds, targets). Wrap in try/except per metric.
          row = {"image": origBaseName}
          for mname, mfn in self.metricsFns.items():
            try:
              # Call the metric function with the prediction and target masks.
              val = mfn(predMask / 255.0, targetMask / 255.0)
              # If the metric returns a tensor, convert to a Python scalar.
              if (hasattr(val, "item")):
                val = val.item()
              # Ensure numeric values are valid; mark invalid values as None.
              if ((val is None) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val)))):
                row[mname] = None
              else:
                row[mname] = float(val)
            except Exception as e:
              # On metric failure, record None for that metric.
              row[mname] = None
              # Record the exception on first failure only for this image.
              if (not any(f.get("image") == origBaseName for f in failedImages)):
                failedImages.append({"image": origBaseName, "reason": f"metric {mname} failed: {e}"})

          # Append the per-image metric row to the list.
          perImageMetrics.append(row)

          # Attempt to create an overlay for this sample; wrap in try/except to avoid breaking the loop on failure.
          try:
            # Convert the input image tensor to a NumPy array for visualization.
            imgTensor = images[i].detach().cpu()
            # Try to obtain a NumPy representation of the tensor.
            try:
              imgNp = imgTensor.numpy()
            except Exception:
              imgNp = np.array(imgTensor)

            # Convert image to HWC and ensure 3 channels for RGB display.
            if (imgNp.ndim == 3):
              # Convert from [C, H, W] to [H, W, C].
              imgVis = np.transpose(imgNp, (1, 2, 0))
              # If single-channel, replicate to RGB.
              if (imgVis.shape[2] == 1):
                imgVis = np.repeat(imgVis, 3, axis=2)
            elif (imgNp.ndim == 2):
              # Convert grayscale [H, W] to RGB by replication.
              imgVis = np.stack([imgNp, imgNp, imgNp], axis=2)
            else:
              # Fallback: squeeze and replicate as needed.
              imgVis = np.squeeze(imgNp)
              if (imgVis.ndim == 2):
                imgVis = np.stack([imgVis, imgVis, imgVis], axis=2)

            # Normalize image to 0-255 uint8 for overlay composition.
            try:
              if ((imgVis.dtype == np.uint8) or (np.max(imgVis) > 1.0)):
                baseRgb = (imgVis.astype(np.float32) / 255.0)
              else:
                baseRgb = imgVis.astype(np.float32)
            except Exception:
              baseRgb = imgVis.astype(np.float32)
            # Clip to [0,1] and convert to uint8 0-255.
            baseRgb = np.clip(baseRgb, 0.0, 1.0)
            baseUint8 = (baseRgb * 255.0).astype(np.uint8)

            # Prepare binary masks for pred and target by thresholding non-zero values.
            try:
              predMaskBin = (predMask > 0).astype(np.uint8)
            except Exception:
              predMaskBin = (predMask != 0).astype(np.uint8)
            try:
              targetMaskBin = (targetMask > 0).astype(np.uint8)
            except Exception:
              targetMaskBin = (targetMask != 0).astype(np.uint8)

            # If mask shapes differ from image, resize masks to image shape using nearest neighbor.
            if (predMaskBin.shape != baseUint8.shape[:2]):
              try:
                predMaskBin = cv2.resize(
                  predMaskBin.astype(np.uint8),
                  (baseUint8.shape[1], baseUint8.shape[0]),
                  interpolation=cv2.INTER_NEAREST,
                )
              except Exception:
                predMaskBin = np.zeros(baseUint8.shape[:2], dtype=np.uint8)
            if (targetMaskBin.shape != baseUint8.shape[:2]):
              try:
                targetMaskBin = cv2.resize(
                  targetMaskBin.astype(np.uint8),
                  (baseUint8.shape[1], baseUint8.shape[0]),
                  interpolation=cv2.INTER_NEAREST
                )
              except Exception:
                targetMaskBin = np.zeros(baseUint8.shape[:2], dtype=np.uint8)

            # Create copies of the base image to draw overlays.
            predOverlay = baseUint8.copy()
            targetOverlay = baseUint8.copy()

            # Apply red tint where prediction mask is present.
            redColor = np.array([255, 0, 0], dtype=np.uint8)
            pmIdx = predMaskBin.astype(bool)
            predOverlay[pmIdx] = (
              0.65 * predOverlay[pmIdx].astype(np.float32) +
              0.35 * redColor.astype(np.float32)
            ).astype(np.uint8)

            # Apply green tint where target mask is present.
            greenColor = np.array([0, 255, 0], dtype=np.uint8)
            tmIdx = targetMaskBin.astype(bool)
            targetOverlay[tmIdx] = (
              0.65 * targetOverlay[tmIdx].astype(np.float32) +
              0.35 * greenColor.astype(np.float32)
            ).astype(np.uint8)

            # Concatenate original, predOverlay, and targetOverlay horizontally.
            # Create a single merged overlay image by tinting prediction and target pixels on the original.
            # Start from a copy of the base image to draw combined overlays.
            try:
              merged = baseUint8.copy()
              # Apply green tint for target mask locations with 50% blending.
              merged[tmIdx] = (
                0.65 * merged[tmIdx].astype(np.float32) +
                0.35 * greenColor.astype(np.float32)
              ).astype(np.uint8)
              # Apply red tint for prediction mask locations with 50% blending.
              merged[pmIdx] = (
                0.65 * merged[pmIdx].astype(np.float32) +
                0.35 * redColor.astype(np.float32)
              ).astype(np.uint8)
              # Use merged as the single output image.
              combined = merged
            except Exception:
              # Fallback to the three-panel concatenation if merging fails.
              try:
                combined = np.hstack([baseUint8, targetOverlay, predOverlay])
              except Exception:
                combined = np.vstack([baseUint8, targetOverlay, predOverlay])

            # Build an output filename for the overlay and write the image to disk.
            overlayName = f"{os.path.splitext(origBaseName)[0]}_Overlay_S{sampleIdx}.png"
            overlayPath = os.path.join(self.overlaysDir, overlayName)
            try:
              Image.fromarray(combined).save(overlayPath)
            except Exception:
              cv2.imwrite(overlayPath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
          except Exception:
            # Ignore overlay generation failures and continue the loop.
            pass

    # Build paths for primary and additional metrics files.
    metricsCsvPath = os.path.join(self.metricsDir, "PerImageMetrics.csv")
    metricsSummaryPath = os.path.join(self.metricsDir, "MetricsSummary.json")
    top75CsvPath = os.path.join(self.metricsDir, "Top75PercentMetrics.csv")
    top75SummaryPath = os.path.join(self.metricsDir, "Top75PercentSummary.json")
    worst10CsvPath = os.path.join(self.metricsDir, "Worst10PercentMetrics.csv")
    worst10SummaryPath = os.path.join(self.metricsDir, "Worst10PercentSummary.json")

    # Define helper to map raw metric values into a 0-1 higher-is-better score.
    def MetricToScore(metricName, value):
      # Return None when value is not available.
      if (value is None):
        return None
      # Handle metrics where higher is better and values are already in [0,1].
      highBetter = {
        "IoU", "Dice", "PixelAccuracy", "Precision",
        "Recall", "Specificity", "F1Score", "mAP",
        "BoundaryF1", "MCC", "CohensKappa"
      }
      # Handle metrics where lower is better.
      lowBetter = {"FPR", "FNR"}
      # Map Hausdorff (unbounded positive) into (0,1] with decreasing score for larger distances.
      if (metricName in highBetter):
        try:
          # Clip within 0-1 for safety and return.
          return max(0.0, min(1.0, float(value)))
        except Exception:
          return None
      if (metricName in lowBetter):
        try:
          # Convert lower-is-better to higher-is-better by inversion within [0,1].
          v = float(value)
          return max(0.0, min(1.0, 1.0 - v))
        except Exception:
          return None
      if (metricName == "Hausdorff"):
        try:
          v = float(value)
          return 1.0 / (1.0 + max(0.0, v))
        except Exception:
          return None
      try:
        # Fallback: try to coerce to a 0-1 clipped float.
        v = float(value)
        return max(0.0, min(1.0, v))
      except Exception:
        return None

    # Compute an AggregateScore per image using the MetricToScore mapping.
    for r in perImageMetrics:
      # Collect per-metric transformed scores.
      scores = []
      for mname in self.metricsFns.keys():
        s = MetricToScore(mname, r.get(mname))
        if (s is not None):
          scores.append(s)
      # Compute the mean of available scores, or None when none available.
      if (len(scores) > 0):
        r["AggregateScore"] = float(np.mean(scores))
      else:
        r["AggregateScore"] = None

    # Sort images by AggregateScore descending with None treated as -inf.
    def ScoreKey(row):
      sc = row.get("AggregateScore")
      if (sc is None):
        return -1.0
      return sc

    sortedRows = sorted(perImageMetrics, key=ScoreKey, reverse=True)
    # Determine counts for top 75% and worst 10% selections.
    nImages = len(sortedRows)
    top75Count = int(math.ceil(0.75 * nImages)) if (nImages > 0) else 0
    worst10Count = int(math.floor(0.10 * nImages)) if (nImages > 0) else 0
    if ((worst10Count == 0) and (nImages > 0)):
      # Ensure at least one image is reported when dataset is small.
      worst10Count = 1

    # Select the top 75% rows and the worst 10% rows.
    top75Rows = sortedRows[:top75Count]
    worst10Rows = sortedRows[-worst10Count:] if (worst10Count > 0) else []

    # Save the primary per-image CSV using CamelCase Image column.
    if (len(perImageMetrics) > 0):
      # CSV fieldnames include Image and metric names plus AggregateScore.
      fieldnames = ["Image"] + [k for k in self.metricsFns.keys()] + ["AggregateScore"]
      try:
        with open(metricsCsvPath, "w", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for r in perImageMetrics:
            # Map existing row keys to CamelCase Image key for output.
            outRow = {"Image": r.get("image")}
            for k in self.metricsFns.keys():
              outRow[k] = r.get(k)
            outRow["AggregateScore"] = r.get("AggregateScore")
            writer.writerow(outRow)
      except Exception as e:
        print(f"Failed to write metrics CSV: {e}")

    # Compute the overall summary statistics for all images.
    overallSummary = {"NImages": len(perImageMetrics), "NFailed": len(failedImages), "Mean": {}, "Std": {}}
    for mname in self.metricsFns.keys():
      vals = [r.get(mname) for r in perImageMetrics if (r.get(mname) is not None)]
      if (len(vals) > 0):
        arr = np.array(vals, dtype=float)
        # print(f"Metric {mname} - Mean: {np.nanmean(arr):.4f}, Std: {np.nanstd(arr):.4f}")
        # print(np.mean(arr), np.std(arr))
        overallSummary["Mean"][mname] = float(np.nanmean(arr))
        overallSummary["Std"][mname] = float(np.nanstd(arr))
      else:
        overallSummary["Mean"][mname] = None
        overallSummary["Std"][mname] = None
    overallSummary["FailedImages"] = failedImages

    # Write the overall summary JSON.
    DumpJsonFile(metricsSummaryPath, overallSummary)

    # Helper to write a subset CSV and compute its aggregated metrics.
    def WriteSubsetFiles(rows, csvPath, summaryPath, subsetName):
      # Write CSV for selected rows.
      try:
        with open(csvPath, "w", newline="") as csvfile:
          fieldnames = ["Image"] + [k for k in self.metricsFns.keys()] + ["AggregateScore"]
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for r in rows:
            outRow = {"Image": r.get("image")}
            for k in self.metricsFns.keys():
              outRow[k] = r.get(k)
            outRow["AggregateScore"] = r.get("AggregateScore")
            writer.writerow(outRow)
      except Exception as e:
        print(f"Failed to write {subsetName} CSV: {e}")

      # Compute summary averages for this subset.
      subsetSummary = {"NImages": len(rows), "Mean": {}, "Std": {}}
      for mname in self.metricsFns.keys():
        vals = [r.get(mname) for r in rows if (r.get(mname) is not None)]
        if (len(vals) > 0):
          arr = np.array(vals, dtype=float)
          subsetSummary["Mean"][mname] = float(np.nanmean(arr))
          subsetSummary["Std"][mname] = float(np.nanstd(arr))
        else:
          subsetSummary["Mean"][mname] = None
          subsetSummary["Std"][mname] = None
      # Include list of images in the subset for traceability.
      subsetSummary["Images"] = [r.get("image") for r in rows]

      # Write JSON summary for the subset.
      DumpJsonFile(summaryPath, subsetSummary)

    # Write top 75% files when available.
    if (len(top75Rows) > 0):
      WriteSubsetFiles(top75Rows, top75CsvPath, top75SummaryPath, "Top75Percent")

    # Write worst 10% files when available.
    if (len(worst10Rows) > 0):
      WriteSubsetFiles(worst10Rows, worst10CsvPath, worst10SummaryPath, "Worst10Percent")

    # Print final status messages about saved files.
    print(f"Saved {globalIdx} predicted masks to: {self.predsDir}.")
    print(f"Per-image metrics: {metricsCsvPath}")
    print(f"Metrics summary: {metricsSummaryPath}")
    print(f"Top 75% metrics: {top75CsvPath} and {top75SummaryPath}")
    print(f"Worst 10% metrics: {worst10CsvPath} and {worst10SummaryPath}")


if (__name__ == "__main__"):
  # Run comprehensive tests for classification and segmentation training.
  print("\n" + "=" * 60)
  print("PyTorchHelper Training Test Suite")
  print("=" * 60)

  # ========================================================================
  # CLASSIFICATION TRAINING TEST
  # ========================================================================
  print("\n" + "-" * 60)
  print("Testing Classification Training...")
  print("-" * 60)

  try:
    # Create dummy classification data.
    numSamples = 64
    inputSize = 100
    numClasses = 10
    batchSize = 16

    # Generate random features and labels.
    X = torch.randn(numSamples, inputSize)
    y = torch.randint(0, numClasses, (numSamples,))


    # Create a custom dataset that includes imagePaths for Inference().
    class DummySegmentationDataset(torch.utils.data.Dataset):
      def __init__(self, X, y):
        self.X = X
        self.y = y
        # Create dummy image paths for Inference() compatibility.
        self.imagePaths = [f"dummy_image_{i}.png" for i in range(len(X))]

      def __len__(self):
        return len(self.X)

      def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    # Create dataset and data loaders.
    dataset = DummySegmentationDataset(X, y)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
    valLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
    allLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)


    # Create a simple MLP model for testing.
    class SimpleClassifier(nn.Module):
      def __init__(self, inputSize, numClasses):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
          nn.Linear(inputSize, 64),
          nn.ReLU(),
          nn.Linear(64, numClasses)
        )

      def forward(self, x):
        return self.net(x)


    model = SimpleClassifier(inputSize, numClasses)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device before training.
    model = model.to(device)

    # Train and evaluate the model.
    # Use a path with a directory component to ensure CheckpointSaver receives a valid savePath.
    bestModelPath = os.path.join("TestClassificationOutput", "TestBestModel.pth")
    os.makedirs(os.path.dirname(bestModelPath), exist_ok=True)

    # Train and evaluate the model.
    history = TrainEvaluateClassificationModel(
      model=model,
      criterion=criterion,
      device=device,
      bestModelStoragePath=bestModelPath,
      noOfClasses=numClasses,
      numEpochs=10,  # Short test run.
      optimizer=optimizer,
      scaler=scaler,
      scheduler=scheduler,
      trainLoader=trainLoader,
      valLoader=valLoader,
      resumeFromCheckpoint=False,
      finalModelStoragePath="TestFinalModel.pth",
      judgeBy="val_loss",
      earlyStoppingPatience=2,
      verbose=True,
      gradAccumSteps=1,
      maxGradNorm=None,
      useAmp=torch.cuda.is_available(),
      useMixupFn=False,
      mixUpAlpha=0.5,
      useEma=False,
      saveEvery=None
    )

    # Verify training completed successfully.
    assert len(history["train_loss"]) == 10, "Training history length mismatch"
    assert len(history["val_loss"]) == 10, "Validation history length mismatch"
    print("✅ Classification training test passed.")

    # Clean up test files.
    import shutil

    if (os.path.exists("TestClassificationOutput")):
      shutil.rmtree("TestClassificationOutput")

  except Exception as e:
    print(f"❌ Classification training test failed: {e}")

  # ========================================================================
  # CLASSIFICATION CLASS TRAINING TEST
  # ========================================================================
  print("\n" + "-" * 60)
  print("Testing Classification Class Training...")
  print("-" * 60)

  try:
    # Create dummy classification data.
    numSamples = 64
    inputSize = 100
    numClasses = 10
    batchSize = 16

    # Generate random features and labels.
    X = torch.randn(numSamples, inputSize)
    y = torch.randint(0, numClasses, (numSamples,))

    # Create dataset and data loaders.
    dataset = torch.utils.data.TensorDataset(X, y)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
    valLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
    allLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)


    # Create a simple MLP model for testing.
    class SimpleClassifier(nn.Module):
      def __init__(self, inputSize, numClasses):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
          nn.Linear(inputSize, 64),
          nn.ReLU(),
          nn.Linear(64, numClasses)
        )

      def forward(self, x):
        return self.net(x)


    model = SimpleClassifier(inputSize, numClasses)
    lossFn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device before training.
    model = model.to(device)

    # Initialize the classification training pipeline.
    pipeline = PyTorchClassificationTrainingPipeline(
      model=model,
      trainLoader=trainLoader,
      valLoader=valLoader,
      allLoader=allLoader,
      optimizer=optimizer,
      scheduler=scheduler,
      lossFn=lossFn,
      noOfClasses=numClasses,
      learningRate=1e-3,
      device=device,
      outputDir="TestClassificationPipelineOutput",
      dpi=72,  # Lower DPI for faster testing.
      logEveryNSteps=5,
      useAmp=torch.cuda.is_available(),
      earlyStoppingPatience=3,
      judgeBy="val_loss",
      verbose=True,
    )

    # Run training for a few epochs.
    pipeline.Train(numEpochs=10)

    # Run inference to verify prediction saving works.
    pipeline.Inference()

    # Verify output directories were created.
    assert os.path.exists("TestClassificationPipelineOutput/Checkpoints"), "Checkpoint directory not created"
    assert os.path.exists("TestClassificationPipelineOutput/Preds"), "Predictions directory not created"
    assert os.path.exists("TestClassificationPipelineOutput/Metrics"), "Metrics directory not created"
    print("✅ Classification class training test passed.")

    # Clean up test files.
    import shutil

    if (os.path.exists("TestClassificationPipelineOutput")):
      shutil.rmtree("TestClassificationPipelineOutput")

  except Exception as e:
    print(f"❌ Classification class training test failed: {e}")

  # ========================================================================
  # SEGMENTATION CLASS TRAINING TEST
  # ========================================================================
  print("\n" + "-" * 60)
  print("Testing Segmentation Training...")
  print("-" * 60)

  try:
    # Create dummy segmentation data.
    numSamples = 32
    inChannels = 3
    outChannels = 1
    imageSize = 64
    batchSize = 8

    # Generate random images and masks.
    X = torch.randn(numSamples, inChannels, imageSize, imageSize)
    y = torch.randint(0, 2, (numSamples, outChannels, imageSize, imageSize)).float()

    # Create dataset and data loaders.
    dataset = torch.utils.data.TensorDataset(X, y)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
    valLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
    allLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)


    # Create a simple U-Net-like model for testing (no upsampling to avoid size mismatch).
    class SimpleUNet(nn.Module):
      def __init__(self, inChannels, outChannels):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
          nn.Conv2d(inChannels, 16, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size=3, padding=1),
          nn.ReLU(),
        )
        # Use Conv2d instead of ConvTranspose2d to preserve spatial dimensions.
        self.decoder = nn.Sequential(
          nn.Conv2d(32, 16, kernel_size=3, padding=1),  # No upsampling
          nn.ReLU(),
          nn.Conv2d(16, outChannels, kernel_size=1),
        )

      def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    model = SimpleUNet(inChannels, outChannels)
    lossFn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the segmentation training pipeline.
    pipeline = PyTorchUNetSegmentationTrainingPipeline(
      model=model,
      trainLoader=trainLoader,
      valLoader=valLoader,
      allLoader=allLoader,
      optimizer=optimizer,
      scheduler=scheduler,
      lossFn=lossFn,
      learningRate=1e-3,
      device=device,
      outputDir="TestSegmentationOutput",
      dpi=72,  # Lower DPI for faster testing.
      logEveryNSteps=5,
    )

    # Run training for a few epochs.
    pipeline.Train(numEpochs=10)

    # Run inference to verify prediction saving works.
    pipeline.Inference()

    # Verify output directories were created.
    assert os.path.exists("TestSegmentationOutput/Checkpoints"), "Checkpoint directory not created"
    assert os.path.exists("TestSegmentationOutput/Preds"), "Predictions directory not created"
    print("✅ Segmentation training test passed.")

    # Clean up test files.
    import shutil

    if (os.path.exists("TestSegmentationOutput")):
      shutil.rmtree("TestSegmentationOutput")

  except Exception as e:
    print(f"❌ Segmentation training test failed: {e}")

  # ========================================================================
  # CALLBACK INTEGRATION TEST
  # ========================================================================
  print("\n" + "-" * 60)
  print("Testing Callback Integration...")
  print("-" * 60)

  try:
    # Test EnableMixedPrecision.
    model = SimpleClassifier(100, 10)
    scaler = EnableMixedPrecision(model)
    assert scaler is not None, "EnableMixedPrecision returned None"
    print("✅ EnableMixedPrecision test passed.")

    # Test EarlyStopping.
    earlyStopping = EarlyStopping(patience=3, mode="min", verbose=False)
    # Simulate improving metric.
    assert not earlyStopping(0.5), "EarlyStopping should not stop on first call"
    assert not earlyStopping(0.3), "EarlyStopping should not stop on improvement"
    # Simulate non-improving metrics.
    for _ in range(3):
      earlyStopping(0.4)
    assert earlyStopping.earlyStop, "EarlyStopping should trigger after patience"
    print("✅ EarlyStopping test passed.")

    # Test CheckpointSaver.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpDir:
      saver = CheckpointSaver(savePath=tmpDir, monitor="val_loss", mode="min", verbose=False)
      filepath = saver(model=model, currentMetric=0.5, epoch=1)
      assert filepath is not None, "CheckpointSaver returned None"
      assert os.path.exists(filepath), "Checkpoint file not saved"
      print("✅ CheckpointSaver test passed.")

  except Exception as e:
    print(f"❌ Callback integration test failed: {e}")

  # ========================================================================
  # FINAL SUMMARY
  # ========================================================================
  print("\n" + "=" * 60)
  print("Test Suite Complete")
  print("=" * 60)
  print("All tests executed. Review output above for pass/fail status.")
  print("Note: Test files were cleaned up after successful runs.")
