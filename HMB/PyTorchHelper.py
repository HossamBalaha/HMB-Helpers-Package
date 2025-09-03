import os, timm, torch, tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix
from HMB.PerformanceMetrics import CalculatePerformanceMetrics


# Function to save a PyTorch model's state dictionary to a file.
def SaveModel(model, filename="model.pth"):
  '''
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


# Function to load a PyTorch model's state dictionary from a file and move it to a device.
def LoadModel(model, filename="model.pth", device="gpu"):
  '''
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


def SaveCheckpoint(model, optimizer, filename="chk.pth.tar"):
  '''
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
  '''
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


class CustomDataset(torch.utils.data.Dataset):
  '''
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
    '''
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
    '''
    Get the total number of samples in the dataset.

    Returns:
      int: Number of samples in the dataset.
    '''

    return len(self.samples)

  def __getitem__(self, idx):
    '''
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
  '''
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


def TrainEvaluateModel(
  model,  # Model to train and evaluate.
  criterion,  # Loss function.
  device,  # Device to run training and evaluation on (CPU or GPU).
  modelStoragePath,  # Path to save the best model.
  noOfClasses,  # Number of classes in the classification task.
  numEpochs,  # Total number of epochs for training.
  optimizer,  # Optimizer for updating model parameters.
  scaler,  # Gradient scaler for mixed precision training.
  scheduler,  # Learning rate scheduler.
  trainLoader,  # DataLoader for training data.
  valLoader,  # DataLoader for validation data.
  verbose=True,  # Verbosity flag to control logging.
):
  '''
  Train and evaluate a classification model for a specified number of epochs.

  Parameters:
    model (torch.nn.Module): Model to train and evaluate.
    criterion (callable): Loss function.
    device (torch.device): Device to run training and evaluation on (CPU or GPU).
    modelStoragePath (str): Path to save the best model.
    noOfClasses (int): Number of classes in the classification task.
    numEpochs (int): Total number of epochs for training.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
    scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    trainLoader (torch.utils.data.DataLoader): DataLoader for training data.
    valLoader (torch.utils.data.DataLoader): DataLoader for validation data.
    verbose (bool, optional): Verbosity flag to control logging. Defaults to True.

  Returns:
    dict: History dictionary containing training and validation metrics.
  '''

  # Initialize history dictionary to store training and validation metrics.
  history = {
    "train_accuracy": [],
    "val_accuracy"  : [],
    "train_loss"    : [],
    "val_loss"      : []
  }

  # Variables to track the best validation loss and accuracy.
  bestValLoss = float("inf")
  bestValAccuracy = 0.0

  # Training loop for the specified number of epochs.
  for epoch in range(numEpochs):
    if (verbose):
      print(f"Starting epoch {epoch + 1}/{numEpochs}")

    # Train for one epoch.
    avgTrainEpochLoss, avgTrainEpochTrain = TrainOneEpoch(
      model, trainLoader, criterion, device, epoch,
      noOfClasses, numEpochs, optimizer, scaler
    )

    avgValEpochLoss, avgValEpochAccuracy = EvaluateOneEpoch(
      model, valLoader, criterion, device, noOfClasses
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
    if ((avgValEpochLoss < bestValLoss) or (avgValEpochAccuracy > bestValAccuracy)):
      bestValLoss = avgValEpochLoss
      bestValAccuracy = avgValEpochAccuracy
      SaveModel(model, modelStoragePath)
      if (verbose):
        print(
          f"Saved new best model with val loss: {bestValLoss:.4f} "
          f"and val accuracy: {bestValAccuracy:.4f}"
        )

    # Update learning rate scheduler.
    scheduler.step()
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
):
  '''
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
    scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.

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

  # Iterate over the training data loader with a progress bar.
  for batchIdx, batch in tqdm.tqdm(
    enumerate(dataLoader),  # Enumerate over batches.
    total=len(dataLoader),  # Total number of batches.
    desc=f"Epoch {epoch + 1}/{numEpochs}",  # Description for the progress bar.
  ):
    # Get data and labels from the batch and move them to the specified device.
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    # Zero the gradients of the optimizer.
    optimizer.zero_grad()

    # Forward pass through the model to get outputs.
    outputs = model(data)

    # Get the predicted class indices.
    outputIdx = outputs.argmax(dim=1)

    # Compute the loss using the specified criterion.
    loss = criterion(outputs, labels)
    # If loss is a tensor, convert it to a scalar value.
    loss = loss.item() if isinstance(loss, torch.Tensor) else loss
    # Accumulate the total loss for the epoch.
    totalEpochLoss += loss

    # Compute the confusion matrix and accuracy.
    cm = confusion_matrix(
      labels.cpu(),  # True labels.
      outputIdx.cpu(),  # Predicted labels.
      labels=list(range(noOfClasses)),  # List of class labels.
    )
    # Calculate performance metrics from the confusion matrix.
    metrics = CalculatePerformanceMetrics(cm)
    accuracy = metrics["Weighted Accuracy"]
    # Accumulate the total accuracy for the epoch.
    totalEpochAccuracy += accuracy

    # Backward pass and optimization step with mixed precision.
    scaler.scale(loss).backward()
    # Make an optimization step using the scaled gradients.
    scaler.step(optimizer)
    # Update the gradient scaler.
    scaler.update()

  # Calculate average loss and accuracy for the epoch.
  avgTrainLoss = totalEpochLoss / len(dataLoader)
  avgTrainAccuracy = totalEpochAccuracy / len(dataLoader)

  return avgTrainLoss, avgTrainAccuracy


def EvaluateOneEpoch(
  model,  # Model to evaluate.
  dataLoader,  # DataLoader for evaluation data.
  criterion,  # Loss function.
  device,  # Device to run evaluation on (CPU or GPU).
  noOfClasses,  # Number of classes in the classification task.
):
  '''
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
      metrics = CalculatePerformanceMetrics(cm)
      accuracy = metrics["Weighted Accuracy"]

      # Accumulate the total accuracy for the validation epoch.
      totalAccuracy += accuracy

  # Calculate average loss and accuracy for the validation epoch.
  avgValLoss = totalLoss / len(dataLoader)
  avgValAccuracy = totalAccuracy / len(dataLoader)

  return avgValLoss, avgValAccuracy
