from HMB.Initializations import CheckInstalledModules

if __name__ == "__main__":
  CheckInstalledModules(["torch", "numpy", "matplotlib", "PIL", "tqdm", "tensorboard", "sklearn"])

# ------------------------------------------------------------------------- #

import os, torch, json, cv2, math, csv, argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from HMB.PyTorchHelper import GetOptimizer, LoadCheckpoint, SaveCheckpoint
from HMB import ImageSegmentationMetrics as ISM
from HMB.DatasetsHelper import CreateSegmentationDataLoaders
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding
from HMB.UNetHelper import PreparePredTensorToNumpy, GetUNetModel

# Define default hyperparameters dictionary with CamelCase keys.
# This dictionary uses camelCase variable name to follow project conventions.
defaultHparams = {
  # Model name to use.
  "ModelName"       : "ResidualAttentionUNet",
  # Data directory containing images and masks.
  "DataDir"         : "Data",
  # Output directory for logs and checkpoints.
  "OutputDir"       : "OutputV2",
  # Phase to run: Train or Infer.
  "Phase"           : "Train",
  # Number of training epochs.
  "NumEpochs"       : 50,
  # Batch size for training.
  "BatchSize"       : 16,
  # Image size (square) used for training and validation.
  "ImageSize"       : 128,
  # Learning rate for optimizer.
  "LearningRate"    : 1e-4,
  # Weight decay for optimizer.
  "WeightDecay"     : 1e-6,
  # Optimizer name.
  "Optimizer"       : "adamw",
  # Scheduler name.
  "Scheduler"       : "ReduceLROnPlateau",
  # Number of data loader worker processes.
  "NumWorkers"      : 1,
  # Random seed for reproducibility.
  "Seed"            : np.random.randint(0, 10000),
  # Number of segmentation classes.
  "NumClasses"      : 1,
  # Device to run on.
  "Device"          : "cuda",
  # Resume from checkpoint path if any.
  "ResumeCheckpoint": "",
  # Whether to use automatic mixed precision.
  "UseAMP"          : False,
  # Whether to log to Weights & Biases.
  "UseWandB"        : False,
  # Maximum number of checkpoints to keep.
  "MaxCheckpoints"  : 5,
}


# Parse command-line arguments and return the hparams dictionary.
def ParseArgs():
  # Create an argument parser with a description.
  parser = argparse.ArgumentParser(description="Training and evaluation CLI for segmentation.")
  # Add arguments corresponding to keys in defaultHparams.
  parser.add_argument(
    "--ModelName",
    type=str,
    default=defaultHparams["ModelName"],
    help="Model name to use."
  )
  parser.add_argument(
    "--DataDir",
    type=str,
    default=defaultHparams["DataDir"],
    help="Data directory path."
  )
  parser.add_argument(
    "--OutputDir",
    type=str,
    default=defaultHparams["OutputDir"],
    help="Output directory for logs and checkpoints."
  )
  parser.add_argument(
    "--Phase",
    type=str,
    default=defaultHparams["Phase"],
    choices=["Train", "Infer"],
    help="Phase to run: Train or Infer."
  )
  parser.add_argument(
    "--NumEpochs",
    type=int,
    default=defaultHparams["NumEpochs"],
    help="Number of training epochs."
  )
  parser.add_argument(
    "--BatchSize",
    type=int,
    default=defaultHparams["BatchSize"],
    help="Batch size for training."
  )
  parser.add_argument(
    "--ImageSize",
    type=int,
    default=defaultHparams["ImageSize"],
    help="Square image size for training and validation."
  )
  parser.add_argument(
    "--LearningRate",
    type=float,
    default=defaultHparams["LearningRate"],
    help="Learning rate."
  )
  parser.add_argument(
    "--WeightDecay",
    type=float,
    default=defaultHparams["WeightDecay"],
    help="Weight decay."
  )
  parser.add_argument(
    "--Device",
    type=str,
    default=defaultHparams["Device"],
    help="Device to train on."
  )
  parser.add_argument(
    "--NumWorkers",
    type=int,
    default=defaultHparams["NumWorkers"],
    help="Number of data loader workers."
  )
  parser.add_argument(
    "--NumClasses",
    type=int,
    default=defaultHparams["NumClasses"],
    help="Number of segmentation classes."
  )
  parser.add_argument(
    "--ResumeCheckpoint", type=str, default=defaultHparams["ResumeCheckpoint"],
    help="Path to checkpoint to resume from."
  )
  parser.add_argument("--UseAMP", action="store_true", help="Use automatic mixed precision for training.")
  # Parse the known args from the command line.
  args = parser.parse_args()
  # Merge parsed args into a CamelCase hparams dict.
  hparams = {
    "ModelName"       : args.ModelName,
    "DataDir"         : args.DataDir,
    "OutputDir"       : args.OutputDir,
    "Phase"           : args.Phase,
    "NumEpochs"       : args.NumEpochs,
    "BatchSize"       : args.BatchSize,
    "ImageSize"       : args.ImageSize,
    "LearningRate"    : args.LearningRate,
    "WeightDecay"     : args.WeightDecay,
    "Device"          : args.Device,
    "NumWorkers"      : args.NumWorkers,
    "NumClasses"      : args.NumClasses,
    "ResumeCheckpoint": args.ResumeCheckpoint,
    "UseAMP"          : args.UseAMP,
  }
  # Ensure output directory exists and return the hparams dictionary.
  os.makedirs(hparams.get("OutputDir", "Output"), exist_ok=True)

  # Print parsed hyperparameters for verification.
  print("Parsed Hyperparameters:")
  for key, value in hparams.items():
    print(f"  {key}: {value}")

  return hparams


# Save hparams dict to a JSON file in the specified output directory.
def SaveHparams(hparams: dict, outputDir: str, outputFileName: str = "ExpConfig.json"):
  # Ensure the output directory exists.
  os.makedirs(outputDir, exist_ok=True)
  # Compose the full path for the config file.
  filePath = os.path.join(outputDir, outputFileName)
  # Write the hparams dictionary to disk as JSON.
  with open(filePath, "w", encoding="utf-8") as file:
    json.dump(hparams, file, indent=2)


class Trainer:
  # Initialize the trainer with model, dataloaders, optimizer, scheduler, and hparams.
  def __init__(
      self,
      model: nn.Module,
      trainLoader,
      valLoader,
      optimizer: optim.Optimizer,
      scheduler,
      lossFn,
      hparams: Dict,
      device: str = "cuda"
  ):
    # Store references to the model and device.
    self.model = model
    self.device = device
    # Store dataloaders.
    self.trainLoader = trainLoader
    self.valLoader = valLoader
    # Store optimizer, scheduler, and loss function.
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.lossFn = lossFn
    # Store the hyperparameters' dictionary.
    self.hparams = hparams
    # Initialize TensorBoard writer in the output directory.
    self.writer = SummaryWriter(
      log_dir=os.path.join(hparams.get("OutputDir", "Output"), "Logs"),
    )
    # Prepare the model on the target device.
    self.model.to(self.device)
    # Initialize the best metric for checkpointing.
    self.bestMetric = -1.0
    # Prepare checkpoint directory.
    self.checkpointDir = os.path.join(hparams.get("OutputDir", "Output"), "Checkpoints")
    os.makedirs(self.checkpointDir, exist_ok=True)

  # Save a checkpoint to disk with a given tag.
  def SaveCheckpoint(self, epoch: int, tag: str = "latest"):
    filePath = os.path.join(self.checkpointDir, f"Checkpoint{tag.lower().capitalize()}.pth")
    SaveCheckpoint(self.model, self.optimizer, filePath, epoch=epoch, hparams=self.hparams)
    return filePath

  # Load checkpoint from disk and restore model and optimizer states.
  def LoadCheckpoint(self, filePath: str, strict: bool = True) -> int:
    checkpoint = LoadCheckpoint(
      filePath,
      self.model,
      self.optimizer,
      lr=self.hparams.get("LearningRate", 1e-4),
      device=self.device,
      strict=strict
    )
    epoch = checkpoint.get("epoch", 0) + 1
    return epoch

  # Run the full training loop for a given number of epochs.
  def Train(self, numEpochs: int):
    # Loop over epochs from 1 to numEpochs inclusive.
    for epoch in range(1, numEpochs + 1):
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
        # If scheduler is ReduceLROnPlateau, step with validation loss.
        if ("ReduceLROnPlateau" in type(self.scheduler).__name__):
          self.scheduler.step(valMetrics.get("Loss", 0.0))
        else:
          self.scheduler.step()
      # Save the latest checkpoint.
      self.SaveCheckpoint(epoch, tag="latest")
      # If validation mean dice improved, save the best checkpoint.
      if (valMetrics.get("MeanDice", 0.0) > self.bestMetric):
        # Update the best metric and save the best checkpoint.
        self.bestMetric = valMetrics.get("MeanDice", 0.0)
        self.SaveCheckpoint(epoch, tag="best")
      print(
        f"Epoch {epoch}/{numEpochs} - Train Loss: {trainLoss:.4f} - "
        f"Val Loss: {valMetrics.get('Loss', 0.0):.4f} - "
        f"Val Dice: {valMetrics.get('MeanDice', 0.0):.4f} - "
        f"Val IoU: {valMetrics.get('MeanIoU', 0.0):.4f} - "
        f"Val Pixel Acc: {valMetrics.get('PixelAccuracy', 0.0):.4f}"
      )

  # Run a single training epoch and return average loss.
  def TrainEpoch(self, epoch: int) -> float:
    # Set model to training mode.
    self.model.train()
    # Initialize running loss and sample count.
    runningLoss = 0.0
    count = 0
    # Iterate over batches from the train loader.
    for batchIdx, (images, masks) in tqdm(
        enumerate(self.trainLoader),
        total=len(self.trainLoader),
        desc=f"Training Epoch {epoch}"
    ):
      # Move images and masks to the configured device.
      images = images.to(self.device)
      masks = masks.to(self.device)
      # Zero gradients on optimizer.
      self.optimizer.zero_grad()
      # Forward pass to obtain logits.
      logits = self.model(images)
      # Compute loss using the provided loss function.
      loss = self.lossFn(logits, masks)
      # Backpropagate gradients.
      loss.backward()
      # Step optimizer to update parameters.
      self.optimizer.step()
      # Accumulate loss and increment count.
      runningLoss += loss.item()
      count += 1
      # Optionally log batch-level loss to TensorBoard.
      if ((batchIdx + 1) % 10 == 0):
        self.writer.add_scalar("Loss/TrainBatch", runningLoss / float(count), epoch * 1000 + batchIdx)
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
            # Compute the output directory path for saving combined PNG files.
            outputDir = os.path.join(self.hparams.get("OutputDir", "Output"), "TrainSamples")
            # Ensure the output directory exists on disk.
            os.makedirs(outputDir, exist_ok=True)

            # Compose the filename for the saved combined image.
            combinedPath = os.path.join(outputDir, f"Epoch{epoch}_Batch{batchIdx}_Sample{i}.png")

            # Prepare image, mask and prediction numpy arrays for plotting.
            # images: [B, C, H, W] or [B, H, W].
            imgTensor = images[i].detach().cpu()
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
            maskTensor = masks[i].detach().cpu()
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
              predTensor = preds[i].detach().cpu()
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
              # Build a filename for the probability map image.
              probPath = os.path.join(outputDir, f"Epoch{epoch}_Batch{batchIdx}_Sample{i}_Prob.png")
              # Create a small figure and save the probability heatmap.
              figProb, axProb = plt.subplots(1, 1, figsize=(4, 4))
              # Display the probability map with a perceptually-uniform colormap.
              axProb.imshow(probNp, cmap="viridis", vmin=0.0, vmax=1.0)
              # Turn off axis decorations.
              axProb.axis("off")
              # Save the probability map to disk.
              figProb.savefig(probPath, dpi=720, bbox_inches="tight")
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
            fig.savefig(combinedPath, dpi=150)
            # Close the figure to release memory resources.
            plt.close(fig)
    # Return average training loss for the epoch.
    return runningLoss / float(max(1, count))

  # Run validation over the validation set and return metrics dictionary.
  def Validate(self, epoch: int) -> Dict:
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
          diceScores.append(ISM.ComputeDice(pred, tgt))
          iouScores.append(ISM.ComputeIoU(pred, tgt))
          pixelAccs.append(ISM.ComputePixelAccuracy(pred, tgt))
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


# Define the main Run function which executes training or inference phases.
def Run():
  # Parse command-line arguments into a hyperparameters dictionary.
  hparams = ParseArgs()
  # Create predictions output directory inside the output directory.
  predsDir = os.path.join(hparams.get("OutputDir", "Output"), "Preds")
  actualDir = os.path.join(hparams.get("OutputDir", "Output"), "Actuals")
  # Ensure the predictions directory exists.
  os.makedirs(predsDir, exist_ok=True)
  os.makedirs(actualDir, exist_ok=True)

  # Decide which device to use for inference based on availability and hparams.
  if (torch.cuda.is_available() and (hparams.get("Device", "cuda").startswith("cuda"))):
    # Select the requested CUDA device.
    device = torch.device(hparams.get("Device", "cuda"))
  else:
    # Fall back to CPU device when CUDA is not available or not requested.
    device = torch.device("cpu")
  # Print which device will be used.
  print(f"Using device: {device}")

  # Create or obtain training, validation and combined loaders as before.
  trainLoader, valLoader, allLoader = CreateSegmentationDataLoaders(
    hparams.get("DataDir", "Data"),
    imageSize=hparams.get("ImageSize", 256),
    batchSize=hparams.get("BatchSize", 8) if (hparams.get("Phase", "Train") == "Train") else 1,
    numWorkers=hparams.get("NumWorkers", 4),
    numClasses=hparams.get("NumClasses", 1)
  )
  # Print counts for sanity.
  print(f"Training samples: {len(trainLoader.dataset)}, Validation samples: {len(valLoader.dataset)}")

  # Validate created dataloaders.
  if ((trainLoader is None) or (valLoader is None)):
    # Raise error when loaders failed to create.
    raise RuntimeError("Failed to create dataloaders for inference.")
  elif (len(valLoader.dataset) == 0 or len(trainLoader.dataset) == 0):
    # Raise error when datasets are empty.
    raise RuntimeError("No samples found for inference in the specified DataDir.")

  # Get the min, max, and mean pixel values from the training dataset.
  # pixelStats = trainLoader.dataset.GetPixelStats()
  # for statName, statValue in pixelStats.items():
  #   print(f"{statName}: {statValue}")

  # Instantiate the requested model via the factory.
  model = GetUNetModel(
    hparams.get("ModelName", "UNet"),
    inputChannels=3,
    numClasses=hparams.get("NumClasses", 1)
  )
  # Print which model was instantiated.
  print(f"Model {hparams.get('ModelName', 'UNet')} instantiated.")
  # Move the model to the selected device.
  model.to(device)

  # If the requested phase is Train, run training and return.
  if (hparams.get("Phase", "Infer") == "Train"):
    # Save hyperparameters to the output directory.
    SaveHparams(hparams, hparams.get("OutputDir", "Output"))

    # Create optimizer using helper function.
    optimizer = GetOptimizer(
      model,
      optimizerType=hparams.get("Optimizer", "adamw"),
      learningRate=hparams.get("LearningRate", 1e-4),
      weightDecay=hparams.get("WeightDecay", 1e-6)
    )
    # Print which optimizer was created.
    print(f"Optimizer {hparams.get('Optimizer', 'Adam')} created.")

    # Create a scheduler; use ReduceLROnPlateau by default.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    # Print scheduler creation message.
    print("Scheduler ReduceLROnPlateau created.")

    # Select an appropriate loss function depending on number of classes.
    if (hparams.get("NumClasses", 1) == 1):
      # Use binary loss for single-class segmentation.
      lossFn = nn.BCEWithLogitsLoss()
    else:
      # Use cross-entropy for multi-class segmentation.
      lossFn = nn.CrossEntropyLoss()
    # Print loss function creation message.
    print("Loss function created.")

    # Create Trainer instance to run training.
    trainer = Trainer(
      model,
      trainLoader,
      valLoader,
      optimizer,
      scheduler,
      lossFn,
      hparams,
      device=device,
    )
    # Print training start message.
    print("Starting training...")

    # If resume checkpoint is provided, load it and resume.
    if (hparams.get("ResumeCheckpoint", "") != ""):
      # Load checkpoint into trainer if requested.
      trainer.LoadCheckpoint(hparams.get("ResumeCheckpoint"))
      # Print checkpoint resume message.
      print("Resumed from checkpoint:", hparams.get("ResumeCheckpoint"))

    # Run training for the specified number of epochs.
    trainer.Train(hparams.get("NumEpochs", 50))
    # Print training completion message.
    print("Training completed.")
    # Return after training phase completes.
    return

  else:
    # If a checkpoint path was provided, load the model weights.
    if (hparams.get("ResumeCheckpoint", "") != ""):
      # Load the checkpoint onto the selected device and print keys for debugging.
      ckpt = torch.load(hparams.get("ResumeCheckpoint"), map_location=device)
      # Extract state dict from common key names.
      if (isinstance(ckpt, dict)):
        # Use nested key "model_state" when present.
        if ("model_state" in ckpt):
          stateDict = ckpt["model_state"]
        # Use nested key "state_dict" when present.
        elif ("state_dict" in ckpt):
          stateDict = ckpt["state_dict"]
        else:
          # Assume the dict itself is the state dict.
          stateDict = ckpt
      else:
        # If checkpoint is not a dict, try to use it directly.
        stateDict = ckpt
      # Load the state dictionary into the model.
      model.load_state_dict(stateDict)

    # Set the model to evaluation mode to disable training-specific layers.
    model.eval()
    # Initialize a global counter for saved prediction files.
    globalIdx = 0

    # Prepare collectors for metrics computation.
    metricsDir = os.path.join(hparams.get("OutputDir", "Output"), "Metrics")
    # Ensure the metrics directory exists.
    os.makedirs(metricsDir, exist_ok=True)

    # Map metric display names to functions in HMB.ImageSegmentationMetrics.
    metricsFns = {
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

    # Initialize a list to store per-image metrics.
    perImageMetrics = []
    # Initialize a list to store any failures encountered.
    failedImages = []
    # Disable gradient computation for inference to save memory and compute.
    with torch.inference_mode():
      # Iterate over the validation dataloader batches.
      for batchIdx, (images, masks) in enumerate(allLoader):
        # Move images to the selected device.
        images = images.to(device)
        # Forward pass through the model to obtain logits.
        logits = model(images)

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

        # Iterate over items in the batch and save each predicted mask and compute metrics.
        for i in range(preds.shape[0]):
          # Convert the predicted mask and target mask tensors to numpy arrays for saving and metric computation.
          predMask = PreparePredTensorToNumpy(preds[i], doScale2Image=True)
          targetMask = PreparePredTensorToNumpy(masks[i], doScale2Image=True)

          # Compute the absolute sample index within the dataset.
          sampleIdx = batchIdx * (
            allLoader.batch_size
            if ((hasattr(allLoader, "batch_size") and allLoader.batch_size is not None)) else 1
          )
          # Add the intra-batch index to obtain the final sample index.
          sampleIdx += i
          origImagePath = allLoader.dataset.imagePaths[sampleIdx]
          # Extract the basename of the original image to use as the output filename.
          origBaseName = os.path.basename(origImagePath)

          # Build the output path using the original filename.
          predOutputPath = os.path.join(predsDir, origBaseName)
          actualOutputPath = os.path.join(actualDir, origBaseName)
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
              predMask = resized.astype(np.int64)
            except Exception as e:
              # Record a failure when resize fails.
              failedImages.append({"image": origBaseName, "reason": f"shape mismatch and resize failed: {e}"})
              # Skip metric computation for this image.
              continue

          # Now call each metric function with (preds, targets). Wrap in try/except per metric.
          row = {"image": origBaseName}
          for mname, mfn in metricsFns.items():
            try:
              # Call the metric function with the prediction and target masks.
              val = mfn(predMask, targetMask)
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

          # Prepare and save overlay images showing Original | PredOverlay (red) | TargetOverlay (green).
          # Build overlays output directory inside the experiment output dir.
          overlaysDir = os.path.join(hparams.get("OutputDir", "Output"), "Overlays")
          # Ensure the overlays directory exists.
          os.makedirs(overlaysDir, exist_ok=True)

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
                  predMaskBin.astype(np.uint8), (baseUint8.shape[1], baseUint8.shape[0]),
                  interpolation=cv2.INTER_NEAREST
                )
              except Exception:
                predMaskBin = np.zeros(baseUint8.shape[:2], dtype=np.uint8)
            if (targetMaskBin.shape != baseUint8.shape[:2]):
              try:
                targetMaskBin = cv2.resize(
                  targetMaskBin.astype(np.uint8), (baseUint8.shape[1], baseUint8.shape[0]),
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
                0.65 * predOverlay[pmIdx].astype(np.float32) + 0.35 * redColor.astype(np.float32)).astype(np.uint8)

            # Apply green tint where target mask is present.
            greenColor = np.array([0, 255, 0], dtype=np.uint8)
            tmIdx = targetMaskBin.astype(bool)
            targetOverlay[tmIdx] = (
                0.65 * targetOverlay[tmIdx].astype(np.float32) + 0.35 * greenColor.astype(np.float32)
            ).astype(np.uint8)

            # Concatenate original, predOverlay, and targetOverlay horizontally.
            # Create a single merged overlay image by tinting prediction and target pixels on the original.
            # Start from a copy of the base image to draw combined overlays.
            try:
              merged = baseUint8.copy()
              # Apply green tint for target mask locations with 50% blending.
              merged[tmIdx] = (
                  0.65 * merged[tmIdx].astype(np.float32) + 0.35 * greenColor.astype(np.float32)
              ).astype(np.uint8)
              # Apply red tint for prediction mask locations with 50% blending.
              merged[pmIdx] = (
                  0.65 * merged[pmIdx].astype(np.float32) + 0.35 * redColor.astype(np.float32)
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
            overlayPath = os.path.join(overlaysDir, overlayName)
            try:
              Image.fromarray(combined).save(overlayPath)
            except Exception:
              try:
                cv2.imwrite(overlayPath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
              except Exception:
                pass
          except Exception:
            # Ignore overlay generation failures and continue the loop.
            pass

    # Build paths for primary and additional metrics files.
    metricsCsvPath = os.path.join(metricsDir, "PerImageMetrics.csv")
    metricsSummaryPath = os.path.join(metricsDir, "MetricsSummary.json")
    top75CsvPath = os.path.join(metricsDir, "Top75PercentMetrics.csv")
    top75SummaryPath = os.path.join(metricsDir, "Top75PercentSummary.json")
    worst10CsvPath = os.path.join(metricsDir, "Worst10PercentMetrics.csv")
    worst10SummaryPath = os.path.join(metricsDir, "Worst10PercentSummary.json")

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
      for mname in metricsFns.keys():
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
      fieldnames = ["Image"] + [k for k in metricsFns.keys()] + ["AggregateScore"]
      try:
        with open(metricsCsvPath, "w", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for r in perImageMetrics:
            # Map existing row keys to CamelCase Image key for output.
            outRow = {"Image": r.get("image")}
            for k in metricsFns.keys():
              outRow[k] = r.get(k)
            outRow["AggregateScore"] = r.get("AggregateScore")
            writer.writerow(outRow)
      except Exception as e:
        print(f"Failed to write metrics CSV: {e}")

    # Compute the overall summary statistics for all images.
    overallSummary = {"NImages": len(perImageMetrics), "NFailed": len(failedImages), "Mean": {}, "Std": {}}
    for mname in metricsFns.keys():
      vals = [r.get(mname) for r in perImageMetrics if (r.get(mname) is not None)]
      if (len(vals) > 0):
        arr = np.array(vals, dtype=float)
        overallSummary["Mean"][mname] = float(np.nanmean(arr))
        overallSummary["Std"][mname] = float(np.nanstd(arr))
      else:
        overallSummary["Mean"][mname] = None
        overallSummary["Std"][mname] = None
    overallSummary["FailedImages"] = failedImages

    # Write the overall summary JSON.
    try:
      with open(metricsSummaryPath, "w") as jf:
        json.dump(overallSummary, jf, indent=2)
    except Exception as e:
      print(f"Failed to write metrics summary JSON: {e}")

    # Helper to write a subset CSV and compute its aggregated metrics.
    def WriteSubsetFiles(rows, csvPath, summaryPath, subsetName):
      # Write CSV for selected rows.
      try:
        with open(csvPath, "w", newline="") as csvfile:
          fieldnames = ["Image"] + [k for k in metricsFns.keys()] + ["AggregateScore"]
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for r in rows:
            outRow = {"Image": r.get("image")}
            for k in metricsFns.keys():
              outRow[k] = r.get(k)
            outRow["AggregateScore"] = r.get("AggregateScore")
            writer.writerow(outRow)
      except Exception as e:
        print(f"Failed to write {subsetName} CSV: {e}")

      # Compute summary averages for this subset.
      subsetSummary = {"NImages": len(rows), "Mean": {}, "Std": {}}
      for mname in metricsFns.keys():
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
      try:
        with open(summaryPath, "w") as jf:
          json.dump(subsetSummary, jf, indent=2)
      except Exception as e:
        print(f"Failed to write {subsetName} summary JSON: {e}")

    # Write top 75% files when available.
    if (len(top75Rows) > 0):
      WriteSubsetFiles(top75Rows, top75CsvPath, top75SummaryPath, "Top75Percent")

    # Write worst 10% files when available.
    if (len(worst10Rows) > 0):
      WriteSubsetFiles(worst10Rows, worst10CsvPath, worst10SummaryPath, "Worst10Percent")

    # Print final status messages about saved files.
    print(f"Saved {globalIdx} predicted masks to: {predsDir}.")
    print(f"Per-image metrics: {metricsCsvPath}")
    print(f"Metrics summary: {metricsSummaryPath}")
    print(f"Top 75% metrics: {top75CsvPath} and {top75SummaryPath}")
    print(f"Worst 10% metrics: {worst10CsvPath} and {worst10SummaryPath}")


# Execute the inference runner when this script is run directly.
if (__name__ == "__main__"):
  # Suppress non-essential warnings for a cleaner run.
  IgnoreWarnings()
  # Set a fixed random seed for reproducibility across runs.
  DoRandomSeeding()

  # Run the main entry-point.
  Run()

  # Examples:
  # python Main.py --DataDir "Data-Tooth" --ModelName "TransUNet" --OutputDir "Test-TransUNet".
  # python Main.py --DataDir "Data-Tooth" --ModelName "ResidualAttentionUNet" --OutputDir "Test-ResidualAttentionUNet".
  # python Main.py --DataDir "Data-Tooth" --ModelName "BoundaryAwareUNet" --OutputDir "Test-BoundaryAwareUNet".
  # python Main.py --DataDir "Data-Pulp" --ModelName "TransUNet" --OutputDir "Pulp-TransUNet".
  # python Main.py --DataDir "Data-Pulp" --ModelName "ResidualAttentionUNet" --OutputDir "Pulp-ResidualAttentionUNet".
  # python Main.py --DataDir "Data-Pulp" --ModelName "BoundaryAwareUNet" --OutputDir "Pulp-BoundaryAwareUNet".
  # python Main.py --DataDir "Data-Bone" --ModelName "TransUNet" --OutputDir "Bone-TransUNet".
  # python Main.py --DataDir "Data-Bone" --ModelName "ResidualAttentionUNet" --OutputDir "Bone-ResidualAttentionUNet".
  # python Main.py --DataDir "Data-Bone" --ModelName "BoundaryAwareUNet" --OutputDir "Bone-BoundaryAwareUNet".
  # python Main.py --DataDir "Data-Tooth" --ModelName "TransUNet" --OutputDir "Test-TransUNet" --Phase "Infer" --ResumeCheckpoint "Test-TransUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Tooth" --ModelName "ResidualAttentionUNet" --OutputDir "Test-ResidualAttentionUNet" --Phase "Infer" --ResumeCheckpoint "Test-ResidualAttentionUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Pulp" --ModelName "TransUNet" --OutputDir "Pulp-TransUNet" --Phase "Infer" --ResumeCheckpoint "Pulp-TransUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Pulp" --ModelName "ResidualAttentionUNet" --OutputDir "Pulp-ResidualAttentionUNet" --Phase "Infer" --ResumeCheckpoint "Pulp-ResidualAttentionUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Bone" --ModelName "TransUNet" --OutputDir "Bone-TransUNet" --Phase "Infer" --ResumeCheckpoint "Bone-TransUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Bone" --ModelName "ResidualAttentionUNet" --OutputDir "Bone-ResidualAttentionUNet" --Phase "Infer" --ResumeCheckpoint "Bone-ResidualAttentionUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Tooth" --ModelName "BoundaryAwareUNet" --OutputDir "Test-BoundaryAwareUNet" --Phase "Infer" --ResumeCheckpoint "Test-BoundaryAwareUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Pulp" --ModelName "BoundaryAwareUNet" --OutputDir "Pulp-BoundaryAwareUNet" --Phase "Infer" --ResumeCheckpoint "Pulp-BoundaryAwareUNet\\Checkpoints\\checkpoint_best.pth".
  # python Main.py --DataDir "Data-Bone" --ModelName "BoundaryAwareUNet" --OutputDir "Bone-BoundaryAwareUNet" --Phase "Infer" --ResumeCheckpoint "Bone-BoundaryAwareUNet\\Checkpoints\\checkpoint_best.pth".
