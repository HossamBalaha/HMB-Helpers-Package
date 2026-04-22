from HMB.Initializations import CheckInstalledModules

if __name__ == "__main__":
  CheckInstalledModules(["torch", "numpy", "matplotlib", "PIL", "tqdm", "tensorboard", "sklearn"])

# ------------------------------------------------------------------------- #

import os, torch, cv2, argparse, builtins
import numpy as np
from tqdm import tqdm
from typing import Dict
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from HMB.PyTorchHelper import GetOptimizer, PyTorchUNetSegmentationModule
from HMB.DatasetsHelper import CreateSegmentationDataLoaders
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding
from HMB.PyTorchUNetHelper import GetUNetModel
from HMB.Utils import DumpJsonFile

# Ensure all prints flush by default to make logs appear promptly.
# Save the original built-in print function for delegation.
_original_print = builtins.print


# Define a wrapper that sets flush=True when not explicitly provided.
def print(*args, **kwargs):
  # Ensure flush is True by default when not provided.
  if ("flush" not in kwargs):
    kwargs["flush"] = True
  # Delegate to the original print implementation.
  return _original_print(*args, **kwargs)

# Override the built-in print with our wrapper to ensure all prints are flushed immediately.
builtins.print = print

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
  # DPI for saving evaluation figures.
  "DPI"             : 720,
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
  parser.add_argument("--DPI", type=int, default=defaultHparams["DPI"], help="DPI for saving evaluation figures.")
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
    "DPI"             : args.DPI,
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
  # with open(filePath, "w", encoding="utf-8") as file:
  #   json.dump(hparams, file, indent=2)
  DumpJsonFile(filePath, hparams)


# Define the main Run function which executes training or inference phases.
def Run():
  # Parse command-line arguments into a hyperparameters dictionary.
  hparams = ParseArgs()

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
  segObj = PyTorchUNetSegmentationModule(
    model,
    trainLoader,
    valLoader,
    allLoader,
    optimizer,
    scheduler,
    lossFn,
    device=device,
    outputDir=hparams.get("OutputDir", "Output"),
    dpi=hparams.get("DPI", 720)
  )

  # If resume checkpoint is provided, load it and resume.
  if (hparams.get("ResumeCheckpoint", "") != ""):
    # Load checkpoint into trainer if requested.
    segObj.LoadCheckpoint(hparams.get("ResumeCheckpoint"))
    # Print checkpoint resume message.
    print("Resumed from checkpoint:", hparams.get("ResumeCheckpoint"))

  # Check if the output directory already exists and contains checkpoints.
  if (os.path.exists(segObj.checkpointDir) and os.path.isdir(segObj.checkpointDir)):
    requiredFile = "CheckpointBest.pth"
    checkpointFiles = os.listdir(segObj.checkpointDir)
    if (requiredFile in checkpointFiles):
      print(f"Warning: Checkpoint directory {segObj.checkpointDir} already contains {requiredFile}.")
      # Load the existing checkpoint to avoid overwriting and to continue training or inference.
      existingCheckpointPath = os.path.join(segObj.checkpointDir, requiredFile)
      segObj.LoadCheckpoint(existingCheckpointPath)
      print(f"Loaded existing checkpoint from {existingCheckpointPath} to continue training/inference.")
    else:
      print(f"No existing checkpoint found in {segObj.checkpointDir}. Starting fresh.")
  else:
    print(f"Checkpoint directory {segObj.checkpointDir} does not exist. Starting fresh.")

  # Check if the requested phase is Train or Infer and execute accordingly.
  # If the requested phase is Train, run training and return.
  if (hparams.get("Phase", "Infer") == "Train"):
    print("Starting training...")
    # Run training for the specified number of epochs.
    segObj.Train(hparams.get("NumEpochs", 50))
    print("Training completed.")

  elif (hparams.get("Phase", "Infer") == "Infer"):
    print("Starting inference...")
    # Run inference to save predicted masks and compute metrics.
    segObj.Inference()
    print("Inference completed.")


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

  # python PyTorch_UNet_Segmentation.py --DataDir "D:\Book Chapter\Second Book Chapter\Brain Tumor Segmentation" --ModelName "CBAMUNet" --OutputDir "D:\Book Chapter\Second Book Chapter\CBAMUNet_1_T1" --Phase "Infer" --ResumeCheckpoint "D:\Book Chapter\Second Book Chapter\CBAMUNet_1_T1\Checkpoints\\CheckpointBest.pth"
