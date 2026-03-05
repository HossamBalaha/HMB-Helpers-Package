import argparse, splitfolders, os, torch, timm, json
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding
from HMB.PyTorchHelper import (
  CustomDataset, TrainEvaluateModel,
  GetOptimizer, LoadPyTorchDict,
  GenericEvaluatePredictPlotSubset,
)

# Ensure all prints flush by default to make logs appear promptly.
# Save the original built-in print function for delegation.
_original_print = print


# Define a wrapper that sets flush=True when not explicitly provided.
def print(*args, **kwargs):
  # Ensure flush is True by default when not provided.
  if ("flush" not in kwargs):
    kwargs["flush"] = True
  # Delegate to the original print implementation.
  return _original_print(*args, **kwargs)


# Define a function to parse command line arguments for training configuration.
def GetArgs():
  # Create an argument parser with a short description.
  parser = argparse.ArgumentParser(description="Fine-tune on custom dataset")
  # Add argument for dataset directory path.
  parser.add_argument(
    "--dataDir",
    type=str,
    required=True,
    help="Path to dataset directory"
  )
  # Add argument for number of classes in the dataset.
  parser.add_argument(
    "--numClasses",
    type=int,
    required=True,
    help="Number of classes in your dataset"
  )
  # Add argument for output directory to store checkpoints.
  parser.add_argument(
    "--outputDir",
    type=str,
    default="./Output",
    help="Directory to save model checkpoints"
  )
  # Add argument for timm model name to use for fine-tuning.
  parser.add_argument(
    "--modelName",
    type=str,
    default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    help="Model name from timm"
  )
  # Add argument for image size to resize to.
  parser.add_argument(
    "--imageSize",
    type=int,
    default=448,
    help="Image size to resize to"
  )
  # Add argument for the optimizer type to use during training.
  parser.add_argument(
    "--optimizer",
    type=str,
    default="adamw",
    help="Optimizer to use (e.g., adamw, sgd)"
  )
  # Add boolean argument to control dataset splitting.
  parser.add_argument(
    "--doSplit",
    action="store_true",
    help="Whether to split the dataset"
  )
  # Add boolean argument to force re-splitting of the dataset.
  parser.add_argument(
    "--forceSplit",
    action="store_true",
    help="Whether to force re-splitting the dataset"
  )
  # Add argument to set validation split ratio.
  parser.add_argument("--splitRatio", type=float, default=0.2, help="Train/validation split ratio")
  # Add argument to set number of training epochs.
  parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
  # Add argument to set training batch size.
  parser.add_argument("--batchSize", type=int, default=8, help="Batch size")
  # Add argument to set learning rate for the optimizer.
  parser.add_argument("--learningRate", type=float, default=1e-5, help="Learning rate")
  # Add argument to set weight decay for the optimizer.
  parser.add_argument("--weightDecay", type=float, default=0.01, help="Weight decay")
  # Add argument to set number of warmup epochs.
  parser.add_argument("--warmupEpochs", type=int, default=1, help="Number of warmup epochs")
  # Add argument to set number of data loader worker processes.
  parser.add_argument("--numWorkers", type=int, default=1, help="Number of data loading workers")
  # Add argument to select device, defaulting to CUDA if available or CPU otherwise.
  parser.add_argument(
    "--device", type=str,
    default=("cuda" if (torch.cuda.is_available()) else "cpu"),
    help="Device to use for training"
  )
  # Add argument for optional checkpoint resume path.
  parser.add_argument(
    "--resumeFromCheckpoint",
    type=str,
    default=None,
    help="Path to checkpoint to resume from"
  )
  # Add verbose flag argument for more detailed logging.
  parser.add_argument("--verbose", action="store_true", help="Whether to print detailed logs")
  # Add argument to select how to judge the best model during training.
  parser.add_argument(
    "--judgeBy",
    type=str,
    default="both",
    help="Criterion to judge the best model (val_loss, val_accuracy, or both)"
  )
  # Add argument for early stopping patience in epochs (None disables early stopping).
  parser.add_argument(
    "--earlyStoppingPatience",
    type=int,
    default=None,
    help="Patience (in epochs) for early stopping, or None to disable"
  )
  # Add argument for gradient accumulation steps.
  parser.add_argument(
    "--gradAccumSteps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps"
  )
  # Add argument for maximum gradient norm for clipping (None disables clipping).
  parser.add_argument(
    "--maxGradNorm",
    type=float,
    default=None,
    help="Maximum gradient norm for clipping, or None to disable"
  )
  # Add argument to enable or disable automatic mixed precision.
  parser.add_argument(
    "--useAmp",
    action="store_true",
    help="Whether to use automatic mixed precision"
  )
  # Add argument to enable or disable MixUp augmentation.
  parser.add_argument(
    "--useMixupFn",
    action="store_true",
    help="Whether to use MixUp data augmentation"
  )
  # Add argument for MixUp alpha value.
  parser.add_argument(
    "--mixUpAlpha",
    type=float,
    default=0.5,
    help="Alpha value for MixUp data augmentation"
  )
  # Add argument to enable or disable exponential moving average for model parameters.
  parser.add_argument(
    "--useEma",
    action="store_true",
    help="Whether to use Exponential Moving Average for model parameters"
  )
  # Add argument to control saving frequency (save every N epochs, or None to rely on best model only).
  parser.add_argument(
    "--saveEvery",
    type=int,
    default=None,
    help="Save model every N epochs, or None to only save best model"
  )
  # Add optional argument to point to an already-split train folder.
  parser.add_argument(
    "--splitTrainFolder",
    type=str,
    default=None,
    help="Optional explicit path to a pre-split training folder"
  )
  # Add optional argument to point to an already-split validation folder.
  parser.add_argument(
    "--splitValFolder",
    type=str,
    default=None,
    help="Optional explicit path to a pre-split validation folder"
  )
  # Add optional argument to point to an already-split test folder.
  parser.add_argument(
    "--splitTestFolder",
    type=str,
    default=None,
    help="Optional explicit path to a pre-split test folder"
  )
  # Example of verbose:
  # xxx.py --dataDir /path/to/dataset --numClasses 10 --modelName resnet50 --imageSize 224 --optimizer adamw --doSplit True --splitRatio 0.2 --epochs 20 --batchSize 16 --learningRate 1e-4 --weightDecay 0.01 --warmupEpochs 2 --numWorkers 4 --device cuda:0 --resumeFromCheckpoint /path/to/checkpoint.pth --verbose True
  # To make it --verbose without needing to specify True, you can change the argument definition to:
  # parser.add_argument("--verbose", action="store_true", help="Whether to print detailed logs")

  # Return the parsed arguments from the command line.
  return parser.parse_args()


# Validate and normalize command-line arguments.
def ValidateArgs(args):
  '''Validate and (lightly) normalize command-line arguments.

  Raises informative exceptions for invalid values and adjusts a couple of
  settings (for example falling back to CPU if CUDA isn't available).
  Returns the (possibly modified) args object.
  '''
  # Ensure dataDir is a string and not empty.
  if (not isinstance(args.dataDir, str) or not args.dataDir):
    # Raise an informative error when dataDir is invalid.
    raise ValueError("--dataDir must be a non-empty string path to your dataset")
  # Ensure the provided dataDir path exists.
  if (not os.path.exists(args.dataDir)):
    # Raise an informative error when the path is missing.
    raise FileNotFoundError(f"dataDir path does not exist: {args.dataDir}")

  # Ensure numClasses is a positive integer.
  if (not isinstance(args.numClasses, int) or args.numClasses <= 0):
    # Raise when numClasses is invalid.
    raise ValueError("--numClasses must be a positive integer")

  # Ensure batchSize is a positive integer.
  if (not isinstance(args.batchSize, int) or args.batchSize <= 0):
    # Raise when batchSize is invalid.
    raise ValueError("--batchSize must be a positive integer")
  # Ensure epochs is a positive integer.
  if (not isinstance(args.epochs, int) or args.epochs <= 0):
    # Raise when epochs is invalid.
    raise ValueError("--epochs must be a positive integer")
  # Ensure warmupEpochs is a non-negative integer.
  if (not isinstance(args.warmupEpochs, int) or args.warmupEpochs < 0):
    # Raise when warmupEpochs is invalid.
    raise ValueError("--warmupEpochs must be a non-negative integer")
  # Ensure numWorkers is an integer >= 0.
  if (not isinstance(args.numWorkers, int) or args.numWorkers < 0):
    # Raise when numWorkers is invalid.
    raise ValueError("--numWorkers must be an integer >= 0")

  # Ensure splitRatio is numeric and convert to float.
  if (not isinstance(args.splitRatio, float) and not isinstance(args.splitRatio, int)):
    # Raise when splitRatio is not numeric.
    raise ValueError("--splitRatio must be a float between 0 and 1")
  # Normalize splitRatio to float.
  args.splitRatio = float(args.splitRatio)
  # Ensure splitRatio is strictly between 0 and 1.
  if (not (0.0 < args.splitRatio < 1.0)):
    # Raise when splitRatio is out of bounds.
    raise ValueError("--splitRatio must be strictly between 0 and 1 (e.g. 0.2)")

  # Ensure modelName is a non-empty string.
  if (not isinstance(args.modelName, str) or not args.modelName):
    # Raise when modelName is invalid.
    raise ValueError("--modelName must be a non-empty timm model name string")
  # Ensure imageSize is a positive integer.
  if (not isinstance(args.imageSize, int) or args.imageSize <= 0):
    # Raise when imageSize is invalid.
    raise ValueError("--imageSize must be a positive integer (e.g. 224 or 448)")
  # Ensure optimizer is a non-empty string.
  if (not isinstance(args.optimizer, str) or not args.optimizer):
    # Raise when optimizer is invalid.
    raise ValueError("--optimizer must be a non-empty string (e.g. 'adamw' or 'sgd')")

  # Ensure device is a string and not empty.
  if (not isinstance(args.device, str) or not args.device):
    # Raise when device is invalid.
    raise ValueError("--device must be a string like 'cpu' or 'cuda' or 'cuda:0'")
  # Fall back to CPU if CUDA was requested but is not available.
  if (args.device.startswith("cuda") and not torch.cuda.is_available()):
    # Inform the user when falling back to CPU.
    print("Warning: CUDA requested but not available. Falling back to CPU.")
    # Set device to CPU for safety.
    args.device = "cpu"

  # Validate the resumeFromCheckpoint path when provided.
  if (args.resumeFromCheckpoint):
    # Ensure resumeFromCheckpoint is a non-empty string.
    if (not isinstance(args.resumeFromCheckpoint, str) or not args.resumeFromCheckpoint):
      # Raise when resumeFromCheckpoint is invalid.
      raise ValueError("--resumeFromCheckpoint must be a non-empty string path or None")
    # Ensure the checkpoint file exists on disk.
    if (not os.path.exists(args.resumeFromCheckpoint)):
      # Raise when the checkpoint file is missing.
      raise FileNotFoundError(f"resumeFromCheckpoint not found: {args.resumeFromCheckpoint}")

  # Validate judgeBy choices.
  validJudges = {"val_loss", "val_accuracy", "both"}
  # Ensure judgeBy is one of the allowed strings.
  if (args.judgeBy not in validJudges):
    # Raise when judgeBy is invalid.
    raise ValueError(f"--judgeBy must be one of {validJudges}")

  # Validate earlyStoppingPatience when provided.
  if (args.earlyStoppingPatience is not None):
    # Ensure earlyStoppingPatience is positive when not None.
    if (not isinstance(args.earlyStoppingPatience, int) or args.earlyStoppingPatience <= 0):
      # Raise when earlyStoppingPatience is invalid.
      raise ValueError("--earlyStoppingPatience must be a positive integer or None")
  # Ensure gradAccumSteps is a positive integer.
  if (not isinstance(args.gradAccumSteps, int) or args.gradAccumSteps <= 0):
    # Raise when gradAccumSteps is invalid.
    raise ValueError("--gradAccumSteps must be a positive integer")
  # Ensure maxGradNorm is positive when provided.
  if (args.maxGradNorm is not None and args.maxGradNorm <= 0):
    # Raise when maxGradNorm is invalid.
    raise ValueError("--maxGradNorm must be positive or None")

  # Ensure boolean flags are booleans.
  for boolAttr in ("useAmp", "useMixupFn", "useEma", "doSplit", "forceSplit", "verbose"):
    # Retrieve the boolean attribute from args.
    val = getattr(args, boolAttr)
    # Ensure val is a boolean type.
    if (not isinstance(val, bool)):
      # Raise when a boolean flag is invalid.
      raise ValueError(f"--{boolAttr} must be a boolean (True/False)")

  # Validate mixUpAlpha when mixup is enabled.
  if (args.useMixupFn):
    # Ensure mixUpAlpha is within the allowed interval.
    if (not (0.0 < args.mixUpAlpha <= 1.0)):
      # Raise when mixUpAlpha is invalid.
      raise ValueError("--mixUpAlpha must be in the interval (0, 1]")

  # Validate saveEvery when provided.
  if (args.saveEvery is not None and (not isinstance(args.saveEvery, int) or args.saveEvery <= 0)):
    # Raise when saveEvery is invalid.
    raise ValueError("--saveEvery must be a positive integer or None")

  # Validate weight decay and learning rate.
  if (args.weightDecay is not None and args.weightDecay < 0):
    # Raise when weightDecay is invalid.
    raise ValueError("--weightDecay must be >= 0")
  if (args.learningRate is not None and args.learningRate <= 0):
    # Raise when learningRate is invalid.
    raise ValueError("--learningRate must be > 0")

  # When splitting is requested, ensure dataDir contains subfolders representing classes.
  if (args.doSplit):
    try:
      # List directory entries that are directories to infer class subfolders.
      entries = [d for d in os.listdir(args.dataDir) if os.path.isdir(os.path.join(args.dataDir, d))]
      # Ensure at least one class subfolder exists.
      if (len(entries) == 0):
        # Raise when no class subfolders are found.
        raise ValueError(f"--dataDir appears to contain no class subfolders: {args.dataDir}")
    except PermissionError:
      # Re-raise permission errors for visibility.
      raise

  # Validate explicit splitTrainFolder when provided.
  if (args.splitTrainFolder):
    # Ensure splitTrainFolder points to an existing directory.
    if (not isinstance(args.splitTrainFolder, str) or not os.path.isdir(args.splitTrainFolder)):
      # Raise when splitTrainFolder is invalid.
      raise ValueError("--splitTrainFolder must be a path to an existing directory or None")

  # Validate explicit splitValFolder when provided.
  if (args.splitValFolder):
    # Ensure splitValFolder points to an existing directory.
    if (not isinstance(args.splitValFolder, str) or not os.path.isdir(args.splitValFolder)):
      # Raise when splitValFolder is invalid.
      raise ValueError("--splitValFolder must be a path to an existing directory or None")

  # Validate explicit splitTestFolder when provided.
  if (args.splitTestFolder):
    # Ensure splitTestFolder points to an existing directory.
    if (not isinstance(args.splitTestFolder, str) or not os.path.isdir(args.splitTestFolder)):
      # Raise when splitTestFolder is invalid.
      raise ValueError("--splitTestFolder must be a path to an existing directory or None")

  print("-" * 40)
  for k, v in vars(args).items():
    print(f"{k}: {v}")
  print("-" * 40)

  # Return the validated and possibly modified args object.
  return args


# Create data loaders using timm's data configuration helpers.
def CreateTimmDataLoaders(args, splitTrainFolder=None, splitValFolder=None):
  # Create a timm model instance to query its data configuration.
  model = timm.create_model(args.modelName, pretrained=True)
  # Resolve the model-specific data configuration from timm.
  dataConfig = timm.data.resolve_model_data_config(model)
  # Create training transforms from the data configuration.
  trainTransforms = timm.data.create_transform(**dataConfig, is_training=True)
  # Create validation transforms from the data configuration.
  valTransforms = timm.data.create_transform(**dataConfig, is_training=False)

  if (splitTrainFolder is None):
    # Create the training dataset using the project's CustomDataset wrapper.
    trainDataset = CustomDataset(os.path.join(args.dataDir + " Split", "train"), transform=trainTransforms)
  else:
    # Create the training dataset directly from the data directory.
    trainDataset = CustomDataset(splitTrainFolder, transform=trainTransforms)

  if (splitValFolder is None):
    # Create the validation dataset using the project's CustomDataset wrapper.
    valDataset = CustomDataset(os.path.join(args.dataDir + " Split", "val"), transform=valTransforms)
  else:
    # Create the validation dataset directly from the data directory.
    valDataset = CustomDataset(splitValFolder, transform=valTransforms)

  if (args.verbose):
    print(
      f"Training dataset created with {len(trainDataset)} samples from "
      f"{splitTrainFolder if splitTrainFolder else os.path.join(args.dataDir + ' Split', 'train')}"
    )
    print(
      f"Validation dataset created with {len(valDataset)} samples from "
      f"{splitValFolder if splitValFolder else os.path.join(args.dataDir + ' Split', 'val')}"
    )
    print(f"Data configuration used for transforms: {dataConfig}")
    print(f"Example training transform pipeline: {trainTransforms}")
    print(f"Example validation transform pipeline: {valTransforms}")
    print("Classes:", trainDataset.classes)
    print("Class to index mapping:", trainDataset.classToIdx)
    print(f"First 5 samples in training dataset: {[trainDataset[i][1] for i in range(min(5, len(trainDataset)))]}")

  # Create the DataLoader for the training dataset with shuffling enabled.
  trainLoader = DataLoader(
    trainDataset,
    batch_size=args.batchSize,
    shuffle=True,
    num_workers=args.numWorkers,
    pin_memory=True
  )

  # Create the DataLoader for the validation dataset without shuffling.
  valLoader = DataLoader(
    valDataset,
    batch_size=args.batchSize,
    shuffle=False,
    num_workers=args.numWorkers,
    pin_memory=True
  )

  # Return both training and validation data loaders.
  return trainLoader, valLoader


# Create and configure the timm model for the current task.
def CreateModel(args):
  # Notify that model creation has started.
  print("Creating model...")
  # Instantiate the timm model with the requested number of classes.
  model = timm.create_model(args.modelName, pretrained=True, num_classes=args.numClasses)
  print(f"Model {args.modelName} created with {args.numClasses} output classes.")
  # Return the constructed model object.
  return model


# Main entry point for script execution.
def MainTrain():
  # Parse command line arguments into the args variable.
  args = GetArgs()
  # Validate and possibly normalize args early.
  args = ValidateArgs(args)
  # Print all arguments when verbose mode is enabled.
  if (getattr(args, "verbose", False)):
    # Print the complete args namespace.
    print("Full arguments:")
    for k, v in vars(args).items():
      print(f"  {k}: {v}")

  # Check whether to split the dataset or whether split folders already exist.
  if (
    (args.forceSplit) or (
    (not os.path.exists(os.path.join(args.dataDir + " Split", "train"))) and
    (not os.path.exists(os.path.join(args.dataDir + " Split", "val")))
  )):
    # Print message about existence of train and validation folders.
    print("Train and validation folders already exist.")
    # Conditionally split the dataset if requested by the argument.
    if (args.doSplit):
      # Compute the training ratio based on the split ratio argument.
      trainRatio = 1.0 - args.splitRatio
      # Perform the folder split using `splitfolders.ratio` with a fixed seed.
      splitfolders.ratio(
        args.dataDir,
        output=args.dataDir + " Split",
        seed=np.random.randint(0, 10000),
        ratio=(trainRatio, args.splitRatio)
      )
      # Print completion message for dataset splitting.
      print("Dataset split completed.")

  # Ensure the output directory exists or create it.
  os.makedirs(args.outputDir, exist_ok=True)
  print(f"Output directory: {args.outputDir}")

  # Save the training configuration arguments to a JSON file in the output directory for future reference.
  argsJsonPath = os.path.join(args.outputDir, "TrainingArgs.json")
  with open(argsJsonPath, "w") as f:
    json.dump(vars(args), f, indent=4)

  # Select the device for computation and wrap it in a torch.device.
  device = torch.device(args.device)
  # Print the selected device to the console.
  print(f"Using device: {device}")

  # Print a message indicating creation of data loaders.
  print("Creating data loaders...")
  # Resolve explicit split train folder path when provided.
  splitTrainFolder = (
    args.splitTrainFolder
    if (args.splitTrainFolder)
    else os.path.join(args.dataDir + " Split", "train")
  )
  # Resolve explicit split validation folder path when provided.
  splitValFolder = args.splitValFolder if (args.splitValFolder) else os.path.join(args.dataDir + " Split", "val")
  # Create the training and validation data loaders with explicit folder paths.
  trainLoader, valLoader = CreateTimmDataLoaders(
    args,
    splitTrainFolder=splitTrainFolder,
    splitValFolder=splitValFolder
  )
  # Print the number of training samples available.
  print(f"Training samples: {len(trainLoader.dataset)}")
  # Print the number of validation samples available.
  print(f"Validation samples: {len(valLoader.dataset)}")

  # Create the model using the CreateModel helper.
  model = CreateModel(args)
  # Move the model to the selected device for training.
  model.to(device)
  print("Model created and moved to device.")

  # Instantiate the cross-entropy loss function for classification.
  criterion = nn.CrossEntropyLoss()
  print("Loss function created.")
  # Instantiate the optimizer via project's GetOptimizer helper.
  optimizer = GetOptimizer(
    model,  # Model whose parameters to optimize.
    optimizerType=args.optimizer,  # Type of optimizer to use.
    learningRate=args.learningRate,  # Learning rate for the optimizer.
    weightDecay=args.weightDecay,  # Weight decay for regularization.
  )
  print(f"Optimizer ({args.optimizer}) created.")

  # Call `TrainEvaluateModel` to perform training and evaluation and store the history.
  history = TrainEvaluateModel(
    model=model,  # Model to train and evaluate.
    criterion=criterion,  # Loss function.
    device=device,  # Device to run training and evaluation on (CPU or GPU).
    bestModelStoragePath=os.path.join(args.outputDir, "BestModel.pth"),  # Path to save the best model.
    noOfClasses=args.numClasses,  # Number of classes in the classification task.
    numEpochs=args.epochs,  # Total number of epochs for training.
    optimizer=optimizer,  # Optimizer for updating model parameters.
    scaler=torch.cuda.amp.GradScaler(),  # Gradient scaler for mixed precision training.
    scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs),  # Learning rate scheduler.
    trainLoader=trainLoader,  # DataLoader for training data.
    valLoader=valLoader,  # DataLoader for validation data.
    resumeFromCheckpoint=args.resumeFromCheckpoint,  # Whether to resume training from a checkpoint.
    # Path to save the final model after training.
    finalModelStoragePath=os.path.join(args.outputDir, "LastModel.pth"),
    verbose=args.verbose,  # Verbosity flag to control logging.
    judgeBy=args.judgeBy,  # Criterion to judge the best model (val_loss, val_accuracy, or both).
    earlyStoppingPatience=args.earlyStoppingPatience,  # Patience for early stopping.
    gradAccumSteps=args.gradAccumSteps,  # Number of gradient accumulation steps.
    maxGradNorm=args.maxGradNorm,  # Maximum gradient norm for clipping.
    useAmp=args.useAmp,  # Whether to use automatic mixed precision.
    useMixupFn=args.useMixupFn,  # Whether to use MixUp data augmentation.
    mixUpAlpha=args.mixUpAlpha,  # Value of alpha for MixUp data augmentation.
    useEma=args.useEma,  # Whether to use Exponential Moving Average for model parameters.
    saveEvery=args.saveEvery,  # Save model every N epochs.
  )
  # Save the training history as a CSV file in the output directory.
  historyCsvPath = os.path.join(args.outputDir, "TrainingHistory.csv")
  pd.DataFrame(history).to_csv(historyCsvPath, index=False)
  print(f"Training history saved to: {historyCsvPath}")
  # Print a message indicating training completion.
  print("Training complete.")


# Create a callable factory for timm model predictions.
def CreateTimmPredictCallable(model, transform, device, imageSize):
  '''Return a callable(imageNp, imagePath=None, trueClassName=None) -> 1D numpy prob vector.'''

  def PredictCallable(img, imagePath=None, trueClassName=None):
    # Convert the input HWC numpy image to a PIL Image for transformation.
    # Resize the image to the expected input size for the model using the provided transform.
    from PIL import Image
    imgPIL = Image.fromarray(img).convert("RGB").resize((imageSize, imageSize))
    # Apply the provided transform to the image and add a batch dimension.
    inputTensor = transform(imgPIL).unsqueeze(0).to(device)
    # Perform inference with the model in evaluation mode and no gradient tracking.
    with torch.inference_mode():
      output = model(inputTensor)
      probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs

  return PredictCallable


def MainTest():
  # Parse and validate arguments as in MainTrain.
  args = GetArgs()
  args = ValidateArgs(args)
  # Create the model and load the best checkpoint.
  model = CreateModel(args)

  bestModelPath = os.path.join(args.outputDir, "BestModel.pth")
  stateDict = LoadPyTorchDict(bestModelPath, device=args.device)

  # isEMA = args.useEma
  # if (isEMA):
  #   modelStateDict = stateDict.get("ema_module_state_dict", None)
  #   if (modelStateDict is None):
  #     raise ValueError(
  #       "EMA state dict not found in checkpoint. Ensure that --useEma was enabled during training and "
  #       "that the checkpoint contains the EMA state."
  #     )
  # else:

  modelStateDict = stateDict.get("model_state_dict", None)
  if (modelStateDict is None):
    raise ValueError(
      "Model state dict not found in checkpoint. Ensure that the checkpoint contains the model state."
    )

  model.load_state_dict(modelStateDict)
  print(f"Model loaded from checkpoint: {bestModelPath}")

  # Prepare a prediction callable that accepts a HWC numpy image and returns 1D probability vector.
  # This matches the expected interface of `GenericEvaluatePredictPlotSubset`.
  # Create a default transform from timm for the model if available.
  try:
    dataConfig = timm.data.resolve_model_data_config(model)
    modelTransform = timm.data.create_transform(**dataConfig, is_training=False)
  except Exception as e:
    raise ValueError(
      "Failed to create default transform from timm data config. "
      "Ensure the model name is correct and timm is properly installed. "
      f"Original error: {e}"
    )

  device = torch.device(args.device)
  model.to(device)
  model.eval()

  for split in ["train", "val", "test"]:
    splitFolder = (
      getattr(args, f"split{split.capitalize()}Folder")
      if (getattr(args, f"split{split.capitalize()}Folder"))
      else os.path.join(args.dataDir + " Split", split)
    )
    print(f"{split.capitalize()} folder: {splitFolder}")

    # Run inference and generate plots for the current split using the `GenericEvaluatePredictPlotSubset` helper.
    (
      predsCsvPath, weightedMetrics, allPredsIndices, allGtsIndices, allPredsProbs,
      allPredsConfs, predictionsRecords, classNames, cm
    ) = GenericEvaluatePredictPlotSubset(
      datasetDir=splitFolder,
      model=CreateTimmPredictCallable(model, modelTransform, device, args.imageSize),
      subset=None,
      prefix=split.capitalize(),
      storageDir=args.outputDir,
      heavy=True,
      computeECE=True,
      exportFailureCases=True,
      saveArtifacts=True,
      maxSamples=None,
      preprocessFn=None,
      dpi=720,
    )

  print(f"Per-sample predictions CSV path: {predsCsvPath}")
  print("Inference and per-sample export complete.")


# Execute the script when run directly.
if (__name__ == "__main__"):
  # Suppress noisy warnings early in execution.
  # IgnoreWarnings()
  # Set random seeds for reproducibility.
  DoRandomSeeding()
  # Run the main (train) entry point to perform training and validation.
  MainTrain()
  # Run the main test entry point to perform inference and plotting.
  MainTest()
