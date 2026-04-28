import os, warnings, builtins, cv2, math, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from HMB.TFUNetHelper import VNet, SegNet
from HMB.TFHelper import TFDataset
from HMB.ImagesHelper import ReadImage, ReadMask
from HMB.ImageSegmentationMetrics import *
from HMB.Initializations import IMAGE_SUFFIXES

# ============================================================================================ #
# Enable eager execution for TensorFlow 2.x (if not already enabled).
tf.config.run_functions_eagerly(True)
warnings.filterwarnings("ignore")

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

# Select a specific GPU when multiple GPUs are available (optional).
gpus = tf.config.list_physical_devices("GPU")
gpuIndex = 0  # Change this index to select a different GPU (e.g., 0, 1, 2, ...).
if (gpus):
  try:
    # Restrict TensorFlow to only use the first GPU.
    tf.config.set_visible_devices(gpus[gpuIndex], "GPU")
    print("Using GPU:", gpus[gpuIndex])
  except RuntimeError as e:
    print("Error setting visible devices:", e)


# ============================================================================================ #


# Parse command-line arguments and return configuration dictionary.
def ParseArgs():
  import argparse

  # Create an argument parser for the evaluation script.
  parser = argparse.ArgumentParser(description="TensorFlow UNet evaluation and prediction script.")

  # Add arguments for model, weights, and data paths.
  parser.add_argument("--ModelName", type=str, default="VNet", help="Model name to instantiate.")
  parser.add_argument("--ModelWeights", type=str, required=True, help="Path to the model weights file.")
  parser.add_argument("--DataDir", type=str, required=True, help="Path to the folder containing images.")
  parser.add_argument(
    "--MasksDir", type=str, required=True,
    help=(
      "Path to the folder containing masks (required). "
      "Provide a folder with mask files corresponding to images in --DataDir."
    )
  )
  parser.add_argument(
    "--MaskPostfix", type=str, default="",
    help=(
      "Optional postfix pattern for mask filenames. If provided, image basename 'name.ext' -> mask basename 'name_<ext>.ext' "
      "when MaskPostfix == 'extformat'. Use empty string for no postfix."
    )
  )
  parser.add_argument("--OutputDir", type=str, default="Output", help="Directory to store predictions and metrics.")
  parser.add_argument("--InputSize", type=int, nargs=3, default=[256, 256, 3], help="Input size H W C.")
  parser.add_argument("--NumClasses", type=int, default=1, help="Number of segmentation classes.")
  parser.add_argument("--SavePredictions", action="store_true", help="Whether to save predicted masks as PNG files.")
  # Add argument for optional hyperparameters JSON file path.
  parser.add_argument(
    "--HyperparamsJson", type=str, default="",
    help="Optional path to a JSON file containing model hyperparameters."
  )

  # Parse arguments.
  args = parser.parse_args()

  # Build the keys configuration dictionary for downstream usage.
  config = {
    "ModelName"      : args.ModelName,
    "ModelWeights"   : args.ModelWeights,
    "DataDir"        : args.DataDir,
    "MasksDir"       : args.MasksDir,
    "MaskPostfix"    : args.MaskPostfix,
    "OutputDir"      : args.OutputDir,
    "InputSize"      : tuple(args.InputSize),
    "NumClasses"     : args.NumClasses,
    "SavePredictions": args.SavePredictions,
    "HyperparamsJson": args.HyperparamsJson,
  }

  # Ensure the output directory exists.
  os.makedirs(config.get("OutputDir", "Output"), exist_ok=True)

  # Return the parsed configuration.
  return config


# Evaluate the model on a dataset with masks and optionally save predictions.
def EvaluateModel(config):
  # Extract configuration values into local variables for clarity.
  modelName = config.get("ModelName")
  modelWeights = config.get("ModelWeights")
  dataDir = config.get("DataDir")
  masksDir = config.get("MasksDir")
  outputDir = config.get("OutputDir")
  inputSize = config.get("InputSize")
  numClasses = config.get("NumClasses")
  savePreds = config.get("SavePredictions")

  # Collect image file paths from the provided data directory.
  allFiles = sorted(os.listdir(dataDir))
  imageFiles = [
    os.path.join(dataDir, f)
    for f in allFiles
    if (f.endswith(tuple(IMAGE_SUFFIXES)))
  ]

  # If no images found, bail out with a helpful message.
  if (len(imageFiles) == 0):
    print(f"No image files found in DataDir='{dataDir}'. Supported extensions: {IMAGE_SUFFIXES}")
    return

  # MasksDir is required now: validate existence and that masks match images.
  if ((not masksDir) or (not os.path.isdir(masksDir))):
    print(f"MasksDir must be provided and be an existing directory. Given MasksDir='{masksDir}'")
    return

  # Build mask path list that corresponds to image filenames.
  maskPostfix = config.get("MaskPostfix", "")
  masksList = []
  for p in imageFiles:
    base = os.path.basename(p)
    name, ext = os.path.splitext(base)
    if (maskPostfix):
      # User requested a postfix that transforms "name.ext" to "name_<ext_no>.ext"
      # e.g. "image.png" -> "image_png.png" where ext_no is "png"
      extNo = ext.lstrip(".")
      maskName = f"{name}{maskPostfix}"
    else:
      maskName = base
    masksList.append(os.path.join(masksDir, maskName))

  # Ensure every mask exists for the corresponding image; otherwise abort with details.
  missingMasks = [m for m in masksList if (not os.path.exists(m))]
  if (len(missingMasks) > 0):
    print(f"Found {len(missingMasks)} missing mask files in MasksDir='{masksDir}'. Missing examples:")
    for mm in missingMasks[:10]:
      print("  ", mm)
    if (len(missingMasks) > 10):
      print("  ...")
    print("Ensure mask filenames match image filenames (same basename) and try again.")
    return

  print(
    f"Found {len(imageFiles)} image files and {len(masksList)} corresponding mask files. "
    f"Proceeding with evaluation."
  )

  # Build the evaluation dataset.
  # dataset = BuildDataset(imageFiles, masksList, inputSize=inputSize, batchSize=batchSize)
  dataset = TFDataset(imageFiles, masksList, batchSize=1)

  # Instantiate the model using the factory helper.
  # model = GetUNetModel(modelName, inputChannels=inputSize[2], numClasses=numClasses)

  # Load hyperparameters from JSON file if provided by the user.
  hyperparamsJson = config.get("HyperparamsJson", "")
  # Initialize hyperparameters dictionary with defaults.
  hyperparams = {}
  # If a hyperparameters JSON path was provided, attempt to read and parse it.
  if (hyperparamsJson):
    # Validate that the JSON file exists on disk.
    if (not os.path.exists(hyperparamsJson)):
      print(f"Hyperparameters JSON file not found: {hyperparamsJson}")
    else:
      # Read the JSON file into the hyperparams dictionary.
      with open(hyperparamsJson, "r", encoding="utf-8") as jsonFile:
        hyperparams = json.load(jsonFile)[0]

  # Instantiate the requested model implementation based on the ModelName argument.
  if (modelName == "VNet"):
    # Extract model hyperparameters from the JSON dictionary or use defaults.
    kernelInitializer = hyperparams.get("Kernelinitializer", "glorot_normal")
    dropoutRatio = float(hyperparams.get("Dropoutratio", 0.1))
    dropoutType = hyperparams.get("Dropouttype", "spatial")
    applyBatchNorm = hyperparams.get("Applybatchnorm", True) in [True, "true", "True", "TRUE"]
    concatenateType = hyperparams.get("Concatenatetype", "concatenate")
    noOfLevels = int(hyperparams.get("Nooflevels", 4))

    # Create a VNet instance using extracted hyperparameters.
    model = VNet(
      inputSize=inputSize,
      kernelInitializer=kernelInitializer,
      dropoutRatio=dropoutRatio,
      dropoutType=dropoutType,
      applyBatchNorm=applyBatchNorm,
      concatenateType=concatenateType,
      noOfLevels=noOfLevels,
    )
  elif (modelName == "SegNet"):
    # Extract model hyperparameters from the JSON dictionary or use defaults.
    level = int(hyperparams.get("Level", 4))
    encoder = hyperparams.get("Encoder", "VGG16")

    # Create a SegNet instance using the same hyperparameters when available.
    model = SegNet(
      inputSize=inputSize,
      encoder=encoder,
      level=level,
      numClasses=numClasses,
    )
  else:
    print(f"Unsupported ModelName='{modelName}'. Supported options: 'VNet', 'SegNet'.")
    return

  # Ensure the model is built before loading weights. Some custom model
  # factories return an un-built subclassed model which requires either
  # calling build(inputShape) or running a single forward pass.
  try:
    if (not getattr(model, "built", False)):
      print("Model is not built. Attempting to build the model before loading weights.")
      # Try build() first — works for models that implement build().
      try:
        model.build((None, inputSize[0], inputSize[1], inputSize[2]))
        print("Model built via build(input_shape).")
      except Exception:
        raise RuntimeError("Model does not implement build(input_shape), trying a forward pass to build.")
  except Exception as e:
    print("Unexpected error while preparing the model for weight loading:", e)

  # Load provided weights into the model.
  model.load_weights(modelWeights)

  # Run predictions on the dataset to obtain model outputs.
  # Calculate explicit steps from number of images to avoid empty/zero-target
  # Progbar errors when dataset cardinality is unknown or zero.
  numSamples = len(imageFiles)
  preds = model.predict(dataset, steps=numSamples, verbose=1)

  # Normalize preds to numpy array and ensure first axis is samples.
  preds = np.asarray(preds)

  # If masks are available, evaluate metrics per image.
  results = []
  # Load masks into memory for metric computation.
  masksArray = [ReadMask(np.bytes_(p), inputSize[:2]) for p in masksList]
  # Iterate through predicted samples and compare with ground truth masks.
  for idx in range(preds.shape[0]):
    p = preds[idx]
    # Squeeze prediction to 2D if single-channel.
    pred2d = np.squeeze(p)

    # Binarize predictions if numClasses <= 2 (assuming sigmoid output).
    if (numClasses <= 2):
      # Normalize first using min-max scaling to ensure values are in [0, 1] before thresholding.
      # minVal = np.min(pred2d)
      # maxVal = np.max(pred2d)
      # if (maxVal > minVal):
      #   pred2d = (pred2d - minVal) / (maxVal - minVal)
      # else:
      #   print("Warning: Predictions have zero variance (max == min). Skipping normalization.")
      pred2d = (pred2d > 0.5).astype("float32")

    gt = masksArray[idx]  # Retrieve the corresponding ground truth mask.
    gt2d = np.squeeze(gt)  # Squeeze ground truth to 2D.
    dice = ComputeDice(pred2d, gt2d)  # Compute Dice coefficient for the sample.
    iou = ComputeIoU(pred2d, gt2d)  # Compute IoU for the sample.
    f1 = ComputeF1Score(pred2d, gt2d)  # Compute F1 score for the sample.
    acc = ComputePixelAccuracy(pred2d, gt2d)  # Compute pixel accuracy for the sample.
    map = ComputeMeanAveragePrecision(pred2d, gt2d)  # Compute mean average precision for the sample.
    bf1 = ComputeBoundaryF1Score(pred2d, gt2d)  # Compute boundary F1 score for the sample.
    mcc = ComputeMatthewsCorrelationCoefficient(pred2d, gt2d)  # Compute MCC for the sample.
    kc = ComputeCohensKappa(pred2d, gt2d)  # Compute Cohen's Kappa for the sample.

    # Append metrics to results.
    results.append(
      {
        "Image"   : os.path.basename(imageFiles[idx]),
        "Dice"    : float(dice),
        "IoU"     : float(iou),
        "F1"      : float(f1),
        "Accuracy": float(acc),
        "mAP"     : float(map),
        "BF1"     : float(bf1),
        "MCC"     : float(mcc),
        "Kappa"   : float(kc),
      }
    )

    # Optionally save predicted mask as PNG file.
    if (savePreds):
      os.makedirs(os.path.join(outputDir, "PredMasks"), exist_ok=True)
      predMaskPath = os.path.join(
        outputDir,
        "PredMasks",
        os.path.splitext(os.path.basename(imageFiles[idx]))[0] + "_PredMask.png"
      )
      # Scale prediction to [0, 255] for saving as PNG.
      predToSave = (pred2d * 255.0).astype("uint8")
      cv2.imwrite(predMaskPath, predToSave)

  if (savePreds):
    # Save the images as matplotlib figures for visualization
    # (original image, predicted mask, and optionally ground truth mask).

    for idx in range(preds.shape[0]):
      img = ReadImage(np.bytes_(imageFiles[idx]), inputSize[:2])
      img = img * 255.0 if (img.max() <= 1.0) else img  # Scale back to [0, 255] if normalized.
      img = img.astype("uint8")  # Ensure image is in uint8 format for saving.

      pred2d = np.squeeze(preds[idx])
      gt2d = np.squeeze(masksArray[idx])

      # Binarize predictions if numClasses <= 2 (assuming sigmoid output).
      if (numClasses <= 2):
        # Normalize first using min-max scaling to ensure values are in [0, 1] before thresholding.
        # minVal = np.min(pred2d)
        # maxVal = np.max(pred2d)
        # if (maxVal > minVal):
        #   pred2d = (pred2d - minVal) / (maxVal - minVal)
        # else:
        #   print("Warning: Predictions have zero variance (max == min). Skipping normalization.")
        pred2d = (pred2d > 0.5).astype("float32")

      plt.figure(figsize=(12, 4))
      plt.subplot(1, 3, 1)
      if (img.shape[2] == 1):
        plt.imshow(img[:, :, 0], cmap="gray")
      else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
      plt.title("Original Image")
      plt.axis("off")

      plt.subplot(1, 3, 2)
      plt.imshow(pred2d, cmap="gray")
      plt.title("Predicted Mask")
      plt.axis("off")

      if (gt2d is not None):
        plt.subplot(1, 3, 3)
        plt.imshow(gt2d, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

      os.makedirs(os.path.join(outputDir, "EvalFigures"), exist_ok=True)
      fileBaseNameNoExt = os.path.splitext(os.path.basename(imageFiles[idx]))[0]
      figPath = os.path.join(
        outputDir,
        "EvalFigures",
        f"{fileBaseNameNoExt}_Eval.png"
      )
      plt.tight_layout()
      plt.savefig(figPath)
      plt.close()

  # When metrics were computed save summary and print averages.
  if (len(results) > 0):
    # Create a DataFrame from the per-image results.
    df = pd.DataFrame(results)
    # Compute mean statistics across evaluated images.
    meanStats = df.mean(numeric_only=True)
    # Save CSV of per-image metrics.
    csvPath = os.path.join(outputDir, "EvaluationMetrics.csv")
    df.to_csv(csvPath, index=False)
    # Save JSON summary with averages.
    jsonPath = os.path.join(outputDir, "EvaluationSummary.json")
    with open(jsonPath, "w", encoding="utf-8") as f:
      json.dump({"Averages": meanStats.to_dict(), "Details": results}, f, indent=2)
    # Print mean metrics for convenience.
    print("Mean Dice:", float(meanStats.get("Dice", 0.0)))
    print("Mean IoU:", float(meanStats.get("IoU", 0.0)))
    print("Mean F1:", float(meanStats.get("F1", 0.0)))
    print("Mean Accuracy:", float(meanStats.get("Accuracy", 0.0)))
    print("Mean mAP:", float(meanStats.get("mAP", 0.0)))
    print("Mean BF1:", float(meanStats.get("BF1", 0.0)))
    print("Mean MCC:", float(meanStats.get("MCC", 0.0)))
    print("Mean Kappa:", float(meanStats.get("Kappa", 0.0)))
  else:
    print("No metrics were computed. Check if masks were loaded correctly and try again.")


def Run():
  # Parse arguments into configuration.
  config = ParseArgs()
  # Execute evaluation using the provided configuration.
  EvaluateModel(config)


if (__name__ == "__main__"):
  # Execute the CLI runner when the script is invoked directly.
  Run()
