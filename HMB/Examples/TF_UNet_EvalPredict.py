import os, warnings, builtins, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from HMB.TFUNetHelper import GetUNetModel
from HMB.ImagesHelper import ReadImage, ReadMask, IMAGE_SUFFIXES
from HMB.ImageSegmentationMetrics import (
  ComputeDice,
  ComputeIoU,
  ComputeF1Score,
  ComputePixelAccuracy,
)

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
  # Import argparse locally to avoid global import side-effects.
  import argparse

  # Create an argument parser for the evaluation script.
  parser = argparse.ArgumentParser(description="TensorFlow UNet evaluation and prediction script.")

  # Add arguments for model, weights, and data paths.
  parser.add_argument("--ModelName", type=str, default="UNet", help="Model name to instantiate.")
  parser.add_argument("--ModelWeights", type=str, required=True, help="Path to the model weights file.")
  parser.add_argument("--DataDir", type=str, required=True, help="Path to the folder containing images.")
  parser.add_argument("--MasksDir", type=str, default="", help="Path to the folder containing masks (optional).")
  parser.add_argument("--OutputDir", type=str, default="Output", help="Directory to store predictions and metrics.")
  parser.add_argument("--InputSize", type=int, nargs=3, default=[256, 256, 3], help="Input size H W C.")
  parser.add_argument("--BatchSize", type=int, default=8, help="Batch size for prediction.")
  parser.add_argument("--NumClasses", type=int, default=1, help="Number of segmentation classes.")
  parser.add_argument("--SavePredictions", action="store_true", help="Whether to save predicted masks as PNG files.")

  # Parse arguments.
  args = parser.parse_args()

  # Build the keys configuration dictionary for downstream usage.
  config = {
    "ModelName"      : args.ModelName,
    "ModelWeights"   : args.ModelWeights,
    "DataDir"        : args.DataDir,
    "MasksDir"       : args.MasksDir,
    "OutputDir"      : args.OutputDir,
    "InputSize"      : tuple(args.InputSize),
    "BatchSize"      : args.BatchSize,
    "NumClasses"     : args.NumClasses,
    "SavePredictions": args.SavePredictions,
  }

  # Ensure the output directory exists.
  os.makedirs(config.get("OutputDir", "Output"), exist_ok=True)

  # Return the parsed configuration.
  return config


# Build a tf.data.Dataset for images (and optional masks).
def BuildDataset(imagesList, masksList=None, inputSize=(256, 256, 3), batchSize=8):
  # Create a TensorFlow dataset from image and optional mask paths.
  if (masksList is None):
    dataset = tf.data.Dataset.from_tensor_slices(imagesList)
  else:
    dataset = tf.data.Dataset.from_tensor_slices((imagesList, masksList))

  # Define a parse function to read images and masks via numpy_function.
  def _ParseSingle(*paths):
    # If masks are not provided parse only the image path.
    if (len(paths) == 1):
      imgPath = paths[0]
      # Read image using helper that expects bytes.
      img = tf.numpy_function(ReadImage, [imgPath, inputSize[:2]], tf.float32)
      # Set shape information for TF graph.
      img.set_shape((inputSize[0], inputSize[1], inputSize[2]))
      return img
    else:
      imgPath, maskPath = paths
      # Read image and mask using helpers.
      img = tf.numpy_function(ReadImage, [imgPath, inputSize[:2]], tf.float32)
      mask = tf.numpy_function(ReadMask, [maskPath, inputSize[:2]], tf.float32)
      # Set static shapes to help TF runtime.
      img.set_shape((inputSize[0], inputSize[1], inputSize[2]))
      mask.set_shape((inputSize[0], inputSize[1], 1))
      return img, mask

  # Map parse function onto the dataset.
  if (masksList is None):
    dataset = dataset.map(lambda p: _ParseSingle(p))
  else:
    dataset = dataset.map(lambda a, b: _ParseSingle(a, b))

  # Batch the dataset for prediction.
  dataset = dataset.batch(batchSize)

  # Prefetch a small number of batches to improve throughput.
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  # Return the prepared dataset.
  return dataset


# Evaluate the model on a dataset with masks and optionally save predictions.
def EvaluateModel(config):
  # Extract configuration values into local variables for clarity.
  modelName = config.get("ModelName")
  modelWeights = config.get("ModelWeights")
  dataDir = config.get("DataDir")
  masksDir = config.get("MasksDir")
  outputDir = config.get("OutputDir")
  inputSize = config.get("InputSize")
  batchSize = config.get("BatchSize")
  numClasses = config.get("NumClasses")
  savePreds = config.get("SavePredictions")

  # Collect image file paths from the provided data directory.
  allFiles = sorted(os.listdir(dataDir))
  imageFiles = [os.path.join(dataDir, f) for f in allFiles if (f.endswith(tuple(IMAGE_SUFFIXES)))]

  # Optionally collect mask paths when masksDir is provided.
  masksList = None
  if (masksDir and len(masksDir) > 0):
    # Build mask path list that corresponds to image filenames when possible.
    masksList = [os.path.join(masksDir, os.path.basename(p)) for p in imageFiles]

  # Build the evaluation dataset.
  dataset = BuildDataset(imageFiles, masksList, inputSize=inputSize, batchSize=batchSize)

  # Instantiate the model using the factory helper.
  model = GetUNetModel(modelName, inputChannels=inputSize[2], numClasses=numClasses)

  # Load provided weights into the model.
  model.load_weights(modelWeights)

  # Run predictions on the dataset to obtain model outputs.
  preds = model.predict(dataset)

  # If masks are available, evaluate metrics per image.
  results = []
  if (masksList is not None):
    # Load masks into memory for metric computation.
    masksArray = [ReadMask(np.bytes_(p), inputSize[:2]) for p in masksList]
    # Iterate through predicted batches and compare with ground truth masks.
    idx = 0
    for batchPred in preds:
      # Iterate through each prediction in the batch.
      for p in batchPred:
        # Squeeze prediction to 2D if single-channel.
        pred2d = np.squeeze(p)
        # Retrieve the corresponding ground truth mask.
        gt = masksArray[idx]
        # Squeeze ground truth to 2D.
        gt2d = np.squeeze(gt)
        # Compute Dice coefficient for the sample.
        dice = ComputeDice(pred2d, gt2d)
        # Compute IoU for the sample.
        iou = ComputeIoU(pred2d, gt2d)
        # Compute F1 score for the sample.
        f1 = ComputeF1Score(pred2d, gt2d)
        # Compute pixel accuracy for the sample.
        acc = ComputePixelAccuracy(pred2d, gt2d)
        # Append metrics to results.
        results.append(
          {"Image"   : os.path.basename(imageFiles[idx]), "Dice": float(dice), "IoU": float(iou), "F1": float(f1),
           "Accuracy": float(acc)})
        # Optionally save predicted mask as PNG file.
        if (savePreds):
          # Convert prediction to uint8 mask (0-255) using threshold 0.5.
          maskOut = (pred2d > 0.5).astype("uint8") * 255
          # Determine output path and write file.
          outPath = os.path.join(outputDir, os.path.basename(imageFiles[idx]).replace(".", "_pred."))
          cv2.imwrite(outPath, maskOut)
        # Increment sample index.
        idx += 1

  else:
    # If no masks provided optionally save raw predictions only.
    idx = 0
    for batchPred in preds:
      for p in batchPred:
        pred2d = np.squeeze(p)
        if (savePreds):
          # Convert prediction probabilities to uint8 PNG.
          outMap = (pred2d * 255.0).astype("uint8")
          outPath = os.path.join(outputDir, os.path.basename(imageFiles[idx]).replace(".", "_pred."))
          cv2.imwrite(outPath, outMap)
        idx += 1

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


def Run():
  # Parse arguments into configuration.
  config = ParseArgs()
  # Execute evaluation using the provided configuration.
  EvaluateModel(config)


if (__name__ == "__main__"):
  # Execute the CLI runner when the script is invoked directly.
  Run()
