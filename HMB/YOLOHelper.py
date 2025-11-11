import os, sys, torch, cv2, json
from tqdm import tqdm
import numpy as np
import pandas as pd
from ultralytics import YOLO
from typing import List, Optional, Tuple
from sklearn.metrics import confusion_matrix
from HMB.Initializations import IgnoreWarnings
from HMB.Initializations import SeedEverything
from HMB.PerformanceMetrics import CalculatePerformanceMetrics


def TrainMultipleYoloClassifiers(
  datasetPath,
  baseDir,
  runsDir,
  targetModels=None,
  epochs=250,
  batchSize=128,
  inputShape=(512, 512),
  trialNum=1,
  exportOnnx=True,
  onnxOpset=11,
  deviceEnvVars=None,
  seed=None,
  overwriteExisting=False,
  enablePlots=True,
  enableSave=True,
):
  r'''
  High-level function to train multiple YOLO classification models on a given dataset.
  It handles environment setup, model training, validation, and optional ONNX export.
  Also, it saves top-1 and top-5 metrics to text files for each model.
  Moreover, it saves the trained model in Keras format.
  Finally, it manages exceptions to ensure robustness across multiple models.

  Parameters:
    datasetPath (str): Path to the dataset folder containing train/val splits.
    baseDir (str): Directory to save experiment outputs.
    runsDir (str): Subdirectory under baseDir to store run outputs.
    targetModels (List[str], optional): List of YOLO model keywords to train. Defaults to None, which uses a predefined set.
    epochs (int, optional): Number of training epochs. Defaults to 250.
    batchSize (int, optional): Batch size for training. Defaults to 128.
    inputShape (Tuple[int, int], optional): Input image shape (height, width). Defaults to (512, 512).
    trialNum (int, optional): Trial number for naming runs. Defaults to 1.
    exportOnnx (bool, optional): Whether to export trained models to ONNX format. Defaults to True.
    onnxOpset (int, optional): ONNX opset version for export. Defaults to 11.
    deviceEnvVars (dict, optional): Environment variables for device configuration. Defaults to None.
    seed (int, optional): Random seed for reproducibility. Defaults to None.
    overwriteExisting (bool, optional): Whether to overwrite existing runs. Defaults to False.
    enablePlots (bool, optional): Whether to enable training plots. Defaults to True.
    enableSave (bool, optional): Whether to save the best model. Defaults to True.
  '''

  # Ensure default model list is provided when None.
  if (targetModels is None):
    targetModels = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

  # Set default device environment variables if provided.
  if (deviceEnvVars is None):
    deviceEnvVars = {
      "KMP_DUPLICATE_LIB_OK": "TRUE",
      "CUDA_VISIBLE_DEVICES": "0",
      "CUDA_LAUNCH_BLOCKING": "1",
      "TORCH_USE_CUDA_DSA"  : "1",
    }

  # Apply environment variables to the process.
  for envKey, envVal in deviceEnvVars.items():
    # Set or overwrite the environment variable.
    os.environ[envKey] = str(envVal)

  # Optionally seed randomness for reproducibility.
  if (seed is not None):
    SeedEverything(seed=seed)

  # Suppress warnings if the project's helper exists.
  IgnoreWarnings()

  # Iterate over requested models and train each one.
  for modelKeyword in targetModels:
    print(f"\nTraining Model: {modelKeyword} ...")
    try:
      expOutputDir = os.path.join(baseDir, runsDir, f"{modelKeyword}-cls-{trialNum}")
      os.makedirs(expOutputDir, exist_ok=True)

      # Instantiate the YOLO classification model from the Ultralytics hub.
      model = YOLO(f"{modelKeyword}-cls.pt", task="classify")

      # Train the model with provided settings.
      model.train(
        data=datasetPath,  # Path to dataset splits.
        batch=batchSize,  # Batch size for training.
        epochs=epochs,  # Number of epochs.
        imgsz=inputShape[0],  # Input image size (height).
        plots=True,  # Enable training plots.
        save=True,  # Save the best model.
        name=f"{modelKeyword}-cls-{trialNum}",  # Naming convention for the run.
        project=runsDir,  # Project directory for outputs.
        exist_ok=overwriteExisting,  # Overwrite existing runs if specified.
      )

      # Retrieve class names from the trained model.
      classes = model.names

      # Validate the model on the splits folder.
      metrics = model.val(
        data=datasetPath,  # Path to dataset splits.
        plots=enablePlots,  # Enable validation plots.
        save=enableSave,  # Save predictions.
        name=f"{modelKeyword}-cls-{trialNum}",  # Naming convention for the run.
        project=runsDir,  # Project directory for outputs.
        exist_ok=overwriteExisting,  # Overwrite existing runs if specified.
      )

      # Store top-1 and top-5 metrics for later use if needed as text files.
      # Ensure the output directory exists.
      os.makedirs(expOutputDir, exist_ok=True)

      # Save top-1 metrics to a text file.
      with open(os.path.join(expOutputDir, "Top1-Metrics.txt"), "w") as f:
        f.write(str(metrics.top1))
      # Save top-5 metrics to a text file.
      with open(os.path.join(expOutputDir, "Top5-Metrics.txt"), "w") as f:
        f.write(str(metrics.top5))

      print("Top-1 Metrics:", metrics.top1)
      print("Top-5 Metrics:", metrics.top5)

      # Save the trained model in Keras format if supported.
      model.save(os.path.join(expOutputDir, "model.keras"))

      # Optionally export to ONNX format; wrap in try/except to avoid failing the loop.
      if (exportOnnx):
        try:
          model.export(
            format="onnx",
            opset=onnxOpset,
            dynamic=False,
            simplify=True,
            name=f"{modelKeyword}-cls-{trialNum}",
            project=runsDir,
            exist_ok=True,
          )
        except Exception as e:
          print(f"Warning: ONNX export failed for {modelKeyword}: {e}")

    except Exception as e:
      print(f"Error training or processing model {modelKeyword}: {e}")
      # Continue with next model on error.
      continue


def EvaluateAndSaveYoloClassifications(
  baseDir,
  datasetPath,
  runsDir,
  extensions=None,
  inputShape=(512, 512),
  trialNum=1,
  categories=None,
  targetModels=None,
):
  r'''
  Evaluate trained YOLO classification models on dataset splits and save CSV results.
  It computes confusion matrices and performance metrics for each model and category.
  The results are saved in CSV format under the specified results subfolder.
  Also, it returns a summary mapping for programmatic access.

  Parameters.
    baseDir (str): Root experiment directory containing runs and dataset.
    datasetPath (str): Path to the dataset containing category subfolders.
    runsDir (str): Subdirectory under baseDir to store evaluation results.
    extensions (List[str] | None): Allowed file extensions for images.
    inputShape (Tuple[int,int]): Input image size as (height, width).
    trialNum (int): Trial number used in run naming.
    categories (List[str] | None): Categories to evaluate, defaults to ['val','test','train'].
    targetModels (List[str] | None): Model list to evaluate, defaults to common yolo11 variants.

  Returns.
    summary (dict): Mapping of model->{category->csvPath, metrics} for quick programmatic access.
  '''

  # Default allowed image extensions when None provided.
  if (extensions is None):
    extensions = ["tiff", "tif", "jpeg", "jpg", "png", "bmp"]

  # Default categories when None provided.
  if (categories is None):
    categories = ["val", "test", "train"]

  # Default target models when None provided.
  if (targetModels is None):
    targetModels = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

  # Prepare a container to return run summaries.
  summary = {}

  # Iterate over requested categories and models.
  for cat in categories:
    print(f"\nEvaluating Category: {cat} ...")
    # Create category-level summary mapping.
    for modelKeyword in targetModels:
      print(f"\nWorking on Model: {modelKeyword} ...")
      # Create a per-model dictionary for results.
      modelSummary = {}

      # Compose the expected weights path for this run.
      weights = os.path.join(
        baseDir, "runs", "classify",
        f"{modelKeyword}-cls-{trialNum}",
        "weights", "best.pt"
      )

      # Check weights existence and continue if missing.
      if (not os.path.exists(weights)):
        # Report missing weights and continue to next model.
        print(f"Weights file not found: {weights}.")
        modelSummary[cat] = {"csv": None, "error": "weights_missing"}
        summary[modelKeyword] = summary.get(modelKeyword, {})
        summary[modelKeyword].update(modelSummary)
        continue

      # Load the model for classification task.
      try:
        # Load model in classify mode.
        model = YOLO(weights, task="classify", verbose=False)
      except Exception as e:
        print(f"Failed to load model {weights}: {e}.")
        modelSummary[cat] = {"csv": None, "error": "load_failed", "exception": str(e)}
        summary[modelKeyword] = summary.get(modelKeyword, {})
        summary[modelKeyword].update(modelSummary)
        continue

      # Prepare history list for building a DataFrame.
      history = []

      # Iterate over classes reported by the model.
      for classKey, className in tqdm(list(model.names.items()), desc=f"Classes {modelKeyword}"):
        # Compute directory containing images for this class and category.
        classDir = os.path.join(datasetPath, cat, className)

        # Skip if class directory does not exist.
        if (not os.path.isdir(classDir)):
          # Warn and continue when expected class folder is missing.
          print(f"Warning: expected folder missing: {classDir}.")
          continue

        # List files in the class directory.
        try:
          files = os.listdir(classDir)
        except Exception as e:
          # If listing fails, skip this class.
          print(f"Failed to list files in {classDir}: {e}.")
          continue

        # Iterate over files found in the class folder.
        for fileName in tqdm(files, desc=f"Processing {className}"):
          # Get file extension for filtering.
          fileExt = fileName.split('.')[-1].lower()

          # Skip files with unsupported extensions.
          if (fileExt not in extensions):
            continue

          # Build the full image path.
          imgPath = os.path.join(classDir, fileName)

          # Run the model on the image and collect probabilities.
          try:
            pred = model(imgPath, imgsz=inputShape[0], verbose=False)
            # Extract probabilities as a CPU numpy array when available.
            toValues = None
            try:
              toValues = pred[0].probs.data.cpu().numpy()
            except Exception:
              # Fallback: try to get probs from different attribute name.
              try:
                toValues = pred[0].prob.data.cpu().numpy()
              except Exception:
                # If probabilities are not available, skip storing per-class probabilities.
                toValues = None

            # Determine predicted top1 index in a robust way.
            try:
              predictedIndex = int(pred[0].probs.top1)
              predictedProb = float(pred[0].probs.values[0])
            except Exception:
              try:
                predictedIndex = int(np.argmax(toValues)) if (toValues is not None) else -1
              except Exception:
                predictedIndex = -1
              try:
                predictedProb = (
                  float(toValues[predictedIndex])
                  if (toValues is not None and predictedIndex >= 0)
                  else 0.0
                )
              except Exception:
                predictedProb = 0.0

            # Compose history record with image path, actual class and predicted info.
            record = {
              "Image"          : imgPath,
              "Actual"         : className,
              "Predicted"      : model.names.get(predictedIndex, "unknown"),
              "Actual Index"   : int(classKey),
              "Predicted Index": int(predictedIndex),
              "Predicted Prob" : float(predictedProb),
            }

            # Add per-class probabilities when available.
            if (toValues is not None):
              for i in range(len(toValues)):
                record[f"{model.names.get(i, str(i))} Prob"] = float(toValues[i])

            # Append record to history list.
            history.append(record)

          except Exception as e:
            # On failure to predict, log and continue to next image.
            print(f"Prediction failed for image {imgPath}: {e}.")
            continue

      # Build a DataFrame from collected history.
      df = pd.DataFrame(history)

      # Compose storage path for the CSV file.
      csvFileName = rf"{modelKeyword} Classification-{cat.capitalize()}.csv"
      storagePath = os.path.join(baseDir, runsDir, csvFileName)

      # Ensure the results folder exists.
      os.makedirs(os.path.dirname(storagePath), exist_ok=True)

      # Save the DataFrame to CSV.
      df.to_csv(storagePath, index=False)

      # Reload the CSV to ensure consistent types for metrics computation.
      df = pd.read_csv(storagePath)

      # Compute reference and prediction arrays for metrics.
      references = np.array(list(df["Actual Index"].values))
      predictions = np.array(list(df["Predicted Index"].values))

      # Print shapes and a small sample of references and predictions.
      print(f"Shape of references: {references.shape}.")
      print(f"Shape of predictions: {predictions.shape}.")
      print(f"Sample references: {references[:5]}.")
      print(f"Sample predictions: {predictions[:5]}.")

      # Compute confusion matrix for the set.
      try:
        cm = confusion_matrix(references, predictions)
      except Exception as e:
        # If confusion matrix computation fails, store the error and continue.
        print(f"Confusion matrix computation failed: {e}.")
        modelSummary[cat] = {"csv": storagePath, "error": "cm_failed", "exception": str(e)}
        summary[modelKeyword] = summary.get(modelKeyword, {})
        summary[modelKeyword].update(modelSummary)
        continue

      # Print the confusion matrix.
      print("Confusion Matrix:")
      print(cm)

      # Compute additional performance metrics using the project's helper.
      try:
        metrics = CalculatePerformanceMetrics(cm, addWeightedAverage=True)
      except Exception as e:
        # Fallback: store the exception if metrics calculation fails.
        print(f"CalculatePerformanceMetrics failed: {e}.")
        modelSummary[cat] = {"csv": storagePath, "cm": cm.tolist(), "metrics_error": str(e)}
        summary[modelKeyword] = summary.get(modelKeyword, {})
        summary[modelKeyword].update(modelSummary)
        continue

      # Print the derived metrics to stdout.
      print(f"Performance Metrics for {modelKeyword} on {cat} set:")
      for key, value in metrics.items():
        print(f"{key}: {value}")

      # Populate model summary with results and metrics.
      modelSummary[cat] = {"csv": storagePath, "cm": cm.tolist(), "metrics": metrics}
      summary[modelKeyword] = summary.get(modelKeyword, {})
      summary[modelKeyword].update(modelSummary)

  # Save the summary mapping as a JSON file for reference.
  summaryPath = os.path.join(baseDir, runsDir, "Classification_Summary.json")
  with open(summaryPath, "w") as f:
    json.dump(summary, f, indent=4)

  # Return the summary mapping for programmatic inspection.
  return summary


if __name__ == "__main__":
  # Generate a random seed for this run.
  rndNumber = np.random.randint(0, 10000)
  # Compute splits and experiment paths used in the example.
  datasetPath = r"/home/hmbala01/[B] Bladder/Data/Data"
  expDir = r"/home/hmbala01/[B] Bladder"

  # Call the high-level training function with defaults.
  TrainMultipleYoloClassifiers(
    baseDir=baseDir,
    datasetPath=datasetPath,
    runsDir="runs/classify",
    epochs=250,
    batchSize=128,
    inputShape=(512, 512),
    trialNum=1,
    targetModels=None,
    exportOnnx=True,
    onnxOpset=11,
    seed=rndNumber,
  )

  # Call the evaluation and saving function with defaults.
  summary = EvaluateAndSaveYoloClassifications(
    baseDir=baseDir,
    datasetPath=datasetPath,
    runsDir="results/classify",
    extensions=None,
    inputShape=(512, 512),
    trialNum=1,
    categories=None,
    targetModels=None,
  )
