import os, sys, torch, cv2, json, time
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix
from HMB.Initializations import IgnoreWarnings, SeedEverything
from HMB.PerformanceMetrics import (
  CalculatePerformanceMetrics, PlotConfusionMatrix, PlotROCAUCCurve, PlotPRCCurve, ComputeECE
)

IMAGE_SUFFIXES = {
  ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif",
  ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".TIF", ".GIF",
}


def ExtractYOLOModelSize(modelName: str) -> str:
  r'''
  Extract model size from YOLO model name. If the model name contains "n", "s", "m", "l", or "x",
  it maps to "Nano", "Small", "Medium", "Large", or "XLarge" respectively. If no match is found,
  the original model name is returned.

  Parameters:
    modelName (str): The YOLO model name to extract size from.

  Returns:
    str: Human-readable model size or original model name if no match.
  '''

  # Normalize model name and map abbreviation to human-readable size.
  modelNameLower = modelName.lower().replace("yolo", "")
  sizeMapping = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "XLarge",
  }
  for abbreviation, fullName in sizeMapping.items():
    if (abbreviation in modelNameLower):
      return fullName
  return modelName


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
    dict: Mapping of model->{category->csvPath, metrics} for quick programmatic access.
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


def EvaluatePredictPlotSubset(
  datasetDir,
  model,
  subset="test",
  prefix="",
  storageDir=None,
  heavy=True,
  computeECE=True,
  exportFailureCases=True,
  eps=1e-10,
) -> dict[str, str]:
  r'''
  Evaluate a trained Ultralytics YOLO classification model on a specified dataset subset
  (train/val/test/all), collect predictions, compute confusion matrix and performance metrics,
  and optionally save predictions to a CSV file. It also generates and saves confusion matrix,
  ROC AUC, and PRC plots.

  Parameters:
    datasetDir (str): Path to the dataset directory containing train/val/test splits.
    model (ultralytics.YOLO): Trained Ultralytics YOLO classification model.
    subset (str): Dataset subset to evaluate ("train", "val", "test", or "all"). Defaults to "test".
    prefix (str): Prefix for saved figure filenames. Defaults to "".
    storageDir (str | None): Directory to save predictions CSV and figures. If None, uses current directory. Defaults to None.
    heavy (bool): Whether to compute heavy metrics and plot ROC/PRC curves. Defaults to True.
    computeECE (bool): Whether to compute Expected Calibration Error (ECE). Defaults to True.
    exportFailureCases (bool): Whether to export misclassified samples to CSV. Defaults to True.
    eps (float): Small epsilon value for numerical stability in metric calculations. Defaults to 1e-10.

  Returns:

  '''

  print(f"Collecting predictions on {subset} split for confusion matrix computation...")
  if (subset not in ("train", "val", "test", "all")):
    raise ValueError(f"Invalid subset name: {subset}")
  if (subset == "all"):
    splitDirs = [Path(datasetDir) / split for split in ("train", "val", "test")]
  else:
    splitDirs = [Path(datasetDir) / subset]

  allPredsIndices: List[int] = []
  allGtsIndices: List[int] = []
  allPredsProbs: List[List[float]] = []
  allPredsNames: List[str] = []
  allGtsNames: List[str] = []
  allPredsConfidences: List[Optional[float]] = []
  predictionsRecords: List[Dict[str, Any]] = []
  classNames = []

  try:
    for splitDir in splitDirs:
      if (not splitDir.exists()):
        print(f"Warning: Split directory does not exist, skipping: {splitDir}")
        continue

      classDirs = sorted([directory for directory in splitDir.iterdir() if directory.is_dir()])
      if (len(classNames) == 0):
        classNames = [directory.name for directory in classDirs]
      for trueClassIndex, classDir in enumerate(classDirs):
        imageFiles = [
          p
          for p in classDir.iterdir()
          if (p.is_file() and (p.suffix.lower() in IMAGE_SUFFIXES))
        ]
        if (len(imageFiles) == 0):
          print(f"Warning: No image files found in class directory, skipping: {classDir}")
          continue
        for imagePath in imageFiles:
          # Run prediction for the image using the model.
          predictionResults = model.predict(str(imagePath), verbose=False)
          if ((len(predictionResults) > 0) and hasattr(predictionResults[0], "probs")):
            try:
              predictedClassIndex = int(predictionResults[0].probs.top1)
            except Exception:
              try:
                predictedClassIndex = int(torch.argmax(predictionResults[0].probs).item())
              except Exception:
                predictedClassIndex = -1
            allPredsIndices.append(predictedClassIndex)
            allGtsIndices.append(trueClassIndex)
            allPredsNames.append(
              classDirs[predictedClassIndex].name
              if (predictedClassIndex >= 0 and predictedClassIndex < len(classDirs))
              else "Unknown"
            )
            allGtsNames.append(classDir.name)

            prob = None
            if (len(predictionResults) > 0):
              try:
                probsAttr = predictionResults[0].probs
                if (probsAttr is None):
                  prob = None
                else:
                  probsAttr = probsAttr.data
                if (isinstance(probsAttr, torch.Tensor)):
                  prob = probsAttr.cpu().numpy().tolist()
                elif (isinstance(probsAttr, np.ndarray)):
                  prob = probsAttr.tolist()
                else:
                  prob = None
              except Exception:
                prob = None
            else:
              prob = None
            allPredsProbs.append(prob if prob is not None else [])

            # Compute confidence for the predicted class if available.
            predictedConfidence = None
            if ((prob is not None) and (predictedClassIndex is not None) and (predictedClassIndex < len(prob))):
              try:
                predictedConfidence = float(prob[predictedClassIndex])
              except Exception:
                predictedConfidence = None
            allPredsConfidences.append(predictedConfidence)
          else:
            allPredsIndices.append(-1)
            allGtsIndices.append(trueClassIndex)
            allPredsNames.append("Unknown")
            allGtsNames.append(classDir.name)
            allPredsProbs.append([])
            allPredsConfidences.append(None)

          eceValue = None
          if (computeECE and (len(allPredsProbs) > 0 and allPredsProbs[-1])):
            eceValue = ComputeECE([allPredsProbs[-1]], [allGtsIndices[-1]])

          predictionsRecords.append({
            "image"              : str(imagePath),
            "split"              : splitDir.name,
            "trueClassIndex"     : int(trueClassIndex),
            "trueClassName"      : classDir.name,
            "predictedClassIndex": allPredsIndices[-1],
            "predictedClassName" : allPredsNames[-1],
            "predictedConfidence": allPredsConfidences[-1],
            "probabilities"      : (json.dumps(allPredsProbs[-1]) if (allPredsProbs[-1]) else None),
            "ece"                : (float(eceValue) if (eceValue is not None) else None),
          })

      print(f"Prediction collection completed for split: {splitDir.name}")
      print(f"Collected predictions for {len(allGtsIndices)} samples across {len(classDirs)} classes.")
      print(f"Total samples collected for confusion matrix: {len(allGtsIndices)}")
      print(f"{'-' * 60}")

    print("Finished collecting predictions for all specified splits.")
    assert len(allPredsIndices) == len(allGtsIndices), "Mismatch in predictions and ground truths count."
    assert len(allPredsIndices) == len(allPredsProbs), "Mismatch in predictions and probabilities count."
    assert len(allPredsIndices) == len(allPredsNames), "Mismatch in predictions and names count."
    assert len(allGtsIndices) == len(allGtsNames), "Mismatch in ground truths and names count."
    assert len(allPredsIndices) == len(allPredsConfidences), "Mismatch in predictions and confidences count."
    assert len(predictionsRecords) == len(allGtsIndices), "Mismatch in prediction records and ground truths count."
    print(f"Total samples collected: {len(allGtsIndices)}")
    print(f"{'-' * 60}")

    print("Computing confusion matrix and performance metrics...")
    if ((len(allPredsIndices) > 0) and (len(allGtsIndices) > 0)):
      confusion = confusion_matrix(allGtsIndices, allPredsIndices)
      metricResults = CalculatePerformanceMetrics(
        confusion,
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
    weightedMetrics = {}
    print(f"Error during prediction collection or metric computation: {ex}")

  if (prefix):
    weightedMetrics = {f"{prefix}{key}": value for key, value in weightedMetrics.items()}

  if (storageDir is None):
    storageDir = Path(".")

  if (not os.path.exists(storageDir)):
    os.makedirs(storageDir, exist_ok=True)
    print(f"Created storage directory: {storageDir}")
  else:
    print(f"Using existing storage directory: {storageDir}")

  storageFileName = f"{prefix}_Predictions_{subset}.csv" if (prefix) else f"Predictions_{subset}.csv"
  storageFilePath = Path(storageDir) / storageFileName
  try:
    dfPreds = pd.DataFrame(predictionsRecords)
    dfPreds.to_csv(storageFilePath, index=False)
    print(f"Predictions for subset '{subset}' saved to: {storageFilePath}")
  except Exception as saveErr:
    print(f"Warning: Could not save predictions CSV: {saveErr}")

  if (exportFailureCases):
    try:
      failureRecords = []
      for i in range(len(allGtsIndices)):
        if (allGtsIndices[i] != allPredsIndices[i]):
          record = {
            "image"              : predictionsRecords[i]["image"],
            "split"              : predictionsRecords[i]["split"],
            "trueClassIndex"     : predictionsRecords[i]["trueClassIndex"],
            "trueClassName"      : predictionsRecords[i]["trueClassName"],
            "predictedClassIndex": predictionsRecords[i]["predictedClassIndex"],
            "predictedClassName" : predictionsRecords[i]["predictedClassName"],
            "predictedConfidence": predictionsRecords[i]["predictedConfidence"],
            "probabilities"      : predictionsRecords[i]["probabilities"],
            "ece"                : predictionsRecords[i]["ece"],
          }
          failureRecords.append(record)
      if (len(failureRecords) > 0):
        dfFailures = pd.DataFrame(failureRecords)
        failureFileName = f"{prefix}_Misclassified_Samples.csv" if (prefix) else "Misclassified_Samples.csv"
        failureFilePath = Path(storageDir) / failureFileName
        dfFailures.to_csv(failureFilePath, index=False)
        print(f"Misclassified samples exported to: {failureFilePath}")
      else:
        print("No misclassified samples to export.")
    except Exception as failErr:
      print(f"Warning: Could not export misclassified samples: {failErr}")

  cm = confusion_matrix(allGtsIndices, allPredsIndices)
  print("Confusion matrix computed.")
  print(cm)
  print("Class names:")
  print(classNames)
  print("All collected ground truth indices:")
  print(allGtsIndices[:10], "..." if len(allGtsIndices) > 10 else "")
  print("All collected predicted indices:")
  print(allPredsIndices[:10], "..." if len(allPredsIndices) > 10 else "")
  print("All collected predicted probabilities (first 3 samples):")
  for probs in allPredsProbs[:3]:
    print(probs)
  print(f"{'-' * 60}")
  figsPath = Path(storageDir)

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
      fileName=str(figsPath / filename),
      fontSize=15,
      annotate=True,
      figSize=(8, 8),
      colorbar=True,
      returnFig=False,
      dpi=720,
    )
    print(f"Confusion matrix figure saved to: {figsPath / filename}")
  except Exception as figErr:
    print(f"Warning: Could not generate confusion matrix figure: {figErr}")

  if (heavy):
    try:
      # Prepare data for ROC AUC curve plotting.
      filename = f"{prefix}_ROC_AUC.pdf" if (prefix) else "ROC_AUC.pdf"
      PlotROCAUCCurve(
        np.array(allGtsIndices),  # True labels.
        np.array(allPredsProbs),  # Predicted probabilities.
        classNames,  # List of class names.
        areProbabilities=True,  # Whether yPred are probabilities.
        title="ROC Curve & AUC",  # Plot title.
        figSize=(5, 5),  # Figure size.
        cmap=None,  # Colormap for ROC curves.
        display=False,  # Display the plot.
        save=True,  # Save the plot.
        fileName=str(figsPath / filename),  # File name to save.
        fontSize=15,  # Font size.
        plotDiagonal=True,  # Plot diagonal reference line.
        annotateAUC=True,  # Annotate AUC value on plot.
        showLegend=True,  # Show legend.
        returnFig=False,  # Return figure object.
        dpi=720,  # DPI for saving the figure.
      )
      print(f"ROC AUC figure saved to: {figsPath / filename}")
    except Exception as rocErr:
      print(f"Warning: Could not generate ROC AUC figure: {rocErr}")

    try:
      filename = f"{prefix}_PRC.pdf" if (prefix) else "PRC.pdf"
      PlotPRCCurve(
        allGtsIndices,  # True labels.
        allPredsProbs,  # Predicted probabilities.
        classNames,  # List of class names.
        areProbabilities=True,  # Whether yPred are probabilities.
        title="PRC Curve",  # Plot title.
        figSize=(5, 5),  # Figure size.
        cmap=None,  # Colormap for PRC curves.
        display=False,  # Display the plot.
        save=True,  # Save the plot.
        fileName=str(figsPath / filename),  # File name to save.
        fontSize=15,  # Font size.
        annotateAvg=True,  # Annotate average precision value on plot.
        showLegend=True,  # Show legend.
        returnFig=False,  # Return figure object.
        dpi=720,  # DPI for saving the figure.
      )
      print(f"PRC figure saved to: {figsPath / filename}")
    except Exception as prcErr:
      print(f"Warning: Could not generate PRC figure: {prcErr}")
  else:
    print("Heavy metrics and plots skipped as per configuration.")

  if (computeECE):
    ece = ComputeECE(allPredsProbs, allGtsIndices)
    weightedMetrics["ECE"] = ece
    print("Expected Calibration Error (ECE):", ece)

  return (
    str(storageFilePath), weightedMetrics, allPredsIndices,
    allGtsIndices, allPredsProbs, allPredsConfidences,
    predictionsRecords, classNames, cm
  )


def MeasureLatencyWithUltralytics(
  modelPath: str,
  runs: int = 20,
  warmup: int = 5,
  inputShape: Tuple[int, int] = (224, 224),
  exampleInput: Optional[np.ndarray] = None,
):
  r'''
  Measure average inference latency (in milliseconds) for an Ultralytics YOLO model or
  other supported model file formats using a single synthetic or user-provided example input.

  Supported model formats/paths:
    - Ultralytics-wrapped .pt weights or hub identifier: loaded via `ultralytics.YOLO`.
    - TorchScript file (.pt saved via torch.jit.save / scripted/traced): loaded via `torch.jit.load`.
    - ONNX file (.onnx): executed via `onnxruntime.InferenceSession` (if onnxruntime installed).

  The helper will attempt to load the model using the best available backend for the
  provided path and run warmup + timed predictions, returning average latency in ms.

  Parameters:
    modelPath (str): Path to the model file or Ultralytics model identifier.
    runs (int): Number of timed runs to average. Defaults to 20.
    warmup (int): Number of warmup runs to perform before timing. Defaults to 5.
    inputShape (Tuple[int,int]): (H, W) used to synthesize an RGB input when `exampleInput` is not provided.
    exampleInput (np.ndarray | torch.Tensor | None): Optional user-provided input. If given, this
      will be used for warmup and timed runs. Expected shape is HxWx3 or batch-like formats.

  Returns:
    float | None: Average latency in milliseconds on success, or None on failure.
  '''

  try:
    # Determine extension lower-case for format guessing.
    _, ext = os.path.splitext(str(modelPath))
    ext = ext.lower()

    # Helper to build a synthetic numpy image in HWC uint8 format.
    def _CreateNumpyRandomImage(h, w, backendName="ultralytics"):
      if (backendName == "onnx"):
        # ONNX expects NCHW float32 input; create accordingly.
        return (np.random.rand(1, 3, h, w) * 255).astype("float32")
      else:
        # Ultralytics and torchscript expect HWC uint8 image.
        return (np.random.rand(h, w, 3) * 255).astype("uint8")

    # Prepare input in three possible backends: ultralytics, torchscript, onnxruntime.
    backendName = None
    loadedModel = None
    ortSess = None

    # Try ultralytics if class available and path doesn't explicitly look like an ONNX file.
    if (ext != ".onnx"):
      try:
        loadedModel = YOLO(modelPath, task="classify", verbose=False)
        backendName = "ultralytics"
      except Exception:
        loadedModel = None
        backendName = None

    # If ultralytics not selected, try TorchScript for .pt or fallback when ultralytics failed.
    if (backendName is None) and (ext in [".pt", ".pth"]):
      try:
        tsModel = torch.jit.load(modelPath, map_location="cpu")
        loadedModel = tsModel
        backendName = "torchscript"
      except Exception:
        loadedModel = None
        backendName = None

    # If .onnx file, try onnxruntime.
    if (backendName is None) and (ext == ".onnx"):
      try:
        import onnxruntime as ort
        ortSess = ort.InferenceSession(modelPath, providers=["CPUExecutionProvider"])
        backendName = "onnx"
      except Exception:
        ortSess = None
        backendName = None

    # If still no backend identified, attempt a last-resort ultralytics instantiation.
    if (backendName is None):
      try:
        loadedModel = YOLO(modelPath, task="classify", verbose=False)
        backendName = "ultralytics"
      except Exception:
        backendName = None

    if (backendName is None):
      # Unsupported model type or required runtime not available.
      return None

    # Prepare the input for warmup and timing.
    if (exampleInput is not None):
      exampleInp = exampleInput
    else:
      h, w = inputShape
      exampleInp = _CreateNumpyRandomImage(h, w, backendName=backendName)

    # Warmup runs to stabilize runtime performance.
    for _ in range(max(1, warmup)):
      try:
        if (backendName == "ultralytics"):
          # Ultralytics accepts numpy HWC or path; pass imgsz when we generated the image.
          loadedModel.predict(
            exampleInp,
            imgsz=max(inputShape) if (exampleInput is None) else None,
            verbose=False
          )
        elif (backendName == "torchscript"):
          # Ensure tensor on CPU and in NCHW float format.
          if (not isinstance(exampleInp, torch.Tensor)):
            if (isinstance(exampleInp, np.ndarray) and exampleInp.ndim == 3):
              tensorInp = torch.from_numpy(exampleInp).permute(2, 0, 1).unsqueeze(0).float()
            elif (isinstance(exampleInp, np.ndarray) and exampleInp.ndim == 4):
              tensorInp = torch.from_numpy(exampleInp).permute(0, 3, 1, 2).float() if (
                exampleInp.shape[-1] == 3) else torch.from_numpy(exampleInp).float()
            else:
              try:
                tensorInp = torch.tensor(exampleInp)
              except Exception:
                tensorInp = None
            if (tensorInp is None):
              continue
          else:
            tensorInp = exampleInp
          # Run the torchscript model.
          loadedModel(tensorInp)
        elif (backendName == "onnx"):
          # ONNX runtime expects numpy inputs; get input name and run.
          try:
            inName = ortSess.get_inputs()[0].name
            ortSess.run(None, {inName: exampleInp})
          except Exception:
            pass
      except Exception:
        # Ignore per-iteration errors during warmup to keep the benchmark robust.
        pass

    # Timed runs to measure average latency.
    startTime = time.perf_counter()
    for _ in range(max(1, runs)):
      try:
        if (backendName == "ultralytics"):
          loadedModel.predict(
            exampleInp,
            imgsz=max(inputShape) if (exampleInput is None) else None,
            verbose=False
          )
        elif (backendName == "torchscript"):
          import torch
          if (not isinstance(exampleInp, torch.Tensor)):
            if (isinstance(exampleInp, np.ndarray) and exampleInp.ndim == 3):
              tensorInp = torch.from_numpy(exampleInp).permute(2, 0, 1).unsqueeze(0).float()
            elif (isinstance(exampleInp, np.ndarray) and exampleInp.ndim == 4):
              tensorInp = (
                torch.from_numpy(exampleInp).permute(0, 3, 1, 2).float()
                if (exampleInp.shape[-1] == 3)
                else torch.from_numpy(exampleInp).float()
              )
            else:
              try:
                tensorInp = torch.tensor(exampleInp)
              except Exception:
                tensorInp = None
            if (tensorInp is None):
              continue
          else:
            tensorInp = exampleInp
          loadedModel(tensorInp)
        elif (backendName == "onnx"):
          inName = ortSess.get_inputs()[0].name
          ortSess.run(None, {inName: exampleInp})
      except Exception:
        # Ignore individual run errors to keep benchmark robust.
        pass
    duration = time.perf_counter() - startTime

    avgMs = (duration / max(1, runs)) * 1000.0
    return float(avgMs)
  except Exception:
    return None


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


def ExportYOLO2TorchScript(
  weightsPath: str,
  outPath: str,
  imgsz: int = 224,
  device: str = "cpu"
) -> Optional[str]:
  r'''
  Export a classification model to a TorchScript file.

  This helper attempts to load a model via Ultralytics' YOLO wrapper when available.
  It extracts the underlying core model, moves it to the requested device, and
  attempts to produce a TorchScript artifact by tracing with a dummy input first
  and falling back to scripting when tracing is not possible.

  Parameters:
    weightsPath (str): Path to weights or a model identifier understood by Ultralytics.
    outPath (str): Desired output path for the TorchScript file (including filename).
    imgsz (int): Spatial size used for the dummy trace input. Defaults to 224.
    device (str): Torch device string to use for the conversion. Defaults to "cpu".

  Returns:
    str | None: Returns the output path on success, otherwise None.
  '''

  try:
    # Instantiate the YOLO classifier wrapper.
    try:
      yoloModel = YOLO(weightsPath, task="classify", verbose=False)
    except Exception:
      return None

    # Prepare dummy input on the requested device.
    deviceT = torch.device(device)
    dummy = torch.randn(1, 3, imgsz, imgsz).to(deviceT)

    # Attempt to extract the core PyTorch module from the wrapper.
    try:
      coreModel = yoloModel.model
      coreModel.to(deviceT)
      coreModel.eval()

      # Try tracing first, and fall back to scripting when tracing fails.
      try:
        traced = torch.jit.trace(coreModel, dummy, strict=False)
      except Exception:
        traced = torch.jit.script(coreModel)

      traced.save(str(outPath))
      return str(outPath)
    except Exception:
      return None
  except Exception:
    return None


def ExportYOLO2ONNX(weightsPath: str, outPath: str, imgsz: int = 224, opset: int = 12) -> Optional[str]:
  r'''
  Export model to ONNX using Ultralytics' export helper when available.

  The function will attempt to call `model.export(format="onnx")` on a YOLO wrapper.
  If Ultralytics returns a path or list of paths, the first candidate is copied to
  the requested output location for convenience.

  Parameters:
    weightsPath (str): Path or identifier understood by Ultralytics.
    outPath (str): Desired ONNX output path including filename.
    imgsz (int): Input spatial size to request from the exporter. Defaults to 224.
    opset (int): ONNX opset version to request. Defaults to 12.

  Returns:
    str | None: Returns the output path on success, otherwise None.
  '''

  try:
    # Instantiate the YOLO classifier wrapper.
    try:
      yoloModel = YOLO(weightsPath, task="classify", verbose=False)
    except Exception:
      return None

    # Attempt export using Ultralytics' helper.
    try:
      exported = yoloModel.export(format="onnx", imgsz=imgsz, opset=opset)

      # Normalize exported return type to a candidate path string.
      if (isinstance(exported, (list, tuple))):
        candidate = exported[0] if (len(exported) > 0) else None
      else:
        candidate = exported

      if (not candidate):
        return None

      if (os.path.exists(str(candidate))):
        import shutil
        shutil.copy2(str(candidate), str(outPath))
        return str(outPath)

      return None
    except Exception:
      return None
  except Exception:
    return None


def ApplyYOLOPruning(weightsPath: str, outPath: str, sparsity: float = 0.5) -> Optional[str]:
  r'''
  Apply global unstructured pruning to Conv2d and Linear weights and save state_dict.

  This helper loads a classifier via the Ultralytics YOLO wrapper when available.
  It collects Conv2d and Linear weight tensors and applies global L1 unstructured
  pruning to the requested sparsity fraction. The function removes pruning
  reparametrizations to make weights dense and then saves the resulting
  state_dict to the requested output path.

  Parameters:
    weightsPath (str): Path to weights or model identifier understood by Ultralytics.
    outPath (str): Path where the pruned state_dict will be saved.
    sparsity (float): Fraction of weights to remove in [0,1). Defaults to 0.5.

  Returns:
    str | None: Returns the output path on success, otherwise None.
  '''

  try:
    # Instantiate the wrapper for classification.
    try:
      modelWrapper = YOLO(weightsPath, task="classify", verbose=False)
    except Exception:
      return None

    # Extract the core PyTorch model from the wrapper.
    try:
      model = modelWrapper.model
    except Exception:
      return None

    # Import pruning utilities from torch.
    import torch.nn.utils.prune as prune

    # Validate sparsity value and coerce to float.
    try:
      amount = float(sparsity)
      if (amount < 0.0 or amount >= 1.0):
        # Accept exact zero to indicate no pruning.
        if (amount == 0.0):
          amount = 0.0
        else:
          return None
    except Exception:
      return None

    # Collect module weight parameters to prune.
    paramsToPrune = []
    for _, module in model.named_modules():
      if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
        if (hasattr(module, "weight")):
          paramsToPrune.append((module, "weight"))

    # Abort when no prunable parameters are found.
    if (len(paramsToPrune) == 0):
      return None

    # Apply global unstructured pruning with L1 criterion.
    try:
      prune.global_unstructured(paramsToPrune, pruning_method=prune.L1Unstructured, amount=amount)
    except Exception:
      return None

    # Remove pruning reparametrizations to leave dense tensors.
    for module, name in paramsToPrune:
      try:
        prune.remove(module, name)
      except Exception:
        # Ignore failures to remove reparametrization for robustness.
        pass

    # Ensure output directory exists and save state_dict.
    try:
      outP = Path(outPath)
      outP.parent.mkdir(parents=True, exist_ok=True)
      torch.save(model.state_dict(), str(outP))
      return str(outP)
    except Exception:
      return None
  except Exception:
    return None
