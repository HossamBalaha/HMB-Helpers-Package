import torch, os, argparse, json, builtins, shutil, tqdm
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from HMB.PyTorchHelper import LoadModel
from HMB.PyTorchTabularModelsZoo import GetModel
from HMB.DatasetsHelper import TabularPreprocessor
from HMB.PyTorchClassificationLosses import FocalLossAlt
from HMB.Initializations import UpdateMatplotlibSettings, DoRandomSeeding
from HMB.PyTorchTrainingPipeline import PyTorchClassificationTrainingPipeline
from HMB.PyTorchTrainingPipeline import GenericTabularEvaluatePredictPlotSubset
from HMB.Utils import ReadProjectConfig, SafeParseProbabilities, SimpleSerializeForJson
from HMB.ExplainabilityHelper import ComputeShapValues, ShapSummaryPlot, TrainSurrogateTree
from HMB.StatisticalAnalysisHelper import ExtractDataFromSummaryFile, PlotMetrics, StatisticalAnalysis
from HMB.PerformanceMetrics import (
  CalculatePerformanceMetrics, PlotMultiTrialROCAUC,
  PlotMultiTrialPRCurve, HistoryPlotter
)

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


def GetArgs():
  parser = argparse.ArgumentParser(description="Run the ML pipeline with configurable settings.")
  parser.add_argument(
    "--dataDir",
    type=str,
    help="Directory containing the dataset.",
    required=True,
  )
  parser.add_argument(
    "--saveDir",
    type=str,
    default="Experiments",
    help=(
      "Directory to save models and artifacts. "
      "Defaults to the same directory name as the dataset but prefixed with 'Results_'."
    )
  )
  parser.add_argument(
    "--configPath",
    type=str,
    default="PyTorch_Tabular_CSV_Pipeline.json",
    help="Path to a JSON file containing configuration settings. Overrides command-line arguments."
  )
  parser.add_argument(
    "--labelColumn",
    type=str,
    default="Label",
    help="Name of the column in the dataset that contains the target labels."
  )
  parser.add_argument(
    "--dropFirstColumn",
    action="store_true",
    help="Whether to drop the first column of the dataset, which may contain non-informative indices or IDs."
  )
  parser.add_argument(
    "--noOfTrials",
    type=int,
    default=1,
    help="Number of independent trials to run for each dataset (creates separate output dirs)."
  )
  parser.add_argument(
    "--phase",
    type=str,
    default="training",
    choices=("training", "statistical", "fused", "explain"),
    help=(
      "Phase to run: 'training' runs the full training pipeline, 'statistical' only computes dataset statistics, "
      "'fused' runs the fusion phase across saved model results, 'explain' runs model-level explainability (SHAP/Surrogate)."
    )
  )
  parser.add_argument(
    "--addFusedModel",
    action="store_true",
    help="Whether to add a fused model that combines predictions from multiple individual models for ensembling."
  )
  parser.add_argument(
    "--fusedModelOnly",
    action="store_true",
    help="Whether to only run the fused model phase without running individual model training or statistical analysis."
  )
  args = parser.parse_args()
  return args


def ValidateArgs(args):
  # Validate data directory exists
  if (not os.path.isdir(args.dataDir)):
    raise ValueError(f"Data directory does not exist: {args.dataDir}")

  # Validate config file exists when provided
  if (args.configPath and not os.path.isfile(args.configPath)):
    raise ValueError(f"Config file does not exist: {args.configPath}")

  # Validate number of trials is a positive integer.
  if (hasattr(args, "noOfTrials")):
    try:
      ntr = int(args.noOfTrials)
    except Exception:
      raise ValueError(f"--noOfTrials must be an integer, got: {args.noOfTrials}")
    if (ntr < 1):
      raise ValueError(f"--noOfTrials must be >= 1, got: {ntr}")

  # Validate phase argument.
  if (hasattr(args, "phase") and args.phase not in ("training", "statistical", "fused", "explain")):
    raise ValueError(f"--phase must be one of 'training', 'statistical', 'fused', or 'explain', got: {args.phase}")


def WeightedMajorityVote(
  predsList,
  weightsList,
  probsList=None,
  uniqueClasses=None,
  verbose=False,
):
  # Convert the input weights list into a numpy array for mathematical operations.
  weightsArr = np.array(weightsList, dtype=float)
  if (verbose):
    print(f"Performing weighted majority vote with predictions: {predsList} and weights: {weightsArr}")
    print(f"Unique classes considered for voting: {uniqueClasses}")
  # Create an accumulator array initialized to zero for weighted class votes.
  classVotes = np.zeros(len(uniqueClasses), dtype=float)
  # Calculate the total sum of the provided weights for normalization.
  totalWeight = weightsArr.sum()
  # Handle cases where weights sum to zero or contain invalid values.
  if ((totalWeight == 0) or (not np.isfinite(totalWeight))):
    # Assign uniform weights to all models when the sum is invalid.
    weightsArr = np.ones(len(weightsArr), dtype=float)
    # Recalculate the total weight after applying uniform distribution.
    totalWeight = weightsArr.sum()
  # Normalize the weights to ensure they sum to unity for averaging.
  normWeights = weightsArr / totalWeight
  if (verbose):
    print(f"Normalized weights for voting: {normWeights}")
  # Iterate through each model's prediction and its corresponding normalized weight.
  for pred, w in zip(predsList, normWeights):
    # Check if the predicted class is among the unique classes considered for voting.
    if (pred in uniqueClasses):
      # Find the index of the predicted class in the unique classes list.
      classIdx = uniqueClasses.index(pred)
      # Accumulate the normalized weight for the predicted class index.
      classVotes[classIdx] += w
  if (verbose):
    print(f"Accumulated class votes: {classVotes}")
  # Determine the index of the class with the highest accumulated vote.
  fusedClassIdx = np.argmax(classVotes)
  # Retrieve the class name corresponding to the fused class index.
  fusedClass = uniqueClasses[fusedClassIdx]
  if (verbose):
    print(f"Fused prediction: {fusedClass} with index: {fusedClassIdx}")
  # If probability vectors are provided, compute the weighted average probability vector.
  fusedProbVec = None
  if (probsList is not None):
    # Initialize an array to accumulate weighted probabilities for each class.
    probAccumulator = np.zeros(len(uniqueClasses), dtype=float)
    # Iterate through each model's probability vector and its corresponding normalized weight.
    for probVec, w in zip(probsList, normWeights):
      if (probVec is not None):
        # Convert the probability vector to a numpy array for mathematical operations.
        probArr = np.array(probVec, dtype=float)
        # Check if the probability vector length matches the number of unique classes.
        if (len(probArr) == len(uniqueClasses)):
          # Accumulate the weighted probabilities for each class index.
          probAccumulator += w * probArr
    if (verbose):
      print(f"Accumulated weighted probabilities: {probAccumulator}")
    # Normalize the accumulated probabilities to ensure they sum to 1.
    if (probAccumulator.sum() > 0):
      fusedProbVec = (probAccumulator / probAccumulator.sum()).tolist()
      if (verbose):
        print(f"Normalized fused probability vector: {fusedProbVec}")
      fusedClass = uniqueClasses[np.argmax(fusedProbVec)]
      if (verbose):
        print(f"Fused prediction based on probabilities: {fusedClass} (index: {np.argmax(fusedProbVec)})")
    else:
      if (verbose):
        print("Warning: Sum of accumulated probabilities is zero; cannot normalize.")

  if (verbose):
    print(f"Final fused prediction: {fusedClass} (index: {fusedClassIdx}), fused probability vector: {fusedProbVec}")
  return fusedClass, fusedProbVec


def LoadPredictionsFile(path):
  if (not os.path.exists(path)):
    raise FileNotFoundError(f"Predictions file not found: {path}")

  df = pd.read_csv(path, low_memory=False)

  probsList = []
  if ("Probabilities" in df.columns):
    for p in df["Probabilities"].values:
      if (isinstance(p, str)):
        try:
          parsed = SafeParseProbabilities([p])[0]
          probsList.append(parsed)
        except Exception:
          try:
            probsList.append(json.loads(p))
          except Exception:
            try:
              probsList.append(eval(p))
            except Exception:
              probsList.append(None)
      else:
        probsList.append(p)
  else:
    probsList = [None] * len(df)

  trueIdx = df["TrueClassIndex"].astype(int).tolist() if ("TrueClassIndex" in df.columns) else [None] * len(df)
  predIdx = (
    df["PredictedClassIndex"].astype(int).tolist()
    if ("PredictedClassIndex" in df.columns)
    else [None] * len(df)
  )
  trueName = df["TrueClassName"].tolist() if ("TrueClassName" in df.columns) else [None] * len(df)
  predName = df["PredictedClassName"].tolist() if ("PredictedClassName" in df.columns) else [None] * len(df)

  idxToName = {}
  for ti, tn in zip(trueIdx, trueName):
    if ((ti is not None) and (tn is not None)):
      idxToName[int(ti)] = tn

  for pi, pn in zip(predIdx, predName):
    if ((pi is not None) and (pn is not None) and (int(pi) not in idxToName)):
      idxToName[int(pi)] = pn

  return {
    "TrueIdx"  : trueIdx,
    "PredIdx"  : predIdx,
    "TrueName" : trueName,
    "PredName" : predName,
    "Probs"    : probsList,
    "IdxToName": idxToName,
    "DfLen"    : len(df),
  }


def FuseModelsFromData(
  experimentPath,
  models,
  outModelName="FusedModel"
):
  # Print the dataset directory currently being processed.
  print(f"Processing dataset folder: {experimentPath}.")
  # Initialize a list to store models that actually exist in the directory.
  existingModels = [m for m in models if (os.path.isdir(os.path.join(experimentPath, m)))]
  # Warn the user if some configured models are missing from the directory.
  if (len(existingModels) < len(models)):
    print(f"Warning: some configured models are missing under {experimentPath}. Found: {existingModels}.")
  # Skip processing if no models were found in the directory.
  if (len(existingModels) == 0):
    print("No models found for this dataset. Skipping.")
    return
  # Initialize a list to hold the set of trials for each model.
  trialsSets = []
  # Iterate through each existing model to find its trial directories.
  for m in existingModels:
    # Construct the full path to the model directory.
    modelPath = os.path.join(experimentPath, m)
    # List directories starting with Trial_x inside the model path.
    entries = [
      d for d in os.listdir(modelPath) if
      (os.path.isdir(os.path.join(modelPath, d)) and d.startswith("Trial_"))
    ]
    # Add the set of trial entries to the trialsSets list.
    trialsSets.append(set(entries))
  # Compute the intersection of trial sets to find common trials across all models.
  commonTrials = set.intersection(*trialsSets) if (len(trialsSets) > 0) else set()
  # Sort the common trials alphabetically for deterministic processing.
  commonTrials = sorted(list(commonTrials))
  # Skip processing if no common trials were found across the models.
  if (len(commonTrials) == 0):
    print(f"No common Trial_x folders found across models in {experimentPath}. Skipping.")
    return
  # Print the list of common trials that will be fused.
  print(f"Common trials to fuse: {commonTrials}.")
  # Construct the output directory path for the fused model.
  outModelDir = os.path.join(experimentPath, outModelName)
  # Create the output model directory if it does not already exist.
  os.makedirs(outModelDir, exist_ok=True)
  # Attempt to copy preprocessor artifacts from the first model Trial_1.
  if (len(existingModels) > 0):
    # Construct the source path for the preprocessor artifacts.
    srcProc = os.path.join(experimentPath, existingModels[0], "Trial_1")
    # Construct the destination path for the preprocessor artifacts.
    destProc = os.path.join(outModelDir, "Trial_1")
    # Check if the source preprocessor directory exists before copying.
    if (os.path.isdir(srcProc)):
      # Create the destination preprocessor directory.
      os.makedirs(destProc, exist_ok=True)
      # Iterate through each file in the source preprocessor directory.
      for fname in os.listdir(srcProc):
        # Construct the full source file path.
        srcf = os.path.join(srcProc, fname)
        # Construct the full destination file path.
        dstf = os.path.join(destProc, fname)
        # Attempt to copy the file to the destination directory.
        try:
          # Skip subdirectories to avoid recursive copying of large artifacts.
          if (os.path.isdir(srcf)):
            continue
          # Copy the file preserving metadata.
          shutil.copy2(srcf, dstf)
        except Exception:
          # Continue execution if the file copy fails.
          continue
  # Iterate through each common trial to perform fusion.
  for trial in commonTrials:
    # Initialize a list to store prediction results for each model.
    modelPredictions = []
    # Iterate through each existing model to load its checkpoint and predict.
    for m in existingModels:
      # Find the best model by looking for the checkpoint file saved during training in the Checkpoints subdirectory.
      # The best checkpoint is expected to have the lowest loss value.
      # Files format: CheckpointEpoch339_Metric_0_0009.pt
      allChecks = sorted([
        f
        for f in os.listdir(os.path.join(experimentPath, m, trial, "Checkpoints"))
        if (f.startswith("CheckpointEpoch") and f.endswith(".pt"))
      ])
      bestCheck = None
      bestLoss = float("inf")
      for check in allChecks:
        try:
          # Extract the loss value from the checkpoint filename using string manipulation.
          lossStr = check.split("_Metric_")[-1].replace(".pt", "").replace("_", ".")
          lossVal = float(lossStr)
          # Update the best checkpoint if a lower loss value is found.
          if (lossVal <= bestLoss):
            bestLoss = lossVal
            bestCheck = check
        except Exception:
          # Skip files that do not conform to the expected naming convention.
          continue
      # Compose the path to the expected best-checkpoint file produced during training.
      checkpointPath = os.path.join(experimentPath, m, trial, "Checkpoints", bestCheck)
      print(f"Model {m} in trial {trial}: best checkpoint found: {checkpointPath} with loss: {bestLoss}.")
      # Construct the path to the raw dataset file for this trial.
      rawDataPath = os.path.join(experimentPath, m, trial, "AllRaw.csv")
      # Check if both the checkpoint and raw data files exist.
      if (os.path.isfile(checkpointPath) and os.path.isfile(rawDataPath)):
        # Store the model path and data path for later processing.
        modelPredictions.append({"ModelName": m, "CheckpointPath": checkpointPath, "DataPath": rawDataPath})
      else:
        # Print a warning if required files are missing for the model.
        print(f"Warning: missing checkpoint or data for model {m} in {trial}. Skipping model.")
    # Skip the trial if no valid models were found for fusion.
    if (len(modelPredictions) == 0):
      print(f"No valid models found for trial {trial}. Skipping trial.")
      continue
    configsPath = os.path.join(experimentPath, existingModels[0], trial, "ConfigsJson.json")
    configs = ReadProjectConfig(configsPath) if os.path.isfile(configsPath) else {}
    # Load the preprocessor from the first valid model trial to ensure consistent transformation.
    preprocessor = TabularPreprocessor(
      ignoreCategorical=configs.get("IgnoreCategorical", True),
      numericScaler=configs.get("NumericScaler", None)
    )
    # Load the preprocessor artifacts from the first model Trial_1 directory.
    preprocessor.Load(os.path.join(experimentPath, existingModels[0], "Trial_1"))
    print(f"Loaded preprocessor for trial {trial} from {os.path.join(experimentPath, existingModels[0], 'Trial_1')}.")
    print(f"Is preprocessor loaded successfully? {preprocessor.IsLoaded()}.")
    # Read the raw dataset using pandas.
    rawDataFrame = pd.read_csv(modelPredictions[0]["DataPath"], low_memory=False)
    print(
      f"Loaded raw data for trial {trial} from {modelPredictions[0]['DataPath']}: "
      f"shape: {rawDataFrame.shape}, columns: {rawDataFrame.columns.tolist()}."
    )
    # Transform the raw data into features and labels using the preprocessor.
    featuresArray, labelsArray = preprocessor.Transform(rawDataFrame, labelColumn="Label")
    print(
      f"Transformed raw data for trial {trial} into features and labels: "
      f"features shape: {featuresArray.shape}, labels shape: {labelsArray.shape}."
    )
    # Convert the numpy arrays to PyTorch tensors for model inference.
    featuresTensor = torch.from_numpy(featuresArray).float()
    # Convert the labels array to a PyTorch tensor for dataset creation.
    labelsTensor = torch.from_numpy(labelsArray).long()
    print(
      f"Converted features and labels to PyTorch tensors for trial {trial}: "
      f"features tensor shape: {featuresTensor.shape}, labels tensor shape: {labelsTensor.shape}."
    )
    # Create a `TensorDataset` to hold the features and labels.
    inferenceDataset = TensorDataset(featuresTensor, labelsTensor)
    # Create a DataLoader for batched inference without shuffling.
    inferenceLoader = DataLoader(inferenceDataset, batch_size=256, shuffle=False)
    # Initialize a list to hold aligned true labels for the trial.
    allTrueIndices = labelsArray.tolist()
    print(
      f"Prepared true class indices for trial {trial}: {len(allTrueIndices)} samples, "
      f"unique classes: {set(allTrueIndices)}."
    )
    # Initialize a list to hold aligned true class names using the encoder.
    encObj = preprocessor.labelEncoder
    allTrueNames = [encObj.inverse_transform([idx])[0] for idx in allTrueIndices]
    print(
      f"Prepared inference data for trial {trial}: {len(allTrueIndices)} samples, "
      f"feature dimension: {featuresArray.shape[1]}, "
      f"number of classes: {len(set(allTrueIndices))}."
    )
    # Initialize lists to store predictions and probabilities for each model.
    modelPredIndices = []
    # Initialize the list to store probability vectors for each model.
    modelProbVectors = []
    # Initialize a dictionary to store model weights based on F1 scores.
    modelWeightsDict = {}
    # Determine the device for inference based on CUDA availability.
    inferenceDevice = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # Iterate through each model to perform inference and collect predictions.
    for modelEntry in modelPredictions:
      print(
        f"Processing model {modelEntry['ModelName']} for trial {trial}: "
        f"checkpoint: {modelEntry['CheckpointPath']}, data: {modelEntry['DataPath']}."
      )
      # Extract the model name from the current entry.
      currentModelName = modelEntry["ModelName"]
      # Extract the checkpoint path from the current entry.
      currentCheckpointPath = modelEntry["CheckpointPath"]
      # Instantiate a fresh model architecture matching the current model.
      tempModel = GetModel(currentModelName, featuresArray.shape[1], len(set(labelsArray)))
      # Load the trained weights from the checkpoint file onto the device.
      tempModel = LoadModel(tempModel, currentCheckpointPath, device=inferenceDevice, weightsOnly=False)
      # Move the model explicitly to the inference device.
      tempModel.to(inferenceDevice)
      # Set the model to evaluation mode to disable dropout and gradients.
      tempModel.eval()
      # Initialize lists to collect predictions and probabilities for this model.
      tempPredIndices = []
      # Initialize the list to collect probability vectors for this model.
      tempProbVectors = []
      # Disable gradient computation for faster and safer inference.
      with torch.no_grad():
        # Iterate through batches in the data loader.
        for featuresBatch, _ in tqdm.tqdm(
          inferenceLoader, desc=f"Model {currentModelName} inference on trial {trial}", unit="batch"
        ):
          # Move the feature batch to the inference device.
          featuresBatch = featuresBatch.to(inferenceDevice)
          # Perform the forward pass through the model.
          outputBatch = tempModel(featuresBatch)
          # Compute class probabilities using the softmax function.
          probabilitiesBatch = torch.nn.functional.softmax(outputBatch, dim=1)
          # Get the predicted class indices with the highest probabilities.
          predictedIndices = torch.argmax(probabilitiesBatch, dim=1)
          # Convert the predicted indices to a list and extend the collection.
          tempPredIndices.extend(predictedIndices.cpu().numpy().tolist())
          # Convert the probability batch to a list and extend the collection.
          tempProbVectors.extend(probabilitiesBatch.cpu().numpy().tolist())
      print(
        f"Model {currentModelName} in trial {trial}: collected predictions and probabilities: "
        f"{len(tempPredIndices)} samples, {len(tempProbVectors)} probability vectors."
      )
      # Store the collected predictions for the current model.
      modelPredIndices.append(tempPredIndices)
      # Store the collected probability vectors for the current model.
      modelProbVectors.append(tempProbVectors)
      # Compute the confusion matrix for the current model predictions.
      currentConfMatrix = confusion_matrix(allTrueIndices, tempPredIndices)
      # Calculate performance metrics from the confusion matrix.
      currentMetrics = CalculatePerformanceMetrics(
        currentConfMatrix,
        eps=configs.get("Eps", 1e-10),
        addWeightedAverage=True,
        addPerClass=True,
      )
      # Extract the weighted F1 score to use as the voting weight.
      currentF1Score = currentMetrics.get("Weighted F1", 0.0)
      # Store the F1 score in the weights dictionary keyed by model name.
      modelWeightsDict[currentModelName] = currentF1Score
      print(
        f"Model {currentModelName} in trial {trial}: computed metrics: "
        f"Weighted F1 score: {currentF1Score}, assigned voting weight: {currentF1Score}."
      )
    # Initialize the list to hold the final fused prediction indices.
    fusedPredIndices = []
    # Initialize the list to hold the final fused probability vectors.
    fusedProbVectors = []
    # Iterate through each sample index to perform ensemble voting.
    for sampleIdx in tqdm.tqdm(
      range(len(allTrueIndices)),
      desc=f"Performing weighted majority voting for trial {trial}",
    ):
      # Collect predictions from all models for the current sample.
      currentSamplePreds = [modelPredIndices[mIdx][sampleIdx] for mIdx in range(len(modelPredIndices))]
      # Collect probability vectors from all models for the current sample.
      currentSampleProbs = [modelProbVectors[mIdx][sampleIdx] for mIdx in range(len(modelProbVectors))]
      # Extract the voting weights corresponding to the active models.
      currentSampleWeights = [modelWeightsDict[modelEntry["ModelName"]] for modelEntry in modelPredictions]
      # Determine the unique classes present in the true labels.
      uniqueClassesList = sorted(list(set(allTrueIndices)))
      # Map the unique indices to their corresponding class names.
      uniqueClassNamesList = [
        allTrueNames[allTrueIndices.index(idx)] if (idx in allTrueIndices) else f"Class_{idx}"
        for idx in uniqueClassesList
      ]
      # Execute the weighted majority voting algorithm for the sample.
      votedClassName, votedProbVector = WeightedMajorityVote(
        currentSamplePreds,
        currentSampleWeights,
        currentSampleProbs,
        uniqueClasses=uniqueClassNamesList,
      )
      # Resolve the class index corresponding to the voted class name.
      votedClassIndex = None
      for idx, name in zip(uniqueClassesList, uniqueClassNamesList):
        if (name == votedClassName):
          votedClassIndex = idx
          break
      # print(
      #   f"Sample index {sampleIdx} in trial {trial}: "
      #   f"Model predictions: {currentSamplePreds}, "
      #   f"Model weights: {currentSampleWeights}, "
      #   f"Voted class name: {votedClassName}, "
      #   f"Resolved voted class index: {votedClassIndex}, "
      #   f"Voted probability vector: {votedProbVector}."
      # )
      # Append the resolved fused prediction index to the results list.
      fusedPredIndices.append(votedClassIndex)
      # Serialize the probability vector or store null if unavailable.
      fusedProbVectors.append(json.dumps(votedProbVector) if (votedProbVector is not None) else None)
    # Construct the output directory for the fused predictions.
    outTrialPredDir = os.path.join(outModelDir, trial, "TabularEvalPredResults")
    # Create the output directory structure if it does not exist.
    os.makedirs(outTrialPredDir, exist_ok=True)
    # Build a pandas DataFrame containing the fused results.
    outDataFrame = pd.DataFrame({
      "TrueClassIndex"     : allTrueIndices,
      "TrueClassName"      : allTrueNames,
      "PredictedClassIndex": fusedPredIndices,
      "PredictedClassName" : [
        allTrueNames[idx]
        if (idx < len(allTrueNames) and idx is not None) else str(idx)
        for idx in fusedPredIndices
      ],
      "Probabilities"      : fusedProbVectors
    })
    # Save the fused predictions DataFrame to a CSV file.
    outDataFrame.to_csv(os.path.join(outTrialPredDir, "Predictions.csv"), index=False)
    # Print confirmation that the fused predictions were saved successfully.
    print(f"Saved fused predictions to: {os.path.join(outTrialPredDir, 'Predictions.csv')}.")
    # Attempt to compute and save the fused metrics summary.
    try:
      # Convert the true indices list to a numpy integer array.
      yTrueArray = np.array(allTrueIndices, dtype=int)
      # Convert the fused prediction indices list to a numpy integer array.
      yPredArray = np.array(fusedPredIndices, dtype=int)
      # Compute the confusion matrix for the fused predictions.
      fusedConfMatrix = confusion_matrix(yTrueArray, yPredArray)
      # Calculate comprehensive performance metrics for the fused model.
      fusedMetrics = CalculatePerformanceMetrics(fusedConfMatrix, eps=1e-10, addWeightedAverage=True, addPerClass=True)
      # Prepare the summary dictionary for JSON serialization.
      summaryDict = {"ConfusionMatrix": fusedConfMatrix.tolist(), "Metrics": SimpleSerializeForJson(fusedMetrics)}
      # Open the output JSON file in write mode.
      with open(os.path.join(outTrialPredDir, "FusedSummary.json"), "w") as f:
        # Dump the summary dictionary to the JSON file with indentation.
        json.dump(summaryDict, f, indent=4)
      # Print confirmation that the fused summary was saved successfully.
      print(f"Saved fused summary to: {os.path.join(outTrialPredDir, 'FusedSummary.json')}.")
    except Exception as e:
      # Print a warning if metric computation or saving fails.
      print(f"Warning: failed to compute/save fused metrics for trial {trial}: {e}.")


def RunStatisticalPhase(
  modelName,
  csvPath,
  configs,
  saveDirLocal,
):
  # Check if the save directory for the statistical phase exists.
  if (not os.path.exists(saveDirLocal)):
    # Print a message indicating the save directory is missing.
    print(f"Save directory for statistical phase does not exist: {saveDirLocal}")
    # Print guidance to ensure training phase was completed before running statistical analysis.
    print("Ensure that the training phase was completed and the directory structure is correct.")
    # Exit the function early when required artifacts are not available.
    return

  # Announce the start of the statistical analysis phase for the given model and CSV.
  print(f"Running statistical analysis phase for model: {modelName} on dataset: {csvPath}")
  # Announce the directory where trial data will be searched.
  print(f"Looking for trial directories in: {saveDirLocal}")
  # List all entries in the save directory to search for trial subfolders.
  allTrials = os.listdir(saveDirLocal)
  # Filter the directory entries to those that match the expected trial naming pattern.
  summaryFiles = [
    f
    for f in allTrials
    if (f.startswith("Trial_")) and (os.path.isdir(os.path.join(saveDirLocal, f)))
  ]

  # Report how many trial directories were discovered for analysis.
  print(f"Found {len(summaryFiles)} trial directories for statistical analysis: {summaryFiles}")
  # If no trial directories are found, warn and exit the function.
  if (len(summaryFiles) <= 0):
    # Print a descriptive warning about missing trial directories.
    print(
      "No trial directories found for statistical analysis; "
      "ensure that the training phase was completed with multiple trials."
    )
    # Stop further processing when no trials are available.
    return

  # Initialize lists to aggregate labels, indices, probabilities, and metrics across trials.
  allTrueLabels = []
  # Initialize a list to hold predicted label names for each trial.
  allPredLabels = []
  # Initialize a list to hold true class indices for each trial.
  allTrueIdx = []
  # Initialize a list to hold predicted class indices for each trial.
  allPredIdx = []
  # Initialize a list to hold prediction probability arrays for each trial.
  allProbabilities = []
  # Initialize a list to hold computed metrics dictionaries for each trial.
  allMetrics = []

  # Iterate over each detected trial directory and extract prediction artifacts.
  for trial in summaryFiles:
    # Print which trial directory is currently being processed.
    print(f"Extracting data for statistical analysis from trial: {trial}")
    # Compose the expected path to the trial's predictions CSV file.
    summaryPath = os.path.join(saveDirLocal, trial, "TabularEvalPredResults", "Predictions.csv")
    # Check whether the predictions file exists at the expected location.
    if (os.path.exists(summaryPath)):
      # Read the predictions CSV into a pandas DataFrame.
      data = pd.read_csv(summaryPath, low_memory=False)
      # Extract the true class indices from the DataFrame as a list.
      trueClassIdx = data["TrueClassIndex"].values.tolist()
      # Extract the predicted class indices from the DataFrame as a list.
      predClassIdx = data["PredictedClassIndex"].values.tolist()
      # Extract the true class names from the DataFrame as a list.
      trueClassName = data["TrueClassName"].values.tolist()
      # Extract the predicted class names from the DataFrame as a list.
      predClassName = data["PredictedClassName"].values.tolist()
      # Extract the probabilities column from the DataFrame.
      probabilities = data["Probabilities"].values
      # Parse the probabilities from string representation to actual lists of floats.
      for i in range(len(probabilities)):
        probabilities[i] = SafeParseProbabilities(probabilities[i])
      # Convert the numpy array of probabilities to a list for easier handling.
      probabilities = probabilities.tolist()
      # Append the trial's true class names to the aggregated list.
      allTrueLabels.append(trueClassName)
      # Append the trial's predicted class names to the aggregated list.
      allPredLabels.append(predClassName)
      # Append the trial's true class indices to the aggregated list.
      allTrueIdx.append(trueClassIdx)
      # Append the trial's predicted class indices to the aggregated list.
      allPredIdx.append(predClassIdx)
      # Append the trial's probability arrays to the aggregated list.
      allProbabilities.append(probabilities)

      # Compute the confusion matrix for this trial using true and predicted indices.
      cm = confusion_matrix(trueClassIdx, predClassIdx)
      # Calculate performance metrics from the confusion matrix using helper function.
      metrics = CalculatePerformanceMetrics(
        cm,  # Confusion matrix (2D list or numpy array).
        eps=configs.get("Eps", 1e-10),  # Small value to avoid division by zero in metric calculations.
        addWeightedAverage=True,  # Whether to include weighted averages in the output.
        addPerClass=True,  # Whether to include per-class metrics in the output.
      )
      # Store the computed metrics for later aggregation and reporting.
      allMetrics.append(metrics)
    else:
      # Warn when a trial's predictions CSV cannot be found at the expected path.
      print(f"Warning: Summary file not found for trial {trial} at expected path: {summaryPath}")

  # Create the Statistics output folder inside the save directory if it does not exist.
  statsFolder = os.path.join(saveDirLocal, "Statistics")
  os.makedirs(statsFolder, exist_ok=True)

  # Instantiate a tabular preprocessor to retrieve saved label encoder information.
  processor = TabularPreprocessor(
    ignoreCategorical=configs.get("IgnoreCategorical", True),
    numericScaler=configs.get("NumericScaler", None)
  )
  # Attempt to load preprocessor artifacts from the first trial to obtain the label encoder.
  processor.Load(saveDirLocal + "/Trial_1")  # Load from the first trial to get the label encoder.
  # Choose class names from the label encoder when available, otherwise build fallback names.
  if (processor.IsLoaded() and processor.labelEncoder is not None):
    # Use classes provided by the saved label encoder artifact.
    classes = list(processor.labelEncoder.classes_)
  else:
    # Warn that the label encoder could not be loaded and fall back to index-based names.
    print("Warning: Could not load label encoder from preprocessor artifacts; using class indices as class names.")
    # Build fallback class names from unique true class indices aggregated across trials.
    classes = [f"Class_{i}" for i in range(len(set(np.concatenate(allTrueIdx))))]

  # Print the class names that will be used for plotting and analysis.
  print(f"Classes for statistical analysis: {classes}")
  # Read the DPI configuration value for saved plots from the configs dictionary.
  dpi = configs.get("PlotDPI", 300)
  # Read the font size configuration value for plots from the configs dictionary.
  fontSize = configs.get("FontSize", 15)

  # Print a concise summary of the number of trials' data prepared for analysis.
  print(
    f"Prepared data for statistical analysis: "
    f"{len(allTrueLabels)} trials with true labels, {len(allPredLabels)} trials with predicted labels, "
    f"{len(allProbabilities)} trials with probabilities."
  )

  # Print small samples of labels and probabilities for each trial to aid debugging.
  for trial in range(len(allTrueLabels)):
    # Print a short sample of true labels, predicted labels, and probabilities for the trial.
    print(
      f"Trial {trial + 1}: True labels sample: {allTrueLabels[trial][:5]}, "
      f"Predicted labels sample: {allPredLabels[trial][:5]}, "
      f"Probabilities sample: {allProbabilities[trial][:2]}"
    )

  # Store the trials values in a CSV file for reference and potential future analysis.
  # Columns should include TrueLabels, PredLabels, Probabilities, and any other relevant information for each trial.
  # Prefixed by the trial name or index to distinguish between them, e.g., Trial_1_TrueLabels, Trial_1_PredLabels, etc.
  trialsDataFile = os.path.join(statsFolder, "Trials_Data_Summary.csv")
  with open(trialsDataFile, "w") as f:
    classes = (
      processor.labelEncoder.classes_
      if (processor.IsLoaded() and processor.labelEncoder is not None)
      else [f"Class_{i}" for i in range(numClasses)]
    )

    # Write the header row with trial-specific column names.
    header = []
    for i in range(len(allTrueLabels)):
      header.append(f"Trial_{i + 1}_TrueLabels")
      header.append(f"Trial_{i + 1}_PredLabels")
      for c in classes:
        header.append(f"Trial_{i + 1}_Prob_{c}")
    f.write(",".join(header) + "\n")

    # Determine the maximum number of rows needed to accommodate all trials' data.
    maxRows = max(len(allTrueLabels[i]) for i in range(len(allTrueLabels)))
    # Determine a default padding length for probability vectors from the first non-empty trial.
    padLengthDefault = 1
    for trialProbs in allProbabilities:
      if (trialProbs and len(trialProbs) > 0):
        try:
          padLengthDefault = len(trialProbs[0])
        except Exception:
          padLengthDefault = 1
        break
    # Write each row of trial data, filling in empty values where trials have fewer entries.
    for row in range(maxRows):
      rowData = []
      for i in range(len(allTrueLabels)):
        rowData.append(str(allTrueLabels[i][row]) if (row < len(allTrueLabels[i])) else "")
        rowData.append(str(allPredLabels[i][row]) if (row < len(allPredLabels[i])) else "")
        if (row < len(allProbabilities[i])):
          toAdd = allProbabilities[i][row]
        else:
          toAdd = [""] * padLengthDefault
        rowData.extend([str(p) for p in toAdd])  # Extend the row data with probabilities for this trial.
      f.write(",".join(rowData) + "\n")

  # Iterate over the two methods for uncertainty aggregation: confidence intervals and standard deviation.
  for which in ["CI", "SD"]:
    # Build the output filename for the precision-recall curve PDF for the chosen method.
    fileName = os.path.join(statsFolder, f"{which}_MultiTrial_PRC_Curve.pdf")
    # Generate and optionally save the multi-trial precision-recall curve using the helper.
    PlotMultiTrialPRCurve(
      allTrueIdx,  # List of true labels arrays from all trials.
      allProbabilities,  # List of predicted probabilities from all trials.
      classes,  # List of class names.
      confidenceLevel=0.95,  # Confidence level for CI.
      which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
      title="Multi-Trial Precision-Recall Curve",
      figSize=(8, 8),  # Figure size in inches.
      cmap=None,  # Colormap for different classes.
      display=False,  # Whether to display the plot.
      save=True,  # Whether to save the plot.
      fileName=fileName,  # File name for saving.
      fontSize=fontSize,  # Font size for labels and annotations.
      showLegend=True,  # Whether to show legend.
      returnFig=False,  # Whether to return the matplotlib figure object.
      dpi=dpi,  # DPI for saving the figure.
      addZoomedInset=True,  # Whether to add a zoomed inset for the top-right corner of the PRC plot.
    )

    # Build the output filename for the ROC AUC plot PDF for the chosen method.
    fileName = os.path.join(statsFolder, f"{which}_MultiTrial_ROC_AUC.pdf")
    # Generate and optionally save the multi-trial ROC AUC plot using the helper.
    PlotMultiTrialROCAUC(
      allTrueIdx,  # List of true labels arrays from all trials.
      allProbabilities,  # List of predicted probabilities from all trials.
      classes,  # List of class names.
      confidenceLevel=0.95,  # Confidence level for CI (default 95%).
      which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
      title="Multi-Trial ROC Curve",  # Plot title.
      figSize=(8, 8),  # Figure size.
      cmap=None,  # Colormap for ROC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=fileName,  # File name.
      fontSize=15,  # Font size.
      plotDiagonal=True,  # Plot diagonal reference line.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
      addZoomedInset=True,  # Whether to add a zoomed inset for the top-left corner.
    )

  # Save the calculated metrics for each trial to a CSV file for comparison.
  # Example of the file structure (if you have a single system):
  #     Precision, Recall, F1, Accuracy, Specificity, Average
  #     Metric, Metric, Metric, Metric, Metric, Metric
  #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133,
  #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282,
  #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406,
  #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339,
  #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813,

  # Create a new list to hold the refined metrics dictionaries for each trial after cleaning for CSV output.
  refinedAllMetrics = []
  # Iterate through the collected metrics for each trial and clean them for CSV output.
  for i in range(len(allMetrics)):
    # Print a message indicating which trial's metrics are being processed.
    print(f"Processing metrics for trial {i + 1} for CSV output.")
    # Retrieve the metrics dictionary for the trial.
    weightedMetrics = allMetrics[i]
    # Keep only keys corresponding to weighted averages for consistent CSV columns.
    weightedMetrics = {k: weightedMetrics[k] for k in weightedMetrics if (k.startswith("Weighted "))}
    # Remove the "Weighted " prefix from metric names to create clean column headers.
    refinedMetrics = {k.replace("Weighted ", ""): v for k, v in weightedMetrics.items()}

    # Optionally remove the "Average" metric from the refined metrics if configured to do so.
    if (configs.get("RemoveAverageFromStats", True)):
      # Remove the Average key for the trial when present.
      refinedMetrics = {k: v for k, v in refinedMetrics.items() if ("Average" not in k)}

    # Append the cleaned metrics dictionary to the refined metrics list.
    refinedAllMetrics.append(refinedMetrics)

  # Build a DataFrame to hold the trial metrics comparison with a helper second row.
  firstRow = list(refinedAllMetrics[0].keys())  # Get metric names from the first trial's metrics for the header.
  # Create a second row that labels the following rows as metrics for clarity.
  secondRow = ["Metric"] * len(firstRow)  # Create a second row with the keyword "Metric" for clarity.
  # Extract metric values from each trial in the same column order as the header.
  data = [list(metrics.values()) for metrics in refinedAllMetrics]  # Extract metric values for each trial.
  # Construct a pandas DataFrame using the header and metric values.
  dfMetrics = pd.DataFrame(data, columns=firstRow)  # Create a DataFrame with metric values and column headers.
  # Insert the secondRow at the top of the DataFrame to label the subsequent metric rows.
  dfMetrics.loc[-1] = secondRow  # Add the second row with "Metric" values.
  # Shift the DataFrame index to accommodate the newly inserted row.
  dfMetrics.index = dfMetrics.index + 1  # Shift the index to accommodate the new row.
  # Sort the DataFrame index so the header, label row, and data rows appear in order.
  dfMetrics.sort_index(inplace=True)
  # Save the trial metrics comparison DataFrame to a CSV file inside the statistics folder.
  trialMetricsComparisonFile = os.path.join(statsFolder, "Trial_Metrics_Comparison.csv")
  dfMetrics.to_csv(trialMetricsComparisonFile, index=False)

  # Print the path where the trial metrics comparison CSV was saved.
  print(f"Trial metrics comparison saved to: {trialMetricsComparisonFile}")
  # Print the DataFrame containing the trial metrics comparison for quick inspection.
  print(f"Trial Metrics Comparison:\n{dfMetrics}")

  # Extract plotting history, plot names, and metric dictionaries from the saved summary CSV.
  history, names, metrics = ExtractDataFromSummaryFile(trialMetricsComparisonFile)

  # Prepare a folder name for storing individual performance metric plots, if required.
  newFolderName = ""
  # Check configuration to determine whether to generate individual metric plots.
  if (configs.get("PlotMetricsIndividual", True)):
    # Build the path for the folder where individual plots will be stored.
    newFolderName = os.path.join(statsFolder, "PerformanceMetricsPlots")
    # Create the folder for individual plots if it does not already exist.
    os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.

    # Generate and save a set of performance metric plots using the helper function.
    PlotMetrics(
      history, names, metrics,
      factor=5,  # Factor to multiply the default figure size.
      keyword="AllMetrics",  # Keyword to append to the filenames of the saved plots.
      dpi=dpi,  # Dots per inch (resolution) of the saved plots.
      xTicksRotation=45,  # Rotation angle for x-axis tick labels.
      whichToPlot=[],  # List of plot types to generate.
      fontSize=fontSize,  # Font size for the plots.
      showFigures=False,  # Whether to display the plots or not.
      storeInsideNewFolder=True,  # Whether to store the plots inside a new folder.
      newFolderName=newFolderName,  # Name of the folder to store the plots.
      noOfPlotsPerRow=3,  # Number of plots per row in the subplot grid.
      cmap="viridis",  # Color map for the plots.
      differentColors=True,  # Whether to use different colors for different plots.
      fixedTicksColors=True,  # Whether to use fixed ticks colors for consistency across plots.
      fixedTicksColor="black",  # Color to use for fixed ticks if `fixedTicksColors` is True.
      extension=".pdf",  # File extension for saved plots.
    )

  # Indicate that performance plots have been generated successfully.
  print("\u2713 Performance plots generated.")
  # Announce that generation of the statistical analysis report is starting.
  print("\nGenerating statistical analysis report...")
  # Initialize a list to accumulate the statistical analysis entries for each metric.
  overallReport = []
  # Iterate over each metric name to compute statistical tests and compile the report entries.
  for metric in metrics:
    for index, data in enumerate(history):
      # Run the configured statistical analysis for the metric across trials.
      report = StatisticalAnalysis(
        data[metric]["Trials"],
        hypothesizedMean=data[metric]["Mean"],
        secondMetricList=None,
      )
      # Add metadata indicating the data type (history name) to the report entry.
      report["Type"] = names[index]
      # Add metadata indicating the metric name to the report entry.
      report["Metric"] = metric
      # Append the populated report entry to the overall report list.
      overallReport.append(report)
  # Convert the list of report entries into a pandas DataFrame for saving.
  reportDF = pd.DataFrame(overallReport)
  # Build the path where the final statistical analysis CSV will be saved.
  reportCsvPath = os.path.join(statsFolder, "Statistical_Analysis_Report.csv")
  # Save the statistical analysis report DataFrame to CSV.
  reportDF.to_csv(reportCsvPath, index=False)
  # Print confirmation that the statistical analysis report has been saved to disk.
  print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")


def RunTabularPipelineForCSV(
  modelName,
  csvPath,
  configs,
  saveDirLocal,
  labelColumn="Label",
  dropFirstColumn=False,
):
  # Ensure deterministic behavior by setting random seeds.
  DoRandomSeeding()

  # Compute the base file name for this CSV without extension.
  baseFileName = os.path.splitext(os.path.basename(csvPath))[0]
  # Print which CSV file is about to be processed and its derived dataset name.
  print(f"\nProcessing CSV: {csvPath} -> Dataset name: {baseFileName}")

  # Load the entire CSV into a pandas DataFrame with an optional row limit from configs.
  dfAll = pd.read_csv(csvPath, nrows=configs.get("MaxRows", None), low_memory=False)

  # Decide whether stratified splitting should use the provided label column.
  if (labelColumn in dfAll.columns):
    # Use the label column series as the stratification key for splits.
    stratifyCol = dfAll[labelColumn]
  else:
    # Assign None to stratifyCol when no label column is present.
    stratifyCol = None

  # Optionally remove the first column of the dataset when requested by the caller.
  if (dropFirstColumn):
    # Drop the first column which may represent indices or non-informative IDs.
    dfAll = dfAll.iloc[:, 1:]
    # Inform the user that the first column was dropped as requested.
    print("Dropped the first column of the dataset as per argument.")

  # Split the dataset into training and temporary sets according to configured train fraction.
  dfTrain, dfTemp = train_test_split(
    dfAll,
    stratify=stratifyCol,
    train_size=configs["TrainFraction"],
    random_state=np.random.randint(0, 10000),
  )

  # Split the temporary set into validation and test sets while preserving class distribution.
  dfVal, dfTest = train_test_split(
    dfTemp,
    stratify=(dfTemp[labelColumn] if (labelColumn in dfTemp.columns) else None),
    test_size=(configs["TestFraction"] / (configs["ValFraction"] + configs["TestFraction"])),
    random_state=np.random.randint(0, 10000),
  )

  # Persist the raw, unprocessed splits to CSV files for traceability.
  dfTrain.to_csv(os.path.join(saveDirLocal, "TrainRaw.csv"), index=False)
  dfVal.to_csv(os.path.join(saveDirLocal, "ValRaw.csv"), index=False)
  dfTest.to_csv(os.path.join(saveDirLocal, "TestRaw.csv"), index=False)
  dfAll.to_csv(os.path.join(saveDirLocal, "AllRaw.csv"), index=False)

  # Print the sizes of the generated splits to help the user verify the operation.
  print(f"Data split sizes (rows) for {baseFileName}: Train={len(dfTrain)} Val={len(dfVal)} Test={len(dfTest)}")

  # Instantiate a tabular preprocessor responsible for scaling and label encoding.
  preprocessor = TabularPreprocessor(
    ignoreCategorical=configs.get("IgnoreCategorical", True),
    numericScaler=configs.get("NumericScaler", None),
  )
  # Attempt to load any previously saved preprocessor artifacts from disk.
  preprocessor.Load(saveDirLocal)
  # If artifacts are present, validate they are compatible with the current training data.
  if (preprocessor.IsLoaded()):
    # Assume artifacts are compatible until checks reveal otherwise.
    compatible = True
    # Retrieve the feature names associated with the saved artifacts.
    savedNums = preprocessor.GetFeatureNames() or []
    # Identify numeric feature names that are expected but missing from the current training split.
    missingCols = [c for c in savedNums if (c not in dfTrain.columns)]
    # If any expected numeric columns are missing, mark artifacts as incompatible.
    if (len(missingCols) > 0):
      # Warn the user about the missing numeric columns in the loaded artifacts.
      print(f"Preprocessor artifact numeric columns missing in training data: {missingCols}")
      # Mark the artifacts as incompatible so they will be re-fitted.
      compatible = False
    # Validate that any saved label encoder covers the labels present in the training partition.
    if ((labelColumn in dfTrain.columns) and (getattr(preprocessor, "labelEncoder", None) is not None)):
      try:
        # Extract the set of classes known to the saved label encoder.
        encClasses = set(preprocessor.labelEncoder.classes_)
        # Extract the set of label values found in the current training data.
        trainLabels = set(dfTrain[labelColumn].astype(str).unique())
        # If the encoder does not cover all training labels, mark as incompatible.
        if (not trainLabels.issubset(encClasses)):
          # Inform the user about the mismatch between encoder classes and training labels.
          print("Label encoder classes in artifacts do not cover training labels.")
          # Mark the artifacts as incompatible to trigger re-fitting.
          compatible = False
      except Exception:
        # Treat unexpected errors during validation as incompatibility.
        compatible = False

    # If artifacts passed validation checks, keep using them.
    if (compatible):
      # Inform the user that preprocessor artifacts were loaded and validated successfully.
      print("Preprocessor artifacts loaded from disk and validated as compatible with training data.")
    else:
      # Inform the user that artifacts are incompatible and will be re-fitted.
      print("Preprocessor artifacts are incompatible with this training split; re-fitting on training data.")
      # Instantiate a fresh preprocessor and fit it on the training partition.
      preprocessor = TabularPreprocessor(
        ignoreCategorical=configs.get("IgnoreCategorical", True),
        numericScaler=configs.get("NumericScaler", None),
      )
      preprocessor.Fit(dfTrain, labelColumn=labelColumn)
      # Persist the newly fitted preprocessor artifacts for reproducibility.
      preprocessor.Save(saveDirLocal)
      # Inform the user that the new artifacts were saved successfully.
      print("Preprocessor fitted on training data and saved successfully.")
  else:
    # Fit a new preprocessor when no artifacts were present on disk.
    preprocessor.Fit(dfTrain, labelColumn=labelColumn)
    # Save the fitted artifacts to the save directory for future runs.
    preprocessor.Save(saveDirLocal)
    # Notify the user the preprocessor was fitted and saved.
    print("Preprocessor fitted on training data and saved successfully.")

  # Transform the training data into numeric matrices and label vectors using the preprocessor.
  xTrain, yTrain = preprocessor.Transform(dfTrain, labelColumn=labelColumn)
  # Transform the validation partition into numeric arrays using the same preprocessor.
  xVal, yVal = preprocessor.Transform(dfVal, labelColumn=labelColumn)
  # Transform the test partition into numeric arrays using the same preprocessor.
  xTest, yTest = preprocessor.Transform(dfTest, labelColumn=labelColumn)

  # Validate transformed arrays for NaN values and warn the user if any are present.
  if (xTrain is not None and np.isnan(xTrain).any()):
    # Warn about NaNs detected in the training features and suggest imputation.
    print(f"Warning: NaN values found in xTrain for {baseFileName}. Consider adding imputation to the pipeline.")
  if (xVal is not None and np.isnan(xVal).any()):
    # Warn about NaNs detected in the validation features and suggest imputation.
    print(f"Warning: NaN values found in xVal for {baseFileName}. Consider adding imputation to the pipeline.")
  if (xTest is not None and np.isnan(xTest).any()):
    # Warn about NaNs detected in the test features and suggest imputation.
    print(f"Warning: NaN values found in xTest for {baseFileName}. Consider adding imputation to the pipeline.")

  # Build pandas DataFrames for the transformed feature matrices using preprocessor feature names.
  trainDF = pd.DataFrame(xTrain, columns=preprocessor.GetFeatureNames())
  valDF = pd.DataFrame(xVal, columns=preprocessor.GetFeatureNames())
  testDF = pd.DataFrame(xTest, columns=preprocessor.GetFeatureNames())
  # Attach label columns to each split DataFrame for clarity and downstream usage.
  trainDF["Label"] = yTrain
  valDF["Label"] = yVal
  testDF["Label"] = yTest
  # Concatenate train, val, and test DataFrames to create a combined dataset view.
  allDF = pd.concat([trainDF, valDF, testDF], ignore_index=True)
  # Save the processed DataFrames to CSV files inside the save directory for reproducibility.
  trainDF.to_csv(os.path.join(saveDirLocal, "TrainData.csv"), index=False)
  valDF.to_csv(os.path.join(saveDirLocal, "ValData.csv"), index=False)
  testDF.to_csv(os.path.join(saveDirLocal, "TestData.csv"), index=False)
  allDF.to_csv(os.path.join(saveDirLocal, "AllData.csv"), index=False)

  # Print the shapes of transformed arrays and label vectors for user inspection.
  print(
    f"Transformed data shapes for {baseFileName}: Xtrain={xTrain.shape if (xTrain is not None) else None}, "
    f"ytrain={yTrain.shape if (yTrain is not None) else None} | "
    f"Xval={xVal.shape if (xVal is not None) else None}, yval={yVal.shape if (yVal is not None) else None} | "
    f"Xtest={xTest.shape if (xTest is not None) else None}, ytest={yTest.shape if (yTest is not None) else None}"
  )
  # Try to obtain a preview of the feature names used after preprocessing for logging.
  try:
    featureNamesPreview = preprocessor.GetFeatureNames() or []
  except Exception:
    featureNamesPreview = []
  # Print the feature names or indicate N/A when none are available.
  print(
    f"Columns after preprocessing: "
    f"{featureNamesPreview if len(featureNamesPreview) > 0 else (preprocessor.GetFeatureNames() if (preprocessor.IsLoaded()) else 'N/A')}"
  )

  # Retrieve the definitive feature names list from the preprocessor for model input construction.
  featureNames = preprocessor.GetFeatureNames()

  # If no numeric features were discovered during preprocessing, warn and skip this dataset.
  if (xTrain is None):
    # Warn that there are no numeric features and abort processing of this CSV.
    print(f"Warning: No numeric features found after preprocessing on {baseFileName}; skipping.")
    return

  # Build PyTorch TensorDataset objects for train, validation, and test splits with appropriate dtypes.
  trainDataset = TensorDataset(torch.from_numpy(xTrain).float(), torch.from_numpy(yTrain).long())
  valDataset = TensorDataset(torch.from_numpy(xVal).float(), torch.from_numpy(yVal).long())
  testDataset = TensorDataset(torch.from_numpy(xTest).float(), torch.from_numpy(yTest).long())
  # Create DataLoader instances for each dataset using configured batch sizes and shuffle settings.
  trainLoader = DataLoader(trainDataset, batch_size=configs["BatchSize"], shuffle=True)
  valLoader = DataLoader(valDataset, batch_size=configs["BatchSize"], shuffle=False)
  testLoader = DataLoader(testDataset, batch_size=configs["BatchSize"], shuffle=False)

  # Combine unique class indices across all splits to determine the set of classes present.
  classesSet = set(np.concatenate([yTrain, yVal, yTest]))

  # Print the number of batches for each DataLoader to give the user execution context.
  print(
    f"DataLoaders created for {baseFileName}: "
    f"Train batches={len(trainLoader)}, "
    f"Val batches={len(valLoader)}, "
    f"Test batches={len(testLoader)}"
  )

  # Determine the number of classes and the model input size from the processed data.
  numClasses = len(classesSet)
  inputSize = xTrain.shape[1]
  # Instantiate the model architecture from the model zoo using input size and class count.
  model = GetModel(modelName, inputSize, numClasses)

  # Decide on runtime device based on configuration and CUDA availability.
  device = configs["Device"]
  # Fall back to CPU when CUDA is not available or not requested in the device string.
  device = torch.device(device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu")

  # Print device selection and model architecture summary for diagnostic purposes.
  print((f"Using device: {device} for dataset {baseFileName}"))
  print(f"Model architecture for {baseFileName}:\n{model}")
  # Print the total number of model parameters to give a sense of model size.
  print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

  # Create the Adam optimizer using the learning rate supplied in the configs.
  optimizer = torch.optim.Adam(model.parameters(), lr=configs["LearningRate"])

  # Select the loss function based on configuration, supporting focal loss as an option.
  if (configs.get("Loss", "CrossEntropy") == "Focal"):
    # Compute per-class counts from training labels to derive class balancing weights.
    _, counts = np.unique(yTrain, return_counts=True)
    # Build weights inversely proportional to class counts to compensate class imbalance.
    weights = torch.tensor((1.0 / counts), dtype=torch.float32)
    # Normalize weights so they sum to 1 for improved numerical stability.
    weights = weights / weights.sum()
    # Move the weights tensor to the selected device to avoid device mismatch errors.
    weights = weights.to(device)
    # Instantiate the alternative focal loss with computed class weights and configured gamma.
    lossFn = FocalLossAlt(gamma=configs.get("FocalGamma", 2.0), weight=weights)
  else:
    # Default to PyTorch's CrossEntropyLoss when focal loss is not configured.
    lossFn = torch.nn.CrossEntropyLoss()

  # Initialize the end-to-end PyTorch training pipeline with the configured components.
  pipeline = PyTorchClassificationTrainingPipeline(
    model=model,
    trainLoader=trainLoader,
    valLoader=valLoader,
    allLoader=testLoader,  # Use testLoader as allLoader for inference.
    optimizer=optimizer,
    scheduler=None,  # No scheduler by default; can be added from configs if needed.
    lossFn=lossFn,
    noOfClasses=numClasses,
    learningRate=configs["LearningRate"],
    device=device,
    outputDir=saveDirLocal,
    dpi=configs.get("PlotDPI", 300),
    logEveryNSteps=100,
    useAmp=configs.get("UseAmp", torch.cuda.is_available()),
    earlyStoppingPatience=configs.get("EarlyStopPatience", 5),
    judgeBy="val_loss",  # Monitor validation loss for checkpointing.
    checkpointSaver=None,  # Use default CheckpointSaver.
    verbose=True,
    saveDataFrames=True,  # Save train/val/test/all DataFrames to outputDir/Data/.
    configs=configs,  # Pass configs to save to outputDir/ConfigsUsed.json.
  )

  # Persist processed DataFrames using the pipeline helper for consistency.
  pipeline.SaveDataFrames(trainDF, valDF, testDF, allDF)

  # Execute training for the configured number of epochs.
  pipeline.Train(numEpochs=configs["NumEpochs"])

  # Run inference on the test set to produce predictions and evaluation artifacts.
  pipeline.Inference()

  # Optionally generate and save a training history plot when a history CSV exists.
  historyCsvPath = os.path.join(saveDirLocal, "History.csv")
  if (os.path.exists(historyCsvPath)):
    # Load the training history from CSV into a pandas DataFrame.
    history = pd.read_csv(historyCsvPath)
    # Create and save a history plot using the helper function with configurable options.
    HistoryPlotter(
      history,
      title=f"Training History - {baseFileName}_{modelName}",
      metrics=("loss", "accuracy"),
      xLabel="Epochs",
      fontSize=configs.get("PlotFontSize", 14),
      save=True,
      savePath=os.path.join(saveDirLocal, "History.png"),
      dpi=configs.get("PlotDPI", 300),
      display=configs.get("DisplayPlot", False),
      figSize=configs.get("PlotFigSize", (10, 5)),
      returnFig=False,
      smooth=configs.get("PlotSmooth", True),
      smoothFactor=configs.get("PlotSmoothFactor", 0.6)
    )
    # Inform the user that the training history plot was saved.
    print(f"Saved training history plot to {os.path.join(saveDirLocal, 'History.png')}.")

  # Perform optional explainability steps such as SHAP and surrogate tree generation when enabled.
  if (configs.get("Explain", False)):
    # Determine the directory where explainability artifacts will be stored.
    explainDir = configs.get("ExplainSaveDir", os.path.join(saveDirLocal, "ExplainArtifacts"))
    # Ensure the explainability directory exists.
    os.makedirs(explainDir, exist_ok=True)
    # Determine the number of background samples to use for SHAP computation.
    bgN = min(configs.get("ExplainBackground", 200), xTrain.shape[0])
    # Select background sample indices randomly without replacement from the training set.
    bgIdx = np.random.choice(xTrain.shape[0], bgN, replace=False)
    # Build the background dataset array for SHAP using selected indices.
    background = xTrain[bgIdx]
    # Determine how many samples from the test set will be used for explanations.
    sampleN = min(configs.get("ExplainSamples", 500), xTest.shape[0])
    # Select a subset of test samples to compute SHAP explanations on.
    xSample = xTest[:sampleN]
    # Inform the user that SHAP computation is starting and may take time.
    print("Computing SHAP values (this may be slow)...")
    # Compute SHAP values for the model predictions with the chosen background and sample data.
    shapVals = ComputeShapValues(
      model,
      backgroundData=background,
      xData=xSample,
      featureNames=featureNames,
      nsamples=configs.get("ExplainSamples", 500)
    )
    # Save a SHAP summary plot to the explainability artifacts directory.
    ShapSummaryPlot(
      shapVals,
      featureNames=featureNames,
      savePath=os.path.join(explainDir, "ShapSummary.png")
    )
    # Inform the user that the SHAP summary plot was persisted.
    print("Saved SHAP summary to explain artifacts.")
    # Train a surrogate decision tree on model predictions to extract interpretable rules.
    surrogateSampleN = min(sampleN, xTrain.shape[0])
    surrogateIdx = np.random.choice(xTrain.shape[0], surrogateSampleN, replace=False)
    surrogateX = xTrain[surrogateIdx]
    # Fit the surrogate tree and retrieve textual rules describing model behavior.
    clf, rules = TrainSurrogateTree(model, surrogateX, maxDepth=3)
    # Write the surrogate rules to a text file in the explainability directory for review.
    with open(os.path.join(explainDir, "SurrogateRules.txt"), "w") as f:
      f.write(rules)
    # Inform the user that surrogate rules were saved to disk.
    print("Saved surrogate rules to explain artifacts.")

  # Backfill a copy of the configs used for this run if the pipeline did not already save them.
  if (configs is not None and not os.path.exists(os.path.join(saveDirLocal, "ConfigsUsed.json"))):
    # Create a shallow copy of the configs and remove the potentially large Models list.
    configsCopy = dict(configs)
    configsCopy.pop("Models", None)
    # Add metadata about which model and input sizes were used during this run.
    configsCopy["ModelUsed"] = modelName
    configsCopy["InputSize"] = inputSize
    configsCopy["NumClasses"] = numClasses
    # Persist the trimmed configs copy to disk for reproducibility.
    with open(os.path.join(saveDirLocal, "ConfigsUsed.json"), "w") as f:
      json.dump(configsCopy, f, indent=4)
    # Inform the user that a copy of the configs was saved.
    print(f"Saved a copy of the configs used for this dataset to {os.path.join(saveDirLocal, 'ConfigsUsed.json')}.")

  # Print a header indicating that a post-training evaluation on the full dataset is about to run.
  print(f"\n=== Post-training evaluation on AllData.csv for {baseFileName}_{modelName} ===")
  try:
    # Find the best model by looking for the checkpoint file saved during training in the Checkpoints subdirectory.
    # The best checkpoint is expected to have the lowest loss value.
    # Files format: CheckpointEpoch339_Metric_0_0009.pt
    allChecks = sorted([
      f
      for f in os.listdir(os.path.join(saveDirLocal, "Checkpoints"))
      if (f.startswith("CheckpointEpoch") and f.endswith(".pt"))
    ])
    bestCheck = None
    bestLoss = float("inf")
    for check in allChecks:
      try:
        # Extract the loss value from the checkpoint filename using string manipulation.
        lossStr = check.split("_Metric_")[-1].replace(".pt", "").replace("_", ".")
        lossVal = float(lossStr)
        # Update the best checkpoint if a lower loss value is found.
        if (lossVal < bestLoss):
          bestLoss = lossVal
          bestCheck = check
      except Exception:
        # Skip files that do not conform to the expected naming convention.
        continue
    # Compose the path to the expected best-checkpoint file produced during training.
    modelPath = os.path.join(saveDirLocal, "Checkpoints", bestCheck) if (bestCheck is not None) else None
    # If a checkpoint exists, load it into a fresh model instance for evaluation.
    if (os.path.exists(modelPath)):
      # Instantiate a fresh model with matching architecture for loading weights.
      evalModel = GetModel(modelName, inputSize, numClasses)
      # Load the saved checkpoint weights into the model instance on the target device.
      evalModel = LoadModel(
        evalModel,
        modelPath,
        device=device,
        weightsOnly=False,
      )
      # Inform the user the best model checkpoint was successfully loaded.
      print("Loaded best model for post-training evaluation.")
    else:
      # Fall back to the in-memory model when the checkpoint file is not available.
      evalModel = model
      # Inform the user that the checkpoint was not found and the in-memory model will be used.
      print("Best checkpoint not found; using in-memory model for evaluation.")

    # Evaluate the selected model on the full raw dataset and export artifacts to disk.
    evalResults = GenericTabularEvaluatePredictPlotSubset(
      dataPath=os.path.join(saveDirLocal, "AllRaw.csv"),
      model=evalModel,
      targetColumn="Label",
      featureColumns=featureNames if (featureNames) else None,
      subset=None,  # Evaluate all samples in the file.
      prefix=None,
      ignoreCategorical=configs.get("IgnoreCategorical", True),
      batchSize=configs.get("EvalBatchSize", 64),
      storageDir=os.path.join(saveDirLocal, "TabularEvalPredResults"),
      tabularProcessorDir=saveDirLocal,
      heavy=configs.get("Heavy", True),
      computeECE=configs.get("ComputeECE", True),
      exportFailureCases=configs.get("ExportFailureCases", True),
      eps=configs.get("Eps", 1e-10),
      saveArtifacts=configs.get("SaveArtifacts", True),
      maxSamplesToEval=configs.get("MaxSamplesToEval", None),  # Evaluate all samples.
      dpi=configs.get("PlotDPI", 300),
      fontSize=configs.get("FontSize", 15),
      device=device,
      figSize=configs.get("EvalFigSize", (10, 10)),
      numericScaler=configs.get("NumericScaler", None),
    )
    # Unpack returned evaluation artifacts and metrics from the evaluation helper.
    (
      predsCsvPath, weightedMetrics, predsIdx, gtsIdx,
      probs, confs, records, classNamesEval, cmEval
    ) = evalResults
    # Print summary metrics returned by the evaluation routine for quick review.
    print(f"Post-training evaluation metrics on AllData.csv for {baseFileName}:")
    for key, value in weightedMetrics.items():
      # Format numeric metric values to four decimal places and print others directly.
      if (isinstance(value, (int, float))):
        print(f"  {key}: {value:.4f}")
      else:
        print(f"  {key}: {value}")
    # Inform the user where the predictions CSV was saved by the evaluator.
    print(f"Predictions CSV saved to: {predsCsvPath}")
  except Exception as evalErr:
    # Catch and print any errors that occurred during post-training evaluation.
    print(f"Warning: Post-training evaluation failed for {baseFileName}: {evalErr}")


def RunExplainabilityPhase(
  modelName,
  csvPath,
  configs,
  saveDirLocal,
  labelColumn="Label",
  dropFirstColumn=False,
):
  # Explainability phase uses saved artifacts (preprocessor + checkpoint) to compute SHAP and surrogate rules.
  if (not os.path.exists(saveDirLocal)):
    print(f"Explainability: save directory not found: {saveDirLocal}")
    return

  print(f"Running explainability for {modelName} on {csvPath} -> using artifacts in {saveDirLocal}")

  # Try to load raw data and preprocessor artifacts.
  rawCandidates = [
    os.path.join(saveDirLocal, "AllRaw.csv"),
    os.path.join(saveDirLocal, "AllData.csv"),
    os.path.join(saveDirLocal, "AllRaw.csv")
  ]
  rawPath = None
  for p in rawCandidates:
    if (p and os.path.exists(p)):
      rawPath = p
      break
  if (rawPath is None):
    print(f"No raw/data CSV found in {saveDirLocal}. Looked for: {rawCandidates}")
    return

  try:
    dfAll = pd.read_csv(rawPath, low_memory=False)
  except Exception as e:
    print(f"Failed to read data file {rawPath}: {e}")
    return

  preprocessor = TabularPreprocessor(
    ignoreCategorical=configs.get("IgnoreCategorical", True),
    numericScaler=configs.get("NumericScaler", None)
  )
  try:
    preprocessor.Load(saveDirLocal)
  except Exception:
    # also try Trial_1
    try:
      preprocessor.Load(os.path.join(saveDirLocal, "Trial_1"))
    except Exception:
      pass

  if (dropFirstColumn):
    try:
      dfAll = dfAll.iloc[:, 1:]
    except Exception:
      pass

  # Transform using available preprocessor; fall back to raw numeric values if not possible.
  try:
    xAll, yAll = preprocessor.Transform(dfAll, labelColumn=labelColumn)
  except Exception:
    try:
      # If transform fails but data already numeric with Label column
      if (labelColumn in dfAll.columns):
        yAll = dfAll[labelColumn].values
        xAll = dfAll.drop(columns=[labelColumn]).values
      else:
        xAll = dfAll.values
        yAll = np.zeros(xAll.shape[0], dtype=int)
    except Exception as e:
      print(f"Failed to prepare features for explainability: {e}")
      return

  if (xAll is None or xAll.size == 0):
    print("No features available for explainability.")
    return

  featureNames = preprocessor.GetFeatureNames() if preprocessor.IsLoaded() else None
  inputSize = xAll.shape[1]
  numClasses = len(set(yAll)) if (yAll is not None) else None

  # Find best checkpoint in Checkpoints folder
  checkpointsDir = os.path.join(saveDirLocal, "Checkpoints")
  bestCheck = None
  if (os.path.isdir(checkpointsDir)):
    allChecks = sorted(
      [f for f in os.listdir(checkpointsDir) if (f.startswith("CheckpointEpoch") and f.endswith('.pt'))])
    bestLoss = float('inf')
    for check in allChecks:
      try:
        lossStr = check.split("_Metric_")[-1].replace('.pt', '').replace('_', '.')
        lossVal = float(lossStr)
        if (lossVal < bestLoss):
          bestLoss = lossVal
          bestCheck = check
      except Exception:
        continue

  modelDevice = torch.device(
    "cuda" if (torch.cuda.is_available() and configs.get("Device", "cpu").startswith("cuda")) else "cpu")
  try:
    model = GetModel(modelName, inputSize, numClasses if (numClasses is not None) else 1)
    if (bestCheck is not None):
      modelPath = os.path.join(checkpointsDir, bestCheck)
      model = LoadModel(model, modelPath, device=modelDevice, weightsOnly=False)
    model.to(modelDevice)
    model.eval()
  except Exception as e:
    print(f"Failed to instantiate/load model for explainability: {e}")
    return

  explainDir = configs.get("ExplainSaveDir", os.path.join(saveDirLocal, "ExplainArtifacts"))
  os.makedirs(explainDir, exist_ok=True)

  bgN = min(configs.get("ExplainBackground", 200), xAll.shape[0])
  bgIdx = np.random.choice(xAll.shape[0], bgN, replace=False)
  background = xAll[bgIdx]
  sampleN = min(configs.get("ExplainSamples", 500), xAll.shape[0])
  xSample = xAll[:sampleN]

  try:
    print("Computing SHAP values for explainability (may be slow)...")
    shapVals = ComputeShapValues(
      model, background, xSample, featureNames=featureNames,
      nsamples=configs.get("ExplainSamples", 500)
    )
    ShapSummaryPlot(shapVals, featureNames=featureNames, savePath=os.path.join(explainDir, "ShapSummary.png"))
    print(f"Saved SHAP summary to {explainDir}")
  except Exception as e:
    print(f"SHAP computation failed: {e}")

  try:
    surrogateSampleN = min(sampleN, xAll.shape[0])
    surrogateX = xAll[np.random.choice(xAll.shape[0], surrogateSampleN, replace=False)]
    clf, rules = TrainSurrogateTree(model, surrogateX, maxDepth=configs.get("SurrogateMaxDepth", 3))
    with open(os.path.join(explainDir, "SurrogateRules.txt"), "w") as f:
      f.write(rules)
    print(f"Saved surrogate tree rules to {os.path.join(explainDir, 'SurrogateRules.txt')}")
  except Exception as e:
    print(f"Surrogate tree training failed: {e}")


# If executed as a script, run the Run function.
if (__name__ == "__main__"):
  # Update default matplotlib settings to configured preferences.
  UpdateMatplotlibSettings()

  # Parse command-line arguments provided to the script into an args namespace.
  args = GetArgs()
  # Validate parsed arguments to ensure required inputs and correct types.
  ValidateArgs(args)
  # Extract the data directory path from the parsed arguments.
  dataDir = args.dataDir
  # Build a list of CSV file paths found inside the provided data directory.
  csvFiles = [os.path.join(dataDir, f) for f in os.listdir(dataDir) if (f.lower().endswith(".csv"))]
  # Raise an error when no CSV files were discovered in the specified directory.
  if (len(csvFiles) == 0):
    # Notify the user that no CSV files were found and abort execution.
    raise RuntimeError(f"No CSV files found in data directory: {dataDir}")

  # Read the project-wide configuration JSON into a dictionary for runtime options.
  configs = ReadProjectConfig(args.configPath)
  # Get the list of model names defined in the configuration file.
  modelsNames = configs.get("Models", [])

  # Transfer commonly used CLI argument values into local variables for convenience.
  labelColumn = args.labelColumn
  dropFirstColumn = args.dropFirstColumn
  noOfTrials = args.noOfTrials
  saveDir = args.saveDir
  phase = args.phase

  # Execute the training phase when requested by the CLI argument.
  if (phase == "training"):
    # Iterate through each model configured in the project.
    for modelName in modelsNames:
      # Iterate through each CSV dataset discovered in the data directory.
      for csvFile in csvFiles:
        try:
          # Print a header announcing which model and dataset are being processed.
          print(f"\n\n=== Running pipeline for model: {modelName} on dataset: {csvFile} ===")
          # Compute a base file name for the dataset from the CSV filename.
          baseFileName = os.path.splitext(os.path.basename(csvFile))[0]
          # Determine the base save directory for this dataset and model.
          if (saveDir is None):
            # Use a default results folder name when no explicit save directory is provided.
            baseSaveDir = f"Results_{baseFileName}_{modelName}"
          else:
            # Use the provided save directory as the parent for per-dataset/model folders.
            baseSaveDir = os.path.join(saveDir, baseFileName, modelName)
          # Run the requested number of independent trials, creating subfolders when >1.
          for t in range(max(1, int(noOfTrials))):
            # Create a trial-specific subdirectory when multiple trials are requested.
            if (int(noOfTrials) > 1):
              trialSaveDir = os.path.join(baseSaveDir, f"Trial_{t + 1}")
            else:
              # Use the base save directory when only a single trial is requested.
              trialSaveDir = baseSaveDir
            # Ensure the trial save directory exists on disk.
            os.makedirs(trialSaveDir, exist_ok=True)
            # Print which trial is starting and where outputs will be saved.
            print(f"--> Trial {t + 1}/{noOfTrials}: saving to {trialSaveDir}")
            # Execute the per-CSV tabular pipeline for the current trial folder.
            RunTabularPipelineForCSV(modelName, csvFile, configs, trialSaveDir, labelColumn, dropFirstColumn)
        except Exception as e:
          # Print any exception that occurs while processing a model/dataset combination.
          print(f"Error occurred while processing: {e}")
    # Inform the user that the training phase has completed for all requested runs.
    print("Training phase completed for all models and CSV files.")

  # Execute only the statistical analysis phase when specified by the CLI argument.
  elif (phase == "statistical"):
    if (args.addFusedModel):
      # If the fused model flag is set, add a "FusedModel" entry to the list of models for analysis.
      modelsNames.append("FusedModel")
      print("Added 'FusedModel' to the list of models for statistical analysis.")
    if (args.fusedModelOnly):
      # If the fused model only flag is set, restrict the list of models to just "FusedModel".
      modelsNames = ["FusedModel"]
      print("Restricted statistical analysis to 'FusedModel' only.")
    # Iterate through configured models for performing statistical aggregation.
    for modelName in modelsNames:
      # Iterate through each CSV dataset to run statistics for their results.
      for csvFile in csvFiles:
        # Print a header announcing the statistical phase for the model and dataset.
        print(f"\n\n=== Running statistical phase for model: {modelName} on dataset: {csvFile} ===")
        # Determine the base file name for the dataset from its CSV path.
        baseFileName = os.path.splitext(os.path.basename(csvFile))[0]
        # Compute the base save directory for the dataset and model combination.
        if (saveDir is None):
          # Use a default results folder pattern when no explicit save directory is provided.
          baseSaveDir = f"Results_{baseFileName}_{modelName}"
        else:
          # Use the provided parent save directory to construct the per-dataset path.
          baseSaveDir = os.path.join(saveDir, baseFileName, modelName)
        # Run the statistical analysis phase using the assembled save directory path.
        RunStatisticalPhase(modelName, csvFile, configs, baseSaveDir)
    # Inform the user that the statistical phase completed for all requested runs.
    print("Statistical phase completed for all models and CSV files.")

  # Execute the fusion phase when specified by the CLI argument.
  elif (phase == "fused"):
    # Determine output fused model folder name from configs or default.
    outModelName = configs.get("FusedModelName", "FusedModel")
    # Experiments root where per-dataset folders are stored is saveDir by convention.
    experimentsRoot = saveDir if (saveDir is not None) else "Experiments"
    if (not os.path.isdir(experimentsRoot)):
      raise RuntimeError(f"Experiments directory for fusion not found: {experimentsRoot}")

    # Iterate over dataset subfolders and run fusion using `FuseModelsFromData`.
    for dataset in os.listdir(experimentsRoot):
      experimentPath = os.path.join(experimentsRoot, dataset)
      if (not os.path.isdir(experimentPath)):
        continue
      try:
        print(f"\n\n=== Running fusion for: {experimentPath} ===")
        FuseModelsFromData(experimentPath, modelsNames, outModelName=outModelName)
      except Exception as e:
        print(f"Warning: fusion failed for {experimentPath}: {e}")

    print("Fusion phase completed for all datasets.")

  # Execute the explainability phase when specified by the CLI argument.
  elif (phase == "explain"):
    # Iterate through each model configured in the project.
    for modelName in modelsNames:
      # Iterate through each CSV dataset discovered in the data directory.
      for csvFile in csvFiles:
        try:
          # Print a header announcing which model and dataset are being processed for explainability.
          print(f"\n\n=== Running explainability for model: {modelName} on dataset: {csvFile} ===")
          # Compute a base file name for the dataset from the CSV filename.
          baseFileName = os.path.splitext(os.path.basename(csvFile))[0]
          # Determine the base save directory for this dataset and model.
          if (saveDir is None):
            # Use a default results folder name when no explicit save directory is provided.
            baseSaveDir = f"Results_{baseFileName}_{modelName}"
          else:
            # Use the provided save directory as the parent for per-dataset/model folders.
            baseSaveDir = os.path.join(saveDir, baseFileName, modelName)

          # Search for the trials subdirectories under the base save directory to find all of them.
          trialDirs = []
          if (os.path.isdir(baseSaveDir)):
            for entry in os.listdir(baseSaveDir):
              entryPath = os.path.join(baseSaveDir, entry)
              if (os.path.isdir(entryPath) and entry.startswith("Trial_")):
                trialDirs.append(entryPath)
          # If no trial subdirectories are found, use the base save directory itself as the trial directory.
          if (len(trialDirs) == 0):
            trialDirs = [baseSaveDir]

          # Run the explainability phase for each trial directory found, which will look for artifacts in those directories.
          for t, trialDir in enumerate(trialDirs):
            # Ensure path exists (RunExplainabilityPhase will check), and announce the trial.
            print(f"--> Explainability for Trial {t + 1}/{len(trialDirs)}: looking for artifacts in {trialDir}")
            RunExplainabilityPhase(modelName, csvFile, configs, trialDir, labelColumn, dropFirstColumn)
        except Exception as e:
          # Print any exception that occurs while processing a model/dataset combination for explainability.
          print(f"Error occurred while processing for explainability: {e}")
    # Inform the user that the explainability phase has completed for all requested runs.
    print("Explainability phase completed for all models and CSV files.")
