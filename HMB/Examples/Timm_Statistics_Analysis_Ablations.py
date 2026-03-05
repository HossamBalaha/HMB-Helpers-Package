import argparse, os, timm, torch, json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding
from HMB.PerformanceMetrics import (
  CalculatePerformanceMetrics, PlotMultiTrialROCAUC, PlotMultiTrialPRCurve,
  PlotCalibrationCurve, SampleMonteCarloDirichletFromProbs,
  ComputeMonteCarloUncertaintyMeasures,
  ComputeECEPlotReliability
)
from HMB.StatisticalAnalysisHelper import ExtractDataFromSummaryFile, PlotMetrics, StatisticalAnalysis
from HMB.ExplainabilityHelper import CAMExplainerPyTorch
from HMB.PyTorchHelper import LoadPyTorchDict, EvaluateModelOnPerturbations
from HMB.Initializations import IMAGE_SUFFIXES
from HMB.PyTorchModelMemoryProfiler import PyTorchModelMemoryProfiler

# Ensure all prints flush by default to make logs appear promptly.
# Use the real built-in print function for delegation to avoid analyzer warnings.
import builtins as _builtins

_original_print = _builtins.print


# Define a wrapper that sets flush=True when not explicitly provided.
def print(*args, **kwargs):
  # Ensure flush is True by default when not provided.
  if ("flush" not in kwargs):
    kwargs["flush"] = True
  # Delegate to the original print implementation.
  return _original_print(*args, **kwargs)


# Define a function to parse command line arguments for this statistics analysis script.
def GetArgs():
  parser = argparse.ArgumentParser(
    description="Aggregate trial predictions and plot performance metrics across systems")
  parser.add_argument(
    "--baseExpDir",
    type=str,
    required=False,
    default=r"/home/hmbala01/BC-Group/Experiments",
    help="Base experiments directory containing system subfolders (default: %(default)s)"
  )
  parser.add_argument(
    "--predCSVFileFix",
    type=str,
    default="_Predictions_",
    help="Substring used to identify per-trial prediction CSV files (default: '%(default)s')"
  )
  parser.add_argument(
    "--subsets",
    nargs='+',
    default=["Train", "Test"],
    help="Subset names to process (space separated), e.g. --subsets Train Val Test"
  )
  parser.add_argument(
    "--actualColName",
    type=str,
    default="trueClassName",
    help="Column name used for ground-truth labels in prediction CSVs"
  )
  parser.add_argument(
    "--actualColIDColName",
    type=str,
    default="trueClassIndex",
    help="Column name used for ground-truth label indices in prediction CSVs (if applicable for ROC/PR curves)"
  )
  parser.add_argument(
    "--predictionColName",
    type=str,
    default="predictedClassName",
    help="Column name used for predicted labels in prediction CSVs"
  )
  parser.add_argument(
    "--probabilityColName",
    type=str,
    default="probabilities",
    help="Column name used for predicted probabilities in prediction CSVs (if applicable for ROC/PR curves)"
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose logging"
  )
  parser.add_argument(
    "--dpi",
    type=int,
    default=720,
    help="DPI to use when saving figures (default: %(default)s)"
  )
  parser.add_argument(
    "--explainMethods",
    nargs='+',
    default=[],
    help="List of explainability methods to compute (e.g. gradcam). Multiple values allowed."
  )
  parser.add_argument(
    "--maxExplainImages",
    type=int,
    default=0,
    help="Maximum number of images to run explainability on per trial (0 to disable)."
  )
  parser.add_argument(
    "--explainDatasetDir",
    type=str,
    default=None,
    help="Path to the dataset directory to use for explainability (required if --explainMethods is specified)"
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for model loading and explainability (default: %(default)s)"
  )
  parser.add_argument(
    "--maxPerturbImages",
    type=int,
    default=0,
    help="Maximum number of images to evaluate for perturbation analysis (0 to disable)"
  )

  return parser.parse_args()


# Validate and normalize command-line arguments for this script.
def ValidateArgs(args):
  if (not isinstance(args.baseExpDir, str) or not args.baseExpDir):
    raise ValueError("--baseExpDir must be a non-empty string path to the experiments folder")
  if (not os.path.exists(args.baseExpDir) or not os.path.isdir(args.baseExpDir)):
    raise FileNotFoundError(f"baseExpDir path does not exist or is not a directory: {args.baseExpDir}")

  if (not isinstance(args.predCSVFileFix, str) or not args.predCSVFileFix):
    raise ValueError("--predCSVFileFix must be a non-empty string")

  if (not isinstance(args.subsets, (list, tuple)) or len(args.subsets) == 0):
    raise ValueError("--subsets must be a non-empty list of subset names (e.g. --subsets Train Val Test)")
  # Normalize subset names to the expected capitalization (first letter uppercase)
  args.subsets = [str(s) for s in args.subsets]

  # Explain methods should be a list of strings
  if (not isinstance(args.explainMethods, (list, tuple))):
    raise ValueError("--explainMethods must be a list of method names or omitted")
  args.explainMethods = [str(m).lower() for m in args.explainMethods if (m is not None and str(m).strip())]

  # maxExplainImages should be non-negative
  if (not isinstance(args.maxExplainImages, int) or args.maxExplainImages < 0):
    raise ValueError("--maxExplainImages must be a non-negative integer")

  # If explainDatasetDir is provided, it must be a valid directory; also require it if explainMethods are specified.
  if (
      args.explainDatasetDir and (
      not os.path.exists(args.explainDatasetDir) or not os.path.isdir(args.explainDatasetDir))
  ):
    raise FileNotFoundError(f"--explainDatasetDir path does not exist or is not a directory: {args.explainDatasetDir}")

  if (not isinstance(args.actualColName, str) or not args.actualColName):
    raise ValueError("--actualColName must be a non-empty string")
  if (not isinstance(args.actualColIDColName, str) or not args.actualColIDColName):
    raise ValueError("--actualColIDColName must be a non-empty string")
  if (not isinstance(args.predictionColName, str) or not args.predictionColName):
    raise ValueError("--predictionColName must be a non-empty string")
  if (not isinstance(args.probabilityColName, str) or not args.probabilityColName):
    raise ValueError("--probabilityColName must be a non-empty string")

  if (not isinstance(args.verbose, bool)):
    # argparse sets this as bool when using store_true; still validate defensively.
    args.verbose = bool(args.verbose)

  if (not isinstance(args.maxPerturbImages, int) or args.maxPerturbImages < 0):
    raise ValueError("--maxPerturbImages must be a non-negative integer")

  # Validate DPI argument
  if (not isinstance(args.dpi, int) or args.dpi <= 0):
    raise ValueError("--dpi must be a positive integer")

  # Print parsed args when verbose to aid debugging.
  if (args.verbose):
    print("Parsed arguments:")
    for k, v in vars(args).items():
      print(f"  {k}: {v}")

  return args


# Create and configure the model for the current task.
def CreateModel(modelName, numClasses):
  # Notify that model creation has started.
  print("Creating model...")
  # Instantiate the timm model with the requested number of classes.
  model = timm.create_model(modelName, pretrained=True, num_classes=numClasses)
  print(f"Model {modelName} created with {numClasses} output classes.")
  # Return the constructed model object.
  return model


def CereateModelTransforms(model):
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
  return modelTransform


# Create a callable factory for the model predictions.
def CreatePredictCallable(model, transform, device, imageSize):
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


# Execute the script when run directly.
def Main():
  # Suppress noisy warnings early in execution.
  # IgnoreWarnings()
  # Set random seeds for reproducibility.
  DoRandomSeeding()

  # Parse and validate CLI arguments.
  args = GetArgs()
  args = ValidateArgs(args)

  def ProcessSystem(
      subsets,
      predCSVFileFix,
      baseOutputDir,
      verbose,
      actualColName,
      actualColIDColName,
      predictionColName,
      probabilityColName,
      dpi,
      explainMethods=None,
      maxExplainImages=0,
      explainDatasetDir=None,
      device="cuda",
      maxPerturbImages=0,
  ):
    trials = [
      el for el in os.listdir(baseOutputDir)
      if (os.path.isdir(os.path.join(baseOutputDir, el)) and el != "PerformanceMetricsPlots")
    ]
    trials = list(sorted(trials))  # Sort trials alphabetically for consistent processing order.

    # Merge predictions from all trials into a single DataFrame for analysis.
    allPredictions = None

    for trial in trials:
      trialDir = os.path.join(baseOutputDir, trial)
      if (not os.path.isdir(trialDir)):
        if (verbose):
          print(f"Skipping non-directory item: {trialDir}")
        continue  # Skip non-directory files.

      # Save the args used for this trial to a JSON file within the trial directory for traceability.
      argsJsonPath = os.path.join(trialDir, "FollowUpArgs.json")
      with open(argsJsonPath, "w") as f:
        json.dump(vars(args), f, indent=4)

      # ---------------------------------------------------------------------------------- #
      argsJsonPath = os.path.join(trialDir, "TrainingArgs.json")
      trainingArgs = {}
      imgSize = 224  # Default image size if not found in training args.
      if (not os.path.exists(argsJsonPath)):
        if (verbose):
          print(f"Warning: `TrainingArgs.json` not found for trial '{trial}' at expected path: {argsJsonPath}")
        modelName = None
      else:
        if (verbose):
          print(f"Found `TrainingArgs.json` for trial '{trial}': {argsJsonPath}")
        # Optionally, you could load and inspect the training arguments here if needed.
        with open(argsJsonPath, "r") as f:
          trainingArgs = json.load(f)
        modelName = trainingArgs.get("modelName", None)
      if (modelName is None):
        if (verbose):
          print(
            f"No model name found in training args for trial '{trial}'. "
            f"Attempting to infer from trial name..."
          )
        parentDirName = os.path.basename(os.path.dirname(trialDir))
        lut = {
          "ConvNeXtV2" : ("convnextv2_nano.fcmae_ft_in22k_in1k", 224),
          "EVA02"      : ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 448),
          "ViTBaseP16" : ("vit_base_patch16_224.augreg2_in21k_ft_in1k", 224),
          "MobileNetV3": ("mobilenetv3_small_100.lamb_in1k", 224),
        }
        modelObject = lut.get(parentDirName, None)
        if (verbose):
          print(
            f"Parent directory name for trial '{trial}': '{parentDirName}'. "
            f"Looking up in LUT for model name and image size..."
          )
        if (modelObject is not None):
          modelName = modelObject[0]
          imgSize = modelObject[1]
          if (verbose):
            print(
              f"Inferred model name '{modelName}' and image size {imgSize} for "
              f"trial '{trial}' from LUT based on trial name."
            )
        else:
          if (verbose):
            print(
              f"No model name found in training args or LUT for trial '{trial}'. "
              f"Explainability will be skipped for this trial."
            )

      model = None
      if (modelName is not None):
        numClasses = trainingArgs.get("numClasses", 4)
        bestModelPath = os.path.join(trialDir, "BestModel.pth")
        model = CreateModel(modelName, numClasses)
        stateDict = LoadPyTorchDict(bestModelPath, device=device)

        modelStateDict = stateDict.get("model_state_dict", None)
        if (modelStateDict is None):
          raise ValueError(
            "Model state dict not found in checkpoint. Ensure that the checkpoint contains the model state."
          )

        model.load_state_dict(modelStateDict)
        if (verbose):
          print(f"Model loaded from checkpoint: {bestModelPath}")

        modelTransform = CereateModelTransforms(model)
        if (verbose):
          print(f"Model transform created for model '{modelName}' with expected input size {imgSize}x{imgSize}.")

        # Create profiler instance with standard ImageNet input dimensions.
        profiler = PyTorchModelMemoryProfiler(
          model=model,
          inputShape=(3, imgSize, imgSize),  # Use the inferred image size for profiling.
          batchSize=1,
          precision="FP32",
          device=device,
        )
        # Profile transformer with sequence length derived from patch embedding.
        try:
          # Execute comprehensive memory profiling.
          memoryReport = profiler.ProfileModelMemory(
            optimizerType="Adam",
            isTransformer=False,
            checkpointing=True,
            checkpointSavingsFactor=0.5,
            deviceFLOPSGFLOPS=5000.0,
            datasetSize=100000,
            trainingMultiplier=3.0,
            runMicroBenchmark=False
          )
        except Exception as e:
          if (verbose):
            print(f"Error during memory profiling for trial '{trial}': {e}")
          memoryReport = profiler.ProfileModelMemory(
            optimizerType="Adam",
            optimizerKwargs={"amsgrad": True},
            isTransformer=True,
            sequenceLength=4096,
            checkpointing=True,
            checkpointSavingsFactor=0.5,
            deviceFLOPSGFLOPS=5000.0,
            datasetSize=100000,
            trainingMultiplier=3.0,
            runMicroBenchmark=False
          )
        profiler.PrintMemoryReport(memoryReport)
        reportFilePath = os.path.join(trialDir, "MemoryReport.json")
        profiler.SaveProfileToJSON(memoryReport, reportFilePath)

        device = torch.device(device)
        model.to(device)
        model.eval()

        callable = CreatePredictCallable(model, modelTransform, device, imgSize)
        if (verbose):
          print(f"Created prediction callable for model '{modelName}' on trial '{trial}'")

        if (explainDatasetDir is not None and maxPerturbImages is not None and maxPerturbImages > 0):
          storeDir = os.path.join(trialDir, "Perturbation_Evaluation")
          EvaluateModelOnPerturbations(
            callable,
            trialDir,
            explainDatasetDir,
            storeDir,
            perturbations=[],
            levels=[],
            maxSamples=maxPerturbImages,
            preprocessFn=None,
            subset=None,
            eps=1e-10,
            dpi=dpi,
          )
        else:
          if (verbose):
            print(
              f"Perturbation evaluation skipped for trial '{trial}' because "
              f"explainDatasetDir is '{explainDatasetDir}' and maxPerturbImages is {maxPerturbImages}."
            )

        if (explainDatasetDir is not None and explainMethods and maxExplainImages > 0):
          for method in explainMethods:
            if (verbose):
              print(f"Initializing CAM explainer for method '{method}' on trial '{trial}' with model '{modelName}'")
            try:
              expl = CAMExplainerPyTorch(
                torchModel=model,
                device=device,
                camType=method,
                imgSize=imgSize,
                outputBase=os.path.join(trialDir, f"CAM_Explanations_{method}"),
                debug=verbose,
                figsize=(14, 16),
              )

              classes = sorted(os.listdir(explainDatasetDir))
              for cls in classes:
                if (verbose):
                  print(f"Processing class '{cls}' for explainability with method '{method}' in trial '{trial}'")
                clsDir = os.path.join(explainDatasetDir, cls)
                if (not os.path.isdir(clsDir)):
                  continue
                imgFiles = [f for f in os.listdir(clsDir) if (f.lower().endswith(tuple(IMAGE_SUFFIXES)))]
                imgFiles = sorted(imgFiles)[:maxExplainImages]  # Limit to maxExplainImages if specified.
                for imgFile in imgFiles:
                  imgPath = os.path.join(clsDir, imgFile)
                  if (verbose):
                    print(f"Processing explainability for image: {imgPath} with method '{method}'")
                  result = expl.ProcessImage(imgPath, classNames={i: cls for i, cls in enumerate(classes)})
                  if (verbose):
                    print(f"Processed explainability for image: {imgPath} with method '{method}'. Result: {result}")

            except Exception as e:
              if (verbose):
                print(f"Failed to initialize CAM explainer for method '{method}': {e}")
        else:
          if (verbose):
            print(
              f"Explainability skipped for trial '{trial}' because explainDatasetDir is '{explainDatasetDir}', "
              f"explainMethods is {explainMethods}, and maxExplainImages is {maxExplainImages}."
            )
      # ---------------------------------------------------------------------------------- #

      trialPredictions = []

      # ---------------------------------------------------------------------------------- #
      #
      # ---------------------------------------------------------------------------------- #
      for subset in subsets:
        # Look for the prediction CSV file for the current trial and subset.
        predCSVFile = [
          f for f in os.listdir(trialDir)
          if ((predCSVFileFix in f) and (subset in f) and (f.endswith(".csv")))
        ]
        if (len(predCSVFile) == 0):
          if (verbose):
            print(f"No prediction CSV file found for trial '{trial}' and subset '{subset}'.")
          continue  # Skip if no prediction file is found.
        elif (len(predCSVFile) > 1):
          if (verbose):
            print(
              f"Multiple prediction CSV files found for trial '{trial}' and "
              f"subset '{subset}': {predCSVFile}. Skipping this subset."
            )
          continue  # Skip if multiple prediction files are found.
        else:
          predCSVFile = os.path.join(trialDir, predCSVFile[0])  # Get the single prediction file.

        if (os.path.isfile(predCSVFile)):
          df = pd.read_csv(predCSVFile)
          # Append the rows to the trialPredictions list.
          dfList = df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries.
          trialPredictions.extend(dfList)  # Add to the trial predictions.
        else:
          if (verbose):
            print(f"Prediction CSV file does not exist: {predCSVFile}")
      # ---------------------------------------------------------------------------------- #

      # Convert list of dictionaries back to DataFrame.
      concDF = pd.DataFrame(trialPredictions)
      # Rename columns except for "image".
      concDF.rename(columns={col: f"{col}_{trial}" for col in concDF.columns if (col != "image")}, inplace=True)

      if (verbose):
        print(f"Trial '{trial}' - Concatenated predictions shape: {concDF.shape}")
        print(f"Trial '{trial}' - Sample of concatenated predictions:\n{concDF.head()}")

      if (allPredictions is None):
        # Update `trialPredictions` with the concatenated DataFrame.
        allPredictions = concDF
      else:
        # Merge the current trial's predictions with the accumulated predictions on the 'image' column.
        allPredictions = pd.merge(
          concDF,
          allPredictions,
          on="image",
          how="outer",  # Use outer join to keep all images across trials.
        )
        if (verbose):
          print(f"After merging trial '{trial}', combined predictions shape: {allPredictions.shape}")

    # Concatenate all trial predictions into a single DataFrame for analysis.
    if (len(allPredictions) > 0):
      # Combine all trial predictions into a single DataFrame, aligning on the 'image' column.
      # Save the combined predictions to a new CSV file for further analysis.
      allPredictions.to_csv(
        os.path.join(baseOutputDir, "Combined_Predictions.csv"),
        index=False,
      )
      if (verbose):
        print(f"Combined predictions saved to: {os.path.join(baseOutputDir, 'Combined_Predictions.csv')}")
    else:
      if (verbose):
        print("No predictions were found across all trials. No combined CSV file created.")

    metrics = {}
    allProbs = []
    actualCol = allPredictions[f"{actualColName}_{trials[0]}"]
    actualColID = allPredictions[f"{actualColIDColName}_{trials[0]}"]

    if (verbose):
      print(f"Trial '{trials[0]}' - Extracted actual labels for metrics calculation and ROC/PR curves.")
      print(f"Trial '{trials[0]}' - Sample actual labels: {actualCol.head()}")
      print(f"Trial '{trials[0]}' - Sample actual label indices: {actualColID.head()}")

    for trial in trials:
      predCol = f"{predictionColName}_{trial}"
      if (predCol in allPredictions.columns):
        predictedCol = allPredictions[predCol]
        cm = confusion_matrix(actualCol, predictedCol)
        trialMetrics = CalculatePerformanceMetrics(cm, addWeightedAverage=True, eps=1e-10)
        metrics[trial] = {k: v for k, v in trialMetrics.items() if ("Weighted" in k)}
        if (verbose):
          print(f"Trial '{trial}' - Performance Metrics:")
          for key, value in trialMetrics.items():
            if ("Weighted" in key):
              print(f"  {key}: {np.round(value, 4)}")
      else:
        if (verbose):
          print(
            f"Prediction column '{predCol}' not found in combined predictions. "
            f"Skipping metrics calculation for trial '{trial}'."
          )

      probCol = f"{probabilityColName}_{trial}"
      if (probCol in allPredictions.columns):
        probs = allPredictions[probCol].tolist()
        # Eval the string representation of the probabilities to convert them back to lists (if they were saved as strings).
        probs = [eval(p) if isinstance(p, str) else p for p in probs]
        allProbs.append(probs)
        if (verbose):
          print(f"Trial '{trial}' - Extracted predicted probabilities for ROC/PR curves.")
          print(f"Trial '{trial}' - Sample predicted probabilities: {probs[:5]}")

        # Plot calibration curve for this trial using the per-trial probabilities and the true label indices.
        # Convert probabilities and labels to numpy arrays suitable for the calibration plotting function.
        probsNp = np.array(probs)
        # actualColID is a pandas Series extracted earlier from the first trial; convert to numpy array here.
        labelsNp = np.array(actualColID.tolist()) if hasattr(actualColID, "tolist") else np.array(actualColID)

        # Only plot if the number of probability rows matches the number of labels.
        if (probsNp.shape[0] != labelsNp.shape[0]):
          if (verbose):
            print(
              f"Skipping calibration plot for trial '{trial}' because number of probability "
              f"rows ({probsNp.shape[0]}) ''"
              f"does not match number of labels ({labelsNp.shape[0]})."
            )
        else:
          calibFile = os.path.join(baseOutputDir, trial, f"Calibration_Curve.pdf")
          PlotCalibrationCurve(
            probsNp, labelsNp,
            nBins=5,
            title=f"Calibration Curve",
            fontSize=12,
            figSize=(5, 5),
            display=False,
            save=True,
            fileName=calibFile,
            dpi=dpi,
            returnFig=False,
            color="green"
          )
          if (verbose):
            print(f"Saved calibration curve for trial '{trial}': {calibFile}")

          # --- Monte Carlo Dirichlet sampling and ECE plotting for this trial ---
          # Sample Monte Carlo Dirichlet distributions from the predicted probabilities.
          # Use sensible defaults: T=500 samples, concentration=30.0 (same as example).
          probsMC = SampleMonteCarloDirichletFromProbs(probsNp, T=500, concentration=30.0)

          # Compute uncertainty measures from the Monte Carlo samples.
          uncertaintyMeasures = ComputeMonteCarloUncertaintyMeasures(probsMC)

          # Extract confidences, predictions and use the true labels we already prepared.
          confidences = np.array(uncertaintyMeasures.get("predictedConfidence"))
          predictedIdx = np.array(uncertaintyMeasures.get("predictedIdx"))
          trueLabels = labelsNp

          # Only compute and save ECE plot if shapes match.
          if (confidences.shape[0] == trueLabels.shape[0] == predictedIdx.shape[0]):
            eceFile = os.path.join(baseOutputDir, trial, f"ECE_Curve.pdf")
            # Compute ECE and save the reliability plot for this trial.
            ece, binAcc, binConf, binCounts = ComputeECEPlotReliability(
              confidences,
              predictedIdx,
              trueLabels,
              nBins=5,
              title=f"ECE",
              fontSize=14,
              figSize=(6, 6),
              display=False,
              save=True,
              fileName=eceFile,
              dpi=dpi,
              returnFig=False,
              cmap="Blues",
              applyXYLimits=True
            )
            if (verbose):
              print(f"Saved ECE/reliability plot for trial '{trial}': {eceFile}")
              print(f"Trial '{trial}' - ECE: {ece}")
          else:
            if (verbose):
              print(
                f"Skipping ECE plot for trial '{trial}' because shapes do not match: "
                f"confidences {confidences.shape}, predicted {predictedIdx.shape}, labels {trueLabels.shape}"
              )
      else:
        if (verbose):
          print(
            f"Probability column '{probCol}' not found in combined predictions. "
            f"Skipping probability extraction and ROC/PR curves for trial '{trial}'."
          )

    actualColID = np.array(actualColID.values.tolist())
    classes = sorted(allPredictions[f"{actualColName}_{trials[0]}"].unique())

    if (verbose):
      print(f"Classes identified for ROC/PR curves: {classes}")
      print(f"Sample of actual label indices for ROC/PR curves: {actualColID[:5]}")

    for which in ["CI", "SD"]:
      fileName = os.path.join(baseOutputDir, f"{which}_MultiTrial_PRC_Curve.pdf")
      PlotMultiTrialPRCurve(
        actualColID,  # List of true labels arrays from all trials.
        allProbs,  # List of predicted probabilities from all trials.
        classes,  # List of class names.
        confidenceLevel=0.95,  # Confidence level for CI.
        which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
        title="Multi-Trial Precision-Recall Curve",
        figSize=(8, 8),  # Figure size in inches.
        cmap=None,  # Colormap for different classes.
        display=False,  # Whether to display the plot.
        save=True,  # Whether to save the plot.
        fileName=fileName,  # File name for saving.
        fontSize=15,  # Font size for labels and annotations.
        showLegend=True,  # Whether to show legend.
        returnFig=False,  # Whether to return the matplotlib figure object.
        dpi=dpi,  # DPI for saving the figure.
        addZoomedInset=True,  # Whether to add a zoomed inset for the top-right corner of the PRC plot.
      )

      fileName = os.path.join(baseOutputDir, f"{which}_MultiTrial_ROC_AUC.pdf")
      PlotMultiTrialROCAUC(
        actualColID,  # List of true labels arrays from all trials.
        allProbs,  # List of predicted probabilities from all trials.
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

    firstRow = [el.split(" ")[1] for el in metrics[trials[0]].keys() if ("Weighted" in el)]
    secondRow = ["Metric"] * len(firstRow)
    metricValues = [
      [metrics[trial][f"Weighted {metric}"] for metric in firstRow]
      for trial in trials
    ]
    # Create a DataFrame to store the metrics for each trial, with the first row containing metric names and the
    # second row containing the keyword "Metric".
    dfMetrics = pd.DataFrame(
      data=metricValues,
      columns=firstRow,
    )
    # Insert the secondRow as the second row in the DataFrame at index 0 (after the header); pushes the metric values down by one row.
    dfMetrics.loc[-1] = secondRow  # Add the second row with "Metric" values.
    dfMetrics.index = dfMetrics.index + 1  # Shift the index to accommodate the new row.
    # Sort the index to maintain the correct order (header, "Metric" row, then metric values).
    dfMetrics.sort_index(inplace=True)
    # Save the DataFrame to a CSV file for comparison.
    trialMetricsComparisonFile = os.path.join(baseOutputDir, "Trial_Metrics_Comparison.csv")
    dfMetrics.to_csv(trialMetricsComparisonFile, index=False)
    if (verbose):
      print(f"Trial metrics comparison saved to: {os.path.join(baseOutputDir, 'Trial_Metrics_Comparison.csv')}")
      print(f"Trial Metrics Comparison:\n{dfMetrics}")

    newFolderName = os.path.join(baseOutputDir, "PerformanceMetricsPlots")
    os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.
    history, names, metrics = ExtractDataFromSummaryFile(trialMetricsComparisonFile)
    PlotMetrics(
      history, names, metrics,
      factor=5,  # Factor to multiply the default figure size.
      keyword="AllMetrics",  # Keyword to append to the filenames of the saved plots.
      dpi=dpi,  # Dots per inch (resolution) of the saved plots.
      xTicksRotation=45,  # Rotation angle for x-axis tick labels.
      whichToPlot=[],  # List of plot types to generate.
      fontSize=14,  # Font size for the plots.
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

    print("\u2713 Performance plots generated.")
    print("\nGenerating statistical analysis report...")
    overallReport = []
    for metric in metrics:
      for index, data in enumerate(history):
        report = StatisticalAnalysis(
          data[metric]["Trials"],
          hypothesizedMean=data[metric]["Mean"],
          secondMetricList=None,
        )
        report["Type"] = names[index]
        report["Metric"] = metric
        overallReport.append(report)
    reportDF = pd.DataFrame(overallReport)
    reportCsvPath = os.path.join(baseOutputDir, "Statistical_Analysis_Report.csv")
    reportDF.to_csv(reportCsvPath, index=False)
    print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")

    if (verbose):
      print(f"Names of the metrics plotted: {names}")
      print(f"Metrics plotted: {metrics}")
      print(f"History of metric values plotted:")
      print(history)
      print(f"Generated performance metric plots saved in: {newFolderName}")
      print(f"Finished processing system: {system} ({idx + 1}/{len(foundSystems)})")

    return dfMetrics, history, names, metrics

  # Use parsed args.
  verbose = args.verbose
  actualColName = args.actualColName
  actualColIDColName = args.actualColIDColName
  predictionColName = args.predictionColName
  subsets = args.subsets
  predCSVFileFix = args.predCSVFileFix
  baseExpDir = args.baseExpDir
  probabilityColName = args.probabilityColName
  dpi = args.dpi
  explainMethods = args.explainMethods
  maxExplainImages = args.maxExplainImages
  explainDatasetDir = args.explainDatasetDir
  device = args.device
  maxPerturbImages = args.maxPerturbImages

  foundSystems = [
    el for el in os.listdir(baseExpDir)
    if (os.path.isdir(os.path.join(baseExpDir, el)))
  ]
  # Sort systems alphabetically for consistent processing order.
  foundSystems = [
    el for el in sorted(foundSystems)
    if (os.path.isdir(os.path.join(baseExpDir, el)) and el != "All_Systems_PerformanceMetricsPlots")
  ]

  if (verbose):
    print(f"Found systems in base experiment directory '{baseExpDir}': {foundSystems}")
  if (len(foundSystems) == 0):
    print(f"No systems found in base experiment directory: {baseExpDir}. Please check the path and try again.")
    exit(1)  # Exit with an error code if no systems are found.

  # Example of the file structure (if you have multiple systems):
  #     System A, , , , , , System B, , , , ,
  #     Precision, Recall, F1, Accuracy, Specificity, Average, Precision, Recall, F1, Accuracy, Specificity, Average
  #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133, 0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133
  #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282, 0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282
  #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406, 0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406
  #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339, 0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339
  #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813, 0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813

  metricsRowNames = []
  systemsRowNames = []
  dataOnly = None

  for idx, system in enumerate(foundSystems):
    print(f"System {idx + 1}/{len(foundSystems)}: {system}")
    baseOutputDir = os.path.join(baseExpDir, system)
    dfMetrics, history, names, metrics = ProcessSystem(
      subsets, predCSVFileFix, baseOutputDir,
      verbose, actualColName, actualColIDColName,
      predictionColName, probabilityColName, dpi,
      explainMethods=explainMethods,
      maxExplainImages=maxExplainImages,
      explainDatasetDir=explainDatasetDir,
      device=device,
      maxPerturbImages=maxPerturbImages,
    )
    # Drop the first row.
    dfMetrics = dfMetrics.drop(index=dfMetrics.index[0])

    if (dataOnly is None):
      # Initialize with the first system's metrics.
      dataOnly = dfMetrics.values
      metricsRowNames.extend(names)
      # Add system name followed by empty strings for alignment.
      systemsRowNames.extend([system] + [""] * (len(names) - 1))
    else:
      noOfRows = dfMetrics.shape[0]
      if (noOfRows != dataOnly.shape[0]):
        if (verbose):
          print(
            f"Warning: Number of rows in metrics for system '{system}' ({noOfRows}) does not match "
            f"the number of rows in previous systems ({dataOnly.shape[0]}). "
            f"This may indicate inconsistent metric extraction."
          )
        if (noOfRows < dataOnly.shape[0]):
          if (verbose):
            print(
              f"System '{system}' has fewer metric rows ({noOfRows}) than previous systems ({dataOnly.shape[0]}). "
              f"Some metrics may be missing for this system."
            )
          continue
        else:
          if (verbose):
            print(
              f"System '{system}' has more metric rows ({noOfRows}) than previous systems ({dataOnly.shape[0]}). "
              f"Some metrics may be extra for this system."
            )
            # Trim the extra rows to match the previous systems for consistency in comparison.
          dfMetrics = dfMetrics.head(dataOnly.shape[0])

      # Stack metrics side-by-side for subsequent systems.
      temp = []
      for i in range(dataOnly.shape[0]):
        temp.append(list(dataOnly[i]) + list(dfMetrics.values[i]))
      dataOnly = np.array(temp)
      metricsRowNames.extend(names)
      # Add system name followed by empty strings for alignment.
      systemsRowNames.extend([system] + [""] * (len(names) - 1))

  if (verbose):
    print(f"Metrics row names (metric names): {metricsRowNames}")
    print(f"Systems row names (system names): {systemsRowNames}")
    print(f"Data only (metric values for all systems):\n{dataOnly}")

  # Create a DataFrame to store the metrics for all systems, with the first row containing metric names and
  # the second row containing system names.
  dfAllMetrics = pd.DataFrame(
    data=dataOnly,
    columns=systemsRowNames,
  )
  # Insert the metricsRowNames as the first row in the DataFrame at index 0 (after the header); pushes the
  # metric values down by one row.
  dfAllMetrics.loc[-1] = metricsRowNames  # Add the first row with system names.
  dfAllMetrics.index = dfAllMetrics.index + 1  # Shift the index to accommodate the new row.

  # Sort the index to maintain the correct order (header, system names row, then metric values).
  dfAllMetrics.sort_index(inplace=True)
  # Save the DataFrame to a CSV file for comparison.
  allSystemsMetricsComparisonFile = os.path.join(baseExpDir, "All_Systems_Metrics_Comparison.csv")
  dfAllMetrics.to_csv(allSystemsMetricsComparisonFile, index=False)
  if (verbose):
    print(f"All systems metrics comparison saved to: {os.path.join(baseExpDir, 'All_Systems_Metrics_Comparison.csv')}")
    print(f"All Systems Metrics Comparison:\n{dfAllMetrics}")

  # Generate performance metric plots for all systems combined.
  newFolderName = os.path.join(baseExpDir, "All_Systems_PerformanceMetricsPlots")
  os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.
  history, names, metrics = ExtractDataFromSummaryFile(allSystemsMetricsComparisonFile)
  PlotMetrics(
    history, names, metrics,
    factor=5,  # Factor to multiply the default figure size.
    keyword="AllSystems_AllMetrics",  # Keyword to append to the filenames of the saved plots.
    dpi=dpi,  # Dots per inch (resolution) of the saved plots.
    xTicksRotation=45,  # Rotation angle for x-axis tick labels.
    whichToPlot=[],  # List of plot types to generate.
    fontSize=14,  # Font size for the plots.
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

  print("\u2713 Performance plots generated.")
  print("\nGenerating statistical analysis report...")
  overallReport = []
  for metric in metrics:
    for index, data in enumerate(history):
      report = StatisticalAnalysis(
        data[metric]["Trials"],
        hypothesizedMean=data[metric]["Mean"],
        secondMetricList=None,
      )
      report["Type"] = names[index]
      report["Metric"] = metric
      overallReport.append(report)
  reportDF = pd.DataFrame(overallReport)
  reportCsvPath = os.path.join(baseExpDir, "All_Systems_Statistical_Analysis_Report.csv")
  reportDF.to_csv(reportCsvPath, index=False)
  print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")

  if (verbose):
    print(f"Generated combined performance metric plots for all systems saved in: {newFolderName}")
    print("Finished processing all systems.")


if (__name__ == "__main__"):
  # Structure of the experiments folder should be (if Train and Test subsets are used and you have multiple systems and trials):
  #     Experiments/
  #         System A/
  #             Trial 1/
  #                 Train_[predCSVFileFix].csv (e.g. Train_Predictions.csv)
  #                 Test_[predCSVFileFix].csv (e.g. Test_Predictions.csv)
  #                 ...
  #             Trial 2/
  #                 Train_[predCSVFileFix].csv (e.g. Train_Predictions.csv)
  #                 Test_[predCSVFileFix].csv (e.g. Test_Predictions.csv)
  #                 ...
  #             ...
  #         System B/
  #             Trial 1/
  #                 Train_[predCSVFileFix].csv (e.g. Train_Predictions.csv)
  #                 Test_[predCSVFileFix].csv (e.g. Test_Predictions.csv)
  #             Trial 2/
  #                 Train_[predCSVFileFix].csv (e.g. Train_Predictions.csv)
  #                 Test_[predCSVFileFix].csv (e.g. Test_Predictions.csv)
  #             ...
  #        ...

  Main()
