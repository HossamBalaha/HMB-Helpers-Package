from HMB.Initializations import CheckInstalledModules

if __name__ == "__main__":
  CheckInstalledModules(["pandas", "numpy", "matplotlib", "tensorflow", "sklearn"])

# ------------------------------------------------------------------------- #

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from HMB.TFHelper import (
  TrainPretrainedAttentionModelFromDataFrame,
  EvaluatePretrainedAttentionModelFromDataFrame,
  StatisticsPretrainedAttentionModelFromDataFrame
)
from HMB.DatasetsHelper import RawImageFolder

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


if (__name__ == "__main__"):
  # Set to "TRAINING", "TESTING", "STATISTICS", "REPORTING", "EXPLAINABILITY", or "ALL"
  # based on the desired phase of execution.
  # Set to "ALL" to run all phases sequentially.
  CURRENT_PHASE = "STATISTICS"

  # Training parameters and image shapes.
  initialEpochs = 25  # Number of initial epochs.
  fineTuneEpochs = 25  # Number of fine-tune epochs.
  batchSize = 32  # Set batch size.
  channels = 3  # Number of channels.
  imgSize = (512, 512)  # Define target image size.
  imgShape = (imgSize[0], imgSize[1], channels)  # Define input shape.
  columnsMap = {"imagePath": "image_path", "categoryEncoded": "category_encoded", "split": "split"}
  baseModelString = "ResNet50V2"  # "Xception"
  attentionBlockStr = "ECA"  # "CBAM"
  noOfTrials = 10  # Number of trials to run for training and evaluation.
  dpi = 300  # Set the DPI for saving evaluation results.
  ensureCUDA = True  # Set to True to ensure CUDA is available for training.
  categories = ["grade0", "grade1", "grade2", "grade3", "grade4"]  # Define the class categories as a list.

  # Define the dataset base paths using camelCase naming for clarity.
  baseDir = r"/path/to/the/project/Dataset"
  basePaths = {
    "train": rf"{baseDir}/Training",
    "test" : rf"{baseDir}/Test",
    "val"  : rf"{baseDir}/Validation"
  }

  # Base directory for storing experiment results.
  baseStorageDir = r"/path/to/the/project/Experiments"
  # Ensure the base storage directory exists.
  if (not os.path.exists(baseStorageDir)):
    os.makedirs(baseStorageDir)

  # Create a RawImageFolder object to read images and labels from the specified base paths and categories.
  obj = RawImageFolder(basePaths, rootType="dict", categories=categories)
  dataFrame = obj.ToDataFrame()

  expFolderName = f"{baseModelString}_{attentionBlockStr}"
  expFolderPath = os.path.join(baseStorageDir, expFolderName)
  os.makedirs(expFolderPath, exist_ok=True)

  if (CURRENT_PHASE in ["TRAINING", "TESTING", "ALL"]):
    for trial in range(1, noOfTrials + 1):
      print(f"Starting trial {trial} with model {baseModelString} and attention block {attentionBlockStr}...")
      expDir = os.path.join(expFolderPath, f"Trial_{trial}")
      os.makedirs(expDir, exist_ok=True)
      modelPath = os.path.join(expDir, "BestModel.keras")

      if (CURRENT_PHASE in ["TRAINING", "ALL"]):
        TrainPretrainedAttentionModelFromDataFrame(
          dataFrame,
          columnsMap=columnsMap,
          labelEncoder=obj.GetLabelEncoder(),
          imgShape=imgShape,
          batchSize=batchSize,
          baseModelString=baseModelString,
          attentionBlockStr=attentionBlockStr,
          initialEpochs=initialEpochs,
          fineTuneEpochs=fineTuneEpochs,
          augmentationConfigs=None,
          monitor="val_loss",
          earlyStoppingPatience=10,
          ensureCUDA=ensureCUDA,
          storageDir=expDir,
          dpi=dpi,
          verbose=2,
        )

      if (CURRENT_PHASE in ["TESTING", "ALL"]):
        EvaluatePretrainedAttentionModelFromDataFrame(
          dataFrame,
          modelPath,
          columnsMap=columnsMap,
          labelEncoder=obj.GetLabelEncoder(),
          imgShape=imgShape,
          batchSize=batchSize,
          storageDir=expDir,
          dpi=dpi,
          verbose=2,
          ensureCUDA=ensureCUDA,
        )

  if (CURRENT_PHASE in ["STATISTICS", "ALL"]):
    statisticsStoragePath = os.path.join(baseStorageDir, "Statistics")  # Path to store statistics results.
    os.makedirs(statisticsStoragePath, exist_ok=True)  # Ensure the statistics storage directory exists.

    StatisticsPretrainedAttentionModelFromDataFrame(
      baseStorageDir,
      statisticsStoragePath,
      dpi=dpi,
      plotMetricsIndividual=True,
      plotMetricsOverall=True,
      includeAverageInPlots=False,
    )
