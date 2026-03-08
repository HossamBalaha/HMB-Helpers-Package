import os
import numpy as np
import pandas as pd
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding, UpdateMatplotlibSettings
from HMB.MachineLearningHelper import OptunaTuningClassification, OptunaTuningClassificationTesting

if (__name__ == "__main__"):
  IgnoreWarnings()
  DoRandomSeeding()
  UpdateMatplotlibSettings()

  # baseDir = r"path\to\your\directory\which\contains\csv\files"
  baseDir = r"C:\Users\Hossam\Downloads\Features v3"
  csvFiles = [
    el
    for el in os.listdir(baseDir)
    if (el.endswith(".csv"))
  ]
  # Sort the list of CSV files alphabetically.
  csvFiles = sorted(csvFiles)
  print("CSV Files:", csvFiles)

  # TRUE: Perform the training and tuning process.
  # FALSE: Perform the testing and evaluation process using the best parameters obtained from a previous training run.
  IS_TRAINING_PHASE = False

  if (IS_TRAINING_PHASE):
    print("Starting the training and tuning process...")
    for file in csvFiles:
      print(f"Processing file: {file}")
      baseName = os.path.splitext(file)[0]
      tuner = OptunaTuningClassification(
        baseDir=baseDir,
        scalers=["Standard", "MinMax", "MaxAbs", "Robust", "L1", "L2", "Normalizer", "QT", None],
        models=["RF", "LR", "XGB", "SGD", "SVC", "KNN", "AB", "LGBM"],
        fsTechs=["PCA", "RF", "LDA", "RFE", "Chi2", "ANOVA", "MI", None],
        fsRatios=[25, 50, 75, 100] + [10, 20, 30, 40, 60, 70, 80, 90],
        dataBalanceTechniques=[None],
        outliersTechniques=[
          "IQR", "ZScore", "IForest", "LOF", "EllipticEnvelope",
          "OCSVM", "DBSCAN", "Mahalanobis", None
        ],
        datasetFilename=file,
        storageFolderPath=os.path.join(baseDir, f"Optuna_Results_{baseName}"),
        testFilename=None,
        testRatio=0.2,
        contamination=0.05,
        numTrials=250,
        prefix="Optuna",
        samplerTech="TPE",
        targetColumn="Class",
        dropFirstColumn=True,
        dropNAColumns=True,
        encodeCategorical=True,
        saveFigures=True,
        eps=1e-8,
        loadStudy=False,
        verbose=True,
      )
      tuner.Tune()
      print(tuner.GetBestParams())
      print(tuner.GetBestValue())
  else:
    print(
      "Starting the testing and evaluation process using the best parameters "
      "obtained from a previous training run..."
    )
    for file in csvFiles:
      print(f"Processing file: {file}")
      # Load the best parameters from a previous training run for the current file.
      # Load from the "Optuna Best Params.csv" file inside the Optuna_Results_{baseName} folder.
      baseName = os.path.splitext(file)[0]
      datasetFilePath = os.path.join(baseDir, file)
      experimentPath = os.path.join(baseDir, f"Optuna_Results_{baseName}")
      OptunaTuningClassificationTesting(datasetFilePath, experimentPath)
      break
