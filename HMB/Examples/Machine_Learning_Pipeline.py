from HMB.Initializations import CheckInstalledModules

if __name__ == "__main__":
  CheckInstalledModules(["pandas", "numpy", "matplotlib", "shap", "sklearn"])

# ------------------------------------------------------------------------- #

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding, UpdateMatplotlibSettings
from HMB.MachineLearningHelper import (
  OptunaTuningClassification,
  OptunaTuningClassificationTesting,
  OptunaTuningClassificationTrials,
  OptunaTuningClassificationTrialsStatistics
)
from HMB.ExplainabilityHelper import SHAPExplainer

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
  IgnoreWarnings()
  DoRandomSeeding()
  UpdateMatplotlibSettings()

  # Set to "TRAINING", "TESTING", "TRIALS", "STATISTICS", "REPORTING", "EXPLAINABILITY", or "ALL"
  # based on the desired phase of execution.
  # Set to "ALL" to run all phases sequentially.
  CURRENT_PHASE = "TRAINING"

  # Set the DPI for saved figures; higher values yield better quality but larger file sizes.
  DPI = 300

  # Set the number of top experiments to consider for testing and trial runs; this will create subfolders
  # for each of the top experiments.
  NUM_OF_TOP_EXPERIMENTS = 5

  # Set the number of trials to run for the trial evaluation phase; this will run the best parameters
  # obtained from training across multiple trials to evaluate their performance and stability.
  NUM_OF_TRIALS = 10

  # Restricted metrics for statistics.
  RESTRICTED_METRICS_FOR_STATISTICS = [
    "Precision", "Recall", "F1", "Accuracy", "Specificity", "BAC",
    # "MCC", "Youden", "Yule", "Average", "Average 9"
  ]

  # The target column name in the CSV files; change if your target column has a different name.
  TARGET_COLUMN = "Class"

  # You can file mapping of system names to more descriptive names if needed,
  # e.g., {"System A": "Random Forest", "System B": "XGBoost"}.
  MAPPING = {
    # Change the keys to match the actual system names in your CSV files and the values to the desired display names.
    "Features_Fundus_Mar_25_V3_Mod"  : "All",
    "Features_GLCM_Fundus_Mar_25_V3" : "GLCM",
    "Features_LBP_Fundus_Mar_25_V3"  : "LBP",
    "Features_GLRLM_Fundus_Mar_25_V3": "GLRLM",
    "Features_Hu_Fundus_Mar_25_V3"   : "Hu Moments",
    "Features_LTE_Fundus_Mar_25_V3"  : "LTE",
    "Features_Shape_Fundus_Mar_25_V3": "Shape",
    "Features_SFM_Fundus_Mar_25_V3"  : "SFM",
    "Features_TAS_Fundus_Mar_25_V3"  : "TAS",
  }

  # BASE_DIR = r"path\to\your\directory\which\contains\csv\files"
  BASE_DIR = r"C:\Users\Hossam\Downloads\Features v3"
  csvFiles = [
    el for el in os.listdir(BASE_DIR)
    if (el.endswith(".csv") and "All_Systems" not in el)
  ]
  # Sort the list of CSV files alphabetically.
  csvFiles = sorted(csvFiles)
  print("CSV Files:", csvFiles)

  if (CURRENT_PHASE in ["TRAINING", "ALL"]):
    print("Starting the training and tuning process...")
    for file in csvFiles:
      print(f"Processing file: {file}")
      baseName = os.path.splitext(file)[0]
      tuner = OptunaTuningClassification(
        baseDir=BASE_DIR,
        scalers=["Standard", "MinMax", "MaxAbs", "Robust", "L1", "L2", "Normalizer", "QT", None],
        models=["RF", "LR", "XGB", "SGD", "SVC", "KNN", "AB", "LGBM"],
        fsTechs=["PCA", "RF", "LDA", "RFE", "Chi2", "ANOVA", "MI", None],
        fsRatios=[10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100],
        dataBalanceTechniques=[None],
        outliersTechniques=[
          "IQR", "ZScore", "IForest", "LOF", "EllipticEnvelope",
          "OCSVM", "DBSCAN", "Mahalanobis", None
        ],
        datasetFilename=file,
        storageFolderPath=os.path.join(BASE_DIR, f"Optuna_Results_{baseName}"),
        testFilename=None,
        testRatio=0.2,
        contamination=0.05,
        numTrials=250,
        prefix="Optuna",
        samplerTech="TPE",
        targetColumn=TARGET_COLUMN,
        dropFirstColumn=True,
        dropNAColumns=True,
        encodeCategorical=True,
        # To avoid saving a large number of figures for all trials; set to True if you want to save them.
        saveFigures=False,
        eps=1e-8,
        loadStudy=False,
        verbose=True,
      )
      tuner.Tune()
      print(tuner.GetBestParams())
      print(tuner.GetBestValue())

  if (CURRENT_PHASE in ["TESTING", "ALL"]):
    print(
      "Starting the testing and evaluation process using the best parameters "
      "obtained from a previous training run..."
    )
    for file in csvFiles:
      print(f"Processing file: {file}")
      # Load the best parameters from a previous training run for the current file.
      # Load from the "Optuna Best Params.csv" file inside the Optuna_Results_{baseName} folder.
      baseName = os.path.splitext(file)[0]
      datasetFilePath = os.path.join(BASE_DIR, file)
      experimentPath = os.path.join(BASE_DIR, f"Optuna_Results_{baseName}")
      storageDir = os.path.join(experimentPath, "Testing Results")
      os.makedirs(storageDir, exist_ok=True)
      OptunaTuningClassificationTesting(
        datasetFilePath,
        experimentPath,
        storageDir,
        dpi=DPI,
        T=500,
        numberOfTopExperiments=NUM_OF_TOP_EXPERIMENTS,
        plotCounterfactualOutcomes=False,
        sortByMetric="Weighted Average",
      )

  if (CURRENT_PHASE in ["TRIALS", "ALL"]):
    print(
      "Starting the trial runs to evaluate the performance of the best parameters "
      "obtained from a previous training run across multiple trials..."
    )
    for file in csvFiles:
      print(f"Processing file: {file}")
      baseName = os.path.splitext(file)[0]
      datasetFilePath = os.path.join(BASE_DIR, file)
      experimentPath = os.path.join(BASE_DIR, f"Optuna_Results_{baseName}")
      storageDir = os.path.join(experimentPath, "Trial Results")
      os.makedirs(storageDir, exist_ok=True)

      OptunaTuningClassificationTrials(
        datasetFilePath,
        experimentPath,
        storageDir,
        noOfTrials=NUM_OF_TRIALS,
        dpi=DPI,
        T=50,
        numberOfTopExperiments=NUM_OF_TOP_EXPERIMENTS,
        sortByMetric="Weighted Average",
      )

  if (CURRENT_PHASE in ["STATISTICS", "ALL"]):
    from sklearn.metrics import confusion_matrix
    from HMB.StatisticalAnalysisHelper import ExtractDataFromSummaryFile, PlotMetrics, StatisticalAnalysis

    print(
      "Starting the statistical analysis of the trial results obtained from multiple runs "
      "using the best parameters from a previous training run..."
    )

    allHistory = {}
    for i, file in enumerate(csvFiles):
      print(f"Processing file: {file}")
      baseName = os.path.splitext(file)[0]
      experimentPath = os.path.join(BASE_DIR, f"Optuna_Results_{baseName}")
      trialResultsPath = os.path.join(experimentPath, "Trial Results")
      statisticsStoragePath = os.path.join(experimentPath, "Statistics")
      os.makedirs(statisticsStoragePath, exist_ok=True)

      fileHistory = OptunaTuningClassificationTrialsStatistics(
        trialResultsPath,
        statisticsStoragePath,
        dpi=DPI,
        plotMetricsIndividual=False,
        plotMetricsOverall=False,
        includeAverageInPlots=False,
      )
      key = list(fileHistory.keys())[0]
      dfMetrics = fileHistory[key]["dfMetrics"]
      # Applying the restriction on metrics to be included in the statistical analysis if specified.
      if (RESTRICTED_METRICS_FOR_STATISTICS):
        # Filter the DataFrame to keep only the specified metrics.
        filteredMetrics = [
          metric
          for metric in dfMetrics.columns
          if (metric in RESTRICTED_METRICS_FOR_STATISTICS)
        ]
        allHistory[MAPPING[baseName]] = dfMetrics[filteredMetrics]
        print(
          f"\u2713 Applied metric restriction. Metrics included in the analysis "
          f"for {MAPPING[baseName]}: {filteredMetrics}"
        )
      print(f"\u2713 Extracted trial history statistics for file: {file} and stored under key: {MAPPING[baseName]}")
      print(f"Current keys in allHistory: {list(allHistory.keys())}")
      print(f"Current record keys for {MAPPING[baseName]}: {list(allHistory[MAPPING[baseName]].keys())}")
      print(f"Sample of the metrics DataFrame for {MAPPING[baseName]}:")
      print(dfMetrics.head())

      # Clear figures to free up memory after processing each file.
      plt.close("all")

    # Example of the file structure (if you have multiple systems):
    #     System A, , , , , , System B, , , , ,
    #     Precision, Recall, F1, Accuracy, Specificity, Average, Precision, Recall, F1, Accuracy, Specificity, Average
    #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133, 0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133
    #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282, 0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282
    #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406, 0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406
    #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339, 0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339
    #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813, 0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813

    systemsRow = []
    metricsRow = []
    dataRows = []
    systems = list(allHistory.keys())
    metricsNames = allHistory[systems[0]].columns.tolist()
    print("Systems:", systems)
    print("Metrics Names:", metricsNames)

    for system in systems:
      if (system in MAPPING):
        system = MAPPING[system]
      systemsRow.extend([system] + [""] * (len(metricsNames) - 1))
      metricsRow.extend(metricsNames)

    noOfRecords = len(allHistory[systems[0]])
    for i in range(1, noOfRecords):
      row = []
      for system in systems:
        dfMetrics = allHistory[system]
        row.extend(dfMetrics.iloc[i].tolist())
      dataRows.append(row)

    finalDf = pd.DataFrame(dataRows, columns=systemsRow)
    # Insert the `metricsRowNames` as the first row in the DataFrame at index 0 (after the header); pushes the
    # metric values down by one row.
    finalDf.loc[-1] = metricsRow  # Add the first row with system names.
    finalDf.index = finalDf.index + 1  # Shift the index to accommodate the new row.

    # Sort the index to maintain the correct order (header, system names row, then metric values).
    finalDf.sort_index(inplace=True)
    print(finalDf.head())

    # Generate performance metric plots for all systems combined.
    newFolderName = os.path.join(BASE_DIR, "All Systems Statistical Analysis")
    os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.
    # Save the DataFrame to a CSV file for comparison.
    top1EachSystem = os.path.join(newFolderName, "Top-1 Each System (for Statistics).csv")
    finalDf.to_csv(top1EachSystem, index=False)
    # Save a LaTeX version of the DataFrame for better presentation in reports or papers.
    top1EachSystemLatex = os.path.join(newFolderName, "Top-1 Each System (for Statistics).tex")
    with open(top1EachSystemLatex, "w") as f:
      f.write(finalDf.to_latex(index=False))

    hist, names, metrics = ExtractDataFromSummaryFile(top1EachSystem)
    PlotMetrics(
      hist, names, metrics,
      factor=5,  # Factor to multiply the default figure size.
      keyword="Summary",  # Keyword to append to the filenames of the saved plots.
      dpi=DPI,  # Dots per inch (resolution) of the saved plots.
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
      for index, data in enumerate(hist):
        report = StatisticalAnalysis(
          data[metric]["Trials"],
          hypothesizedMean=data[metric]["Mean"],
          secondMetricList=None,
        )
        report["Type"] = names[index]
        report["Metric"] = metric
        overallReport.append(report)
    reportDF = pd.DataFrame(overallReport)
    reportCsvPath = os.path.join(newFolderName, "Statistical Analysis Report.csv")
    reportDF.to_csv(reportCsvPath, index=False)
    print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")

  if (CURRENT_PHASE in ["REPORTING", "ALL"]):
    print(
      "Starting the reporting phase to generate comprehensive reports "
      "based on the best parameters and trial results..."
    )

    reportingFolder = os.path.join(BASE_DIR, "Reporting")
    os.makedirs(reportingFolder, exist_ok=True)

    # ======================================================================================= #
    # Reporting the final table for all systems and metrics in a well-formatted manner.
    # This will use the output in the "Testing Results" folder.
    # Initialize container to collect per-system final history DataFrames.
    merged = {}
    # Iterate over each CSV file detected earlier to collect reporting artifacts.
    for file in csvFiles:
      # Print progress for the current file being processed.
      print(f"Processing file: {file}")
      # Derive the base filename without extension to locate experiment folders.
      baseName = os.path.splitext(file)[0]
      # Construct path to the experiment results folder for this dataset.
      experimentPath = os.path.join(BASE_DIR, f"Optuna_Results_{baseName}")
      # Path to the testing results subfolder where FinalHistory CSV is expected.
      testingResultsPath = os.path.join(experimentPath, "Testing Results")

      # Store the cleaned final history DataFrame in a CSV file for reference.
      cleanedFinalHistoryFilePath = os.path.join(
        testingResultsPath,
        f"Cleaned_Summary_of_Top_{NUM_OF_TOP_EXPERIMENTS}_Experiments.csv"
      )
      if (not os.path.exists(cleanedFinalHistoryFilePath)):
        raise ValueError(f"\u2717 Cleaned summary CSV not found at expected path: {cleanedFinalHistoryFilePath}")
      # Load the cleaned summary CSV which contains the final history for this system.
      finalHistoryDf = pd.read_csv(cleanedFinalHistoryFilePath)
      # Store the DataFrame in the merged mapping using the system name from MAPPING.
      merged[MAPPING[baseName]] = finalHistoryDf
      print(f"\u2713 Loaded cleaned summary for {MAPPING[baseName]} from: {cleanedFinalHistoryFilePath}")

    # After collecting all per-system DataFrames, concatenate them into a single DataFrame.
    # After processing all files, we can merge the final history DataFrames for all systems into a single DataFrame for comparison.
    mergedDf = pd.DataFrame()
    # Iterate over the merged mapping and stack DataFrames together with a System column.
    for system, df in merged.items():
      # Make a defensive copy of the DataFrame to avoid side effects on the original.
      df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original.
      # Add a column that identifies which system the row belongs to.
      df["System"] = system  # Add a column for the system name.
      # Concatenate the current system's DataFrame into the accumulating merged DataFrame.
      mergedDf = pd.concat([mergedDf, df], ignore_index=True)  # Concatenate the DataFrames for all systems.
    # Reorder columns to place the System identifier up front for readability.
    # Reorder the columns to have "System" as the first column.
    cols = mergedDf.columns.tolist()
    cols = ["System"] + [col for col in cols if (col != "System")]
    mergedDf = mergedDf[cols]
    # Persist the merged summary to CSV for external use.
    # Save the merged DataFrame to a CSV file for comparison.
    mergedReportFile = os.path.join(reportingFolder, "All_Systems_Merged_Testing_Results_Summary.csv")
    mergedDf.to_csv(mergedReportFile, index=False)
    # Print where the merged CSV was written.
    print(f"\u2713 Merged testing results summary saved to: {mergedReportFile}")
    # Also write a LaTeX version suitable for papers/reports.
    # Save as Latex table for better presentation in reports or papers.
    mergedLatexFile = os.path.join(reportingFolder, "All_Systems_Merged_Testing_Results_Summary.tex")
    # Number should be formatted to 4 decimal places in the LaTeX table for better readability.
    with open(mergedLatexFile, "w") as f:
      # Write the merged DataFrame as a LaTeX table with fixed float formatting.
      f.write(mergedDf.to_latex(index=False, float_format="%.2f"))
    print(f"\u2713 Merged testing results summary saved as LaTeX table to: {mergedLatexFile}")
    # print(f"\u2713 Merged Testing Results Summary:\n{mergedDf}")
    # ======================================================================================= #

    # ======================================================================================= #
    finalSummary = {}
    # Report the trials history for all systems in a well-formatted manner.
    # This will use the output in the "Trial Results" folder.
    # Iterate again over input CSV files to assemble trial-history reports for each system.
    for file in csvFiles:
      # Print the filename being processed for trial-history reporting.
      print(f"Processing file: {file}")
      # Extract base name again for folder path construction.
      baseName = os.path.splitext(file)[0]
      # Build the experiment path for this dataset.
      experimentPath = os.path.join(BASE_DIR, f"Optuna_Results_{baseName}")
      # Path to the trial results' subfolder.
      refinedSummaryPath = os.path.join(experimentPath, "Trial Results", "Refined_Summary.csv")
      if (not os.path.exists(refinedSummaryPath)):
        raise ValueError(f"\u2717 Refined summary CSV not found at expected path: {refinedSummaryPath}")
      # Load the refined summary CSV which contains the trial history for this system.
      trialHistoryDf = pd.read_csv(refinedSummaryPath)
      finalSummary[MAPPING[baseName]] = trialHistoryDf
      print(f"\u2713 Loaded trial history summary for {MAPPING[baseName]} from: {refinedSummaryPath}")

    finalSummaryDf = pd.DataFrame()
    for system, df in finalSummary.items():
      df = df.copy()  # Create a copy to avoid modifying the original.
      df["System"] = system  # Add a column for the system name.
      finalSummaryDf = pd.concat(
        [finalSummaryDf, df],
        ignore_index=True
      )  # Concatenate into the final summary DataFrame.
    # Reorder columns to have "System" first for readability.
    cols = finalSummaryDf.columns.tolist()
    cols = ["System"] + [col for col in cols if (col != "System")]
    finalSummaryDf = finalSummaryDf[cols]
    # Save the final summary DataFrame to CSV for comparison.
    finalSummaryReportFile = os.path.join(reportingFolder, "All_Systems_Merged_Trials_History_Summary.csv")
    finalSummaryDf.to_csv(finalSummaryReportFile, index=False)
    print(f"\u2713 All systems merged trials history summary saved to: {finalSummaryReportFile}")
    # Also write a LaTeX table version of the final summary.
    finalSummaryLatexFile = os.path.join(reportingFolder, "All_Systems_Merged_Trials_History_Summary.tex")
    with open(finalSummaryLatexFile, "w") as f:
      f.write(finalSummaryDf.to_latex(index=False, float_format="%.2f"))
    print(f"\u2713 All systems merged trials history summary saved as LaTeX table to: {finalSummaryLatexFile}")
    # Get the mean and standard deviation of the performance metrics for each system/model and save to a summary CSV.
    # Group by "System" and "Model" to get mean and std for each combination of system and model.
    finalSummaryStatisticsDf = finalSummaryDf.groupby(["System", "Model"]).agg(["mean", "std"])
    finalSummaryStatisticsReportFile = os.path.join(
      reportingFolder,
      "All_Systems_Merged_Trials_History_Summary_Statistics.csv"
    )
    finalSummaryStatisticsDf.to_csv(finalSummaryStatisticsReportFile)
    print(f"\u2713 All systems merged trials history summary statistics saved to: {finalSummaryStatisticsReportFile}")
    # Also write a LaTeX table version of the final summary statistics.
    finalSummaryStatisticsLatexFile = os.path.join(
      reportingFolder,
      "All_Systems_Merged_Trials_History_Summary_Statistics.tex"
    )
    with open(finalSummaryStatisticsLatexFile, "w") as f:
      f.write(finalSummaryStatisticsDf.to_latex(float_format="%.2f"))
    print(
      f"\u2713 All systems merged trials history summary statistics saved as LaTeX table to: "
      f"{finalSummaryStatisticsLatexFile}"
    )
    # Select the top-1 in each system/model combination based on the mean of the main performance metric (e.g., "Accuracy") and save to a CSV.
    # Assuming "Accuracy" is the main performance metric we want to use for selecting the top-1 model in each system/model combination.
    # Get the index of the row with the max mean accuracy for each system.
    top1DIds = finalSummaryStatisticsDf["Accuracy"]["mean"].groupby(level=0).idxmax()
    top1Df = finalSummaryStatisticsDf.loc[top1DIds].reset_index()
    top1ReportFile = os.path.join(reportingFolder, "All_Systems_Merged_Trials_History_Top1_Summary.csv")
    top1Df.to_csv(top1ReportFile, index=False)
    print(f"\u2713 All systems merged trials history top-1 summary saved to: {top1ReportFile}")
    # Also write a LaTeX table version of the top-1 summary.
    top1LatexFile = os.path.join(reportingFolder, "All_Systems_Merged_Trials_History_Top1_Summary.tex")
    with open(top1LatexFile, "w") as f:
      f.write(top1Df.to_latex(float_format="%.2f"))
    print(f"\u2713 All systems merged trials history top-1 summary saved as LaTeX table to: {top1LatexFile}")

    # ======================================================================================= #

    # Final message to indicate the entire reporting phase is done.
    print("\u2713 Reporting phase completed.")

  if (CURRENT_PHASE in ["EXPLAINABILITY", "ALL"]):
    print(
      "Starting the explainability phase to generate insights and explanations "
      "for the best models and parameters identified in previous phases..."
    )

    baseDir = BASE_DIR
    for file in csvFiles:
      print(f"Processing file: {file}")
      # Extract base name again for folder path construction.
      baseName = os.path.splitext(file)[0]
      # Build the experiment path for this dataset.
      experimentPath = os.path.join(BASE_DIR, f"Optuna_Results_{baseName}")
      csvName = os.path.join(experimentPath, "Optuna Best Params.csv")

      explainer = SHAPExplainer(
        baseDir=baseDir,
        experimentFolderName=experimentPath,
        testFilename=file,
        targetColumn=TARGET_COLUMN,
        pickleFilePath=None,
        shapStorageKeyword="SHAP Results",
        dpi=DPI,
        csvName=csvName,
      )
      explainer.LoadModelAndData(maxNoRecords=500)
      explainer.ComputeShapValues()
      explainer.MakePredictions()
      explainer.VisualizeExplanations(
        instanceIndex=0,
        categoryToExplain="all",
        noOfRecords=500,
        noOfFeatures=10,
      )

    print("\u2713 Explainability phase completed.")
