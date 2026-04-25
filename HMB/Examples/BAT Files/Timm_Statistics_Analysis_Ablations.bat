@echo off
REM Statistics_Analysis_Ablations.bat - Windows equivalent of Statistics_Analysis_Ablations.sh
REM Edit the variables below to suit your environment before running.

setlocal enabledelayedexpansion

:: Path to the Python script (edit as needed)
set "SCRIPT=Statistics_Analysis_Ablations.py"

:: Base experiments directory containing system subfolders (edit or add more below)
set "BASE_EXP_DIR=C:\path\to\Experiments"
:: To specify multiple base dirs, set BASE_DIRS to a space-separated list of quoted paths, e.g.:
:: set "BASE_DIRS=""C:\Path With Spaces\Exp1" "D:\OtherExp"""
set "BASE_DIRS="%BASE_EXP_DIR%""

:: Prediction CSV file fix substring used to identify per-trial prediction CSVs
set "PRED_CSV_FIX=_Predictions_"

:: Subsets to process (space separated). These will be added as multiple --subsets args.
set "SUBSETS=Train Test"

:: Column names for actual and predicted labels
set "ACTUAL_COL_NAME=trueClassName"
set "ACTUAL_COL_ID_COL_NAME=trueClassIndex"
set "PREDICTION_COL_NAME=predictedClassName"
:: Column name for probabilities.
set "PROBABILITY_COL_NAME=probabilities"

:: Verbose flag (1 to enable)
set "VERBOSE=1"

:: DPI for saving figures
set "DPI=1080"

:: Explainability methods to analyze (space separated, e.g. "gradcam")
set "EXPLAIN_METHODS=gradcam gradcampp layercam integratedgradients"
:: Max number of images for the explainability analysis (set to a positive integer or 0 for none)
set "MAX_EXPLAIN_IMAGES=10"
:: Dataset/subset directory path for the explainability analysis (if needed, otherwise can be left empty)
set "EXPLAIN_DATASET_DIR=/path/to/dataset/subset/for/explainability"

:: Perturbation max number of images (set to a positive integer or leave empty for no limit)
set "MAX_PERTURB_IMAGES="

:: Device to use for the analysis (e.g., "cuda" or "cpu")
set "DEVICE=cuda"

:: Iterate over base dirs
for %%B in (%BASE_DIRS%) do (
  set "BASE=%%~B"
  echo Running statistics analysis on base experiments dir: !BASE!

  :: Start building the command. We quote each path/argument to handle spaces properly.
  set "CMD=python "%SCRIPT%" --baseExpDir "!BASE!" --predCSVFileFix "%PRED_CSV_FIX%""

  :: Append subsets as separate arguments (e.g. --subsets Train --subsets Test)
  for %%S in (%SUBSETS%) do (
    set "CMD=!CMD! --subsets "%%S""
  )

  :: Append the rest of the fixed args
  set "CMD=!CMD! --actualColName "%ACTUAL_COL_NAME%" --predictionColName "%PREDICTION_COL_NAME%" --probabilityColName "%PROBABILITY_COL_NAME%" --actualColIDColName "%ACTUAL_COL_ID_COL_NAME%" --dpi "%DPI%""

  :: Append explainability methods (multiple entries allowed)
  for %%E in (%EXPLAIN_METHODS%) do (
    set "CMD=!CMD! --explainMethods "%%E""
  )
  :: Append max explain images
  set "CMD=!CMD! --maxExplainImages "%MAX_EXPLAIN_IMAGES%""

  :: Append explain dataset dir if specified
  if not "%EXPLAIN_DATASET_DIR%"=="" (
    set "CMD=!CMD! --explainDatasetDir "%EXPLAIN_DATASET_DIR%""
  )

  :: Append the device
  set "CMD=!CMD! --device "%DEVICE%""

  :: Append perturbation max images if specified
  set "CMD=!CMD! --maxPerturbImages "%MAX_PERTURB_IMAGES%""

  :: Verbose flag
  if "%VERBOSE%"=="1" (
    set "CMD=!CMD! --verbose"
  )

  echo Running: !CMD!

  :: Execute the constructed command. Use delayed expansion to expand CMD at runtime.
  cmd /V:ON /C "!CMD!"
  if errorlevel 1 (
    echo Error: Command failed with exit code %ERRORLEVEL%
    endlocal
    exit /b 1
  )
)

endlocal

echo All runs completed.
exit /b 0
