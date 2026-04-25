#!/bin/bash
#SBATCH --job-name=stats_ablation
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output="/Jobs/Statistics_Analysis_Ablations_Output_%j.log"
#SBATCH --error="/Jobs/Statistics_Analysis_Ablations_Error_%j.log"

# Path to the Python script
SCRIPT="/path/to/Statistics_Analysis_Ablations.py"

# Base experiments directory containing system subfolders
BASE_EXP_DIR="/home/hmbala01/BC-Group/Experiments"
# Prediction CSV file fix substring used to identify per-trial prediction CSVs
PRED_CSV_FIX="_Predictions_"
# Subsets to process (space separated)
SUBSETS=("Train" "Test")
# Column names for actual and predicted labels
ACTUAL_COL_NAME="trueClassName"
ACTUAL_COL_ID_COL_NAME="trueClassIndex"
PREDICTION_COL_NAME="predictedClassName"
# Column name for probabilities.
PROBABILITY_COL_NAME="probabilities"
# Verbose flag (1 to enable)
VERBOSE=1
# DPI for saving figures
DPI=1080
# Explainability methods to analyze (space separated, e.g. "GradCAM")
EXPLAIN_METHODS=("gradcam" "gradcampp" "layercam" "integratedgradients")
# Max number of images for the explainability analysis (set to a positive integer or leave empty for no limit)
MAX_EXPLAIN_IMAGES=10
# Dataset/subset directory path for the explainability analysis (if needed, otherwise can be left empty)
EXPLAIN_DATASET_DIR="/path/to/dataset/subset/for/explainability"
# Device to use for analysis (e.g., "cuda" or "cpu")
DEVICE="cuda"
# Perturbation max number of images (set to a positive integer or leave empty for no limit)
MAX_PERTURB_IMAGES=

# Optionally loop over different base experiment directories or systems
BASE_DIRS=("${BASE_EXP_DIR}")

for BASE in "${BASE_DIRS[@]}"; do
  echo "Running statistics analysis on base experiments dir: ${BASE}"

  CMD=(python "$SCRIPT"
    --baseExpDir "$BASE"
    --predCSVFileFix "$PRED_CSV_FIX"
    --subsets ${SUBSETS[@]}
    --actualColName "$ACTUAL_COL_NAME"
    --predictionColName "$PREDICTION_COL_NAME"
    --probabilityColName "$PROBABILITY_COL_NAME"
    --actualColIDColName "$ACTUAL_COL_ID_COL_NAME"
    --dpi "$DPI"
    --explainMethods ${EXPLAIN_METHODS[@]}
    --maxExplainImages "$MAX_EXPLAIN_IMAGES"
    --explainDatasetDir "$EXPLAIN_DATASET_DIR"
    --device "$DEVICE"
    --maxPerturbImages "$MAX_PERTURB_IMAGES"
  )

  if [ -n "$VERBOSE" ] && [ "$VERBOSE" -eq 1 ]; then
    CMD+=(--verbose)
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
  if [[ $? -ne 0 ]]; then
    echo "Error: Command failed..."
    exit 1
  fi

done
