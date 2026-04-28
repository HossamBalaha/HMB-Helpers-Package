#!/bin/bash
# TF_UNet_EvalPredict.sh.
# Simple launcher for TF_UNet_EvalPredict.py.
# Edit variables below to match your environment before running.

# SBATCH headers (optional, remove or edit if not using slurm).
#SBATCH --job-name=tf_unet_eval
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

SCRIPT="/path/to/TF_UNet_EvalPredict.py"

# Configurable defaults (change as needed).
ModelName="VNet"
ModelWeights="/path/to/weights.h5"
DataDir="/path/to/images"
MasksDir="/path/to/masks"
MaskPostfix=""  # If non-empty, enables mask postfix mapping: name.ext -> name_<postfix>.
OutputDir="Output"
InputSize="256 256 3"  # H W C as a space-separated string.
NumClasses=2
SavePredictions=0  # 1 to save predicted masks.
# Optional path to a hyperparameters JSON for the model.
HyperparamsJson=""
# Trials are discovered under a parent folder when TrialsParent is set.
# The script will iterate subdirectories matching TrialsPattern under TrialsParent.
TrialsParent=""
# Pattern to match trial directories within TrialsParent.
TrialsPattern="Trial*"

# Helper that runs a single evaluation invocation.
# Uses the environment variables ModelWeights, HyperparamsJson and OutputDir.

# Iterate trial subfolders under TrialsParent when provided.
if [ -n "$TrialsParent" ] && [ -d "$TrialsParent" ]; then
  # Discover matching trial directories using find to handle spaces and globbing robustly.
  mapfile -t trials < <(find "$TrialsParent" -maxdepth 1 -type d -name "$TrialsPattern" -print)

  # If no trials were found, exit with an informative message.
  if [ ${#trials[@]} -eq 0 ]; then
    echo "No trial subfolders matching pattern '$TrialsPattern' were found under '$TrialsParent'."
    echo "Ensure trials are named like 'Trial 1', 'Trial 2' or adjust TrialsPattern."
    exit 1
  fi

  # Print discovered trials for visibility.
  echo "Found ${#trials[@]} trials under '$TrialsParent':"
  for t in "${trials[@]}"; do
    echo "  - $t"
  done

  # Iterate the discovered trial directories.
  for d in "${trials[@]}"; do
    trialRoot="$d"

    # Discover a weights file inside the trial folder.
    # Prefer files that contain the token "weights" in their name.
    weightFile=""
    if ls "$trialRoot"/*weights*.h5 1> /dev/null 2>&1; then
      weightFile=$(ls "$trialRoot"/*weights*.h5 | head -n1)
    elif ls "$trialRoot"/*.h5 1> /dev/null 2>&1; then
      weightFile=$(ls "$trialRoot"/*.h5 | head -n1)
    fi

    # Use discovered weight file or fall back to conventional filename.
    if [ -n "$weightFile" ]; then
      ModelWeights="$weightFile"
    else
      ModelWeights="$trialRoot/Trial.weights.h5"
    fi

    # Use Hyperparameters.json if present in the trial folder.
    if [ -f "$trialRoot/Hyperparameters.json" ]; then
      HyperparamsJson="$trialRoot/Hyperparameters.json"
    else
      HyperparamsJson=""
    fi

    # Prepare output directory for this trial.
    OutputDir="$trialRoot/Output"
    mkdir -p "$OutputDir"

      # Build the python command as an array to preserve quoting and spaces, and run it.
      CMD=(python "$SCRIPT"
        --ModelName "$ModelName"
        --ModelWeights "$ModelWeights"
        --DataDir "$DataDir"
        --OutputDir "$OutputDir"
        --NumClasses "$NumClasses"
      )

      # Append InputSize as three values (unquoted expansion splits into three args).
      CMD+=(--InputSize $InputSize)

      # Forward masks directory if provided.
      if [ -n "$MasksDir" ]; then
        CMD+=(--MasksDir "$MasksDir")
      fi

      # Forward mask postfix if provided.
      if [ -n "$MaskPostfix" ]; then
        CMD+=(--MaskPostfix "$MaskPostfix")
      fi

      # Forward hyperparameters JSON if provided.
      if [ -n "$HyperparamsJson" ]; then
        CMD+=(--HyperparamsJson "$HyperparamsJson")
      fi

      # Optionally request saving predictions.
      if [ "$SavePredictions" -eq 1 ]; then
        CMD+=(--SavePredictions)
      fi

      # Execute the assembled command and report errors but continue to next trial.
      echo "Running: ${CMD[*]}"
      "${CMD[@]}"
      RET=$?
      if [ $RET -ne 0 ]; then
        echo "Error: command failed for trial '$trialRoot' with exit code $RET." >&2
        echo "Continuing to next trial." >&2
        # Do not abort the whole run; proceed with remaining trials.
        continue
      fi
  done
else
  # Single-run mode using variables above. Build and execute the command inline.
  CMD=(python "$SCRIPT"
    --ModelName "$ModelName"
    --ModelWeights "$ModelWeights"
    --DataDir "$DataDir"
    --OutputDir "$OutputDir"
    --NumClasses "$NumClasses"
  )

  # Append InputSize as three values.
  CMD+=(--InputSize $InputSize)

  if [ -n "$MasksDir" ]; then
    CMD+=(--MasksDir "$MasksDir")
  fi

  if [ -n "$MaskPostfix" ]; then
    CMD+=(--MaskPostfix "$MaskPostfix")
  fi

  if [ -n "$HyperparamsJson" ]; then
    CMD+=(--HyperparamsJson "$HyperparamsJson")
  fi

  if [ -n "$OutputDir" ]; then
    CMD+=(--OutputDir "$OutputDir")
  fi

  if [ "$SavePredictions" -eq 1 ]; then
    CMD+=(--SavePredictions)
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
  RET=$?
  if [ $RET -ne 0 ]; then
    echo "Error: command failed with exit code $RET" >&2
    exit $RET
  fi
fi

echo "Evaluation completed."
