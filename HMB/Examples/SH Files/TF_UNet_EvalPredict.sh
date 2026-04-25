#!/bin/bash
# TF_UNet_EvalPredict.sh
# Simple launcher for TF_UNet_EvalPredict.py
# Edit variables below to match your environment before running.

# SBATCH headers (optional, remove or edit if not using slurm)
#SBATCH --job-name=tf_unet_eval
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

SCRIPT="/path/to/TF_UNet_EvalPredict.py"

# Configurable defaults (change as needed)
ModelName="UNet"
ModelWeights="/path/to/weights.h5"
DataDir="/path/to/images"
MasksDir=""             # leave empty if no masks available
MaskPostfix=""          # If non-empty, enables mask postfix mapping: name.ext -> name_<ext>.ext
OutputDir="Output"
InputSize=(256 256 3)    # H W C
BatchSize=8
NumClasses=1
SavePredictions=0        # 1 to save predicted masks

mkdir -p "$OutputDir"

# Build command
CMD=(python "$SCRIPT"
  --ModelName "$ModelName"
  --ModelWeights "$ModelWeights"
  --DataDir "$DataDir"
  --OutputDir "$OutputDir"
  --BatchSize "$BatchSize"
  --NumClasses "$NumClasses"
)

# InputSize is three values
CMD+=(--InputSize "${InputSize[0]}" "${InputSize[1]}" "${InputSize[2]}")

if [ -n "$MasksDir" ]; then
  CMD+=(--MasksDir "$MasksDir")
fi

if [ -n "$MaskPostfix" ]; then
  CMD+=(--MaskPostfix "$MaskPostfix")
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

echo "Evaluation completed."
