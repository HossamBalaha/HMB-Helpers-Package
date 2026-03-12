#!/bin/bash
#SBATCH --job-name=hmb         # Job name.
#SBATCH --partition=gpu             # Specify the GPU partition (e.g., "gpu", "compute").
#SBATCH --cpus-per-task=48           # Request CPU cores.
#SBATCH --mem=250G                     # Request memory (adjust as needed).
#SBATCH --nodes=1                   # Request one or more nodes.
#SBATCH --gpus=1                   # Request one or more GPUs (adjust as needed).
#SBATCH --time=48:00:00             # Wall time limit (HH:MM:SS).
#SBATCH --output="/Jobs/PyTorch_UNet_Segmentation_Output_%j.log"
#SBATCH --error="/Jobs/PyTorch_UNet_Segmentation_Error_%j.log"

# Path to the training/inference script (edit if needed)
SCRIPT="/path/to/PyTorch_UNet_Segmentation.py"

# General settings (defaults follow the script's defaults)
ModelName="ResidualAttentionUNet"
DataDir="./Data"
OutputDir="./OutputUNet"
Phase="Train"            # Train or Infer
NumEpochs=50
BatchSizes=(16)
ImageSize=128
LearningRate=1e-4
WeightDecay=1e-6
Device="cuda"
NumWorkers=4
NumClasses=1
DPI=720
# Resume checkpoint path (leave empty to disable)
ResumeCheckpoint=""
# Use automatic mixed precision (set to 1 to enable)
USE_AMP=0
# Trials - useful for repeating experiments with different seeds
TRIALS=(1)

# Loop over trials and batch sizes
for TRIAL in "${TRIALS[@]}"; do
  for BATCH_SIZE in "${BatchSizes[@]}"; do
    OUTPUT_RUN_DIR="${OutputDir}/${ModelName}_${BATCH_SIZE}_T${TRIAL}"
    echo "Starting run: Trial ${TRIAL}, Batch Size ${BATCH_SIZE}"

    CMD=(python "$SCRIPT"
      --ModelName "$ModelName"
      --DataDir "$DataDir"
      --OutputDir "$OUTPUT_RUN_DIR"
      --Phase "$Phase"
      --NumEpochs "$NumEpochs"
      --BatchSize "$BATCH_SIZE"
      --ImageSize "$ImageSize"
      --LearningRate "$LearningRate"
      --WeightDecay "$WeightDecay"
      --Device "$Device"
      --NumWorkers "$NumWorkers"
      --NumClasses "$NumClasses"
      --DPI "$DPI"
    )

    # Conditionally append optional args
    if [ -n "$ResumeCheckpoint" ] && [ -f "$ResumeCheckpoint" ]; then
      CMD+=(--ResumeCheckpoint "$ResumeCheckpoint")
    fi

    if [ "$USE_AMP" -eq 1 ]; then
      CMD+=(--UseAMP)
    fi

    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
    if [[ $? -ne 0 ]]; then
      echo "Error: command failed"
      exit 1
    fi
  done
done

echo "All runs completed."
