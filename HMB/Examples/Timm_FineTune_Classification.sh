#!/bin/bash
#SBATCH --job-name=hmb         # Job name.
#SBATCH --partition=gpu             # Specify the GPU partition (e.g., "gpu", "compute").
#SBATCH --cpus-per-task=48           # Request CPU cores.
#SBATCH --mem=250G                     # Request memory (adjust as needed).
#SBATCH --nodes=1                   # Request one or more nodes.
#SBATCH --gpus=1                   # Request one or more GPUs (adjust as needed).
#SBATCH --time=48:00:00             # Wall time limit (HH:MM:SS).
#SBATCH --output="/Jobs/Timm_FineTune_Classification_Output_%j.log"      # Standard output file.
#SBATCH --error="/Jobs/Timm_FineTune_Classification_Error_%j.log"        # Standard error file.

# Example multi-run loop using all CLI args for Timm_FineTune_Classification.py.

# Path to the training script.
SCRIPT="/path/to/Timm_FineTune_Classification.py"

# Global settings that can be adjusted.
NUM_CLASSES=3
DATA_DIR="/path/to/dataset/"
BASE_OUTPUT_DIR="/path/to/output/"
MODEL_NAME="densenet"
OPTIMIZER="adamw"
DO_SPLIT=1
FORCE_SPLIT=0
SPLIT_RATIO=0.2
EPOCHS=125
WARMUP_EPOCHS=1
BATCH_SIZES=(256 128)
TRIALS=(1 2 3 4 5)
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
NUM_WORKERS=8
DEVICE="cuda"
RESUME_FROM_CHECKPOINT=""  # Leave empty to not pass the argument.
VERBOSE=1
JUDGE_BY="both"               # val_loss, val_accuracy, or both.
EARLY_STOPPING_PATIENCE=""    # Empty means disabled; set an integer to enable.
GRAD_ACCUM_STEPS=1
MAX_GRAD_NORM=""              # Empty means disabled; set float to enable.
USE_AMP=1
USE_MIXUP_FN=0
MIXUP_ALPHA=0.5
USE_EMA=0
SAVE_EVERY=""                 # Empty means disabled; set integer to save every N epochs.
SPLIT_TRAIN_FOLDER=""         # Optional explicit path to a pre-split training folder.
SPLIT_VAL_FOLDER=""           # Optional explicit path to a pre-split validation folder.

# Loop over trials and batch sizes and run the training script with constructed args.
for TRIAL in "${TRIALS[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    # Construct output directory per run.
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/Results_${BATCH_SIZE}_T${TRIAL}"

    echo "Starting training: Trial ${TRIAL}, Batch Size ${BATCH_SIZE}"

    # Start building the command with required args.
    CMD=(python "$SCRIPT"
      --numClasses "$NUM_CLASSES"
      --dataDir "$DATA_DIR"
      --doSplit "$DO_SPLIT"
      --forceSplit "$FORCE_SPLIT"
      --splitRatio "$SPLIT_RATIO"
      --epochs "$EPOCHS"
      --modelName "$MODEL_NAME"
      --optimizer "$OPTIMIZER"
      --batchSize "$BATCH_SIZE"
      --learningRate "$LEARNING_RATE"
      --weightDecay "$WEIGHT_DECAY"
      --warmupEpochs "$WARMUP_EPOCHS"
      --numWorkers "$NUM_WORKERS"
      --device "$DEVICE"
      --outputDir "$OUTPUT_DIR"
      --verbose "$VERBOSE"
    )

    # Conditionally append optional args only if they are set (non-empty).
    if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
      CMD+=(--resumeFromCheckpoint "$RESUME_FROM_CHECKPOINT")
    fi

    if [ -n "$JUDGE_BY" ]; then
      CMD+=(--judgeBy "$JUDGE_BY")
    fi

    # Conditionally append explicit split folders when provided.
    if [ -n "$SPLIT_TRAIN_FOLDER" ]; then
      CMD+=(--splitTrainFolder "$SPLIT_TRAIN_FOLDER")
    fi

    if [ -n "$SPLIT_VAL_FOLDER" ]; then
      CMD+=(--splitValFolder "$SPLIT_VAL_FOLDER")
    fi

    if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
      CMD+=(--earlyStoppingPatience "$EARLY_STOPPING_PATIENCE")
    fi

    CMD+=(--gradAccumSteps "$GRAD_ACCUM_STEPS")

    if [ -n "$MAX_GRAD_NORM" ]; then
      CMD+=(--maxGradNorm "$MAX_GRAD_NORM")
    fi

    CMD+=(--useAmp "$USE_AMP")
    CMD+=(--useMixupFn "$USE_MIXUP_FN")
    CMD+=(--mixUpAlpha "$MIXUP_ALPHA")
    CMD+=(--useEma "$USE_EMA")

    if [ -n "$SAVE_EVERY" ]; then
      CMD+=(--saveEvery "$SAVE_EVERY")
    fi

    # Print and run the command.
    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
    if [[ $? -ne 0 ]]; then
      echo "Error: Command failed for Batch Size ${BATCH_SIZE}, Trial ${TRIAL}"
      exit 1
    fi

  done
done
