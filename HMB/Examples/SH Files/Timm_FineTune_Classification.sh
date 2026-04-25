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
# NUM_CLASSES: Number of output classes for the classifier (integer).
NUM_CLASSES=3
# DATA_DIR: Root path to your dataset. Expected structure depends on the training script
#           (commonly a folder per class or a pre-split train/val directory).
DATA_DIR="/path/to/dataset/"
# BASE_OUTPUT_DIR: Base directory where results, logs and checkpoints will be saved.
BASE_OUTPUT_DIR="/path/to/output/"
# MODEL_NAME: Name of the timm model architecture to use (e.g. resnet50, densenet121, vit_base_patch16_224).
MODEL_NAME="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
# IMAGE_SIZE: Input image size expected by the model (e.g. 224 for 224x224). The training script should handle resizing.
IMAGE_SIZE=448
# OPTIMIZER: Optimizer type to use (e.g. adamw, sgd). Must be supported by the training script.
OPTIMIZER="adamw"
# DO_SPLIT: If set (1) the dataset will be split into train/val according to SPLIT_RATIO; 0 disables splitting.
DO_SPLIT=1
# FORCE_SPLIT: If set (1) force re-creating the train/val split even if split folders exist.
FORCE_SPLIT=0
# SPLIT_RATIO: Fraction of the data to reserve for validation (0.0 - 1.0).
SPLIT_RATIO=0.2
# EPOCHS: Total number of training epochs.
EPOCHS=125
# WARMUP_EPOCHS: Number of warmup epochs used by the LR scheduler (if supported).
WARMUP_EPOCHS=1
# BATCH_SIZES: Array of batch sizes to try in separate runs (loops over these values).
BATCH_SIZES=(256 128)
# TRIALS: Array of trial identifiers; useful for repeating experiments with different seeds.
TRIALS=(1 2 3 4 5 6 7 8 9 10)
# LEARNING_RATE: Initial learning rate used by the optimizer.
LEARNING_RATE=1e-5
# WEIGHT_DECAY: Weight decay (L2 regularization) for the optimizer.
WEIGHT_DECAY=0.01
# NUM_WORKERS: Number of DataLoader worker processes for loading data.
NUM_WORKERS=8
# DEVICE: Device to run training on (e.g. "cuda" or "cpu"). The training script should accept this flag.
DEVICE="cuda"
# RESUME_FROM_CHECKPOINT: Path to a checkpoint file to resume training from. Leave empty to disable.
RESUME_FROM_CHECKPOINT=""  # Leave empty to not pass the argument.
# VERBOSE: Verbosity level passed to the script (e.g. 0=quiet, 1=normal, 2=debug). Check the script for exact meanings.
VERBOSE=1
# JUDGE_BY: Metric used to select the best model when saving ("val_loss", "val_accuracy", or "both").
JUDGE_BY="both"
# EARLY_STOPPING_PATIENCE: If set to an integer, training stops after this many epochs without improvement. Leave empty to disable.
EARLY_STOPPING_PATIENCE=""
# GRAD_ACCUM_STEPS: Number of steps to accumulate gradients before performing an optimizer step.
GRAD_ACCUM_STEPS=1
# MAX_GRAD_NORM: If set (float), gradients will be clipped to this maximum norm. Leave empty to disable.
MAX_GRAD_NORM=""
# USE_AMP: Enable automatic mixed precision (1) or disable (0) if supported by the training script.
USE_AMP=1
# USE_MIXUP_FN: Enable (1) or disable (0) mixup augmentation function during training.
USE_MIXUP_FN=0
# MIXUP_ALPHA: Mixup alpha parameter controlling strength of mixup augmentation.
MIXUP_ALPHA=0.5
# USE_EMA: Use Exponential Moving Average (EMA) of model weights for evaluation/saving (1/0).
USE_EMA=0
# SAVE_EVERY: Save a checkpoint every N epochs. Leave empty to rely on the script's default/save-on-improvement.
SAVE_EVERY=""
# SPLIT_TRAIN_FOLDER: Optional explicit path to a pre-split training folder. If provided, splitting is skipped/overridden.
SPLIT_TRAIN_FOLDER="C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset\Training"
# SPLIT_VAL_FOLDER: Optional explicit path to a pre-split validation folder. If provided, splitting is skipped/overridden.
SPLIT_VAL_FOLDER="C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset\Testing"
# SPLIT_TEST_FOLDER: Optional explicit path to a pre-split test folder. If provided, splitting is skipped/overridden.
SPLIT_TEST_FOLDER="C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset\Testing"

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
    )

    # Conditionally append optional args only if they are set (non-empty).
    # -n means "if the string is non-empty". For boolean flags, we check if they are set to 1.

    if [ -n "$DO_SPLIT" ] && [ "$DO_SPLIT" -eq 1 ]; then
      CMD+=(--doSplit)
    fi

    if [ -n "$FORCE_SPLIT" ] && [ "$FORCE_SPLIT" -eq 1 ]; then
      CMD+=(--forceSplit)
    fi

    if [ -n "$VERBOSE" ] && [ "$VERBOSE" -eq 1 ]; then
      CMD+=(--verbose)
    fi

    if [ -n "$IMAGE_SIZE" ]; then
      CMD+=(--imageSize "$IMAGE_SIZE")
    fi

    if [ -n "$RESUME_FROM_CHECKPOINT" ] && [ -f "$RESUME_FROM_CHECKPOINT" ]; then
      CMD+=(--resumeFromCheckpoint "$RESUME_FROM_CHECKPOINT")
    fi

    if [ -n "$JUDGE_BY" ]; then
      CMD+=(--judgeBy "$JUDGE_BY")
    fi

    if [ -n "$SPLIT_TRAIN_FOLDER" ]; then
      CMD+=(--splitTrainFolder "$SPLIT_TRAIN_FOLDER")
    fi

    if [ -n "$SPLIT_VAL_FOLDER" ]; then
      CMD+=(--splitValFolder "$SPLIT_VAL_FOLDER")
    fi

    if [ -n "$SPLIT_TEST_FOLDER" ]; then
      CMD+=(--splitTestFolder "$SPLIT_TEST_FOLDER")
    fi

    if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
      CMD+=(--earlyStoppingPatience "$EARLY_STOPPING_PATIENCE")
    fi

    if [ -n "$GRAD_ACCUM_STEPS" ]; then
      CMD+=(--gradAccumSteps "$GRAD_ACCUM_STEPS")
    fi

    if [ -n "$MAX_GRAD_NORM" ]; then
      CMD+=(--maxGradNorm "$MAX_GRAD_NORM")
    fi

    if [ -n "$USE_AMP" ] && [ "$USE_AMP" -eq 1 ]; then
      CMD+=(--useAmp)
    fi

    if [ -n "$USE_MIXUP_FN" ] && [ "$USE_MIXUP_FN" -eq 1 ]; then
      CMD+=(--useMixupFn)
      CMD+=(--mixUpAlpha "$MIXUP_ALPHA")
    fi

    if [ -n "$USE_EMA" ] && [ "$USE_EMA" -eq 1 ]; then
      CMD+=(--useEma)
    fi

    if [ -n "$SAVE_EVERY" ]; then
      CMD+=(--saveEvery "$SAVE_EVERY")
    fi

    # Print and run the command.
    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
    if [[ $? -ne 0 ]]; then
      echo "Error: Command failed..."
      exit 1
    fi

  done
done
