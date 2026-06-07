@echo off
REM Timm_FineTune_Classification.bat
REM Windows batch equivalent of the provided SLURM bash script.
REM Edit variables below to suit your environment before running.

setlocal enabledelayedexpansion

:: Path to the training script (edit to point to your python script)
:: Update the path to your local copy of Timm_FineTune_Classification.py
set "SCRIPT=Timm_FineTune_Classification.py"

:: Global settings (defaults copied from the SH file)
:: NUM_CLASSES: Number of output classes for the classifier (integer).
set "NUM_CLASSES=3"
:: DATA_DIR: Root path to your dataset. Expected structure depends on the training script
::           (commonly a folder per class or a pre-split train/val directory).
set "DATA_DIR=path\to\your\dataset"
:: BASE_OUTPUT_DIR: Base directory where results, logs and checkpoints will be saved.
set "BASE_OUTPUT_DIR=path\to\output\directory"
:: MODEL_NAME: Name of the timm model architecture to use (e.g. resnet50, densenet121, vit_base_patch16_224).
set "MODEL_NAME=eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
:: IMAGE_SIZE: Input image size expected by the model (e.g. 224 for 224x224). The training script should handle resizing.
set "IMAGE_SIZE=448"
:: OPTIMIZER: Optimizer type to use (e.g. adamw, sgd). Must be supported by the training script.
set "OPTIMIZER=adamw"
:: DO_SPLIT: If set (1) the dataset will be split into train/val according to SPLIT_RATIO; 0 disables splitting.
set "DO_SPLIT=0"
:: FORCE_SPLIT: If set (1) force re-creating the train/val split even if split folders exist.
set "FORCE_SPLIT=0"
:: SPLIT_RATIO: Fraction of the data to reserve for validation (0.0 - 1.0).
set "SPLIT_RATIO=0.2"
:: EPOCHS: Total number of training epochs.
set "EPOCHS=125"
:: WARMUP_EPOCHS: Number of warmup epochs used by the LR scheduler (if supported).
set "WARMUP_EPOCHS=1"
:: BATCH_SIZES: Space-separated list of batch sizes to try in separate runs (loops over these values).
set "BATCH_SIZES=16 32 64 128"
:: TRIALS: Space-separated list of trial identifiers; useful for repeating experiments with different seeds.
set "TRIALS=1 2 3 4 5 6 7 8 9 10"
:: LEARNING_RATE: Initial learning rate used by the optimizer.
set "LEARNING_RATE=1e-5"
:: WEIGHT_DECAY: Weight decay (L2 regularization) for the optimizer.
set "WEIGHT_DECAY=0.01"
:: NUM_WORKERS: Number of DataLoader worker processes for loading data.
set "NUM_WORKERS=8"
:: DEVICE: Device to run training on (e.g. "cuda" or "cpu"). The training script should accept this flag.
set "DEVICE=cuda"
:: RESUME_FROM_CHECKPOINT: Path to a checkpoint file to resume training from. Leave empty to disable.
set "RESUME_FROM_CHECKPOINT="
:: VERBOSE: Verbosity level passed to the script (e.g. 0=quiet, 1=normal, 2=debug). Check the script for exact meanings.
set "VERBOSE=1"
:: JUDGE_BY: Metric used to select the best model when saving ("val_loss", "val_accuracy", or "both").
set "JUDGE_BY=both"
:: EARLY_STOPPING_PATIENCE: If set to an integer, training stops after this many epochs without improvement. Leave empty to disable.
set "EARLY_STOPPING_PATIENCE="
:: GRAD_ACCUM_STEPS: Number of steps to accumulate gradients before performing an optimizer step.
set "GRAD_ACCUM_STEPS=1"
:: MAX_GRAD_NORM: If set (float), gradients will be clipped to this maximum norm. Leave empty to disable.
set "MAX_GRAD_NORM="
:: USE_AMP: Enable automatic mixed precision (1) or disable (0) if supported by the training script.
set "USE_AMP=1"
:: USE_MIXUP_FN: Enable (1) or disable (0) mixup augmentation function during training.
set "USE_MIXUP_FN=0"
:: MIXUP_ALPHA: Mixup alpha parameter controlling strength of mixup augmentation.
set "MIXUP_ALPHA=0.5"
:: USE_EMA: Use Exponential Moving Average (EMA) of model weights for evaluation/saving (1/0).
set "USE_EMA=0"
:: SAVE_EVERY: Save a checkpoint every N epochs. Leave empty to rely on the script's default/save-on-improvement.
set "SAVE_EVERY="
:: SPLIT_TRAIN_FOLDER: Optional explicit path to a pre-split training folder. If provided, splitting is skipped/overridden.
set "SPLIT_TRAIN_FOLDER=path\to\your\dataset\train"
:: SPLIT_VAL_FOLDER: Optional explicit path to a pre-split validation folder. If provided, splitting is skipped/overridden.
set "SPLIT_VAL_FOLDER=path\to\your\dataset\val"
:: SPLIT_TEST_FOLDER: Optional explicit path to a pre-split test folder. If provided, splitting is skipped/overridden.
set "SPLIT_TEST_FOLDER=path\to\your\dataset\test"

:: Iterate trials and batch sizes
for %%T in (%TRIALS%) do (
  for %%B in (%BATCH_SIZES%) do (
    set "OUTPUT_DIR=%BASE_OUTPUT_DIR%\Results_%%B_T%%T"
    echo.
    echo ==================================================
    echo Starting training: Trial %%T, Batch Size %%B
    echo Output dir: !OUTPUT_DIR!

    :: Build base command
    set CMD=python "%SCRIPT%" --numClasses %NUM_CLASSES% --dataDir "%DATA_DIR%" --splitRatio %SPLIT_RATIO% --epochs %EPOCHS% --modelName "%MODEL_NAME%" --optimizer %OPTIMIZER% --batchSize %%B --learningRate %LEARNING_RATE% --weightDecay %WEIGHT_DECAY% --warmupEpochs %WARMUP_EPOCHS% --numWorkers %NUM_WORKERS% --device "%DEVICE%" --outputDir "!OUTPUT_DIR!"

    :: Conditionally append flags
    if "%DO_SPLIT%"=="1" set "CMD=!CMD! --doSplit"
    if "%FORCE_SPLIT%"=="1" set "CMD=!CMD! --forceSplit"
    if "%VERBOSE%"=="1" set "CMD=!CMD! --verbose"
    if defined IMAGE_SIZE set "CMD=!CMD! --imageSize %IMAGE_SIZE%"
    if defined RESUME_FROM_CHECKPOINT (
      if exist "%RESUME_FROM_CHECKPOINT%" (
        set CMD=!CMD! --resumeFromCheckpoint "!RESUME_FROM_CHECKPOINT!"
      )
    )
    if defined JUDGE_BY set "CMD=!CMD! --judgeBy %JUDGE_BY%"
    if defined SPLIT_TRAIN_FOLDER if not "!SPLIT_TRAIN_FOLDER!"=="" set CMD=!CMD! --splitTrainFolder "!SPLIT_TRAIN_FOLDER!"
    if defined SPLIT_VAL_FOLDER if not "!SPLIT_VAL_FOLDER!"=="" set CMD=!CMD! --splitValFolder "!SPLIT_VAL_FOLDER!"
    if defined SPLIT_TEST_FOLDER if not "!SPLIT_TEST_FOLDER!"=="" set CMD=!CMD! --splitTestFolder "!SPLIT_TEST_FOLDER!"
    if defined EARLY_STOPPING_PATIENCE if not "%EARLY_STOPPING_PATIENCE%"=="" set "CMD=!CMD! --earlyStoppingPatience %EARLY_STOPPING_PATIENCE%"
    if defined GRAD_ACCUM_STEPS if not "%GRAD_ACCUM_STEPS%"=="" set "CMD=!CMD! --gradAccumSteps %GRAD_ACCUM_STEPS%"
    if defined MAX_GRAD_NORM if not "%MAX_GRAD_NORM%"=="" set "CMD=!CMD! --maxGradNorm %MAX_GRAD_NORM%"
    if "%USE_AMP%"=="1" set "CMD=!CMD! --useAmp"
    if "%USE_MIXUP_FN%"=="1" (
      set "CMD=!CMD! --useMixupFn"
      set "CMD=!CMD! --mixUpAlpha %MIXUP_ALPHA%"
    )
    if "%USE_EMA%"=="1" set "CMD=!CMD! --useEma"
    if not "%SAVE_EVERY%"=="" set "CMD=!CMD! --saveEvery %SAVE_EVERY%"

    echo Running: !CMD!

    :: Execute the constructed command
    cmd /V:ON /C "!CMD!"
    if errorlevel 1 (
      echo Error: Command failed with exit code %ERRORLEVEL%
      endlocal
      exit /b 1
    )
  )
)

endlocal

echo All runs completed.
exit /b 0
