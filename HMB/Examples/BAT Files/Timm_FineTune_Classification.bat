@echo off
REM Timm_FineTune_Classification.bat
REM Windows batch equivalent of the provided SLURM bash script.
REM Edit variables below to suit your environment before running.

setlocal enabledelayedexpansion

:: Path to the training script (edit to point to your python script)
set "SCRIPT=Timm_FineTune_Classification.py"

:: Global settings (defaults copied from the SH file)
set "NUM_CLASSES=3"
set "DATA_DIR=C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset"
set "BASE_OUTPUT_DIR=C:\Users\Hossam\Downloads\Brain Tumor MRI Output"
set "MODEL_NAME=eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
set "IMAGE_SIZE=448"
set "OPTIMIZER=adamw"
set "DO_SPLIT=0"
set "FORCE_SPLIT=0"
set "SPLIT_RATIO=0.2"
set "EPOCHS=125"
set "WARMUP_EPOCHS=1"
set "BATCH_SIZES=32"
set "TRIALS=1 2 3 4 5"
set "LEARNING_RATE=1e-5"
set "WEIGHT_DECAY=0.01"
set "NUM_WORKERS=8"
set "DEVICE=cuda"
set "RESUME_FROM_CHECKPOINT="
set "VERBOSE=1"
set "JUDGE_BY=both"
set "EARLY_STOPPING_PATIENCE="
set "GRAD_ACCUM_STEPS=1"
set "MAX_GRAD_NORM="
set "USE_AMP=1"
set "USE_MIXUP_FN=0"
set "MIXUP_ALPHA=0.5"
set "USE_EMA=0"
set "SAVE_EVERY="
set "SPLIT_TRAIN_FOLDER=C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset\Training"
set "SPLIT_VAL_FOLDER=C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset\Testing"
set "SPLIT_TEST_FOLDER=C:\Users\Hossam\Downloads\Brain Tumor MRI Dataset\Testing"

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
