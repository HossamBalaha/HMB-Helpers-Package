@echo off
REM PyTorch_UNet_Segmentation.bat
REM Windows batch launcher for PyTorch_UNet_Segmentation.py

setlocal enabledelayedexpansion

n:: Path to the script (edit as needed)
set "SCRIPT=PyTorch_UNet_Segmentation.py"

n:: Defaults (mirror script defaults)
set "ModelName=ResidualAttentionUNet"
set "DataDir=Data"
set "OutputDir=OutputUNet"
set "Phase=Train"
set "NumEpochs=50"
set "BATCH_SIZES=16"
set "ImageSize=128"
set "LearningRate=1e-4"
set "WeightDecay=1e-6"
set "Device=cuda"
set "NumWorkers=4"
set "NumClasses=1"
set "ResumeCheckpoint="
set "USE_AMP=0"
set "TRIALS=1"

n:: Iterate trials and batch sizes
for %%T in (%TRIALS%) do (
  for %%B in (%BATCH_SIZES%) do (
    set "OUTPUT_RUN_DIR=%OutputDir%\%ModelName%_%%B_T%%T"
    echo.
    echo ==================================================
    echo Starting run: Trial %%T, Batch Size %%B
    echo Output dir: !OUTPUT_RUN_DIR!

    set CMD=python "%SCRIPT%" --ModelName "%ModelName%" --DataDir "%DataDir%" --OutputDir "!OUTPUT_RUN_DIR!" --Phase "%Phase%" --NumEpochs %NumEpochs% --BatchSize %%B --ImageSize %ImageSize% --LearningRate %LearningRate% --WeightDecay %WeightDecay% --Device "%Device%" --NumWorkers %NumWorkers% --NumClasses %NumClasses%

    if not "%ResumeCheckpoint%"=="" (
      if exist "%ResumeCheckpoint%" (
        set CMD=!CMD! --ResumeCheckpoint "%ResumeCheckpoint%"
      )
    )
    if "%USE_AMP%"=="1" set "CMD=!CMD! --UseAMP"

    echo Running: !CMD!
    cmd /C "!CMD!"
    if errorlevel 1 (
      echo Error: command failed with exit code %ERRORLEVEL%
      endlocal
      exit /b 1
    )
  )
)

endlocal

echo All runs completed.
exit /b 0
