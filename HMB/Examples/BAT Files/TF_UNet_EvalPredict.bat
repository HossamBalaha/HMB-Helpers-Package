@echo off
REM TF_UNet_EvalPredict.bat
REM Windows batch launcher for TF_UNet_EvalPredict.py

setlocal enabledelayedexpansion

:: Path to the script (edit as needed)
set "SCRIPT=TF_UNet_EvalPredict.py"

:: Defaults (edit to match your environment)
set "ModelName=UNet"
set "ModelWeights=C:\Users\Hossam\Downloads\VNet Trial 1\Trial 1.weights.h5"
set "DataDir=C:\Users\Hossam\Downloads\PH2\Images"
set "MasksDir=C:\Users\Hossam\Downloads\PH2\Lesion"
set "MaskPostfix=_lesion.bmp"
set "OutputDir=C:\Users\Hossam\Downloads\VNet Trial 1\Output"
set "InputSize=256 256 3"
set "BatchSize=8"
set "NumClasses=1"
set "SavePredictions=1"  :: 1 to save predicted masks

if not exist "%OutputDir%" mkdir "%OutputDir%"

:: Build base command
set "CMD=python "%SCRIPT%" --ModelName "%ModelName%" --ModelWeights "%ModelWeights%" --DataDir "%DataDir%" --OutputDir "%OutputDir%" --BatchSize %BatchSize% --NumClasses %NumClasses%"

:: Append InputSize (three values)
for /f "tokens=1-3" %%a in ("%InputSize%") do set "CMD=!CMD! --InputSize %%a %%b %%c"

if not "%MasksDir%"=="" (
  set "CMD=!CMD! --MasksDir "%MasksDir%""
)

if not "%MaskPostfix%"=="" (
  set "CMD=!CMD! --MaskPostfix "%MaskPostfix%""
)

if "%SavePredictions%"=="1" (
  set "CMD=!CMD! --SavePredictions"
)

echo Running: !CMD!
cmd /C "!CMD!"
if errorlevel 1 (
  echo Error: command failed with exit code %ERRORLEVEL%
  endlocal
  exit /b 1
)

endlocal

echo Evaluation completed.
exit /b 0
