@echo off
REM TF_UNet_EvalPredict.bat.
REM Windows batch launcher for TF_UNet_EvalPredict.py.

setlocal enabledelayedexpansion

:: Path to the script (edit as needed).
set "SCRIPT=TF_UNet_EvalPredict.py"

:: Defaults (edit to match your environment).
set "ModelName=UNet"
set "ModelWeights=C:\Users\Hossam\Downloads\VNet Trial 1\Trial 1.weights.h5"
set "DataDir=C:\Users\Hossam\Downloads\PH2\Images"
set "MasksDir=C:\Users\Hossam\Downloads\PH2\Lesion"
set "MaskPostfix=_lesion.bmp"
set "OutputDir=C:\Users\Hossam\Downloads\VNet Trial 1\Output"
set "InputSize=256 256 3"
set "BatchSize=8"
set "NumClasses=1"
set "SavePredictions=1"  :: 1 to save predicted masks.
:: Optional path to hyperparameters JSON file.
set "HyperparamsJson="
:: Trials are discovered under a parent folder when TrialsParent is set.
:: The batch will iterate directories matching TrialsPattern under TrialsParent.
set "TrialsParent="
:: Pattern to match trial directories, example: Trial*.
set "TrialsPattern=Trial*"

:: Ensure output dir exists when running single-run mode.
if not exist "%OutputDir%" mkdir "%OutputDir%"

:: -----------------------------
:: Subroutine :run_one
:: Builds and executes the python command for a single trial or single run.
:: Optional first parameter is the trial root path.
:: -----------------------------
goto :main

:run_one
:: Parameter expansion: %~1 is trialRoot when called with an argument.
set "trialRoot=%~1"
if not "%trialRoot%"=="" (
  :: If a trial root was supplied, derive files from it.
  set "OutputDir=%trialRoot%\Output"
  if not exist "%OutputDir%" mkdir "%OutputDir%"

  :: Auto-discover a weights file inside the trial folder.
  set "ModelWeights="
  for %%F in ("%trialRoot%\*weights*.h5") do (
    if "!ModelWeights!"=="" set "ModelWeights=%%~fF"
  )
  if "%ModelWeights%"=="" (
    for %%F in ("%trialRoot%\*.h5") do (
      if "!ModelWeights!"=="" set "ModelWeights=%%~fF"
    )
  )
  if "%ModelWeights%"=="" set "ModelWeights=%trialRoot%\Trial.weights.h5"

  :: Auto-discover hyperparameters JSON if present.
  if exist "%trialRoot%\Hyperparameters.json" (
    set "HyperparamsJson=%trialRoot%\Hyperparameters.json"
  ) else (
    set "HyperparamsJson="
  )
)

:: Build the command string; keep quoting for paths with spaces.
set "CMD=python "%SCRIPT%" --ModelName "%ModelName%" --ModelWeights "%ModelWeights%" --DataDir "%DataDir%" --OutputDir "%OutputDir%" --BatchSize %BatchSize% --NumClasses %NumClasses%"

:: Append InputSize (three values).
for /f "tokens=1-3" %%a in ("%InputSize%") do set "CMD=!CMD! --InputSize %%a %%b %%c"

:: Optionally forward masks and postfix.
if not "%MasksDir%"=="" (
  set "CMD=!CMD! --MasksDir \"%MasksDir%\""
)
if not "%MaskPostfix%"=="" (
  set "CMD=!CMD! --MaskPostfix \"%MaskPostfix%\""
)

:: Optionally forward hyperparameters JSON.
if not "%HyperparamsJson%"=="" (
  set "CMD=!CMD! --HyperparamsJson \"%HyperparamsJson%\""
)

:: Optionally request saving predictions.
if not "%SavePredictions%"=="0" (
  set "CMD=!CMD! --SavePredictions"
)

:: Execute the command and handle errors by returning a non-zero code to the caller.
echo Running: !CMD!
:: Also append the executed command to a per-run log inside the OutputDir.
echo !CMD! >> "%OutputDir%\EvalCommand.log"
cmd /C "!CMD!"
if errorlevel 1 (
  :: Persist the failure to a per-trial error log.
  echo Error: command failed with exit code %ERRORLEVEL% >> "%OutputDir%\RunErrors.log"
  :: Print concise error to console and return to caller.
  echo Error: command failed with exit code %ERRORLEVEL%.
  exit /b 1
)

echo Evaluation completed for %trialRoot%.
goto :eof

:main
:: Main loop: when TrialsParent is set iterate directories, otherwise run single invocation.
if not "%TrialsParent%"=="" (
  :: Check for any matching trial directories.
  if not exist "%TrialsParent%\%TrialsPattern%" (
    echo No trial subfolders matching pattern "%TrialsPattern%" were found under "%TrialsParent%".
    echo Ensure trials are named like "Trial 1", "Trial 2" or adjust TrialsPattern.
    endlocal
    exit /b 1
  )

  :: Print discovered trials for visibility.
  echo Found trial directories under "%TrialsParent%":
  for /d %%D in ("%TrialsParent%\%TrialsPattern%") do (
    echo   - %%~fD
  )

  :: Iterate matching trial directories and run evaluation for each.
  for /d %%D in ("%TrialsParent%\%TrialsPattern%") do (
    call :run_one "%%~fD"
    if errorlevel 1 (
      echo Error: trial %%~fD failed with exit code %ERRORLEVEL%.
      echo Continuing to next trial.
    )
  )
) else (
  call :run_one
  if errorlevel 1 (
    echo Error: command failed with exit code %ERRORLEVEL%.
    endlocal
    exit /b 1
  )
)

endlocal
