# Examples — HMB/Examples

<!-- Badges -->
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/hmb-helpers?label=PyPI)](https://pypi.org/project/hmb-helpers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Docs Status](https://img.shields.io/badge/docs-built-green.svg)](https://hmb-helpers-package.readthedocs.io/en/latest/)
[![Examples](https://img.shields.io/badge/examples-ready-brightgreen.svg)](https://github.com/HossamBalaha/HMB-Helpers-Package/tree/main/HMB/Examples)

This folder contains runnable example scripts that demonstrate common workflows built with the HMB Helpers
library. Each example below includes:

- A short purpose statement
- Expected inputs and outputs
- A compact CLI table showing flags, types, defaults and descriptions
- One or more example run commands

Keeping the CLI tables up to date: the tables mirror the scripts' argparse help or top-level constants. When you
change a script's flags or in-file defaults, please update the corresponding table in this README so the docs stay
authoritative.

Getting started (for beginners)
-------------------------------

Prerequisites

- Python 3.8 or later
- pip (or your preferred package manager)
- Recommended: create and use a virtual environment (venv or conda)
- For GPU work: appropriate CUDA/cuDNN and drivers installed for your PyTorch/TensorFlow build

Quick setup

1. From the repository root, create and activate a virtual environment (Windows example):

```bat
python -m venv .venv
.venv\Scripts\activate
```

2. Install project requirements:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer an editable install (so local changes to the package are immediately available):

```bat
python -m pip install -e .
```

Note: Starting with package version 0.2.0 the `HMB/Examples` folder is included in the published
distribution (wheel and sdist). If you installed `hmb-helpers` from PyPI at or after v0.2.1, these examples
are available under the installed `HMB/Examples` package; otherwise install from this repository to access them.

Run an example (overview)

- From Windows (cmd.exe) you can run the bundled BAT wrappers in `HMB\Examples\BAT Files\` using `call`.
- On POSIX (bash) use the shell scripts in `HMB/Examples/SH Files/`.
- Or run any example script directly with Python. Many examples accept argparse flags; run `--help` to show
  current options.

Windows wrapper example (cmd.exe):

```bat
call "HMB\Examples\BAT Files\Timm_Statistics_Analysis_Ablations.bat"
```

POSIX wrapper example (bash):

```bash
bash "HMB/Examples/SH Files/Timm_Statistics_Analysis_Ablations.sh"
```

Run a single example directly (Windows):

```bat
python "HMB\Examples\PyTorch_Tabular_CSV_Pipeline.py" --dataDir data --label ColumnName
```

Hints for first runs

- Use a very small dataset or a few images to smoke-test the script before committing large runs.
- Use `--help` on any script to verify flags and defaults: e.g. `python PATH/To/Script.py --help`.
- Keep experiment outputs out of source folders: create an `outputs/` directory or similar.

Examples
---------------------------

The following sections describe each example script. Each section keeps the CLI/args as a compact table
showing the flag, the type, the default value and a short description. These tables are intended as a quick
reference — always use `--help` on the script to view the authoritative, live argument list.

### PyTorch_UNet_Segmentation.py

- Purpose: Train or run inference with UNet-style segmentation models using PyTorch helpers.
- Inputs: training/validation image folders (or a dataset loader) and optional config file.
- Outputs: checkpoints, logs, evaluation metrics and predicted masks.
- Example run (Windows):
  ```bat
  python "HMB\Examples\PyTorch_UNet_Segmentation.py" --DataDir Data --OutputDir OutputV2
  ```

CLI / key parameters

| Flag               | Type  | Default               | Description                               |
|--------------------|:-----:|-----------------------|-------------------------------------------|
| --ModelName        |  str  | ResidualAttentionUNet | Model to instantiate                      |
| --DataDir          |  str  | Data                  | Dataset directory (images & masks)        |
| --OutputDir        |  str  | OutputV2              | Output folder for checkpoints & artifacts |
| --Phase            |  str  | Train                 | Train or Infer                            |
| --NumEpochs        |  int  | 50                    | Number of training epochs                 |
| --BatchSize        |  int  | 16                    | Training batch size                       |
| --ImageSize        |  int  | 128                   | Square image size (resize)                |
| --LearningRate     | float | 1e-4                  | Learning rate                             |
| --WeightDecay      | float | 1e-6                  | Weight decay                              |
| --Device           |  str  | cuda                  | Device selection (cuda or cpu)            |
| --NumWorkers       |  int  | 1                     | DataLoader workers                        |
| --NumClasses       |  int  | 1                     | Number of segmentation classes            |
| --ResumeCheckpoint |  str  |                       | Resume checkpoint path                    |
| --UseAMP           | flag  | False                 | Use automatic mixed precision (PyTorch)   |
| --DPI              |  int  | 720                   | DPI for saved figures                     |

Notes: the script saves the hyperparameter JSON to the output folder and supports resuming from checkpoints.

---

### PyTorch_Tabular_CSV_Pipeline.py

- Purpose: End‑to‑end tabular workflow — preprocessing, training, evaluation and reporting.
- Inputs: one or more CSV files (each treated as a dataset). The script reads `Models` from the project config.
- Outputs: per-dataset outputs (checkpoints, CSV reports, plots, per-sample predictions).
- Example run:
  ```bat
  python "HMB\Examples\PyTorch_Tabular_CSV_Pipeline.py" --dataDir data/csvs --saveDir outputs
  ```

CLI / key parameters

| Flag              | Type | Default      | Description                                                    |
|-------------------|:----:|--------------|----------------------------------------------------------------|
| --dataDir         | str  | (required)   | Directory containing CSV files (each file is a dataset)        |
| --saveDir         | str  | Experiments  | Root folder for per-dataset outputs                            |
| --configPath      | str  | Configs.json | Path to JSON config (contains Models list & training settings) |
| --labelColumn     | str  | Label        | Name of target column in CSV                                   |
| --dropFirstColumn | flag | False        | Drop the first column (ID/index)                               |
| --noOfTrials      | int  | 1            | Number of independent trials per dataset                       |

Notes: The pipeline uses `TabularPreprocessor` artifacts stored under each dataset output folder. It also
saves the transformed data CSVs for inspection.

Config file
-----------

This example ships with an optional JSON config file `PyTorch_Tabular_CSV_Pipeline.json` (located next to the
example). The pipeline will load this file when present and use the settings as defaults. The script still accepts
command-line flags which override values in the config file.

Example config (full file included with the example):

```json
{
  "BatchSize": 16384,
  "NumEpochs": 500,
  "TrainFraction": 0.80,
  "ValFraction": 0.1,
  "TestFraction": 0.1,
  "LearningRate": 0.001,
  "NumericScaler": "Standard",
  "Device": "cuda",
  "Models": [
    "TabTransformerModel",
    "VAEClassifier",
    "ContrastiveClassifier"
  ],
  "UseAmp": true,
  "Explain": true,
  "PlotDPI": 720,
  "PlotFigSize": [
    10,
    5
  ],
  "EvalBatchSize": 64,
  "SaveArtifacts": true
}
```

Recommended / commonly-used fields
----------------------------------

- BatchSize: Large-batch default for training. The script may override this per-model or per-experiment.
- NumEpochs: Number of training epochs.
- TrainFraction / ValFraction / TestFraction: Data split fractions (should sum to 1.0 when used together).
- Device: "cuda" or "cpu". Use "cpu" for debugging on non-GPU machines.
- Models: List of model factory names the pipeline will try (order is respected when running multiple models).
- UseAmp: Enable PyTorch automatic mixed precision when available (faster mixed-precision training on GPUs).
- Explain: When true, the pipeline will run the configured explainability methods (may be slow for large datasets).
- PlotDPI / PlotFigSize: Controls figure resolution and default size for saved plots.
- EvalBatchSize: Batch size used during evaluation/inference.
- SaveArtifacts: When true, intermediate artifacts (preprocessed datasets, encoders, model checkpoints) are persisted.

Other advanced options (present in the example JSON)
-------------------------------------------------

- Heavy: Toggle to enable more expensive diagnostics/plots (set true for full-run experiments).
- ComputeECE: Compute expected calibration error plots during evaluation (may increase runtime).
- ExportFailureCases: Save per-sample misclassification records for post-hoc analysis.
- MaxRows / MaxSamplesToEval: Useful for smoke-tests or CI to limit dataset size.

Edit the JSON to tune large-run defaults; prefer overriding on the command line for ad-hoc runs or CI smoke-tests.

---

### TF_UNet_Training.py

- Purpose: TF/Keras UNet training example with randomized hyperparameter trials.
- Inputs: dataset image and mask folders — configured in-file.
- Outputs: checkpoints, training curves and per-trial artifacts.
- Example run: edit the top-level variables in the file then run with Python.

Configuration (edit in script)

| Variable                  | Type  | Example / Default | Description                   |
|---------------------------|:-----:|-------------------|-------------------------------|
| datasetBase / datasetName |  str  | see file          | Base path to dataset files    |
| imagesPath / masksPath    |  str  | see file          | Paths to images and masks     |
| inputSize                 | tuple | (256,256,3)       | Model input shape             |
| epochs                    |  int  | 1                 | Number of epochs (in example) |
| trials                    |  int  | 3                 | Number of randomized trials   |
| whichModel                |  str  | VNet              | Model choice (VNet or SegNet) |

Notes: this script uses in-file constants and does not expose an argparse CLI; update variables directly.

---

### TF_UNet_EvalPredict.py

- Purpose: Run inference and per-image evaluation for TF UNet models.
- Inputs: model weights and an images folder with matching mask files.
- Outputs: per-image metrics CSV, evaluation summary JSON and optional saved predicted masks / figures.
- Example run:
  ```bash
  python "HMB/Examples/TF_UNet_EvalPredict.py" --ModelWeights /path/to/weights --DataDir /path/to/images --MasksDir /path/to/masks
  ```

CLI / key parameters

| Flag              |    Type     | Default    | Description                                       |
|-------------------|:-----------:|------------|---------------------------------------------------|
| --ModelName       |     str     | VNet       | Model factory name                                |
| --ModelWeights    |     str     | (required) | Path to model weights                             |
| --DataDir         |     str     | (required) | Images folder                                     |
| --MasksDir        |     str     | (required) | Masks folder (must match images)                  |
| --MaskPostfix     |     str     | ""         | Postfix to map image->mask filenames              |
| --OutputDir       |     str     | Output     | Output folder for metrics & predictions           |
| --InputSize       | int,int,int | 256 256 3  | Input H W C                                       |
| --NumClasses      |     int     | 1          | Number of segmentation classes                    |
| --SavePredictions |    flag     | False      | Save predicted mask PNGs                          |
| --HyperparamsJson |     str     | ""         | Optional JSON of hyperparameters used at training |

Notes: the script validates that masks exist for all listed images and raises a helpful error listing missing files.

---

### Timm_FineTune_Classification.py

- Purpose: Fine-tune image classification models using timm backbones and a flexible training configuration.
- Inputs: image dataset (folder-per-class) or pre-split train/val folders.
- Outputs: Best/Last checkpoints, training history CSV, per-split evaluation artifacts and plots.
- Example run:
  ```bat
  python "HMB\Examples\Timm_FineTune_Classification.py" --dataDir C:\data\mydataset --numClasses 5 --outputDir ./out
  ```

CLI / key parameters

| Flag                               | Type  | Default                    | Description                                             |
|------------------------------------|:-----:|----------------------------|---------------------------------------------------------|
| --dataDir                          |  str  | (required)                 | Dataset root (folder-per-class)                         |
| --numClasses                       |  int  | (required)                 | Number of classes                                       |
| --outputDir                        |  str  | ./Output                   | Output folder                                           |
| --modelName                        |  str  | eva02_large_patch14_448... | timm model name                                         |
| --imageSize                        |  int  | 448                        | Resize size                                             |
| --optimizer                        |  str  | adamw                      | Optimizer                                               |
| --epochs                           |  int  | 10                         | Training epochs                                         |
| --batchSize                        |  int  | 8                          | Batch size                                              |
| --learningRate                     | float | 1e-5                       | Learning rate                                           |
| --weightDecay                      | float | 0.01                       | Weight decay                                            |
| --doSplit / --forceSplit           | flag  | False                      | Create or force dataset split into `dataDir + ' Split'` |
| --splitRatio                       | float | 0.2                        | Validation split ratio when splitting                   |
| --numWorkers                       |  int  | 1                          | DataLoader workers                                      |
| --device                           |  str  | cuda / cpu                 | Device selection                                        |
| --resumeFromCheckpoint             |  str  |                            | Resume checkpoint                                       |
| --useAmp / --useMixupFn / --useEma | flag  | False                      | Optional training features                              |
| --verbose                          | flag  | False                      | Print detailed logs                                     |

Notes: the script saves `TrainingArgs.json` in the output folder for reproducibility.

---

### Timm_Statistics_Analysis_Ablations.py

- Purpose: Aggregate trial-level predictions across systems, generate ROC/PR/ECE plots and statistical reports.
- Inputs: experiment folders containing per-trial prediction CSVs.
- Outputs: combined predictions CSV, PR/ROC/AUC/ECE plots, and statistical analysis CSVs.

CLI / key parameters

| Flag                 | Type | Default              | Description                                        |
|----------------------|:----:|----------------------|----------------------------------------------------|
| --baseExpDir         | str  | /path/to/Experiments | Base folder containing system subfolders           |
| --predCSVFileFix     | str  | _Predictions_        | Substring used to locate per-trial prediction CSVs |
| --subsets            | list | [Train, Test]        | Subset names to process                            |
| --actualColName      | str  | trueClassName        | Column name for ground truth labels                |
| --actualColIDColName | str  | trueClassIndex       | Column name for ground truth label indices         |
| --predictionColName  | str  | predictedClassName   | Column name for predicted label names              |
| --probabilityColName | str  | probabilities        | Column name for predicted probability vectors      |
| --explainMethods     | list | []                   | Explainability methods to run (optional)           |
| --maxExplainImages   | int  | 0                    | Max images per trial to run explainability on      |
| --explainDatasetDir  | str  | None                 | Dataset to use for explainability processing       |
| --device             | str  | cuda                 | Device for model loading/explainability            |
| --dpi                | int  | 720                  | DPI for saved figures                              |
| --verbose            | flag | False                | Enable verbose logging                             |

Notes: this helper stitches per-trial CSVs across trials in a system, infers classes, and computes multi‑trial
confidence intervals.

---

### Train_TF_Pretrained_Attention_Model_from_DataFrame.py

- Purpose: Train/evaluate an attention-based TF model when inputs are provided as a DataFrame (paths + labels).
- Inputs: DataFrame or CSV with image paths and labels.
- Outputs: checkpoints, evaluation curves and statistics.

Configuration (edit in script)

| Variable                            | Type  | Default / example | Description                                             |
|-------------------------------------|:-----:|-------------------|---------------------------------------------------------|
| CURRENT_PHASE                       |  str  | STATISTICS        | Which phase(s) to run (TRAINING/TESTING/STATISTICS/ALL) |
| initialEpochs / fineTuneEpochs      |  int  | 25 / 25           | Training schedule                                       |
| batchSize                           |  int  | 32                | Batch size                                              |
| imgSize                             | tuple | (512,512)         | Target image size                                       |
| baseModelString / attentionBlockStr |  str  | ResNet50V2 / ECA  | Model components                                        |
| baseDir / baseStorageDir            |  str  | set in file       | Dataset & experiment output locations                   |

Notes: this example uses RawImageFolder.ToDataFrame() and orchestrates training/testing/statistics flows via file
constants.

---

### Explain_TF_Pretrained_Attention_Model_from_DataFrame.py

- Purpose: Produce TSNE/UMAP visualizations and Grad-CAM style explanations for TF attention models trained from a
  DataFrame.
- Inputs: trained model (.keras/.h5) and dataset folder with Training/Validation/Test splits.
- Outputs: TSNE/UMAP plots and CAM visualizations saved to the experiment folder.

CLI / key parameters

| Flag                | Type  | Default    | Description                               |
|---------------------|:-----:|------------|-------------------------------------------|
| --model             |  str  | (required) | Path to trained model                     |
| --baseModelString   |  str  | (required) | Base architecture used in training        |
| --attentionBlockStr |  str  | (required) | Attention block string used in training   |
| --dataRoot          |  str  | Dataset    | Root folder with Training/Validation/Test |
| --imgSize           |  int  | 512        | Image size                                |
| --batchSize         |  int  | 16         | Batch size for feature extraction         |
| --numUMAP           |  int  | 1000       | Number of samples for TSNE/UMAP (0 = all) |
| --useCAM            | flag  | False      | Enable CAM visualizations                 |
| --numCAM            |  int  | 32         | Number of CAM images to render            |
| --camType           |  str  | gradcam    | CAM method                                |
| --useGPU            | flag  | False      | Run CAM on GPU if available               |
| --alpha             | float | 0.45       | Overlay alpha for heatmaps                |
| --dpi               |  int  | 720        | DPI for saved figures                     |

Notes: the script builds a DataFrame that matches the training label encoder, ensuring class index consistency.

---

### Machine_Learning_Classification_Pipeline.py

- Purpose: Demonstrates an Optuna-based ML pipeline: tuning, testing, trial runs and SHAP explainability for tabular
  features.
- Inputs: CSV feature files.
- Outputs: Optuna results, trial metrics, statistical reports and SHAP visualizations.

Configuration (edit in script)

| Variable                   | Type | Default / example | Description                                                                           |
|----------------------------|:----:|-------------------|---------------------------------------------------------------------------------------|
| CURRENT_PHASE              | str  | TRAINING          | Which phases to run (TRAINING/TESTING/TRIALS/STATISTICS/REPORTING/EXPLAINABILITY/ALL) |
| BASE_DIR                   | str  | path/to/features  | Folder with CSV feature files and where Optuna results live                           |
| NUM_OF_TUNING_TRIALS       | int  | 250               | Number of Optuna trials                                                               |
| NUM_OF_TRIALS              | int  | 10                | Number of stability trials                                                            |
| MODELS_LIST / SCALERS_LIST | list | see file          | Model and preprocessor candidates for tuning                                          |

Notes: edit `BASE_DIR` and constants at the top of the file before running.

Platform wrappers
-----------------

- Windows wrappers: `HMB/Examples/BAT Files/` — run from repository root with `call`.
- POSIX wrappers: `HMB/Examples/SH Files/` — run on Linux/macOS.

Tips & best practices
---------------------

- Inspect each example's top-level docstring: many scripts provide argparse help and usage examples.
- Use small synthetic datasets for fast smoke tests before full runs — this saves time and GPU costs.
- Keep outputs in dedicated `outputs/` or experiment folders and avoid writing checkpoints into source directories.
- When in doubt, run the script with `--help` to confirm live flags and default values.

Troubleshooting (common issues)
-------------------------------

- "File not found" or missing datasets: verify that file paths and dataset folder names match the flags you passed.
- CUDA / device errors: ensure your PyTorch/TensorFlow installation matches your CUDA driver and toolkit. Use
  `--device cpu` to test CPU-only runs.
- Permission errors when writing outputs: create the output folder manually or run with an account that has write
  permission for the target folder.

Contributing and help
---------------------

- Please read `CONTRIBUTING.md` for how to propose changes and submit pull requests.
- If you find a bug in an example, update the example script and the CLI table in this README so users have the
  correct guidance.

License
-------

This repository is licensed under the MIT license. See the top-level `LICENSE` file for details.

Links
-----

Helpful links and references:

- Documentation (built): https://hmb-helpers-package.readthedocs.io/en/latest/
- Project homepage / top-level README: ../../README.md
- Contributing guidelines: ../../CONTRIBUTING.md
- License: ../../LICENSE
- PyPI package page: https://pypi.org/project/hmb-helpers/

If you need to report a bug or request a feature, please consult `CONTRIBUTING.md` and open an issue using the
project's issue tracker (see the top-level README for the repository URL).
