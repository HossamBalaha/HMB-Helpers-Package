# Changelog

All notable changes to the **HMB Helpers Package** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-07

### Added

- `Examples/README.md`: added a friendly, beginner-oriented README for the `HMB/Examples` folder that
  documents prerequisites, setup, per-script CLI tables, run examples, and troubleshooting guidance.
- `HMB/Utils.py` (`SimpleSerializeForJson`): Added a detailed docstring describing behavior, parameters and return
  value for the JSON-serialization helper. Added a cross-reference note recommending `ConvertToJsonSerializable`
  when callers need a more robust serializer that preserves dtype/shape metadata and includes type tags for
  reconstruction of tensors/arrays (NumPy, PyTorch, TensorFlow).
- `HMB/Utils.py` (`SafeParseProbabilities`): Added a robust probabilities parsing helper that accepts `None`,
  numeric scalars, lists/tuples, NumPy arrays, and string representations (JSON or Python literals). The helper
  gracefully handles NaN variants and returns an empty list on unrecoverable parse errors. A backward-compatible
  lowercase alias `safe_parse_probabilities` was provided for callers relying on the previous name.
- `HMB/Utils.py` (`fprint`): Added a `fprint` utility function that wraps the built-in `print` with `flush=True`
  to ensure messages appear immediately in the console. The function accepts additional positional and keyword
  arguments to be forwarded to the underlying `print` call.
- `HMB/PlotsHelper.py` (`PlotClassDistribution`): Added a utility to plot per-class sample counts (bar chart),
  annotate counts, and optionally save/display the figure. This complements dataset inspection utilities such as
  `GenericImagesDatasetHandler.PrintSummary` and `GenericImagesDatasetHandler.CreateYAML`.
- `HMB/PlotsHelper.py` (`SaveMatplotlibFigure`): Added a small helper to standardize saving Matplotlib figures with
  publication-friendly defaults (dpi, bbox/tight layout handling, optional vector output with embedded fonts, file
  format selection, and safe fallbacks). This centralizes figure-export behavior across helpers and examples.
- `HMB/ExplainabilityHelper.py` & `HMB/Examples/PyTorch_Tabular_CSV_Pipeline.py`: Added an explainability runner
  and improved SHAP visualization exports. The example pipeline exposes a new `--phase explain` CLI option and
  helper functions now export SHAP summary/beeswarm/bar/scatter/decision/dependence figures. Figures are saved as
  both PDF (vector) and PNG (raster) by default to simplify inclusion in reports and publications.
- `HMB/Initializations.py` (`VIDEO_SUFFIXES`): Added a centralized `VIDEO_SUFFIXES` constant listing
  recognized video file extensions (for example, `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`). This constant
  standardizes video file discovery across dataset loaders and examples and reduces duplication of extension
  lists in video-related helpers and examples.
- `HMB/PyTorchHelper.py` (`PyTorchVideoTransforms`): Added `PyTorchVideoTransforms`, a small helper/class to
  construct common video preprocessing pipelines. The helper supports per-frame resizing, normalization using
  model `data_config` when available, temporal sampling strategies (uniform, fixed-rate, center sampling), and
  avoids double-resizing when a transform already includes a `Resize` op. This centralizes video transform logic
  for training, evaluation and explainability flows.
- `HMB/DatasetsHelper.py` (`PyTorchVideoClassificationDataset`): Added `PyTorchVideoClassificationDataset`, a
  dataset class that decodes video files using PyAV (`av`), supports configurable frame sampling (num frames,
  stride, sample strategy), optional per-frame transforms, lazy decoding for large datasets, and safe behavior
  with PyTorch `DataLoader` workers. This class provides a simple, drop-in dataset for video classification
  experiments and examples.
- `requirements.txt`: Added `av` (PyAV) to support efficient video decoding in `PyTorchVideoClassificationDataset`
  and other video helpers. Users running video examples should ensure system-level dependencies for PyAV are
  available for their platform (see PyAV installation notes).
- Packaging: Added `PyYAML` to default install requirements to ensure YAML parsing support is available for
  configuration loading and dataset utilities (e.g., `ReadProjectConfig`, `GenericImagesDatasetHandler.CreateYAML`).
- `HMB/Utils.py`: Added a lightweight energy/CO2 estimation helper that wraps `codecarbon` (pinned as
  `codecarbon` in `requirements.txt`). The helper can be invoked from examples and pipelines to
  produce per-run energy and CO2 estimates (useful for reproducibility notes and sustainability reporting).
- Packaging: Promoted commonly-used packages to the core `install_requires` to simplify the default install
  (pandas, matplotlib, tqdm, scikit-learn) and removed duplicates from related extras (`scientific`, `plotting`,
  `utils`). This reduces redundancy and makes typical helper scripts usable without selecting extras.
- `StatisticalAnalysisHelper.py` (`CohensDPaired`, `BenjaminiHochberg`): Verified implementations and improved
  documentation. `CohensDPaired` computes Cohen's d for paired samples as the mean of differences divided by the
  standard deviation of differences (sample ddof=1 by default); it returns NaN when inputs are empty or the
  difference standard deviation is zero. `BenjaminiHochberg` performs BH FDR adjustment (returns adjusted p-values
  in the original input order) and enforces monotonicity (non-decreasing adjusted p-values). Both functions have
  clearer docstrings and example usages added to `HMB/StatisticalAnalysisHelper.py`.

### Changed

- `PerformanceMetrics.py`: Improved docstrings, fixed typos, and enhanced the comments for better clarity and
  maintainability.
- `PerformanceMetrics.py` (`PlotClasswisePRFBar`): Added `xAxisRotation` parameter (default 45) to control x-axis label
  rotation.
- `PyTorchTrainingPipeline.py` (`GenericTabularEvaluatePredictPlotSubset`): Updated the flow to include additional
  performance plots and to utilize the `TabularPreprocessor` for preprocessing of tabular inputs.
- `Examples/`: Updated the existing examples and added more examples to reflect the API and the new changes.
- `Examples/`: Renamed example script `HMB/Examples/Machine_Learning_Pipeline.py` to
  `HMB/Examples/Machine_Learning_Classification_Pipeline.py` and updated related example wrappers and
  documentation to match the new filename and updated example content.
- `HMB/Examples/Timm_FineTune_Classification.py`: Improved checkpoint discovery and selection used by the
  inference/test flow. The script now scans the `--outputDir` for both `.pt` and `.pth` checkpoints, extracts a
  numeric metric value from filenames using the `_Metric_<value>` convention, converts metrics to floats and
  selects the best checkpoint automatically (printing the chosen file). This makes the example more robust when
  multiple checkpoints exist and simplifies selecting the best model for evaluation.
- `HMB/Examples/Timm_Statistics_Analysis_Ablations.py`: Hardened checkpoint discovery/selection during system
  processing. The script now scans each trial directory for `.pt` and `.pth` files, validates the presence of
  checkpoints (raises a clear error when none are found), prints an informational message when only a single
  checkpoint is available (and uses it), and otherwise extracts numeric `_Metric_<value>` suffixes to automatically
  select the best checkpoint. `judgeBy` is read from saved training args for reporting. This reduces noisy failures
  and improves clarity when evaluating multiple trials.
- `HMB/Examples/Timm_Statistics_Analysis_Ablations.py`: Use `imageSize` from saved `TrainingArgs.json` when
  available instead of unconditionally overriding from the parent-directory LUT. The script now calls
  `imgSize = trainingArgs.get("imageSize", imgSize)` so that explicit training configurations take precedence
  while retaining a sensible fallback inferred from directory names when the training args are missing. This
  improves behavior for experiments where the image size was specified at training time.
- `HMB/Examples/Timm_Statistics_Analysis_Ablations.py` & `HMB/Examples/Timm_FineTune_Classification.py`: When a
  user- or training-specified `imageSize` disagrees with the model's expected input size (for example, user set
  448 but the model expects 224), the evaluation and explainability flows now detect this mismatch and prefer
  the model's expected input shape for preprocessing. The scripts warn when they adjust the size and avoid
  feeding incompatible shapes to the model (prevents runtime errors and incorrect resizing semantics).
- `Examples/README.md` (`PyTorch_Tabular_CSV_Pipeline`): Documented example JSON config and explained commonly used and
  advanced configuration fields such as `BatchSize`, `NumEpochs`, `Models`, `UseAmp`, `Explain`, `PlotDPI`,
  `SaveArtifacts`, and recommended evaluation/sample limits (for example, `MaxRows`/`MaxSamplesToEval`).
- `HMB/Examples/BAT Files/Timm_FineTune_Classification.bat`: Synchronized Windows batch example comments and
  variable documentation with the SLURM shell example (`HMB/Examples/SH Files/Timm_FineTune_Classification.sh`). The
  `.bat` now includes detailed inline descriptions for common configuration variables (data paths, model name,
  optimizer, split settings, training hyperparameters, and device flags) to improve usability for Windows users.
- `PerformanceMetrics.py` (`PlotMultiTrialROCAUC`, `PlotMultiTrialPRCurve`): Improved input validation and handling
  for `allYTrue`. If a single numpy array or a single-item list is supplied, it is now reliably repeated to match
  the number of prediction trials in `allYPred`. Additionally, both `PlotMultiTrialPRCurve` and `PlotMultiTrialROCAUC`
  now perform robust validation of prediction scores: non-finite values (NaN / Inf) are filtered out prior to calling
  scikit-learn metric functions, and trials that become degenerate after filtering (for example, only a single label
  present) are skipped with an informative warning instead of raising an uncaught exception. These changes make the
  multi-trial plotting functions more resilient to upstream data/model issues and provide clearer diagnostics when
  inputs are invalid.
- `PerformanceMetrics.py`: Standardized figure saving across the module to use `HMB.PlotsHelper.SaveMatplotlibFigure`.
  Plotting helpers such as `PlotConfusionMatrix`, `PlotROCAUCCurve`, `PlotPRCCurve`, `PlotMultiTrialROCAUC`,
  `PlotMultiTrialPRCurve`, and `PlotInteractionEffect` now delegate file export to the centralized helper which
  ensures consistent PDF/PNG sibling outputs, tight layout handling, directory creation and robust error handling.
- `PerformanceMetrics.py` (`PlotMultiTrialROCAUC`, `PlotMultiTrialPRCurve`): Improved the zoomed-in inset behavior
  by computing the inset bounds dynamically from the plotted data (instead of using fixed hard-coded axis ranges).
  The inset position is also chosen dynamically to avoid occluding the zoomed region. A safe fallback to the
  original fixed inset is used when no appropriate zoom region can be detected or on error. This makes ROC and
  PR multi-trial insets more robust across a wider set of curve shapes and score distributions.
- `HMB/DatasetsHelper.py` (`TabularPreprocessor`): Added a selectable `numericScaler` constructor parameter
  (accepts "Standard" (default), "MinMax", "Robust", a scaler instance, or `None` to disable scaling).
  The preprocessor now configures and fits the requested scaler during `Fit` and applies it during `Transform`
  only when enabled. This preserves previous default behavior while allowing explicit disabling of feature
  scaling or selecting alternative scalers.
- `PyTorchTrainingPipeline.py` (`GenericTabularEvaluatePredictPlotSubset`): Added a `numericScaler` argument that is
  forwarded to `TabularPreprocessor` when constructing or loading preprocessing artifacts. Accepts the same values
  as `TabularPreprocessor` ("Standard" (default), "MinMax", "Robust", a sklearn-like scaler instance, or `None`).
- `HMB/DatasetsHelper.py` (`GenericImagesDatasetHandler.PlotClassDistribution`): Now delegates to
  `HMB.PlotsHelper.PlotClassDistribution` for plotting when available (falls back to legacy bar-chart utility
  if the centralized plot helper fails). This ensures a single, consistent plotting implementation and
  improved annotation and pareto-style cumulative curves.
- Packaging: Added missing extras mapping for previously-listed requirements:
    - `tensorboard` added to the `tensorflow` extra (useful for TensorFlow runs and logging).
    - `opencv-python-headless` added to the `cv` extra (headless CI/server installs).
    - `openslide-bin` added to the `medical` extra (system-level OpenSlide helper package).
- Packaging: Updated packaging metadata to improve consistency and ease of installation:
    - Added the missing extras mapping shown above.
    - Brought `setup.py` minima into alignment with the development `requirements.txt` for core runtime
      dependencies (numpy, pillow, PyYAML, pandas, matplotlib, tqdm, scikit-learn) and refreshed the `pytorch`
      extra minima. The `setup.py` entries remain intentionally looser than the full pinned `requirements.txt`
      so downstream users can select platform-appropriate wheels (for example, CUDA-enabled `torch` builds).
    - The `all` extra was adjusted to avoid installing large, platform-specific frameworks (PyTorch, TensorFlow,
      Keras, tensorboard, etc.) by default. Users who need those frameworks should install them explicitly via the
      `pytorch` or `tensorflow` extras or follow the framework vendor installer instructions (this reduces accidental
      long-running installations on systems without the appropriate device/platform support).
- `Initializations.py` (`UpdateMatplotlibSettings`): Enhanced plotting defaults for publication-quality figures.
  The function now applies a colorblind-friendly Seaborn palette and sets the axes color cycle (using `cycler` when
  available), embeds TrueType fonts in PDF/PS outputs (`pdf.fonttype=42`, `ps.fonttype=42`), adjusts default line
  width to 1.5 for a lighter publication look, and enables automatic figure layout (`figure.autolayout=True`) while
  disabling `constrained_layout` by default to avoid conflicts with calls to `tight_layout()` (reduces repeated
  "The figure layout has changed to tight" warnings). Safe fallbacks were added so these enhancements do not fail in
  environments missing optional packages. Added `snsStyle` parameter (default "whitegrid") to allow users to select
  their preferred Seaborn style when Seaborn is available.
- Added runtime font detection: `UpdateMatplotlibSettings` now inspects installed fonts and only sets
  `mpl.rcParams['font.serif']` to families that exist on the host system (includes common Linux-friendly fallbacks
  such as `DejaVu Serif`, `Liberation Serif`, and `Noto Serif`). If no suitable serif is found, the function falls
  back to a safe `sans-serif` default. This prevents the Matplotlib warning "Generic family 'serif' not found ..."
  on Linux and headless/container environments.
- `StatisticalAnalysisHelper.py`: Refined docstrings and added concrete usage examples.
  The `HMB/StatisticalAnalysisHelper.py` module received clearer, more detailed docstrings for several
  public functions (parameter descriptions, return values, and behavior notes). In addition, compact example
  snippets demonstrating common workflows (residual plots, Bland–Altman, QQ/residual diagnostics, and
  correlation heatmaps) were added to improve discoverability and make the helper functions easier to adopt.
- `StatisticalAnalysisHelper.py` (`kstest` usage): Updated internal calls to SciPy's `stats.kstest` to
  explicitly provide a `method` parameter (defaulting to `'interpolate'` when a p-value is required) and to
  consume the returned `pvalue` attribute when available. This change silences the SciPy 1.17+ FutureWarning,
  ensures correct p-value extraction across SciPy versions, and improves forward-compatibility with future
  SciPy releases where legacy attributes will be removed.
- `PlotsHelper.py` (`SaveMatplotlibFigure` and global save behavior): Reupdated and hardened the figure-saving helper
  and standardized export across the package. Highlights:
    - `SaveMatplotlibFigure` now normalizes input paths (accepts paths with or without extensions), exposes
      `exportPdf`/`exportPng` flags, and performs robust tight-layout handling that avoids calling `tight_layout()`
      for `constrained_layout=True` figures (reduces repeated "The figure layout has changed to tight" warnings).
    - Safer save fallbacks: vector (PDF) export is attempted first and failures no longer abort raster export.
    - A non-invasive wrapper was added to `matplotlib.pyplot.savefig` so existing `plt.savefig(...)` calls across
      modules will route through the centralized helper by default (the wrapper falls back to the original
      `savefig` if it encounters errors).
    - The helper was applied in core modules to centralize export behavior: `HMB/ExplainabilityHelper.py`,
      `HMB/AttentionMapsHelper.py`, `HMB/PerformanceMetrics.py` (and other plotting modules benefit from the new
      wrapper).
- `HMB/WSIHelper.py` (`IsSimilarityAccepted`, `GetEmptyPercentage`, `TileExtractionAlignmentHandler`):
    - Added PIL to NumPy conversion support to `IsSimilarityAccepted` and `GetEmptyPercentage`.
    - Enhanced `IsSimilarityAccepted` by introducing configurable threshold arguments (`miThreshold=0.35`,
      `cosSimThreshold=0.75`, `pHashThreshold=20`) and a `strict` boolean flag. The hardcoded condition check
      was replaced with a dynamic evaluation that counts the number of met conditions and appends this count to
      the status string. The function now returns `True` if all conditions are met (when `strict=True`) or if
      at least one condition is met (when `strict=False`).
    - Added `miThreshold`, `cosSimThreshold`, `pHashThreshold`, and `strict` arguments to
      `TileExtractionAlignmentHandler` to propagate these configurable thresholds to the similarity check.

### Fixed

- `PerformanceMetrics.py` (`CalculatePerformanceMetrics`): Fixed incorrect calculations for MCC and Yule macro values.
- `PerformanceMetrics.py`: Ensure that `probs` is always treated as a 2D numpy array (prevents incorrect shapes/inputs).
- `PerformanceMetrics.py` (`ComputeBrierScore`): Ensure inputs are converted/validated as numpy arrays for vectorized
  operations and added input validation.
- `HMB/Examples/Timm_Statistics_Analysis_Ablations.py`: Fixed a crash when no per-trial prediction CSVs are present.
  The script now detects when no predictions are found across trials, skips the affected system gracefully, and
  returns empty results instead of raising a TypeError when attempting to compute `len()` on a `None` object.
- `HMB/Utils.py` (`SafeTrapz`) and callers: Added `SafeTrapz`, a robust trapezoidal-integration helper that
  prefers `numpy.trapz` when available and falls back to a pure-Python implementation when necessary. Core
  modules that computed area-under-curve values now use `SafeTrapz` to avoid crashes in environments where
  `numpy.trapz` is missing or unavailable. Affected modules include `HMB/PyTorchHelper.py` (e.g.
  `EvaluateModelOnPerturbations`), `HMB/ExplainabilityHelper.py`, and `HMB/PerformanceMetrics.py`.
- Packaging: Ensure `HMB/Examples` is included in published distributions (wheel and sdist). Added `MANIFEST.in`,
  updated `setup.py` package data globs, and added `HMB/Examples/__init__.py` so example scripts are shipped and
  the folder is importable. These packaging fixes were applied as part of the 0.2.0 release.
- `HMB/WSIHelper.py`: Fixed the import of `IsSimilarityAccepted`.

### Notes

- `requirements.txt` remains the canonical, fully-pinned development environment (including CUDA-specific
  wheel pins). The `setup.py` minima are intentionally looser so end users on different platforms can pick
  appropriate platform-specific wheels (for example, CUDA-enabled `torch` builds).

- Note: `requirements.txt` now includes `av` (PyAV) to enable video decoding support used by the new
  `PyTorchVideoClassificationDataset` and related video helpers. On some platforms PyAV requires additional
  system libraries (FFmpeg development headers); consult PyAV installation instructions if `pip install av`
  fails on your platform.

## [0.1.0] - 2026-04-28

### Added

- Initial stable release of the HMB Helpers Package.
- Core utility modules for image processing, segmentation, deep learning workflows, and text/PDF handling.
- Cross-platform development support for Windows and POSIX environments.
- Comprehensive test suite, documentation, and PyPI distribution.
- Zenodo archival record for long-term citation.

[//]: # (### Changed)

[//]: # (- )

[//]: # ()

[//]: # (### Deprecated)

[//]: # (- )

[//]: # ()

[//]: # (### Removed)

[//]: # (- )

[//]: # ()

[//]: # (### Fixed)

[//]: # (- )

[//]: # ()

[//]: # (### Security)

[//]: # (-)
