import hashlib, time, json, shutil, os, cv2, torch, math, joblib, av
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.impute import SimpleImputer
from PIL import Image, ImageOps, ImageEnhance
from sklearn.model_selection import train_test_split
from typing import *
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from HMB.Utils import DumpJsonFile, ReadJsonFile
from HMB.PyTorchHelper import PyTorchVideoTransforms
from HMB.MachineLearningHelper import GetScalerObject
from HMB.Initializations import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from HMB.DataAugmentationHelper import PerformDataAugmentation
from HMB.PlotsHelper import PlotBarChart, PlotClassDistribution


class TabularPreprocessor:
  r'''
  Generic tabular data preprocessor for PyTorch classification pipelines.

  Handles numeric and categorical features, missing value imputation,
  feature scaling, ordinal encoding, and label encoding. Artifacts can be
  saved/loaded via joblib for reproducibility across training and inference.

  Parameters:
    ignoreCategorical (bool): If True, categorical/object columns are completely
      ignored during fitting and transformation. Only numeric features are processed.
      Defaults to False.
    numericScaler (str|object|None): Specifies the scaler to apply to numeric features.

  Attributes:
    ignoreCategorical (bool): Flag indicating whether categorical columns are skipped.
    numericColumns (list): List of detected numeric column names.
    categoricalColumns (list): List of detected categorical column names (empty if ignored).
    numericImputer (SimpleImputer): Imputer for numeric columns (fitted on training data).
    categoricalImputer (SimpleImputer): Imputer for categorical columns (fitted on training data).
    numericScaler (sklearn-like estimator or None): Scaler for numeric columns (fitted on training data).
      Can be None to disable scaling. The scaler may be a StandardScaler, MinMaxScaler,
      RobustScaler or any object implementing fit/transform.
    categoricalEncoder (OrdinalEncoder): Encoder for categorical columns (fitted on training data).
    labelEncoder (LabelEncoder): Encoder for target labels (fitted on training data).
    _featureNames (list): Cached ordered list of feature names used during transformation.
    _isLoaded (bool): Internal flag indicating whether artifacts were loaded or preprocessor was fitted.

  Notes:
    - The Fit method detects numeric and categorical columns, fits imputers, scalers, and encoders on the provided DataFrame.
    - The Transform method applies the fitted transformations to a new DataFrame and returns feature arrays and label vectors suitable for PyTorch models.
    - The Save and Load methods handle persistence of all artifacts to/from disk using joblib.
    - The IsLoaded method allows checking whether the preprocessor is ready for transformation (either fitted or loaded).
    - The GetFeatureNames method provides the ordered list of feature names corresponding to the transformed feature matrix columns.
    - The ValidateSchema method checks that a new DataFrame contains the expected columns before transformation, raising an error if there are mismatches.

  Example usage:
  .. code-block:: python

    # Initialize the preprocessor and ignore categorical columns.
    preprocessor = TabularPreprocessor(ignoreCategorical=True, numericScaler="Standard")
    preprocessor.Fit(trainDf, labelColumn="Label")

    # Transform training, validation, and test DataFrames.
    xTrain, yTrain = preprocessor.Transform(trainDf, labelColumn="Label")
    xVal, yVal = preprocessor.Transform(valDf, labelColumn="Label")
    xTest, yTest = preprocessor.Transform(testDf, labelColumn="Label")

    # Save the preprocessor artifacts for later use.
    preprocessor.Save("PreprocessorArtifacts")
  '''

  # Initialize the preprocessor with default attributes.
  def __init__(self, ignoreCategorical: bool = False, numericScaler: object = "Standard"):
    # Track whether categorical columns should be ignored.
    self.ignoreCategorical = ignoreCategorical
    # Track numeric and categorical column names.
    self.numericColumns = []
    self.categoricalColumns = []
    # Initialize imputers for missing values.
    self.numericImputer = None
    self.categoricalImputer = None
    # Initialize scaler selection and encoders.
    # `numericScaler` parameter can be one of the listed in `GetScalerObject`
    # or a custom scaler instance, or None to disable scaling.
    self.numericScalerSpec = numericScaler
    # `numericScaler` will hold the fitted scaler instance (or `None` if disabled).
    self.numericScaler = None
    self.categoricalEncoder = None
    self.labelEncoder = None
    # Store the final ordered feature names.
    self._featureNames = []
    # Track whether artifacts were loaded or fitted.
    self._isLoaded = False

  # Fit the preprocessor on a training DataFrame.
  def Fit(self, df: pd.DataFrame, labelColumn: str = None):
    # Automatically detect numeric columns.
    self.numericColumns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Conditionally detect categorical columns based on ignoreCategorical flag.
    self.categoricalColumns = [] if (self.ignoreCategorical) else df.select_dtypes(
      include=["object", "category"]).columns.tolist()
    # Remove label column from feature lists if present.
    if (labelColumn in self.numericColumns):
      self.numericColumns.remove(labelColumn)
    if (labelColumn in self.categoricalColumns):
      self.categoricalColumns.remove(labelColumn)

    # Fit numeric imputer and scaler when numeric columns exist.
    if (len(self.numericColumns) > 0):
      # Use median imputation for robustness to outliers.
      self.numericImputer = SimpleImputer(strategy="median")
      self.numericImputer.fit(df[self.numericColumns])
      # Configure and fit the numeric scaler according to the provided spec.
      try:
        spec = self.numericScalerSpec
        # If user explicitly disabled scaling by passing None, keep scaler None.
        if (spec is None):
          self.numericScaler = None
        # If spec is a string, map common choices to sklearn scalers.
        elif (isinstance(spec, str)):
          # This function should return a fitted scaler instance based on the string spec.
          self.numericScaler = GetScalerObject(spec)
        else:
          # Assume the user passed a scaler-like object (with fit/transform).
          self.numericScaler = spec

        # Fit the scaler when it is enabled (not None).
        if (self.numericScaler is not None):
          self.numericScaler.fit(self.numericImputer.transform(df[self.numericColumns]))
      except Exception:
        # On any failure configuring/scaling, fall back to no scaler to avoid breaking Fit.
        self.numericScaler = None

    # Fit categorical imputer and encoder when categorical columns exist and are not ignored.
    if ((not self.ignoreCategorical) and (len(self.categoricalColumns) > 0)):
      # Use most frequent value imputation for categorical data.
      self.categoricalImputer = SimpleImputer(strategy="most_frequent")
      self.categoricalImputer.fit(df[self.categoricalColumns])
      # Fit ordinal encoder with safe handling for unseen categories.
      self.categoricalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
      self.categoricalEncoder.fit(self.categoricalImputer.transform(df[self.categoricalColumns]))

    # Fit label encoder when label column is provided.
    if (labelColumn is not None):
      # Create and fit the label encoder on string-converted labels.
      self.labelEncoder = LabelEncoder()
      self.labelEncoder.fit(df[labelColumn].astype(str))

    # Build ordered feature name list for downstream consistency.
    self._featureNames = self.numericColumns.copy()
    if (len(self.categoricalColumns) > 0):
      self._featureNames.extend(self.categoricalColumns)
    # Mark preprocessor as fitted in memory.
    self._isLoaded = True

  # Transform a DataFrame to numerical arrays suitable for PyTorch models.
  def Transform(self, df: pd.DataFrame, labelColumn: str = None):
    # Initialize container for transformed feature parts.
    xParts = []
    # Transform numeric features when imputer and scaler are available.
    if (len(self.numericColumns) > 0 and self.numericImputer is not None):
      # Copy data to avoid SettingWithCopyWarning.
      numData = df[self.numericColumns].copy()
      # Apply median imputation to numeric columns.
      numData = self.numericImputer.transform(numData)
      # Apply scaling to imputed numeric columns only when a scaler is configured.
      if (self.numericScaler is not None):
        numData = self.numericScaler.transform(numData)
      # Append numeric features to output list.
      xParts.append(numData.astype(np.float32))

    # Transform categorical features when encoder is available and not ignored.
    if ((not self.ignoreCategorical) and (len(self.categoricalColumns) > 0) and (self.categoricalEncoder is not None)):
      # Copy data to avoid SettingWithCopyWarning.
      catData = df[self.categoricalColumns].copy()
      # Apply most-frequent imputation to categorical columns.
      catData = self.categoricalImputer.transform(catData)
      # Apply ordinal encoding to imputed categorical columns.
      catData = self.categoricalEncoder.transform(catData)
      # Append categorical features to output list.
      xParts.append(catData.astype(np.float32))

    # Concatenate feature parts if any features exist.
    x = np.hstack(xParts) if (xParts) else None
    # Initialize label array.
    y = None

    # Transform labels when label column and encoder are provided.
    if (labelColumn is not None and self.labelEncoder is not None):
      # Transform string labels to integer indices.
      y = self.labelEncoder.transform(df[labelColumn].astype(str))

    # Return feature matrix and label vector.
    return x, y

  # Save the preprocessor artifacts to a specified directory.
  def Save(self, path: str):
    # Create the target directory if it does not exist.
    os.makedirs(path, exist_ok=True)
    # Save numeric imputer when fitted.
    if (self.numericImputer is not None):
      joblib.dump(self.numericImputer, os.path.join(path, "NumericImputer.joblib"))
    # Save numeric scaler when fitted.
    if (self.numericScaler is not None):
      joblib.dump(self.numericScaler, os.path.join(path, "NumericScaler.joblib"))
    # Save categorical imputer when fitted.
    if (self.categoricalImputer is not None):
      joblib.dump(self.categoricalImputer, os.path.join(path, "CategoricalImputer.joblib"))
    # Save categorical encoder when fitted.
    if (self.categoricalEncoder is not None):
      joblib.dump(self.categoricalEncoder, os.path.join(path, "CategoricalEncoder.joblib"))
    # Save label encoder when fitted.
    if (self.labelEncoder is not None):
      joblib.dump(self.labelEncoder, os.path.join(path, "LabelEncoder.joblib"))
    # Save column metadata for schema validation.
    joblib.dump(self.numericColumns, os.path.join(path, "NumericColumns.joblib"))
    joblib.dump(self.categoricalColumns, os.path.join(path, "CategoricalColumns.joblib"))
    # Save the ignoreCategorical flag for reproducibility.
    joblib.dump(self.ignoreCategorical, os.path.join(path, "IgnoreCategorical.joblib"))
    # Mark preprocessor as loaded from disk artifacts.
    self._isLoaded = True

  # Load the preprocessor artifacts from a specified directory.
  def Load(self, path: str):
    # Load numeric imputer when artifact exists.
    if (os.path.exists(os.path.join(path, "NumericImputer.joblib"))):
      self.numericImputer = joblib.load(os.path.join(path, "NumericImputer.joblib"))
    # Load numeric scaler when artifact exists.
    if (os.path.exists(os.path.join(path, "NumericScaler.joblib"))):
      self.numericScaler = joblib.load(os.path.join(path, "NumericScaler.joblib"))
    # Load categorical imputer when artifact exists.
    if (os.path.exists(os.path.join(path, "CategoricalImputer.joblib"))):
      self.categoricalImputer = joblib.load(os.path.join(path, "CategoricalImputer.joblib"))
    # Load categorical encoder when artifact exists.
    if (os.path.exists(os.path.join(path, "CategoricalEncoder.joblib"))):
      self.categoricalEncoder = joblib.load(os.path.join(path, "CategoricalEncoder.joblib"))
    # Load label encoder when artifact exists.
    if (os.path.exists(os.path.join(path, "LabelEncoder.joblib"))):
      self.labelEncoder = joblib.load(os.path.join(path, "LabelEncoder.joblib"))
    # Load numeric columns list when artifact exists.
    if (os.path.exists(os.path.join(path, "NumericColumns.joblib"))):
      self.numericColumns = joblib.load(os.path.join(path, "NumericColumns.joblib"))
    # Load categorical columns list when artifact exists.
    if (os.path.exists(os.path.join(path, "CategoricalColumns.joblib"))):
      self.categoricalColumns = joblib.load(os.path.join(path, "CategoricalColumns.joblib"))
    # Load the ignoreCategorical flag when artifact exists.
    if (os.path.exists(os.path.join(path, "IgnoreCategorical.joblib"))):
      self.ignoreCategorical = joblib.load(os.path.join(path, "IgnoreCategorical.joblib"))
    # Set loaded flag when any core artifact is present.
    self._isLoaded = os.path.exists(os.path.join(path, "NumericColumns.joblib"))

  # Return whether artifacts were loaded or preprocessor was fitted.
  def IsLoaded(self) -> bool:
    # Return the internal loaded state.
    return bool(self._isLoaded)

  # Return the ordered list of feature names used during transformation.
  def GetFeatureNames(self):
    # Return cached feature names if available.
    if (self._featureNames):
      return self._featureNames
    # Fallback: reconstruct from tracked columns to prevent empty-list pandas errors.
    names = self.numericColumns.copy()
    if (len(self.categoricalColumns) > 0):
      names.extend(self.categoricalColumns)
    return names

  # Validate that a DataFrame contains the expected columns before transformation.
  def ValidateSchema(self, df: pd.DataFrame) -> bool:
    # Check for missing numeric columns.
    missingNumeric = [c for c in self.numericColumns if (c not in df.columns)]
    # Check for missing categorical columns.
    missingCategorical = [c for c in self.categoricalColumns if (c not in df.columns)]
    # Raise error when required columns are missing.
    if (missingNumeric or missingCategorical):
      raise ValueError(f"Schema mismatch. Missing columns: {missingNumeric + missingCategorical}")
    # Return true when schema validation passes.
    return True


class RawImageFolder(object):
  r'''
  Lightweight loader to get image paths and labels without transforms.

  Attributes:
    samples (list): List of tuples (imagePath, labelIndex) for all images in the dataset.
    classToIdx (dict): Mapping of class names to integer indices.
    classes (list): Sorted list of class names detected in the dataset.

  Notes:
    - The constructor scans the provided root directory for subdirectories (each representing a class) and collects
      image file paths along with their corresponding class labels.
    - Only files with extensions matching those in IMAGE_SUFFIXES are considered as valid images.
    - The __getitem__ method returns the file path and label index for a given sample index, allowing for lazy loading of images.
  '''

  def __init__(self, root, rootType="str", categories=None):
    r'''
    Initialize the `RawImageFolder` by scanning the root directory for class subfolders and image files.

    Parameters:
      root (str|dict): If `rootType` is "str", this should be a string path to the dataset root directory containing
        class subfolders. If `rootType` is "dict", this should be a dictionary mapping split names
        ("train", "val", "test") to their respective root directories.
      rootType (str): Type of the root parameter, either "str" for a single directory or "dict" for split-based
        directories. Default is "str".
      categories (list|None): Optional list of category names to include. If None, all subdirectories in the
        root will be considered as classes.
    '''

    from sklearn.preprocessing import LabelEncoder

    self.classToIdx = {}
    self.splits = []
    self.labels = []
    self.paths = []
    self.labelEncoder = LabelEncoder()
    self.encodedLabels = []

    if (rootType == "str"):
      if (not os.path.exists(root)):
        raise ValueError(f"Root directory not found: {root}")

      if (categories is None):
        self.classes = sorted(
          entry.name for entry in os.scandir(root) if (entry.is_dir())
        )
      else:
        self.classes = sorted(categories)

      self.classToIdx = {clsName: idx for idx, clsName in enumerate(self.classes)}

      for targetClass in self.classes:
        classDir = os.path.join(root, targetClass)
        for fname in sorted(os.listdir(classDir)):
          if (fname.lower().endswith(tuple(IMAGE_SUFFIXES))):
            path = os.path.join(classDir, fname)
            # self.samples.append((path, self.classToIdx[targetClass]))
            self.paths.append(path)
            self.labels.append(self.classToIdx[targetClass])

      self.encodedLabels = self.labelEncoder.fit_transform(self.labels)

    elif (rootType == "dict"):
      if (not isinstance(root, dict)):
        raise ValueError("Expected root to be a dict mapping splits to container paths.")

      # Ensure that the keys are "train", "val", and "test".
      expectedSplits = {"train", "val", "test"}
      if (not expectedSplits.issubset(root.keys())):
        raise ValueError(f"Expected root dict to contain keys: {expectedSplits}")

      # Walk the dataset folders and collect all image paths with labels and split info.
      for splitName, basePath in root.items():
        for category in categories:
          # Build category directory path.
          categoryPath = os.path.join(basePath, category)
          # Check whether the category path exists before listing files.
          if (os.path.exists(categoryPath)):
            for imageName in os.listdir(categoryPath):
              # Build full image path.
              imagePath = os.path.join(categoryPath, imageName)
              # Append image path to list.
              self.paths.append(imagePath)
              # Append label to list.
              self.labels.append(category)
              # Append split (train/val/test) to list.
              self.splits.append(splitName)

      self.encodedLabels = self.labelEncoder.fit_transform(self.labels)

    else:
      raise ValueError(f"Unsupported rootType: {rootType}. Expected 'str' or 'dict'.")

  def __len__(self):
    r'''
    Calculate the total number of samples (images) in the dataset.

    Returns:
      int: Total number of samples (images) in the dataset.
    '''

    return len(self.paths)

  def __getitem__(self, idx):
    r'''
    Retrieve the image path and label index for a given sample index.

    Parameters:
      idx (int): Index of the sample to retrieve.

    Returns:
      tuple: A tuple containing the image file path (str) and the corresponding label index (int).
    '''

    return (self.paths[idx], self.labels[idx])

  def GetPaths(self):
    r'''
    Get the list of image file paths in the dataset.

    Returns:
      list: A list of image file paths (str) in the dataset.
    '''

    return self.paths

  def GetLabels(self):
    r'''
    Get the list of labels corresponding to the image paths in the dataset.

    Returns:
      list: A list of labels (str) corresponding to the image paths in the dataset.
    '''

    return self.labels

  def ToDataFrame(self):
    r'''
    Convert the dataset samples into a pandas DataFrame for easier manipulation.

    Returns:
      pandas.DataFrame: A DataFrame with columns "image_path", "label", and "split" (if available).
    '''

    import pandas as pd

    data = {
      "image_path"      : self.paths,
      "label"           : self.labels,
      "category_encoded": self.encodedLabels.tolist(),
    }
    if (self.splits):
      data["split"] = self.splits

    df = pd.DataFrame(data)

    # Ensure that the DataFrame labels columns are strings.
    df["label"] = df["label"].astype(str)
    df["category_encoded"] = df["category_encoded"].astype(str)

    return df

  def GetLabelEncoder(self):
    r'''
    Get the fitted LabelEncoder instance for encoding and decoding labels.

    Returns:
      sklearn.preprocessing.LabelEncoder: The fitted LabelEncoder instance.
    '''

    return self.labelEncoder

  def EncodeLabel(self, label):
    r'''
    Encode a string label into its corresponding integer index using the fitted LabelEncoder.

    Parameters:
      label (str): The string label to encode.

    Returns:
      int: The integer index corresponding to the provided string label.
    '''

    return self.labelEncoder.transform([label])[0]

  def DecodeLabel(self, index):
    r'''
    Decode an integer label index back into its corresponding string label using the fitted LabelEncoder.

    Parameters:
      index (int): The integer index to decode.

    Returns:
      str: The string label corresponding to the provided integer index.
    '''

    return self.labelEncoder.inverse_transform([index])[0]


class GenericImagesDatasetHandler(object):
  r'''
  Generic handler for image classification datasets.

  The handler can auto-detect common dataset layouts (nested class folders
  or flat filenames with class prefix) or load a provided configuration JSON.
  It exposes utilities to create a standard train/val/test layout (folders
  per class), generate a lightweight dataset YAML for training tools, and
  write a small DatasetManifest.json with provenance and file checksums.

  Attributes:
    imageExtensions (set): Allowed image file extensions used when scanning.

  Notes:
    - Auto-detection supports three strategies: pre-split (train/val/test),
      nested class folders, and flat filenames with a class prefix.
    - Prepared datasets are written with `train/`, `val/` and `test/` folders
      under the requested output directory. A `Dataset.yaml` and
      `DatasetManifest.json` are written alongside the splits.

  Key methods:
    - Prepare(outputDir, valSplit, testSplit, balance, balanceMethod, balanceTarget, randomSeed)
      Prepare the dataset in a standard layout. Balancing (optional) supports
      `duplication` and `augmentation` methods and applies only to the `train` split.
    - CreateConfigFile(outputPath, splits, minSamplesPerClass, description)
      Generate and write a DatasetConfig.json describing classes, counts and metadata.
    - ValidateDatasetStructure(datasetPath, minSamplesPerClass=1, returnStructured=False)
      Validate a pre-split dataset; when `returnStructured=True` returns a dict
      with per-split/per-class counts, sample file paths and readability flags.
    - CreateYAML(outputDir)
      Create a small `Dataset.yaml` suitable for training tools (also includes
      YOLO-compatible keys `nc` and `names`).
    - BuildManifest(outputDir, splitMapping)
      Write `DatasetManifest.json` with basic provenance and per-file SHA256 where possible.

  Examples
  --------
  .. code-block:: python

    from pathlib import Path
    from HMB.DatasetsHelper import GenericImagesDatasetHandler

    # Initialize handler for a dataset folder and auto-detect structure.
    datasetPath = Path("/path/to/dataset")
    handler = GenericImagesDatasetHandler(datasetPath)

    # Print detected configuration summary.
    handler.PrintSummary()

    # Prepare dataset in standard train/val/test layout under output folder with balancing.
    outputPath = Path("/path/to/output_dataset")
    # Enable balancing, choose method and target, and set a reproducible seed.
    handler.Prepare(
      outputPath,
      valSplit=0.1,
      testSplit=0.1,
      balance=True,
      balanceMethod="augmentation",
      balanceTarget="max",
      randomSeed=42
    )

    # Create a dataset configuration JSON file (saved in the output dataset folder).
    configPath = handler.CreateConfigFile(
      outputPath=outputPath / "DatasetConfig.json",
      splits={"train": 0.8, "val": 0.1, "test": 0.1},
      minSamplesPerClass=20,
      description="My Custom Dataset"
    )

    # Validate an existing dataset structure and report issues (legacy list form).
    issues = handler.ValidateDatasetStructure(outputPath, minSamplesPerClass=5)
    if (isinstance(issues, list) and len(issues) > 0):
      print("Dataset validation issues found:")
      for issue in issues:
        print(f" - {issue}")
    else:
      print("Dataset structure is valid (no issues found in legacy list form).")

    # If you want a structured report with per-split/per-class details, request it explicitly:
    structured = handler.ValidateDatasetStructure(outputPath, minSamplesPerClass=5, returnStructured=True)
    # `structured` is a dict: {"splits": {..}, "issues": [...]}
    if (structured.get("issues")):
      print("Structured validation issues:")
      for issue in structured["issues"]:
        print(f" - {issue}")
    else:
      print("Structured validation passed.")
  '''

  def __init__(
    self,
    sourceDir: Path,
    configPath: Path = None,
    imageExtensions: set = None,
    autoDetect: bool = True,
    requiredSplits: set = {"train", "val", "test"},
  ):
    r'''
    Initialize the dataset handler.

    Parameters:
      sourceDir (Path): Path to the dataset root folder to analyze.
      configPath (Path|None): Optional path to a dataset configuration JSON.
      imageExtensions (set|None): Optional iterable of image file extensions to consider
                                  (for example {".jpg", ".png"}). If None, uses the
                                  class default set.
      autoDetect (bool): If True (default) auto-detect dataset structure during init.
                         If False, skip auto-detection so the instance can be configured
                         manually (useful in tests and programmatic workflows).
      requiredSplits (set): Set of required split folder names to check for when
                            detecting pre-split structure. Defaults to {"train", "val", "test"}.

    Raises:
      ValueError: If the provided sourceDir does not exist.
    '''

    self.sourceDir = Path(sourceDir)
    self.configPath = configPath
    self.config = None
    self.classMapping = None
    self.requiredSplits = requiredSplits

    # Configure image extensions; normalize to lower-case with leading dot.
    if (imageExtensions is None):
      # normalize default class-level extensions to a canonical lower-case set
      self.imageExtensions = {
        (ext.lower() if (ext.startswith(".")) else f".{ext.lower()}")
        for ext in IMAGE_SUFFIXES
      }
    else:
      self.imageExtensions = {
        (ext.lower() if (str(ext).startswith(".")) else f".{str(ext).lower()}")
        for ext in imageExtensions
      }

    # Validate source directory exists only when auto-detection is requested.
    if (autoDetect):
      if (not self.sourceDir.exists()):
        raise ValueError(f"Dataset directory not found: {self.sourceDir}")
      # Load configuration when autoDetect is True.
      self.LoadDetectConfig()
      print("Dataset handler initialized successfully.", flush=True)
    else:
      # When autoDetect is False, allow non-existent sourceDir so callers can
      # programmatically set up the instance in tests or workflows.
      self.config = None
      self.classMapping = None

  def AutoDetectStructure(self):
    r'''
    Auto-detect dataset structure and populate the internal config.

    The method attempts three strategies:
      Strategy 1: Pre-split layout (highest priority).
      Strategy 2: Look for nested class folders (generic).
      Strategy 3: Look for flat structure with labels in filename (generic).

    Raises:
      ValueError: If neither strategy detects a valid multi-class layout.
    '''

    print("Scanning dataset directory...", flush=True)

    # Strategy 1: Pre-split layout (highest priority).
    if (self.DetectPreSplitStructure()):
      print("Detected pre-split structure.", flush=True)
      # Infer class mapping from train split (most representative).
      trainPath = self.sourceDir / list(self.requiredSplits)[0]
      classMapping = {}
      for item in trainPath.iterdir():
        if (item.is_dir() and not item.name.startswith(".")):
          images = self.GetImages(item)
          if (images):
            classMapping[item.name] = item.name
      if (len(classMapping) >= 2):
        self.classMapping = classMapping
        self.config = {
          "datasetFormat" : "pre-split",
          "sourceDir"     : str(self.sourceDir),
          "classes"       : list(set(classMapping.values())),
          "classMappings" : classMapping,
          "requiredSplits": list(self.requiredSplits),
        }
        return

    # Strategy 2: Look for nested class folders (generic).
    classMapping = self.DetectNestedStructure()

    if (classMapping):
      print(f"Detected nested structure with {len(classMapping)} classes.", flush=True)
      self.classMapping = classMapping
      self.config = {
        "datasetFormat" : "nested",
        "sourceDir"     : str(self.sourceDir),
        "classes"       : list(set(classMapping.values())),
        "classMappings" : classMapping,
        "requiredSplits": list(self.requiredSplits),
      }
      return

    # Strategy 3: Look for flat structure with labels in filename (generic).
    classMapping = self.DetectFlatStructure()

    if (classMapping):
      print(f"Detected flat structure with {len(classMapping)} classes.", flush=True)
      self.classMapping = classMapping
      self.config = {
        "datasetFormat" : "flat",
        "sourceDir"     : str(self.sourceDir),
        "classes"       : list(set(classMapping.values())),
        "classMappings" : classMapping,
        "requiredSplits": list(self.requiredSplits),
      }
      return

    raise ValueError(
      f"Could not auto-detect dataset structure in {self.sourceDir}\n"
      "Expected structure:\n"
      "  - Pre-split: dataset/train/class1/, dataset/val/class1/, etc.\n"
      "  - Nested: dataset/class1/, dataset/class2/, etc.\n"
      "  - Flat: dataset/class1_image1.jpg, etc.\n"
      "Provide DatasetConfig.json for custom structures."
    )

  def DetectPreSplitStructure(self) -> bool:
    r'''
    Check if the source directory already contains train/val/test subdirectories
    with at least one class folder each.

    Returns:
      bool: True if a valid pre-split structure is detected.
    '''

    existingSplits = {item.name for item in self.sourceDir.iterdir() if (item.is_dir())}

    if (not set(self.requiredSplits).issubset(existingSplits)):
      print("Pre-split structure not detected: missing required splits.", flush=True)
      return False

    # Verify each split has at least one non-hidden class folder with images.
    for split in self.requiredSplits:
      splitPath = self.sourceDir / split
      classDirs = [
        d
        for d in splitPath.iterdir()
        if (d.is_dir() and not d.name.startswith("."))
      ]
      if (not classDirs):
        print(f"Pre-split structure not detected: no class folders in split '{split}'.", flush=True)
        return False
      # Check at least one image in one class.
      hasImage = any(self.GetImages(classDir) for classDir in classDirs)
      if (not hasImage):
        print(f"Pre-split structure not detected: no images found in split '{split}'.", flush=True)
        return False

    return True

  def DetectNestedStructure(self) -> dict:
    r'''
    Detect nested class folder structure dynamically.

    Returns:
      dict: Mapping of folderName->className when more than one class folder is found, otherwise an empty dict.
    '''

    classMapping = {}

    # Scan top-level directories to identify class folders.
    for item in self.sourceDir.iterdir():
      if (not item.is_dir()):
        continue

      # Skip hidden/special folders.
      if (item.name.startswith(".")):
        continue

      # Count images in folder.
      images = self.GetImages(item)

      if (len(images) > 0):
        # Use folder name as class name.
        className = item.name
        classMapping[className] = className
        print(f"  Found class '{className}' with {len(images)} images.", flush=True)

    return classMapping if (len(classMapping) >= 2) else {}

  def DetectFlatStructure(self) -> dict:
    r'''
    Detect flat filename structure where class is encoded as a prefix in the filename
    (for example: class_imagename.jpg).

    Returns:
      dict: Mapping of detected prefix->className or empty dict if detection fails.
    '''

    classMapping = {}
    images = self.GetImages(self.sourceDir)

    if (len(images) == 0):
      return {}

    # Extract class from filename prefix.
    for imagePath in images:
      filename = imagePath.stem
      if ("_" in filename):
        className = filename.split("_")[0]
        if (className not in classMapping):
          classMapping[className] = className

    print(f"  Found {len(classMapping)} classes in flat structure", flush=True)
    return classMapping if (len(classMapping) >= 2) else {}

  def GetImages(self, dirPath: Path) -> list:
    r'''
    Get all image files in the given directory (non-recursive) using the
    handler's allowed extensions.

    Parameters:
      dirPath (Path): Directory to scan for images.

    Returns:
      list[Path]: List of Path objects for detected image files.
    '''

    images = []
    for extension in self.imageExtensions:
      images.extend(dirPath.glob(f"*{extension}"))
      images.extend(dirPath.glob(f"*{extension.upper()}"))
    return list(set(images))  # Remove duplicates.

  def LoadDetectConfig(self):
    r'''
    Load configuration from a JSON file if provided, otherwise auto-detect layout.

    Side effects:
      Sets self.config and self.classMapping accordingly.
    '''

    if (self.configPath and Path(self.configPath).exists()):
      print(f"Loading dataset config from: {self.configPath}", flush=True)
      self.config = ReadJsonFile(self.configPath)
      self.classMapping = self.config.get("classMappings", {})
      self.requiredSplits = set(self.config.get("requiredSplits", []))
      print("Dataset config loaded successfully.", flush=True)
    else:
      print("Auto-detecting dataset structure...", flush=True)
      self.AutoDetectStructure()

      # Calculate the total number of images per class.
      requiredSplits = self.config["requiredSplits"]  # Example: train/val/test.
      format = self.config["datasetFormat"]
      configs = self.config.copy()

      if (format == "pre-split"):
        print("Calculating class distributions for pre-split dataset...", flush=True)

        classCounts = {}
        for split in requiredSplits:
          classCounts[split] = {}
          for folderName, className in self.classMapping.items():
            folder = self.sourceDir / split / folderName
            count = len(self.GetImages(folder)) if (folder.exists()) else 0
            classCounts[split][className] = count
        configs["classCounts"] = classCounts
        # Calculate ratios.
        ratios = {}
        for split in requiredSplits:
          totalImages = sum(classCounts[split].values())
          ratios[split] = {}
          for className, count in classCounts[split].items():
            ratio = (count / totalImages) if (totalImages > 0) else 0.0
            ratios[split][className] = round(ratio, 2)
        configs["classRatios"] = ratios
        splitsCount = {}
        for split in requiredSplits:
          splitsCount[split] = sum(classCounts[split].values())
        configs["totalSamples"] = sum(splitsCount.values())
        configs["splitsCount"] = splitsCount
        splitsRatios = {}
        totalSamples = sum(splitsCount.values())
        for split, count in splitsCount.items():
          ratio = (count / totalSamples) if (totalSamples > 0) else 0.0
          splitsRatios[split] = round(ratio, 2)
        configs["splitsRatios"] = splitsRatios
        # Calculate per class totals across splits.
        totalClassCounts = {}
        for folderName, className in self.classMapping.items():
          totalCount = 0
          for split in requiredSplits:
            totalCount += classCounts[split].get(className, 0)
          totalClassCounts[className] = totalCount
        configs["totalClassCounts"] = totalClassCounts
        # Ratios across total samples.
        totalClassRatios = {}
        for className, count in totalClassCounts.items():
          ratio = (count / totalSamples) if (totalSamples > 0) else 0.0
          totalClassRatios[className] = round(ratio, 2)
        configs["totalClassRatios"] = totalClassRatios
      elif (format in {"nested", "flat"}):
        print("Calculating class distributions for generic dataset...", flush=True)

        classCounts = {}
        ratios = {}
        for folderName, className in self.classMapping.items():
          folder = self.sourceDir / folderName
          count = len(self.GetImages(folder)) if folder.exists() else 0
          classCounts[className] = count
          ratio = (count / sum(classCounts.values())) if (sum(classCounts.values()) > 0) else 0.0
          ratios[className] = round(ratio, 2)
        configs["classCounts"] = classCounts
        configs["classRatios"] = ratios
        totalSamples = sum(classCounts.values())
        configs["totalSamples"] = totalSamples
        splitsRatios = {}
        for split in requiredSplits:
          splitsRatios[split] = 0.0
        configs["splitsRatios"] = splitsRatios
        configs["splitsCount"] = {}
        for split in requiredSplits:
          configs["splitsCount"][split] = 0
        configs["totalClassCounts"] = classCounts
        configs["totalClassRatios"] = ratios
      self.config = configs

  def CreateConfigFile(
    self,
    outputPath: Path = None,
    splits: dict = None,
    minSamplesPerClass: int = None,
    description: str = None
  ) -> Path:
    r'''
    Auto-generate a complete dataset configuration JSON based on detected mapping
    and source data. The file contains class mappings, splits defaults and basic
    metadata including per-class sample counts.

    Parameters:
      outputPath (Path|None): Destination path for the generated JSON. Defaults to <sourceDir>/DatasetConfig.json when None.
      splits (dict|None): Optional custom split fractions to override defaults.
      minSamplesPerClass (int|None): Minimum suggested samples per class.
      description (str|None): Optional human-readable dataset description.

    Returns:
      Path: Path to the written configuration JSON.
    '''

    if (self.classMapping is None):
      raise ValueError("Class mapping not available. Run auto-detection or load a config first.")

    outPath = Path(outputPath) if (outputPath is not None) else (self.sourceDir / "DatasetConfig.json")
    minSamples = minSamplesPerClass if (minSamplesPerClass is not None) else 100
    imageExts = sorted({ext.lower() for ext in self.imageExtensions})

    currentConfigs = self.config
    currentConfigs.update({
      "name"              : description or f"{self.sourceDir.name} Dataset",
      "description"       : description or "Auto-generated dataset configuration for training",
      "splits"            : splits if (splits is not None) else currentConfigs.get("splits", {}),
      "imageExtensions"   : imageExts,
      "minSamplesPerClass": minSamples,
      "metadata"          : {
        "source"  : str(self.sourceDir),
        "modality": "Image",
        "notes"   : "Auto-generated dataset configuration."
      }
    })

    outPath.parent.mkdir(parents=True, exist_ok=True)
    DumpJsonFile(str(outPath), currentConfigs)
    print(f"Dataset config written: {outPath}", flush=True)

    # Update internal state to reflect new config.
    self.config = currentConfigs
    self.configPath = outPath

    return outPath

  def GetConfig(self) -> dict:
    r'''
    Return a compact representation of the detected configuration.

    Returns:
      dict: Configuration dictionary.
    '''

    return self.config if (self.config is not None) else {}

  def PlotClassDistribution(
    self,
    fileName="ClassDistribution.pdf",
    title="Class Distribution",
    save=False,
    display=True,
    fontSize=12,
    dpi=720,
  ):
    r'''
    Plot a bar chart of the class distribution in the dataset.

    This method prefers to use the centralized `PlotClassDistribution` utility from `PlotsHelper` when available,
    which creates a more polished and informative plot. If that utility is not available or fails, it falls back
    to a simpler bar chart implementation.

    Parameters:
      fileName (str): Path to save the plot PDF. Default: "ClassDistribution.pdf".
      title (str): Title of the plot. Default: "Class Distribution".
      save (bool): If True, save the plot to fileName. Default: False.
      display (bool): If True, display the plot interactively. Default: True.
      returnPreds (bool): If True, return the matplotlib figure and axis objects. Default: False.
      fontSize (int): Font size for labels and title. Default: 12.
      dpi (int): DPI for saved figure. Default: 720.
    '''

    # Guard against missing class mapping.
    if (not self.classMapping):
      print("Warning: class mapping is empty or not set; nothing to plot.", flush=True)
      return None

    classNames = self.classMapping.keys()
    counts = self.config.get("totalClassCounts", {})
    counts = [counts.get(className, 0) for className in classNames]

    # Prefer the centralized PlotClassDistribution utility in PlotsHelper when available.
    try:
      # Build an expanded labels array by repeating each class name according to its count.
      labelsExpanded = []
      for name, cnt in zip(classNames, counts):
        labelsExpanded.extend([name] * int(cnt))
      outPath = Path(fileName) if (fileName is not None) else Path("ClassDistribution.pdf")
      PlotClassDistribution(
        labelsExpanded,
        outPath=outPath,
        title=title,
        dpi=dpi,
        exportPng=save,
        save=save,
        display=display,
        colormap="viridis",
        fontSize=fontSize,
      )
    except Exception as e:
      print(
        f"Warning: could not generate class distribution plot with `PlotClassDistribution` utility: {e}",
        flush=True
      )

      # Fallback to the older bar chart utility when the new plot helper fails.
      try:
        PlotBarChart(
          values=counts,
          labels=classNames,
          title=title,
          ylabel="Number of Images",
          save=save,
          fileName=fileName,
          dpi=dpi,
          display=display,
          annotate=True,
          annotateFormat="{:.0f}",  # Format as integer counts.
          fontSize=fontSize,
          rotation=45,
          returnFig=False,
        )
      except Exception as e2:
        print(f"Warning: could not generate class distribution plot: {e} / {e2}", flush=True)

  def PrintSummary(self):
    r'''
    Print a short summary of the dataset: source path, format and per-class counts.
    '''

    print("\n" + "=" * 80, flush=True)
    print("DATASET SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Source Directory: {self.sourceDir}", flush=True)
    print(f"Dataset Format: {self.config.get('datasetFormat', 'unknown')}", flush=True)
    print(f"Classes: {len(set(self.classMapping.values()))}", flush=True)

    for folderName, className in self.classMapping.items():
      imageCount = len(self.GetImages(self.sourceDir / folderName))
      print(f"  - {className}: {imageCount} images", flush=True)

    configs = self.GetConfig()
    for k, v in configs.items():
      if (type(configs[k]) == dict):
        for k2, v2 in configs[k].items():
          print(f"{k}.{k2}: {v2}", flush=True)
      else:
        print(f"{k}: {v}", flush=True)

    print("=" * 80 + "\n", flush=True)

  def ValidateDatasetStructure(self, datasetPath, minSamplesPerClass: int = 1, returnStructured: bool = False):
    r'''
    Quick validation helper for datasets organized with "train", "val", "test"
    top-level folders and class subdirectories.

    Parameters:
      datasetPath (str|Path): Path to dataset root containing train/val/test.
      minSamplesPerClass (int): Minimal images expected per class in a split.
      returnStructured (bool): If True, return a structured dict with per-split
        and per-class details (counts, sample path, readable) in addition to collected issues.
        If False, return only the list of issue strings (legacy behavior).

    Returns:
      list[str] | dict: If `returnStructured` is False returns list[str] of issue messages.
                       If True returns dict with keys:
                         - "splits": {split: {className: {"count": int, "sample": str|None, "readable": bool}}}
                         - "issues": list[str]
    '''

    issues = []
    structured = {"splits": {}, "issues": issues}

    base = Path(datasetPath)
    if (not base.exists()):
      issues.append(f"Dataset path does not exist: {base}")
      return structured if returnStructured else issues

    # Use the handler's configured image extensions (lower-case entries).
    exts = sorted({ext.lower() if (ext.startswith(".")) else f".{ext.lower()}" for ext in self.imageExtensions})

    for split in ("train", "val", "test"):
      splitPath = base / split
      structured["splits"][split] = {}

      if (not splitPath.exists()):
        issues.append(f"Missing split folder: {split}")
        continue

      classDirs = [d for d in splitPath.iterdir() if d.is_dir()]
      if (len(classDirs) == 0):
        issues.append(f"No class subfolders found under {split}")
        continue

      for classDir in classDirs:
        # Count files for known extensions (case-insensitive) without double-counting.
        count = 0
        sample = None
        for entry in classDir.iterdir():
          if (not entry.is_file()):
            continue
          if (entry.suffix.lower()) in exts:
            count += 1
            if sample is None:
              sample = entry

        readable = True
        if (count < minSamplesPerClass):
          issues.append(f"Too few images in {split}/{classDir.name}: {count} (<{minSamplesPerClass})")
          readable = bool(sample is not None)
        else:
          # Try opening one sample to ensure basic readability.
          try:
            if (sample is not None):
              Image.open(sample).verify()
          except Exception:
            readable = False
            issues.append(f"Unreadable image file in {classDir}")

        structured["splits"][split][classDir.name] = {
          "count"   : int(count),
          "sample"  : str(sample) if sample is not None else None,
          "readable": bool(readable),
        }

    return structured if returnStructured else issues

  def BuildManifest(self, outputDir: Path, splitMapping: dict) -> Path:
    r'''
    Build a lightweight JSON manifest describing the produced dataset and
    including SHA256 checksums for files when possible.

    Parameters:
      outputDir (Path): Output dataset directory used as reference for relative paths.
      splitMapping (dict): Mapping of splitName->[file Paths] produced by Prepare*.

    Returns:
      Path: Path to the written DatasetManifest.json file.
    '''

    if (type(outputDir) == str):
      outputDir = Path(outputDir)

    manifest = {
      "built_at"     : time.strftime("%Y-%m-%d %H:%M:%S"),
      "source"       : str(self.sourceDir),
      "config"       : self.config,
      "class_mapping": self.classMapping,
      "splits"       : {},
    }

    for splitName, splitPaths in splitMapping.items():
      entries = []
      for pathItem in splitPaths:
        relativePath = (
          str(pathItem.relative_to(outputDir))
          if (pathItem.is_relative_to(outputDir))
          else str(pathItem)
        )
        try:
          hashValue = hashlib.sha256()
          with open(pathItem, "rb") as file:
            hashValue.update(file.read())
          shaValue = hashValue.hexdigest()
        except Exception:
          shaValue = hashlib.sha256(relativePath.encode("utf-8")).hexdigest()
        entries.append({"file": relativePath, "sha256": shaValue, "class": pathItem.parent.name})
      manifest["splits"][splitName] = {
        "count": len(entries),
        "files": entries,
      }

    totalFiles = sum(v["count"] for v in manifest["splits"].values())
    if (totalFiles > 0):
      manifest["split_ratios"] = {
        k: v["count"] / totalFiles for k, v in manifest["splits"].items()
      }

    manifestPath = outputDir / "DatasetManifest.json"
    try:
      DumpJsonFile(str(manifestPath), manifest)
      print(f"Dataset manifest saved: {manifestPath}", flush=True)
    except Exception as error:
      print(f"Warning: could not write manifest: {error}", flush=True)
    return manifestPath

  def Prepare(
    self,
    outputDir: Path,
    valSplit: float = 0.1,
    testSplit: float = 0.1,
    balance: bool = None,
    balanceMethod: str = "duplication",
    balanceTarget: Union[str, int] = "max",
    randomSeed: int = 42,
  ):
    r'''
    Prepare dataset in a standard train/val/test layout by creating train/val/test
    folders per class and copying files according to the requested splits.

    Parameters:
      outputDir (Path): Destination folder where the train/val/test layout will be created.
      valSplit (float): Fraction of data to use for validation (default 0.1).
      testSplit (float): Fraction of data to use for testing (default 0.1).
      balance (bool|None): If True, apply class balancing after creating splits.
                           Balancing is applied only to the "train" split.
      balanceMethod (str): One of {"duplication", "augmentation"}. Default: "duplication".
      balanceTarget (str|int): Target per-class count. If "max" (default) expand all
                               classes up to the current maximum class size. If an int,
                               expand all classes up to that number.
      randomSeed (int): Random seed for reproducibility.

    Returns:
      Path: Path to the created dataset YAML file.
    '''

    if (self.classMapping is None):
      raise ValueError("Dataset configuration not loaded or detected.")

    if (self.DetectPreSplitStructure()):
      print("Detected pre-existing train/val/test splits. Copying as-is...", flush=True)
      splitMapping = self.PreparePreSplit(outputDir)
    else:
      print("Preparing dataset in standard train/val/test layout...", flush=True)
      trainSplit = 1.0 - valSplit - testSplit
      # Prepare data based on detected format (nested or flat).
      if (self.config.get("datasetFormat") == "nested"):
        splitMapping = self.PrepareNested(outputDir, trainSplit, valSplit, testSplit, randomSeed)
      elif (self.config.get("datasetFormat") == "flat"):
        splitMapping = self.PrepareFlat(outputDir, trainSplit, valSplit, testSplit, randomSeed)
      else:
        raise ValueError(f"Unsupported dataset format: {self.config.get('datasetFormat')}")

    # Create dataset YAML (generic, includes YOLO-compatible keys).
    yamlPath = self.CreateYAML(outputDir)
    if (balance):
      try:
        # Pass random seed to balancing helper for reproducibility.
        self._ApplyBalancing(outputDir, method=balanceMethod, target=balanceTarget, randomSeed=randomSeed)
      except Exception as e:
        print(f"Warning: balancing failed: {e}", flush=True)

    # Build manifest capturing provenance and file hashes.
    self.BuildManifest(outputDir, splitMapping)
    print(f"Dataset preparation complete: {outputDir}", flush=True)

    return yamlPath

  def PreparePreSplit(self, outputDir: Path) -> dict:
    r'''
    Copy an existing train/val/test layout from sourceDir to outputDir.
    Assumes sourceDir/train/, sourceDir/val/, sourceDir/test/ exist and contain class folders.

    Parameters:
      outputDir (Path): Destination base path.

    Returns:
      dict: Mapping of splitName -> list[Path] of copied files.
    '''

    splitMapping = {"train": [], "val": [], "test": []}
    splitCounts = {"train": 0, "val": 0, "test": 0}

    splitDictMappings = {
      "train"     : "train",
      "Train"     : "train",
      "Training"  : "train",
      "val"       : "val",
      "Val"       : "val",
      "validation": "val",
      "Validation": "val",
      "test"      : "test",
      "Test"      : "test",
      "testing"   : "test",
      "Testing"   : "test",
    }

    for split in self.requiredSplits:
      srcSplit = self.sourceDir / split
      dstSplit = outputDir / splitDictMappings[split]

      # Discover class folders dynamically.
      classDirs = [
        d for d in srcSplit.iterdir()
        if (d.is_dir() and not d.name.startswith("."))
      ]

      count = 0
      for classDir in classDirs:
        dstClass = dstSplit / classDir.name
        dstClass.mkdir(parents=True, exist_ok=True)
        images = self.GetImages(classDir)
        for img in images:
          dstFile = dstClass / img.name
          shutil.copy2(img, dstFile)
          splitMapping[splitDictMappings[split]].append(dstFile)
          count += 1
      splitCounts[splitDictMappings[split]] = count
      print(
        f"Copied {len(splitMapping[splitDictMappings[split]])} files for split '{splitDictMappings[split]}'.",
        flush=True
      )

    return splitMapping

  def PrepareNested(
    self,
    outputDir: Path,
    trainSplit: float,
    valSplit: float,
    testSplit: float,
    randomSeed: int = 42,
  ):
    r'''
    Prepare dataset from nested class folder structure by splitting each class
    folder into train/val/test and copying files accordingly.

    Parameters:
      outputDir (Path): Destination base path.
      trainSplit (float): Fraction of non-test data to use for training.
      valSplit (float): Fraction of non-test data to use for validation.
      testSplit (float): Fraction to reserve for test.
      randomSeed (int): Random seed for reproducibility.

    Returns:
      dict: Mapping of splitName->list[Path] of copied files.
    '''

    if (type(outputDir) == str):
      outputDir = Path(outputDir)

    print("Processing nested structure...", flush=True)
    splitMapping = {"train": [], "val": [], "test": []}

    for folderName, className in self.classMapping.items():
      sourceClassDir = self.sourceDir / folderName
      images = self.GetImages(sourceClassDir)

      print(
        f"Processing class '{className}' ({len(images)} images)... "
        f"valSplit={valSplit} testSplit={testSplit}",
        flush=True
      )

      # Split data.
      # Skip test split when testSplit <= 0 to avoid sklearn defaults.
      if (testSplit <= 0):
        trainVal = images
        testItems = []
      else:
        expectedTest = math.ceil(testSplit * len(images))
        if (expectedTest < 1):
          trainVal = images
          testItems = []
        else:
          trainVal, testItems = train_test_split(
            images,
            test_size=testSplit,
            random_state=randomSeed,
          )

      if (len(trainVal) > 0):
        # If no validation split was requested, keep all non-test samples as training.
        if (valSplit <= 0):
          trainItems, valItems = trainVal, []
        else:
          # Compute validation fraction relative to non-test data and split accordingly.
          denom = (valSplit + trainSplit)
          valFraction = (valSplit / denom) if (denom > 0) else 0.0
          trainItems, valItems = train_test_split(
            trainVal,
            test_size=valFraction,
            random_state=randomSeed,
          )
      else:
        trainItems, valItems = [], []

      # Copy files to train/val/test directories.
      for imageItem in trainItems:
        destDir = outputDir / "train" / className
        destDir.mkdir(parents=True, exist_ok=True)
        destination = destDir / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["train"].append(destination)

      for imageItem in valItems:
        destDir = outputDir / "val" / className
        destDir.mkdir(parents=True, exist_ok=True)
        destination = destDir / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["val"].append(destination)

      for imageItem in testItems:
        destDir = outputDir / "test" / className
        destDir.mkdir(parents=True, exist_ok=True)
        destination = destDir / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["test"].append(destination)

      print(f"  Train: {len(trainItems)}, Val: {len(valItems)}, Test: {len(testItems)}", flush=True)

    return splitMapping

  def PrepareFlat(
    self,
    outputDir: Path,
    trainSplit: float,
    valSplit: float,
    testSplit: float,
    randomSeed: int = 42,
  ):
    r'''
    Prepare dataset from a flat filename layout where class is encoded as the
    filename prefix. Groups files by class, splits them and copies to the
    train/val/test layout.

    Parameters:
      outputDir (Path): Destination base path.
      trainSplit (float): Fraction of non-test data to use for training.
      valSplit (float): Fraction of non-test data to use for validation.
      testSplit (float): Fraction to reserve for test.
      randomSeed (int): Random seed for reproducibility.

    Returns:
      dict: Mapping of splitName->list[Path] of copied files.
    '''

    if (type(outputDir) == str):
      outputDir = Path(outputDir)

    print("Processing flat structure...", flush=True)
    splitMapping = {"train": [], "val": [], "test": []}

    images = self.GetImages(self.sourceDir)
    classImages = {}

    # Group images by class (extracted from filename).
    for imagePath in images:
      filename = imagePath.stem
      if ("_" in filename):
        className = filename.split("_")[0]
        if (className not in classImages):
          classImages[className] = []
        classImages[className].append(imagePath)

    # Split and copy for each class.
    for className, imageList in classImages.items():
      mappedClass = self.classMapping.get(className, className)
      print(
        f"Processing class '{mappedClass}' ({len(imageList)} images)... "
        f"valSplit={valSplit} testSplit={testSplit}",
        flush=True
      )

      # Split data. Skip test split when testSplit <= 0 to avoid sklearn defaults.
      if (testSplit <= 0):
        trainVal = imageList
        testItems = []
      else:
        expectedTest = math.ceil(testSplit * len(imageList))
        if (expectedTest < 1):
          trainVal = imageList
          testItems = []
        else:
          trainVal, testItems = train_test_split(
            imageList,
            test_size=testSplit,
            random_state=randomSeed,
          )

      if (len(trainVal) > 0):
        # If no validation split was requested, keep all non-test samples as training.
        if (valSplit <= 0):
          trainItems, valItems = trainVal, []
        else:
          # Compute validation fraction relative to non-test data and split accordingly.
          denom = (valSplit + trainSplit)
          valFraction = (valSplit / denom) if (denom > 0) else 0.0
          trainItems, valItems = train_test_split(
            trainVal,
            test_size=valFraction,
            random_state=randomSeed,
          )
      else:
        trainItems, valItems = [], []

      # Copy files to train/val/test directories.
      for imageItem in trainItems:
        destDir = outputDir / "train" / className
        destDir.mkdir(parents=True, exist_ok=True)
        destination = destDir / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["train"].append(destination)

      for imageItem in valItems:
        destDir = outputDir / "val" / className
        destDir.mkdir(parents=True, exist_ok=True)
        destination = destDir / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["val"].append(destination)

      for imageItem in testItems:
        destDir = outputDir / "test" / className
        destDir.mkdir(parents=True, exist_ok=True)
        destination = destDir / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["test"].append(destination)

      print(f"  Train: {len(trainItems)}, Val: {len(valItems)}, Test: {len(testItems)}", flush=True)

    return splitMapping

  def CreateYAML(self, outputDir: Path) -> Path:
    r'''
    Create a simple dataset YAML usable by common training tools. The file
    includes generic keys (num_classes, class_names) and retains YOLO-compatible
    keys (nc, names) for convenience and backward compatibility.

    Parameters:
      outputDir (Path): Base path where train/val/test folders reside.

    Returns:
      Path: Path to the written Dataset.yaml file.
    '''

    if (type(outputDir) == str):
      outputDir = Path(outputDir)

    classNames = sorted(set(self.classMapping.values()))
    classNameDict = {index: name for index, name in enumerate(classNames)}

    yamlContent = f'''path: {outputDir}
train: train
val: val
test: test
num_classes: {len(classNames)}
class_names: {classNameDict}
# YOLO-compatible keys
nc: {len(classNames)}
names: {classNameDict}
'''

    yamlPath = outputDir / "Dataset.yaml"
    with open(yamlPath, "w") as file:
      file.write(yamlContent)

    print(f"Created Dataset.yaml: {yamlPath}", flush=True)
    return yamlPath

  def _ApplyBalancing(
    self,
    outputDir: Path,
    method: str = "duplication",
    target: Union[str, int] = "max",
    randomSeed: int = 42
  ):
    r'''
    Internal helper to balance class counts inside the "train" split.

    Parameters:
      outputDir (Path): Dataset root containing train/val/test folders.
      method (str): "duplication" or "augmentation".
      target (str|int): "max" to match the largest class, or an integer target count.
      randomSeed (int): Random seed for reproducibility.

    Notes:
      - Balancing is applied only to the train split. Validation and test splits are left unchanged.
      - Duplication simply copies existing images with a suffix to avoid collisions.
      - Augmentation delegates to PerformDataAugmentation from DataAugmentationHelper.
    '''

    method = (method or "duplication").lower()
    trainDir = Path(outputDir) / "train"
    if (not trainDir.exists()):
      print("No train directory found for balancing; skipping.", flush=True)
      return

    # Collect class directories and counts.
    classDirs = [d for d in trainDir.iterdir() if d.is_dir()]
    counts = {d.name: len([f for f in d.iterdir() if f.is_file()]) for d in classDirs}
    if (len(counts) == 0):
      print("No classes found under train/ for balancing; skipping.", flush=True)
      return
    else:
      print(f"Current class counts before balancing: {counts}", flush=True)

    # Determine target count. When target is "max", use counts from the source data
    # so balancing aims to restore representation based on the original dataset.
    maxCount = max(counts.values())
    if (isinstance(target, str) and target == "max"):
      if (self.config.get("datasetFormat") == "pre-split"):
        # Use current train split as reference.
        targetCount = max(counts.values())
      else:
        # Compute source-level class counts depending on dataset format.
        sourceCounts = {}
        if (self.config and self.config.get("datasetFormat") == "nested"):
          for folderName, mappedClass in self.classMapping.items():
            folder = self.sourceDir / folderName
            sourceCounts[mappedClass] = len(self.GetImages(folder)) if (folder.exists()) else 0
        else:
          # Flat format: count by filename prefix in sourceDir.
          images = self.GetImages(self.sourceDir)
          tempCounts = {}
          for img in images:
            fname = img.stem
            if ("_" in fname):
              key = fname.split("_")[0]
              mapped = self.classMapping.get(key, key)
            else:
              mapped = self.classMapping.get(fname, fname)
            tempCounts[mapped] = tempCounts.get(mapped, 0) + 1
          sourceCounts = tempCounts
        targetCount = max(sourceCounts.values()) if (len(sourceCounts) > 0) else maxCount
    else:
      try:
        targetCount = int(target)
      except Exception:
        targetCount = maxCount

    if (targetCount <= 0):
      print("Invalid balance target; skipping.", flush=True)
      return

    for classDir in classDirs:
      filesList = [p for p in classDir.iterdir() if p.is_file()]
      currentCount = len(filesList)
      if (currentCount >= targetCount or currentCount == 0):
        continue

      neededCount = targetCount - currentCount
      print(
        f"Balancing class ({classDir.name}): current={currentCount}, "
        f"target={targetCount}, needed={neededCount}",
        flush=True
      )

      # Ensure reproducibility using provided random seed.
      rng = np.random.default_rng(randomSeed)
      for i in range(neededCount):
        src = rng.choice(filesList)
        destName = f"{src.stem}_bal_{i}{src.suffix}"
        destPath = classDir / destName
        if (method == "duplication"):
          # Perform a simple file copy.
          shutil.copy2(src, destPath)
        else:
          # Augmentation path: delegate to DataAugmentationHelper.PerformDataAugmentation.
          try:
            augConfig = {
              "rotation"  : {"enabled": True, "range": (-15, 15)},
              "flip"      : {"enabled": True, "horizontal": True, "vertical": False},
              "brightness": {"enabled": True, "range": (0.85, 1.15)},
              "contrast"  : {"enabled": True, "range": (0.9, 1.1)},
            }

            augmented = PerformDataAugmentation(
              str(src),
              augConfig,
              numResultantImages=1,
              auxImagesList=None,
              seed=randomSeed
            )
            if (len(augmented) > 0):
              augImg = augmented[0]
              j = 0
              while True:
                destName = f"{src.stem}_bal_{i}_{j}{src.suffix}"
                destPath = classDir / destName
                if (not destPath.exists()):
                  break
                j += 1
              augImg.save(destPath)
            else:
              # Fallback to duplication when augmentation returns nothing.
              destName = f"{src.stem}_bal_{i}{src.suffix}"
              destPath = classDir / destName
              shutil.copy2(src, destPath)
          except Exception:
            # On any failure, fallback to duplication copy.
            destName = f"{src.stem}_bal_{i}{src.suffix}"
            destPath = classDir / destName
            try:
              shutil.copy2(src, destPath)
            except Exception:
              # Last resort: skip this augmentation if copy also fails.
              print(f"Warning: could not create balanced sample for {src}", flush=True)

    print("Balancing complete.", flush=True)


class PyTorchCustomDataset(torch.utils.data.Dataset):
  r'''
  PyTorch dataset for image classification tasks, loading images from a directory
  structure where each class has its own subfolder.

  Parameters:
    dataDir (str): Path to the root directory containing class subfolders with images.
    transform (callable, optional): Optional transform to be applied on a sample.
    allowedExtensions (tuple, optional): Tuple of allowed image file extensions. Defaults to IMAGE_SUFFIXES.
  '''

  def __init__(
    self,
    dataDir,
    transform=None,
    allowedExtensions=tuple(IMAGE_SUFFIXES)
  ):
    r'''
    Initialize the custom dataset for image classification tasks.

    Parameters:
      dataDir (str): Path to the root directory containing class subfolders with images.
      transform (callable, optional): Optional transform to be applied on a sample.
      allowedExtensions (tuple, optional): Tuple of allowed image file extensions. Defaults to IMAGE_SUFFIXES.
    '''

    # Setting data directory.
    self.dataDir = dataDir
    # Setting transform.
    self.transform = transform
    # Getting classes.
    self.classes = sorted(os.listdir(dataDir))
    # Creating class to index mapping.
    self.classToIdx = {}
    # Initializing samples list.
    self.samples = []
    for idx, cls in enumerate(self.classes):
      self.classToIdx[cls] = idx
      clsDir = os.path.join(dataDir, cls)
      if (not os.path.isdir(clsDir)):
        continue
      for fname in os.listdir(clsDir):
        if (fname.lower().endswith(allowedExtensions)):
          # Getting image path.
          path = os.path.join(clsDir, fname)
          self.samples.append((path, idx))

  def __len__(self):
    r'''
    Get the total number of samples in the dataset.

    Returns:
      int: Number of samples in the dataset.
    '''

    return len(self.samples)

  def __getitem__(self, idx):
    r'''
    Retrieve an image and its label by index.

    Parameters:
      idx (int): Index of the sample to retrieve.

    Returns:
      tuple: (image, label) where image is a PIL Image or transformed tensor, and label is an int class index.
    '''

    path, label = self.samples[idx]
    img = Image.open(path).convert("RGB")
    if (self.transform):
      img = self.transform(img)
    return img, label

  def GetClassMapping(self):
    r'''
    Get the mapping of class names to their corresponding indices.

    Returns:
      dict: A dictionary mapping class names to integer indices.
    '''

    return self.classToIdx

  def GetClasses(self):
    r'''
    Get the list of class names in the dataset.

    Returns:
      list: A list of class names.
    '''

    return self.classes

  def GetSamples(self):
    r'''
    Get the list of samples in the dataset, where each sample is a tuple of (image path, class index).

    Returns:
      list: A list of tuples, each containing the image path and its corresponding class index.
    '''

    return self.samples


class PyTorchSegmentationDataset(torch.utils.data.Dataset):
  r'''
  PyTorch Dataset class for image segmentation tasks. Loads images and
  corresponding masks from provided file paths, applies optional transformations,
  encodes masks into class indices, and converts data to tensors.

  Parameters:
    imagePaths (List[str]): List of file paths to input images.
    maskPaths (List[str]): List of file paths to corresponding segmentation masks.
    transforms (Callable|None): Optional callable for data augmentation/transforms.
    imageSize (int): Target size to resize images and masks (square). Default is 256.
    numClasses (int): Number of segmentation classes. Default is 2.
  Notes:
    This dataset implements the standard PyTorch dataset protocol with the
    usual methods such as ``__len__`` and ``__getitem__`` provided on the
    instance. Detailed behaviour and return types for these methods are
    documented in their respective method docstrings below.
  '''

  def __init__(
    self,
    imagePaths: List[str],
    maskPaths: List[str],
    transforms: Optional[Callable] = None,
    imageSize: int = 256,
    numClasses: int = 1
  ):
    r'''
    Initialize the segmentation dataset instance.

    Parameters:
      imagePaths (List[str]): List of image file paths.
      maskPaths (List[str]): List of corresponding mask file paths.
      transforms (Callable|None): Optional augmentation/transform callable that accepts and returns a dict with keys "image" and "mask".
      imageSize (int): Target square size (height,width) to resize images and masks.
      numClasses (int): Number of segmentation classes used for encoding masks.
    '''

    # Store provided paths and transforms on the instance.
    self.imagePaths = imagePaths
    # Store provided mask paths on the instance.
    self.maskPaths = maskPaths
    # Store transforms callable on the instance.
    self.transforms = transforms
    # Store target image size for resizing.
    self.imageSize = imageSize
    # Store number of segmentation classes.
    self.numClasses = numClasses

  def __len__(self) -> int:
    r'''
    Return the number of samples available in the dataset.

    Returns:
      int: Number of image-mask pairs.
    '''

    # Return the length of the image paths list.
    return len(self.imagePaths)

  def LoadImage(self, path):
    r'''
    Load an image from disk and return an RGB numpy array.

    The function first attempts to use OpenCV for performance and falls back
    to PIL if OpenCV fails or is unavailable.

    Parameters:
      path (str|Path): Filesystem path to the image file.

    Returns:
      numpy.ndarray: HxWx3 uint8 RGB image array.
    '''

    # Try importing cv2 for fast image I/O.
    try:
      # Read the image using cv2 in BGR format.
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      # Convert BGR to RGB for consistency.
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # Return the image array.
      return img
    # Fallback to PIL if cv2 is not available.
    except Exception:
      # Open image with PIL and convert to RGB.
      img = Image.open(path).convert("RGB")
      # Convert to numpy array and return.
      return np.array(img)

  def LoadMask(self, path):
    r'''
    Load a segmentation mask from disk as a 2D numpy array.

    Attempts to use OpenCV to read the mask in grayscale. If OpenCV is not
    available, falls back to PIL.

    Parameters:
      path (str|Path): Filesystem path to the mask image.

    Returns:
      numpy.ndarray: HxW integer mask array (single-channel).
    '''

    # Try importing cv2 for mask reading.
    try:
      # Read the mask as grayscale to preserve labels.
      mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      # Return the mask array.
      return mask
    # Fallback to PIL when cv2 is not available.
    except Exception:
      # Open mask and convert to grayscale.
      mask = Image.open(path).convert("L")
      # Convert to numpy array and return.
      return np.array(mask)

  def EncodeMask(self, mask):
    r'''
    Encode a grayscale mask into integer class indices.
    - For binary (numClasses == 1): threshold at 128 → {0, 1}
    - For multi-class (numClasses >= 2): assume pixel values are class IDs; clip to [0, numClasses-1]

    Parameters:
      mask (numpy.ndarray): HxW grayscale mask array.

    Returns:
      numpy.ndarray: (H, W) of dtype int64
    '''

    if (self.numClasses == 1):
      # Binary: foreground = 1, background = 0.
      encoded = (mask > 128).astype(np.int64)
    else:
      # Multi-class: clip to valid range.
      encoded = np.clip(mask, 0, self.numClasses - 1).astype(np.int64)
    return encoded

  # Convert image and mask numpy arrays to torch tensors.
  def ToTensors(self, image, mask):
    r'''
    Convert numpy image and mask arrays to PyTorch tensors.

    The image is normalized to [0,1] float32 and transposed to channel-first
    format (C,H,W). The mask is returned as a torch tensor of shape (H,W).

    Parameters:
      image (numpy.ndarray): HxWx3 image array (uint8 or float).
      mask (numpy.ndarray): HxWx1 integer mask array.

    Returns:
      tuple: (imageTensor, maskTensor) where imageTensor is torch.FloatTensor shape (C,H,W) and maskTensor is torch tensor shape (H,W).
    '''

    # Normalize image to [0,1] and convert to (C,H,W).
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    imageTensor = torch.from_numpy(image)

    # Handle mask based on numClasses.
    if (self.numClasses == 1):
      # Binary: mask should be float32 with 0.0/1.0.
      mask = mask.astype(np.float32)
      maskTensor = torch.from_numpy(mask).unsqueeze(0)  # (H, W) → (1, H, W).
    else:
      # Multi-class: mask should be long (class indices).
      maskTensor = torch.from_numpy(mask.astype(np.int64))

    return imageTensor, maskTensor

  def __getitem__(self, index):
    r'''
    Retrieve the image and mask tensors for a given index.

    The method loads image and mask files, resizes them to the configured
    `imageSize`, applies optional augmentation/transforms (expects a callable
    that accepts and returns a dict with "image" and "mask"), encodes the mask
    into class indices and converts both to torch tensors.

    Parameters:
      index (int): Index of the sample to retrieve.

    Returns:
      tuple: (imageTensor, maskTensor) as returned by `ToTensors`.
    '''

    # Obtain image and mask paths for the given index.
    imgPath = self.imagePaths[index]
    # Obtain the corresponding mask path.
    maskPath = self.maskPaths[index]
    # Load image and mask from disk.
    image = self.LoadImage(imgPath)
    mask = self.LoadMask(maskPath)
    # Resize image using cv2 when available, otherwise use PIL for resizing.
    try:
      # Resize image to target size using bilinear interpolation.
      image = cv2.resize(image, (self.imageSize, self.imageSize), interpolation=cv2.INTER_LINEAR)
      # Resize mask to target size using nearest neighbor interpolation.
      mask = cv2.resize(mask, (self.imageSize, self.imageSize), interpolation=cv2.INTER_NEAREST)
    except Exception:
      # Use PIL for resizing when cv2 is not available.
      image = np.array(Image.fromarray(image).resize((self.imageSize, self.imageSize), resample=Image.BILINEAR))
      mask = np.array(Image.fromarray(mask).resize((self.imageSize, self.imageSize), resample=Image.NEAREST))
    # Apply optional transforms if provided.
    if (self.transforms is not None):
      # Albumentations expects dict with "image" and "mask" keys.
      augmented = self.transforms(image=image, mask=mask)
      # Extract augmented image and mask.
      image = augmented["image"]
      mask = augmented["mask"]
    # Encode the mask into class indices.
    encoded = self.EncodeMask(mask)
    # Convert to tensors and return.
    imageTensor, maskTensor = self.ToTensors(image, encoded)
    # Return the image and mask tensors.
    return imageTensor, maskTensor

  def GetPixelStats(self):
    r'''
    Compute per-channel min, max, mean, and standard deviation across the dataset
    for both images and masks. This can be useful for normalization and analysis.

    Returns:
      dict: Dictionary with keys "image" and "mask", each containing a dict with keys
            "min", "max", "mean", "std" mapping to numpy arrays of shape (3,) for images and (1,) for masks.
    '''

    from tqdm import tqdm

    # Initialize accumulators for statistics.
    pixelSumImages = np.zeros(3, dtype=np.float64)
    pixelSumMasks = np.zeros(3, dtype=np.float64)
    pixelSumSqImages = np.zeros(3, dtype=np.float64)
    pixelSumSqMasks = np.zeros(3, dtype=np.float64)
    pixelMinImages = np.full(3, np.inf, dtype=np.float64)
    pixelMaxImages = np.full(3, -np.inf, dtype=np.float64)
    pixelMinMasks = np.full(3, np.inf, dtype=np.float64)
    pixelMaxMasks = np.full(3, -np.inf, dtype=np.float64)
    totalPixels = 0

    # Iterate over the dataset to accumulate statistics using a progress bar.
    for idx in tqdm(range(len(self)), desc="Computing pixel statistics"):
      # Use __getitem__ to get image and mask tensors.
      imageTensor, maskTensor = self[idx]
      # Convert tensors to numpy arrays.
      image = imageTensor.numpy().transpose(1, 2, 0)  # (H, W, C)
      mask = maskTensor.numpy().transpose(1, 2, 0)  # (H, W, C)
      # Update total pixel count.
      numPixels = image.shape[0] * image.shape[1]
      totalPixels += numPixels
      # Update sums and sums of squares for images.
      pixelSumImages += image.sum(axis=(0, 1))
      pixelSumSqImages += (image ** 2).sum(axis=(0, 1))
      # Update min and max for images.
      pixelMinImages = np.minimum(pixelMinImages, image.min(axis=(0, 1)))
      pixelMaxImages = np.maximum(pixelMaxImages, image.max(axis=(0, 1)))
      # Update sums and sums of squares for masks.
      pixelSumMasks += mask.sum(axis=(0, 1))
      pixelSumSqMasks += (mask ** 2).sum(axis=(0, 1))
      # Update min and max for masks.
      pixelMinMasks = np.minimum(pixelMinMasks, mask.min(axis=(0, 1)))
      pixelMaxMasks = np.maximum(pixelMaxMasks, mask.max(axis=(0, 1)))

    # Compute mean and std for images.
    meanImages = pixelSumImages / totalPixels
    stdImages = np.sqrt((pixelSumSqImages / totalPixels) - (meanImages ** 2))
    # Compute mean and std for masks.
    meanMasks = pixelSumMasks / totalPixels
    stdMasks = np.sqrt((pixelSumSqMasks / totalPixels) - (meanMasks ** 2))

    # Prepare results.
    stats = {
      "images": {
        "mean"       : meanImages.tolist(),
        "std"        : stdImages.tolist(),
        "min"        : pixelMinImages.tolist(),
        "max"        : pixelMaxImages.tolist(),
        "overallMin" : float(np.mean(pixelMinImages)),
        "overallMax" : float(np.mean(pixelMaxImages)),
        "overallMean": float(np.mean(meanImages)),
        "overallStd" : float(np.mean(stdImages))
      },
      "masks" : {
        "mean"       : meanMasks.tolist(),
        "std"        : stdMasks.tolist(),
        "min"        : pixelMinMasks.tolist(),
        "max"        : pixelMaxMasks.tolist(),
        "overallMin" : float(np.mean(pixelMinMasks)),
        "overallMax" : float(np.mean(pixelMaxMasks)),
        "overallMean": float(np.mean(meanMasks)),
        "overallStd" : float(np.mean(stdMasks))
      }
    }

    return stats


# Define the generic dataset class for video classification tasks.
class PyTorchVideoClassificationDataset(torch.utils.data.Dataset):
  r'''
  Generic dataset class for loading video files with class labels.

  Expected directory structure:
    rootDir/
      split/  # e.g., "train", "val", "test"
        class1/
          video1.mp4
          video2.mp4
        class2/
          video3.mp4
          video4.mp4

  Each video file is read and processed into a tensor, and the corresponding class label is
  returned as an integer.
  The dataset supports configurable frame sampling and optional transformations.

  Key features:
    - Configurable root directory and dataset split.
    - Automatic discovery of video files based on class subdirectories.
    - Mapping of class names to numeric labels.
    - Robust video reading with PyAV and safe handling of edge cases (e.g., empty videos).
    - Optional transformation pipeline for data augmentation or preprocessing.
  '''

  # Initialize the dataset with directory paths and processing parameters.
  def __init__(
    self,
    rootDir: str,
    split: str,
    transform: Optional[PyTorchVideoTransforms] = None,
    numFrames: int = None,
    sampleRate: int = None,
    classNames: List[str] = None,
    fileExtensions: Tuple[str, ...] = None
  ) -> None:
    r'''
    Initialize the dataset.

    Parameters:
      rootDir: Root directory containing train/val/test splits.
      split: Dataset split name.
      transform: Optional video transformation pipeline.
      numFrames: Number of frames to sample per video.
      sampleRate: Frame sampling rate.
      classNames: List of expected class directory names.
      fileExtensions: Tuple of supported video file extensions.
    '''

    # Store the root directory path for dataset access.
    self.rootDir = rootDir
    # Store the current dataset split identifier.
    self.split = split
    # Store the optional transformation pipeline reference.
    self.transform = transform
    # Set the target frame count using configuration fallbacks.
    self.numFrames = numFrames
    # Set the frame sampling rate using configuration fallbacks.
    self.sampleRate = sampleRate
    # Set the class names list using configuration fallbacks.
    self.classNames = classNames

    self._ResolveClasses()

    # Set the supported file extensions using default values.
    self.fileExtensions = fileExtensions or tuple(VIDEO_SUFFIXES)
    # Build a dictionary mapping class names to numeric identifiers.
    self.labelToId = {
      labelName: labelIndex
      for labelIndex, labelName in enumerate(self.classNames)
    }
    # Build a dictionary mapping numeric identifiers to class names.
    self.idToLabel = {
      labelIndex: labelName
      for labelName, labelIndex in self.labelToId.items()
    }
    # Initialize the list for storing discovered video file paths.
    self.videoPaths = []
    # Initialize the list for storing corresponding numeric labels.
    self.labels = []
    # Execute the directory scanning method to populate sample lists.
    self._LoadVideoList()
    # Validate that the dataset discovery process yielded valid samples.
    if (len(self.videoPaths) == 0):
      # Generate a detailed diagnostic error message for debugging.
      self._HandleEmptyDatasetError()

  def _ResolveClasses(self):
    r'''
    Resolve class names from directory structure if not provided.
    This method checks if the classNames list was provided during initialization. If it is None,
    it attempts to discover class names by listing subdirectories within the specified dataset split folder.
    If the split directory does not exist or contains no subdirectories, it raises a ValueError with a clear
    message indicating the issue. This allows for flexibility in dataset organization while ensuring that
    the class names are properly set for subsequent processing steps. By resolving class names dynamically,
    the dataset can adapt to different directory structures without requiring hardcoded class lists, as
    long as the expected folder organization is maintained.
    '''

    if (self.classNames is None):
      # Find the split directory and list subdirectories as class names if not provided.
      splitPath = os.path.join(self.rootDir, self.split)
      if (os.path.exists(splitPath) and os.path.isdir(splitPath)):
        self.classNames = [
          entry for entry in os.listdir(splitPath)
          if os.path.isdir(os.path.join(splitPath, entry))
        ]
      else:
        self.classNames = []
    if (self.classNames is None or len(self.classNames) == 0):
      raise ValueError(
        f"No class directories found in split path '{os.path.join(self.rootDir, self.split)}'. "
        f"Please provide classNames or ensure the directory structure is correct."
      )

  # Scan the filesystem and populate internal lists with video metadata.
  def _LoadVideoList(self) -> None:
    r'''
    Scan directory and build list of video paths with labels.
    This method iterates through the expected class subdirectories within the specified dataset split,
    checks for the presence of valid video files based on configured extensions, and populates the
    internal lists of video paths and corresponding numeric labels. It also handles cases where
    directories may be missing or empty without crashing, allowing for a comprehensive error message
    to be generated later if no valid samples are found.
    '''

    # Construct the absolute path to the current split directory.
    splitPath = os.path.join(self.rootDir, self.split)
    # Iterate through each expected class directory name.
    for className in self.classNames:
      # Construct the absolute path to the specific class subdirectory.
      classDir = os.path.join(splitPath, className)
      # Verify that the class directory actually exists on the filesystem.
      if (not os.path.exists(classDir)):
        continue
      # Retrieve all entries contained within the class directory.
      for fileName in os.listdir(classDir):
        # Check if the current entry matches any supported video extension.
        if (any(fileName.lower().endswith(ext) for ext in self.fileExtensions)):
          # Construct the complete absolute path to the valid video file.
          fullPath = os.path.join(classDir, fileName)
          # Append the valid video path to the internal tracking list.
          self.videoPaths.append(fullPath)
          # Append the corresponding numeric class identifier to the label list.
          self.labels.append(self.labelToId[className])

  # Decode a video file and return a uniformly sampled frame array.
  def _ReadVideo(self, videoPath: str) -> np.ndarray:
    r'''
    Read video frames using PyAV with safe sampling fallbacks.
    This method uses the PyAV library to decode video frames from the specified file path.
    It collects frames into a list, ensuring that it does not exceed the maximum required
    count based on the configured number of frames and sampling rate.
    If no frames are successfully decoded, it returns a placeholder array of zeros.
    When sampling frames, it checks if the total decoded frame count is sufficient to perform
    uniform sampling; if not, it simply returns all available frames without attempting to sample
    beyond the existing count. This approach ensures robustness against videos of varying lengths
    and potential decoding issues.

    Parameters:
      videoPath: Path to the video file to be decoded.
    '''

    # Initialize an empty list to store decoded frame arrays.
    frameList = []
    # Open the video container for sequential frame decoding operations.
    with av.open(videoPath) as container:
      # Iterate through each decoded frame from the video stream.
      for frame in container.decode(video=0):
        # Convert the PyAV frame object to an RGB numpy array.
        imageArray = frame.to_rgb().to_ndarray()
        # Append the converted array to the frame collection list.
        frameList.append(imageArray)
        # Terminate the decoding loop once the maximum required count is reached.
        if (len(frameList) >= self.numFrames * self.sampleRate):
          break
    # Generate a placeholder array if the video decoding yielded zero frames.
    if (len(frameList) == 0):
      return np.zeros((self.numFrames, 224, 224, 3), dtype=np.uint8)
    # Calculate sampling indices safely when video length exceeds frame requirement.
    if (len(frameList) >= self.numFrames):
      sampledIndices = np.linspace(0, len(frameList) - 1, self.numFrames, dtype=int)
    else:
      sampledIndices = np.arange(len(frameList))
    # Extract frames from the collection using the computed index array.
    selectedFrames = [frameList[i] for i in sampledIndices]
    # Stack the selected frame arrays into a single unified tensor array.
    return np.stack(selectedFrames, axis=0)

  # Return the total count of available video samples in the dataset.
  def __len__(self) -> int:
    r'''
    Return total number of video samples.
    This method simply returns the length of the internal list that tracks all discovered video file paths,
    which corresponds to the total number of valid samples available for loading and processing.
    It relies on the assumption that the _LoadVideoList method has been executed successfully during
    initialization to populate this list with valid entries. If no valid videos were found, this method
    would return zero, which is handled by the error checking logic in the constructor.
    '''

    # Calculate and return the length of the stored video path list.
    return len(self.videoPaths)

  # Retrieve and process a single video sample by its integer index.
  def __getitem__(self, sampleIndex: int) -> dict:
    r'''
    Load and preprocess a single video sample.
    This method retrieves the file path and corresponding label for the requested sample index,
    decodes the video into a sequence of frames, applies any specified transformations, and constructs
    a standardized output dictionary containing the processed video tensor, numeric label, and
    original file path.
    It ensures that the video data is returned in a format suitable for model ingestion, with pixel
    values normalized and dimensions ordered correctly.
    The method also handles the case where no transformations
    are provided by applying a default conversion and normalization process to the raw video array.

    Parameters:
      sampleIndex: Integer index of the video sample to retrieve.
    '''

    # Retrieve the file system path for the requested sample index.
    currentPath = self.videoPaths[sampleIndex]
    # Retrieve the numeric class label for the requested sample index.
    currentLabel = self.labels[sampleIndex]
    # Decode the target video file into a sequence of frame arrays.
    videoArray = self._ReadVideo(currentPath)
    # Apply the external transformation pipeline if one was provided.
    if (self.transform is not None):
      processedTensor = self.transform(videoArray)
    else:
      # Convert the raw array to a floating point tensor and scale pixel intensity.
      processedTensor = torch.from_numpy(videoArray).float() / 255.0
      # Reorder the tensor dimensions to match the expected channel-first layout.
      processedTensor = processedTensor.permute(3, 0, 1, 2)
    # Construct and return the standardized output dictionary.
    return {
      # Store the processed video tensor for model ingestion.
      "PixelValues": processedTensor,
      # Store the numeric label wrapped in a PyTorch tensor.
      "Labels"     : torch.tensor(currentLabel, dtype=torch.long),
      # Store the original file path for debugging or logging purposes.
      "VideoPath"  : currentPath
    }

  # Generate a comprehensive error message when dataset discovery fails.
  def _HandleEmptyDatasetError(self) -> None:
    r'''
    Raise a diagnostic error when no valid videos are found.
    This method constructs a detailed error message that includes the expected directory structure,
    the results of checks for the existence of class subdirectories, and the count of valid video
    files found in each class folder. The message is designed to guide the user in verifying their dataset
    root path and ensuring that their directory structure matches the expected class names. By providing
    this level of diagnostic information, the method helps users quickly identify and resolve issues related
    to dataset organization, missing folders, or incorrect file naming conventions that may be preventing the
    discovery of valid video samples.
    '''

    # Construct the absolute path to the expected split directory.
    splitPath = os.path.join(self.rootDir, self.split)
    # Initialize a list to accumulate diagnostic status strings.
    diagnosticChecks = []
    # Iterate through each expected class name to verify filesystem structure.
    for className in self.classNames:
      # Construct the absolute path to the specific class folder.
      classDir = os.path.join(splitPath, className)
      # Check whether the directory currently exists on the filesystem.
      directoryExists = os.path.exists(classDir)
      validFileCount = 0
      # Attempt to count valid video files if the directory is present.
      if (directoryExists):
        try:
          # Count entries matching the configured video extensions.
          validFileCount = len([
            fileEntry for fileEntry in os.listdir(classDir)
            if (any(fileEntry.lower().endswith(ext) for ext in self.fileExtensions))
          ])
        except Exception:
          # Assign zero to the count if filesystem access fails.
          validFileCount = 0
      # Format the diagnostic status for the current class directory.
      diagnosticChecks.append(f"{className}: exists={directoryExists}, files={validFileCount}")
    # Construct the complete error message with embedded diagnostics.
    errorMessage = (
      f"No video files found for split '{self.split}' in '{splitPath}'. "
      f"Checked class folders: {diagnosticChecks}. "
      f"Please verify the dataset root path and ensure your directory structure matches the "
      f"expected class names: {self.classNames}"
    )
    # Raise a value error containing the constructed diagnostic message.
    raise ValueError(errorMessage)


def CreateSegmentationDataLoaders(
  dataDir: str,
  imageSize: int = 256,
  batchSize: int = 8,
  numWorkers: int = 4,
  numClasses: int = 1,
  pinMemory: bool = False,
  imagesFolderName: str = "Images",
  masksFolderName: str = "Masks"
):
  r'''
  Create PyTorch DataLoader objects for segmentation tasks from a directory
  that contains either an `Images/` and `Masks/` folder pair or a looser
  tree where image/mask pairs are discovered recursively.

  The function will look for `Images/` and `Masks/` under `dataDir`. When the
  standard layout is missing it will attempt to gather image-mask pairs using
  `GatherSegmentationPairs` (non-recursive pairing by basename matching).

  Parameters:
    dataDir (str): Root path to the dataset containing images and masks.
    imageSize (int): Target square size to resize images and masks. Default 256.
    batchSize (int): Batch size for the returned DataLoaders. Default 8.
    numWorkers (int): Number of worker processes for the DataLoader. Default 4.
    numClasses (int): Number of segmentation classes (used by `SegmentationDataset`).
    pinMemory (bool): Whether to use pinned memory in DataLoaders for faster GPU transfer.
    imagesFolderName (str): Name of the folder containing input images (default "Images").
    masksFolderName (str): Name of the folder containing segmentation masks (default "Masks").

  Returns:
    tuple: (trainLoader, valLoader) PyTorch DataLoader instances for training and validation.
  '''

  # Build paths for images and masks inside the provided data directory.
  imagesDir = os.path.join(dataDir, imagesFolderName)
  # Build masks directory path.
  masksDir = os.path.join(dataDir, masksFolderName)
  # If standard Images/Masks structure is missing, attempt to gather pairs recursively.
  if (not os.path.isdir(imagesDir)) or (not os.path.isdir(masksDir)):
    # Gather image and mask pairs across the dataset tree.
    imageFiles, maskFiles = GatherSegmentationPairs(dataDir)
  else:
    # Collect image file paths sorted for reproducibility.
    imageFiles = sorted(
      [
        os.path.join(imagesDir, f)
        for f in os.listdir(imagesDir)
        if (f.lower().endswith(tuple(IMAGE_SUFFIXES)))
      ]
    )
    # Collect mask file paths sorted for reproducibility.
    maskFiles = sorted(
      [
        os.path.join(masksDir, f)
        for f in os.listdir(masksDir)
        if (f.lower().endswith(tuple(IMAGE_SUFFIXES)))
      ]
    )
  # Create a simple train/validation split by index.
  split = int(len(imageFiles) * 0.8)
  # Assign train and validation image lists.
  trainImages = imageFiles[:split]
  trainMasks = maskFiles[:split]
  valImages = imageFiles[split:]
  valMasks = maskFiles[split:]
  # Create dataset instances for train and validation.
  trainDataset = PyTorchSegmentationDataset(
    trainImages,
    trainMasks,
    transforms=None,
    imageSize=imageSize,
    numClasses=numClasses
  )
  valDataset = PyTorchSegmentationDataset(
    valImages,
    valMasks,
    transforms=None,
    imageSize=imageSize,
    numClasses=numClasses
  )
  allDataset = PyTorchSegmentationDataset(
    imageFiles,
    maskFiles,
    transforms=None,
    imageSize=imageSize,
    numClasses=numClasses
  )
  # Create dataloaders for train and validation.
  trainLoader = torch.utils.data.DataLoader(
    trainDataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers=numWorkers,
    pin_memory=pinMemory,
  )
  valLoader = torch.utils.data.DataLoader(
    valDataset,
    batch_size=batchSize,
    shuffle=False,
    num_workers=numWorkers,
    pin_memory=pinMemory,
  )
  allLoader = torch.utils.data.DataLoader(
    allDataset,
    batch_size=batchSize,
    shuffle=False,
    num_workers=numWorkers,
    pin_memory=pinMemory,
  )
  # Return train and validation dataloaders.
  return trainLoader, valLoader, allLoader


def GatherSegmentationPairs(rootDir: str):
  r'''
  Discover image/mask file pairs by walking a directory tree.

  The function treats directories whose name contains the substring "mask"
  (case-insensitive) as mask folders and everything else as image folders. It
  matches images and masks by basename (ignoring extension) using a best-match
  heuristic (prefix containment) and returns two lists: paired images and
  paired masks in identical order.

  Parameters:
    rootDir (str): Root directory to recursively scan for images and masks.

  Returns:
    tuple: (pairedImagePaths, pairedMaskPaths) where each is a list of filesystem paths (strings).
  '''

  # Initialize lists for images and masks.
  imagePaths = []
  maskPaths = []
  # Walk the directory tree to collect all image files and mask files.
  for dirpath, dirnames, filenames in os.walk(rootDir):
    # Normalize the directory name for mask detection.
    lowName = os.path.basename(dirpath).lower()
    # If the directory name suggests it contains masks, collect mask files.
    if ("mask" in lowName):
      for f in filenames:
        # Only consider image-like files.
        if (f.lower().endswith(tuple(IMAGE_SUFFIXES))):
          maskPaths.append(os.path.join(dirpath, f))
    elif ("mask" not in lowName):
      # Otherwise, collect image files as potential images.
      for f in filenames:
        if (f.lower().endswith(tuple(IMAGE_SUFFIXES))):
          imagePaths.append(os.path.join(dirpath, f))

  # Map masks by basename for quick lookup.
  maskMap = {os.path.splitext(os.path.basename(p))[0]: p for p in maskPaths}
  # Build paired lists by matching basenames between images and masks.
  pairedImages = []
  pairedMasks = []
  for img in sorted(imagePaths):
    # Compute basename without extension.
    base = os.path.splitext(os.path.basename(img))[0]
    # If a corresponding mask exists, pair them.
    whichIsClosest = [m for m in list(maskMap.keys()) if (base in m)]
    whichIsClosest = whichIsClosest[0] if (len(whichIsClosest) > 0) else None
    if (whichIsClosest):
      pairedImages.append(img)
      pairedMasks.append(maskMap[whichIsClosest])
  print(f"Found {len(imagePaths)} images and {len(maskPaths)} masks in '{rootDir}'")
  print(f"Paired {len(pairedImages)} image-mask pairs.")
  # Return the paired image and mask file lists.
  return pairedImages, pairedMasks


# Define a function to discover CSV files in a directory.
def DiscoverCsvFiles(dataDir: str) -> List[str]:
  r'''
  Recursively discover CSV files in the given directory.

  Parameters:
    dataDir (str): The root directory to search for CSV files.

  Returns:
    List[str]: A list of file paths to discovered CSV files.
  '''

  # Create an empty list to collect CSV file paths.
  csvFiles = []
  # Walk the directory tree to find CSV files.
  for root, _, files in os.walk(dataDir):
    for file in files:
      # Check if the file name ends with .csv.
      if (file.lower().endswith(".csv")):
        # Append the absolute path of the CSV file to the list.
        csvFiles.append(os.path.join(root, file))
  # Return the list of discovered CSV files.
  return csvFiles


def SanitizeArray(inputData):
  r'''
  Replace infinite values and fill NaNs in array-like inputs.

  - If inputData is a pandas DataFrame, replace +/-inf with NaN and fill numeric columns with
    the column mean (or 0 if mean is not finite). Non-numeric columns are forward/back
    filled where possible.
  - If inputData is a numpy array or other sequence, convert to float64, replace non-finite
    entries with NaN and fill each column with the column mean (or 0 if no finite vals).

  Returns the same type as input where practical (DataFrame -> DataFrame, ndarray -> ndarray).
  Note: List inputs will be converted to numpy.ndarray.

  Parameters:
    inputData (array-like): Input data to sanitize. It can be a pandas DataFrame, a numpy array, or a list of lists.

  Returns:
    array-like: Sanitized data with infinities replaced and NaNs filled.
  '''

  import numpy as np
  import pandas as pd

  # Check if the input data is None.
  if (inputData is None):
    # Return None immediately if input is None.
    return inputData

  # Check if the input data is a pandas DataFrame.
  if (isinstance(inputData, pd.DataFrame)):
    # Create a copy of the DataFrame to avoid modifying the original.
    inputData = inputData.copy()
    # Replace infinity values with NaN in the DataFrame.
    inputData.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Check if there are any null values in the DataFrame.
    if (inputData.isnull().values.any()):
      # Iterate over each column in the DataFrame.
      for col in inputData.columns:
        # Attempt to process the column data.
        try:
          # Check if the column data type is numeric.
          if (np.issubdtype(inputData[col].dtype, np.number)):
            # Calculate the mean of the column skipping NaN values.
            colMean = inputData[col].mean(skipna=True)
            # Check if the mean is None or not finite.
            if ((colMean is None) or (not np.isfinite(colMean))):
              # Fill NaN values with 0.0 if mean is invalid.
              inputData[col].fillna(0.0, inplace=True)
            else:
              # Fill NaN values with the calculated mean.
              inputData[col].fillna(colMean, inplace=True)
          else:
            # Handle non-numeric columns by forward and backward filling.
            inputData[col] = inputData[col].ffill().bfill()
            # Fill remaining NaN values with an empty string.
            inputData[col].fillna("", inplace=True)
        # Catch any exceptions during column processing.
        except Exception:
          # Fill NaN values with 0.0 as a last resort.
          inputData[col].fillna(0.0, inplace=True)
    # Return the sanitized DataFrame.
    return inputData

  # Attempt to convert the input to a numpy array of float64.
  try:
    # Convert input data to a numpy array.
    arr = np.array(inputData, dtype=np.float64)
  # Catch exceptions if conversion fails.
  except (ValueError, TypeError):
    # Return the original input if conversion fails.
    return inputData

  # Replace non-finite values with NaN in the numpy array.
  arr[~np.isfinite(arr)] = np.nan

  # Check if the array is one-dimensional.
  if (arr.ndim == 1):
    # Check if there are any NaN values in the array.
    if (np.isnan(arr).any()):
      # Extract finite values from the array.
      finite = arr[np.isfinite(arr)]
      # Calculate the fill value based on finite values or default to 0.0.
      fill = np.nanmean(finite) if (finite.size > 0) else 0.0
      # Replace NaN values with the calculated fill value.
      arr[np.isnan(arr)] = fill
    # Return the sanitized one-dimensional array.
    return arr

  # Iterate over each column index in the two-dimensional array.
  for j in range(arr.shape[1]):
    # Extract the current column from the array.
    col = arr[:, j]
    # Check if there are any NaN values in the current column.
    if (np.isnan(col).any()):
      # Extract finite values from the current column.
      finite = col[np.isfinite(col)]
      # Calculate the fill value based on finite values or default to 0.0.
      fill = np.nanmean(finite) if (finite.size > 0) else 0.0
      # Replace NaN values in the column with the calculated fill value.
      col[np.isnan(col)] = fill
  # Return the sanitized two-dimensional array.
  return arr


def ReadAndConcatCsv(files: List[str], nrows: int = None) -> pd.DataFrame:
  r'''
  Read multiple CSV files into pandas DataFrames, add a "SourceFile" column to each, and concatenate them.

  Parameters:
    files (List[str]): List of file paths to CSV files to read and concatenate.
    nrows (int, optional): If provided, limits the number of rows read from each CSV file for testing purposes.
      Defaults to None (read all rows).

  Returns:
    pd.DataFrame: A single DataFrame containing the concatenated data from all CSV files, with an additional "SourceFile" column indicating the origin of each row.
  '''

  # Create an empty list to store individual DataFrames.
  frames = []
  # Iterate over provided files and read each into a DataFrame.
  for file in files:
    print(f"Reading CSV file: {file} with nrows={nrows}")
    # Read the CSV file using pandas, optionally limiting rows.
    df = pd.read_csv(file, nrows=nrows, low_memory=False)
    # Add a SourceFile column to indicate origin of the rows.
    df["SourceFile"] = os.path.basename(file)
    print(f"Read {len(df)} rows and {len(df.columns)} columns from {file}.")
    # Append the DataFrame to the list.
    frames.append(df)
  print(f"Read {len(frames)} DataFrames from CSV files.")
  print(f"Number of rows in each DataFrame: {[len(df) for df in frames]}")
  print(f"Number of columns in each DataFrame: {[len(df.columns) for df in frames]}")
  # Concatenate all DataFrames along rows and reset the index.
  if (len(frames) > 0):
    # Concatenate and return the result.
    return pd.concat(frames, ignore_index=True)
  # Return empty DataFrame if no frames were read.
  return pd.DataFrame()


def LoadAllCsvs(dataDir: str, nrows: int = None) -> pd.DataFrame:
  r'''
  Discover all CSV files in the specified directory, read them into pandas DataFrames, and concatenate
  them into a single DataFrame.

  Parameters:
    dataDir (str): The directory to search for CSV files.
    nrows (int, optional): If provided, limits the number of rows read from each CSV file for testing purposes.
      Defaults to None (read all rows).

  Returns:
    pd.DataFrame: A single DataFrame containing the concatenated data from all discovered CSV files, with an additional "SourceFile" column indicating the origin of each row.
  '''

  # Discover CSV files in the specified directory.
  files = DiscoverCsvFiles(dataDir)
  print(f"Discovered {len(files)} CSV files in {dataDir}.")
  # Read and concatenate all discovered CSV files.
  return ReadAndConcatCsv(files, nrows=nrows)


class TFFolderBasedDataPipeline:
  r"""
  A unified pipeline to handle folder scanning, auto-splitting, and tf.data.Dataset creation.

  This class is designed to simplify the process of creating TensorFlow data pipelines for image
  classification tasks. It automatically checks for the presence of "train", "val", and "test"
  subdirectories, and if they are not found, it attempts to split the dataset based on class subdirectories.
  It then builds tf.data.Dataset pipelines for each split, applying appropriate transformations and
  caching strategies for training and evaluation. The class also calculates and stores the number of samples
  in each split for easy access.

  Expected directory structure:
  Dataset/
  ├── Class_A/
  │   ├── image1.jpg
  │   └── image2.png
  ├── Class_B/
  │   ├── image3.jpg
  │   └── image4.bmp
  └── Class_C/
      └── image5.tiff

  If "train", "val", and "test" folders are not found, it will create them by splitting the class folders
  using an 80/10/10 ratio. The resulting structure will be:
  SplitDataset/
    ├── train/
    │   ├── Class_A/
    │   │   ├── image1.jpg
    │   │   └── image2.png
    │   ├── Class_B/
    │   │   ├── image3.jpg
    │   │   └── image4.bmp
    │   └── Class_C/
    │       └── image5.tiff
    ├── val/
    │   ├── Class_A/
    │   │   ├── image6.jpg
    │   │   └── image7.png
    │   ├── Class_B/
    │   │   ├── image8.jpg
    │   │   └── image9.bmp
    │   └── Class_C/
    │       └── image10.tiff
    └── test/
        ├── Class_A/
        │   ├── image11.jpg
        │   └── image12.png
        ├── Class_B/
        │   ├── image13.jpg
        │   └── image14.bmp
        └── Class_C/
            └── image15.tiff

  .. note:: This class requires the `splitfolders` library for auto-splitting functionality.
    If the library is not installed, the auto-splitting feature will not work, and the class
    will raise an error if it cannot find the expected "train", "val", and "test" directories
    or class subdirectories to split. To install `splitfolders`, you can use pip: `pip install splitfolders`

  .. warning:: The auto-splitting process will create a new directory named "SplitDataset" in the
    parent directory of the original dataset. Ensure that you have write permissions to the parent
    directory and that you do not have an existing "SplitDataset" directory that you do not want to
    be overwritten, as the splitting process will create this directory if it does not already exist.
    Always back up your data before performing operations that modify the filesystem.

  Examples
  --------
  .. code-block:: python

    import matplotlib.pyplot as plt
    from HMB.DatasetsHelper import TFFolderBasedDataPipeline

    # Define the root directory containing the dataset.
    dataDir = "/path/to/your/dataset"
    # Define the batch size for the data pipeline.
    batchSize = 32
    # Define the target image size for resizing.
    imageSize = 224

    # Instantiate the unified data pipeline.
    pipeline = TFFolderBasedDataPipeline(
      dataDir=dataDir,
      batchSize=batchSize,
      imageSize=imageSize
    )

    # Access the training, validation, and test datasets.
    trainDataset = pipeline.train
    valDataset = pipeline.val
    testDataset = pipeline.test

    # Print the dataset lengths and class names for verification.
    print("Dataset Lengths:", pipeline.lengths)
    # Print the discovered class names.
    print("Class Names:", pipeline.classNames)

    # Create a dictionary of available loaders for easy iteration.
    availableLoaders = {
      "Train": pipeline.train,
      "Val"  : pipeline.val,
      "Test" : pipeline.test,
    }
    # Filter out None values for missing splits.
    availableLoaders = {k: v for k, v in availableLoaders.items() if (v is not None)}

    # Print the available loader keys.
    print("Available Loaders:", list(availableLoaders.keys()))
    # Print the sample batch shapes header.
    print("Sample batch shapes:")

    numOfClasses = len(pipeline.classNames)
    print(f"Number of classes: {numOfClasses}")

    # Iterate through the available loaders to print batch shapes.
    for split, loader in availableLoaders.items():
      # Take one batch from the loader.
      for batch in loader.take(1):
        # Unpack the batch into images, labels, and filenames.
        images, labels, filenames = batch
        # Print the shapes for the current split.
        print(f"{split} - Images shape: {images.shape}, Labels shape: {labels.shape}, Filenames shape: {filenames.shape}")

    # Visualize a few samples from the training set to verify augmentations.
    if (pipeline.train is not None):
      # Create a matplotlib figure for the visualization.
      plt.figure(figsize=(12, 6))
      # Take one batch from the training loader.
      for i, batch in enumerate(pipeline.train.take(1)):
        # Unpack the batch.
        images, labels, filenames = batch
        # Iterate through the first 8 images in the batch.
        for j in range(min(8, images.shape[0])):
          # Add a subplot for the current image.
          plt.subplot(2, 4, j + 1)
          # Display the image.
          plt.imshow(images[j].numpy())
          # Set the title with the label and class name.
          plt.title(f"Label: {labels[j].numpy()} - {pipeline.classNames[labels[j].numpy()]}")
          # Turn off the axis.
          plt.axis("off")
      # Set the main title for the figure.
      plt.suptitle("Sample Augmented Training Images")
      # Display the plot.
      plt.show()
  """

  def __init__(
    self,
    dataDir: str,
    batchSize: int = 32,
    imageSize: int = 224,
    ratioTuple: tuple = (0.8, 0.1, 0.1),
  ) -> None:
    r"""
    Initialize the data pipeline. This method sets up the internal state, checks for directory structure,
    prepares splits if necessary, and builds the tf.data.Dataset pipelines for training, validation, and testing.

    The method performs the following steps:
      1. Stores the provided parameters as instance variables.
      2. Calls the _prepareSplits method to check for the presence of "train", "val", and "test" directories and to perform auto-splitting if they are not found.
      3. Calls the _buildIndex method to scan the directories, discover class names, and calculate the number of samples in each split.
      4. Calls the _buildPipeline method three times to create tf.data.Dataset pipelines for the "train", "val", and "test" splits, applying appropriate transformations and caching strategies for each.

    Parameters:
      dataDir (str): The root directory containing the dataset. It should contain "train", "val", and "test" subdirectories, or class subdirectories for auto-splitting.
      batchSize (int): The batch size to use for the tf.data.Dataset pipelines. Default is 32.
      imageSize (int): The target size to which images will be resized. Default is 224 (for 224x224 images).
      ratioTuple (tuple): A tuple specifying the train/val/test split ratios for auto-splitting. Default is (0.8, 0.1, 0.1).
    """

    # Store the root data directory.
    self.dataDir = Path(dataDir)
    # Store the batch size.
    self.batchSize = batchSize
    # Store the target image size.
    self.imageSize = imageSize
    # Store the train/val/test split ratios for auto-splitting.
    self.ratioTuple = ratioTuple
    # Initialize the dictionary for dataset lengths.
    self.lengths = {}
    # Initialize the list for class names.
    self.classNames = []
    # Prepare the directory splits.
    self.rootPath = self._prepareSplits()
    # Build the dataset index to discover classes and calculate lengths.
    self._buildIndex()
    # Build the training pipeline.
    self.train = self._buildPipeline("train", isTraining=True, useCache=True)
    # Build the validation pipeline.
    self.val = self._buildPipeline("val", isTraining=False, useCache=False)
    # Build the test pipeline.
    self.test = self._buildPipeline("test", isTraining=False, useCache=False)

  def _prepareSplits(self) -> Path:
    r"""
    Check for train/val/test folders and auto-split if necessary. This method checks if the expected
    "train", "val", and "test" subdirectories exist in the root dataset directory. If they are not found,
    it looks for class subdirectories to perform an automatic split using the `splitfolders` library.
    The method creates a new "SplitDataset" directory in the parent directory of the original dataset
    if auto-splitting is performed. It returns the path to the directory that contains the "train",
    "val", and "test" subdirectories, which will be used for building the tf.data.Dataset pipelines.

    Returns:
      pathlib.Path: The path to the directory containing the "train", "val", and "test" subdirectories, either the original root path or the new split directory if auto-splitting was performed.
    """

    # Define the root path.
    rootPath = self.dataDir
    # Check if the train directory exists.
    if ((rootPath / "train").exists()):
      # Return the original root path.
      return rootPath
    # Check if there are class directories in the root path to split.
    classDirs = [
      d for d in rootPath.iterdir()
      if (d.is_dir() and d.name not in ["train", "val", "test", "SplitDataset"])
    ]
    # Check if class directories were found.
    if (len(classDirs) > 0):
      # Print a message about creating subsets.
      print("Train directory not found. Using split-folders to create train/val/test subsets...")
      # Import the splitfolders module dynamically.
      import splitfolders
      # Define the output directory for the split dataset.
      rootParent = rootPath.parent
      # Define the split output directory path.
      splitOutputDir = rootParent / "SplitDataset"
      # Check if the split directory does not exist.
      if (not splitOutputDir.exists()):
        # Use splitfolders to create the train/val/test subsets.
        splitfolders.ratio(
          input=str(rootPath),
          output=str(splitOutputDir),
          seed=1337,
          ratio=self.ratioTuple,
        )
      # Return the new split directory path.
      return splitOutputDir
    else:
      # Raise an error if train directory is missing and no class directories are found.
      raise FileNotFoundError("Train directory not found and no class directories found to split.")

  def _buildIndex(self) -> None:
    r"""
    Scan the directories to discover class names and calculate split lengths. This method scans the
    "train" directory to discover class subdirectories, which are assumed to represent different
    classes in the dataset. It stores the class names in a list. Then, it iterates through the "train",
    "val", and "test" splits to count the number of valid image files in each split, storing these
    counts in a dictionary. The method also prints out the discovered class names and the number of
    samples found in each split for verification.
    """

    # Determine the target train directory.
    targetDir = self.rootPath / "train"
    # Get sorted list of class directories.
    classDirs = sorted([d for d in targetDir.iterdir() if d.is_dir()])
    # Store the class names.
    self.classNames = [d.name for d in classDirs]
    # Print the discovered classes.
    print(f"Found {len(classDirs)} classes: {self.classNames}")
    # Iterate through train, val, and test splits to calculate lengths.
    for split in ["train", "val", "test"]:
      # Determine the split directory.
      splitDir = self.rootPath / split
      # Check if the split directory exists.
      if (splitDir.exists()):
        # Initialize the count for the split.
        count = 0
        # Iterate through class directories in the split.
        for classDir in classDirs:
          # Determine the class directory in the split.
          splitClassDir = splitDir / classDir.name
          # Check if the class directory exists in the split.
          if (splitClassDir.exists()):
            # Count the valid image files.
            for imgPath in splitClassDir.rglob("*"):
              # Filter by common image extensions.
              if (imgPath.suffix.lower() in tuple(IMAGE_SUFFIXES)):
                # Increment the count.
                count += 1
        # Store the length for the split.
        self.lengths[split.capitalize()] = count
        # Print the number of samples for the split.
        print(f"{split.capitalize()} samples: {count}")

  def _loadAndProcessImage(
    self,
    imagePath: str,
    isTraining: bool,
    useCache: bool,
    imageCache: dict
  ) -> tf.Tensor:
    r"""
    Load and process a single image. This method attempts to read an image from the given path, decode it,
    and apply transformations. If `isTraining` is True, it applies data augmentations such as random cropping,
    flipping, brightness, and contrast adjustments. If `useCache` is True, it checks if the processed image
    is already in the `imageCache` dictionary to avoid redundant processing. The method also includes error
    handling to catch exceptions that may occur during image loading (e.g., due to corrupt files) and returns
    a dummy tensor in such cases. The processed image tensor is normalized to the range [0, 1] and resized
    to the target image size before being returned.

    Parameters:
      imagePath (str): The filesystem path to the image to be loaded and processed.
      isTraining (bool): A flag indicating whether the image is being processed for training (True) or evaluation (False). This determines whether data augmentations are applied.
      useCache (bool): A flag indicating whether to use the `imageCache` for storing and retrieving processed images to improve performance by avoiding redundant processing of the same image.
      imageCache (dict): A dictionary used for caching processed images. The keys are image paths and the values are the corresponding processed image tensors. This cache is used to speed up loading during training by storing already processed images in memory.

    Returns:
      tf.Tensor: A tensor representing the processed image, normalized to [0, 1] and resized to the target image size. If an error occurs during loading, a dummy tensor of zeros with the appropriate shape is returned instead.
    """

    # Try to load and process the image to handle corrupt files.
    try:
      # Check if caching is enabled and the image is in the cache.
      if (useCache and imagePath in imageCache):
        # Retrieve the cached image tensor.
        img = imageCache[imagePath]
      else:
        # Read the image file from disk.
        imgRaw = tf.io.read_file(imagePath)
        # Decode the image in RGB mode.
        img = tf.image.decode_image(imgRaw, channels=3, expand_animations=False)
        # Convert the image to float32.
        img = tf.cast(img, tf.float32)
        # Normalize the image to [0, 1].
        img = img / 255.0
        # Apply transforms if training.
        if (isTraining):
          # Resize the image slightly larger for random crop.
          img = tf.image.resize(img, [int(self.imageSize * 1.1), int(self.imageSize * 1.1)])
          # Apply random resized cropping.
          img = tf.image.random_crop(img, size=[self.imageSize, self.imageSize, 3])
          # Apply random horizontal flipping.
          img = tf.image.random_flip_left_right(img)
          # Apply random vertical flipping.
          img = tf.image.random_flip_up_down(img)
          # Apply random brightness.
          img = tf.image.random_brightness(img, max_delta=0.2)
          # Apply random contrast.
          img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
          # Clip the image values to [0, 1].
          img = tf.clip_by_value(img, 0.0, 1.0)
        else:
          # Resize the image to the target size.
          img = tf.image.resize(img, [self.imageSize, self.imageSize])
        # Store the image in the cache if caching is enabled.
        if (useCache):
          # Add the image to the cache dictionary.
          imageCache[imagePath] = img
    except Exception as e:
      # Print a warning about the corrupt image.
      print(f"Warning: Could not load image {imagePath}: {e}")
      # Create a dummy zero tensor for the image.
      img = tf.zeros((self.imageSize, self.imageSize, 3), dtype=tf.float32)
    # Return the processed image tensor.
    return img

  def _createGenerator(
    self,
    split: str,
    isTraining: bool,
    useCache: bool
  ):
    r"""
    Create a generator function for tf.data.Dataset.from_generator. This generator function iterates through the
    specified split directory, loads and processes each image, and yields a tuple of (image tensor, label, filename)
    for each sample. The method initializes an image cache dictionary to store processed images if caching is
    enabled. It scans the split directory for class subdirectories, collects valid image paths and their
    corresponding labels based on the class index, and then iterates through these paths to load and process
    each image using the `_loadAndProcessImage` method. The generator yields the processed image tensor, the
    integer label corresponding to the class, and the filename of the image for each sample in the dataset.

    Parameters:
      split (str): The name of the dataset split to create the generator for (e.g., "train", "val", "test"). This determines which subdirectory of the dataset to scan for images and labels.
      isTraining (bool): A flag indicating whether the generator is being created for training (True) or evaluation (False). This determines whether data augmentations will be applied to the images when they are loaded and processed.
      useCache (bool): A flag indicating whether to use an image cache to store processed images. If True, the generator will check if an image has already been processed and stored in the cache before loading and processing it again, which can improve performance during training by avoiding redundant processing of the same images.

    Yields:
      tuple: A tuple containing the processed image tensor (tf.Tensor), the integer label (int) corresponding to the class, and the filename (str) of the image for each sample in the specified split. The image tensor is normalized to [0, 1] and resized to the target image size. The label is an integer index representing the class of the image, and the filename is the name of the image file (without the path) for reference.
    """

    # Initialize the dictionary for image caching.
    imageCache = {}
    # Initialize the list for image paths.
    imagePaths = []
    # Initialize the list for labels.
    labels = []
    # Determine the target split directory.
    targetDir = self.rootPath / split
    # Get sorted list of class directories.
    classDirs = sorted([d for d in targetDir.iterdir() if d.is_dir()])
    # Iterate through each class directory.
    for classIdx, classDir in enumerate(classDirs):
      # Recursively find all image files in the class directory.
      for imgPath in classDir.rglob("*"):
        # Filter by common image extensions.
        if (imgPath.suffix.lower() not in tuple(IMAGE_SUFFIXES)):
          # Skip non-image files.
          continue
        # Add the valid image path.
        imagePaths.append(str(imgPath))
        # Add the corresponding label.
        labels.append(classIdx)
    # Iterate through the dataset indices.
    for idx in range(len(imagePaths)):
      # Load and process the image.
      img = self._loadAndProcessImage(imagePaths[idx], isTraining, useCache, imageCache)
      # Retrieve the corresponding label.
      label = labels[idx]
      # Get the filename of the image.
      filename = Path(imagePaths[idx]).name
      # Yield the image, label, and filename.
      yield img, label, filename

  def _buildPipeline(
    self,
    split: str,
    isTraining: bool,
    useCache: bool
  ) -> Optional[tf.data.Dataset]:
    r"""
    Build the tf.data.Dataset pipeline for a specific split. This method checks if the specified split directory
    exists and, if it does, creates a tf.data.Dataset using the `_createGenerator` method.
    The dataset is configured with the appropriate output signature for images, labels, and filenames. It then
    applies batching and prefetching transformations to optimize performance. If the split directory does not exist,
    the method returns None, indicating that the dataset for that split is not available.

    Parameters:
      split (str): The name of the dataset split to build the pipeline for (e.g., "train", "val", "test"). This determines which subdirectory of the dataset to use for creating the tf.data.Dataset pipeline.
      isTraining (bool): A flag indicating whether the pipeline is being built for training (True) or evaluation (False). This determines whether data augmentations will be applied to the images when they are loaded and processed in the generator function.
      useCache (bool): A flag indicating whether to use an image cache to store processed images. If True, the generator function will check if an image has already been processed and stored in the cache before loading and processing it again, which can improve performance during training by avoiding redundant processing of the same images.
    """

    # Determine the target split directory.
    splitDir = self.rootPath / split
    # Check if the split directory exists.
    if (not splitDir.exists()):
      # Return None for the missing split.
      return None

    # Retrieve the total number of elements for this split from the pre-calculated lengths.
    numElements = self.lengths.get(split.capitalize(), 0)

    # Create the tf.data.Dataset using from_generator.
    dataset = tf.data.Dataset.from_generator(
      lambda: self._createGenerator(split, isTraining, useCache),
      output_signature=(
        tf.TensorSpec(shape=(self.imageSize, self.imageSize, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
      )
    )

    # Explicitly assert the cardinality so model.fit() knows the exact number of steps.
    # This must be done BEFORE batching.
    if (numElements > 0):
      dataset = dataset.apply(tf.data.experimental.assert_cardinality(numElements))

    # Batch the dataset.
    dataset = dataset.batch(self.batchSize)
    # Prefetch the dataset for performance.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # Return the configured dataset.
    return dataset


if __name__ == "__main__":
  # Example usage:
  dataset = PyTorchVideoClassificationDataset(
    rootDir=r"path/to/dataset/root",
    split="train",
    transform=None,
    numFrames=16,
    sampleRate=2
  )
  print(f"Total samples: {len(dataset)}")
  print("Testing sample retrieval...")
  sample = dataset[0]
  print(f"Sample keys: {sample.keys()}")
  print(f"PixelValues shape: {sample['PixelValues'].shape}")
  print(f"Label: {sample['Labels']}")
  print(f"Video path: {sample['VideoPath']}")
  print("`VideoClassificationDataset` test completed successfully.")
