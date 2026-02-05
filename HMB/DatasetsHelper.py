import hashlib, time, json, shutil, os, cv2, torch
import numpy as np
import math
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps, ImageEnhance
from HMB.DataAugmentationHelper import PerformDataAugmentation
from HMB.PlotsHelper import PlotBarChart
from HMB.Initializations import IMAGE_SUFFIXES
from HMB.Utils import DumpJsonFile, ReadJsonFile


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

    # Delegate plotting to the shared PlotBarChart utility to avoid redundancy.
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
    except Exception as e:
      print(f"Warning: could not generate class distribution plot: {e}", flush=True)

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


class SegmentationDataset(torch.utils.data.Dataset):
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

  Methods:
    __len__(): Returns the number of samples in the dataset.
    __getitem__(index): Returns the image and mask tensors for the given index.
  '''

  # Initialize the dataset with lists of image and mask paths, transforms, and target size.
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

  # Return the number of samples in the dataset.
  def __len__(self) -> int:
    r'''
    Return the number of samples available in the dataset.

    Returns:
      int: Number of image-mask pairs.
    '''

    # Return the length of the image paths list.
    return len(self.imagePaths)

  # Load an image from disk and return as a numpy array.
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

  # Load a mask from disk and return as a numpy array.
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

  # Get a single item by index, apply transforms, and return tensors.
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


def CreateSegmentationDataLoaders(
  dataDir: str,
  imageSize: int = 256,
  batchSize: int = 8,
  numWorkers: int = 4,
  numClasses: int = 1
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

  Returns:
    tuple: (trainLoader, valLoader) PyTorch DataLoader instances for training and validation.
  '''

  # Build paths for images and masks inside the provided data directory.
  imagesDir = os.path.join(dataDir, "Images")
  # Build masks directory path.
  masksDir = os.path.join(dataDir, "Masks")
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
  trainDataset = SegmentationDataset(
    trainImages,
    trainMasks,
    transforms=None,
    imageSize=imageSize,
    numClasses=numClasses
  )
  valDataset = SegmentationDataset(
    valImages,
    valMasks,
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
    pin_memory=True
  )
  valLoader = torch.utils.data.DataLoader(
    valDataset,
    batch_size=batchSize,
    shuffle=False,
    num_workers=numWorkers,
    pin_memory=True
  )
  # Return train and validation dataloaders.
  return trainLoader, valLoader


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
