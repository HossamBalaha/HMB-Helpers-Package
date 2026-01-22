import hashlib, time, json, shutil
import numpy as np
import math
from pathlib import Path
from typing import Union
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps, ImageEnhance
from HMB.DataAugmentationHelper import PerformDataAugmentation
from HMB.PlotsHelper import PlotBarChart


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

    # Create a dataset configuration JSON file with custom splits and metadata.
    configPath = handler.CreateConfigFile(
      outputPath=datasetPath / "DatasetConfig.json",
      splits={"train": 0.8, "val": 0.1, "test": 0.1},
      minSamplesPerClass=20,
      description="My Custom Dataset"
    )

    # Validate an existing dataset structure and report issues.
    issues = handler.ValidateDatasetStructure(outputPath, minSamplesPerClass=5)
    if (len(issues) > 0):
      print("Dataset validation issues found:")
      for issue in issues:
        print(f" - {issue}")
    else:
      print("Dataset structure is valid.")
  '''

  imageExtensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".WEBP"
  }

  def __init__(
      self,
      sourceDir: Path,
      configPath: Path = None,
      imageExtensions: set = None,
      autoDetect: bool = True,
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

    Raises:
      ValueError: If the provided sourceDir does not exist.
    '''

    self.sourceDir = Path(sourceDir)
    self.configPath = configPath
    self.config = None
    self.classMapping = None

    # Configure image extensions; normalize to lower-case with leading dot.
    if (imageExtensions is None):
      # normalize default class-level extensions to a canonical lower-case set
      self.imageExtensions = {
        (ext.lower() if (ext.startswith(".")) else f".{ext.lower()}")
        for ext in GenericImagesDatasetHandler.imageExtensions
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
      self.LoadConfig()
    else:
      # When autoDetect is False, allow non-existent sourceDir so callers can
      # programmatically set up the instance in tests or workflows.
      self.config = None
      self.classMapping = None

  def LoadConfig(self):
    r'''
    Load configuration from a JSON file if provided, otherwise auto-detect layout.

    Side effects:
      Sets self.config and self.classMapping accordingly.
    '''

    if (self.configPath and Path(self.configPath).exists()):
      print(f"Loading dataset config from: {self.configPath}", flush=True)
      with open(self.configPath, "r") as file:
        self.config = json.load(file)
      self.classMapping = self.config.get("classMappings", {})
    else:
      print("Auto-detecting dataset structure...", flush=True)
      self.AutoDetectStructure()

  def AutoDetectStructure(self):
    r'''
    Auto-detect dataset structure and populate the internal config.

    The method attempts two strategies:
      1. Nested folders per class (dataset/class1/, dataset/class2/,...)
      2. Flat filenames with class prefix (class_imagename.jpg)

    Raises:
      ValueError: If neither strategy detects a valid multi-class layout.
    '''

    print("Scanning dataset directory...", flush=True)

    # Strategy 1: Look for nested class folders (generic).
    classMapping = self.DetectNestedStructure()

    if (classMapping):
      print(f"Detected nested structure with {len(classMapping)} classes.", flush=True)
      self.classMapping = classMapping
      self.config = {
        "datasetFormat": "nested",
        "classes"      : list(set(classMapping.values())),
        "classMappings": classMapping,
      }
      return

    # Strategy 2: Look for flat structure with labels in filename (generic).
    classMapping = self.DetectFlatStructure()

    if (classMapping):
      print(f"Detected flat structure with {len(classMapping)} classes.", flush=True)
      self.classMapping = classMapping
      self.config = {
        "datasetFormat": "flat",
        "classes"      : list(set(classMapping.values())),
        "classMappings": classMapping,
      }
      return

    raise ValueError(
      f"Could not auto-detect dataset structure in {self.sourceDir}\n"
      "Expected structure:\n"
      "  - Nested: dataset/class1/, dataset/class2/, etc.\n"
      "  - Flat: dataset/class1_image1.jpg, dataset/class2_image1.jpg, etc.\n"
      "Provide datasetConfig.json for custom structures."
    )

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

    manifestPath = outputDir / "DatasetManifest.json"
    try:
      with open(manifestPath, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
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

    print("Preparing dataset in standard train/val/test layout...", flush=True)
    trainSplit = 1.0 - valSplit - testSplit

    # Create output directory structure.
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    for splitName in ["train", "val", "test"]:
      for className in set(self.classMapping.values()):
        (outputDir / splitName / className).mkdir(parents=True, exist_ok=True)

    # Prepare data based on detected format (nested or flat).
    # Initialize splitMapping to ensure defined in all code paths.
    splitMapping = {"train": [], "val": [], "test": []}
    if (self.config.get("datasetFormat") == "nested"):
      splitMapping = self.PrepareNested(outputDir, trainSplit, valSplit, testSplit)
    elif (self.config.get("datasetFormat") == "flat"):
      splitMapping = self.PrepareFlat(outputDir, trainSplit, valSplit, testSplit)

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

  def PrepareNested(self, outputDir: Path, trainSplit: float, valSplit: float, testSplit: float):
    r'''
    Prepare dataset from nested class folder structure by splitting each class
    folder into train/val/test and copying files accordingly.

    Parameters:
      outputDir (Path): Destination base path.
      trainSplit (float): Fraction of non-test data to use for training.
      valSplit (float): Fraction of non-test data to use for validation.
      testSplit (float): Fraction to reserve for test.

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
            random_state=42,
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
            random_state=42,
          )
      else:
        trainItems, valItems = [], []

      # Copy files to train/val/test directories.
      for imageItem in trainItems:
        destination = outputDir / "train" / className / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["train"].append(destination)

      for imageItem in valItems:
        destination = outputDir / "val" / className / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["val"].append(destination)

      for imageItem in testItems:
        destination = outputDir / "test" / className / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["test"].append(destination)

      print(f"  Train: {len(trainItems)}, Val: {len(valItems)}, Test: {len(testItems)}", flush=True)

    return splitMapping

  def PrepareFlat(self, outputDir: Path, trainSplit: float, valSplit: float, testSplit: float):
    r'''
    Prepare dataset from a flat filename layout where class is encoded as the
    filename prefix. Groups files by class, splits them and copies to the
    train/val/test layout.

    Parameters:
      outputDir (Path): Destination base path.
      trainSplit (float): Fraction of non-test data to use for training.
      valSplit (float): Fraction of non-test data to use for validation.
      testSplit (float): Fraction to reserve for test.

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
            random_state=42,
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
            random_state=42,
          )
      else:
        trainItems, valItems = [], []

      # Copy files to train/val/test directories.
      for imageItem in trainItems:
        destination = outputDir / "train" / mappedClass / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["train"].append(destination)

      for imageItem in valItems:
        destination = outputDir / "val" / mappedClass / imageItem.name
        shutil.copy2(imageItem, destination)
        splitMapping["val"].append(destination)

      for imageItem in testItems:
        destination = outputDir / "test" / mappedClass / imageItem.name
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

    # Use the handler"s configured image extensions (lower-case entries).
    exts = sorted({ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in self.imageExtensions})

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

    outPath = Path(outputPath) if outputPath is not None else (self.sourceDir / "DatasetConfig.json")

    # Determine dataset format.
    datasetFormat = (self.config.get("datasetFormat") if self.config else "nested")

    # Classes and mappings.
    classes = sorted(set(self.classMapping.values()))
    classMappings = self.classMapping

    # Splits defaults.
    defaultSplits = {"train": 0.8, "val": 0.1, "test": 0.1}
    if (splits):
      merged = defaultSplits.copy()
      merged.update(splits)
      defaultSplits = merged

    # Image extensions (normalized lower-case list).
    imageExts = sorted({ext.lower() for ext in self.imageExtensions})

    # Compute class distribution and total samples.
    classDistribution = {}
    totalSamples = 0

    if (datasetFormat == "nested"):
      for folderName, mappedClass in self.classMapping.items():
        folder = self.sourceDir / folderName
        count = len(self.GetImages(folder)) if folder.exists() else 0
        classDistribution[mappedClass] = count
        totalSamples += count
    else:
      # Flat structure: count by filename prefix.
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
        totalSamples += 1
      classDistribution = tempCounts

    minSamples = minSamplesPerClass if (minSamplesPerClass is not None) else 10

    metadata = {
      "source"           : str(self.sourceDir),
      "modality"         : "Image",
      "totalSamples"     : totalSamples,
      "classDistribution": classDistribution,
      "notes"            : "Auto-generated dataset configuration."
    }

    configObj = {
      "name"              : description or f"{self.sourceDir.name} Dataset",
      "description"       : description or "Auto-generated dataset configuration for training",
      "datasetFormat"     : datasetFormat,
      "classes"           : classes,
      "classMappings"     : classMappings,
      "splits"            : defaultSplits,
      "imageExtensions"   : imageExts,
      "minSamplesPerClass": minSamples,
      "metadata"          : metadata
    }

    try:
      outPath.parent.mkdir(parents=True, exist_ok=True)
      with open(outPath, "w", encoding="utf-8") as fh:
        json.dump(configObj, fh, indent=2, ensure_ascii=False)
      print(f"Dataset config written: {outPath}", flush=True)

      # Update internal state to reflect new config.
      self.config = configObj
      self.configPath = outPath

      return outPath
    except Exception as e:
      raise RuntimeError(f"Failed to write dataset config to {outPath}: {e}")

  def GetConfig(self) -> dict:
    r'''
    Return a compact representation of the detected configuration.

    Returns:
      dict: Keys include datasetFormat, classes, classMappings and sourceDir.
    '''

    return {
      "datasetFormat": self.config.get("datasetFormat", "unknown"),
      "classes"      : list(set(self.classMapping.values())),
      "classMappings": self.classMapping,
      "sourceDir"    : str(self.sourceDir),
    }

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

    # Build labels and counts from detected mapping.
    classNames = []
    counts = []

    for folderName, className in self.classMapping.items():
      imageCount = len(self.GetImages(self.sourceDir / folderName))
      classNames.append(className)
      counts.append(imageCount)

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
        annotateFormat="{:.0f}", # Format as integer counts.
        fontSize=fontSize,
        rotation=45,
        returnFig=False,
      )
    except Exception as e:
      print(f"Warning: could not generate class distribution plot: {e}", flush=True)

  def CollectSummary(self) -> dict:
    r'''
    Collect a short summary of the dataset: source path, format and per-class counts.

    Returns:
      dict: Summary with keys sourceDir, datasetFormat, numClasses, classCounts.
    '''

    summary = {
      "sourceDir"    : str(self.sourceDir),
      "datasetFormat": self.config.get("datasetFormat", "unknown"),
      "numClasses"   : len(set(self.classMapping.values())),
      "classCounts"  : {},
    }

    for folderName, className in self.classMapping.items():
      imageCount = len(self.GetImages(self.sourceDir / folderName))
      summary["classCounts"][className] = imageCount

    return summary

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

    print("=" * 80 + "\n", flush=True)

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
