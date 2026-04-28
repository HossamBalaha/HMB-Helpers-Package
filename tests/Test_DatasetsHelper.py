import unittest, tempfile, shutil, os, json
from PIL import Image
from pathlib import Path
from HMB.DatasetsHelper import GenericImagesDatasetHandler
from HMB.Initializations import IMAGE_SUFFIXES


class TestDatasetsHelper(unittest.TestCase):
  """
  Unit tests for the dataset helper and validator.
  """

  def makeImage(self, path: Path):
    """Create a small valid JPEG image at the requested path.

    Pillow requires a known extension to save, so save as .jpg and rename
    when a different suffix is requested.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = path.with_suffix(".jpg")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(tmpPath)
    if (tmpPath != path):
      tmpPath.replace(path)

  def testValidateMissingPath(self):
    """Validator returns an issue when dataset path does not exist."""
    # Use a clearly non-existent path for the test.
    dsPath = Path("this_path_does_not_exist_12345")
    handlerInstance = GenericImagesDatasetHandler(
      dsPath,
      imageExtensions=IMAGE_SUFFIXES,
      autoDetect=False
    )
    issues = handlerInstance.ValidateDatasetStructure("this_path_does_not_exist")
    self.assertIsInstance(issues, list)
    self.assertTrue(any("does not exist" in i for i in issues))

  def testValidateMissingSplits(self):
    """Validator detects missing split folders."""
    with tempfile.TemporaryDirectory() as tmpDir:
      dsPath = Path(tmpDir) / "ds"
      dsPath.mkdir()
      # Only create train split to simulate missing others.
      (dsPath / "train").mkdir()

      handlerInstance = GenericImagesDatasetHandler(
        dsPath,
        imageExtensions=IMAGE_SUFFIXES,
        autoDetect=False,
      )
      issues = handlerInstance.ValidateDatasetStructure(dsPath)
      self.assertTrue(any("Missing split folder" in i for i in issues))

  def testValidateTooFewAndStructured(self):
    """Validator returns too-few-images and structured report when requested."""
    with tempfile.TemporaryDirectory() as tmpDir:
      dsPath = Path(tmpDir) / "ds"
      # Create nested structure with classes and one sample image.
      for split in ("train", "val", "test"):
        for cls in ("cat", "dog"):
          (dsPath / split / cls).mkdir(parents=True, exist_ok=True)
      sampleImage = dsPath / "train" / "cat" / "img1.jpg"
      self.makeImage(sampleImage)

      handlerInstance = GenericImagesDatasetHandler(
        dsPath,
        imageExtensions=IMAGE_SUFFIXES,
        autoDetect=False
      )
      issues = handlerInstance.ValidateDatasetStructure(dsPath, minSamplesPerClass=2)
      self.assertTrue(any("Too few images" in i for i in issues))

      structured = handlerInstance.ValidateDatasetStructure(
        dsPath, minSamplesPerClass=1, returnStructured=True
      )
      self.assertIsInstance(structured, dict)
      self.assertIn("splits", structured)
      self.assertIn("issues", structured)
      self.assertGreaterEqual(structured["splits"]["train"]["cat"]["count"], 1)
      self.assertTrue(structured["splits"]["train"]["cat"]["readable"])

  def testCustomExtensions(self):
    """Handler honors custom imageExtensions passed via constructor."""
    with tempfile.TemporaryDirectory() as tmpDir:
      dsPath = Path(tmpDir) / "ds"
      (dsPath / "train" / "cls").mkdir(parents=True, exist_ok=True)
      imgCustom = dsPath / "train" / "cls" / "img.custom"
      self.makeImage(imgCustom)

      # Create instance bypassing __init__ to avoid auto-detect side-effects.
      handlerInstance = GenericImagesDatasetHandler(dsPath, imageExtensions={".custom"}, autoDetect=False)
      structured = handlerInstance.ValidateDatasetStructure(
        dsPath, minSamplesPerClass=1, returnStructured=True
      )
      self.assertEqual(structured["splits"]["train"]["cls"]["count"], 1)

  def testUnreadableImage(self):
    """Validator reports unreadable image files."""
    with tempfile.TemporaryDirectory() as tmpDir:
      dsPath = Path(tmpDir) / "ds"
      (dsPath / "train" / "cls").mkdir(parents=True, exist_ok=True)
      badFile = dsPath / "train" / "cls" / "bad.jpg"
      with open(badFile, "wb") as fh:
        fh.write(b"notanimage")

      handlerInstance = GenericImagesDatasetHandler(dsPath, imageExtensions=IMAGE_SUFFIXES,
                                                    autoDetect=False)
      structured = handlerInstance.ValidateDatasetStructure(
        dsPath, minSamplesPerClass=1, returnStructured=True
      )
      self.assertFalse(structured["splits"]["train"]["cls"]["readable"])
      self.assertTrue(any("Unreadable image" in s for s in structured["issues"]))

  def testCreateYAMLCreateConfigAndGetConfig(self):
    """Test CreateYAML, CreateConfigFile and GetConfig functionality."""
    with tempfile.TemporaryDirectory() as tmpDir:
      outPath = Path(tmpDir) / "out"
      outPath.mkdir(parents=True, exist_ok=True)

      # Prepare a minimal handler instance with classMapping and sourceDir.
      handlerInstance = GenericImagesDatasetHandler(outPath,
                                                    imageExtensions=IMAGE_SUFFIXES,
                                                    autoDetect=False)
      handlerInstance.classMapping = {"cat": "cat", "dog": "dog"}
      handlerInstance.config = {"datasetFormat": "nested"}
      handlerInstance.sourceDir = Path(tmpDir) / "src"
      handlerInstance.imageExtensions = IMAGE_SUFFIXES

      yamlPath = handlerInstance.CreateYAML(outPath)
      self.assertTrue(yamlPath.exists())
      # YAML should contain both generic and compatibility keys.
      content = yamlPath.read_text()
      self.assertIn("num_classes", content)
      self.assertIn("nc:", content)

      # Create a DatasetConfig.json and validate GetConfig.
      configPath = handlerInstance.CreateConfigFile(
        outputPath=(outPath / "DatasetConfig.json"),
        splits={"train": 0.7, "val": 0.2, "test": 0.1},
        minSamplesPerClass=1, description="Demo"
      )
      self.assertTrue(configPath.exists())

      cfg = handlerInstance.GetConfig()
      self.assertIn("datasetFormat", cfg)

      # Ensure PrintSummary does not raise.
      handlerInstance.PrintSummary()

  def testPrepareCreatesLayoutAndManifest(self):
    """Prepare creates train/val/test layout and writes manifest and YAML."""
    with tempfile.TemporaryDirectory() as tmpDir:
      src = Path(tmpDir) / "src"
      # create nested source with multiple images per class.
      for cls in ("cat", "dog"):
        (src / cls).mkdir(parents=True, exist_ok=True)
        # create enough images per class to allow splits.
        for i in range(8):
          self.makeImage(src / cls / f"img{i}.jpg")

      # Use real constructor to auto-detect structure.
      handler = GenericImagesDatasetHandler(src)
      outDir = Path(tmpDir) / "out"
      yamlPath = handler.Prepare(outDir, valSplit=0.2, testSplit=0.2)

      # Verify layout and files.
      self.assertTrue((outDir / "train" / "cat").exists())
      self.assertTrue((outDir / "val" / "cat").exists())
      self.assertTrue((outDir / "test" / "cat").exists())
      self.assertTrue((outDir / "Dataset.yaml").exists())
      self.assertTrue((outDir / "DatasetManifest.json").exists())
      manifest = json.load(open(outDir / "DatasetManifest.json", "r"))
      self.assertIn("splits", manifest)

  def testConstructorImageExtensionsNormalization(self):
    """Constructor normalizes imageExtensions to lower-case with leading dot."""
    with tempfile.TemporaryDirectory() as tmpDir:
      src = Path(tmpDir) / "src"
      # Ensure at least two classes exist so auto-detection does not fail.
      (src / "cat").mkdir(parents=True, exist_ok=True)
      (src / "dog").mkdir(parents=True, exist_ok=True)
      self.makeImage(src / "cat" / "img1.jpg")
      self.makeImage(src / "dog" / "img1.jpg")
      handler = GenericImagesDatasetHandler(src, imageExtensions=IMAGE_SUFFIXES,
                                            autoDetect=True)
      self.assertIn(".jpg", handler.imageExtensions)
      self.assertIn(".png", handler.imageExtensions)

  def testGetImagesRespectsExtensions(self):
    """GetImages returns only files matching handler.imageExtensions."""
    with tempfile.TemporaryDirectory() as tmpDir:
      d = Path(tmpDir) / "d"
      d.mkdir()
      (d / "a.JPG").write_text("x")
      (d / "b.png").write_text("x")
      (d / "c.custom").write_text("x")
      handlerInstance = GenericImagesDatasetHandler(d, imageExtensions={".custom", ".jpg"}, autoDetect=False)
      imgs = handlerInstance.GetImages(d)
      # Should include custom and jpg, but not png.
      names = [p.name for p in imgs]
      self.assertTrue(any(n.lower().endswith(".custom") for n in names))
      self.assertTrue(any(n.lower().endswith(".jpg") for n in names))
      self.assertFalse(any(n.lower().endswith(".png") for n in names))


if (__name__ == "__main__"):
  unittest.main()
