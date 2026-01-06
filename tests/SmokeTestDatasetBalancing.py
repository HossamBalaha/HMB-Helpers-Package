import sys
import types
import importlib
from pathlib import Path
from PIL import Image


def makeImage(path: Path) -> None:
  """Create a tiny RGB image at the given path."""
  img = Image.new("RGB", (32, 32), (255, 0, 0))
  path.parent.mkdir(parents=True, exist_ok=True)
  img.save(path)


def insertAugmentationStub() -> None:
  """Insert a lightweight augmentation stub into sys.modules to avoid heavy deps during tests."""
  mod = types.ModuleType("HMB.DataAugmentationHelper")

  def PerformDataAugmentation(imagePath: str, config, numResultantImages: int, auxImagesList=None,
                              extensions=(".png", ".jpg", ".jpeg", ".bmp", ".gif")):
    """Return simple copies of the source image as augmented results."""
    img = Image.open(imagePath).convert("RGB")
    return [img.copy() for _ in range(numResultantImages)]

  mod.PerformDataAugmentation = PerformDataAugmentation
  sys.modules["HMB.DataAugmentationHelper"] = mod


def reloadDatasetsHelper():
  """Reload the datasets helper to pick up the augmentation stub."""
  if ("HMB.DatasetsHelper" in sys.modules):
    importlib.reload(sys.modules["HMB.DatasetsHelper"])
  else:
    # Import the module so it appears in sys.modules and then reload it.
    import HMB.DatasetsHelper
    importlib.reload(sys.modules["HMB.DatasetsHelper"])


def testDuplicationBalancing(tmp_path: Path):
  """Smoke test exercising duplication balancing path."""
  insertAugmentationStub()
  reloadDatasetsHelper()

  from HMB.DatasetsHelper import GenericImagesDatasetHandler

  source = tmp_path / "sourceDataset"
  classA = source / "classA"
  classB = source / "classB"

  # Create imbalanced source where classA has 2 images and classB has 5 images.
  for i in range(2):
    makeImage(classA / f"img{i}.jpg")
  for i in range(5):
    makeImage(classB / f"img{i}.jpg")

  output = tmp_path / "outputDup"

  handler = GenericImagesDatasetHandler(source, autoDetect=True)
  handler.Prepare(output, valSplit=0.0, testSplit=0.0, balance=True, balanceMethod="duplication", balanceTarget="max")

  a_count = len(list((output / "train" / "classA").glob("*.*")))
  b_count = len(list((output / "train" / "classB").glob("*.*")))

  assert (a_count == b_count == 5)


def testAugmentationBalancing(tmp_path: Path):
  """Smoke test exercising augmentation balancing path."""
  insertAugmentationStub()
  reloadDatasetsHelper()

  from HMB.DatasetsHelper import GenericImagesDatasetHandler

  source = tmp_path / "sourceDataset2"
  classA = source / "classA"
  classB = source / "classB"

  # Create imbalanced source where classA has 1 image and classB has 4 images.
  for i in range(1):
    makeImage(classA / f"img{i}.jpg")
  for i in range(4):
    makeImage(classB / f"img{i}.jpg")

  output = tmp_path / "outputAug"

  handler = GenericImagesDatasetHandler(source, autoDetect=True)
  handler.Prepare(output, valSplit=0.0, testSplit=0.0, balance=True, balanceMethod="augmentation", balanceTarget="max")

  a_count = len(list((output / "train" / "classA").glob("*.*")))
  b_count = len(list((output / "train" / "classB").glob("*.*")))

  assert (a_count == b_count == 4)
