import unittest
import numpy as np
import tempfile
import os
from PIL import Image
from HMB.DataAugmentationHelper import (
  PerformDataAugmentation,
  ApplyAugmentation,
)


class TestDataAugmentationHelper(unittest.TestCase):
  """
  Unit tests for DataAugmentationHelper using PerformDataAugmentation and ApplyAugmentation APIs.
  """

  def setUp(self):
    # Create a temporary test image.
    self.tmpDir = tempfile.mkdtemp()
    self.testImg = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    self.testImgPath = os.path.join(self.tmpDir, "test.png")
    self.testImg.save(self.testImgPath)

  def tearDown(self):
    import shutil
    if os.path.exists(self.tmpDir):
      shutil.rmtree(self.tmpDir, ignore_errors=True)

  def test_perform_augmentation_rotation(self):
    config = {"rotation": {"enabled": True, "range": (-30, 30)}}
    augmented = PerformDataAugmentation(self.testImgPath, config, numResultantImages=2)
    self.assertEqual(len(augmented), 2)
    for img in augmented:
      self.assertIsInstance(img, Image.Image)

  def test_perform_augmentation_flip(self):
    config = {"flip": {"enabled": True, "horizontal": True, "vertical": False}}
    augmented = PerformDataAugmentation(self.testImgPath, config, numResultantImages=2)
    self.assertEqual(len(augmented), 2)

  def test_perform_augmentation_brightness(self):
    config = {"brightness": {"enabled": True, "range": (0.8, 1.2)}}
    augmented = PerformDataAugmentation(self.testImgPath, config, numResultantImages=2)
    self.assertEqual(len(augmented), 2)

  def test_apply_augmentation_rotation(self):
    params = {"range": (-45, 45)}
    augmented = ApplyAugmentation(self.testImg, "rotation", params)
    self.assertIsInstance(augmented, Image.Image)
    self.assertEqual(augmented.size, self.testImg.size)

  def test_no_enabled_augmentations(self):
    config = {"rotation": {"enabled": False}}
    augmented = PerformDataAugmentation(self.testImgPath, config, numResultantImages=2)
    self.assertEqual(len(augmented), 2)
    # Should return copies of original when no augmentation enabled.


if (__name__ == "__main__"):
  unittest.main()
