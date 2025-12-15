import unittest
import numpy as np
from HMB.HandCraftedFeatures import (
  CalculateGLCMCooccuranceMatrix,
  CalculateGLCMFeaturesOptimized,
  LocalBinaryPattern2D,
)


class TestHandCraftedFeatures(unittest.TestCase):
  '''
  Unit tests for handcrafted feature extractors on small synthetic images.
  '''

  def setUp(self):
    self.img = (np.random.rand(32, 32) * 255).astype(np.uint8)

  def test_glcm_features(self):
    co = CalculateGLCMCooccuranceMatrix(self.img, d=1, theta=0, isSymmetric=True, isNorm=True)
    feats = CalculateGLCMFeaturesOptimized(co)
    self.assertTrue(isinstance(feats, dict))
    self.assertTrue(len(feats) > 0)

  def test_glcm_multiple_params(self):
    for d in [1, 2, 3]:
      for theta in [0, 45, 90, 135]:
        co = CalculateGLCMCooccuranceMatrix(self.img, d=d, theta=theta, isSymmetric=False, isNorm=False)
        feats = CalculateGLCMFeaturesOptimized(co)
        self.assertTrue(isinstance(feats, dict))

  def test_glcm_invalid_input(self):
    with self.assertRaises(Exception):
      _ = CalculateGLCMCooccuranceMatrix(np.array([1, 2, 3]), d=1, theta=0, isSymmetric=True, isNorm=True)

  def test_lbp_features(self):
    lbp = LocalBinaryPattern2D(
      self.img,
      distance=1,
      theta=135,
      isClockwise=False,
      normalizeLBP=False,
    )
    self.assertTrue(isinstance(lbp, np.ndarray))
    self.assertGreater(lbp.size, 0)

  def test_lbp_param_variations(self):
    for dist in [1, 2]:
      for theta in [0, 90, 180, 270]:
        lbp = LocalBinaryPattern2D(self.img, distance=dist, theta=theta, isClockwise=True, normalizeLBP=True)
        self.assertTrue(isinstance(lbp, np.ndarray))

  def test_histogram_features(self):
    # Simple histogram check similar to ComputeHistogramFeatures
    hist, _ = np.histogram(self.img.flatten(), bins=16, range=(0, 255))
    self.assertEqual(hist.size, 16)


if __name__ == "__main__":
  unittest.main()
