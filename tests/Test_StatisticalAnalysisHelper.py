import unittest
import numpy as np
from HMB.StatisticalAnalysisHelper import GeneralStatisticsHelper


class TestStatisticalAnalysisHelper(unittest.TestCase):
  """
  Unit tests for GeneralStatisticsHelper.
  Focus on affine transforms, area/centroid, and chi-squared.
  """

  def setUp(self):
    self.gsh = GeneralStatisticsHelper()

  # ========== Affine Transforms ==========

  def test_affine_covariance_from_raw(self):
    # Raw data: samples along axis 0
    data = np.array([
      [1.0, 2.0],
      [2.0, 3.0],
      [3.0, 4.0],
      [4.0, 5.0],
    ])
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    b = np.array([0.0, 0.0])
    covT = self.gsh.AffineCovariance(data, A, b, isCovariance=False)
    # Check symmetry and shape
    self.assertEqual(covT.shape, (2, 2))
    self.assertTrue(np.allclose(covT, covT.T))

  def test_affine_mean_from_raw(self):
    data = np.array([
      [1.0, 2.0],
      [2.0, 3.0],
      [3.0, 4.0],
      [4.0, 5.0],
    ])
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    b = np.array([1.0, -2.0])
    meanT = self.gsh.AffineMean(data, A, b, isMean=False)
    self.assertEqual(meanT.shape, (2,))

  # ========== Spatial Moments ==========

  def test_area_simple_image(self):
    img = np.ones((5, 5), dtype=np.float32)
    area = self.gsh.Area(img)
    self.assertAlmostEqual(area, 25.0, places=5)

  def test_centroid_centered_square(self):
    img = np.zeros((5, 5), dtype=np.float32)
    img[2, 2] = 1.0
    x, y = self.gsh.Centroid(img)
    # Centroid should be at (2,2)
    self.assertAlmostEqual(x, 2.0, places=5)
    self.assertAlmostEqual(y, 2.0, places=5)

  def test_area_with_zeros_and_negatives(self):
    img = np.zeros((4, 4), dtype=np.float32)
    area_zero = self.gsh.Area(img)
    self.assertAlmostEqual(area_zero, 0.0, places=6)
    img_neg = -np.ones((3, 3), dtype=np.float32)
    area_neg = self.gsh.Area(img_neg)
    self.assertTrue(np.isfinite(area_neg))

  def test_centroid_uniform_image(self):
    img = np.ones((2, 2), dtype=np.float32)
    x, y = self.gsh.Centroid(img)
    self.assertTrue(np.isfinite(x) and np.isfinite(y))

  # ========== Chi-Squared ==========

  def test_chi_squared_no_correction(self):
    X = [0, 0, 1, 1, 1, 2, 2]
    y = [1, 1, 1, 2, 2, 2, 2]
    chi = self.gsh.ChiSquared(X, y, withCorrection=False)
    self.assertTrue(isinstance(chi, (int, float, np.floating)))
    self.assertGreaterEqual(chi, 0.0)

  def test_chi_squared_with_correction(self):
    X = [0, 0, 1, 1, 1, 2, 2]
    y = [1, 1, 1, 2, 2, 2, 2]
    chi = self.gsh.ChiSquared(X, y, withCorrection=True)
    self.assertTrue(isinstance(chi, (int, float, np.floating)))
    self.assertGreaterEqual(chi, 0.0)

  def test_chi_squared_mismatched_lengths(self):
    X = [0, 1]
    y = [1]
    with self.assertRaises(Exception):
      _ = self.gsh.ChiSquared(X, y, withCorrection=False)

  # ========== Descriptive Stats ==========
  def test_columns_rows_mean(self):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colsMean = self.gsh.ColumnsMean(data)
    rowsMean = self.gsh.RowsMean(data)
    # Verify shapes (columns mean -> shape equals number of rows; rows mean -> equals number of columns)
    self.assertEqual(colsMean.shape, (2,))
    self.assertEqual(rowsMean.shape, (3,))
    # Verify aggregate properties
    self.assertAlmostEqual(colsMean.mean(), 3.5, places=5)
    self.assertAlmostEqual(rowsMean.mean(), 3.5, places=5)

  def test_variance_std(self):
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    var = self.gsh.Variance(vec)
    std = self.gsh.StandardDeviation(vec)
    self.assertGreaterEqual(var, 0.0)
    self.assertGreaterEqual(std, 0.0)

  def test_entropy_image(self):
    img = (np.random.rand(16, 16) * 255).astype(np.uint8)
    ent = self.gsh.Entropy(img)
    self.assertTrue(np.isfinite(ent))

  def test_descriptive_stats_with_nans(self):
    data = np.array([[np.nan, 1.0], [2.0, 3.0]])
    colsMean = self.gsh.ColumnsMean(data)
    rowsMean = self.gsh.RowsMean(data)
    self.assertTrue(colsMean.shape[0] == data.shape[0])
    self.assertTrue(rowsMean.shape[0] == data.shape[1])


if (__name__ == "__main__"):
  unittest.main()
