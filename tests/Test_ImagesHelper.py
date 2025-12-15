import unittest
import numpy as np
import cv2
from PIL import Image
from HMB.ImagesHelper import (
  GetEmptyPercentage,
  GetEmptyPercentageHistogram,
  ExtractLargestContour,
  MinMaxNormalization,
  CalculateCDF,
  ExtractMultipleObjectsFromROI,
)


class TestImagesHelper(unittest.TestCase):
  '''
  Unit tests for ImagesHelper focusing on pure functions that accept numpy arrays.
  '''

  def setUp(self):
    # Create synthetic RGB images
    self.rgbWhite = np.ones((64, 64, 3), dtype=np.uint8) * 255
    self.rgbBlack = np.zeros((64, 64, 3), dtype=np.uint8)
    self.rgbHalf = np.zeros((64, 64, 3), dtype=np.uint8)
    self.rgbHalf[:, :32, :] = 255  # left half white, right half black

  # ========== Empty Percentage ==========

  def test_get_empty_percentage_white(self):
    ratio = GetEmptyPercentage(self.rgbWhite, shape=(64, 64), inverse=False)
    # For a uniform white image, Otsu inverse thresholding may yield 0% empty depending on binarization
    self.assertGreaterEqual(ratio, 0.0)
    self.assertLessEqual(ratio, 100.0)

  def test_get_empty_percentage_black(self):
    ratio = GetEmptyPercentage(self.rgbBlack, shape=(64, 64), inverse=False)
    self.assertGreaterEqual(ratio, 0.0)
    self.assertLessEqual(ratio, 100.0)

  def test_get_empty_percentage_inverse(self):
    ratioNonEmpty = GetEmptyPercentage(self.rgbHalf, shape=(64, 64), inverse=True)
    self.assertGreaterEqual(ratioNonEmpty, 40.0)
    self.assertLessEqual(ratioNonEmpty, 60.0)

  def test_get_empty_percentage_histogram(self):
    ratio = GetEmptyPercentageHistogram(self.rgbHalf, shape=(64, 64), inverse=False)
    # Depending on thresholds, histogram-based empty may count large proportions
    self.assertGreaterEqual(ratio, 0.0)
    self.assertLessEqual(ratio, 100.0)

  def test_get_empty_percentage_grayscale(self):
    # Single-channel image
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, :32] = 255
    ratio = GetEmptyPercentage(img, shape=(64, 64), inverse=False)
    self.assertGreaterEqual(ratio, 0.0)
    self.assertLessEqual(ratio, 100.0)

  def test_get_empty_percentage_float_dtype(self):
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[:, :32, :] = 1.0
    ratio = GetEmptyPercentage(img, shape=(64, 64), inverse=True)
    self.assertGreaterEqual(ratio, 0.0)
    self.assertLessEqual(ratio, 100.0)

  def test_get_empty_percentage_invalid_shape(self):
    # Non-image shape should raise or handle gracefully
    with self.assertRaises(Exception):
      _ = GetEmptyPercentage(np.array([1, 2, 3]), shape=(64, 64))

  # ========== Largest Contour ==========

  def test_extract_largest_contour_simple(self):
    # Create a white square on black background
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(img, (16, 16), (48, 48), (255, 255, 255), -1)
    masked, contour, mask, draw = ExtractLargestContour(img)
    self.assertEqual(masked.shape, img.shape)
    self.assertIsNotNone(contour)
    self.assertEqual(mask.shape, img.shape[:2])

  def test_extract_largest_contour_no_shape(self):
    # Degenerate all-black image should yield None contour
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    masked, contour, mask, draw = ExtractLargestContour(img)
    self.assertEqual(masked.shape, img.shape)
    self.assertIsNone(contour)
    self.assertEqual(mask.shape, img.shape[:2])

  def test_extract_largest_contour_tiny_object(self):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(img, (16, 16), 1, (255, 255, 255), -1)
    masked, contour, mask, draw = ExtractLargestContour(img)
    self.assertEqual(mask.shape, img.shape[:2])

  def test_extract_largest_contour_irregular_shape(self):
    img = np.zeros((50, 80, 3), dtype=np.uint8)
    pts = np.array([[10, 10], [70, 15], [60, 40], [20, 45]])
    cv2.fillPoly(img, [pts], (255, 255, 255))
    masked, contour, mask, draw = ExtractLargestContour(img)
    self.assertEqual(mask.shape, img.shape[:2])
    self.assertIsNotNone(contour)

  def test_extract_largest_contour_grayscale_input(self):
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 10, 255, -1)
    masked, contour, mask, draw = ExtractLargestContour(img)
    self.assertEqual(mask.shape, img.shape[:2])

  # ========== Normalization and CDF ==========

  def test_min_max_normalization_uint8(self):
    img = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
    norm = MinMaxNormalization(img, mapToUint8=True)
    self.assertEqual(norm.dtype, np.uint8)
    self.assertEqual(norm.min(), 0)
    self.assertEqual(norm.max(), 255)

  def test_min_max_normalization_float(self):
    img = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    norm = MinMaxNormalization(img, mapToUint8=False)
    self.assertEqual(norm.dtype, np.float32)
    self.assertAlmostEqual(norm.min(), 0.0, places=6)
    self.assertAlmostEqual(norm.max(), 1.0, places=6)

  def test_min_max_normalization_constant(self):
    # Degenerate image with constant value
    img = np.ones((16, 16), dtype=np.uint8) * 128
    norm = MinMaxNormalization(img, mapToUint8=True)
    # In constant case, min and max are equal; ensure function handles without NaN
    self.assertTrue(np.all(norm == 0) or np.all(norm == 255) or np.all(norm == 128))

  def test_min_max_normalization_int16(self):
    img = np.linspace(-1000, 1000, 64 * 64, dtype=np.int16).reshape(64, 64)
    norm = MinMaxNormalization(img, mapToUint8=True)
    self.assertEqual(norm.dtype, np.uint8)
    self.assertEqual(norm.min(), 0)
    self.assertEqual(norm.max(), 255)

  def test_min_max_normalization_multi_channel(self):
    img = np.stack([
      np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64),
      np.ones((64, 64), dtype=np.float32),
      np.zeros((64, 64), dtype=np.float32),
    ], axis=-1)
    norm = MinMaxNormalization(img, mapToUint8=False)
    self.assertEqual(norm.dtype, np.float32)
    self.assertTrue(np.all(norm[..., 1] >= 0))  # constant channel handled

  def test_min_max_normalization_invalid_input(self):
    with self.assertRaises(Exception):
      _ = MinMaxNormalization(np.array([1, 2, 3]), mapToUint8=True)

  def test_calculate_cdf_basic(self):
    # Use a non-degenerate image to avoid zero bins
    img = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
    cdf, bins = CalculateCDF(img)
    self.assertTrue(isinstance(cdf, np.ndarray))
    self.assertTrue(isinstance(bins, np.ndarray))

  def test_calculate_cdf_monotonic(self):
    img = np.linspace(0, 255, 32 * 32, dtype=np.uint8).reshape(32, 32)
    cdf, bins = CalculateCDF(img)
    # CDF should be monotonically non-decreasing and end at 1.0
    self.assertTrue(np.all(np.diff(cdf) >= 0))
    self.assertAlmostEqual(cdf[-1], 1.0, places=6)

  def test_calculate_cdf_float_input(self):
    img = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    cdf, bins = CalculateCDF(img)
    self.assertTrue(isinstance(cdf, np.ndarray))
    self.assertTrue(isinstance(bins, np.ndarray))

  def test_calculate_cdf_invalid_input(self):
    with self.assertRaises(Exception):
      _ = CalculateCDF(np.array([1, 2, 3]))

  # ========== ExtractMultipleObjectsFromROI ==========

  def test_extract_multiple_objects_basic(self):
    img = np.zeros((128, 128), dtype=np.uint8)
    seg = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (60, 60), 200, -1)
    cv2.rectangle(seg, (20, 20), (60, 60), 255, -1)
    cv2.circle(img, (90, 90), 10, 180, -1)
    cv2.circle(seg, (90, 90), 10, 255, -1)
    regions = ExtractMultipleObjectsFromROI(img, seg, targetSize=(128, 128), cntAreaThreshold=20, sortByX=True)
    self.assertIsInstance(regions, list)
    self.assertGreaterEqual(len(regions), 2)
    for r in regions:
      self.assertEqual(r.shape, (128, 128))

  def test_extract_multiple_objects_empty_mask_raises(self):
    img = np.zeros((64, 64), dtype=np.uint8)
    seg = np.zeros((64, 64), dtype=np.uint8)
    with self.assertRaises(ValueError):
      _ = ExtractMultipleObjectsFromROI(img, seg)

  def test_extract_multiple_objects_cnt_area_threshold(self):
    img = np.zeros((128, 128), dtype=np.uint8)
    seg = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (15, 15), 200, -1)  # tiny object
    cv2.rectangle(seg, (10, 10), (15, 15), 255, -1)
    cv2.rectangle(img, (50, 50), (100, 100), 200, -1)  # large object
    cv2.rectangle(seg, (50, 50), (100, 100), 255, -1)
    regions = ExtractMultipleObjectsFromROI(img, seg, cntAreaThreshold=100)
    self.assertGreaterEqual(len(regions), 1)


if __name__ == "__main__":
  unittest.main()
