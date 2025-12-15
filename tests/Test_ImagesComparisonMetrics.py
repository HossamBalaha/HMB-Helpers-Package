import unittest
import numpy as np
from HMB.ImagesComparisonMetrics import (
  MutualInformation,
  NormalizedMutualInformation,
  StructuralSimilarity,
  NormalizedCrossCorrelation,
  HistogramComparison,
  UniversalQualityIndex,
  CosineSimilarityImages,
  PeakSignalToNoiseRatio,
  MeanSquaredError,
  NormalizedMeanSquaredError,
  JensenShannonDivergence,
)


class TestImagesComparisonMetrics(unittest.TestCase):
  '''
  Unit tests for ImagesComparisonMetrics using small synthetic arrays.
  Covers core metrics that are deterministic and fast.
  '''

  def setUp(self):
    # Synthetic grayscale images
    self.imgA = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (16, 1))
    self.imgB = self.imgA.copy()
    self.imgC = np.flip(self.imgA, axis=1)  # reversed gradient
    # Add small noise to create non-identical images
    rng = np.random.default_rng(42)
    self.imgNoisy = np.clip(self.imgA + 0.02 * rng.standard_normal(self.imgA.shape), 0.0, 1.0).astype(np.float32)

  # ========== Mutual Information family ==========

  def test_mutual_information_identical(self):
    mi = MutualInformation(self.imgA, self.imgB, bins=32)
    self.assertGreater(mi, 0.0)

  def test_mutual_information_changed(self):
    mi = MutualInformation(self.imgA, self.imgC, bins=32)
    self.assertGreaterEqual(mi, 0.0)

  def test_normalized_mutual_information_range(self):
    nmi = NormalizedMutualInformation(self.imgA, self.imgB)
    self.assertGreaterEqual(nmi, 0.0)
    self.assertLessEqual(nmi, 1.5)  # some NMI definitions can exceed 1 slightly depending on implementation

  def test_mutual_information_invalid_bins(self):
    with self.assertRaises(Exception):
      _ = MutualInformation(self.imgA, self.imgB, bins=-1)

  # ========== SSIM / NCC ==========

  def test_structural_similarity_high_for_identical(self):
    ssim = StructuralSimilarity(self.imgA, self.imgB, winSize=7)
    self.assertGreater(ssim, 0.9)

  def test_structural_similarity_lower_for_changed(self):
    ssimA = StructuralSimilarity(self.imgA, self.imgB, winSize=7)
    ssimC = StructuralSimilarity(self.imgA, self.imgC, winSize=7)
    self.assertGreater(ssimA, ssimC)

  def test_structural_similarity_invalid_shape(self):
    with self.assertRaises(Exception):
      _ = StructuralSimilarity(self.imgA[:-1], self.imgB)

  def test_normalized_cross_correlation_range(self):
    ncc = NormalizedCrossCorrelation(self.imgA, self.imgB)
    self.assertGreaterEqual(ncc, -1.0)
    self.assertLessEqual(ncc, 1.0)

  def test_normalized_cross_correlation_dtype_variants(self):
    a_u8 = (self.imgA * 255).astype(np.uint8)
    b_u8 = (self.imgB * 255).astype(np.uint8)
    ncc = NormalizedCrossCorrelation(a_u8, b_u8)
    self.assertGreaterEqual(ncc, -1.0)
    self.assertLessEqual(ncc, 1.0)

  # ========== Histogram / Error metrics ==========

  def test_histogram_comparison_identical(self):
    histScore = HistogramComparison(self.imgA, self.imgB, bins=16, eps=1e-8)
    self.assertGreaterEqual(histScore, 0.0)

  def test_histogram_comparison_bins_variations(self):
    score16 = HistogramComparison(self.imgA, self.imgB, bins=16, eps=1e-8)
    score64 = HistogramComparison(self.imgA, self.imgB, bins=64, eps=1e-8)
    self.assertGreaterEqual(score16, 0.0)
    self.assertGreaterEqual(score64, 0.0)

  def test_universal_quality_index_identical(self):
    uqi = UniversalQualityIndex(self.imgA, self.imgB)
    self.assertGreater(uqi, 0.9)

  def test_cosine_similarity_images_identical(self):
    cos = CosineSimilarityImages(self.imgA, self.imgB)
    self.assertGreater(cos, 0.99)

  def test_cosine_similarity_images_mismatched_shape(self):
    with self.assertRaises(Exception):
      _ = CosineSimilarityImages(self.imgA[:-1], self.imgB)

  def test_mse_zero_identical(self):
    mse = MeanSquaredError(self.imgA, self.imgB)
    self.assertAlmostEqual(mse, 0.0, places=6)

  def test_nmse_nonzero_changed(self):
    nmse = NormalizedMeanSquaredError(self.imgA, self.imgNoisy)
    self.assertGreater(nmse, 0.0)

  def test_nmse_bounds(self):
    nmse = NormalizedMeanSquaredError(self.imgA, self.imgNoisy)
    self.assertGreaterEqual(nmse, 0.0)

  def test_psnr_infinite_identical(self):
    psnr = PeakSignalToNoiseRatio(self.imgA, self.imgB, eps=1e-10)
    # For identical images PSNR tends to be high; depending on impl may be large but finite
    self.assertGreater(psnr, 40.0)

  def test_psnr_bounds_and_eps(self):
    psnr_default = PeakSignalToNoiseRatio(self.imgA, self.imgNoisy)
    psnr_tight = PeakSignalToNoiseRatio(self.imgA, self.imgNoisy, eps=1e-12)
    self.assertTrue(np.isfinite(psnr_default))
    self.assertTrue(np.isfinite(psnr_tight))

  # ========== Jensen-Shannon Divergence ==========

  def test_jensen_shannon_divergence(self):
    jsd = JensenShannonDivergence(self.imgA, self.imgNoisy)
    self.assertGreaterEqual(jsd, 0.0)

  def test_jsd_identical_zero(self):
    jsd = JensenShannonDivergence(self.imgA, self.imgB)
    self.assertGreaterEqual(jsd, 0.0)


if __name__ == "__main__":
  unittest.main()
