import unittest
import math
from HMB.VotingHelper import VotingHelper


class TestVotingHelper(unittest.TestCase):
  """
  Unit tests for the VotingHelper class.
  Tests cover all voting and aggregation methods including edge cases.
  """

  def setUp(self):
    """Initialize VotingHelper instance before each test."""
    self.vh = VotingHelper()

  # ========== WeightedMajorityVoting Tests. ==========

  def testWeightedMajorityVotingBasic(self):
    """Test weighted majority voting with basic case."""
    predictions = ["cat", "dog", "cat"]
    weights = [0.6, 0.2, 0.2]
    result = self.vh.WeightedMajorityVoting(predictions, weights)
    self.assertEqual(result, "cat")

  def testWeightedMajorityVotingTieBreaker(self):
    """Test weighted majority voting where weight determines winner."""
    predictions = ["A", "B"]
    weights = [1.0, 2.0]
    result = self.vh.WeightedMajorityVoting(predictions, weights)
    self.assertEqual(result, "B")

  def testWeightedMajorityVotingNumericLabels(self):
    """Test weighted majority voting with numeric labels."""
    predictions = [1, 2, 1, 2]
    weights = [1.0, 1.0, 2.0, 1.0]
    result = self.vh.WeightedMajorityVoting(predictions, weights)
    self.assertEqual(result, 1)

  def testWeightedMajorityVotingEqualWeights(self):
    """Test weighted majority voting with equal weights."""
    predictions = ["x", "y", "x"]
    weights = [1.0, 1.0, 1.0]
    result = self.vh.WeightedMajorityVoting(predictions, weights)
    self.assertEqual(result, "x")

  def testWeightedMajorityVotingSinglePrediction(self):
    """Test weighted majority voting with single prediction."""
    predictions = ["single"]
    weights = [1.0]
    result = self.vh.WeightedMajorityVoting(predictions, weights)
    self.assertEqual(result, "single")

  # ========== MajorityVoting Tests. ==========

  def testMajorityVotingBasic(self):
    """Test simple majority voting."""
    predictions = ["cat", "dog", "cat"]
    result = self.vh.MajorityVoting(predictions)
    self.assertEqual(result, "cat")

  def testMajorityVotingClearWinner(self):
    """Test majority voting with clear winner."""
    predictions = ["A", "A", "A", "B"]
    result = self.vh.MajorityVoting(predictions)
    self.assertEqual(result, "A")

  def testMajorityVotingNumeric(self):
    """Test majority voting with numeric labels."""
    predictions = [1, 2, 1, 1, 2]
    result = self.vh.MajorityVoting(predictions)
    self.assertEqual(result, 1)

  def testMajorityVotingTieAdditional(self):
    """Test majority voting with tie (additional case)."""
    labels = ["cat", "dog"]
    result = self.vh.MajorityVoting(labels)
    self.assertIn(result, labels)

  # ========== WeightedAverageVoting Tests. ==========

  def testWeightedAverageVotingBasic(self):
    """Test weighted average with basic case."""
    predictions = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 1.0]
    result = self.vh.WeightedAverageVoting(predictions, weights)
    expected = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 1.0) / (1.0 + 2.0 + 1.0)  # 8/4 = 2.0.
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedAverageVotingEqualWeights(self):
    """Test weighted average with equal weights (should equal simple average)."""
    predictions = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    result = self.vh.WeightedAverageVoting(predictions, weights)
    expected = 2.0
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedAverageVotingZeroWeight(self):
    """Test weighted average with one zero weight."""
    predictions = [1.0, 2.0, 3.0]
    weights = [1.0, 0.0, 1.0]
    result = self.vh.WeightedAverageVoting(predictions, weights)
    expected = (1.0 * 1.0 + 3.0 * 1.0) / 2.0  # 2.0.
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedAverageVotingSingleValue(self):
    """Test weighted average with single value."""
    predictions = [5.0]
    weights = [2.0]
    result = self.vh.WeightedAverageVoting(predictions, weights)
    self.assertAlmostEqual(result, 5.0, places=10)

  # ========== AverageVoting Tests. ==========

  def testAverageVotingBasic(self):
    """Test simple average."""
    predictions = [1.0, 2.0, 3.0]
    result = self.vh.AverageVoting(predictions)
    self.assertAlmostEqual(result, 2.0, places=10)

  def testAverageVotingNegativeValues(self):
    """Test average with negative values."""
    predictions = [-1.0, 0.0, 1.0]
    result = self.vh.AverageVoting(predictions)
    self.assertAlmostEqual(result, 0.0, places=10)

  def testAverageVotingIntegers(self):
    """Test average with integer predictions."""
    predictions = [1, 2, 3, 4, 5]
    result = self.vh.AverageVoting(predictions)
    self.assertAlmostEqual(result, 3.0, places=10)

  def testAverageVotingSingleValue(self):
    """Test average with single value."""
    predictions = [7.5]
    result = self.vh.AverageVoting(predictions)
    self.assertAlmostEqual(result, 7.5, places=10)

  # ========== WeightedMedianVoting Tests. ==========

  def testWeightedMedianVotingBasic(self):
    """Test weighted median with basic case."""
    predictions = [1, 2, 3, 4]
    weights = [1, 1, 1, 1]
    result = self.vh.WeightedMedianVoting(predictions, weights)
    # Should return 2 (first value where cumulative weight >= 2).
    self.assertEqual(result, 2)

  def testWeightedMedianVotingSkewedWeights(self):
    """Test weighted median with skewed weights."""
    predictions = [1, 2, 3]
    weights = [1, 10, 1]
    result = self.vh.WeightedMedianVoting(predictions, weights)
    # Total weight = 12, half = 6, cumulative at value 2 is 11 >= 6.
    self.assertEqual(result, 2)

  def testWeightedMedianVotingUnsortedInput(self):
    """Test weighted median with unsorted input."""
    predictions = [3, 1, 2]
    weights = [1, 1, 1]
    result = self.vh.WeightedMedianVoting(predictions, weights)
    # After sorting: [(1,1), (2,1), (3,1)], total=3, half=1.5, returns 2.
    self.assertEqual(result, 2)

  def testWeightedMedianVotingSingleValue(self):
    """Test weighted median with single value."""
    predictions = [5]
    weights = [1]
    result = self.vh.WeightedMedianVoting(predictions, weights)
    self.assertEqual(result, 5)

  # ========== MedianVoting Tests. ==========

  def testMedianVotingOddCount(self):
    """Test median with odd number of predictions."""
    predictions = [1, 2, 3]
    result = self.vh.MedianVoting(predictions)
    self.assertEqual(result, 2)

  def testMedianVotingEvenCount(self):
    """Test median with even number of predictions."""
    predictions = [1, 2, 3, 4]
    result = self.vh.MedianVoting(predictions)
    expected = (2 + 3) / 2.0  # 2.5.
    self.assertAlmostEqual(result, expected, places=10)

  def testMedianVotingUnsorted(self):
    """Test median with unsorted predictions."""
    predictions = [5, 1, 3, 2, 4]
    result = self.vh.MedianVoting(predictions)
    self.assertEqual(result, 3)

  def testMedianVotingNegativeValues(self):
    """Test median with negative values."""
    predictions = [-3, -1, 0, 1, 3]
    result = self.vh.MedianVoting(predictions)
    self.assertEqual(result, 0)

  def testMedianVotingSingleValue(self):
    """Test median with single value."""
    predictions = [42]
    result = self.vh.MedianVoting(predictions)
    self.assertEqual(result, 42)

  # ========== WeightedModeVoting Tests. ==========

  def testWeightedModeVotingNumeric(self):
    """Test weighted mode with numeric values."""
    predictions = [1, 2, 1]
    weights = [1.0, 3.0, 1.0]
    result = self.vh.WeightedModeVoting(predictions, weights)
    # Value 2 has weight 3, values 1 have combined weight 2.
    self.assertEqual(result, 2)

  def testWeightedModeVotingLabels(self):
    """Test weighted mode with string labels."""
    predictions = ["x", "y", "x"]
    weights = [1.0, 2.0, 3.0]
    result = self.vh.WeightedModeVoting(predictions, weights)
    # x has weight 4, y has weight 2.
    self.assertEqual(result, "x")

  def testWeightedModeVotingSingleValue(self):
    """Test weighted mode with single value."""
    predictions = [5]
    weights = [1.0]
    result = self.vh.WeightedModeVoting(predictions, weights)
    self.assertEqual(result, 5)

  # ========== ModeVoting Tests. ==========

  def testModeVotingBasic(self):
    """Test simple mode voting."""
    predictions = ["x", "y", "x", "x"]
    result = self.vh.ModeVoting(predictions)
    self.assertEqual(result, "x")

  def testModeVotingNumeric(self):
    """Test mode with numeric values."""
    predictions = [1, 2, 1, 3, 1]
    result = self.vh.ModeVoting(predictions)
    self.assertEqual(result, 1)

  def testModeVotingSingleValue(self):
    """Test mode with single value."""
    predictions = ["only"]
    result = self.vh.ModeVoting(predictions)
    self.assertEqual(result, "only")

  # ========== WeightedGeometricMeanVoting Tests. ==========

  def testWeightedGeometricMeanVotingBasic(self):
    """Test weighted geometric mean with basic case."""
    predictions = [1.0, 4.0]
    weights = [1.0, 1.0]
    result = self.vh.WeightedGeometricMeanVoting(predictions, weights)
    expected = 2.0  # sqrt(1 * 4) = 2
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedGeometricMeanVotingUnequalWeights(self):
    """Test weighted geometric mean with unequal weights."""
    predictions = [2.0, 8.0]
    weights = [3.0, 1.0]
    result = self.vh.WeightedGeometricMeanVoting(predictions, weights)
    # (2^3 * 8^1)^(1/4) = (8 * 8)^(1/4) = 64^(1/4) = 2.828...
    expected = (2.0 ** 3 * 8.0 ** 1) ** (1.0 / 4.0)
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedGeometricMeanVotingEqualValues(self):
    """Test weighted geometric mean with equal values."""
    predictions = [5.0, 5.0, 5.0]
    weights = [1.0, 2.0, 3.0]
    result = self.vh.WeightedGeometricMeanVoting(predictions, weights)
    self.assertAlmostEqual(result, 5.0, places=10)

  # ========== GeometricMeanVoting Tests. ==========

  def testGeometricMeanVotingBasic(self):
    """Test simple geometric mean."""
    predictions = [1.0, 4.0]
    result = self.vh.GeometricMeanVoting(predictions)
    expected = 2.0  # sqrt(1 * 4) = 2
    self.assertAlmostEqual(result, expected, places=10)

  def testGeometricMeanVotingThreeValues(self):
    """Test geometric mean with three values."""
    predictions = [2.0, 4.0, 8.0]
    result = self.vh.GeometricMeanVoting(predictions)
    expected = (2.0 * 4.0 * 8.0) ** (1.0 / 3.0)  # 64^(1/3) = 4
    self.assertAlmostEqual(result, expected, places=10)

  def testGeometricMeanVotingEqualValues(self):
    """Test geometric mean with equal values."""
    predictions = [3.0, 3.0, 3.0]
    result = self.vh.GeometricMeanVoting(predictions)
    self.assertAlmostEqual(result, 3.0, places=10)

  # ========== WeightedHarmonicMeanVoting Tests. ==========

  def testWeightedHarmonicMeanVotingBasic(self):
    """Test weighted harmonic mean with basic case."""
    predictions = [1.0, 2.0, 4.0]
    weights = [1.0, 1.0, 1.0]
    result = self.vh.WeightedHarmonicMeanVoting(predictions, weights)
    # H = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 = 1.714...
    expected = 3.0 / (1.0 / 1.0 + 1.0 / 2.0 + 1.0 / 4.0)
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedHarmonicMeanVotingUnequalWeights(self):
    """Test weighted harmonic mean with unequal weights."""
    predictions = [2.0, 4.0]
    weights = [1.0, 3.0]
    result = self.vh.WeightedHarmonicMeanVoting(predictions, weights)
    # H = (1+3) / (1/2 + 3/4) = 4 / 1.25 = 3.2
    expected = 4.0 / (1.0 / 2.0 + 3.0 / 4.0)
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedHarmonicMeanVotingEqualValues(self):
    """Test weighted harmonic mean with equal values."""
    predictions = [5.0, 5.0, 5.0]
    weights = [1.0, 2.0, 3.0]
    result = self.vh.WeightedHarmonicMeanVoting(predictions, weights)
    self.assertAlmostEqual(result, 5.0, places=10)

  # ========== HarmonicMeanVoting Tests. ==========

  def testHarmonicMeanVotingBasic(self):
    """Test simple harmonic mean."""
    predictions = [1.0, 2.0, 4.0]
    result = self.vh.HarmonicMeanVoting(predictions)
    expected = 3.0 / (1.0 / 1.0 + 1.0 / 2.0 + 1.0 / 4.0)
    self.assertAlmostEqual(result, expected, places=10)

  def testHarmonicMeanVotingTwoValues(self):
    """Test harmonic mean with two values."""
    predictions = [2.0, 8.0]
    result = self.vh.HarmonicMeanVoting(predictions)
    # H = 2 / (1/2 + 1/8) = 2 / 0.625 = 3.2
    expected = 2.0 / (1.0 / 2.0 + 1.0 / 8.0)
    self.assertAlmostEqual(result, expected, places=10)

  def testHarmonicMeanVotingEqualValues(self):
    """Test harmonic mean with equal values."""
    predictions = [7.0, 7.0, 7.0]
    result = self.vh.HarmonicMeanVoting(predictions)
    self.assertAlmostEqual(result, 7.0, places=10)

  # ========== WeightedQuadraticMeanVoting Tests. ==========

  def testWeightedQuadraticMeanVotingBasic(self):
    """Test weighted quadratic mean (RMS) with basic case."""
    predictions = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 1.0]
    result = self.vh.WeightedQuadraticMeanVoting(predictions, weights)
    # (1^2*1 + 2^2*2 + 3^2*1) / 4 = (1 + 8 + 9) / 4 = 18/4 = 4.5
    # sqrt(4.5) = 2.121...
    expected = math.sqrt((1.0 ** 2 * 1.0 + 2.0 ** 2 * 2.0 + 3.0 ** 2 * 1.0) / 4.0)
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedQuadraticMeanVotingEqualWeights(self):
    """Test weighted quadratic mean with equal weights."""
    predictions = [3.0, 4.0]
    weights = [1.0, 1.0]
    result = self.vh.WeightedQuadraticMeanVoting(predictions, weights)
    expected = math.sqrt((9.0 + 16.0) / 2.0)  # sqrt(12.5) = 3.535...
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedQuadraticMeanVotingEqualValues(self):
    """Test weighted quadratic mean with equal values."""
    predictions = [5.0, 5.0, 5.0]
    weights = [1.0, 2.0, 3.0]
    result = self.vh.WeightedQuadraticMeanVoting(predictions, weights)
    self.assertAlmostEqual(result, 5.0, places=10)

  # ========== QuadraticMeanVoting Tests. ==========

  def testQuadraticMeanVotingBasic(self):
    """Test simple quadratic mean (RMS)."""
    predictions = [1.0, 2.0, 3.0]
    result = self.vh.QuadraticMeanVoting(predictions)
    expected = math.sqrt((1.0 + 4.0 + 9.0) / 3.0)  # sqrt(14/3) = 2.160...
    self.assertAlmostEqual(result, expected, places=10)

  def testQuadraticMeanVoting3_4Triangle(self):
    """Test quadratic mean with 3-4-5 triangle values."""
    predictions = [3.0, 4.0]
    result = self.vh.QuadraticMeanVoting(predictions)
    expected = math.sqrt((9.0 + 16.0) / 2.0)  # sqrt(12.5) = 3.535...
    self.assertAlmostEqual(result, expected, places=10)

  def testQuadraticMeanVotingEqualValues(self):
    """Test quadratic mean with equal values."""
    predictions = [6.0, 6.0, 6.0]
    result = self.vh.QuadraticMeanVoting(predictions)
    self.assertAlmostEqual(result, 6.0, places=10)

  # ========== WeightedCubicMeanVoting Tests. ==========

  def testWeightedCubicMeanVotingBasic(self):
    """Test weighted cubic mean with basic case."""
    predictions = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 1.0]
    result = self.vh.WeightedCubicMeanVoting(predictions, weights)
    # (1^3*1 + 2^3*2 + 3^3*1) / 4 = (1 + 16 + 27) / 4 = 44/4 = 11
    # 11^(1/3) = 2.224...
    expected = ((1.0 ** 3 * 1.0 + 2.0 ** 3 * 2.0 + 3.0 ** 3 * 1.0) / 4.0) ** (1.0 / 3)
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedCubicMeanVotingEqualWeights(self):
    """Test weighted cubic mean with equal weights."""
    predictions = [2.0, 4.0]
    weights = [1.0, 1.0]
    result = self.vh.WeightedCubicMeanVoting(predictions, weights)
    expected = ((8.0 + 64.0) / 2.0) ** (1.0 / 3)  # (36)^(1/3) = 3.301...
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedCubicMeanVotingEqualValues(self):
    """Test weighted cubic mean with equal values."""
    predictions = [7.0, 7.0, 7.0]
    weights = [1.0, 2.0, 3.0]
    result = self.vh.WeightedCubicMeanVoting(predictions, weights)
    self.assertAlmostEqual(result, 7.0, places=10)

  # ========== CubicMeanVoting Tests. ==========

  def testCubicMeanVotingBasic(self):
    """Test simple cubic mean."""
    predictions = [1.0, 2.0, 3.0]
    result = self.vh.CubicMeanVoting(predictions)
    expected = ((1.0 + 8.0 + 27.0) / 3.0) ** (1.0 / 3)  # (12)^(1/3) = 2.289...
    self.assertAlmostEqual(result, expected, places=10)

  def testCubicMeanVotingTwoValues(self):
    """Test cubic mean with two values."""
    predictions = [1.0, 8.0]
    result = self.vh.CubicMeanVoting(predictions)
    expected = ((1.0 + 512.0) / 2.0) ** (1.0 / 3)  # (256.5)^(1/3) = 6.353...
    self.assertAlmostEqual(result, expected, places=10)

  def testCubicMeanVotingEqualValues(self):
    """Test cubic mean with equal values."""
    predictions = [4.0, 4.0, 4.0]
    result = self.vh.CubicMeanVoting(predictions)
    self.assertAlmostEqual(result, 4.0, places=10)

  # ========== WeightedQuarticMeanVoting Tests. ==========

  def testWeightedQuarticMeanVotingBasic(self):
    """Test weighted quartic mean with basic case."""
    predictions = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 1.0]
    result = self.vh.WeightedQuarticMeanVoting(predictions, weights)
    # (1^4*1 + 2^4*2 + 3^4*1) / 4 = (1 + 32 + 81) / 4 = 114/4 = 28.5
    # 28.5^(1/4) = 2.305...
    expected = ((1.0 ** 4 * 1.0 + 2.0 ** 4 * 2.0 + 3.0 ** 4 * 1.0) / 4.0) ** (1.0 / 4)
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedQuarticMeanVotingEqualWeights(self):
    """Test weighted quartic mean with equal weights."""
    predictions = [1.0, 2.0]
    weights = [1.0, 1.0]
    result = self.vh.WeightedQuarticMeanVoting(predictions, weights)
    expected = ((1.0 + 16.0) / 2.0) ** (1.0 / 4)  # (8.5)^(1/4) = 1.711...
    self.assertAlmostEqual(result, expected, places=10)

  def testWeightedQuarticMeanVotingEqualValues(self):
    """Test weighted quartic mean with equal values."""
    predictions = [3.0, 3.0, 3.0]
    weights = [1.0, 2.0, 3.0]
    result = self.vh.WeightedQuarticMeanVoting(predictions, weights)
    self.assertAlmostEqual(result, 3.0, places=10)

  # ========== QuarticMeanVoting Tests. ==========

  def testQuarticMeanVotingBasic(self):
    """Test simple quartic mean."""
    predictions = [1.0, 2.0, 3.0]
    result = self.vh.QuarticMeanVoting(predictions)
    expected = ((1.0 + 16.0 + 81.0) / 3.0) ** (1.0 / 4)  # (98/3)^(1/4) = 2.379...
    self.assertAlmostEqual(result, expected, places=10)

  def testQuarticMeanVotingTwoValues(self):
    """Test quartic mean with two values."""
    predictions = [1.0, 3.0]
    result = self.vh.QuarticMeanVoting(predictions)
    expected = ((1.0 + 81.0) / 2.0) ** (1.0 / 4)  # (41)^(1/4) = 2.530...
    self.assertAlmostEqual(result, expected, places=10)

  def testQuarticMeanVotingEqualValues(self):
    """Test quartic mean with equal values."""
    predictions = [5.0, 5.0, 5.0]
    result = self.vh.QuarticMeanVoting(predictions)
    self.assertAlmostEqual(result, 5.0, places=10)

  # ========== Integration Tests. ==========

  def testMeanRelationships(self):
    """Test that harmonic <= geometric <= arithmetic <= quadratic for positive numbers."""
    predictions = [1.0, 2.0, 4.0]
    weights = [1.0, 1.0, 1.0]

    harmonic = self.vh.HarmonicMeanVoting(predictions)
    geometric = self.vh.GeometricMeanVoting(predictions)
    arithmetic = self.vh.AverageVoting(predictions)
    quadratic = self.vh.QuadraticMeanVoting(predictions)

    self.assertLessEqual(harmonic, geometric)
    self.assertLessEqual(geometric, arithmetic)
    self.assertLessEqual(arithmetic, quadratic)

  def testAllEqualValuesConsistency(self):
    """Test that all mean methods return same value when all predictions are equal."""
    predictions = [5.0, 5.0, 5.0]
    weights = [1.0, 2.0, 3.0]

    avg = self.vh.AverageVoting(predictions)
    wavg = self.vh.WeightedAverageVoting(predictions, weights)
    geom = self.vh.GeometricMeanVoting(predictions)
    wgeom = self.vh.WeightedGeometricMeanVoting(predictions, weights)
    harm = self.vh.HarmonicMeanVoting(predictions)
    wharm = self.vh.WeightedHarmonicMeanVoting(predictions, weights)
    quad = self.vh.QuadraticMeanVoting(predictions)
    wquad = self.vh.WeightedQuadraticMeanVoting(predictions, weights)
    cub = self.vh.CubicMeanVoting(predictions)
    wcub = self.vh.WeightedCubicMeanVoting(predictions, weights)
    quart = self.vh.QuarticMeanVoting(predictions)
    wquart = self.vh.WeightedQuarticMeanVoting(predictions, weights)

    for mean_val in [avg, wavg, geom, wgeom, harm, wharm, quad, wquad, cub, wcub, quart, wquart]:
      self.assertAlmostEqual(mean_val, 5.0, places=10)


if __name__ == "__main__":
  unittest.main()
