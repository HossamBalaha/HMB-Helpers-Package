import unittest
import numpy as np
from HMB.VectorsHelper import VectorsHelper


class TestVectorsHelper(unittest.TestCase):
  """
  Unit tests for VectorsHelper covering basic operations and edge cases.
  """

  def setUp(self):
    """Initialize VectorsHelper instance before each test."""
    self.vh = VectorsHelper()

  def testLengthSimpleVector(self):
    """Test Length method with simple 2D vector."""
    vector = np.array([3.0, 4.0])
    result = self.vh.Length(vector)
    expected = 5.0  # sqrt(3^2 + 4^2) = sqrt(25) = 5.
    self.assertAlmostEqual(result, expected, places=10)

  def testLengthUnitVector(self):
    """Test Length method with unit vector."""
    vector = np.array([1.0, 0.0, 0.0])
    result = self.vh.Length(vector)
    self.assertAlmostEqual(result, 1.0, places=10)

  def testLengthZeroVector(self):
    """Test Length method with zero vector."""
    vector = np.array([0.0, 0.0, 0.0])
    result = self.vh.Length(vector)
    self.assertAlmostEqual(result, 0.0, places=10)

  def testLengthNegativeValues(self):
    """Test Length method with negative values."""
    vector = np.array([-3.0, -4.0])
    result = self.vh.Length(vector)
    expected = 5.0
    self.assertAlmostEqual(result, expected, places=10)

  def testLengthHighDimension(self):
    """Test Length method with high-dimensional vector."""
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = self.vh.Length(vector)
    expected = np.sqrt(1 + 4 + 9 + 16 + 25)  # sqrt(55).
    self.assertAlmostEqual(result, expected, places=10)

  def testLengthListInput(self):
    """Test Length method with list input."""
    vector = [3.0, 4.0]
    result = self.vh.Length(vector)
    expected = 5.0
    self.assertAlmostEqual(result, expected, places=10)

  def testDotproductOrthogonalVectors(self):
    """Test DotProduct with orthogonal vectors (should be 0)."""
    vector1 = np.array([1.0, 0.0])
    vector2 = np.array([0.0, 1.0])
    result = self.vh.DotProduct(vector1, vector2)
    self.assertAlmostEqual(result, 0.0, places=10)

  def testDotproductParallelVectors(self):
    """Test DotProduct with parallel vectors."""
    vector1 = np.array([2.0, 3.0])
    vector2 = np.array([4.0, 6.0])
    result = self.vh.DotProduct(vector1, vector2)
    expected = 2.0 * 4.0 + 3.0 * 6.0  # 8 + 18 = 26.
    self.assertAlmostEqual(result, expected, places=10)

  def testDotproductIdenticalVectors(self):
    """Test DotProduct with identical vectors."""
    vector = np.array([3.0, 4.0])
    result = self.vh.DotProduct(vector, vector)
    expected = 3.0 * 3.0 + 4.0 * 4.0  # 9 + 16 = 25.
    self.assertAlmostEqual(result, expected, places=10)

  def testDotproductNegativeValues(self):
    """Test DotProduct with negative values."""
    vector1 = np.array([1.0, -2.0, 3.0])
    vector2 = np.array([-1.0, 2.0, -3.0])
    result = self.vh.DotProduct(vector1, vector2)
    expected = 1.0 * (-1.0) + (-2.0) * 2.0 + 3.0 * (-3.0)  # -1 - 4 - 9 = -14.
    self.assertAlmostEqual(result, expected, places=10)

  def testCrossproductStandardBasis(self):
    """Test CrossProduct with standard basis vectors."""
    vector1 = np.array([1.0, 0.0, 0.0])
    vector2 = np.array([0.0, 1.0, 0.0])
    result = self.vh.CrossProduct(vector1, vector2)
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=10)

  def testCrossproductReversedOrder(self):
    """Test CrossProduct with reversed order (should negate result)."""
    vector1 = np.array([0.0, 1.0, 0.0])
    vector2 = np.array([1.0, 0.0, 0.0])
    result = self.vh.CrossProduct(vector1, vector2)
    expected = np.array([0.0, 0.0, -1.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=10)

  def testCrossproductParallelVectors(self):
    """Test CrossProduct with parallel vectors (should be zero vector)."""
    vector1 = np.array([1.0, 2.0, 3.0])
    vector2 = np.array([2.0, 4.0, 6.0])
    result = self.vh.CrossProduct(vector1, vector2)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=10)

  def testCrossproductGeneralCase(self):
    """Test CrossProduct with general 3D vectors."""
    vector1 = np.array([2.0, 3.0, 4.0])
    vector2 = np.array([5.0, 6.0, 7.0])
    result = self.vh.CrossProduct(vector1, vector2)
    # Manual calculation: [(3*7 - 4*6), (4*5 - 2*7), (2*6 - 3*5)] = [-3, 6, -3].
    expected = np.array([-3.0, 6.0, -3.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=10)

  def testDistanceIdenticalVectors(self):
    """Test Distance between identical vectors (should be 0)."""
    vector = np.array([1.0, 2.0, 3.0])
    result = self.vh.Distance(vector, vector)
    self.assertAlmostEqual(result, 0.0, places=10)

  def testDistanceSimpleCase(self):
    """Test Distance with simple 2D vectors."""
    vector1 = np.array([0.0, 0.0])
    vector2 = np.array([3.0, 4.0])
    result = self.vh.Distance(vector1, vector2)
    expected = 5.0  # sqrt(3^2 + 4^2).
    self.assertAlmostEqual(result, expected, places=10)

  def testDistanceNegativeCoordinates(self):
    """Test Distance with negative coordinates."""
    vector1 = np.array([1.0, 2.0])
    vector2 = np.array([-2.0, -2.0])
    result = self.vh.Distance(vector1, vector2)
    # diff = [3.0, 4.0], distance = 5.0.
    expected = 5.0
    self.assertAlmostEqual(result, expected, places=10)

  def testDistanceSymmetry(self):
    """Test that Distance is symmetric."""
    vector1 = np.array([1.0, 2.0, 3.0])
    vector2 = np.array([4.0, 5.0, 6.0])
    result1 = self.vh.Distance(vector1, vector2)
    result2 = self.vh.Distance(vector2, vector1)
    self.assertAlmostEqual(result1, result2, places=10)

  def testAngleOrthogonalVectorsRad(self):
    """Test Angle between orthogonal vectors in radians."""
    vector1 = np.array([1.0, 0.0])
    vector2 = np.array([0.0, 1.0])
    result = self.vh.Angle(vector1, vector2, mode="rad")
    expected = np.pi / 2  # 90 degrees.
    self.assertAlmostEqual(result, expected, places=10)

  def testAngleOrthogonalVectorsDeg(self):
    """Test Angle between orthogonal vectors in degrees."""
    vector1 = np.array([1.0, 0.0])
    vector2 = np.array([0.0, 1.0])
    result = self.vh.Angle(vector1, vector2, mode="deg")
    expected = 90.0
    self.assertAlmostEqual(result, expected, places=10)

  def testAngleParallelVectors(self):
    """Test Angle between parallel vectors (should be 0)."""
    vector1 = np.array([1.0, 2.0, 3.0])
    vector2 = np.array([2.0, 4.0, 6.0])
    result = self.vh.Angle(vector1, vector2, mode="rad")
    self.assertAlmostEqual(result, 0.0, places=10)

  def testAngleOppositeVectorsRad(self):
    """Test Angle between opposite vectors in radians."""
    vector1 = np.array([1.0, 0.0])
    vector2 = np.array([-1.0, 0.0])
    result = self.vh.Angle(vector1, vector2, mode="rad")
    expected = np.pi  # 180 degrees.
    self.assertAlmostEqual(result, expected, places=10)

  def testAngleOppositeVectorsDeg(self):
    """Test Angle between opposite vectors in degrees."""
    vector1 = np.array([1.0, 0.0])
    vector2 = np.array([-1.0, 0.0])
    result = self.vh.Angle(vector1, vector2, mode="deg")
    expected = 180.0
    self.assertAlmostEqual(result, expected, places=10)

  def testAngle45Degrees(self):
    """Test Angle for vectors at 45 degrees."""
    vector1 = np.array([1.0, 0.0])
    vector2 = np.array([1.0, 1.0])
    result = self.vh.Angle(vector1, vector2, mode="deg")
    expected = 45.0
    self.assertAlmostEqual(result, expected, places=10)

  def testAngle3dVectors(self):
    """Test Angle with 3D vectors."""
    vector1 = np.array([1.0, 0.0, 0.0])
    vector2 = np.array([0.0, 1.0, 0.0])
    result = self.vh.Angle(vector1, vector2, mode="deg")
    expected = 90.0
    self.assertAlmostEqual(result, expected, places=10)

  def testChangebasisSingleBasisVector(self):
    """Test ChangeBasis with a single basis vector."""
    v = np.array([3.0, 4.0])
    basis = np.array([1.0, 0.0])
    result = self.vh.ChangeBasis(v, basis)
    # Projection of [3,4] onto [1,0] = (3*1 + 4*0) / (1^2 + 0^2) = 3/1 = 3.
    expected = [3.0]
    self.assertEqual(len(result), 1)
    self.assertAlmostEqual(result[0], expected[0], places=10)

  def testChangebasisMultipleBasisVectors(self):
    """Test ChangeBasis with multiple basis vectors."""
    v = np.array([6.0, 8.0])
    basis1 = np.array([1.0, 0.0])
    basis2 = np.array([0.0, 1.0])
    result = self.vh.ChangeBasis(v, basis1, basis2)
    # Projection onto [1,0] = 6, projection onto [0,1] = 8.
    expected = [6.0, 8.0]
    self.assertEqual(len(result), 2)
    self.assertAlmostEqual(result[0], expected[0], places=10)
    self.assertAlmostEqual(result[1], expected[1], places=10)

  def testChangebasisNonOrthogonalBasis(self):
    """Test ChangeBasis with non-orthogonal basis vectors."""
    v = np.array([10.0, 5.0])
    basis1 = np.array([2.0, 1.0])
    basis2 = np.array([-2.0, 4.0])
    result = self.vh.ChangeBasis(v, basis1, basis2)
    # Projection onto [2,1]: (10*2 + 5*1) / (2^2 + 1^2) = 25 / 5 = 5.
    # Projection onto [-2,4]: (10*(-2) + 5*4) / ((-2)^2 + 4^2) = 0 / 20 = 0.
    expected = [5.0, 0.0]
    self.assertEqual(len(result), 2)
    self.assertAlmostEqual(result[0], expected[0], places=10)
    self.assertAlmostEqual(result[1], expected[1], places=10)

  def testChangebasis3dVector(self):
    """Test ChangeBasis with 3D vector and basis."""
    v = np.array([1.0, 2.0, 3.0])
    basis1 = np.array([1.0, 0.0, 0.0])
    basis2 = np.array([0.0, 1.0, 0.0])
    basis3 = np.array([0.0, 0.0, 1.0])
    result = self.vh.ChangeBasis(v, basis1, basis2, basis3)
    expected = [1.0, 2.0, 3.0]
    self.assertEqual(len(result), 3)
    for i in range(3):
      self.assertAlmostEqual(result[i], expected[i], places=10)

  def testChangebasisZeroVector(self):
    """Test ChangeBasis with zero vector."""
    v = np.array([0.0, 0.0])
    basis = np.array([1.0, 1.0])
    result = self.vh.ChangeBasis(v, basis)
    expected = [0.0]
    self.assertEqual(len(result), 1)
    self.assertAlmostEqual(result[0], expected[0], places=10)

  def testChangebasisScaledBasis(self):
    """Test ChangeBasis with scaled basis vector."""
    v = np.array([6.0, 8.0])
    basis = np.array([3.0, 4.0])  # length = 5.
    result = self.vh.ChangeBasis(v, basis)
    # dot(v, basis) = 6*3 + 8*4 = 18 + 32 = 50.
    # sum(basis^2) = 9 + 16 = 25.
    # projection = 50 / 25 = 2.
    expected = [2.0]
    self.assertEqual(len(result), 1)
    self.assertAlmostEqual(result[0], expected[0], places=10)

  def testInputTypesNumpyArrays(self):
    """Test that methods work with numpy arrays."""
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])

    length = self.vh.Length(v1)
    dot = self.vh.DotProduct(v1, v2)
    distance = self.vh.Distance(v1, v2)

    self.assertIsInstance(length, (float, np.floating, np.ndarray))
    self.assertIsInstance(dot, (float, np.floating, np.ndarray))
    self.assertIsInstance(distance, (float, np.floating, np.ndarray))

  def testInputTypesLists(self):
    """Test that methods work with Python lists."""
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]

    length = self.vh.Length(v1)
    dot = self.vh.DotProduct(v1, v2)
    distance = self.vh.Distance(v1, v2)

    self.assertIsInstance(length, (float, np.floating, np.ndarray))
    self.assertIsInstance(dot, (float, np.floating, np.ndarray))
    self.assertIsInstance(distance, (float, np.floating, np.ndarray))

  def testCosineSimilarityBasic(self):
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    self.assertAlmostEqual(self.vh.CosineSimilarity(a, b), 1.0, places=6)

  def testCosineSimilarityOrthogonal(self):
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    self.assertAlmostEqual(self.vh.CosineSimilarity(a, b), 0.0, places=6)

  def testCosineSimilarityZeroVector(self):
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    val = self.vh.CosineSimilarity(a, b)
    self.assertTrue(np.isfinite(val))

  def testNormalizeVector(self):
    v = np.array([3.0, 4.0])
    n = self.vh.NormalizeVector(v)
    self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=6)

  def testNormalizeZeroVector(self):
    v = np.array([0.0, 0.0])
    n = self.vh.NormalizeVector(v)
    self.assertTrue(np.all(np.isfinite(n)))

  def testProjectVector(self):
    v = np.array([2.0, 2.0])
    basis = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    p = self.vh.ProjectVector(v, basis)
    self.assertAlmostEqual(p[0], 2.0, places=6)
    self.assertAlmostEqual(p[1], 2.0, places=6)

  def testHighDimensionalVectors(self):
    a = np.random.rand(1000)
    b = np.random.rand(1000)
    val = self.vh.CosineSimilarity(a, b)
    self.assertTrue(-1.0 <= val <= 1.0)

  def testAngleZeroVectorHandling(self):
    """Angle with a zero vector should result in NaN; ensure it's a float and is NaN."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    ang = self.vh.Angle(v1, v2, mode="rad")
    # Depending on implementation, this may produce nan due to division by zero
    self.assertTrue(np.isnan(ang))

  def testChangebasisZeroBasisVector(self):
    """ChangeBasis with a zero-norm basis vector should handle division by zero (inf) in projection."""
    v = np.array([1.0, 2.0])
    zero_basis = np.array([0.0, 0.0])
    res = self.vh.ChangeBasis(v, zero_basis)
    # Expect inf or nan due to division by zero; assert it's not finite
    self.assertEqual(len(res), 1)
    self.assertTrue(not np.isfinite(res[0]))

  def testAngleNumericStability(self):
    """Angle numerical stability near parallel vectors should stay within [0, pi]."""
    a = np.array([1e8, 1e8])
    b = np.array([2e8, 2e8])
    ang = self.vh.Angle(a, b, mode="rad")
    self.assertTrue(0.0 <= ang <= np.pi)


if __name__ == "__main__":
  unittest.main()
