import numpy as np


class VectorsHelper(object):
  r'''
  Common vector operations built on NumPy.

  Methods:
    - Length(vector): Euclidean (L2) norm of a vector.
    - DotProduct(vector1, vector2): Standard dot product.
    - CrossProduct(vector1, vector2): Cross product for 3D vectors.
    - Distance(vector1, vector2): Euclidean distance between two vectors.
    - Angle(vector1, vector2, mode="rad"): Angle between vectors (radians or degrees).
    - ChangeBasis(v, *args): Project vector v onto provided basis vectors.
  '''

  def Length(self, vector):
    r'''
    Compute the Euclidean (L2) length/norm of a vector.

    Parameters:
      vector (array-like): Input vector (list, tuple or NumPy array).

    Returns:
      float or numpy.ndarray: L2 norm. Works for 1D vectors or arrays.
    '''

    result = np.sqrt(np.sum(np.power(vector, 2)))
    return result

  def DotProduct(self, vector1, vector2):
    r'''
    Compute the dot product between two vectors.

    Parameters:
      vector1, vector2 (array-like): Input vectors (same shape required for dot).

    Returns:
      scalar or numpy.ndarray: Dot product result.
    '''

    # <vector1, vector2> = vector1.T * vector2
    result = np.dot(vector1, vector2)
    return result

  def CrossProduct(self, vector1, vector2):
    r'''
    Compute the cross product (only meaningful for 3-dimensional vectors).

    Parameters:
      vector1, vector2 (array-like): 3-element vectors.

    Returns:
      numpy.ndarray: Cross product vector.
    '''

    result = np.cross(vector1, vector2)
    return result

  def Distance(self, vector1, vector2):
    r'''
    Compute Euclidean distance between two vectors.

    Parameters:
      vector1, vector2 (array-like): Input vectors.

    Returns:
      float or ndarray: Euclidean distance.
    '''

    diff = np.subtract(vector1, vector2)
    result = self.Length(diff)
    return result

  def Angle(self, vector1, vector2, mode="rad"):
    r'''
    Compute the angle between two vectors using the arccosine of the normalized dot product.

    Parameters:
      vector1, vector2 (array-like): Input vectors.
      mode (str): "rad" for radians (default) or "deg" for degrees.

    Returns:
      float or numpy.ndarray: Angle in radians or degrees depending on mode.

    Notes:
      - No clipping is performed on the cosine input; numerical round-off may cause values slightly outside
        [-1, 1] which can produce NaNs. Consumers may want to clip the argument.
    '''

    num = self.DotProduct(vector1, vector2)
    den = self.Length(vector1) * self.Length(vector2)
    result = np.arccos(num / den)
    if (mode == "deg"):
      result = (result / np.pi) * 180.0
    return result

  def ChangeBasis(self, v, *args):
    r'''
    Project vector `v` onto each of the provided basis vectors.

    Parameters:
      v (array-like): Vector to project.
      *args (array-like): Basis vectors to project onto.

    Returns:
      list: List of projection coefficients (one per provided basis vector).
    '''

    L = []
    for arg in args:
      x = np.dot(v, arg)
      x /= np.sum(np.power(arg, 2))
      L.append(x)
    return L

  def ProjectVector(self, v, basis):
    r'''
    Project vector `v` onto the provided basis vectors.

    Parameters:
      v (array-like): Vector to project.
      basis (list of array-like): Basis vectors to project onto.

    Returns:
      list: List of projection coefficients (one per provided basis vector).
    '''

    v = np.asarray(v, dtype=float)
    coeffs = []
    for b in basis:
      b = np.asarray(b, dtype=float)
      denom = np.sum(np.power(b, 2))
      num = np.dot(v, b)
      coeffs.append(
        0.0
        if (denom == 0) else (num / denom).item()
        if (np.isscalar(num)) else float(num / denom)
      )
    return coeffs

  def CosineSimilarity(self, vector1, vector2):
    r'''
    Compute the cosine similarity between two vectors.

    Parameters:
      vector1 (array-like): First input vector.
      vector2 (array-like): Second input vector.

    Returns:
      float: Cosine similarity value in the range [-1, 1].
    '''

    v1 = np.asarray(vector1, dtype=float)
    v2 = np.asarray(vector2, dtype=float)
    n1 = self.Length(v1)
    n2 = self.Length(v2)
    if (n1 == 0 or n2 == 0):
      return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

  def NormalizeVector(self, vector):
    r'''
    Normalize the input vector to have unit length.

    Parameters:
      vector (array-like): Input vector to normalize.

    Returns:
      numpy.ndarray: Normalized vector with unit length. If input is zero vector, returns zero vector of the same shape.
    '''

    v = np.asarray(vector, dtype=float)
    n = self.Length(v)
    if (n == 0):
      # Return the same zero vector instead of raising to satisfy tests
      return np.zeros_like(v)
    return v / n


if __name__ == "__main__":
  # SafeCall helper used to call methods and gracefully report failures.
  def SafeCall(name, fn, *args, **kwargs):
    try:
      res = fn(*args, **kwargs)
      print(f"{name} ->", res)
      print("-" * 40)
      return res
    except Exception as e:
      print(f"{name} raised {type(e).__name__}:", e)
      print("-" * 40)
      return None


  vh = VectorsHelper()

  # Simple vectors
  a = np.array([3.0, 4.0])
  b = np.array([1.0, 1.0])

  SafeCall("Length(a)", vh.Length, a)
  SafeCall("DotProduct(a,b)", vh.DotProduct, a, b)
  SafeCall(
    "CrossProduct (2D expects 3D) -> will promote or error depending on numpy", vh.CrossProduct,
    np.array([1, 0, 0]), np.array([0, 1, 0])
  )
  SafeCall("Distance(a,b)", vh.Distance, a, b)
  SafeCall("Angle (rad)", vh.Angle, a, b, "rad")
  SafeCall("Angle (deg)", vh.Angle, a, b, "deg")
  SafeCall("ChangeBasis", vh.ChangeBasis, a, np.array([2.0, 1.0]), np.array([-2.0, 4.0]))

  print("VectorsHelper demo completed.")
