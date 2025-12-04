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
