import math
from collections import Counter
from functools import reduce


class VotingHelper(object):
  r'''
  VotingHelper: Collection of voting and aggregation utilities.

  Each method takes a list of predictions and optionally a list of weights (same length).
  Methods return a single aggregated scalar or label depending on the strategy.
  '''

  def WeightedMajorityVoting(self, predictions, weights):
    r'''
    Compute a weighted majority vote over labels.

    Parameters:
      predictions (iterable): Iterable of labels.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      label: The label with the highest cumulative weight.
    '''

    cumulativeVotes = Counter()
    for pred, weight in zip(predictions, weights):
      cumulativeVotes[pred] += weight
    majorityLabel = max(cumulativeVotes, key=cumulativeVotes.get)
    return majorityLabel

  def MajorityVoting(self, predictions):
    r'''
    Simple majority vote (unweighted).

    Parameters:
      predictions (iterable): Iterable of labels.

    Returns:
      label: The most common label.
    '''

    majorityLabel = max(set(predictions), key=predictions.count)
    return majorityLabel

  def WeightedAverageVoting(self, predictions, weights):
    r'''
    Weighted average of numeric predictions.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted mean.
    '''

    return sum([pred * weight for pred, weight in zip(predictions, weights)]) / sum(weights)

  def AverageVoting(self, predictions):
    r'''
    Simple arithmetic mean of numeric predictions.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Arithmetic mean.
    '''

    return sum(predictions) / len(predictions)

  def WeightedMedianVoting(self, predictions, weights):
    r'''
    Weighted median: returns a scalar median taking weights into account.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted median.
    '''

    # Pair and sort by prediction value.
    items = sorted(zip(predictions, weights), key=lambda x: x[0])
    total = sum(weights)
    cum = 0.0
    for val, w in items:
      cum += w
      if (cum >= total / 2.0):
        return val
    return items[-1][0]

  def MedianVoting(self, predictions):
    r'''
    Unweighted median for numeric predictions.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Median value.
    '''

    s = sorted(predictions)
    mid = len(s) // 2
    if (len(s) % 2 == 1):
      return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

  def WeightedModeVoting(self, predictions, weights):
    r'''
    Weighted mode: returns the weighted-average of the mode(s) if numeric, otherwise the highest-weight label.

    Parameters:
      predictions (iterable): Iterable of labels (numeric or non-numeric).
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      label or float: Weighted mode label or numeric value.
    '''

    # If labels are non-numeric, return the label with largest cumulative weight.
    try:
      # try numeric path: compute weighted frequencies per unique value
      cumulative = {}
      for p, w in zip(predictions, weights):
        cumulative[p] = cumulative.get(p, 0) + w
      # return the key with max cumulative weight
      return max(cumulative, key=cumulative.get)
    except Exception:
      return self.WeightedMajorityVoting(predictions, weights)

  def ModeVoting(self, predictions):
    r'''
    Unweighted mode (most common element).

    Parameters:
      predictions (iterable): Iterable of labels.

    Returns:
      label: Most common label.
    '''

    return max(set(predictions), key=predictions.count)

  def WeightedGeometricMeanVoting(self, predictions, weights):
    r'''
    Weighted geometric mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted geometric mean.
    '''

    # Compute product of powers: prod(p_i ** w_i) ^ (1 / sum(weights)).
    logSum = sum([w * math.log(p) for p, w in zip(predictions, weights)])
    return math.exp(logSum / sum(weights))

  def GeometricMeanVoting(self, predictions):
    r'''
    Geometric mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.

    Returns:
      float: Geometric mean.
    '''

    prod = reduce(lambda x, y: x * y, predictions)
    return prod ** (1.0 / len(predictions))

  def WeightedHarmonicMeanVoting(self, predictions, weights):
    r'''
    Weighted harmonic mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted harmonic mean.
    '''

    num = sum(weights)
    den = sum([w / p for p, w in zip(predictions, weights)])
    return num / den

  def HarmonicMeanVoting(self, predictions):
    r'''
    Unweighted harmonic mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.

    Returns:
      float: Harmonic mean.
    '''

    return len(predictions) / sum([1.0 / pred for pred in predictions])

  def WeightedQuadraticMeanVoting(self, predictions, weights):
    r'''
    Weighted root-mean-square (quadratic mean).

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted root-mean-square.
    '''

    num = sum([w * (p ** 2) for p, w in zip(predictions, weights)])
    return (num / sum(weights)) ** 0.5

  def QuadraticMeanVoting(self, predictions):
    r'''
    Unweighted root-mean-square.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Root-mean-square.
    '''

    return (sum([pred ** 2 for pred in predictions]) / len(predictions)) ** 0.5

  def WeightedCubicMeanVoting(self, predictions, weights):
    r'''
    Weighted cubic mean (signed) -> (mean of cubes)^(1/3).

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted cubic mean.
    '''

    num = sum([w * (p ** 3) for p, w in zip(predictions, weights)])
    return (num / sum(weights)) ** (1.0 / 3)

  def CubicMeanVoting(self, predictions):
    r'''
    Unweighted cubic mean.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Cubic mean.
    '''

    return (sum([pred ** 3 for pred in predictions]) / len(predictions)) ** (1.0 / 3)

  def WeightedQuarticMeanVoting(self, predictions, weights):
    r'''
    Weighted quartic mean -> 4th root of mean of 4th powers.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted quartic mean.
    '''

    num = sum([w * (p ** 4) for p, w in zip(predictions, weights)])
    return (num / sum(weights)) ** (1.0 / 4)

  def QuarticMeanVoting(self, predictions):
    r'''
    Unweighted quartic mean.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Quartic mean.
    '''

    return (sum([pred ** 4 for pred in predictions]) / len(predictions)) ** (1.0 / 4)


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


  vh = VotingHelper()

  # Label-based voting
  labels = ["cat", "dog", "cat"]
  weights = [0.6, 0.2, 0.2]
  SafeCall("WeightedMajorityVoting", vh.WeightedMajorityVoting, labels, weights)
  SafeCall("MajorityVoting", vh.MajorityVoting, labels)

  # Numeric aggregation examples
  nums = [1.0, 2.0, 3.0]
  wnums = [1.0, 2.0, 1.0]
  SafeCall("WeightedAverageVoting", vh.WeightedAverageVoting, nums, wnums)
  SafeCall("AverageVoting", vh.AverageVoting, nums)

  # Median variants
  SafeCall("WeightedMedianVoting", vh.WeightedMedianVoting, [1, 2, 3, 4], [1, 1, 1, 1])
  SafeCall("MedianVoting", vh.MedianVoting, [1, 2, 3])

  # Mode variants
  SafeCall("WeightedModeVoting", vh.WeightedModeVoting, ["x", "y", "x"], [1, 2, 3])
  SafeCall("ModeVoting", vh.ModeVoting, ["x", "y", "x", "x"])

  # Geometric / harmonic means (positive inputs required)
  SafeCall("WeightedGeometricMeanVoting", vh.WeightedGeometricMeanVoting, [1.0, 4.0], [1.0, 1.0])
  SafeCall("GeometricMeanVoting", vh.GeometricMeanVoting, [1.0, 4.0])
  SafeCall("WeightedHarmonicMeanVoting", vh.WeightedHarmonicMeanVoting, [1.0, 2.0, 4.0], [1.0, 1.0, 1.0])
  SafeCall("HarmonicMeanVoting", vh.HarmonicMeanVoting, [1.0, 2.0, 4.0])

  # Power means
  SafeCall("WeightedQuadraticMeanVoting", vh.WeightedQuadraticMeanVoting, [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
  SafeCall("QuadraticMeanVoting", vh.QuadraticMeanVoting, [1.0, 2.0, 3.0])
  SafeCall("WeightedCubicMeanVoting", vh.WeightedCubicMeanVoting, [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
  SafeCall("CubicMeanVoting", vh.CubicMeanVoting, [1.0, 2.0, 3.0])
  SafeCall("WeightedQuarticMeanVoting", vh.WeightedQuarticMeanVoting, [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
  SafeCall("QuarticMeanVoting", vh.QuarticMeanVoting, [1.0, 2.0, 3.0])

  print("VotingHelper demo completed.")
