import unittest
import numpy as np
from HMB.PerformanceMetrics import CalculatePerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
  """
  Unit tests for PerformanceMetrics.CalculatePerformanceMetrics focusing on small and edge confusions.
  """

  def test_binary_confusion_basic(self):
    cm = np.array([[50, 2], [5, 45]])
    m = CalculatePerformanceMetrics(cm, addWeightedAverage=True, addPerClass=False)
    # Check presence of keys
    for k in [
      "Macro Precision", "Macro Recall", "Macro F1", "Macro Accuracy",
      "Micro Precision", "Micro Recall", "Micro F1", "Micro Accuracy",
      "Weighted Precision", "Weighted Recall", "Weighted F1", "Weighted Accuracy",
    ]:
      self.assertIn(k, m)

  def test_multiclass_confusion(self):
    cm = np.array([[10, 2, 1], [3, 9, 0], [0, 1, 11]])
    m = CalculatePerformanceMetrics(cm, addWeightedAverage=True, addPerClass=True)
    # Per-class keys should exist
    self.assertIn("Class 0 Precision", m)
    self.assertIn("Class 1 Recall", m)
    self.assertIn("Class 2 F1", m)

  def test_unbalanced_classes(self):
    cm = np.array([[100, 0, 0], [0, 1, 0], [0, 0, 1]])
    m = CalculatePerformanceMetrics(cm, addWeightedAverage=True, addPerClass=True)
    self.assertTrue(m["Weighted Accuracy"] >= m["Macro Accuracy"])  # tendency in unbalanced cases

  def test_string_input_confusion(self):
    # Ensure function handles list input (converted to numpy)
    cm = [[3, 1], [0, 2]]
    m = CalculatePerformanceMetrics(cm)
    self.assertIn("Macro F1", m)


if __name__ == "__main__":
  unittest.main()
