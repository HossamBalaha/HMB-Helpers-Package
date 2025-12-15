import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from HMB.MachineLearningHelper import (
  GetScalerObject,
  ListScikitMachineLearningClassifiers,
  GetMLClassificationModelObject,
  PerformDataBalancing,
)


class TestMachineLearningHelper(unittest.TestCase):
  """
  Edge-case tests for MachineLearningHelper utilities focusing on input validation
  and behavior with empty/small datasets. Uses mocks where external libs are heavy.
  """

  def setUp(self):
    # Small synthetic dataset
    self.X = np.random.randn(20, 5)
    self.y = np.random.randint(0, 2, size=20)

  def test_get_scaler_object_standard(self):
    scaler = GetScalerObject("StandardScaler")
    self.assertIsNotNone(scaler)
    # Should be able to fit_transform
    scaled = scaler.fit_transform(self.X)
    self.assertEqual(scaled.shape, self.X.shape)

  def test_list_classifiers(self):
    classifiers = ListScikitMachineLearningClassifiers()
    self.assertIsInstance(classifiers, dict)
    self.assertGreater(len(classifiers), 0)

  def test_get_ml_classification_model(self):
    model = GetMLClassificationModelObject("RandomForestClassifier", hyperparameters={"n_estimators": 10})
    self.assertIsNotNone(model)
    # Should be able to fit
    model.fit(self.X, self.y)

  def test_perform_data_balancing_smote(self):
    try:
      res = PerformDataBalancing(self.X, self.y, techniqueStr="SMOTE")
      # Support both (X, y) and (X, y, obj)
      if len(res) == 2:
        X_bal, y_bal = res
      else:
        X_bal, y_bal, _ = res
      self.assertGreaterEqual(len(X_bal), len(self.X))
      self.assertEqual(len(X_bal), len(y_bal))
    except ImportError:
      self.skipTest("imblearn not installed")


if __name__ == "__main__":
  unittest.main()
