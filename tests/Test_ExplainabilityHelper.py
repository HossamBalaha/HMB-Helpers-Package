import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from HMB.ExplainabilityHelper import OptunaMLPipelineSHAPExplainer


class TestExplainabilityHelper(unittest.TestCase):
  """
  Unit tests for SHAPExplainer covering initialization and main workflow with mocks.
  """

  @patch("HMB.ExplainabilityHelper.shap")
  @patch("HMB.ExplainabilityHelper.plt")
  def test_shap_explainer_workflow(self, mockPlt, mockShap):
    # Mock shap explainer and values.
    mockExpl = MagicMock()
    mockShap.TreeExplainer.return_value = mockExpl
    mockExpl.shap_values.return_value = np.random.randn(10, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
      baseDir = tmpdir
      expName = "Exp1"
      testCsv = "test.csv"
      targetCol = "y"
      shapKey = "SHAP_Results"

      # Prepare dummy data.
      df = pd.DataFrame({
        "f1": np.random.randn(20),
        "f2": np.random.randn(20),
        "y" : np.random.randint(0, 2, size=20)
      })
      expDir = os.path.join(baseDir, expName)
      os.makedirs(expDir, exist_ok=True)
      df.to_csv(os.path.join(expDir, testCsv), index=False)

      # Create dummy pickle file path.
      picklePath = os.path.join(expDir, "model.pkl")
      with open(picklePath, "wb") as f:
        f.write(b"dummy")

      expl = OptunaMLPipelineSHAPExplainer(
        baseDir=baseDir,
        experimentFolderName=expName,
        testFilename=testCsv,
        targetColumn=targetCol,
        pickleFilePath=picklePath,
        shapStorageKeyword=shapKey,
      )

      # Stub methods to avoid real model loading; simulate minimal flow.
      expl.objects = {"model": MagicMock()}
      expl.testData = df.copy()
      expl.XTest = df[["f1", "f2"]]
      expl.yTest = df["y"]
      expl.model = MagicMock()

      # Compute shap values.
      expl.explainer = mockExpl

      # Create a SHAP-like object with .values and .data.
      class ShapVals:
        def __init__(self, vals, data):
          self.values = vals
          self.data = data

        def __getitem__(self, idx):
          return self.values[idx]

      sv = ShapVals(np.random.randn(len(expl.XTest), expl.XTest.shape[1]), expl.XTest.values)
      expl.shapValues = sv

      # Make predictions.
      expl.yPred = np.random.randint(0, 2, size=len(expl.XTest))
      expl.yPredDecoded = expl.yPred

      # Visualize with mocks (should not raise).
      expl.VisualizeExplanations(instanceIndex=0, categoryToExplain="all", noOfRecords=10, noOfFeatures=2)

      # Assert storage path exists and some plot calls occurred.
      self.assertTrue(os.path.isdir(expl.storagePath))
      self.assertTrue(mockPlt.savefig.called or mockPlt.figure.called)


if (__name__ == "__main__"):
  unittest.main()
