import unittest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch

from HMB.YOLOHelper import EvaluateAndSaveYoloClassifications


class TestYOLOHelper(unittest.TestCase):
  """Unit tests for YOLOHelper.EvaluateAndSaveYoloClassifications using mocks and temp dirs."""

  def test_weights_missing(self):
    with tempfile.TemporaryDirectory() as baseDir:
      # Create minimal dataset structure (no weights created)
      datasetPath = os.path.join(baseDir, "dataset")
      os.makedirs(os.path.join(datasetPath, "val"), exist_ok=True)
      # Ensure runs/results folder exists so summary can be written
      os.makedirs(os.path.join(baseDir, "results"), exist_ok=True)

      summary = EvaluateAndSaveYoloClassifications(
        baseDir=baseDir,
        datasetPath=datasetPath,
        runsDir="results",
        targetModels=["ymock"],
        trialNum=1,
      )

      # The weights file does not exist -> each category for the model should report weights_missing
      self.assertIn("ymock", summary)
      for cat in ["val", "test", "train"]:
        self.assertIn(cat, summary["ymock"])
        self.assertIsNone(summary["ymock"][cat]["csv"])
        self.assertEqual(summary["ymock"][cat]["error"], "weights_missing")

  def test_load_failed(self):
    with tempfile.TemporaryDirectory() as baseDir:
      # Create dummy weights file at expected location
      weight_dir = os.path.join(baseDir, "runs", "classify", f"ymock-cls-1", "weights")
      os.makedirs(weight_dir, exist_ok=True)
      open(os.path.join(weight_dir, "best.pt"), "w").close()
      # Ensure runs/results folder exists so summary can be written
      os.makedirs(os.path.join(baseDir, "results"), exist_ok=True)

      # Patch YOLO to raise on instantiation
      with patch("HMB.YOLOHelper.YOLO", side_effect=Exception("load error")):
        datasetPath = os.path.join(baseDir, "dataset")
        os.makedirs(os.path.join(datasetPath, "val"), exist_ok=True)

        summary = EvaluateAndSaveYoloClassifications(
          baseDir=baseDir,
          datasetPath=datasetPath,
          runsDir="results",
          targetModels=["ymock"],
          trialNum=1,
        )

      # Loading failed -> should report load_failed and include exception text
      self.assertIn("ymock", summary)
      for cat in ["val", "test", "train"]:
        self.assertIn(cat, summary["ymock"])
        self.assertIsNone(summary["ymock"][cat]["csv"])
        self.assertEqual(summary["ymock"][cat]["error"], "load_failed")
        self.assertIn("exception", summary["ymock"][cat])

  def test_successful_evaluation_with_mocked_model(self):
    with tempfile.TemporaryDirectory() as baseDir:
      # Create dummy weights file at expected location
      weight_dir = os.path.join(baseDir, "runs", "classify", f"ymock-cls-1", "weights")
      os.makedirs(weight_dir, exist_ok=True)
      open(os.path.join(weight_dir, "best.pt"), "w").close()

      # Prepare a minimal dataset with one image for class 'cat'
      datasetPath = os.path.join(baseDir, "dataset")
      # Create class folders and a dummy image for each category so CSVs are non-empty
      for cat in ["val", "test", "train"]:
        class_dir = os.path.join(datasetPath, cat, "cat")
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, "img1.jpg")
        with open(img_path, "wb") as f:
          f.write(b"\xff\xd8\xff")
      # Ensure runs/results folder exists so summary can be written
      os.makedirs(os.path.join(baseDir, "results"), exist_ok=True)

      # Build a mock YOLO model that returns predictable probabilities
      class MockPredProbs:
        def __init__(self, arr):
          self.data = type("_", (), {"cpu": lambda self: type("__", (), {"numpy": lambda self: arr})()})()
          self.top1 = int(np.argmax(arr))
          self.values = [float(arr[self.top1])]

      class MockPred:
        def __init__(self, arr):
          self.probs = MockPredProbs(arr)

      class MockModel:
        def __init__(self, names):
          # names should be a dict like {0: 'cat', 1: 'dog'}
          self.names = names

        def __call__(self, imgPath, imgsz=None, verbose=False):
          # return a single-prediction object with higher probability for index 1
          arr = np.array([0.1, 0.9])
          return [MockPred(arr)]

      def fake_YOLO(*args, **kwargs):
        # Return a model whose names correspond to indexes 0 and 1
        return MockModel({0: "cat", 1: "dog"})

      # Provide a JSON dump that can handle numpy types to avoid TypeError during summary write
      def safe_json_dump(obj, fp, indent=4):
        def default(o):
          try:
            # numpy arrays -> list
            if hasattr(o, "tolist"):
              return o.tolist()
            # numpy scalar -> Python scalar
            if isinstance(o, np.generic):
              return o.item()
          except Exception:
            pass
          return str(o)

        fp.write(json.dumps(obj, default=default, indent=indent))

      with patch("HMB.YOLOHelper.YOLO", new=fake_YOLO), patch("HMB.YOLOHelper.json.dump", new=safe_json_dump):
        summary = EvaluateAndSaveYoloClassifications(
          baseDir=baseDir,
          datasetPath=datasetPath,
          runsDir="results",
          targetModels=["ymock"],
          trialNum=1,
        )

      # The evaluation should have produced a CSV for 'val' and metrics
      self.assertIn("ymock", summary)
      self.assertIn("val", summary["ymock"])
      csv_path = summary["ymock"]["val"]["csv"]
      self.assertIsNotNone(csv_path)
      self.assertTrue(os.path.exists(csv_path))
      # Confusion matrix and metrics entries should be present
      self.assertIn("cm", summary["ymock"]["val"])
      self.assertIn("metrics", summary["ymock"]["val"])

  def test_prediction_fallback_prob_attribute(self):
    """Ensure EvaluateAndSaveYoloClassifications handles pred[0].prob (fallback path)."""
    with tempfile.TemporaryDirectory() as baseDir:
      # Create dummy weights and dataset
      weight_dir = os.path.join(baseDir, "runs", "classify", f"ymock-cls-1", "weights")
      os.makedirs(weight_dir, exist_ok=True)
      open(os.path.join(weight_dir, "best.pt"), "w").close()

      datasetPath = os.path.join(baseDir, "dataset")
      for cat in ["val", "test", "train"]:
        class_dir = os.path.join(datasetPath, cat, "cat")
        os.makedirs(class_dir, exist_ok=True)
        with open(os.path.join(class_dir, "img1.jpg"), "wb") as f:
          f.write(b"\xff\xd8\xff")
      os.makedirs(os.path.join(baseDir, "results"), exist_ok=True)

      # Mock model that exposes .prob instead of .probs
      class MockPredProb:
        def __init__(self, arr):
          # provide .data.cpu().numpy() chain
          self.prob = type("P", (), {"data": type("D", (), {"cpu": lambda self: type("N", (), {"numpy": lambda self: arr})()}), "top1": int(np.argmax(arr)), "values": [float(arr[int(np.argmax(arr))])]})()

      class MockPred:
        def __init__(self, arr):
          self.prob = MockPredProb(arr)

      class MockModel:
        def __init__(self, names):
          self.names = names

        def __call__(self, imgPath, imgsz=None, verbose=False):
          arr = np.array([0.8, 0.2])
          return [MockPred(arr)]

      def fake_YOLO(*args, **kwargs):
        return MockModel({0: "cat", 1: "dog"})

      # safe JSON dump to handle numpy types
      def safe_json_dump(obj, fp, indent=4):
        def default(o):
          try:
            if hasattr(o, "tolist"):
              return o.tolist()
            if isinstance(o, np.generic):
              return o.item()
          except Exception:
            pass
          return str(o)
        fp.write(json.dumps(obj, default=default, indent=indent))

      with patch("HMB.YOLOHelper.YOLO", new=fake_YOLO), patch("HMB.YOLOHelper.json.dump", new=safe_json_dump):
        summary = EvaluateAndSaveYoloClassifications(
          baseDir=baseDir,
          datasetPath=datasetPath,
          runsDir="results",
          targetModels=["ymock"],
          trialNum=1,
        )

      self.assertIn("ymock", summary)
      self.assertIn("val", summary["ymock"])
      self.assertTrue(os.path.exists(summary["ymock"]["val"]["csv"]))

  def test_train_multiple_yolo_classifiers_mocked(self):
    """Test TrainMultipleYoloClassifiers with a mocked YOLO that simulates training/validation/save."""
    from HMB.YOLOHelper import TrainMultipleYoloClassifiers
    with tempfile.TemporaryDirectory() as baseDir:
      datasetPath = os.path.join(baseDir, "dataset")
      os.makedirs(os.path.join(datasetPath, "train", "cat"), exist_ok=True)
      # make a tiny image so nothing complains if accessed
      with open(os.path.join(datasetPath, "train", "cat", "img1.jpg"), "wb") as f:
        f.write(b"\xff\xd8\xff")

      os.makedirs(os.path.join(baseDir, "results"), exist_ok=True)

      class FakeYOLO:
        def __init__(self, *args, **kwargs):
          self.names = {0: "cat"}

        def train(self, *args, **kwargs):
          # no-op training
          return None

        def val(self, *args, **kwargs):
          # return object with top1 and top5
          return type("M", (), {"top1": 0.11, "top5": 0.22})()

        def save(self, path):
          # create the file to emulate saving in Keras format
          os.makedirs(os.path.dirname(path), exist_ok=True)
          with open(path, "w") as f:
            f.write("fake-model")

        def export(self, *args, **kwargs):
          return None

      with patch("HMB.YOLOHelper.YOLO", new=FakeYOLO):
        # call the training function; keep exportOnnx False to skip optional export
        TrainMultipleYoloClassifiers(
          datasetPath=datasetPath,
          baseDir=baseDir,
          runsDir="results",
          targetModels=["ymock"],
          epochs=1,
          batchSize=1,
          inputShape=(32, 32),
          trialNum=1,
          exportOnnx=False,
        )

      expOutputDir = os.path.join(baseDir, "results", f"ymock-cls-1")
      top1_path = os.path.join(expOutputDir, "Top1-Metrics.txt")
      top5_path = os.path.join(expOutputDir, "Top5-Metrics.txt")
      model_path = os.path.join(expOutputDir, "model.keras")

      self.assertTrue(os.path.exists(top1_path))
      self.assertTrue(os.path.exists(top5_path))
      self.assertTrue(os.path.exists(model_path))
      # Check contents include the numeric metric string
      with open(top1_path, "r") as f:
        self.assertIn("0.11", f.read())
      with open(top5_path, "r") as f:
        self.assertIn("0.22", f.read())


if (__name__ == "__main__"):
  unittest.main()
