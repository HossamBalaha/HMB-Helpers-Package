import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch
from HMB.AttentionMapsHelper import AttentionMapsVisualizer


class TestAttentionMapsHelper(unittest.TestCase):
  '''
  Unit tests for attention maps visualization internals using mocks to avoid heavy dependencies.
  '''

  def test_attention_maps_visualizer_init_with_mocks(self):
    # Prepare temp data folder with class subfolders and a dummy checkpoint file
    with tempfile.TemporaryDirectory() as tmpdir:
      dataFolder = os.path.join(tmpdir, "data")
      os.makedirs(dataFolder, exist_ok=True)
      for cls in ["A", "B"]:
        os.makedirs(os.path.join(dataFolder, cls), exist_ok=True)
      ckpt = os.path.join(tmpdir, "model.ckpt")
      open(ckpt, "w").close()
      # Mock TimmModel to avoid real loading
      with patch("HMB.AttentionMapsHelper.TimmModel", return_value=(MagicMock(), MagicMock(), MagicMock())):
        amv = AttentionMapsVisualizer(
          baseFolder=tmpdir,
          dataFolder=dataFolder,
          modelName="dummy-model",
          modelCheckpointPath=ckpt,
          modelType="Timm",
          size=224,
          device="cpu",
        )
        self.assertIsNotNone(amv)
        self.assertEqual(amv.numClasses, 2)
        self.assertEqual(sorted(amv.classes), ["A", "B"])


if __name__ == "__main__":
  unittest.main()
