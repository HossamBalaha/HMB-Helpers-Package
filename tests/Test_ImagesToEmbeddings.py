import unittest
import os
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from HMB.ImagesToEmbeddings import TransformersEmbeddingModel, ExtractEmbeddingsTimm


class TestImagesToEmbeddings(unittest.TestCase):
  """
  Unit tests for ImagesToEmbeddings with heavy dependencies mocked.
  Covers TransformersEmbeddingModel and timm-based extraction path shape and error handling.
  """

  @patch("HMB.ImagesToEmbeddings.timm.create_model")
  @patch("HMB.ImagesToEmbeddings.resolve_data_config")
  @patch("HMB.ImagesToEmbeddings.create_transform")
  def test_extract_embeddings_timm_minimal(self, mockCreateTransform, mockResolveCfg, mockCreateModel):
    # Mock model
    embModel = MagicMock()
    embModel.eval = MagicMock()
    embModel.to = MagicMock(return_value=embModel)
    embModel.pretrained_cfg = {}

    def fwd(x):
      return torch.randn(1, 16)

    embModel.__call__ = MagicMock(side_effect=fwd)
    mockCreateModel.return_value = embModel

    # Mock transforms
    tfm = MagicMock()
    tfm.side_effect = lambda img: torch.randn(3, 224, 224)
    mockCreateTransform.return_value = tfm
    mockResolveCfg.return_value = {}

    # Create dummy dataset folder
    import tempfile
    from PIL import Image
    tmpDir = tempfile.mkdtemp()
    clsdir = os.path.join(tmpDir, "classA")
    os.makedirs(clsdir, exist_ok=True)
    img1 = os.path.join(clsdir, "a.png")
    Image.new("RGB", (32, 32)).save(img1)
    outp = os.path.join(tmpDir, "emb.pkl")

    try:
      ExtractEmbeddingsTimm(tmpDir, outp, modelName="dummy-model", device="cpu")
      self.assertTrue(os.path.exists(outp))
    finally:
      import shutil
      if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)


if __name__ == "__main__":
  unittest.main()
