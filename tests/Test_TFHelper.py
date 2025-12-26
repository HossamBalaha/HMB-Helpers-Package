import unittest
import os
import tempfile
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

from HMB.TFHelper import TFGradCam, SaveGradCamsForSamples


class DummyModel:
  def __init__(self):
    # Minimal Keras-like API.
    import tensorflow as tf
    inp = tf.keras.Input(shape=(None, None, 3))
    x = tf.keras.layers.Conv2D(4, (3, 3), activation="relu", name="conv")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    self.model = tf.keras.Model(inputs=inp, outputs=out)

  @property
  def layers(self):
    return self.model.layers

  @property
  def inputs(self):
    return self.model.inputs

  @property
  def output(self):
    return self.model.output

  def __call__(self, x, training=False):
    return self.model(x, training=training)

  def get_layer(self, name):
    return self.model.get_layer(name)


class TestTFHelper(unittest.TestCase):
  """
  Unit tests for TFHelper covering TFGradCam and SaveGradCamsForSamples using a tiny Keras model.
  """

  def setUp(self):
    self.dm = DummyModel()

  def test_tfgradcam_basic(self):
    # Create synthetic image tensor (1,H,W,3).
    img = np.random.rand(1, 32, 32, 3).astype(np.float32)
    heatmap = TFGradCam(self.dm, img, classIdx=0, lastConvLayerName="conv")
    self.assertTrue(isinstance(heatmap, np.ndarray))
    self.assertEqual(heatmap.ndim, 2)

  def test_tfgradcam_invalid_layer(self):
    img = np.random.rand(1, 32, 32, 3).astype(np.float32)
    with self.assertRaises(Exception):
      _ = TFGradCam(self.dm, img, classIdx=0, lastConvLayerName="not_a_layer")

  def test_tfgradcam_class_index_out_of_bounds(self):
    img = np.random.rand(1, 32, 32, 3).astype(np.float32)
    with self.assertRaises(Exception):
      _ = TFGradCam(self.dm, img, classIdx=99, lastConvLayerName="conv")

  def test_save_gradcams_for_samples(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create synthetic images.
      imgPaths = []
      for i in range(3):
        p = os.path.join(tmpdir, f"img_{i}.png")
        Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8)).save(p)
        imgPaths.append(p)
      # Run and verify files generated (overlay created).
      SaveGradCamsForSamples(self.dm, imgPaths, [0, 1], tmpdir, imgSize=(32, 32), lastConvLayerName="conv")
      outs = [f for f in os.listdir(tmpdir) if f.startswith("GradCAM_IDx")]
      self.assertGreaterEqual(len(outs), 1)

  def test_save_gradcams_empty_inputs(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(Exception):
        SaveGradCamsForSamples(self.dm, [], [], tmpdir, imgSize=(32, 32), lastConvLayerName="conv")


if (__name__ == "__main__"):
  unittest.main()
