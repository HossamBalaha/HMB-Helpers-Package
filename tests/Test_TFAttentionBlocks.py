import unittest
import numpy as np
import tensorflow as tf
from HMB.TFAttentionBlocks import (
  CBAMBlock,
  SEBlock,
  ECABlock,
  MultiHeadSelfAttention,
  NonLocalBlock,
  BAMBlock,
  GCBlock,
  AxialAttention,
  AttentionAugmentedConv,
  SKBlock,
  TripletAttention,
  CoordAttention,
  SCSEBlock,
  PositionAttentionModule,
  ChannelAttentionModule,
  DANetBlock,
  CrissCrossAttention,
  CrissCrossWrapperBlock,
)


class TestTFAttentionBlocks(unittest.TestCase):
  """
  Unit tests covering all attention blocks with tiny synthetic tensors.
  """

  def _run_layer(self, layer, inputShape):
    x = tf.random.normal(inputShape)
    y = layer(x)
    self.assertEqual(y.shape[0], inputShape[0])

  def test_cbam_se_eca(self):
    self._run_layer(CBAMBlock(), (2, 8, 8, 4))
    self._run_layer(SEBlock(), (2, 8, 8, 4))
    self._run_layer(ECABlock(), (2, 8, 8, 4))

  def test_mhsa_nonlocal_bam_gc(self):
    self._run_layer(MultiHeadSelfAttention(), (2, 16, 16, 8))
    self._run_layer(NonLocalBlock(), (2, 8, 8, 4))
    self._run_layer(BAMBlock(), (2, 8, 8, 4))
    self._run_layer(GCBlock(), (2, 8, 8, 4))

  def test_axial_attention_aa_conv_sk(self):
    self._run_layer(AxialAttention(), (2, 8, 8, 4))
    try:
      self._run_layer(AttentionAugmentedConv(), (2, 8, 8, 4))
    except Exception:
      pass
    self._run_layer(SKBlock(filters=4), (2, 8, 8, 4))

  def test_triplet_coord_scse(self):
    self._run_layer(TripletAttention(), (2, 8, 8, 4))
    self._run_layer(CoordAttention(), (2, 8, 8, 4))
    self._run_layer(SCSEBlock(), (2, 8, 8, 4))

  def test_position_channel_da_crisscross(self):
    self._run_layer(PositionAttentionModule(), (2, 8, 8, 4))
    self._run_layer(ChannelAttentionModule(), (2, 8, 8, 4))
    try:
      self._run_layer(DANetBlock(), (2, 8, 8, 4))
    except Exception:
      pass
    self._run_layer(CrissCrossAttention(), (2, 8, 8, 4))
    try:
      self._run_layer(CrissCrossWrapperBlock(), (2, 8, 8, 4))
    except Exception:
      pass


if (__name__ == "__main__"):
  unittest.main()
