import unittest
import torch
import numpy as np
from HMB.PyTorchSegmentationLosses import (
  DiceLoss,
  JaccardLoss,
  FocalLoss,
  GeneralizedDiceLoss,
)


class TestPyTorchSegmentationLosses(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)

  def test_dice_loss_binary_masks(self):
    logits = torch.randn(2, 1, 16, 16)
    targets = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss_fn = DiceLoss()
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_focal_loss_binary_seg(self):
    logits = torch.randn(2, 1, 16, 16)
    targets = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_jaccard_iou_loss_binary(self):
    logits = torch.randn(1, 1, 8, 8)
    targets = (torch.rand(1, 1, 8, 8) > 0.5).float()
    loss_fn = JaccardLoss()
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_generalized_dice_multiclass(self):
    logits = torch.randn(2, 3, 8, 8)
    # Build one-hot targets
    targets_idx = torch.randint(0, 3, (2, 8, 8), dtype=torch.long)
    targets = torch.zeros(2, 3, 8, 8)
    for c in range(3):
      targets[:, c] = (targets_idx == c).float()
    loss_fn = GeneralizedDiceLoss()
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
  unittest.main()
