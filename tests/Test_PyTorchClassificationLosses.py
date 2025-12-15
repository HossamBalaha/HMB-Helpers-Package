import unittest
import torch
import numpy as np
from HMB.PyTorchClassificationLosses import (
  CrossEntropyLossWrapper,
  LabelSmoothingCrossEntropy,
  FocalLoss,
  BinaryFocalLoss,
)


class TestPyTorchClassificationLosses(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)

  def test_label_smoothing_cross_entropy_basic(self):
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    loss_fn = LabelSmoothingCrossEntropy(labelSmoothing=0.1)
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_cross_entropy_wrapper_with_class_weights(self):
    logits = torch.randn(5, 4)
    targets = torch.tensor([0, 1, 1, 2, 3], dtype=torch.long)
    weights = torch.tensor([1.0, 2.0, 1.0, 0.5], dtype=torch.float32)
    loss_fn = CrossEntropyLossWrapper(classWeight=weights, reductionMode="mean")
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_cross_entropy_ignore_index_like(self):
    # CrossEntropyLossWrapper does not expose ignore_index; simulate by masking
    logits = torch.randn(3, 2)
    targets = torch.tensor([0, -100, 1], dtype=torch.long)
    mask = targets != -100
    loss_fn = CrossEntropyLossWrapper()
    loss = loss_fn.forward(logits[mask], targets[mask])
    self.assertTrue(torch.isfinite(loss))

  def test_focal_loss_multiclass_basic(self):
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_focal_loss_multiclass_class_vector_alpha(self):
    logits = torch.randn(6, 3)
    targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    alpha = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))

  def test_binary_focal_loss_basic(self):
    logits = torch.randn(6)
    targets = torch.randint(0, 2, (6,)).float()
    loss_fn = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn.forward(logits, targets)
    self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
  unittest.main()
