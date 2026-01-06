import unittest
import os
import tempfile
import torch
import torch.nn as nn
from HMB.PyTorchHelper import (
  SaveModel,
  LoadModel,
  SavePyTorchDict,
  LoadPyTorchDict,
  SaveCheckpoint,
  LoadCheckpoint,
  GetOptimizer,
  MixupFn,
  MixupCriterion,
  ExponentialMovingAverage,
)


class TinyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(4, 2)

  def forward(self, x):
    return self.fc(x)


class TestPyTorchHelperUtilities(unittest.TestCase):
  def test_save_and_load_model_cpu(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "model.pth")
      model = TinyModel()
      SaveModel(model, path)
      self.assertTrue(os.path.exists(path))
      # Modify weights and then load back.
      for p in model.parameters():
        with torch.no_grad():
          p.add_(1.0)
      LoadModel(model, path, device="cpu")
      # Just ensure no exception and model is on CPU.
      self.assertEqual(next(model.parameters()).device.type, "cpu")

  def test_save_and_load_dict(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "state.pth")
      state = {"epoch": 1, "weights": TinyModel().state_dict()}
      SavePyTorchDict(state, path)
      loaded = LoadPyTorchDict(path, device="cpu")
      self.assertIsInstance(loaded, dict)
      self.assertIn("epoch", loaded)

  def test_load_dict_file_not_found(self):
    loaded = LoadPyTorchDict("does_not_exist.pth", device="cpu")
    self.assertIsNone(loaded)

  def test_checkpoint_roundtrip(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "chk.pth.tar")
      model = TinyModel()
      opt = GetOptimizer(model, optimizerType="adam", learningRate=0.01)
      SaveCheckpoint(model, opt, filename=path)
      self.assertTrue(os.path.exists(path))
      # Change LR via load.
      new_lr = 0.001
      LoadCheckpoint(path, model, opt, lr=new_lr, device=torch.device("cpu"))
      for pg in opt.param_groups:
        self.assertAlmostEqual(pg["lr"], new_lr)

  def test_get_optimizer_invalid(self):
    model = TinyModel()
    with self.assertRaises(ValueError):
      _ = GetOptimizer(model, optimizerType="not_an_opt")

  def test_get_optimizer_variants(self):
    model = TinyModel()
    self.assertIsNotNone(GetOptimizer(model, optimizerType="sgd", learningRate=0.1))
    self.assertIsNotNone(GetOptimizer(model, optimizerType="adam", learningRate=0.001))
    self.assertIsNotNone(GetOptimizer(model, optimizerType="adamw", learningRate=0.0005))

  def test_checkpoint_multiple_param_groups(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "multi_chk.pth.tar")
      model = TinyModel()
      # Create optimizer with two param groups.
      params = [
        {"params": [p for p in model.fc.parameters()], "lr": 0.01},
        {"params": [torch.nn.Parameter(torch.randn(2, requires_grad=True))], "lr": 0.02}
      ]
      opt = torch.optim.SGD(params, lr=0.01)
      SaveCheckpoint(model, opt, filename=path)
      self.assertTrue(os.path.exists(path))
      LoadCheckpoint(path, model, opt, lr=0.005, device=torch.device("cpu"))
      for pg in opt.param_groups:
        self.assertAlmostEqual(pg["lr"], 0.005)

  # ========== New tests: Mixup and EMA ==========
  def test_mixup_fn_alpha_zero_returns_one_hot(self):
    # Create simple inputs and integer targets
    inputs = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    mixed_inputs, mixed_targets = MixupFn(inputs, targets, alpha=0.0, numClasses=3)
    # Inputs should be unchanged
    self.assertTrue(torch.allclose(mixed_inputs, inputs))
    # Targets should be one-hot with correct shape and dtype
    self.assertEqual(mixed_targets.shape, (4, 3))
    self.assertTrue(torch.all((mixed_targets.sum(dim=1) - 1.0).abs() < 1e-6))

  def test_mixup_fn_alpha_positive_shapes_and_sum_to_one(self):
    inputs = torch.randn(6, 5)
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    mixed_inputs, mixed_targets = MixupFn(inputs, targets, alpha=0.4, numClasses=3)
    self.assertEqual(mixed_inputs.shape, inputs.shape)
    self.assertEqual(mixed_targets.shape[0], inputs.shape[0])
    # Each soft target row should sum to ~1
    sums = mixed_targets.sum(dim=1)
    self.assertTrue(torch.all((sums - 1.0).abs() < 1e-5))

  def test_mixup_criterion_outputs_scalar(self):
    logits = torch.randn(4, 3)
    soft = torch.softmax(torch.randn(4, 3), dim=1)
    loss = MixupCriterion(logits, soft)
    self.assertTrue(torch.is_tensor(loss))
    self.assertEqual(loss.dim(), 0)

  def test_ema_load_state_without_module_raises(self):
    model = TinyModel()
    ema = ExponentialMovingAverage(model)
    sd = ema.state_dict()
    ema2 = ExponentialMovingAverage()
    with self.assertRaises(RuntimeError):
      ema2.load_state_dict(sd)


if (__name__ == "__main__"):
  unittest.main()
