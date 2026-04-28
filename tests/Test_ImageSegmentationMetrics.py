import unittest
import numpy as np
from HMB.ImageSegmentationMetrics import (
  ComputeIoU,
  ComputeDice,
  ComputePixelAccuracy,
  ComputePrecision,
  ComputeRecall,
  ComputeSpecificity,
  ComputeFPR,
  ComputeFNR,
  ComputeF1Score,
  ComputeMeanAveragePrecision,
  ComputeHausdorffDistance,
  ComputeBoundaryF1Score,
  ComputeMatthewsCorrelationCoefficient,
)


class TestImageSegmentationMetrics(unittest.TestCase):
  '''
  Unit tests for ImageSegmentationMetrics.
  Tests IoU, Dice, PixelAccuracy, Precision, Recall, Specificity, FPR, FNR, F1, mAP, Hausdorff.
  '''

  def setUp(self):
    # Create small synthetic predictions and targets
    self.predsBinary = np.array([
      [[0, 1, 0], [1, 1, 0], [0, 0, 1]]
    ], dtype=np.float32)
    self.targetsBinary = np.array([
      [[0, 1, 1], [1, 0, 0], [0, 0, 1]]
    ], dtype=np.float32)
    # Add channel and batch dims: (N, C, H, W) by expanding
    self.predsBinary = self.predsBinary[np.newaxis, np.newaxis, ...]
    self.targetsBinary = self.targetsBinary[np.newaxis, np.newaxis, ...]

    # Soft predictions in [0,1]
    self.predsSoft = np.array([
      [[0.2, 0.8, 0.3], [0.9, 0.7, 0.4], [0.1, 0.2, 0.95]]
    ], dtype=np.float32)
    self.predsSoft = self.predsSoft[np.newaxis, np.newaxis, ...]

  # ========== IoU ==========

  def test_iou_binary(self):
    iou = ComputeIoU(self.predsBinary, self.targetsBinary, iouType="binary")
    self.assertGreaterEqual(iou, 0.0)
    self.assertLessEqual(iou, 1.0)

  def test_iou_soft(self):
    iou = ComputeIoU(self.predsSoft, self.targetsBinary, iouType="soft")
    self.assertGreaterEqual(iou, 0.0)
    self.assertLessEqual(iou, 1.0)

  def test_iou_weighted_requires_weights(self):
    with self.assertRaises(ValueError):
      ComputeIoU(self.predsBinary, self.targetsBinary, iouType="weighted")

  def test_iou_weighted_valid(self):
    weight = np.array([1.0])
    iou = ComputeIoU(self.predsBinary, self.targetsBinary, iouType="weighted", weight=weight)
    self.assertGreaterEqual(iou, 0.0)
    self.assertLessEqual(iou, 1.0)

  def test_iou_empty_masks(self):
    preds = np.zeros_like(self.targetsBinary)
    targets = np.zeros_like(self.targetsBinary)
    iou = ComputeIoU(preds, targets, iouType="binary")
    self.assertGreaterEqual(iou, 0.0)
    self.assertLessEqual(iou, 1.0)

  def test_iou_full_overlap(self):
    preds = np.ones_like(self.targetsBinary)
    targets = np.ones_like(self.targetsBinary)
    iou = ComputeIoU(preds, targets, iouType="binary")
    self.assertGreaterEqual(iou, 0.99)

  def test_iou_mismatched_shapes(self):
    with self.assertRaises(Exception):
      _ = ComputeIoU(self.predsBinary[..., :-1], self.targetsBinary, iouType="binary")

  def test_iou_invalid_type(self):
    with self.assertRaises(ValueError):
      _ = ComputeIoU(self.predsBinary, self.targetsBinary, iouType="invalid")

  def test_iou_weighted_multiclass(self):
    # Simulate 2-class masks by stacking channels
    preds = np.stack([self.predsBinary.squeeze(), 1.0 - self.predsBinary.squeeze()], axis=0)
    targets = np.stack([self.targetsBinary.squeeze(), 1.0 - self.targetsBinary.squeeze()], axis=0)
    weight = np.array([0.7, 0.3])
    val = ComputeIoU(preds, targets, iouType="weighted", weight=weight)
    self.assertGreaterEqual(val, 0.0)
    self.assertLessEqual(val, 1.0)

  # ========== Dice ==========

  def test_dice_basic(self):
    dice = ComputeDice(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(dice, 0.0)
    self.assertLessEqual(dice, 1.0)

  def test_dice_perfect(self):
    dice = ComputeDice(self.targetsBinary, self.targetsBinary)
    self.assertGreaterEqual(dice, 0.99)

  def test_dice_empty(self):
    preds = np.zeros_like(self.targetsBinary)
    targets = np.zeros_like(self.targetsBinary)
    dice = ComputeDice(preds, targets)
    self.assertGreaterEqual(dice, 0.0)

  # ========== Pixel Accuracy ==========

  def test_pixel_accuracy_basic(self):
    acc = ComputePixelAccuracy(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(acc, 0.0)
    self.assertLessEqual(acc, 1.0)

  def test_pixel_accuracy_perfect(self):
    acc = ComputePixelAccuracy(self.targetsBinary, self.targetsBinary)
    self.assertEqual(acc, 1.0)

  def test_pixel_accuracy_mismatched_shapes(self):
    with self.assertRaises(Exception):
      _ = ComputePixelAccuracy(self.predsBinary[..., :-1], self.targetsBinary)

  # ========== Precision/Recall/Specificity/FPR ==========

  def test_precision_basic(self):
    prec = ComputePrecision(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(prec, 0.0)
    self.assertLessEqual(prec, 1.0)

  def test_precision_zero_division(self):
    preds = np.zeros_like(self.targetsBinary)
    prec = ComputePrecision(preds, self.targetsBinary)
    self.assertEqual(prec, 0.0)

  def test_recall_basic(self):
    rec = ComputeRecall(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(rec, 0.0)
    self.assertLessEqual(rec, 1.0)

  def test_recall_zero_division(self):
    preds = np.ones_like(self.targetsBinary) * 0.0
    rec = ComputeRecall(preds, np.zeros_like(self.targetsBinary))
    self.assertEqual(rec, 0.0)

  def test_specificity_basic(self):
    spec = ComputeSpecificity(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(spec, 0.0)
    self.assertLessEqual(spec, 1.0)

  def test_fpr_basic(self):
    fpr = ComputeFPR(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(fpr, 0.0)
    self.assertLessEqual(fpr, 1.0)

  def test_fpr_zero_division(self):
    preds = np.zeros_like(self.targetsBinary)
    fpr = ComputeFPR(preds, np.ones_like(self.targetsBinary))
    self.assertEqual(fpr, 0.0)

  def test_fnr_basic(self):
    fnr = ComputeFNR(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(fnr, 0.0)
    self.assertLessEqual(fnr, 1.0)

  def test_fnr_zero_division(self):
    preds = np.ones_like(self.targetsBinary)
    fnr = ComputeFNR(preds, np.zeros_like(self.targetsBinary))
    self.assertEqual(fnr, 0.0)

  def test_f1_score_consistency(self):
    f1 = ComputeF1Score(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(f1, 0.0)
    self.assertLessEqual(f1, 1.0)

  def test_boundary_f1_basic(self):
    val = ComputeBoundaryF1Score(self.predsBinary, self.targetsBinary)
    self.assertGreaterEqual(val, 0.0)
    self.assertLessEqual(val, 1.0)

  def test_mcc_basic(self):
    val = ComputeMatthewsCorrelationCoefficient(self.predsBinary, self.targetsBinary)
    self.assertTrue(np.isfinite(val))

  def test_map_batch(self):
    preds = np.concatenate([self.predsBinary, self.predsBinary], axis=0)  # batch of 2
    targets = np.concatenate([self.targetsBinary, self.targetsBinary], axis=0)
    m = ComputeMeanAveragePrecision(preds, targets)
    self.assertGreaterEqual(m, 0.0)
    self.assertLessEqual(m, 1.0)

  def test_hausdorff_simple_shapes(self):
    # Two small masks with a single pixel each
    preds = np.zeros((1, 1, 8, 8), dtype=np.float32)
    targets = np.zeros((1, 1, 8, 8), dtype=np.float32)
    preds[0, 0, 1, 1] = 1.0
    targets[0, 0, 6, 6] = 1.0
    d = ComputeHausdorffDistance(preds, targets)
    self.assertGreaterEqual(d, 0.0)

  def test_hausdorff_empty_masks(self):
    preds = np.zeros_like(self.targetsBinary)
    targets = np.zeros_like(self.targetsBinary)
    d = ComputeHausdorffDistance(preds, targets)
    self.assertTrue(np.isinf(d))


if __name__ == "__main__":
  unittest.main()
