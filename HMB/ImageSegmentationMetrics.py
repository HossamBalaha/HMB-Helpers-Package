'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Aug 17th, 2025
# Last Modification Date: Aug 17th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import numpy as np


def ComputeIoU(preds, targets, smooth=1):
  '''
  Compute the Intersection over Union (IoU) metric.

  Args:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    IoU value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Flatten the input and target tensors into 1D arrays for computation.
  intersection = (preds * targets).sum()
  # Calculate the union of the predicted and target tensors.
  union = preds.sum() + targets.sum() - intersection
  # Compute the IoU using the formula.
  iou = (intersection + smooth) / (union + smooth)
  return iou


def ComputeDice(preds, targets, smooth=1):
  '''
  Compute the Dice coefficient.

  Args:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Dice coefficient value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Flatten the input and target tensors into 1D arrays for computation.
  intersection = (preds * targets).sum()
  # Calculate the Dice coefficient using the formula.
  dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
  return dice


def ComputeF1Score(preds, targets, smooth=1):
  '''
  Compute the F1 score.

  Args:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    F1 score value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Flatten the input and target tensors into 1D arrays for computation.
  intersection = (preds * targets).sum()
  # Calculate precision and recall.
  precision = intersection / (preds.sum() + smooth)
  recall = intersection / (targets.sum() + smooth)
  # Compute the F1 score using the formula.
  f1Score = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
  return f1Score


if __name__ == "__main__":
  # Simulate a model output tensor with random values (e.g., logits).
  predictions = np.random.rand(1, 1, 256, 256).astype(np.float32)  # Simulated model output.
  # Normalize the predictions to the range [0, 1].
  predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

  # Simulate a ground truth mask tensor with binary values (0 or 1).
  groundTruthMask = np.random.randint(0, 2, size=(1, 1, 256, 256)).astype(np.float32)

  # Compute metrics.
  iou = ComputeIoU(predictions, groundTruthMask)
  dice = ComputeDice(predictions, groundTruthMask)
  f1Score = ComputeF1Score(predictions, groundTruthMask)

  print(f"IoU: {iou}, Dice: {dice}, F1 Score: {f1Score}")

  # Compute metrics.
  iou = ComputeIoU(groundTruthMask, groundTruthMask)
  dice = ComputeDice(groundTruthMask, groundTruthMask)
  f1Score = ComputeF1Score(groundTruthMask, groundTruthMask)

  print(f"IoU: {iou}, Dice: {dice}, F1 Score: {f1Score}")
