import numpy as np


# Define a function to compute Intersection over Union (IoU) metric.
def ComputeIoU(preds, targets, smooth=1.0):
  r'''
  Compute the Intersection over Union (IoU) metric.

  .. math::
    IoU=\frac{|\mathrm{Prediction}\cap \mathrm{Ground\ Truth}|+\mathrm{smooth}}{|\mathrm{Prediction}\cup\mathrm{Ground\ Truth}|+\mathrm{smooth}}

  where:
    - :math:`|\mathrm{Prediction}\cap \mathrm{Ground\ Truth}|` is the intersection of the predicted and ground truth tensors.
    - :math:`|\mathrm{Prediction}\cup\mathrm{Ground\ Truth}|` is the union of the predicted and ground truth tensors.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    IoU value.
   '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate intersection between prediction and target.
  intersection = (preds * targets).sum()
  # Calculate the union of the predicted and target tensors.
  union = preds.sum() + targets.sum() - intersection
  # Calculate final IoU value with smoothing.
  iou = (intersection + smooth) / (union + smooth)
  # Return the computed IoU value.
  return iou


# Define a function to compute the Dice coefficient.
def ComputeDice(preds, targets, smooth=1.0):
  r'''
  Compute the Dice coefficient.

  .. math::
    Dice = \frac{2 \times |Prediction \cap Ground\ Truth| + smooth}{|Prediction| + |Ground\ Truth| + smooth}

  where:
    - :math:`|Prediction \cap Ground\ Truth|` is the intersection of the predicted and ground truth tensors.
    - :math:`|Prediction|` is the sum of the predicted tensor.
    - :math:`|Ground\ Truth|` is the sum of the ground truth tensor.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Dice coefficient value.
  '''

  # Threshold predictions at 0.5 to obtain binary mask.
  preds = np.float32(preds > 0.5)  # Convert logits to binary predictions.
  # Calculate intersection between prediction and target.
  intersection = (preds * targets).sum()
  # Calculate the Dice coefficient using the formula.
  dice = (
    (2.0 * intersection + smooth) /
    (preds.sum() + targets.sum() + smooth)
  )
  # Return the computed Dice coefficient.
  return dice


# Define a function to compute the F1 score.
def ComputeF1Score(preds, targets, smooth=1.0):
  r'''
  Compute the F1 score.

  .. math::
    F1 = \frac{2 \times Precision \times Recall + smooth}{Precision + Recall + smooth}

  where:
    - Precision is the ratio of true positives to the sum of true positives and false positives.
    - Recall is the ratio of true positives to the sum of true positives and false negatives.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    F1 score value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate intersection between prediction and target.
  intersection = (preds * targets).sum()
  # Compute precision from intersection.
  precision = intersection / (preds.sum() + smooth)
  # Compute recall from intersection.
  recall = intersection / (targets.sum() + smooth)
  # Compute the F1 score using the formula.
  f1Score = (
    (2.0 * precision * recall + smooth) /
    (precision + recall + smooth)
  )
  # Return the computed F1 score.
  return f1Score


# Define a function to compute Pixel Accuracy.
def ComputePixelAccuracy(preds, targets):
  r'''
  Compute the pixel accuracy metric.

  .. math::
    Pixel\ Accuracy = \frac{Number\ of\ Correct\ Pixels}{Total\ Number\ of\ Pixels}

  where:
    - Number of Correct Pixels is the sum of pixels where predictions match targets.
    - Total Number of Pixels is the product of the dimensions of the predicted tensor.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    Pixel accuracy value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of correct pixels.
  correct = (preds == targets).sum()
  # Calculate the total number of pixels.
  total = np.prod(preds.shape)
  # Return the pixel accuracy value.
  return correct / total


# Define a function to compute Precision.
def ComputePrecision(preds, targets, smooth=1.0):
  r'''
  Compute the precision metric.

  .. math::
    Precision = \frac{TP + smooth}{TP + FP + smooth}

  where:
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`smooth` is a small constant to avoid division by zero.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Precision value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  # Return the precision value.
  return (TP + smooth) / (TP + FP + smooth)


# Define a function to compute Recall.
def ComputeRecall(preds, targets, smooth=1.0):
  r'''
  Compute the recall metric.

  .. math::
    Recall = \frac{TP + smooth}{TP + FN + smooth}

  where:
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).
    - :math:`smooth` is a small constant to avoid division by zero.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Recall value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Calculate the number of false negatives.
  FN = ((1.0 - preds) * targets).sum()
  # Return the recall value.
  return (TP + smooth) / (TP + FN + smooth)


# Define a function to compute Specificity.
def ComputeSpecificity(preds, targets, smooth=1.0):
  r'''
  Compute the specificity metric.

  .. math::
    Specificity = \frac{TN + smooth}{TN + FP + smooth}

  where:
    - :math:`TN` is the number of true negatives (predicted negative and actually negative).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`smooth` is a small constant to avoid division by zero.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Specificity value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true negatives.
  TN = ((1.0 - preds) * (1.0 - targets)).sum()
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  # Return the specificity value.
  return (TN + smooth) / (TN + FP + smooth)


# Define a function to compute False Positive Rate (FPR).
def ComputeFPR(preds, targets, smooth=1.0):
  r'''
  Compute the false positive rate (FPR).

  .. math::
    FPR = \frac{FP + smooth}{FP + TN + smooth}

  where:
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`TN` is the number of true negatives (predicted negative and actually negative).
    - :math:`smooth` is a small constant to avoid division by zero.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    FPR value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  # Calculate the number of true negatives.
  TN = ((1.0 - preds) * (1.0 - targets)).sum()
  # Return the false positive rate value.
  return (FP + smooth) / (FP + TN + smooth)


# Define a function to compute False Negative Rate (FNR).
def ComputeFNR(preds, targets, smooth=1.0):
  r'''
  Compute the false negative rate (FNR).

  .. math::
    FNR = \frac{FN + smooth}{FN + TP + smooth}

  where:
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`smooth` is a small constant to avoid division by zero.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    FNR value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of false negatives.
  FN = ((1.0 - preds) * targets).sum()
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Return the false negative rate value.
  return (FN + smooth) / (FN + TP + smooth)


# Define a function to compute mean Average Precision (mAP) for binary masks.
def ComputeMeanAveragePrecision(preds, targets, smooth=1.0):
  r'''
  Compute the mean average precision (mAP) for binary masks.

  .. math::
    mAP = \frac{1}{N} \times \sum_{i=1}^{N} Precision_i

  where:
    - :math:`Precision_i` is the precision for the i-th image in the batch.
    - :math:`N` is the total number of images in the batch.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    mAP value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)

  # For binary mask, mAP is just the precision averaged over all images (if batch).
  if (preds.ndim > 2):
    precisions = [
      ComputePrecision(p, t, smooth)
      for p, t in zip(preds, targets)
    ]
    return np.mean(precisions)
  else:
    return ComputePrecision(preds, targets, smooth)


def ComputeHausdorffDistance(preds, targets):
  r'''
  Compute the Hausdorff distance between predicted and ground truth masks.

  .. math::
    H(A, B) = \max\{\sup_{a \in A} \inf_{b \in B} d(a, b), \sup_{b \in B} \inf_{a \in A} d(a, b)\}

  where:
    - :math:`A` is the set of points in the predicted mask.
    - :math:`B` is the set of points in the ground truth mask.
    - :math:`d(a, b)` is the Euclidean distance between points `a` and `b`.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    Hausdorff distance value.
  '''

  from scipy.spatial.distance import directed_hausdorff

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  targets = np.float32(targets > 0.5)

  # Get the coordinates of the predicted and target points.
  predPoints = np.argwhere(preds)
  targetPoints = np.argwhere(targets)

  # If either of the sets of points is empty, return infinity.
  if (len(predPoints) == 0 or len(targetPoints) == 0):
    return float("inf")

  # Compute directed Hausdorff distance in both directions.
  hd1 = directed_hausdorff(predPoints, targetPoints)[0]
  hd2 = directed_hausdorff(targetPoints, predPoints)[0]

  # Return the maximum of the two directed distances.
  return max(hd1, hd2)


# Define a function to compute Boundary F1 Score (BF Score).
def ComputeBoundaryF1Score(preds, targets, dilationRatio=0.02):
  r'''
  Compute the Boundary F1 Score (BF Score).

  .. math::
    BF = \frac{2 \times Precision_{boundary} \times Recall_{boundary}}{Precision_{boundary} + Recall_{boundary}}

  where:
    - :math:`Precision_{boundary}` is the precision of the predicted boundary.
    - :math:`Recall_{boundary}` is the recall of the predicted boundary.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    dilationRatio: Ratio for boundary dilation.

  Returns:
    Boundary F1 Score value.
  '''

  from scipy.ndimage import binary_dilation

  preds = np.float32(preds > 0.5)
  targets = np.float32(targets > 0.5)
  # Get boundaries by subtracting eroded mask from mask
  from scipy.ndimage import binary_erosion
  def GetBoundary(mask, dilationRatio):
    h, w = mask.shape[-2], mask.shape[-1]
    dilation = int(np.round(dilationRatio * np.sqrt(h * w)))
    eroded = binary_erosion(mask, iterations=dilation)
    boundary = mask - eroded
    return boundary

  predBoundary = GetBoundary(preds.squeeze(), dilationRatio)
  targetBoundary = GetBoundary(targets.squeeze(), dilationRatio)
  # Dilate boundaries.
  predDil = binary_dilation(predBoundary, iterations=1)
  targetDil = binary_dilation(targetBoundary, iterations=1)
  # Precision and recall for boundaries
  precision = (predBoundary * targetDil).sum() / (predBoundary.sum() + 1e-7)
  recall = (targetBoundary * predDil).sum() / (targetBoundary.sum() + 1e-7)
  bfScore = 2 * precision * recall / (precision + recall + 1e-7)
  return bfScore


def ComputeMatthewsCorrelationCoefficient(preds, targets, smooth=1.0):
  r'''
  Compute the Matthews Correlation Coefficient (MCC).

  .. math::
    MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}

  where
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`TN` is the number of true negatives (predicted negative and actually negative).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    MCC value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Calculate the number of true negatives.
  TN = ((1.0 - preds) * (1.0 - targets)).sum()
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  # Calculate the number of false negatives.
  FN = ((1.0 - preds) * targets).sum()
  # Calculate the numerator for MCC.
  numerator = TP * TN - FP * FN
  # Calculate the denominator for MCC.
  denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + smooth
  # Return the MCC value.
  return numerator / denominator


def ComputeCohensKappa(preds, targets, smooth=1.0):
  r'''
  Compute Cohen's Kappa metric.

  .. math::
    \kappa = \frac{p_o - p_e}{1 - p_e}

  where:
    - :math:`p_o` is the observed agreement between predictions and targets.
    - :math:`p_e` is the expected agreement by chance.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Cohen's Kappa value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate observed agreement.
  po = (preds == targets).sum() / np.prod(preds.shape)
  # Calculate expected agreement.
  pyes = (
    (preds.sum() / np.prod(preds.shape)) *
    (targets.sum() / np.prod(targets.shape))
  )
  pno = (
    ((1.0 - preds).sum() / np.prod(preds.shape)) *
    ((1.0 - targets).sum() / np.prod(targets.shape))
  )
  pe = pyes + pno
  # Return Cohen's Kappa value.
  return (po - pe) / (1.0 - pe + smooth)


def ComputeBalancedAccuracy(preds, targets, smooth=1.0):
  r'''
  Compute the balanced accuracy metric.

  .. math::
    Balanced\ Accuracy = \frac{Recall + Specificity}{2}

  where:
    - :math:`Recall` is the true positive rate.
    - :math:`Specificity` is the true negative rate.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    Balanced accuracy value.
  '''

  # Calculate recall.
  recall = ComputeRecall(preds, targets, smooth)
  # Calculate specificity.
  specificity = ComputeSpecificity(preds, targets, smooth)
  # Return balanced accuracy value.
  return (recall + specificity) / 2.0


def ComputeMeanSurfaceDistance(preds, targets):
  r'''
  Compute the Mean Surface Distance (MSD) between predicted and ground truth masks.

  .. math::
    MSD = \frac{1}{|S_P|} \sum_{p \in S_P} \min_{q \in S_T} d(p, q)

  where:
    - :math:`S_P` is the set of points on the predicted mask boundary.
    - :math:`S_T` is the set of points on the ground truth mask boundary
    - :math:`d(p, q)` is the Euclidean distance between points `p` and `q`.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    MSD value.
  '''

  from scipy.ndimage import binary_erosion
  from scipy.spatial.distance import cdist

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  targets = np.float32(targets > 0.5)

  # Get boundaries.
  predBoundary = preds - binary_erosion(preds)
  targetBoundary = targets - binary_erosion(targets)

  # Get coordinates of boundary points.
  predPoints = np.argwhere(predBoundary)
  targetPoints = np.argwhere(targetBoundary)

  # If either boundary is empty, return infinity.
  if (len(predPoints) == 0 or len(targetPoints) == 0):
    return float('inf')

  # Compute distances from each predicted boundary point to all target boundary points.
  distances = cdist(predPoints, targetPoints)
  # Calculate mean of minimum distances.
  msd = np.mean(np.min(distances, axis=1))
  # Return MSD value.
  return msd


def ComputeAverageSymmetricSurfaceDistance(preds, targets):
  r'''
  Compute the Average Symmetric Surface Distance (ASSD) between predicted and ground truth masks.

  .. math::
    ASSD = \frac{MSD(P, T) + MSD(T, P)}{2}

  where:
    - :math:`MSD(P, T)` is the Mean Surface Distance from prediction to target.
    - :math:`MSD(T, P)` is the Mean Surface Distance from target to prediction.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    ASSD value.
  '''

  # Calculate MSD from prediction to target.
  msd1 = ComputeMeanSurfaceDistance(preds, targets)
  # Calculate MSD from target to prediction.
  msd2 = ComputeMeanSurfaceDistance(targets, preds)
  # Return ASSD value.
  return (msd1 + msd2) / 2.0


def ComputeVolumetricOverlapError(preds, targets, smooth=1.0):
  r'''
  Compute the Volumetric Overlap Error (VOE).

  .. math::
    VOE = 1 - IoU

  where:
    - :math:`IoU` is the Intersection over Union value.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.

  Returns:
    VOE value.
  '''

  # Calculate IoU value.
  iou = ComputeIoU(preds, targets, smooth)
  # Return VOE value.
  return 1.0 - iou


def ComputeGlobalConsistencyError(preds, targets):
  r'''
  Compute the Global Consistency Error (GCE).

  .. math::
    GCE = \frac{1}{N} \times \sum_{i=1}^{N} \min(E(S_1, S_2, p_i), E(S_2, S_1, p_i))

  where:
    - :math:`E(S_1, S_2, p_i)` is the error for pixel `p_i` in the predicted mask compared to the ground truth mask.
    - :math:`N` is the total number of pixels in the mask.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    GCE value.
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  targets = np.float32(targets > 0.5)
  # Flatten arrays.
  predsFlat = preds.flatten()
  targetsFlat = targets.flatten()
  # Calculate error for each pixel.
  error1 = np.sum(predsFlat != targetsFlat) / len(predsFlat)
  error2 = np.sum(targetsFlat != predsFlat) / len(targetsFlat)
  # Return GCE value.
  return min(error1, error2)


# Main block to test the metric functions with simulated data.
if __name__ == "__main__":
  # Simulate a model output tensor with random values (e.g., logits).
  predictions = np.random.rand(1, 1, 256, 256).astype(np.float32)  # Simulated model output tensor.
  # Normalize the predictions to the range [0, 1].
  predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())  # Normalize predictions.

  # Simulate a ground truth mask tensor with binary values (0 or 1).
  groundTruthMask = np.random.randint(0, 2, size=(1, 1, 256, 256)).astype(np.float32)  # Simulated ground truth mask.

  # Calculate IoU for predictions.
  iou = ComputeIoU(predictions, groundTruthMask)  # Compute metrics.
  # Calculate Dice coefficient for predictions.
  dice = ComputeDice(predictions, groundTruthMask)  # Compute metrics.
  # Calculate F1 score for predictions.
  f1Score = ComputeF1Score(predictions, groundTruthMask)  # Compute metrics.

  # Print the computed metrics for predictions.
  print(f"IoU: {iou}, Dice: {dice}, F1 Score: {f1Score}")  # Print the computed metrics.

  # Calculate IoU for ground truth.
  iou = ComputeIoU(groundTruthMask, groundTruthMask)  # Compute metrics.
  # Calculate Dice coefficient for ground truth.
  dice = ComputeDice(groundTruthMask, groundTruthMask)  # Compute metrics.
  # Calculate F1 score for ground truth.
  f1Score = ComputeF1Score(groundTruthMask, groundTruthMask)  # Compute metrics.

  # Print the computed metrics for ground truth.
  print(f"IoU: {iou}, Dice: {dice}, F1 Score: {f1Score}")  # Print the computed metrics for ground truth.

  # Calculate Pixel Accuracy for predictions.
  pixelAccuracy = ComputePixelAccuracy(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Pixel Accuracy.
  print(f"Pixel Accuracy: {pixelAccuracy}")  # Print the computed Pixel Accuracy.

  # Calculate Precision for predictions.
  precision = ComputePrecision(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Precision.
  print(f"Precision: {precision}")  # Print the computed Precision.

  # Calculate Recall for predictions.
  recall = ComputeRecall(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Recall.
  print(f"Recall: {recall}")  # Print the computed Recall.

  # Calculate Specificity for predictions.
  specificity = ComputeSpecificity(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Specificity.
  print(f"Specificity: {specificity}")  # Print the computed Specificity.

  # Calculate False Positive Rate (FPR) for predictions.
  fpr = ComputeFPR(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed FPR.
  print(f"FPR: {fpr}")  # Print the computed FPR.

  # Calculate False Negative Rate (FNR) for predictions.
  fnr = ComputeFNR(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed FNR.
  print(f"FNR: {fnr}")  # Print the computed FNR.

  # Calculate mean Average Precision (mAP) for predictions.
  mapScore = ComputeMeanAveragePrecision(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed mAP.
  print(f"mAP: {mapScore}")  # Print the computed mAP.

  # Calculate Hausdorff Distance for predictions.
  hd = ComputeHausdorffDistance(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Hausdorff Distance.
  print(f"Hausdorff Distance: {hd}")  # Print the computed Hausdorff Distance.

  # Calculate Boundary F1 Score for predictions.
  bfScore = ComputeBoundaryF1Score(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Boundary F1 Score.
  print(f"Boundary F1 Score: {bfScore}")  # Print the computed Boundary F1 Score.

  # Calculate Matthews Correlation Coefficient for predictions.
  mcc = ComputeMatthewsCorrelationCoefficient(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed MCC.
  print(f"MCC: {mcc}")  # Print the computed MCC.

  # Calculate Cohen's Kappa for predictions.
  kappa = ComputeCohensKappa(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Cohen's Kappa.
  print(f"Cohen's Kappa: {kappa}")  # Print the computed Cohen's Kappa.

  # Calculate Balanced Accuracy for predictions.
  balancedAcc = ComputeBalancedAccuracy(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Balanced Accuracy.
  print(f"Balanced Accuracy: {balancedAcc}")  # Print the computed Balanced Accuracy.

  # Calculate Mean Surface Distance for predictions.
  msd = ComputeMeanSurfaceDistance(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed Mean Surface Distance.
  print(f"MSD: {msd}")  # Print the computed Mean Surface Distance.

  # Calculate Average Symmetric Surface Distance for predictions.
  assd = ComputeAverageSymmetricSurfaceDistance(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed ASSD.
  print(f"ASSD: {assd}")  # Print the computed ASSD.

  # Calculate Volumetric Overlap Error for predictions.
  voe = ComputeVolumetricOverlapError(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed VOE.
  print(f"VOE: {voe}")  # Print the computed VOE.

  # Calculate Global Consistency Error for predictions.
  gce = ComputeGlobalConsistencyError(predictions, groundTruthMask)  # Compute metrics.
  # Print the computed GCE.
  print(f"GCE: {gce}")  # Print the computed GCE.
