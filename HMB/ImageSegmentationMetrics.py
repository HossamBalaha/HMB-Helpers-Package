import numpy as np


def ComputeIoU(preds, targets, smooth=1.0, iouType="binary", weight=None):
  r'''
  Compute the Intersection over Union (IoU) metric.

  .. math::
    IoU = \frac{|Prediction \cap Ground\ Truth| + smooth}{|Prediction \cup Ground\ Truth| + smooth}
  where:
    - :math:`|Prediction \cap Ground\ Truth|` is the intersection of the predicted and ground truth tensors.
    - :math:`|Prediction \cup Ground\ Truth|` is the union of the predicted and ground truth tensors.
    - :math:`smooth` is a small constant to avoid division by zero.

  .. note::
    The `iouType` parameter determines how the IoU is computed:
      - `binary`: Threshold predictions at 0.5 to obtain binary masks.
      - `soft`: Use raw predictions for soft IoU.
      - `weighted`: Use class weights for weighted IoU (requires `weight` parameter).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    smooth: Smoothing factor to avoid division by zero.
    iouType: Type of IoU to compute ("binary", "soft", or "weighted").
    weight: Class weights for multiclass IoU.

  Returns:
    float: IoU value.

  Raises:
    ValueError: If `iouType` is not one of "binary", "soft", or "weighted".
    ValueError: If `weight` is not provided when `iouType` is "weighted".

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    iou = ism.ComputeIoU(preds, targets, iouType="binary")
    print(f"IoU: {iou}")
    iouSoft = ism.ComputeIoU(preds, targets, iouType="soft")
    print(f"Soft IoU: {iouSoft}")
    weight = np.array([0.7, 0.3])  # Example weights for two classes.
    iouWeighted = ism.ComputeIoU(preds, targets, iouType="weighted", weight=weight)
    print(f"Weighted IoU: {iouWeighted}")
  '''

  if (iouType == "binary"):
    # Threshold predictions at 0.5 to obtain binary mask.
    preds = np.float32(preds > 0.5)
    # Calculate intersection and union between prediction and target.
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
  elif (iouType == "soft"):
    # Use raw predictions for soft IoU.
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
  elif (iouType == "weighted"):
    if (weight is None):
      raise ValueError("Weight must be provided for weighted IoU.")
    intersection = (weight * preds * targets).sum()
    union = (
      (weight * preds).sum() +
      (weight * targets).sum() -
      intersection
    )
  else:
    raise ValueError("Invalid iouType. Must be 'binary', 'soft', or 'weighted'.")
  # Calculate the IoU using the formula.
  iou = (intersection + smooth) / (union + smooth)
  # Return the computed IoU value.
  return iou


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
    float: Dice coefficient value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    dice = ism.ComputeDice(preds, targets)
    print(f"Dice: {dice}")
  '''

  # Threshold predictions at 0.5 to obtain binary mask.
  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate intersection between prediction and target.
  intersection = (preds * targets).sum()
  # Calculate the Dice coefficient using the formula.
  dice = (
    (2.0 * intersection + smooth) /
    (preds.sum() + targets.sum() + smooth)
  )
  # Return the computed Dice coefficient.
  return dice


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
    float: Pixel accuracy value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    acc = ism.ComputePixelAccuracy(preds, targets)
    print(f"Pixel Accuracy: {acc}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  targets = np.float32(targets)
  # Calculate the number of correct pixels.
  correct = np.sum(preds == targets)
  # Calculate the total number of pixels.
  total = np.prod(preds.shape)
  # Return the pixel accuracy value.
  return correct / total


def ComputePrecision(preds, targets):
  r'''
  Compute the precision metric.

  .. math::
    Precision = \frac{TP}{TP + FP}

  where:
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: Precision value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    precision = ism.ComputePrecision(preds, targets)
    print(f"Precision: {precision}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  if ((TP + FP) == 0):
    return 0.0
  # Return the precision value.
  return TP / (TP + FP)


def ComputeRecall(preds, targets):
  r'''
  Compute the recall metric.

  .. math::
    Recall = \frac{TP}{TP + FN}

  where:
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: Recall value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    recall = ism.ComputeRecall(preds, targets)
    print(f"Recall: {recall}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Calculate the number of false negatives.
  FN = ((1.0 - preds) * targets).sum()
  if ((TP + FN) == 0):
    return 0.0
  # Return the recall value.
  return TP / (TP + FN)


def ComputeSpecificity(preds, targets):
  r'''
  Compute the specificity metric.

  .. math::
    Specificity = \frac{TN}{TN + FP}

  where:
    - :math:`TN` is the number of true negatives (predicted negative and actually negative).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: Specificity value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    specificity = ism.ComputeSpecificity(preds, targets)
    print(f"Specificity: {specificity}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true negatives.
  TN = ((1.0 - preds) * (1.0 - targets)).sum()
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  if ((TN + FP) == 0):
    return 0.0
  # Return the specificity value.
  return TN / (TN + FP)


def ComputeFPR(preds, targets):
  r'''
  Compute the false positive rate (FPR).

  .. math::
    FPR = \frac{FP}{FP + TN}

  where:
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`TN` is the number of true negatives (predicted negative and actually negative).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: FPR value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    fpr = ism.ComputeFPR(preds, targets)
    print(f"FPR: {fpr}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  # Calculate the number of true negatives.
  TN = ((1.0 - preds) * (1.0 - targets)).sum()
  if ((FP + TN) == 0):
    return 0.0
  # Return the false positive rate value.
  return FP / (FP + TN)


def ComputeFNR(preds, targets):
  r'''
  Compute the false negative rate (FNR).

  .. math::
    FNR = \frac{FN}{FN + TP}

  where:
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).
    - :math:`TP` is the number of true positives (predicted positive and actually positive).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: FNR value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    fnr = ism.ComputeFNR(preds, targets)
    print(f"FNR: {fnr}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of false negatives.
  FN = ((1.0 - preds) * targets).sum()
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  if ((TP + FN) == 0):
    return 0.0
  # Return the false negative rate value.
  return FN / (FN + TP)


def ComputeF1Score(preds, targets):
  r'''
  Compute the F1 score.

  .. math::
    F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}

  where:
    - Precision is the ratio of true positives to the sum of true positives and false positives.
    - Recall is the ratio of true positives to the sum of true positives and false negatives.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: F1 score value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    f1 = ism.ComputeF1Score(preds, targets)
    print(f"F1 Score: {f1}")
  '''

  # Calculate precision and recall.
  precision = ComputePrecision(preds, targets)
  recall = ComputeRecall(preds, targets)
  if ((precision + recall) == 0):
    return 0.0
  elif (np.isnan(precision) or np.isnan(recall)):
    return 0.0
  # Calculate the F1 Score using the formula.
  f1 = (2.0 * precision * recall) / (precision + recall)
  # Return the computed F1 Score.
  return f1


def ComputeMeanAveragePrecision(preds, targets):
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

  Returns:
    float: mAP value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(2, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(2, 1, 256, 256))
    mapScore = ism.ComputeMeanAveragePrecision(preds, targets)
    print(f"mAP: {mapScore}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)

  # For binary mask, mAP is just the precision averaged
  # over all images (if batch).
  if (preds.ndim > 2):
    precisions = [
      ComputePrecision(p, t)
      for p, t in zip(preds, targets)
    ]
    return np.mean(precisions)
  else:
    return ComputePrecision(preds, targets)


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
    float: Hausdorff distance value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    hd = ism.ComputeHausdorffDistance(preds, targets)
    print(f"Hausdorff Distance: {hd}")
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

  # Compute directed Hausdorff distances.
  d1 = directed_hausdorff(predPoints, targetPoints)[0]
  d2 = directed_hausdorff(targetPoints, predPoints)[0]

  # Return the Hausdorff distance value.
  return max(d1, d2)


def ComputeBoundaryF1Score(preds, targets, dilationRatio=0.02, eps=1e-7):
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
    eps: Small constant to avoid division by zero.

  Returns:
    float: Boundary F1 Score value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    bfScore = ism.ComputeBoundaryF1Score(preds, targets)
    print(f"Boundary F1 Score: {bfScore}")
  '''

  from scipy.ndimage import binary_dilation

  preds = np.float32(preds > 0.5)
  targets = np.float32(targets > 0.5)
  # Get boundaries by subtracting eroded mask from mask.
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
  # Precision and recall for boundaries.
  precision = (
    (predBoundary * targetDil).sum() /
    (predBoundary.sum() + eps)
  )
  recall = (
    (targetBoundary * predDil).sum() /
    (targetBoundary.sum() + eps)
  )
  bfScore = 2.0 * precision * recall / (precision + recall + eps)
  return bfScore


def ComputeMatthewsCorrelationCoefficient(preds, targets):
  r'''
  Compute the Matthews Correlation Coefficient (MCC).

  .. math::
    MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP) \times (TP + FN) \times (TN + FP) \times (TN + FN)}}

  where
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`TN` is the number of true negatives (predicted negative and actually negative).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: MCC value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    mcc = ism.ComputeMatthewsCorrelationCoefficient(preds, targets)
    print(f"Matthews Correlation Coefficient: {mcc}")
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
  # Calculate the numerator and denominator for MCC.
  num = (TP * TN) - (FP * FN)
  den = np.sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
  )
  if (den == 0):
    return 0.0
  # Return the Matthews Correlation Coefficient value.
  return num / den


def ComputeCohensKappa(preds, targets):
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

  Returns:
    float: Cohen's Kappa value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    kappa = ism.ComputeCohensKappa(preds, targets)
    print(f"Cohen's Kappa: {kappa}")
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
  # Total number of pixels.
  total = TP + TN + FP + FN
  if (total == 0):
    return 0.0
  # Calculate observed accuracy.
  Po = (TP + TN) / total
  # Calculate expected accuracy.
  Pe = (
    ((TP + FP) * (TP + FN) +
     (FN + TN) * (FP + TN)) /
    (total * total)
  )
  if ((1.0 - Pe) == 0):
    return 0.0
  # Return the Cohen's Kappa value.
  return (Po - Pe) / (1.0 - Pe)


def ComputeBalancedAccuracy(preds, targets):
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

  Returns:
    float: Balanced accuracy value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    balancedAcc = ism.ComputeBalancedAccuracy(preds, targets)
    print(f"Balanced Accuracy: {balancedAcc}")
  '''

  # Calculate recall.
  recall = ComputeRecall(preds, targets)
  # Calculate specificity.
  specificity = ComputeSpecificity(preds, targets)
  # Return balanced accuracy value.
  return (recall + specificity) / 2.0


def ComputeMeanSurfaceDistance(preds, targets):
  r'''
  Compute the Mean Surface Distance (MSD) between predicted and ground truth masks.

  .. math::
    MSD = \frac{1}{|S_P|} \times \sum_{p \in S_P} \min_{q \in S_T} d(p, q)

  where:
    - :math:`S_P` is the set of points on the predicted mask boundary.
    - :math:`S_T` is the set of points on the ground truth mask boundary
    - :math:`d(p, q)` is the Euclidean distance between points `p` and `q`.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: MSD value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    msd = ism.ComputeMeanSurfaceDistance(preds, targets)
    print(f"Mean Surface Distance: {msd}")
  '''

  from scipy.ndimage import distance_transform_edt

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  targets = np.float32(targets > 0.5)

  # Get the coordinates of the predicted and target points.
  predPoints = np.argwhere(preds)
  targetPoints = np.argwhere(targets)

  # If either of the sets of points is empty, return infinity.
  if (len(predPoints) == 0 or len(targetPoints) == 0):
    return float("inf")

  # Compute distance transform for target points.
  dtTarget = distance_transform_edt(1 - targets)
  # For each predicted point, find the distance to the nearest target point.
  distances = [dtTarget[tuple(coord)] for coord in predPoints]
  # Return the mean surface distance value.
  return np.mean(distances)


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
    float: ASSD value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    assd = ism.ComputeAverageSymmetricSurfaceDistance(preds, targets)
    print(f"Average Symmetric Surface Distance: {assd}")
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
    float: VOE value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    voe = ism.ComputeVolumetricOverlapError(preds, targets)
    print(f"Volumetric Overlap Error: {voe}")
  '''

  # Calculate IoU value.
  iou = ComputeIoU(preds, targets, smooth=smooth, iouType="binary")
  # Return VOE value.
  return 1.0 - iou


def ComputeGlobalConsistencyError(preds, targets):
  r'''
  Compute the Global Consistency Error (GCE).

  .. math::
    GCE = \frac{1}{N} \times \sum_{i=1}^{N} \min(E(S_1, S_2, p_i), E(S_2, S_1, p_i))

  where:
    - :math:`E(S_1, S_2, p_i)` is the error for pixel `p_i` in the predicted
      mask compared to the ground truth mask.
    - :math:`N` is the total number of pixels in the mask.

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: GCE value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    gce = ism.ComputeGlobalConsistencyError(preds, targets)
    print(f"Global Consistency Error: {gce}")
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


def ComputeTversky(preds, targets, alpha=0.5):
  r'''
  Compute the Tversky index metric.

  .. math::
    Tversky = \frac{TP}{TP + \alpha \times FP + (1-\alpha) \times FN}

  where:
    - :math:`TP` is the number of true positives (predicted positive and actually positive).
    - :math:`FP` is the number of false positives (predicted positive but actually negative).
    - :math:`FN` is the number of false negatives (predicted negative but actually positive).
    - :math:`\alpha` is the weight for false positives (default 0.5).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    alpha: Weight for false positives (default 0.5).

  Returns:
    float: Tversky index value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    tversky = ism.ComputeTversky(preds, targets, alpha=0.7)
    print(f"Tversky Index: {tversky}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the number of true positives.
  TP = (preds * targets).sum()
  # Calculate the number of false positives.
  FP = (preds * (1.0 - targets)).sum()
  # Calculate the number of false negatives.
  FN = ((1.0 - preds) * targets).sum()
  beta = 1.0 - alpha
  # Return the Tversky index value.
  return TP / (TP + alpha * FP + beta * FN)


def ComputeFocalTverskyLoss(preds, targets, alpha=0.5, gamma=np.round(4 / 3.0, 5)):
  r'''
  Compute the Focal Tversky index metric.

  .. math::
    FocalTversky = (1 - Tversky)^{1/\gamma}

  where:
    - :math:`Tversky` is the Tversky index (see ComputeTversky).
    - :math:`\gamma` is the focusing parameter (default 4/3).
    - :math:`\alpha` is the weight for false positives (default 0.5).

  Parameters:
    preds: Predicted tensor (logits).
    targets: Ground truth tensor (binary mask).
    alpha: Weight for false positives (default 0.5).
    gamma: Focusing parameter (default 4/3).

  Returns:
    float: Focal Tversky index value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    focalTversky = ism.ComputeFocalTverskyLoss(preds, targets, alpha=0.7, gamma=1.33)
    print(f"Focal Tversky Index: {focalTversky}")
  '''

  # Calculate the Tversky index.
  tversky = ComputeTversky(preds, targets, alpha)
  # Return the Focal Tversky value.
  return np.power((1.0 - tversky), 1.0 / gamma)


def ComputeFocalLoss(preds, targets, beta=0.5, gamma=2.0, eps=1e-7):
  r'''
  Compute the Focal Loss for binary segmentation.

  .. math::
    FL = -\beta \times (1 - p)^{\gamma} \times y \times \log(p) - (1 - \beta) \times p^{\gamma} \times (1 - y) \times \log(1 - p)

  where:
    - :math:`p` is the predicted probability.
    - :math:`y` is the ground truth label.
    - :math:`\beta` balances positive/negative examples.
    - :math:`\gamma` focuses on hard examples.
    - :math:`\epsilon` is a small constant to avoid log(0).

  Parameters:
    preds: Predicted tensor (probabilities).
    targets: Ground truth tensor (binary mask).
    beta: Balance parameter (default 0.5).
    gamma: Focusing parameter (default 2.0).
    eps: Small constant to avoid log(0).

  Returns:
    float: Focal loss value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    loss = ism.ComputeFocalLoss(preds, targets)
    print(f"Focal Loss: {loss}")
  '''

  # Clip predictions to avoid log(0) error.
  preds = np.clip(preds, eps, 1.0 - eps)
  # Calculate the focal loss.
  loss = (
    beta * np.power((1.0 - preds), gamma) *
    targets * np.log(preds)
    + (1.0 - beta) * np.power(preds, gamma) *
    (1.0 - targets) * np.log(1.0 - preds)
  )
  # Return the mean focal loss value.
  return -np.mean(loss)


def ComputeComboLoss(preds, targets, alpha=0.5, beta=0.5, smooth=1.0, eps=1e-7):
  r'''
  Compute the Combo Loss, combining weighted cross-entropy and Dice loss.

  .. math::
    ComboLoss = \beta \times CE + (1 - \beta) \times -\log(Dice)

  where:
    - :math:`CE` is the weighted cross-entropy.
    - :math:`Dice` is the Dice coefficient.
    - :math:`\alpha` weights positive/negative classes in CE.
    - :math:`\beta` balances CE and Dice.

  Parameters:
    preds: Predicted tensor (probabilities).
    targets: Ground truth tensor (binary mask).
    alpha: Weight for positive class in CE (default 0.5).
    beta: Balance between CE and Dice (default 0.5).
    smooth: Smoothing factor for Dice (default 1.0).
    eps: Small constant to avoid log(0).

  Returns:
    float: Combo loss value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    loss = ism.ComputeComboLoss(preds, targets)
    print(f"Combo Loss: {loss}")
  '''

  # Calculate the Dice coefficient.
  dice = float(ComputeDice(preds, targets, smooth))
  # Clip predictions to avoid log(0) error.
  preds = np.clip(preds, eps, 1.0 - eps)
  # Calculate the weighted cross-entropy.
  tLnP = alpha * targets * np.log(preds)
  pLnT = (1.0 - alpha) * (1.0 - targets) * np.log(1.0 - preds)
  out = -tLnP + pLnT
  weightedCE = np.mean(out)
  # Calculate the Combo Loss.
  loss = beta * weightedCE - (1.0 - beta) * np.log(dice)
  return loss


def ComputeTanimotoLoss(preds, targets):
  r'''
  Compute the Tanimoto Loss for binary segmentation.

  .. math::
    Tanimoto = 1 - \frac{\sum p t}{\sum p^2 + \sum t^2 - \sum p t}

  where:
    - :math:`p` is the predicted mask.
    - :math:`t` is the ground truth mask.

  Parameters:
    preds: Predicted tensor (logits or probabilities).
    targets: Ground truth tensor (binary mask).

  Returns:
    float: Tanimoto loss value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    loss = ism.ComputeTanimotoLoss(preds, targets)
    print(f"Tanimoto Loss: {loss}")
  '''

  # Convert logits to binary predictions.
  preds = np.float32(preds > 0.5)
  # Calculate the numerator and denominator for Tanimoto loss.
  num = (preds * targets).sum()
  den = (
    (preds ** 2).sum() +
    (targets ** 2).sum() -
    (preds * targets).sum()
  )
  # Return the Tanimoto loss value.
  return 1.0 - num / den


def ComputeMSELoss(preds, targets):
  r'''
  Compute the Mean Squared Error (MSE) loss.

  .. math::
    MSE = \frac{1}{N} \times \sum_{i=1}^N (p_i - t_i)^2

  where:
    - :math:`p_i` is the predicted value.
    - :math:`t_i` is the ground truth value.
    - :math:`N` is the number of elements.

  Parameters:
    preds: Predicted tensor.
    targets: Ground truth tensor.

  Returns:
    float: MSE loss value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.rand(1, 1, 256, 256)
    loss = ism.ComputeMSELoss(preds, targets)
    print(f"MSE Loss: {loss}")
  '''

  # Calculate the Mean Squared Error (MSE) loss.
  return np.mean((preds - targets) ** 2)


def ComputeBCELoss(preds, targets, smooth=1e-7):
  r'''
  Compute the Binary Cross-Entropy (BCE) loss.

  .. math::
    BCE = -\frac{1}{N} \times \sum_{i=1}^N [t_i \times \log(p_i) + (1 - t_i) \times \log(1 - p_i)]

  where:
    - :math:`p_i` is the predicted probability.
    - :math:`t_i` is the ground truth label.
    - :math:`N` is the number of elements.
    - :math:`smooth` is a small constant to avoid log(0).

  Parameters:
    preds: Predicted tensor (probabilities).
    targets: Ground truth tensor (binary mask).
    smooth: Small constant to avoid log(0) (default 1e-7).

  Returns:
    float: BCE loss value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    loss = ism.ComputeBCELoss(preds, targets)
    print(f"BCE Loss: {loss}")
  '''

  # Clip predictions to avoid log(0) error.
  preds = np.clip(preds, smooth, 1.0 - smooth)
  # Calculate the Binary Cross-Entropy (BCE) loss.
  bce = (
    -np.mean(
      targets * np.log(preds) +
      (1.0 - targets) * np.log(1.0 - preds)
    )
  )
  return bce


def ComputeHMBLoss(preds, targets):
  r'''
  Compute the HMB Loss, a weighted combination of multiple loss functions for segmentation.

  The HMB Loss (H-Loss), as suggested by Hossam (the author), introduces a weighted sum
  of various loss functions: Dice, IoU, MSE, BCE, Tversky, and Tanimoto losses [1]_.
  This idea is presented in a research article that can be accessed from:
  https://doi.org/10.1109/ACCESS.2024.3483661

  The HMB Loss combines:
    - MSE Loss (distance-based)
    - Dice Loss (region-based)
    - IoU Loss (region-based)
    - Tversky Loss (region-based)
    - BCE Loss (distribution-based)
    - Tanimoto Loss (distribution-based)

  Parameters:
    preds: Predicted tensor.
    targets: Ground truth tensor.

  Returns:
    float: Weighted average loss value.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.ImageSegmentationMetrics as ism
    preds = np.random.rand(1, 1, 256, 256)
    targets = np.random.randint(0, 2, size=(1, 1, 256, 256))
    loss = ism.ComputeHMBLoss(preds, targets)
    print(f"HMB Loss: {loss}")

  References
  ----------
  .. [1] Sharaby, I., Balaha, H. M., Alksas, A., Mahmoud, A., Abou El-Ghar, M., Khalil, A., ...
    & El-Baz, A. (2024). Artificial intelligence-based kidney segmentation with modified
    cycle-consistent generative adversarial network and appearance-based shape prior. IEEE Access.
    https://doi.org/10.1109/ACCESS.2024.3483661

  '''

  # Calculate individual loss components.
  mseLoss = ComputeMSELoss(preds, targets)  # Distance-based.
  diceLoss = 1.0 - ComputeDice(preds, targets)  # Region-based.
  iouLoss = 1.0 - ComputeIoU(preds, targets)  # Region-based.
  tverskyLoss = 1.0 - ComputeTversky(preds, targets)  # Region-based.
  bceLoss = ComputeBCELoss(preds, targets)  # Distribution-based.
  tanimotoLoss = ComputeTanimotoLoss(preds, targets)  # Distribution-based.

  # Define weights for each loss component.
  weights = np.array(
    [
      200,  # Weight for MSE Loss.
      50,  # Weight for Dice Loss.
      50,  # Weight for IoU Loss.
      50,  # Weight for Tversky Loss.
      25,  # Weight for BCE Loss.
      25,  # Weight for Tanimoto Loss.
    ]
  )

  # Normalize the weights to sum to 1.
  weights = weights / np.sum(weights)

  # Calculate the weighted average loss.
  avgLoss = (
    weights[0] * mseLoss
    + weights[1] * diceLoss
    + weights[2] * iouLoss
    + weights[3] * tverskyLoss
    + weights[4] * bceLoss
    + weights[5] * tanimotoLoss
  )

  return avgLoss


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
