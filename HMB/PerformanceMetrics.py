'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Aug 13th, 2025
# Last Modification Date: Aug 13th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import numpy as np


def CalculatePerformanceMetrics(
  confMatrix,
  eps=1e-10,
  addWeightedAverage=False,
):
  """
  Calculate performance metrics from a confusion matrix.

  Parameters:
  confMatrix (list of list): Confusion matrix as a nested list.
  eps (float): Small value to avoid division by zero.
  addWeightedAverage (bool): Whether to include weighted averages in the output.

  Returns:
  dict: A dictionary containing performance metrics including:
        - True Positives (TP)
        - False Positives (FP)
        - False Negatives (FN)
        - True Negatives (TN)
        - Macro Precision
        - Macro Recall
        - Macro F1
        - Macro Accuracy
        - Macro Specificity
        - Micro Precision
        - Micro Recall
        - Micro F1
        - Micro Accuracy
        - Micro Specificity
        - Weights
        - Weighted Precision
        - Weighted Recall
        - Weighted F1
        - Weighted Accuracy
        - Weighted Specificity
  """
  # Convert the confusion matrix to a NumPy array for easier manipulation.
  confMatrix = np.array(confMatrix)

  # Check if the confusion matrix is for binary classification.
  noOfClasses = confMatrix.shape[0]
  if (noOfClasses > 2):
    # Calculate True Positives (TP) as the diagonal elements of the confusion matrix.
    TP = np.diag(confMatrix)
    # Calculate False Positives (FP) as the sum of each column minus the TP.
    FP = np.sum(confMatrix, axis=0) - TP
    # Calculate False Negatives (FN) as the sum of each row minus the TP.
    FN = np.sum(confMatrix, axis=1) - TP
    # Calculate True Negatives (TN) as the total sum of the matrix minus TP, FP, and FN.
    TN = np.sum(confMatrix) - (TP + FP + FN)
  else:
    # For binary classification, the confusion matrix is a 2x2 matrix.
    # Unravel the confusion matrix to get the TP, FP, FN, and TN.
    # The order of the elements is TN, FP, FN, TP.
    TN, FP, FN, TP = confMatrix.ravel()

  # Avoid division by zero by adding a small epsilon value.
  TP = TP + eps
  FP = FP + eps
  FN = FN + eps
  TN = TN + eps

  # Create a dictionary to hold the calculated performance metrics and
  # the TP, FP, FN, TN vectors.
  metrics = {
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
  }

  # Calculate precision using macro averaging: mean of TP / (TP + FP) for each class.
  precision = np.mean(TP / (TP + FP))
  # Calculate recall using macro averaging: mean of TP / (TP + FN) for each class.
  recall = np.mean(TP / (TP + FN))
  # Calculate F1 score using macro averaging: harmonic mean of precision and recall.
  f1 = 2 * precision * recall / (precision + recall)
  # Calculate accuracy using macro averaging: sum of TP and TN divided by the total sum of the matrix.
  accuracy = np.mean(TP + TN) / np.sum(confMatrix)
  # Calculate specificity using macro averaging: mean of TN / (TN + FP) for each class.
  specificity = np.mean(TN / (TN + FP))

  metrics.update({
    "Macro Precision"  : precision,
    "Macro Recall"     : recall,
    "Macro F1"         : f1,
    "Macro Accuracy"   : accuracy,
    "Macro Specificity": specificity,
  })

  # If requested, calculate the macro average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity) / 5.0
    metrics.update({
      "Macro Average": avg,
    })

  # Calculate precision using micro averaging: sum of TP divided by the sum of TP and FP.
  precision = np.sum(TP) / np.sum(TP + FP)
  # Calculate recall using micro averaging: sum of TP divided by the sum of TP and FN.
  recall = np.sum(TP) / np.sum(TP + FN)
  # Calculate F1 score using micro averaging: harmonic mean of precision and recall.
  f1 = 2 * precision * recall / (precision + recall)
  # Calculate accuracy using micro averaging: sum of TP and TN divided by TP, TN, FP, and FN.
  accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
  # Calculate specificity using micro averaging: sum of TN divided by the sum of TN and FP.
  specificity = np.sum(TN) / np.sum(TN + FP)

  metrics.update({
    "Micro Precision"  : precision,
    "Micro Recall"     : recall,
    "Micro F1"         : f1,
    "Micro Accuracy"   : accuracy,
    "Micro Specificity": specificity,
  })

  # If requested, calculate the micro average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity) / 5.0
    metrics.update({
      "Micro Average": avg,
    })

  # Calculate the number of samples per class by summing the rows of the confusion matrix.
  samples = np.sum(confMatrix, axis=1)

  # Calculate the weights for each class as the proportion of samples in that class.
  weights = samples / np.sum(confMatrix)

  # Calculate precision using weighted averaging: sum of precision per class multiplied by weights.
  precision = np.sum(TP / (TP + FP) * weights)
  # Calculate recall using weighted averaging: sum of recall per class multiplied by weights.
  recall = np.sum(TP / (TP + FN) * weights)
  # Calculate F1 score using weighted averaging: harmonic mean of weighted precision and recall.
  f1 = 2 * precision * recall / (precision + recall)
  # Calculate accuracy using weighted averaging: sum of TP and TN divided by the total sum of the matrix.
  accuracy = np.sum((TP + TN) * weights) / np.sum(confMatrix)
  # Calculate specificity using weighted averaging: sum of specificity per class multiplied by weights.
  specificity = np.sum(TN / (TN + FP) * weights)

  metrics.update({
    "Weights"             : weights,
    "Weighted Precision"  : precision,
    "Weighted Recall"     : recall,
    "Weighted F1"         : f1,
    "Weighted Accuracy"   : accuracy,
    "Weighted Specificity": specificity,
  })

  # If requested, calculate the weighted average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity) / 5.0
    metrics.update({
      "Weighted Average": avg,
    })

  return metrics


if __name__ == "__main__":
  # Example usage of the CalculatePerformanceMetrics function.
  confMatrix = [
    [50, 2, 1],
    [5, 45, 0],
    [0, 3, 47]
  ]

  metrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True)

  for key, value in metrics.items():
    print(f"{key}: {np.round(value, 4)}")
