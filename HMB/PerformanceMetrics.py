'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Aug 13th, 2025
# Last Modification Date: Aug 19th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import numpy as np


def CalculatePerformanceMetrics(
  confMatrix,
  eps=1e-10,
  addWeightedAverage=False,
):
  '''
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
  '''

  # Convert the confusion matrix to a NumPy array for easier manipulation.
  confMatrix = np.array(confMatrix)

  # Get the number of classes from the shape of the confusion matrix.
  noOfClasses = confMatrix.shape[0]
  # Check if the confusion matrix is for binary classification or multiclass.
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
    # Unravel the confusion matrix to get the TN, FP, FN, and TP.
    TN, FP, FN, TP = confMatrix.ravel()

  # Add a small epsilon value to avoid division by zero in metric calculations.
  TP = TP + eps
  FP = FP + eps
  FN = FN + eps
  TN = TN + eps

  # Create a dictionary to hold the calculated performance metrics and the TP, FP, FN, TN vectors.
  metrics = {
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
  }

  # Calculate macro-averaged precision, recall, F1, accuracy, and specificity.
  precision = np.mean(TP / (TP + FP))
  recall = np.mean(TP / (TP + FN))
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.mean(TP + TN) / np.sum(confMatrix)
  specificity = np.mean(TN / (TN + FP))

  # Add macro metrics to the dictionary.
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

  # Calculate micro-averaged precision, recall, F1, accuracy, and specificity.
  precision = np.sum(TP) / np.sum(TP + FP)
  recall = np.sum(TP) / np.sum(TP + FN)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
  specificity = np.sum(TN) / np.sum(TN + FP)

  # Add micro metrics to the dictionary.
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

  # Calculate weighted-averaged precision, recall, F1, accuracy, and specificity.
  precision = np.sum(TP / (TP + FP) * weights)
  recall = np.sum(TP / (TP + FN) * weights)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum((TP + TN) * weights) / np.sum(confMatrix)
  specificity = np.sum(TN / (TN + FP) * weights)

  # Add weights and weighted metrics to the dictionary.
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

  # Return the dictionary containing all calculated metrics.
  return metrics


if __name__ == "__main__":
  # Example confusion matrix for a 3-class classification problem.
  confMatrix = [
    [50, 2, 1],
    [5, 45, 0],
    [0, 3, 47]
  ]

  # Calculate metrics and include weighted averages in the output.
  metrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True)

  # Print each metric name and its rounded value.
  for key, value in metrics.items():
    print(f"{key}: {np.round(value, 4)}")
