import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter
from HMB.PlotsHelper import GetCmapColors


def CalculatePerformanceMetrics(
  confMatrix,  # Confusion matrix (2D list or numpy array).
  eps=1e-10,  # Small value to avoid division by zero.
  addWeightedAverage=False,  # Whether to include weighted averages in the output.
  addPerClass=False,  # Whether to include per-class metrics in the output.
):
  r'''
  Calculate performance metrics from a confusion matrix.

  Parameters:
    confMatrix (list or numpy.ndarray): Confusion matrix representing the classification results.
    eps (float): Small value to avoid division by zero. Default is 1e-10.
    addWeightedAverage (bool): Whether to include weighted averages in the output. Default is False.
    addPerClass (bool): Whether to include per-class metrics in the output. Default is False.

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

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    confMatrix = [[50, 2, 1], [5, 45, 0], [0, 3, 47]]
    metrics = pm.CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True)
    for key, value in metrics.items():
      print(f"{key}: {np.round(value, 4)}")


  Another example to use the confusion matrix from sklearn:

  .. code-block:: python

    import numpy as np
    from sklearn.metrics import confusion_matrix
    import HMB.PerformanceMetrics as pm

    yTrue = [0, 1, 2, 0, 1, 2]
    yPred = [0, 2, 1, 0, 0, 1]
    confMatrix = confusion_matrix(yTrue, yPred)
    metrics = pm.CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True)
    for key, value in metrics.items():
      print(f"{key}: {np.round(value, 4)}")
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
    "TP": str(TP),
    "FP": str(FP),
    "FN": str(FN),
    "TN": str(TN),
  }

  # If requested, calculate per-class precision, recall, F1, accuracy, and specificity.
  if (addPerClass and noOfClasses > 2):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2.0 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    bac = 0.5 * (recall + specificity)

    for i in range(len(precision)):
      metrics.update({
        f"Class {i} Precision"  : precision[i],
        f"Class {i} Recall"     : recall[i],
        f"Class {i} F1"         : f1[i],
        f"Class {i} Accuracy"   : accuracy[i],
        f"Class {i} Specificity": specificity[i],
        f"Class {i} BAC"        : bac[i],
        f"TP Class {i}"         : TP[i],
        f"FP Class {i}"         : FP[i],
        f"FN Class {i}"         : FN[i],
        f"TN Class {i}"         : TN[i],
      })

  # Calculate macro-averaged precision, recall, F1, accuracy, and specificity.
  precision = np.mean(TP / (TP + FP))
  recall = np.mean(TP / (TP + FN))
  f1 = 2.0 * precision * recall / (precision + recall)
  accuracy = np.mean(TP + TN) / np.sum(confMatrix)
  specificity = np.mean(TN / (TN + FP))
  bac = 0.5 * (recall + specificity)

  # Add macro metrics to the dictionary.
  metrics.update({
    "Macro Precision"  : precision,
    "Macro Recall"     : recall,
    "Macro F1"         : f1,
    "Macro Accuracy"   : accuracy,
    "Macro Specificity": specificity,
    "Macro BAC"        : bac,
  })

  # If requested, calculate the macro average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0
    metrics.update({
      "Macro Average": avg,
    })

  # Calculate micro-averaged precision, recall, F1, accuracy, and specificity.
  precision = np.sum(TP) / np.sum(TP + FP)
  recall = np.sum(TP) / np.sum(TP + FN)
  f1 = 2.0 * precision * recall / (precision + recall)
  accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
  specificity = np.sum(TN) / np.sum(TN + FP)
  bac = 0.5 * (recall + specificity)

  # Add micro metrics to the dictionary.
  metrics.update({
    "Micro Precision"  : precision,
    "Micro Recall"     : recall,
    "Micro F1"         : f1,
    "Micro Accuracy"   : accuracy,
    "Micro Specificity": specificity,
    "Micro BAC"        : bac,
  })

  # If requested, calculate the micro average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0
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
  f1 = 2.0 * precision * recall / (precision + recall)
  accuracy = np.sum((TP + TN) * weights) / np.sum(confMatrix)
  specificity = np.sum(TN / (TN + FP) * weights)
  bac = 0.5 * (recall + specificity)

  # Add weights and weighted metrics to the dictionary.
  metrics.update({
    "Weights"             : weights,
    "Weighted Precision"  : precision,
    "Weighted Recall"     : recall,
    "Weighted F1"         : f1,
    "Weighted Accuracy"   : accuracy,
    "Weighted Specificity": specificity,
    "Weighted BAC"        : bac,
  })

  # If requested, calculate the weighted average of the metrics.
  if (addWeightedAverage):
    avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0
    metrics.update({
      "Weighted Average": avg,
    })

  # Return the dictionary containing all calculated metrics.
  return metrics


def PlotConfusionMatrix(
  cm,  # Confusion matrix (2D list or numpy array).
  classes,  # List of class labels.
  normalize=False,  # Whether to normalize the confusion matrix.
  roundDigits=3,  # Number of decimal places to round normalized values.
  title="Confusion Matrix",  # Title of the plot.
  cmap=plt.cm.Blues,  # Colormap for the plot.
  display=True,  # Whether to display the plot.
  save=False,  # Whether to save the plot.
  fileName="ConfusionMatrix.pdf",  # File name to save the plot.
  fontSize=15,  # Font size for labels and annotations.
  annotate=True,  # Whether to annotate cells with values.
  figSize=(8, 8),  # Figure size in inches.
  colorbar=True,  # Whether to show colorbar.
  returnFig=False,  # Whether to return the figure object.
  dpi=720,  # DPI for saving the figure.
):
  r'''
  Plot a confusion matrix with options for normalization, annotation, saving, and display.

  Parameters:
    cm (list or numpy.ndarray): Confusion matrix representing the classification results.
    classes (list): List of class labels to display on axes.
    normalize (bool): Whether to normalize the confusion matrix by row sums. Default is False.
    roundDigits (int): Number of decimal places to round normalized values. Default is 3.
    title (str): Title of the plot. Default is "Confusion Matrix".
    cmap (matplotlib.colors.Colormap or None): Colormap for the plot. Default is plt.cm.Blues.
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "ConfusionMatrix.pdf".
    fontSize (int): Font size for labels and annotations. Default is 15.
    annotate (bool): Whether to annotate cells with values. Default is True.
    figSize (tuple): Figure size in inches. Default is (8, 8).
    colorbar (bool): Whether to show colorbar. Default is True.
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    dpi (int): DPI for saving the figure. Default is 720.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - The confusion matrix can be normalized by row sums for better interpretability.
    - Annotated values in each cell help visualize the distribution of predictions.
    - Saving and displaying the plot are optional and controlled by parameters.

  .. math::
    \text{Normalized CM}_{i,j} = \frac{CM_{i,j}}{\sum_j CM_{i,j}}
  .. math::
    \text{Precision} = \frac{TP}{TP + FP}
    \qquad
    \text{Recall} = \frac{TP}{TP + FN}
    \qquad
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    confMatrix = [[50, 2, 1], [5, 45, 0], [0, 3, 47]]
    classLabels = ["Class 0", "Class 1", "Class 2"]
    pm.PlotConfusionMatrix(
      confMatrix,
      classes=classLabels,
      normalize=False,
      title="Confusion Matrix",
      annotate=True,
      fontSize=15,
      figSize=(6, 6),
      colorbar=True,
      display=True,
      save=False,
    )
  '''

  # Make the grid lines behind the confusion matrix.
  plt.rcParams["axes.axisbelow"] = True

  # Check if normalization is requested.
  if (normalize):  # Normalize the confusion matrix.
    # Normalize the confusion matrix by row sums.
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

  if (cmap is None):
    cmap = plt.cm.Blues  # Default colormap.

  if (save or display or returnFig):
    # Create a new figure with the specified size.
    fig = plt.figure(figsize=figSize)  # Create a new figure.

    # Display the confusion matrix as an image.
    plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # Set the plot title if provided.
    if (title and len(title) > 0):
      # Set the plot title with the specified font size.
      plt.title(title, fontsize=fontSize)  # Set the title.

    # Add the color bar and change the color bar font size.
    if (colorbar):
      # Set colorbar tick label size.
      plt.colorbar().ax.tick_params(labelsize=fontSize)

    # Create tick marks for each class label.
    tickMarks = np.arange(len(classes))  # Create a range of values.
    # Set x-axis tick labels with rotation and font size.
    plt.xticks(tickMarks, classes, rotation=45, fontsize=fontSize)
    # Set y-axis tick labels with font size.
    plt.yticks(tickMarks, classes, fontsize=fontSize)

    # Choose format for cell annotation based on normalization.
    fmt = f"0.{roundDigits}f" if normalize else "d"  # Set the format.
    # Calculate threshold for text color contrast.
    thresh = cm.max() / 2.0  # Threshold.

    # Annotate cells with values if requested.
    if (annotate):
      for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
          # Place text annotation in each cell.
          plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=fontSize,
          )

    # Set y-axis label.
    plt.ylabel("True Label", fontsize=fontSize)
    # Set x-axis label.
    plt.xlabel("Predicted Label", fontsize=fontSize)
    # Tight the layout to ignore wasted spaces.
    plt.tight_layout()

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    # Display the plot if requested.
    if (display):  # Display the plot.
      plt.show()

    plt.close(fig)  # Close the plot.

    # Return the figure object if requested.
    if (returnFig):
      return fig


def PlotRegressionResults(
  yTrue,
  yPred,
  title="Regression Results",
  fontSize=14,
  figsize=(10, 5),
  display=True,
  save=False,
  fileName="RegressionResults.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot regression results: predicted vs. true values and residuals.

  Parameters:
    yTrue (array-like): True target values.
    yPred (array-like): Predicted target values.
    title (str): Plot title.
    fontSize (int): Font size for labels and title.
    figsize (tuple): Figure size.
    display (bool): Whether to display the plot.
    save (bool): Whether to save the plot.
    fileName (str): File name to save the plot.
    dpi (int): DPI for saving the figure.
    returnFig (bool): Whether to return the figure object.

  Returns:
    fig: The matplotlib figure object if returnFig is True, else None.
  '''
  yTrue = np.array(yTrue)
  yPred = np.array(yPred)
  residuals = yTrue - yPred

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

  # Scatter plot: True vs. Predicted.
  ax1.scatter(yTrue, yPred, alpha=0.7, color="royalblue", edgecolor="k")
  minVal = min(np.min(yTrue), np.min(yPred))
  maxVal = max(np.max(yTrue), np.max(yPred))
  ax1.plot([minVal, maxVal], [minVal, maxVal], "r--", lw=2, label="Ideal")
  ax1.set_xlabel("True Values", fontsize=fontSize)
  ax1.set_ylabel("Predicted Values", fontsize=fontSize)
  ax1.set_title("True vs. Predicted", fontsize=fontSize + 2)
  ax1.legend(fontsize=fontSize * 0.8)
  ax1.grid(alpha=0.3)

  # Residuals plot.
  ax2.scatter(yPred, residuals, alpha=0.7, color="darkorange", edgecolor="k")
  ax2.axhline(0, color="gray", linestyle="--", lw=2)
  ax2.set_xlabel("Predicted Values", fontsize=fontSize)
  ax2.set_ylabel("Residuals (True - Pred)", fontsize=fontSize)
  ax2.set_title("Residuals Plot", fontsize=fontSize + 2)
  ax2.grid(alpha=0.3)

  # Summary statistics.
  mse = np.mean(residuals ** 2)
  mae = np.mean(np.abs(residuals))
  r2 = 1 - np.sum(residuals ** 2) / np.sum((yTrue - np.mean(yTrue)) ** 2)
  stats = f"MSE: {mse:.3f}\nMAE: {mae:.3f}\nR2: {r2:.3f}"
  ax2.text(
    0.05, 0.95, stats, transform=ax2.transAxes,
    fontsize=fontSize * 0.9, verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
  )

  plt.suptitle(title, fontsize=fontSize + 4)
  plt.tight_layout(rect=[0, 0, 1, 0.95])

  if (save and fileName):
    plt.savefig(fileName, dpi=dpi)
  if (display):
    plt.show()

  plt.close(fig)

  if (returnFig):
    return fig

  return None


def PlotROCAUCCurve(
  yTrue,  # True labels (one-hot or binary).
  yPred,  # Predicted labels (one-hot or binary).
  classes,  # List of class names.
  areProbabilities=False,  # Whether yPred are probabilities.
  title="ROC Curve & AUC",  # Plot title.
  figSize=(5, 5),  # Figure size.
  cmap=None,  # Colormap for ROC curves.
  display=True,  # Display the plot.
  save=False,  # Save the plot.
  fileName="ROC_AUC.pdf",  # File name.
  fontSize=15,  # Font size.
  plotDiagonal=True,  # Plot diagonal reference line.
  annotateAUC=True,  # Annotate AUC value on plot.
  showLegend=True,  # Show legend.
  returnFig=False,  # Return figure object.
  dpi=720,  # DPI for saving the figure.
):
  r'''
  Plot ROC curves and calculate AUC for each class, with options for annotation, saving, and display.

  Parameters:
    yTrue (array-like or numpy.ndarray): True labels (one-hot encoded or binary).
    yPred (array-like or numpy.ndarray): Predicted labels or probabilities
      (one-hot encoded, binary, or probabilities).
    classes (list): List of class names for labeling curves.
    areProbabilities (bool): Whether yPred contains probabilities. Default is False.
    title (str): Title of the plot. Default is "ROC Curve & AUC".
    figSize (tuple): Figure size in inches. Default is (5, 5).
    cmap (matplotlib.colors.Colormap or None): Colormap for ROC curves. Default is None.
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "ROC_AUC.pdf".
    fontSize (int): Font size for labels and annotations. Default is 15.
    plotDiagonal (bool): Whether to plot the diagonal reference line. Default is True.
    annotateAUC (bool): Whether to annotate AUC value on each curve. Default is True.
    showLegend (bool): Whether to show legend. Default is True.
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    dpi (int): DPI for saving the figure. Default is 720.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - ROC curves visualize the trade-off between true positive rate and false positive rate for each class.
    - AUC (Area Under Curve) quantifies the overall ability of the classifier to distinguish between classes.
    - Saving and displaying the plot are optional and controlled by parameters.

  .. math::
    \text{TPR} = \frac{TP}{TP + FN}
    \qquad
    \text{FPR} = \frac{FP}{FP + TN}
  .. math::
    \text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    yTrue = [0, 1, 2, 0, 1, 2]
    yPred = [
      [0.8, 0.1, 0.1],
      [0.2, 0.7, 0.1],
      [0.1, 0.2, 0.7],
      [0.9, 0.05, 0.05],
      [0.1, 0.8, 0.1],
      [0.05, 0.1, 0.85]
    ]
    classLabels = ["Class 0", "Class 1", "Class 2"]
    pm.PlotROCAUCCurve(
      np.array(yTrue),
      np.array(yPred),
      classes=classLabels,
      areProbabilities=True,
      title="ROC Curve & AUC",
      display=True,
      save=False
    )
  '''

  from sklearn.metrics import roc_auc_score, roc_curve

  # Ensure that yTrue and yPred are numpy arrays.
  yTrue = np.array(yTrue)
  yPred = np.array(yPred)

  # Get the number of classes.
  numClasses = len(classes)

  if (not areProbabilities):
    # One-hot encode the true and predicted labels.
    yTrue = np.eye(numClasses)[yTrue]
    yPred = np.eye(numClasses)[yPred]

  # Get colors for each class from colormap.
  colors = (
    cmap(np.linspace(0, 1, numClasses))
    if (cmap) else [None] * numClasses
  )

  if (save or display or returnFig):
    # Create a figure.
    fig = plt.figure(figsize=figSize)

    # Calculate the ROC curve and AUC for each class.
    for i in range(numClasses):
      yTrueC = (
        yTrue[:, i]
        if (not areProbabilities)
        else ((yTrue == i).astype(np.float32))
      )
      # Calculate ROC curve and AUC.
      aucRoc = roc_auc_score(
        yTrueC,  # True labels for class i.
        yPred[:, i],  # Predicted labels for class i.
      )
      FPR, TPR, _ = roc_curve(
        yTrueC,  # True labels for class i.
        yPred[:, i],  # Predicted labels for class i.
      )

      # Plot ROC curve for each class.
      plt.plot(
        FPR, TPR,
        label=(
          f"{classes[i]} (AUC={aucRoc:.3f})"
          if (annotateAUC) else f"{classes[i]}"
        ),
        color=colors[i] if (colors[i] is not None) else None
      )

    if (plotDiagonal):
      # Plot the diagonal line.
      plt.plot([0, 1], [0, 1], "k--")

    # Set the plot title with the specified font size.
    plt.title(title, fontsize=fontSize)
    # Set x-axis label.
    plt.xlabel("False Positive Rate", fontsize=fontSize)
    # Set y-axis label.
    plt.ylabel("True Positive Rate", fontsize=fontSize)

    # Add grid lines to the plot.
    plt.grid(True)

    # Update the font of tick labels.
    plt.xticks(fontsize=fontSize * 0.75)
    plt.yticks(fontsize=fontSize * 0.75)

    if (showLegend):
      # Show legend if requested.
      plt.legend(fontsize=fontSize * 0.75)

    # Tight the layout to ignore wasted spaces.
    plt.tight_layout()

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    if (display):
      # Display the plot if requested.
      plt.show()

    plt.close(fig)  # Close the plot.

    if (returnFig):
      # Return the figure object if requested.
      return fig


def PlotPRCCurve(
  yTrue,  # True labels (one-hot or binary).
  yPred,  # Predicted labels (one-hot or binary).
  classes,  # List of class names.
  areProbabilities=False,  # Whether yPred are probabilities.
  title="PRC Curve",  # Plot title.
  figSize=(5, 5),  # Figure size.
  cmap=None,  # Colormap for PRC curves.
  display=True,  # Display the plot.
  save=False,  # Save the plot.
  fileName="PRC.pdf",  # File name.
  fontSize=15,  # Font size.
  annotateAvg=True,  # Annotate average precision value on plot.
  showLegend=True,  # Show legend.
  returnFig=False,  # Return figure object.
  dpi=720,  # DPI for saving the figure.
):
  r'''
  Plot Precision-Recall curves (PRC) and calculate average precision for each class, with options for annotation, saving, and display.

  Parameters:
    yTrue (array-like or numpy.ndarray): True labels (one-hot encoded or binary).
    yPred (array-like or numpy.ndarray): Predicted labels or probabilities (one-hot encoded, binary, or probabilities).
    classes (list): List of class names for labeling curves.
    areProbabilities (bool): Whether yPred contains probabilities. Default is False.
    title (str): Title of the plot. Default is "PRC Curve".
    figSize (tuple): Figure size in inches. Default is (5, 5).
    cmap (matplotlib.colors.Colormap or None): Colormap for PRC curves. Default is None.
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "PRC.pdf".
    fontSize (int): Font size for labels and annotations. Default is 15.
    annotateAvg (bool): Whether to annotate average precision value on each curve. Default is True.
    showLegend (bool): Whether to show legend. Default is True.
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    dpi (int): DPI for saving the figure. Default is 720.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Precision-Recall curves visualize the trade-off between precision (PPV) and recall (TPR) for each class.
    - Average precision quantifies the overall ability of the classifier to balance precision and recall.
    - Saving and displaying the plot are optional and controlled by parameters.

  .. math::
    \text{Precision} = \frac{TP}{TP + FP}
    \qquad
    \text{Recall} = \frac{TP}{TP + FN}
  .. math::
    \text{Average Precision} = \int_0^1 \text{Precision}(\text{Recall}) \, d\text{Recall}

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    yTrue = [0, 1, 2, 0, 1, 2]
    yPred = [
      [0.8, 0.1, 0.1],
      [0.2, 0.7, 0.1],
      [0.1, 0.2, 0.7],
      [0.9, 0.05, 0.05],
      [0.1, 0.8, 0.1],
      [0.05, 0.1, 0.85]
    ]
    classLabels = ["Class 0", "Class 1", "Class 2"]
    pm.PlotPRCCurve(
      np.array(yTrue),
      np.array(yPred),
      classes=classLabels,
      areProbabilities=True,
      title="PRC Curve",
      display=True,
      save=False
    )
  '''

  from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
  )

  # Ensure that yTrue and yPred are numpy arrays.
  yTrue = np.array(yTrue)
  yPred = np.array(yPred)

  # Get the number of classes.
  numClasses = len(classes)

  if (not areProbabilities):
    # One-hot encode the true and predicted labels.
    yTrue = np.eye(numClasses)[yTrue]
    yPred = np.eye(numClasses)[yPred]

  # Get colors for each class from colormap.
  colors = (
    cmap(np.linspace(0, 1, numClasses))
    if (cmap) else [None] * numClasses
  )

  if (save or display or returnFig):
    # Create a figure.
    fig = plt.figure(figsize=figSize)

    # Calculate the PRC curve and Avg. for each class.
    for i in range(numClasses):
      yTrueC = (
        yTrue[:, i]
        if (not areProbabilities)
        else ((yTrue == i).astype(np.float32))
      )
      # Calculate the average precision score.
      avgScore = average_precision_score(
        yTrueC,  # True labels for class i.
        yPred[:, i],  # Predicted labels for class i.
      )
      # Extract the PPV and TPR values.
      PPV, TPR, _ = precision_recall_curve(
        yTrueC,  # True labels for class i.
        yPred[:, i],  # Predicted labels for class i.
      )

      # Plot the PPV and TPR values.
      plt.step(
        TPR, PPV,  # TPR and PPV values.
        where="post",
        label=(
          f"{classes[i]} (AVG={avgScore:.3f})"
          if (annotateAvg) else f"{classes[i]}"
        ),
        color=colors[i] if (colors[i] is not None) else None
      )

    # Set the plot title with the specified font size.
    plt.title(title, fontsize=fontSize)
    # Set the x- and y-labels.
    plt.xlabel("Recall", fontsize=fontSize)
    plt.ylabel("Precision", fontsize=fontSize)

    # Set the x- and y-limits.
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    # Update the font of tick labels.
    plt.xticks(fontsize=fontSize * 0.75)
    plt.yticks(fontsize=fontSize * 0.75)

    if (showLegend):
      # Show legend if requested.
      plt.legend(fontsize=fontSize * 0.75)

    # Tight the layout to ignore wasted spaces.
    plt.tight_layout()

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    if (display):
      # Display the plot if requested.
      plt.show()

    plt.close(fig)  # Close the plot.

    if (returnFig):
      # Return the figure object if requested.
      return fig


def PlotMultiTrialROCAUC(
  allYTrue,  # List of true labels arrays from all trials (list of arrays).
  allYPred,  # List of predicted probabilities from all trials (list of arrays).
  classes,  # List of class names.
  confidenceLevel=0.95,  # Confidence level for CI (default 95%).
  which="CI",  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
  title="Multi-Trial ROC Curve with Confidence Intervals",  # Plot title.
  figSize=(8, 8),  # Figure size.
  cmap=None,  # Colormap for ROC curves.
  display=True,  # Display the plot.
  save=False,  # Save the plot.
  fileName="MultiTrial_ROC_AUC.pdf",  # File name.
  fontSize=15,  # Font size.
  plotDiagonal=True,  # Plot diagonal reference line.
  showLegend=True,  # Show legend.
  returnFig=False,  # Return figure object.
  dpi=720,  # DPI for saving the figure.
  addZoomedInset=True,  # Whether to add a zoomed inset for the top-left corner.
):
  '''
  Plot averaged ROC curves across multiple trials with confidence intervals.

  Parameters:
    allYTrue (list): List of ground truth arrays from all trials. Each element shape: (nSamples, nClasses) or (nSamples,).
    allYPred (list): List of prediction probability arrays from all trials. Each element shape: (nSamples, nClasses).
    classes (list): List of class names.
    confidenceLevel (float): Confidence level for intervals (default 0.95).
    which (str): Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
    title (str): Plot title.
    figSize (tuple): Figure size in inches.
    cmap (colormap): Matplotlib colormap for different classes.
    display (bool): Whether to display the plot.
    save (bool): Whether to save the plot.
    fileName (str): File name for saving.
    fontSize (int): Font size for labels.
    plotDiagonal (bool): Whether to plot the diagonal reference line.
    showLegend (bool): Whether to show legend.
    returnFig (bool): Whether to return figure object.
    dpi (int): DPI for saving the figure.
    addZoomedInset (bool): Whether to add a zoomed inset for the top-left corner of the ROC plot.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - This function computes and plots the mean ROC curve across multiple trials for each class.
    - Confidence intervals are calculated using the normal approximation method.
    - The plot includes options for saving, displaying, and customizing appearance.

  .. math::
    \text{TPR} = \frac{TP}{TP + FN}
    \qquad
    \text{FPR} = \frac{FP}{FP + TN}
  .. math::
    \text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    # Simulated data for 3 trials and 3 classes.
    allYTrue = [
      np.array([0, 1, 2, 0, 1, 2]),
      np.array([0, 1, 2, 0, 1, 2]),
      np.array([0, 1, 2, 0, 1, 2])
    ]
    allYPred = [
      np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.1, 0.85]
      ]),
      np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.3, 0.5],
        [0.85, 0.1, 0.05],
        [0.15, 0.75, 0.1],
        [0.1, 0.2, 0.7]
      ]),
      np.array([
        [0.75, 0.15, 0.1],
        [0.25, 0.65, 0.1],
        [0.15, 0.25, 0.6],
        [0.88, 0.07, 0.05],
        [0.12, 0.78, 0.1],
        [0.08, 0.15, 0.77]
      ])
    ]
    classLabels = ["Class 0", "Class 1", "Class 2"]
    pm.PlotMultiTrialROCAUC(
      allYTrue,
      allYPred,
      classes=classLabels,
      confidenceLevel=0.95,
      which="CI",
      title="Multi-Trial ROC Curve with Confidence Intervals",
      display=True,
      save=False
    )
  '''

  from scipy import stats
  from sklearn.metrics import roc_curve, auc

  assert which in ["CI", "SD"], "Parameter 'which' must be either 'CI' or 'SD'."

  # Convert to numpy arrays.
  # Check if allYTrue is just a single array (not a list of arrays), and if so, repeat it to match the length of allYPred.
  if (isinstance(allYTrue, np.ndarray)):
    allYTrue = [allYTrue] * len(allYPred)
  elif (isinstance(allYTrue, list) and len(allYTrue) == 1 and isinstance(allYTrue[0], np.ndarray)):
    allYTrue = allYTrue * len(allYPred)

  allYPred = [np.array(yp) for yp in allYPred]

  numClasses = len(classes)
  numTrials = len(allYTrue)

  # Get colors for each class.
  if (cmap is None):
    cmap = plt.cm.get_cmap("tab10")
  colors = [cmap(i / numClasses) for i in range(numClasses)]

  # Create figure.
  fig = plt.figure(figsize=figSize)

  # Storage for results.
  resultsDict = {}

  # Common FPR values for interpolation.
  meanFPR = np.linspace(0, 1, 100)

  # Process each class.
  for classIdx in range(numClasses):
    className = classes[classIdx]

    # Storage for this class across trials.
    tprs = []
    aucs = []

    # Collect ROC data from all trials.
    for trialIdx in range(numTrials):
      yTrue = allYTrue[trialIdx]
      yPred = allYPred[trialIdx]

      # Check if yTrue is one-hot encoded or class indices.
      if (yTrue.ndim == 1):
        # Class indices - convert to binary for this class.
        yTrueBinary = (yTrue == classIdx).astype(int)
      else:
        # One-hot encoded.
        yTrueBinary = yTrue[:, classIdx]

      # Get predictions for this class.
      yPredClass = yPred[:, classIdx]

      # Calculate ROC curve.
      fpr, tpr, _ = roc_curve(yTrueBinary, yPredClass)

      # Calculate AUC.
      rocAuc = auc(fpr, tpr)
      aucs.append(rocAuc)

      # Interpolate TPR at common FPR values.
      interpTpr = np.interp(meanFPR, fpr, tpr)
      interpTpr[0] = 0.0  # Ensure it starts at 0.
      tprs.append(interpTpr)

    # Convert to numpy array.
    tprs = np.array(tprs)
    aucs = np.array(aucs)

    # Calculate mean and std.
    meanTpr = np.mean(tprs, axis=0)
    stdTpr = np.std(tprs, axis=0)
    meanAuc = np.mean(aucs)
    stdAuc = np.std(aucs)

    if (which == "CI"):
      # Calculate alpha for confidence intervals.
      alpha = 1 - confidenceLevel
      # Calculate confidence intervals using normal approximation for CI.
      z = stats.norm.ppf(1 - alpha / 2)  # Z-score for confidence level.
      # CI for TPR.
      tprUpper = np.minimum(meanTpr + z * stdTpr / np.sqrt(numTrials), 1.0)
      tprLower = np.maximum(meanTpr - z * stdTpr / np.sqrt(numTrials), 0.0)
      # CI for AUC.
      aucUpper = meanAuc + z * stdAuc / np.sqrt(numTrials)
      aucLower = meanAuc - z * stdAuc / np.sqrt(numTrials)

      labelPlot = f"{className} (AUC={meanAuc:.3f} ± {stdAuc:.3f} CI)"
      labelFill = f"{className} {int(confidenceLevel * 100)}% CI"
    else:
      # Use standard deviation as bounds.
      tprUpper = np.minimum(meanTpr + stdTpr, 1.0)
      tprLower = np.maximum(meanTpr - stdTpr, 0.0)
      aucUpper = meanAuc + stdAuc
      aucLower = meanAuc - stdAuc

      labelPlot = f"{className} (AUC={meanAuc:.3f} ± {stdAuc:.3f} SD)"
      labelFill = f"{className} ± 1 SD"

    # Store results.
    resultsDict[className] = {
      "meanAuc"   : meanAuc,
      "stdAuc"    : stdAuc,
      "aucCiLower": aucLower,
      "aucCiUpper": aucUpper,
      "meanTpr"   : meanTpr,
      "tprCiLower": tprLower,
      "tprCiUpper": tprUpper,
    }

    # Plot mean ROC curve.
    plt.plot(
      meanFPR,
      meanTpr,
      color=colors[classIdx],
      linewidth=2,
      label=labelPlot,
    )

    # Plot confidence interval as shaded region.
    plt.fill_between(
      meanFPR,
      tprLower,
      tprUpper,
      color=colors[classIdx],
      alpha=0.2,
      label=labelFill,
    )

  # Plot diagonal reference line.
  if (plotDiagonal):
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

  # Set labels and title.
  plt.xlabel("False Positive Rate", fontsize=fontSize)
  plt.ylabel("True Positive Rate", fontsize=fontSize)
  plt.title(f"{title}\n({numTrials} trials)", fontsize=fontSize)

  # Set axis limits.
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])

  # Add grid.
  plt.grid(True, alpha=0.3)

  # Update tick font sizes.
  plt.xticks(fontsize=fontSize * 0.75)
  plt.yticks(fontsize=fontSize * 0.75)

  # Show legend.
  if (showLegend):
    plt.legend(loc="lower right", fontsize=fontSize * 0.65)

  # Tight layout.
  plt.tight_layout()

  if (addZoomedInset):
    # Add a small plot on the plot for a zoomed-in view of the top-left corner.
    axInset = fig.add_axes([0.2, 0.2, 0.25, 0.25])  # [left, bottom, width, height].
    for classIdx in range(numClasses):
      className = classes[classIdx]
      meanTpr = resultsDict[className]["meanTpr"]
      tprLower = resultsDict[className]["tprCiLower"]
      tprUpper = resultsDict[className]["tprCiUpper"]

      axInset.plot(
        meanFPR,
        meanTpr,
        color=colors[classIdx],
        linewidth=2,
      )
      axInset.fill_between(
        meanFPR,
        tprLower,
        tprUpper,
        color=colors[classIdx],
        alpha=0.2,
      )
    axInset.plot([0, 1], [0, 1], "k--", linewidth=1)
    axInset.set_xlim(0.0, 0.2)
    axInset.set_ylim(0.8, 1.0)
    axInset.set_title("Zoomed In", fontsize=fontSize * 0.75)
    axInset.set_xlabel("FPR", fontsize=fontSize * 0.6)
    axInset.set_ylabel("TPR", fontsize=fontSize * 0.6)
    axInset.tick_params(axis="both", which="major", labelsize=fontSize * 0.5)
    axInset.grid(True, alpha=0.3)

  # Save if requested.
  if (save):
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  # Display if requested.
  if (display):
    plt.show()

  # Close figure if not returning.
  if (not returnFig):
    plt.close(fig)

  # Return results and optionally figure.
  if (returnFig):
    return resultsDict, fig
  else:
    return resultsDict


def PlotMultiTrialPRCurve(
  allYTrue,  # List of true labels arrays from all trials.
  allYPred,  # List of predicted probabilities from all trials.
  classes,  # List of class names.
  confidenceLevel=0.95,  # Confidence level for CI.
  which="CI",  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
  title="Multi-Trial Precision-Recall Curve with Confidence Intervals",
  figSize=(8, 8), # Figure size in inches.
  cmap=None, # Colormap for different classes.
  display=True, # Whether to display the plot.
  save=False, # Whether to save the plot.
  fileName="MultiTrial_PRC.pdf", # File name for saving.
  fontSize=15, # Font size for labels and annotations.
  showLegend=True, # Whether to show legend.
  returnFig=False, # Whether to return the matplotlib figure object.
  dpi=720, # DPI for saving the figure.
  addZoomedInset=True,  # Whether to add a zoomed inset for the top-right corner of the PRC plot.
):
  '''
  Plot averaged Precision-Recall curves across multiple trials with confidence intervals.

  Parameters:
    allYTrue (list): List of ground truth arrays from all trials. Each element shape: (nSamples, nClasses) or (nSamples,).
    allYPred (list): List of prediction probability arrays from all trials. Each element shape: (nSamples, nClasses).
    classes (list): List of class names.
    confidenceLevel (float): Confidence level for intervals (default 0.95).
    title (str): Plot title.
    figSize (tuple): Figure size in inches.
    cmap (colormap): Matplotlib colormap for different classes.
    display (bool): Whether to display the plot.
    save (bool): Whether to save the plot.
    fileName (str): File name for saving.
    fontSize (int): Font size for labels.
    showLegend (bool): Whether to show legend.
    returnFig (bool): Whether to return figure object.
    dpi (int): DPI for saving.
    addZoomedInset (bool): Whether to add a zoomed inset for the top-right corner of the PRC plot.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - This function computes and plots the mean Precision-Recall curve across multiple trials for each class.
    - Confidence intervals are calculated using the normal approximation method.
    - The plot includes options for saving, displaying, and customizing appearance.

  .. math::
    \text{Precision} = \frac{TP}{TP + FP}
    \qquad
    \text{Recall} = \frac{TP}{TP + FN}
  .. math::
    \text{Average Precision} = \int_0^1 \text{Precision}(\text{Recall}) \, d\text{Recall}

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    # Simulated data for 3 trials and 3 classes.
    allYTrue = [
      np.array([0, 1, 2, 0, 1, 2]),
      np.array([0, 1, 2, 0, 1, 2]),
      np.array([0, 1, 2, 0, 1, 2])
    ]
    allYPred = [
      np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.1, 0.85]
      ]),
      np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.3, 0.5],
        [0.85, 0.1, 0.05],
        [0.15, 0.75, 0.1],
        [0.1, 0.2, 0.7]
      ]),
      np.array([
        [0.75, 0.15, 0.1],
        [0.25, 0.65, 0.1],
        [0.15, 0.25, 0.6],
        [0.88, 0.07, 0.05],
        [0.12, 0.78, 0.1],
        [0.08, 0.15, 0.77]
      ])
    ]
    classLabels = ["Class 0", "Class 1", "Class 2"]
    pm.PlotMultiTrialPRCurve(
      allYTrue,
      allYPred,
      classes=classLabels,
      confidenceLevel=0.95,
      title="Multi-Trial Precision-Recall Curve with Confidence Intervals",
      display=True,
      save=False
    )
  '''

  from scipy import stats
  from sklearn.metrics import precision_recall_curve, average_precision_score

  assert which in ["CI", "SD"], "Parameter 'which' must be either 'CI' or 'SD'."

  # Convert to numpy arrays.
  # Check if allYTrue is just a single array (not a list of arrays), and if so, repeat it to match the length of allYPred.
  if (isinstance(allYTrue, np.ndarray)):
    allYTrue = [allYTrue] * len(allYPred)
  elif (isinstance(allYTrue, list) and len(allYTrue) == 1 and isinstance(allYTrue[0], np.ndarray)):
    allYTrue = allYTrue * len(allYPred)
  allYPred = [np.array(yp) for yp in allYPred]

  numClasses = len(classes)
  numTrials = len(allYTrue)

  # Get colors for each class.
  if (cmap is None):
    cmap = plt.cm.get_cmap("tab10")
  colors = [cmap(i / numClasses) for i in range(numClasses)]

  # Create figure.
  fig = plt.figure(figsize=figSize)

  # Storage for results.
  resultsDict = {}

  # Common recall values for interpolation.
  meanRecall = np.linspace(0, 1, 100)

  # Process each class.
  for classIdx in range(numClasses):
    className = classes[classIdx]

    # Storage for this class across trials.
    precisions = []
    aps = []

    # Collect PR data from all trials.
    for trialIdx in range(numTrials):
      yTrue = allYTrue[trialIdx]
      yPred = allYPred[trialIdx]

      # Check if yTrue is one-hot encoded or class indices.
      if (yTrue.ndim == 1):
        yTrueBinary = (yTrue == classIdx).astype(int)
      else:
        yTrueBinary = yTrue[:, classIdx]

      # Get predictions for this class.
      yPredClass = yPred[:, classIdx]

      # Calculate PR curve.
      precision, recall, _ = precision_recall_curve(yTrueBinary, yPredClass)

      # Calculate Average Precision.
      ap = average_precision_score(yTrueBinary, yPredClass)
      aps.append(ap)

      # Interpolate precision at common recall values.
      # Reverse for interpolation (recall is decreasing).
      interpPrecision = np.interp(meanRecall[::-1], recall[::-1], precision[::-1])[::-1]
      precisions.append(interpPrecision)

    # Convert to numpy array.
    precisions = np.array(precisions)
    aps = np.array(aps)

    # Calculate mean and std.
    meanPrecision = np.mean(precisions, axis=0)
    stdPrecision = np.std(precisions, axis=0)
    meanAp = np.mean(aps)
    stdAp = np.std(aps)

    if (which == "CI"):
      # Calculate alpha for confidence intervals.
      alpha = 1 - confidenceLevel
      # Calculate confidence intervals.
      z = stats.norm.ppf(1 - alpha / 2)

      # CI for Precision.
      precisionUpper = np.minimum(meanPrecision + z * stdPrecision / np.sqrt(numTrials), 1.0)
      precisionLower = np.maximum(meanPrecision - z * stdPrecision / np.sqrt(numTrials), 0.0)

      # CI for AP.
      apUpper = meanAp + z * stdAp / np.sqrt(numTrials)
      apLower = meanAp - z * stdAp / np.sqrt(numTrials)

      labelPlot = f"{className} (AP={meanAp:.3f} ± {stdAp:.3f} CI)"
      labelFill = f"{className} {int(confidenceLevel * 100)}% CI"
    else:
      # Use standard deviation as bounds.
      precisionUpper = np.minimum(meanPrecision + stdPrecision, 1.0)
      precisionLower = np.maximum(meanPrecision - stdPrecision, 0.0)
      apUpper = meanAp + stdAp
      apLower = meanAp - stdAp

      labelPlot = f"{className} (AP={meanAp:.3f} ± {stdAp:.3f} SD)"
      labelFill = f"{className} ± 1 SD"

    # Store results.
    resultsDict[className] = {
      "meanAp"   : meanAp,
      "stdAp"    : stdAp,
      "apCiLower": apLower,
      "apCiUpper": apUpper,
      "meanPrecision": meanPrecision,
      "precisionCiLower": precisionLower,
      "precisionCiUpper": precisionUpper,
    }

    # Plot mean PR curve.
    plt.plot(
      meanRecall,
      meanPrecision,
      color=colors[classIdx],
      linewidth=2,
      label=labelPlot,
    )

    # Plot confidence interval.
    plt.fill_between(
      meanRecall,
      precisionLower,
      precisionUpper,
      color=colors[classIdx],
      alpha=0.2,
      label=labelFill,
    )

  # Set labels and title.
  plt.xlabel("Recall", fontsize=fontSize)
  plt.ylabel("Precision", fontsize=fontSize)
  plt.title(f"{title}\n({numTrials} trials)", fontsize=fontSize)

  # Set axis limits.
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])

  # Add grid.
  plt.grid(True, alpha=0.3)

  # Update tick font sizes.
  plt.xticks(fontsize=fontSize * 0.75)
  plt.yticks(fontsize=fontSize * 0.75)

  # Show legend.
  if (showLegend):
    plt.legend(loc="lower left", fontsize=fontSize * 0.65)

  # Tight layout.
  plt.tight_layout()

  if (addZoomedInset):
    # Add a small plot on the plot for a zoomed-in view of the top-right corner.
    axInset = fig.add_axes([0.5, 0.5, 0.25, 0.25])  # [left, bottom, width, height].
    for classIdx in range(numClasses):
      className = classes[classIdx]
      meanPrecision = resultsDict[className]["meanPrecision"]
      precisionLower = resultsDict[className]["precisionCiLower"]
      precisionUpper = resultsDict[className]["precisionCiUpper"]

      axInset.plot(
        meanRecall,
        meanPrecision,
        color=colors[classIdx],
        linewidth=2,
      )
      axInset.fill_between(
        meanRecall,
        precisionLower,
        precisionUpper,
        color=colors[classIdx],
        alpha=0.2,
      )
    axInset.set_xlim(0.8, 1.0)
    axInset.set_ylim(0.8, 1.05)
    axInset.set_title("Zoomed In", fontsize=fontSize * 0.75)
    axInset.set_xlabel("Recall", fontsize=fontSize * 0.6)
    axInset.set_ylabel("Precision", fontsize=fontSize * 0.6)
    axInset.tick_params(axis="both", which="major", labelsize=fontSize * 0.5)
    axInset.grid(True, alpha=0.3)

  # Save if requested.
  if (save):
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  # Display if requested.
  if (display):
    plt.show()

  # Close figure if not returning.
  if (not returnFig):
    plt.close(fig)

  # Return results and optionally figure.
  if (returnFig):
    return resultsDict, fig
  else:
    return resultsDict


def PlotCounterfactualOutcomes(
  X,
  classifier,
  treatmentCol=None,
  lowVal=None,
  highVal=None,
  classNames=None,
  title="Counterfactual Outcome Distributions",
  save=False,
  fileName="CounterfactualOutcomePlot.pdf",
  display=True,
  returnPreds=False,
  fontSize=12,
  dpi=720,
):
  r'''
  Plot counterfactual outcome distributions for two treatment scenarios using a histogram.

  Parameters:
    X (pandas.DataFrame): Feature matrix.
    classifier: Trained classifier with predict method.
    treatmentCol (str or None): Name of the treatment column. If None, prints available columns and returns.
    lowVal (numeric or None): Value representing "no/low" treatment. If None, uses min value in column.
    highVal (numeric or None): Value representing "high" treatment. If None, uses max value in column.
    classNames (list or None): List of class names for x-axis ticks. If None, inferred from classifier or predictions.
    title (str): Title of the plot. Default is "Counterfactual Outcome Distributions".
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "CounterfactualOutcomePlot.pdf".
    display (bool): Whether to display the plot. Default is True.
    returnPreds (bool): If True, returns (yPredLow, yPredHigh). Default is False.
    fontSize (int): Font size for labels and title. Default is 12.
    dpi (int): DPI for saving the figure. Default is 720.

  Returns:
    None or tuple: None if returnPreds is False, otherwise (yPredLow, yPredHigh).

  Notes:
    - The function creates two copies of X, sets the treatment column to lowVal and highVal, and predicts outcomes for both scenarios.
    - The results are plotted as overlaid histograms, with x-axis ticks labeled by classNames if provided.
    - The plot is saved to fileName and optionally displayed.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    X = ...  # Load or create your feature DataFrame.
    classifier = ...  # Load or train your classifier.
    classNames = []  # Specify class names.

    pm.PlotCounterfactualOutcomes(
      X, classifier,
      treatmentCol="Inflight Wi-Fi service",
      classNames=classNames,
      title="Counterfactual Outcome Distributions",
      save=True,
      fileName="CounterfactualOutcomePlot.pdf",
      display=True,
      fontSize=14
    )
  '''

  def _PreviewColumnsAndValues(xdf, maxUnique=10):
    # Print available columns and their unique values for user guidance.
    print("Available columns:")
    for col in xdf.columns:
      uniqueVals = xdf[col].unique()
      if (len(uniqueVals) <= maxUnique):
        print(f"- {col}: unique values = {uniqueVals}")
      else:
        print(
          f"- {col}: {len(uniqueVals)} unique values (showing first {maxUnique}): "
          f"{uniqueVals[:maxUnique]}"
        )

  # Plot counterfactual outcomes for two treatment scenarios.
  if (treatmentCol is None or treatmentCol not in X.columns):
    # If treatmentCol is not specified or invalid, print available columns and return.
    print("[Counterfactual Plot] Please specify a valid treatmentCol. Available columns:")
    _PreviewColumnsAndValues(X)
    return None

  # Get sorted unique values in treatment column.
  uniqueVals = np.sort(X[treatmentCol].unique())
  if (len(uniqueVals) < 2):
    # If not enough unique values, print warning and return.
    print(f"Not enough unique values in treatment column '{treatmentCol}' to create counterfactuals.")
    return None

  if (lowVal is None or highVal is None):
    # If lowVal or highVal not specified, use min and max and print info.
    if (lowVal is None):
      lowVal = uniqueVals[0]
    if (highVal is None):
      highVal = uniqueVals[-1]
    print(f"[Counterfactual Plot] Using lowVal={lowVal}, highVal={highVal}")

  if (hasattr(classifier, "_estimator_type") and classifier._estimator_type == "regressor"):
    # If classifier is a regressor, skip plotting and print warning.
    print(
      "[Counterfactual Plot] Classifier is a regressor. "
      "Counterfactual outcome plots are only for classifiers."
    )
    return None

  xLow = X.copy()  # Copy dataframe for low treatment scenario.
  xHigh = X.copy()  # Copy dataframe for high treatment scenario.
  xLow[treatmentCol] = lowVal  # Set treatment column to low value.
  xHigh[treatmentCol] = highVal  # Set treatment column to high value.
  yPredLow = classifier.predict(xLow)  # Predict outcomes for low treatment.
  yPredHigh = classifier.predict(xHigh)  # Predict outcomes for high treatment.

  if (classNames is None):
    # Infer class names if not provided.
    if (hasattr(classifier, "classes_")):
      classNames = [str(c) for c in classifier.classes_]
    else:
      classNames = [str(c) for c in np.unique(np.concatenate([yPredLow, yPredHigh]))]

  if (save or display):
    plt.figure(figsize=(8, 6))  # Create new figure for plot.
    bins = np.arange(-0.5, np.max([yPredLow.max(), yPredHigh.max()]) + 1.5, 1)  # Set histogram bins.

    plt.hist(
      yPredLow,
      bins=bins,
      alpha=0.6,
      label=f"{treatmentCol}={lowVal}",
      color="red",
      rwidth=0.8,
    )  # Plot low treatment histogram.
    plt.hist(
      yPredHigh,
      bins=bins,
      alpha=0.6,
      label=f"{treatmentCol}={highVal}",
      color="blue",
      rwidth=0.8,
    )  # Plot high treatment histogram.

    # Set x-axis label with font size.
    plt.xlabel("Predicted Class", fontsize=fontSize)

    # Set x-ticks to class names if provided.
    if (classNames is not None):
      # If classNames is provided, set x-ticks to class names.
      numClasses = len(classNames)
      plt.xticks(ticks=np.arange(numClasses), labels=classNames, fontsize=fontSize)

    # Set y-axis label with font size.
    plt.ylabel("Number of Samples", fontsize=fontSize)

    if (title and len(title) > 0):
      plt.title(title, fontsize=fontSize + 2)  # Set plot title if provided.
    else:
      # Set plot title with font size.
      plt.title(f"Counterfactual Outcome Plot: {treatmentCol}", fontsize=fontSize + 2)

    plt.legend(fontsize=fontSize)  # Show legend with font size.
    plt.grid(axis="y", alpha=0.3)  # Add grid to y-axis.
    plt.tight_layout()  # Adjust layout.

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          plt.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      plt.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    if (display):
      plt.show()  # Show plot if requested.

    plt.close()  # Close the plot to free memory.

  if (returnPreds):
    return yPredLow, yPredHigh  # Optionally return predictions.

  print(f"[Counterfactual Plot] Unique values in '{treatmentCol}': {uniqueVals}")
  return None


def PlotInteractionEffect(
  X,
  classifier,
  feature1,
  feature2,
  gridSize=30,
  title="Interaction Effect Plot",
  save=False,
  fileName="InteractionEffectPlot.pdf",
  display=True,
  fontSize=12,
  plotType="surface",  # or "contour"
  backend="matplotlib",  # or "plotly"
  dpi=720,
):
  r'''
  Plot the interaction effect between two features on the predicted outcome using a 3D surface or contour plot.

  Parameters:
    X (pandas.DataFrame): Feature matrix.
    classifier: Trained classifier with predict or predict_proba method.
    feature1 (str): Name of the first feature (x-axis).
    feature2 (str): Name of the second feature (y-axis).
    gridSize (int): Number of grid points for each feature. Default is 30.
    title (str): Title of the plot. Default is "Interaction Effect Plot".
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "InteractionEffectPlot.pdf".
    display (bool): Whether to display the plot. Default is True.
    fontSize (int): Font size for labels and title. Default is 12.
    plotType (str): "surface" for 3D surface plot, "contour" for contour plot. Default is "surface".
    backend (str): "matplotlib" for static plots, "plotly" for interactive HTML. Default is "matplotlib".
    dpi (int): DPI for saving the figure. Default is 720.

  Returns:
    Figure object (matplotlib or plotly) or None.

  Notes:
    - For classifiers with predict_proba, the positive class probability is plotted if binary, otherwise the max probability.
    - For classifiers without predict_proba, the predicted class is plotted.
    - The plot is saved to fileName and optionally displayed.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    X = ...  # Load or create your feature DataFrame.
    classifier = ...  # Load or train your classifier.

    pm.PlotInteractionEffect(
      X, classifier,
      feature1="Flight Distance",
      feature2="Seat Comfort",
      gridSize=30,
      title="Interaction Effect Plot",
      save=True,
      fileName="InteractionEffectPlot.pdf",
      display=True,
      fontSize=14,
      plotType="surface",
      backend="matplotlib"
    )
  '''

  # Check if the specified features exist in the DataFrame.
  if (feature1 not in X.columns or feature2 not in X.columns):
    # Print an error message if features are not found.
    print(f"[Interaction Effect Plot] '{feature1}' or '{feature2}' not found in columns.")
    print(f"Available columns: {list(X.columns)}")

    return None
  # Get the minimum and maximum values for feature1.
  f1Min, f1Max = X[feature1].min(), X[feature1].max()
  # Get the minimum and maximum values for feature2.
  f2Min, f2Max = X[feature2].min(), X[feature2].max()
  # Create a grid of values for feature1.
  f1Grid = np.linspace(f1Min, f1Max, gridSize)
  # Create a grid of values for feature2.
  f2Grid = np.linspace(f2Min, f2Max, gridSize)
  # Create a meshgrid for the two features.
  f1Mesh, f2Mesh = np.meshgrid(f1Grid, f2Grid)
  # Use the median values of all features as a base row.
  baseRow = X.median().to_dict()
  # Initialize a grid to store predictions.
  Z = np.zeros_like(f1Mesh, dtype=float)

  # Iterate over the grid to compute predictions for each combination.
  for i in range(gridSize):
    # Iterate over the second feature's grid.
    for j in range(gridSize):
      # Copy the base row for each grid point.
      row = baseRow.copy()
      # Set the value for feature1 at this grid point.
      row[feature1] = f1Mesh[i, j]
      # Set the value for feature2 at this grid point.
      row[feature2] = f2Mesh[i, j]
      # Create a DataFrame for prediction.
      rowDf = pd.DataFrame([row])
      # Use predict_proba if available, otherwise use predict.
      if (hasattr(classifier, "predict_proba")):
        # Get the predicted probabilities.
        proba = classifier.predict_proba(rowDf)[0]
        # Use the probability of the positive class if binary, else use the max probability.
        if (len(proba) == 2):
          Z[i, j] = proba[1]
        else:
          Z[i, j] = np.max(proba)
      else:
        # Use the predicted class if predict_proba is not available.

        Z[i, j] = classifier.predict(rowDf)[0]
  # Set the x-axis label.
  xLabel = feature1
  # Set the y-axis label.
  yLabel = feature2
  # Set the z-axis label depending on the classifier type.
  zLabel = "Predicted Probability" if (hasattr(classifier, "predict_proba")) else "Predicted Class"

  # Check which backend to use for plotting.
  if (backend == "matplotlib"):
    # Import matplotlib and 3D plotting toolkit.
    from mpl_toolkits.mplot3d import Axes3D
    # Create a new figure for the plot.
    fig = plt.figure(figsize=(8, 6))
    # Plot a 3D surface if requested.
    if (plotType == "surface"):
      # Add a 3D subplot.
      ax = fig.add_subplot(111, projection="3d")
      # Plot the surface.
      surf = ax.plot_surface(f1Mesh, f2Mesh, Z, cmap="viridis", edgecolor="none", alpha=0.85)
      # Set the x-axis label with font size.
      ax.set_xlabel(xLabel, fontsize=fontSize)
      # Set the y-axis label with font size.
      ax.set_ylabel(yLabel, fontsize=fontSize)
      # Set the z-axis label with font size.
      ax.set_zlabel(zLabel, fontsize=fontSize)
      # Set the plot title with font size.
      ax.set_title(f"Interaction Effect: {feature1} vs {feature2}", fontsize=fontSize + 2)
      # Add a colorbar to the plot.
      fig.colorbar(surf, shrink=0.5, aspect=10)
    else:
      # Plot a filled contour plot if requested.
      cp = plt.contourf(f1Mesh, f2Mesh, Z, cmap="viridis")
      # Set the x-axis label with font size.
      plt.xlabel(xLabel, fontsize=fontSize)
      # Set the y-axis label with font size.
      plt.ylabel(yLabel, fontsize=fontSize)
      # Set the plot title with font size.
      plt.title(f"Interaction Effect: {feature1} vs {feature2}", fontsize=fontSize + 2)
      # Add a colorbar to the plot.

      plt.colorbar(cp)
    # Adjust the layout for better appearance.
    plt.tight_layout()

    if (title and len(title) > 0):
      plt.suptitle(title, fontsize=fontSize + 4)  # Set overall title if provided.
      plt.subplots_adjust(top=0.9)  # Adjust the top to fit the suptitle.

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    # Show the plot if requested.
    if (display):
      plt.show()

    # Close the plot to free memory.
    plt.close(fig)

    # Print a message indicating where the plot was saved.
    print(f"[Interaction Effect Plot] Saved to {fileName}")
    # Return the figure object.
    return fig
  else:
    # Import plotly for interactive plotting.
    import plotly.graph_objs as go
    import plotly.io as pio

    # Create a plotly surface plot if requested.
    if (plotType == "surface"):
      fig = go.Figure(data=[go.Surface(z=Z, x=f1Grid, y=f2Grid, colorscale="Viridis")])
    else:
      # Create a plotly contour plot if requested.
      fig = go.Figure(
        data=[
          go.Contour(z=Z, x=f1Grid, y=f2Grid, colorscale="Viridis", contours_coloring="heatmap")
        ]
      )

    # Update the layout with axis labels and font size.
    fig.update_layout(
      title=f"Interaction Effect: {feature1} vs {feature2}",
      scene=dict(
        xaxis_title=xLabel,
        yaxis_title=yLabel,
        zaxis_title=zLabel,
        xaxis=dict(title_font=dict(size=fontSize)),
        yaxis=dict(title_font=dict(size=fontSize)),
        zaxis=dict(title_font=dict(size=fontSize)),
      ),
      font=dict(size=fontSize),
      margin=dict(l=0, r=0, b=0, t=40),
    )

    if (title and len(title) > 0):
      fig.update_layout(title=title)  # Set overall title if provided.

    # Save the plot as an interactive HTML file if requested.
    if (save):
      # Save the plot as an interactive HTML file.
      pio.write_html(fig, file=fileName, auto_open=display)
      # Print a message indicating where the plot was saved.
      print(f"[Interaction Effect Plot] Saved to {fileName}")

    if (display):
      fig.show()  # Show the plot if requested.

    # Return the figure object.
    return fig


def PlotCalibrationCurveFromModel(
  classifier,
  X,
  y,
  classNames=None,
  nBins=10,
  strategy="uniform",
  title="Calibration Curve",
  save=False,
  fileName="CalibrationCurve.pdf",
  display=True,
  fontSize=12,
  plotHistogram=False,
  returnFig=False,
  cmap=None,
  dpi=720,
):
  r'''
  Plot calibration curves to evaluate how well predicted probabilities align with observed outcomes.

  Parameters:
    classifier: Trained classifier with predict_proba method.
    X (pandas.DataFrame or numpy.ndarray): Feature matrix.
    y (array-like): True labels.
    classNames (list or None): List of class names. If None, inferred from classifier.
    nBins (int): Number of bins to discretize the [0, 1] interval. Default is 10.
    strategy (str): Binning strategy ("uniform" or "quantile"). Default is "uniform".
    title (str): Title of the plot. Default is "Calibration Curve".
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "CalibrationCurve.pdf".
    display (bool): Whether to display the plot. Default is True.
    fontSize (int): Font size for labels and title. Default is 12.
    plotHistogram (bool): Whether to plot a histogram of predicted probabilities below the calibration curve. Default is False.
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    cmap (str or Colormap or None): Colormap to use for the curves. If None, uses "tab10". Default is None.
    dpi (int): DPI for saving the figure. Default is 720.

  Returns:
    None or matplotlib.figure.Figure: The figure object if returnFig is True, otherwise None.

  Notes:
    - For binary classification, plots the calibration curve for the positive class.
    - For multiclass, plots one-vs-rest calibration curves for each class.
    - The plot is saved to fileName and optionally displayed.
    - If plotHistogram is True, a histogram of predicted probabilities is shown below the calibration curve.
    - For best results, use with probabilistic models and sufficient sample size per bin.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    X = ...  # Load or create your feature DataFrame.
    y = ...  # Load or create your true labels.
    classifier = ...  # Load or train your classifier.
    classNames = []  # Specify class names.

    pm.PlotCalibrationCurveFromModel(
      classifier, X, y,
      classNames=classNames,
      nBins=10,
      strategy="uniform",
      title="Calibration Curve",
      save=True,
      fileName="CalibrationCurve.pdf",
      display=True,
      fontSize=14,
      plotHistogram=True,
      returnFig=False,
      cmap="tab20"
    )
  '''

  import random
  from sklearn.calibration import calibration_curve

  # Check if classifier supports predict_proba.
  if (not hasattr(classifier, "predict_proba")):
    raise ValueError("Classifier must support 'predict_proba' for calibration curves.")
  # Check if X and y are not None.
  if (X is None or y is None):
    raise ValueError("X and y must not be None.")
  # Check if X and y are not empty.
  if (len(X) == 0 or len(y) == 0):
    raise ValueError("X and y must not be empty.")

  # Convert y to numpy array.
  y = np.array(y)
  # Get predicted probabilities from classifier.
  yProb = classifier.predict_proba(X)
  # Get number of classes.
  nClasses = yProb.shape[1] if (len(yProb.shape) > 1) else 2
  # Check if there are at least two classes.
  if (nClasses < 2):
    raise ValueError("Calibration curve requires at least two classes.")

  # Infer class names if not provided.
  if (classNames is None):
    if (hasattr(classifier, "classes_")):
      classNames = [str(c) for c in classifier.classes_]
    else:
      classNames = [str(i) for i in range(nClasses)]

  # Prepare figure and axes.
  if (plotHistogram):
    fig, (ax1, ax2) = plt.subplots(
      2, 1,
      figsize=(8, 8),
      gridspec_kw={"height_ratios": [2, 1]},
    )
  else:
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = None

  # Get color map for plotting.
  if (cmap is None):
    cmapObj = plt.cm.get_cmap("tab10", nClasses)
  elif isinstance(cmap, str):
    cmapObj = plt.cm.get_cmap(cmap, nClasses)
  else:
    cmapObj = cmap

  # Pick random colors from the colormap for each class.
  colorIndices = random.sample(range(cmapObj.N), nClasses) if cmapObj.N >= nClasses else list(range(nClasses))
  random.shuffle(colorIndices)
  colors = [cmapObj(i) for i in colorIndices]

  # Initialize lines and labels lists.
  lines = []
  labels = []

  # Plot calibration curve(s).
  for i in range(nClasses):
    # For binary classification, plot only the positive class.
    if (nClasses == 2):
      prob = yProb[:, 1]
      yTrue = (y == classifier.classes_[1]) if (hasattr(classifier, "classes_")) else (y == 1)
      label = f"{classNames[1]} (n={np.sum(yTrue)})" if (len(classNames) > 1) else "Class 1"
      fracPos, meanPred = calibration_curve(yTrue, prob, n_bins=nBins, strategy=strategy)
      line, = ax1.plot(meanPred, fracPos, marker="o", label=label, color=colors[1 % len(colors)])
      lines.append(line)
      labels.append(label)
      if (plotHistogram and ax2 is not None):
        ax2.hist(prob, range=(0, 1), bins=nBins, color=colors[1 % len(colors)], alpha=0.6, label=label)
      break
    else:
      prob = yProb[:, i]
      yTrue = (y == i) if (not hasattr(classifier, "classes_")) else (y == classifier.classes_[i])
      label = f"{classNames[i]} (n={np.sum(yTrue)})"
      fracPos, meanPred = calibration_curve(yTrue, prob, n_bins=nBins, strategy=strategy)
      line, = ax1.plot(meanPred, fracPos, marker="o", label=label, color=colors[i % len(colors)])
      lines.append(line)
      labels.append(label)
      if (plotHistogram and ax2 is not None):
        ax2.hist(prob, range=(0, 1), bins=nBins, color=colors[i % len(colors)], alpha=0.6, label=label)

  # Plot perfectly calibrated line.
  ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
  ax1.set_xlabel("Mean Predicted Probability", fontsize=fontSize)
  ax1.set_ylabel("Fraction of Positives", fontsize=fontSize)
  ax1.set_title("Calibration Curve", fontsize=fontSize + 2)
  ax1.legend(fontsize=fontSize)
  ax1.grid(True, alpha=0.3)

  if (plotHistogram and ax2 is not None):
    ax2.set_xlabel("Predicted Probability", fontsize=fontSize)
    ax2.set_ylabel("Count", fontsize=fontSize)
    ax2.set_title("Predicted Probability Histogram", fontsize=fontSize)
    ax2.legend(fontsize=fontSize)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
  else:
    plt.tight_layout()

  if (title and len(title) > 0):
    plt.suptitle(title, fontsize=fontSize + 4)
    plt.subplots_adjust(top=0.9)

  # Save the plot if requested.
  if (save):  # Save the plot.
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()

  plt.close(fig)
  print(f"[Calibration Curve] Saved to {fileName}")

  if (returnFig):
    return fig
  return None


def PlotCalibrationCurve(
  probs,
  labels,
  nBins=10,
  title="Calibration Curve",
  fontSize=14,
  figSize=(6, 6),
  display=True,
  save=False,
  fileName="CalibrationCurve.pdf",
  dpi=720,
  returnFig=False,
  color="blue",
):
  r'''
  Plot calibration curve given predicted probabilities and true labels.

  Parameters:
    probs (numpy.ndarray): Predicted probabilities of shape (nSamples, nClasses).
    labels (array-like): True labels of shape (nSamples,).
    nBins (int): Number of bins to discretize the [0, 1] interval. Default is 10.
    title (str): Title of the plot. Default is "Calibration Curve".
    fontSize (int): Font size for labels and title. Default is 14.
    figSize (tuple): Figure size in inches. Default is (6, 6).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "CalibrationCurve.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    color (str): Color of the calibration curve. Default is "blue".

  Returns:
    tuple: (binCenters, accuracy, confidence, support) where:
      - binCenters (numpy.ndarray): Centers of the bins.
      - accuracy (numpy.ndarray): Accuracy in each bin.
      - confidence (numpy.ndarray): Average confidence in each bin.
      - support (numpy.ndarray): Number of samples in each bin.

  Notes:
    - The function computes the calibration data and plots the calibration curve.
    - The plot is saved to fileName and optionally displayed.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    # Example predicted probabilities and true labels.
    probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
    labels = np.array([1, 0, 1, 0])

    pm.PlotCalibrationCurve(
      probs, labels,
      nBins=5,
      title="Calibration Curve Example",
      fontSize=12,
      figSize=(5, 5),
      display=True,
      save=True,
      fileName="CalibrationCurveExample.pdf",
      dpi=300,
      returnFig=False,
      color="green"
    )
  '''

  # Compute calibration data.
  labels = np.asarray(labels)
  confidences = np.max(probs, axis=1)
  preds = np.argmax(probs, axis=1)
  bins = np.linspace(0.0, 1.0, nBins + 1)
  binCenters = 0.5 * (bins[:-1] + bins[1:])
  acc = np.zeros(nBins)
  conf = np.zeros(nBins)
  support = np.zeros(nBins, dtype=int)
  for i in range(nBins):
    mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
    support[i] = np.sum(mask)

    # Calculate accuracy and confidence for the bin.
    if (support[i] > 0):
      acc[i] = np.mean(preds[mask] == labels[mask])
      conf[i] = np.mean(confidences[mask])
    else:
      acc[i] = np.nan
      conf[i] = np.nan

  # Plotting.
  if (display or save or returnFig):
    plt.figure(figsize=figSize)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.plot(conf, acc, marker="o", color=color, label="Model")
    plt.xlabel("Confidence", fontsize=fontSize)
    plt.ylabel("Accuracy", fontsize=fontSize)
    plt.title(title, fontsize=fontSize)

    # Add grid, limits, and legend.
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend(fontsize=fontSize)
    plt.tight_layout()

    # Save the plot if requested.
    if (save):
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          plt.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      plt.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    # Display the plot if requested.
    if (display):
      plt.show()

    # Close the plot to free memory.
    plt.close()

    if (returnFig):
      return plt.gcf(), binCenters, acc, conf, support

  return binCenters, acc, conf, support


def PlotTopKAccuracyCurve(
  probs,
  labels,
  maxK=10,
  title="Top-k Accuracy Curve",
  figSize=(6, 6),
  save=False,
  fileName="TopKAccuracyCurve.pdf",
  display=True,
  fontSize=12,
  returnFig=False,
  dpi=720,
  color="blue",
):
  r'''
  Plot Top-k accuracy curve given predicted probabilities and true labels.

  Parameters:
    probs (numpy.ndarray): Predicted probabilities of shape (nSamples, nClasses).
    labels (array-like): True labels of shape (nSamples,).
    maxK (int): Maximum value of k for Top-k accuracy. Default is 10.
    title (str): Title of the plot. Default is "Top-k Accuracy Curve".
    figSize (tuple): Figure size in inches. Default is (6, 6).
    save (bool): Whether to save the plot to fileName. Default is False.
    fileName (str): File name to save the plot. Default is "TopKAccuracyCurve.pdf".
    display (bool): Whether to display the plot. Default is True.
    fontSize (int): Font size for labels and title. Default is 12.
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    dpi (int): DPI for saving the figure. Default is 720.
    color (str): Color of the Top-k accuracy curve. Default is "blue".

  Returns:
    None or matplotlib.figure.Figure: The figure object if returnFig is True, otherwise None.

  Notes:
    - The function computes Top-k accuracy for k=1 to maxK and plots the curve.
    - The plot is saved to fileName and optionally displayed.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    # Example predicted probabilities and true labels.
    probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
    labels = np.array([1, 0, 1, 0])

    pm.PlotTopKAccuracyCurve(
      probs, labels,
      maxK=2,
      title="Top-k Accuracy Curve Example",
      figSize=(5, 5),
      save=True,
      fileName="TopKAccuracyCurveExample.pdf",
      display=True,
      fontSize=12,
      returnFig=False,
      dpi=300,
      color="green"
    )
  '''

  labels = np.asarray(labels)
  N, C = probs.shape
  ks = list(range(1, min(maxK, C) + 1))

  topkAcc = []
  order = np.argsort(probs, axis=1)[:, ::-1]  # Descending.

  for k in ks:
    topk = order[:, :k]
    hits = np.any(topk == labels[:, None], axis=1)
    topkAcc.append(np.mean(hits))

  if (save or display or returnFig):
    plt.figure(figsize=figSize)
    plt.plot(ks, topkAcc, marker="o", color=color)
    plt.xlabel("k", fontsize=fontSize)
    plt.ylabel("Accuracy", fontsize=fontSize)
    plt.title(title, fontsize=fontSize + 2)
    plt.xticks(ks, fontsize=fontSize)
    plt.tight_layout()
    plt.grid(True, linestyle=":", alpha=0.4)

    # Save the plot if requested.
    if (save):
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          plt.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      plt.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    # Display the plot if requested.
    if (display):
      plt.show()

    # Close the plot to free memory.
    plt.close()

    if (returnFig):
      return plt.gcf()


def HistoryPlotter(
  history,  # Dictionary containing training history.
  title,  # Title of the plot.
  metrics=("loss",),  # Tuple or list of metrics to plot.
  xLabel="Epochs",  # Label for x-axis.
  fontSize=14,  # Font size for labels and title.
  save=False,  # Whether to save the plot.
  savePath=None,  # Path to save the plot.
  dpi=720,  # DPI for saving the figure.
  colors=None,  # Optional dict of colors for each metric.
  labels=None,  # Optional dict of labels for each metric.
  display=True,  # Whether to display the plot.
  figSize=(10, 5),  # Figure size.
  returnFig=False,  # Whether to return the figure object.
  smooth=True,  # Whether to apply smoothing to the curves.
  smoothFactor=0.6,  # Smoothing factor for curves (0 to 1).
):
  r'''
  Plot training history metrics (e.g., loss, accuracy) for train and validation sets.

  Parameters:
    history (dict): Dictionary containing training history with keys like "train_loss", "val_loss", etc.
    title (str): Title of the plot.
    metrics (tuple or list): Metrics to plot (e.g., ("loss", "accuracy")). Default is ("loss",).
    xLabel (str): Label for x-axis. Default is "Epochs".
    fontSize (int): Font size for labels and title. Default is 14.
    save (bool): Whether to save the plot. Default is False.
    savePath (str or None): Path to save the plot. Default is None.
    dpi (int): DPI for saving the figure. Default is 720.
    colors (dict or None): Optional dict mapping metric names to colors.
    labels (dict or None): Optional dict mapping metric names to custom labels.
    display (bool): Whether to display the plot. Default is True.
    figSize (tuple): Figure size in inches. Default is (10, 5).
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.
    smooth (bool): Whether to apply smoothing to the curves. Default is True.
    smoothFactor (float): Smoothing factor for curves (0 to 1). Default is 0.6.

  Returns:
    matplotlib.figure.Figure or None: The axes object, figure object if returnFig is True, otherwise None.

  Notes:
    - Supports plotting multiple metrics for both training and validation.
    - Custom colors and labels can be provided for each metric.
    - Saving and displaying the plot are optional and controlled by parameters.

  Examples
  --------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    history = {
      "train_loss": [0.8, 0.6, 0.4],
      "val_loss": [0.9, 0.7, 0.5],
      "train_accuracy": [0.5, 0.7, 0.9],
      "val_accuracy": [0.4, 0.6, 0.8]
    }
    pm.HistoryPlotter(
      history,
      title="Training History",
      metrics=("loss", "accuracy"),
      doSave=True,
      savePath="history_plot.pdf"
    )
  '''

  if (save or display or returnFig):
    # Create a new figure with the specified size.
    plt.figure(figsize=figSize)  # Create figure.

    # Set default colors if not provided.
    if (colors is None):
      defaultColors = ["blue", "orange", "green", "red", "purple", "brown"]
      colors = {}
      for idx, metric in enumerate(metrics):
        colors[f"train_{metric}"] = defaultColors[idx % len(defaultColors)]
        colors[f"val_{metric}"] = defaultColors[(idx + 1) % len(defaultColors)]

    # Set default labels if not provided.
    if (labels is None):
      labels = {}
      for metric in metrics:
        labels[f"train_{metric}"] = f"Train {metric.capitalize()}"
        labels[f"val_{metric}"] = f"Validation {metric.capitalize()}"

    # Plot each metric for train and validation.
    for metric in metrics:
      trainKey = f"train_{metric}"
      valKey = f"val_{metric}"

      # Get the values for train and validation metrics from history.
      trainValues = history.get(trainKey, [])
      valValues = history.get(valKey, [])

      if (smooth):
        # Apply exponential moving average smoothing if requested.
        smoothedTrainValues, smoothedValValues = [], []
        for i in range(len(trainValues)):
          if (len(smoothedTrainValues) == 0):
            smoothedTrainValues.append(trainValues[i])
            smoothedValValues.append(valValues[i])
          else:
            # Exponential moving average.
            # smoothed(t) = alpha * value(t) + (1 - alpha) * smoothed(t - 1).
            smoothedTrainValues.append(
              smoothFactor * trainValues[i] +
              (1 - smoothFactor) * smoothedTrainValues[-1]
            )
            smoothedValValues.append(
              smoothFactor * valValues[i] +
              (1 - smoothFactor) * smoothedValValues[-1]
            )
        # Replace the original values with the smoothed values.
        trainValues = smoothedTrainValues
        valValues = smoothedValValues

      if (trainKey in history):
        # Plot training metric.
        plt.plot(
          history[trainKey],  # Training metric values.
          label=labels.get(trainKey, trainKey),  # Label for legend.
          color=colors.get(trainKey, None)  # Color for line.
        )
      if (valKey in history):
        # Plot validation metric.
        plt.plot(
          history[valKey],  # Validation metric values.
          label=labels.get(valKey, valKey),  # Label for legend.
          color=colors.get(valKey, None)  # Color for line.
        )

    # Set the plot title.
    plt.title(title, fontsize=fontSize * 1.2)  # Set title.

    # Set x-axis label.
    plt.xlabel(xLabel.capitalize(), fontsize=fontSize)  # Set x-label.

    # Set y-axis label.
    if (len(metrics) == 1):
      plt.ylabel(metrics[0].capitalize(), fontsize=fontSize)  # Set y-label.
    else:
      # Set y-label for multiple metrics.
      plt.ylabel("Metric Value", fontsize=fontSize)

    # Tight layout to minimize wasted space.
    plt.tight_layout()  # Tight layout.

    # Add legend.
    plt.legend()  # Add legend.

    # Add grid lines.
    plt.grid()  # Add grid.

    # Save the plot if requested.
    if (save and savePath):  # Save plot.
      ext = savePath.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          plt.savefig(savePath, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      plt.savefig(savePath.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    # Display the plot if requested.
    if (display):  # Display plot.
      plt.show()

    plt.close()  # Close the plot.

    # Return the figure or axes object if requested.
    if (returnFig):
      return plt.gcf()  # Return figure object.


def PlotMetricCurve(
  dataToPlot,  # Data to plot.
  xData=None,  # X-axis data.
  title="Metric Curve",  # Title of the plot.
  xLabel="X",  # X-axis label.
  yLabel="Y",  # Y-axis label.
  fontSize=15,  # Font size.
  xTicks=None,  # X-axis ticks.
  yTicks=None,  # Y-axis ticks.
  xTicksRotation=0,  # X-axis ticks rotation.
  yTicksRotation=0,  # Y-axis ticks rotation.
  save=False,  # Whether to save the plot.
  savePath=None,  # Path to save the plot.
  dpi=720,  # DPI for saving the figure.
  display=True,  # Whether to display the plot.
  figSize=(5, 5),  # Figure size.
  returnFig=False,  # Whether to return the figure object.
):
  r'''
  Plot a metric curve given data and optional x-axis values.

  Parameters:
    dataToPlot (array-like): Data to plot on the y-axis.
    xData (array-like or None): Data for the x-axis. If None, uses indices of dataToPlot. Default is None.
    title (str): Title of the plot. Default is "Metric Curve".
    xLabel (str): Label for the x-axis. Default is "X".
    yLabel (str): Label for the y-axis. Default is "Y".
    fontSize (int): Font size for labels and title. Default is 15.
    xTicks (array-like or None): Ticks for the x-axis. Default is None.
    yTicks (array-like or None): Ticks for the y-axis. Default is None.
    xTicksRotation (int): Rotation angle for x-axis ticks. Default is 0.
    yTicksRotation (int): Rotation angle for y-axis ticks. Default is 0.
    save (bool): Whether to save the plot. Default is False.
    savePath (str or None): Path to save the plot. Default is None.
    dpi (int): DPI for saving the figure. Default is 720.
    display (bool): Whether to display the plot. Default is True.
    figSize (tuple): Figure size in inches. Default is (5, 5).
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The figure object if returnFig is True, otherwise None
  '''

  # Get old font size.
  oldFontSize = plt.rcParams.get("font.size")

  # Update the overall font size.
  plt.rcParams.update({"font.size": fontSize})

  # Create a figure.
  plt.figure(1, figsize=figSize)

  # Plot the data.
  if (xData is None):
    xData = np.arange(len(dataToPlot))

  # Plot the data.
  plt.plot(xData, dataToPlot)

  # Set the x- and y-labels.
  plt.xlabel(xLabel, fontsize=fontSize)
  plt.ylabel(yLabel, fontsize=fontSize)

  # Set the title of the plot.
  plt.title(title, fontsize=fontSize)

  # Set the x- and y-ticks.
  if (xTicks is not None):
    plt.xticks(xTicks, rotation=xTicksRotation, fontsize=fontSize)
  if (yTicks is not None):
    plt.yticks(yTicks, rotation=yTicksRotation, fontsize=fontSize)

  plt.grid()  # Show the grid.
  plt.tight_layout()  # Tight the layout to ignore wasted spaces.

  # Save the plot if requested.
  if (save and savePath):  # Save plot.
    ext = savePath.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        plt.savefig(savePath, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    plt.savefig(savePath.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  # Display the plot if requested.
  if (display):  # Display plot.
    plt.show()

  plt.close()  # Close the plot.

  # Restore the old font size.
  plt.rcParams.update({"font.size": oldFontSize})

  # Return the figure or axes object if requested.
  if (returnFig):
    return plt.gcf()  # Return figure object.


def PlotCumulativeGainLiftChart(
  yTrue,
  yScores,
  posLabel=1,
  title="Cumulative Gain & Lift Chart",
  classNames=None,
  figsize=(10, 5),
  fontSize=14,
  display=True,
  save=False,
  fileName="CumulativeGainLiftChart.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot Cumulative Gain and Lift charts to visualize model effectiveness for ranking tasks.

  Parameters:
    yTrue (array-like): True binary labels.
    yScores (array-like): Target scores/probabilities for the positive class.
    posLabel (int or str): The label of the positive class. Default is 1.
    title (str): Plot title. Default is "Cumulative Gain & Lift Chart".
    classNames (list or None): Optional class names for legend. Default is None.
    figsize (tuple): Figure size. Default is (10, 5).
    fontSize (int): Font size for labels and title. Default is 14.
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "CumulativeGainLiftChart.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object(s) if returnFig is True, otherwise None.

  Notes:
    - The Cumulative Gain chart shows the proportion of positives captured as more samples are included, sorted by model score.
    - The Lift chart shows the improvement over random selection at each proportion of the sample.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    yTrue = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    yScores = [0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.4, 0.5, 0.05]
    pm.PlotCumulativeGainLiftChart(
      yTrue, yScores,
      title="Model Ranking Effectiveness",
      display=True,
      save=False
    )
  '''

  # Convert the inputs to numpy arrays.
  yTrue = np.array(yTrue)
  yScores = np.array(yScores)

  # Find the number of samples and sort by predicted scores.
  numSamples = len(yTrue)
  order = np.argsort(-yScores)
  # Sort true labels according to the order of predicted scores.
  yTrueSorted = yTrue[order]
  # Calculate cumulative positives and proportions.
  positives = (yTrueSorted == posLabel).astype(int)
  cumPositives = np.cumsum(positives)
  # Calculate total positives, proportion of samples, and proportion of positives captured.
  totalPositives = np.sum(positives)
  percSamples = np.arange(1, numSamples + 1) / numSamples
  percPositives = cumPositives / totalPositives

  # Cumulative Gain Chart.
  fig, ax1 = plt.subplots(figsize=figsize)
  ax1.plot(percSamples, percPositives, label="Model", color="blue", lw=2)
  ax1.plot([0, 1], [0, 1], "--", color="gray", label="Random", lw=1.5)
  ax1.set_xlabel("Proportion of Samples", fontsize=fontSize)
  ax1.set_ylabel("Proportion of Positives Captured", fontsize=fontSize)
  ax1.set_title(f"{title} - Cumulative Gain", fontsize=fontSize + 2)
  ax1.legend(fontsize=fontSize * 0.9)
  ax1.grid(axis="y", alpha=0.2)
  plt.tight_layout()

  # Save the Cumulative Gain chart if requested.
  if (save):
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName.replace(f".{ext}", "_CumulativeGain.pdf"), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", "_CumulativeGain.png"), dpi=dpi, bbox_inches="tight")

  # Display the Cumulative Gain chart if requested.
  if (display):
    plt.show()

  plt.close(fig)

  # Lift Chart.
  # Calculate lift, avoiding division by zero.
  lift = percPositives / percSamples
  lift[0] = lift[1]  # avoid inf at 0

  # Create Lift chart.
  fig2, ax2 = plt.subplots(figsize=figsize)
  ax2.plot(percSamples, lift, label="Model", color="green", lw=2)
  ax2.axhline(1, color="gray", linestyle="--", label="Random")
  ax2.set_xlabel("Proportion of Samples", fontsize=fontSize)
  ax2.set_ylabel("Lift", fontsize=fontSize)
  ax2.set_title(f"{title} - Lift Chart", fontsize=fontSize + 2)
  ax2.legend(fontsize=fontSize * 0.9)
  ax2.grid(True, alpha=0.3)
  plt.tight_layout()

  # Save the Lift chart if requested.
  if (save):
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig2.savefig(fileName.replace(f".{ext}", f"_Lift.pdf"), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig2.savefig(fileName.replace(f".{ext}", f"_Lift.png"), dpi=dpi, bbox_inches="tight")

  # Display the Lift chart if requested.
  if (display):
    plt.show()

  plt.close(fig2)

  if (returnFig):
    return fig, fig2

  return None


def PlotErrorAnalysis(
  yTrue,
  yPred,
  X=None,
  classNames=None,
  maxExamples=5,
  fontSize=12,
  figsize=(12, 8),
  display=True,
  save=False,
  fileName="ErrorAnalysis.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot error analysis showing examples of false positives, false negatives, true positives, and true negatives.

  Parameters:
    yTrue (array-like): True labels.
    yPred (array-like): Predicted labels.
    X (array-like, DataFrame, or None): Optional input samples to display. Default is None.
    classNames (list or None): Optional class names for display. Default is None.
    maxExamples (int): Max examples per error type to show. Default is 5.
    fontSize (int): Font size for text. Default is 12.
    figsize (tuple): Figure size. Default is (12, 8).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "ErrorAnalysis.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Shows up to maxExamples for each of FP, FN, TP, TN, with sample indices and optionally sample data.
    - Useful for qualitative error analysis and debugging.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    yTrue = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    yPred = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    PlotErrorAnalysis(
      yTrue, yPred,
      maxExamples=5,
      display=True,
      save=False
    )
  '''

  yTrue = np.array(yTrue)
  yPred = np.array(yPred)
  if (classNames is None and len(np.unique(yTrue)) == 2):
    classNames = ["Negative", "Positive"]
  elif (classNames is None):
    classNames = [str(c) for c in np.unique(yTrue)]

  # Binary only for now.
  FPIdx = np.where((yTrue == 0) & (yPred == 1))[0]
  FNIdx = np.where((yTrue == 1) & (yPred == 0))[0]
  TPIdx = np.where((yTrue == 1) & (yPred == 1))[0]
  TNIdx = np.where((yTrue == 0) & (yPred == 0))[0]

  types = [
    ("False Positives", FPIdx, "#FFCCCC"),
    ("False Negatives", FNIdx, "#FFE5B4"),
    ("True Positives", TPIdx, "#CCFFCC"),
    ("True Negatives", TNIdx, "#CCE5FF"),
  ]
  errorCounts = [len(FPIdx), len(FNIdx), len(TPIdx), len(TNIdx)]
  errorLabels = ["FP", "FN", "TP", "TN"]
  errorColors = ["#FF6666", "#FFB266", "#66FF66", "#66B2FF"]

  if (save or display or returnFig):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1, 1])

    # Top: summary bar chart.
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.bar(errorLabels, errorCounts, color=errorColors, edgecolor="black")
    for i, count in enumerate(errorCounts):
      ax_bar.text(
        i, count + max(errorCounts) * 0.02,
        str(count),
        ha="center",
        va="bottom",
        fontsize=fontSize + 2,
        fontweight="bold"
      )
    ax_bar.set_ylabel("Count", fontsize=fontSize)
    ax_bar.set_title("Error Type Counts", fontsize=fontSize + 4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.2)

    # 2x2 grid for error examples
    axes = [
      fig.add_subplot(gs[1, 0]),
      fig.add_subplot(gs[1, 1]),
      fig.add_subplot(gs[2, 0]),
      fig.add_subplot(gs[2, 1]),
    ]

    for i, (label, idxs, color) in enumerate(types):
      ax = axes[i]
      n = min(maxExamples, len(idxs))
      ax.set_title(f"{label} (n={len(idxs)})", fontsize=fontSize + 1, backgroundcolor=color, pad=8)
      ax.axis("off")

      if (n == 0):
        ax.text(0.5, 0.5, "None", ha="center", va="center", fontsize=fontSize, color="gray")
        continue

      # Table header.
      header = "Idx | True | Pred" + (" | Sample" if X is not None else "")
      ax.text(0, 1, header, fontsize=fontSize, fontweight="bold", va="top", family="monospace")

      for j in range(n):
        idx = idxs[j]
        tval = yTrue[idx]
        pval = yPred[idx]
        row = f"{idx:<3} | {tval:<4} | {pval:<4}"

        if (X is not None):
          sample = X.iloc[idx] if (hasattr(X, "iloc")) else X[idx]
          sampleStr = str(sample)

          # Limit sample string length for readability.
          if (len(sampleStr) > 60):
            sampleStr = sampleStr[:57] + "..."
          row += f" | {sampleStr}"
        ax.text(
          0, 1 - (j + 1) * 0.13, row,
          fontsize=fontSize * 0.95,
          va="top",
          family="monospace",
          bbox=dict(facecolor=color, edgecolor="none", alpha=0.25)
        )

    plt.suptitle("Error Analysis: FP, FN, TP, TN", fontsize=fontSize + 6, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    if (display):
      plt.show()

    plt.close(fig)

    if (returnFig):
      return fig

  return None


def PlotClasswisePRFBar(
  cm,
  classNames=None,
  title="Classwise Performance Metrics",
  fontSize=14,
  figsize=(8, 5),
  display=True,
  save=False,
  fileName="ClasswisePRFBar.pdf",
  dpi=720,
  returnFig=False,
):
  '''
  Plot classwise Precision, Recall, and F1-score as a grouped bar chart.

  Parameters:
    cm (array-like): Confusion matrix (2D array).
    classNames (list or None): List of class names. If None, uses class indices.
    title (str): Plot title. Default is "Classwise Performance Metrics".
    fontSize (int): Font size for labels and title. Default is 14.
    figsize (tuple): Figure size. Default is (8, 5).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "ClasswisePRFBar.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Computes precision, recall, and F1-score from the confusion matrix.
    - Displays a grouped bar chart for each class.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    cm = [[50, 2, 1],
          [10, 45, 5],
          [0, 3, 47]]
    classNames = ["Class A", "Class B", "Class C"]
    pm.PlotClasswisePRFBar(
      cm, classNames=classNames,
      fontSize=12,
      figsize=(9, 6),
      display=True,
      save=True,
      fileName="ClasswisePRFBar.pdf",
      dpi=300,
      returnFig=False
    )
  '''

  cm = np.array(cm)
  numClasses = cm.shape[0]

  if (classNames is None):
    classNames = [str(i) for i in range(numClasses)]

  pm = CalculatePerformanceMetrics(
    cm,  # Confusion matrix (2D list or numpy array).
    eps=1e-10,  # Small value to avoid division by zero.
    addWeightedAverage=True,  # Whether to include weighted averages in the output.
    addPerClass=True,  # Whether to include per-class metrics in the output.
  )
  precision, recall, f1, specificity, accuracy, bac = [], [], [], [], [], []
  for i in range(numClasses):
    precision.append(pm[f"Class {i} Precision"])
    recall.append(pm[f"Class {i} Recall"])
    f1.append(pm[f"Class {i} F1"])
    specificity.append(pm[f"Class {i} Specificity"])
    accuracy.append(pm[f"Class {i} Accuracy"])
    bac.append(pm[f"Class {i} BAC"])

  metrics = np.vstack([precision, recall, f1, specificity, accuracy, bac])
  labels = ["Precision", "Recall", "F1", "Specificity", "Accuracy", "BAC"]

  x = np.arange(numClasses)
  width = 0.15

  fig, ax = plt.subplots(figsize=figsize)

  for i in range(metrics.shape[0]):
    ax.bar(
      x + i * width,
      metrics[i],
      width,
      label=labels[i],
      edgecolor="black"  # Add border around bars
    )

  ax.set_xticks(x + width * 2.5)
  ax.set_xticklabels(classNames, fontsize=fontSize)
  ax.set_ylim(0, 1.05)
  ax.set_ylabel("Score", fontsize=fontSize)
  ax.set_title(title, fontsize=fontSize + 2)
  ax.legend(fontsize=fontSize * 0.85)
  plt.tight_layout()

  # Save the plot if requested.
  if (save):  # Save the plot.
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()

  plt.close(fig)

  if (returnFig):
    return fig

  return None


def PlotErrorMatrix(
  cm,
  classNames=None,
  fontSize=14,
  figsize=(7, 6),
  display=True,
  save=False,
  fileName="ErrorMatrix.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot confusion matrix (error matrix) with highlighted most common errors.

  Parameters:
    cm (array-like): Confusion matrix (2D array).
    classNames (list or None): List of class names. If None, uses class indices.
    fontSize (int): Font size for labels and title. Default is 14.
    figsize (tuple): Figure size. Default is (7, 6).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "ErrorMatrix.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Highlights the most common misclassifications in red.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm
    import numpy as np

    cm = np.array([[50, 2, 1],
                   [10, 45, 5],
                   [0, 3, 47]])
    classNames = ["Class A", "Class B", "Class C"]
    pm.PlotErrorMatrix(
      cm, classNames=classNames,
      fontSize=12,
      figsize=(8, 7),
      display=True,
      save=True,
      fileName="ErrorMatrix.pdf",
      dpi=300,
      returnFig=False
    )
  '''

  cm = np.array(cm)
  numClasses = cm.shape[0]
  if (classNames is None):
    classNames = [str(i) for i in range(numClasses)]

  fig, ax = plt.subplots(figsize=figsize)
  im = ax.imshow(cm, cmap="Blues")
  ax.set_xticks(np.arange(numClasses))
  ax.set_yticks(np.arange(numClasses))
  ax.set_xticklabels(classNames, fontsize=fontSize)
  ax.set_yticklabels(classNames, fontsize=fontSize)
  plt.xlabel("Predicted", fontsize=fontSize)
  plt.ylabel("True", fontsize=fontSize)
  plt.title("Error Matrix (Confusion Matrix)", fontsize=fontSize + 2)

  # Highlight most common errors (off-diagonal max per row).
  for i in range(numClasses):
    for j in range(numClasses):
      color = "red" if (i != j and cm[i, j] == np.max(np.delete(cm[i], i))) else "black"
      ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=fontSize * 0.9)

  plt.colorbar(im, ax=ax)
  plt.tight_layout()

  if (save):
    fig.savefig(fileName, dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()

  plt.close(fig)

  if (returnFig):
    return fig
  return None


def PlotMisclassificationExamples(
  yTrue,
  yPred,
  X=None,
  maxExamples=5,
  fontSize=12,
  figsize=(10, 5),
  display=True,
  save=False,
  fileName="MisclassificationExamples.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot most frequent misclassifications with example indices and optional sample data.

  Parameters:
    yTrue (array-like): True labels.
    yPred (array-like): Predicted labels.
    X (array-like, DataFrame, or None): Optional input samples to display. Default is None.
    maxExamples (int): Max misclassification types to show. Default is 5.
    fontSize (int): Font size for text. Default is 12.
    figsize (tuple): Figure size. Default is (10, 5).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "MisclassificationExamples.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Shows the most frequent misclassification pairs (true, predicted) with counts and optional samples.
    - Useful for qualitative error analysis and debugging.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm
    import numpy as np

    yTrue = np.array([0, 1, 1, 0, 1, 0, 1])
    yPred = np.array([0, 1, 0, 0, 1, 1, 1])
    X = np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6", "sample7"])
    pm.PlotMisclassificationExamples(
      yTrue, yPred, X=X,
      maxExamples=3,
      fontSize=12,
      figsize=(9, 6),
      display=True,
      save=True,
      fileName="MisclassificationExamples.pdf",
      dpi=300,
      returnFig=False
    )
  '''

  yTrue = np.array(yTrue)
  yPred = np.array(yPred)
  mask = yTrue != yPred
  errors = list(zip(yTrue[mask], yPred[mask]))
  counter = Counter(errors)
  mostCommon = counter.most_common(maxExamples)

  fig, ax = plt.subplots(figsize=figsize)
  ax.axis("off")
  lines = []

  for idx, ((t, p), count) in enumerate(mostCommon):
    line = f"{idx + 1}. True: {t}  Pred: {p}  Count: {count}"
    if (X is not None):
      sampleIdxs = np.where((yTrue == t) & (yPred == p))[0][:1]
      for si in sampleIdxs:
        line += f" | Sample idx: {si} | Sample: {X[si] if not hasattr(X, 'iloc') else X.iloc[si]}"
    lines.append(line)

  text = "\n".join(lines) if lines else "No misclassifications."
  ax.text(0, 1, text, fontsize=fontSize, va="top")
  plt.title("Most Frequent Misclassifications", fontsize=fontSize + 2)
  plt.tight_layout()

  if (save):
    fig.savefig(fileName, dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()

  plt.close(fig)

  if (returnFig):
    return fig

  return None


def PlotPredictionConfidenceHistogram(
  yPredProba,
  yPred=None,
  fontSize=14,
  figsize=(8, 5),
  bins=20,
  display=True,
  save=False,
  fileName="PredictionConfidenceHistogram.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot histogram of prediction confidences (predicted probabilities).

  Parameters:
    yPredProba (array-like): Predicted probabilities (2D array for multi-class).
    yPred (array-like or None): Optional predicted classes. If None, uses argmax of yPredProba.
    fontSize (int): Font size for labels and title. Default is 14.
    figsize (tuple): Figure size. Default is (8, 5).
    bins (int): Number of bins for histogram. Default is 20.
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "PredictionConfidenceHistogram.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Shows histogram of predicted probabilities for the predicted class.
    - Useful for assessing model confidence and calibration.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm
    import numpy as np

    yPredProba = np.array([[0.1, 0.7, 0.2],
                          [0.8, 0.1, 0.1],
                          [0.3, 0.4, 0.3],
                          [0.2, 0.2, 0.6]])
    yPred = np.array([1, 0, 1, 2])
    pm.PlotPredictionConfidenceHistogram(
      yPredProba, yPred=yPred,
      fontSize=12,
      figsize=(9, 6),
      bins=10,
      display=True,
      save=True,
      fileName="PredictionConfidenceHistogram.pdf",
      dpi=300,
      returnFig=False
    )
  '''

  yPredProba = np.array(yPredProba)

  if (yPred is None):
    yPred = np.argmax(yPredProba, axis=1)

  confidences = yPredProba[np.arange(len(yPred)), yPred]
  fig, ax = plt.subplots(figsize=figsize)
  ax.hist(confidences, bins=bins, color="skyblue", edgecolor="black", alpha=0.8)
  ax.set_xlabel("Predicted Probability (Confidence)", fontsize=fontSize)
  ax.set_ylabel("Count", fontsize=fontSize)
  ax.set_title("Prediction Confidence Histogram", fontsize=fontSize + 2)
  plt.tight_layout()

  if (save):
    fig.savefig(fileName, dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()
  plt.close(fig)
  if returnFig:
    return fig
  return None


def PlotClassificationResiduals(
  yTrue,
  yPred,
  fontSize=14,
  figsize=(8, 5),
  display=True,
  save=False,
  fileName="ClassificationResiduals.pdf",
  dpi=720,
  returnFig=False,
):
  r'''
  Plot residuals for classification tasks (true - predicted).

  Parameters:
    yTrue (array-like): True labels.
    yPred (array-like): Predicted labels.
    fontSize (int): Font size for labels and title. Default is 14.
    figsize (tuple): Figure size. Default is (8, 5).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "ClassificationResiduals.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm

    yTrue = [0, 1, 1, 0, 1, 0, 1]
    yPred = [0, 1, 0, 0, 1, 1, 1]
    pm.PlotClassificationResiduals(
      yTrue, yPred,
      fontSize=12,
      figsize=(9, 6),
      display=True,
      save=True,
      fileName="ClassificationResiduals.pdf",
      dpi=300,
      returnFig=False
    )
  '''

  import seaborn as sns

  yTrue = np.array(yTrue)
  yPred = np.array(yPred)
  residuals = yTrue - yPred

  fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=figsize,
    gridspec_kw={"height_ratios": [2, 1]}
  )

  # Histogram of residuals.
  bins = np.arange(residuals.min() - 0.5, residuals.max() + 1.5, 1)
  ax1.hist(
    residuals,
    bins=bins,
    color="purple",
    edgecolor="black",
    alpha=0.7,
    rwidth=0.85
  )
  ax1.set_xlabel("Residual (True - Predicted)", fontsize=fontSize)
  ax1.set_ylabel("Count", fontsize=fontSize)
  ax1.set_title("Classification Residuals", fontsize=fontSize + 2)
  ax1.grid(axis="y", alpha=0.3)

  # Swarm plot of residuals (shows distribution and outliers).
  sns.swarmplot(
    x=residuals,
    ax=ax2,
    color="purple",
    size=6
  )
  ax2.set_xlabel("Residual (True - Predicted)", fontsize=fontSize)
  ax2.set_yticks([])
  ax2.set_ylabel("")
  ax2.set_title("Residuals Distribution (Swarm Plot)", fontsize=fontSize)

  # Summary statistics.
  meanRes = np.mean(residuals)
  stdRes = np.std(residuals)
  minRes = np.min(residuals)
  maxRes = np.max(residuals)
  textstr = f"Mean: {meanRes:.2f} | Std: {stdRes:.2f} | Min: {minRes} | Max: {maxRes}"
  ax1.text(
    0.98, 0.98, textstr,
    transform=ax1.transAxes,
    fontsize=fontSize * 0.85,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="gray", alpha=0.7)
  )

  plt.tight_layout()

  # Save the plot if requested.
  if (save):  # Save the plot.
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()

  plt.close(fig)

  if (returnFig):
    return fig

  return None


def PlotFeatureImportance(
  model,
  featureNames,
  title="Feature Importance",
  fontSize=14,
  figsize=(8, 5),
  display=True,
  save=False,
  fileName="FeatureImportance.pdf",
  dpi=720,
  returnFig=False,
  topN=None,
):
  r'''
  Plot feature importance from a trained model.

  Parameters:
    model: Trained model with `feature_importances_` or `coef_` attribute.
    featureNames (list): List of feature names.
    title (str): Title of the plot. Default is "Feature Importance".
    fontSize (int): Font size for labels and title. Default is 14.
    figsize (tuple): Figure size. Default is (8, 5).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "FeatureImportance.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.
    topN (int or None): If specified, show only the top N features. Default is None (show all).

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - Supports models with `feature_importances_` (e.g., tree-based) or `coef_` (e.g., linear models).
    - Displays a bar chart of feature importances.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    y = data.target
    featureNames = data.feature_names

    model = RandomForestClassifier()
    model.fit(X, y)
    pm.PlotFeatureImportance(
      model, featureNames,
      title="Iris Feature Importance",
      fontSize=12,
      figsize=(9, 6),
      display=True,
      save=True,
      fileName="IrisFeatureImportance.pdf",
      dpi=300,
      returnFig=False,
      topN=3
    )
  '''

  if (hasattr(model, "feature_importances_")):
    importances = model.feature_importances_
  elif (hasattr(model, "coef_")):
    importances = np.abs(model.coef_.ravel())
  else:
    raise ValueError("Model does not have feature_importances_ or coef_ attribute.")

  indices = np.argsort(importances)[::-1]
  if (topN is not None):
    indices = indices[:topN]

  fig, ax = plt.subplots(figsize=figsize)
  ax.bar(range(len(indices)), importances[indices], color="royalblue")
  ax.set_xticks(range(len(indices)))
  ax.set_xticklabels(
    np.array(featureNames)[indices],
    rotation=45,
    ha="right",
    fontsize=fontSize * 0.8,
  )
  ax.set_ylabel("Importance", fontsize=fontSize)
  ax.set_title(title, fontsize=fontSize + 2)
  plt.tight_layout()

  # Save the plot if requested.
  if (save):  # Save the plot.
    ext = fileName.split(".")[-1]
    if (ext.lower() == "pdf"):
      try:
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
      except Exception as e:
        print(f"Error saving plot: {e}")
    fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

  if (display):
    plt.show()

  plt.close(fig)

  if (returnFig):
    return fig

  return None


def SampleMonteCarloDirichletFromProbs(probs, T=100, concentration=50.0, rng=None):
  '''
  Create T Monte Carlo softmax probability samples for each row in probs
  by sampling from a Dirichlet distribution with concentration proportional
  to the provided probs.

  Parameters:
    probs (numpy.ndarray): 2D array of shape (N, C) with base probabilities.
    T (int): Number of Monte Carlo samples to draw per instance. Default is 100.
    concentration (float): Concentration parameter for the Dirichlet distribution. Higher values lead to samples closer to the base probs.
    rng (numpy.random.Generator or None): Optional random number generator for reproducibility. If None, a new generator is created.

  Returns:
    numpy.ndarray: 3D array of shape (T, N, C) with Monte Carlo probability samples.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    probs = np.array([[0.7, 0.2, 0.1],
                      [0.1, 0.8, 0.1]])
    T = 500
    samples = pm.SampleMonteCarloDirichletFromProbs(probs, T=T, concentration=30.0)
    print(samples.shape)  # Should be (500, 2, 3).
  '''

  if (rng is None):
    rng = np.random.default_rng()

  probs = np.asarray(probs)

  if (probs.ndim != 2):
    raise ValueError("`probs` must be a 2D array of shape (N, C).")

  N, C = probs.shape
  samples = np.empty((T, N, C), dtype=float)

  for i in range(N):
    # Alpha proportional to probs; add small epsilon to avoid zeros.
    alpha = probs[i] * float(concentration) + 1e-8
    # Draw T samples for this instance.
    samples[:, i, :] = rng.dirichlet(alpha, size=T)
  return samples


def ComputeMonteCarloUncertaintyMeasures(probsMC, eps=1e-12):
  '''
  Given Monte Carlo probability samples, compute useful uncertainty measures.

  Parameters:
    probsMC (numpy.ndarray): 3D array of shape (T, N, C) with Monte Carlo probability samples.
    eps (float): Small value to avoid log(0). Default is 1e-12.

  Returns:
    dict: Dictionary containing the following keys:
      - "predictiveMean": (N, C) array of predictive mean probabilities.
      - "predictiveEntropy": (N,) array of predictive entropy values.
      - "expectedEntropy": (N,) array of expected entropy values.
      - "mutualInformation": (N,) array of mutual information values (epistemic uncertainty).
      - "varTop": (N,) array of variance of top class probabilities across T samples.
      - "predictedIdx": (N,) array of predicted class indices from predictive mean.
      - "predictedConfidence": (N,) array of predicted class confidences from predictive mean.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    probs = np.array([[0.7, 0.2, 0.1],
                      [0.1, 0.8, 0.1]])
    T = 500
    probsMC = pm.SampleMonteCarloDirichletFromProbs(probs, T=T, concentration=30.0)
    uncertaintyMeasures = pm.ComputeMonteCarloUncertaintyMeasures(probsMC)
    print(uncertaintyMeasures["predictiveMean"]) # Should be close to original probs.
  '''

  probsMC = np.asarray(probsMC)

  if (probsMC.ndim != 3):
    raise ValueError("`probsMC` must have shape (T, N, C).")

  T, N, C = probsMC.shape

  # Predictive mean (expected predictive distribution).
  predictiveMean = probsMC.mean(axis=0)  # (N, C).

  # Predictive entropy H[ p(y|x, D) ].
  predictiveEntropy = -np.sum(predictiveMean * np.log(predictiveMean + eps), axis=1)

  # Expected entropy E_t[ H[ p_t(y|x) ] ].
  entropyPerT = -np.sum(probsMC * np.log(probsMC + eps), axis=2)  # (T, N).
  expectedEntropy = entropyPerT.mean(axis=0)  # (N,).

  # Mutual information = predictiveEntropy - expectedEntropy (epistemic uncertainty).
  mutualInformation = predictiveEntropy - expectedEntropy

  # Variance of the top class probability across T samples (another epistemic proxy).
  topProbsPerT = np.max(probsMC, axis=2)  # (T, N).
  varTop = np.var(topProbsPerT, axis=0)

  # Predicted label and confidence from predictive mean.
  predictedIdx = predictiveMean.argmax(axis=1)
  predictedConfidence = predictiveMean[np.arange(N), predictedIdx]

  return {
    "predictiveMean"     : predictiveMean,
    "predictiveEntropy"  : predictiveEntropy,
    "expectedEntropy"    : expectedEntropy,
    "mutualInformation"  : mutualInformation,
    "varTop"             : varTop,
    "predictedIdx"       : predictedIdx,
    "predictedConfidence": predictedConfidence,
  }


def ComputeECE(probabilities, labels, binCount: int = 15, nBins: int | None = None) -> float:
  '''Compute Expected Calibration Error (ECE).

  This function accepts either:
    - a 2D array of per-class probabilities (N x C) with integer class labels.
    - a 1D array of confidences (N,) with labels as 0/1 correctness indicators or class labels.

  The optional legacy keyword `nBins` is accepted for compatibility and overrides binCount when provided.

  Parameters:
    probabilities (list or numpy.ndarray): List/array of predicted probabilities or confidences.
    labels (list or numpy.ndarray): List/array of true labels or correctness indicators.
    binCount (int): Number of bins to use. Default is 15.
    nBins (int or None): Legacy argument name for number of bins. If provided, overrides binCount.

  Returns:
    float: Expected Calibration Error (ECE) value.
  '''

  # Normalize bin count allowing legacy argument name.
  if (nBins is not None):
    bins = int(nBins)
  else:
    bins = int(binCount)

  # Coerce inputs to numpy arrays.
  probsArr = np.asarray(probabilities)
  labelsArr = np.asarray(labels)

  # Return zero for empty inputs.
  if (probsArr.size == 0 or labelsArr.size == 0):
    return 0.0

  # Handle 2D per-class probability arrays by deriving confidences and predictions.
  if (probsArr.ndim == 2):
    confidences = probsArr.max(axis=1)
    predictions = probsArr.argmax(axis=1)
    if (labelsArr.shape[0] != confidences.shape[0]):
      return 0.0
    trueMask = (predictions == labelsArr)
  else:
    # Handle 1D confidences where labels may be correctness indicators or class labels.
    confidences = probsArr.ravel()
    if (labelsArr.shape[0] != confidences.shape[0]):
      return 0.0
    uniqueLabels = set(np.unique(labelsArr))
    if (uniqueLabels.issubset({0, 1})):
      # Labels represent correctness directly.
      trueMask = (labelsArr.astype(int) == 1)
    else:
      # Fallback: convert confidences to binary predictions at 0.5 threshold.
      preds = (confidences >= 0.5).astype(int)
      trueMask = (preds == labelsArr.astype(int))

  # Compute ECE by binning confidences and comparing average confidence vs accuracy per bin.
  ece = 0.0
  boundaries = np.linspace(0.0, 1.0, bins + 1)
  n = len(confidences)
  for i in range(bins):
    lo, hi = boundaries[i], boundaries[i + 1]
    if (i == 0):
      inBin = (confidences >= lo) & (confidences <= hi)
    else:
      inBin = (confidences > lo) & (confidences <= hi)
    cnt = np.sum(inBin)
    if (cnt == 0):
      continue
    accInBin = np.mean(trueMask[inBin].astype(float))
    avgConfInBin = np.mean(confidences[inBin])
    ece += (cnt / n) * np.abs(accInBin - avgConfInBin)

  return float(ece)


def ComputeECEPlotReliability(
  confidences,
  predictions,
  labels,
  nBins=15,
  title="Expected Calibration Error (ECE)",
  fontSize=14,
  figSize=(6, 6),
  display=True,
  save=False,
  fileName="ECE.pdf",
  dpi=720,
  returnFig=False,
  cmap="Blues",
  applyXYLimits=True,
):
  r'''
  Compute Expected Calibration Error (ECE) and plot reliability diagram.

  Parameters:
    confidences (list or numpy.ndarray): List/array of predicted confidences (max class prob).
    predictions (list or numpy.ndarray): List/array of predicted labels.
    labels (list or numpy.ndarray): List/array of true labels.
    nBins (int): number of bins to use.
    title (str): Title of the plot. Default is "Expected Calibration Error (ECE)".
    fontSize (int): Font size for labels and title. Default is 14.
    figSize (tuple): Figure size. Default is (6, 6).
    display (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot. Default is False.
    fileName (str): File name to save the plot. Default is "ECE.pdf".
    dpi (int): DPI for saving the figure. Default is 720.
    returnFig (bool): Whether to return the figure object. Default is False.
    cmap (str): Colormap for the plot. Default is "Blues".
    applyXYLimits (bool): Whether to apply x and y limits [0, 1]. Default is True.

  Returns:
    ece (float): Expected calibration error.
    binAcc (list): List of accuracies per bin.
    binConf (list): List of average confidences per bin.
    binCounts (list): List of sample counts per bin.
    fig (matplotlib.figure.Figure, optional): The matplotlib figure object if returnFig is True.

  Notes:
    - ECE quantifies the difference between predicted confidence and actual accuracy.
    - The reliability diagram visualizes calibration across confidence bins.
    - Saving and displaying the plot are optional and controlled by parameters.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    # You would typically get confidences and correctness from model predictions.
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    T = 500
    probsMC = pm.SampleMonteCarloDirichletFromProbs(probs, T=T, concentration=30.0)
    uncertaintyMeasures = pm.ComputeMonteCarloUncertaintyMeasures(probsMC)
    confidences = uncertaintyMeasures["predictedConfidence"]
    predictions = uncertaintyMeasures["predictedIdx"]
    labels = np.array([0, 1])  # True labels for the examples.

    # Sample data for demonstration:
    # confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    # predictions = np.array([1, 0, 1, 1, 0])
    # labels = np.array([1, 0, 0, 1, 0])

    ece, binAcc, binConf, binCounts = pm.ComputeECEPlotReliability(
      confidences,
      predictions,
      labels,
      nBins=5,
      title="ECE Example",
      fontSize=14,
      figSize=(6, 6),
      display=True,
      save=False,
      fileName="ECE_Example.pdf",
      dpi=300,
      returnFig=False,
      cmap="Blues",
      applyXYLimits=True
    )
    print(f"ECE: {ece}")
    print(f"Bin Accuracies: {binAcc}")
    print(f"Bin Confidences: {binConf}")
    print(f"Bin Counts: {binCounts}")
  '''

  # Ensure inputs are numpy arrays for safe indexing operations.
  confidences = np.asarray(confidences)
  predictions = np.asarray(predictions).astype(int)
  labels = np.asarray(labels).astype(int)

  # Build bin edges and initialize accumulators.
  binEdges = np.linspace(0.0, 1.0, nBins + 1)
  ece = 0.0
  binAcc = []
  binConf = []
  binCounts = []
  nSamples = len(confidences)

  # Validate input lengths to avoid silent errors.
  if ((len(predictions) != nSamples) or (len(labels) != nSamples)):
    raise ValueError(
      "ComputeECEPlotReliability: confidences, predictions and labels must have the same length."
    )

  # Iterate over bins and compute accuracy and average confidence per bin.
  for i in range(nBins):
    low = binEdges[i]
    high = binEdges[i + 1]

    # Include left edge but handle the last bin inclusively on both ends.
    if (i < nBins - 1):
      mask = (confidences > low) & (confidences <= high)
    else:
      mask = (confidences >= low) & (confidences <= high)

    count = int(np.sum(mask))
    if (count == 0):
      binAcc.append(0.0)
      binConf.append(0.0)
      binCounts.append(0)
      continue

    # Compute accuracy and mean confidence for samples in this bin.
    acc = float(np.mean(predictions[mask] == labels[mask]))
    avgConf = float(np.mean(confidences[mask]))
    binAcc.append(acc)
    binConf.append(avgConf)
    binCounts.append(count)

    # Accumulate weighted calibration gap to compute ECE.
    ece += (count / float(nSamples)) * abs(avgConf - acc)

  # Create bin centers for plotting if needed.
  bins = np.arange(len(binAcc)) + 0.5

  # Get colors from the specified colormap.
  if (cmap is None):
    cmap = "Blues"
  cmapColors = GetCmapColors(
    cmap,
    noColors=10,
    darkColorsOnly=True,
    darknessThreshold=0.6
  )
  rndThreeIdxs = np.random.choice(len(cmapColors), size=3, replace=False)
  cmapColors = [cmapColors[i] for i in rndThreeIdxs]
  firstColor = cmapColors[0]
  secondColor = cmapColors[1]
  thirdColor = cmapColors[2]

  if (save or display or returnFig):
    # Create a figure.
    fig = plt.figure(figsize=figSize)

    # Plot bars for difference between acc and conf.
    plt.plot(
      [0, 1], [0, 1],
      linestyle="--",
      color=firstColor,
      label="Perfectly Calibrated"
    )

    # Plot accuracy bars.
    plt.bar(
      bins / float(nBins),
      binAcc,
      width=1.0 / nBins,
      alpha=0.7,
      color=secondColor,
      edgecolor="black",
      label="Accuracy",
    )
    # Plot confidence line.
    plt.plot(
      bins / float(nBins),
      binConf,
      marker="o",
      color=thirdColor,
      label="Confidence",
      alpha=0.9,
    )

    # Apply limits if requested.
    if (applyXYLimits):
      # Set limits and labels.
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])

    plt.xlabel("Confidence", fontsize=fontSize)
    plt.ylabel("Accuracy", fontsize=fontSize)
    plt.title(title, fontsize=fontSize + 2)

    # Update the font of tick labels.
    plt.xticks(fontsize=fontSize * 0.75)
    plt.yticks(fontsize=fontSize * 0.75)

    # Add legend.
    plt.legend(fontsize=fontSize * 0.75)

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}.")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    if (display):
      # Display the plot if requested.
      plt.show()

    plt.close(fig)  # Close the plot.

    if (returnFig):
      return ece, binAcc, binConf, binCounts, fig

  return ece, binAcc, binConf, binCounts


# Plot risk-coverage curve and compute AUC for selective prediction.
def RiskCoverageCurve(
  confidences,
  correctness,
  title="Risk-Coverage (Accuracy vs Coverage)",
  fontSize=14,
  figSize=(6, 6),
  display=True,
  save=False,
  fileName="RiskCoverage.pdf",
  dpi=720,
  returnFig=False,
  color="blue",
):
  '''
  Compute and plot risk (error) vs coverage curve sorted by confidence. The risk-coverage
  curve shows accuracy as a function of coverage when rejecting low-confidence predictions.
  The more area under the curve (AUC), the better the selective prediction performance.
  For example, a model that is perfectly calibrated and accurate will have AUC=1.0,
  while a random model will have AUC close to the accuracy level.

  Parameters:
    confidences (numpy.ndarray): 1D array of prediction confidences.
    correctness (numpy.ndarray): 1D boolean array of correctness (True=correct).
    title (str): Plot title.
    fontSize (int): Font size for plot.
    figSize (tuple): Figure size.
    display (bool): Whether to display the plot.
    save (bool): Whether to save the plot.
    fileName (str): File name to save the plot.
    dpi (int): DPI for saved figure.
    returnFig (bool): Whether to return the figure object.
    color (str): Color for the plot line.

  Returns:
    coverage (numpy.ndarray): coverage levels.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm

    # You would typically get confidences and correctness from model predictions.
    probs = np.array([[0.7, 0.2, 0.1],
                        [0.1, 0.8, 0.1]])
    T = 500
    probsMC = pm.SampleMonteCarloDirichletFromProbs(probs, T=T, concentration=30.0)
    uncertaintyMeasures = pm.ComputeMonteCarloUncertaintyMeasures(probsMC)
    confidences = uncertaintyMeasures["predictedConfidence"]
    predictions = uncertaintyMeasures["predictedIdx"]
    labels = np.array([0, 1])  # True labels.
    correctness = (predictions == labels).astype(int)

    # Sample data for demonstration:
    # confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    # correctness = np.array([1, 0, 1, 1, 0])  # 1=correct, 0=incorrect.

    coverage, accuracy, aucVal, fig = pm.RiskCoverageCurve(
      confidences,
      correctness,
      title="Risk-Coverage (Accuracy vs Coverage)",
      fontSize=14,
      figSize=(6, 6),
      display=True,
      save=False,
      fileName="RiskCoverage.pdf",
      dpi=720,
      returnFig=False,
      color="blue"
    )
  '''

  confidences = np.asarray(confidences)
  correctness = np.asarray(correctness).astype(bool)
  order = np.argsort(-confidences)
  sortedCorrect = correctness[order]
  n = len(sortedCorrect)
  cumCorrect = np.cumsum(sortedCorrect)
  coverage = np.arange(1, n + 1) / float(n)
  accuracy = cumCorrect / np.arange(1, n + 1)

  # Compute AUC under accuracy vs coverage.
  aucVal = np.trapz(accuracy, coverage)

  if (save or display or returnFig):
    fig = plt.figure(figsize=figSize)
    plt.plot(
      coverage,
      accuracy,
      label=f"AUC={aucVal:.3f}",
      color=color,
      linewidth=2,
    )

    plt.xlabel("Coverage", fontsize=fontSize)
    plt.ylabel("Accuracy", fontsize=fontSize)
    plt.title(title, fontsize=fontSize + 2)

    # Update the font of tick labels.
    plt.xticks(fontsize=fontSize * 0.75)
    plt.yticks(fontsize=fontSize * 0.75)

    plt.legend()  # Add legend.
    plt.tight_layout()  # Adjust layout.

    # Save the plot if requested.
    if (save):  # Save the plot.
      ext = fileName.split(".")[-1]
      if (ext.lower() == "pdf"):
        try:
          fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
        except Exception as e:
          print(f"Error saving plot: {e}.")
      fig.savefig(fileName.replace(f".{ext}", ".png"), dpi=dpi, bbox_inches="tight")

    if (display):
      # Display the plot if requested.
      plt.show()

    plt.close(fig)  # Close the plot.

    if (returnFig):
      return coverage, accuracy, aucVal, fig

  return coverage, accuracy, aucVal


def ComputeBrierScore(confidences: List[float], correctness: List[float]) -> float:
  r'''
  Compute Brier Score given prediction confidences and correctness indicators.
  Compute Brier score = mean((conf - correct)^2).

  Parameters:
    confidences (list or numpy.ndarray): List/array of predicted confidences (0.0 to 1.0).
    correctness (list or numpy.ndarray): List/array of correctness indicators (0.0 or 1.0).

  Returns:
    float: Brier score value, or None if inputs are invalid.

  Example
  -------
  .. code-block:: python

    import numpy as np
    import HMB.PerformanceMetrics as pm
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    correctness = [1, 0, 1, 1, 0]
    brierScore = pm.ComputeBrierScore(confidences, correctness)
    print(f"Brier Score: {brierScore}")
  '''

  if (not confidences or len(confidences) != len(correctness)):
    return None
  try:
    diffs = [(c - y) ** 2 for c, y in zip(confidences, correctness)]
    return float(np.mean(diffs))
  except Exception:
    return None


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

  # Define class labels for the confusion matrix.
  classLabels = ["Class 0", "Class 1", "Class 2"]
  # Plot and display the confusion matrix with annotations.
  # To save the figure, set save=True and provide a fileName.
  # To get the figure object, set returnFig=True.
  # PlotConfusionMatrix(
  #   confMatrix,  # Confusion matrix to plot.
  #   classes=classLabels,  # Class labels for the axes.
  #   normalize=False,  # Set to True to normalize the matrix.
  #   title="Confusion Matrix",  # Title of the plot.
  #   annotate=True,  # Annotate cells with values.
  #   fontSize=12,  # Font size for labels and annotations.
  #   figSize=(6, 6),  # Size of the figure.
  #   colorbar=True,  # Show colorbar.
  #   display=True,  # Display the figure.
  #   save=False,  # Set to True to save the figure.
  # )

  # Test the PlotCumulativeGainLiftChart.
  # yTrue = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  # yScores = [0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.4, 0.5, 0.05]
  # PlotCumulativeGainLiftChart(
  #   yTrue, yScores,
  #   title="Cumulative Gain & Lift Chart Example",
  #   display=True,
  #   save=False,
  # )

  # # Test the PlotErrorAnalysis.
  # yTrue = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  # yPred = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0]
  # PlotErrorAnalysis(
  #   yTrue, yPred,
  #   maxExamples=5,
  #   display=True,
  #   save=False
  # )

  # # Test the PlotClasswisePRFBar.
  # cm = [
  #   [50, 2, 1],
  #   [10, 45, 5],
  #   [0, 3, 47]
  # ]
  # classNames = ["Class A", "Class B", "Class C"]
  # PlotClasswisePRFBar(
  #   cm, classNames=classNames,
  #   fontSize=12,
  #   figsize=(9, 6),
  #   display=True,
  #   save=False,
  #   fileName="ClasswisePRFBar.pdf",
  #   dpi=300,
  #   returnFig=False
  # )

  # # Test the PlotErrorMatrix.
  # PlotErrorMatrix(
  #   cm, classNames=classNames,
  #   fontSize=12,
  #   figsize=(7, 6),
  #   display=True,
  #   save=False,
  #   fileName="ErrorMatrix.pdf",
  #   dpi=300,
  #   returnFig=False
  # )

  # # Test the PlotMisclassificationExamples.
  # yTrue = [0, 1, 1, 0, 1, 0, 1]
  # yPred = [0, 1, 0, 0, 1, 1, 1]
  # PlotMisclassificationExamples(
  #   yTrue, yPred,
  #   maxExamples=3,
  #   display=True,
  #   save=False
  # )

  # # Test the PlotPredictionConfidenceHistogram.
  # yPredProba = [
  #   [0.9, 0.1],
  #   [0.2, 0.8],
  #   [0.6, 0.4],
  #   [0.3, 0.7],
  #   [0.95, 0.05],
  #   [0.4, 0.6],
  #   [0.1, 0.9]
  # ]
  # PlotPredictionConfidenceHistogram(
  #   yPredProba,
  #   fontSize=12,
  #   figsize=(8, 5),
  #   bins=10,
  #   display=True,
  #   save=False
  # )

  # # Test the PlotClassificationResiduals.
  # yTrue = [0, 1, 1, 0, 1, 0, 1]
  # yPred = [0, 1, 0, 0, 1, 1, 1]
  # PlotClassificationResiduals(
  #   yTrue, yPred,
  #   fontSize=12,
  #   figsize=(8, 5),
  #   display=True,
  #   save=False
  # )
