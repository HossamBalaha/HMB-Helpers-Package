# Import the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
  Here is how to use this function in a script:

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
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
  }

  # If requested, calculate per-class precision, recall, F1, accuracy, and specificity.
  if (addPerClass):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2.0 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    bac = 0.5 * (recall + specificity)

    metrics.update({
      "Per Class Precision"  : precision,
      "Per Class Recall"     : recall,
      "Per Class F1"         : f1,
      "Per Class Accuracy"   : accuracy,
      "Per Class Specificity": specificity,
      "Per Class BAC"        : bac,
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
  Here is how to use this function in a script:

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

  # Check if normalization is requested.
  if (normalize):  # Normalize the confusion matrix.
    # Normalize the confusion matrix by row sums.
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

  if (cmap is None):
    cmap = plt.cm.Blues  # Default colormap.

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
  Here is how to use this function in a script:

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
    if cmap else [None] * numClasses
  )

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
  Here is how to use this function in a script:

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
    if cmap else [None] * numClasses
  )

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


def PlotCalibrationCurve(
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
    X (pandas.DataFrame or np.ndarray): Feature matrix.
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

    pm.PlotCalibrationCurve(
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


def HistoryPlotter(
  history,  # Dictionary containing training history.
  title,  # Title of the plot.
  metrics=("loss",),  # Tuple or list of metrics to plot.
  xLabel="Epochs",  # Label for x-axis.
  fontSize=14,  # Font size for labels and title.
  doSave=False,  # Whether to save the plot.
  savePath=None,  # Path to save the plot.
  dpi=720,  # DPI for saving the figure.
  colors=None,  # Optional dict of colors for each metric.
  labels=None,  # Optional dict of labels for each metric.
  display=True,  # Whether to display the plot.
  figSize=(10, 5),  # Figure size.
  returnFig=False,  # Whether to return the figure object.
):
  r'''
  Plot training history metrics (e.g., loss, accuracy) for train and validation sets.

  Parameters:
    history (dict): Dictionary containing training history with keys like "train_loss", "val_loss", etc.
    title (str): Title of the plot.
    metrics (tuple or list): Metrics to plot (e.g., ("loss", "accuracy")). Default is ("loss",).
    xLabel (str): Label for x-axis. Default is "Epochs".
    fontSize (int): Font size for labels and title. Default is 14.
    doSave (bool): Whether to save the plot. Default is False.
    savePath (str or None): Path to save the plot. Default is None.
    dpi (int): DPI for saving the figure. Default is 720.
    colors (dict or None): Optional dict mapping metric names to colors.
    labels (dict or None): Optional dict mapping metric names to custom labels.
    display (bool): Whether to display the plot. Default is True.
    figSize (tuple): Figure size in inches. Default is (10, 5).
    returnFig (bool): Whether to return the matplotlib figure object. Default is False.

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
  if (doSave and savePath):  # Save plot.
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
  precision = pm["Per Class Precision"]
  recall = pm["Per Class Recall"]
  f1 = pm["Per Class F1"]
  specificity = pm["Per Class Specificity"]
  accuracy = pm["Per Class Accuracy"]
  bac = pm["Per Class BAC"]

  metrics = np.vstack([precision, recall, f1, specificity, accuracy, bac])
  labels = ["Precision", "Recall", "F1-score", "Specificity", "Accuracy", "BAC"]

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
  ax.set_title(
    "Classwise Performance Metrics",
    fontsize=fontSize + 2
  )
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

  from collections import Counter

  yTrue = np.array(yTrue)
  yPred = np.array(yPred)
  mask = yTrue != yPred
  errors = list(zip(yTrue[mask], yPred[mask]))
  counter = Counter(errors)
  most_common = counter.most_common(maxExamples)

  fig, ax = plt.subplots(figsize=figsize)
  ax.axis("off")
  lines = []

  for idx, ((t, p), count) in enumerate(most_common):
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

  # Histogram of residuals
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


def PlotAll(
  X,
  yTrue,
  yPred,
  yPredProba,
  classNames,
  classifier,
  display=True,
  save=False,
  fontSize=14,
  dpi=720,
):
  r'''
  Generate all file plots.

  Parameters:
    X (array-like or DataFrame): Input samples (optional, for error analysis).
    yTrue (array-like): True labels.
    yPred (array-like): Predicted labels.
    yPredProba (array-like or None): Predicted probabilities (optional, for confidence histogram).
    classNames (list or None): List of class names. If None, uses class indices.
    classifier (object or None): Classifier object with "predict_proba" method (optional).
    display (bool): Whether to display the plots. Default is True.
    save (bool): Whether to save the plots. Default is False.
    fontSize (int): Font size for labels and titles. Default is 14.
    dpi (int): DPI for saving the figures. Default is 720.

  Example
  -------
  .. code-block:: python

    import HMB.PerformanceMetrics as pm
    import numpy as np

    X = np.array(["sample1", "sample2", "sample3", "sample4", "sample5", "sample6", "sample7"])
    yTrue = np.array([0, 1, 1, 0, 1, 0, 1])
    yPred = np.array([0, 1, 0, 0, 1, 1, 1])
    yPredProba = np.array([[0.9, 0.1],
                          [0.2, 0.8],
                          [0.6, 0.4],
                          [0.3, 0.7],
                          [0.95, 0.05],
                          [0.4, 0.6],
                          [0.1, 0.9]])
    pm.PlotAll(
      X, yTrue, yPred, yPredProba,
      classNames=["Class 0", "Class 1"],
      classifier=None,
      display=True,
      save=False,
      fontSize=12,
      dpi=300,
    )
  '''

  # Confusion Matrix.
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(yTrue, yPred)

  PlotConfusionMatrix(
    cm,
    classes=classNames if (classNames) is not None else [str(i) for i in range(cm.shape[0])],
    normalize=False,
    title="Confusion Matrix",
    annotate=True,
    fontSize=fontSize,
    figSize=(6, 6),
    colorbar=True,
    display=display,
    save=save,
    fileName="ConfusionMatrix.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotROCAUCCurve(
    yTrue,
    yPred=yPredProba if (yPredProba is not None) else yPred,
    classes=classNames if (classNames) is not None else [str(i) for i in range(cm.shape[0])],
    areProbabilities=True if (yPredProba is not None) else False,
    fontSize=fontSize,
    title="ROC Curve",
    display=display,
    save=save,
    fileName="ROCAUCCurve.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotPRCCurve(
    yTrue,
    yPred=yPredProba if (yPredProba is not None) else yPred,
    classes=classNames if (classNames) is not None else [str(i) for i in range(cm.shape[0])],
    areProbabilities=True if (yPredProba is not None) else False,
    fontSize=fontSize,
    title="Precision-Recall Curve",
    display=display,
    save=save,
    fileName="PRCCurve.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotCumulativeGainLiftChart(
    yTrue,
    yScores=yPredProba if (yPredProba is not None) else yPred,
    title="Cumulative Gain & Lift Chart",
    display=display,
    save=save,
    fileName="CumulativeGainLiftChart.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotErrorAnalysis(
    yTrue, yPred, X=X,
    maxExamples=5,
    display=display,
    save=save,
    fileName="ErrorAnalysis.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotClasswisePRFBar(
    cm, classNames=classNames if (classNames) is not None else [str(i) for i in range(cm.shape[0])],
    fontSize=fontSize,
    figsize=(9, 6),
    display=display,
    save=save,
    fileName="ClasswisePRFBar.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotErrorMatrix(
    cm, classNames=classNames if (classNames) is not None else [str(i) for i in range(cm.shape[0])],
    fontSize=fontSize,
    figsize=(7, 6),
    display=display,
    save=save,
    fileName="ErrorMatrix.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  PlotMisclassificationExamples(
    yTrue, yPred, X=X,
    maxExamples=5,
    fontSize=fontSize,
    figsize=(10, 5),
    display=display,
    save=save,
    fileName="MisclassificationExamples.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )

  if ((yPredProba is not None) or (classifier is not None)):
    if ((yPredProba is None) and (classifier is not None) and hasattr(classifier, "predict_proba")):
      yPredProba = classifier.predict_proba(X)
    PlotPredictionConfidenceHistogram(
      yPredProba,
      yPred=yPred,
      fontSize=fontSize,
      figsize=(8, 5),
      bins=20,
      display=display,
      save=save,
      fileName="PredictionConfidenceHistogram.pdf" if (save) else "",
      dpi=dpi,
      returnFig=False,
    )

  PlotClassificationResiduals(
    yTrue, yPred,
    fontSize=fontSize,
    figsize=(8, 5),
    display=display,
    save=save,
    fileName="ClassificationResiduals.pdf" if (save) else "",
    dpi=dpi,
    returnFig=False,
  )


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
