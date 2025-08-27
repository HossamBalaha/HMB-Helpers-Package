# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt


def CalculatePerformanceMetrics(
  confMatrix,  # Confusion matrix (2D list or numpy array).
  eps=1e-10,  # Small value to avoid division by zero.
  addWeightedAverage=False,  # Whether to include weighted averages in the output.
):
  '''
  Calculate performance metrics from a confusion matrix.

  Parameters:
    confMatrix (list or numpy.ndarray): Confusion matrix representing the classification results.
    eps (float): Small value to avoid division by zero. Default is 1e-10.
    addWeightedAverage (bool): Whether to include weighted averages in the output. Default is False.

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
  f1 = 2.0 * precision * recall / (precision + recall)
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
  f1 = 2.0 * precision * recall / (precision + recall)
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
  f1 = 2.0 * precision * recall / (precision + recall)
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
    fig.savefig(fileName, dpi=720, bbox_inches="tight")

  # Display the plot if requested.
  if (display):  # Display the plot.
    plt.show()

  plt.close()  # Close the plot.

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

  if (save):
    # Save the plot if requested.
    fig.savefig(fileName, dpi=dpi, bbox_inches="tight")

  if (display):
    # Display the plot if requested.
    plt.show()

  plt.close()  # Close the plot.

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

  # Add grid lines to the plot.
  plt.grid(True)

  if (showLegend):
    # Show legend if requested.
    plt.legend(fontsize=fontSize * 0.75)

  # Tight the layout to ignore wasted spaces.
  plt.tight_layout()

  if (save):
    # Save the plot if requested.
    fig.savefig(fileName, dpi=dpi, bbox_inches="tight")

  if (display):
    # Display the plot if requested.
    plt.show()

  plt.close()  # Close the plot.

  if (returnFig):
    # Return the figure object if requested.
    return fig


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
  PlotConfusionMatrix(
    confMatrix,  # Confusion matrix to plot.
    classes=classLabels,  # Class labels for the axes.
    normalize=False,  # Set to True to normalize the matrix.
    title="Confusion Matrix",  # Title of the plot.
    annotate=True,  # Annotate cells with values.
    fontSize=12,  # Font size for labels and annotations.
    figSize=(6, 6),  # Size of the figure.
    colorbar=True,  # Show colorbar.
    display=True,  # Display the figure.
    save=False,  # Set to True to save the figure.
  )
