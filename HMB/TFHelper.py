import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


def MCDropoutPredictions(model, genObj, steps, T=30):
  r'''
  Run T stochastic forward passes with dropout enabled (model called with training=True).
  Returns a numpy array of shape (noSamples, noClasses, T).

  Parameters:
    model (tensorflow.keras.Model): Trained model with Dropout layers.
    genObj (tensorflow.keras.utils.Sequence): Data generator.
    steps (int): Number of steps to cover the subset.
    T (int): Number of stochastic forward passes.

  Returns:
    numpy.ndarray: Array of shape (noSamples, noClasses, T) with all predictions.
  '''

  # Validate T is at least 1.
  assert T >= 1, "MCDropoutPredictions: T must be at least 1."
  # Validate steps is at least 1.
  assert steps >= 1, "MCDropoutPredictions: steps must be at least 1."
  # Get number of samples from generator.
  noSamples = genObj.samples
  # Reset generator to start from beginning.
  genObj.reset()
  # Get one batch to determine number of classes.
  x0, _ = next(genObj)
  preds0 = model(x0, training=True).numpy()
  noClasses = preds0.shape[1]
  # Allocate array to store all predictions.
  predsAll = np.zeros((noSamples, noClasses, T), dtype=np.float32)
  # Iterate over T Monte Carlo dropout passes.
  for t in range(T):
    # Reset generator for each pass.
    genObj.reset()
    idx = 0
    # Iterate over batches in this pass.
    for s in range(steps):
      try:
        batchX, _ = next(genObj)
      except StopIteration:
        break
      # Call model with training=True to activate dropout.
      predsBatch = model(batchX, training=True).numpy()
      batchSize = predsBatch.shape[0]
      # Ensure we do not exceed noSamples boundary.
      endIdx = min(idx + batchSize, noSamples)
      actualBatchSize = endIdx - idx
      predsAll[idx:endIdx, :, t] = predsBatch[:actualBatchSize]
      idx += batchSize
      # Break if all samples are processed.
      if (idx >= noSamples):
        break
  return predsAll


def ComputeUncertaintyStats(predsAll, eps=1e-12):
  r'''
  Given predsAll of shape (noSamples, noClasses, T), compute uncertainty metrics.

  Parameters:
    predsAll (numpy.ndarray): Array of shape (noSamples, noClasses, T).
    eps (float): Small value to avoid log(0).

  Returns:
    dict: Dictionary with computed statistics including entropy and mutual information.
  '''
  # Compute mean probability across T draws.
  meanProbs = np.mean(predsAll, axis=2)
  # Compute standard deviation across T draws.
  stdProbs = np.std(predsAll, axis=2)
  # Predicted label from mean probability.
  predLabels = np.argmax(meanProbs, axis=1)
  # Mean confidence is max of mean probabilities.
  meanConfidence = np.max(meanProbs, axis=1)
  # Compute confidence per sample per T (max per sample per T).
  maxPerT = np.max(predsAll, axis=1)
  # Standard deviation of confidence across T draws.
  stdConfidence = np.std(maxPerT, axis=1)
  # Predictive entropy: H[ E_t p ].
  predictiveEntropy = -np.sum(meanProbs * np.log(meanProbs + eps), axis=1)
  # Expected entropy: E_t [ H[ p_t ] ].
  entPerT = -np.sum(predsAll * np.log(predsAll + eps), axis=1)
  expectedEntropy = np.mean(entPerT, axis=1)
  # Mutual information (BALD): predictiveEntropy - expectedEntropy.
  mutualInfo = predictiveEntropy - expectedEntropy
  return {
    "meanProbs"        : meanProbs,
    "stdProbs"         : stdProbs,
    "predLabels"       : predLabels,
    "meanConfidence"   : meanConfidence,
    "stdConfidence"    : stdConfidence,
    "predictiveEntropy": predictiveEntropy,
    "expectedEntropy"  : expectedEntropy,
    "mutualInfo"       : mutualInfo,
  }


def SaveTopUncertainImages(
  testDf,
  meanProbs,
  mutualInfo,
  storePath,
  labelEncoder,
  topN=16,
  imgSize=(128, 128),
  dpi=720,
):
  r'''
  Save a grid of the top-N images ranked by mutual information for inspection.

  Parameters:
    testDf (pandas.DataFrame): DataFrame with test image paths and labels.
    meanProbs (numpy.ndarray): Array of shape (noSamples, noClasses) with mean probabilities.
    mutualInfo (numpy.ndarray): Array of shape (noSamples,) with mutual information values.
    storePath (str): File path to save the resulting figure (e.g., "TopUncertainImages.pdf").
    labelEncoder (sklearn.preprocessing.LabelEncoder): Fitted label encoder.
    topN (int): Number of top uncertain images to save.
    imgSize (tuple): (height, width) for resizing images.
    dpi (int): Dots per inch for saved figure.
  '''

  import matplotlib.pyplot as plt

  # Reset test DataFrame index so integer positions match 0..N-1.
  df = testDf.reset_index(drop=True).copy()
  noDf = len(df)
  # Ensure mutualInfo is a numpy 1D array.
  mutualInfo = np.asarray(mutualInfo).reshape(-1)
  # Validate mutualInfo length matches DataFrame length.
  if (mutualInfo.size != noDf):
    raise ValueError(f"SaveTopUncertainImages: mutualInfo size ({mutualInfo.size}) != DataFrame length ({noDf})")
  # Validate DataFrame is not empty.
  if (noDf == 0):
    return
  # Rank by descending mutual information and select top entries.
  ranks = np.argsort(-mutualInfo)
  selCount = min(topN, len(ranks))
  sel = ranks[:selCount]
  # Compute grid size from actual selection count.
  cols = int(np.ceil(np.sqrt(selCount))) if (selCount > 0) else 1
  rows = int(np.ceil(selCount / cols)) if (selCount > 0) else 1
  # Create figure with subplots.
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
  # Ensure axes is a flat array for easy indexing.
  axes = np.array(axes).reshape(-1)
  # Iterate over selected uncertain samples.
  for i, idx in enumerate(sel):
    # Get image path from DataFrame.
    try:
      imgPath = df.iloc[int(idx)]["image_path"]
    except Exception:
      imgPath = None
    # Load image or create placeholder.
    if (imgPath is None):
      im = Image.new("RGB", imgSize, (255, 255, 255))
    else:
      try:
        im = Image.open(imgPath).convert("RGB")
        im = im.resize(imgSize)
      except Exception:
        im = Image.new("RGB", imgSize, (255, 255, 255))
    # Display image on subplot.
    axes[i].imshow(im)
    # Get true label from DataFrame.
    trueLabel = df.iloc[int(idx)]["label"] if ("label" in df.columns) else ""
    # Get predicted label from meanProbs (not category_encoded).
    predLabelIdx = np.argmax(meanProbs[int(idx)])
    try:
      predLabel = labelEncoder.inverse_transform([predLabelIdx])[0]
    except Exception:
      predLabel = str(predLabelIdx)
    # Set subplot title with uncertainty and labels.
    axes[i].set_title(
      f"U: {float(mutualInfo[int(idx)]):.3f}\nTrue: {trueLabel}\nPred: {predLabel}",
      fontsize=10,
    )
    axes[i].axis("off")
  # Turn off any leftover axes.
  for j in range(selCount, len(axes)):
    axes[j].axis("off")
  # Apply tight layout once.
  plt.tight_layout()
  # Save figure to PDF.
  plt.savefig(storePath, dpi=dpi)
  plt.close()


def AggregateClasswiseUncertainty(exportDf, meanProbs, mutualInfo):
  r'''
  Return a DataFrame with class-wise mean mutual information and mean confidence.

  Parameters:
    exportDf (pandas.DataFrame): DataFrame with per-image predictions and uncertainties.
    meanProbs (numpy.ndarray): Array of shape (noSamples, noClasses) with mean probabilities.
    mutualInfo (numpy.ndarray): Array of shape (noSamples,) with mutual information values.

  Returns:
    pandas.DataFrame: DataFrame with class-wise aggregated uncertainty metrics.
  '''

  # Create copy of export DataFrame.
  df = exportDf.copy()
  # Add mutual information column.
  df["mutualInfo"] = mutualInfo
  # Compute mean confidence from meanProbs.
  df["meanConfidence"] = np.max(meanProbs, axis=1)
  # Validate required columns exist.
  if ("label" not in df.columns):
    raise ValueError("AggregateClasswiseUncertainty: 'label' column not found in `exportDf`.")
  # Group by label and aggregate metrics.
  grouped = df.groupby("label").agg(
    n_samples=("image_path", "count"),
    mean_mutual_info=("mutualInfo", "mean"),
    median_mutual_info=("mutualInfo", "median"),
    mean_confidence=("meanConfidence", "mean"),
  ).reset_index()
  return grouped


def ComputeEnergyScore(logits):
  r'''
  Compute energy score = -logsumexp(logits) per sample for OOD detection.

  Parameters:
    logits (numpy.ndarray): Shape (N, C) logits from model.

  Returns:
    numpy.ndarray: Shape (N,) energy scores (lower = more in-distribution).
  '''

  # Use numerically stable logsumexp.
  a = np.max(logits, axis=1, keepdims=True)
  lse = a + np.log(np.sum(np.exp(logits - a), axis=1, keepdims=True))
  # Return negative log-sum-exp as energy score.
  return -np.squeeze(lse)


def FindGlobalPoolingLayer(model):
  r'''
  Find global pooling layer. First look for standard Keras global pooling layers, then use name heuristics,
  then fallback to any layer with output rank 2.

  Parameters:
    model (keras.models.Model): Keras model object.

  Returns:
    keras.models.Model: Keras model object.
  '''

  from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
  for layer in model.layers[::-1]:
    if (isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPooling2D))):
      return layer
    # Name heuristics.
    lname = layer.name.lower()
    if ("gap" in lname or "global" in lname):
      return layer
  # Fallback: look for layer with output rank 2.
  for layer in model.layers[::-1]:
    try:
      if (len(layer.output_shape) == 2):
        return layer
    except Exception:
      continue
  return None


def BuildPretrainedAttentionModel(
  baseModelString,
  attentionBlockStr,
  inputShape,
  numClasses,
  optimizer=None,
  compile=True,
):
  r'''
  Reconstruct model architecture used in training so weights can be loaded.
  This mirrors the model construction portion of Code.py (no training code here).

  Parameters:
    baseModelString (str): backbone model name (e.g. "Xception").
    attentionBlockStr (str): attention block name (e.g. "CBAM").
    inputShape (tuple): input shape used for training (H, W, C).
    numClasses (int): number of target classes.
    optimizer (tensorflow.keras.optimizers.Optimizer or None): optional optimizer to use;
      if None, defaults to Adam with lr=1e-4.
    compile (bool): whether to compile the model; if False, returns uncompiled model.

  Returns:
    model (tensorflow.keras.Model): compiled model ready to load weights and predict.
  '''

  from tensorflow.keras.models import Model
  from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
  from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

  # Load the specified backbone model.
  if (baseModelString == "Xception"):
    from tensorflow.keras.applications import Xception
    baseModel = Xception(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  elif (baseModelString == "ResNet50V2"):
    from tensorflow.keras.applications import ResNet50V2
    baseModel = ResNet50V2(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  elif (baseModelString == "InceptionV3"):
    from tensorflow.keras.applications import InceptionV3
    baseModel = InceptionV3(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  elif (baseModelString == "DenseNet121"):
    from tensorflow.keras.applications import DenseNet121
    baseModel = DenseNet121(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  elif (baseModelString == "MobileNetV2"):
    from tensorflow.keras.applications import MobileNetV2
    baseModel = MobileNetV2(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  elif (baseModelString == "EfficientNetB0"):
    from tensorflow.keras.applications import EfficientNetB0
    baseModel = EfficientNetB0(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  elif (baseModelString == "NASNetMobile"):
    from tensorflow.keras.applications import NASNetMobile
    baseModel = NASNetMobile(
      weights="imagenet",
      include_top=False,
      input_shape=inputShape
    )
  else:
    raise ValueError(f"Unsupported backbone model: {baseModelString}")

  # Freeze the base model initially.
  baseModel.trainable = False

  # Create the specified attention block.
  if (attentionBlockStr == "CBAM"):
    from HMB.TFAttentionBlocks import CBAMBlock
    attnBlock = CBAMBlock(ratio=8, kernelSize=7)
    print("Using CBAM attention block with ratio=8 and kernel size=7.")
  elif (attentionBlockStr == "BAM"):
    from HMB.TFAttentionBlocks import BAMBlock
    attnBlock = BAMBlock(reduction=16, dilationRates=(1, 2, 4))
    print("Using BAM attention block with reduction=16 and dilation rates (1, 2, 4).")
  elif (attentionBlockStr == "SE"):
    from HMB.TFAttentionBlocks import SEBlock
    attnBlock = SEBlock(ratio=16)
    print("Using SE attention block with ratio=16.")
  elif (attentionBlockStr == "ECA"):
    from HMB.TFAttentionBlocks import ECABlock
    attnBlock = ECABlock(gamma=2, b=1)
    print("Using ECA attention block with gamma=2 and b=1.")
  elif (attentionBlockStr == "GC"):
    from HMB.TFAttentionBlocks import GCBlock
    attnBlock = GCBlock(reduction=16)
    print("Using GC attention block with reduction=16.")
  else:
    raise ValueError(f"Unsupported attention block: {attentionBlockStr}")

  # Build the model architecture.
  x = baseModel.output
  x = attnBlock(x)
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation="relu")(x)
  x = Dropout(0.5)(x)

  if (numClasses == 2):
    predictions = Dense(1, activation="sigmoid", dtype="float32")(x)
  else:
    predictions = Dense(numClasses, activation="softmax", dtype="float32")(x)

  model = Model(inputs=baseModel.input, outputs=predictions)

  if (compile):
    if (optimizer is None):
      from tensorflow.keras.optimizers import Adam
      # Default optimizer with lower LR for fine-tuning.
      optimizer = Adam(learning_rate=1e-4)
    else:
      # Clone the provided optimizer to avoid modifying the original optimizer's state.
      optimizer = tf.keras.optimizers.deserialize(
        tf.keras.optimizers.serialize(optimizer),
        custom_objects={"Adam": tf.keras.optimizers.Adam}
      )

    lossFn = SparseCategoricalCrossentropy() if (numClasses > 2) else BinaryCrossentropy()

    # Compile the model.
    model.compile(
      optimizer=optimizer,
      loss=lossFn,
      metrics=["accuracy"],
    )

  return model


def CreateFitPretrainedAttentionModel(
  trainGenNew,
  validGenNew,
  baseModelString,
  attentionBlockStr,
  inputShape,
  numClasses,
  callbacks,
  modelCheckpointPath=None,
  initialEpochs=10,
  fineTuneEpochs=20,
  fineTuneAt=100,
  optimizer=None,
  storageDir="History",
  verbose=1,
):
  r'''
  Create a CNN model with a specified backbone and attention block.

  Parameters:
    trainGenNew (ImageDataGenerator): Training data generator.
    validGenNew (ImageDataGenerator): Validation data generator.
    baseModelString (str): Backbone model name (e.g., "Xception").
    attentionBlockStr (str): Attention block name (e.g., "CBAM").
    inputShape (tuple): Input shape for the model.
    numClasses (int): Number of output classes.
    callbacks (list): List of Keras callbacks for training.
    modelCheckpointPath (str or None): Optional path to save the best model; if None, defaults to "BestModel.keras".
    initialEpochs (int): Number of initial training epochs.
    fineTuneEpochs (int): Number of fine-tuning epochs.
    fineTuneAt (int): Layer index to start fine-tuning from (default: 100).
    optimizer (tensorflow.keras.optimizers.Optimizer or None): Optional optimizer to use; if None, defaults to Adam with lr=1e-3.
    storageDir (str): Directory to save the best model and history.
    verbose (int): Verbosity level for training (0 = silent, 1 = progress bar, 2 = one line per epoch).

  Returns:
    model (tensorflow.keras.Model): The trained Keras model.
    history (tensorflow.keras.callbacks.History): Training history for initial training.
    historyFine (tensorflow.keras.callbacks.History): Training history for fine-tuning.
    configs (dict): Dictionary of training configurations and parameters.
  '''

  from tensorflow.keras.losses import SparseCategoricalCrossentropy

  model = BuildPretrainedAttentionModel(
    baseModelString=baseModelString,
    attentionBlockStr=attentionBlockStr,
    inputShape=inputShape,
    numClasses=numClasses,
    optimizer=optimizer,
  )

  # Step 1: Frozen training.
  # Train model on frozen backbone.
  history = model.fit(
    trainGenNew,  # Training data generator.
    validation_data=validGenNew,  # Validation data generator.
    epochs=initialEpochs,  # Number of epochs for initial training.
    callbacks=callbacks,  # Callbacks for training.
    verbose=verbose,  # Verbosity level.
  )

  # Step 2: Fine-tuning.
  # Unfreeze part of the base model for fine-tuning.
  model.trainable = True  # Set base model trainable.
  for layer in model.layers[:fineTuneAt]:
    layer.trainable = False  # Keep earlier layers frozen.

  if (optimizer is None):
    from tensorflow.keras.optimizers import Adam
    # Default optimizer with lower LR for fine-tuning.
    optimizer = Adam(learning_rate=1e-4)
  else:
    # Clone the provided optimizer to avoid modifying the original optimizer's state.
    optimizer = tf.keras.optimizers.deserialize(
      tf.keras.optimizers.serialize(optimizer),
      custom_objects={"Adam": tf.keras.optimizers.Adam}
    )

  model.compile(
    # Lower LR for fine-tuning.
    optimizer=optimizer,
    # Use sparse categorical crossentropy loss.
    loss=SparseCategoricalCrossentropy(),
    # Track accuracy metric.
    metrics=["accuracy"],
  )

  totalEpochs = initialEpochs + fineTuneEpochs  # Compute total epochs.
  # Continue training with unfrozen layers.
  historyFine = model.fit(
    trainGenNew,  # Training data generator.
    validation_data=validGenNew,  # Validation data generator.
    epochs=totalEpochs,  # Total number of epochs.
    initial_epoch=history.epoch[-1] + 1,  # Start from last epoch + 1.
    callbacks=callbacks,  # Callbacks for training.
    verbose=verbose,  # Verbosity level.
  )

  # Load the best model weights after training.
  # model.load_weights(f"BestModel.weights.h5") --- IGNORE ---
  if (modelCheckpointPath is None):
    modelCheckpointPath = os.path.join(storageDir, f"BestModel.keras")
  model = tf.keras.models.load_model(modelCheckpointPath)  # Load the best saved model.
  # # Store the best model after fine-tuning as pickle file.
  # picklePath = os.path.join(storageDir, f"BestModel.pkl")
  # with open(picklePath) as f:
  #   pickle.dump(model, f)

  configs = {
    "baseModelString"    : str(baseModelString),
    "attentionBlockStr"  : str(attentionBlockStr),
    "inputShape"         : inputShape,
    "numClasses"         : int(numClasses),
    "initialEpochs"      : int(initialEpochs),
    "fineTuneEpochs"     : int(fineTuneEpochs),
    "fineTuneAt"         : int(fineTuneAt),
    "optimizer"          : str(optimizer),
    "modelCheckpointPath": str(modelCheckpointPath),
    "storageDir"         : str(storageDir),
  }
  # Return the trained model and histories.
  return model, history, historyFine, configs


def TrainPretrainedAttentionModelFromDataFrame(
  dataFrame,
  columnsMap={"imagePath": "image_path", "categoryEncoded": "category_encoded", "split": "split"},
  labelEncoder=None,
  imgShape=(512, 512, 3),
  batchSize=32,
  baseModelString="Xception",
  attentionBlockStr="CBAM",
  initialEpochs=10,
  fineTuneEpochs=20,
  augmentationConfigs=None,
  monitor="val_loss",
  earlyStoppingPatience=10,
  ensureCUDA=True,
  storageDir="History",
  dpi=720,
  verbose=1,
):
  r'''
  Perform training, evaluation, and reporting for the model.
  This function handles data preparation, model training with callbacks, evaluation on the test set,
  and saving of results including confusion matrix, classification report, performance metrics, and
  training history.

  Parameters:
    dataFrame (pandas.DataFrame): DataFrame containing image paths and labels.
    columnsMap (dict): Mapping of required column names in the DataFrame.
    labelEncoder (sklearn.preprocessing.LabelEncoder or None): Optional label encoder for encoding string labels to integers; if None, a new LabelEncoder will be created and fitted on the "categoryEncoded" column.
    imgShape (tuple): Input shape for the model (height, width, channels).
    batchSize (int): Batch size for training and evaluation.
    baseModelString (str): Base model architecture (e.g., "Xception").
    attentionBlockStr (str): Attention block type (e.g., "CBAM").
    initialEpochs (int): Number of initial training epochs with the base model frozen.
    fineTuneEpochs (int): Number of fine-tuning epochs after unfreezing the base model.
    augmentationConfigs (dict or None): Optional dictionary of augmentation parameters for `ImageDataGenerator`.
    monitor (str): Metric to monitor for callbacks (e.g., "val_loss").
    earlyStoppingPatience (int): Number of epochs with no improvement before early stopping.
    ensureCUDA (bool): Whether to check for CUDA availability and raise an error if not found.
    storageDir (str): Directory where training history and results will be saved.
    dpi (int): Dots per inch for saving figures.
    verbose (int): Verbosity level for training (0 = silent, 1 = progress bar, 2 = one line per epoch).
  '''

  # Verify that the dataFrame contains the required columns.
  requiredColumns = set(columnsMap.values())
  if (not requiredColumns.issubset(dataFrame.columns)):
    raise ValueError(f"DataFrame must contain columns: {requiredColumns}")

  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix, classification_report
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
  from HMB.Utils import WritePickleFile
  from HMB.Initializations import (
    DoRandomSeeding, EnsureCUDAAvailable, ClearTensorFlowSession, UpdateMatplotlibSettings
  )
  from HMB.PerformanceMetrics import PlotConfusionMatrix, CalculatePerformanceMetrics, HistoryPlotter

  ClearTensorFlowSession()
  DoRandomSeeding()
  UpdateMatplotlibSettings()

  if (ensureCUDA):
    EnsureCUDAAvailable("tensorflow")

  if (augmentationConfigs is None):
    augmentationConfigs = {
      "rotation_range"    : 2,  # Small rotation.
      "zoom_range"        : 0.05,  # Slight zoom.
      "width_shift_range" : 0.05,  # Small width shift.
      "height_shift_range": 0.05,  # Small height shift.
      "shear_range"       : 0.05,  # Small shear.
      "horizontal_flip"   : True,  # Horizontal flip.
      "vertical_flip"     : True,  # Vertical flip.
      "fill_mode"         : "nearest",  # Fill mode for new pixels.
    }

  # Define augmentation for training.
  trGen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values.
    **augmentationConfigs  # Unpack augmentation parameters.
  )

  # Define generator for validation and test without augmentation.
  tsGen = ImageDataGenerator(rescale=1.0 / 255)

  # Create data generators from the paths.
  splitCol = columnsMap["split"]
  trainDfNew = dataFrame[dataFrame[splitCol] == "train"].copy()  # Training DataFrame.
  validDfNew = dataFrame[dataFrame[splitCol] == "val"].copy()  # Validation DataFrame.
  testDfNew = dataFrame[dataFrame[splitCol] == "test"].copy()  # Test DataFrame.

  xCol = columnsMap["imagePath"]
  yCol = columnsMap["categoryEncoded"]
  imgSize = imgShape[:2]  # Extract image size from shape.
  trainGenNew = trGen.flow_from_dataframe(
    trainDfNew,
    x_col=xCol,
    y_col=yCol,
    target_size=imgSize,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=True,
    batch_size=batchSize,
  )

  validGenNew = tsGen.flow_from_dataframe(
    validDfNew,
    x_col=xCol,
    y_col=yCol,
    target_size=imgSize,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=True,
    batch_size=batchSize,
  )

  testGenNew = tsGen.flow_from_dataframe(
    testDfNew,
    x_col="image_path",
    y_col="category_encoded",
    target_size=imgSize,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=False,
    batch_size=batchSize,
  )

  # Create storage directory if it doesn't exist.
  os.makedirs(storageDir, exist_ok=True)

  # Create a figure to display some augmented images.
  plt.figure(figsize=(12, 8))  # Initialize augmentation samples figure.
  # Get a batch of augmented images.
  augmentedImages, augmentedLabels = next(trainGenNew)
  # Display a few augmented images.
  for i in range(6):
    labelIdx = int(augmentedLabels[i])
    if (labelEncoder is not None):
      labelName = labelEncoder.inverse_transform([labelIdx])[0]
    else:
      labelName = str(labelIdx)
    plt.subplot(1, 6, i + 1)
    plt.imshow(augmentedImages[i])
    plt.axis("off")
    plt.title("Label: {}".format(labelName))
  plt.tight_layout()  # Adjust layout.
  # Save augmentation samples.
  figStoragePath = os.path.join(storageDir, "AugmentedImagesSamples.pdf")
  plt.savefig(figStoragePath, dpi=dpi, bbox_inches="tight")

  # Determine number of classes from the training generator.
  # Count classes from generator mapping.
  numClasses = len(trainGenNew.class_indices)

  modelCheckpointPath = os.path.join(storageDir, f"BestModel.keras")
  callbacks = [
    ModelCheckpoint(
      modelCheckpointPath,  # File name for best model.
      save_best_only=True,  # Save only the best model.
      save_weights_only=False,  # Save the entire model.
      monitor=monitor,  # Monitor validation accuracy.
      mode="min",  # Maximize validation accuracy.
      verbose=verbose,  # Verbose output when saving best model.
    ),
    # Stop early on no improvement.
    EarlyStopping(
      monitor=monitor,  # Monitor validation accuracy.
      patience=earlyStoppingPatience,  # Number of epochs with no improvement before stopping.
      restore_best_weights=True,  # Restore best weights after stopping.
      verbose=verbose,  # Verbose output when stopping early.
    ),
    # Reduce LR on plateau.
    ReduceLROnPlateau(
      monitor=monitor,  # Monitor validation loss.
      factor=0.5,  # Reduce LR by this factor.
      patience=5,  # Number of epochs with no improvement before reducing LR.
      min_lr=1e-6,  # Minimum learning rate.
      verbose=verbose,  # Verbose output when reducing LR.
    )
  ]

  # Reset the generators before training.
  trainGenNew.reset()
  validGenNew.reset()
  testGenNew.reset()

  # Create the model using a backbone and an attention module.
  model, history, historyFine, configs = CreateFitPretrainedAttentionModel(
    trainGenNew=trainGenNew,
    validGenNew=validGenNew,
    baseModelString=baseModelString,
    attentionBlockStr=attentionBlockStr,
    inputShape=imgShape,
    numClasses=numClasses,
    callbacks=callbacks,
    initialEpochs=initialEpochs,
    fineTuneEpochs=fineTuneEpochs,
    modelCheckpointPath=modelCheckpointPath,
    verbose=verbose,
    storageDir=storageDir,
  )

  configs.update({
    "trainSamples"         : int(len(trainGenNew.filenames)),
    "validSamples"         : int(len(validGenNew.filenames)),
    "testSamples"          : int(len(testGenNew.filenames)),
    "trainBatchSize"       : int(batchSize),
    "trainStepsPerEpoch"   : int(len(trainGenNew) // batchSize),
    "validStepsPerEpoch"   : int(len(validGenNew) // batchSize),
    "testSteps"            : int(len(testGenNew) // batchSize),
    "augmentationConfigs"  : augmentationConfigs,
    "monitor"              : str(monitor),
    "earlyStoppingPatience": int(earlyStoppingPatience),
    "ensureCUDA"           : bool(ensureCUDA),
    "storageDir"           : str(storageDir),
    "dpi"                  : int(dpi),
    "modelCheckpointPath"  : str(modelCheckpointPath),
    "initialEpochs"        : int(initialEpochs),
    "fineTuneEpochs"       : int(fineTuneEpochs),
    "imgSize"              : imgSize,
    "imgShape"             : imgShape,
    "baseModelString"      : str(baseModelString),
    "attentionBlockStr"    : str(attentionBlockStr),
    "numClasses"           : int(numClasses),
    "batchSize"            : int(batchSize),
  })

  if (labelEncoder is not None):
    configs["labelEncoderClasses"] = labelEncoder.classes_.tolist()
    configs["labelEncoderClassMapping"] = dict(
      zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))

  # Evaluate on test set.
  # Evaluate model on test set.
  testLoss, testAccuracy = model.evaluate(testGenNew, verbose=verbose)
  # Print test metrics.
  print(f"Test Accuracy: {testAccuracy:.4f}, Test Loss: {testLoss:.4f}")

  # Compute predictions for confusion matrix and classification report.
  yTrue = []  # Initialize list for true labels.
  yPred = []  # Initialize list for predicted labels.
  testGenNew.reset()  # Reset generator to start.

  for i in range(len(testGenNew)):
    # Get next batch from generator.
    images, labelsBatch = next(testGenNew)
    # Predict on batch.
    preds = model.predict(images, verbose=verbose)
    # Extend true labels.
    yTrue.extend(labelsBatch)
    # Extend predictions.
    yPred.extend(np.argmax(preds, axis=1))

  # Convert true labels to numpy array.
  yTrue = np.array(yTrue)
  # Convert predicted labels to numpy array.
  yPred = np.array(yPred)

  # Get class label names.
  if (labelEncoder is not None):
    classLabels = labelEncoder.classes_.tolist()
  else:
    classLabels = list(trainGenNew.class_indices.keys())

  # Compute confusion matrix.
  cm = confusion_matrix(yTrue, yPred)
  fileName = os.path.join(storageDir, "ConfusionMatrix.pdf")
  PlotConfusionMatrix(
    cm,  # Confusion matrix (2D list or numpy array).
    classLabels,  # List of class labels.
    normalize=False,  # Whether to normalize the confusion matrix.
    roundDigits=3,  # Number of decimal places to round normalized values.
    title="Confusion Matrix",  # Title of the plot.
    cmap="random",  # Colormap for the heatmap.
    display=False,  # Whether to display the plot.
    save=True,  # Whether to save the plot.
    fileName=fileName,  # File name to save the plot.
    fontSize=15,  # Font size for labels and annotations.
    annotate=True,  # Whether to annotate cells with values.
    figSize=(8, 8),  # Figure size in inches.
    colorbar=True,  # Whether to show colorbar.
    returnFig=False,  # Whether to return the figure object.
    dpi=dpi,  # DPI for saving the figure.
  )

  # Print header for classification report.
  print("\nClassification Report:")
  # Print classification metrics.
  print(classification_report(yTrue, yPred, target_names=classLabels))

  # Calculate detailed performance metrics.
  pm = CalculatePerformanceMetrics(
    confMatrix=cm,
    eps=1e-10,  # Small value to avoid division by zero.
    addWeightedAverage=True,  # Whether to include weighted averages in the output.
    addPerClass=True,  # Whether to include per-class metrics in the output.
  )
  # Convert the performance metrics (stored as a dictionary) to a DataFrame.
  pmDf = pd.DataFrame(pm)
  pmFilePath = os.path.join(storageDir, "PerformanceMetrics.csv")
  # Save the performance metrics to a CSV file.
  pmDf.to_csv(pmFilePath, index=False)
  # Print the performance metrics dictionary.
  for key, value in pm.items():
    print(f"{key}: {value}")

  configs.update({
    "testAccuracy"          : float(testAccuracy),
    "testLoss"              : float(testLoss),
    "performanceMetrics"    : pm,
    "performanceMetricsFile": str(pmFilePath),
  })

  # Combine training accuracy across phases.
  trainAccuracy = history.history["accuracy"] + historyFine.history["accuracy"]
  # Combine validation accuracy.
  valAccuracy = history.history["val_accuracy"] + historyFine.history["val_accuracy"]
  # Combine training loss.
  trainLoss = history.history["loss"] + historyFine.history["loss"]
  # Combine validation loss.
  valLoss = history.history["val_loss"] + historyFine.history["val_loss"]

  # Store the history as CSV file.
  history = {
    "train_accuracy": trainAccuracy,
    "val_accuracy"  : valAccuracy,
    "train_loss"    : trainLoss,
    "val_loss"      : valLoss,
  }
  historyDf = pd.DataFrame(history)
  historyFilePath = os.path.join(storageDir, "TrainingHistory.csv")
  historyDf.to_csv(historyFilePath, index=False)  # Save history to CSV.

  title = f"Training History: {baseModelString} + {attentionBlockStr}"
  savePath = os.path.join(storageDir, "TrainingHistory.pdf")
  HistoryPlotter(
    history,  # Dictionary containing training history.
    title,  # Title of the plot.
    metrics=("loss", "accuracy"),  # Tuple or list of metrics to plot.
    xLabel="Epochs",  # Label for x-axis.
    fontSize=15,  # Font size for labels and title.
    save=True,  # Whether to save the plot.
    savePath=savePath,  # Path to save the plot.
    dpi=dpi,  # DPI for saving the figure.
    colors=None,  # Optional dict of colors for each metric.
    labels=None,  # Optional dict of labels for each metric.
    display=False,  # Whether to display the plot.
    figSize=(10, 5),  # Figure size.
    returnFig=False,  # Whether to return the figure object.
    smooth=True,  # Whether to apply smoothing to the curves.
    smoothFactor=0.6,  # Smoothing factor for curves (0 to 1).
  )

  # Store the last model.
  lastModelPath = os.path.join(storageDir, "LastModel.keras")
  model.save(lastModelPath)  # Save the last model after training.

  configs.update({
    "testGenSize"        : int(len(testGenNew.filenames)),
    "trainGenSize"       : int(len(trainGenNew.filenames)),
    "validGenSize"       : int(len(validGenNew.filenames)),
    "historyFilePath"    : str(historyFilePath),
    "trainingHistoryPlot": str(savePath),
    "lastModelPath"      : str(lastModelPath),
  })

  # Store the final results and configurations in a JSON file.
  resultsFilePath = os.path.join(storageDir, "FinalResults.json")
  WritePickleFile(resultsFilePath, configs)


def EvaluatePretrainedAttentionModelFromDataFrame(
  dataFrame,
  modelPath,
  columnsMap={"imagePath": "image_path", "categoryEncoded": "category_encoded", "split": "split"},
  labelEncoder=None,
  imgShape=(512, 512, 3),
  batchSize=32,
  storageDir="History",
  dpi=720,
  verbose=1,
  ensureCUDA=True,
  T=50  # Number of stochastic forward passes for MC-dropout.
):
  r'''
  Evaluate a trained model on the test set and save results.

  Parameters:
    dataFrame (pandas.DataFrame): DataFrame containing image paths and labels.
    modelPath (str): Path to the saved Keras model (.keras file).
    columnsMap (dict): Mapping of required column names in the DataFrame.
    labelEncoder (sklearn.preprocessing.LabelEncoder or None): Optional label encoder to decode class labels; if None, class labels will be taken from the generator's class indices.
    imgShape (tuple): Input shape for the model (height, width, channels).
    batchSize (int): Batch size for evaluation.
    storageDir (str): Directory where evaluation results will be saved.
    dpi (int): Dots per inch for saving figures.
    verbose (int): Verbosity level for evaluation (0 = silent, 1 = progress bar, 2 = one line per epoch).
    ensureCUDA (bool): Whether to check for CUDA availability and raise an error if not found.
    T (int): Number of stochastic forward passes for MC-dropout uncertainty estimation (if applicable).
  '''

  # Verify that the dataFrame contains the required columns.
  requiredColumns = set(columnsMap.values())
  if (not requiredColumns.issubset(dataFrame.columns)):
    raise ValueError(f"DataFrame must contain columns: {requiredColumns}")

  if (not os.path.isfile(modelPath) or not os.path.exists(modelPath)):
    raise ValueError(f"Model file not found at path: {modelPath}")

  import numpy as np
  import pandas as pd
  from sklearn.metrics import confusion_matrix, classification_report
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from HMB.Initializations import (
    DoRandomSeeding, EnsureCUDAAvailable, ClearTensorFlowSession,
    UpdateMatplotlibSettings
  )
  from HMB.PerformanceMetrics import (
    PlotConfusionMatrix, CalculatePerformanceMetrics,
    PlotROCAUCCurve, PlotPRCCurve, PlotClasswisePRFBar,
    ComputeECEPlotReliability, RiskCoverageCurve,
    PlotErrorAnalysis, PlotErrorMatrix, PlotPredictionConfidenceHistogram,
    PlotClassificationResiduals, ComputeMonteCarloUncertaintyMeasures,
    SampleMonteCarloDirichletFromProbs, PlotTopKAccuracyCurve,
    PlotCalibrationCurve,
  )

  ClearTensorFlowSession()
  DoRandomSeeding()
  UpdateMatplotlibSettings()

  if (ensureCUDA):
    EnsureCUDAAvailable("tensorflow")

  customObjects = {
    "CBAMBlock": None,
    "BAMBlock" : None,
    "SEBlock"  : None,
    "ECABlock" : None,
    "GCBlock"  : None,
  }
  if ("CBAM" in modelPath):
    from HMB.TFAttentionBlocks import CBAMBlock
    customObjects["CBAMBlock"] = CBAMBlock
  elif ("BAM" in modelPath):
    from HMB.TFAttentionBlocks import BAMBlock
    customObjects["BAMBlock"] = BAMBlock
  elif ("SE" in modelPath):
    from HMB.TFAttentionBlocks import SEBlock
    customObjects["SEBlock"] = SEBlock
  elif ("ECA" in modelPath):
    from HMB.TFAttentionBlocks import ECABlock
    customObjects["ECABlock"] = ECABlock
  elif ("GC" in modelPath):
    from HMB.TFAttentionBlocks import GCBlock
    customObjects["GCBlock"] = GCBlock
  else:
    raise ValueError(
      "Could not determine attention block from model path. "
      "Ensure the model file name contains one of the attention block identifiers (CBAM, BAM, SE, ECA, GC)."
    )

  # Load the trained model.
  model = tf.keras.models.load_model(modelPath, custom_objects=customObjects)

  for subsetKey in ["train", "val", "test"]:
    keyword = subsetKey.capitalize()  # Capitalize the subset key for display and file naming.
    print(f"\nEvaluating on {keyword} set:")
    subsetStorageDir = os.path.join(storageDir, keyword)
    os.makedirs(subsetStorageDir, exist_ok=True)  # Create directory for this subset's results.

    # Define generator for evaluation without augmentation.
    generator = ImageDataGenerator(rescale=1.0 / 255)
    # Create data generator for the test set.
    splitCol = columnsMap["split"]
    dfNew = dataFrame[dataFrame[splitCol] == subsetKey].copy()
    xCol = columnsMap["imagePath"]
    yCol = columnsMap["categoryEncoded"]
    genObj = generator.flow_from_dataframe(
      dfNew,
      x_col=xCol,
      y_col=yCol,
      target_size=imgShape[:2],
      class_mode="sparse",
      color_mode="rgb",
      shuffle=False,
      batch_size=batchSize,
    )

    # Evaluate on test set.
    testLoss, testAccuracy = model.evaluate(genObj, verbose=verbose)
    print(f"{keyword} Accuracy: {testAccuracy:.4f}, {keyword} Loss: {testLoss:.4f}")

    # Compute predictions for confusion matrix and classification report.
    genObj.reset()
    yTrue = []
    yPred = []
    yPredProb = []
    yPredConfidences = []
    for i in range(len(genObj)):
      images, labelsBatch = next(genObj)
      preds = model.predict(images, verbose=verbose)
      yTrue.extend(labelsBatch)
      yPred.extend(np.argmax(preds, axis=1))
      yPredProb.extend(preds)
      yPredConfidences.extend(np.max(preds, axis=1))

    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    yPredProb = np.array(yPredProb)
    yPredConfidences = np.array(yPredConfidences)

    if (labelEncoder is not None):
      yTrueLabels = labelEncoder.inverse_transform(yTrue.astype(int))
      yPredLabels = labelEncoder.inverse_transform(yPred.astype(int))
    else:
      yTrueLabels = yTrue
      yPredLabels = yPred

    # Store evaluation results as DataFrame and save to CSV.
    evalResults = {
      "yTrue"           : yTrue.tolist(),
      "yPred"           : yPred.tolist(),
      "yPredProb"       : yPredProb.tolist(),
      "yPredConfidences": yPredConfidences.tolist(),
      "yTrueLabels"     : yTrueLabels.tolist(),
      "yPredLabels"     : yPredLabels.tolist(),
    }
    evalResultsDf = pd.DataFrame(evalResults)
    evalResultsFilePath = os.path.join(subsetStorageDir, f"{keyword}Results.csv")
    evalResultsDf.to_csv(evalResultsFilePath, index=False)

    # Get class label names from the generator's class indices.
    if (labelEncoder is not None):
      classLabels = labelEncoder.classes_.tolist()
    else:
      classLabels = list(genObj.class_indices.keys())
    classLabels = [str(label) for label in classLabels]  # Ensure labels are strings.

    cm = confusion_matrix(yTrue, yPred)
    cmPath = os.path.join(subsetStorageDir, f"{keyword}ConfusionMatrix.pdf")
    PlotConfusionMatrix(
      cm,  # Confusion matrix (2D list or numpy array).
      classLabels,  # List of class labels.
      normalize=False,  # Whether to normalize the confusion matrix.
      roundDigits=3,  # Number of decimal places to round normalized values.
      title="Confusion Matrix",  # Title of the plot.
      cmap="random",  # Colormap for the heatmap.
      display=False,  # Whether to display the plot.
      save=True,  # Whether to save the plot.
      fileName=cmPath,  # File name to save the plot.
      fontSize=15,  # Font size for labels and annotations.
      annotate=True,  # Whether to annotate cells with values.
      figSize=(8, 8),  # Figure size in inches.
      colorbar=True,  # Whether to show colorbar.
      returnFig=False,  # Whether to return the figure object.
      dpi=dpi,  # DPI for saving the figure.
    )
    print("\nClassification Report:")
    print(classification_report(yTrue, yPred, target_names=classLabels))

    pm = CalculatePerformanceMetrics(
      confMatrix=cm,
      eps=1e-10,  # Small value to avoid division by zero.
      addWeightedAverage=True,  # Whether to include weighted averages in the output.
      addPerClass=True,  # Whether to include per-class metrics in the output.
    )
    pmDf = pd.DataFrame(pm)
    pmFilePath = os.path.join(subsetStorageDir, f"{keyword}PerformanceMetrics.csv")
    pmDf.to_csv(pmFilePath, index=False)
    for key, value in pm.items():
      print(f"{key}: {value}")

    rocPath = os.path.join(subsetStorageDir, f"{keyword}ROCCurve.pdf")
    PlotROCAUCCurve(
      yTrue,  # True labels (one-hot or binary).
      yPredProb,  # Predicted labels (one-hot or binary).
      classLabels,  # List of class names.
      areProbabilities=True,  # Whether yPred are probabilities.
      title="ROC Curve & AUC",  # Plot title.
      figSize=(5, 5),  # Figure size.
      cmap="random",  # Colormap for ROC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=rocPath,  # File name.
      fontSize=16,  # Font size.
      plotDiagonal=True,  # Plot diagonal reference line.
      annotateAUC=True,  # Annotate AUC value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    prcPath = os.path.join(subsetStorageDir, f"{keyword}PRCCurve.pdf")
    PlotPRCCurve(
      yTrue,  # True labels (one-hot or binary).
      yPredProb,  # Predicted labels (one-hot or binary).
      classLabels,  # List of class names.
      areProbabilities=True,  # Whether yPred are probabilities.
      title="PRC Curve",  # Plot title.
      figSize=(5, 5),  # Figure size.
      cmap="random",  # Colormap for PRC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=prcPath,  # File name.
      fontSize=16,  # Font size.
      annotateAvg=True,  # Annotate average precision value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    clsWisePRFPath = os.path.join(subsetStorageDir, f"{keyword}ClasswisePRFBarPlot.pdf")
    PlotClasswisePRFBar(
      cm,
      classNames=classLabels,
      fontSize=15,
      figsize=(8, 5),
      display=False,
      save=True,
      fileName=clsWisePRFPath,
      dpi=dpi,
      returnFig=False,
    )

    # Check presence of dropout layers.
    hasDropout = any([isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers])
    if (not hasDropout):
      print("[WARN] Model contains no Dropout layers; MC-dropout will not produce stochasticity.\n")
    else:
      print("[INFO] Model contains Dropout layers; proceeding with MC-dropout inference.\n")

    # Run MC-dropout predictions.
    steps = int(np.ceil(genObj.samples / genObj.batch_size))
    print(f"[MC] Running T={T} stochastic forward passes (steps={steps})...")
    mcPreds = MCDropoutPredictions(
      model=model,
      genObj=genObj,
      steps=steps,
      T=T,  # Number of stochastic forward passes.
    )
    print(f"[MC] Completed: Shape = {mcPreds.shape}.\n")

    # Compute uncertainty statistics.
    print("[STATS] Computing uncertainty statistics...")
    stats = ComputeUncertaintyStats(mcPreds)
    for key, value in stats.items():
      print(f"{key}: {value}")
    meanProbs = stats["meanProbs"]
    stdProbs = stats["stdProbs"]
    predLabels = stats["predLabels"]
    print("[STATS] Completed uncertainty statistics computation.\n")

    # Build export DataFrame aligned with the DataFrame order.
    exportDf = dfNew.reset_index(drop=True).copy()
    exportDf["predEncoded"] = predLabels
    exportDf["predLabel"] = labelEncoder.inverse_transform(predLabels)
    exportDf["predConfidenceMean"] = stats["meanConfidence"]
    exportDf["predConfidenceStd"] = stats["stdConfidence"]
    exportDf["predictiveEntropy"] = stats["predictiveEntropy"]
    exportDf["expectedEntropy"] = stats["expectedEntropy"]
    exportDf["mutualInfo"] = stats["mutualInfo"]

    # Optionally store a few summary columns for class probabilities (first K classes).
    noClasses = meanProbs.shape[1]
    K = min(5, noClasses)
    for k in range(K):
      exportDf[f"class{k}_mean_prob"] = meanProbs[:, k]
      exportDf[f"class{k}_std_prob"] = stdProbs[:, k]

    exportFilePath = os.path.join(subsetStorageDir, f"{keyword}MCUncertaintyResults.csv")
    exportDf.to_csv(exportFilePath, index=False)
    print(f"[EXPORT] Exported MC-dropout uncertainty results to: {exportFilePath}\n")

    # Compute and save reliability diagram (ECE) using encoded predicted labels.
    print("[EVAL] Computing Expected Calibration Error (ECE) and plotting reliability diagram...")
    eceOutPath = os.path.join(subsetStorageDir, f"{keyword}ReliabilityDiagram.pdf")
    ece, binAcc, binConf, binCounts = ComputeECEPlotReliability(
      exportDf["predConfidenceMean"].values,
      exportDf["predEncoded"].values,
      exportDf["category_encoded"].astype(int).values,
      nBins=15,
      title="Expected Calibration Error (ECE) - MC Dropout",
      fontSize=15,
      figSize=(6, 6),
      display=False,
      save=True,
      fileName=eceOutPath,
      dpi=dpi,
      returnFig=False,
    )
    print(f"[EVAL] ECE = {ece:.4f}")
    print(f"[EVAL] Saved reliability diagram to {eceOutPath}.\n")
    # Store the ECE and bin details in a CSV for further analysis.
    eceDetailsDf = pd.DataFrame({
      "binUpperBound": binConf,
      "binAccuracy"  : binAcc,
      "binConfidence": binConf,
      "binCount"     : binCounts,
    })
    eceDetailsPath = os.path.join(subsetStorageDir, f"{keyword}ECEDetailsInitial.csv")
    eceDetailsDf.to_csv(eceDetailsPath, index=False)
    print(f"[EVAL] Saved ECE bin details to {eceDetailsPath}.\n")

    # Save top-N uncertain images.
    print("[EXPORT] Saving top-N uncertain images grid...")
    storePath = os.path.join(subsetStorageDir, f"{keyword}TopUncertainImages.pdf")
    SaveTopUncertainImages(
      dfNew,
      meanProbs,
      exportDf["mutualInfo"],
      storePath,
      labelEncoder,
      topN=16,
      imgSize=(128, 128),
    )
    print(f"[EXPORT] Saved top-N uncertain images to {storePath}.\n")

    # Aggregate and save class-wise uncertainty metrics.
    print("[EXPORT] Computing and saving class-wise uncertainty metrics...")
    classwiseStats = AggregateClasswiseUncertainty(exportDf, meanProbs, exportDf["mutualInfo"])
    classwiseStatsPath = os.path.join(subsetStorageDir, f"{keyword}ClasswiseUncertaintyMetrics.csv")
    classwiseStats.to_csv(classwiseStatsPath, index=False)
    print(f"[EXPORT] Saved class-wise uncertainty metrics to {classwiseStatsPath}.\n")

    # Compute energy score from pseudo-logits (log probs) as an interpretability metric.
    print("[EVAL] Computing energy scores for interpretability...")
    energyScores = ComputeEnergyScore(meanProbs)
    exportDf["energyScore"] = energyScores
    energyScoresPath = os.path.join(subsetStorageDir, f"{keyword}EnergyScores.csv")
    exportDf[["image_path", "category_encoded", "predEncoded", "energyScore"]].to_csv(energyScoresPath, index=False)
    print(f"[EVAL] Computed energy scores and saved to {energyScoresPath}.\n")

    # Save updated exportDf (with energyScore maybe added).
    finalExportPath = os.path.join(subsetStorageDir, f"{keyword}FinalMCUncertaintyResults.csv")
    exportDf.to_csv(finalExportPath, index=False)
    print(f"[EXPORT] Saved final MC uncertainty results with energy scores to {finalExportPath}.\n")

    # Risk-Coverage curve.
    confidences = exportDf["predConfidenceMean"].values
    correctness = (exportDf["predEncoded"].values == exportDf["category_encoded"].astype(int).values)
    riskCovPath = os.path.join(subsetStorageDir, f"{keyword}RiskCoverageCurve.pdf")
    coverage, accuracy, aucVal = RiskCoverageCurve(
      confidences,
      correctness,
      title="Risk-Coverage (Accuracy vs Coverage)",
      fontSize=15,
      figSize=(6, 6),
      display=False,
      save=True,
      fileName=riskCovPath,
      dpi=dpi,
      returnFig=False,
    )
    print(f"[INTERP] Saved risk-coverage plot to {riskCovPath} (AUC={aucVal:.3f}).\n")

    errorAnalysisPath = os.path.join(subsetStorageDir, f"{keyword}ErrorAnalysis.pdf")
    PlotErrorAnalysis(
      yTrue.astype(int),
      yPred.astype(int),
      X=None,
      classNames=classLabels,
      maxExamples=5,
      fontSize=15,
      figsize=(12, 10),
      display=False,
      save=True,
      fileName=errorAnalysisPath,
      dpi=dpi,
      returnFig=False,
    )
    print(f"[EVAL] Saved error analysis plot to {errorAnalysisPath}.\n")

    errorMatrixPath = os.path.join(subsetStorageDir, f"{keyword}ErrorMatrix.pdf")
    PlotErrorMatrix(
      cm,
      classNames=classLabels,
      fontSize=15,
      figsize=(7, 6),
      display=False,
      save=True,
      fileName=errorMatrixPath,
      dpi=dpi,
      returnFig=False,
    )
    print(f"[EVAL] Saved error matrix plot to {errorMatrixPath}.\n")

    predConfHistPath = os.path.join(subsetStorageDir, f"{keyword}PredictionConfidenceHistogram.pdf")
    PlotPredictionConfidenceHistogram(
      yPredProb,
      yPred=yPred,
      fontSize=15,
      figsize=(8, 5),
      bins=20,
      display=False,
      save=True,
      fileName=predConfHistPath,
      dpi=dpi,
      returnFig=False,
    )
    print(f"[EVAL] Saved prediction confidence histogram to {predConfHistPath}.\n")

    clsResidualsPath = os.path.join(subsetStorageDir, f"{keyword}ClassificationResiduals.pdf")
    PlotClassificationResiduals(
      yTrue,
      yPred,
      fontSize=15,
      figsize=(8, 5),
      display=False,
      save=True,
      fileName=clsResidualsPath,
      dpi=dpi,
      returnFig=False,
    )
    print(f"[EVAL] Saved classification residuals plot to {clsResidualsPath}.\n")

    probsMC = SampleMonteCarloDirichletFromProbs(yPredProb, T=T, concentration=30.0)
    uncertaintyMeasures = ComputeMonteCarloUncertaintyMeasures(probsMC)
    confidences = uncertaintyMeasures["predictedConfidence"]
    predictions = uncertaintyMeasures["predictedIdx"]
    ecePath = os.path.join(subsetStorageDir, f"{keyword}ECEReliabilityPlot.pdf")
    ece, binAcc, binConf, binCounts = ComputeECEPlotReliability(
      confidences,
      predictions,
      labels=yPred,
      nBins=5,
      title="ECE Example",
      fontSize=15,
      figSize=(6, 6),
      display=False,
      save=True,
      fileName=ecePath,
      dpi=dpi,
      returnFig=False,
      cmap="random",
      applyXYLimits=True,
    )
    print(f"ECE: {ece}")
    print(f"Bin Accuracies: {binAcc}")
    print(f"Bin Confidences: {binConf}")
    print(f"Bin Counts: {binCounts}")
    # Store ECE details in a CSV for further analysis.
    eceDetailsDf = pd.DataFrame({
      "binUpperBound": binConf,
      "binAccuracy"  : binAcc,
      "binConfidence": binConf,
      "binCount"     : binCounts,
    })
    eceDetailsPath = os.path.join(subsetStorageDir, f"{keyword}ECEDetails.csv")
    eceDetailsDf.to_csv(eceDetailsPath, index=False)
    print(f"Saved ECE bin details to {eceDetailsPath}.\n")

    riskCoveragePath = os.path.join(subsetStorageDir, f"{keyword}RiskCoverageCurve.pdf")
    correctness = (predictions == yPred).astype(int)
    RiskCoverageCurve(
      confidences,
      correctness,
      title="Risk-Coverage (Accuracy vs Coverage)",
      fontSize=15,
      figSize=(6, 6),
      display=False,
      save=True,
      fileName=riskCoveragePath,
      dpi=dpi,
      returnFig=False,
      color="blue",
    )
    print(f"Saved risk-coverage curve to {riskCoveragePath}.\n")

    topKAccPath = os.path.join(subsetStorageDir, f"{keyword}TopKAccuracyCurve.pdf")
    PlotTopKAccuracyCurve(
      yPredProb,
      yPred,
      maxK=10,
      title="Top-k Accuracy Curve",
      figSize=(6, 6),
      save=True,
      fileName=topKAccPath,
      display=False,
      fontSize=15,
      returnFig=False,
      dpi=dpi,
      color="blue",
    )
    print(f"Saved top-k accuracy curve to {topKAccPath}.\n")

    calibCurvePath = os.path.join(subsetStorageDir, f"{keyword}CalibrationCurve.pdf")
    PlotCalibrationCurve(
      yPredProb,
      yPred,
      nBins=10,
      title="Calibration Curve",
      fontSize=15,
      figSize=(6, 6),
      display=False,
      save=True,
      fileName=calibCurvePath,
      dpi=dpi,
      returnFig=False,
      color="blue",
    )
    print(f"Saved calibration curve to {calibCurvePath}.\n")


def StatisticsPretrainedAttentionModelFromDataFrame(
  trialResultsPath,
  statisticsStoragePath,
  dpi=720,
  plotMetricsIndividual=False,
  plotMetricsOverall=False,
  includeAverageInPlots=False,
  whichSubset="test",
):
  r'''
  Analyze the results from multiple trials of inference using a saved preprocessing + model objects bundle
  with the best parameters from a previous tuning run.

  Parameters:
    trialResultsPath (str): Path to the directory containing the results of multiple trials.
    statisticsStoragePath (str): Directory where the aggregated performance plots and statistics will be saved.
    dpi (int, optional): DPI for saving the performance plots. Default is 720.
    plotMetricsIndividual (bool, optional): Whether to plot performance metrics for individual trials. Default is False.
    plotMetricsOverall (bool, optional): Whether to plot aggregated performance metrics across all trials. Default is False.
    includeAverageInPlots (bool, optional): Whether to include the average performance across trials in the plots. Default is False.
    whichSubset (str, optional): Which data subset's results to analyze (e.g., "test", "val"). Default is "test".

  Returns:
    dict: A dictionary containing the aggregated performance metrics and results from all trials.
  '''

  from sklearn.metrics import confusion_matrix
  from HMB.PerformanceMetrics import CalculatePerformanceMetrics, PlotMultiTrialROCAUC, PlotMultiTrialPRCurve
  from HMB.StatisticalAnalysisHelper import ExtractDataFromSummaryFile, PlotMetrics, StatisticalAnalysis

  storageFolderName = os.path.basename(statisticsStoragePath)
  print(f"\n\u2728 Starting statistics analysis for trial results in: {trialResultsPath}")
  print(f"\u2728 Aggregated performance plots and statistics will be saved in: {statisticsStoragePath}\n")

  # Check first if the folder contains multiple trial result folders with the expected structure or just a single
  # trial result folder.
  isFound = len([
    el for el in os.listdir(trialResultsPath)
    if (os.path.isdir(os.path.join(trialResultsPath, el)) and el != storageFolderName)
  ]) > 0
  filesToProcess = {}
  if (isFound):
    # Inform the user that trial result files were found in the directory.
    print("\u2713 Found multiple models trial results file in:", trialResultsPath)
    # Find subdirectories (per-model) inside the trial results' path.
    foundModels = [
      el
      for el in os.listdir(trialResultsPath)
      if (os.path.isdir(os.path.join(trialResultsPath, el)) and el != storageFolderName)
    ]
    # Print which models were discovered with trial results.
    print(f"\u2713 Found the following models with trial results: {foundModels}")
    for record in foundModels:
      recordPath = os.path.join(trialResultsPath, record)
      if (os.path.isdir(recordPath)):
        trialFiles = [
          el
          for el in os.listdir(recordPath)
          if (el.startswith("Trial_") and os.path.isdir(os.path.join(recordPath, el)))
        ]
        if (len(trialFiles) == 0):
          print(
            f"\u26A0 No trial result folders found for model '{record}' (skipping). "
            f"Expected subdirectories named 'Trial_X'."
          )
        else:
          filesToProcess[record] = trialFiles
      else:
        print(f"\u26A0 Found unexpected file in trial results directory (skipping): {recordPath}")
  else:
    trialFiles = [
      el
      for el in os.listdir(trialResultsPath)
      if (el.startswith("Trial_") and os.path.isdir(os.path.join(trialResultsPath, el)))
    ]
    if (len(trialFiles) == 0):
      raise ValueError(
        f"No trial result folders found in the specified directory: {trialResultsPath}. "
        f"Ensure that the directory contains subdirectories named 'Trial_X' for each trial."
      )
    filesToProcess["Current"] = trialFiles

  if (len(filesToProcess) == 0):
    raise ValueError(
      f"No trial result folders found in the specified directory: {trialResultsPath}. "
      f"Ensure that the directory contains subdirectories named 'Trial_X' for each trial."
    )
  print(f"\u2713 Found the following trial files to process: {list(filesToProcess.keys())}")
  # Print the trial files that will be processed for each model.
  for modelKey, trialFiles in filesToProcess.items():
    print(f"\u2713 For model '{modelKey}', found trial files: {trialFiles}")

  allHistory = {}
  for k, trialFiles in filesToProcess.items():
    print(f"\nProcessing trial results for: {k}")
    if (k == "Current"):
      newStorageDir = statisticsStoragePath
    else:
      newStorageDir = os.path.join(statisticsStoragePath, k)
    os.makedirs(newStorageDir, exist_ok=True)

    allYTrue = []
    allYPred = []
    classes = []
    kHistory = {}

    for trialFile in trialFiles:
      if (k == "Current"):
        csvFolder = os.path.join(trialResultsPath, trialFile)
      else:
        csvFolder = os.path.join(trialResultsPath, k, trialFile)

      if (not os.path.isdir(csvFolder) or not os.path.exists(csvFolder)):
        raise FileNotFoundError(
          f"Expected directory not found: {csvFolder}. Please ensure that the trial results are "
          f"generated correctly and the file structure is as expected."
        )

      # Search for the `whichSubset` folder.
      testFolder = None
      for el in os.listdir(csvFolder):
        if (os.path.isdir(os.path.join(csvFolder, el)) and el.lower().startswith(whichSubset.lower())):
          testFolder = os.path.join(csvFolder, el)
          break
      if (testFolder is None):
        raise FileNotFoundError(
          f"Expected the {whichSubset} folder not found in: {csvFolder}. "
          f"Please ensure that the trial results are "
          f"generated correctly and the file structure is as expected."
        )
      # Search for the evaluation results CSV file inside the folder.
      # It can be "{whichSubset}Results.csv" or "EvaluationResults.csv" (case-insensitive).
      csvPath = None
      for el in os.listdir(testFolder):
        if (
          os.path.isfile(os.path.join(testFolder, el)) and
          el.lower() in [f"{whichSubset.lower()}results.csv", "evaluationresults.csv"]
        ):
          csvPath = os.path.join(testFolder, el)
          break

      if (not os.path.isfile(csvPath)):
        raise FileNotFoundError(
          f"Expected file not found: {csvPath}. Please ensure that the trial results are "
          f"generated correctly and the file structure is as expected."
        )

      # yTrue	yPred	yPredProb	yPredConfidences	yTrueLabels	yPredLabels.
      df = pd.read_csv(csvPath)
      yTrue = df["yTrue"].values
      yPred = df["yPred"].values
      try:
        yPredProb = df["yPredProb"].values
        # Parse the yPredProb column from string representation of list to actual list of floats.
        yPredProb = np.array([np.fromstring(probStr.strip("[]"), sep=",") for probStr in yPredProb])
      except:
        yPredProb = None

      yPredConfidences = df["yPredConfidences"].values
      yTrueLabels = df["yTrueLabels"].values
      yPredLabels = df["yPredLabels"].values
      classes = np.unique(yTrueLabels)
      cm = confusion_matrix(yTrue, yPred)
      metrics = CalculatePerformanceMetrics(cm, addWeightedAverage=True, eps=1e-8)

      if (not includeAverageInPlots):
        # Remove the columns that contain the word "average".
        metrics = {key: value for key, value in metrics.items() if ("average" not in key.lower())}

      kHistory[trialFile] = {
        "yTrue"           : yTrue,
        "yPred"           : yPred,
        "yPredProb"       : yPredProb,
        "yPredConfidences": yPredConfidences,
        "yTrueLabels"     : yTrueLabels,
        "yPredLabels"     : yPredLabels,
        "confusionMatrix" : cm,
        "metrics"         : metrics,
      }
      allYTrue = yTrue
      allYPred.append(np.array(yPredProb) if (yPredProb is not None) else np.array(yPred))

    print(f"Total samples across all trials: {len(allYTrue)}")
    print(f"Unique classes: {classes}")

    for which in ["CI", "SD"]:
      fileName = os.path.join(newStorageDir, f"{which}_MultiTrial_PRC_Curve.pdf")
      PlotMultiTrialPRCurve(
        allYTrue,  # List of true labels arrays from all trials.
        allYPred,  # List of predicted probabilities from all trials.
        classes,  # List of class names.
        confidenceLevel=0.95,  # Confidence level for CI.
        which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
        title="Multi-Trial Precision-Recall Curve",
        figSize=(8, 8),  # Figure size in inches.
        cmap="random",  # Colormap for different classes.
        display=False,  # Whether to display the plot.
        save=True,  # Whether to save the plot.
        fileName=fileName,  # File name for saving.
        fontSize=15,  # Font size for labels and annotations.
        showLegend=True,  # Whether to show legend.
        returnFig=False,  # Whether to return the matplotlib figure object.
        dpi=dpi,  # DPI for saving the figure.
        addZoomedInset=True,  # Whether to add a zoomed inset for the top-right corner of the PRC plot.
      )

      fileName = os.path.join(newStorageDir, f"{which}_MultiTrial_ROC_AUC.pdf")
      PlotMultiTrialROCAUC(
        allYTrue,  # List of true labels arrays from all trials.
        allYPred,  # List of predicted probabilities from all trials.
        classes,  # List of class names.
        confidenceLevel=0.95,  # Confidence level for CI (default 95%).
        which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
        title="Multi-Trial ROC Curve",  # Plot title.
        figSize=(8, 8),  # Figure size.
        cmap="random",  # Colormap for ROC curves.
        display=False,  # Display the plot.
        save=True,  # Save the plot.
        fileName=fileName,  # File name.
        fontSize=15,  # Font size.
        plotDiagonal=True,  # Plot diagonal reference line.
        showLegend=True,  # Show legend.
        returnFig=False,  # Return figure object.
        dpi=dpi,  # DPI for saving the figure.
        addZoomedInset=True,  # Whether to add a zoomed inset for the top-left corner.
      )

    # Save the calculated metrics for each trial to a CSV file for comparison.
    # Example of the file structure (if you have a single system):
    #     Precision, Recall, F1, Accuracy, Specificity, Average
    #     Metric, Metric, Metric, Metric, Metric, Metric
    #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133,
    #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282,
    #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406,
    #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339,
    #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813,
    firstRow = [
      " ".join(el.split(" ")[1:])  # Remove the "Weighted " prefix from the metric names for cleaner column headers.
      for el in kHistory[trialFiles[0]]["metrics"].keys()
      if ("Weighted" in el)
    ]
    secondRow = ["Metric"] * len(firstRow)
    metricValues = [
      [kHistory[trial]["metrics"][f"Weighted {metric}"] for metric in firstRow]
      for trial in trialFiles
    ]
    # Create a DataFrame to store the metrics for each trial, with the first row containing metric names and the
    # second row containing the keyword "Metric".
    dfMetrics = pd.DataFrame(
      data=metricValues,
      columns=firstRow,
    )
    # Insert the secondRow as the second row in the DataFrame at index 0 (after the header); pushes the metric values down by one row.
    dfMetrics.loc[-1] = secondRow  # Add the second row with "Metric" values.
    dfMetrics.index = dfMetrics.index + 1  # Shift the index to accommodate the new row.
    # Sort the index to maintain the correct order (header, "Metric" row, then metric values).
    dfMetrics.sort_index(inplace=True)
    # Save the DataFrame to a CSV file for comparison.
    trialMetricsComparisonFile = os.path.join(newStorageDir, "Trial_Metrics_Comparison.csv")
    dfMetrics.to_csv(trialMetricsComparisonFile, index=False)
    print(f"Trial metrics comparison saved to: {trialMetricsComparisonFile}")
    print(f"Trial Metrics Comparison:\n{dfMetrics}")

    history, names, metrics = ExtractDataFromSummaryFile(trialMetricsComparisonFile)

    newFolderName = ""
    if (plotMetricsIndividual):
      newFolderName = os.path.join(newStorageDir, "PerformanceMetricsPlots")
      os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.

      PlotMetrics(
        history, names, metrics,
        factor=5,  # Factor to multiply the default figure size.
        keyword="AllMetrics",  # Keyword to append to the filenames of the saved plots.
        dpi=dpi,  # Dots per inch (resolution) of the saved plots.
        xTicksRotation=45,  # Rotation angle for x-axis tick labels.
        whichToPlot=[],  # List of plot types to generate.
        fontSize=14,  # Font size for the plots.
        showFigures=False,  # Whether to display the plots or not.
        storeInsideNewFolder=True,  # Whether to store the plots inside a new folder.
        newFolderName=newFolderName,  # Name of the folder to store the plots.
        noOfPlotsPerRow=3,  # Number of plots per row in the subplot grid.
        cmap="random",  # Color map for the plots.
        differentColors=True,  # Whether to use different colors for different plots.
        fixedTicksColors=True,  # Whether to use fixed ticks colors for consistency across plots.
        fixedTicksColor="black",  # Color to use for fixed ticks if `fixedTicksColors` is True.
        extension=".pdf",  # File extension for saved plots.
      )

    print("\u2713 Performance plots generated.")
    print("\nGenerating statistical analysis report...")
    overallReport = []
    for metric in metrics:
      for index, data in enumerate(history):
        report = StatisticalAnalysis(
          data[metric]["Trials"],
          hypothesizedMean=data[metric]["Mean"],
          secondMetricList=None,
        )
        report["Type"] = names[index]
        report["Metric"] = metric
        overallReport.append(report)
    reportDF = pd.DataFrame(overallReport)
    reportCsvPath = os.path.join(newStorageDir, "Statistical_Analysis_Report.csv")
    reportDF.to_csv(reportCsvPath, index=False)
    print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")

    kHistory.update({
      "allYTrue"                  : allYTrue,
      "allYPred"                  : allYPred,
      "classes"                   : classes,
      "reportDF"                  : reportDF,
      "dfMetrics"                 : dfMetrics,
      "averageMetrics"            : {
        # Calculate the average of the metric values across all trials, skipping the first two rows (header and "Metric" row).
        metric: dfMetrics[metric][1:].mean()
        for metric in firstRow
      },
      "overallReport"             : overallReport,
      "trialMetricsComparisonFile": trialMetricsComparisonFile,
      "statisticsStoragePath"     : newStorageDir,
      "performancePlotsFolder"    : newFolderName,
    })

    allHistory[k] = kHistory

  noOfSystems = len(allHistory.keys())
  print(
    f"\n\u2713 Completed processing all trials for {noOfSystems} systems. "
    f"Aggregated history and results are stored in the `allHistory` dictionary."
  )
  if (noOfSystems > 1):
    print(
      f"\n\u2713 Note: Since multiple top experiments were processed, the `allHistory` dictionary contains "
      f"separate entries for each model with their respective trial results and analyses."
    )

    # Now we need to report the statistics summary across the different top experiments (if there are multiple)
    # for easier comparison.
    for k in allHistory.keys():
      # Example of the file structure (if you have multiple systems):
      #     System A, , , , , , System B, , , , ,
      #     Precision, Recall, F1, Accuracy, Specificity, Average, Precision, Recall, F1, Accuracy, Specificity, Average
      #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133, 0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133
      #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282, 0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282
      #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406, 0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406
      #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339, 0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339
      #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813, 0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813
      systems = list(allHistory.keys())
      metricsNames = allHistory[systems[0]]["dfMetrics"].columns.tolist()
      print("Systems:", systems)
      print("Metrics Names:", metricsNames)

      systemsRow = []
      for system in systems:
        systemsRow.extend([system] + [""] * (len(metricsNames) - 1))
      metricsRow = []
      for system in systems:
        metricsRow.extend(metricsNames)
      dataRows = []
      dfMetrics = allHistory[k]["dfMetrics"]
      # Start from 1 to skip the "Metric" row.
      for i in range(1, len(dfMetrics)):
        row = []
        for system in systems:
          row.extend(dfMetrics.iloc[i].values.tolist())
        dataRows.append(row)
      finalDf = pd.DataFrame(
        dataRows,
        columns=systemsRow,
      )
      # Insert the metricsRowNames as the first row in the DataFrame at index 0 (after the header); pushes the
      # metric values down by one row.
      finalDf.loc[-1] = metricsRow  # Add the first row with system names.
      finalDf.index = finalDf.index + 1  # Shift the index to accommodate the new row.

      # Sort the index to maintain the correct order (header, system names row, then metric values).
      finalDf.sort_index(inplace=True)
      print(finalDf.head())

      # Save the DataFrame to a CSV file for comparison.
      newFolderName = os.path.join(statisticsStoragePath, "Statistics Summary")
      os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.
      allSystemsMetricsComparisonFile = os.path.join(newFolderName, "All_Systems_Metrics_Comparison.csv")
      finalDf.to_csv(allSystemsMetricsComparisonFile, index=False)
      # Store as Latex table as well for easier inclusion in reports and papers.
      allSystemsMetricsComparisonLatexFile = os.path.join(newFolderName, "All_Systems_Metrics_Comparison.tex")
      with open(allSystemsMetricsComparisonLatexFile, "w") as f:
        f.write(finalDf.to_latex(index=False))
      print(f"All systems metrics comparison saved to: {allSystemsMetricsComparisonFile}")

      # Generate performance metric plots for all systems combined.
      hist, names, metrics = ExtractDataFromSummaryFile(allSystemsMetricsComparisonFile)

      if (plotMetricsOverall):
        PlotMetrics(
          hist, names, metrics,
          factor=5,  # Factor to multiply the default figure size.
          keyword="AllSystems_AllMetrics",  # Keyword to append to the filenames of the saved plots.
          dpi=dpi,  # Dots per inch (resolution) of the saved plots.
          xTicksRotation=45,  # Rotation angle for x-axis tick labels.
          whichToPlot=[],  # List of plot types to generate.
          fontSize=14,  # Font size for the plots.
          showFigures=False,  # Whether to display the plots or not.
          storeInsideNewFolder=True,  # Whether to store the plots inside a new folder.
          newFolderName=newFolderName,  # Name of the folder to store the plots.
          noOfPlotsPerRow=3,  # Number of plots per row in the subplot grid.
          cmap="random",  # Color map for the plots.
          differentColors=True,  # Whether to use different colors for different plots.
          fixedTicksColors=True,  # Whether to use fixed ticks colors for consistency across plots.
          fixedTicksColor="black",  # Color to use for fixed ticks if `fixedTicksColors` is True.
          extension=".pdf",  # File extension for saved plots.
        )

      print("\u2713 Performance plots generated.")
      print("\nGenerating statistical analysis report...")
      overallReport = []
      for metric in metrics:
        for index, data in enumerate(hist):
          report = StatisticalAnalysis(
            data[metric]["Trials"],
            hypothesizedMean=data[metric]["Mean"],
            secondMetricList=None,
          )
          report["Type"] = names[index]
          report["Metric"] = metric
          overallReport.append(report)
      reportDF = pd.DataFrame(overallReport)
      reportCsvPath = os.path.join(newFolderName, "All_Systems_Statistical_Analysis_Report.csv")
      reportDF.to_csv(reportCsvPath, index=False)
      # Save the report as Latex table as well for easier inclusion in reports and papers.
      reportLatexPath = os.path.join(newFolderName, "All_Systems_Statistical_Analysis_Report.tex")
      with open(reportLatexPath, "w") as f:
        f.write(reportDF.to_latex(index=False))
      print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")
      print(f"Generated combined performance metric plots for {k} and saved to: {newFolderName}")

  if (noOfSystems > 1):
    print(
      f"\n\u2713 Since multiple top experiments were processed, the `allHistory` dictionary contains "
      f"separate entries for each model with their respective trial results and analyses. "
      f"The statistics summary across the different top experiments has been generated and saved in the "
      f"`Statistics Summary` folder inside the `statisticsStoragePath` directory for easier comparison."
    )
    # Return the top-1 experiment history for each model in the `allHistory` dictionary for further analysis and comparison.
    topSystemDict = {}
    topValue = -1
    for k in allHistory.keys():
      averageMetrics = allHistory[k]["averageMetrics"]
      avg = averageMetrics["Average"]
      if (avg > topValue):
        topValue = avg
        topSystemDict = {
          k: allHistory[k],
        }
    allHistory = topSystemDict
  else:
    print(
      f"\n\u2713 Since there is only one top experiment, the `allHistory` dictionary "
      f"contains the history and results for that single experiment."
    )

  return allHistory
