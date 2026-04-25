import os, pickle, patchify
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.nn import depth_to_space
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.backend import clear_session


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

  from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy

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

  lossFn = SparseCategoricalCrossentropy() if (numClasses > 2) else BinaryCrossentropy()

  model.compile(
    optimizer=optimizer,
    loss=lossFn,
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


def ImageToViTPatches(image, noOfPatches, patchSize):
  r'''
  Convert an image to flattened patch embeddings suitable for Vision Transformer input.
  This function takes a preprocessed image, extracts non-overlapping patches of a specified size,
  and flattens each patch into a vector. The output is a numpy array of shape (noOfPatches, patchSize*patchSize*3)
  containing the flattened patch embeddings for each image.

  Parameters:
    image (numpy.ndarray): A preprocessed image array of shape (height, width, channels) that is compatible with the
      expected input shape of the Vision Transformer model.
    noOfPatches (int): The number of patches to extract from the image. This should match the expected number of
      patches for the Vision Transformer model architecture (e.g., 256 for ViT-Base with 16x16 patches on
      224x224 images).
    patchSize (int): The size of each square patch (e.g., 16 for 16x16 patches). The function will extract
      non-overlapping patches of this size from the image.

  Returns:
    numpy.ndarray: A numpy array of shape (noOfPatches, patchSize*patchSize*3) containing the flattened patch embeddings for each image. Each row corresponds to a patch, and the columns represent the pixel values of the patch flattened into a vector (patchSize*patchSize*3, where 3 is the number of color channels).
  '''

  # Extract patches using patchify: returns (gridH, gridW, 1, ph, pw, c).
  patches = patchify.patchify(image, (patchSize, patchSize, 3), step=patchSize)
  # Reshape to flat list: (num_patches, patchSize, patchSize, 3).
  patches = patches.reshape(-1, patchSize, patchSize, 3)
  # Truncate or pad to expected number of patches.
  if (len(patches) < noOfPatches):
    print("Warning: Number of extracted patches is less than the expected number of patches.")
    # Pad with zeros if insufficient patches (edge case for non-divisible dimensions).
    padding = np.zeros((noOfPatches - len(patches), patchSize, patchSize, 3))
    patches = np.concatenate([patches, padding], axis=0)
  else:
    patches = patches[:noOfPatches]
  # Flatten each patch to vector: (noOfPatches, patchSize*patchSize*3).
  patches = patches.reshape(-1, patchSize * patchSize * 3)
  return patches


class ViTPatchDataGeneratorFromDataFrame(Sequence):
  r'''
  Keras Sequence generator for Vision Transformer training using a pandas DataFrame.
  This generator loads images from file paths specified in a DataFrame, extracts
  non-overlapping patches, flattens them to embedding vectors, and yields batches
  compatible with the VisionTransformer model architecture.

  Parameters:
    dataFrame (pandas.DataFrame): DataFrame containing at least 'image_path' and 'label' columns.
    inputShape (tuple): Expected input shape of the images (height, width, channels).
    batchSize (int): Number of samples per batch.
    classMode (str): "categorical" for one-hot encoded labels, "sparse" for integer labels.
    noOfPatches (int): Number of patches to extract from each image (default 256).
    patchSize (int): Size of each square patch (default 16, resulting in 16x16 patches).
    embedDimension (int): Dimension of the flattened patch embeddings (default 768 for ViT-Base).
    shuffle (bool): Whether to shuffle the data at the end of each epoch (default False).

  Returns:
    A batch of data in the form (images, labels) where:
      - images: A numpy array of shape (batchSize, noOfPatches, embedDimension) containing the flattened patch embeddings for each image in the batch.
      - labels: A numpy array of shape (batchSize, numClasses) for "categorical" classMode or (batchSize, 1) for "sparse" classMode containing the corresponding labels for each image in the batch.
  '''

  def __init__(
    self, dataFrame, inputShape, batchSize, classMode="categorical",
    noOfPatches=256, patchSize=16, embedDimension=768, shuffle=False,
  ):
    # Validate required DataFrame columns.
    requiredCols = {"image_path", "label"}
    if (not requiredCols.issubset(dataFrame.columns)):
      raise ValueError(f"DataFrame must contain columns: {requiredCols}")

    self.dataFrame = dataFrame.reset_index(drop=True)
    self.inputShape = inputShape
    self.batchSize = batchSize
    self.classMode = classMode
    self.noOfPatches = noOfPatches
    self.patchSize = patchSize
    self.embedDimension = embedDimension
    self.shuffle = shuffle

    # Build class mapping for categorical encoding.
    self.classes = sorted(dataFrame["label"].unique())
    self.classIndices = {label: idx for idx, label in enumerate(self.classes)}
    self.numClasses = len(self.classes)
    self.yTrue = dataFrame["label"].values
    self.yTrueIndices = np.array([self.classIndices[label] for label in self.yTrue])

    self.numImages = len(self.dataFrame)
    self.indices = np.arange(self.numImages)

    # Warn about dropped samples if batch size doesn't divide evenly.
    remainder = self.numImages % self.batchSize
    if (remainder != 0):
      print(
        f"Warning: {remainder} samples will be dropped per epoch "
        f"(batchSize={batchSize}, total={self.numImages})"
      )
      self.yTrue = self.yTrue[:-remainder]
      self.yTrueIndices = self.yTrueIndices[:-remainder]

    if (self.shuffle):
      np.random.shuffle(self.indices)

  def __len__(self):
    '''Return number of batches per epoch.'''
    return self.numImages // self.batchSize

  def _image_to_patches(self, image):
    '''Extract and flatten non-overlapping patches from a preprocessed image.'''

    # Extract patches using patchify: returns (grid_h, grid_w, 1, ph, pw, c).
    patches = patchify.patchify(image, (self.patchSize, self.patchSize, 3), step=self.patchSize)
    # Reshape to flat list: (num_patches, patchSize, patchSize, 3)
    patches = patches.reshape(-1, self.patchSize, self.patchSize, 3)
    # Truncate or pad to expected number of patches
    if (len(patches) < self.noOfPatches):
      # Pad with zeros if insufficient patches (edge case for non-divisible dimensions).
      padding = np.zeros((self.noOfPatches - len(patches), self.patchSize, self.patchSize, 3))
      patches = np.concatenate([patches, padding], axis=0)
    else:
      patches = patches[:self.noOfPatches]
    # Flatten each patch to vector: (noOfPatches, patchSize*patchSize*3).
    return patches.reshape(self.noOfPatches, -1)

  def __getitem__(self, index):
    '''Generate one batch of data.'''
    # Select batch indices.
    batchIndices = self.indices[index * self.batchSize: (index + 1) * self.batchSize]

    # Pre-allocate batch arrays (use empty for slight efficiency gain)
    images = np.empty((self.batchSize, self.noOfPatches, self.embedDimension), dtype=np.float32)

    if (self.classMode == "categorical"):
      labels = np.zeros((self.batchSize, self.numClasses), dtype=np.float32)
    else:
      labels = np.zeros((self.batchSize, 1), dtype=np.int32)

    for i, idx in enumerate(batchIndices):
      # Load image path and label from DataFrame.
      row = self.dataFrame.iloc[idx]
      imgPath = row["image_path"]
      label = row["label"]

      # Load and validate image.
      image = cv2.imread(imgPath)
      if (image is None):
        raise ValueError(f"Failed to load image: {imgPath}")

      # Convert BGR (OpenCV default) to RGB for model compatibility.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Resize to expected input dimensions.
      image = cv2.resize(
        image,
        (self.inputShape[1], self.inputShape[0]),  # (width, height) for cv2.resize.
        interpolation=cv2.INTER_CUBIC
      )

      # Extract and flatten patches.
      patches = self._image_to_patches(image)
      images[i] = patches

      # Encode label.
      if (self.classMode == "categorical"):
        labels[i, self.classIndices[label]] = 1.0
      else:
        labels[i, 0] = self.classIndices[label]

    # Normalize pixel values to [0, 1].
    return images / 255.0, labels

  def on_epoch_end(self):
    '''Callback invoked at the end of each epoch for shuffling.'''
    if self.shuffle:
      np.random.shuffle(self.indices)

  def get_class_indices(self):
    '''Return the mapping from class names to integer indices.'''
    return self.classIndices.copy()


class ViTPatchDataGeneratorFromFolder(Sequence):
  r'''
  Keras Sequence generator for Vision Transformer training using images from a folder structure.
  This generator loads images from a specified folder structure, extracts non-overlapping patches,
  flattens them to embedding vectors, and yields batches compatible with the VisionTransformer model architecture.

  Parameters:
    folder (str): Path to the root folder containing subfolders for each class, with images inside those subfolders.
    inputShape (tuple): Expected input shape of the images (height, width, channels).
    batchSize (int): Number of samples per batch.
    classMode (str): "categorical" for one-hot encoded labels, "sparse" for integer labels.
    noOfPatches (int): Number of patches to extract from each image (default 256).
    patchSize (int): Size of each square patch (default 16, resulting in 16x16 patches).
    embedDimension (int): Dimension of the flattened patch embeddings (default 768 for ViT-Base).
    shuffle (bool): Whether to shuffle the data at the end of each epoch (default False).

  Returns:
    A batch of data in the form (images, labels) where:
      - images: A numpy array of shape (batchSize, noOfPatches, embedDimension) containing the flattened patch embeddings for each image in the batch.
      - labels: A numpy array of shape (batchSize, numClasses) for "categorical" classMode or (batchSize, 1) for "sparse" classMode containing the corresponding labels for each image in the batch.
  '''

  def __init__(
    self, folder, inputShape, batchSize, classMode="categorical",
    noOfPatches=256, patchSize=16, embedDimension=768, shuffle=False,
  ):
    self.folder = folder
    self.inputShape = inputShape
    self.batchSize = batchSize
    self.classMode = classMode
    self.noOfPatches = noOfPatches
    self.patchSize = patchSize
    self.embedDimension = embedDimension
    self.classes = os.listdir(folder)
    self.shuffle = shuffle

    self.listOfImages = []
    self.listOfLabels = []

    for label in os.listdir(folder):
      labelFolder = os.path.join(folder, label)
      for image in os.listdir(labelFolder):
        self.listOfImages.append(os.path.join(labelFolder, image))
        self.listOfLabels.append(label)

    self.numImages = len(self.listOfImages)
    self.indices = np.arange(self.numImages)
    self.classIndices = {label: index for index, label in enumerate(self.classes)}

    np.random.shuffle(self.indices)

  def __len__(self):
    return self.numImages // self.batchSize

  def __getitem__(self, index):
    indices = self.indices[index * self.batchSize: (index + 1) * self.batchSize]
    images = np.zeros((self.batchSize, self.noOfPatches, self.embedDimension))
    if (self.classMode == "categorical"):
      labels = np.zeros((self.batchSize, len(self.classIndices)))
    else:
      labels = np.zeros((self.batchSize, 1))

    for i, index in enumerate(indices):
      image = self.listOfImages[index]
      label = self.listOfLabels[index]

      image = cv2.imread(image)
      image = cv2.resize(image, (self.inputShape[1], self.inputShape[0]), interpolation=cv2.INTER_CUBIC)
      patches = ImageToPatches(image, self.noOfPatches, self.patchSize)

      images[i] = patches

      if (self.classMode == "categorical"):
        labels[i, self.classIndices[label]] = 1
      else:
        labels[i] = self.classIndices[label]

    return images / 255.0, labels

  def on_epoch_end(self):
    if (self.shuffle):
      np.random.shuffle(self.indices)

  def __iter__(self):
    for index in range(0, len(self)):
      yield self.__getitem__(index)

  def __next__(self):
    return self.__iter__()


def Conv2DBlock(
  inputLayer,  # Input layer.
  filters,  # Number of filters.
  kernelSize=3,  # Kernel size.
  padding="same",  # Padding.
  strides=(1, 1),  # Stride.
  kernelInitializer="he_normal",  # Kernel initializer.
  activation="relu",  # Activation function.
  applyBatchNorm=False,  # Apply batch normalization.
  applyActivation=True,  # Apply activation.
):
  r'''
  Shorthand helper to apply a Conv2D followed optionally by BatchNormalization and Activation.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): Input Keras tensor to the convolutional block.
    filters (int): Number of convolution filters.
    kernelSize (int or tuple): Convolution kernel size.
    padding (str): Padding mode, e.g. "same" or "valid".
    strides (tuple): Stride of the convolution.
    kernelInitializer (str): Kernel initializer name.
    activation (str or callable): Activation to apply if applyActivation is True.
    applyBatchNorm (bool): If True, apply BatchNormalization after convolution.
    applyActivation (bool): If True, apply the activation after (optional) BatchNorm.

  Returns:
    tensorflow.keras.layers.Layer: Output Keras tensor after applying the convolutional block with the specified options.
  '''

  # Apply convolution.
  conv = Conv2D(
    filters,  # Number of filters.
    kernelSize,  # Kernel size.
    padding=padding,  # Padding.
    strides=strides,  # Stride.
    use_bias=True,  # Use bias.
    kernel_initializer=kernelInitializer,  # Kernel initializer.
  )(inputLayer)

  # Check if batch normalization is required.
  if (applyBatchNorm):
    conv = BatchNormalization()(conv)

  # Check if activation is required.
  if (applyActivation):
    conv = Activation(activation)(conv)

  return conv


def DropoutBlock(inputLayer, dropoutRatio, dropoutType="spatial"):
  r'''
  Apply either spatial or standard dropout according to the requested type.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): Input Keras tensor to the dropout block.
    dropoutRatio (float): Drop probability in (0, 1]. If 0.0 returns input unchanged.
    dropoutType (str): One of {"spatial", "feature"}. "spatial" -> SpatialDropout2D, "feature" -> Dropout.

  Returns:
    tensorflow.keras.layers.Layer: Output Keras tensor after applying the specified dropout. If dropoutRatio is 0.0, returns inputLayer unchanged.
  '''

  # Apply the requested dropout type if dropoutRatio is greater than 0. Otherwise, return the input layer unchanged.
  if (dropoutRatio > 0.0):
    if (dropoutType == "spatial"):
      # Spatial dropout layer.
      output = SpatialDropout2D(dropoutRatio)(inputLayer)
    elif (dropoutType == "feature"):
      # Dropout layer.
      output = Dropout(dropoutRatio)(inputLayer)
    else:
      output = inputLayer
  else:
    output = inputLayer
  return output


def DownSamplingBlock(
  inputLayer,  # Input layer.
  filters,  # Number of filters.
  kernelSize=3,  # Kernel size.
  padding="same",  # Padding.
  kernelInitializer="he_normal",  # Kernel initializer.
  downsmaplingType="maxpooling",  # Downsampling type.
):
  r'''
  Generic downsampling block supporting different strategies.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): input to the downsampling block.
    filters (int): Number of filters for conv variants (ignored for max-pooling path).
    kernelSize (int or tuple): Kernel size for convolutional options.
    padding (str): Padding mode for conv variants.
    kernelInitializer (str): Kernel initializer for conv variants.
    downsmaplingType (str): One of {"maxpooling", "stridedconvolution", "dilatedconvolution"}.

  Returns:
    tensorflow.keras.layers.Layer:  Keras tensor after applying the requested downsampling operation.
  '''

  # Check if max pooling or strided convolution.
  if (downsmaplingType == "stridedconvolution"):
    output = Conv2D(
      filters,  # Number of filters.
      kernelSize,  # Kernel size.
      strides=(2, 2),  # Strides.
      padding=padding,  # Padding.
      kernel_initializer=kernelInitializer,  # Kernel initializer.
    )(inputLayer)
  elif (downsmaplingType == "dilatedconvolution"):
    output = Conv2D(
      filters,  # Number of filters.
      kernelSize,  # Kernel size.
      dilation_rate=(2, 2),  # Dilation rate.
      padding=padding,  # Padding.
      kernel_initializer=kernelInitializer,  # Kernel initializer.
    )(inputLayer)
  else:  # Default is max pooling.
    output = MaxPooling2D(pool_size=(2, 2))(inputLayer)

  return output


class SubpixelConv2D(Layer):
  r'''
  Keras Layer implementing sub-pixel convolution upsampling via depth_to_space.

  This layer expects the incoming channel dimension to be divisible by scale^2 and
  performs TensorFlow's depth_to_space to rearrange depth into spatial dimensions.

  Parameters:
    scale (int): Upscaling factor (e.g. 2).

  Notes:
    The layer is registered in Keras custom objects for model (de)serialization.
  '''

  def __init__(self, scale=2, **kwargs):
    # Scale is the upscaling ratio, a single integer.
    self.scale = scale
    super(SubpixelConv2D, self).__init__(**kwargs)

  def call(self, inputs):
    # Upscaling is done by depth to space operation.
    # Sub-pixel convolution is learnable.
    # See https://arxiv.org/abs/1609.05158
    return depth_to_space(inputs, self.scale)

  def compute_output_shape(self, inputShape):
    return (
      inputShape[0],  # Batch
      # Height
      inputShape[1] * self.scale if inputShape[1] else None,
      # Width
      inputShape[2] * self.scale if inputShape[2] else None,
      int(inputShape[3] / (self.scale ** 2))  # Channels
    )

  def get_config(self):
    config = super(SubpixelConv2D, self).get_config()
    config["scale"] = self.scale
    return config


# Register the custom layer so that it can be serialized.
# See https://www.tensorflow.org/guide/keras/custom_layers_and_models#serializing_custom_objects
get_custom_objects().update({"SubpixelConv2D": SubpixelConv2D})


def UpSamplingBlock(
  inputLayer,  # Input layer.
  filters,  # Number of filters.
  kernelSize=3,  # Kernel size.
  padding="same",  # Padding.
  kernelInitializer="he_normal",  # Kernel initializer.
  upsamplingType="upsampling",  # Upsampling type.
  activation="relu",  # Activation function.
  strides=(2, 2),  # Strides for transposed convolution.
):
  r'''
  Flexible upsampling block supporting standard Upsampling2D + Conv, Conv2DTranspose, or Subpixel conv.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): input to the upsampling block.
    filters (int): Number of filters for conv outputs (for subpixel, `filters*4` is used before depth_to_space).
    kernelSize (int or tuple): Kernel size for convolutional layers.
    padding (str): Padding mode for conv variants.
    kernelInitializer (str): Kernel initializer name.
    upsamplingType (str): One of {"upsampling", "transposedconvolution", "subpixelconvolution"}.
    activation (str or callable): Activation applied in certain branches.
    strides (tuple): Strides for Conv2DTranspose when used.

  Returns:
    tensorflow.keras.layers.Layer: Keras tensor after applying the requested upsampling operation.
  '''

  # Check if up sampling or transposed convolution.
  if (upsamplingType == "transposedconvolution"):
    # Transposed convolution.
    output = Conv2DTranspose(
      filters,  # Number of filters.
      kernelSize,  # Kernel size.
      strides=strides,  # Strides.
      padding=padding,  # Padding.
      use_bias=True,  # Use bias.
      kernel_initializer=kernelInitializer,  # Kernel initializer.
    )(inputLayer)
  elif (upsamplingType == "subpixelconvolution"):
    # Subpixel convolution.
    output = Conv2D(
      filters * 4,  # Number of filters.
      kernelSize,  # Kernel size.
      activation=activation,  # Activation function.
      padding=padding,  # Padding.
      use_bias=True,  # Use bias.
      kernel_initializer=kernelInitializer,  # Kernel initializer.
    )(inputLayer)
    output = SubpixelConv2D(scale=2)(output)
  else:  # Default is upsampling.
    # Up sampling.
    output = UpSampling2D(size=(2, 2))(inputLayer)
    output = Conv2D(
      filters,  # Number of filters.
      kernelSize,  # Kernel size.
      activation=activation,  # Activation function.
      padding=padding,  # Padding.
      use_bias=True,  # Use bias.
      kernel_initializer=kernelInitializer,  # Kernel initializer.
    )(output)

  return output


def AttentionLayer(
  input1,  # Input 1.
  input2,  # Input 2.
  filters,  # Number of filters.
  kernelSize=1,  # Kernel size.
  strides=(1, 1),  # Strides.
  padding="same",  # Padding.
  kernelInitializer="he_normal",  # Kernel initializer.
):
  r'''
  Lightweight attention gating layer inspired by attention U-Net style blocks.

  This block computes a gating mask between two inputs (e.g., encoder skip and decoder feature maps)
  by convolving both to a common number of filters, summing, applying ReLU then a 1-channel sigmoid
  to produce an attention map which is multiplied with `input1`.

  Parameters:
    input1 (tensorflow.keras.layers.Layer): the tensor to be gated (e.g., skip connection features).
    input2 (tensorflow.keras.layers.Layer): the gating tensor (e.g., decoder features).
    filters (int): Number of filters used internally when projecting inputs to the same space.
    kernelSize (int): Kernel size for internal convolutions.
    strides (tuple): Stride for internal convolutions.
    padding (str): Padding mode for internal convolutions.
    kernelInitializer (str): Kernel initializer for internal convolutions.

  Returns:
    tensorflow.keras.layers.Layer: The result of applying the attention gating to `input1`, where `input1` is multiplied by the attention mask computed from both inputs. The output has the same shape as `input1` but with its features modulated by the attention mechanism.
  '''

  configs = {
    # Kernel size for the internal convolutions that project both inputs to a common feature space.
    "kernelSize"       : kernelSize,
    # Padding mode for the internal convolutions (e.g., "same" to preserve spatial dimensions).
    "padding"          : padding,
    # Strides for the internal convolutions (e.g., (1, 1) for no spatial downsampling).
    "strides"          : strides,
    # Kernel initializer for the internal convolutional layers (e.g., "he_normal" for good initialization in ReLU networks).
    "kernelInitializer": kernelInitializer,
    # Whether to apply activation after the internal convolutions (set to False here since we apply ReLU explicitly after summing).
    "applyActivation"  : False,
    # Whether to apply batch normalization in the internal convolutions (set to False for simplicity in this attention block).
    "applyBatchNorm"   : False,
  }

  # Apply convolution to both inputs.
  input1Conv = Conv2DBlock(input1, **configs, filters=filters)
  input2Conv = Conv2DBlock(input2, **configs, filters=filters)

  # Add both inputs.
  addBoth = add([input1Conv, input2Conv])

  # Apply ReLU activation.
  f = Activation("relu")(addBoth)

  # Apply convolution to the result.
  g = Conv2DBlock(f, **configs, filters=1)

  # Apply Sigmoid activation.
  h = Activation("sigmoid")(g)

  # Multiply the input with the result.
  result = multiply([input1, h])

  # Return the result.
  return result


def AttentionConcatenate(convLayer, skipConnection):
  r'''
  Apply attention gating to the `skipConnection` and concatenate the gated features with `convLayer`.

  Parameters:
    convLayer (tensorflow.keras.layers.Layer): the decoder/upsampled features.
    skipConnection (tensorflow.keras.layers.Layer): encoder skip connection to be gated and concatenated.

  Returns:
    tensorflow.keras.layers.Layer: The result of concatenating `convLayer` with the attention-gated version of `skipConnection`. This allows the model to focus on relevant features from the skip connection when merging with the decoder features.
  '''

  # Number of filters for the attention layer is typically set to the number of filters in the `convLayer`
  # to ensure compatibility when applying the attention mechanism.
  filters = convLayer.get_shape().as_list()[-1]
  # Apply attention gating to the skip connection using the `convLayer` as the gating signal.
  # The `AttentionLayer` computes an attention mask that modulates the `skipConnection` features based on
  # their relevance to the `convLayer` features.
  attention = AttentionLayer(skipConnection, convLayer, filters)
  # Concatenate the original `convLayer` features with the attention-gated `skipConnection` features.
  # This allows the model to combine the decoder features with the most relevant information from the
  # encoder skip connection, enhancing the feature representation for subsequent layers.
  attConc = concatenate([convLayer, attention])
  return attConc


def ConcatenateBlock(
  inputLayer,  # Input layer.
  forwardLayer,  # Skip layer.
  concatenateType="concatenate"  # Concatenate type.
):
  r'''
  Concatenate helper that supports plain concatenation or attention-based concatenation.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): typically the upsampled/decoder features.
    forwardLayer (tensorflow.keras.layers.Layer): the skip/encoder features to merge.
    concatenateType (str): "concatenate" or "attention".

  Returns:
    tensorflow.keras.layers.Layer: The result of merging `inputLayer` and `forwardLayer` using the specified concatenation strategy. If "concatenate", it performs a simple concatenation along the channel axis. If "attention", it applies attention gating to `forwardLayer` before concatenating with `inputLayer`, allowing the model to focus on relevant features from the skip connection.
  '''

  # Check if concatenation or attention.
  if (concatenateType == "attention"):
    # Concatenate with attention.
    output = AttentionConcatenate(inputLayer, forwardLayer)
  else:  # Default is concatenation.
    # Concatenate the skip layer and up convolution.
    output = concatenate([inputLayer, forwardLayer], axis=3)

  return output


def DownResidualBlock(
  inputLayer,  # Input layer.
  stage=1,  # Stage.
  activation="relu",  # Activation function.
  applyActivation=True,  # Apply activation.
  dropoutRatio=0.0,  # Dropout rate.
  dropoutType="spatial",  # Dropout type.
  applyBatchNorm=True,  # Apply batch normalization.
  kernelInitializer="he_normal",  # Kernel initializer.
):
  r'''
  Residual block for the encoder (downsampling) path.

  The block applies a small stack of convolutional layers (number controlled by `stage`),
  adds a residual connection to the input, optionally applies activation and dropout, then
  performs a downsampling convolution to reduce spatial resolution and increase filters.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): input to the residual block.
    stage (int): Controls number of internal convolutional layers and base filter count (16 * 2^(stage-1)).
    activation (str or callable): Activation function to use.
    applyActivation (bool): Whether to apply activation after the residual addition.
    dropoutRatio (float): Dropout rate applied after residual addition (0.0 disables dropout).
    dropoutType (str): "spatial" or "feature": type of dropout used when dropoutRatio > 0.
    applyBatchNorm (bool): Whether to use BatchNormalization in each conv block.
    kernelInitializer (str): Kernel initializer name for conv layers.

  Returns:
    tuple: (downsampledOutput, outputBeforeDownSampling)
      - downsampledOutput: Keras tensor after residual + downsampling conv.
      - outputBeforeDownSampling: Keras tensor of the residual output prior to downsampling (for skip connections).
  '''

  conv = inputLayer

  filters = 16 * (2 ** (stage - 1))  # Number of filters.
  stages = min(stage, 3)  # Number of stages.

  for _ in range(stages):
    conv = Conv2DBlock(
      conv,
      filters,  # Number of filters.
      kernelSize=5,  # Kernel size.
      padding="same",  # Padding.
      strides=(1, 1),  # Stride.
      kernelInitializer=kernelInitializer,  # Kernel initializer.
      activation=activation,  # Activation function.
      applyBatchNorm=applyBatchNorm,  # Apply batch normalization.
      applyActivation=applyActivation,  # Apply activation.
    )

  output = add([inputLayer, conv])

  # Check if activation is required.
  if (applyActivation):
    output = Activation(activation)(output)

  # Check if dropout is required.
  if (dropoutRatio > 0.0):
    output = DropoutBlock(output, dropoutRatio, dropoutType)

  outputBeforeDownSampling = output

  filtersAlt = 16 * (2 ** stage)  # Number of filters.

  # Down sampling.
  output = Conv2DBlock(
    output,
    filtersAlt,  # Number of filters.
    kernelSize=2,  # Kernel size.
    padding="same",  # Padding.
    strides=(2, 2),  # Stride.
    kernelInitializer=kernelInitializer,  # Kernel initializer.
    activation=activation,  # Activation function.
    applyBatchNorm=applyBatchNorm,  # Apply batch normalization.
    applyActivation=applyActivation,  # Apply activation.
  )

  return output, outputBeforeDownSampling


def UpResidualBlock(
  inputLayer,  # Input layer.
  forwardLayer,  # Skip layer.
  stage=1,  # Stage.
  activation="relu",  # Activation function.
  applyActivation=True,  # Apply activation.
  applyBatchNorm=True,  # Apply batch normalization.
  kernelInitializer="he_normal",  # Kernel initializer.
  concatenateType="concatenate",  # Concatenate type.
):
  r'''
  Residual block for the decoder (upsampling) path.

  The block merges the upsampled tensor with a skip connection (either plain concat or attention),
  applies a small stack of convolutions, adds a residual connection to the upsampled input,
  and optionally upsamples when `stages > 1`.

  Parameters:
    inputLayer (tensorflow.keras.layers.Layer): the decoder/upsampled features.
    forwardLayer (tensorflow.keras.layers.Layer): encoder skip connection to be merged.
    stage (int): Controls number of internal convolutional layers and base filter count.
    activation (str or callable): Activation function to use.
    applyActivation (bool): Whether to apply activation after residual addition.
    applyBatchNorm (bool): Whether to use BatchNormalization after ConvTranspose/Upsampling.
    kernelInitializer (str): Kernel initializer for conv layers.
    concatenateType (str): "concatenate" or "attention" for merging.

  Returns:
    tensorflow.keras.layers.Layer: Keras tensor after applying the up residual block, which includes merging with the skip connection, convolutional processing, and optional upsampling. The output has enhanced features due to the residual connection and the merged skip features, and is ready for further processing in the decoder path.
  '''

  filters = 16 * (2 ** stage)  # Number of filters.
  stages = min(stage + 1, 3)  # Number of stages.

  # Apply concatenation.
  merge = ConcatenateBlock(inputLayer, forwardLayer, concatenateType)

  conv = merge

  for _ in range(stages):
    conv = Conv2DBlock(
      conv,
      filters,  # Number of filters.
      kernelSize=5,  # Kernel size.
      padding="same",  # Padding.
      strides=(1, 1),  # Stride.
      kernelInitializer=kernelInitializer,  # Kernel initializer.
      activation=activation,  # Activation function.
      applyBatchNorm=applyBatchNorm,  # Apply batch normalization.
      applyActivation=applyActivation,  # Apply activation.
    )

  # Add the skip connection.
  output = add([conv, inputLayer])

  # Check if activation is required.
  if (applyActivation):
    # Apply activation function.
    output = Activation(activation)(output)

  if (stages > 1):
    filters = 16 * (2 ** (stage - 1))  # Number of filters.

    output = UpSamplingBlock(
      output,
      filters,  # Number of filters.
      kernelSize=2,  # Kernel size.
      padding="valid",  # Padding.
      kernelInitializer=kernelInitializer,  # Kernel initializer.
      strides=(2, 2),  # Strides for transposed convolution.
      upsamplingType="transposedconvolution",  # Upsampling type.
      activation=None,  # Activation function.
    )

    # Check if batch normalization is required.
    if (applyBatchNorm):
      output = BatchNormalization()(output)

    # Check if activation is required.
    if (applyActivation):
      output = Activation(activation)(output)

  return output


def BuildOptimizer(optimizerSpec):
  r'''
  Build a Keras optimizer instance from a flexible specification.

  Parameters:
    optimizerSpec: Can be one of the following:
      - An instance of a Keras optimizer (e.g., `tf.keras.optimizers.Adam()`). A fresh instance will be created from its config.
      - A tuple of (optimizerClass, optimizerKwargs) where `optimizerClass` is a Keras optimizer class and `optimizerKwargs` is a dict of keyword arguments to instantiate it.
      - A Keras optimizer class (e.g., `tf.keras.optimizers.Adam`) which will be instantiated with default parameters.
      - A callable that returns an instance of a Keras optimizer when called with no arguments.

  Returns:
    tensorflow.keras.optimizers.Optimizer: A new instance of the specified Keras optimizer.
  '''

  # Always return a fresh optimizer instance for each trial/model.
  if (isinstance(optimizerSpec, tf.keras.optimizers.Optimizer)):
    return optimizerSpec.__class__.from_config(optimizerSpec.get_config())

  if (isinstance(optimizerSpec, tuple) and len(optimizerSpec) == 2):
    optimizerClass, optimizerKwargs = optimizerSpec
    return optimizerClass(**optimizerKwargs)

  if (isinstance(optimizerSpec, type) and issubclass(
    optimizerSpec,
    tf.keras.optimizers.Optimizer,
  )):
    return optimizerSpec()

  if (callable(optimizerSpec)):
    optimizer = optimizerSpec()
    if (isinstance(optimizer, tf.keras.optimizers.Optimizer)):
      return optimizer

  raise ValueError(f"Invalid optimizer specification: {optimizerSpec}")


def TFDataset(X, y, batchSize=8):
  r'''
  Create a TensorFlow dataset from the given data and batch size, with shuffling, parsing, batching, prefetching, and repeating.

  Parameters:
    X (list or array-like): List of input data (e.g., image paths).
    y (list or array-like): List of corresponding labels or masks.
    batchSize (int): Number of samples per batch.

  Returns:
    tensorflow.data.Dataset: A TensorFlow dataset that yields batches of parsed data, shuffled and prefetched for optimized training. The dataset is set to repeat indefinitely, so it can be used directly in model training loops without needing to specify the number of epochs in the dataset itself.
  '''

  from HMB.ImagesHelper import ReadImage, ReadMask

  def _TFParse(X, y):
    # Read the image and mask.
    image = tf.numpy_function(ReadImage, [X], tf.float64)
    mask = tf.numpy_function(ReadMask, [y], tf.float64)

    # Set the shape of the images and masks.
    image.set_shape([image.shape[0], image.shape[1], 3])
    mask.set_shape([mask.shape[0], mask.shape[1], 1])

    # Return the image and mask.
    return image, mask

  # Create a TensorFlow dataset.
  dataset = tf.data.Dataset.from_tensor_slices((X, y))

  # Shuffle the dataset.
  dataset = dataset.shuffle(buffer_size=batchSize)

  # Parse images and masks.
  dataset = dataset.map(_TFParse)

  # Batch the dataset.
  dataset = dataset.batch(batchSize)

  # Prefetch the dataset to optimize training.
  dataset = dataset.prefetch(1)

  # Repeat the dataset indefinitely.
  dataset = dataset.repeat()

  # Return the dataset.
  return dataset


def CompileTrainTFUNetModel(
  model,
  trialNo,
  inputSize,
  imagesList,
  masksList,
  hyperparameters,
  pretrainedWeights=None,
  epochs=25,
  outputFolder=None,
  keyword=None,
  testSize=0.25,
  randomState=42,
):
  r'''
  Compile and train a TFUNet model with the given parameters, and store the results.

  Parameters:
    model (tf.keras.Model): The TFUNet model instance to be trained.
    trialNo (int): The trial number for this training run, used for organizing outputs.
    inputSize (tuple): The expected input size of the model (height, width, channels).
    imagesList (str): Path to the directory containing input images.
    masksList (str): Path to the directory containing corresponding segmentation masks.
    hyperparameters (dict): A dictionary of hyperparameters for training (e.g., optimizer, loss, batchSize).
    pretrainedWeights (str, optional): Path to pretrained weights to initialize the model, if any.
    epochs (int): Number of epochs to train the model.
    outputFolder (str): Path to the folder where training outputs (weights, logs, hyperparameters) will be stored.
    keyword (str): A keyword to uniquely identify this training run, used in naming output files and directories.
    testSize (float): Ratio of the dataset to be used as the test set when splitting the data. The remaining will be used for training and validation.
    randomState (int): Random state for reproducibility when splitting the dataset.
  '''

  from HMB.ImagesHelper import ReadImage, ReadMask
  from sklearn.model_selection import train_test_split

  def _LoadData(imagesList, masksList, testSize=0.2):
    # Split the dataset into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      imagesList,  # Images.
      masksList,  # Masks.
      test_size=testSize,  # Ratio of the testing set.
      random_state=randomState,  # Random state.
    )

    # Split the training set into training and validation sets.
    xTrain, xVal, yTrain, yVal = train_test_split(
      xTrain,  # Images.
      yTrain,  # Masks.
      test_size=testSize,  # Ratio of the validation set.
      random_state=randomState,  # Random state.
    )

    # Return the dataset.
    return xTrain, xVal, xTest, yTrain, yVal, yTest

  def _HyperparameterToString(key, value):
    if (key.lower() == "optimizer"):
      optimizer = BuildOptimizer(value)
      config = optimizer.get_config()
      return f"{optimizer.__class__.__name__}({config})"
    return str(value)

  # Create the output directory for the trial.
  storageDir = os.path.join(outputFolder, keyword)
  os.makedirs(storageDir, exist_ok=True)

  # Clear the session.
  clear_session()

  # Load the dataset (split paths).
  xTrain, xVal, xTest, yTrain, yVal, yTest = _LoadData(
    imagesList,  # Path to the images' directory.
    masksList,  # Path to the masks' directory.
    testSize=testSize,  # Ratio of the testing set.
  )

  batchSize = hyperparameters["batchSize"]

  # Create the training and validation datasets.
  trainDataset = TFDataset(xTrain, yTrain, batchSize=batchSize)
  valDataset = TFDataset(xVal, yVal, batchSize=batchSize)

  # Calculate the number of steps per epoch.
  trainSteps = len(xTrain) // batchSize
  valSteps = len(xVal) // batchSize

  # Add the remaining samples.
  if (len(xTrain) % batchSize != 0):
    trainSteps += 1
  if (len(xVal) % batchSize != 0):
    valSteps += 1

  path = os.path.join(storageDir, keyword)

  callbacks = [
    ModelCheckpoint(
      f"{path}.weights.h5",
      save_best_only=True,
      save_weights_only=True,
      monitor="val_loss",
      mode="min",
      verbose=0,
    ),
    EarlyStopping(
      monitor="val_loss",
      # Set the minimum change in the monitored quantity
      # to be considered an improvement.
      patience=int(epochs * 0.1) + 1,
      restore_best_weights=True,
      verbose=0,
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4),
    CSVLogger(f"{path}.csv"),
    TensorBoard(
      log_dir=os.path.join(storageDir, "Logs"),
      histogram_freq=1,
    ),
  ]

  # Compile the model with a new optimizer instance for this model's variables.
  optimizer = BuildOptimizer(hyperparameters["optimizer"])
  model.compile(
    optimizer=optimizer,  # Optimizer.
    loss=hyperparameters["loss"],  # Loss function.
    metrics=metrics,  # Metrics.
  )

  # Load the pretrained weights if provided.
  if (pretrainedWeights):
    model.load_weights(pretrainedWeights)

  # Print the model summary.
  # model.summary()

  # Train the model.
  history = model.fit(
    trainDataset,  # Training data.
    batch_size=batchSize,  # Batch size.
    epochs=epochs,  # Epochs.
    validation_data=valDataset,  # Data for evaluation.
    steps_per_epoch=trainSteps,  # Number of steps per epoch.
    validation_steps=valSteps,  # Number of steps per validation.
    shuffle=True,  # Whether to shuffle the training data before each epoch.
    callbacks=callbacks,  # List of callbacks.
    verbose=2,  # Verbosity mode: 0, 1, or 2.
  )

  trainMetrics = model.evaluate(trainDataset, steps=trainSteps, verbose=0)
  valMetrics = model.evaluate(valDataset, steps=valSteps, verbose=0)

  print("Train Metrics:", trainMetrics)
  print("Val Metrics:", valMetrics)

  # Store the hyperparameters.
  dictToStore = {}
  for key in hyperparameters.keys():
    dictToStore[key.capitalize()] = _HyperparameterToString(
      key,
      hyperparameters[key],
    )
  dictToStore["Trial No."] = "Trial " + str(trialNo)
  dictToStore["Epochs"] = str(epochs)
  dictToStore["Train Steps"] = str(trainSteps)
  dictToStore["Val Steps"] = str(valSteps)
  dictToStore["Pretrained Weights"] = str(pretrainedWeights)
  dictToStore["Output Folder"] = str(outputFolder)
  dictToStore["Storage Dir"] = str(storageDir)
  dictToStore["Keyword"] = str(keyword)
  dictToStore["Path"] = str(path)
  dictToStore["Input Size"] = str(inputSize)
  for i, metric in enumerate(model.metrics_names):
    dictToStore[metric.capitalize()] = trainMetrics[i]
    dictToStore["Val " + metric.capitalize()] = valMetrics[i]

  hyperparametersPath = os.path.join(storageDir, "Hyperparameters.csv")
  if (os.path.exists(hyperparametersPath)):
    df = pd.read_csv(hyperparametersPath)
    dfList = df.values.tolist()
    dfList.append(list(dictToStore.values()))
    df = pd.DataFrame(dfList, columns=list(dictToStore.keys()))
  else:
    df = pd.DataFrame(dictToStore, index=[0])
  df.to_csv(hyperparametersPath, index=False)
  # Save hyperparameters and metrics to a JSON file.
  df.to_json(
    hyperparametersPath.replace(".csv", ".json"),
    orient="records",
    indent=4,
  )

  # Plot training and validation accuracy values.
  plt.figure(figsize=(10, 10))
  metricsNames = ["loss", "accuracy"]
  for i, metric in enumerate(metricsNames):
    plt.subplot(2, 1, i + 1)
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model " + metric)
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.tight_layout()
    plt.grid()
  plt.savefig(
    f"{path}.pdf",
    dpi=720,
    bbox_inches="tight"
  )

  print("Trial", trialNo, "completed.")
