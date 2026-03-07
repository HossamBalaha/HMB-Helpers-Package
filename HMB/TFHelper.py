import os
import numpy as np
import tensorflow as tf
from PIL import Image


# Compute Grad-CAM heatmap for a single image and target class.
def TFGradCam(model, imgTensor, classIdx=None, lastConvLayerName=None):
  '''
  Compute Grad-CAM heatmap for imgTensor and target class index.

  Parameters:
    model (tensorflow.keras.Model): Trained Keras model.
    imgTensor (numpy.ndarray or tf.Tensor): Shape (1,H,W,3) preprocessed input.
    classIdx (int or None): Target class index; if None uses model prediction.
    lastConvLayerName (str|None): Specify conv layer name; if None pick last Conv2D.

  Returns.
    heatmap (2D numpy array): normalized heatmap in [0,1].

  Example
  -------
  .. code-block:: python

  import HMB.TFHelper as tfh

  model = ...  # Load or build model.
  img = ...    # Load and preprocess image to shape (1, H, W, 3).
  heatmap = tfh.TFGradCam(model, img, classIdx=2, lastConvLayerName=None)
  '''

  # Convert to tensor and ensure batch dimension.
  x = tf.convert_to_tensor(imgTensor, dtype=tf.float32)

  # Find last convolutional 2D layer if name not provided.
  if (lastConvLayerName is None):
    lastConv = None
    for layer in reversed(model.layers):
      if (isinstance(layer, tf.keras.layers.Conv2D)):
        lastConv = layer
        break
    if (lastConv is None):
      raise ValueError("TFGradCam: no Conv2D layer found in model.")
    lastConvLayerName = lastConv.name

  # Build a model that outputs conv layer activations and predictions.
  convLayer = model.get_layer(lastConvLayerName).output
  # Use the same input structure as the original model to avoid Keras warnings about input nesting
  # gradModel = tf.keras.models.Model([model.inputs], [convLayer, model.output])
  gradModel = tf.keras.models.Model(model.inputs, [convLayer, model.output])

  with tf.GradientTape() as tape:
    convOutputs, predictions = gradModel(x)
    if (classIdx is None):
      classIdx = tf.argmax(predictions[0])
    classScore = predictions[:, classIdx]

  # Compute gradients of the class score w.r.t conv outputs.
  grads = tape.gradient(classScore, convOutputs)

  # Compute channel-wise mean of gradients.
  weights = tf.reduce_mean(grads, axis=(1, 2))
  convOutputs = convOutputs[0]
  weights = weights[0]

  # Weighted combination of activations.
  cam = tf.zeros(shape=convOutputs.shape[:2], dtype=tf.float32)
  for i in range(int(convOutputs.shape[-1])):
    cam += weights[i] * convOutputs[:, :, i]

  # Relu and normalize.
  cam = tf.nn.relu(cam)
  cam = cam.numpy()
  if (cam.max() != 0):
    cam = (cam - cam.min()) / (cam.max() - cam.min())
  else:
    cam = np.zeros_like(cam)

  return cam


# Save Grad-CAM overlays for a list of sample indices.
def SaveGradCamsForSamples(
    model,
    imgPaths,
    sampleIndices,
    outFolder,
    imgSize=(512, 512),
    lastConvLayerName=None
):
  '''
  Compute and save Grad-CAM overlays for the provided samples.

  Parameters:
    model (tensorflow.keras.Model): Trained Keras model.
    imgPaths (list): List of image file paths in the same order as indices refer to.
    sampleIndices (array-like): Indices to visualize.
    outFolder (str): Output folder where overlays will be saved.
    imgSize (tuple): Size to resize images for model input.
    lastConvLayerName (str): Optional conv layer to use.

  Example
  -------
  .. code-block:: python

    import HMB.TFHelper as tfh

    model = ...  # Load or build model.
    imgPaths = [...]  # List of image file paths.
    sampleIndices = [0, 5, 10]  # Indices of samples to visualize.
    outFolder = "GradCAM_Overlays"

    tfh.SaveGradCamsForSamples(
      model,
      imgPaths,
      sampleIndices,
      outFolder,
      imgSize=(512, 512),
      lastConvLayerName=None
    )
  '''

  from HMB.ImagesHelper import OverlayHeatmapOnImage

  # Input validation.
  if ((model is None) or (imgPaths is None) or (sampleIndices is None) or (outFolder is None)):
    raise ValueError("Model, imgPaths, sampleIndices, and outFolder are required.")
  if (not isinstance(imgPaths, list) or (len(imgPaths) == 0)):
    raise ValueError("imgPaths must be a non-empty list.")
  if (not isinstance(outFolder, str) or (len(outFolder.strip()) == 0)):
    raise ValueError("Invalid output folder.")
  # Raise if path exists and is a file, or cannot be created.
  if (os.path.exists(outFolder) and not os.path.isdir(outFolder)):
    raise ValueError("Output path is not a directory.")
  try:
    os.makedirs(outFolder, exist_ok=True)
  except Exception:
    raise ValueError("Failed to create output folder.")

  for idx in sampleIndices:
    imgPath = imgPaths[int(idx)]
    try:
      orig = Image.open(imgPath).convert("RGB")
    except Exception:
      orig = Image.new("RGB", imgSize, (255, 255, 255))

    # Prepare model input.
    inp = orig.resize(imgSize)
    inpArr = np.asarray(inp).astype(np.float32) / 255.0
    inpBatch = np.expand_dims(inpArr, axis=0)

    # Compute prediction to get predicted class.
    preds = model(inpBatch, training=False).numpy()
    predClass = int(np.argmax(preds[0]))

    # Compute Grad-CAM heatmap.
    try:
      heatmap = TFGradCam(model, inpBatch, classIdx=predClass, lastConvLayerName=lastConvLayerName)
    except Exception as e:
      # If Grad-CAM fails, skip and continue.
      print(f"[WARN] TFGradCam failed for {imgPath}: {e}.")
      continue

    # Create and save overlay.
    overlay = OverlayHeatmapOnImage(orig, heatmap, alpha=0.5)

    outPath = os.path.join(outFolder, f"GradCAM_IDx{idx}_Pred{predClass}.pdf")
    overlay.save(outPath)


def BuildPretrainedAttentionModel(
    baseModelString,
    attentionBlockStr,
    inputShape,
    numClasses,
    optimizer=None,
):
  r'''
  Reconstruct model architecture used in training so weights can be loaded.
  This mirrors the model construction portion of Code.py (no training code here).

  Parameters:
    baseModelString (str): backbone model name (e.g. "Xception").
    attentionBlockStr (str): attention block name (e.g. "CBAM").
    inputShape (tuple): input shape used for training (H, W, C).
    numClasses (int): number of target classes.
    optimizer (tensorflow.keras.optimizers.Optimizer or None): optional optimizer to use; if None, defaults to Adam with lr=1e-4.

  Returns:
    model (tensorflow.keras.Model): compiled model ready to load weights and predict.
  '''

  from tensorflow.keras.models import Model
  from tensorflow.keras.losses import SparseCategoricalCrossentropy
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

  # Compile the model.
  model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalCrossentropy(),
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
  baseModel.trainable = True  # Set base model trainable.
  for layer in baseModel.layers[:fineTuneAt]:
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
  # Store the best model after fine-tuning as pickle file.
  picklePath = os.path.join(storageDir, f"TrainedModel.pkl")
  with open(picklePath) as f:
    pickle.dump(model, f)

  configs = {
    "baseModelString"    : baseModelString,
    "attentionBlockStr"  : attentionBlockStr,
    "inputShape"         : inputShape,
    "numClasses"         : numClasses,
    "initialEpochs"      : initialEpochs,
    "fineTuneEpochs"     : fineTuneEpochs,
    "fineTuneAt"         : fineTuneAt,
    "optimizer"          : str(optimizer),
    "modelCheckpointPath": modelCheckpointPath,
    "storageDir"         : storageDir,
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

  import os
  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix, classification_report
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
  from HMB.Utils import DumpJsonFile
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
  )

  configs.update({
    "trainSamples"         : len(trainGenNew.filenames),
    "validSamples"         : len(validGenNew.filenames),
    "testSamples"          : len(testGenNew.filenames),
    "trainBatchSize"       : batchSize,
    "trainStepsPerEpoch"   : len(trainGenNew) // batchSize,
    "validStepsPerEpoch"   : len(validGenNew) // batchSize,
    "testSteps"            : len(testGenNew) // batchSize,
    "augmentationConfigs"  : augmentationConfigs,
    "monitor"              : monitor,
    "earlyStoppingPatience": earlyStoppingPatience,
    "ensureCUDA"           : ensureCUDA,
    "storageDir"           : storageDir,
    "dpi"                  : dpi,
    "modelCheckpointPath"  : modelCheckpointPath,
    "initialEpochs"        : initialEpochs,
    "fineTuneEpochs"       : fineTuneEpochs,
    "imgSize"              : imgSize,
    "imgShape"             : imgShape,
    "baseModelString"      : baseModelString,
    "attentionBlockStr"    : attentionBlockStr,
    "numClasses"           : numClasses,
    "batchSize"            : batchSize,
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
    cmap="Blues",  # Colormap for the heatmap.
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
    "testAccuracy"          : testAccuracy,
    "testLoss"              : testLoss,
    "performanceMetrics"    : pm,
    "performanceMetricsFile": pmFilePath,
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
    "testGenSize"        : len(testGenNew.filenames),
    "trainGenSize"       : len(trainGenNew.filenames),
    "validGenSize"       : len(validGenNew.filenames),
    "historyFilePath"    : historyFilePath,
    "trainingHistoryPlot": savePath,
    "lastModelPath"      : lastModelPath,
  })

  # Store the final results and configurations in a JSON file.
  resultsFilePath = os.path.join(storageDir, "FinalResults.json")
  # Save final results and configs to JSON.
  DumpJsonFile(resultsFilePath, configs)


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
  '''

  # Verify that the dataFrame contains the required columns.
  requiredColumns = set(columnsMap.values())
  if (not requiredColumns.issubset(dataFrame.columns)):
    raise ValueError(f"DataFrame must contain columns: {requiredColumns}")

  if (not os.path.isfile(modelPath) or not os.path.exists(modelPath)):
    raise ValueError(f"Model file not found at path: {modelPath}")

  import os
  import numpy as np
  import pandas as pd
  from sklearn.metrics import confusion_matrix, classification_report
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from HMB.Initializations import (
    DoRandomSeeding, EnsureCUDAAvailable, ClearTensorFlowSession, UpdateMatplotlibSettings
  )
  from HMB.PerformanceMetrics import (
    PlotConfusionMatrix, CalculatePerformanceMetrics,
    PlotROCAUCCurve, PlotPRCCurve, PlotClasswisePRFBar
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
    yPredConfidences = np.array(predConfidences)

    if (labelEncoder is not None):
      yTrueLabels = labelEncoder.inverse_transform(yTrue)
      yPredLabels = labelEncoder.inverse_transform(yPred)
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
    evalResultsFilePath = os.path.join(storageDir, f"{keyword}Results.csv")
    evalResultsDf.to_csv(evalResultsFilePath, index=False)

    # Get class label names from the generator's class indices.
    if (labelEncoder is not None):
      classLabels = labelEncoder.classes_.tolist()
    else:
      classLabels = list(genObj.class_indices.keys())
    cm = confusion_matrix(yTrue, yPred)
    cmPath = os.path.join(storageDir, f"{keyword}ConfusionMatrix.pdf")
    PlotConfusionMatrix(
      cm,  # Confusion matrix (2D list or numpy array).
      classLabels,  # List of class labels.
      normalize=False,  # Whether to normalize the confusion matrix.
      roundDigits=3,  # Number of decimal places to round normalized values.
      title="Confusion Matrix",  # Title of the plot.
      cmap="Blues",  # Colormap for the heatmap.
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
    pmFilePath = os.path.join(storageDir, f"{keyword}PerformanceMetrics.csv")
    pmDf.to_csv(pmFilePath, index=False)
    for key, value in pm.items():
      print(f"{key}: {value}")

    rocPath = os.path.join(storageDir, f"{keyword}ROCCurve.pdf")
    PlotROCAUCCurve(
      yTrue,  # True labels (one-hot or binary).
      yPredProb,  # Predicted labels (one-hot or binary).
      classLabels,  # List of class names.
      areProbabilities=True,  # Whether yPred are probabilities.
      title="ROC Curve & AUC",  # Plot title.
      figSize=(5, 5),  # Figure size.
      cmap=None,  # Colormap for ROC curves.
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

    prcPath = os.path.join(storageDir, f"{keyword}PRCCurve.pdf")
    PlotPRCCurve(
      yTrue,  # True labels (one-hot or binary).
      yPredProb,  # Predicted labels (one-hot or binary).
      classLabels,  # List of class names.
      areProbabilities=True,  # Whether yPred are probabilities.
      title="PRC Curve",  # Plot title.
      figSize=(5, 5),  # Figure size.
      cmap=None,  # Colormap for PRC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=prcPath,  # File name.
      fontSize=16,  # Font size.
      annotateAvg=True,  # Annotate average precision value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    clsWisePRFPath = os.path.join(storageDir, f"{keyword}ClasswisePRFBarPlot.pdf")
    PlotClasswisePRFBar(
      cm,
      classNames=classLabels,
      fontSize=14,
      figsize=(8, 5),
      display=False,
      save=True,
      fileName=clsWisePRFPath,
      dpi=dpi,
      returnFig=False,
    )
