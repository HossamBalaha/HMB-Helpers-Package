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
  gradModel = tf.keras.models.Model([model.inputs], [convLayer, model.output])

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
    lastConvLayerName=None,
  )
  '''

  from HMB.ImagesHelper import OverlayHeatmapOnImage

  os.makedirs(outFolder, exist_ok=True)

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

    outPath = os.path.join(outFolder, f"GradCAM_IDx{idx}_Pred{predClass}.png")
    overlay.save(outPath)
