import os, random, warnings, builtins, cv2, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from HMB.TFUNetHelper import VNet, SegNet
from HMB.ImageSegmentationMetrics import *
from HMB.TFHelper import BuildOptimizer, CompileTrainTFUNetModel
from HMB.TFSegmentationLosses import *
from HMB.ImagesHelper import ReadImage, ReadMask
from HMB.Initializations import (
  IMAGE_SUFFIXES,
  DoRandomSeeding,
  UpdateMatplotlibSettings
)

# ============================================================================================ #
# Enable eager execution for TensorFlow 2.x (if not already enabled).
tf.config.run_functions_eagerly(True)
warnings.filterwarnings("ignore")

# Ensure all prints flush by default to make logs appear promptly.
# Save the original built-in print function for delegation.
_original_print = builtins.print


# Define a wrapper that sets flush=True when not explicitly provided.
def print(*args, **kwargs):
  # Ensure flush is True by default when not provided.
  if ("flush" not in kwargs):
    kwargs["flush"] = True
  # Delegate to the original print implementation.
  return _original_print(*args, **kwargs)


# Override the built-in print with our wrapper to ensure all prints are flushed immediately.
builtins.print = print

# Select a specific GPU when multiple GPUs are available (optional).
gpus = tf.config.list_physical_devices("GPU")
gpuIndex = 0  # Change this index to select a different GPU (e.g., 0, 1, 2, ...).
if (gpus):
  try:
    # Restrict TensorFlow to only use the first GPU.
    tf.config.set_visible_devices(gpus[gpuIndex], "GPU")
    print("Using GPU:", gpus[gpuIndex])
  except RuntimeError as e:
    print("Error setting visible devices:", e)


# ============================================================================================ #


def HyperparameterToString(key, value):
  if (key.lower() == "optimizer"):
    optimizer = BuildOptimizer(value)
    config = optimizer.get_config()
    return f"{optimizer.__class__.__name__}({config})"
  return str(value)


def LoadData(imagesList, masksList, testSize=0.2):
  # Split the dataset into training and testing sets.
  xTrain, xTest, yTrain, yTest = train_test_split(
    imagesList,  # Images.
    masksList,  # Masks.
    test_size=testSize,  # Ratio of the testing set.
    random_state=42  # Random state.
  )

  # Split the training set into training and validation sets.
  xTrain, xVal, yTrain, yVal = train_test_split(
    xTrain,  # Images.
    yTrain,  # Masks.
    test_size=testSize,  # Ratio of the validation set.
    random_state=42  # Random state.
  )

  # Return the dataset.
  return xTrain, xVal, xTest, yTrain, yVal, yTest


def TFParse(X, y):
  # Read the image and mask.
  image = tf.numpy_function(ReadImage, [X], tf.float64)
  mask = tf.numpy_function(ReadMask, [y], tf.float64)

  # Set the shape of the images and masks.
  image.set_shape([image.shape[0], image.shape[1], 3])
  mask.set_shape([mask.shape[0], mask.shape[1], 1])

  # Return the image and mask.
  return image, mask


def TFDataset(X, y, batchSize=8):
  # Create a TensorFlow dataset.
  dataset = tf.data.Dataset.from_tensor_slices((X, y))

  # Shuffle the dataset.
  dataset = dataset.shuffle(buffer_size=batchSize)

  # Parse images and masks.
  dataset = dataset.map(TFParse)

  # Batch the dataset.
  dataset = dataset.batch(batchSize)

  # Prefetch the dataset to optimize training.
  dataset = dataset.prefetch(1)

  # Repeat the dataset indefinitely.
  dataset = dataset.repeat()

  # Return the dataset.
  return dataset


if __name__ == "__main__":
  DoRandomSeeding()
  UpdateMatplotlibSettings()

  pretrainedWeights = None  # Set the pretrained weights.
  inputSize = (256, 256, 3)  # Set the input size.
  epochs = 1  # Set the number of epochs.
  trials = 3  # Set the number of trials.
  metrics = ["accuracy"]  # Create the metrics.
  numClasses = 2  # Set the number of classes.
  testSize = 0.25  # Set the ratio of the testing set.
  whichModel = "VNet"  # Set the model to train.

  # Set the base directory for storing the results.
  storageBaseDir = os.getcwd()

  # For PH2 Dataset.
  datasetBase = r"F:\[A] Skin Cancer ViT HAM10K PH2\Datasets"
  datasetName = r"PH2"  # Set the path to the dataset directory.
  imagesPath = os.path.join(datasetBase, datasetName, "Images")
  masksPath = os.path.join(datasetBase, datasetName, "Lesion")
  # For Skin cancer HAM10000 Dataset.
  # datasetName = r"HAM10000"  # Set the path to the dataset directory.
  # imagesPath = os.path.join(datasetBase, datasetName, "images")
  # masksPath = os.path.join(datasetBase, datasetName, "masks")

  # Get the list of the images.
  imagesList = sorted(os.listdir(imagesPath))

  # Get the list of the masks.
  # For PH2 Dataset.
  masksList = [
    os.path.join(masksPath, image.replace(".", "_lesion."))
    for image in imagesList
    if (image.endswith(tuple(IMAGE_SUFFIXES)))
  ]
  # For Skin cancer HAM10000 Dataset.
  # masksList = [
  #   os.path.join(masksPath, image.replace(".jpg", "_segmentation.png"))
  #   for image in imagesList
  #   if (image.endswith(tuple(IMAGE_SUFFIXES)))
  # ]

  imagesList = [
    os.path.join(imagesPath, image)
    for image in imagesList
    if (image.endswith(tuple(IMAGE_SUFFIXES)))
  ]

  assert len(imagesList) == len(masksList), "The number of images and masks must be the same."

  for i in range(len(imagesList) - 1, -1, -1):
    if (not os.path.exists(imagesList[i]) or not os.path.exists(masksList[i])):
      imagesList.pop(i)
      masksList.pop(i)

  print("Number of images:", len(imagesList))
  print("Number of masks:", len(masksList))
  print("First image:", imagesList[0])
  print("First mask:", masksList[0])

  baseHyperparametersRanges = {
    "batchSize": [1, 3, 5, 7, 10],
    "optimizer": [
      (Adam, {"learning_rate": 5e-3, "beta_1": 0.5}),
      (Nadam, {"learning_rate": 5e-3, "beta_1": 0.5}),
      (RMSprop, {"learning_rate": 5e-3, "momentum": 0.5}),
      (SGD, {"learning_rate": 5e-3, "momentum": 0.5}),
    ],
    "loss"     : [
      "binary_crossentropy",
      TverskyLoss,
      DiceLoss,
      JaccardLoss,
      FocalLoss,
    ],
  }

  vnetHyperparametersRanges = {
    **baseHyperparametersRanges,
    "dropoutRatio"     : [0.0, 0.1, 0.25, 0.5],
    "dropoutType"      : ["spatial", "feature"],
    "applyBatchNorm"   : [True, False],
    "concatenateType"  : ["concatenate", "attention"],
    "noOfLevels"       : [
      3, 4, 5, 6, 7,
    ],
    "kernelInitializer": [
      "he_normal",
      "he_uniform",
      "glorot_normal",
      "glorot_uniform",
      "lecun_normal",
      "lecun_uniform",
      "random_normal",
      "random_uniform",
      "truncated_normal",
      "variance_scaling",
      "zeros",
    ],
  }

  segnetHyperparametersRanges = {
    **baseHyperparametersRanges,
    "level"  : [3, 4],
    "encoder": ["VGG16", "ResNet50", "MobileNet", "Vanilla"],
  }

  # Loop over the trials.
  for i in range(trials):
    try:
      print("Trial:", i + 1)
      outputFolder = os.path.join(
        storageBaseDir,
        f"{datasetName} ({whichModel} Training Example)",
      )
      os.makedirs(outputFolder, exist_ok=True)

      # Set the hyperparameters randomly.
      hyperparameters = {}
      for key in baseHyperparametersRanges.keys():
        hyperparameters[key] = random.choice(baseHyperparametersRanges[key])
      if (whichModel == "VNet"):
        for key in vnetHyperparametersRanges.keys():
          if (key not in baseHyperparametersRanges):
            hyperparameters[key] = random.choice(vnetHyperparametersRanges[key])
      elif (whichModel == "SegNet"):
        for key in segnetHyperparametersRanges.keys():
          if (key not in baseHyperparametersRanges):
            hyperparameters[key] = random.choice(segnetHyperparametersRanges[key])
      print("Working with:", hyperparameters)

      if (whichModel == "VNet"):
        model = VNet(
          inputSize=inputSize,  # Input size.
          kernelInitializer=hyperparameters["kernelInitializer"],  # Kernel initializer.
          dropoutRatio=hyperparameters["dropoutRatio"],  # Dropout rate.
          dropoutType=hyperparameters["dropoutType"],  # Dropout type.
          applyBatchNorm=hyperparameters["applyBatchNorm"],  # Apply batch normalization.
          concatenateType=hyperparameters["concatenateType"],  # Concatenate type.
          noOfLevels=hyperparameters["noOfLevels"],  # Number of levels.
        )
      elif (whichModel == "SegNet"):
        model = SegNet(
          inputSize=inputSize,  # Input size.
          numClasses=numClasses,  # Number of classes.
          level=hyperparameters["level"],  # Level.
          encoder=hyperparameters["encoder"],  # Encoder.
        )
      else:
        raise ValueError("Invalid model name.")

      # Compile the model with the hyperparameters.
      CompileTrainTFUNetModel(
        model,
        trialNo=i + 1,
        inputSize=inputSize,
        imagesList=imagesList,
        masksList=masksList,
        hyperparameters=hyperparameters,
        pretrainedWeights=pretrainedWeights,
        epochs=epochs,
        outputFolder=outputFolder,
        keyword=f"Trial {i + 1}",
        testSize=testSize,
      )
    except Exception as e:
      print("Error:", e)
      continue
