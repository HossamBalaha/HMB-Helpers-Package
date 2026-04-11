import os, argparse, math, tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from HMB.TFHelper import BuildPretrainedAttentionModel, FindGlobalPoolingLayer
from HMB.ExplainabilityHelper import (
  CAMExplainerTensorFlow,
  TSNEFeaturesExplainability,
  UMAPFeaturesExplainability
)
from HMB.Initializations import UpdateMatplotlibSettings


def PrepareDataFrame(basePaths, categories):
  r'''
  Build the dataset DataFrame so label -> encoded mapping is consistent with training runs.

  Parameters:
    basePaths (dict): dictionary with keys "train","val","test" -> path strings
    categories (list): List of category folder names

  Returns:
    dataFrame (pandas.DataFrame): DataFrame with columns ["image_path","label","split","category_encoded"].
    labelEncoder (sklearn.preprocessing.LabelEncoder): Fitted label encoder.
  '''

  # Collect image paths, labels and split names.
  imagePaths = []
  labels = []
  splits = []

  # Walk each split folder and each category to collect files.
  for splitName, splitPath in basePaths.items():
    # Skip non-existing split folders.
    if (not os.path.exists(splitPath)):
      continue
    for category in categories:
      # Compute category folder path.
      categoryPath = os.path.join(splitPath, category)
      # Skip missing category folders.
      if (not os.path.exists(categoryPath)):
        continue
      # Iterate files in category folder.
      for imageName in os.listdir(categoryPath):
        # Build full image path.
        imagePath = os.path.join(categoryPath, imageName)
        # Append collected info to lists.
        imagePaths.append(imagePath)
        labels.append(category)
        splits.append(splitName)

  # Build DataFrame from collected lists.
  dataFrame = pd.DataFrame({
    "image_path": imagePaths,
    "label"     : labels,
    "split"     : splits
  })

  # Encode string labels to integer classes using LabelEncoder.
  # Create a label encoder instance.
  from sklearn.preprocessing import LabelEncoder
  labelEncoder = LabelEncoder()
  # Encode labels.
  dataFrame["category_encoded"] = labelEncoder.fit_transform(dataFrame["label"])
  # Convert labels to string.
  dataFrame["category_encoded"] = dataFrame["category_encoded"].astype(str)

  return dataFrame, labelEncoder


def CreateTestGenerator(
  dataFrame,
  imgSize,
  batchSize,
):
  r'''
  Create Keras ImageDataGenerator + flow_from_dataframe for the test split.
  shuffle=False is important to preserve image order for mapping predictions to paths.

  Parameters:
    dataFrame (pandas.DataFrame): full dataset DataFrame with split column
    imgSize (tuple): (height, width)
    batchSize (int): batch size for prediction

  Returns:
    testGen (DirectoryIterator): Keras generator for test images
  '''

  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  # Create ImageDataGenerator with rescaling only.
  testGenFactory = ImageDataGenerator(rescale=1.0 / 255)

  # Subset DataFrame for test split.
  testDf = dataFrame[dataFrame["split"] == "test"].copy()

  # Create generator from DataFrame.
  testGen = testGenFactory.flow_from_dataframe(
    dataframe=testDf,
    x_col="image_path",
    y_col="category_encoded",
    target_size=imgSize,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=False,
    batch_size=batchSize
  )

  return testGen, testDf


def Explain(args):
  # Prepare dataset DataFrame & generator
  # Determine category folders (grade0..grade4) from Training/Validation/Test if possible.
  trainFolder = os.path.join(args.dataRoot, "Training")
  testFolder = os.path.join(args.dataRoot, "Test")
  valFolder = os.path.join(args.dataRoot, "Validation")

  assert os.path.exists(trainFolder)
  assert os.path.exists(testFolder)
  assert os.path.exists(valFolder)

  categories = [
    d
    for d in sorted(os.listdir(trainFolder))
    if (os.path.isdir(os.path.join(trainFolder, d)))
  ]

  basePaths = {
    "train": trainFolder,
    "val"  : valFolder,
    "test" : testFolder,
  }

  dataFrame, labelEncoder = PrepareDataFrame(basePaths, categories)
  imgSize = (args.imgSize, args.imgSize)
  testGen, testDf = CreateTestGenerator(dataFrame, imgSize, args.batchSize)

  baseModelString = str(args.baseModelString)
  attentionBlockStr = str(args.attentionBlockStr)
  inputShape = (args.imgSize, args.imgSize, 3)
  numClasses = len(labelEncoder.classes_)

  print("Base Model String:", baseModelString)
  print("Attention Block String:", attentionBlockStr)
  print("Input Shape:", inputShape)
  print("Num Classes:", numClasses)

  model = BuildPretrainedAttentionModel(
    baseModelString,
    attentionBlockStr,
    inputShape,
    numClasses,
    optimizer=None,
    compile=False,
  )
  model.load_weights(args.model)

  # Find GAP layer and feature extractor.
  gapLayer = FindGlobalPoolingLayer(model)
  if (gapLayer is None):
    raise RuntimeError("No global pooling / GAP layer found in model.")
  featModel = tf.keras.Model(inputs=model.inputs, outputs=gapLayer.output)

  # Extract features for the test set.
  steps = math.ceil(testGen.samples / float(args.batchSize))
  feats = []
  predsAll = []
  paths = []
  testGen.reset()
  for _ in tqdm.tqdm(range(steps), desc="Extracting features and predictions"):
    try:
      x, y = next(testGen)
    except StopIteration:
      break
    # Use Keras predict() to avoid mixing symbolic KerasTensors and eager tensors.
    featsBatch = featModel.predict(x, verbose=0)
    predsBatch = model.predict(x, verbose=0)
    feats.append(featsBatch)
    predsAll.append(predsBatch)
    # Collect paths from generator -- generator stores filenames attribute.
    if (hasattr(testGen, "filenames")):
      # Filenames correspond to entire epoch; compute slice using index.
      pass
    # Fallback: pull filepaths from testDf using generator index.
    # We will build paths at the end using testDf.reset_index.
  feats = np.concatenate(feats, axis=0)
  predsAll = np.concatenate(predsAll, axis=0)
  paths = testDf.reset_index(drop=True)["image_path"].values[: feats.shape[0]]

  # Predicted labels from `predsAll`.
  predLabels = np.argmax(predsAll, axis=1)
  trueLabels = testDf.reset_index(drop=True)["label"].values[: feats.shape[0]]
  trueLabelIdx = testDf.reset_index(drop=True)["category_encoded"].astype(int).values[: feats.shape[0]]

  # Optionally subsample for faster TSNE/UMAP.
  nSamples = min(args.numUMAP, feats.shape[0]) if (args.numUMAP > 0) else feats.shape[0]
  idxs = np.linspace(0, feats.shape[0] - 1, nSamples).astype(int)
  featsSub = feats[idxs]
  trueIdxSub = trueLabelIdx[idxs]
  predIdxSub = predLabels[idxs]

  outDir = Path(args.expDir if args.expDir else os.path.dirname(args.model))
  outDir.mkdir(parents=True, exist_ok=True)

  TSNEFeaturesExplainability(
    featsSub,
    labelEncoder,
    int(nSamples),
    outDir,
    predIdxSub,
    trueIdxSub,
  )

  UMAPFeaturesExplainability(
    featsSub,
    labelEncoder,
    int(nSamples),
    outDir,
    predIdxSub,
    trueIdxSub,
  )

  if (args.useCAM):
    classNames = {}
    try:
      for i, name in enumerate(labelEncoder.classes_):
        classNames[int(i)] = str(name)
    except Exception:
      # Fallback: infer from unique labels in testDf
      uniq = sorted(list(set(testDf["label"].values)))
      for i, name in enumerate(uniq):
        classNames[int(i)] = str(name)

    if (args.camType.lower() == "all"):
      camTypes = [
        "gradcam",
        "gradcampp",
        "xgradcam",
        "eigencam",
        "layercam",
        "scorecam",
        "ablationcam",
        "saliency",
        "smoothgrad",
        "integratedgradients",
        "occlusion",
        "gradxinput",
        "smoothgradcampp",
      ]
    else:
      camTypes = [args.camType]

    for camType in camTypes:
      camExplainer = CAMExplainerTensorFlow(
        tfModel=model,
        device="gpu" if (args.useGPU) else "cpu",
        camType=camType,
        imgSize=args.imgSize,
        outputBase=str(outDir),
        alpha=args.alpha,
        dpi=args.dpi,
        debug=False,
      )

      # Pick numCAM samples (evenly spaced).
      numCAM = min(args.numCAM, feats.shape[0])
      camIdxs = np.linspace(0, feats.shape[0] - 1, numCAM).astype(int)
      for i, idx in enumerate(camIdxs):
        imgPath = paths[int(idx)]
        try:
          result = camExplainer.ProcessImage(
            imgPath,
            classNames=classNames,
          )
        except Exception as ex:
          print(f"Warning: CAM failed for {imgPath}: {ex}")

  print("Saved visualizations to", outDir)


def ParseArgs():
  p = argparse.ArgumentParser()
  p.add_argument(
    "--model", type=str, required=True,
    help="Path to trained model file (.keras/.h5) or name inside expDir"
  )
  p.add_argument(
    "--baseModelString", type=str, required=True,
    help="Base model string (e.g., ResNet50, InceptionV3)"
  )
  p.add_argument(
    "--attentionBlockStr", type=str, required=True,
    help="AttentionBlock string (e.g., ResNet50, InceptionV3)"
  )
  p.add_argument(
    "--dataRoot", type=str, default="Dataset",
    help="Root dataset folder containing Training/Test/Validation"
  )
  p.add_argument("--expDir", type=str, default=".", help="Experiment folder where outputs will be saved")
  p.add_argument("--imgSize", type=int, default=512)
  p.add_argument("--batchSize", type=int, default=16)
  p.add_argument("--numUMAP", type=int, default=1000, help="Number of samples to run UMAP/TSNE on (0 = all)")
  p.add_argument("--useCAM", action="store_true", help="Apply CAM to images")
  p.add_argument("--numCAM", type=int, default=32, help="Number of Grad-CAM images to render")
  p.add_argument(
    "--camType", type=str, default="gradcam",
    help="CAM method to use (gradcam, gradcampp, layercam, etc.)"
  )
  p.add_argument("--useGPU", action="store_true", help="Run CAM computations on GPU if available")
  p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for heatmaps")
  p.add_argument("--dpi", type=int, default=720, help="DPI for figures")
  return p.parse_args()


if (__name__ == "__main__"):
  UpdateMatplotlibSettings()

  args = ParseArgs()
  Explain(args)
