import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from HMB.TFHelper import TrainPretrainedAttentionModelFromDataFrame, EvaluatePretrainedAttentionModelFromDataFrame

if (__name__ == "__main__"):
  # Define the dataset base paths using camelCase naming for clarity.
  basePaths = {
    "train": r"/path/to/the/project/Dataset/Training",
    "test" : r"/path/to/the/project/Dataset/Test",
    "val"  : r"/path/to/the/project/Dataset/Validation"
  }

  # Base directory for storing experiment results.
  baseStorageDir = r"/path/to/the/project/Experiments"
  # Ensure the base storage directory exists.
  if (not os.path.exists(baseStorageDir)):
    os.makedirs(baseStorageDir)

  # Define the class categories as a list.
  categories = ["grade0", "grade1", "grade2", "grade3", "grade4"]

  # Initialize list for image paths.
  imagePaths = []
  # Initialize list for labels.
  labels = []
  # Initialize list for split names.
  splits = []

  # Walk the dataset folders and collect all image paths with labels and split info.
  for splitName, basePath in basePaths.items():
    for category in categories:
      # Build category directory path.
      categoryPath = os.path.join(basePath, category)
      # Check whether the category path exists before listing files.
      if (os.path.exists(categoryPath)):
        for imageName in os.listdir(categoryPath):
          # Build full image path.
          imagePath = os.path.join(categoryPath, imageName)
          # Append image path to list.
          imagePaths.append(imagePath)
          # Append label to list.
          labels.append(category)
          # Append split (train/val/test) to list.
          splits.append(splitName)

  # Create a DataFrame containing image paths, labels and split information.
  dataFrame = pd.DataFrame({
    "image_path": imagePaths,
    "label"     : labels,
    "split"     : splits
  })

  # Encode string labels to integer classes using LabelEncoder.
  # Create a label encoder instance.
  labelEncoder = LabelEncoder()
  # Encode labels.
  dataFrame["category_encoded"] = labelEncoder.fit_transform(dataFrame["label"])
  # Convert labels to string.
  dataFrame["category_encoded"] = dataFrame["category_encoded"].astype(str)

  # Training parameters and image shapes.
  initialEpochs = 25  # Number of initial epochs.
  fineTuneEpochs = 25  # Number of fine-tune epochs.
  batchSize = 32  # Set batch size.
  channels = 3  # Number of channels.
  imgSize = (512, 512)  # Define target image size.
  imgShape = (imgSize[0], imgSize[1], channels)  # Define input shape.
  columnsMap = {"imagePath": "image_path", "categoryEncoded": "category_encoded", "split": "split"}

  # Define experiment directory.
  baseModelString = "Xception"
  attentionBlockStr = "CBAM"
  for trial in range(1, 11):
    print(f"Starting trial {trial} with model {baseModelString} and attention block {attentionBlockStr}...")
    expDir = os.path.join(baseStorageDir, f"{baseModelString}_{attentionBlockStr}_T{trial}")
    os.makedirs(expDir, exist_ok=True)
    modelPath = os.path.join(expDir, "BestModel.keras")

    TrainPretrainedAttentionModelFromDataFrame(
      dataFrame,
      columnsMap=columnsMap,
      labelEncoder=labelEncoder,
      imgShape=imgShape,
      batchSize=batchSize,
      baseModelString=baseModelString,
      attentionBlockStr=attentionBlockStr,
      initialEpochs=initialEpochs,
      fineTuneEpochs=fineTuneEpochs,
      augmentationConfigs=None,
      monitor="val_loss",
      earlyStoppingPatience=10,
      ensureCUDA=True,
      storageDir=expDir,
      dpi=720,
      verbose=2,
    )

    EvaluatePretrainedAttentionModelFromDataFrame(
      dataFrame,
      modelPath,
      columnsMap=columnsMap,
      labelEncoder=labelEncoder,
      imgShape=imgShape,
      batchSize=batchSize,
      storageDir=expDir,
      dpi=720,
      verbose=2,
      ensureCUDA=True,
    )
