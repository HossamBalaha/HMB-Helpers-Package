# Import the operating system interface.
import os
# Import the pickle serialization module.
import pickle
# Import the garbage collection module.
import gc
# Import the PyTorch package.
import torch
# Import the pandas package.
import pandas
# Import the tqdm progress bar with a CamelCase alias.
from tqdm import tqdm as Tqdm
# Import the embedding model helper from the local package.
from HMB.ImagesToEmbeddings import TransformersEmbeddingModel


# Define a function that builds a Phikon V2 embedding lookup table.
def CreateLookupTablePhikonV2(datasetFolder, storageKeyword, modelName="owkin/phikon-v2", device=None):
  # Verify that the dataset folder exists.
  if (not os.path.isdir(datasetFolder)):
    # Raise an error for the missing dataset folder.
    raise ValueError(f"Dataset folder not found: {datasetFolder}")

  # Default to the CPU device.
  if (device is None):
    # Assign the CPU device.
    device = "cpu"
    # Check whether CUDA is available.
    if (torch.cuda.is_available()):
      # Assign the CUDA device.
      device = "cuda"

  # Build the pickle output path.
  outputPicklePath = f"{storageKeyword}.p"
  # Build the CSV output path.
  outputCsvPath = f"{storageKeyword}.csv"
  # Determine the output directory from the pickle path.
  outputDirectory = os.path.dirname(outputPicklePath)
  # Check whether an output directory is specified and missing.
  if (outputDirectory and not os.path.isdir(outputDirectory)):
    # Create the missing output directory.
    os.makedirs(outputDirectory, exist_ok=True)

  # Create the embedding model helper.
  embeddingModel = TransformersEmbeddingModel(modelName, device)
  # Load the model and processor before processing images.
  embeddingModel.LoadModel()

  # Initialize the lookup table.
  lookupTablePhikonV2 = {}
  # Initialize the tabular row list.
  lookupDictList = []
  # Define the accepted image file extensions.
  imageExtensionTuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
  # List the class folders in the dataset folder.
  classList = sorted(os.listdir(datasetFolder))

  # Iterate over each class folder.
  for className in Tqdm(classList, desc="Processing Classes"):
    # Build the class folder path.
    classPath = os.path.join(datasetFolder, className)

    # Skip entries that are not directories.
    if (not os.path.isdir(classPath)):
      # Continue to the next class entry.
      continue

    # List the image files in the class folder.
    imageNameList = sorted(os.listdir(classPath))

    # Iterate over each image file.
    for imageName in Tqdm(imageNameList, desc=f"Class: {className}", leave=False):
      # Build the image path.
      imagePath = os.path.join(classPath, imageName)

      # Skip entries that are not files.
      if (not os.path.isfile(imagePath)):
        # Continue to the next image entry.
        continue

      # Skip files without an accepted image extension.
      if (not imagePath.lower().endswith(imageExtensionTuple)):
        # Continue to the next image entry.
        continue

      # Skip empty files.
      if (os.path.getsize(imagePath) == 0):
        # Continue to the next image entry.
        continue

      # Extract the embedding for the image.
      embedding = embeddingModel.GetEmbedding(imagePath)
      # Flatten the embedding into a one dimensional vector.
      embeddingVector = embedding.reshape(-1)
      # Reshape the embedding vector into a single row matrix.
      embeddingMatrix = embeddingVector.reshape(1, -1)

      # Store the embedding matrix in the lookup table.
      lookupTablePhikonV2[f"{className}_{imageName}"] = embeddingMatrix

      # Initialize the tabular row with the file name and class label.
      rowDict = {"Filename": imageName, "Class": className}

      # Iterate over each embedding dimension.
      for columnIndex in range(embeddingVector.shape[0]):
        # Build the embedding column name.
        columnName = f"EmbeddingPhikonV2_{columnIndex}"
        # Store the embedding value in the row.
        rowDict[columnName] = embeddingVector[columnIndex]

      # Append the row to the tabular row list.
      lookupDictList.append(rowDict)

  # Open the pickle output file for binary writing.
  with open(outputPicklePath, "wb") as outputFile:
    # Write the lookup table to the pickle file.
    pickle.dump(lookupTablePhikonV2, outputFile)

  # Create a DataFrame from the tabular row list.
  dataFrame = pandas.DataFrame(lookupDictList)
  # Save the DataFrame to the CSV output path.
  dataFrame.to_csv(outputCsvPath, index=False)

  # Delete the embedding model helper.
  del embeddingModel
  # Delete the lookup table.
  del lookupTablePhikonV2
  # Delete the tabular row list.
  del lookupDictList
  # Delete the DataFrame.
  del dataFrame
  # Collect unused memory.
  gc.collect()
  # Check whether CUDA is available.
  if (torch.cuda.is_available()):
    # Empty the CUDA cache.
    torch.cuda.empty_cache()


# Check whether the script is executed directly.
if (__name__ == "__main__"):
  # Set the dataset keyword.
  keyword = "BreakHis"
  # Set the dataset folder path.
  datasetFolder = r"path/to/BreakHis/dataset/images"
  # Build the storage keyword.
  storageKeyword = f"{keyword}PhikonV2"
  # Run the lookup table creation function.
  CreateLookupTablePhikonV2(datasetFolder, storageKeyword)
