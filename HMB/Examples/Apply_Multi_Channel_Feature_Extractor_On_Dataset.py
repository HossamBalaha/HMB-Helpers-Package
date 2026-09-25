import os  # Import the operating system module.
import tqdm  # Import the tqdm library for progress bars.
import cv2  # Import the OpenCV library for image processing.
import numpy  # Import the numpy library for numerical operations.
import imageio  # Import the imageio library for saving standard images.
from pathlib import Path  # Import the Path class from pathlib.
from HMB.ImagesHelper import MultiChannelFeatureExtractor
from HMB.Utils import fprint  # Import custom print function from HMB utilities.
from HMB.Initializations import IMAGE_SUFFIXES


def ProcessDataset(
  inputDir,
  outputDir,
  splits,
  extensions,
  featureList,
  referenceImage,
  newSize=(512, 512),
):
  r'''
  Process a dataset of images to extract specified features and save them in a structured format.
  It handles multiple dataset splits (e.g., train, val, test) and supports various image formats.

  Parameters:
    inputDir (str): The root directory of the original dataset.
    outputDir (str): The root directory where the modified dataset will be saved.
    splits (list): A list of dataset splits to process (e.g., ["train", "val", "test"]).
    extensions (list): A list of image file extensions to consider (e.g., [".png", ".jpg", ".jpeg"]).
    featureList (list): A list of features to extract from each image (e.g., ["Hematoxylin", "Clustering", "B"]).
    referenceImage (str): Path to the reference image for feature extraction.
    newSize (tuple): A tuple specifying the new size (width, height) to resize images to. Default is (512, 512).
  '''

  # Instantiate the feature extractor.
  extractor = MultiChannelFeatureExtractor()

  # Fit the clustering model if "Clustering" is in the feature list.
  if ("Clustering" in featureList):
    fprint("Fitting clustering model...")
    extractor.FitClusteringModel(referenceImage)
    fprint("Clustering model fitted successfully.")

  # Convert paths to Path objects for easier manipulation.
  inputRoot = Path(inputDir)
  # Convert the output directory path to a Path object.
  outputRoot = Path(outputDir)
  layersOutputRoot = Path(str(outputRoot) + "_Layers")
  # Iterate through each split.
  for split in splits:
    # Construct the input path for the current split.
    splitInputPath = inputRoot / split
    # Construct the output path for the current split.
    splitOutputPath = outputRoot / split
    # Construct the layers output path for the current split.
    splitLayersOutputPath = layersOutputRoot / split
    # Create the layers output directory for the current split if it does not exist.
    splitLayersOutputPath.mkdir(parents=True, exist_ok=True)
    # Check if the split directory exists.
    if (not splitInputPath.exists()):
      # Print a warning if the split is missing.
      fprint(f"Warning: Split directory not found: {splitInputPath}")
      # Continue to the next split.
      continue
    # Print the current split being processed.
    fprint(f"Processing Split: {split}")
    # Get all class directories within the current split.
    classDirs = sorted([d for d in splitInputPath.iterdir() if (d.is_dir())])
    # Iterate through each class directory.
    for classDir in classDirs:
      # Extract the class name from the directory.
      className = classDir.name
      # Construct the output path for the current class.
      classOutputPath = splitOutputPath / className
      # Create the output class directory if it does not exist.
      classOutputPath.mkdir(parents=True, exist_ok=True)
      # Construct the layers output path for the current class.
      classLayersOutputPath = splitLayersOutputPath / className
      # Create the layers output class directory if it does not exist.
      classLayersOutputPath.mkdir(parents=True, exist_ok=True)
      # Print the current class being processed.
      fprint(f"  Processing Class: {className}")
      # Initialize a counter for processed images in this class.
      processedCount = 0
      # Iterate through each image extension to find all images.
      for ext in extensions:
        tBar = tqdm.tqdm(classDir.glob(f"*{ext}"), desc=f"    Processing {className} ({ext})", unit="image")
        # Find all files with the current extension.
        for imgPath in tBar:
          # Attempt to process the current image.
          try:
            # Load the image from the path.
            loadedImage = extractor.LoadImageFromPath(str(imgPath))
            # Ensure that the number of channels in the loaded image is 3 (RGB).
            if (loadedImage.ndim != 3 or loadedImage.shape[-1] != 3):
              # Convert the image to RGB if it is grayscale or has a different number of channels.
              if (loadedImage.ndim == 2):
                # Convert grayscale to RGB by stacking the single channel.
                loadedImage = cv2.cvtColor(loadedImage, cv2.COLOR_GRAY2RGB)
              elif (loadedImage.ndim == 3 and loadedImage.shape[-1] == 1):
                # Convert single-channel to RGB by stacking the single channel.
                loadedImage = cv2.cvtColor(loadedImage, cv2.COLOR_GRAY2RGB)
              elif (loadedImage.ndim == 3 and loadedImage.shape[-1] == 4):
                # Convert RGBA to RGB by removing the alpha channel.
                loadedImage = cv2.cvtColor(loadedImage, cv2.COLOR_RGBA2RGB)
            if (newSize is not None):
              # Resize the image to the specified new size.
              loadedImage = cv2.resize(loadedImage, newSize, interpolation=cv2.INTER_CUBIC)
            # Generate the custom multi-channel feature image.
            featureImage = extractor.GenerateCustomFeatureImage(loadedImage, featureList)
            # Determine the number of channels in the feature image.
            numChannels = featureImage.shape[-1] if (featureImage.ndim == 3) else 1
            # Check if the image has exactly 3 channels.
            if (numChannels == 3):
              # Extract the original file extension in lowercase.
              originalExt = imgPath.suffix.lower()
              # Check if the original extension is a standard format.
              if (originalExt in IMAGE_SUFFIXES):
                # Use the original extension for the output file.
                outExt = originalExt
              else:
                # Default to PNG for lossless quality if the original is not standard.
                outExt = ".png"
              # Construct the output filename with the determined extension.
              outFilename = f"{imgPath.stem}{outExt}"
              # Construct the full output file path.
              outFilePath = classOutputPath / outFilename
              # Check if the output file already exists to avoid reprocessing.
              if (outFilePath.exists()):
                # Skip to the next image if it already exists.
                continue
              # Convert the feature image to uint8 if it is in float format.
              if (featureImage.dtype != numpy.uint8):
                # Scale the float image to the 0-255 range and convert to uint8.
                imageToSave = (featureImage * 255).astype(numpy.uint8)
              else:
                # Use the image directly if it is already uint8.
                imageToSave = featureImage
              # Save the 3-channel image using imageio.
              imageio.imwrite(str(outFilePath), imageToSave)
            else:
              # Construct the output filename and changing extension to tiff.
              outFilename = f"{imgPath.stem}.tiff"
              # Construct the full output file path.
              outFilePath = classOutputPath / outFilename
              # Check if the output file already exists to avoid reprocessing.
              if (outFilePath.exists()):
                # Skip to the next image if it already exists.
                continue
              # Save the multi-channel image as a TIFF file.
              extractor.SaveMultiChannelImage(featureImage, str(outFilePath))
            # Save each feature channel as a separate grayscale image.
            for idx, featureName in enumerate(featureList):
              # Construct the output filename for the current feature channel.
              featureOutFilename = f"{imgPath.stem}_{featureName}.png"
              # Construct the full output file path for the feature channel.
              featureOutFilePath = classLayersOutputPath / featureOutFilename
              # Check if the feature output file already exists to avoid reprocessing.
              if (featureOutFilePath.exists()):
                # Skip to the next feature if it already exists.
                tBar.set_postfix({"Status": "Skipped"})
                tBar.update(1)
                continue
              # Extract the specific channel from the feature image.
              channelImage = featureImage[..., idx] if (featureImage.ndim == 3) else featureImage
              # Convert the channel image to uint8 if it is in float format.
              if (channelImage.dtype != numpy.uint8):
                # Scale the float image to the 0-255 range and convert to uint8.
                channelToSave = (channelImage * 255).astype(numpy.uint8)
              else:
                # Use the channel image directly if it is already uint8.
                channelToSave = channelImage
              # Save the individual feature channel as a grayscale PNG image.
              imageio.imwrite(str(featureOutFilePath), channelToSave)
            # Increment the processed counter.
            processedCount += 1
          # Catch any exceptions that occur during processing.
          except Exception as e:
            # Print an error message if processing fails for an image.
            tBar.set_postfix({"Status": "Error"})
            tBar.update(1)
            fprint(f"    Error processing {imgPath.name}: {e}")
            import traceback
            traceback.print_exc()
      # Print the summary for the current class.
      fprint(f"    Saved {processedCount} images to {classOutputPath}")
  # Print a completion message.
  fprint("Dataset processing complete!")


# Check if the script is being run as the main module.
if (__name__ == "__main__"):
  # Define the root directory of your original dataset.
  inputDatasetDir = r"/path/to/your/original/dataset"

  # Define the root directory where the modified dataset will be saved.
  outputDatasetDir = r"/path/to/save/modified/dataset"

  # Define the path to the reference image for feature extraction.
  referenceImage = r"/path/to/your/reference/image.png"

  # Define the splits to process.
  datasetSplits = ["train", "test", "val"]

  # You can pick up from the following list of features to extract.
  # [
  #   "Hog",
  #   "Clustering",
  #   "Edge",
  #   "Texture",
  #   "Stain",
  #   "Hematoxylin",
  #   "Gabor",
  #   "Canny",
  #   "Entropy",
  #   "DoG",
  #   "MultiGabor",
  #   "Eosin",
  #   "Laplacian",
  #   "Frangi",
  #   "Sato",
  #   "LocalVariance",
  #   "Hue",
  #   "Saturation",
  #   "Lightness",
  #   "AChannel",
  #   "BChannel"
  # ]

  # Define the feature list to extract.
  featureList = ["DoG", "AChannel", "Lightness"]

  # Execute the dataset processing function with the configured parameters.
  ProcessDataset(
    inputDir=inputDatasetDir,
    outputDir=outputDatasetDir,
    splits=datasetSplits,
    extensions=IMAGE_SUFFIXES,
    featureList=featureList,
    referenceImage=referenceImage,
  )
