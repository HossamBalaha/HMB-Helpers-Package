import os, cv2, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Dict, List, Any


def PerformDataAugmentation(
  imagePath: str,
  config: Dict[str, Any],
  numResultantImages: int,
  auxImagesList: List[str] = None,
) -> List[Image.Image]:
  r'''
  Performs data augmentation on an image by randomly applying one augmentation technique per generated image.

  This function loads an image from the specified path and generates multiple augmented versions by randomly
  selecting and applying one augmentation technique to each generated image. The augmentation techniques are
  configured through a dictionary where each technique can be enabled or disabled.

  Parameters:
    imagePath (str): Path to the input image file.
    config (Dict[str, Any]): Dictionary containing augmentation configurations with "enabled" flags and parameter ranges.
      Each augmentation type should have an "enabled" key (bool) and relevant parameters for that technique.
    numResultantImages (int): Number of augmented images to generate.
    auxImagesList (List[str]): Optional list of auxiliary image paths for advanced augmentations like mixup, cutmix, mosaic.
      Default is None.

  Returns:
    List[Image.Image]: List of augmented PIL Image objects.

  Examples
  --------
  .. code-block:: python

    configs = {
      "rotation"            : {"enabled": True, "range": (-30, 30)},
      "flip"                : {"enabled": True, "horizontal": True, "vertical": False},
      "brightness"          : {"enabled": True, "range": (0.7, 1.3)},
      "contrast"            : {"enabled": True, "range": (0.8, 1.2)},
      "saturation"          : {"enabled": True, "range": (0.8, 1.2)},
      "blur"                : {"enabled": True, "range": (0, 2)},
      "sharpness"           : {"enabled": True, "range": (0.5, 2.0)},
      "zoom"                : {"enabled": True, "range": (0.8, 1.2)},
      "translation"         : {"enabled": True, "range": (-20, 20)},
      "noise"               : {"enabled": True, "range": (0, 25)},
      "cutout"              : {"enabled": True, "numHoles": (1, 3), "holeSize": (0.1, 0.2)},
      "hideAndSeek"         : {"enabled": True, "gridSize": (4, 8), "hideProb": 0.5},
      "gridmask"            : {"enabled": True, "dRange": (0.3, 0.5), "rRange": (0.4, 0.7)},
      "randomErasing"       : {"enabled": True, "probability": 0.5, "area": (0.02, 0.2), "aspectRatio": (0.3, 3.3)},
      "colorJitter"         : {
        "enabled": True, "hueShift": (-0.05, 0.05), "saturationShift": (-0.1, 0.1), "valueShift": (-0.1, 0.1)
      },
      "elasticDeformation"  : {"enabled": True, "alpha": (30, 40), "sigma": (4, 6)},
      "perspectiveTransform": {"enabled": True, "scale": (0.0, 0.15)},
      "affineTransform"     : {"enabled": True, "scale": (0.9, 1.1), "shear": (-10, 10), "rotate": (-20, 20)},
      "clahe"               : {"enabled": True, "clipLimit": (2.0, 4.0), "tileGridSize": (8, 8)},
      "speckleNoise"        : {"enabled": True, "intensity": (0.0, 0.15)},
      "saltPepperNoise"     : {"enabled": True, "amount": (0.01, 0.03), "saltVsPepper": 0.5},
      "poissonNoise"        : {"enabled": True, "scale": (1.0, 5.0)},
      "motionBlur"          : {"enabled": True, "kernelSize": (5, 11), "angle": (0, 360)},
      "medianBlur"          : {"enabled": True, "kernelSize": (3, 5)},
      "bilateralFilter"     : {"enabled": True, "d": (5, 9), "sigmaColor": (75, 125), "sigmaSpace": (75, 125)},
      "channelShuffle"      : {"enabled": True},
      "invert"              : {"enabled": True},
      "solarize"            : {"enabled": True, "threshold": (128, 200)},
      "posterize"           : {"enabled": True, "bits": (3, 5)},
      "equalize"            : {"enabled": True},
      "emboss"              : {"enabled": True},
      "edgeEnhance"         : {"enabled": True, "factor": (0.3, 0.6)},
      "coarseDropout"       : {"enabled": True, "numHoles": (3, 6), "holeSize": (0.05, 0.12), "fillValue": (0, 0, 0)},
      "mixup"               : {"enabled": True, "alpha": 0.4, "auxImageDir": classPath},
      "cutmix"              : {"enabled": True, "alpha": 1.0, "auxImageDir": classPath},
      "mosaic"              : {"enabled": True, "auxImageDir": classPath, "numImages": 4}
    }

    # Performing data augmentation to generate 5 augmented images.
    augmentedImages = PerformDataAugmentation(
      imagePath="path/to/image.jpg",
      config=configs,
      numResultantImages=5,
      auxImagesList=None
    )

    # Saving the augmented images to a specified folder.
    outputFolder = "path/to/augmented_images"
    os.makedirs(outputFolder, exist_ok=True)
    SaveAugmentedImages(
      augmentedImages,
      outputFolder,
      baseFilename="augmented_image",
    )
  '''

  # Loading the original image from the provided path.
  originalImage = Image.open(imagePath).convert("RGB")

  # Creating a list to store augmented images.
  augmentedImages = []

  # Collecting all enabled augmentation techniques from the configuration.
  enabledAugmentations = []
  for augmentationType, augmentationConfig in config.items():
    if (augmentationConfig.get("enabled", False)):
      enabledAugmentations.append(augmentationType)

  # Checking if there are any enabled augmentation techniques.
  if (len(enabledAugmentations) == 0):
    print("Warning: No augmentation techniques are enabled in the configuration.")
    return [originalImage.copy() for _ in range(numResultantImages)]

  # Generating the specified number of augmented images.
  for i in range(numResultantImages):
    # Selecting a random augmentation technique from the enabled options.
    selectedAugmentation = random.choice(enabledAugmentations)

    # Applying the selected augmentation technique to create a new image.
    augmentedImage = ApplyAugmentation(
      originalImage,  # Original image to augment.
      selectedAugmentation,  # Selected augmentation type.
      config[selectedAugmentation],  # Parameters for the selected augmentation.
      auxImagesList,  # Optional auxiliary images for advanced augmentations.
    )

    # Adding the augmented image to the results list.
    augmentedImages.append(augmentedImage)

  # Returning the list of augmented images.
  return augmentedImages


def ApplyAugmentation(
  image: Image.Image,
  augmentationType: str,
  augmentationParams: Dict[str, Any],
  auxImagesList: List[str] = None,
) -> Image.Image:
  r'''
  Applies a specific augmentation technique to an image.

  This function takes an input image and applies one of the supported augmentation techniques based on the
  specified augmentation type and parameters. It supports a wide range of augmentation techniques including
  geometric transformations, color adjustments, noise addition, and advanced techniques like mixup and cutmix.

  Parameters:
    image (Image.Image): Input PIL Image object to be augmented.
    augmentationType (str): Type of augmentation to apply (e.g., "rotation", "flip", "brightness", etc.).
    augmentationParams (Dict[str, Any]): Parameters for the selected augmentation technique, including ranges
      and specific settings for that augmentation type.
    auxImagesList (List[str]): Optional list of auxiliary image paths for advanced augmentations like mixup,
      cutmix, and mosaic. Default is None.

  Returns:
    Image.Image: Augmented PIL Image object.
  '''

  # Creating a copy of the original image to avoid modifying it.
  augmentedImage = image.copy()

  # Applying rotation augmentation if selected.
  if (augmentationType == "rotation"):
    minAngle, maxAngle = augmentationParams["range"]
    # Generating a random rotation angle within the specified range.
    angle = random.uniform(minAngle, maxAngle)
    # Rotating the image by the selected angle.
    augmentedImage = augmentedImage.rotate(
      angle,  # Rotation angle in degrees.
      resample=Image.BICUBIC,  # Resampling method.
      expand=False,  # Keep original image size.
      fillcolor=(0, 0, 0),  # Fill color for areas outside the rotated image.
    )

  # Applying flip augmentation if selected.
  elif (augmentationType == "flip"):
    # Checking if horizontal flip is enabled.
    if (
      augmentationParams.get("horizontal", True) and
      augmentationParams.get("vertical", False)
    ):
      # Randomly selecting between horizontal and vertical flip.
      flipType = random.choice(["horizontal", "vertical"])
      if (flipType == "horizontal"):
        augmentedImage = augmentedImage.transpose(Image.FLIP_LEFT_RIGHT)
      else:
        augmentedImage = augmentedImage.transpose(Image.FLIP_TOP_BOTTOM)
    elif (augmentationParams.get("horizontal", True)):
      # Applying horizontal flip.
      augmentedImage = augmentedImage.transpose(Image.FLIP_LEFT_RIGHT)
    elif (augmentationParams.get("vertical", False)):
      # Applying vertical flip.
      augmentedImage = augmentedImage.transpose(Image.FLIP_TOP_BOTTOM)

  # Applying brightness augmentation if selected.
  elif (augmentationType == "brightness"):
    minFactor, maxFactor = augmentationParams["range"]
    # Generating a random brightness factor within the specified range.
    brightnessFactor = random.uniform(minFactor, maxFactor)
    # Creating a brightness enhancer object.
    enhancer = ImageEnhance.Brightness(augmentedImage)
    # Adjusting the brightness of the image.
    augmentedImage = enhancer.enhance(brightnessFactor)

  # Applying contrast augmentation if selected.
  elif (augmentationType == "contrast"):
    minFactor, maxFactor = augmentationParams["range"]
    # Generating a random contrast factor within the specified range.
    contrastFactor = random.uniform(minFactor, maxFactor)
    # Creating a contrast enhancer object.
    enhancer = ImageEnhance.Contrast(augmentedImage)
    # Adjusting the contrast of the image.
    augmentedImage = enhancer.enhance(contrastFactor)

  # Applying saturation augmentation if selected.
  elif (augmentationType == "saturation"):
    minFactor, maxFactor = augmentationParams["range"]
    # Generating a random saturation factor within the specified range.
    saturationFactor = random.uniform(minFactor, maxFactor)
    # Creating a color enhancer object.
    enhancer = ImageEnhance.Color(augmentedImage)
    # Adjusting the saturation of the image.
    augmentedImage = enhancer.enhance(saturationFactor)

  # Applying blur augmentation if selected.
  elif (augmentationType == "blur"):
    minRadius, maxRadius = augmentationParams["range"]
    # Generating a random blur radius within the specified range.
    blurRadius = random.uniform(minRadius, maxRadius)
    # Applying Gaussian blur to the image.
    augmentedImage = augmentedImage.filter(ImageFilter.GaussianBlur(radius=blurRadius))

  # Applying sharpness augmentation if selected.
  elif (augmentationType == "sharpness"):
    minFactor, maxFactor = augmentationParams["range"]
    # Generating a random sharpness factor within the specified range.
    sharpnessFactor = random.uniform(minFactor, maxFactor)
    # Creating a sharpness enhancer object.
    enhancer = ImageEnhance.Sharpness(augmentedImage)
    # Adjusting the sharpness of the image.
    augmentedImage = enhancer.enhance(sharpnessFactor)

  # Applying zoom augmentation if selected.
  elif (augmentationType == "zoom"):
    minFactor, maxFactor = augmentationParams["range"]
    # Generating a random zoom factor within the specified range.
    zoomFactor = random.uniform(minFactor, maxFactor)
    # Getting the original image dimensions.
    width, height = augmentedImage.size
    # Calculating the new dimensions after zooming.
    newWidth = int(width * zoomFactor)
    newHeight = int(height * zoomFactor)
    # Resizing the image with the zoom factor.
    augmentedImage = augmentedImage.resize((newWidth, newHeight), Image.BICUBIC)
    # Cropping or padding the image to maintain original dimensions.
    if (zoomFactor > 1.0):
      # Calculating crop coordinates for zoomed-in image.
      left = (newWidth - width) // 2
      top = (newHeight - height) // 2
      right = left + width
      bottom = top + height
      # Cropping the image to original size.
      augmentedImage = augmentedImage.crop((left, top, right, bottom))
    else:
      # Creating a new image with original dimensions and black background.
      paddedImage = Image.new("RGB", (width, height), (0, 0, 0))
      # Calculating paste coordinates for zoomed-out image.
      pasteX = (width - newWidth) // 2
      pasteY = (height - newHeight) // 2
      # Pasting the resized image onto the padded background.
      paddedImage.paste(augmentedImage, (pasteX, pasteY))
      augmentedImage = paddedImage

  # Applying translation augmentation if selected.
  elif (augmentationType == "translation"):
    minShift, maxShift = augmentationParams["range"]
    # Generating random horizontal translation within the specified range.
    shiftX = random.randint(minShift, maxShift)
    # Generating random vertical translation within the specified range.
    shiftY = random.randint(minShift, maxShift)
    # Translating the image by the selected offsets.
    augmentedImage = augmentedImage.transform(
      augmentedImage.size,
      Image.AFFINE,
      (1, 0, -shiftX, 0, 1, -shiftY),
      resample=Image.BICUBIC,
      fillcolor=(0, 0, 0)
    )

  # Applying noise augmentation if selected.
  elif (augmentationType == "noise"):
    minIntensity, maxIntensity = augmentationParams["range"]
    # Generating a random noise intensity within the specified range.
    noiseIntensity = random.uniform(minIntensity, maxIntensity)
    # Converting the image to a numpy array for noise addition.
    imageArray = np.array(augmentedImage).astype(np.float32)
    # Generating random Gaussian noise with the selected intensity.
    noise = np.random.normal(0, noiseIntensity, imageArray.shape)
    # Adding the noise to the image array.
    noisyImageArray = imageArray + noise
    # Clipping the values to valid pixel range.
    noisyImageArray = np.clip(noisyImageArray, 0, 255).astype(np.uint8)
    # Converting the numpy array back to a PIL Image.
    augmentedImage = Image.fromarray(noisyImageArray)

  # Applying cutout augmentation if selected.
  elif (augmentationType == "cutout"):
    minHoles, maxHoles = augmentationParams["numHoles"]
    minSize, maxSize = augmentationParams["holeSize"]
    # Getting image dimensions.
    width, height = augmentedImage.size
    # Converting image to numpy array for manipulation.
    imageArray = np.array(augmentedImage)
    # Generating random number of holes.
    numHoles = random.randint(minHoles, maxHoles)
    # Creating holes in the image.
    for _ in range(numHoles):
      # Calculating hole size as fraction of image dimensions.
      holeSizeFraction = random.uniform(minSize, maxSize)
      holeWidth = int(width * holeSizeFraction)
      holeHeight = int(height * holeSizeFraction)
      # Generating random position for the hole.
      x = random.randint(0, max(0, width - holeWidth))
      y = random.randint(0, max(0, height - holeHeight))
      # Filling the hole with zeros (black).
      imageArray[y:y + holeHeight, x:x + holeWidth, :] = 0
    # Converting the numpy array back to PIL Image.
    augmentedImage = Image.fromarray(imageArray)

  # Applying mixup augmentation if selected.
  elif (augmentationType == "mixup"):
    # Checking if auxiliary images list is provided.
    if (auxImagesList is not None and len(auxImagesList) > 0):
      alpha = augmentationParams["alpha"]
      # Selecting a random auxiliary image.
      auxImagePath = random.choice(auxImagesList)
      # Loading the auxiliary image.
      auxImage = Image.open(auxImagePath).convert("RGB")
      # Resizing auxiliary image to match original image dimensions.
      auxImage = auxImage.resize(augmentedImage.size, Image.BICUBIC)
      # Generating mixup lambda from beta distribution.
      mixupLambda = np.random.beta(alpha, alpha)
      # Converting images to numpy arrays.
      imageArray1 = np.array(augmentedImage).astype(np.float32)
      imageArray2 = np.array(auxImage).astype(np.float32)
      # Mixing the two images.
      mixedArray = mixupLambda * imageArray1 + (1 - mixupLambda) * imageArray2
      # Converting back to uint8 and PIL Image.
      mixedArray = np.clip(mixedArray, 0, 255).astype(np.uint8)
      augmentedImage = Image.fromarray(mixedArray)
    elif (augmentationParams.get("auxImageDir") is not None):
      # Loading images from the auxiliary image directory.
      auxImageDir = augmentationParams["auxImageDir"]
      if (os.path.exists(auxImageDir)):
        auxImages = [
          os.path.join(auxImageDir, f)
          for f in os.listdir(auxImageDir)
          if (f.lower().endswith((".png", ".jpg", ".jpeg")))
        ]
        if (len(auxImages) > 0):
          alpha = augmentationParams["alpha"]
          # Selecting a random auxiliary image.
          auxImagePath = random.choice(auxImages)
          # Loading the auxiliary image.
          auxImage = Image.open(auxImagePath).convert("RGB")
          # Resizing auxiliary image to match original image dimensions.
          auxImage = auxImage.resize(augmentedImage.size, Image.BICUBIC)
          # Generating mixup lambda from beta distribution.
          mixupLambda = np.random.beta(alpha, alpha)
          # Converting images to numpy arrays.
          imageArray1 = np.array(augmentedImage).astype(np.float32)
          imageArray2 = np.array(auxImage).astype(np.float32)
          # Mixing the two images.
          mixedArray = mixupLambda * imageArray1 + (1 - mixupLambda) * imageArray2
          # Converting back to uint8 and PIL Image.
          mixedArray = np.clip(mixedArray, 0, 255).astype(np.uint8)
          augmentedImage = Image.fromarray(mixedArray)

  # Applying hide and seek augmentation if selected.
  elif (augmentationType == "hideAndSeek"):
    minGrid, maxGrid = augmentationParams["gridSize"]
    hideProb = augmentationParams["hideProb"]
    # Generating random grid size.
    gridSize = random.randint(minGrid, maxGrid)
    # Getting image dimensions.
    width, height = augmentedImage.size
    # Calculating cell dimensions.
    cellWidth = width // gridSize
    cellHeight = height // gridSize
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Iterating through grid cells.
    for i in range(gridSize):
      for j in range(gridSize):
        # Randomly deciding whether to hide this cell.
        if (random.random() < hideProb):
          # Calculating cell boundaries.
          x1 = j * cellWidth
          y1 = i * cellHeight
          x2 = min((j + 1) * cellWidth, width)
          y2 = min((i + 1) * cellHeight, height)
          # Hiding the cell by setting it to zero.
          imageArray[y1:y2, x1:x2, :] = 0
    # Converting the numpy array back to PIL Image.
    augmentedImage = Image.fromarray(imageArray)

  # Applying gridmask augmentation if selected.
  elif (augmentationType == "gridmask"):
    dMinRatio, dMaxRatio = augmentationParams["dRange"]
    rMinRatio, rMaxRatio = augmentationParams["rRange"]
    # Getting image dimensions.
    width, height = augmentedImage.size
    # Generating random ratios.
    rRatio = random.uniform(rMinRatio, rMaxRatio)
    dRatio = random.uniform(dMinRatio, dMaxRatio)
    # Calculating grid size.
    gridSize = int(min(width, height) * rRatio)
    # Calculating hole size.
    holeSize = int(gridSize * dRatio)
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Generating random offset.
    offsetX = random.randint(0, gridSize)
    offsetY = random.randint(0, gridSize)
    # Creating grid mask.
    for y in range(-gridSize, height, gridSize):
      for x in range(-gridSize, width, gridSize):
        # Calculating hole position with offset.
        holeX1 = x + offsetX
        holeY1 = y + offsetY
        holeX2 = holeX1 + holeSize
        holeY2 = holeY1 + holeSize
        # Clipping to image boundaries.
        holeX1 = max(0, holeX1)
        holeY1 = max(0, holeY1)
        holeX2 = min(width, holeX2)
        holeY2 = min(height, holeY2)
        # Applying the mask.
        if (holeX1 < holeX2 and holeY1 < holeY2):
          imageArray[holeY1:holeY2, holeX1:holeX2, :] = 0
    # Converting the numpy array back to PIL Image.
    augmentedImage = Image.fromarray(imageArray)

  # Applying random erasing augmentation if selected.
  elif (augmentationType == "randomErasing"):
    probability = augmentationParams["probability"]
    # Checking if erasing should be applied based on probability.
    if (random.random() < probability):
      minArea, maxArea = augmentationParams["area"]
      minAspect, maxAspect = augmentationParams["aspectRatio"]
      # Getting image dimensions.
      width, height = augmentedImage.size
      imageArea = width * height
      # Generating random area.
      erasingArea = random.uniform(minArea, maxArea) * imageArea
      # Generating random aspect ratio.
      aspectRatio = random.uniform(minAspect, maxAspect)
      # Calculating erasing dimensions.
      erasingHeight = int(np.sqrt(erasingArea / aspectRatio))
      erasingWidth = int(aspectRatio * erasingHeight)
      # Checking if dimensions are valid.
      if (erasingWidth < width and erasingHeight < height):
        # Generating random position.
        x = random.randint(0, width - erasingWidth)
        y = random.randint(0, height - erasingHeight)
        # Converting image to numpy array.
        imageArray = np.array(augmentedImage)
        # Generating random color for erasing.
        erasingColor = np.random.randint(0, 256, size=3, dtype=np.uint8)
        # Applying erasing.
        imageArray[y:y + erasingHeight, x:x + erasingWidth, :] = erasingColor
        # Converting the numpy array back to PIL Image.
        augmentedImage = Image.fromarray(imageArray)

  # Applying cutmix augmentation if selected.
  elif (augmentationType == "cutmix"):
    # Checking if auxiliary images list is provided.
    if (auxImagesList is not None and len(auxImagesList) > 0):
      alpha = augmentationParams["alpha"]
      # Selecting a random auxiliary image.
      auxImagePath = random.choice(auxImagesList)
      # Loading the auxiliary image.
      auxImage = Image.open(auxImagePath).convert("RGB")
      # Resizing auxiliary image to match original image dimensions.
      auxImage = auxImage.resize(augmentedImage.size, Image.BICUBIC)
      # Generating cutmix lambda from beta distribution.
      cutmixLambda = np.random.beta(alpha, alpha)
      # Getting image dimensions.
      width, height = augmentedImage.size
      # Calculating bounding box dimensions.
      cutWidth = int(width * np.sqrt(1 - cutmixLambda))
      cutHeight = int(height * np.sqrt(1 - cutmixLambda))
      # Generating random center position.
      cx = random.randint(0, width)
      cy = random.randint(0, height)
      # Calculating bounding box coordinates.
      x1 = np.clip(cx - cutWidth // 2, 0, width)
      y1 = np.clip(cy - cutHeight // 2, 0, height)
      x2 = np.clip(cx + cutWidth // 2, 0, width)
      y2 = np.clip(cy + cutHeight // 2, 0, height)
      # Converting images to numpy arrays.
      imageArray1 = np.array(augmentedImage)
      imageArray2 = np.array(auxImage)
      # Applying cutmix by replacing the region.
      imageArray1[y1:y2, x1:x2, :] = imageArray2[y1:y2, x1:x2, :]
      # Converting the numpy array back to PIL Image.
      augmentedImage = Image.fromarray(imageArray1)
    elif (augmentationParams.get("auxImageDir") is not None):
      # Loading images from the auxiliary image directory.
      auxImageDir = augmentationParams["auxImageDir"]
      if (os.path.exists(auxImageDir)):
        auxImages = [
          os.path.join(auxImageDir, f)
          for f in os.listdir(auxImageDir)
          if (f.lower().endswith((".png", ".jpg", ".jpeg")))
        ]
        if (len(auxImages) > 0):
          alpha = augmentationParams["alpha"]
          # Selecting a random auxiliary image.
          auxImagePath = random.choice(auxImages)
          # Loading the auxiliary image.
          auxImage = Image.open(auxImagePath).convert("RGB")
          # Resizing auxiliary image to match original image dimensions.
          auxImage = auxImage.resize(augmentedImage.size, Image.BICUBIC)
          # Generating cutmix lambda from beta distribution.
          cutmixLambda = np.random.beta(alpha, alpha)
          # Getting image dimensions.
          width, height = augmentedImage.size
          # Calculating bounding box dimensions.
          cutWidth = int(width * np.sqrt(1 - cutmixLambda))
          cutHeight = int(height * np.sqrt(1 - cutmixLambda))
          # Generating random center position.
          cx = random.randint(0, width)
          cy = random.randint(0, height)
          # Calculating bounding box coordinates.
          x1 = np.clip(cx - cutWidth // 2, 0, width)
          y1 = np.clip(cy - cutHeight // 2, 0, height)
          x2 = np.clip(cx + cutWidth // 2, 0, width)
          y2 = np.clip(cy + cutHeight // 2, 0, height)
          # Converting images to numpy arrays.
          imageArray1 = np.array(augmentedImage)
          imageArray2 = np.array(auxImage)
          # Applying cutmix by replacing the region.
          imageArray1[y1:y2, x1:x2, :] = imageArray2[y1:y2, x1:x2, :]
          # Converting the numpy array back to PIL Image.
          augmentedImage = Image.fromarray(imageArray1)

  # Applying mosaic augmentation if selected.
  elif (augmentationType == "mosaic"):
    numImages = augmentationParams.get("numImages", 4)
    # Checking if auxiliary images list is provided.
    if (auxImagesList is not None and len(auxImagesList) >= numImages - 1):
      # Selecting random auxiliary images.
      selectedAuxImages = random.sample(auxImagesList, numImages - 1)
      # Creating list of all images including original.
      allImages = [augmentedImage] + [Image.open(path).convert("RGB") for path in selectedAuxImages]
      # Getting original image dimensions.
      width, height = augmentedImage.size
      # Creating mosaic based on number of images.
      if (numImages == 4):
        # Creating 2x2 mosaic.
        halfWidth = width // 2
        halfHeight = height // 2
        # Creating new image for mosaic.
        mosaicImage = Image.new("RGB", (width, height))
        # Resizing and pasting images.
        mosaicImage.paste(allImages[0].resize((halfWidth, halfHeight), Image.BICUBIC), (0, 0))
        mosaicImage.paste(allImages[1].resize((halfWidth, halfHeight), Image.BICUBIC), (halfWidth, 0))
        mosaicImage.paste(allImages[2].resize((halfWidth, halfHeight), Image.BICUBIC), (0, halfHeight))
        mosaicImage.paste(allImages[3].resize((halfWidth, halfHeight), Image.BICUBIC), (halfWidth, halfHeight))
        augmentedImage = mosaicImage
      elif (numImages == 9):
        # Creating 3x3 mosaic.
        thirdWidth = width // 3
        thirdHeight = height // 3
        # Creating new image for mosaic.
        mosaicImage = Image.new("RGB", (width, height))
        # Resizing and pasting images.
        for i in range(3):
          for j in range(3):
            idx = i * 3 + j
            if (idx < len(allImages)):
              mosaicImage.paste(
                allImages[idx].resize((thirdWidth, thirdHeight), Image.BICUBIC),
                (j * thirdWidth, i * thirdHeight)
              )
        augmentedImage = mosaicImage
    elif (augmentationParams.get("auxImageDir") is not None):
      # Loading images from the auxiliary image directory.
      auxImageDir = augmentationParams["auxImageDir"]
      if (os.path.exists(auxImageDir)):
        auxImages = [
          os.path.join(auxImageDir, f)
          for f in os.listdir(auxImageDir)
          if (f.lower().endswith((".png", ".jpg", ".jpeg")))
        ]
        if (len(auxImages) >= numImages - 1):
          # Selecting random auxiliary images.
          selectedAuxImages = random.sample(auxImages, numImages - 1)
          # Creating list of all images including original.
          allImages = [augmentedImage] + [Image.open(path).convert("RGB") for path in selectedAuxImages]
          # Getting original image dimensions.
          width, height = augmentedImage.size
          # Creating mosaic based on number of images.
          if (numImages == 4):
            # Creating 2x2 mosaic.
            halfWidth = width // 2
            halfHeight = height // 2
            # Creating new image for mosaic.
            mosaicImage = Image.new("RGB", (width, height))
            # Resizing and pasting images.
            mosaicImage.paste(allImages[0].resize((halfWidth, halfHeight), Image.BICUBIC), (0, 0))
            mosaicImage.paste(allImages[1].resize((halfWidth, halfHeight), Image.BICUBIC), (halfWidth, 0))
            mosaicImage.paste(allImages[2].resize((halfWidth, halfHeight), Image.BICUBIC), (0, halfHeight))
            mosaicImage.paste(allImages[3].resize((halfWidth, halfHeight), Image.BICUBIC), (halfWidth, halfHeight))
            augmentedImage = mosaicImage
          elif (numImages == 9):
            # Creating 3x3 mosaic.
            thirdWidth = width // 3
            thirdHeight = height // 3
            # Creating new image for mosaic.
            mosaicImage = Image.new("RGB", (width, height))
            # Resizing and pasting images.
            for i in range(3):
              for j in range(3):
                idx = i * 3 + j
                if (idx < len(allImages)):
                  mosaicImage.paste(allImages[idx].resize((thirdWidth, thirdHeight), Image.BICUBIC),
                                    (j * thirdWidth, i * thirdHeight))
            augmentedImage = mosaicImage

  # Applying color jitter augmentation if selected.
  elif (augmentationType == "colorJitter"):
    hueMin, hueMax = augmentationParams["hueShift"]
    satMin, satMax = augmentationParams["saturationShift"]
    valMin, valMax = augmentationParams["valueShift"]
    # Converting image to HSV color space.
    hsvImage = augmentedImage.convert("HSV")
    hsvArray = np.array(hsvImage).astype(np.float32)
    # Generating random shifts.
    hueShift = random.uniform(hueMin, hueMax) * 255
    satShift = random.uniform(satMin, satMax) * 255
    valShift = random.uniform(valMin, valMax) * 255
    # Applying shifts to HSV channels.
    hsvArray[:, :, 0] = (hsvArray[:, :, 0] + hueShift) % 256
    hsvArray[:, :, 1] = np.clip(hsvArray[:, :, 1] + satShift, 0, 255)
    hsvArray[:, :, 2] = np.clip(hsvArray[:, :, 2] + valShift, 0, 255)
    # Converting back to uint8 and PIL Image.
    hsvArray = hsvArray.astype(np.uint8)
    hsvImage = Image.fromarray(hsvArray, mode="HSV")
    # Converting back to RGB.
    augmentedImage = hsvImage.convert("RGB")

  # Applying elastic deformation augmentation if selected.
  elif (augmentationType == "elasticDeformation"):
    alphaMin, alphaMax = augmentationParams["alpha"]
    sigmaMin, sigmaMax = augmentationParams["sigma"]
    # Generating random alpha and sigma values.
    alpha = random.uniform(alphaMin, alphaMax)
    sigma = random.uniform(sigmaMin, sigmaMax)
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Getting image shape.
    shape = imageArray.shape[:2]
    # Generating random displacement fields.
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    # Creating meshgrid for coordinates.
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Computing distorted coordinates.
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # Applying deformation to each channel.
    distortedChannels = []
    for i in range(imageArray.shape[2]):
      distortedChannel = map_coordinates(imageArray[:, :, i], indices, order=1, mode="reflect")
      distortedChannels.append(distortedChannel.reshape(shape))
    # Stacking channels back together.
    distortedArray = np.stack(distortedChannels, axis=2).astype(np.uint8)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(distortedArray)

  # Applying perspective transform augmentation if selected.
  elif (augmentationType == "perspectiveTransform"):
    scaleMin, scaleMax = augmentationParams["scale"]
    # Generating random scale value.
    scale = random.uniform(scaleMin, scaleMax)
    # Getting image dimensions.
    width, height = augmentedImage.size
    # Converting to numpy array for OpenCV processing.
    imageArray = np.array(augmentedImage)
    # Defining source points (corners of the image).
    srcPoints = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # Generating random perspective distortion.
    dstPoints = np.float32([
      [random.uniform(0, width * scale), random.uniform(0, height * scale)],
      [random.uniform(width * (1 - scale), width), random.uniform(0, height * scale)],
      [random.uniform(width * (1 - scale), width), random.uniform(height * (1 - scale), height)],
      [random.uniform(0, width * scale), random.uniform(height * (1 - scale), height)]
    ])
    # Computing perspective transform matrix.
    matrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    # Applying perspective transformation.
    warpedArray = cv2.warpPerspective(imageArray, matrix, (width, height), borderValue=(0, 0, 0))
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(warpedArray)

  # Applying affine transform augmentation if selected.
  elif (augmentationType == "affineTransform"):
    scaleMin, scaleMax = augmentationParams["scale"]
    shearMin, shearMax = augmentationParams["shear"]
    rotateMin, rotateMax = augmentationParams["rotate"]
    # Generating random transformation parameters.
    scale = random.uniform(scaleMin, scaleMax)
    shear = random.uniform(shearMin, shearMax)
    rotate = random.uniform(rotateMin, rotateMax)
    # Getting image center.
    width, height = augmentedImage.size
    centerX, centerY = width / 2, height / 2
    # Converting to numpy array for OpenCV processing.
    imageArray = np.array(augmentedImage)
    # Computing rotation matrix.
    rotationMatrix = cv2.getRotationMatrix2D((centerX, centerY), rotate, scale)
    # Adding shear to the transformation matrix.
    shearRadians = np.deg2rad(shear)
    shearMatrix = np.array([[1, np.tan(shearRadians), 0], [0, 1, 0]], dtype=np.float32)
    # Combining rotation and shear.
    affineMatrix = np.dot(shearMatrix, np.vstack([rotationMatrix, [0, 0, 1]]))[:2]
    # Applying affine transformation.
    transformedArray = cv2.warpAffine(imageArray, affineMatrix, (width, height), borderValue=(0, 0, 0))
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(transformedArray)

  # Applying CLAHE augmentation if selected.
  elif (augmentationType == "clahe"):
    clipLimitMin, clipLimitMax = augmentationParams["clipLimit"]
    tileGridSize = augmentationParams["tileGridSize"]
    # Generating random clip limit.
    clipLimit = random.uniform(clipLimitMin, clipLimitMax)
    # Converting image to LAB color space for better results.
    imageArray = np.array(augmentedImage)
    labImage = cv2.cvtColor(imageArray, cv2.COLOR_RGB2LAB)
    # Creating CLAHE object.
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    # Applying CLAHE to L channel only.
    labImage[:, :, 0] = clahe.apply(labImage[:, :, 0])
    # Converting back to RGB.
    rgbArray = cv2.cvtColor(labImage, cv2.COLOR_LAB2RGB)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(rgbArray)

  # Applying speckle noise augmentation if selected.
  elif (augmentationType == "speckleNoise"):
    minIntensity, maxIntensity = augmentationParams["intensity"]
    # Generating random speckle intensity.
    intensity = random.uniform(minIntensity, maxIntensity)
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage).astype(np.float32)
    # Generating speckle noise (multiplicative).
    speckle = np.random.randn(*imageArray.shape) * intensity
    # Applying speckle noise.
    noisyArray = imageArray + imageArray * speckle
    # Clipping values to valid range.
    noisyArray = np.clip(noisyArray, 0, 255).astype(np.uint8)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(noisyArray)

  # Applying salt and pepper noise augmentation if selected.
  elif (augmentationType == "saltPepperNoise"):
    minAmount, maxAmount = augmentationParams["amount"]
    saltVsPepper = augmentationParams["saltVsPepper"]
    # Generating random noise amount.
    amount = random.uniform(minAmount, maxAmount)
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage).copy()
    # Calculating number of pixels to corrupt.
    numSalt = int(amount * imageArray.size * saltVsPepper / imageArray.shape[2])
    numPepper = int(amount * imageArray.size * (1 - saltVsPepper) / imageArray.shape[2])
    # Adding salt (white pixels).
    saltCoords = [np.random.randint(0, i, numSalt) for i in imageArray.shape[:2]]
    imageArray[saltCoords[0], saltCoords[1], :] = 255
    # Adding pepper (black pixels).
    pepperCoords = [np.random.randint(0, i, numPepper) for i in imageArray.shape[:2]]
    imageArray[pepperCoords[0], pepperCoords[1], :] = 0
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(imageArray)

  # Applying Poisson noise augmentation if selected.
  elif (augmentationType == "poissonNoise"):
    minScale, maxScale = augmentationParams["scale"]
    # Generating random scale value.
    scale = random.uniform(minScale, maxScale)
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage).astype(np.float32)
    # Normalizing to range suitable for Poisson.
    normalized = imageArray / 255.0 * scale
    # Applying Poisson noise.
    poissonArray = np.random.poisson(normalized) / scale * 255.0
    # Clipping values to valid range.
    poissonArray = np.clip(poissonArray, 0, 255).astype(np.uint8)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(poissonArray)

  # Applying motion blur augmentation if selected.
  elif (augmentationType == "motionBlur"):
    minKernel, maxKernel = augmentationParams["kernelSize"]
    minAngle, maxAngle = augmentationParams["angle"]
    # Generating random kernel size (must be odd).
    kernelSize = random.randint(minKernel, maxKernel)
    if (kernelSize % 2 == 0):
      kernelSize += 1
    # Generating random angle.
    angle = random.uniform(minAngle, maxAngle)
    # Creating motion blur kernel.
    kernel = np.zeros((kernelSize, kernelSize))
    kernel[int((kernelSize - 1) / 2), :] = np.ones(kernelSize)
    kernel = kernel / kernelSize
    # Rotating the kernel.
    kernelRotated = cv2.warpAffine(
      kernel,
      cv2.getRotationMatrix2D((kernelSize / 2, kernelSize / 2), angle, 1.0),
      (kernelSize, kernelSize)
    )
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Applying motion blur.
    blurredArray = cv2.filter2D(imageArray, -1, kernelRotated)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(blurredArray)

  # Applying median blur augmentation if selected.
  elif (augmentationType == "medianBlur"):
    minKernel, maxKernel = augmentationParams["kernelSize"]
    # Generating random kernel size (must be odd).
    kernelSize = random.randint(minKernel, maxKernel)
    if (kernelSize % 2 == 0):
      kernelSize += 1
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Applying median blur.
    blurredArray = cv2.medianBlur(imageArray, kernelSize)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(blurredArray)

  # Applying bilateral filter augmentation if selected.
  elif (augmentationType == "bilateralFilter"):
    minD, maxD = augmentationParams["d"]
    minSigmaColor, maxSigmaColor = augmentationParams["sigmaColor"]
    minSigmaSpace, maxSigmaSpace = augmentationParams["sigmaSpace"]
    # Generating random parameters.
    d = random.randint(minD, maxD)
    sigmaColor = random.uniform(minSigmaColor, maxSigmaColor)
    sigmaSpace = random.uniform(minSigmaSpace, maxSigmaSpace)
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Applying bilateral filter.
    filteredArray = cv2.bilateralFilter(imageArray, d, sigmaColor, sigmaSpace)
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(filteredArray)

  # Applying channel shuffle augmentation if selected.
  elif (augmentationType == "channelShuffle"):
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Getting list of channel indices.
    channels = [0, 1, 2]
    # Shuffling the channels randomly.
    random.shuffle(channels)
    # Reordering channels.
    shuffledArray = imageArray[:, :, channels]
    # Converting back to PIL Image.
    augmentedImage = Image.fromarray(shuffledArray)

  # Applying invert augmentation if selected.
  elif (augmentationType == "invert"):
    # Inverting the image colors.
    augmentedImage = ImageOps.invert(augmentedImage)

  # Applying solarize augmentation if selected.
  elif (augmentationType == "solarize"):
    minThreshold, maxThreshold = augmentationParams["threshold"]
    # Generating random threshold.
    threshold = random.randint(minThreshold, maxThreshold)
    # Applying solarization.
    augmentedImage = ImageOps.solarize(augmentedImage, threshold)

  # Applying posterize augmentation if selected.
  elif (augmentationType == "posterize"):
    minBits, maxBits = augmentationParams["bits"]
    # Generating random bit depth.
    bits = random.randint(minBits, maxBits)
    # Applying posterization.
    augmentedImage = ImageOps.posterize(augmentedImage, bits)

  # Applying equalize augmentation if selected.
  elif (augmentationType == "equalize"):
    # Applying histogram equalization.
    augmentedImage = ImageOps.equalize(augmentedImage)

  # Applying emboss augmentation if selected.
  elif (augmentationType == "emboss"):
    # Applying emboss filter.
    augmentedImage = augmentedImage.filter(ImageFilter.EMBOSS)

  # Applying edge enhance augmentation if selected.
  elif (augmentationType == "edgeEnhance"):
    minFactor, maxFactor = augmentationParams["factor"]
    # Generating random blending factor.
    blendFactor = random.uniform(minFactor, maxFactor)
    # Applying edge enhancement.
    edgeEnhanced = augmentedImage.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # Blending with original image.
    augmentedImage = Image.blend(augmentedImage, edgeEnhanced, blendFactor)

  # Applying coarse dropout augmentation if selected.
  elif (augmentationType == "coarseDropout"):
    minHoles, maxHoles = augmentationParams["numHoles"]
    minSize, maxSize = augmentationParams["holeSize"]
    fillValue = augmentationParams["fillValue"]
    # Getting image dimensions.
    width, height = augmentedImage.size
    # Converting image to numpy array.
    imageArray = np.array(augmentedImage)
    # Generating random number of holes.
    numHoles = random.randint(minHoles, maxHoles)
    # Creating coarse dropout holes.
    for _ in range(numHoles):
      # Calculating hole size as fraction of image dimensions.
      holeSizeFraction = random.uniform(minSize, maxSize)
      holeWidth = int(width * holeSizeFraction)
      holeHeight = int(height * holeSizeFraction)
      # Generating random position for the hole.
      x = random.randint(0, max(0, width - holeWidth))
      y = random.randint(0, max(0, height - holeHeight))
      # Filling the hole with specified color.
      imageArray[y:y + holeHeight, x:x + holeWidth, :] = fillValue
    # Converting the numpy array back to PIL Image.
    augmentedImage = Image.fromarray(imageArray)

  # Returning the augmented image.
  return augmentedImage


def LoadAuxiliaryImages(directory: str, maxImages: int = 100) -> List[str]:
  r'''
  Loads auxiliary image paths from a directory for advanced augmentations.

  This function scans a directory for image files and returns a list of their paths. These paths can be used
  for advanced data augmentation techniques that require additional images, such as mixup, cutmix, and mosaic.
  The function supports common image formats including PNG, JPG, JPEG, BMP, and GIF.

  Parameters:
    directory (str): Path to the directory containing auxiliary images.
    maxImages (int): Maximum number of image paths to load. Default is 100.

  Returns:
    List[str]: List of image file paths found in the directory.
  '''

  # Checking if the directory exists.
  if (not os.path.exists(directory)):
    print(f"Warning: Auxiliary image directory not found: {directory}")
    return []

  # Creating a list to store image paths.
  imagePaths = []

  # Iterating through files in the directory.
  for filename in os.listdir(directory):
    # Checking if the file has an image extension.
    if (filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))):
      # Adding the full path to the list.
      imagePaths.append(os.path.join(directory, filename))
      # Checking if we have reached the maximum number of images.
      if (len(imagePaths) >= maxImages):
        break

  # Returning the list of image paths.
  return imagePaths


def SaveAugmentedImages(
  augmentedImages: List[Image.Image],
  outputDir: str,
  baseFilename: str,
  outputExtension: str = ".png"
) -> None:
  r'''
  Saves augmented images to the specified output directory.

  This function takes a list of augmented PIL Image objects and saves them to disk with sequential numbering.
  The output directory is created automatically if it does not exist. Each saved image is named using the
  base filename followed by "_Aug_" and a sequential number.

  Parameters:
    augmentedImages (List[Image.Image]): List of augmented PIL Image objects to save.
    outputDir (str): Directory where augmented images will be saved.
    baseFilename (str): Base name for the output files (without extension).
    outputExtension (str): File extension for the saved images (default is ".png").
  '''

  # Creating the output directory if it does not exist.
  os.makedirs(outputDir, exist_ok=True)

  # Iterating through the augmented images and saving them.
  for idx, augmentedImage in enumerate(augmentedImages):
    # Constructing the output filename with index.
    outputFilename = f"{baseFilename}_Aug_{idx + 1}{outputExtension}"
    # Creating the full output path.
    outputPath = os.path.join(outputDir, outputFilename)
    # Saving the augmented image to disk.
    augmentedImage.save(outputPath)
    print(f"Saved augmented image: {outputPath}")
