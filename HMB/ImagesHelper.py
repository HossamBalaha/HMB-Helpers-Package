# Import the required libraries.
import os  # Import os for file path operations.
import cv2, PIL  # Import OpenCV and PIL for image processing.
import numpy as np  # Import numpy for numerical operations.
from PIL import Image # Import Image module from PIL for image handling.
import matplotlib.pyplot as plt  # Import matplotlib for plotting.


def ReadVolume(caseImgPaths, caseSegPaths, raiseErrors=True):
  r'''
  Read and preprocess a 3D volume from a set of 2D slices and their corresponding segmentation masks.

  Parameters:
    caseImgPaths (list): List of paths to the 2D slices of the volume.
    caseSegPaths (list): List of paths to the segmentation masks of the slices.

  Returns:
    numpy.ndarray: A 3D NumPy array representing the preprocessed volume.

  Raises:
    FileNotFoundError: If any of the image or segmentation files are not found.
    ValueError: If a cropped image is empty and raiseErrors is set to True.
  '''

  # Initialize a list to store the cropped slices.
  volumeCropped = []

  # Loop through each slice and its corresponding segmentation mask.
  for i in range(len(caseImgPaths)):
    # Check if the files exist.
    if (not os.path.exists(caseImgPaths[i])) or (not os.path.exists(caseSegPaths[i])):
      raise FileNotFoundError("One or more files were not found. Please check the file paths.")

    # Load the slice and segmentation mask in grayscale mode.
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)  # Load the slice.
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

    # Extract the Region of Interest (ROI) using the segmentation mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)  # Apply bitwise AND operation to extract the ROI.

    # Crop the ROI to remove unnecessary background.
    x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
    cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

    if (np.sum(cropped) <= 0):
      # Raise an error if the cropped image is empty and raiseErrors is True.
      if (raiseErrors):
        raise ValueError("The cropped image is empty. Please check the segmentation mask.")
      else:
        # If raiseErrors is False, skip the empty slice and continue processing.
        continue

    # Append the cropped slice to the list.
    volumeCropped.append(cropped)

  # Determine the maximum width and height across all cropped slices.
  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])  # Maximum width.
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])  # Maximum height.

  # Pad each cropped slice to match the maximum width and height.
  for i in range(len(volumeCropped)):
    # Calculate the padding size.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]  # Horizontal padding.
    deltaHeight = maxHeight - volumeCropped[i].shape[0]  # Vertical padding.

    # Add padding to the cropped image and place the image in the center.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],  # Image to pad.
      deltaHeight // 2,  # Top padding.
      deltaHeight - deltaHeight // 2,  # Bottom padding.
      deltaWidth // 2,  # Left padding.
      deltaWidth - deltaWidth // 2,  # Right padding.
      cv2.BORDER_CONSTANT,  # Padding type.
      value=0  # Padding value.
    )

    # Replace the cropped slice with the padded slice.
    volumeCropped[i] = padded.copy()

  # Convert the list of slices to a 3D NumPy array.
  volumeCropped = np.array(volumeCropped)

  return volumeCropped  # Return the preprocessed 3D volume.


def ReadVolumeSpecificClasses(caseImgPaths, caseSegPaths, specificClasses=[]):
  r'''
  Read and preprocess a 3D volume from a set of 2D slices and their corresponding segmentation masks.

  Parameters:
    caseImgPaths (list): List of file paths to medical image slices in BMP format.
    caseSegPaths (list): List of file paths to segmentation masks matching the slices.
    specificClasses (list): List of specific classes to include in the segmentation. If empty, all classes are included.

  Returns:
    numpy.ndarray: 3D array of preprocessed and aligned medical imaging data.

  Raises:
    FileNotFoundError: If any of the image or segmentation files are not found.
    ValueError: If no slices were successfully processed.
  '''

  # Initialize empty list to store processed slices.
  volumeCropped = []

  # Process each image-segmentation pair in the input lists.
  for i in range(len(caseImgPaths)):
    # Verify both image and segmentation files exist before processing.
    if (not os.path.exists(caseImgPaths[i])) or (not os.path.exists(caseSegPaths[i])):
      raise FileNotFoundError("One or more files were not found. Please check the file paths.")

    # Load grayscale medical image slice (8-bit depth).
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)
    # Load corresponding binary segmentation mask.
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)

    # Check if specific classes are provided for segmentation.
    if (specificClasses):
      # Create a mask for the specific classes.
      mask = np.zeros_like(caseSeg)
      for classId in specificClasses:
        mask[caseSeg == classId] = 255
      caseSeg = mask

    # Extract region of interest using bitwise AND operation between image and mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)

    # Calculate bounding box coordinates of non-zero region in ROI.
    x, y, w, h = cv2.boundingRect(roi)
    # Crop image to tight bounding box around segmented area.
    cropped = roi[y:y + h, x:x + w]

    # Validate cropped slice contains actual data (not just background).
    if (np.sum(cropped) <= 0):
      continue  # Skip empty slices.

    # Add processed slice to volume list.
    volumeCropped.append(cropped)

  # Check if any slices were successfully processed.
  if (len(volumeCropped) == 0):
    raise ValueError("No slices were successfully processed. Please check the input data.")

  # Determine maximum dimensions across all slices for padding alignment.
  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])

  # Standardize slice dimensions through symmetric padding.
  for i in range(len(volumeCropped)):
    # Calculate required padding for width and height dimensions.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]
    deltaHeight = maxHeight - volumeCropped[i].shape[0]

    # Apply padding to create uniform slice dimensions.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],
      deltaHeight // 2,  # Top padding (integer division)
      deltaHeight - deltaHeight // 2,  # Bottom padding (remainder)
      deltaWidth // 2,  # Left padding
      deltaWidth - deltaWidth // 2,  # Right padding
      cv2.BORDER_CONSTANT,  # Padding style (constant zero values)
      value=0
    )

    # Update volume with padded slice.
    volumeCropped[i] = padded.copy()

  # Convert list of 2D slices into 3D numpy array (z, y, x).
  volumeCropped = np.array(volumeCropped)

  return volumeCropped


def ExtractMultipleObjectsFromROI(
  caseImg,
  caseSeg,
  targetSize=(256, 256),
  cntAreaThreshold=0,
  sortByX=True,
):
  r'''
  Extracts multiple objects from a region of interest (ROI) in a medical image.

  Parameters:
    caseImg (numpy.ndarray): The input medical image.
    caseSeg (numpy.ndarray): The segmentation mask indicating regions of interest.
    targetSize (tuple): The target size for resizing images.
    cntAreaThreshold (int): Minimum contour area to consider for extraction.
    sortByX (bool): Whether to sort extracted regions by their x-coordinate.

  Returns:
    list: A list of extracted regions from the image.

  Raises:
    ValueError: If the segmentation mask is completely black/empty.
  '''

  # Resize images to standard dimensions using cubic interpolation for quality.
  caseImg = cv2.resize(caseImg, targetSize, interpolation=cv2.INTER_CUBIC)  # Resize scan image.
  caseSeg = cv2.resize(caseSeg, targetSize, interpolation=cv2.INTER_CUBIC)  # Resize mask image.

  # Binarize segmentation mask by thresholding to ensure only 0/255 values.
  caseSeg[caseSeg > 0] = 255  # Convert any positive values to pure white.

  # or you can apply:
  # caseSeg = cv2.threshold(caseSeg, 127, 255, cv2.THRESH_BINARY)[1]  # Binarize mask.

  # Perform sanity check on the segmentation mask to ensure valid content.
  if (np.sum(caseSeg) <= 0):  # Calculate sum of all pixel values in mask.
    # Raise error if mask contains no white pixels to prevent empty processing.
    raise ValueError("The mask is completely black/empty. Please check the segmentation mask.")

  # Detect contours in the segmentation mask using simple approximation method.
  contours = cv2.findContours(caseSeg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if (sortByX):  # Check if sorting by x-coordinate is requested.
    # Sort detected contours from left-to-right based on their x-coordinate.
    contours = sorted(contours[0], key=lambda x: cv2.boundingRect(x)[0], reverse=False)
  else:
    # If not sorting, just extract contours without sorting.
    contours = contours[0] if (len(contours) == 2) else contours[1]

  # Initialize empty list to store extracted region-of-interest (ROI) images.
  regions = []

  # Process each detected contour to extract individual anatomical structures.
  for i in range(len(contours)):
    # Calculate the area of the current contour for size filtering.
    cntArea = cv2.contourArea(contours[i])
    # Skip contours smaller than threshold to ignore noise/artifacts.
    if (cntArea <= cntAreaThreshold):
      continue

    # Create blank mask matching image dimensions for current ROI.
    regionMask = np.zeros_like(caseSeg)
    # Select current contour from the list of detected contours.
    regionCnt = contours[i]
    # Fill contour area in the mask to create binary ROI representation.
    cv2.fillPoly(regionMask, [regionCnt], 255)
    # Apply mask to original image to isolate the anatomical structure.
    roi = cv2.bitwise_and(caseImg, regionMask)
    # Calculate bounding box coordinates around the masked region.
    x, y, w, h = cv2.boundingRect(roi)
    # Crop the region from the original image using bounding box coordinates.
    cropped = roi[y:y + h, x:x + w]
    # Add cropped ROI to the collection of extracted regions.
    regions.append(cropped)

  if (sortByX):  # Check if sorting by x-coordinate is requested.
    # Sort extracted regions by their x-coordinate to maintain left-to-right order.
    regions = sorted(regions, key=lambda x: cv2.boundingRect(x)[0], reverse=False)

  # Return the list of extracted regions for further processing.
  return regions


def GetEmptyPercentage(img, shape=(256, 256), inverse=False):
  r'''
  Calculate the percentage of empty (black or white) regions in an image.

  Parameters:
    img (numpy.ndarray): Input RGB image.
    shape (tuple): Desired shape for calculating the percentage.
    inverse (bool): If True, calculates the percentage of non-empty regions instead.

  Returns:
    float: Ratio of empty regions to the total area.
  '''

  # Convert the input image to grayscale.
  imgGray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

  # Resize the grayscale image to the specified shape if needed.
  if ((shape is not None) and (imgGray.shape[0] != shape[0]) or (imgGray.shape[1] != shape[1])):
    imgGray = cv2.resize(imgGray, shape, interpolation=cv2.INTER_CUBIC)

  # If inverse is True, calculate non-empty regions instead.
  if (inverse):
    imgGray = 255 - imgGray

  # Apply Otsu's thresholding to binarize the image.
  # THRESH_BINARY_INV means background will be white (255) and foreground black (0).
  _, thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

  # Count non-zero pixels (which represent the background in this case).
  backgroundPixels = cv2.countNonZero(thresh)

  # Calculate the ratio of empty (black or white) regions to the total area.
  ratio = backgroundPixels * 100.0 / (shape[0] * shape[1])

  # Ensure the ratio is within the range [0, 100].
  return ratio


def GetEmptyPercentageHistogram(img, shape=(256, 256), inverse=False, thresholdLow=10, thresholdHigh=245):
  r'''
  Calculate the percentage of empty (black or white) regions in an image using histogram analysis.

  Parameters:
    img (numpy.ndarray): Input RGB image.
    shape (tuple): Desired shape for calculating the percentage.
    inverse (bool): If True, calculates the percentage of non-empty regions instead.
    thresholdLow (int): Threshold for detecting dark regions (default: 10).
    thresholdHigh (int): Threshold for detecting bright regions (default: 245).

  Returns:
    float: Ratio of empty regions to the total area.
  '''

  # Convert the input image to grayscale.
  imgGray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

  # Resize the grayscale image to the specified shape if needed.
  if ((shape is not None) and (imgGray.shape[0] != shape[0]) or (imgGray.shape[1] != shape[1])):
    imgGray = cv2.resize(imgGray, shape, interpolation=cv2.INTER_CUBIC)

  # If inverse is True, invert the grayscale image.
  if (inverse):
    imgGray = 255 - imgGray  # Invert the grayscale image if inverse is True.

  # Calculate histogram.
  hist = cv2.calcHist([imgGray], [0], None, [256], [0, 256])

  # Calculate empty pixels (near black and near white).
  # Near black pixels (0-10).
  emptyBlack = np.sum(hist[:thresholdLow + 1])

  # Near white pixels (245-255).
  emptyWhite = np.sum(hist[thresholdHigh:])

  # Total empty pixels
  totalEmpty = emptyBlack + emptyWhite

  # Calculate the ratio of empty regions to the total area.
  totalPixels = shape[0] * shape[1]
  ratio = (totalEmpty * 100.0) / totalPixels

  # Ensure the ratio is within the range [0, 100].
  return min(max(ratio, 0), 100)


def ExtractLargestContour(img):
  r'''
  Extract the largest contour from an image and create a mask.

  Parameters:
    img (numpy.ndarray): Input RGB image.

  Returns:
    tuple: Masked image, contour, mask, and visualization.
  '''

  # Convert the input RGB image to grayscale.
  imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # Apply Gaussian blur to reduce noise using a 5x5 kernel.
  imgGray = cv2.GaussianBlur(imgGray, (5, 5), 0)
  # Apply Otsu's thresholding to create a binary image.
  imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  # Find contours in the binary image using external retrieval mode and simple chain approximation.
  contours, _ = cv2.findContours(
    imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  # Sort the contours by area in descending order.
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:]
  # Select the largest contour.
  contour = contours[0]
  # Create a mask for the largest contour.
  mask = np.zeros(imgGray.shape, np.uint8)
  # Draw the contour on the mask.
  cv2.drawContours(mask, [contour], -1, 255, -1)
  # Apply the mask to the original image.
  img = cv2.bitwise_and(img, img, mask=mask)
  # Fill any black regions in the masked image with white.
  img[img == 0] = 255
  # Draw the contour on a copy of the image for visualization.
  draw = cv2.drawContours(
    img.copy(), [contour], -1, (0, 255, 0), 2
  )
  # Return the masked image, the contour, the mask, and the visualization.
  return img, contour, mask, draw


def MatchTwoImagesViaSIFT(
  img1,  # First input RGB image.
  img2,  # Second input RGB image.
  shape=(1024, 1024),  # Desired output shape for the aligned images.
  tolerance=0.50,  # Ratio threshold for filtering good matches (default is 0.50).
):
  r'''
  Match two images using SIFT (Scale-Invariant Feature Transform) feature detection.
  This function detects keypoints and computes descriptors for both images,
  then matches them using a brute-force matcher with a ratio test to filter good matches.
  The function returns the aligned images, matches, homography matrix, and output shape.
  This is useful for aligning images that may have different perspectives or scales.

  .. math::

     H = \begin{bmatrix}
         h_{11} & h_{12} & h_{13} \\
         h_{21} & h_{22} & h_{23} \\
         h_{31} & h_{32} & h_{33}
     \end{bmatrix}

  where:
    - :math:`H` is the homography matrix.
    - :math:`h_{ij}` are the elements of the homography matrix.

  Parameters:
    img1 (numpy.ndarray): First input RGB image.
    img2 (numpy.ndarray): Second input RGB image.
    shape (tuple): Desired output shape for the aligned images.
    tolerance (float): Ratio threshold for filtering good matches.

  Returns:
    tuple: Aligned images, matches, homography matrix, and output shape.
  '''

  # If no shape is provided, use the maximum dimensions of the input images.
  if (shape is None):
    shape = (
      max(img1.shape[1], img2.shape[1]),
      max(img1.shape[0], img2.shape[0]),
    )
  # Create a SIFT detector.
  sift = cv2.SIFT_create()
  # Detect keypoints and compute descriptors for the first image.
  kp1, des1 = sift.detectAndCompute(img1, None)
  # Detect keypoints and compute descriptors for the second image.
  kp2, des2 = sift.detectAndCompute(img2, None)
  # Create a Brute-Force Matcher.
  bf = cv2.BFMatcher()
  # Match the descriptors using k-nearest neighbors (k=2).
  matches = bf.knnMatch(des1, des2, k=2)
  # Initialize a list to store good matches.
  good = []
  # Apply the ratio test to filter out poor matches.
  for (m, n) in matches:
    if (m.distance < tolerance * n.distance):
      good.append([m])
  # Extract the coordinates of the matched keypoints in the first image.
  srcPoints = np.float32([kp1[match[0].queryIdx].pt for match in good]).reshape(-1, 1, 2)
  # Extract the coordinates of the matched keypoints in the second image.
  dstPoints = np.float32([kp2[match[0].trainIdx].pt for match in good]).reshape(-1, 1, 2)
  # Compute the homography matrix using RANSAC.
  homography, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
  # Warp the first image to align with the second image using the homography.
  img1Trans = cv2.warpPerspective(img1, homography, shape)
  # Fill any black regions in the warped image with white.
  img1Trans[img1Trans == 0] = 255
  # Draw the matches on a new image.
  imgMatches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
  # Crop the images to the minimum dimensions to ensure they are the same size.
  minShape = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
  # Crop the warped image.
  img1Trans = img1Trans[:minShape[1], :minShape[0], :]
  # Crop the second image.
  img2 = img2[:minShape[1], :minShape[0], :]
  # Return the aligned images, the matches, the homography, and the shape.
  return img1Trans, img2, imgMatches, homography, shape


# Match two images using ORB feature detection and alignment.
def MatchTwoImagesViaORB(
  img1,  # First input RGB image.
  img2,  # Second input RGB image.
  shape=(1024, 1024),  # Desired output shape for the aligned images.
  maxNumFeatures=5000,  # Maximum number of features to detect.
  maxGoodMatches=50,  # Maximum number of good matches to consider for alignment.
):
  r'''
  Match two images using ORB (Oriented FAST and Rotated BRIEF) feature detection.
  This function detects keypoints and computes descriptors for both images using ORB,
  then matches them using a brute-force matcher. The function filters the matches to retain
  the best ones and computes a homography matrix to align the images.
  The function returns the aligned images, matches, homography matrix, and output shape.

  .. math::

     H = \begin{bmatrix}
         h_{11} & h_{12} & h_{13} \\
         h_{21} & h_{22} & h_{23} \\
         h_{31} & h_{32} & h_{33}
     \end{bmatrix}

  where:
    - :math:`H` is the homography matrix.
    - :math:`h_{ij}` are the elements of the homography matrix.

  Parameters:
    img1 (numpy.ndarray): First input RGB image.
    img2 (numpy.ndarray): Second input RGB image.
    shape (tuple): Desired output shape for the aligned images.
    maxNumFeatures (int): Maximum number of features to detect.
    maxGoodMatches (int): Maximum number of good matches to use.

  Returns:
    tuple: Aligned images, matches, homography matrix, and output shape.
  '''

  # If no shape is provided, use the maximum dimensions of the input images.
  if (shape is None):
    shape = (
      max(img1.shape[1], img2.shape[1]),
      max(img1.shape[0], img2.shape[0]),
    )
  # Create an ORB detector with the specified maximum number of features.
  orbDetector = cv2.ORB_create(maxNumFeatures)
  # Detect keypoints and compute descriptors for the first image.
  kp1, d1 = orbDetector.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
  # Detect keypoints and compute descriptors for the second image.
  kp2, d2 = orbDetector.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
  # Create a Brute-Force Matcher using Hamming distance.
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  # Match the descriptors of the two images.
  matches = list(matcher.match(d1, d2))
  # Sort the matches by distance (best matches first).
  matches = sorted(matches, key=lambda x: x.distance)
  # Select the top N good matches.
  good = matches[:maxGoodMatches]
  # Draw the matches on a new image.
  imgMatches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
  # Extract the coordinates of the matched keypoints in the first image.
  srcPoints = np.float32([kp1[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
  # Extract the coordinates of the matched keypoints in the second image.
  dstPoints = np.float32([kp2[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
  # Compute the homography matrix using RANSAC.
  homography, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
  # Warp the first image to align with the second image using the homography.
  img1Trans = cv2.warpPerspective(img1, homography, shape)
  # Fill any black regions in the warped image with white.
  img1Trans[img1Trans == 0] = 255
  # Crop the images to the minimum dimensions to ensure they are the same size.
  minShape = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
  # Crop the warped image.
  img1Trans = img1Trans[:minShape[1], :minShape[0], :]
  # Crop the second image.
  img2 = img2[:minShape[1], :minShape[0], :]
  # Return the aligned images, the matches, the homography, and the shape.
  return img1Trans, img2, imgMatches, homography, shape


# Perform Free Form Deformation (FFD) using B-spline transformation to align two images.
def FreeFormDeformationImproved(
  imagePath1,  # Path to the source image.
  imagePath2,  # Path to the target image.
  gridSize=[10, 10],  # Grid size for the B-spline transform.
  numberOfHistogramBins=50,  # Number of histogram bins for the metric.
  samplingPercentage=0.1,  # Percentage of pixels to sample for the metric.
  learningRate=0.01,  # Learning rate for the optimizer.
  numberOfIterations=500,  # Maximum number of iterations.
  convergenceMinimumValue=1e-6,  # Convergence threshold.
  convergenceWindowSize=10,  # Window size for convergence determination.
):
  r'''
  Perform Free Form Deformation (FFD) using B-spline transformation to align two images.

  Parameters:
    imagePath1 (str): Path to the source image.
    imagePath2 (str): Path to the target image.
    gridSize (list): Grid size for the B-spline transform (default is [10, 10]).
    numberOfHistogramBins (int): Number of histogram bins for the metric (default is 50).
    samplingPercentage (float): Percentage of pixels to sample for the metric (default is 0.1).
    learningRate (float): Learning rate for the optimizer (default is 0.01).
    numberOfIterations (int): Maximum number of iterations for the optimizer (default is 500).
    convergenceMinimumValue (float): Convergence threshold for the optimizer (default is 1e-6).
    convergenceWindowSize (int): Window size for convergence determination (default is 10).

  Returns:
    tuple: A tuple containing the original source image, target image, and deformed image.
  '''

  import SimpleITK as sitk  # Import SimpleITK for image registration and transformation.

  # Load the source and target images using PIL.
  image1 = PIL.Image.open(imagePath1)  # Open the source image from the given file path.
  image2 = PIL.Image.open(imagePath2)  # Open the target image from the given file path.

  # Convert the images to grayscale for easier processing.
  sourceGray = np.array(image1.convert("L"))  # Convert the source image to grayscale.
  targetGray = np.array(image2.convert("L"))  # Convert the target image to grayscale.

  # Convert the grayscale images to float32 for numerical processing.
  sourceChannel = sourceGray.astype(np.float32)  # Ensure the source image is in float32 format.
  targetChannel = targetGray.copy().astype(np.float32)  # Ensure the target image is in float32 format.

  # Convert the NumPy arrays to SimpleITK images for registration.
  sourceChannel = sitk.GetImageFromArray(sourceChannel)  # Convert the source image to a SimpleITK image.
  targetChannel = sitk.GetImageFromArray(targetChannel)  # Convert the target image to a SimpleITK image.

  # Define the B-spline transform for the grid (Free Form Deformation).
  initialTransform = sitk.BSplineTransformInitializer(targetChannel, gridSize)  # Initialize the B-spline transform.

  # Define the registration method for aligning the images.
  registrationMethod = sitk.ImageRegistrationMethod()  # Create an instance of the registration method.
  # Use Mattes Mutual Information as the similarity metric for registration.
  registrationMethod.SetMetricAsMattesMutualInformation(numberOfHistogramBins=numberOfHistogramBins)
  # Use regular sampling for the metric to reduce computation time.
  registrationMethod.SetMetricSamplingStrategy(registrationMethod.REGULAR)  # Set the sampling strategy.
  # Set the sampling percentage for the metric to balance accuracy and speed.
  registrationMethod.SetMetricSamplingPercentage(samplingPercentage)  # Set the sampling percentage.
  # Use linear interpolation for resampling during registration.
  registrationMethod.SetInterpolator(sitk.sitkLinear)  # Set the interpolation method.
  # Use gradient descent as the optimizer for minimizing the registration error.
  registrationMethod.SetOptimizerAsGradientDescent(
    learningRate=learningRate,  # Learning rate for the optimizer.
    numberOfIterations=numberOfIterations,  # Maximum number of iterations.
    convergenceMinimumValue=convergenceMinimumValue,  # Convergence threshold.
    convergenceWindowSize=convergenceWindowSize,  # Window size for convergence determination.
  )
  # Set the optimizer scales based on physical shifts to improve optimization stability.
  registrationMethod.SetOptimizerScalesFromPhysicalShift()  # Adjust optimizer scales dynamically.
  # Set the initial transform for the registration process.
  registrationMethod.SetInitialTransform(initialTransform)  # Provide the initial B-spline transform.

  # Execute the registration to compute the optimal transform.
  transform = registrationMethod.Execute(targetChannel, sourceChannel)  # Perform the registration.

  # Initialize the deformed image with the same size as the source image.
  deformed = np.zeros_like(np.array(image1)).astype(np.float32)  # Pre-allocate memory for the deformed image.
  source = np.array(image1)  # Convert the source image to a NumPy array for processing.

  # Apply the transform to each channel (RGB) of the source image.
  for i in range(deformed.shape[2]):  # Loop through each channel (e.g., Red, Green, Blue).

    # Extract the current channel from the source image.
    sourceChannel = source[:, :, i].copy().astype(np.float32)  # Extract and convert the channel to float32.
    sourceChannel = sitk.GetImageFromArray(sourceChannel)  # Convert the channel to a SimpleITK image.

    # Define the resampler for applying the computed transform.
    resampler = sitk.ResampleImageFilter()  # Create a resampler object.
    resampler.SetTransform(transform)  # Set the computed transform for resampling.
    resampler.SetDefaultPixelValue(0)  # Set the default pixel value for areas outside the source image.
    resampler.SetReferenceImage(targetChannel)  # Set the target image as the reference for resampling.
    resampler.SetOutputSpacing(targetChannel.GetSpacing())  # Set the output spacing to match the target image.
    resampler.SetSize(targetChannel.GetSize())  # Set the output size to match the target image.
    resampler.SetOutputOrigin(targetChannel.GetOrigin())  # Set the output origin to match the target image.

    # Resample the source channel using the computed transform.
    deformedChannel = resampler.Execute(sourceChannel)  # Apply the transform to the current channel.

    # Update the deformed image with the resampled channel.
    deformed[:, :, i] = sitk.GetArrayFromImage(deformedChannel)  # Convert back to NumPy array and update.

  # Compute the deformation field.
  deformationField = sitk.TransformToDisplacementField(
    transform,
    outputPixelType=sitk.sitkVectorFloat64,  # Set the output pixel type to vector of float64 for the deformation field.
    size=targetChannel.GetSize(),  # Set the size of the deformation field to match the target image.
    outputOrigin=targetChannel.GetOrigin(),  # Set the output origin to match the target image.
    outputSpacing=targetChannel.GetSpacing(),  # Set the output spacing to match the target image.
    outputDirection=targetChannel.GetDirection(),  # Set the output direction to match the target image.
  )

  # Convert the deformation field to a NumPy array for further processing
  deformationFieldArray = sitk.GetArrayFromImage(deformationField)  # Shape: (Height, Width, 2)

  # Return the original source image, target image, deformed image, and deformation field array.
  return image1, image2, deformed.astype(np.uint8), deformationFieldArray


# Reads an image file and ignores ICC profile information.
def IgnoreICCFile(imgPath):
  r'''
  Reads an image file and ignores ICC profile information.
  This is useful for ensuring consistent color representation across different platforms.

  Parameters:
    imgPath (str): Path to the image file.
  '''

  # Open the image using PIL.
  img = PIL.Image.open(imgPath)
  # Remove ICC profile if it exists.
  img.info.pop("icc_profile", None)
  # Save the image without ICC profile.
  img.save(imgPath, quality=100)


# Check if a PNG image is not truncated by reading its header.
def CheckIfPNGImageIsNotTruncated(imgPath):
  r'''
  Check if a PNG image is not truncated by reading its header.
  This is useful for validating the integrity of PNG files.
  Returns True if the image is a valid PNG, False otherwise.
  If the file cannot be read, it returns False and prints an error message.

  Parameters:
    imgPath (str): Path to the PNG image file.

  Returns:
    bool: True if the image is a valid PNG, False otherwise.
  '''

  try:
    # Open the file in binary read mode.
    with open(imgPath, "rb") as f:
      # Read the first 8 bytes of the PNG file.
      header = f.read(8)
      # Check if the header matches the PNG signature.
      if (header != b"\x89PNG\r\n\x1a\n"):
        return False
      return True
  except Exception as e:
    print(f"Error reading PNG file: {e}")
    return False


# Fix a truncated PNG image by appending a valid PNG end chunk.
def FixTruncatedPNGImage(imgPath):
  r'''
  Fix a truncated PNG image by appending a valid PNG end chunk.
  This is useful for recovering partially downloaded or corrupted PNG files.

  Parameters:
    imgPath (str): Path to the PNG image file to be fixed.
  '''

  try:
    # Open the file in append mode.
    with open(imgPath, "ab") as f:
      # Append the PNG end chunk.
      f.write(b"\x00\x00\x00\x00IEND\xaeB`\x82")
      print(f"Fixed truncated PNG image: {imgPath}")
  except Exception as e:
    print(f"Error fixing truncated PNG image: {e}")


def LoadDicom(filePath, useVOILUT=True):
  r'''
  Load a DICOM file and extract its pixel array.

  Parameters:
    filePath (str): Path to the DICOM file.

  Returns:
    numpy.ndarray: The pixel data extracted from the DICOM file.
  '''

  import pydicom  # Import pydicom for handling DICOM files.
  from pydicom.pixels.processing import apply_voi_lut

  # Read the DICOM file.
  ds = pydicom.dcmread(filePath)

  # Extract the pixel array source the DICOM file.
  if (useVOILUT):
    # Apply VOI LUT if available for better visualization.
    image2D = apply_voi_lut(ds.pixel_array, ds)
  else:
    image2D = ds.pixel_array

  return image2D


def MinMaxNormalization(image, mapToUint8=True):
  r'''
  Normalize an image using min-max normalization.
  The pixel values are scaled to the range [0, 255] if mapToUint8 is True,
  otherwise they are scaled to the range [0, 1].

  Parameters:
    image (numpy.ndarray): Input image.
    mapToUint8 (bool): If True, the output image will be converted to uint8.

  Returns:
    numpy.ndarray: The normalized image. If mapToUint8 is True, the output will be of type uint8.
  '''

  # Calculate the delta between the maximum and minimum pixel intensities.
  delta = image.max() - image.min()

  # Normalize the image to range source 0 to 1.
  normSim = (image - image.min()) / delta

  if (not mapToUint8):
    return normSim

  # Scale the image to range source 0 to 255.
  normSim = normSim * 255.0

  # Convert the image to unsigned 8-bit integers.
  image = normSim.astype(np.uint8)

  return image


def CalculateCDF(image):
  r'''
  Calculate the cumulative distribution function (CDF) of an image.

  Parameters:
    image (numpy.ndarray): Input image.

  Returns:
    tuple: A typle containing:
      - cdfNormalized (numpy.ndarray): The normalized CDF of the image.
      - bins (numpy.ndarray): The bin values as integers.
  '''

  # Flatten the image to a 1D array.
  flattened = image.flatten()

  # Get the maximum and minimum pixel intensities.
  maxInt = int(flattened.max())
  minInt = int(flattened.min())

  # Compute the histogram and bin values using numpy's histogram function.
  # The number of bins is determined by the maximum value in the image.
  histogram, bins = np.histogram(
    flattened,  # Source image.
    bins=maxInt,  # Number of bins.
    range=[minInt, maxInt],  # Range of values.
  )

  # Calculate the cumulative sum of the histogram values.
  cdf = histogram.cumsum()

  # Normalize the cumulative sum to range source 0 to 1.
  cdfNormalized = cdf / cdf.max()

  # Return the normalized CDF and the bins as integers.
  return cdfNormalized, bins.astype(int)


def CalculateAverageCDFs(
  sourceFolder,  # The folder containing the source images.
  applyThresholding=False,  # If True, the source images will be thresholded.
  applyNormalization=False,  # If True, the source images will be normalized.
  isDicom=True,  # If True, the source files are DICOM files.
  specialIndex=None,  # If not None, only the files with the indices in this list will be used.
):
  r'''
  Calculate the average cumulative distribution functions (CDFs) for a set of images in a folder.

  Parameters:
    sourceFolder (str): Path to the folder containing the images.
    applyThresholding (bool): Whether to apply thresholding to the images.
    applyNormalization (bool): Whether to normalize the images.
    isDicom (bool): Whether the images are DICOM files.
    specialIndex (list or None): List of indices to process, or None to process all.

  Returns:
    tuple: A tuple containing:
      - avgCDFs (numpy.ndarray): The average CDFs across all images.
      - counts (numpy.ndarray): The counts of occurrences for each bin.
      - maxBins (int): The maximum number of bins used.
  '''

  # Get a sorted list of files in the "source" folder.
  sourceFiles = sorted(os.listdir(sourceFolder))

  # Check if the special index is inside the range of the "source" files range.
  if ((specialIndex is not None) and (max(specialIndex) >= len(sourceFiles))):
    return None

  # Initialize lists to store CDFs and bins.
  cdfs, bins = [], []

  # Initialize variables to keep track of maximum number of bins.
  maxBins = 0

  # Define the working range.
  workingRange = range(len(sourceFiles)) if specialIndex is None else specialIndex

  # Loop through each file in the "source" folder.
  for i in workingRange:
    # Get the file path
    filePath = os.path.join(sourceFolder, sourceFiles[i])

    if (isDicom):  # If the file is a DICOM file.
      image2D = LoadDicom(filePath)
    else:
      # Read the image file.
      image2D = cv2.imread(filePath)

    # Normalize the image if the applyNormalization flag is True.
    if (applyNormalization):
      image2D = MinMaxNormalization(image2D)

      # Convert the image to grayscale if it is not already.
    if (len(image2D.shape) > 2):
      # Convert the image to grayscale.
      image2D = cv2.cvtColor(image2D, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image if the applyThresholding flag is True.
    if (applyThresholding):
      # Apply thresholding to the image.
      _, image2D = cv2.threshold(
        image2D,  # Source image.
        0,  # Threshold value.
        255,  # Maximum value.
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,  # Thresholding type.
      )

    # Calculate the CDF and bins for the image.
    cdf_, bins_ = CalculateCDF(image2D)

    # Append the CDF and bins to the respective lists.
    cdfs.append(cdf_)
    bins.append(bins_[:-1])

    # Update the maximum number of bins if necessary.
    if (np.max(bins_) > maxBins):
      maxBins = int(np.max(bins_))

  # Initialize arrays to store cumulative CDFs and counts.
  cumCDFs = np.zeros(maxBins)
  counts = np.zeros(maxBins)

  # Loop through each CDF and its corresponding bins.
  for i in range(len(cdfs)):
    # Accumulate the CDF values based on the corresponding bins.
    cumCDFs[bins[i]] += cdfs[i]

    # Increment the counts for the corresponding bins.
    counts[bins[i]] += 1

  # Calculate the average CDFs by dividing cumulative CDFs by counts.
  avgCDFs = cumCDFs / counts

  # Return the average CDFs, counts, and maximum number of bins.
  return avgCDFs, counts, maxBins


def PriorInformationTrainingGeneric(
  image,  # The grayscale image.
  mask,  # The mask of the included region.
  startingRadius=5,  # The starting radius for drawing circles.
  stepRadius=5,  # The step size for increasing the radius.
  maxValue=255,  # The maximum intensity value.
):
  r'''
  Generate histograms for included and non-included regions at multiple radii for prior information training.

  Parameters:
    image (numpy.ndarray): Grayscale image.
    mask (numpy.ndarray): Mask of the included region.
    startingRadius (int): Starting radius for drawing circles.
    stepRadius (int): Step size for increasing the radius.
    maxValue (int): Maximum intensity value.

  Returns:
    list: List of dictionaries containing histograms and related information for each radius.
  '''

  # Calculate the moments of the included region mask to find the centroid.
  moments = cv2.moments(mask)
  centroidX = int(moments["m10"] / moments["m00"])  # Calculate the x coordinate of the centroid.
  centroidY = int(moments["m01"] / moments["m00"])  # Calculate the y coordinate of the centroid.

  # Find the maximum radius that can be drawn from the centroid to the image borders.
  maxRadius = np.min(
    [
      centroidX,  # Distance from centroid to the left border.
      centroidY,  # Distance from centroid to the top border.
      image.shape[0] - centroidX,  # Distance from centroid to the right border.
      image.shape[1] - centroidY,  # Distance from centroid to the bottom border.
    ]
  )

  # Create an empty list to store histograms.
  listOfHistograms = []

  # Iterate through different radii.
  for radius in range(startingRadius, maxRadius + 1, stepRadius):
    # Create an empty image to draw a white circle.
    circleImage = np.zeros_like(mask).astype(np.uint8)

    # Draw a white circle at the centroid with the specified radius.
    cv2.circle(circleImage, (centroidX, centroidY), radius, (255, 255, 255), -1)

    # Separate included and non-included pixels based on the circle mask.
    includedPixels = image[circleImage > 0]
    nonIncludedPixels = image[circleImage <= 0]

    # Calculate histograms for the included and non-included regions.
    histIncluded, binIncluded = np.histogram(
      includedPixels,  # Pixels from the included region.
      bins=np.max(image),  # Number of bins.
      range=(0, np.max(image)),  # Range of histogram bins.
    )

    histNonIncluded, binNonIncluded = np.histogram(
      nonIncludedPixels,  # Pixels from the non-included region.
      bins=np.max(image),  # Number of bins.
      range=(0, np.max(image)),  # Range of histogram bins.
    )

    # Add the histograms and relevant information to the list of histograms.
    listOfHistograms.append(
      {
        "included"       : histIncluded,
        "nonIncluded"    : histNonIncluded,
        "radius"         : radius,
        "includedBins"   : binIncluded,
        "nonIncludedBins": binNonIncluded,
        "maxIntensity"   : np.max(image),
        "centroidX"      : centroidX,
        "centroidY"      : centroidY,
      }
    )

  # Return the list of histograms for different radii.
  return listOfHistograms


def PriorInformationTestingGeneric(image, histogramsDict, startingSigma=1, stepSigma=1, position=0):
  r'''
  Generate probability maps for included and non-included regions using prior histograms.

  Parameters:
    image (numpy.ndarray): Grayscale image.
    histogramsDict (dict): Dictionary of histograms for different radii.
    startingSigma (int): Initial sigma for intensity range.
    stepSigma (int): Step size for increasing sigma.
    position (int): Progress bar position for tqdm.

  Returns:
    tuple: A tuple containing:
      - sumIncludedMaps (numpy.ndarray): The summed included probability map.
      - sumNonIncludedMaps (numpy.ndarray): The summed non-included probability map.
  '''
  import tqdm  # Import tqdm for progress bar.

  # Calculate the centroids of the image in the X and Y dimensions.
  centroidX, centroidY = image.shape[0] // 2, image.shape[1] // 2

  # Initialize two lists to store included and non-included probability maps.
  listOfIncludedMaps = []
  listOfNonIncludedMaps = []

  # Iterate through the radii in the histograms dictionary.
  for radius in tqdm.tqdm(
    list(histogramsDict.keys()),  # Get the list of radii from the histograms dictionary.
    desc=f"Prior Information ({position + 1})",  # Set a description for the progress bar.
    leave=(position == 0),  # Specify if the progress bar should leave (clear) when done.
    position=position,  # Set the position of the progress bar.
  ):
    # Get the included and non-included histograms for the current radius.
    histIncluded = histogramsDict[radius]["included"]
    histNonIncluded = histogramsDict[radius]["nonIncluded"]

    # Create an empty image to draw a white circle.
    circleImage = np.zeros_like(image)

    # Draw a white circle at the centroid with the specified radius.
    cv2.circle(circleImage, (centroidX, centroidY), radius, (255, 255, 255), -1)

    # Create empty probability maps for included and non-included.
    includedProbabilityMap = np.zeros_like(image, dtype=np.float32)
    nonIncludedProbabilityMap = np.zeros_like(image, dtype=np.float32)

    # Find the locations where the circle mask is not zero.
    locations = np.argwhere(circleImage > 0).tolist()

    # Iterate through the contour points.
    for point in locations:
      # Get the X and Y coordinates of the current point.
      x = point[1]
      y = point[0]

      # Get the intensity of the current point in the image.
      intensity = image[y, x]

      # Set the starting sigma for intensity range calculation.
      sigma = startingSigma

      # Initialize the starting histogram included range.
      histIncludedRange = 0

      # Calculate the intensity range for included probability while ensuring it's not zero.
      while (histIncludedRange == 0):
        # Create a range of intensities based on the current intensity and sigma value.
        rangeIntensity = np.arange(intensity - sigma, intensity + sigma + 1)

        # Filter out negative and out-of-range intensity values.
        rangeIntensity = rangeIntensity[(rangeIntensity >= 0) & (rangeIntensity < np.max(image))]

        # Calculate the included probability within the current intensity range.
        histIncludedRange = np.sum(histIncluded[rangeIntensity])

        # Increment sigma for the next iteration.
        sigma += stepSigma

      # Calculate the non-included probability within the same intensity range.
      histNonIncludedRange = np.sum(histNonIncluded[rangeIntensity])

      # Calculate the denominator of the probability map.
      den = histIncludedRange + histNonIncludedRange

      # Calculate the probability of being included at the current point.
      probIncluded = histIncludedRange / den

      # Calculate the probability of being non-included at the current point.
      probNonIncluded = histNonIncludedRange / den

      # Update the included and non-included probability maps with the calculated probabilities.
      includedProbabilityMap[y, x] = probIncluded
      nonIncludedProbabilityMap[y, x] = probNonIncluded

    # Normalize the included and non-included probability maps to the range [0, 1].
    includedProbabilityMap = MinMaxNormalization(includedProbabilityMap)
    nonIncludedProbabilityMap = MinMaxNormalization(nonIncludedProbabilityMap)

    # Add the probability maps to their respective lists.
    listOfIncludedMaps.append(includedProbabilityMap)
    listOfNonIncludedMaps.append(nonIncludedProbabilityMap)

  # Normalize the sum of included and non-included probability maps to the range [0, 1].
  sumIncludedMaps = np.zeros_like(image, dtype=np.float32)
  sumNonIncludedMaps = np.zeros_like(image, dtype=np.float32)

  # Calculate the sum of included maps.
  for includedMap in listOfIncludedMaps:
    sumIncludedMaps += includedMap

  # Calculate the sum of non-included maps.
  for nonIncludedMap in listOfNonIncludedMaps:
    sumNonIncludedMaps += nonIncludedMap

  # Normalize the sum of included and non-included probability maps.
  sumIncludedMaps = MinMaxNormalization(sumIncludedMaps)
  sumNonIncludedMaps = MinMaxNormalization(sumNonIncludedMaps)

  # Return the sum of included and non-included probability maps.
  return sumIncludedMaps, sumNonIncludedMaps


def PriorInformationGeneric(image, mask, startingRadius=10, stepRadius=10, startingSigma=1, stepSigma=1):
  r'''
  Generate lists of probability maps for included and non-included regions at multiple radii.

  Parameters:
    image (numpy.ndarray): Grayscale image.
    mask (numpy.ndarray): Mask of the included region.
    startingRadius (int): Starting radius for drawing circles.
    stepRadius (int): Step size for increasing the radius.
    startingSigma (int): Initial sigma for intensity range.
    stepSigma (int): Step size for increasing sigma.

  Returns:
    tuple: A tuple containing:
      - listOfIncludedMaps (list): List of included probability maps.
      - listOfNonIncludedMaps (list): List of non-included probability maps.
  '''

  listOfIncludedMaps = []
  listOfNonIncludedMaps = []

  # Find the centroid of the included region.
  moments = cv2.moments(mask)
  centroidX = int(moments["m10"] / moments["m00"])
  centroidY = int(moments["m01"] / moments["m00"])

  # Find the maximum radius that can be drawn from the centroid to the image borders.
  maxRadius = np.min(
    [
      centroidX,  # Distance from centroid to the left border.
      centroidY,  # Distance from centroid to the top border.
      image.shape[0] - centroidX,  # Distance from centroid to the right border.
      image.shape[1] - centroidY,  # Distance from centroid to the bottom border.
    ]
  )

  # Iterate through different radii.
  for radius in range(startingRadius, maxRadius + 1, stepRadius):
    # Create an empty image to draw a white circle.
    circleImage = np.zeros_like(mask)

    # Draw a white circle at the centroid with the specified radius.
    cv2.circle(circleImage, (centroidX, centroidY), radius, (255, 255, 255), -1)

    # Separate included and non-included pixels based on the circle mask.
    includedPixels = image[circleImage > 0]
    nonIncludedPixels = image[circleImage <= 0]

    # Calculate histograms for the included and non-included regions.
    histIncluded, binIncluded = np.histogram(
      includedPixels,  # Pixels from the included region.
      bins=np.max(image),  # Number of bins.
      range=(0, np.max(image)),  # Range of histogram bins.
    )

    histNonIncluded, binNonIncluded = np.histogram(
      nonIncludedPixels,  # Pixels from the non-included region.
      bins=np.max(image),  # Number of bins.
      range=(0, np.max(image)),  # Range of histogram bins.
    )

    # Create empty probability maps for included and non-included.
    includedProbabilityMap = np.zeros_like(image, dtype=np.float32)
    nonIncludedProbabilityMap = np.zeros_like(image, dtype=np.float32)

    # Find the locations where the circle mask is not zero.
    locations = np.argwhere(circleImage > 0).tolist()

    # Iterate through the contour points.
    for point in locations:
      # Get the X and Y coordinates of the current point.
      x = point[1]
      y = point[0]

      # Get the intensity of the current point in the image.
      intensity = image[y, x]

      # Set the starting sigma for intensity range calculation.
      sigma = startingSigma

      # Initialize the starting histogram included range.
      histIncludedRange = 0

      # Calculate the intensity range for included probability while ensuring it's not zero.
      while (histIncludedRange == 0):
        # Create a range of intensities based on the current intensity and sigma value.
        rangeIntensity = np.arange(intensity - sigma, intensity + sigma + 1)

        # Filter out negative and out-of-range intensity values.
        rangeIntensity = rangeIntensity[(rangeIntensity >= 0) & (rangeIntensity < np.max(image))]

        # Calculate the included probability within the current intensity range.
        histIncludedRange = np.sum(histIncluded[rangeIntensity])

        # Increment sigma for the next iteration.
        sigma += stepSigma

      # Calculate the non-included probability within the same intensity range.
      histNonIncludedRange = np.sum(histNonIncluded[rangeIntensity])

      # Calculate the denominator of the probability map.
      den = histIncludedRange + histNonIncludedRange

      # Calculate the probability of being included at the current point.
      probIncluded = histIncludedRange / den

      # Calculate the probability of being non-included at the current point.
      probNonIncluded = histNonIncludedRange / den

      # Update the included and non-included probability maps with the calculated probabilities.
      includedProbabilityMap[y, x] = probIncluded
      nonIncludedProbabilityMap[y, x] = probNonIncluded

    # Normalize the included and non-included probability maps to the range [0, 1].
    includedProbabilityMap = (
      (includedProbabilityMap - np.min(includedProbabilityMap)) /
      (np.max(includedProbabilityMap) - np.min(includedProbabilityMap))
    )
    nonIncludedProbabilityMap = (
      (nonIncludedProbabilityMap - np.min(nonIncludedProbabilityMap)) /
      (np.max(nonIncludedProbabilityMap) - np.min(nonIncludedProbabilityMap))
    )
    listOfIncludedMaps.append(includedProbabilityMap)
    listOfNonIncludedMaps.append(nonIncludedProbabilityMap)

  return listOfIncludedMaps, listOfNonIncludedMaps


def ReadRGBA(imgPath):
  r'''
  Read an image from a file and convert it to RGBA format.

  Parameters:
    imgPath (str): Path to the image file.

  Returns:
    numpy.ndarray: The image in RGBA format.
  '''

  img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)  # returns BGR or BGRA.
  if (img is None):
    raise FileNotFoundError(f"Image not found: `{imgPath}`")
  # If grayscale convert to RGBA.
  if (img.ndim == 2):
    rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
  elif (img.shape[2] == 4):
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
  else:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    rgba = np.concatenate((rgb, alpha), axis=2)
  return rgba


def SaveFigureRGBA(fig, savePath, dpi=720):
  r'''
  Save a Matplotlib figure as a PNG image with RGBA format.

  Parameters:
    fig (matplotlib.figure.Figure): The Matplotlib figure to save.
    savePath (str): Path to save the PNG image.
    dpi (int): Dots per inch for the saved image.
  '''

  from PIL import Image

  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  # Get ARGB buffer and convert to RGBA.
  argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
  rgba = argb[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA.
  im = Image.fromarray(rgba)
  im.save(savePath, format="PNG", dpi=(dpi, dpi))


def ComputeAndPlotDeformationFieldViaFarneback(
  img1,  # Path to the source image.
  img2,  # Path to the target image.
  step=10,  # Subsampling step for stream plot visualization.
  pyrScale=0.5,  # Pyramid scale for Farneback optical flow.
  levels=3,  # Number of pyramid levels for Farneback optical flow.
  winsize=15,  # Window size for Farneback optical flow.
  iterations=3,  # Number of iterations for Farneback optical flow.
  polyN=5,  # Size of the pixel neighborhood for polynomial expansion in Farneback optical flow.
  polySigma=1.2,  # Standard deviation of the Gaussian used to smooth derivatives in Farneback optical flow.
  flags=0,  # Flags for Farneback optical flow computation.
  backgroundAlpha=0.7,  # Alpha blending for the background image in the plot.
  density=1.5,  # Density parameter for plt.streamplot.
  addGradientBar=False,  # If True, adds a gradient color bar to the plot.
  sourceTitle="Source Image",  # Title for the source image subplot.
  targetTitle="Target Image",  # Title for the target image subplot.
  optFlowTitle="Deformation Field via Farneback Optical Flow",  # Title of the plot.
  showPlot=True,  # If True, displays the plot.
  savePlotPath=None,  # If provided, saves the plot to the specified path.
  cmap=plt.cm.viridis,  # Colormap for the plot.
  fontSize=12,  # Font size for plot labels.
  returnFigure=False,  # If True, returns the figure object.
):
  r'''
  Compute a dense deformation field between two images using Farneback optical flow
  and display a stream plot of the displacement vectors over the source image.

  Parameters:
    img1 (numpy.ndarray): Source image as a NumPy array.
    img2 (numpy.ndarray): Target image as a NumPy array.
    step (int): Subsampling step for plotting vectors to improve readability.
    pyrScale (float): Pyramid scale parameter for Farneback optical flow.
    levels (int): Number of pyramid levels for Farneback optical flow.
    winsize (int): Window size for Farneback optical flow.
    iterations (int): Number of iterations for Farneback optical flow.
    polyN (int): Size of the pixel neighborhood for polynomial expansion in Farneback optical flow.
    polySigma (float): Standard deviation of the Gaussian used to smooth derivatives in Farneback optical flow.
    flags (int): Flags for Farneback optical flow computation.    
    backgroundAlpha (float): Alpha blending for the background image in the plot.
    density (float): Density parameter passed to plt.streamplot for vector density.
    addGradientBar (bool): If True, adds a gradient color bar to the plot.
    sourceTitle (str): Title for the source image subplot.
    targetTitle (str): Title for the target image subplot.
    optFlowTitle (str): Title of the optical flow plot.
    showPlot (bool): If True, displays the plot.
    savePlotPath (str or None): If provided, saves the plot to the specified path.
    cmap (matplotlib.colors.Colormap): Colormap for the plot.
    fontSize (int): Font size for plot labels.
    returnFigure (bool): If True, returns the figure object.

  Returns:
    numpy.ndarray: Deformation field \(`H x W x 2`\) where channels are horizontal and vertical displacements.
  '''

  # Local import to avoid adding module at file top.
  import matplotlib.pyplot as plt

  if ((img1 is None) or (img2 is None)):
    raise ValueError("Failed to read one or both images. Ensure the files are valid image files.")

  # If sizes differ, resize target to match source for optical flow computation.
  if ((img1.shape[:2] != img2.shape[:2])):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)

  # Convert to grayscale for optical flow calculation.
  if (img1.shape[2] == 3):
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  elif (img1.shape[2] == 4):
    img1Gray = cv2.cvtColor(img1[..., :3], cv2.COLOR_RGB2GRAY)
  elif (img1.ndim == 2):
    img1Gray = img1
  else:
    raise ValueError("Source image must have 2 (grayscale), 3 (BGR) or 4 (BGRA) channels.")
  if (img2.shape[2] == 3):
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  elif (img2.shape[2] == 4):
    img2Gray = cv2.cvtColor(img2[..., :3], cv2.COLOR_RGB2GRAY)
  elif (img2.ndim == 2):
    img2Gray = img2
  else:
    raise ValueError("Target image must have 2 (grayscale), 3 (BGR) or 4 (BGRA) channels.")

  # Compute dense optical flow (Farneback).
  flow = cv2.calcOpticalFlowFarneback(
    img1Gray,  # Source image (grayscale).
    img2Gray,  # Target image (grayscale).
    None,  # Output flow (None to allocate new).
    pyr_scale=pyrScale,  # Pyramid scale.
    levels=levels,  # Number of pyramid levels.
    winsize=winsize,  # Window size.
    iterations=iterations,  # Number of iterations.
    poly_n=polyN,  # Size of pixel neighborhood.
    poly_sigma=polySigma,  # Standard deviation of Gaussian.
    flags=flags,  # Operation flags.
  )

  # Split into horizontal (u) and vertical (v) components.
  u = flow[..., 0]
  v = flow[..., 1]

  # Prepare downsampled grid for stream plot.
  step = max(1, int(step))
  x = np.arange(0, u.shape[1], step)
  y = np.arange(0, u.shape[0], step)
  X, Y = np.meshgrid(x, y)

  # Downsample the flow vectors for plotting.
  U = u[::step, ::step]
  V = v[::step, ::step]

  # Use default colormap if none provided.
  if (cmap is None):
    cmap = plt.cm.viridis

  # Create 1x3 subplots: original source, original target, streamplot.
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  ax0, ax1, ax2 = axes.ravel()

  # Left: source image.
  ax0.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
  ax0.set_title(sourceTitle, fontsize=fontSize * 1.2)
  ax0.axis("off")

  # Middle: target image.
  ax1.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
  ax1.set_title(targetTitle, fontsize=fontSize * 1.2)
  ax1.axis("off")

  # Right: streamplot overlaid on source.
  ax2.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), alpha=backgroundAlpha)

  # When gradient bar requested, color streamlines by magnitude; otherwise use a fixed color.
  if (addGradientBar):
    import matplotlib as mpl
    # Magnitude for downsampled vectors and full-field for normalization.
    mag = np.sqrt(U ** 2 + V ** 2)
    fullMagMax = float(np.max(np.sqrt(u ** 2 + v ** 2)))
    vmax = fullMagMax if (fullMagMax > 0) else 1e-8
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    # Draw streamlines colored by magnitude.
    strm = ax2.streamplot(
      X, Y, U, V,  # Flow field components.
      color=mag,  # Color by magnitude.
      cmap=cmap,  # Colormap for streamlines.
      norm=norm,  # Normalize colors to full-field magnitude.
      density=density,  # Density of the streamlines.
      linewidth=1,  # Line width of the streamlines.
      arrowsize=1.2,  # Arrow size scaling factor.
    )
    # Create scalar mappable for colorbar and attach to the same axes.
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar.
    cbar = fig.colorbar(
      sm,  # mappable for colorbar.
      ax=ax2,  # Axes to attach colorbar to.
      orientation="vertical",  # Vertical colorbar.
      fraction=0.046,  # Fraction of original axes size.
      pad=0.04,  # Padding between axes and colorbar.
    )
    cbar.set_label(
      "Displacement Magnitude",  # Colorbar label.
      rotation=270,  # Rotate label vertically.
      labelpad=15,  # Padding for label.
      fontsize=fontSize,  # Font size for label.
    )
  else:
    # Fixed-color streamplot (no colorbar).
    ax2.streamplot(
      X, Y, U, V,  # Flow field components.
      color="blue",  # Fixed color for streamlines.
      density=density,  # Density of the streamlines.
      linewidth=1,  # Line width of the streamlines.
      arrowsize=1.2,  # Arrow size scaling factor.
    )

  # Set title and remove axes for cleaner look.
  ax2.set_title(optFlowTitle, fontsize=fontSize * 1.2)
  ax2.axis("off")

  # Tight layout for better spacing.
  plt.tight_layout()

  if (savePlotPath is not None):
    fig.savefig(savePlotPath, bbox_inches="tight", dpi=720)
  if (showPlot):
    plt.show()
  plt.close(fig)

  # Return figure if requested.
  if (returnFigure):
    return flow, fig

  # Return the computed deformation field.
  return flow


# Overlay a heatmap on a PIL image and return the overlay image.
def OverlayHeatmapOnImage(origPIL, heatmap, alpha=0.4, cmap=plt.cm.jet):
  '''
  Overlay heatmap (2D array) over a PIL Image and return a PIL Image.

  Parameters:
    origPIL (PIL.Image): original RGB image.
    heatmap (numpy.ndarray): heatmap 2D array.
    alpha (float): overlay alpha.
    cmap (matplotlib.colors.Colormap): colormap to map heatmap values to colors.

  Returns.
    overlayPIL (PIL.Image): image with heatmap overlay.
  '''

  # Resize heatmap to image size and map to colors.
  heatmapImg = Image.fromarray(np.uint8(cmap(heatmap) * 255))
  heatmapImg = heatmapImg.resize(origPIL.size, resample=Image.BILINEAR)
  heatmapRGB = heatmapImg.convert("RGBA")

  base = origPIL.convert("RGBA")
  blended = Image.blend(base, heatmapRGB, alpha=alpha)
  return blended.convert("RGB")
