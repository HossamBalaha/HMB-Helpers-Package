# Import the required libraries.
import os  # Import os for file path operations.
import cv2, PIL  # Import OpenCV and PIL for image processing.
import numpy as np  # Import numpy for numerical operations.



def ReadVolume(caseImgPaths, caseSegPaths, raiseErrors=True):
  '''
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
  '''
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
  '''
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
  '''
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
  '''
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
  '''
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
    - :math: `H` is the homography matrix.
    - :math: `h_{ij}` are the elements of the homography matrix.

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
  '''
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
    - :math: `H` is the homography matrix.
    - :math: `h_{ij}` are the elements of the homography matrix.

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
  imagePath1,  # Path to the source image (HE image).
  imagePath2,  # Path to the target image (MT image).
  gridSize=[10, 10],  # Grid size for the B-spline transform.
  numberOfHistogramBins=50,  # Number of histogram bins for the metric.
  samplingPercentage=0.1,  # Percentage of pixels to sample for the metric.
  learningRate=0.01,  # Learning rate for the optimizer.
  numberOfIterations=500,  # Maximum number of iterations.
  convergenceMinimumValue=1e-6,  # Convergence threshold.
  convergenceWindowSize=10,  # Window size for convergence determination.
):
  '''
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
  '''
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
  '''
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
  '''
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
