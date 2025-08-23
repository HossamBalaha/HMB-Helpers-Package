'''
This module provides functions for calculating handcrafted features from images and their masks (if existing).

.. note::
  All functions in this module are authored by me and shared publicly mainly for educational purposes.

References:
   - **YouTube Playlist**: For detailed explanations and demonstrations of the functions, refer to the following playlist:
     `Handcrafted Features Tutorial
     <https://www.youtube.com/playlist?list=PLVrN2LRb7eT2GOJS8YKf1TcP6X1jr-9Dn>`_

   - **GitHub Repository**: For practical examples and code implementations, visit the repository:
     `BE-645 Artificial Intelligence and Radiomics
     <https://github.com/HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics>`_

Usage:
   These functions are designed to extract meaningful features from images,
   which can be used in various applications such as image analysis, machine learning, and radiomics.
'''
# Import the required libraries.
import cv2  # OpenCV for image processing.
import numpy as np  # NumPy for numerical operations.


# ===========================================================================================
# Function(s) for calculating first-order statistical features from an image.
# ===========================================================================================

def FirstOrderFeatures2D(img, mask, isNorm=True, ignoreZeros=True):
  '''
  Calculate first-order statistical features from an image using a mask.

  Parameters:
    img (numpy.ndarray): The input image as a 2D NumPy array.
    mask (numpy.ndarray): The binary mask as a 2D NumPy array.
    isNorm (bool): Flag to indicate whether to normalize the histogram.
    ignoreZeros (bool): Flag to indicate whether to ignore zeros in the histogram.

  Returns:
    tuple: A tuple containing:
      - dict: A dictionary containing the calculated first-order features.
      - numpy.ndarray: The histogram of the pixel values in the region of interest.
  '''

  # Extract the Region of Interest (ROI) using the mask.
  roi = cv2.bitwise_and(img, mask)  # Apply bitwise AND operation to extract the ROI.

  # Crop the ROI to remove unnecessary background.
  x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
  cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

  # Calculate the histogram of the cropped ROI.
  minVal = int(np.min(cropped))  # Find the minimum pixel value in the cropped ROI.
  maxVal = int(np.max(cropped))  # Find the maximum pixel value in the cropped ROI.
  hist2D = []  # Initialize an empty list to store the histogram values.

  # Loop through each possible value in the range [minVal, maxVal].
  for i in range(minVal, maxVal + 1):
    hist2D.append(np.count_nonzero(cropped == i))  # Count occurrences of the value `i` in the cropped ROI.
  hist2D = np.array(hist2D)  # Convert the histogram list to a NumPy array.

  # If ignoreZeros is True, set the first bin (background) to zero.
  if (ignoreZeros and (minVal == 0)):
    # Ignore the background (assumed to be the first bin in the histogram).
    hist2D = hist2D[1:]  # Remove the first bin (background).
    minVal += 1  # Adjust the minimum value to exclude the background.

  # Calculate the total count of values in the histogram before normalization.
  freqCount = np.sum(hist2D)  # Sum all frequencies in the histogram.

  # Normalize the histogram if the flag is set.
  if (isNorm):
    # Normalize the histogram to represent probabilities.
    hist2D = hist2D / np.sum(hist2D)  # Divide each bin by the total count to normalize.

  # Calculate the total count of values from the histogram after normalization.
  count = np.sum(hist2D)  # Sum all probabilities in the normalized histogram.

  # Determine the range of values in the histogram.
  rng = np.arange(minVal, maxVal + 1)  # Create an array of values from `minVal` to `maxVal`.

  # Calculate the sum of values from the histogram.
  sumVal = np.sum(hist2D * rng)  # Multiply each value by its frequency and sum the results.

  # Calculate the mean (average) value from the histogram.
  mean = sumVal / count  # Divide the total sum by the total count.

  # Calculate the variance from the histogram.
  variance = np.sum(hist2D * (rng - mean) ** 2) / count  # Measure of the spread of the data.

  # Calculate the standard deviation from the histogram.
  stdDev = np.sqrt(variance)  # Square root of the variance.

  # Calculate the skewness from the histogram.
  skewness = np.sum(hist2D * (rng - mean) ** 3) / (count * stdDev ** 3)  # Measure of asymmetry in the data.

  # Calculate the kurtosis from the histogram.
  kurtosis = np.sum(hist2D * (rng - mean) ** 4) / (count * stdDev ** 4)  # Measure of the "tailedness" of the data.

  # Calculate the excess kurtosis from the histogram.
  exKurtosis = kurtosis - 3  # Excess kurtosis relative to a normal distribution.

  # Store the results in a dictionary.
  results = {
    "Min"               : minVal,  # Minimum pixel value.
    "Max"               : maxVal,  # Maximum pixel value.
    "Count"             : count,  # Total count of pixels after normalization.
    "Frequency Count"   : freqCount,  # Total count of pixels before normalization.
    "Sum"               : sumVal,  # Sum of pixel values.
    "Mean"              : mean,  # Mean pixel value.
    "Variance"          : variance,  # Variance of pixel values.
    "Standard Deviation": stdDev,  # Standard deviation of pixel values.
    "Skewness"          : skewness,  # Skewness of pixel values.
    "Kurtosis"          : kurtosis,  # Kurtosis of pixel values.
    "Excess Kurtosis"   : exKurtosis,  # Excess kurtosis of pixel values.
  }

  return results, hist2D


def FirstOrderFeatures2DV2(data, isNorm=True, ignoreZeros=True):
  '''
  Calculate first-order statistical features from an image using a mask.

  Parameters:
    data (numpy.ndarray): The input image as a 2D NumPy array.
    isNorm (bool): Flag to indicate whether to normalize the histogram.
    ignoreZeros (bool): Flag to indicate whether to ignore zeros in the histogram.

  Returns:
    tuple: A tuple containing:
      - dict: A dictionary containing the calculated first-order features.
      - numpy.ndarray: The histogram of the pixel values in the region of interest.
  '''

  # Calculate the histogram of the cropped ROI.
  minVal = int(np.min(data))  # Find the minimum pixel value in the cropped ROI.
  maxVal = int(np.max(data))  # Find the maximum pixel value in the cropped ROI.
  hist2D = []  # Initialize an empty list to store the histogram values.

  # Loop through each possible value in the range [minVal, maxVal].
  for i in range(minVal, maxVal + 1):
    hist2D.append(np.count_nonzero(data == i))  # Count occurrences of the value `i` in the cropped ROI.
  hist2D = np.array(hist2D)  # Convert the histogram list to a NumPy array.

  # If ignoreZeros is True, set the first bin (background) to zero.
  if (ignoreZeros and (minVal == 0)):
    # Ignore the background (assumed to be the first bin in the histogram).
    hist2D = hist2D[1:]  # Remove the first bin (background).
    minVal += 1  # Adjust the minimum value to exclude the background.

  # Calculate the total count of values in the histogram before normalization.
  freqCount = np.sum(hist2D)  # Sum all frequencies in the histogram.

  # Normalize the histogram if the flag is set.
  if (isNorm):
    # Normalize the histogram to represent probabilities.
    hist2D = hist2D / np.sum(hist2D)  # Divide each bin by the total count to normalize.

  # Calculate the total count of values from the histogram after normalization.
  count = np.sum(hist2D)  # Sum all probabilities in the normalized histogram.

  # Determine the range of values in the histogram.
  rng = np.arange(minVal, maxVal + 1)  # Create an array of values from `minVal` to `maxVal`.

  # Calculate the sum of values from the histogram.
  sumVal = np.sum(hist2D * rng)  # Multiply each value by its frequency and sum the results.

  # Calculate the mean (average) value from the histogram.
  mean = sumVal / count  # Divide the total sum by the total count.

  # Calculate the variance from the histogram.
  variance = np.sum(hist2D * (rng - mean) ** 2) / count  # Measure of the spread of the data.

  # Calculate the standard deviation from the histogram.
  stdDev = np.sqrt(variance)  # Square root of the variance.

  # Calculate the skewness from the histogram.
  skewness = np.sum(hist2D * (rng - mean) ** 3) / (count * stdDev ** 3)  # Measure of asymmetry in the data.

  # Calculate the kurtosis from the histogram.
  kurtosis = np.sum(hist2D * (rng - mean) ** 4) / (count * stdDev ** 4)  # Measure of the "tailedness" of the data.

  # Calculate the excess kurtosis from the histogram.
  exKurtosis = kurtosis - 3  # Excess kurtosis relative to a normal distribution.

  # Store the results in a dictionary.
  results = {
    "Min"               : minVal,  # Minimum pixel value.
    "Max"               : maxVal,  # Maximum pixel value.
    "Count"             : count,  # Total count of pixels after normalization.
    "Frequency Count"   : freqCount,  # Total count of pixels before normalization.
    "Sum"               : sumVal,  # Sum of pixel values.
    "Mean"              : mean,  # Mean pixel value.
    "Variance"          : variance,  # Variance of pixel values.
    "Standard Deviation": stdDev,  # Standard deviation of pixel values.
    "Skewness"          : skewness,  # Skewness of pixel values.
    "Kurtosis"          : kurtosis,  # Kurtosis of pixel values.
    "Excess Kurtosis"   : exKurtosis,  # Excess kurtosis of pixel values.
  }

  return results, hist2D


# ===========================================================================================
# Function(s) for calculating Gray-Level Co-occurrence Matrix (GLCM) and its features.
# ===========================================================================================

def CalculateGLCMCooccuranceMatrix(image, d, theta, isSymmetric=False, isNorm=True, ignoreZeros=True, epsilon=1e-6):
  '''
  Calculate the Gray-Level Co-occurrence Matrix (GLCM) for a given image.

  Parameters:
    image (numpy.ndarray): The input image as a 2D NumPy array.
    d (int): The distance between pixel pairs.
    theta (float): The angle (in radians) for the direction of pixel pairs.
    isSymmetric (bool): Whether to make the GLCM symmetric. Default is False.
    isNorm (bool): Whether to normalize the GLCM. Default is True.
    ignoreZeros (bool): Whether to ignore zero-valued pixels. Default is True.
    epsilon (float): A small value to avoid division by zero during normalization. Default is 1e-6.

  Returns:
    coMatrix (numpy.ndarray): The calculated GLCM.

  Raises:
    ValueError: If the distance 'd' is less than 1 or greater than or equal to the number of unique intensity levels.
  '''

  # Determine the number of unique intensity levels in the matrix.
  minA = np.min(image)  # Minimum intensity value.
  maxA = np.max(image)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  if (d < 1):
    raise ValueError("The distance between voxel pairs should be greater than or equal to 1.")
  elif (d >= N):
    raise ValueError("The distance between voxel pairs should be less than the number of unique intensity levels.")

  # Initialize the co-occurrence matrix with zeros.
  coMatrix = np.zeros((N, N))  # Create an N x N matrix filled with zeros.

  # Iterate over each pixel in the image to calculate the GLCM.
  for xLoc in range(image.shape[1]):  # Loop through columns.
    for yLoc in range(image.shape[0]):  # Loop through rows.
      startLoc = (yLoc, xLoc)  # Current pixel location (row, column).

      # Calculate the target pixel location based on distance and angle.
      xTarget = xLoc + np.round(d * np.cos(theta))  # Target column.
      yTarget = yLoc - np.round(d * np.sin(theta))  # Target row.
      endLoc = (int(yTarget), int(xTarget))  # Target pixel location.

      # Check if the target location is within the bounds of the image.
      if (
        (endLoc[0] < 0)  # Target row is above the top edge.
        or (endLoc[0] >= image.shape[0])  # Target row is below the bottom edge.
        or (endLoc[1] < 0)  # Target column is to the left of the left edge.
        or (endLoc[1] >= image.shape[1])  # Target column is to the right of the right edge.
      ):
        continue  # Skip this pair if the target is out of bounds.

      if (ignoreZeros):
        # Skip the calculation if the pixel values are zero.
        if ((image[startLoc] == 0) or (image[endLoc] == 0)):
          continue

      # (- minA) is added to work with matrices that does not start from 0.
      # Increment the count for the pair (start, end).
      # image[startLoc] and image[endLoc] are the intensity values at the start and end locations.
      startPixel = image[startLoc] - minA  # Adjust start pixel value.
      endPixel = image[endLoc] - minA  # Adjust end pixel value.

      # Increment the co-occurrence matrix at the corresponding location.
      coMatrix[endPixel, startPixel] += 1

  # If symmetric, add the transpose of the co-occurrence matrix to itself.
  if (isSymmetric):
    coMatrix += coMatrix.T  # Make the GLCM symmetric.

  # Normalize the co-occurrence matrix if requested.
  if (isNorm):
    # Divide each element by the sum of all elements.
    # epsilon is added to avoid division by zero.
    coMatrix = coMatrix / (np.sum(coMatrix) + epsilon)

  return coMatrix  # Return the calculated GLCM.


def CalculateGLCMCooccuranceMatrix3D(volume, d, theta, isSymmetric=False, isNorm=True, ignoreZeros=True, epsilon=1e-6):
  '''
  Calculate the 3D Gray-Level Co-occurrence Matrix (GLCM) for a given volume.

  Parameters:
    volume (numpy.ndarray): The 3D volume as a NumPy array.
    d (int): The distance between voxel pairs.
    theta (float): The angle (in radians) for the direction of voxel pairs.
    isSymmetric (bool): Whether to make the GLCM symmetric. Default is False.
    isNorm (bool): Whether to normalize the GLCM. Default is True.
    ignoreZeros (bool): Whether to ignore zero-valued voxels. Default is True.
    epsilon (float): A small value to avoid division by zero during normalization. Default is 1e-6.

  Returns:
    coMatrix (numpy.ndarray): The calculated GLCM.

  Raises:
    ValueError: If the distance 'd' is less than 1, greater than or equal to the number of slices,
      or greater than or equal to the number of unique intensity levels.
  '''

  # Determine the number of unique intensity levels in the volume.
  minA = np.min(volume)  # Minimum intensity value.
  maxA = np.max(volume)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  noOfSlices = volume.shape[0]  # Number of slices in the volume.

  # Initialize the co-occurrence matrix with zeros.
  coMatrix = np.zeros((N, N))

  if (d < 1):
    raise ValueError("The distance between voxel pairs should be greater than or equal to 1.")
  elif (d >= noOfSlices):
    raise ValueError("The distance between voxel pairs should be less than the number of slices.")
  elif (d >= N):
    raise ValueError("The distance between voxel pairs should be less than the number of unique intensity levels.")

  # Iterate over each voxel in the volume to calculate the GLCM.
  for xLoc in range(volume.shape[2]):  # Loop through columns.
    for yLoc in range(volume.shape[1]):  # Loop through rows.
      for zLoc in range(volume.shape[0]):  # Loop through slices.
        startLoc = (zLoc, yLoc, xLoc)  # Current voxel location (slice, row, column).

        # Calculate the target voxel location based on distance and angle.
        xTarget = xLoc + np.round(d * np.cos(theta) * np.sin(theta))  # Target column.
        yTarget = yLoc - np.round(d * np.sin(theta) * np.sin(theta))  # Target row.
        zTarget = zLoc + np.round(d * np.cos(theta))  # Target slice.
        endLoc = (int(zTarget), int(yTarget), int(xTarget))  # Target voxel location.

        # Check if the target location is within the bounds of the volume.
        if (
          (endLoc[0] < 0)  # Target slice is below the bottom slice.
          or (endLoc[0] >= volume.shape[0])  # Target slice is above the top slice.
          or (endLoc[1] < 0)  # Target row is above the top edge.
          or (endLoc[1] >= volume.shape[1])  # Target row is below the bottom edge.
          or (endLoc[2] < 0)  # Target column is to the left of the left edge.
          or (endLoc[2] >= volume.shape[2])  # Target column is to the right of the right edge.
        ):
          continue  # Skip this pair if the target is out of bounds.

        if (ignoreZeros):
          # Skip the calculation if the pixel values are zero.
          if ((volume[startLoc] == 0) or (volume[endLoc] == 0)):
            continue

        # (- minA) is added to work with matrices that does not start from 0.
        # Increment the count for the pair (start, end).
        # volume[startLoc] and volume[endLoc] are the intensity values at the start and end locations.
        startPixel = volume[startLoc] - minA  # Adjust start pixel value.
        endPixel = volume[endLoc] - minA  # Adjust end pixel value.

        # Increment the co-occurrence matrix at the corresponding location.
        coMatrix[endPixel, startPixel] += 1

  # If symmetric, add the transpose of the co-occurrence matrix to itself.
  if (isSymmetric):
    coMatrix += coMatrix.T  # Make the GLCM symmetric.

  # Normalize the co-occurrence matrix if requested.
  if (isNorm):
    # Divide each element by the sum of all elements.
    # epsilon is added to avoid division by zero.
    coMatrix = coMatrix / (np.sum(coMatrix) + epsilon)

  return coMatrix  # Return the calculated GLCM.


def CalculateGLCMFeaturesOptimized(coMatrix):
  '''
  Calculate texture features from a Gray-Level Co-occurrence Matrix (GLCM).

  Parameters:
    coMatrix (numpy.ndarray): The GLCM as a 2D NumPy array.

  Returns:
    dict: A dictionary containing the calculated texture features. This includes:
      - Energy: Measure of textural uniformity.
      - Contrast: Measure of local intensity variation.
      - Homogeneity: Measure of closeness of the distribution of elements in the GLCM to the GLCM diagonal.
      - Entropy: Measure of randomness in the texture.
      - Correlation: Measure of how correlated a pixel is to its neighbor over the whole image.
      - Dissimilarity: Measure of how different the elements of the GLCM are from each other.
      - TotalSum: Sum of all elements in the GLCM.
      - MeanX: Mean of the rows in the GLCM.
      - MeanY: Mean of the columns in the GLCM.
      - StdDevX: Standard deviation of the rows in the GLCM.
      - StdDevY: Standard deviation of the columns in the GLCM.
  '''

  N = coMatrix.shape[0]  # Number of unique intensity levels.

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)  # Sum of the squares of all elements in the GLCM.

  # Initialize variables for texture features.
  contrast = 0.0  # Initialize contrast.
  homogeneity = 0.0  # Initialize homogeneity.
  entropy = 0.0  # Initialize entropy.
  dissimilarity = 0.0  # Initialize dissimilarity.
  meanX = 0.0  # Initialize mean of rows.
  meanY = 0.0  # Initialize mean of columns.

  # Loop through each element in the GLCM to calculate texture features.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      # Calculate the contrast in the direction of theta.
      contrast += (i - j) ** 2 * coMatrix[i, j]  # Weighted sum of squared differences.

      # Calculate the homogeneity of the co-occurrence matrix.
      homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)  # Weighted sum of inverse differences.

      # Calculate the entropy of the co-occurrence matrix.
      if (coMatrix[i, j] > 0):  # Check if the value is greater than zero.
        entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])  # Sum of -p * log(p).

      # Calculate the dissimilarity of the co-occurrence matrix.
      dissimilarity += np.abs(i - j) * coMatrix[i, j]  # Weighted sum of absolute differences.

      # Calculate the mean of the co-occurrence matrix.
      meanX += i * coMatrix[i, j]  # Weighted sum of row indices.
      meanY += j * coMatrix[i, j]  # Weighted sum of column indices.

  totalSum = np.sum(coMatrix)  # Calculate the sum of all elements in the GLCM.
  meanX /= totalSum  # Calculate mean of rows.
  meanY /= totalSum  # Calculate mean of columns.

  # Calculate the standard deviation of rows and columns.
  stdDevX = 0.0  # Initialize standard deviation of rows.
  stdDevY = 0.0  # Initialize standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      stdDevX += (i - meanX) ** 2 * coMatrix[i, j]  # Weighted sum of squared row differences.
      stdDevY += (j - meanY) ** 2 * coMatrix[i, j]  # Weighted sum of squared column differences.

  # Calculate the correlation of the co-occurrence matrix.
  correlation = 0.0  # Initialize correlation.
  stdDevX = np.sqrt(stdDevX)  # Calculate standard deviation of rows.
  stdDevY = np.sqrt(stdDevY)  # Calculate standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      correlation += (
        (i - meanX) * (j - meanY) * coMatrix[i, j] / (stdDevX * stdDevY)
      )  # Weighted sum of normalized differences.

  # Return the calculated features as a dictionary.
  return {
    "Energy"       : energy,  # Energy of the GLCM.
    "Contrast"     : contrast,  # Contrast of the GLCM.
    "Homogeneity"  : homogeneity,  # Homogeneity of the GLCM.
    "Entropy"      : entropy,  # Entropy of the GLCM.
    "Correlation"  : correlation,  # Correlation of the GLCM.
    "Dissimilarity": dissimilarity,  # Dissimilarity of the GLCM.
    "TotalSum"     : totalSum,  # Sum of all elements in the GLCM.
    "MeanX"        : meanX,  # Mean of rows.
    "MeanY"        : meanY,  # Mean of columns.
    "StdDevX"      : stdDevX,  # Standard deviation of rows.
    "StdDevY"      : stdDevY,  # Standard deviation of columns.
  }


# ===========================================================================================
# Function(s) for calculating Gray-Level Run-Length Matrix (GLRLM) and its features.
# ===========================================================================================

def CalculateGLRLMRunLengthMatrix(matrix, theta, isNorm=True, ignoreZeros=True, epsilon=1e-6):
  '''
  Calculate the Gray-Level Run-Length Matrix (GLRLM) for a given 2D matrix.
  The GLRLM is a statistical tool used to quantify the texture of an image by
  analyzing the runs of pixels with the same intensity level in a specific direction.

  Parameters:
    matrix (numpy.ndarray): A 2D matrix representing the image or data for which the GLRLM is to be calculated.
    theta (float): The angle (in radians) specifying the direction in which runs are to be analyzed.
      The direction is determined by the cosine and sine of this angle.
    isNorm (bool): If True, the resulting GLRLM is normalized by dividing by the total number of runs.
      Normalization ensures that the matrix represents probabilities rather than counts. Default is True.
    ignoreZeros (bool): If True, runs with zero intensity are ignored in the calculation of the GLRLM.
      This is useful when zero values represent background or irrelevant data. Default is True.
    epsilon (float): A small value added to the denominator during normalization to prevent division by zero.
      Default is 1e-6.

  Returns:
    numpy.ndarray: A 2D matrix representing the Gray-Level Run-Length Matrix. The rows correspond to intensity levels, and the columns correspond to run lengths. If `isNorm` is True, the matrix is normalized.
  '''

  # Calculate minimum intensity value in the input matrix for intensity range adjustment.
  minA = np.min(matrix)
  # Calculate maximum intensity value in the input matrix for intensity range adjustment.
  maxA = np.max(matrix)
  # Determine total number of distinct gray levels by calculating intensity range span.
  N = maxA - minA + 1
  # Find maximum potential run length based on largest matrix dimension.
  R = np.max(matrix.shape)

  # Initialize empty GLRLM matrix with dimensions (intensity levels × max run length).
  rlMatrix = np.zeros((N, R))
  # Create tracking matrix to prevent duplicate processing of pixels in runs.
  seenMatrix = np.zeros(matrix.shape)
  # Calculate x-direction step using cosine (negative for coordinate system alignment).
  dx = int(np.round(np.cos(theta)))
  # Calculate y-direction step using sine of the analysis angle.
  dy = int(np.round(np.sin(theta)))

  # Adjust direction for specific angles to ensure consistent run direction.
  if (theta in [np.radians(45), np.radians(135)]):
    dx = -dx  # Adjust x-direction for 45 and 135 degrees.
    dy = dy  # Keep y-direction unchanged for 45 and 135 degrees.

  # Iterate through each row index of the input matrix.
  for i in range(matrix.shape[0]):
    # Iterate through each column index of the input matrix.
    for j in range(matrix.shape[1]):
      # Skip already processed pixels to prevent duplicate counting.
      if (seenMatrix[i, j] == 1):
        continue

      # Mark current pixel as processed in tracking matrix.
      seenMatrix[i, j] = 1
      # Store intensity value of current pixel for run comparison.
      currentPixel = matrix[i, j]
      # Initialize run length counter for current streak.
      d = 1

      # Explore consecutive pixels in specified direction until boundary or value change.
      while (
        (i + d * dy >= 0) and
        (i + d * dy < matrix.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < matrix.shape[1])
      ):
        # Check if subsequent pixel matches current intensity value.
        if (matrix[i + d * dy, j + d * dx] == currentPixel):
          # Mark matching pixel as processed in tracking matrix.
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
          # Increment run length counter for continued streak.
          d += 1
        else:
          # Exit loop when streak breaks (different value encountered).
          break

      # Skip zero-value runs if configured to ignore background.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Update GLRLM by incrementing count at corresponding intensity-runlength position.
      # (Adjust intensity index by minimum value for proper matrix positioning).
      rlMatrix[currentPixel - minA, d - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if (isNorm):
    # Add small epsilon to prevent division by zero in empty matrices.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + epsilon)

  # Return computed Gray-Level Run-Length Matrix.
  return rlMatrix


def CalculateGLRLMFeatures(rlMatrix, image):
  '''
  Calculate texture features from a Gray-Level Run-Length Matrix (GLRLM).
  This function computes various texture features based on the GLRLM, which is derived
  from an image. These features are commonly used in texture analysis and image processing.

  Parameters:
    rlMatrix (numpy.ndarray): A 2D Gray-Level Run-Length Matrix (GLRLM) computed from an image.
      The rows represent intensity levels, and the columns represent run lengths.
    image (numpy.ndarray): The original 2D image from which the GLRLM was derived. This is used to determine
      the number of gray levels and the total number of pixels.

  Returns:
    dict: A dictionary containing the following texture features:
      - "Short Run Emphasis"          : Emphasizes short runs in the image.
      - "Long Run Emphasis"           : Emphasizes long runs in the image.
      - "Gray Level Non-Uniformity"   : Measures the variability of gray levels.
      - "Run Length Non-Uniformity"   : Measures the variability of run lengths.
      - "Run Percentage"              : Ratio of runs to the total number of pixels.
      - "Low Gray Level Run Emphasis" : Emphasizes runs with low gray levels.
      - "High Gray Level Run Emphasis": Emphasizes runs with high gray levels.
  '''

  # Determine minimum intensity value in the original image.
  minA = np.min(image)
  # Determine maximum intensity value in the original image.
  maxA = np.max(image)
  # Calculate total number of distinct gray levels in the image.
  N = maxA - minA + 1
  # Get maximum possible run length from image dimensions.
  R = np.max(image.shape)

  # Calculate total number of runs recorded in the GLRLM.
  rlN = np.sum(rlMatrix)

  # Calculate Short Run Emphasis (SRE) emphasizing shorter runs through inverse squared weighting.
  sre = np.sum(
    rlMatrix / (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Calculate Long Run Emphasis (LRE) emphasizing longer runs through squared weighting.
  lre = np.sum(
    rlMatrix * (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Calculate Gray Level Non-Uniformity (GLN) measuring gray level distribution consistency.
  gln = np.sum(
    np.sum(rlMatrix, axis=1) ** 2,  # Row sums squared
  ) / rlN

  # Calculate Run Length Non-Uniformity (RLN) measuring run length distribution consistency.
  rln = np.sum(
    np.sum(rlMatrix, axis=0) ** 2,  # Column sums squared
  ) / rlN

  # Calculate Run Percentage (RP) indicating proportion of image occupied by runs.
  rp = rlN / np.prod(image.shape)

  # Calculate Low Gray Level Run Emphasis (LGRE) weighting low intensities more heavily.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # Calculate High Gray Level Run Emphasis (HGRE) weighting high intensities more heavily.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # Package computed features into a dictionary with descriptive keys.
  return {
    "Total Runs"                         : rlN,
    "Short Run Emphasis (SRE)"           : sre,
    "Long Run Emphasis (LRE)"            : lre,
    "Gray Level Non-Uniformity (GLN)"    : gln,
    "Run Length Non-Uniformity (RLN)"    : rln,
    "Run Percentage (RP)"                : rp,
    "Low Gray Level Run Emphasis (LGRE)" : lgre,
    "High Gray Level Run Emphasis (HGRE)": hgre,
  }


def CalculateGLRLMRunLengthMatrix3D(volume, theta, isNorm=True, ignoreZeros=True, epsilon=1e-6):
  '''
  Calculate 3D Gray-Level Run-Length Matrix (GLRLM) for volumetric texture analysis.

  Parameters:
    volume (numpy.ndarray): 3D array of intensity values (z, y, x dimensions).
    theta (float): Analysis angle in radians determining 3D direction vector.
    isNorm (bool): Enable matrix normalization to probability distribution. Default is True.
    ignoreZeros (bool): Exclude zero-valued voxels from run calculations. Default is True.
    epsilon (float): Small value to prevent division by zero during normalization. Default is 1e-6.

  Returns:
    rlMatrix (numpy.ndarray): 2D matrix of size (intensity levels × max run length).
  '''

  # Calculate intensity range parameters for matrix indexing.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1  # Number of discrete intensity levels
  R = np.max(volume.shape)  # Maximum possible run length

  # Initialize empty GLRLM and pixel tracking matrix.
  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(volume.shape)

  # Calculate directional components using spherical coordinates.
  dx = int(np.round(np.cos(theta) * np.sin(theta)))  # X-axis step
  dy = int(np.round(np.sin(theta) * np.sin(theta)))  # Y-axis step
  dz = int(np.round(np.cos(theta)))  # Z-axis step

  # Iterate through all voxels in 3D volume.
  for i in range(volume.shape[0]):  # Z-dimension
    for j in range(volume.shape[1]):  # Y-dimension
      for k in range(volume.shape[2]):  # X-dimension
        # Skip previously processed voxels.
        if (seenMatrix[i, j, k] == 1):
          continue

        # Mark current voxel as processed.
        seenMatrix[i, j, k] = 1
        currentVal = volume[i, j, k]
        runLength = 1

        # Extend run along specified direction until value change.
        while (
          (i + runLength * dz >= 0) and
          (i + runLength * dz < volume.shape[0]) and
          (j + runLength * dy >= 0) and
          (j + runLength * dy < volume.shape[1]) and
          (k + runLength * dx >= 0) and
          (k + runLength * dx < volume.shape[2])
        ):
          if (volume[i + runLength * dz, j + runLength * dy, k + runLength * dx] == currentVal):
            seenMatrix[i + runLength * dz, j + runLength * dy, k + runLength * dx] = 1
            runLength += 1
          else:
            break

        # Skip zero-value runs if configured.
        if (ignoreZeros and currentVal == 0):
          continue

        # Update GLRLM with current run information.
        # (Adjust intensity index by minimum value for proper matrix positioning).
        rlMatrix[currentVal - minA, runLength - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if (isNorm):
    # Add epsilon to avoid division by zero.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + epsilon)

  return rlMatrix


# ===========================================================================================
# Function(s) for calculating Gray-Level Size-Zone Matrix (GLSZM) and its features.
# ===========================================================================================

def FindConnectedRegions(image, connectivity=4):
  '''
  Finds connected regions in a 2D image based on pixel connectivity.

  Parameters:
    image (numpy.ndarray): A 2D NumPy array representing the input image. Each element represents a pixel value.
    connectivity (int): The type of connectivity to use for determining connected regions. Options are (a)
      4 for 4-connectivity (up, down, left, right) and (b) 8 for 8-connectivity (includes diagonals).

  Returns:
    dict: A dictionary where keys are unique pixel values from the image, and values are lists of sets. Each set contains the coordinates (i, j) of pixels belonging to a connected region for that pixel value.
  '''

  def RecursiveHelper(i, j, currentPixel, region, seenMatrix, connectivity=4):
    '''
    Recursive helper function to find all connected pixels for a given starting pixel.

    Parameters:
      i (int): Row index of the current pixel.
      j (int): Column index of the current pixel.
      currentPixel (int): The pixel value being processed.
      region (set): A set to store the coordinates of connected pixels.
      seenMatrix (numpy.ndarray): A 2D matrix to track visited pixels.
      connectivity (int): The type of connectivity (4 or 8).
    '''

    # Check if the current pixel is out of bounds, already seen, or not matching the current pixel value.
    if (
      (i < 0) or  # Check if row index is out of bounds.
      (i >= image.shape[0]) or
      (j < 0) or
      (j >= image.shape[1]) or
      (image[i, j] != currentPixel) or  # Check if pixel value matches the current pixel value.
      ((i, j) in region) or  # Check if the pixel has already been added to the region.
      (seenMatrix[i, j] == 1)  # Check if the pixel has already been seen.
    ):
      return  # Exit if any condition is met.

    # Add the current pixel to the region and mark it as seen.
    region.add((i, j))
    seenMatrix[i, j] = 1

    # Recursively check the neighboring pixels (up, left, down, right).
    RecursiveHelper(i - 1, j, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i, j - 1, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i + 1, j, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i, j + 1, currentPixel, region, seenMatrix, connectivity)

    # If 8-connectivity is specified, also check diagonal neighbors.
    if (connectivity == 8):
      RecursiveHelper(i - 1, j - 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i - 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i + 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i + 1, j - 1, currentPixel, region, seenMatrix, connectivity)

  # Initialize a matrix to keep track of seen pixels.
  seenMatrix = np.zeros(image.shape)

  # Dictionary to store regions grouped by pixel values.
  regions = {}

  # Iterate over each pixel in the image.
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      # Skip if the pixel has already been processed.
      if (seenMatrix[i, j]):
        continue

      # Get the current pixel value.
      currentPixel = image[i, j]

      # Initialize a list for this pixel value if it doesn't exist.
      if (currentPixel not in regions):
        regions[currentPixel] = []

      # Initialize a new region set for the current pixel.
      region = set()

      # Use the helper function to find all connected pixels.
      RecursiveHelper(i, j, currentPixel, region, seenMatrix, connectivity)

      # Add the region to the dictionary if it contains any pixels.
      if (len(region) > 0):
        regions[currentPixel].append(region)

  # Return the dictionary of regions.
  return regions


def CalculateGLSZMSizeZoneMatrix(image, connectivity=4, isNorm=False, ignoreZeros=False, epsilon=1e-6):
  '''
  Calculate the Size-Zone Matrix for a given image based on connected regions.

  Parameters:
    image (numpy.ndarray): A 2D NumPy array representing the input image. Each element represents a pixel value.
    connectivity (int): The type of connectivity to use for determining connected regions. Options are (a)
      4 for 4-connectivity (up, down, left, right) and (b) 8 for 8-connectivity (includes diagonals).
    isNorm (bool): Whether to normalize the size-zone matrix.
    ignoreZeros (bool): Whether to ignore zero pixel values.
    epsilon (float): A small value to avoid division by zero during normalization. Default is 1e-6.

  Returns:
    tuple: A tuple containing:
      - szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      - szDict (dict): A dictionary where keys are unique pixel values from the image,
        and values are lists of sets. Each set contains the coordinates
        (i, j) of pixels belonging to a connected region for that pixel value.
      - N (int): The number of unique pixel values in the image.
      - Z (int): The maximum size of any region in the dictionary.

  Raises:
    ValueError: If the input image is not 2D, if connectivity is not 4 or 8, if the image is empty,
      or if the image is completely black.
  '''

  if (image.ndim != 2):
    raise ValueError("The input image must be a 2D array.")

  if (connectivity not in [4, 8]):
    raise ValueError("Connectivity must be either 4 or 8.")

  if (image.size == 0):
    raise ValueError("The input image is empty.")

  if (np.max(image) == 0):
    raise ValueError("The input image is completely black.")

  # Find connected regions in the image.
  szDict = FindConnectedRegions(image, connectivity=connectivity)

  # Determine the number of unique pixel values in the image.
  minA = np.min(image)  # Minimum intensity value.
  maxA = np.max(image)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  # Find the maximum size of any region in the dictionary.
  # By iterating over all zones of all pixel values and getting the length of the largest zone.
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])

  # Initialize a size-zone matrix with zeros.
  szMatrix = np.zeros((N, Z))

  # Populate the size-zone matrix with counts of regions for each pixel value.
  for currentPixel, zones in szDict.items():
    for zone in zones:
      # Ignore zeros if needed.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Increment the count for the corresponding pixel value and region size.
      # (Adjust intensity index by minimum value for proper matrix positioning).
      szMatrix[currentPixel - minA, len(zone) - 1] += 1

  szMatrixSum = np.sum(szMatrix)

  if (szMatrixSum == 0):
    return szMatrix, szDict, N, Z

  # Normalize the size-zone matrix if required.
  if (isNorm):
    # Normalize the size-zone matrix.
    # Add small epsilon to avoid division by zero.
    szMatrix = szMatrix / (np.sum(szMatrix) + epsilon)

  return szMatrix, szDict, N, Z


def CalculateGLSZMFeatures(szMatrix, data, N, Z, epsilon=1e-6):
  '''
  Calculate the features of the Size-Zone Matrix (GLSZM).

  Parameters:
    szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
    data (numpy.ndarray): The original 2D image data from which the GLSZM was derived.
    N (int): The number of unique pixel values in the image.
    Z (int): The maximum size of any region in the dictionary.
   epsilon (float): A small value to avoid division by zero during calculations. Default is 1e-6.

  Returns:
    dict: A dictionary containing the calculated features. This includes:
      - "Small Zone Emphasis (SZE)": Emphasizes small zones in the image.
      - "Large Zone Emphasis (LZE)": Emphasizes large zones in the image.
      - "Gray Level Non-Uniformity (GLN)": Measures the variability of gray levels.
      - "Zone Size Non-Uniformity (ZSN)": Measures the variability of zone sizes.
      - "Zone Percentage (ZP)": Ratio of zones to the total number of pixels.
      - "Gray Level Variance (GLV)": Measures the variance of gray levels.
      - "Zone Size Variance (ZSV)": Measures the variance of zone sizes.
      - "Zone Size Entropy (ZSE)": Measures the randomness of zone sizes.
      - "Low Gray Level Zone Emphasis (LGZE)": Emphasizes zones with low gray levels.
      - "High Gray Level Zone Emphasis (HGZE)": Emphasizes zones with high gray levels.
      - "Small Zone Low Gray Level Emphasis (SZLGE)": Emphasizes small zones with low gray levels.
      - "Small Zone High Gray Level Emphasis (SZHGE)": Emphasizes small zones with high gray levels.
      - "Large Zone Low Gray Level Emphasis (LZGLE)": Emphasizes large zones with low gray levels.
      - "Large Zone High Gray Level Emphasis (LZHGE)": Emphasizes large zones with high gray levels.
  '''

  # Calculate the total number of zones in the size-zone matrix.
  # Sum all values in the size-zone matrix to get the total zone count.
  szN = np.sum(szMatrix)

  # Small Zone Emphasis.
  sze = np.sum(
    szMatrix / ((np.arange(1, Z + 1) ** 2) + epsilon),  # Divide each zone by its size squared.
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone Emphasis.
  lze = np.sum(
    szMatrix * ((np.arange(1, Z + 1) ** 2) + epsilon),  # Multiply each zone by its size squared.
  ).sum() / szN  # Normalize by the total number of zones.

  # Gray Level Non-Uniformity.
  gln = np.sum(
    np.sum(szMatrix, axis=1) ** 2,  # Sum each row and square the result.
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Non-Uniformity.
  zsn = np.sum(
    np.sum(szMatrix, axis=0) ** 2,  # Sum each column and square the result.
  ) / szN  # Normalize by the total number of zones.

  # Zone Percentage.
  # Divide the total number of zones by the total number of pixels.
  zp = szN / np.prod(data.shape)

  # Gray Level Variance.
  glv = np.sum(
    # Compute variance for each gray level.
    (np.sum(szMatrix, axis=1)) *
    ((np.arange(1, N + 1) - np.mean(np.arange(1, N + 1))) ** 2),
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Variance.
  zsv = np.sum(
    # Compute variance for zone sizes.
    (np.sum(szMatrix, axis=0)) *
    ((np.arange(1, Z + 1) - np.mean(np.arange(1, Z + 1))) ** 2),
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Entropy.
  log = np.log2(szMatrix + epsilon)  # Compute log base 2 of the size-zone matrix.
  log[log == -np.inf] = 0  # Replace -inf with 0.
  log[log < 0] = 0  # Replace negative values with 0.
  zse = np.sum(
    # Compute entropy for zone sizes.
    szMatrix * log,
  ) / szN  # Normalize by the total number of zones.

  # Low Gray Level Zone Emphasis.
  lgze = np.sum(
    # Divide each gray level by its squared value.
    szMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # High Gray Level Zone Emphasis.
  hgze = np.sum(
    # Multiply each gray level by its squared value.
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # Small Zone Low Gray Level Emphasis.
  # Adding epsilon to avoid division by zero.
  szlge = np.sum(
    # Combine small zone and low gray level emphasis.
    szMatrix / ((np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2) + epsilon),
  ).sum() / szN  # Normalize by the total number of zones.

  # Small Zone High Gray Level Emphasis.
  szhge = np.sum(
    # Combine small zone and high gray level emphasis.
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2) / ((np.arange(1, Z + 1) ** 2) + epsilon),
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone Low Gray Level Emphasis.
  lzgle = np.sum(
    # Combine large zone and low gray level emphasis.
    szMatrix * (np.arange(1, Z + 1) ** 2) / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone High Gray Level Emphasis.
  lzhge = np.sum(
    # Combine large zone and high gray level emphasis.
    szMatrix * (np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  return {
    "Small Zone Emphasis (SZE)"                  : sze,
    "Large Zone Emphasis (LZE)"                  : lze,
    "Gray Level Non-Uniformity (GLN)"            : gln,
    "Zone Size Non-Uniformity (ZSN)"             : zsn,
    "Zone Percentage (ZP)"                       : zp,
    "Gray Level Variance (GLV)"                  : glv,
    "Zone Size Variance (ZSV)"                   : zsv,
    "Zone Size Entropy (ZSE)"                    : zse,
    "Low Gray Level Zone Emphasis (LGZE)"        : lgze,
    "High Gray Level Zone Emphasis (HGZE)"       : hgze,
    "Small Zone Low Gray Level Emphasis (SZLGE)" : szlge,
    "Small Zone High Gray Level Emphasis (SZHGE)": szhge,
    "Large Zone Low Gray Level Emphasis (LZGLE)" : lzgle,
    "Large Zone High Gray Level Emphasis (LZHGE)": lzhge,
  }


def FindConnectedRegions3D(volume, connectivity=6):
  '''
  Finds connected regions in a 3D volume based on pixel connectivity.

  Parameters:
    volume (numpy.ndarray): A 3D NumPy array representing the input volume.
    connectivity (int): The type of connectivity to use for determining
      connected regions. Options are (a) 6: 6-connectivity (faces only) and
      (b) 26: 26-connectivity (faces, edges, and corners).

  Returns:
    dict: A dictionary where keys are unique pixel values from the volume, and values are lists of sets. Each set contains the coordinates (i, j, k) of pixels belonging to a connected region for that pixel value.
  '''

  def RecursiveHelper(i, j, k, currentPixel, region, seenMatrix, connectivity=6):
    '''
    Recursive helper function to find all connected pixels for a given starting pixel.

    Parameters:
      i (int): Z-axis index of the current pixel.
      j (int): Y-axis index of the current pixel.
      k (int): X-axis index of the current pixel.
      currentPixel (int): The pixel value being processed.
      region (set): A set to store the coordinates of connected pixels.
      seenMatrix (numpy.ndarray): A 3D matrix to track visited pixels.
      connectivity (int): The type of connectivity (6 or 26).
    '''

    # Check if the current pixel is out of bounds, already seen, or not matching the current pixel value.
    if (
      (i < 0) or  # Check if Z-axis index is out of bounds.
      (i >= volume.shape[0]) or
      (j < 0) or  # Check if Y-axis index is out of bounds.
      (j >= volume.shape[1]) or
      (k < 0) or  # Check if X-axis index is out of bounds.
      (k >= volume.shape[2]) or
      (volume[i, j, k] != currentPixel) or  # Check if pixel value matches the current pixel value.
      ((i, j, k) in region) or  # Check if the pixel has already been added to the region.
      (seenMatrix[i, j, k] == 1)  # Check if the pixel has already been seen.
    ):
      return  # Exit if any condition is met.

    # Add the current pixel to the region and mark it as seen.
    region.add((i, j, k))  # Add the pixel coordinates to the region set.
    seenMatrix[i, j, k] = 1  # Mark the pixel as seen.

    # Recursively check the neighboring pixels (faces only for 6-connectivity).
    RecursiveHelper(i - 1, j, k, currentPixel, region, seenMatrix, connectivity)  # Check Z-axis neighbor below.
    RecursiveHelper(i + 1, j, k, currentPixel, region, seenMatrix, connectivity)  # Check Z-axis neighbor above.
    RecursiveHelper(i, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Check Y-axis neighbor left.
    RecursiveHelper(i, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Check Y-axis neighbor right.
    RecursiveHelper(i, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Check X-axis neighbor behind.
    RecursiveHelper(i, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Check X-axis neighbor front.

    # If 26-connectivity is specified, also check diagonal neighbors (edges and corners).
    if (connectivity == 26):
      # k is fixed => same slice (4 pixels).
      RecursiveHelper(i - 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

      # k - 1 => pre-slice (8 pixels).
      RecursiveHelper(i - 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

      # k + 1 => post-slice (8 pixels).
      RecursiveHelper(i - 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

  # Initialize a matrix to keep track of seen pixels.
  seenMatrix = np.zeros(volume.shape)  # Create a 3D matrix of zeros.

  # Dictionary to store regions grouped by pixel values.
  regions = {}  # Keys are pixel values, values are lists of sets.

  # Iterate over each voxel in the volume.
  for i in range(volume.shape[0]):  # Loop over Z-axis.
    for j in range(volume.shape[1]):  # Loop over Y-axis.
      for k in range(volume.shape[2]):  # Loop over X-axis.
        # Skip if the voxel has already been processed.
        if (seenMatrix[i, j, k]):
          continue  # Skip already processed voxels.

        # Get the current voxel value.
        currentPixel = volume[i, j, k]  # Retrieve the intensity value of the voxel.

        # Initialize a list for this pixel value if it doesn't exist.
        if (currentPixel not in regions):
          regions[currentPixel] = []  # Create a new list for this intensity value.

        # Initialize a new region set for the current voxel.
        region = set()  # Create an empty set to store connected voxel coordinates.

        # Use the helper function to find all connected voxels.
        RecursiveHelper(i, j, k, currentPixel, region, seenMatrix, connectivity)  # Find connected region.

        # Add the region to the dictionary if it contains any voxels.
        if (len(region) > 0):
          regions[currentPixel].append(region)  # Append the region to the list for this intensity value.

  # Return the dictionary of regions.
  return regions  # Return the dictionary containing connected regions.


def CalculateGLSZMSizeZoneMatrix3D(volume, connectivity=6, isNorm=True, ignoreZeros=True, epsilon=1e-6):
  '''
  Calculate the Size-Zone Matrix for a 3D volume based on connected regions.

  Parameters:
    volume (numpy.ndarray): A 3D NumPy array representing the input volume.
    connectivity (int): The type of connectivity to use for determining connected regions. Options are
      (a) 6 for 6-connectivity (faces only) and (b) 26 for 26-connectivity (faces, edges, and corners).
      Default is 6.
    isNorm (bool): Whether to normalize the size-zone matrix. Default is True.
    ignoreZeros (bool): Whether to ignore zero pixel values. Default is True.
    epsilon (float): A small value to avoid division by zero during normalization. Default is 1e-6.

  Returns:
    tuple: A tuple containing the following elements:
      - szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      - szDict (dict): A dictionary where keys are unique pixel values from the volume,
        and values are lists of sets. Each set contains the coordinates
        (i, j, k) of pixels belonging to a connected region for that pixel value.
      - N (int): The number of unique pixel values in the volume.
      - Z (int): The maximum size of any region in the dictionary.

  Raises:
    ValueError: If the input volume is not a 3D array, if connectivity is not 6 or 26,
      if the input volume is empty, or if the input volume is completely black.
  '''

  if (volume.ndim != 3):
    raise ValueError("The input volume must be a 3D array.")

  if (connectivity not in [6, 26]):
    raise ValueError("Connectivity must be either 6 or 26.")

  if (volume.size == 0):
    raise ValueError("The input volume is empty.")

  if (np.max(volume) == 0):
    raise ValueError("The input volume is completely black.")

  # Find connected regions in the volume.
  szDict = FindConnectedRegions3D(volume, connectivity=connectivity)  # Identify connected regions.

  # Determine the number of unique pixel values in the volume.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1  # Number of discrete intensity levels

  # Find the maximum size of any region in the dictionary.
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])  # Find the largest connected region size.

  # Initialize a size-zone matrix with zeros.
  szMatrix = np.zeros((N, Z))  # Create a 2D matrix to store size-zone counts.

  # Populate the size-zone matrix with counts of regions for each pixel value.
  for currentVal, zones in szDict.items():
    for zone in zones:
      # Ignore zeros if needed.
      if (ignoreZeros and (currentVal == 0)):
        continue  # Skip zero-valued regions if ignoreZeros is True.

      # Increment the count for the corresponding pixel value and region size.
      szMatrix[currentVal - minA, len(zone) - 1] += 1  # Update the size-zone matrix.

  # Normalize the size-zone matrix if required.
  if (isNorm):
    # Normalize by total sum to avoid division by zero.
    # Add small epsilon to avoid division by zero.
    szMatrix = szMatrix / (np.sum(szMatrix) + epsilon)

    # Return the size-zone matrix, dictionary, and metadata.
  return szMatrix, szDict, N, Z  # Return the computed outputs.


# ===========================================================================================
# Function(s) for handling the local binary patterns (LBP).
# ===========================================================================================
def BuildLBPKernel(
  distance=1,  # Distance parameter to determine the size of the kernel.
  theta=135,  # Angle parameter to rotate the kernel (default is 135 degrees).
  isClockwise=False,  # Direction of rotation (False means counterclockwise).
):
  '''
  Build a kernel matrix for Local Binary Pattern (LBP) computation.
  The kernel is generated based on the specified distance and angle (theta).
  The kernel is a square matrix of size (2 * distance + 1) x (2 * distance + 1).
  The kernel is filled with powers of 2, representing the weights of the pixels
  in the LBP computation. The kernel is rotated by the specified angle (theta)
  in a clockwise or counterclockwise direction.

  Parameters:
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the kernel rotation.
    isClockwise (bool): Direction of rotation (True for clockwise, False for counterclockwise).

  Returns:
    numpy.ndarray: A kernel matrix representing the LBP pattern weights.

  Raises:
    ValueError: If the distance is less than 1, or if theta is not in the range [0, 360], or if theta is not a multiple of the angle between elements.
  '''

  # Check if the distance is less than 1, raising a ValueError if true.
  if (distance < 1):
    raise ValueError("Distance must be greater than or equal to 1.")

  # Calculate the total number of elements on the edges.
  noOfElements = 8 * distance  # Total number of edge elements is 8 * distance.

  # Calculate the angle between consecutive elements.
  angle = 360.0 / float(noOfElements)  # Divide 360 degrees by the total number of edge elements.

  # Check if the angle (theta) is outside the valid range (0 to 360 degrees), raising a ValueError if true.
  if (theta < 0 or theta > 360):
    raise ValueError("Theta must be between 0 and 360 degrees.")

  # Check if the angle (theta) is not a multiple of (angle) degrees, raising a ValueError if true.
  if (theta % angle != 0):
    raise ValueError("Theta must be a multiple of the angle between elements.")

  # Calculate the size of the matrix.
  n = 2 * distance + 1  # The size of the kernel is (2 * distance + 1) x (2 * distance + 1).

  # Initialize the matrix with zeros.
  kernel = np.zeros((n, n), dtype=np.uint32)  # Create a zero-filled matrix of size n x n.

  # Generate the coordinates for the edges of the kernel in a clockwise order.
  coordinates = []  # List to store the edge coordinates of the kernel.

  # Add coordinates for the leftmost column (top to bottom).
  for row in range(0, n):  # Iterate over rows from top to bottom.
    coordinates.append((row, 0))  # Append (row, 0) for the leftmost column.

  # Add coordinates for the bottommost row (left to right).
  for col in range(0, n):  # Iterate over columns from left to right.
    coordinates.append((n - 1, col))  # Append (n-1, col) for the bottommost row.

  # Add coordinates for the rightmost column (bottom to top).
  for row in range(n - 1, -1, -1):  # Iterate over rows from bottom to top.
    coordinates.append((row, n - 1))  # Append (row, n-1) for the rightmost column.

  # Add coordinates for the topmost row (right to left).
  for col in range(n - 1, -1, -1):  # Iterate over columns from right to left.
    coordinates.append((0, col))  # Append (0, col) for the topmost row.

  # Remove the repeated coordinates.
  for i in range(len(coordinates) - 1, 0, -1):  # Iterate from the end to the beginning.
    if (coordinates[i] == coordinates[i - 1]):  # Check if the current coordinate is equal to the previous one.
      coordinates.pop(i)  # Remove the current coordinate if it is a duplicate.
  # Remove the last coordinate if it is equal to the first one.
  if (coordinates[-1] == coordinates[0]):  # Check if the last coordinate is equal to the first one.
    coordinates.pop(-1)  # Remove the last coordinate if it is a duplicate.

  # Calculate the shift required to rotate the kernel by the given theta.
  thetaShift = int((theta - 135) / angle)  # Determine how many positions to shift based on theta.

  # Rotate the coordinates list by thetaShift positions.
  coordinates = coordinates[thetaShift:] + coordinates[:thetaShift]  # Shift the coordinates list.

  # If the rotation direction is clockwise, rotate the kernel counterclockwise.
  if (isClockwise):
    # Reverse the order of coordinates except the first one.
    coordinates = [coordinates[0]] + coordinates[1:][::-1]

  # Assign powers of 2 to the edge elements in the kernel.
  counter = 0  # Counter to track the current power of 2.

  # Iterate through the shifted coordinates and assign values to the kernel.
  for i in range(len(coordinates)):  # Loop through all edge coordinates.
    x = coordinates[i][0]  # Extract the x-coordinate.
    y = coordinates[i][1]  # Extract the y-coordinate.
    if (kernel[y, x] == 0):  # Check if the position is still zero (not yet assigned).
      kernel[y, x] = 2 ** counter  # Assign 2^counter to the current position.
      counter += 1  # Increment the counter for the next power of 2.

  # Transpose the kernel to match the expected orientation.
  kernel = kernel.T

  return kernel  # Return the final kernel matrix.


def LocalBinaryPattern2D(
  matrix,
  distance=1,
  theta=135,
  isClockwise=False,
  normalizeLBP=False,
):
  '''
  Compute the Local Binary Pattern (LBP) matrix for a given 2D matrix.
  This function calculates the LBP values based on the specified distance,
  angle (theta), and direction (clockwise or counterclockwise).
  The LBP is a texture descriptor that encodes local patterns in the image,
  making it useful for various image analysis tasks.

  Parameters:
    matrix (numpy.ndarray): Input 2D matrix (grayscale) for LBP computation.
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the LBP computation (must be a multiple of 45).
    isClockwise (bool): Direction of LBP computation (True for clockwise, False for counterclockwise).
    normalizeLBP (bool): Flag to normalize the LBP values (default is False).

  Returns:
    numpy.ndarray: LBP matrix with the same shape as the input image, containing LBP values.

  Raises:
    ValueError: If the distance is less than 1, exceeds half of the image dimensions, or if the angle (theta) is outside the valid range (0 to 360 degrees).
  '''

  # Check if the distance is less than 1, raising a ValueError if true.
  if (distance < 1):
    raise ValueError("Distance must be greater than or equal to 1.")
  # Check if the distance exceeds half of the image dimensions, raising a ValueError if true.
  if (distance > matrix.shape[0] // 2 or distance > matrix.shape[1] // 2):
    raise ValueError("Distance must be less than half of the matrix dimensions.")
  # Check if the angle (theta) is outside the valid range (0 to 360 degrees), raising a ValueError if true.
  if (theta < 0 or theta > 360):
    raise ValueError("Theta must be between 0 and 360 degrees.")

  # Calculate the total number of elements on the edges.
  noOfElements = 8 * distance  # Total number of edge elements is 8 * distance.

  # Calculate the angle between consecutive elements.
  angle = 360.0 / float(noOfElements)  # Divide 360 degrees by the total number of edge elements.

  # Check if the angle (theta) is not a multiple of {angle} degrees, raising a ValueError if true.
  if (theta % angle != 0):
    raise ValueError(f"Theta must be a multiple of {angle} degrees.")

  # Calculate the size of the kernel window based on the distance parameter.
  windowSize = distance * 2 + 1
  # Determine the center coordinates of the kernel window.
  centerX = windowSize // 2
  centerY = windowSize // 2

  # Build the LBP kernel using the specified parameters.
  kernel = BuildLBPKernel(
    distance=distance,
    theta=theta,
    isClockwise=isClockwise,
  )

  # Initialize an empty matrix to store the computed LBP values.
  lbpMatrix = np.zeros(matrix.shape, dtype=np.uint32)

  # Pad the input matrix with zeros to handle boundary conditions during convolution.
  paddedA = np.pad(matrix, distance, mode="constant", constant_values=0)

  # Iterate through each pixel in the input matrix to compute its LBP value.
  for y in range(distance, matrix.shape[0] + distance):
    for x in range(distance, matrix.shape[1] + distance):
      # Extract the region of interest (ROI) around the current pixel.
      region = paddedA[
        y - distance:y + distance + 1,
        x - distance:x + distance + 1
      ]
      # Compare each pixel in the ROI with the center pixel to create a binary mask.
      comp = region >= region[centerY, centerX]
      # Compute the LBP value for the current pixel by summing the weighted kernel values.
      lbpMatrix[y - distance, x - distance] = np.sum(kernel[comp])

  # Normalize the LBP values if the flag is set to True.
  if (normalizeLBP):
    # Find the minimum and maximum LBP values in the matrix.
    minValue = np.min(lbpMatrix)
    maxValue = np.max(lbpMatrix)
    # Normalize the LBP values to the range [0, 255].
    lbpMatrix = ((lbpMatrix - minValue) / (maxValue - minValue) * 255)
    # Ensure the LBP matrix is of type uint8.
    lbpMatrix = lbpMatrix.astype(np.uint8)

  # Return the computed LBP matrix.
  return lbpMatrix


def UniformLocalBinaryPattern2D(
  matrix,
  distance=1,
  theta=135,
  isClockwise=False,
  normalizeLBP=False,
):
  '''
  Compute the Uniform Local Binary Pattern (LBP) matrix for a given 2D matrix.
  This function calculates the LBP values based on the specified distance,
  angle (theta), and direction (clockwise or counterclockwise).
  The Uniform LBP is a variant of LBP that focuses on uniform patterns,
  making it useful for texture analysis and classification tasks.
  The uniform patterns are defined as those with at most two transitions
  between 0 and 1 in the binary representation of the LBP value.

  Parameters:
    matrix (numpy.ndarray): Input 2D matrix (grayscale) for LBP computation.
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the LBP computation (must be a multiple of 45).
    isClockwise (bool): Direction of LBP computation (True for clockwise, False for counterclockwise).
    normalizeLBP (bool): Flag to normalize the LBP values (default is False).

  Returns:
    numpy.ndarray: Uniform LBP matrix with the same shape as the input image, containing LBP values.
  '''

  # Run the standard LBP function to get the LBP matrix.
  lbpMatrix = LocalBinaryPattern2D(
    matrix,
    distance=distance,
    theta=theta,
    isClockwise=isClockwise,
    normalizeLBP=False,  # No need to normalize here.
  )

  # Initialize an empty matrix to store the uniform LBP values.
  uniformMatrix = np.zeros(matrix.shape, dtype=np.uint32)

  # Iterate through each pixel in the LBP matrix to compute uniform LBP values.
  for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
      # Convert the LBP value to binary representation with 8 * distance bits.
      binary = np.binary_repr(
        lbpMatrix[y, x],
        width=8 * distance,
      )
      # Count the number of transitions (0 to 1 or 1 to 0) in the binary representation.
      transitions = 0
      for i in range(1, len(binary)):
        # Count transitions between consecutive bits.
        if (binary[i] != binary[i - 1]):
          transitions += 1

      # If the number of transitions is less than or equal to 2, assign the LBP value.
      if (transitions <= 2):
        # Assign the LBP value to the uniform matrix.
        uniformMatrix[y, x] = int(binary, 2)

  # Normalize the uniform LBP values if the flag is set to True.
  if (normalizeLBP):
    # Find the minimum and maximum uniform LBP values in the matrix.
    minValue = np.min(uniformMatrix)
    maxValue = np.max(uniformMatrix)
    # Normalize the uniform LBP values to the range [0, 255].
    uniformMatrix = ((uniformMatrix - minValue) / (maxValue - minValue) * 255).astype(np.uint8)

  # Ensure the uniform LBP matrix is of type uint8.
  uniformMatrix = uniformMatrix.astype(np.uint8)

  # Return the computed uniform LBP matrix.
  return uniformMatrix


# ===========================================================================================
# Function(s) for calculating shape features.
# ===========================================================================================
def ShapeFeatures2D(matrix):
  '''
  Calculate shape features of a given binary matrix in 2D.
  The function computes various shape features such as area, perimeter,
  centroid, bounding box, aspect ratio, compactness, eccentricity,
  convex hull area, extent, solidity, major and minor axis lengths,
  orientation, and roundness.

  Parameters:
    matrix (numpy.ndarray): A binary 2D NumPy array representing the shape.

  Returns:
    dict: A dictionary containing the calculated shape features. This includes:
      - Area: The number of non-zero pixels in the matrix.
      - Perimeter: The perimeter of the largest contour in the matrix.
      - Centroid: The coordinates of the centroid of the shape.
      - Bounding Box: The bounding box of the shape in the format (x, y, w, h).
      - Aspect Ratio: The ratio of width to height of the bounding box.
      - Compactness: A measure of how compact the shape is.
      - Eccentricity: A measure of how elongated the shape is.
      - Convex Hull Area: The area of the convex hull of the shape.
      - Extent: The ratio of the area of the shape to the area of its bounding box.
      - Solidity: The ratio of the area of the shape to the area of its convex hull.
      - Major Axis Length: The length of the major axis of the shape.
      - Minor Axis Length: The length of the minor axis of the shape.
      - Orientation: The orientation angle of the major axis of the shape.
      - Roundness: A measure of how round the shape is.
      - Symmetry: A measure of the symmetry of the shape.
      - Elongation: The ratio of the major axis length to the minor axis length.
      - Thinness Ratio: The ratio of the square of the perimeter to the area.
      - Convexity: The ratio of the perimeter of the convex hull to the perimeter of the shape.
      - Sparseness: A measure of how spread out the shape is.
      - Curvature: A measure of how sharply the contour bends at each point.

  Raises:
    ValueError: If the input matrix is empty or not a valid binary image.
  '''

  # Check if the input matrix is empty or not.
  if (matrix is None or matrix.size == 0):
    # Raise error if the matrix is empty.
    raise ValueError("The input matrix is empty. Please provide a valid matrix.")

  # Calculate the Shape Features:

  # 1. Area.
  # Counts the number of non-zero pixels in the cropped image.
  area = cv2.countNonZero(matrix)

  # 2. Perimeter.
  # Finds contours in the matrix image.
  contours, _ = cv2.findContours(matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Identifies the largest contour.
  largestContour = max(contours, key=cv2.contourArea)
  # Calculates the perimeter of the largest contour.
  perimeter = cv2.arcLength(largestContour, True)

  # 3. Centroid.
  # Computes moments of the matrix.
  moments = cv2.moments(matrix)
  # Calculates the X-coordinate of the centroid.
  centroidX = int(moments["m10"] / moments["m00"])
  # Calculates the Y-coordinate of the centroid.
  centroidY = int(moments["m01"] / moments["m00"])

  # 4. Bounding Box.
  # Recalculates the bounding box for the matrix.
  x, y, w, h = cv2.boundingRect(matrix)

  # 5. Aspect Ratio.
  # Computes the aspect ratio of the bounding box.
  aspectRatio = w / h

  # 6. Compactness.
  # Calculates compactness using perimeter and area.
  compactness = (perimeter ** 2) / (4 * np.pi * area)

  # 7. Eccentricity.
  # Computes normalized second-order moment mu20.
  mu20 = moments["mu20"] / moments["m00"]
  # Computes normalized second-order moment mu02.
  mu02 = moments["mu02"] / moments["m00"]
  # Calculates eccentricity based on moments.
  eccentricity = np.sqrt(1 - (mu02 / mu20))

  # 8. Convex Hull.
  # Finds the convex hull of the largest contour.
  smallestConvexHull = cv2.convexHull(largestContour)
  # Calculates the area of the convex hull.
  convexHullArea = cv2.contourArea(smallestConvexHull)

  # 9. Extent (or Rectangularity).
  # Computes the extent as the ratio of contour area to bounding box area.
  extent = area / (w * h)

  # 10. Solidity.
  # Calculates solidity as the ratio of contour area to convex hull area.
  solidity = area / convexHullArea

  # 11. Major Axis Length.
  # Computes the length of the major axis using the second-order moment mu20.
  majorAxisLength = 2 * np.sqrt(moments["m20"] / moments["m00"])

  # 12. Minor Axis Length.
  # Computes the length of the minor axis using the second-order moment mu02.
  minorAxisLength = 2 * np.sqrt(moments["m02"] / moments["m00"])

  # 13. Orientation.
  # Calculates orientation angle as the angle of the major axis of the ellipse.
  orientation = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])

  # 14. Roundness.
  # Computes roundness based on area and perimeter.
  roundness = (4 * area) / (np.pi * perimeter ** 2)

  # 15. Symmetry.
  # Flip the matrix horizontally and vertically.
  flippedHorizontal = np.fliplr(matrix)
  flippedVertical = np.flipud(matrix)
  # Calculate the symmetry score for horizontal flipping.
  horizontalSymmetry = np.sum(matrix == flippedHorizontal) / area
  # Calculate the symmetry score for vertical flipping.
  verticalSymmetry = np.sum(matrix == flippedVertical) / area
  # Calculate the average symmetry score.
  symmetry = (horizontalSymmetry + verticalSymmetry) / 2.0

  # 16. Elongation.
  # Calculate the elongation based on the major and minor axis lengths.
  elongation = majorAxisLength / minorAxisLength

  # 17. Thinness Ratio.
  # Calculate the thinness ratio based on the perimeter and area.
  thinnessRatio = np.power(perimeter, 2) / area

  # 18. Convexity.
  # Convexity measures how close the shape is to being convex.
  # It is the ratio of the perimeter of the convex hull to the perimeter of the shape.
  # True: The contour is closed, False: The contour is open.
  convexHullPerimeter = cv2.arcLength(smallestConvexHull, True)
  convexity = convexHullPerimeter / perimeter

  # 19. Sparseness.
  # Sparseness measures how "spread out" the shape is.
  # Calculate the area of the bounding box.
  boundingBoxArea = w * h
  # Compute sparseness as a measure of spread.
  sparseness = (np.sqrt(area / boundingBoxArea) - (area / boundingBoxArea))

  # 20. Curvature.
  # Curvature measures how sharply the contour bends at each point.
  curvatures = []
  for i in range(len(largestContour)):
    # Loop through all points in the largest contour.
    p1 = largestContour[i - 1][0]  # Previous point.
    p2 = largestContour[i][0]  # Current point.
    p3 = largestContour[(i + 1) % len(largestContour)][0]  # Next point.
    # Calculate the curvature using the cross product and dot product.
    v1 = p2 - p1  # Vector from p1 to p2.
    v2 = p3 - p2  # Vector from p2 to p3.
    crossProduct = np.cross(v1, v2)  # Cross product of the vectors.
    dotProduct = np.dot(v1, v2)  # Dot product of the vectors.
    angle = np.arctan2(crossProduct, dotProduct)  # Angle between the vectors.
    curvatures.append(angle)  # Append the curvature to the list.
  # Calculate the average curvature.
  averageCurvature = np.mean(curvatures)
  # Calculate the standard deviation of curvature.
  stdCurvature = np.std(curvatures)

  # Return all calculated features as a dictionary.
  return {
    "Area"               : area,
    "Perimeter"          : perimeter,
    "Centroid X"         : centroidX,
    "Centroid Y"         : centroidY,
    "Bounding Box X"     : x,
    "Bounding Box Y"     : y,
    "Bounding Box W"     : w,
    "Bounding Box H"     : h,
    "Aspect Ratio"       : aspectRatio,
    "Compactness"        : compactness,
    "Eccentricity"       : eccentricity,
    "Convex Hull Area"   : convexHullArea,
    "Extent"             : extent,
    "Solidity"           : solidity,
    "Major Axis Length"  : majorAxisLength,
    "Minor Axis Length"  : minorAxisLength,
    "Orientation"        : orientation,
    "Roundness"          : roundness,
    "Horizontal Symmetry": horizontalSymmetry,
    "Vertical Symmetry"  : verticalSymmetry,
    "Symmetry"           : symmetry,
    "Elongation"         : elongation,
    "Thinness Ratio"     : thinnessRatio,
    "Convexity"          : convexity,
    "Sparseness"         : sparseness,
    "Curvature"          : averageCurvature,
    "Std Curvature"      : stdCurvature,
  }


def ShapeFeatures3D(volume):
  '''
  Calculate 3D shape features of a given binary or labeled volume.
  The function computes various geometric and topological properties such as volume,
  surface area, compactness, sphericity, elongation, flatness, rectangularity,
  spherical disproportion, and Euler number. These features are derived from the
  mesh representation of the input volume using marching cubes.

  Parameters:
    volume (numpy.ndarray): A 3D binary or labeled matrix representing the object.

  Returns:
    dict: A dictionary containing the calculated 3D shape features. This includes:
      - Volume: Total number of non-zero voxels in the volume.
      - Surface Area: Total surface area of the mesh generated by marching cubes.
      - Surface to Volume Ratio: Ratio of surface area to volume.
      - Compactness: A measure of how closely the shape resembles a sphere.
      - Sphericity: A measure of how spherical the shape is.
      - Elongation: Ratio of the longest dimension to the shortest dimension of the bounding box.
      - Flatness: Ratio of the shortest dimension to the intermediate dimension of the bounding box.
      - Rectangularity: Ratio of volume to bounding box volume.
      - Euler Number: Topological characteristic of the shape.
  '''

  # Converts an (n, m, p) matrix into a mesh, using marching_cubes.
  # Marching cubes algorithm generates a triangular mesh from the volume data.
  mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volume)

  # 1. Volume.
  # Computes the total number of non-zero voxels in the volume.
  volume = np.sum(volume)

  # 2. Surface Area.
  # Calculates the total surface area of the mesh generated by marching cubes.
  surfaceArea = mesh.area

  # 3. Surface to Volume Ratio.
  # Measures the ratio of surface area to volume, indicating compactness.
  surfaceToVolumeRatio = surfaceArea / volume

  # 4. Compactness.
  # Quantifies how closely the shape resembles a sphere, based on volume and surface area.
  compactness = (volume ** (2 / 3)) / (6 * np.sqrt(np.pi) * surfaceArea)

  # 5. Sphericity.
  # Measures how spherical the shape is, normalized by volume and surface area.
  sphericity = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surfaceArea

  # Bounding Box.
  # Computes the bounding box of the mesh and extracts its dimensions.
  bbox = mesh.bounding_box.bounds
  Lmax = np.max(bbox[1] - bbox[0])  # Maximum length of the bounding box.
  Lmin = np.min(bbox[1] - bbox[0])  # Minimum length of the bounding box.
  Lint = np.median(bbox[1] - bbox[0])  # Intermediate length of the bounding box.

  # 6. Elongation.
  # Measures the ratio of the longest dimension to the shortest dimension of the bounding box.
  elongation = Lmax / Lmin

  # 7. Flatness.
  # Measures the ratio of the shortest dimension to the intermediate dimension of the bounding box.
  flatness = Lmin / Lint

  # 8. Rectangularity.
  # Measures how efficiently the shape fills its bounding box, as the ratio of volume to bounding box volume.
  bboxVolume = np.prod(bbox[1] - bbox[0])  # Volume of the bounding box.
  rectangularity = volume / bboxVolume

  # 9. Euler Number.
  # Represents the topological characteristic of the shape, computed from the mesh.
  eulerNumber = mesh.euler_number

  # Return all calculated features as a dictionary.
  return {
    "Volume"                 : volume,
    "Surface Area"           : surfaceArea,
    "Surface to Volume Ratio": surfaceToVolumeRatio,
    "Compactness"            : compactness,
    "Sphericity"             : sphericity,
    "Elongation"             : elongation,
    "Flatness"               : flatness,
    "Rectangularity"         : rectangularity,
    "Euler Number"           : eulerNumber
  }
