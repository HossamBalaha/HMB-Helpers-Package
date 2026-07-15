# Import standard libraries for image normalization and processing.
import os, glob, pickle, random
import cv2  # OpenCV for image processing.
import numpy as np  # NumPy for numerical operations.
import spams  # SPAMS for sparse matrix computations.
import tqdm  # TQDM for progress bars.
from HMB.Initializations import IMAGE_SUFFIXES, DoRandomSeeding


def RGB2LAB(img, isNorm=True):
  r'''
  Convert an RGB image to the LAB color space and normalize the channels.

  Parameters:
    img (numpy.ndarray): Input RGB image.
    isNorm (bool): Normalize the LAB channels.

  Returns:
    tuple: Normalized LAB channels (L, A, B).
  '''

  # Validate input image shape.
  if ((img.ndim != 3) or (img.shape[2] != 3)):
    raise ValueError("RGB2LAB expects an HxWx3 image.")

  # Convert image to uint8 if it is in uint16 or float format.
  if (img.dtype == np.uint16):
    img = (img / 257).astype(np.uint8)
  elif ((img.dtype == np.float32) or (img.dtype == np.float64)):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

  # Convert the input RGB image to LAB color space using OpenCV.
  I = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # Convert RGB to LAB color space.

  # Split the LAB image into its three channels: L, A, and B.
  I1, I2, I3 = cv2.split(I)  # Split the LAB image into its three channels: L, A, and B.

  # If normalization is requested, normalize each channel accordingly.
  if (isNorm):  # Normalize the LAB channels.
    # Normalize the L channel to the range [0, 100].
    I1 = I1 / 2.55  # Normalize the L channel to the range [0, 100].
    # Normalize the A channel to the range [-128, 127].
    I2 = I2 - 128.0  # Normalize the A channel to the range [-128, 127].
    # Normalize the B channel to the range [-128, 127].
    I3 = I3 - 128.0  # Normalize the B channel to the range [-128, 127].

  # Return the normalized LAB channels as a tuple.
  return I1, I2, I3  # Return the normalized LAB channels.


def LAB2RGB(I1, I2, I3, isNorm=True):
  r'''
  Convert normalized LAB channels back to the RGB color space.

  Parameters:
    I1 (numpy.ndarray): Normalized L channel.
    I2 (numpy.ndarray): Normalized A channel.
    I3 (numpy.ndarray): Normalized B channel.

  Returns:
    numpy.ndarray: RGB image.
  '''

  # Validate inputs are numpy arrays with matching shapes
  for arr in (I1, I2, I3):
    if (arr is None or not hasattr(arr, "shape")):
      raise ValueError("LAB2RGB expects numpy arrays for I1, I2, I3.")
  if (I1.shape != I2.shape or I1.shape != I3.shape):
    raise ValueError("LAB2RGB channel shapes must match.")

  # If denormalization is requested, reverse the normalization for each channel.
  if (isNorm):
    # Denormalize the L channel back to the range [0, 255].
    I1 = I1 * 2.55  # Denormalize the L channel back to the range [0, 255].
    # Denormalize the A channel back to the range [0, 255].
    I2 = I2 + 128.0  # Denormalize the A channel back to the range [0, 255].
    # Denormalize the B channel back to the range [0, 255].
    I3 = I3 + 128.0  # Denormalize the B channel back to the range [0, 255].

  # Merge the LAB channels back into a single image.
  I = cv2.merge((I1, I2, I3))  # Merge the LAB channels back into a single image.
  # Replace NaNs/infs with numeric values then Clip the pixel values to ensure they are within the valid range [0, 255].
  I = np.nan_to_num(I, nan=0.0, posinf=255.0, neginf=0.0)
  I = np.clip(I, 0, 255).astype(np.uint8)  # Clip the pixel values to ensure they are within the valid range [0, 255].
  # Convert the LAB image back to the RGB color space using OpenCV.
  I = cv2.cvtColor(I, cv2.COLOR_LAB2RGB)  # Convert the LAB image back to the RGB color space.
  # Return the resulting RGB image.
  return I  # Return the resulting RGB image.


def LABSplit2RGB(I1, I2, I3):
  r'''
  Convert LAB channels back to RGB color space.

  Parameters:
    I1 (numpy.ndarray): L channel.
    I2 (numpy.ndarray): A channel.
    I3 (numpy.ndarray): B channel.

  Returns:
    numpy.ndarray: RGB image.
  '''

  # Denormalize L channel (avoid in-place modification).
  I1 = I1 * 2.55  # Denormalize L channel.
  # Denormalize A channel.
  I2 = I2 + 128.0  # Denormalize A channel.
  # Denormalize B channel.
  I3 = I3 + 128.0  # Denormalize B channel.
  # Merge channels back into an image.
  I = cv2.merge((I1, I2, I3))  # Merge channels back into an image.
  # Replace NaNs/infs with numeric values then Clip values to the range [0, 255].
  I = np.nan_to_num(I, nan=0.0, posinf=255.0, neginf=0.0)
  I = np.clip(I, 0, 255)  # Clip values to the range [0, 255].
  # Convert to unsigned 8-bit integer.
  I = I.astype(np.uint8)  # Convert to unsigned 8-bit integer.
  # Convert LAB to RGB color space using OpenCV.
  I = cv2.cvtColor(I, cv2.COLOR_LAB2RGB)  # Convert LAB to RGB color space.
  # Return the RGB image.
  return I  # Return the RGB image.


def LabSplitMeanStd(img):
  r'''
  Compute the mean and standard deviation of the LAB channels of an RGB image.

  Parameters:
    img (numpy.ndarray): Input RGB image.

  Returns:
    tuple: Means and standard deviations of the LAB channels.
  '''

  # Convert the input RGB image to LAB and split into channels.
  I1, I2, I3 = RGB2LAB(img)  # Convert the input RGB image to LAB and split into channels.
  # Calculate the mean and standard deviation of the L channel.
  m1, s1 = cv2.meanStdDev(I1)  # Calculate the mean and standard deviation of the L channel.
  # Calculate the mean and standard deviation of the A channel.
  m2, s2 = cv2.meanStdDev(I2)  # Calculate the mean and standard deviation of the A channel.
  # Calculate the mean and standard deviation of the B channel.
  m3, s3 = cv2.meanStdDev(I3)  # Calculate the mean and standard deviation of the B channel.
  # Store the means in a list.
  means = [m1, m2, m3]  # Store the means in a list.
  # Store the standard deviations in a list.
  stds = [s1, s2, s3]  # Store the standard deviations in a list.
  # Return the means and standard deviations.
  return means, stds  # Return the means and standard deviations.


def FindStainMatrixVahadane(img, beta=0.15, lambda1=0.01):
  r'''
  Find the stain matrix using Vahadane's method (Sparse Non-negative Matrix Factorization).
  It has multiple safeguards to ensure robustness, including fallbacks to a default stain matrix if the input
  image is mostly blank or if SPAMS fails.
  First converts the image to Optical Density (OD) space, filters out low OD values, and then applies SPAMS to
  learn the stain matrix.
  The resulting matrix is normalized and ordered to ensure Hematoxylin is the first row.

  Parameters:
    img (numpy.ndarray): Input RGB image (range [0, 255]).
    beta (float): Threshold for OD values.
    lambda1 (float): Sparsity penalty for dictionary learning.

  Returns:
    numpy.ndarray: Stain matrix (2x3).
  '''

  # Ensure the input is a 3-channel RGB image.
  if ((len(img.shape) != 3) or (img.shape[2] != 3)):
    raise ValueError("Input image must be a 3-channel RGB image.")

  # Convert to OD space and reshape.
  odImage = RGB2OD(img)
  odImageReshaped = odImage.reshape(-1, odImage.shape[-1])

  # Filter out low OD values (background).
  validPixels = (odImageReshaped > beta).any(axis=1)
  odImageReshaped = odImageReshaped[validPixels, :]

  # Default, standard H&E stain matrix fallback (normalized).
  # Source: Standard Macenko/Vahadane literature values.
  defaultStain = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin.
    [0.072, 0.990, 0.105]  # Eosin.
  ], dtype=np.float64)

  # Safeguard 1: If the image is mostly blank, return the default stain matrix immediately.
  if (odImageReshaped.shape[0] < 50):
    return defaultStain

  try:
    # Use SPAMS for Non-negative Dictionary Learning (SNMF).
    W = spams.trainDL(
      np.asfortranarray(odImageReshaped.T),
      K=2,  # Number of stains (Hematoxylin and Eosin)
      lambda1=lambda1,  # Sparsity penalty
      mode=2,  # L2 penalty on the codes
      modeD=0,  # L2 penalty on the dictionary
      posAlpha=True,  # Positive codes
      posD=True,  # Positive dictionary
      verbose=False
    )
  except Exception:
    # Safeguard 2: Fallback on SPAMS failure.
    return defaultStain

  # Safeguard 3: Ensure W is valid and contains no NaNs/Infs.
  if (W is None or np.isnan(W).any() or np.isinf(W).any()):
    return defaultStain

  # Safeguard 4: Normalize the stain matrix rows safely.
  # W is 3x2, so W.T is 2x3. We normalize along axis 1 of W.T.
  norms = np.linalg.norm(W.T, axis=1, keepdims=True)
  norms = np.maximum(norms, 1e-10)  # Prevent division by zero.
  stainMatrix = W.T / norms

  # Safeguard 5: Final check for NaNs in the normalized matrix.
  if (np.isnan(stainMatrix).any()):
    return defaultStain

  # Ensure Hematoxylin is the first row (typically has a higher Red channel value).
  if (stainMatrix[0, 0] < stainMatrix[1, 0]):
    stainMatrix = stainMatrix[[1, 0], :]

  return stainMatrix.astype(np.float64)


class VahadaneColorNormalization(object):
  r'''
  Class for performing Vahadane color normalization on histopathology images.
  Uses Sparse Non-negative Matrix Factorization (SNMF) to estimate the stain matrix.

  Attributes:
    targetStainMatrix (numpy.ndarray): Target stain matrix.
    targetConcentrationsMax (numpy.ndarray): Target maximum concentrations for each stain.
    isFit (bool): Flag to indicate if the model has been fitted.
    beta (float): Threshold for OD values.
    lambda1 (float): Sparsity penalty for dictionary learning.
  '''

  def __init__(self, beta=0.15, lambda1=0.01):
    r'''
    Initialize the VahadaneColorNormalization object.
    '''

    # Initialize target stain matrix as None.
    self.targetStainMatrix = None  # Initialize target stain matrix as None.
    # Initialize target max concentrations as None.
    self.targetConcentrationsMax = None  # Initialize target max concentrations as None.
    # Flag to check if the model has been fitted.
    self.isFit = False  # Flag to check if the model has been fitted.
    # Store the beta threshold.
    self.beta = beta  # Store the beta threshold.
    # Store the lambda1 penalty.
    self.lambda1 = lambda1  # Store the lambda1 penalty.

  def Fit(self, img):
    r'''
    Fit the model to a target image by calculating its stain matrix and max concentrations.

    Parameters:
      img (numpy.ndarray): Target RGB image.
    '''

    # Find the stain matrix for the target image using SNMF.
    self.targetStainMatrix = FindStainMatrixVahadane(
      img, beta=self.beta,
      lambda1=self.lambda1
    )  # Find the stain matrix.

    # Convert to OD space.
    odImage = RGB2OD(img)  # Convert to OD space.
    # Reshape to 2D array (pixels x channels) and transpose to (channels x pixels).
    odImageReshaped = odImage.reshape(-1, odImage.shape[-1]).T  # Reshape and transpose.

    # Calculate stain concentrations using the pseudo-inverse.
    # .T is required because targetStainMatrix is (2, 3). Its transpose is (3, 2),
    # and its pseudo-inverse is (2, 3), which correctly multiplies with odImageReshaped (3, N) to yield (2, N).
    stainConcentrations = np.linalg.pinv(self.targetStainMatrix.T) @ odImageReshaped  # Calculate concentrations.

    # Get the 99th percentile of each stain concentration as the target max.
    self.targetConcentrationsMax = np.percentile(stainConcentrations, 99, axis=1)  # Get the 99th percentile.

    # Set the flag to indicate the model has been fitted.
    self.isFit = True  # Set the flag to indicate the model has been fitted.

  def Normalize(self, img):
    r'''
    Normalize an input image using the fitted target stain matrix and concentrations.

    Parameters:
      img (numpy.ndarray): Input RGB image.

    Returns:
      numpy.ndarray: Normalized RGB image.
    '''

    # Check if the model has been fitted.
    if (not self.isFit):
      raise RuntimeError("Model has not been fitted. Call Fit() first.")  # Raise an error if not fitted.

    # Find the stain matrix for the source image using SNMF.
    sourceStainMatrix = FindStainMatrixVahadane(
      img, beta=self.beta,
      lambda1=self.lambda1
    )  # Find the source stain matrix.

    # Convert to OD space.
    odImage = RGB2OD(img)  # Convert to OD space.
    # Reshape and transpose.
    odImageReshaped = odImage.reshape(-1, odImage.shape[-1]).T  # Reshape and transpose.

    # Calculate source concentrations.
    sourceConcentrations = np.linalg.pinv(sourceStainMatrix.T) @ odImageReshaped  # Calculate source concentrations.

    # Get the 99th percentile of source concentrations.
    sourceConcentrationsMax = np.percentile(sourceConcentrations, 99, axis=1)  # Get the 99th percentile.

    # Guard against division by zero.
    sourceConcentrationsMax = np.maximum(sourceConcentrationsMax, 1e-6)  # Guard against division by zero.

    # Normalize concentrations by scaling to the target max.
    normalizedConcentrations = sourceConcentrations * (
      self.targetConcentrationsMax[:, np.newaxis] / sourceConcentrationsMax[
      :, np.newaxis]
    )  # Normalize concentrations.

    # Recombine the normalized OD using the target stain matrix.
    # .T is required because targetStainMatrix is (2, 3). Its transpose is (3, 2),
    # which correctly multiplies with normalizedConcentrations (2, N) to yield (3, N).
    normalizedOD = self.targetStainMatrix.T @ normalizedConcentrations  # Recombine the normalized OD.

    # Reshape back to original image dimensions (H x W x C).
    normalizedOD = normalizedOD.T.reshape(img.shape)  # Reshape back to image dimensions.

    # Convert back to RGB.
    normalizedImg = OD2RGB(normalizedOD)

    # Return the normalized image.
    return normalizedImg


class MacenkoColorNormalization(object):
  r'''
  Class for performing Macenko color normalization on histopathology images.

  Attributes:
    targetStainMatrix (numpy.ndarray): Target stain matrix.
    targetConcentrationsMax (numpy.ndarray): Target maximum concentrations for each stain.
    isFit (bool): Flag to indicate if the model has been fitted.
    beta (float): Threshold for OD values.
    alpha (float): Percentile for stain orientation.
  '''

  def __init__(self, beta=0.15, alpha=1.0):
    r'''
    Initialize the MacenkoColorNormalization object.
    '''

    # Initialize target stain matrix as None.
    self.targetStainMatrix = None  # Initialize target stain matrix as None.
    # Initialize target max concentrations as None.
    self.targetConcentrationsMax = None  # Initialize target max concentrations as None.
    # Flag to check if the model has been fitted.
    self.isFit = False  # Flag to check if the model has been fitted.
    # Store the beta threshold.
    self.beta = beta  # Store the beta threshold.
    # Store the alpha percentile.
    self.alpha = alpha  # Store the alpha percentile.

  def Fit(self, img):
    r'''
    Fit the model to a target image by calculating its stain matrix and max concentrations.

    Parameters:
      img (numpy.ndarray): Target RGB image.
    '''

    # Find the stain matrix for the target image.
    self.targetStainMatrix = FindStainMatrixMacenko(img, beta=self.beta, alpha=self.alpha)  # Find the stain matrix.

    # Convert to OD space.
    odImage = RGB2OD(img)  # Convert to OD space.
    # Reshape to 2D array (pixels x channels) and transpose to (channels x pixels).
    odImageReshaped = odImage.reshape(-1, odImage.shape[-1]).T  # Reshape and transpose.

    # Calculate stain concentrations using the pseudo-inverse.
    stainConcentrations = np.linalg.pinv(self.targetStainMatrix.T) @ odImageReshaped  # Calculate concentrations.

    # Get the 99th percentile of each stain concentration as the target max.
    self.targetConcentrationsMax = np.percentile(stainConcentrations, 99, axis=1)  # Get the 99th percentile.

    # Set the flag to indicate the model has been fitted.
    self.isFit = True  # Set the flag to indicate the model has been fitted.

  def Normalize(self, img):
    r'''
    Normalize an input image using the fitted target stain matrix and concentrations.

    Parameters:
      img (numpy.ndarray): Input RGB image.

    Returns:
      numpy.ndarray: Normalized RGB image.
    '''

    # Check if the model has been fitted.
    if (not self.isFit):
      raise RuntimeError("Model has not been fitted. Call Fit() first.")  # Raise an error if not fitted.

    # Find the stain matrix for the source image.
    sourceStainMatrix = FindStainMatrixMacenko(img, beta=self.beta, alpha=self.alpha)  # Find the source stain matrix.

    # Convert to OD space.
    odImage = RGB2OD(img)  # Convert to OD space.
    # Reshape and transpose.
    odImageReshaped = odImage.reshape(-1, odImage.shape[-1]).T  # Reshape and transpose.

    # Calculate source concentrations.
    sourceConcentrations = np.linalg.pinv(sourceStainMatrix.T) @ odImageReshaped  # Calculate source concentrations.

    # Get the 99th percentile of source concentrations.
    sourceConcentrationsMax = np.percentile(sourceConcentrations, 99, axis=1)  # Get the 99th percentile.

    # Guard against division by zero.
    sourceConcentrationsMax = np.maximum(sourceConcentrationsMax, 1e-6)  # Guard against division by zero.

    # Normalize concentrations by scaling to the target max.
    normalizedConcentrations = sourceConcentrations * (
      self.targetConcentrationsMax[:, np.newaxis] / sourceConcentrationsMax[
      :, np.newaxis]
    )  # Normalize concentrations.

    # Recombine the normalized OD using the target stain matrix.
    # .T is required because targetStainMatrix is (2, 3). Its transpose is (3, 2),
    # which correctly multiplies with normalizedConcentrations (2, N) to yield (3, N).
    normalizedOD = self.targetStainMatrix.T @ normalizedConcentrations  # Recombine the normalized OD.

    # Reshape back to original image dimensions (H x W x C).
    normalizedOD = normalizedOD.T.reshape(img.shape)  # Reshape back to image dimensions.

    # Convert back to RGB.
    normalizedImg = OD2RGB(normalizedOD)  # Convert back to RGB.

    # Return the normalized image.
    return normalizedImg  # Return the normalized image.


class ReinhardColorNormalization(object):
  r'''
  Class for performing Reinhard color normalization on histopathology images.

  Attributes:
    targetMeans (list): Target means for LAB channels.
    targetStds (list): Target standard deviations for LAB channels.
    isFit (bool): Flag to indicate if the model has been fitted.
    eps (float): Small epsilon to avoid division by zero.
  '''

  def __init__(self):
    r'''
    Initialize the ReinhardColorNormalization object.
    '''

    # Initialize target means as None.
    self.targetMeans = None  # Initialize target means as None.
    # Initialize target standard deviations as None.
    self.targetStds = None  # Initialize target standard deviations as None.
    # Flag to check if the model has been fitted.
    self.isFit = False  # Flag to check if the model has been fitted.
    self.eps = 1e-6  # Small epsilon to avoid division by zero.

  def Fit(self, img):
    r'''
    Fit the model to a target image by calculating its LAB means and standard deviations.

    Parameters:
      img (numpy.ndarray): Target RGB image.
    '''

    # Standardize the brightness of the input image.
    target = StandarizeBrightness(img)  # Standardize the brightness of the input image.
    # Convert the input RGB image to LAB and split into channels.
    L, A, B = RGB2LAB(target)  # Convert the input RGB image to LAB and split into channels.

    # Calculate mean and standard deviation.
    # Store the target means.
    self.targetMeans = [np.mean(L), np.mean(A), np.mean(B)]
    # Store the target standard deviations.
    self.targetStds = [np.std(L), np.std(A), np.std(B)]

    # Set the flag to indicate the model has been fitted.
    self.isFit = True  # Set the flag to indicate the model has been fitted.

  def Normalize(self, img):
    r'''
    Normalize an input image using the fitted target means and standard deviations.

    Parameters:
      img (numpy.ndarray): Input RGB image.

    Returns:
      numpy.ndarray: Normalized RGB image.
    '''

    # Check if the model has been fitted.
    if (not self.isFit):
      raise RuntimeError("Model has not been fitted. Call Fit() first.")

    # Ensure the input is a valid RGB image.
    if ((len(img.shape) != 3) or (img.shape[2] != 3)):
      raise ValueError("Input image must be a 3-channel RGB image.")

    # Standardize the brightness of the input image.
    target = StandarizeBrightness(img)  # Standardize the brightness of the input image.
    # Convert the input RGB image to LAB and split into channels.
    L, A, B = RGB2LAB(target)  # Convert the input RGB image to LAB and split into channels.

    # Calculate mean and standard deviation.
    means = [np.mean(L), np.mean(A), np.mean(B)]
    stds = [np.std(L), np.std(A), np.std(B)]

    # Subtract the mean from each LAB channel.
    f1 = [
      L - means[0],  # Subtract the mean from the L channel.
      A - means[1],  # Subtract the mean from the A channel.
      B - means[2]  # Subtract the mean from the B channel.
    ]
    # Scale each LAB channel by the ratio of target std to source std.
    # Guard against division by zero by clamping the source std to EPS.
    f2 = [
      self.targetStds[0] / max(stds[0], self.eps),  # Scale the L channel by the ratio of target std to source std.
      self.targetStds[1] / max(stds[1], self.eps),  # Scale the A channel by the ratio of target std to source std.
      self.targetStds[2] / max(stds[2], self.eps)  # Scale the B channel by the ratio of target std to source std.
    ]
    # Add the target mean to each LAB channel.
    f3 = [
      self.targetMeans[0],  # Add the target mean to the L channel.
      self.targetMeans[1],  # Add the target mean to the A channel.
      self.targetMeans[2]  # Add the target mean to the B channel.
    ]
    # Apply the normalization transformation to each LAB channel.
    I1 = f1[0] * f2[0] + f3[0]  # Apply the normalization transformation to the L channel.
    I2 = f1[1] * f2[1] + f3[1]  # Apply the normalization transformation to the A channel.
    I3 = f1[2] * f2[2] + f3[2]  # Apply the normalization transformation to the B channel.
    # Convert the normalized LAB image back to RGB.
    I = LAB2RGB(I1, I2, I3)  # Convert the normalized LAB image back to RGB.
    # Return the normalized RGB image.
    return I  # Return the normalized RGB image.


def GetFitNormalizer(
  slide,  # The OpenSlide object for the slide.
  normSize=(1024 * 8, 1024 * 8),  # The size of the region to fit the normalizer.
  method="reinhard",  # The normalization method to use.
):
  r'''
  Create and fit a normalizer for a given slide.

  Parameters:
    slide (OpenSlide): The OpenSlide object for the slide.
    normSize (tuple): The size of the region to fit the normalizer.
    method (str): The normalization method to use ("reinhard", "macenko", "vahadane", or "histogram").

  Returns:
    object: The fitted normalizer.
  '''

  # Extract the region to fit the normalizer.
  centerLocation = (
    slide.dimensions[0] // 2 - normSize[0] // 2,
    slide.dimensions[1] // 2 - normSize[1] // 2,
  )
  regionToFit = slide.read_region(centerLocation, 0, normSize)
  regionToFit = np.array(regionToFit)

  # Initialize the normalizer based on the specified method.
  if (method == "reinhard"):
    normalizer = ReinhardColorNormalization()  # Initialize the Reinhard color normalizer.
  elif (method == "macenko"):
    normalizer = MacenkoColorNormalization()  # Initialize the Macenko color normalizer.
  elif (method == "vahadane"):
    normalizer = VahadaneColorNormalization()  # Initialize the Vahadane color normalizer.
  elif (method == "histogram"):
    normalizer = HistogramColorNormalization()  # Initialize the Histogram color normalizer.
  else:
    raise ValueError("Unsupported normalization method.")  # Raise an error for unsupported methods.

  # Fit the normalizer to the region.
  normalizer.Fit(regionToFit)  # Fit the normalizer to the region.
  # Return the fitted normalizer.
  return normalizer  # Return the fitted normalizer.


def CreateAverageHistogram(refImages, noChannels=4):
  r'''
  Create an average histogram from a list of reference images.

  Parameters:
    refImages (list): List of reference images.
    noChannels (int): Number of channels in the images.

  Returns:
    list: Average histogram for each channel.
  '''

  # Number of reference images.
  noImages = len(refImages)  # Number of reference images.
  # Initialize list to store histograms for each channel.
  channelHist = [[] for _ in range(noChannels)]  # Initialize list to store histograms for each channel.

  # Iterate over each image with a progress bar.
  for i in tqdm.tqdm(range(noImages)):  # Iterate over each image with a progress bar.
    # Iterate over each channel.
    for channel in range(noChannels):  # Iterate over each channel.
      # Compute histogram for the current channel.
      hist = np.histogram(
        np.array(refImages[i])[..., channel],  # Compute histogram for the current channel.
        bins=256,  # Number of bins for the histogram.
        range=(0, 255),  # Range of pixel values.
        density=True,  # Normalize the histogram.
      )[0]
      # Append the histogram to the list.
      channelHist[channel].append(hist)  # Append the histogram to the list.

  # Compute the average histogram for each channel.
  avgHist = [
    np.mean(channelHist[i], axis=0)  # Compute the average histogram for each channel.
    for i in tqdm.tqdm(range(noChannels))
  ]

  # Return the average histogram.
  return avgHist  # Return the average histogram.


def CreateLUTFromHistogram(avgHist):
  r'''
  Create a Look-Up Table (LUT) from an average histogram.

  Parameters:
    avgHist (list): Average histogram for each channel.

  Returns:
    list: LUT for each channel.
  '''

  # Compute the cumulative distribution function (CDF) for each histogram.
  cdf = [np.cumsum(hist) for hist in avgHist]  # Compute the cumulative distribution function (CDF) for each histogram.
  # Find the minimum value of each CDF.
  cdfMin = [np.min(c) for c in cdf]  # Find the minimum value of each CDF.
  # Find the maximum value of each CDF.
  cdfMax = [np.max(c) for c in cdf]  # Find the maximum value of each CDF.

  # Normalize the CDF to the range [0, 255].
  cdfNorm = [
    ((cdf[i] - cdfMin[i]) * 255 / (cdfMax[i] - cdfMin[i]))  # Normalize the CDF to the range [0, 255].
    for i in tqdm.tqdm(range(len(cdf)))
  ]

  # Create the LUT using linear interpolation.
  lut = [
    np.interp(range(256), range(256), cdfNorm[i])  # Create the LUT using linear interpolation.
    for i in tqdm.tqdm(range(len(cdfNorm)))
  ]

  # Return the LUT.
  return lut  # Return the LUT.


def ApplyLUT(image, lut, noChannels=4):
  r'''
  Apply a Look-Up Table (LUT) to an image.

  Parameters:
    image (numpy.ndarray): Input image.
    lut (list): LUT for each channel.
    noChannels (int): Number of channels in the image.

  Returns:
    numpy.ndarray: Image with LUT applied.
  '''

  # Iterate over each channel.
  for channel in range(noChannels):  # Iterate over each channel.
    # Apply the LUT to the channel.
    image[..., channel] = lut[channel][image[..., channel]]  # Apply the LUT to the channel.
    # Clip values to the range [0, 255].
    image[..., channel] = np.clip(image[..., channel], 0, 255)  # Clip values to the range [0, 255].
    # Convert to unsigned 8-bit integer.
    image[..., channel] = image[..., channel].astype(np.uint8)  # Convert to unsigned 8-bit integer.

  # Return the modified image.
  return image  # Return the modified image.


def RGB2OD(img):
  r'''
  Convert an RGB image to Optical Density (OD) space.

  Parameters:
    img (numpy.ndarray): Input RGB image (range [0, 255]).

  Returns:
    numpy.ndarray: Image in OD space.
  '''

  # Ensure the input is in the range [0, 255].
  img = np.clip(img, 0, 255).astype(np.float64)

  # Normalize to [0, 1] and avoid division by zero.
  image = img / 255.0
  image = np.clip(image, 1e-10, 1.0)  # Use a smaller clipping value for better precision.

  # Compute optical density.
  odImage = -np.log(image)
  # Return the image in OD space.
  return odImage


def OD2RGB(img):
  r'''
  Convert an image from Optical Density (OD) space back to RGB.

  Parameters:
    img (numpy.ndarray): Image in OD space.

  Returns:
    numpy.ndarray: RGB image (range [0, 255]).
  '''

  # Ensure the input is valid (non-negative).
  # if (np.any(img < 0)):
  #   raise ValueError("Input OD values must be non-negative.")

  # Clip any small negative values to 0.0 to handle floating-point inaccuracies
  # that arise from matrix multiplications (e.g., pseudo-inverse calculations).
  img = np.maximum(img, 0.0)

  # Convert OD back to RGB using the inverse transform.
  image = np.exp(-img) * 255.0
  image = np.clip(image, 0, 255).astype(np.uint8)  # Clip and convert to uint8.
  # Return the RGB image.
  return image


def StandarizeBrightness(img):
  r'''
  Standardize the brightness of an image using the 90th percentile.

  Parameters:
    img (numpy.ndarray): Input image (assumed to be in range [0, 255]).

  Returns:
    numpy.ndarray: Brightness-standardized image.
  '''

  # Ensure the image is in the range [0, 255].
  img = np.clip(img, 0, 255).astype(np.float32)

  # Calculate the 90th percentile for each channel independently.
  if (len(img.shape) == 3):  # Multi-channel image (e.g., RGB).
    # Percentile for each channel.
    p = np.percentile(img, 90, axis=(0, 1))  # Percentile for each channel.
    # Scale each channel independently.
    I = img * 255.0 / p[None, None, :]  # Scale each channel independently.
  else:  # Single-channel image (e.g., grayscale).
    p = np.percentile(img, 90)
    I = img * 255.0 / p

  # Clip values to ensure they are within the valid range [0, 255].
  I = np.clip(I, 0, 255).astype(np.uint8)
  # Return the brightness-standardized image.
  return I  # Return the brightness-standardized image.


def NormalizeRows(A):
  r'''
  Normalize the rows of a matrix.

  Parameters:
    A (numpy.ndarray): Input matrix.

  Returns:
    numpy.ndarray: Matrix with normalized rows.
  '''

  # Normalize each row (L2 normalization).
  aNorm = A / np.linalg.norm(A, axis=1)[:, np.newaxis]  # Normalize each row.
  # Return the normalized matrix.
  return aNorm  # Return the normalized matrix.


def FindStainMatrixMacenko(img, beta=0.15, alpha=1.0):
  r'''
  Find the stain matrix using Macenko's method.

  Parameters:
    img (numpy.ndarray): Input RGB image (range [0, 255]).
    beta (float): Threshold for OD values.
    alpha (float): Percentile for stain orientation.

  Returns:
    numpy.ndarray: Stain matrix (2x3).
  '''

  # Ensure the input is a 3-channel RGB image.
  if ((len(img.shape) != 3) or (img.shape[2] != 3)):
    raise ValueError("Input image must be a 3-channel RGB image.")

  # Convert to OD space and reshape.
  odImage = RGB2OD(img)
  odImageReshaped = odImage.reshape(-1, odImage.shape[-1])  # Reshape the image.

  # Filter out low OD values.
  odImageReshaped = odImageReshaped[(odImageReshaped > beta).any(axis=1), :]

  # Compute the covariance matrix and perform eigen-decomposition.
  cov_matrix = np.cov(odImageReshaped, rowvar=False)
  eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

  # Select the two most significant eigenvectors.
  V = eigenvectors[:, [2, 1]]  # Corresponding to the two largest eigenvalues.

  # Ensure the first element of each eigenvector is positive.
  if (V[0, 0] < 0):
    V[:, 0] *= -1
  if (V[0, 1] < 0):
    V[:, 1] *= -1

  # Project the data onto the eigenvectors.
  projectedData = np.dot(odImageReshaped, V)
  phi = np.arctan2(projectedData[:, 1], projectedData[:, 0])

  # Find the min and max orientation of the stains.
  minPhi = np.percentile(phi, alpha)
  maxPhi = np.percentile(phi, 100 - alpha)

  # Compute the stain vectors.
  vec1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
  vec2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

  # Ensure the first stain vector is dominant.
  if (vec1[0] > vec2[0]):
    stainMatrix = np.array([vec1, vec2])
  else:
    stainMatrix = np.array([vec2, vec1])

  # Normalize the stain matrix.
  stainMatrix = stainMatrix / np.linalg.norm(stainMatrix, axis=1, keepdims=True)

  # Return the stain matrix.
  return stainMatrix.astype(np.float64)  # Return the stain matrix.


def ConcentrateStainMatrixMacenko(img, stainMatrix, lambdaVal=0.01):
  r'''
  Concentrate the stain matrix using LASSO.

  Parameters:
    img (numpy.ndarray): Input RGB image (range [0, 255]).
    stainMatrix (numpy.ndarray): Stain matrix (2x3).
    lambdaVal (float): L1 penalty for LASSO.

  Returns:
    numpy.ndarray: Concentrated stain matrix (2x3).
  '''

  # Ensure the input is a 3-channel RGB image.
  if ((len(img.shape) != 3) or (img.shape[2] != 3)):
    raise ValueError("Input image must be a 3-channel RGB image.")

  # Ensure the stain matrix has the correct shape.
  if (stainMatrix.shape != (2, 3)):
    raise ValueError("Stain matrix must have shape (2, 3).")

  # Convert to OD space and reshape.
  odImage = RGB2OD(img).astype(np.float64)
  odImageReshaped = odImage.reshape(-1, img.shape[-1])

  # Apply LASSO to concentrate the stains.
  value = spams.lasso(
    odImageReshaped.T,  # Transpose the OD image.
    D=stainMatrix.T,  # Transpose the stain matrix.
    mode=2,  # LASSO mode.
    lambda1=lambdaVal,  # L1 penalty.
    pos=True,  # Ensure positive coefficients.
  ).toarray()  # Convert to a dense array.
  value = value.T  # Transpose back to the original shape.

  # Normalize the concentrated stain matrix.
  # value = value / np.linalg.norm(value, axis=1, keepdims=True)

  # Return the concentrated stain matrix.
  return value.astype(np.float64)  # Return the concentrated stain matrix.


def ApplyReinhard(img, targetMeans, targetStds, eps=1e-6):
  r'''
  Apply Reinhard color normalization to an image.

  Parameters:
    img (numpy.ndarray): Input image.
    targetMeans (list): Target means for LAB channels.
    targetStds (list): Target standard deviations for LAB channels.
    eps (float): Small epsilon to avoid division by zero.

  Returns:
    numpy.ndarray: Normalized image.
  '''

  # Ensure the input is a valid RGB image.
  if ((len(img.shape) != 3) or (img.shape[2] != 3)):
    raise ValueError("Input image must be a 3-channel RGB image.")

  # Standardize the brightness of the image.
  imgMod = StandarizeBrightness(img)  # Standardize the brightness of the image.

  # Convert the input RGB image to LAB and split into channels.
  L, A, B = RGB2LAB(imgMod)  # Convert the input RGB image to LAB and split into channels.

  # Calculate mean and standard deviation.
  means = [np.mean(L), np.mean(A), np.mean(B)]
  stds = [np.std(L), np.std(A), np.std(B)]

  # Normalize L channel using Reinhard transform (guard against division by zero).
  I1 = (L - means[0]) * (targetStds[0] / max(stds[0], eps)) + targetMeans[0]  # Normalize L channel.
  # Normalize A channel.
  I2 = (A - means[1]) * (targetStds[1] / max(stds[1], eps)) + targetMeans[1]  # Normalize A channel.
  # Normalize B channel.
  I3 = (B - means[2]) * (targetStds[2] / max(stds[2], eps)) + targetMeans[2]  # Normalize B channel.

  # Convert back to RGB color space.
  merged = LABSplit2RGB(I1, I2, I3)  # Convert back to RGB color space.
  # Return the normalized image.
  return merged  # Return the normalized image.


class HistogramColorNormalization(object):
  r'''
  Class for performing Histogram Matching (Specification) color normalization.

  Attributes:
    targetHist (list): Target histograms for each channel.
    isFit (bool): Flag to indicate if the model has been fitted.
  '''

  def __init__(self):
    r'''
    Initialize the HistogramColorNormalization object.
    '''

    # Initialize target histograms as None.
    self.targetHist = None  # Initialize target histograms as None.
    # Flag to check if the model has been fitted.
    self.isFit = False  # Flag to check if the model has been fitted.

  def Fit(self, img):
    r'''
    Fit the model to a target image by calculating its channel histograms.

    Parameters:
      img (numpy.ndarray): Target RGB image.
    '''

    # Ensure the image is in the range [0, 255].
    img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure the image is in the range [0, 255].

    # Initialize list to store histograms.
    self.targetHist = []  # Initialize list to store histograms.

    # Iterate over each channel.
    for channel in range(3):  # Iterate over each channel.
      # Compute histogram for the current channel.
      hist, _ = np.histogram(
        img[..., channel],  # Compute histogram for the current channel.
        bins=256,  # Number of bins for the histogram.
        range=(0, 256),  # Range of pixel values.
        density=True  # Normalize the histogram.
      )
      # Append the histogram to the list.
      self.targetHist.append(hist)  # Append the histogram to the list.

    # Set the flag to indicate the model has been fitted.
    self.isFit = True  # Set the flag to indicate the model has been fitted.

  def Normalize(self, img):
    r'''
    Normalize an input image using the fitted target histograms.

    Parameters:
      img (numpy.ndarray): Input RGB image.

    Returns:
      numpy.ndarray: Normalized RGB image.
    '''

    # Check if the model has been fitted.
    if (not self.isFit):
      raise RuntimeError("Model has not been fitted. Call Fit() first.")  # Raise an error if not fitted.

    # Ensure the image is in the range [0, 255].
    img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure the image is in the range [0, 255].
    # Initialize the normalized image.
    normalizedImg = np.zeros_like(img)  # Initialize the normalized image.

    # Iterate over each channel.
    for channel in range(3):  # Iterate over each channel.
      # Compute source CDF.
      srcHist, _ = np.histogram(
        img[..., channel],  # Compute histogram for the current channel.
        bins=256,  # Number of bins for the histogram.
        range=(0, 256),  # Range of pixel values.
        density=True  # Normalize the histogram.
      )
      srcCdf = np.cumsum(srcHist)  # Compute the cumulative distribution function.
      srcCdf = (srcCdf - srcCdf.min()) * 255 / (srcCdf.max() - srcCdf.min() + 1e-10)  # Normalize the CDF.

      # Compute target CDF.
      tgtCdf = np.cumsum(self.targetHist[channel])  # Compute the cumulative distribution function.
      tgtCdf = (tgtCdf - tgtCdf.min()) * 255 / (tgtCdf.max() - tgtCdf.min() + 1e-10)  # Normalize the CDF.

      # Create Look-Up Table (LUT) using linear interpolation.
      lut = np.interp(srcCdf, tgtCdf, range(256)).astype(np.uint8)  # Create the LUT.

      # Apply the LUT to the channel.
      normalizedImg[..., channel] = lut[img[..., channel]]  # Apply the LUT to the channel.

    # Return the normalized image.
    return normalizedImg  # Return the normalized image.
