'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 31th, 2025
# Last Modification Date: Jul 31th, 2025
# Permissions and Citation: Refer to the README file.
'''

import cv2  # OpenCV for image processing.
import numpy as np  # NumPy for numerical operations.
import spams  # SPAMS for sparse matrix computations.
import tqdm  # TQDM for progress bars.


def RGB2LAB(img, isNorm=True):
  """
  Convert an RGB image to the LAB color space and normalize the channels.

  Parameters:
      img (numpy.ndarray): Input RGB image.
      isNorm (bool): Normalize the LAB channels.

  Returns:
      tuple: Normalized LAB channels (L, A, B).
  """
  I = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # Convert RGB to LAB color space.
  I = I.astype(np.float32)  # Convert to float32 for further processing.
  I1, I2, I3 = cv2.split(I)  # Split the LAB image into its three channels: L, A, and B.
  if (isNorm):  # Normalize the LAB channels.
    I1 = I1 / 2.55  # Normalize the L channel to the range [0, 100].
    I2 = I2 - 128.0  # Normalize the A channel to the range [-128, 127].
    I3 = I3 - 128.0  # Normalize the B channel to the range [-128, 127].
  return I1, I2, I3  # Return the normalized LAB channels.


def LAB2RGB(I1, I2, I3, isNorm=True):
  """
  Convert normalized LAB channels back to the RGB color space.

  Parameters:
      I1 (numpy.ndarray): Normalized L channel.
      I2 (numpy.ndarray): Normalized A channel.
      I3 (numpy.ndarray): Normalized B channel.

  Returns:
      numpy.ndarray: RGB image.
  """
  if (isNorm):
    I1 = I1 * 2.55  # Denormalize the L channel back to the range [0, 255].
    I2 = I2 + 128.0  # Denormalize the A channel back to the range [0, 255].
    I3 = I3 + 128.0  # Denormalize the B channel back to the range [0, 255].
  I = cv2.merge((I1, I2, I3))  # Merge the LAB channels back into a single image.
  I = np.clip(I, 0, 255).astype(np.uint8)  # Clip the pixel values to ensure they are within the valid range [0, 255].
  I = cv2.cvtColor(I, cv2.COLOR_LAB2RGB)  # Convert the LAB image back to the RGB color space.
  return I  # Return the resulting RGB image.


def LabSplitMeanStd(img):
  """
  Compute the mean and standard deviation of the LAB channels of an RGB image.

  Parameters:
      img (numpy.ndarray): Input RGB image.

  Returns:
      tuple: Means and standard deviations of the LAB channels.
  """
  I1, I2, I3 = RGB2LAB(img)  # Convert the input RGB image to LAB and split into channels.
  m1, s1 = cv2.meanStdDev(I1)  # Calculate the mean and standard deviation of the L channel.
  m2, s2 = cv2.meanStdDev(I2)  # Calculate the mean and standard deviation of the A channel.
  m3, s3 = cv2.meanStdDev(I3)  # Calculate the mean and standard deviation of the B channel.
  means = [m1, m2, m3]  # Store the means in a list.
  stds = [s1, s2, s3]  # Store the standard deviations in a list.
  return means, stds  # Return the means and standard deviations.


class ReinhardColorNormalization(object):
  """
  A class for performing Reinhard color normalization on images.
  """

  def __init__(self):
    """
    Initialize the ReinhardColorNormalization object.
    """
    self.targetMeans = None  # Initialize target means as None.
    self.targetStds = None  # Initialize target standard deviations as None.
    self.isFit = False  # Flag to check if the model has been fitted.

  def Fit(self, img):
    """
    Fit the model to a target image by calculating its LAB means and standard deviations.

    Parameters:
        img (numpy.ndarray): Target RGB image.
    """
    target = StandarizeBrightness(img)  # Standardize the brightness of the input image.
    L, A, B = RGB2LAB(target)  # Convert the input RGB image to LAB and split into channels.

    # Calculate mean and standard deviation.
    self.targetMeans = [np.mean(L), np.mean(A), np.mean(B)]
    self.targetStds = [np.std(L), np.std(A), np.std(B)]

    # means, stds = LabSplitMeanStd(target)  # Calculate the mean and standard deviation of the LAB channels.
    # self.targetMeans = means  # Store the target means.
    # self.targetStds = stds  # Store the target standard deviations.
    self.isFit = True  # Set the flag to indicate the model has been fitted.

  def Normalize(self, img):
    """
    Normalize an input image using the fitted target means and standard deviations.

    Parameters:
        img (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: Normalized RGB image.
    """
    if (not self.isFit):
      raise RuntimeError("Model has not been fitted. Call Fit() first.")

    # Ensure the input is a valid RGB image.
    if ((len(img.shape) != 3) or (img.shape[2] != 3)):
      raise ValueError("Input image must be a 3-channel RGB image.")

    target = StandarizeBrightness(img)  # Standardize the brightness of the input image.
    L, A, B = RGB2LAB(target)  # Convert the input RGB image to LAB and split into channels.

    # Calculate mean and standard deviation.
    means = [np.mean(L), np.mean(A), np.mean(B)]
    stds = [np.std(L), np.std(A), np.std(B)]

    # target = StandarizeBrightness(img)  # Standardize the brightness of the input image.
    # I1, I2, I3 = RGB2LAB(target)  # Convert the RGB image to LAB and split into channels.
    # means, stds = LabSplitMeanStd(target)  # Calculate the mean and standard deviation of the LAB channels.

    f1 = [
      L - means[0],  # Subtract the mean from the L channel.
      A - means[1],  # Subtract the mean from the A channel.
      B - means[2]  # Subtract the mean from the B channel.
    ]
    f2 = [
      self.targetStds[0] / stds[0],  # Scale the L channel by the ratio of target std to source std.
      self.targetStds[1] / stds[1],  # Scale the A channel by the ratio of target std to source std.
      self.targetStds[2] / stds[2]  # Scale the B channel by the ratio of target std to source std.
    ]
    f3 = [
      self.targetMeans[0],  # Add the target mean to the L channel.
      self.targetMeans[1],  # Add the target mean to the A channel.
      self.targetMeans[2]  # Add the target mean to the B channel.
    ]
    I1 = f1[0] * f2[0] + f3[0]  # Apply the normalization transformation to the L channel.
    I2 = f1[1] * f2[1] + f3[1]  # Apply the normalization transformation to the A channel.
    I3 = f1[2] * f2[2] + f3[2]  # Apply the normalization transformation to the B channel.
    I = LAB2RGB(I1, I2, I3)  # Convert the normalized LAB image back to RGB.
    return I  # Return the normalized RGB image.


# Create the normalizers
def GetFitNormalizer(
  slide,  # The OpenSlide object for the slide.
  normSize=(1024 * 8, 1024 * 8),  # The size of the region to fit the normalizer.
):
  # Initialize the Reinhard color normalizer.
  normalizer = ReinhardColorNormalization()
  # Calculate the center location of the slide.
  centerLocation = (
    slide.dimensions[0] // 2,
    slide.dimensions[1] // 2,
  )
  # Adjust the center location to fit the normalization region.
  centerLocation = (
    centerLocation[0] - normSize[0] // 2,
    centerLocation[1] - normSize[1] // 2,
  )
  # Extract the region to fit the normalizer.
  regionToFit = slide.read_region(centerLocation, 0, normSize)
  # Convert the region to a NumPy array.
  regionToFit = np.array(regionToFit)
  # Fit the normalizer to the region.
  normalizer.Fit(regionToFit)
  # Return the fitted normalizer.
  return normalizer


def CreateAverageHistogram(refImages, noChannels=4):
  """
  Create an average histogram from a list of reference images.

  Parameters:
      refImages (list): List of reference images.
      noChannels (int): Number of channels in the images.

  Returns:
      list: Average histogram for each channel.
  """
  noImages = len(refImages)  # Number of reference images.
  channelHist = [[] for _ in range(noChannels)]  # Initialize list to store histograms for each channel.

  for i in tqdm.tqdm(range(noImages)):  # Iterate over each image with a progress bar.
    for channel in range(noChannels):  # Iterate over each channel.
      hist = np.histogram(
        np.array(refImages[i])[..., channel],  # Compute histogram for the current channel.
        bins=256,  # Number of bins for the histogram.
        range=(0, 255),  # Range of pixel values.
        density=True,  # Normalize the histogram.
      )[0]
      channelHist[channel].append(hist)  # Append the histogram to the list.

  avgHist = [
    np.mean(channelHist[i], axis=0)  # Compute the average histogram for each channel.
    for i in tqdm.tqdm(range(noChannels))
  ]

  return avgHist  # Return the average histogram.


def CreateLUTFromHistogram(avgHist):
  """
  Create a Look-Up Table (LUT) from an average histogram.

  Parameters:
      avgHist (list): Average histogram for each channel.

  Returns:
      list: LUT for each channel.
  """
  cdf = [np.cumsum(hist) for hist in avgHist]  # Compute the cumulative distribution function (CDF) for each histogram.
  cdfMin = [np.min(c) for c in cdf]  # Find the minimum value of each CDF.
  cdfMax = [np.max(c) for c in cdf]  # Find the maximum value of each CDF.

  cdfNorm = [
    ((cdf[i] - cdfMin[i]) * 255 / (cdfMax[i] - cdfMin[i]))  # Normalize the CDF to the range [0, 255].
    for i in tqdm.tqdm(range(len(cdf)))
  ]

  lut = [
    np.interp(range(256), range(256), cdfNorm[i])  # Create the LUT using linear interpolation.
    for i in tqdm.tqdm(range(len(cdfNorm)))
  ]

  return lut  # Return the LUT.


def ApplyLUT(image, lut, noChannels=4):
  """
  Apply a Look-Up Table (LUT) to an image.

  Parameters:
      image (numpy.ndarray): Input image.
      lut (list): LUT for each channel.
      noChannels (int): Number of channels in the image.

  Returns:
      numpy.ndarray: Image with LUT applied.
  """
  for channel in range(noChannels):  # Iterate over each channel.
    image[..., channel] = lut[channel][image[..., channel]]  # Apply the LUT to the channel.
    image[..., channel] = np.clip(image[..., channel], 0, 255)  # Clip values to the range [0, 255].
    image[..., channel] = image[..., channel].astype(np.uint8)  # Convert to unsigned 8-bit integer.

  return image  # Return the modified image.


def LABSplit2RGB(I1, I2, I3):
  """
  Convert LAB channels back to RGB color space.

  Parameters:
      I1 (numpy.ndarray): L channel.
      I2 (numpy.ndarray): A channel.
      I3 (numpy.ndarray): B channel.

  Returns:
      numpy.ndarray: RGB image.
  """
  I1 *= 2.55  # Denormalize L channel.
  I2 += 128.0  # Denormalize A channel.
  I3 += 128.0  # Denormalize B channel.
  I = cv2.merge((I1, I2, I3))  # Merge channels back into an image.
  I = np.clip(I, 0, 255)  # Clip values to the range [0, 255].
  I = I.astype(np.uint8)  # Convert to unsigned 8-bit integer.
  I = cv2.cvtColor(I, cv2.COLOR_LAB2RGB)  # Convert LAB to RGB color space.
  return I  # Return the RGB image.


def RGB2OD(img):
  """
  Convert an RGB image to Optical Density (OD) space.

  Parameters:
      img (numpy.ndarray): Input RGB image (range [0, 255]).

  Returns:
      numpy.ndarray: Image in OD space.
  """
  # Ensure the input is in the range [0, 255].
  img = np.clip(img, 0, 255).astype(np.float64)

  # Normalize to [0, 1] and avoid division by zero.
  image = img / 255.0
  image = np.clip(image, 1e-10, 1.0)  # Use a smaller clipping value for better precision.

  # Compute optical density.
  odImage = -np.log(image)
  return odImage


def OD2RGB(img):
  """
  Convert an image from Optical Density (OD) space back to RGB.

  Parameters:
      img (numpy.ndarray): Image in OD space.

  Returns:
      numpy.ndarray: RGB image (range [0, 255]).
  """
  # Ensure the input is valid (non-negative).
  if (np.any(img < 0)):
    raise ValueError("Input OD values must be non-negative.")

  # Convert OD back to RGB.
  image = np.exp(-img) * 255.0
  image = np.clip(image, 0, 255).astype(np.uint8)  # Clip and convert to uint8.
  return image


def StandarizeBrightness(img):
  """
  Standardize the brightness of an image using the 90th percentile.

  Parameters:
      img (numpy.ndarray): Input image (assumed to be in range [0, 255]).

  Returns:
      numpy.ndarray: Brightness-standardized image.
  """
  # Ensure the image is in the range [0, 255].
  img = np.clip(img, 0, 255).astype(np.float32)

  # Calculate the 90th percentile for each channel independently.
  if (len(img.shape) == 3):  # Multi-channel image (e.g., RGB).
    p = np.percentile(img, 90, axis=(0, 1))  # Percentile for each channel.
    I = img * 255.0 / p[None, None, :]  # Scale each channel independently.
  else:  # Single-channel image (e.g., grayscale).
    p = np.percentile(img, 90)
    I = img * 255.0 / p

  # Clip values to ensure they are within the valid range [0, 255].
  I = np.clip(I, 0, 255).astype(np.uint8)
  return I  # Return the brightness-standardized image.


def NormalizeRows(A):
  """
  Normalize the rows of a matrix.

  Parameters:
      A (numpy.ndarray): Input matrix.

  Returns:
      numpy.ndarray: Matrix with normalized rows.
  """
  aNorm = A / np.linalg.norm(A, axis=1)[:, np.newaxis]  # Normalize each row.
  return aNorm  # Return the normalized matrix.


def FindStainMatrixMacenko(img, beta=0.15, alpha=1.0):
  """
  Find the stain matrix using Macenko's method.

  Parameters:
      img (numpy.ndarray): Input RGB image (range [0, 255]).
      beta (float): Threshold for OD values.
      alpha (float): Percentile for stain orientation.

  Returns:
      numpy.ndarray: Stain matrix (2x3).
  """
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
  projectedData = np.dot(odImageReshaped, V)  # That.
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

  return stainMatrix.astype(np.float64)  # Return the stain matrix.


def ConcentrateStainMatrixMacenko(img, stainMatrix, lambdaVal=0.01):
  """
  Concentrate the stain matrix using LASSO.

  Parameters:
      img (numpy.ndarray): Input RGB image (range [0, 255]).
      stainMatrix (numpy.ndarray): Stain matrix (2x3).
      lambdaVal (float): L1 penalty for LASSO.

  Returns:
      numpy.ndarray: Concentrated stain matrix (2x3).
  """
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

  return value.astype(np.float64)  # Return the concentrated stain matrix.


def ApplyReinhard(img, targetMeans, targetStds):
  """
  Apply Reinhard color normalization to an image.

  Parameters:
      img (numpy.ndarray): Input image.
      targetMeans (list): Target means for LAB channels.
      targetStds (list): Target standard deviations for LAB channels.

  Returns:
      numpy.ndarray: Normalized image.
  """

  # Ensure the input is a valid RGB image.
  if ((len(img.shape) != 3) or (img.shape[2] != 3)):
    raise ValueError("Input image must be a 3-channel RGB image.")

  imgMod = StandarizeBrightness(img)  # Standardize the brightness of the image.

  L, A, B = RGB2LAB(imgMod)  # Convert the input RGB image to LAB and split into channels.

  # Calculate mean and standard deviation.
  means = [np.mean(L), np.mean(A), np.mean(B)]
  stds = [np.std(L), np.std(A), np.std(B)]

  # I1, I2, I3 = RGB2LAB(imgMod)  # Convert to LAB color space.
  # means, stds = LabSplitMeanStd(imgMod)  # Compute the mean and standard deviation of LAB channels.

  I1 = (L - means[0]) * (targetStds[0] / stds[0]) + targetMeans[0]  # Normalize L channel.
  I2 = (A - means[1]) * (targetStds[1] / stds[1]) + targetMeans[1]  # Normalize A channel.
  I3 = (B - means[2]) * (targetStds[2] / stds[2]) + targetMeans[2]  # Normalize B channel.

  merged = LABSplit2RGB(I1, I2, I3)  # Convert back to RGB color space.
  return merged  # Return the normalized image.
