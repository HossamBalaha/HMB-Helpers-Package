'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 31th, 2025
# Last Modification Date: Aug 19th, 2025
# Permissions and Citation: Refer to the README file.
'''

import cv2, PIL
import numpy as np


def MutualInformation(image1, image2, bins=100):
  from scipy.stats import entropy

  # Too Few Bins: Using very few bins (e.g., bins=5) can oversimplify the distribution,
  # leading to loss of important information about intensity relationships between the two images.
  # Too Many Bins: Using too many bins (e.g., bins=100) can result in sparse histograms,
  # especially if the images are small or have limited intensity variation.
  # Sparse histograms can lead to unreliable entropy estimates and noisy mutual information values.

  def _ComputeEntropy(hist):
    # Normalize the histogram to get probabilities
    probabilities = hist / float(hist.sum())
    # Compute entropy using scipy.stats.entropy
    return entropy(probabilities)

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions for both images.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Flatten the first image into a 1D array.
  img1Flat = np.asarray(image1).flatten().astype(np.float64)
  # Flatten the second image into a 1D array.
  img2Flat = np.asarray(image2).flatten().astype(np.float64)

  # Compute the joint histogram of the two flattened images using 20 bins.
  # The joint histogram represents the frequency of co-occurrence of pixel intensity values between the two images.
  # The output `hist2d` is a 2D array where each entry corresponds to the count of pixel pairs falling into specific bins.
  hist2d, _, _ = np.histogram2d(img1Flat, img2Flat, bins=bins)

  # Calculate the joint entropy from the joint histogram.
  # Joint entropy measures the uncertainty or randomness in the joint distribution of pixel intensities from both images.
  # The `entropy` function computes entropy based on the probabilities derived from the histogram.
  jointEntropy = _ComputeEntropy(hist2d.flatten())

  # Compute the marginal histogram for the first image using 20 bins.
  # The marginal histogram represents the distribution of pixel intensities in the first image.
  histImg1, _ = np.histogram(img1Flat, bins=bins)

  # Compute the marginal histogram for the second image using 20 bins.
  # Similarly, this histogram represents the distribution of pixel intensities in the second image.
  histImg2, _ = np.histogram(img2Flat, bins=bins)

  # Calculate the entropy of the first image's marginal histogram.
  # Entropy measures the uncertainty or randomness in the pixel intensity distribution of the first image.
  entropyImg1 = _ComputeEntropy(histImg1)

  # Calculate the entropy of the second image's marginal histogram.
  # This entropy quantifies the uncertainty or randomness in the pixel intensity distribution of the second image.
  entropyImg2 = _ComputeEntropy(histImg2)

  # Compute the Mutual Information (MI) between the two images.
  # MI quantifies the amount of information shared between the two images.
  # It is calculated as the sum of the individual entropies minus the joint entropy.
  mi = entropyImg1 + entropyImg2 - jointEntropy

  # Return the computed Mutual Information value.
  # This value indicates how much information one image provides about the other.
  return mi


def MutualInformationColor(image1, image2, bins=100):
  # Split the images into their respective color channels.
  split1 = cv2.split(image1)  # Split the first image into its color channels.
  split2 = cv2.split(image2)  # Split the second image into its color channels.

  # Compute MI for each channel.
  total = 0.0
  for i in range(len(split1)):
    # Calculate Mutual Information for each channel.
    miChannel = MutualInformation(split1[i], split2[i], bins=bins)
    # Add the MI of the current channel to the total.
    total += miChannel

  # Combine the results (e.g., average).
  miTotal = total / float(len(split1))

  # Return the total Mutual Information across all channels.
  return miTotal


def NormalizedMutualInformation(image1, image2):
  from sklearn.metrics import normalized_mutual_info_score

  # Convert the first input image into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values, which is necessary for comparison.
  img1Flat = np.asarray(image1).flatten().astype(np.float64)

  # Convert the second input image into a NumPy array and flatten it into a 1D array.
  # This step ensures both images are in the same format and ready for calculation of normalized mutual information.
  img2Flat = np.asarray(image2).flatten().astype(np.float64)

  # Calculate the Normalized Mutual Information (NMI) between the two flattened images.
  # The `normalized_mutual_info_score` function computes NMI, which measures the similarity between two datasets.
  # NMI is normalized to a range of [0, 1], where 0 indicates no mutual information and 1 indicates perfect correlation.
  nmi = normalized_mutual_info_score(img1Flat, img2Flat)

  # Return the computed Normalized Mutual Information value.
  # This value provides a normalized measure of the shared information between the two images,
  # making it easier to interpret.
  return nmi


def StructuralSimilarity(image1, image2, winSize=7):
  from skimage.metrics import structural_similarity as ssim

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Calculate the data range of the second image.
  # The data range is the difference between the maximum and minimum pixel values in the image.
  # This value is used to normalize the SSIM calculation.
  dRange = max(image1.max() - image1.min(), image2.max() - image2.min())

  # Compute the Structural Similarity Index (SSIM) between the two images.
  # The `ssim` function compares the reference image (`image1`) with the comparison image (`image2`).
  # The `data_range` parameter specifies the dynamic range of pixel values, ensuring proper normalization.
  ssimScore = ssim(
    image1,  # Reference image used as the baseline for comparison.
    image2,  # Comparison image being evaluated against the reference.
    data_range=dRange,  # Data range of the input images for normalization.
    win_size=winSize,  # Window size for local comparisons (default is 7x7).
  )

  # Return the computed SSIM score.
  # The SSIM score ranges from -1 to 1, where 1 indicates perfect similarity,
  # 0 indicates no correlation, and negative values indicate anti-correlation.
  return ssimScore


def NormalizedCrossCorrelation(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Normalize the first image.
  eps = 1e-8  # Small constant to avoid division by zero.
  image1 = (image1 - np.mean(image1)) / (np.std(image1) + eps)

  # Normalize the second image
  image2 = (image2 - np.mean(image2)) / (np.std(image2) + eps)

  # Calculate the Normalized Cross-Correlation (NCC).
  ncc = (
    np.sum(image1 * image2) /  # Dot product of the two normalized images.
    (np.sqrt(np.sum(image1 ** 2)) * np.sqrt(np.sum(image2 ** 2)))  # Product of the magnitudes of the two images.
  )

  # Return the computed NCC value.
  return ncc


def HistogramComparison(image1, image2, bins=256, eps=1e-10):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Dynamically determine the range of pixel values.
  pixelRange = (min(image1.min(), image2.min()), max(image1.max(), image2.max()))

  # Calculate the histogram of the first image using 256 bins and a range of [0, 256].
  # The histogram represents the distribution of pixel intensity values in the image.
  # The `np.histogram` function returns the histogram values and bin edges,
  # but only the histogram values are used here.
  hist1, _ = np.histogram(image1, bins=bins, range=pixelRange)

  # Calculate the histogram of the second image using the same parameters as the first image.
  # This ensures consistency in the comparison process.
  hist2, _ = np.histogram(image2, bins=bins, range=pixelRange)

  # Normalize the histogram of the first image by dividing each bin count by the total number of pixels.
  # Normalization ensures that the histograms are comparable regardless of the size of the images.
  hist1 = hist1 / (np.sum(hist1) + eps)

  # Normalize the histogram of the second image using the same method as the first image.
  # This step ensures that both histograms are scaled to the range [0, 1].
  hist2 = hist2 / (np.sum(hist2) + eps)

  # Calculate the histogram intersection between the two normalized histograms.
  # Histogram intersection is computed as the sum of the minimum values of corresponding bins in the two histograms.
  # This metric quantifies the overlap between the two histograms, with higher values indicating greater similarity.
  intersection = np.sum(np.minimum(hist1, hist2))

  # Return the computed histogram intersection value.
  # The result ranges from 0 to 1, where 0 indicates no overlap and 1 indicates identical histograms.
  return intersection


def UniversalQualityIndex(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Flatten the images into 1D arrays.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Compute the mean of the flattened first image.
  # The mean represents the average pixel intensity value of the image, providing a measure of its brightness.
  mean1 = np.mean(img1Flat)

  # Compute the mean of the flattened second image.
  # This provides a similar measure of brightness for the second image.
  mean2 = np.mean(img2Flat)

  # Compute the variance of the flattened first image.
  # Variance measures the spread or variability of pixel
  # intensity values around the mean, indicating texture or contrast.
  var1 = np.var(img1Flat)

  # Compute the variance of the flattened second image.
  # This provides a similar measure of texture or contrast for the second image.
  var2 = np.var(img2Flat)

  # Compute the covariance between the two flattened images.
  # Covariance measures how much the pixel intensity values of the two images vary together.
  # The `[0, 1]` index extracts the off-diagonal element of the covariance matrix, which represents the covariance.
  covar = np.cov(img1Flat, img2Flat)[0, 1]

  # Handle edge cases where the denominator becomes zero.
  if (var1 + var2 == 0 or mean1 ** 2 + mean2 ** 2 == 0):
    # If either the sum of variances or the sum of squared means is zero,
    # return a UQI value of 0.0 to indicate no similarity.
    return 0.0

  # Compute the Universal Quality Index (UQI) using the formula:
  # UQI = (4 * covariance * mean1 * mean2) / ((variance1 + variance2) * (mean1^2 + mean2^2))
  # This metric evaluates the similarity between the two images based on their means, variances, and covariance.
  # Higher UQI values indicate greater similarity, with a maximum value of 1 for identical images.
  uqi = (4.0 * covar * mean1 * mean2) / ((var1 + var2) * (mean1 ** 2 + mean2 ** 2))

  # Return the computed Universal Quality Index value.
  # The result quantifies the structural similarity between the two images,
  # accounting for luminance, contrast, and correlation.
  return uqi


def CosineSimilarityImages(image1, image2):
  from sklearn.metrics.pairwise import cosine_similarity

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64).reshape(1, -1)
  img2Flat = image2.flatten().astype(np.float64).reshape(1, -1)

  # Handle edge cases where one or both images are constant.
  if (np.all(img1Flat == img1Flat[0, 0]) or np.all(img2Flat == img2Flat[0, 0])):
    # If either image is constant (all pixel values are the same),
    # return a cosine similarity of 1.0, indicating perfect similarity.
    return 0.0

  # Calculate the cosine similarity between the two flattened images.
  # Cosine similarity measures the cosine of the angle between two vectors, providing a metric for their alignment.
  # The `cosine_similarity` function computes the similarity between the two row vectors (images).
  # The result is a 2D array where the value at position `[0][0]` represents the similarity score.
  cosSim = cosine_similarity(img1Flat, img2Flat)

  # Return the computed cosine similarity value.
  # The `[0][0]` index extracts the scalar similarity score from the 2D array returned by the function.
  # The score ranges from -1 to 1:
  #   - A score of 1 indicates perfect alignment (identical images).
  #   - A score of 0 indicates orthogonality (no similarity).
  #   - A score of -1 indicates opposite alignment (inverted images).
  return cosSim[0][0]


def PeakSignalToNoiseRatio(image1, image2, eps=1e-10):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Ensure the images are in floating-point format for numerical stability.
  image1 = image1.astype(np.float64)
  image2 = image2.astype(np.float64)

  # Calculate the Mean Squared Error (MSE) between the two images.
  # MSE measures the average squared difference between corresponding pixels in the two images.
  # A lower MSE indicates greater similarity between the images.
  mse = np.mean((image1 - image2) ** 2)

  # Check if the MSE is zero.
  # If the MSE is zero, it means the two images are identical, and the PSNR is infinite.
  # Return `float("inf")` to represent an infinite PSNR value in this case.
  if (mse == 0):
    return float("inf")

  # Dynamically determine the maximum possible pixel value.
  maxPixel = max(image1.max(), image2.max())

  # Calculate the Peak Signal-to-Noise Ratio (PSNR) using the formula:
  # PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
  # PSNR quantifies the quality of the image by comparing the maximum possible signal power to the noise level (MSE).
  # Higher PSNR values indicate better image quality and greater similarity between the two images.
  psnr = 20.0 * np.log10(maxPixel / np.sqrt(mse + eps))

  # Return the computed PSNR value.
  # The result is expressed in decibels (dB), with higher values indicating better quality or similarity.
  return psnr


def FeatureBasedSimilarity(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Create a SIFT (Scale-Invariant Feature Transform) object using OpenCV's `SIFT_create` method.
  # SIFT is used to detect keypoints and compute descriptors that represent distinctive features in the images.
  sift = cv2.SIFT_create()

  # Detect keypoints and compute descriptors for the first grayscale image.
  # Keypoints are distinctive points in the image (e.g., corners or edges), and descriptors encode their local appearance.
  # The `detectAndCompute` method returns keypoints and their corresponding descriptors.
  keypoints1, descriptors1 = sift.detectAndCompute(image1, None)

  # Detect keypoints and compute descriptors for the second grayscale image.
  # This step ensures that both images have their features extracted for comparison.
  keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

  # Handle edge cases where no keypoints are detected.
  if (descriptors1 is None or descriptors2 is None):
    # If either image has no keypoints, return a similarity score of 0.0,
    # indicating no similarity due to lack of detectable features.
    return 0.0

  # Create a Brute-Force Matcher (`BFMatcher`) object from OpenCV.
  # The BFMatcher performs exhaustive matching between descriptors from the two images.
  bf = cv2.BFMatcher()

  # Perform k-nearest neighbors (k=2) matching between the descriptors of the two images.
  # For each descriptor in the first image, the matcher finds the two closest matches in the second image.
  # This helps identify robust matches by comparing distances between descriptors.
  matches = bf.knnMatch(descriptors1, descriptors2, k=2)

  # #  Create a FLANN-based matcher
  # FLANN_INDEX_KDTREE = 1
  # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  # search_params = dict(checks=50)  # Higher checks improve accuracy at the cost of speed
  # flann = cv2.FlannBasedMatcher(index_params, search_params)
  #
  # # Perform k-nearest neighbors (k=2) matching
  # matches = flann.knnMatch(descriptors1, descriptors2, k=2)

  # Filter the matches using the Lowe's ratio test to identify "good" matches.
  # A match is considered good if the distance to the closest neighbor is significantly smaller than the distance to the second-closest neighbor.
  # The threshold (0.75) determines the strictness of the filtering.
  goodMatches = [m for m, n in matches if m.distance < 0.75 * n.distance]

  # Calculate the similarity score as the ratio of good matches to the minimum number of keypoints in either image.
  # This normalization ensures that the similarity score is not biased by the number of keypoints detected in each image.
  similarity = len(goodMatches) / min(len(keypoints1), len(keypoints2))

  # Return the computed similarity score.
  # The score ranges from 0 to 1, where:
  #   - A score of 1 indicates perfect similarity (all keypoints match).
  #   - A score of 0 indicates no similarity (no keypoints match).
  return similarity


def MeanSquaredError(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Calculate the Mean Squared Error (MSE) between the two images.
  # MSE measures the average squared difference between corresponding pixels in the two images.
  # A lower MSE indicates greater similarity between the images.
  mse = np.mean((image1 - image2) ** 2)

  # Return the computed MSE value.
  # The result quantifies the dissimilarity between the two images, with lower values indicating better agreement.
  return mse


def NormalizedMeanSquaredError(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Normalize the first image by subtracting its mean and dividing by its standard deviation.
  # This step centers the pixel values around zero (mean subtraction) and scales them to unit variance (division by standard deviation).
  # Normalization ensures that the comparison is not affected by differences in brightness or contrast between the images.
  image1 = (image1 - np.mean(image1)) / np.std(image1)

  # Normalize the second image using the same process as the first image.
  # This ensures both images are on the same scale for accurate comparison.
  image2 = (image2 - np.mean(image2)) / np.std(image2)

  # Calculate the Normalized Mean Squared Error (NMSE) between the two normalized images.
  # NMSE measures the average squared difference between corresponding pixels in the normalized images.
  # Lower NMSE values indicate greater similarity between the images.
  nmse = np.mean((image1 - image2) ** 2)

  # Return the computed NMSE value.
  # The result quantifies the dissimilarity between the two images after normalization, making it robust to differences in intensity scaling.
  return nmse


def EarthMoversDistance(image1, image2):
  from scipy.stats import wasserstein_distance

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Calculate the Earth Mover's Distance (EMD) between the two flattened images.
  # EMD measures the minimum "work" required to transform one distribution (image) into another.
  # The `wasserstein_distance` function computes the EMD based on the pixel intensity distributions of the two images.
  emd = wasserstein_distance(img1Flat, img2Flat)

  # Return the computed Earth Mover's Distance value.
  # The result quantifies the dissimilarity between the two images based on their pixel intensity distributions.
  # Lower EMD values indicate greater similarity between the images.
  return emd


def SpectralResidual(image1, image2):
  from scipy.fftpack import fft2, fftshift, ifft2

  # Define a helper function to compute the spectral residual of an image.
  def _SpectralResidual(image):
    # Convert the input image into a NumPy array for numerical processing.
    # This ensures compatibility with Fourier Transform operations.
    image = np.asarray(image)

    # Compute the Fourier Transform of the image using `fft2`.
    # The Fourier Transform decomposes the image into its frequency components.
    fftImg = fft2(image)

    # Compute the magnitude of the Fourier Transform.
    # The magnitude represents the strength of each frequency component.
    magnitude = np.abs(fftImg)

    # Compute the logarithm of the amplitude (magnitude) to emphasize smaller variations.
    # A small constant (1e-8) is added to avoid taking the log of zero.
    logAmplitude = np.log(magnitude + 1e-8)

    # Extract the phase information from the Fourier Transform.
    # The phase encodes spatial relationships between frequency components.
    phase = np.angle(fftImg)

    # Compute the average log amplitude across all frequency components.
    # This provides a baseline for the overall distribution of frequencies.
    avgLogAmplitude = np.mean(logAmplitude)

    # Compute the spectral residual by subtracting the average log amplitude from the log amplitude.
    # The spectral residual highlights deviations from the average frequency distribution.
    spectralResidual = logAmplitude - avgLogAmplitude

    # Reconstruct the saliency map by applying the inverse Fourier Transform.
    # The reconstructed image emphasizes regions of interest based on the spectral residual.
    saliencyMap = np.abs(ifft2(np.exp(spectralResidual + 1j * phase))) ** 2
    return saliencyMap

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Compute the spectral residuals for both images using the helper function `_SpectralResidual`.
  # The spectral residual highlights the unique features of each image.
  sr1 = _SpectralResidual(image1)
  sr2 = _SpectralResidual(image2)

  # Compute the Spectral Residual Similarity (SRS) using the correlation coefficient.
  # The correlation coefficient measures the linear relationship between the two saliency maps.
  srs = np.corrcoef(sr1.flatten(), sr2.flatten())[0, 1]

  # Return the computed Spectral Residual Similarity value.
  # The result quantifies the similarity between the two images based on their spectral residuals.
  return srs


def PhaseCongruency(image1, image2):
  from skimage.filters import prewitt

  # Define a helper function to compute the phase congruency of an image.
  def _PhaseCongruency(image):
    # Approximate phase congruency using the Prewitt filter.
    # Phase congruency measures the alignment of edges and corners in an image.
    pc = prewitt(np.array(image))
    return pc

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Compute the phase congruency maps for both images using the helper function `_PhaseCongruency`.
  # The phase congruency map highlights edges and structural features in the images.
  pc1 = _PhaseCongruency(image1)
  pc2 = _PhaseCongruency(image2)

  # Compute the similarity between the two phase congruency maps using the correlation coefficient.
  # The correlation coefficient measures the linear relationship between the two maps.
  pcSim = np.corrcoef(pc1.flatten(), pc2.flatten())[0, 1]

  # Return the computed phase congruency similarity value.
  # The result quantifies the similarity between the two images based on their structural features.
  return pcSim


def NoiseQualityMeasure(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Compute the noise as the difference between the two images.
  # The noise represents the discrepancies or errors introduced in the second image relative to the first.
  noise = image1 - image2

  # Compute the Noise Quality Measure (NQM) using the formula:
  # NQM = 20 * log10(MAX_IMAGE1 / STD_NOISE)
  # NQM evaluates the quality of the second image by comparing the maximum intensity of the first image to the standard deviation of the noise.
  nqm = 20.0 * np.log10(np.max(image1) / np.std(noise))

  # Return the computed Noise Quality Measure value.
  # The result quantifies the quality of the second image, with higher values indicating better quality.
  return nqm


def HellingerDistance(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Dynamically determine the range of pixel values.
  pixelRange = (min(image1.min(), image2.min()), max(image1.max(), image2.max()))

  # Calculate the histogram of the first flattened image using 256 bins and a range of [0, 256].
  # The histogram represents the distribution of pixel intensity values in the image.
  hist1, _ = np.histogram(img1Flat, bins=256, range=pixelRange)

  # Calculate the histogram of the second flattened image using the same parameters as the first image.
  # This ensures consistency in the comparison process.
  hist2, _ = np.histogram(img2Flat, bins=256, range=pixelRange)

  # Normalize the histogram of the first image by dividing each bin count by the total number of pixels.
  # Normalization ensures that the histograms are comparable regardless of the size of the images.
  hist1 = hist1 / np.sum(hist1)

  # Normalize the histogram of the second image using the same method as the first image.
  # This step ensures that both histograms are scaled to the range [0, 1].
  hist2 = hist2 / np.sum(hist2)

  # Calculate the Hellinger Distance between the two normalized histograms.
  # Hellinger Distance measures the similarity between two probability distributions.
  # It is computed as the square root of half the sum of squared differences between the square roots of the histograms.
  hellinger = np.sqrt(
    np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2)
  ) / np.sqrt(2)

  # Return the computed Hellinger Distance value.
  # The result quantifies the dissimilarity between the two images based on their histograms, with lower values indicating greater similarity.
  return hellinger


def BhattacharyyaDistance(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Dynamically determine the range of pixel values.
  pixelRange = (min(image1.min(), image2.min()), max(image1.max(), image2.max()))

  # Calculate the histogram of the first flattened image using 256 bins and a range of [0, 256].
  # The histogram represents the distribution of pixel intensity values in the image.
  hist1, _ = np.histogram(img1Flat, bins=256, range=pixelRange)

  # Calculate the histogram of the second flattened image using the same parameters as the first image.
  # This ensures consistency in the comparison process.
  hist2, _ = np.histogram(img2Flat, bins=256, range=pixelRange)

  # Normalize the histogram of the first image by dividing each bin count by the total number of pixels.
  # Normalization ensures that the histograms are comparable regardless of the size of the images.
  hist1 = hist1 / np.sum(hist1)

  # Normalize the histogram of the second image using the same method as the first image.
  # This step ensures that both histograms are scaled to the range [0, 1].
  hist2 = hist2 / np.sum(hist2)

  # Calculate the Bhattacharyya Coefficient (BC) between the two normalized histograms.
  # The BC measures the overlap between the two probability distributions.
  bc = np.sum(np.sqrt(hist1 * hist2))

  # Calculate the Bhattacharyya Distance using the formula: -log(BC).
  # The Bhattacharyya Distance quantifies the dissimilarity between the two histograms, with higher values indicating less overlap.
  bhattacharyya = -np.log(bc)

  # Return the computed Bhattacharyya Distance value.
  # The result quantifies the dissimilarity between the two images based on their histograms, with lower values indicating greater similarity.
  return bhattacharyya


def PerceptualHash(image1, image2):
  from imagehash import phash

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Calculate the Perceptual Hash (pHash) of the first image.
  # The `imagehash.phash` function computes a hash value based on the perceptual features of the image.
  hash1 = phash(PIL.Image.fromarray(image1))

  # Calculate the Perceptual Hash (pHash) of the second image.
  # This step ensures that both images have their perceptual hashes computed for comparison.
  hash2 = phash(PIL.Image.fromarray(image2))

  # Compute the Hamming Distance between the two perceptual hashes.
  # The Hamming Distance measures the number of differing bits between the two hash values, providing a measure of similarity.
  hammingDistance = hash1 - hash2

  # Return the computed Hamming Distance value.
  # The result quantifies the dissimilarity between the two images based on their perceptual hashes, with lower values indicating greater similarity.
  return hammingDistance


def JensenShannonDivergence(image1, image2):
  from scipy.spatial.distance import jensenshannon

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Calculate the Jensen-Shannon Divergence (JSD) between the two flattened images.
  # JSD measures the similarity between two probability distributions based on the Kullback-Leibler divergence.
  # The `jensenshannon` function from SciPy computes the square root of the JSD, ensuring the result is bounded between 0 and 1.
  jsd = jensenshannon(img1Flat, img2Flat)

  # Return the computed Jensen-Shannon Divergence value.
  # The result quantifies the dissimilarity between the two images based on their pixel distributions, with lower values indicating greater similarity.
  return jsd


def KLDivergence(image1, image2, eps=1e-10):
  from scipy.stats import entropy

  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Dynamically determine the range of pixel values.
  pixelRange = (min(image1.min(), image2.min()), max(image1.max(), image2.max()))

  # Compute histograms for the first flattened image using 256 bins and a range of [0, 256].
  # The `density=True` parameter normalizes the histogram so that it represents a probability distribution.
  hist1, _ = np.histogram(img1Flat, bins=256, range=pixelRange, density=True)

  # Compute histograms for the second flattened image using the same parameters as the first image.
  # This ensures consistency in the comparison process.
  hist2, _ = np.histogram(img2Flat, bins=256, range=pixelRange, density=True)

  # Add a small constant to both histograms to avoid division by zero during KL divergence computation.
  # This ensures numerical stability when working with very small probabilities.
  hist1 = hist1 + eps
  hist2 = hist2 + eps

  # Normalize the histograms to ensure they sum to 1.
  # This step ensures that the histograms represent valid probability distributions.
  hist1 = hist1 / float(np.sum(hist1))
  hist2 = hist2 / float(np.sum(hist2))

  # Compute the Kullback-Leibler (KL) Divergence between the two normalized histograms.
  # KL Divergence measures the difference between two probability distributions.
  # The `entropy` function from SciPy computes the KL divergence as entropy(hist1, hist2).
  kld = entropy(hist1, hist2)

  # Return the computed KL Divergence value.
  # The result quantifies the dissimilarity between the two images based on their pixel intensity distributions, with higher values indicating greater dissimilarity.
  return kld


def GradientMagnitudeSimilarityDeviation(image1, image2, eps=1e-10):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Compute the gradient magnitude for the first image using Sobel operators along both axes.
  # The Sobel operator detects edges by approximating the gradient of the image intensity.
  grad1 = np.sqrt(sobel(image1, axis=0) ** 2 + sobel(image1, axis=1) ** 2)

  # Compute the gradient magnitude for the second image using Sobel operators along both axes.
  # This step ensures that both images have their gradients computed for comparison.
  grad2 = np.sqrt(sobel(image2, axis=0) ** 2 + sobel(image2, axis=1) ** 2)

  # Compute the numerator of the Gradient Magnitude Similarity (GMS) formula.
  # The numerator emphasizes regions where the gradient magnitudes are similar.
  numerator = 2.0 * grad1 * grad2

  # Compute the denominator of the GMS formula, adding a small constant to avoid division by zero.
  # This ensures numerical stability during the division operation.
  denominator = grad1 ** 2 + grad2 ** 2 + eps

  # Compute the Gradient Magnitude Similarity (GMS) as the ratio of the numerator to the denominator.
  gms = numerator / denominator

  # Compute the Gradient Magnitude Similarity Deviation (GMSD) as the standard deviation of the GMS map.
  # GMSD quantifies the overall dissimilarity between the two images based on their gradient magnitudes.
  gmsd = np.std(gms)

  # Return the computed GMSD value.
  # The result quantifies the dissimilarity between the two images based on their structural gradients, with lower values indicating greater similarity.
  return gmsd


def SpectralAngleMapper(image1, image2):
  # Convert the first input image into a NumPy array.
  image1 = np.asarray(image1)

  # Convert the second input image into a NumPy array.
  image2 = np.asarray(image2)

  # Validate input dimensions.
  if (image1.shape != image2.shape):
    # Raise an error if the two images do not have the same dimensions.
    raise ValueError("Both images must have the same dimensions.")

  # Handle color images by converting them to grayscale if necessary.
  if (len(image1.shape) == 3):  # Check if the first image is color (3D array).
    # Convert the first image to grayscale using OpenCV's color conversion function.
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  if (len(image2.shape) == 3):  # Check if the second image is color (3D array).
    # Convert the second image to grayscale using the same method.
    image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # Convert the images into a NumPy array and flatten it into a 1D array.
  # Flattening ensures that the image is treated as a single list of pixel values,
  # which simplifies vector-based operations.
  # The `reshape(1, -1)` operation converts the flattened array into a 2D row vector with one row and multiple columns.
  # This reshaping is required for compatibility with the `cosine_similarity` function, which expects 2D inputs.
  img1Flat = image1.flatten().astype(np.float64)
  img2Flat = image2.flatten().astype(np.float64)

  # Compute the dot product of the two flattened images.
  # The dot product measures the alignment between the two vectors (images).
  dotProduct = np.dot(img1Flat, img2Flat)

  # Compute the L2 norm (magnitude) of the first flattened image.
  # The L2 norm represents the length of the vector in the high-dimensional space.
  norm1 = np.linalg.norm(img1Flat)

  # Compute the L2 norm (magnitude) of the second flattened image.
  # This ensures both vectors are normalized before computing the spectral angle.
  norm2 = np.linalg.norm(img2Flat)

  # Compute the Spectral Angle Mapper (SAM) as the arccosine of the normalized dot product.
  # SAM measures the angular difference between the two vectors, with smaller angles indicating greater similarity.
  sam = np.arccos(dotProduct / (norm1 * norm2))

  # Return the computed SAM value.
  # The result quantifies the dissimilarity between the two images based on their spectral characteristics, with lower values indicating greater similarity.
  return sam


def BRISQUE(image):
  from brisque import BRISQUE

  # Convert the input image into a NumPy array.
  # This ensures that the image is in a format suitable for processing by the BRISQUE algorithm.
  image = np.asarray(image)

  # Calculate the BRISQUE score using the `brisque.BRISQUE().score` method.
  # BRISQUE evaluates the perceptual quality of an image based on its natural scene statistics.
  # Lower scores indicate better image quality, while higher scores indicate poorer quality or distortion.
  score = BRISQUE().score(image)

  # Return the computed BRISQUE score.
  # The result quantifies the no-reference image quality assessment, making it useful for evaluating standalone images without a reference.
  return score


def SummaryTable(image1, image2):
  mi = MutualInformation(image1, image2, bins=150)  # Updated.
  nmi = NormalizedMutualInformation(image1, image2)  # Updated.
  ssimScore = StructuralSimilarity(image1, image2, winSize=25)  # Updated.
  ncc = NormalizedCrossCorrelation(image1, image2)  # Updated.
  histIntersect = HistogramComparison(image1, image2, bins=256, eps=1e-10)  # Updated.
  uqi = UniversalQualityIndex(image1, image2)  # Updated.
  cosSim = CosineSimilarityImages(image1, image2)  # Updated.
  srs = SpectralResidual(image1, image2)  # Updated.
  pc = PhaseCongruency(image1, image2)  # Updated.
  nqm = NoiseQualityMeasure(image1, image2)  # Updated.

  mse = MeanSquaredError(image1, image2)  # Updated.
  nmse = NormalizedMeanSquaredError(image1, image2)  # Updated.
  emd = EarthMoversDistance(image1, image2)  # Updated.
  hellinger = HellingerDistance(image1, image2)  # Updated.
  bhattacharyya = BhattacharyyaDistance(image1, image2)  # Updated.
  hammingDistance = PerceptualHash(image1, image2)  # Updated.
  jsd = JensenShannonDivergence(image1, image2)  # Updated.
  kld = KLDivergence(image1, image2, eps=1e-10)  # Updated.
  sam = SpectralAngleMapper(image1, image2)  # Updated.

  # psnr = PeakSignalToNoiseRatio(image1, image2, eps=1e-10)  # Updated X.
  # featureBasedSimilarity = FeatureBasedSimilarity(image1, image2)  # Updated X.
  # bri = BRISQUE(image1)  # Updated X.
  # gmsd = GradientMagnitudeSimilarityDeviation(image1, image2, eps=1e-10)  # Updated X.

  scores = {
    "MI (U)"     : mi,
    "NMI (U)"    : nmi,
    "SSIM (U)"   : ssimScore,
    "NCC (U)"    : ncc,
    "HistInt (U)": histIntersect,
    "UQI (U)"    : uqi,
    "CS (U)"     : cosSim,
    "SRS (U)"    : srs,
    "PC (U)"     : pc,
    "NQM (U)"    : nqm,

    "MSE (D)"    : mse,
    "NMSE (D)"   : nmse,
    "EMD (D)"    : emd,
    "HD (D)"     : hellinger,
    "BhD (D)"    : bhattacharyya,
    "pHash (D)"  : hammingDistance,
    "JSD (D)"    : jsd,
    "KLD (D)"    : kld,
    "SAM (D)"    : sam,

    # "PSNR (X)"   : psnr,
    # "FBS (X)"    : featureBasedSimilarity,
    # "BRISQUE (D)": bri,
    # "GMSD (D)"   : gmsd,
  }
  return scores


def IsSimilarityAccepted(image1, image2):
  scores = SummaryTable(image1, image2)
  mi = scores["MI (U)"]
  cosSim = scores["CS (U)"]
  pHash = scores["pHash (D)"]

  s = f"Mutual Information: {mi:.4f}\nCosine Similarity: {cosSim:.4f}\nPerceptual Hash: {pHash:.4f}"
  if (mi >= 0.35 and cosSim >= 0.75 and pHash <= 20):
    return True, s
  return False, s


if __name__ == "__main__":
  # Generate two random images for testing.
  image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
  image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

  # Call the SummaryTable function to compute various similarity and dissimilarity metrics.
  results = SummaryTable(image1, image2)
  # Print the results.
  for metric, score in results.items():
    print(f"{metric}: {score:.4f}")

  # Check if the images are similar based on the defined criteria.
  similar, message = IsSimilarityAccepted(image1, image2)
  print(f"Are the images similar? {similar}\n{message}")
