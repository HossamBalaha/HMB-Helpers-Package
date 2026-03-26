import os, matplotlib, traceback
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestPower
from statsmodels.stats.weightstats import zconfint
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import lilliefors
from skimage.measure import shannon_entropy, moments
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew, kurtosis, bootstrap, wilcoxon
from HMB.PlotsHelper import GetCmapColors, GetRandomCMAPalette
from HMB.Initializations import UpdateMatplotlibSettings

# Parametric Methods
# These assume an underlying probability distribution (typically normal) and often rely on parameters such as the mean and variance.
# Note: Geometric Mean, Harmonic Mean, Skewness, Kurtosis, and Autocorrelation are marked with an asterisk (*)
# because they are distribution-sensitive; while they can be computed without assuming normality,
# their inferential use often presumes parametric conditions.
# Non-Parametric Methods
# These make minimal or no assumptions about the underlying data distribution. They typically rely on ranks, order statistics, or resampling.
# (or) These do not assume a specific distribution and often rely on ranks or medians instead of means.
# Robust or Context-Dependent Methods
# These methods can be used in both parametric and non-parametric contexts depending on the data and analysis goals.
# They often provide robustness to outliers or distributional assumptions.
# (or) These either relax strict parametric assumptions or adapt based on data characteristics.
# Their classification depends on implementation and purpose.
# - Trimmed Mean (10%): Reduces influence of outliers; often treated as non-parametric or robust.
# - Winsorized Mean (10%): Similar to trimmed mean; robust against extremes.
# - Modified Z-Score Method: Uses median absolute deviation; more robust than classic Z-scores.
# - Effect Size: Can be parametric (e.g., Cohen’s d) or non-parametric (e.g., Cliff’s delta).
# - Statistical Power: Depends on the test being powered; not inherently parametric or non-parametric.
# - Anderson-Darling Test: Primarily used for normality (parametric), but applicable to other distributions.
# - DistributionFitTesting: General term; method determines category.
# - Outlier Handling Suggestion: Strategy varies by context and assumptions.
# Comparison Table (Assumptions, Parameters, Power, Data Type, and Sensitivity):
# |---------|--------------------|------------------------|
# | Feature | Parametric Methods | Non-Parametric Methods |
# |---------|--------------------|------------------------|
# | Assumptions | Assume specific distribution (often normal) | Minimal or no distributional assumptions |
# | Parameters | Rely on parameters like mean and variance | Often rely on ranks, medians, or order statistics |
# | Power | Generally more powerful if assumptions are met | Less powerful but more robust to assumption violations |
# | Power | Generally higher statistical power if assumptions hold | Lower power under normality, but more reliable when assumptions fail |
# | Data Type | Best for interval/ratio data with symmetry | Suitable for ordinal data or skewed distributions |
# | Sensitivity to Outliers | Sensitive to outliers and assumption violations | More robust to outliers and skewness |
# |---------|--------------------|------------------------|

# Define the list of statistical methods/techniques
STAT_METHODS = [
  "Mean", "Median", "Mode", "Standard Deviation (Sample)", "Standard Deviation (Population)",
  "Coefficient of Variation (CV)", "Variance (Sample)", "Variance (Population)", "Minimum",
  "Maximum", "Range", "Interquartile Range (IQR)", "Geometric Mean", "Harmonic Mean",
  "Trimmed Mean (10%)", "Winsorized Mean (10%)", "Percentiles", "Skewness", "Kurtosis (Fisher)",
  "Kurtosis (Pearson)", "Confidence Interval (Mean)", "Bootstrap CI (Mean)", "Bootstrap CI (Median)",
  "Prediction Interval", "One-Sample T-Test", "Effect Size", "Statistical Power",
  "Shapiro-Wilk Test", "D'Agostino's K^2 Test", "Jarque-Bera Test", "Anderson-Darling Test",
  "Kolmogorov-Smirnov Test", "DistributionFitTesting", "IQR Method", "Z-Score Method",
  "Modified Z-Score Method", "Outlier Handling Suggestion", "Log Transformation",
  "Yeo-Johnson Transformation", "Standardized Data", "Min-Max Scaled Data", "Autocorrelation",
  "Ljung-Box Test"
]

STATS_CATEGORIZATION = {
  "Mean"                           : "Parametric",
  "Median"                         : "Non-Parametric",
  "Mode"                           : "Non-Parametric",
  "Standard Deviation (Sample)"    : "Parametric",
  "Standard Deviation (Population)": "Parametric",
  "Coefficient of Variation (CV)"  : "Parametric",
  "Variance (Sample)"              : "Parametric",
  "Variance (Population)"          : "Parametric",
  "Minimum"                        : "Non-Parametric",
  "Maximum"                        : "Non-Parametric",
  "Range"                          : "Non-Parametric",
  "Interquartile Range (IQR)"      : "Non-Parametric",
  "Geometric Mean"                 : "Parametric*",
  "Harmonic Mean"                  : "Parametric*",
  "Trimmed Mean (10%)"             : "Robust (Often Non-Parametric)",
  "Winsorized Mean (10%)"          : "Robust (Often Non-Parametric)",
  "Percentiles"                    : "Non-Parametric",
  "Skewness"                       : "Parametric*",
  "Kurtosis (Fisher)"              : "Parametric*",
  "Kurtosis (Pearson)"             : "Parametric*",
  "Confidence Interval (Mean)"     : "Parametric",
  "Bootstrap CI (Mean)"            : "Non-Parametric",
  "Bootstrap CI (Median)"          : "Non-Parametric",
  "Prediction Interval"            : "Parametric",
  "One-Sample T-Test"              : "Parametric",
  "Effect Size"                    : "Both (Context-Dependent)",
  "Statistical Power"              : "Both (Context-Dependent)",
  "Shapiro-Wilk Test"              : "Parametric (Tests Normality)",
  "D'Agostino's K^2 Test"          : "Parametric (Tests Normality)",
  "Jarque-Bera Test"               : "Parametric (Tests Normality)",
  "Anderson-Darling Test"          : "Both (Can be used for any distribution, but often for normality)",
  "Kolmogorov-Smirnov Test"        : "Non-Parametric",
  "DistributionFitTesting"         : "Both (Context-Dependent)",
  "IQR Method"                     : "Non-Parametric",
  "Z-Score Method"                 : "Parametric",
  "Modified Z-Score Method"        : "Robust (Often Non-Parametric)",
  "Outlier Handling Suggestion"    : "Both (Context-Dependent)",
  "Log Transformation"             : "Parametric (Used to meet parametric assumptions)",
  "Yeo-Johnson Transformation"     : "Parametric (Used to meet parametric assumptions)",
  "Standardized Data"              : "Parametric",
  "Min-Max Scaled Data"            : "Non-Parametric",
  "Autocorrelation"                : "Parametric*",
  "Ljung-Box Test"                 : "Parametric"
}


class GeneralStatisticsHelper(object):
  r'''
  GeneralStatisticsHelper: Collection of general-purpose statistical utilities.

  This helper groups descriptive and inferential statistical routines commonly used
  in data analysis and scientific computing. The class provides convenience wrappers
  around NumPy, SciPy and statsmodels functionality and also implements a number
  of custom summary and dynamic aggregation helpers that operate along flexible axes.

  Key capabilities:
    - Central tendency: mean, median, mode, percentiles, quantiles.
    - Dispersion and shape: variance, standard deviation, IQR, skewness, kurtosis.
    - Frequency and distribution: histogram, empirical and cumulative distribution functions, relative frequency.
    - Inferential measures: chi-squared, one-way ANOVA F-value, bootstrap utilities, hypothesis helpers.
    - Signal/image helpers: area, centroid, moments-based utilities and entropy measures.
    - Dynamic helpers: vectorized "Dynamic" wrappers that apply numpy/scipy functions along configurable axes and optionally return aggregated means.

  Inputs and outputs:
    - Inputs are typically numpy arrays, Python lists or pandas Series/DataFrames where appropriate.
    - Outputs vary by method and include scalars (floats/ints), 1-D/2-D numpy arrays or tuples containing summary statistics.

  Notes:
    - Most methods accept an axis argument or provide a dynamic variant (e.g., MeanDynamic) to compute results along specific axes.
    - Methods are thin wrappers and aim to preserve the semantics and edge-case behavior of the underlying libraries (NumPy, SciPy, statsmodels).

  References:
    - SciPy statistics reference: https://docs.scipy.org/doc/scipy/reference/stats.html
    - NumPy documentation: https://numpy.org/doc/stable/
    - Statsmodels documentation: https://www.statsmodels.org/
  '''

  # Shape (N, ROWS, COLS, CH).
  # axis = None => all elements.
  # axis = 0 => alongside N.
  # axis = 1 => alongside ROWS.
  # axis = 2 => alongside COLS.
  # axis = 3 => alongside CH.

  def AffineCovariance(self, data, A, b, isCovariance=True):
    r'''
    Compute the covariance after an affine transform.

    This implements Var[A*X + b] = A * Var[X] * A.T when provided with a
    covariance matrix, or computes the covariance of `data` first when
    isCovariance=False.

    Parameters:
      data (numpy.ndarray or numpy.ndarray-like): Either a covariance matrix (2-D)
        or raw data with samples along axis 0 when isCovariance=False.
      A (numpy.ndarray): Linear transform matrix to apply.
      b (numpy.ndarray): Additive bias (ignored for covariance calculation).
      isCovariance (bool): If True `data` is treated as a covariance matrix;
        if False `data` is treated as raw samples and the covariance will be
        computed first.

    Returns:
      numpy.ndarray: Transformed covariance matrix.

    Notes:
      - The additive term `b` does not affect covariance but is kept for API
        symmetry with affine mean.
    '''

    # Var[D] = Var[D + a]
    # Var[b * D] = b^2 * Var[D]
    # Var[A * D + b] = A * Var[D] * A.T
    if (not isCovariance):
      data = self.CovarianceMatrix(data)
    result = A @ data @ A.T
    return result

  def AffineMean(self, data, A, b, isMean=True):
    r'''
    Compute the mean after an affine transform.

    Implements E[A*X + b] = A * E[X] + b. If isMean=False the method will
    compute the mean from raw samples first.

    Parameters:
      data (numpy.ndarray): Either a mean vector (1-D) or raw samples (2-D) if isMean=False.
      A (numpy.ndarray): Linear transform matrix.
      b (numpy.ndarray or scalar): Additive bias.
      isMean (bool): If True `data` is treated as a mean vector; if False the
        mean will be computed via RowsMean.

    Returns:
      numpy.ndarray: Transformed mean vector.
    '''

    # E[b * D] = b * E[D]
    # E[D + a] = E[D] + a
    # E[b * D + a] = b * E[D] + a
    if (not isMean):
      data = self.RowsMean(data)
    result = A @ data + b
    return result

  def Area(self, data):
    r'''
    Compute the area (zeroth moment) of an image or 2D array.

    Parameters:
      data (numpy.ndarray): 2D array representing an image or spatial distribution.

    Returns:
      float: Zeroth spatial moment (sum over pixels) which corresponds to area.

    References:
      - skimage.measure.moments
    '''

    M = moments(data)
    return M[0, 0]

  def Centroid(self, data):
    r'''
    Compute the centroid coordinates of a 2D image/array using spatial moments.

    Parameters:
      data (numpy.ndarray): 2D array representing an image or spatial distribution.

    Returns:
      tuple: (x_centroid, y_centroid) as floats.

    Notes:
      - Uses raw (not central) moments so typical centroid formula applies: (M10/M00, M01/M00).
    '''

    M = moments(data)
    return (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

  def ChiSquared(self, X, y, withCorrection=False):
    r'''
    Compute the Pearson chi-squared statistic from two categorical arrays.

    Parameters:
      X (array-like): Categorical observations for variable X.
      y (array-like): Categorical observations for variable Y.
      withCorrection (bool): If True apply Yates' continuity correction.

    Returns:
      float: Chi-squared statistic.

    Notes:
      - This function returns the test statistic only (no p-value).
    '''
    # Find the unique elements from the X and y.
    setX, setY = list(set(X)), list(set(y))

    # Create the observed frequency matrix.
    O = np.zeros((len(setX), len(setY)))

    # Add a very small value to avoid any zero divisions.
    O += np.finfo(float).eps

    # Generate the O table.
    for i in range(len(X)):
      O[setX.index(X[i]), setY.index(y[i])] += 1.0

    totalRows = np.sum(O, axis=1)  # Calculate RT.

    totalColumns = np.sum(O, axis=0)  # Calculate CT.

    N = np.sum(O)  # Calculate N.

    # Calculate the E table.
    E = np.array(
      [
        [el1 * el2 for el1 in totalColumns]
        for el2 in totalRows
      ]
    ) / float(N)

    # Calculate the chi-squared value with or without correction.
    if (withCorrection):
      chi2 = np.sum(np.square(np.abs(O - E) - 0.5) / E)
    else:
      chi2 = np.sum(np.square(O - E) / E)

    return chi2

  def ColumnsMean(self, data):
    r'''
    Compute the mean for each column of a 2D array.

    Parameters:
      data (numpy.ndarray): 2D array with shape (rows, cols).

    Returns:
      numpy.ndarray: 1-D array with mean computed along axis=1.
    '''

    # Calculate the mean of the data.
    assert len(data.shape) == 2
    result = self.Mean(data, axis=1)
    return result

  def Count(self, data):
    r'''
    Return the total number of elements in the input.

    Parameters:
      data (array-like): Input array or sequence.

    Returns:
      int: Number of elements.
    '''

    result = np.size(data)
    return result

  def CountDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Count non-masked elements using a dynamic wrapper.

    This delegates to Dynamic(np.ma.count, ...), allowing axis and aggregation control.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return the mean of the counts.
      perLastAxis (bool): If True count along the last axis.
      axis (int, optional): Axis to count along.
      keepdims (bool): Whether to keep dimension for reduction.

    Returns:
      int or numpy.ndarray: Count or aggregated count depending on flags.
    '''

    result = self.Dynamic(np.ma.count, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def CovarianceMatrix(self, data):
    r'''
    Compute the sample covariance matrix of row-wise observations.

    Parameters:
      data (numpy.ndarray): 2-D array with samples along axis 0 (shape: N x D).

    Returns:
      numpy.ndarray: Covariance matrix (D x D) computed as (X - mean).T @ (X - mean) / N.
    '''

    mue = self.RowsMean(data)
    temp = data - mue
    result = temp.T @ temp / data.shape[0]
    return result

  def CumulativeDistributionFunction(self, data, bins=10, range=None):
    r'''
    Compute a discrete cumulative distribution function by binning data.

    Parameters:
      data (array-like): Input samples.
      bins (int): Number of histogram bins.
      range (tuple or None): Range for histogram bins.

    Returns:
      numpy.ndarray: CDF values for the histogram bins.
    '''

    hist, binEdges = self.Histogram(data, bins=bins, range=range)
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    return cdf

  def CumulativeFrequency(self, data, bins=10, returnFreqOnly=True, returnMean=False):
    r'''
    Wrapper around scipy.stats.cumfreq to compute cumulative frequencies.

    Parameters:
      data (array-like): Input samples.
      bins (int): Number of bins to use for cumulative frequency.
      returnFreqOnly (bool): If True return only the frequency array.
      returnMean (bool): If True return the mean of returned arrays instead of arrays.

    Returns:
      array or tuple: Depending on flags returns frequency array or full cumfreq output.
    '''

    a, lowerLimit, binWidth, extraPoints = stats.cumfreq(data, numbins=bins)
    if (returnFreqOnly):
      if (returnMean):
        return np.mean(a)
      return a
    if (returnMean):
      return (np.mean(a), np.mean(lowerLimit), np.mean(binWidth), np.mean(extraPoints))
    return (a, lowerLimit, binWidth, extraPoints)

  def DescriptiveStatistics(self, data, returnMean=False):
    r'''
    Return a tuple of descriptive statistics using scipy.stats.describe.

    Parameters:
      data (array-like): Input data sample.
      returnMean (bool): If True aggregate vector results into means.

    Returns:
      tuple: (nobs, mean, variance, kurtosis) where elements may be scalars or arrays depending on input.
    '''

    result = stats.describe(data)
    if (returnMean):
      return (result.nobs, np.mean(result.mean), np.mean(result.variance), np.mean(result.kurtosis))
    return (result.nobs, result.mean, result.variance, result.kurtosis)

  def DispersionRatio(self, X):
    r'''
    Compute the dispersion ratio between arithmetic and geometric means.

    Parameters:
      X (numpy.ndarray): Input 2D array with samples along axis 0.

    Returns:
      numpy.ndarray: Ratio of arithmetic mean to geometric mean for each column.
    '''

    # The arithmetic mean (AM).
    am = np.mean(X, axis=0)

    # The geometric mean (GM).
    gm = np.power(np.prod(X, axis=0), (1.0 / float(X.shape[0])))
    gm += np.finfo(float).eps  # Avoid zero divisions.

    # Calculate the dispersion ratio.
    rm = am / gm

    return rm

  def Dynamic(self, func, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False, **kwargs):
    r'''
    Generic dynamic wrapper to apply a reduction function along configurable axes.

    Parameters:
      func (callable): Reduction function (e.g., np.mean, np.sum).
      data (array-like): Input array.
      returnMean (bool): If True return the mean of the reduced result.
      perLastAxis (bool): If True apply reduction along the last axis.
      axis (int or None): Axis to apply the reduction.
      keepdims (bool): Whether to keep reduced dimensions.
      **kwargs: Additional keyword args forwarded to func.

    Returns:
      scalar or numpy.ndarray: The reduced result (optionally averaged).
    '''

    if (perLastAxis):
      result = func(data, axis=data.shape[-1], keepdims=keepdims, **kwargs)
    elif (axis is not None):
      result = func(data, axis=axis, keepdims=keepdims, **kwargs)
    else:
      result = func(data, **kwargs)
    if (returnMean):
      result = np.mean(result)
    return result

  def EmpiricalCumulativeDistributionFunction(self, data, returnMean=False):
    r'''
    Compute the empirical cumulative distribution function (ECDF) values for data.

    Parameters:
      data (array-like): Input samples.
      returnMean (bool): If True return the mean of ECDF values.

    Returns:
      numpy.ndarray or float: ECDF values array or its mean.
    '''

    xs = np.sort(data)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    if (returnMean):
      return np.mean(ys)
    return ys

  def Entropy(self, data):
    r'''
    Compute Shannon entropy using scipy.stats.entropy averaged over provided distribution(s).

    Parameters:
      data (array-like): Input probabilities or counts.

    Returns:
      float: Mean entropy value.
    '''

    from scipy.stats import entropy
    scipyEntropy = np.mean(entropy(data))
    return scipyEntropy

  def HistEntropy(self, pdf):
    r'''
    Compute entropy from a discrete probability mass function (base 2).

    Parameters:
      pdf (array-like): Discrete probability distribution (must sum to 1).

    Returns:
      float: Entropy in bits.
    '''

    en = pdf * np.log2(pdf)
    en[np.isnan(en)] = 0
    en = -np.sum(en)
    return en

  def HistEnergy(self, pdf):
    r'''
    Compute the energy of a discrete distribution (sum of squared probabilities).

    Parameters:
      pdf (array-like): Probability distribution.

    Returns:
      float: Energy value (sum of squares).
    '''

    en = pdf * pdf
    en = np.sum(en)
    return en

  def FValueUsingOneWayANOVA(self, X, y):
    r'''
    Compute the one-way ANOVA F statistic for groups defined by `y`.

    Parameters:
      X (array-like): Numeric observations (1-D or 2-D flattened to 1-D).
      y (array-like): Group labels for each observation.

    Returns:
      float: F-statistic computed from between- and within-group variances.
    '''

    classes = list(set(y))
    X, y = np.array(X), np.array(y)

    # Grouping the data by class.
    xG = np.array([X[y == c] for c in classes], dtype=object)

    # Calculate the mean for each group.
    xMean = np.array([el.mean() for el in xG])

    # Calculate the variance and sum of squares between the samples.
    grandAvg = np.mean(X)
    C = [
      ((xMean[i] - grandAvg) ** 2) * len(xG[i])
      for i in range(len(classes))
    ]
    SSC = np.sum(C)

    # Calculate the variance and sum of squares within the samples.
    S = [
      (xG[i] - xMean[i]) ** 2
      for i in range(len(classes))
    ]
    SSE = np.sum(
      [np.sum(S[i]) for i in range(len(classes))]
    )

    # Calculate the degrees of freedom.
    df1 = len(classes) - 1
    df2 = X.shape[0] - len(classes)

    # Calculate the mean sum of squares.
    MSC = SSC / df1
    MSE = SSE / df2

    # Calculate the F-value.
    F = MSC / MSE
    return F

  def Histogram(self, data, bins=10, range=None, returnMean=False):
    r'''
    Compute a histogram using numpy and optionally return averaged results.

    Parameters:
      data (array-like): Input data.
      bins (int): Number of bins.
      range (tuple): Range for histogram.
      returnMean (bool): If True return means of histogram counts and bin edges.

    Returns:
      tuple: (hist, binEdges) or their means if returnMean=True.
    '''

    # DeprecationWarning: scipy.histogram is deprecated. Use numpy.histogram instead.
    hist, binEdges = np.histogram(data, bins=bins, range=range)
    if (returnMean):
      return (np.mean(hist), np.mean(binEdges))
    return (hist, binEdges)

  def InterquartileRange(self, X, axis=None):
    r'''
    Compute the interquartile range (IQR) along the specified axis.

    Parameters:
      X (array-like): Input data.
      axis (int or None): Axis along which to compute percentiles.

    Returns:
      numpy.ndarray or scalar: IQR value(s).
    '''

    # Calculate the interquartile range.
    # q75, q25 = np.percentile(data, [75, 25])
    q75, q25 = np.percentile(X, [75, 25], axis=axis)
    iqr = q75 - q25
    return iqr

  def KurtosisDynamic(self, data, ddof=0, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Compute kurtosis using a dynamic (axis-flexible) implementation.

    Parameters:
      data (array-like): Input samples.
      ddof (int): Delta degrees of freedom for variance.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Kurtosis values.
    '''

    # bias=False => The calculations are corrected for statistical bias.
    # fisher=False => Pearson’s definition is used.
    # kurtosis(el, fisher=False, bias=False)
    mue = self.MeanDynamic(data, returnMean, perLastAxis, axis, keepdims)
    N = self.CountDynamic(data, returnMean, perLastAxis, axis, keepdims)
    diff = data - mue
    std = self.StandardDeviationDynamic(data, ddof, returnMean, perLastAxis, axis, keepdims)
    num = self.SumDynamic(np.power(diff, 4), returnMean, perLastAxis, axis, keepdims)
    den = N * np.power(std, 4)
    result = num / den
    if (returnMean):
      result = np.mean(result)
    return result

  def Max(self, data):
    r'''
    Return the maximum value in the input.

    Parameters:
      data (array-like): Input data.

    Returns:
      scalar: Maximum value.
    '''

    result = np.max(data)
    return result

  def MaxDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for np.max with axis control.

    Parameters and returns mirror Dynamic's contract.
    '''

    result = self.Dynamic(np.max, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def Mean(self, data, axis=None):
    r'''
    Compute the arithmetic mean along the specified axis.

    Parameters:
      data (array-like): Input array.
      axis (int or None): Axis to compute the mean over.

    Returns:
      numpy.ndarray or scalar: Mean value(s).
    '''

    # Calculate the mean of the data.
    return np.mean(data, axis=axis)

  def HistMean(self, pdf, range):
    r'''
    Compute the mean of a discrete histogram (pdf weighted by bin centers).

    Parameters:
      pdf (array-like): Probability per bin.
      range (array-like): Bin center locations or range values.

    Returns:
      float: Histogram mean.
    '''

    # Calculate the mean of the data.
    mean = np.sum(pdf * range)
    return mean

  def MeanAbsoluteDifference(self, X, axis=0):
    r'''
    Compute the mean absolute deviation from the mean along the given axis.

    Parameters:
      X (array-like): Input data.
      axis (int): Axis along which to compute MAD.

    Returns:
      numpy.ndarray or scalar: Mean absolute deviation.
    '''

    # Absolute / Mean Deviation.
    # Calculate the mean absolute difference between the target and the predicted values.
    absDiff = np.abs(X - X.mean(axis=axis))
    mad = np.mean(absDiff, axis=axis)
    return mad

  def MeanAbsoluteDifferenceDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic variant of mean absolute difference.

    Mirrors MeanDynamic signature.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: MAD values.
    '''

    mean = self.MeanDynamic(data, returnMean, perLastAxis, axis, keepdims)
    mad = self.MeanDynamic(np.abs(data - mean), returnMean, perLastAxis, axis, keepdims)
    return mad

  def RobustMeanAbsoluteDifference(self, X, axis=0, keepdims=False):
    r'''
    Compute median-based mean absolute deviation (robust MAD).

    Parameters:
      X (array-like): Input array.
      axis (int): Axis along which to compute the measure.
      keepdims (bool): Keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Robust MAD.
    '''

    # Absolute / Mean Deviation.
    # Calculate the mean absolute difference between
    # the target and the predicted values.
    absDiff = np.abs(X - np.median(X, axis=axis, keepdims=keepdims))
    mad = np.mean(absDiff, axis=axis)
    return mad

  def RobustMeanAbsoluteDifferenceDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for the robust MAD function.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Robust MAD values.
    '''

    result = self.Dynamic(self.RobustMeanAbsoluteDifference, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def MeanDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for numpy mean with axis control.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Mean values.
    '''

    result = self.Dynamic(np.mean, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def Median(self, X, axis=None):
    r'''
    Compute the median along an axis.

    Parameters:
      X (array-like): Input data.
      axis (int or None): Axis along which to compute the median.

    Returns:
      scalar or array: Median value(s).
    '''

    # Equal to np.median(data)
    # result = np.median(hypsecant.median(data))
    # Calculate the median of the data.
    median = np.median(X, axis=axis)
    return median

  def MedianAbsoluteDeviation(self, data):
    r'''
    Compute the median absolute deviation (MAD) for 1-D data.

    Parameters:
      data (array-like): Input 1-D array.

    Returns:
      float: Median absolute deviation.
    '''

    median = self.Median(data)
    mad = self.Median(np.abs(data - median))
    return mad

  def MedianAbsoluteDeviationDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for MAD.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: MAD values.
    '''

    median = self.MedianDynamic(data, returnMean, perLastAxis, axis, keepdims)
    mad = self.MedianDynamic(np.abs(data - median), returnMean, perLastAxis, axis, keepdims)
    return mad

  def MedianDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for numpy.median.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Median values.
    '''

    result = self.Dynamic(np.median, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def RootMeanSquare(self, data, axis=None, keepdims=False):
    r'''
    Compute the root mean square along an axis.

    Parameters:
      data (array-like): Input data.
      axis (int or None): Axis over which to compute RMS.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: RMS values.
    '''

    # Calculate the root mean square of the data.
    rms = np.sqrt(np.mean(np.square(data), axis=axis, keepdims=keepdims))
    return rms

  def RootMeanSquareDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for RMS computation.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: RMS values.
    '''

    result = self.Dynamic(self.RootMeanSquare, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def Min(self, data):
    r'''
    Return the minimum value from the input.

    Parameters:
      data (array-like): Input data.

    Returns:
      scalar: Minimum value.
    '''

    result = np.min(data)
    return result

  def MinDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for np.min.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Minimum values.
    '''

    result = self.Dynamic(np.min, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def Mode(self, X):
    r'''
    Compute the mode of integer-valued data using bincount.

    Parameters:
      X (array-like): 1-D integer-valued data.

    Returns:
      int: The mode (most frequent value).
    '''

    # Calculate the mode of the data.
    mode = np.argmax(np.bincount(X))
    return mode

  def Percentile(self, X, p, axis=0):
    r'''
    Compute the p-th percentile of data.

    Parameters:
      X (array-like): Input data.
      p (float): Percentile to compute (0-100).
      axis (int): Axis along which to compute percentile.

    Returns:
      scalar or array: Percentile value(s).
    '''

    # Calculate the percentile of the data.
    pr = np.percentile(X, p, axis=axis)
    return pr

  def Percentiles(self, data, ranges=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    r'''
    Convenience to compute multiple percentiles at once.

    Parameters:
      data (array-like): Input data.
      ranges (list): List of percentiles to compute.

    Returns:
      numpy.ndarray: Percentile values corresponding to `ranges`.
    '''

    result = np.nanpercentile(data, ranges)
    return result

  def Quantile(self, X, q, axis=0):
    r'''
    Compute quantiles of the data.

    Parameters:
      X (array-like): Input data.
      q (float or array-like): Quantile(s) in [0, 1].
      axis (int): Axis to compute quantile along.

    Returns:
      scalar or array: Quantile values.
    '''

    # Calculate the quantile of the data.
    qr = np.quantile(X, q, axis=axis)
    return qr

  def Range(self, X, axis=0):
    r'''
    Compute range (max - min) along an axis.

    Parameters:
      X (array-like): Input data.
      axis (int): Axis along which to compute range.

    Returns:
      numpy.ndarray or scalar: Range value(s).
    '''

    # Calculate the range of the data.
    rng = np.max(X, axis=axis) - np.min(X, axis=axis)
    return rng

  def RelativeFrequency(self, data, bins=10, returnFreqOnly=True, returnMean=False):
    r'''
    Compute relative frequency using scipy.stats.relfreq.

    Parameters:
      data (array-like): Input data.
      bins (int): Number of bins.
      returnFreqOnly (bool): If True return only frequency array.
      returnMean (bool): If True return the mean of results.

    Returns:
      array or tuple: Relative frequency outputs depending on flags.
    '''

    a, lowerLimit, binWidth, extraPoints = stats.relfreq(data, numbins=bins)
    if (returnFreqOnly):
      if (returnMean):
        return np.mean(a)
      return a
    if (returnMean):
      return (np.mean(a), np.mean(lowerLimit), np.mean(binWidth), np.mean(extraPoints))
    return (a, lowerLimit, binWidth, extraPoints)

  def RowsMean(self, data):
    r'''
    Compute the mean per row for a 2D array (asserts 2D input).

    Parameters:
      data (numpy.ndarray): 2D input array.

    Returns:
      numpy.ndarray: Mean per row.
    '''

    # Calculate the mean of the data.
    assert len(data.shape) == 2
    result = self.Mean(data, axis=0)
    return result

  def SciPyFisherKurtosis(self, data, returnMean=False):
    r'''
    Compute Fisher (excess) kurtosis via scipy.stats.kurtosis.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of kurtosis results.

    Returns:
      float or numpy.ndarray: Kurtosis values.
    '''

    result = stats.kurtosis(data, fisher=True, bias=True)
    if (returnMean):
      result = np.mean(result)
    return result

  def SciPyFisherKurtosisDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper around scipy kurtosis (Fisher definition).

    Parameters:
      data (array-like): Input samples.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Kurtosis values.
    '''

    result = self.Dynamic(stats.kurtosis, data, returnMean, perLastAxis, axis, keepdims, fisher=True, bias=True)
    return result

  def SciPyPearsonKurtosisDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper around scipy kurtosis (Pearson definition).

    Parameters:
      data (array-like): Input samples.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Kurtosis values.
    '''

    result = self.Dynamic(stats.kurtosis, data, returnMean, perLastAxis, axis, keepdims, fisher=False, bias=True)
    return result

  def SciPyPearsonKurtosis(self, data, returnMean=False):
    r'''
    Compute Pearson kurtosis via scipy.stats.kurtosis.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of kurtosis results.

    Returns:
      float or numpy.ndarray: Kurtosis values.
    '''

    result = stats.kurtosis(data, fisher=False, bias=True)
    if (returnMean):
      result = np.mean(result)
    return result

  def HistPearsonKurtosis(self, pdf, range):
    r'''
    Compute kurtosis from a histogram PDF and bin locations.

    Parameters:
      pdf (array-like): Probability mass per bin.
      range (array-like): Bin centers.

    Returns:
      float: Pearson kurtosis estimate from histogram.
    '''

    mean = self.HistMean(pdf, range)
    std = self.HistVariance(pdf, range) ** 0.5
    kurtosis = np.sum((range - mean) ** 4 * pdf) / (std ** 4)
    return kurtosis

  def SciPySkewnessDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper around scipy.stats.skew.

    Parameters:
      data (array-like): Input samples.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Skewness values.
    '''

    result = self.Dynamic(stats.skew, data, returnMean, perLastAxis, axis, keepdims, bias=True)
    return result

  def SciPySkewness(self, data, returnMean=False):
    r'''
    Compute skewness using scipy.stats.skew.

    Parameters:
      data (array-like): Input data.
      returnMean (bool): If True return mean of skewness results.

    Returns:
      float or numpy.ndarray: Skewness values.
    '''

    result = stats.skew(data, bias=True)
    if (returnMean):
      result = np.mean(result)
    return result

  def HistSkewness(self, pdf, range):
    r'''
    Estimate skewness from a histogram PDF.

    Parameters:
      pdf (array-like): Probability mass per bin.
      range (array-like): Bin center locations.

    Returns:
      float: Skewness of the histogram distribution.
    '''

    mean = self.HistMean(pdf, range)
    std = self.HistVariance(pdf, range) ** 0.5
    skew = np.sum((range - mean) ** 3 * pdf) / (std ** 3)
    return skew

  def ShannonEntropy(self, data):
    r'''
    Compute Shannon entropy using skimage.measure.shannon_entropy averaged over inputs.

    Parameters:
      data (array-like): Input image or distribution to compute entropy for.

    Returns:
      float: Mean Shannon entropy.
    '''

    shannonEntropy = np.mean(shannon_entropy(data))
    return shannonEntropy

  def SkewnessDynamic(self, data, ddof=0, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Compute skewness using moment-based formula with flexible axes.

    Parameters:
      data (array-like): Input samples.
      ddof (int): Delta degrees of freedom for variance.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Skewness values.
    '''

    # bias=False => The calculations are corrected for statistical bias.
    # skew(el, bias=False)
    mue = self.MeanDynamic(data, returnMean, perLastAxis, axis, keepdims)
    N = self.CountDynamic(data, returnMean, perLastAxis, axis, keepdims)
    diff = data - mue
    std = self.StandardDeviationDynamic(data, ddof, returnMean, perLastAxis, axis, keepdims)
    num = self.SumDynamic(np.power(diff, 3), returnMean, perLastAxis, axis, keepdims)
    den = N * np.power(std, 3)
    result = num / den
    if (returnMean):
      result = np.mean(result)
    return result

  def StandardDeviation(self, X, axis=0, ddof=0):
    r'''
    Compute standard deviation.

    Parameters:
      X (array-like): Input data.
      axis (int): Axis to compute along.
      ddof (int): Degrees of freedom correction.

    Returns:
      numpy.ndarray or scalar: Standard deviation.
    '''

    # Calculate the standard deviation of the data.
    return np.std(X, axis=axis, ddof=ddof)

  def StandardDeviationDynamic(self, data, ddof=0, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper computing standard deviation via VarianceDynamic.
    '''
    result = np.sqrt(self.VarianceDynamic(data, ddof, returnMean, perLastAxis, axis, keepdims))
    return result

  def Sum(self, data):
    r'''
    Compute the total sum of elements.

    Parameters:
      data (array-like): Input data.

    Returns:
      scalar: Sum of all elements.
    '''

    result = np.sum(data)
    return result

  def SumDynamic(self, data, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic wrapper for np.sum.
    '''
    result = self.Dynamic(np.sum, data, returnMean, perLastAxis, axis, keepdims)
    return result

  def TValueUsingTwoGroups(self, X, y):
    r'''
    Compute an approximate independent two-sample t-value between two groups.

    Parameters:
      X (array-like): First sample array.
      y (array-like): Second sample array.

    Returns:
      float: t-statistic.
    '''

    # Calculate the N for each group.
    nX = float(len(X))
    nY = float(len(y))

    # Calculate the mean for each group.
    meanX = np.mean(X)
    meanY = np.mean(y)

    # Calculate the std. for each group.
    # ddof=1 is used to correct for the sample size.
    # ddof=0 is used to correct for the population size.
    stdX = np.std(X, ddof=1)
    stdY = np.std(y, ddof=1)

    num = np.abs(meanX - meanY)
    den = np.sqrt((stdX ** 2 / nX) + (stdY ** 2 / nY))
    tValue = num / den

    return tValue

  def Variance(self, X, axis=0, ddof=0):
    r'''
    Compute variance along an axis.

    Parameters:
      X (array-like): Input data.
      axis (int): Axis along which to compute.
      ddof (int): Degrees of freedom correction.

    Returns:
      numpy.ndarray or scalar: Variance value(s).
    '''

    # Calculate the variance of the data.
    var = np.var(X, axis=axis, ddof=ddof)
    return var

  def HistVariance(self, pdf, range):
    r'''
    Compute variance for a histogram-defined distribution.

    Parameters:
      pdf (array-like): Probability mass per bin.
      range (array-like): Bin center locations.

    Returns:
      float: Variance of the histogram distribution.
    '''

    # Calculate the variance of the data.
    mean = self.HistMean(pdf, range)
    var = np.sum(np.power(range - mean, 2) * pdf)
    return var

  def VarianceDynamic(self, data, ddof=0, returnMean=False, perLastAxis=False, axis=None, keepdims=False):
    r'''
    Dynamic variance computation supporting flexible axes.

    Parameters:
      data (array-like): Input samples.
      ddof (int): Delta degrees of freedom for variance.
      returnMean (bool): If True return mean of the result.
      perLastAxis (bool): If True operate along the last axis.
      axis (int): Axis to operate over.
      keepdims (bool): Whether to keep reduced dimensions.

    Returns:
      numpy.ndarray or scalar: Variance values.
    '''

    # Var[D] = Var[D + a]
    # Var[b * D] = b^2 * Var[D]
    # Var[A * D + b] = A * Var[D] * A.T
    mue = self.MeanDynamic(data, returnMean=returnMean, perLastAxis=perLastAxis, axis=axis, keepdims=keepdims)
    diff = data - mue
    diffPow = np.power(diff, 2)
    num = self.SumDynamic(diffPow, returnMean=returnMean, perLastAxis=perLastAxis, axis=axis, keepdims=keepdims)
    N = self.CountDynamic(data, returnMean=returnMean, perLastAxis=perLastAxis, axis=axis, keepdims=keepdims)
    result = num / (N - ddof)
    if (returnMean):
      result = np.mean(result)
    return result

  def ZValueUsingTwoGroups(self, X, y, value=0.0):
    r'''
    Compute Z-statistic for difference of two group means (assuming known population std).

    Parameters:
      X (array-like): First sample.
      y (array-like): Second sample.
      value (float): Hypothesized difference (default 0.0).

    Returns:
      float: z-statistic.
    '''

    # Calculate the N for each group.
    nX = float(len(X))
    nY = float(len(y))

    # Calculate the mean for each group.
    meanX = np.mean(X)
    meanY = np.mean(y)

    # Calculate the std. for each group.
    # ddof=1 is used to correct for the sample size.
    # ddof=0 is used to correct for the population size.
    stdX = np.std(X, ddof=0)
    stdY = np.std(y, ddof=0)

    num = np.abs(meanX - meanY) - value
    den = np.sqrt((stdX ** 2 / nX) + (stdY ** 2 / nY))
    zValue = num / den

    return zValue

  def ConfidenceInterval(self, data, confidence=0.95):
    r'''
    Compute a Student-t based confidence interval for the mean.

    Parameters:
      data (array-like): Sample observations.
      confidence (float): Confidence level (0-1).

    Returns:
      tuple: (lower bound, upper bound)
        - float: Lower bound of the confidence interval.
        - float: Upper bound of the confidence interval.
    '''

    n = len(data)
    m = np.mean(data)
    stdErr = stats.sem(data)
    h = stdErr * stats.t.ppf((1 + confidence) / 2, n - 1)
    start = m - h
    end = m + h
    return (start, end)


class StatisticalAnalysisFramework(object):
  r'''
  A modular validator for repeated-trial system evaluation.
  Automatically selects, runs, and explains applicability of statistical methods.
  The framework is designed to be extensible, allowing for new methods and tests to be added as needed.

  Mermaid diagram of the validation pipeline:
  flowchart TD
      A["Start: Input Data<br>(baseline, threshold, etc.)"] --> B{Infer Data Structure}
      B -->|1D Array| C1["Set isMultiConfig = False<br>sampleSize = len(data)"]
      B -->|" 2D Array<br>(subjects × configs) "| C2["Set isMultiConfig = True<br>nConfigs = cols<br>sampleSize = rows"]
      C1 --> D{Infer Data Type}
      C2 --> D
      D -->|Binary| E1["Set dataType = Binary"]
      D -->|Continuous| E2["Set dataType = Continuous"]
      E1 --> F["Skip Normality/Symmetry Checks"]
      E2 --> G{Check Assumptions}
      G --> H1["Shapiro-Wilk (n ≤ 5000)?"]
      H1 -->|Yes| I1["Run Shapiro-Wilk<br>Set isNormal"]
      H1 -->|No| I2["Skip Shapiro-Wilk"]
      G --> H2["D'Agostino (n > 20)?"]
      H2 -->|Yes| I3["Run D'Agostino<br>Update isNormal if n>5000"]
      H2 -->|No| I4["Skip D'Agostino"]
      G --> H3["Lilliefors (n > 5000)?"]
      H3 -->|Yes| I5["Run Lilliefors<br>Set isNormal"]
      H3 -->|No| I6["Skip Lilliefors"]
      G --> H4["Baseline Provided?"]
      H4 -->|Yes| I7["Compute Skewness<br>Set isSymmetric"]
      H4 -->|No| I8["Assume Symmetric"]
      I1 --> J
      I2 --> J
      I3 --> J
      I4 --> J
      I5 --> J
      I6 --> J
      I7 --> J
      I8 --> J
      F --> J["Assess Method Applicability"]
      J --> K1{Has Threshold?}
      K1 -->|Yes| L1["Evaluate One-Sample Tests:<br>t-test, Wilcoxon, Sign, Binomial"]
      K1 -->|No| L2["Skip One-Sample Tests"]
      J --> K2{Has Baseline?}
      K2 -->|Yes| L3["Evaluate Paired Tests:<br>Paired t, Wilcoxon, Permutation,<br>McNemar, TOST"]
      K2 -->|No| L4["Skip Paired Tests"]
      J --> K3{isMultiConfig?}
      K3 -->|Yes| L5["Evaluate Multi-Config Tests:<br>- ICC (≥2)<br>- Friedman / RM-ANOVA (≥3)<br>- Cochran's Q (Binary, ≥3)"]
      K3 -->|No| L6["Skip Multi-Config Tests"]
      J --> K4{Has predictions & labels?}
      K4 -->|Yes| L7["Evaluate Calibration Tests:<br>CalibrationECE, HosmerLemeshow"]
      K4 -->|No| L8["Skip Calibration Tests"]
      J --> K5{Data Type?}
      K5 -->|Binary| L9["Evaluate Binomial Methods:<br>Clopper-Pearson, Beta-Binomial"]
      K5 -->|Continuous| L10["Evaluate CV, MAD, Effect Sizes"]
      L1 --> M
      L2 --> M
      L3 --> M
      L4 --> M
      L5 --> M
      L6 --> M
      L7 --> M
      L8 --> M
      L9 --> M
      L10 --> M
      M["Generate Method Assessments:<br>Applied/Not Applied + Reasons"] --> N["Execute Applicable Tests"]
  %% One-Sample Branch
      N --> O1{One-Sample Tests?}
      O1 -->|Yes| P1["Run: t-test / Wilcoxon / Sign / Binomial"]
      O1 -->|No| P2["Skip One-Sample Tests"]
  %% Paired Branch
      N --> O2{Paired Tests?}
      O2 -->|Yes| P3["Run: Paired t / Wilcoxon / Permutation /<br>McNemar / TOST"]
      O2 -->|No| P4["Skip Paired Tests"]
  %% Multi-Config Branch
      N --> O3{Multi-Config Tests?}
      O3 -->|Yes| P5["Run: ICC / Friedman / RM-ANOVA /<br>Cochran's Q"]
      O3 -->|No| P6["Skip Multi-Config Tests"]
  %% Calibration Branch
      N --> O4{Calibration Tests?}
      O4 -->|Yes| P7["Run: CalibrationECE / HosmerLemeshow"]
      O4 -->|No| P8["Skip Calibration Tests"]
  %% Descriptive Stats
      N --> O5{Descriptive Stats?}
      O5 -->|Yes| P9["Compute: CV, MAD"]
      O5 -->|No| P10["Skip Descriptive Stats"]
  %% Effect Size
      N --> O6{Effect Size?}
      O6 -->|Yes| P11["Compute: Cohen's d / Hedges' g / A12"]
      O6 -->|No| P12["Skip Effect Size"]
  %% Bootstrap
      N --> O7{Bootstrap?}
      O7 -->|Yes| P13["Compute BCa CIs:<br>Mean/Median/Proportion/5th %ile"]
      O7 -->|No| P14["Skip Bootstrap (should not occur)"]
  %% Tolerance Interval
      N --> O8{Tolerance Interval?}
      O8 -->|Yes| P15["Compute Bootstrap Approximation"]
      O8 -->|No| P16["Skip Tolerance Interval"]
  %% Extra Normality Tests
      N --> O9{Normality Tests?}
      O9 -->|Yes| P17["Run: Anderson-Darling / Jarque-Bera"]
      O9 -->|No| P18["Skip Extra Normality Tests"]
  %% Converge all paths
      P1 --> Q
      P2 --> Q
      P3 --> Q
      P4 --> Q
      P5 --> Q
      P6 --> Q
      P7 --> Q
      P8 --> Q
      P9 --> Q
      P10 --> Q
      P11 --> Q
      P12 --> Q
      P13 --> Q
      P14 --> Q
      P15 --> Q
      P16 --> Q
      P17 --> Q
      P18 --> Q
      Q["Compile Results + Execution Log"] --> R["Return Results Dictionary"]
      R --> S["End"]
  '''

  # Initialize the validator with data and optional references.
  def __init__(self, data, baseline=None, threshold=None, equivalenceMargin=None, alpha=0.05):
    r'''
    Initialize the `StatisticalAnalysisFramework` with input data and parameters.

    Parameters:
      data (array-like): The primary dataset to analyze (e.g., performance metrics).
      baseline (array-like, optional): A reference dataset for comparison (e.g., human performance).
      threshold (float, optional): A performance threshold for certain tests (e.g., above chance).
      equivalenceMargin (float, optional): Margin for equivalence testing.
      alpha (float, optional): Significance level for hypothesis testing (default 0.05).

    This constructor sets up the internal state of the validator, including storing the input data,
    baseline, threshold, equivalence margin, and significance level. It also prepares structures for results
    and logging decisions throughout the validation process.
    '''

    # Store the input data as a numpy array.
    self.data = np.asarray(data)
    # Store the baseline as a numpy array when provided.
    self.baseline = np.asarray(baseline) if (baseline is not None) else None
    # Store the threshold value when provided.
    self.threshold = threshold
    # Store the equivalence margin when provided.
    self.equivalenceMargin = equivalenceMargin
    # Store the significance level alpha.
    self.alpha = alpha
    # Prepare a results dictionary to collect outputs.
    self.results = {}
    # Prepare a log list to collect human-readable decisions.
    self.log = []

  # Run the overall validation pipeline and return results.
  def RunValidation(self):
    r'''
    Execute the full validation process, including data type inference, assumption checks,
    method applicability evaluation, method assessment, test execution, and result compilation.

    Returns:
      dict: A comprehensive results dictionary containing test outcomes, execution log, and method assessments.
    '''

    # Infer the data type from the sample values.
    self.InferDataType()
    # Check statistical assumptions such as normality and symmetry.
    self.CheckAssumptions()
    # Evaluate which methods are applicable based on assumptions.
    self.EvaluateMethodApplicability()
    # Assess methods to build an applied/skipped report with reasons.
    self.AssessMethods()
    # Select and run the applicable tests and collect results.
    self.SelectAndRunTests()
    # Attach the execution log to results under a CamelCase key.
    self.results["ExecutionLog"] = self.log
    # Attach the method assessments mapping to results under a CamelCase key.
    self.results["MethodAssessments"] = self.methodAssessments
    # Return the assembled the results' dictionary.
    return self.results

  # Infer if the data is binary or continuous based on unique values.
  def InferDataType(self):
    r'''
    Infer the data type (Binary vs Continuous) based on unique values in the input data.
    '''

    # Compute unique values present in the data.
    uniqueVals = np.unique(self.data)
    # Detect whether the input is multi-configuration (2D: subjects x configurations/raters).
    self.isMultiConfig = (hasattr(self.data, "ndim") and (self.data.ndim == 2) and (self.data.shape[1] > 1))
    if (self.isMultiConfig):
      # When multi-config, number of configurations equals number of columns.
      self.nConfigs = int(self.data.shape[1])
      # Number of subjects / repeated trials equals number of rows.
      self.sampleSize = int(self.data.shape[0])
      self.log.append(
        f"Multi-configuration data detected with {self.sampleSize} observations and {self.nConfigs} configurations.")
      # For multi-config inputs we leave per-configuration normality to specific tests; default conservative flags.
      # Continue to flatten unique value detection on the full matrix to infer binary/continuous.
      uniqueVals = np.unique(self.data)

    # Determine if data is binary by checking for two unique values 0 and 1.
    if ((len(uniqueVals) == 2) and (set(uniqueVals).issubset({0, 1}))):
      # Set dataType to Binary when values are exactly 0 and 1.
      self.dataType = "Binary"
      # Log the inference decision.
      self.log.append("`dataType` inferred as Binary because data contains only 0 and 1.")
    else:
      # Set dataType to Continuous for all other cases.
      self.dataType = "Continuous"
      # Log the inference decision for continuous data.
      self.log.append(
        "`dataType` inferred as Continuous because data has more "
        "than two unique values or non-binary values."
      )

  # Check assumptions like normality and symmetry for continuous data.
  def CheckAssumptions(self):
    r'''
    Check statistical assumptions such as normality and symmetry based on data type and sample size.
    For Continuous data:
      - Normality is assessed using Shapiro-Wilk for n <= 5000, D'Agostino's K^2 for n > 20, and Lilliefors for n > 5000.
      - Symmetry is assessed via skewness of differences from baseline when a baseline is provided (paired design).
      - Logs are generated for each test applied or skipped with reasons.
    For Binary data:
      - Normality is not applicable and set to False.
      - Symmetry is treated as True by default.
      - Logs are generated indicating that normality and symmetry checks were skipped for binary data.
    '''

    # Record the sample size n.
    n = len(self.data)
    # Store the sample size as an attribute.
    self.sampleSize = n

    # Only perform normality and symmetry checks when data is Continuous.
    if (self.dataType == "Continuous"):
      # Create a dictionary to hold normality test p-values.
      self.normalityPValues = {}

      # Use Shapiro-Wilk test for moderate sample sizes (n <= 5000).
      if (n <= 5000):
        # Perform Shapiro-Wilk normality test and capture the p-value.
        _, pShapiro = stats.shapiro(self.data)
        # Record the Shapiro-Wilk p-value.
        self.normalityPValues["ShapiroWilk"] = pShapiro
        # Determine normality flag based on p-value and alpha.
        self.isNormal = (pShapiro > self.alpha)
        # Log the application of Shapiro-Wilk test with p-value.
        self.log.append(f"ShapiroWilk test applied because sample size ({n}) <= 5000; p-value = {pShapiro:.4f}.")
      else:
        # Log that Shapiro-Wilk is skipped for large sample sizes.
        self.log.append(f"ShapiroWilk skipped because sample size ({n}) > 5000.")

      # Use D'Agostino's K^2 test when sample size is greater than 20.
      if (n > 20):
        # Perform the D'Agostino K^2 normality test and capture the p-value.
        _, pDagostino = stats.normaltest(self.data)
        # Record the D'Agostino K2 p-value.
        self.normalityPValues["DAgostinoK2"] = pDagostino
        # When sample is very large use D'Agostino to set isNormal.
        if (n > 5000):
          # Set normality flag based on D'Agostino p-value.
          self.isNormal = (pDagostino > self.alpha)
        # Log the application of D'Agostino test with p-value.
        self.log.append(f"DAgostinoK2 test applied because sample size ({n}) > 20; p-value = {pDagostino:.4f}.")
      else:
        # Log that D'Agostino is skipped for small sample sizes.
        self.log.append(f"DAgostinoK2 skipped because sample size ({n}) <= 20.")

      # Use Lilliefors test for very large sample sizes (n > 5000).
      if (n > 5000):
        # Perform the Lilliefors test and capture the p-value.
        _, pLilliefors = lilliefors(self.data)
        # Record the Lilliefors p-value.
        self.normalityPValues["Lilliefors"] = pLilliefors
        # Set the normality flag based on Lilliefors p-value.
        self.isNormal = (pLilliefors > self.alpha)
        # Log application of Lilliefors test with p-value.
        self.log.append(f"Lilliefors test applied because sample size ({n}) > 5000; p-value = {pLilliefors:.4f}.")
      else:
        # Log that Lilliefors is skipped for moderate sample sizes.
        self.log.append(f"Lilliefors skipped because sample size ({n}) <= 5000.")

      # Only assess symmetry when a baseline is provided for paired designs.
      if (self.baseline is not None):
        # Compute differences between data and baseline.
        diffs = self.data - self.baseline
        # Compute skewness of differences as a symmetry proxy.
        skewness = stats.skew(diffs)
        # Consider symmetric when absolute skewness is below 0.5.
        self.isSymmetric = (abs(skewness) < 0.5)
        # Log the symmetry assessment result with skewness.
        self.log.append(
          f"Symmetry assessed via skewness = {skewness:.3f}; "
          f"{'assumed symmetric' if self.isSymmetric else 'assumed asymmetric'}."
        )
      else:
        # Default to symmetric when no baseline is available (one-sample context).
        self.isSymmetric = True
        # Log that symmetry was not assessed due to missing baseline.
        self.log.append("Symmetry not assessed because no baseline provided (one-sample context).")
    else:
      # For binary data, normality is not applicable.
      self.isNormal = False
      # For binary data, treat symmetry as True by default.
      self.isSymmetric = True
      # Log that normality and symmetry checks were skipped for binary data.
      self.log.append("Normality and symmetry not assessed because data is Binary.")

  # Evaluate which statistical methods are applicable to this dataset.
  def EvaluateMethodApplicability(self):
    r'''
    Evaluate the applicability of various statistical methods based on the inferred data type, sample size, and assumption checks.
      - For Continuous data, applicability of normality tests, one-sample tests (t-test, Wilcoxon, Sign), paired
          tests (Paired t-test, Wilcoxon signed-rank, Permutation), and effect size calculations (Cohen's d, Hedges' g)
          are determined based on normality, symmetry, presence of baseline, and sample size.
      - For Binary data, applicability of methods like Exact Binomial Test, Clopper-Pearson CI, Bayesian Beta-Binomial,
          McNemar's test, and Cochran's Q is determined based on the presence of a threshold, baseline, and whether
          the data is multi-config.
      - Logs are generated for each method indicating whether it is applicable or not, along with the reasons based
          on the data characteristics and assumptions.
      - The results of this evaluation are stored in an `applicableMethods` dictionary mapping method names to
          boolean flags indicating applicability.
    '''

    # Initialize an applicability dictionary to record decisions.
    self.applicableMethods = {}
    # Local alias of sample size for readability.
    n = self.sampleSize

    # Mark ShapiroWilk as applicable for Continuous when n <= 5000.
    self.applicableMethods["ShapiroWilk"] = (self.dataType == "Continuous") and (n <= 5000)
    # AndersonDarling applicable for Continuous when tail-sensitive normality is relevant.
    self.applicableMethods["AndersonDarling"] = (self.dataType == "Continuous")
    # DAgostinoK2 applicable for Continuous when n > 20.
    self.applicableMethods["DAgostinoK2"] = (self.dataType == "Continuous") and (n > 20)
    # Lillfors applicable for Continuous when n > 5000.
    self.applicableMethods["Lillfors"] = (self.dataType == "Continuous") and (n > 5000)
    # Jarque-Bera applicable for Continuous typically in large-sample contexts.
    self.applicableMethods["JarqueBera"] = (self.dataType == "Continuous") and (n >= 30)
    # TOST equivalence test applicable when baseline and equivalence margin provided.
    self.applicableMethods["TOST"] = (self.baseline is not None) and (self.dataType == "Continuous") and (
        self.equivalenceMargin is not None)

    # Determine if one-sample threshold based methods should be considered.
    hasThreshold = (self.threshold is not None)
    # OneSampleTTest applicable for Continuous when threshold provided and either normal or large n.
    self.applicableMethods["OneSampleTTest"] = (
        hasThreshold and (self.dataType == "Continuous") and
        ((self.isNormal) or (n >= 30))
    )
    # WilcoxonSignedRankOneSample applicable when non-normal but symmetric.
    self.applicableMethods["WilcoxonSignedRankOneSample"] = (
        hasThreshold and (self.dataType == "Continuous") and
        (not ((self.isNormal) or (n >= 30))) and (self.isSymmetric)
    )
    # SignTest applicable when non-normal and not symmetric.
    self.applicableMethods["SignTest"] = (
        hasThreshold and (self.dataType == "Continuous") and
        (not ((self.isNormal) or (n >= 30))) and (not (self.isSymmetric))
    )
    # ExactBinomialTest applicable for binary one-sample tests when threshold provided.
    self.applicableMethods["ExactBinomialTest"] = hasThreshold and (self.dataType == "Binary")
    # ClopperPearsonCI always applicable for binary data.
    self.applicableMethods["ClopperPearsonCI"] = (self.dataType == "Binary")
    # BayesianBetaBinomial always applicable for binary data.
    self.applicableMethods["BayesianBetaBinomial"] = (self.dataType == "Binary")

    # Determine applicability of paired tests when a baseline is present.
    hasBaseline = (self.baseline is not None)
    # PairedTTest applicable when baseline present, data is continuous and normal.
    self.applicableMethods["PairedTTest"] = hasBaseline and (self.dataType == "Continuous") and (self.isNormal)
    # PairedFTest applicable for variance comparisons in paired continuous data when normal.
    self.applicableMethods["PairedFTest"] = hasBaseline and (self.dataType == "Continuous") and (self.isNormal)
    # WilcoxonSignedRankPaired applicable when non-normal but symmetric.
    self.applicableMethods["WilcoxonSignedRankPaired"] = (
        hasBaseline and (self.dataType == "Continuous") and
        (not (self.isNormal)) and (self.isSymmetric)
    )
    # PermutationTest applicable when non-normal and asymmetric.
    self.applicableMethods["PermutationTest"] = (
        hasBaseline and (self.dataType == "Continuous")
        and (not (self.isNormal)) and (not (self.isSymmetric))
    )
    # McNemarTest applicable for binary paired comparisons.
    self.applicableMethods["McNemarTest"] = hasBaseline and (self.dataType == "Binary")
    # Cochran's Q applicable for binary multi-config repeated measures when k >= 3.
    self.applicableMethods["CochransQ"] = self.isMultiConfig and (self.nConfigs >= 3) and (self.dataType == "Binary")

    # Coefficient of Variation applicable for continuous data.
    self.applicableMethods["CV"] = (self.dataType == "Continuous")
    # Median Absolute Deviation applicable for continuous data.
    self.applicableMethods["MAD"] = (self.dataType == "Continuous")

    # ICC applicable when data is multi-config (subjects x raters/configs) with >=2 raters.
    self.applicableMethods["ICC"] = self.isMultiConfig and (self.nConfigs >= 2)
    # Friedman and RMANOVA: applicable for multi-config continuous or ordinal repeated measures with >=3 configurations.
    self.applicableMethods["Friedman"] = self.isMultiConfig and (self.nConfigs >= 3) and (self.dataType == "Continuous")
    self.applicableMethods["RMANOVA"] = self.isMultiConfig and (self.nConfigs >= 3) and (self.dataType == "Continuous")

    # Effect size applicability depending on normality and baseline.
    self.applicableMethods["CohensD"] = (hasBaseline and (self.dataType == "Continuous") and (self.isNormal))
    self.applicableMethods["HedgesG"] = (hasBaseline and (self.dataType == "Continuous") and (self.isNormal))
    self.applicableMethods["VarghaDelaneyA12"] = (
        hasBaseline and (self.dataType == "Continuous") and
        (not (self.isNormal))
    )
    # Determine applicability for tolerance interval estimation using bootstrap for continuous data.
    self.applicableMethods["ToleranceInterval"] = (self.dataType == "Continuous") and (n >= 2)

    # Calibration/ECE and Hosmer-Lemeshow require `predictions` (probabilities) and `labels` (binary true labels) to be set on the instance.
    hasPreds = (hasattr(self, "predictions") and (getattr(self, "predictions") is not None)) and (
        hasattr(self, "labels") and (getattr(self, "labels") is not None))
    self.applicableMethods["CalibrationECE"] = hasPreds
    self.applicableMethods["HosmerLemeshow"] = hasPreds

    # Bootstrap is always applicable as a fallback.
    self.applicableMethods["BCaBootstrap"] = True

    # Multi-group tests allowed when multi-config repeated-measures are present.
    # (Do not forcibly disable here; AssessMethods/GetSkipReason will explain when not applicable.)

    # Log the applicability decisions for traceability.
    for method, applicable in self.applicableMethods.items():
      # Log applicable methods as APPLICABLE.
      if (applicable):
        self.log.append(f"{method} marked as APPLICABLE based on data properties.")
      else:
        # Compute and log a readable reason for skipping a method.
        reason = self.GetSkipReason(method)
        self.log.append(f"{method} SKIPPED because {reason}.")

  # Provide a brief human-readable reason why an applicable method is chosen.
  def GetApplyReason(self, methodName):
    r'''
    Get a human-readable reason for why a method is applied based on the data characteristics and assumptions.

    Parameters:
      methodName (str): The name of the method for which to generate the apply reason.

    Returns:
      str: A human-readable explanation for why the method is applied.
    '''

    # Normality checks: ShapiroWilk when sample size is moderate.
    if (methodName == "ShapiroWilk"):
      return f"Applied because data is Continuous and sample size ({self.sampleSize}) <= 5000."
    # Anderson-Darling for tail-sensitive normality checks.
    if (methodName == "AndersonDarling"):
      return f"Applied because data is Continuous and Anderson–Darling provides tail-sensitive normality diagnostics."
    # Normality checks: DAgostinoK2 when n > 20.
    if (methodName == "DAgostinoK2"):
      return f"Applied because data is Continuous and sample size ({self.sampleSize}) > 20."
    # Normality checks: Lillfors for very large n.
    if (methodName == "Lillfors"):
      return f"Applied because data is Continuous and sample size ({self.sampleSize}) > 5000."
    # Jarque-Bera for large-sample asymptotic normality testing.
    if (methodName == "JarqueBera"):
      return "Applied because data is Continuous and sample size is large; Jarque–Bera tests skewness and kurtosis."
    # One-sample or paired t-tests applied when normality or large n holds.
    if (methodName in ["OneSampleTTest", "PairedTTest"]):
      return "Applied because the data meets approximate normality or large-sample conditions."
    if (methodName in ["PairedFTest"]):
      return "Applied to compare variances of paired differences under normality assumption."
    # Wilcoxon applied when data is non-normal but symmetric.
    if (methodName in ["WilcoxonSignedRankOneSample", "WilcoxonSignedRankPaired"]):
      return "Applied because data is non-normal but symmetry of differences is assumed."
    # Sign test applied when non-normal and asymmetric.
    if (methodName == "SignTest"):
      return "Applied because data is non-normal and asymmetric, so sign test is distribution-free."
    # Exact binomial for binary one-sample.
    if (methodName == "ExactBinomialTest"):
      return "Applied because data is Binary and a threshold proportion was specified."
    # Clopper-Pearson for binary CI.
    if (methodName == "ClopperPearsonCI"):
      return "Applied because data is Binary; Clopper–Pearson provides exact interval coverage."
    # BayesianBetaBinomial for binary posterior insights.
    if (methodName == "BayesianBetaBinomial"):
      return "Applied to produce a posterior distribution for the Bernoulli rate with a conjugate beta prior."
    # Permutation test applied when non-parametric paired testing is required.
    if (methodName == "PermutationTest"):
      return "Applied because paired differences are non-normal and asymmetric; permutation provides an assumption-free test."
    # McNemar applied for paired binary comparisons.
    if (methodName == "McNemarTest"):
      return "Applied because data is Binary and baseline is provided for paired comparisons."
    # TOST equivalence when equivalence margin is provided.
    if (methodName == "TOST"):
      return "Applied because equivalence margin was provided and paired comparisons are possible."
    # Effect sizes and variability metrics.
    if (methodName in ["CohensD", "HedgesG", "VarghaDelaneyA12", "CV", "MAD"]):
      return "Applied to quantify effect magnitude or robust variability as appropriate."
    # Bootstrap always applied as fallback.
    if (methodName == "BCaBootstrap"):
      return "Applied as a distribution-free method to estimate confidence intervals for key statistics."
    # Tolerance interval for coverage guarantees using bootstrap approximation.
    if (methodName == "ToleranceInterval"):
      return "Applied to estimate a distribution-free tolerance interval (bootstrap approximation for quantiles)."
    if (methodName == "Friedman"):
      return "Applied because multi-configuration repeated-measures data with >=3 configurations was provided."
    if (methodName == "RMANOVA"):
      return "Applied because multi-configuration repeated-measures data with >=3 configurations was provided; attempts repeated-measures ANOVA using statsmodels if available."
    if (methodName == "CochransQ"):
      return "Applied because binary multi-configuration repeated-measures data with >=3 configurations was provided."
    if (methodName == "ICC"):
      return "Applied to estimate intraclass correlation from multi-configuration (subjects x raters) data."
    if (methodName == "CalibrationECE"):
      return "Applied because predicted probabilities and true labels were provided; computes expected calibration error (ECE)."
    if (methodName == "HosmerLemeshow"):
      return "Applied because predicted probabilities and true labels were provided; computes Hosmer-Lemeshow goodness-of-fit statistic."
    # Default reason when nothing specific is defined.
    return "Applied because applicability rules were satisfied for this test."

  # Assess methods and produce an applied/skipped mapping with reasons.
  def AssessMethods(self):
    r'''
    Assess each method's applicability and produce a mapping of method names to whether they were applied or
      skipped, along with human-readable reasons for each decision.
      - Iterates over the `applicableMethods` dictionary to determine which methods are applied or skipped.
      - For applied methods, it uses `GetApplyReason` to generate a reason for application and records this in
          the `methodAssessments` dictionary.
      - For skipped methods, it uses `GetSkipReason` to generate a reason for skipping and records this in
          the `methodAssessments` dictionary.
      - Logs the decision for each method in the execution log for transparency and traceability of the
          validation process.
      - The resulting `methodAssessments` dictionary maps each method name to a dictionary containing an
          "Applied" boolean and a "Reason" string explaining the decision.
      - This structured assessment allows for clear communication of which methods were used in the analysis and
          why, as well as which methods were not used and the rationale for their exclusion.
    '''

    # Initialize the assessments' dictionary.
    self.methodAssessments = {}
    # Iterate over all methods declared in applicableMethods.
    for method, applicable in self.applicableMethods.items():
      # When a method is applicable, mark it as Applied and provide a reason.
      if (applicable):
        # Get a human-readable apply reason for the method.
        reason = self.GetApplyReason(method)
        # Record the assessment entry for applied method.
        self.methodAssessments[method] = {"Applied": True, "Reason": reason}
        # Log the application decision.
        self.log.append(f"{method} APPLIED: {reason}")
      else:
        # For skipped methods, get the skip reason.
        reason = self.GetSkipReason(method)
        # Record the assessment entry for skipped method.
        self.methodAssessments[method] = {"Applied": False, "Reason": reason}
        # Log the skip decision.
        self.log.append(f"{method} SKIPPED: {reason}")

  # Provide human-readable reasons for skipping specific methods.
  def GetSkipReason(self, methodName):
    r'''
    Get a human-readable reason for why a method is skipped based on the data characteristics and assumptions.

    Parameters:
      methodName (str): The name of the method for which to generate the skip reason.

    Returns:
      str: A human-readable explanation for why the method is skipped.
    '''

    # Return reason for multi-configuration methods.
    if (methodName in ["Friedman", "RMANOVA", "CochransQ"]):
      return "requires comparison of three or more configurations; only one system (or one vs baseline) provided"
    # Return reason for ICC.
    if (methodName in ["ICC"]):
      return "requires multiple raters or repeated measurements per subject; not applicable to single-system trial metrics"
    # Return reason for calibration tests.
    if (methodName in ["CalibrationECE", "HosmerLemeshow"]):
      return "requires predicted probabilities, not just binary outcomes or scalar metrics"
    # Return reason for one-sample methods lacking threshold.
    if (("OneSample" in methodName) and (self.threshold is None)):
      return "no performance threshold specified"
    # Return reason for methods that require a baseline when none provided.
    if ((("Paired" in methodName) or (methodName in ["TOST", "McNemarTest"])) and (self.baseline is None)):
      return "no baseline system provided for comparison"
    # Return reason when data type mismatches method expectation.
    if ((self.dataType == "Binary") and ("Continuous" in methodName)):
      return "data is binary, not continuous"
    if (
        (self.dataType == "Continuous") and
        (methodName in ["ExactBinomialTest", "ClopperPearsonCI", "BayesianBetaBinomial"])
    ):
      return "data is continuous, not binary"
    # Return reason for coefficient of variation when data contains non-positive values.
    if ((methodName == "CV") and (np.any(self.data <= 0))):
      return "data contains non-positive values; CV undefined"
    # Default fallback reason for skipping.
    return "assumptions not met or redundant given other tests"

  # Select the applicable tests and execute them to populate results.
  def SelectAndRunTests(self):
    r'''
    Select and execute the applicable statistical tests based on the `applicableMethods` mapping, and populate the results dictionary with test outcomes.
      - Iterates over the `applicableMethods` to identify which tests are marked as applicable.
      - For each applicable test, the corresponding statistical test is executed using SciPy or custom implementations as needed.
      - The results of each test (e.g., test statistics, p-values, confidence intervals) are stored in the `results` dictionary under keys corresponding to each test.
      - For one-sample tests, if a threshold is provided, the relevant tests are executed and their results are stored under a "OneSampleTests" key in the results.
      - For multi-configuration tests, the appropriate methods are executed and results are stored under keys corresponding to each method.
      - For effect size calculations and variability metrics, the computed values are stored under descriptive keys in the results.
      - The method assessments are included in the results for transparency, allowing users to see which methods were applied and the reasons for their application or exclusion.
    '''

    # Prepare an output dictionary for test results.
    out = {}

    # Include method assessments in the output for transparency.
    out["MethodAssessments"] = self.methodAssessments

    # Compute descriptive statistics when CV is applicable.
    if (self.applicableMethods.get("CV", False)):
      # Only compute CV when all data values are positive.
      if (np.all(self.data > 0)):
        # Compute coefficient of variation using sample standard deviation.
        cv = np.std(self.data, ddof=1) / np.mean(self.data)
        # Store the CV in the output dictionary under a CamelCase key.
        out["CoefficientOfVariation"] = cv

    # Compute MAD when applicable.
    if (self.applicableMethods.get("MAD", False)):
      # Compute median absolute deviation from the median.
      mad = np.median(np.abs(self.data - np.median(self.data)))
      # Store MAD under a CamelCase key.
      out["MedianAbsoluteDeviation"] = mad

    # Run Anderson-Darling normality test when applicable.
    if (self.applicableMethods.get("AndersonDarling", False)):
      # Use SciPy's Anderson test for the normal distribution.
      adRes = stats.anderson(self.data, dist="norm")
      # Store statistic and critical values for interpretation.
      out["AndersonDarling"] = {
        "Statistic"         : float(adRes.statistic),
        "CriticalValues"    : list(adRes.critical_values),
        "SignificanceLevels": list(adRes.significance_level)
      }

    # Run Jarque-Bera test when applicable.
    if (self.applicableMethods.get("JarqueBera", False)):
      # Use SciPy's jarque_bera for skewness/kurtosis-based normality.
      jbStat, jbP = stats.jarque_bera(self.data)
      # Record Jarque-Bera statistic and p-value.
      out["JarqueBera"] = {"Statistic": float(jbStat), "pValue": float(jbP)}

    # Run one-sample tests when a threshold was provided.
    if (self.threshold is not None):
      # Prepare a dictionary to collect one-sample test results.
      oneSample = {}
      # Run OneSampleTTest when applicable.
      if (self.applicableMethods.get("OneSampleTTest", False)):
        # Perform one-sample t-test against the threshold.
        tStat, pVal = stats.ttest_1samp(self.data, self.threshold)
        # Record t-test results under a CamelCase key.
        oneSample["OneSampleTTest"] = {"tStatistic": tStat, "pValue": pVal}
      # Run Wilcoxon signed-rank one-sample when applicable.
      if (self.applicableMethods.get("WilcoxonSignedRankOneSample", False)):
        # Perform Wilcoxon signed-rank test for differences to threshold.
        wStat, pVal = stats.wilcoxon(self.data - self.threshold)
        # Record Wilcoxon results under a CamelCase key.
        oneSample["WilcoxonSignedRank"] = {"wStatistic": wStat, "pValue": pVal}
      # Run sign test when applicable.
      if (self.applicableMethods.get("SignTest", False)):
        # Compute signs of deviations from threshold.
        signs = np.sign(self.data - self.threshold)
        # Count the number of non-zero deviations.
        nNonZero = np.sum(signs != 0)
        # Compute binomial p-value for the sign distribution when non-zero observations exist.
        if (nNonZero > 0):
          # Count the number of positive signs.
          k = np.sum(signs > 0)
          # Compute two-sided binomial p-value.
          pVal = 2 * min(stats.binom.cdf(k, nNonZero, 0.5), 1 - stats.binom.cdf(k - 1, nNonZero, 0.5))
        else:
          # Default p-value when all differences are zero.
          pVal = 1.0
        # Record the sign test p-value.
        oneSample["SignTest"] = {"pValue": pVal}
      # Run exact binomial test when applicable for binary one-sample.
      if (self.applicableMethods.get("ExactBinomialTest", False)):
        # Count the number of successes in binary data.
        successes = np.sum(self.data)
        # Determine the number of trials.
        nTrials = len(self.data)
        # Compute an exact binomial test p-value against the threshold proportion using a compatibility wrapper.
        pVal = self._safe_binom_test(successes, nTrials, self.threshold, alternative="two-sided")
        # Record the exact binomial test result under a CamelCase key.
        oneSample["ExactBinomialTest"] = {"pValue": pVal}
      # Attach one-sample results to output when any tests were run.
      if (oneSample):
        out["OneSampleTests"] = oneSample

    # Compute Clopper-Pearson confidence interval for binary data when applicable.
    if (self.applicableMethods.get("ClopperPearsonCI", False)):
      # Count successes and trials for binary proportion.
      successes = np.sum(self.data)
      # Determine the number of trials.
      nTrials = len(self.data)
      # Compute Clopper-Pearson interval using the beta distribution quantiles with edge-case handling.
      if (successes == 0):
        ciLow = 0.0
        ciHigh = float(stats.beta.ppf(1 - self.alpha / 2, successes + 1, nTrials - successes))
      elif (successes == nTrials):
        ciLow = float(stats.beta.ppf(self.alpha / 2, successes, nTrials - successes + 1))
        ciHigh = 1.0
      else:
        ciLow = float(stats.beta.ppf(self.alpha / 2, successes, nTrials - successes + 1))
        ciHigh = float(stats.beta.ppf(1 - self.alpha / 2, successes + 1, nTrials - successes))
      out["ClopperPearsonCI"] = {"Lower": ciLow, "Upper": ciHigh}

    # Run Bayesian Beta-Binomial posterior calculations for binary data when applicable.
    if (self.applicableMethods.get("BayesianBetaBinomial", False)):
      # Count successes and trials for the binary posterior.
      successes = np.sum(self.data)
      # Determine the number of trials.
      nTrials = len(self.data)
      # Compute posterior alpha and beta parameters with a uniform prior (1,1).
      alphaPost = successes + 1.0
      betaPost = nTrials - successes + 1.0
      # Compute posterior mean of the binomial probability.
      postMean = alphaPost / (alphaPost + betaPost)
      # Compute posterior probability the rate exceeds threshold when threshold provided or use 0.5 default.
      pGreater = 1 - stats.beta.cdf(self.threshold if (self.threshold is not None) else 0.5, alphaPost, betaPost)
      # Store Bayesian results under a CamelCase key.
      out["BayesianBetaBinomial"] = {
        "PosteriorMean"         : postMean,
        "P_GreaterThanThreshold": pGreater
      }

    # Run paired tests when a baseline is provided and applicable.
    if (self.baseline is not None):
      # Prepare a dictionary to collect paired test results.
      paired = {}
      # Run PairedTTest when assumptions permit.
      if (self.applicableMethods.get("PairedTTest", False)):
        # Perform paired t-test between data and baseline.
        tStat, pVal = stats.ttest_rel(self.data, self.baseline)
        # Record paired t-test results.
        paired["PairedTTest"] = {"tStatistic": tStat, "pValue": pVal}
      if (self.applicableMethods.get("PairedFTest", False)):
        # Perform paired F-test for variance comparison (Levene's test for paired samples).
        fStat, pVal = stats.levene(self.data, self.baseline)
        # Record paired F-test results.
        paired["PairedFTest"] = {"fStatistic": fStat, "pValue": pVal}
      # Run Wilcoxon signed-rank for paired samples when applicable.
      if (self.applicableMethods.get("WilcoxonSignedRankPaired", False)):
        # Perform Wilcoxon signed-rank paired test.
        wStat, pVal = stats.wilcoxon(self.data, self.baseline)
        # Record Wilcoxon paired results.
        paired["WilcoxonSignedRank"] = {"wStatistic": wStat, "pValue": pVal}
      # Run a simple permutation test for paired differences when applicable.
      if (self.applicableMethods.get("PermutationTest", False)):
        # Compute differences for paired observations.
        diffs = self.data - self.baseline
        # Compute observed mean difference.
        obsMean = np.mean(diffs)
        # Set number of permutations for the permutation test.
        nPerm = 5000
        # Prepare a list to collect permutation means.
        permMeans = []
        # Run permutation sampling with random sign flips.
        for _ in range(nPerm):
          # Randomly assign signs to differences.
          signs = np.random.choice([-1, 1], size=len(diffs))
          # Append the mean of signed differences to the permutation distribution.
          permMeans.append(np.mean(signs * diffs))
        # Compute two-sided permutation p-value as the proportion exceeding the observed mean.
        pVal = (np.abs(permMeans) >= np.abs(obsMean)).mean()
        # Record permutation test results with the permutation count.
        paired["PermutationTest"] = {"pValue": pVal, "nPermutations": nPerm}
      # Run McNemar test for binary paired comparisons when applicable.
      if (self.applicableMethods.get("McNemarTest", False)):
        # Count observations where both new and baseline are correct.
        bothCorrect = int(np.sum((self.data == 1) & (self.baseline == 1)))
        # Count observations where new is correct only.
        newOnly = int(np.sum((self.data == 1) & (self.baseline == 0)))
        # Count observations where baseline is correct only.
        baseOnly = int(np.sum((self.data == 0) & (self.baseline == 1)))
        # Count neither correct (both zero)
        neither = int(np.sum((self.data == 0) & (self.baseline == 0)))
        discordant = newOnly + baseOnly
        # Use exact binomial test for small discordant counts.
        if (discordant > 0) and (discordant < 25):
          pVal = self._safe_binom_test(min(newOnly, baseOnly), discordant, 0.5, alternative="two-sided")
        elif (discordant == 0):
          pVal = 1.0
        else:
          # Use continuity-corrected McNemar chi-squared approximation for larger discordant counts.
          stat = (abs(newOnly - baseOnly) - 1) ** 2 / (discordant + 1e-12)
          pVal = 1 - stats.chi2.cdf(stat, df=1)
        paired["McNemarTest"] = {"pValue": pVal, "DiscordantPairs": int(discordant)}
      # Run TOST equivalence testing when applicable.
      if (self.applicableMethods.get("TOST", False)):
        # Implement TOST using t-distribution one-sided p-values computed from the sample differences.
        diffs = self.data - self.baseline
        nDiff = len(diffs)
        meanDiff = float(np.mean(diffs))
        sdDiff = float(np.std(diffs, ddof=1))
        se = sdDiff / (nDiff ** 0.5) if (nDiff > 0) else float('nan')
        # lower test: H0 mean <= -margin  vs H1 mean > -margin
        tLower = (meanDiff - (-self.equivalenceMargin)) / (se + 1e-12)
        pLower = 1.0 - stats.t.cdf(tLower, df=nDiff - 1)
        # upper test: H0 mean >= margin vs H1 mean < margin
        tUpper = (meanDiff - (self.equivalenceMargin)) / (se + 1e-12)
        pUpper = stats.t.cdf(tUpper, df=nDiff - 1)
        pTost = max(pLower, pUpper)
        paired["TOST"] = {"pValue": float(pTost), "EquivalenceMargin": self.equivalenceMargin}
      # Attach paired test results to output when any were run.
      if (paired):
        out["PairedTests"] = paired

    # Compute effect size metrics when a baseline is provided.
    if (self.baseline is not None):
      # Prepare an effect size dictionary.
      effect = {}
      # Compute Cohen's d and Hedges' g when applicable.
      if ((self.applicableMethods.get("CohensD", False)) or (self.applicableMethods.get("HedgesG", False))):
        # Compute differences between data and baseline for effect size calculations.
        diffs = self.data - self.baseline
        # Compute Cohen's d as mean difference over sample standard deviation.
        cohensD = np.mean(diffs) / np.std(diffs, ddof=1)
        # Store Cohen's d under a CamelCase key.
        effect["CohensD"] = cohensD
        # Compute Hedges' g adjustment when applicable.
        if (self.applicableMethods.get("HedgesG", False)):
          # Determine sample size for the pairwise differences.
          nDiff = len(diffs)
          # Apply small-sample correction factor to Cohen's d for Hedges' g.
          hedgeG = cohensD * (1 - 3 / (4 * nDiff - 9))
          # Store Hedges' g under a CamelCase key.
          effect["HedgesG"] = hedgeG
      # Compute Vargha-Delaney A12 when non-normal effect size is required.
      if (self.applicableMethods.get("VarghaDelaneyA12", False)):
        # Determine sizes of the two groups for ranking.
        m = len(self.data)
        n = len(self.baseline)
        # Concatenate the two samples for rank-based computations.
        combined = np.concatenate([self.data, self.baseline])
        # Compute ranks of the combined sample.
        ranks = stats.rankdata(combined)
        # Compute rank sum for the first group portion.
        rankSum = np.sum(ranks[:m])
        # Compute the Vargha-Delaney A12 measure from the rank sum.
        a12 = (rankSum - m * (m + 1) / 2) / (m * n)
        # Store the A12 measure under a CamelCase key.
        effect["VarghaDelaneyA12"] = a12
      # Attach effect size results to output when any were computed.
      if (effect):
        out["EffectSize"] = effect
    # Compute Cochran's Q for binary multi-configuration repeated-measures when applicable.
    if (self.applicableMethods.get("CochransQ", False)):
      # Ensure data is a NumPy array for matrix operations.
      dataMat = np.asarray(self.data)
      # Determine the number of subjects from the data rows.
      nSubjects = int(dataMat.shape[0])
      # Determine the number of configurations from the data columns.
      kConfigs = int(dataMat.shape[1])
      # Compute the column-wise sums of successes for each configuration.
      colSums = np.sum(dataMat, axis=0)
      # Compute the total number of successes across all cells.
      totalSuccesses = float(np.sum(colSums))
      # Compute the numerator for Cochran's Q statistic.
      numer = kConfigs * (kConfigs - 1) * np.sum((colSums - totalSuccesses / kConfigs) ** 2)
      # Compute the denominator for Cochran's Q statistic.
      rowSums = np.sum(dataMat, axis=1)
      denom = np.sum(rowSums * (kConfigs - rowSums))
      # Handle degenerate case where denominator is zero or negative.
      if (denom <= 0):
        out["CochransQ"] = {"Statistic": None, "pValue": None, "Message": "Degenerate data; denominator is zero."}
      else:
        # Compute the Cochran's Q statistic value.
        qStat = float(numer / denom)
        # Compute the p-value from the chi-square distribution with k-1 degrees of freedom.
        qP = float(stats.chi2.sf(qStat, df=(kConfigs - 1)))
        # Store the Cochran's Q results under a CamelCase key.
        out["CochransQ"] = {"Statistic": qStat, "pValue": qP}

    # Compute ICC(2,1) (two-way random effects single-measure) for multi-config continuous data when applicable.
    if (self.applicableMethods.get("ICC", False)):
      # Ensure the data matrix is a NumPy array.
      dataMat = np.asarray(self.data)
      # Determine the number of subjects.
      nSubjects = int(dataMat.shape[0])
      # Determine the number of configurations (raters).
      kConfigs = int(dataMat.shape[1])
      # Compute the grand mean of all observations.
      grandMean = float(np.mean(dataMat))
      # Compute the mean for each subject (row mean).
      rowMeans = np.mean(dataMat, axis=1)
      # Compute the mean for each rater/configuration (column mean).
      colMeans = np.mean(dataMat, axis=0)
      # Compute the sum of squares for rows (subjects).
      ssr = float(kConfigs * np.sum((rowMeans - grandMean) ** 2))
      # Compute the sum of squares for columns (raters/configs).
      ssc = float(nSubjects * np.sum((colMeans - grandMean) ** 2))
      # Compute the total sum of squares.
      sst = float(np.sum((dataMat - grandMean) ** 2))
      # Compute the sum of squares for residuals (error) by subtraction.
      sse = float(sst - ssr - ssc)
      # Compute mean square for rows.
      msr = float(ssr / (nSubjects - 1)) if (nSubjects > 1) else float('nan')
      # Compute mean square for columns.
      msc = float(ssc / (kConfigs - 1)) if (kConfigs > 1) else float('nan')
      # Compute mean square error.
      mse = float(sse / ((nSubjects - 1) * (kConfigs - 1))) if ((nSubjects > 1) and (kConfigs > 1)) else float('nan')
      # Compute the ICC denominator using the ICC(2,1) formula.
      denomIcc = msr + (kConfigs - 1) * mse + (kConfigs * (msc - mse) / float(nSubjects))
      # Handle degenerate denominator for ICC calculation.
      if (denomIcc == 0) or (np.isnan(denomIcc)):
        out["ICC"] = {"ICC": None, "Message": "Degenerate ICC computation; denominator is zero or NaN."}
      else:
        # Compute the ICC(2,1) value.
        iccVal = float((msr - mse) / denomIcc)
        # Store the ICC results under a CamelCase key.
        out["ICC"] = {"ICC": iccVal, "n": nSubjects, "k": kConfigs}

    # Compute Friedman test for multi-config repeated-measures when applicable.
    if (self.applicableMethods.get("Friedman", False)):
      # Convert data to a NumPy array for column-wise passing into scipy's function.
      dataMat = np.asarray(self.data)
      try:
        # Run the Friedman test using column-wise samples as arguments.
        friedmanStat, friedmanP = stats.friedmanchisquare(*tuple(dataMat.T))
        # Store the Friedman test results under a CamelCase key.
        out["Friedman"] = {"Statistic": float(friedmanStat), "pValue": float(friedmanP)}
      except Exception as e:
        # Store a failure message if the Friedman computation failed.
        out["Friedman"] = {"Statistic": None, "pValue": None, "Message": str(e)}

    # Attempt repeated-measures ANOVA using statsmodels AnovaRM when applicable.
    if (self.applicableMethods.get("RMANOVA", False)):
      try:
        # Import AnovaRM from statsmodels for repeated-measures ANOVA.
        from statsmodels.stats.anova import AnovaRM
        # Prepare the data in long format for AnovaRM.
        dataMat = np.asarray(self.data)
        # Determine number of subjects and configs.
        nSubjects = int(dataMat.shape[0])
        kConfigs = int(dataMat.shape[1])
        # Build subject indices repeated for each configuration.
        subjectIdx = np.repeat(np.arange(nSubjects), kConfigs)
        # Build configuration indices tiled for each subject.
        configIdx = np.tile(np.arange(kConfigs), nSubjects)
        # Flatten the data matrix to a 1-D array of values.
        values = dataMat.ravel()
        # Create a pandas DataFrame in long format for AnovaRM.
        dfLong = pd.DataFrame({"subject": subjectIdx, "config": configIdx, "value": values})
        # Fit AnovaRM to the long-format DataFrame.
        anovaRes = AnovaRM(dfLong, "value", "subject", within=["config"]).fit()
        # Attempt to extract the anova table for the within-subject effect.
        anovaTable = anovaRes.anova_table
        # Attempt to locate the row corresponding to the config effect.
        if ("config" in anovaTable.index):
          row = anovaTable.loc["config"]
        else:
          row = anovaTable.iloc[0]
        # Extract F and p-value from the anova row with defensive key handling.
        fVal = float(row.get("F Value", row.get("F", np.nan)))
        pVal = float(row.get("Pr > F", row.get("PR(>F)", np.nan)))
        # Store the RMANOVA results under a CamelCase key.
        out["RMANOVA"] = {"F": fVal, "pValue": pVal}
      except Exception:
        # Store a message indicating RMANOVA could not be computed if an error occurred.
        out["RMANOVA"] = {"Message": "statsmodels AnovaRM not available or failed to run."}

    # Compute expected calibration error (ECE) when predicted probabilities and labels are available.
    if (self.applicableMethods.get("CalibrationECE", False)):
      # Convert predictions and labels to NumPy arrays for vectorized operations.
      preds = np.asarray(self.predictions)
      # Convert true labels to a NumPy array.
      labs = np.asarray(self.labels)
      # Define the number of bins to compute ECE over.
      nBins = int(getattr(self, "eceBins", 10))
      # Initialize the ECE accumulator.
      eceVal = 0.0
      # Compute bin edges using equal-width intervals over [0,1].
      binEdges = np.linspace(0.0, 1.0, nBins + 1)
      # Digitize predictions into bins according to edges.
      binIdx = np.digitize(preds, binEdges, right=False) - 1
      # Iterate over each bin to compute weighted absolute gaps.
      for b in range(nBins):
        # Select indices that belong to the current bin.
        idx = (binIdx == b)
        # Continue when the bin is empty to avoid division by zero.
        if (np.sum(idx) == 0):
          continue
        # Compute the average predicted probability in the bin.
        avgPred = float(np.mean(preds[idx]))
        # Compute the average observed label in the bin.
        avgLabel = float(np.mean(labs[idx]))
        # Compute the weight of the bin as its fraction of total samples.
        weight = float(np.sum(idx)) / float(len(preds))
        # Accumulate the weighted absolute difference into the ECE value.
        eceVal += weight * abs(avgPred - avgLabel)
      # Store the ECE result under a CamelCase key.
      out["CalibrationECE"] = {"ECE": float(eceVal), "nBins": nBins}

    # Compute Hosmer-Lemeshow goodness-of-fit statistic when predicted probabilities and labels are available.
    if (self.applicableMethods.get("HosmerLemeshow", False)):
      # Convert predictions and labels to NumPy arrays for vectorized operations.
      preds = np.asarray(self.predictions)
      # Convert true labels to a NumPy array.
      labs = np.asarray(self.labels)
      # Define the number of groups (deciles) for the Hosmer-Lemeshow test.
      groups = int(getattr(self, "hlGroups", 10))
      # Attempt to create group edges using quantiles to ensure roughly equal-sized bins.
      try:
        edges = np.quantile(preds, np.linspace(0.0, 1.0, groups + 1))
      except Exception:
        # Fall back to equal-width edges when quantile computation fails.
        edges = np.linspace(0.0, 1.0, groups + 1)
      # Digitize predictions into HL groups according to edges.
      groupIdx = np.digitize(preds, edges, right=True) - 1
      # Initialize the Hosmer-Lemeshow statistic accumulator.
      hlStat = 0.0
      # Small epsilon to avoid division by zero.
      eps = 1e-8
      # Iterate over each group to compute observed and expected counts.
      for g in range(groups):
        # Select indices that belong to the current group.
        idx = (groupIdx == g)
        # Continue when the group is empty to avoid divisions by zero.
        if (np.sum(idx) == 0):
          continue
        # Compute the number of observations in the group.
        nGroup = float(np.sum(idx))
        # Compute observed number of events in the group.
        obs = float(np.sum(labs[idx]))
        # Compute expected number of events as the sum of predicted probabilities in the group.
        exp = float(np.sum(preds[idx]))
        # Compute observed and expected non-events for the group.
        obs0 = nGroup - obs
        exp0 = nGroup - exp
        # Accumulate the two components into the Hosmer-Lemeshow statistic.
        hlStat += ((obs - exp) ** 2) / (exp + eps) + ((obs0 - exp0) ** 2) / (exp0 + eps)
      # Compute p-value using chi-square survival function with groups-2 degrees of freedom.
      hlP = float(stats.chi2.sf(hlStat, df=max(groups - 2, 1)))
      # Store the Hosmer-Lemeshow results under a CamelCase key.
      out["HosmerLemeshow"] = {"Chi2": float(hlStat), "pValue": hlP, "Groups": groups}

    # Compute bootstrap BCa confidence intervals for key statistics when applicable.
    if (self.applicableMethods.get("BCaBootstrap", False)):
      # Compute BCa bootstrap intervals using a helper method.
      boot = self.ComputeBCaBootstrap()
      # Attach bootstrap results to output when available.
      if (boot):
        out["BootstrapCIs"] = boot

    # Compute a distribution-free approximate tolerance interval using bootstrap when applicable.
    if (self.applicableMethods.get("ToleranceInterval", False)):
      # Define default coverage and confidence if not specified on the instance.
      coverage = getattr(self, "toleranceCoverage", 0.95)
      confidence = getattr(self, "toleranceConfidence", 0.95)
      # Compute quantile levels for a two-sided interval.
      qLow = (1 - coverage) / 2
      qHigh = 1 - qLow
      # Run bootstrap sampling to estimate sampling distribution of these quantiles.
      nBoot = 2000
      bootLow = []
      bootHigh = []
      for _ in range(nBoot):
        # Draw a bootstrap sample.
        sample = np.random.choice(self.data, size=len(self.data), replace=True)
        # Compute the sample quantiles.
        bootLow.append(np.percentile(sample, qLow * 100))
        bootHigh.append(np.percentile(sample, qHigh * 100))
      # Compute confidence intervals for the population quantiles using bootstrap percentiles.
      alphaCI = (1 - confidence) / 2
      tolLow = np.percentile(bootLow, 100 * alphaCI)
      tolHigh = np.percentile(bootHigh, 100 * (1 - alphaCI))
      # Record the tolerance interval approximation and parameters.
      out["ToleranceInterval"] = {
        "Coverage"  : coverage,
        "Confidence": confidence,
        "Lower"     : float(tolLow),
        "Upper"     : float(tolHigh)
      }

    # Update the object's results dictionary with the computed outputs.
    self.results.update(out)

  # Compute BCa bootstrap confidence intervals for relevant statistics.
  def ComputeBCaBootstrap(self, nBoot=5000, ciLevel=0.95):
    r'''
    Compute bias-corrected and accelerated (BCa) bootstrap confidence intervals for key statistics based on the provided data.
      - This method implements the BCa bootstrap procedure to compute confidence intervals for statistics such as mean, median, and proportions.
      - It generates bootstrap samples by resampling the original data with replacement and computes the statistic of interest on each bootstrap sample to create a distribution of bootstrap estimates.
      - The bias-correction (z0) and acceleration (acc) parameters are computed using the bootstrap distribution and jackknife replicates, respectively, to adjust the percentile positions for the confidence interval.
      - The method returns a dictionary containing the point estimate and the lower and upper bounds of the BCa confidence interval for each relevant statistic, formatted with CamelCase keys for clarity.

    Parameters:
      nBoot (int, optional): Number of bootstrap resamples to perform. Default is 5000.
      ciLevel (float, optional): Confidence level for the intervals (e.g., 0.95 for 95% CI). Default is 0.95.

    Returns:
      dict: A dictionary containing the BCa confidence intervals and point estimates for relevant statistics based on the data type (proportion for binary data, mean/median/fifth percentile for continuous data).
    '''

    # Import norm for quantile functions used in BCa computations.
    from scipy.stats import norm

    # Define an inner function to compute a BCa CI for a statistic function.
    def bcaCI(data, statFunc, nBoot=nBoot, ciLevel=ciLevel):
      # Compute sample size for the provided data.
      n = len(data)
      # Return None when no data is available.
      if (n == 0):
        return None
      # Generate bootstrap replicates of the statistic.
      bootStats = []
      for _ in range(nBoot):
        # Draw a bootstrap sample with replacement.
        sample = np.random.choice(data, size=n, replace=True)
        # Evaluate the statistic on the bootstrap sample.
        bootStats.append(statFunc(sample))
      # Convert bootstrap replicates to a numpy array.
      bootStats = np.array(bootStats)
      # Compute the observed statistic on the original sample.
      thetaHat = statFunc(data)
      # Compute the bias-correction z0 parameter for BCa.
      z0 = norm.ppf(np.mean(bootStats < thetaHat) + 1e-8)
      # Compute jackknife replicates for acceleration computation.
      jackStats = []
      for i in range(n):
        # Remove a single observation to form a jackknife sample.
        jackSample = np.delete(data, i)
        # Compute the statistic on the jackknife sample.
        jackStats.append(statFunc(jackSample))
      # Convert jackknife replicates to a numpy array.
      jackStats = np.array(jackStats)
      # Compute mean of jackknife replicates.
      meanJack = np.mean(jackStats)
      # Compute denominator used in acceleration formula.
      denominator = np.sum((meanJack - jackStats) ** 2)
      # Handle the degenerate case where denominator is zero.
      if (denominator == 0):
        # Set acceleration to zero when denominator is zero.
        acc = 0.0
      else:
        # Compute acceleration parameter using third central moment of jackknife.
        acc = np.sum((meanJack - jackStats) ** 3) / (6 * (denominator) ** 1.5)
      # Compute alpha tail probability for the requested CI level.
      alpha = (1 - ciLevel) / 2
      # Compute z-scores for alpha and (1-alpha).
      zAlpha = norm.ppf(alpha)
      zBeta = norm.ppf(1 - alpha)
      # Compute adjusted percentile positions using BCa formulas.
      p1 = norm.cdf(z0 + (z0 + zAlpha) / (1 - acc * (z0 + zAlpha) + 1e-8))
      p2 = norm.cdf(z0 + (z0 + zBeta) / (1 - acc * (z0 + zBeta) + 1e-8))
      # Extract percentiles from bootstrap replicates for lower and upper CI.
      ciLow = np.percentile(bootStats, p1 * 100)
      ciHigh = np.percentile(bootStats, p2 * 100)
      # Return the BCa CI and point estimate as a dictionary.
      return {"Estimate": float(thetaHat), "Lower": float(ciLow), "Upper": float(ciHigh)}

    # Prepare the output dictionary for bootstrap CIs.
    out = {}
    # Compute BCa CI for proportions when data is binary.
    if (self.dataType == "Binary"):
      # Compute BCa CI for the sample proportion.
      out["Proportion"] = bcaCI(self.data, np.mean)
    else:
      # Compute BCa CI for mean, median, and fifth percentile for continuous data.
      out["Mean"] = bcaCI(self.data, np.mean)
      out["Median"] = bcaCI(self.data, np.median)
      out["FifthPercentile"] = bcaCI(self.data, lambda x: np.percentile(x, 5))
    # Return the assembled bootstrap CI dictionary.
    return out


def StatisticalAnalysis(results, hypothesizedMean=0, secondMetricList=None, confidenceLevel=0.95, nBootstraps=1000):
  r'''
  Perform comprehensive statistical analysis on a list of results.

  Parameters:
    results (list): List of performance metrics (e.g., accuracy, scores).
    hypothesizedMean (float, optional): Mean value for one-sample t-test. Default is 0.
    secondMetricList (list, optional): Second list of metrics for correlation/regression analysis. Default is None.
    confidenceLevel (float, optional): Confidence level for confidence intervals. Default is 0.95.
    nBootstraps (int, optional): Number of bootstrap resamples for confidence intervals. Default is 1000.

  Returns:
    dict: Dictionary containing all statistical analysis results.

  Examples
  --------
  .. code-block:: python

    import HMB.StatisticalAnalysisHelper as sah

    results = [0.8, 0.82, 0.78, 0.81, 0.79]
    analysisReport = sah.StatisticalAnalysis(results, hypothesizedMean=0.75)
    print("Statistical Analysis Report:")
    print(analysisReport)
  '''

  # Initialize an empty dictionary to store the analysis results.
  report = {}

  # Convert the list of results to a numpy array for easier calculations.
  results = np.array(results)

  # ==============================================================================================================
  # Descriptive Statistics
  # Descriptive statistics summarize and describe the main features of a dataset. They provide a quick overview
  # of the data's central tendency, dispersion, and shape, making them essential for initial data exploration.
  # ==============================================================================================================
  report.update(
    {
      "Mean"                           : np.mean(results),  # Central tendency: average value.
      "Median"                         : np.median(results),  # Central tendency: middle value.
      "Mode"                           : stats.mode(results)[0] if (len(results) > 0) else np.nan,
      # Most frequent value.
      # Dispersion: spread of the data (sample standard deviation).
      "Standard Deviation (Sample)"    : np.std(results, ddof=1),
      "Standard Deviation (Population)": np.std(results, ddof=0),
      "Coefficient of Variation (CV)"  : (
        np.std(results, ddof=1) / np.mean(results) * 100 if (np.mean(results) != 0) else np.nan
      ),  # Relative measure of dispersion: standard deviation as a percentage of the mean.
      "Variance (Sample)"              : np.var(results, ddof=1),  # Dispersion: squared standard deviation.
      "Variance (Population)"          : np.var(results, ddof=0),
      # Dispersion: squared standard deviation for population.
      "Minimum"                        : np.min(results),  # Minimum value in the dataset.
      "Maximum"                        : np.max(results),  # Maximum value in the dataset.
      "Range"                          : np.max(results) - np.min(results),  # Difference between max and min.
      "Interquartile Range (IQR)"      : stats.iqr(results),  # Dispersion: range of the middle 50% of data.
      "Geometric Mean"                 : stats.gmean(results),  # Central tendency: multiplicative average.
      "Harmonic Mean"                  : stats.hmean(results),
      # Central tendency: reciprocal of the arithmetic mean of reciprocals.
      "Trimmed Mean (10%)"             : stats.trim_mean(results, proportiontocut=0.1),
      # Robust central tendency: mean after trimming outliers.
      "Winsorized Mean (10%)"          : stats.mstats.winsorize(results, limits=[0.1, 0.1]).mean(),
      # Robust central tendency: mean after capping outliers.
      "Percentiles"                    : {  # Distribution: key percentiles.
        "10th": np.percentile(results, 10),
        "25th": np.percentile(results, 25),
        "50th": np.percentile(results, 50),
        "75th": np.percentile(results, 75),
        "90th": np.percentile(results, 90),
      },
      "Skewness"                       : skew(results),  # Measure of asymmetry of the distribution.
      # Measure of the "tailedness" of the distribution.
      "Kurtosis (Fisher)"              : kurtosis(results, fisher=True),
      "Kurtosis (Pearson)"             : kurtosis(results, fisher=False),
    }
  )

  # ==============================================================================================================
  # Inferential Statistics
  # Inferential statistics allow us to make predictions or inferences about a population based on sample data.
  # They include confidence intervals, hypothesis testing, and effect sizes, which are crucial for drawing
  # conclusions from data.
  # ==============================================================================================================
  # Apply t-stat using scipy.
  mean = np.mean(results)
  stdDev = np.std(results, ddof=1)
  n = len(results)
  confInt = zconfint(results, alpha=1 - confidenceLevel)  # Confidence interval for the mean.

  def _bootstrapCI(data, statFunc=np.mean, nBootstraps=nBootstraps, alpha=1 - confidenceLevel):
    r'''
    Calculate bootstrap confidence interval for a statistic (e.g., mean).

    Parameters:
      data (array-like): Data to resample.
      statFunc (callable, optional): Statistic function to apply (default: np.mean).
      nBootstraps (int, optional): Number of bootstrap resamples.
      alpha (float, optional): Significance level (default: 1 - confidenceLevel).

    Returns:
      tuple: Lower and upper bounds of the confidence interval.
    '''

    bootstrappedStats = []
    for _ in range(nBootstraps):
      sample = np.random.choice(data, size=len(data), replace=True)
      bootstrappedStats.append(statFunc(sample))
    lowerPercentile = (alpha / 2.0) * 100.0
    upperPercentile = (1.0 - alpha / 2.0) * 100.0
    ciLower = np.percentile(bootstrappedStats, lowerPercentile)
    ciUpper = np.percentile(bootstrappedStats, upperPercentile)
    return ciLower, ciUpper

  def _bootstrapCIAdvanced(data, statFunc=np.mean, nBootstraps=nBootstraps, confidenceLevel=confidenceLevel):
    r'''
    Calculate advanced bootstrap confidence interval using scipy's bootstrap.

    Parameters:
      data (array-like): Data to resample.
      statFunc (callable, optional): Statistic function to apply (default: np.mean).
      nBootstraps (int, optional): Number of bootstrap resamples.
      confidenceLevel (float, optional): Confidence level for the interval.

    Returns:
      tuple: Lower and upper bounds of the confidence interval.
    '''

    if (len(data) == 0):
      return np.nan, np.nan
    try:
      bootstrappedStats = bootstrap(
        (data,),  # Tuple of arrays to bootstrap from.
        statFunc,  # Function to compute the statistic (e.g., mean, median).
        n_resamples=nBootstraps,  # Number of bootstrap resamples.
        confidence_level=confidenceLevel,  # Confidence level for the interval.
        random_state=42,  # Optional: for reproducibility.
      )
    except Exception as e:
      print(f"Error in bootstrap calculation: {e}")
      return np.nan, np.nan
    return bootstrappedStats.confidence_interval.low, bootstrappedStats.confidence_interval.high

  def _predicitonInterval(data, confidenceLevel=confidenceLevel):
    r'''
    Calculate approximate prediction interval for future observations.

    Parameters:
      data (array-like): Data to analyze.
      confidenceLevel (float, optional): Confidence level for the interval.

    Returns:
      tuple: Lower and upper bounds of the prediction interval.
    '''

    if (len(data) < 2):
      return np.nan, np.nan
    try:
      meanVal = np.mean(data)
      stdVal = np.std(data, ddof=1)
      nVal = len(data)
      alpha = 1.0 - confidenceLevel
      tVal = stats.t.ppf(1.0 - alpha / 2.0, df=nVal - 1)
      # Prediction interval formula
      margin = tVal * stdVal * np.sqrt(1.0 + 1.0 / nVal)
      return meanVal - margin, meanVal + margin
    except Exception:
      return np.nan, np.nan

  bootCILower, bootCIUpper = _bootstrapCI(results)
  medianCILow, medianCIHigh = _bootstrapCIAdvanced(results, statFunc=np.median, nBootstraps=nBootstraps)
  predIntLow, predIntHigh = _predicitonInterval(results, confidenceLevel=confidenceLevel)  # Prediction interval.

  report.update(
    {
      "Confidence Interval (Mean)": {  # Range within which the true population mean is likely to fall.
        f"{int(confidenceLevel * 100)}%": confInt,
        "Lower Bound"                   : confInt[0],
        "Upper Bound"                   : confInt[1],
      },
      "Bootstrap CI (Mean)"       : {
        f"{int(confidenceLevel * 100)}%": (bootCILower, bootCIUpper),
        "Lower Bound"                   : bootCILower,
        "Upper Bound"                   : bootCIUpper,
      },
      "Bootstrap CI (Median)"     : {
        f"{int(confidenceLevel * 100)}%": (medianCILow, medianCIHigh),
        "Lower Bound"                   : medianCILow,
        "Upper Bound"                   : medianCIHigh,
      },
      "Prediction Interval"       : {  # Range within which future observations are likely to fall.
        f"{int(confidenceLevel * 100)}%": (predIntLow, predIntHigh),
        "Lower Bound"                   : predIntLow,
        "Upper Bound"                   : predIntHigh,
      },
      "One-Sample T-Test"         : {  # Test if the sample mean differs from a hypothesized mean.
        "T-statistic" : stats.ttest_1samp(results, hypothesizedMean).statistic,
        "P-value"     : stats.ttest_1samp(results, hypothesizedMean).pvalue,
        "Significance": (
          "Significant (p < 0.05); Reject the Null Hypothesis" if stats.ttest_1samp(
            results, hypothesizedMean
          ).pvalue < 0.05 else "Not Significant (p >= 0.05); Fail to Reject the Null Hypothesis"
        ),
      },
      "Effect Size"               : {  # Measure the magnitude of the difference or relationship.
        "Cohen's d": (mean - hypothesizedMean) / stdDev,  # Standardized difference between means.
        "Hedges' g": (mean - hypothesizedMean) / stdDev * (1 - (3) / (4 * (n - 1) - 1)),
        # Adjusted Cohen's d for small samples.
      },
      "Statistical Power"         : TTestPower().power(  # Probability of detecting an effect if it exists.
        effect_size=(mean - hypothesizedMean) / stdDev,
        nobs=n,
        alpha=0.05
      ),
    }
  )

  # ==============================================================================================================
  # Normality Tests
  # Normality tests assess whether a dataset follows a normal distribution. This is important because many
  # statistical methods assume normality. These tests help determine if parametric or non-parametric methods
  # should be used.
  # ==============================================================================================================
  shapiroStat, shapiroPValue = stats.shapiro(results)  # Shapiro-Wilk test for normality.
  jbStat, jbP = stats.jarque_bera(results)  # Jarque-Bera test for normality.

  if (len(results) >= 20):
    dagostinoStat, dagostinoP = stats.normaltest(results)  # D’Agostino’s K² test for normality.
    toAdd = {
      "Statistic": dagostinoStat,
      "P-value"  : dagostinoP,
    }
  else:
    toAdd = "Not applicable (n < 20)"

  report.update(
    {
      "Shapiro-Wilk Test"      : {  # Test for normality, especially for small samples.
        "Statistic"     : shapiroStat,
        "P-value"       : shapiroPValue,
        "Interpretation": "Normally Distributed (p > 0.05)" if (
            shapiroPValue > 0.05) else "Not Normally Distributed (p <= 0.05)",
      },
      "D'Agostino's K^2 Test"  : toAdd,  # Test for skewness and kurtosis.
      "Jarque-Bera Test"       : {  # Test for normality based on skewness and kurtosis.
        "Statistic": jbStat,
        "P-value"  : jbP,
      },
      "Anderson-Darling Test"  : stats.anderson(results),  # Test for normality, sensitive to tails.
      "Kolmogorov-Smirnov Test": stats.kstest(results, "norm"),
      # Test for normality by comparing to a reference distribution.
    }
  )

  # ==============================================================================================================
  # Distribution Fit Testing
  # Distribution fit testing evaluates how well a dataset fits a specific probability distribution.
  # This is important for understanding the underlying distribution of the data, which can inform further
  # analysis and modeling choices. Common distributions tested include normal, log-normal, exponential,
  # gamma, and Weibull distributions.
  # ==============================================================================================================

  # Distribution Fit Testing.
  distributionsToTest = ["norm", "lognorm", "expon", "gamma", "weibull_min"]

  bestDistName = None
  bestKsStatistic = np.inf
  bestKsPvalue = np.nan
  bestDistParams = None

  distFitResults = {}
  for distName in distributionsToTest:
    try:
      dist = getattr(stats, distName)
      # Fit distribution.
      params = dist.fit(results)
      # Perform KS test.
      ksStat, ksPvalue = stats.kstest(results, lambda x: dist.cdf(x, *params))
      distFitResults[distName] = {
        "Parameters"    : params,
        "KSStatistic"   : ksStat,
        "KSPValue"      : ksPvalue,
        "Interpretation": "Good Fit (p > 0.05)" if (ksPvalue > 0.05) else "Poor Fit (p <= 0.05)"
      }
      # Update best fit if this is better (lower KS statistic).
      if (ksStat < bestKsStatistic):
        bestKsStatistic = ksStat
        bestKsPvalue = ksPvalue
        bestDistName = distName
        bestDistParams = params
    except Exception:
      # Handle cases where fitting fails (e.g., lognorm with negative data).
      distFitResults[distName] = {
        "Parameters"    : "Fit Failed",
        "KS Statistic"  : np.nan,
        "KS PValue"     : np.nan,
        "Interpretation": "Fit Failed"
      }

  report["DistributionFitTesting"] = {
    "Individual Distributions": distFitResults,
    "Best Fit Distribution"   : bestDistName,
    "Best Fit Parameters"     : bestDistParams,
    "Best KS Statistic"       : bestKsStatistic,
    "Best KS PValue"          : bestKsPvalue,
  }

  # ==============================================================================================================
  # Outlier Detection
  # Outlier detection identifies data points that deviate significantly from the rest of the data. Outliers can
  # skew results and affect the validity of statistical analyses, so it's important to detect and handle them.
  # ==============================================================================================================
  Q1 = np.percentile(results, 25)
  Q3 = np.percentile(results, 75)
  iqr = Q3 - Q1
  lowerBound = Q1 - 1.5 * iqr
  upperBound = Q3 + 1.5 * iqr
  outliers = results[(results < lowerBound) | (results > upperBound)]

  zScores = (results - mean) / stdDev  # Z-scores for outlier detection.
  den = stats.median_abs_deviation(results)
  modifiedZScores = 0.6745 * (
      results - np.median(results)) / den if (den != 0) else np.nan  # Modified Z-scores for outlier detection.

  report.update(
    {
      "IQR Method"                 : {  # Outliers detected using the interquartile range.
        "Outliers"   : outliers,
        "Lower Bound": lowerBound,
        "Upper Bound": upperBound,
      },
      "Z-Score Method"             : {  # Outliers detected using Z-scores.
        "Outliers": results[np.abs(zScores) > 3],
      },
      "Modified Z-Score Method"    : {  # Outliers detected using modified Z-scores.
        "Outliers": results[np.abs(modifiedZScores) > 3.5] if (den != 0) else "Not applicable (MAD = 0)",
      },
      "Outlier Handling Suggestion": (  # Suggestions for handling outliers.
        "Consider using robust statistical methods (e.g., trimmed mean, Winsorized mean)."
        if (len(outliers) > 0) else "No significant outliers detected."
      ),
    }
  )

  # ==============================================================================================================
  # Transformations
  # Data transformations are applied to make data more suitable for analysis. Common transformations include
  # log, Box-Cox, and Yeo-Johnson, which can stabilize variance, normalize data, or improve model performance.
  # ==============================================================================================================
  report.update(
    {
      "Log Transformation"        : np.log(results),  # Stabilizes variance and reduces skewness.
      "Yeo-Johnson Transformation": stats.yeojohnson(results)[0],  # Handles positive and negative values.
      "Standardized Data"         : (results - mean) / stdDev,  # Scales data to have mean 0 and standard deviation 1.
      "Min-Max Scaled Data"       : (results - np.min(results)) / (np.max(results) - np.min(results)),
      # Scales data to a range of 0 to 1.
    }
  )

  # ==============================================================================================================
  # Correlation and Regression
  # Correlation and regression analysis explore relationships between variables. Correlation measures the strength
  # and direction of the relationship, while regression models the relationship to make predictions.
  # ==============================================================================================================
  if (secondMetricList is not None):
    secondMetricList = np.array(secondMetricList)
    if (len(results) == len(secondMetricList)):
      pearsonCorr, pearsonPValue = stats.pearsonr(results, secondMetricList)  # Pearson correlation.
      spearmanCorr, spearmanP = stats.spearmanr(results, secondMetricList)  # Spearman correlation.
      kendallCorr, kendallP = stats.kendalltau(results, secondMetricList)  # Kendall’s Tau correlation.

      X = secondMetricList.reshape(-1, 1)
      y = results
      model = LinearRegression()
      model.fit(X, y)  # Simple linear regression.
      slope = model.coef_[0]
      intercept = model.intercept_

      XSm = sm.add_constant(secondMetricList)
      modelSM = sm.OLS(results, XSm).fit()  # Regression diagnostics.

      report.update(
        {
          "Pearson Correlation" : {  # Measures linear relationship.
            "Coefficient" : pearsonCorr,
            "P-value"     : pearsonPValue,
            "Significance": "Significant (p < 0.05)" if (pearsonPValue < 0.05) else "Not Significant (p >= 0.05)",
          },
          "Spearman Correlation": {  # Measures monotonic relationship.
            "Coefficient": spearmanCorr,
            "P-value"    : spearmanP,
          },
          "Kendall's Tau"       : {  # Measures ordinal association.
            "Coefficient": kendallCorr,
            "P-value"    : kendallP,
          },
          "Linear Regression"   : {  # Models the relationship between variables.
            "Slope"              : slope,
            "Intercept"          : intercept,
            "Regression Equation": f"y = {slope:.4f} * x + {intercept:.4f}",
            "R²"                 : modelSM.rsquared,
            "Adjusted R²"        : modelSM.rsquared_adj,
            "F-statistic"        : modelSM.fvalue,
            "F-test P-value"     : modelSM.f_pvalue,
          },
        }
      )
    else:
      report["Correlation and Regression"] = "Skipped: The two metric lists must have the same length."

  # ==============================================================================================================
  # Time Series Analysis
  # Time series analysis examines data points collected over time to identify patterns, trends, and dependencies.
  # Autocorrelation measures the relationship between a variable's current and past values.
  # ==============================================================================================================
  if (len(results) > 1):
    autocorrelation = np.correlate(results - np.mean(results), results - np.mean(results), mode="full")

    try:
      # Test up to min(10, len/2) lags or at least 1.
      maxLag = max(1, min(10, len(results) // 2 - 1))
      lbStat, lbPvalue = acorr_ljungbox(results, lags=maxLag, return_df=False)
      ljungboxResult = {
        "Lags Tested"    : list(range(1, len(lbStat) + 1)),
        "Statistics"     : lbStat.tolist(),
        "P Values"       : lbPvalue.tolist(),
        "Any Significant": any(p < 0.05 for p in lbPvalue)
      }
    except Exception as e:
      ljungboxResult = f"Ljung-Box test failed: {e}"

    report.update(
      {
        # Measures the correlation of a variable with itself over time.
        "Autocorrelation": autocorrelation[len(results) - 1:],
        "Ljung-Box Test" : ljungboxResult,  # Tests for autocorrelation in time series data.
      }
    )

  return report


def ExtractDataFromSummaryFile(file):
  r'''
  Extract and organize data from a summary CSV file containing names, metrics, and trial data. The file is expected to be structured as follows:
    - First line: Comma-separated names (headers for individuals or categories).
    - Second line: Comma-separated metrics (headers for performance or evaluation criteria).
    - Subsequent lines: Numerical values corresponding to metrics for each trial.

  Example of the file structure (if you have multiple systems):
    System A, , , , , , System B, , , , ,
    Precision, Recall, F1, Accuracy, Specificity, Average, Precision, Recall, F1, Accuracy, Specificity, Average
    0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133, 0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133
    0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282, 0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282
    0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406, 0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406
    0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339, 0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339
    0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813, 0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813

  Example of the file structure (if you have a single system):
    Precision, Recall, F1, Accuracy, Specificity, Average
    Metric, Metric, Metric, Metric, Metric, Metric
    0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133,
    0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282,
    0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406,
    0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339,
    0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813,

  Parameters:
    file (str): Path to the summary CSV file.

  Returns:
    tuple: A tuple containing:
      - history (list): A list of dictionaries, each representing a name with its metrics and trial data.
      - names (list): A list of cleaned names extracted from the file.
      - metrics (list): A list of cleaned metrics extracted from the file.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file format is invalid or data cannot be parsed.

  Examples
  --------
  .. code-block:: python

    import HMB.StatisticalAnalysisHelper as sah

    history, names, metrics = sah.ExtractDataFromSummaryFile("path/to/your/summaryFile.csv")
    print("Names:", names)
    print("Metrics:", metrics)
    for record in history:
      print(record)
  '''

  # Read all lines from the input file.
  with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()

  # Extract names from the first line and split by commas.
  names = lines[0].strip().split(",")

  # Strip whitespace and remove empty strings from names list.
  names = [name.strip() for name in names if (name.strip() != "")]

  # Remove non-ASCII characters from each name to ensure clean text formatting.
  names = [name.encode("ascii", "ignore").decode("utf-8") for name in names]

  # Extract metrics from the second line and split by commas.
  metrics = lines[1].strip().split(",")

  # Strip whitespace and remove empty strings from metrics list.
  metrics = [metric.strip() for metric in metrics if (metric.strip() != "")]

  # Truncate metrics list so that its length matches the number of metrics per name.
  metrics = metrics[:int(len(metrics) / len(names))]

  # Remove non-ASCII characters from each metric to ensure clean text formatting.
  metrics = [metric.encode("ascii", "ignore").decode("utf-8") for metric in metrics]

  # Initialize an empty list to store parsed numerical data from trials.
  data = []

  # Process each line starting from the third line (index 2), which contains trial data.
  for line in lines[2:]:
    values = line.strip().split(",")
    values = [float(value.strip()) for value in values if (value.strip() != "")]
    data.append(values)

  # Convert the list of trial data into a NumPy array for easier manipulation.
  data = np.array(data)

  # Initialize an empty list to store structured history records.
  history = []

  # Build a dictionary for each name containing their metrics with trials and mean values.
  for i in range(len(names)):
    record = {}
    for j in range(len(metrics)):
      # Extract all values for the current metric of the current name across all trials.
      values = data[:, i * len(metrics) + j].tolist()

      # Store the trials and their mean under the metric name.
      record[metrics[j]] = {
        "Trials": values,
        "Mean"  : np.mean(values),
      }
    history.append(record)

  # Return the structured history, cleaned names, and cleaned metrics.
  return history, names, metrics


def PlotDistributionEDA(
    df,  # DataFrame to analyze.
    baseDir,  # Base directory to save the plots.
    figsize=(18, 14),  # Figure size for subplots.
    maxUnique=100,  # Maximum number of unique values to include in the plots.
    minUnique=2,  # Minimum number of unique values to include in the plots.
    dpi=720,  # Resolution for saved images (DPI).
    keyword="X",  # Keyword to filter columns by name.
    maxUniqueLabels=10,  # Maximum number of unique labels to include in the plots.
):
  r'''
  Perform exploratory data analysis (EDA) by plotting the distributions of columns in a DataFrame.

  This function automatically generates and saves distribution plots (histograms for numeric columns,
  bar plots for categorical columns) for each column in the provided DataFrame that meets the criteria
  for unique value counts and data type. It is useful for quickly visualizing the spread, modality,
  and frequency of values in each column, and for identifying potential issues such as outliers or
  highly imbalanced categories.

  Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    baseDir (str): Directory where the plots will be saved.
    figsize (tuple, optional): Figure size for the subplots (default: (18, 14)).
    maxUnique (int, optional): Maximum number of unique values a column can have to be included (default: 100).
    minUnique (int, optional): Minimum number of unique values a column must have to be included (default: 2).
    dpi (int, optional): Resolution (dots per inch) for saved images (default: 720).
    keyword (str, optional): Keyword to include in the saved filenames (default: "X").
    maxUniqueLabels (int, optional): Maximum number of unique labels to show on the x-axis (default: 10).

  Notes:
    - Numeric columns are plotted as histograms.
    - Non-numeric (categorical) columns are plotted as bar plots.
    - Only columns with a number of unique values between minUnique and maxUnique (and <= maxUniqueLabels)
      are included.
    - The function is intended for quick EDA and may not be suitable for very large DataFrames
      or columns with extremely high cardinality.

  Examples
  --------
  .. code-block:: python

    import pandas as pd
    import HMB.StatisticalAnalysisHelper as sah

    # This will generate and save EDA distribution plots for both numeric and non-numeric columns
    # in the DataFrame "df" to the "plots" directory, with filenames containing "MyData".
    df = pd.read_csv("path/to/your/data.csv")
    sah.PlotDistributionEDA(
      df,
      baseDir="paths/to/your/plots",
      figsize=(20, 15),
      maxUnique=50,
      minUnique=2,
      dpi=300,
      keyword="MyData",
      maxUniqueLabels=15
    )
  '''

  def _PlotColumnsByType(
      df,  # DataFrame to analyze.
      baseDir,  # Base directory to save the plots.
      numeric=True,  # Whether to plot numeric columns (True) or non-numeric columns (False).
      figsize=(18, 14),  # Figure size for subplots.
      maxUnique=100,  # Maximum number of unique values to include in the plots.
      minUnique=2,  # Minimum number of unique values to include in the plots.
      dpi=720,  # Resolution for saved images (DPI).
      keyword="X",  # Keyword to filter columns by name.
      maxUniqueLabels=10,  # Maximum number of unique labels to include in the plots.
  ):
    r'''
    Dynamic function to plot the distribution of columns in a DataFrame.
    This function filters columns based on their data type and unique value count,
    then plots histograms for each column in subplots.
    It saves the plots in the specified base directory.
    The function can plot either numeric or non-numeric columns based on the `numeric` parameter.
    The plots are saved as a PDF file in the specified base directory.
    The function also allows customization of the figure size, number of rows,
    maximum and minimum unique values, number of bins for histograms, and DPI for the saved
    images.

    Parameters:
      df (pandas.DataFrame): The DataFrame to analyze.
      baseDir (str): Base directory to save the plots.
      numeric (bool): Whether to plot numeric columns (True) or non-numeric columns (False).
      figsize (tuple): Figure size for subplots.
      maxUnique (int): Maximum number of unique values to include in the plots.
      minUnique (int): Minimum number of unique values to include in the plots.
      bins (int): Number of bins for histograms.
      dpi (int): Resolution for saved images (DPI).
      keyword (str): Keyword to filter columns by name.
      maxUniqueLabels (int): Maximum number of unique labels to include in the plots.
    '''

    # Filter columns based on data type and unique value count.
    filteredColumns = [
      col for col in df.columns
      if (
          len(df[col].unique()) >= minUnique and
          len(df[col].unique()) <= maxUnique and
          (
            df[col].dtype in ["int64", "float64"]
            if numeric else df[col].dtype not in ["int64", "float64"]
          )
      )
    ]

    # Remove columns that contain too many unique values.
    # This is to avoid plotting columns with too many unique values.
    filteredColumns = [
      col for col in filteredColumns
      if (len(df[col].unique()) <= maxUniqueLabels)
    ]

    # Check if there are any columns to plot.
    if (not filteredColumns):
      print(f"No {'numeric' if numeric else 'non-numeric'} columns found matching criteria.")
      return

    if (len(filteredColumns) <= 5):
      noCols, noRows = len(filteredColumns), 1
    elif (len(filteredColumns) <= 50):
      noCols = 5
      # Calculate number of rows based on number of columns.
      noRows = int(np.ceil(len(filteredColumns) / noCols))
    else:
      raise ValueError(
        "Too many columns to plot. Please reduce the number of columns or increase the figure size."
      )

    print(f"Plotting {'numeric' if numeric else 'non-numeric'} columns:")
    print(f"Number of columns for subplots: {noCols}")
    print(f"Number of rows for subplots: {noRows}")
    print(f"Columns to plot: {filteredColumns}")

    if (figsize is None):
      # Make sure figsize is a dynamic tuple based on the number of columns and rows.
      # Adjust figure size based on the number of columns and rows.
      factor = 5  # Factor to adjust the figure size.
      figWidth = noCols * factor
      figHeight = noRows * factor
      figsize = (figWidth, figHeight)

    # Create figure for subplots.
    plt.figure(figsize=figsize)

    # Plot each column in a separate subplot.
    for i, col in enumerate(filteredColumns):
      plt.subplot(noRows, noCols, i + 1)

      # To use dynamic colors for each subplot, we can use a simple rule:
      # color = plt.cm.viridis(i / len(filteredColumns))  # Use colormap for dynamic colors.
      # color = plt.cm.tab10(i % 10)  # Use tab10 colormap for distinct colors.
      # color = plt.cm.get_cmap("Set1", len(filteredColumns))(i)  # Use Set1 colormap for distinct colors.
      # color = plt.cm.get_cmap("tab20", len(filteredColumns))(i)  # Use tab20 colormap for distinct colors.
      # color = plt.cm.get_cmap("Accent", len(filteredColumns))(i)  # Use Accent colormap for distinct colors.
      # More can be found at: https://matplotlib.org/stable/tutorials/colors/colormaps.html

      color = plt.cm.get_cmap("tab20", len(filteredColumns))(i)  # Use tab20 colormap for distinct colors.

      # Plot histogram
      sns.histplot(
        df[col],  # Column to plot distribution for.
        kde=False,  # Disable kernel density estimate.
        # bins=30,  # Number of bins for histogram.
        color=color,  # Use dynamic color for each subplot.
        stat="count",  # Use count for histogram.
        edgecolor="black",  # Edge color for histogram bars.
        alpha=0.7,  # Transparency of histogram bars.
        linewidth=2,  # Width of the edges of histogram bars.
        zorder=2,  # Set the z-order for the histogram bars.
      )

      # Trim the length of labels contents to 15 characters for better readability.
      labels = [str(label)[:15] for label in df[col].unique()]
      # Set x-axis labels to the unique values of the column.
      plt.xticks(
        ticks=np.arange(len(labels)),  # Set x-ticks to the range of unique values.
        labels=labels,  # Set x-tick labels to the unique values.
        rotation=45,  # Rotate x-axis labels for better readability.
        fontsize=12,  # Set font size for x-axis labels.
      )
      # Set the grid behind the bars for better visibility.
      # Add grid lines for better readability.
      plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
      plt.yticks(fontsize=12)  # Set y-axis tick font size.
      plt.title(col, fontsize=14, fontweight="bold")  # Set title for the subplot.
      plt.xlabel(None)  # Remove x-axis label for subplots.
      plt.ylabel(None)  # Remove x-axis label for subplots.
      plt.tight_layout()

    # Save plot as a PDF file in the specified base directory.
    filename = (
      f"{keyword}_EDA_Distribution_Numeric_Plots.pdf"
      if (numeric) else f"{keyword}_EDA_Distribution_NonNumeric_Plots.pdf"
    )
    plt.savefig(
      os.path.join(baseDir, filename),  # Save path for the plot.
      dpi=dpi,  # Resolution for the saved image.
      bbox_inches="tight",  # Adjust bounding box to fit the plot.
    )
    # Close the plot to free up memory.
    plt.close()  # Close the figure to free memory.
    print(f"Saved: {filename}")

  # Plot the distribution of numeric columns.
  print("Plotting numeric columns...")
  _PlotColumnsByType(
    df=df,  # DataFrame to analyze.
    baseDir=baseDir,  # Base directory to save the plots.
    numeric=True,  # Plot numeric columns.
    figsize=figsize,  # Figure size for subplots.
    maxUnique=maxUnique,  # Maximum number of unique values to include in the plots.
    minUnique=minUnique,  # Minimum number of unique values to include in the plots.
    dpi=dpi,  # Resolution for saved images (DPI).
    keyword=keyword,  # Keyword to filter columns by name.
    maxUniqueLabels=maxUniqueLabels,  # Maximum number of unique labels to include in the plots.
  )
  # Plot the distribution of non-numeric columns.
  print("Plotting non-numeric columns...")
  _PlotColumnsByType(
    df=df,  # DataFrame to analyze.
    baseDir=baseDir,  # Base directory to save the plots.
    numeric=False,  # Plot non-numeric columns.
    figsize=figsize,  # Figure size for subplots.
    maxUnique=maxUnique,  # Maximum number of unique values to include in the plots.
    minUnique=minUnique,  # Minimum number of unique values to include in the plots.
    dpi=dpi,  # Resolution for saved images (DPI).
    keyword=keyword,  # Keyword to filter columns by name.
    maxUniqueLabels=maxUniqueLabels,  # Maximum number of unique labels to include in the plots.
  )


def PlotMetrics(
    data, names, metrics,
    factor=5,  # Factor to multiply the default figure size.
    keyword="AllMetrics",  # Keyword to append to the filenames of the saved plots.
    dpi=1080,  # Dots per inch (resolution) of the saved plots.
    xTicksRotation=45,  # Rotation angle for x-axis tick labels.
    whichToPlot=[],  # List of plot types to generate.
    fontSize=14,  # Font size for the plots.
    showFigures=False,  # Whether to display the plots or not.
    storeInsideNewFolder=False,  # Whether to store the plots inside a new folder.
    newFolderName="PerformanceMetricsPlots",  # Name of the folder to store the plots.
    noOfPlotsPerRow=3,  # Number of plots per row in the subplot grid.
    cmap="viridis",  # Color map for the plots.
    differentColors=True,  # Whether to use different colors for different plots.
    fixedTicksColors=True,  # Whether to use fixed ticks colors for consistency across plots.
    fixedTicksColor="black",  # Color to use for fixed ticks if `fixedTicksColors` is True.
    extension=".pdf",  # File extension for saved plots.
):
  r'''
  Plot boxplots, violin plots, Q-Q plots, histograms, density plots, scatter plots,
  heatmaps, line plots, bar plots, pair plots, CDF plots, pie charts, and swarm plots
  for each metric in the data.

  Parameters:
    data (dict): Dictionary containing performance metrics and trial results.
    names (list): List of names or labels for each dataset.
    metrics (list): List of performance metrics to plot.
    factor (int, optional): Factor by which to multiply the default figure size.
    keyword (str, optional): Keyword to append to the filenames of the saved plots.
    dpi (int, optional): Dots per inch (resolution) of the saved plots.
    xTicksRotation (int, optional): Rotation angle for x-axis tick labels.
    whichToPlot (list, optional): List of plot types to generate.
    fontSize (int, optional): Font size for the plots.
    showFigures (bool, optional): Whether to display the plots or not.
    storeInsideNewFolder (bool, optional): Whether to store the plots inside a new folder.
    newFolderName (str, optional): Name of the folder to store the plots.
    noOfPlotsPerRow (int, optional): Number of plots per row in the subplot grid.
    cmap (str, optional): Color map to use for the plots (default: "viridis").
    differentColors (bool, optional): Whether to use different colors for different plots.
    fixedTicksColors (bool, optional): Whether to use fixed ticks colors for consistency across plots.
    fixedTicksColor (str, optional): Color to use for fixed ticks if fixedTicksColors is True.
    extension (str, optional): File extension for saved plots (default: ".pdf").

  Notes:
    - The function uses Seaborn and Matplotlib for plotting.
    - The plots are saved in the current working directory or inside a new folder if specified.
    - The function supports various plot types, which can be specified in the `whichToPlot` parameter.
    - The function automatically adjusts the figure size and layout based on the number of metrics.
    - The function allows customization of colors, font sizes, and other plot aesthetics.
    - Reduce the DPI value if you got an error related to memory while saving the plots. Error example:
      "_tkinter.TclError: not enough free memory for image buffer".
    - If you got this error "Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize",
      try to apply "matplotlib.use("Agg")" after importing "matplotlib".

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.StatisticalAnalysisHelper as sah

    # Example data: 3 datasets with 100 trials each and 2 metrics (accuracy and loss).
    data = {
      "Dataset1": {
        "accuracy": np.random.rand(100) * 0.2 + 0.8,  # Random accuracies between 0.8 and 1.0.
        "loss": np.random.rand(100) * 0.5 + 0.5,      # Random losses between 0.5 and 1.0.
      },
      "Dataset2": {
        "accuracy": np.random.rand(100) * 0.3 + 0.7,  # Random accuracies between 0.7 and 1.0.
        "loss": np.random.rand(100) * 0.4 + 0.6,      # Random losses between 0.6 and 1.0.
      },
      "Dataset3": {
        "accuracy": np.random.rand(100) * 0.25 + 0.75, # Random accuracies between 0.75 and 1.0.
        "loss": np.random.rand(100) * 0.45 + 0.55,     # Random losses between 0.55 and 1.0.
      },
    }
    names = list(data.keys())
    metrics = ["accuracy", "loss"]
    sah.PlotMetrics(
      data, names, metrics,
      factor=4,
      keyword="ModelPerformance",
      dpi=300,
      xTicksRotation=30,
      whichToPlot=["BoxPlots", "ViolinPlots", "Histograms", "ScatterPlots", "LinePlots"],
      fontSize=12,
      showFigures=True,
      storeInsideNewFolder=True,
      newFolderName="ModelPerformancePlots",
      noOfPlotsPerRow=2,
      cmap="plasma",
      differentColors=True,
      fixedTicksColors=True,
      fixedTicksColor="black",
      extension=".pdf",
    )
  '''

  UpdateMatplotlibSettings(fontSize=fontSize, figSize=(factor * 5, factor * 5))

  originalDir = os.getcwd()  # Store the current working directory.
  if (storeInsideNewFolder and newFolderName):
    # Create a new folder to store the plots if it does not exist.
    os.makedirs(newFolderName, exist_ok=True)
    # Change the current working directory to the new folder.
    os.chdir(newFolderName)

  # Initialize the number of metrics and datasets.
  noOfMetrics = len(metrics)
  noOfDatasets = len(data)

  # # Determine the number of rows and columns for subplots.
  # if (noOfMetrics <= 5):
  #   noRows = 1
  #   noCols = noOfMetrics
  # else:
  #   noRows = 2 if (noOfMetrics < 8) else (3 if (noOfMetrics < 12) else 4)
  #   noCols = (noOfMetrics // noRows + 1) if (noOfMetrics % noRows != 0) else (noOfMetrics // noRows)

  # Determine the number of rows and columns for subplots based on noOfPlotsPerRow.
  if (noOfMetrics <= noOfPlotsPerRow):
    noRows = 1
  elif (noOfMetrics <= noOfPlotsPerRow * 2):
    noRows = 2
  elif (noOfMetrics <= noOfPlotsPerRow * 3):
    noRows = 3
  else:
    noRows = 4

  if (noRows == 1):
    noCols = noOfMetrics
  else:
    noCols = (noOfMetrics // noRows + 1) if (noOfMetrics % noRows != 0) else (noOfMetrics // noRows)

  if (len(whichToPlot) == 0):
    whichToPlot = [
      # --- Distribution & Single Metric Analysis ---
      "Histograms",  # Frequency distribution of a single metric.
      "DensityPlots",  # Smoothed probability density (often with RugPlots).
      "BoxPlots",  # Summarize distribution (median, quartiles, outliers).
      "ViolinPlots",  # Combine density shape with box plot summary.
      "QQPlots",  # Compare distribution to a theoretical one (e.g., Normal).
      "CDFPlots",  # Cumulative distribution function.
      "ECDFPlots",  # Empirical cumulative distribution function.
      "SwarmPlots",  # Show individual data points, especially for small datasets.
      "StripPlots",  # Like swarm but allows overlap.
      "DotPlots",  # Dot plot for small counts.
      "StackedBarPlots",  # Stacked bar plot for group comparison.
      "StackedAreaPlots",  # Stacked area plot for cumulative trends.
      "Histogram2DPlots",  # 2D histogram for joint distribution of two metrics.
      "StepPlots",  # Step plot for discrete changes.

      # --- Comparative Analysis (Multiple Datasets/Metrics) ---
      "BarPlots",  # Compare aggregated values (e.g., means) across datasets.
      "LinePlots",  # Show trends over trials/iterations for each dataset.

      # --- Relationships & Correlations (Between Metrics) ---
      "ScatterPlots",  # Show relationship between two metrics.
      "HexbinPlots",  # 2D density plot for large datasets in scatter plots.
      "PairPlots",  # Matrix of scatter plots for multiple metrics (often includes CorrelationHeatmaps).
      "CorrelationHeatmaps",  # Standalone heatmap of correlation matrix (can be part of PairPlots).
      "BlandAltmanPlots",  # Compare agreement between two measurement methods/metrics.

      # --- Advanced Diagnostics (often related to others) ---
      "ResidualPlots",  # Diagnostics for regression (ScatterPlot related)
      "QQResidualPlots",  # Diagnostics for regression normality (ScatterPlot related)

      # --- Other/Advanced ---
      "ContourPlots",  # Show 3D relationships in 2D (requires specific data structure).
      "PieCharts",  # Show proportions (use sparingly, often better replaced by BarPlots).
      "RaincloudPlots",  # Raincloud plot: distribution + box/violin + raw data.
      "AndrewsCurves",  # Andrews curves for high-dimensional data.
      "ParallelCoordinates",  # Parallel coordinates for multi-metric comparison.
      "RadarPlots",  # Radar (spider) plots for profile comparison.
      "BoxenPlots",  # Boxen (letter value) plots for large data.
      "LollipopPlots",  # Lollipop plots for mean/median comparison.
      "SlopeCharts",  # Slope charts for before/after or paired data.
      "DumbbellPlots",  # Dumbbell plots for paired difference visualization.
      "TreemapPlots",  # Treemap for hierarchical metric visualization.
      "SunburstPlots",  # Sunburst for hierarchical metric visualization.
    ]

  # Get colors from the specified colormap.
  if (cmap is None):
    cmap = "Blues"
  elif (cmap.lower() == "random"):
    cmap = GetRandomCMAPalette()

  if (differentColors):
    # Get colors from the specified colormap.
    cmapColors = GetCmapColors(
      cmap,
      (len(names) * len(metrics)) * 10,
      darkColorsOnly=True,
      darknessThreshold=0.6,
    )
    print(f"Using colormap '{cmap}' with {len(cmapColors)} colors.")
  else:
    cmapColors = ["blue"] * (len(names) * len(metrics) * 10)
    print("Using single color 'blue' for all plots.")

  def GetTickColor(idx):
    if (fixedTicksColors):
      return fixedTicksColor
    else:
      return cmapColors[idx % len(cmapColors)]

  # =================================================================================================================
  # Residual Plots (vs Trial Index)
  # Residual plots are used to assess the fit of a model. Here, we fit a simple linear model
  # of the metric value against the trial index (1, 2, 3, ...) and plot the residuals
  # (differences between observed and predicted values) against the trial index.
  # This helps identify patterns over time that may indicate issues with the model, such as trends
  # or changing variance.
  # =================================================================================================================
  if ("ResidualPlots" in whichToPlot):
    try:
      print("Generating Residual plots...")
      # Counters for subplot grid based on total plots needed.
      totalPlots = noOfMetrics * noOfDatasets
      if (totalPlots > 0):
        resRows = int(np.ceil(np.sqrt(totalPlots)))
        resCols = int(np.ceil(totalPlots / resRows)) if (resRows > 0) else 1
        resRows = max(1, resRows)
        resCols = max(1, resCols)

        plt.figure(figsize=(factor * resCols, factor * resRows))
        plotIdx = 1

        for i, metric in enumerate(metrics):
          for j, dataset in enumerate(data):
            trialsData = np.array(dataset[metric]["Trials"])
            numTrials = len(trialsData)

            if (numTrials > 1):  # Need at least 2 points to fit a line.
              trialIndices = np.arange(1, numTrials + 1)  # 1, 2, 3, ...

              # Fit linear model: metricValue ~ trialIndex.
              X = sm.add_constant(trialIndices)  # Add intercept.
              try:
                model = sm.OLS(trialsData, X).fit()
                predictedVals = model.fittedvalues
                residuals = model.resid

                plt.subplot(resRows, resCols, plotIdx)
                plt.scatter(trialIndices, residuals, alpha=0.6)
                plt.axhline(0, color="red", linestyle="--", linewidth=1)
                plt.xlabel("Trial Index")
                plt.ylabel("Residuals")
                plt.title(
                  f"Residuals vs Trial Index\n{metric} - {names[j]}",
                  color=cmapColors[j]
                )
                plt.grid(True, alpha=0.5)
                plt.xticks(color=GetTickColor(j))
                plt.yticks(color=GetTickColor(j))
                plotIdx += 1
              except Exception as e:
                # Handle potential fitting errors (e.g., singular matrix if all X values are the same).
                plt.subplot(resRows, resCols, plotIdx)
                plt.text(
                  0.5, 0.5, f"Fit Error:\n{str(e)[:30]}...", ha="center", va="center",
                  transform=plt.gca().transAxes,
                  # fontsize=10
                )
                plt.title(f"Residuals vs Trial Index\n{metric} - {names[j]}\n(Fit Failed)", color="red")
                plt.xlabel("Trial Index")
                plt.ylabel("Residuals")
                plotIdx += 1
            else:
              # Not enough data points.
              plt.subplot(resRows, resCols, plotIdx)
              plt.text(0.5, 0.5, "N/A\n(< 2 trials)", ha="center", va="center", transform=plt.gca().transAxes)
              plt.title(f"Residuals vs Trial Index\n{metric} - {names[j]}\n(Insufficient Data)", color="orange")
              plt.xlabel("Trial Index")
              plt.ylabel("Residuals")
              plotIdx += 1

        plt.tight_layout(pad=1.0)  # Add padding to prevent title overlap.
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"ResidualPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()
        plt.clf()  # Clear the current figure.
      else:
        print("ResidualPlots: No data available to plot.")
    except Exception as e:
      print(f"Error generating Residual Plots: {str(e)}")
      print(traceback.format_exc())
      print("Skipping Residual Plots.")
      print("=" * 80)

  # ===============================================================================================================
  # Q-Q Residual Plots (vs Normal Distribution)
  # Q-Q (Quantile-Quantile) plots compare the distribution of residuals
  # (calculated from fitting metric value vs trial index) to a theoretical normal distribution.
  # If the points fall approximately along the reference line, it suggests that the residuals are
  # normally distributed. Deviations indicate departures from normality.
  # This checks the normality assumption often made in statistical models.
  # ===============================================================================================================
  try:
    if ("QQResidualPlots" in whichToPlot):
      print("Generating Q-Q Residual plots...")
      # Counters for subplot grid based on total plots needed.
      totalPlots = noOfMetrics * noOfDatasets
      if (totalPlots > 0):
        qqRows = int(np.ceil(np.sqrt(totalPlots)))
        qqCols = int(np.ceil(totalPlots / qqRows)) if (qqRows > 0) else 1
        qqRows = max(1, qqRows)
        qqCols = max(1, qqCols)

        plt.figure(figsize=(factor * qqCols, factor * qqRows))
        plotIdx = 1

        for i, metric in enumerate(metrics):
          for j, dataset in enumerate(data):
            trialsData = np.array(dataset[metric]["Trials"])
            numTrials = len(trialsData)

            if (numTrials > 1):  # Need at least 2 points to fit a line.
              trialIndices = np.arange(1, numTrials + 1)  # 1, 2, 3, ...

              # Fit linear model: metricValue ~ trialIndex to get residuals.
              X = sm.add_constant(trialIndices)  # Add intercept.
              try:
                model = sm.OLS(trialsData, X).fit()
                residuals = model.resid

                plt.subplot(qqRows, qqCols, plotIdx)
                # Use statsmodels qqplot
                sm.qqplot(residuals, line="s", ax=plt.gca())
                # Manually get the line for labeling if needed (qqplot usually handles it)
                # Get current axes lines to potentially modify labels
                # line = plt.gca().getLines()[-1] # Example to access line if needed.
                plt.title(
                  f"Q-Q Plot of Residuals\n{metric} - {names[j]}",
                  color=cmapColors[j]
                )
                plt.xticks(color=GetTickColor(j))
                plt.yticks(color=GetTickColor(j))
                # Labels are usually set by sm.qqplot, but ensure they are present.
                if (not plt.gca().get_ylabel()):
                  plt.xlabel("Theoretical Quantiles (Normal)")
                if (not plt.gca().get_ylabel()):
                  plt.ylabel("Sample Quantiles (Residuals)")

                plotIdx += 1
              except Exception as e:
                # Handle potential fitting errors.
                plt.subplot(qqRows, qqCols, plotIdx)
                plt.text(
                  0.5, 0.5,
                  f"Fit/Q-Q Error:\n{str(e)[:30]}...",
                  ha="center", va="center",
                  transform=plt.gca().transAxes,
                  # fontsize=10,
                )
                plt.title(f"Q-Q Plot of Residuals\n{metric} - {names[j]}\n(Fit/Q-Q Failed)", color="red")
                plotIdx += 1
                plt.xlabel("Theoretical Quantiles")  # Fallback labels.
                plt.ylabel("Sample Quantiles")
            else:
              # Not enough data points.
              plt.subplot(qqRows, qqCols, plotIdx)
              plt.text(0.5, 0.5, "N/A\n(< 2 trials)", ha="center", va="center",
                       transform=plt.gca().transAxes)
              plt.title(f"Q-Q Plot of Residuals\n{metric} - {names[j]}\n(Insufficient Data)", color="orange")
              plotIdx += 1
              plt.xlabel("Theoretical Quantiles")  # Fallback labels.
              plt.ylabel("Sample Quantiles")

        plt.tight_layout(pad=1.5)  # Increase padding due to longer titles.
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"QQResidualPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
      else:
        print("QQResidualPlots: No data available to plot.")
  except Exception as e:
    print(f"Error generating Q-Q Residual Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Q-Q Residual Plots.")
    print("=" * 80)

  # ===============================================================================================================
  # Bland-Altman Plots
  # Bland-Altman plots are used to visualize the agreement between two different measurement methods.
  # They plot the difference between the two methods against their average, helping to identify any systematic bias.
  # ===============================================================================================================
  try:
    if ("BlandAltmanPlots" in whichToPlot and len(metrics) >= 2):
      print("Generating Bland-Altman plots...")
      totalBaNeeded = sum(1 for i in range(len(metrics)) for j in range(len(metrics)) if (i < j))
      if (totalBaNeeded > 0):
        baRows = int(np.ceil(np.sqrt(totalBaNeeded)))
        baCols = int(np.ceil(totalBaNeeded / baRows)) if (baRows > 0) else 1
        for k, dataset in enumerate(data):
          plt.figure(figsize=(factor * baCols, factor * baRows))
          plotIndex = 1
          for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
              if (i < j):
                xVals = np.array(dataset[metric1]["Trials"])
                yVals = np.array(dataset[metric2]["Trials"])
                if (len(xVals) == len(yVals) and len(xVals) > 0):
                  meanVals = (xVals + yVals) / 2
                  diffVals = xVals - yVals
                  meanDiff = np.mean(diffVals)
                  stdDiff = np.std(diffVals, ddof=1)  # Sample standard deviation

                  plt.subplot(baRows, baCols, plotIndex)
                  plt.scatter(meanVals, diffVals, alpha=0.6)
                  plt.axhline(meanDiff, color="red", linestyle="-", label=f"Mean Diff: {meanDiff:.4f}")
                  plt.axhline(meanDiff + 1.96 * stdDiff, color="gray", linestyle="--", label="+1.96 SD")
                  plt.axhline(meanDiff - 1.96 * stdDiff, color="gray", linestyle="--", label="-1.96 SD")
                  plt.xlabel(f"Mean of {metric1} and {metric2}")
                  plt.ylabel(f"Difference ({metric1} - {metric2})")
                  plt.title(
                    f"Bland-Altman: {metric1} vs {metric2}\n({names[k]})",
                    color=cmapColors[k]
                  )
                  plt.xticks(color=GetTickColor(k))
                  plt.yticks(color=GetTickColor(k))
                  plt.legend()
                  plt.grid(True)
                  plotIndex += 1
          plt.tight_layout()
          keywordRep = keyword.replace("\n", "_")
          plt.savefig(f"BlandAltmanPlot_{keywordRep}_{names[k]}{extension}", dpi=dpi, bbox_inches="tight")
          if (showFigures):
            plt.show()
          plt.close()
          plt.clf()  # Clear the current figure.
    elif ("BlandAltmanPlots" in whichToPlot and len(metrics) < 2):
      print("BlandAltmanPlots: Not enough metrics to generate the plots.")
  except Exception as e:
    print(f"Error generating Bland-Altman Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Bland-Altman Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Histograms
  # Histograms are fundamental plots for visualizing the frequency distribution of data.
  # They help understand the shape, central tendency, and spread of the data.
  # ==============================================================================================================
  try:
    if ("Histograms" in whichToPlot):
      print("Generating histograms...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          plt.hist(
            dataset[metric]["Trials"],
            bins="auto",
            alpha=0.5,
            label=names[j],
            edgecolor="black",
            color=cmapColors[j]
          )
        color = cmapColors[i]
        plt.title(
          f"Histogram of {metric} Results",
          color=color
        )
        plt.xlabel("Performance Metric", color=GetTickColor(i))
        plt.ylabel("Frequency", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.legend()
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"Histogram_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Histograms: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Histograms.")
    print("=" * 80)

  # ==============================================================================================================
  # Boxplots
  # Boxplots are useful for visualizing the distribution of data based on a five-number summary:
  # minimum, first quartile (Q1), median, third quartile (Q3), and maximum. They are particularly
  # effective for identifying outliers and comparing distributions across different groups.
  # ==============================================================================================================
  try:
    if ("BoxPlots" in whichToPlot):
      print("Generating boxplots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        bplot = plt.boxplot(
          [el[metric]["Trials"] for el in data],
          vert=True,
          patch_artist=True,
          boxprops=dict(facecolor="lightblue"),
          medianprops=dict(color="black"),
          whiskerprops=dict(color="black"),
          capprops=dict(color="black"),
          flierprops=dict(
            marker="o",
            markersize=5,
            markerfacecolor="red",
            markeredgecolor="red"
          )
        )
        # Color each box with the cmap colors
        for patch, color in zip(bplot["boxes"], cmapColors):
          patch.set_facecolor(color)
        color = cmapColors[i]
        plt.title(
          f"Boxplot of {metric} Results",
          color=color
        )
        plt.xticks(
          list(range(1, noOfDatasets + 1)),
          names,
          rotation=xTicksRotation,
          color=GetTickColor(i)
        )
        plt.yticks(color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"BoxPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Box Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Box Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Violin Plots
  # Violin plots combine the benefits of boxplots and density plots. They show the distribution of data
  # across different groups, including the probability density of the data at different values. This makes
  # them ideal for comparing the shape and spread of distributions.
  # ==============================================================================================================
  try:
    if ("ViolinPlots" in whichToPlot):
      print("Generating violin plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        vplot = plt.violinplot(
          [el[metric]["Trials"] for el in data],
          showmeans=True,
          showmedians=True
        )
        # Color each violin with the cmap colors
        for idx, body in enumerate(vplot["bodies"]):
          body.set_facecolor(cmapColors[idx])
          body.set_edgecolor("black")
          body.set_alpha(0.7)
        color = cmapColors[i]
        plt.title(
          f"Violin Plot of {metric} Results",
          color=color
        )
        plt.xticks(list(range(1, noOfDatasets + 1)), names, rotation=xTicksRotation, color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"ViolinPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Violin Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Violin Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Q-Q Plots
  # Q-Q (Quantile-Quantile) plots are used to assess whether a dataset follows a particular distribution,
  # often the normal distribution. They compare the quantiles of the dataset to the quantiles of a theoretical
  # distribution, helping to identify deviations from normality.
  # ==============================================================================================================
  try:
    if ("QQPlots" in whichToPlot):
      print("Generating Q-Q plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          stats.probplot(dataset[metric]["Trials"], dist="norm", plot=plt)
        color = cmapColors[i]
        plt.title(
          f"Q-Q Plot of {metric} Results",
          color=color
        )
        plt.xlabel("Theoretical Quantiles", color=GetTickColor(i))
        plt.ylabel("Sample Quantiles", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"QQPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Q-Q Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Q-Q Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Density Plots (KDE)
  # Density plots are a powerful visualization tool that provides insights into the distribution of data,
  # particularly in understanding the shape, central tendency, and spread of the data. They are useful for
  # identifying peaks, valleys, and overlaps in distributions.
  # ==============================================================================================================
  try:
    if ("DensityPlots" in whichToPlot):
      print("Generating density plots...")
      plt.figure(figsize=(8 * noCols, 8 * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          sns.kdeplot(
            dataset[metric]["Trials"],
            label=names[j],
            fill=True,
            color=cmapColors[j]
          )
          sns.rugplot(
            dataset[metric]["Trials"],
            height=0.05,
            alpha=0.5,
            color=cmapColors[j]
          )
        color = cmapColors[i]
        plt.title(
          f"Density Plot of {metric} Results",
          color=color
        )
        plt.xlabel("Performance Metric", color=GetTickColor(i))
        plt.ylabel("Density", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.legend()
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"DensityPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Density Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Density Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Scatter Plots
  # Scatter plots are used to visualize the relationship between two variables. They are ideal for identifying
  # correlations, trends, and outliers in paired data.
  # ==============================================================================================================
  try:
    if ("ScatterPlots" in whichToPlot):
      print("Generating scatter plots...")
      # --- Calculate the number of unique pairs to determine subplot grid ---
      uniquePairs = [(i, j) for i in range(len(metrics)) for j in range(len(metrics)) if (i < j)]
      numPairs = len(uniquePairs)

      if (numPairs > 0):
        # Determine subplot grid for the number of unique pairs.
        if (numPairs <= 5):
          spRows, spCols = 1, numPairs
        else:
          spRows = 2 if (numPairs < 8) else (3 if (numPairs < 12) else 4)
          spCols = (numPairs // spRows + 1) if (numPairs % spRows != 0) else (numPairs // spRows)
        spRows = max(1, spRows)  # Ensure at least 1 row.
        spCols = max(1, spCols)  # Ensure at least 1 column.

        plt.figure(figsize=(factor * spCols, factor * spRows))
        # --- Loop through unique pairs and create subplots ---
        for plotIdx, (i, j) in enumerate(uniquePairs):
          metric1 = metrics[i]
          metric2 = metrics[j]
          plt.subplot(spRows, spCols, plotIdx + 1)  # Correct subplot index.

          # --- Plot data for each dataset ---
          for k, dataset in enumerate(data):
            xVals = np.array(dataset[metric1]["Trials"])
            yVals = np.array(dataset[metric2]["Trials"])
            if (len(xVals) == len(yVals)):
              plt.scatter(
                xVals, yVals,
                label=names[k],
                color=cmapColors[k]
              )
          color = cmapColors[plotIdx]
          plt.title(
            f"Scatter Plot: {metric1} vs {metric2}",
            color=color
          )
          plt.xlabel(metric1, color=GetTickColor(i))
          plt.ylabel(metric2, color=GetTickColor(j))
          plt.xticks(color=GetTickColor(i))
          plt.yticks(color=GetTickColor(j))
          if (noOfDatasets > 1):
            plt.legend()

        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"ScatterPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()
        plt.clf()  # Clear the current figure.
      else:
        print("ScatterPlots: Not enough metrics to generate pairs for plotting.")
  except Exception as e:
    print(f"Error generating Scatter Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Scatter Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Line Plots (Trend Analysis)
  # Line plots are used to visualize trends over time or across ordered categories. They are particularly
  # useful for showing changes in metrics over trials or iterations.
  # ==============================================================================================================
  try:
    if ("LinePlots" in whichToPlot):
      print("Generating line plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          plt.plot(
            dataset[metric]["Trials"],
            label=names[j],
            color=cmapColors[j]
          )
        color = cmapColors[i]
        plt.title(
          f"Line Plot of {metric} Results",
          color=color
        )
        plt.xlabel("Trial", color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.legend()
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"LinePlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Line Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Line Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Bar Plots
  # Bar plots are used to compare the mean or median of metrics across different groups. They are effective
  # for summarizing and comparing aggregated data.
  # ==============================================================================================================
  try:
    if ("BarPlots" in whichToPlot):
      print("Generating bar plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        means = [np.mean(dataset[metric]["Trials"]) for dataset in data]
        plt.bar(names, means, color=cmapColors)
        color = cmapColors[i]
        plt.title(
          f"Bar Plot of {metric} Results",
          color=color
        )
        plt.xlabel("Dataset", color=GetTickColor(i))
        plt.ylabel("Mean Performance Metric", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"BarPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Bar Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Bar Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Correlation Heatmaps
  # Correlation heatmaps visualize the correlation matrix of multiple metrics. They are useful for identifying
  # relationships and dependencies between different metrics.
  # ==============================================================================================================
  try:
    if ("CorrelationHeatmaps" in whichToPlot and len(metrics) > 1):
      print("Generating correlation heatmaps...")
      for i, dataset in enumerate(data):
        plt.figure(figsize=(factor * noCols, factor * noRows))
        df = pd.DataFrame({metric: dataset[metric]["Trials"] for metric in metrics})
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", square=True)
        plt.title(f"Correlation Heatmap for {names[i]}")
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"CorrelationHeatmap_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
    elif ("CorrelationHeatmaps" in whichToPlot and len(metrics) <= 1):
      print("CorrelationHeatmaps: Not enough metrics to generate the plots.")
  except Exception as e:
    print(f"Error generating Correlation Heatmaps: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Correlation Heatmaps.")
    print("=" * 80)

  # ==============================================================================================================
  # Pair Plots (Pairwise Relationships)
  # Pair plots are used to visualize pairwise relationships between multiple variables. They are useful for
  # identifying correlations and patterns across multiple metrics.
  # ==============================================================================================================
  try:
    if ("PairPlots" in whichToPlot):
      print("Generating pair plots...")
      for i, dataset in enumerate(data):
        df = pd.DataFrame({metric: dataset[metric]["Trials"] for metric in metrics})
        sns.pairplot(df)
        plt.suptitle(f"Pair Plot for {names[i]}", y=1.02)
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"PairPlot_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Pair Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Pair Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # CDF Plots
  # CDF (Cumulative Distribution Function) plots show the cumulative probability of a variable. They are useful
  # for understanding the distribution and comparing the spread of data across different groups.
  # ==============================================================================================================
  try:
    if ("CDFPlots" in whichToPlot):
      print("Generating CDF plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          sortedData = np.sort(dataset[metric]["Trials"])
          yVals = np.arange(len(sortedData)) / float(len(sortedData) - 1)
          plt.plot(sortedData, yVals, label=names[j])
        plt.title(
          f"CDF of {metric} Results",
          color=cmapColors[i]
        )
        plt.xlabel("Performance Metric", color=GetTickColor(i))
        plt.ylabel("Cumulative Probability", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.legend()
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"CDF_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating CDF Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping CDF Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # ECDF Plots
  # ECDF (Empirical Cumulative Distribution Function) plots are similar to CDF plots but focus on the empirical
  # distribution of the data. They are useful for visualizing the distribution of individual datasets.
  # ==============================================================================================================
  try:
    if ("ECDFPlots" in whichToPlot):
      print("Generating ECDF plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          sortedData = np.sort(dataset[metric]["Trials"])
          yVals = np.arange(1, len(sortedData) + 1) / float(len(sortedData))
          plt.step(sortedData, yVals, label=names[j], where="post")
        plt.title(
          f"ECDF of {metric} Results",
          color=cmapColors[i]
        )
        plt.xlabel("Performance Metric", color=GetTickColor(i))
        plt.ylabel("Empirical Cumulative Probability", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.legend()
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"ECDF_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating ECDF Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping ECDF Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Swarm Plots
  # Swarm plots are used to visualize individual data points and their distribution. They are useful for
  # showing the density of data points and identifying patterns or clusters.
  # ==============================================================================================================
  try:
    if ("SwarmPlots" in whichToPlot):
      print("Generating swarm plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        sns.swarmplot(
          data=[dataset[metric]["Trials"] for dataset in data],
          palette=cmapColors[:len(names)],  # Slice palette to match groups.
          dodge=True,
          size=3,  # Smaller markers to reduce placement failures.
          edgecolor="0.5",  # Numeric gray to avoid grayscale FutureWarning.
          linewidth=0.4,
          alpha=0.8  # Transparency for better visibility.
        )
        color = cmapColors[i]
        plt.title(f"Swarm Plot of {metric} Results", color=color)
        plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"SwarmPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Swarm Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Swarm Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Pie Charts
  # Pie charts are used to visualize the proportion of different categories within a dataset.
  # They are effective for showing the relative sizes of different groups or categories.
  # In this context, it shows the relative average values of different metrics for each dataset/system.
  # ==============================================================================================================
  try:
    if ("PieCharts" in whichToPlot):
      print("Generating pie charts...")
      for i, dataset in enumerate(data):
        # Calculate the mean value for each metric within the current dataset.
        sizes = [np.mean(dataset[metric]["Trials"]) for metric in metrics]

        # Handle potential issues with data for pie charts.
        # Check for non-positive values which can cause issues or misleading representations.
        if (any(s <= 0 for s in sizes)):
          print(f"Warning: Non-positive values found for dataset '{names[i]}'. Pie chart might be misleading or empty.")
          # Optionally, you could filter out non-positive values or adjust them.
          # For now, we'll proceed but matplotlib will handle <=0 values by not showing them or showing warnings.

        # Ensure there is data to plot.
        if (sum(sizes) <= 0):
          print(f"Warning: Sum of sizes is zero or negative for dataset '{names[i]}'. Skipping Pie Chart.")
          continue  # Skip plotting this dataset.

        # Create the figure with the specified size
        plt.figure(figsize=(factor, factor))

        # Create the pie chart.
        # autopct displays percentages, startangle rotates the start, textprops adjusts label size.
        wedges, texts, autotexts = plt.pie(
          sizes,
          labels=metrics,
          autopct="%1.1f%%",
          startangle=140,
          # textprops={"fontsize": max(8, fontSize - 2)},
          textprops={"fontsize": fontSize},
          colors=GetCmapColors(cmap, len(metrics))
        )

        # Set the title for the current dataset.
        plt.title(
          f"Pie Chart of Metrics for {names[i]}",
          # fontsize=fontSize,
          color=cmapColors[i]
        )
        plt.xlabel("", color=GetTickColor(i))
        plt.ylabel("", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis("equal")

        # Improve layout to prevent label clipping (though pie charts can be tricky).
        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"PieChart_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Pie Charts: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Pie Charts.")
    print("=" * 80)

  # ==============================================================================================================
  # Area Plots
  # Area plots are used to visualize the cumulative total of a metric over time or across categories
  # They are effective for showing trends and the relative contribution of different groups.
  # ==============================================================================================================
  try:
    if ("AreaPlots" in whichToPlot):
      print("Generating area plots...")
      plt.figure(figsize=(factor * 2, factor * 2))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          plt.fill_between(
            range(len(dataset[metric]["Trials"])),
            dataset[metric]["Trials"],
            label=names[j],
            alpha=0.5
          )
        plt.title(
          f"Area Plot of {metric} Results",
          color=cmapColors[i]
        )
        plt.xlabel("Trial", color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.legend()
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"AreaPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Area Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Area Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Hexbin Plots
  # Hexbin plots are used to visualize the density of data points in a two-dimensional space
  # They are effective for showing the distribution of data points and identifying clusters.
  # ==============================================================================================================
  try:
    if ("HexbinPlots" in whichToPlot):
      # Print a message indicating the start of hexbin plot generation.
      print("Generating hexbin plots...")
      # Count the number of unique pairs for subplots.
      uniquePairs = [(i, j) for i in range(len(metrics)) for j in range(len(metrics)) if (i < j)]
      numPairs = len(uniquePairs)

      if (numPairs > 0):
        # Determine subplot grid for pairs.
        if (numPairs <= 5):
          hbRows, hbCols = 1, numPairs
        else:
          hbRows = 2 if (numPairs < 8) else (3 if (numPairs < 12) else 4)
          hbCols = (numPairs // hbRows + 1) if (numPairs % hbRows != 0) else (numPairs // hbRows)

        # Create a new figure for the hexbin plots.
        plt.figure(figsize=(factor * hbCols, factor * hbRows))

        # Loop through each unique pair and create subplots.
        for plotIdx, (i, j) in enumerate(uniquePairs):
          metric1 = metrics[i]  # X-axis.
          metric2 = metrics[j]  # Y-axis.
          plt.subplot(hbRows, hbCols, plotIdx + 1)

          # Combine data from all datasets for this pair to create a single hexbin.
          # Or create overlaid hexbins per dataset (less common but possible).
          # Here, we will combine all data points for the pair across datasets.
          allXVals = []
          allYVals = []
          for dataset in data:
            allXVals.extend(dataset[metric1]["Trials"])
            allYVals.extend(dataset[metric2]["Trials"])

          if (len(allXVals) > 0 and len(allYVals) > 0):
            hb = plt.hexbin(
              allXVals,
              allYVals,
              gridsize=30,  # Number of hexagons in the x-direction.
              cmap=cmap,  # Colormap for the hexagons.
              mincnt=1  # Only show hexagons with at least one count.
            )
            plt.title(
              f"Hexbin Plot: {metric1} vs {metric2}",
              color=cmapColors[plotIdx]
            )
            plt.xlabel(metric1, color=GetTickColor(i))
            plt.ylabel(metric2, color=GetTickColor(j))
            plt.xticks(color=GetTickColor(i))
            plt.yticks(color=GetTickColor(j))
            plt.colorbar(hb, ax=plt.gca(), label="Count")
          else:
            plt.title(f"Hexbin Plot: {metric1} vs {metric2}\n(No Data)", color="orange")
            plt.xlabel(metric1, color=GetTickColor(i))
            plt.ylabel(metric2, color=GetTickColor(j))
            plt.xticks(color=GetTickColor(i))
            plt.yticks(color=GetTickColor(j))

        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"HexbinPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
      else:
        print("HexbinPlots: Not enough metrics to generate pairs for plotting.")
  except Exception as e:
    print(f"Error generating Hexbin Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Hexbin Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Contour Plots
  # Contour plots are used to visualize three-dimensional data in two dimensions by plotting contour lines.
  # They are effective for showing the relationship between two variables and a third variable represented
  # by contour lines.
  # ==============================================================================================================
  try:
    if ("ContourPlots" in whichToPlot and noOfMetrics >= 2):
      print("Generating contour plots...")
      for i, dataset in enumerate(data):
        plt.figure(figsize=(factor * 5, factor * 5))
        x = np.array(dataset[metrics[0]]["Trials"])
        y = np.array(dataset[metrics[1]]["Trials"])
        z = np.array(dataset[metrics[2]]["Trials"]) if (len(metrics) > 2) else np.zeros_like(x)
        plt.tricontourf(x, y, z, levels=14, cmap=cmap)
        plt.colorbar(label="Metric Value")
        plt.title(f"Contour Plot for {names[i]}")
        plt.xlabel(metrics[0], color=GetTickColor(i))
        plt.ylabel(metrics[1], color=GetTickColor(i))
        plt.xticks(color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"ContourPlot_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
    elif ("ContourPlots" in whichToPlot and noOfMetrics < 2):
      print("ContourPlots: Not enough metrics to generate the plots.")
  except Exception as e:
    print(f"Error generating Contour Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Contour Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Strip Plots
  # Strip plots show individual data points for each group, allowing overlap.
  # ==============================================================================================================
  try:
    if ("StripPlots" in whichToPlot):
      print("Generating strip plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        sns.stripplot(
          data=[dataset[metric]["Trials"] for dataset in data],
          jitter=True,  # Add jitter to avoid overlap.
          palette=cmapColors[:len(names)],  # Slice palette to match groups.
          dodge=True,
          size=3,  # Smaller markers to reduce placement failures.
          edgecolor="0.5",  # Numeric gray to avoid grayscale FutureWarning.
          linewidth=0.4,
          alpha=0.8,  # Transparency for better visibility.
        )
        color = cmapColors[i]
        plt.title(f"Strip Plot of {metric} Results", color=color)
        plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"StripPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Strip Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Strip Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Dot Plots
  # Dot plots show dots for each data point, grouped by category.
  # ==============================================================================================================
  try:
    if ("DotPlots" in whichToPlot):
      print("Generating dot plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        for j, dataset in enumerate(data):
          y = dataset[metric]["Trials"]
          x = np.full_like(y, j, dtype=float) + np.random.uniform(-0.1, 0.1, size=len(y))
          plt.plot(x, y, "o", color=cmapColors[j], alpha=0.7, label=names[j] if i == 0 else None)
        color = cmapColors[i]
        plt.title(f"Dot Plot of {metric} Results", color=color)
        plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"DotPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Dot Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Dot Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Stacked Bar Plots
  # Stacked bar plots show the mean values of each metric for each dataset, stacked.
  # ==============================================================================================================
  try:
    if ("StackedBarPlots" in whichToPlot):
      print("Generating stacked bar plots...")
      means = np.array([[np.mean(dataset[metric]["Trials"]) for metric in metrics] for dataset in data])
      plt.figure(figsize=(factor * 2, factor * 2))
      bottom = np.zeros(len(data))
      for i, metric in enumerate(metrics):
        plt.bar(names, means[:, i], bottom=bottom, label=metric, color=cmapColors[i])
        bottom += means[:, i]
      plt.title("Stacked Bar Plot of Metrics", color="black")
      # Set the x-axis label for the stacked bar plot.
      plt.xlabel("Dataset", color="black")
      # Set the y-axis label for the stacked bar plot.
      plt.ylabel("Cumulative Mean", color="black")
      # Set the x-tick colors for the stacked bar plot.
      plt.xticks(color="black")
      # Set the y-tick colors for the stacked bar plot.
      plt.yticks(color="black")
      # Add legend to the stacked bar plot.
      plt.legend()
      # Adjust the layout for the stacked bar plot.
      plt.tight_layout()
      # Save the stacked bar plot as a PDF file.
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"StackedBarPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Stacked Bar Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Stacked Bar Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Stacked Area Plots
  # Stacked area plots show cumulative trends for each group/metric over trials.
  # ==============================================================================================================
  try:
    if ("StackedAreaPlots" in whichToPlot):
      # Print a message indicating the start of stacked area plot generation.
      print("Generating stacked area plots...")
      # Loop through each dataset to create a stacked area plot.
      for i, dataset in enumerate(data):
        # Create a new figure for the stacked area plot.
        plt.figure(figsize=(factor * 2, factor * 2))
        # Get the length of each metric's trials in the current dataset.
        trialLens = [len(dataset[metric]["Trials"]) for metric in metrics]
        # Find the minimum length among all metrics' trials.
        minLen = min(trialLens)
        # Create an array of metric values up to the minimum length for stacking.
        arr = np.array([dataset[metric]["Trials"][:minLen] for metric in metrics])
        plt.stackplot(
          range(minLen), arr,
          labels=metrics,
          colors=cmapColors[:len(metrics)]
        )
        plt.title(f"Stacked Area Plot for {names[i]}", color=cmapColors[i])
        # Set the x-axis label for the stacked area plot.
        plt.xlabel("Trial", color=GetTickColor(i))
        # Set the y-axis label for the stacked area plot.
        plt.ylabel("Metric Value (Cumulative)", color=GetTickColor(i))
        # Set the x-tick colors for the stacked area plot.
        plt.xticks(color=GetTickColor(i))
        # Set the y-tick colors for the stacked area plot.
        plt.yticks(color=GetTickColor(i))
        # Add legend to the stacked area plot.
        plt.legend()
        # Adjust the layout for the stacked area plot.
        plt.tight_layout()
        # Save the stacked area plot as a PDF file.
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"StackedAreaPlot_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Stacked Area Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Stacked Area Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Histogram2DPlots
  # 2D histograms for all unique pairs of metrics.
  # ==============================================================================================================
  try:
    if ("Histogram2DPlots" in whichToPlot):
      # Print a message indicating the start of 2D histogram plot generation.
      print("Generating 2D histogram plots...")
      # Create a list of all unique pairs of metrics (i, j) where i < j.
      uniquePairs = [(i, j) for i in range(len(metrics)) for j in range(len(metrics)) if (i < j)]
      # Count the number of unique pairs.
      numPairs = len(uniquePairs)
      # Check if there are any pairs to plot.
      if (numPairs > 0):
        # Determine subplot grid size based on the number of pairs.
        if (numPairs <= 5):
          h2dRows, h2dCols = 1, numPairs
        else:
          h2dRows = 2 if (numPairs < 8) else (3 if (numPairs < 12) else 4)
          h2dCols = (numPairs // h2dRows + 1) if (numPairs % h2dRows != 0) else (numPairs // h2dRows)
        # Create a new figure for the 2D histograms.
        plt.figure(figsize=(factor * h2dCols, factor * h2dRows))
        # Loop through each unique pair and create a subplot for each.
        for plotIdx, (i, j) in enumerate(uniquePairs):
          # Get the metric names for the current pair.
          metric1 = metrics[i]
          metric2 = metrics[j]
          # Create a subplot for the current pair.
          plt.subplot(h2dRows, h2dCols, plotIdx + 1)
          # Initialize lists to hold all x and y values for the current pair.
          allXVals = []
          allYVals = []
          # Loop through each dataset to collect values for the current pair.
          for dataset in data:
            allXVals.extend(dataset[metric1]["Trials"])
            allYVals.extend(dataset[metric2]["Trials"])
          # Plot the 2D histogram for the current pair.
          plt.hist2d(allXVals, allYVals, bins=30, cmap=cmap)
          # Set the title for the current subplot.
          plt.title(f"2D Histogram: {metric1} vs {metric2}", color=cmapColors[plotIdx])
          # Set the x-axis label for the current subplot.
          plt.xlabel(metric1, color=GetTickColor(i))
          # Set the y-axis label for the current subplot.
          plt.ylabel(metric2, color=GetTickColor(j))
          # Set the x-tick colors for the current subplot.
          plt.xticks(color=GetTickColor(i))
          # Set the y-tick colors for the current subplot.
          plt.yticks(color=GetTickColor(j))
          # Add a colorbar to the current subplot.
          plt.colorbar(label="Count")
        # Adjust the layout to prevent overlap.
        plt.tight_layout()
        # Save the 2D histogram plot as a PDF file.
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"Histogram2DPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        # Show the figure if requested.
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
      else:
        print("Histogram2DPlots: Not enough metrics to generate pairs for plotting.")
  except Exception as e:
    print(f"Error generating 2D Histogram Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping 2D Histogram Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # StepPlots
  # Step plots for all unique pairs of metrics.
  # ==============================================================================================================
  try:
    if ("StepPlots" in whichToPlot):
      print("Generating step plots...")
      # Create a new figure for the step plots.
      plt.figure(figsize=(factor * noCols, factor * noRows))
      # Loop through each metric to create a subplot for each.
      for i, metric in enumerate(metrics):
        # Create a subplot for the current metric.
        plt.subplot(noRows, noCols, i + 1)
        # Loop through each dataset to plot the step plot for the current metric.
        for j, dataset in enumerate(data):
          # Plot the step plot for the current dataset and metric.
          plt.step(
            range(len(dataset[metric]["Trials"])),
            dataset[metric]["Trials"],
            label=names[j],
            where="mid",
            color=cmapColors[j]
          )
        # Get the color for the current metric.
        color = cmapColors[i]
        # Set the title for the current subplot.
        plt.title(f"Step Plot of {metric} Results", color=color)
        # Set the x-axis label for the current subplot.
        plt.xlabel("Trial", color=GetTickColor(i))
        # Set the y-axis label for the current subplot.
        plt.ylabel("Performance Metric", color=GetTickColor(i))
        # Set the x-tick colors for the current subplot.
        plt.xticks(color=GetTickColor(i))
        # Set the y-tick colors for the current subplot.
        plt.yticks(color=GetTickColor(i))
        # Add legend to the current subplot.
        plt.legend()
      # Adjust the layout to prevent overlap.
      plt.tight_layout()
      # Save the step plot as a PDF file.
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"StepPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Step Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Step Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Raincloud Plots
  # ==============================================================================================================
  try:
    if ("RaincloudPlots" in whichToPlot):
      print("Generating raincloud plots...")
      try:
        # Use Seaborn's violinplot + stripplot for a modern raincloud plot (ptitprince is not required)
        for i, metric in enumerate(metrics):
          plt.figure(figsize=(factor * 2, factor))
          allTrials = []
          allNames = []
          for j, dataset in enumerate(data):
            allTrials.extend(dataset[metric]["Trials"])
            allNames.extend([names[j]] * len(dataset[metric]["Trials"]))
          df = pd.DataFrame({"Metric": allTrials, "Group": allNames})
          # Violin plot (density)
          sns.violinplot(
            x="Group", y="Metric", data=df,
            hue="Group",
            palette=cmapColors[:len(names)],  # Use only needed colors.
            inner=None,  # No inner bars.
            linewidth=1,  # Outline width.
            cut=0, bw_method=0.2, alpha=0.7,
            legend=False
          )
          # Strip plot (raw data points).
          sns.stripplot(
            x="Group", y="Metric", data=df,
            palette=cmapColors[:len(names)],
            legend=False, hue="Group",
            dodge=False, jitter=True, alpha=0.5,
            size=4, edgecolor="0.5", linewidth=0.5
          )
          plt.title(f"Raincloud Plot of {metric}", color=cmapColors[i])
          plt.xlabel("Group", fontsize=fontSize)
          plt.ylabel("Metric", fontsize=fontSize)
          plt.xticks(color=GetTickColor(i))
          plt.yticks(color=GetTickColor(i))
          plt.tight_layout()
          keywordRep = keyword.replace("\n", "_")
          plt.savefig(f"RaincloudPlot_{metric}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
          if (showFigures):
            plt.show()
          plt.close()  # Close the figure to free memory.
          plt.clf()  # Clear the current figure.

        # Generate a combined raincloud plot for all metrics and datasets as subplots.
        plt.figure(figsize=(factor * noCols, factor * noRows))
        for i, metric in enumerate(metrics):
          plt.subplot(noRows, noCols, i + 1)
          allTrials = []
          allNames = []
          for j, dataset in enumerate(data):
            allTrials.extend(dataset[metric]["Trials"])
            allNames.extend([names[j]] * len(dataset[metric]["Trials"]))
          df = pd.DataFrame({"Metric": allTrials, "Group": allNames})
          sns.violinplot(
            x="Group", y="Metric", data=df,
            hue="Group",
            palette=cmapColors[:len(names)],
            inner=None,
            linewidth=1,
            cut=0, bw_method=0.2, alpha=0.7,
            legend=False
          )
          sns.stripplot(
            x="Group", y="Metric", data=df,
            palette=cmapColors[:len(names)],
            legend=False, hue="Group",
            dodge=False, jitter=True, alpha=0.5,
            size=4, edgecolor="0.5", linewidth=0.5
          )
          plt.title(f"Raincloud Plot of {metric}", color=cmapColors[i])
          plt.xlabel("Group", fontsize=fontSize)
          plt.ylabel("Metric", fontsize=fontSize)
          plt.xticks(color=GetTickColor(i), rotation=xTicksRotation)
          plt.yticks(color=GetTickColor(i))
        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"RaincloudPlot_Combined_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
      except Exception as e:
        print(f"RaincloudPlots: {e}")
  except Exception as e:
    print(f"Error generating Raincloud Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Raincloud Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Andrews Curves
  # ==============================================================================================================
  try:
    if ("AndrewsCurves" in whichToPlot):
      print("Generating Andrews curves...")
      for i, dataset in enumerate(data):
        df = pd.DataFrame({metric: dataset[metric]["Trials"] for metric in metrics})
        df["Group"] = names[i]
        plt.figure(figsize=(factor * 2, factor * 2))
        try:
          pd.plotting.andrews_curves(df, "Group", color=cmapColors[i])
          plt.title(f"Andrews Curves for {names[i]}", color=cmapColors[i])
          plt.tight_layout()
          keywordRep = keyword.replace("\n", "_")
          plt.savefig(f"AndrewsCurves_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
          if (showFigures):
            plt.show()
          plt.close()
          plt.clf()
        except Exception as e:
          print(f"AndrewsCurves: {e}")
      # Create a combined Andrews curves plot for all datasets.
      plt.figure(figsize=(factor * 2, factor * 2))
      combinedDF = pd.DataFrame()
      for i, dataset in enumerate(data):
        temp_df = pd.DataFrame({metric: dataset[metric]["Trials"] for metric in metrics})
        temp_df["Group"] = names[i]
        combinedDF = pd.concat([combinedDF, temp_df], ignore_index=True)
      try:
        pd.plotting.andrews_curves(combinedDF, "Group", color=cmapColors)
        plt.title("Combined Andrews Curves", color="black")
        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"AndrewsCurves_Combined_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
      except Exception as e:
        print(f"AndrewsCurves Combined: {e}")
  except Exception as e:
    print(f"Error generating Andrews Curves: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Andrews Curves.")
    print("=" * 80)

  # ==============================================================================================================
  # Parallel Coordinates
  # ==============================================================================================================
  try:
    if ("ParallelCoordinates" in whichToPlot):
      print("Generating parallel coordinates plots...")
      for i, dataset in enumerate(data):
        df = pd.DataFrame({metric: dataset[metric]["Trials"] for metric in metrics})
        df["Group"] = names[i]
        plt.figure(figsize=(factor * 2, factor * 2))
        try:
          pd.plotting.parallel_coordinates(df, "Group", color=[cmapColors[i]])
          plt.title(f"Parallel Coordinates for {names[i]}", color=cmapColors[i])
          plt.tight_layout()
          keywordRep = keyword.replace("\n", "_")
          plt.savefig(f"ParallelCoordinates_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
          if (showFigures):
            plt.show()
          plt.close()  # Close the figure to free memory.
          plt.clf()  # Clear the current figure.
        except Exception as e:
          print(f"ParallelCoordinates: {e}")
  except Exception as e:
    print(f"Error generating Parallel Coordinates: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Parallel Coordinates.")
    print("=" * 80)

  # ==============================================================================================================
  # Radar (Spider) Plots
  # ==============================================================================================================
  try:
    if ("RadarPlots" in whichToPlot):
      print("Generating radar plots...")
      for i, dataset in enumerate(data):
        values = [np.mean(dataset[metric]["Trials"]) for metric in metrics]
        values += values[:1]  # close the circle
        angles = np.linspace(0, 2 * np.pi, len(metrics) + 1, endpoint=True)
        plt.figure(figsize=(factor, factor))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, color=cmapColors[i], linewidth=2)
        ax.fill(angles, values, color=cmapColors[i], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        # Annotate each value on the radar plot
        for angle, value, label in zip(angles, values, metrics + [metrics[0]]):
          ax.text(
            angle, value + 0.02 * max(values), f"{value:.2f}",
            color=cmapColors[i],
            fontsize=fontSize,
            ha="center",
            va="center"
          )
        plt.title(f"Radar Plot for {names[i]}", color=cmapColors[i])
        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"RadarPlot_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()
        plt.clf()
      # Create a combined radar plot for all datasets.
      plt.figure(figsize=(factor * 1.75, factor * 1.75))
      ax = plt.subplot(111, polar=True)  # Create the polar axes ONCE
      for i, dataset in enumerate(data):
        values = [np.mean(dataset[metric]["Trials"]) for metric in metrics]
        values += values[:1]  # close the circle
        angles = np.linspace(0, 2 * np.pi, len(metrics) + 1, endpoint=True)
        ax.plot(angles, values, label=names[i], color=cmapColors[i], linewidth=2)
        ax.fill(angles, values, color=cmapColors[i], alpha=0.05)
        # Annotate each value on the combined radar plot
        for angle, value, label in zip(angles, values, metrics + [metrics[0]]):
          ax.text(
            angle + i * 0.15,  # Slightly offset angle for better visibility.
            value - 0.125 * max(values),
            f"{value:.4f}",
            color=cmapColors[i],
            fontsize=fontSize,
            ha="center",
            va="center"
          )
      ax.set_xticks(angles[:-1])
      ax.set_xticklabels(metrics)
      plt.title("Combined Radar Plot", color="black")
      plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"RadarPlot_Combined_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Radar Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Radar Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Boxen Plots
  # ==============================================================================================================
  try:
    if ("BoxenPlots" in whichToPlot):
      print("Generating boxen plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        sns.boxenplot(
          data=[dataset[metric]["Trials"] for dataset in data],
          palette=cmapColors[:len(names)],  # Slice palette to match groups.
          dodge=True,
          edgecolor="0.5",  # Numeric gray to avoid grayscale FutureWarning.
          linewidth=0.4,
          alpha=0.8,  # Transparency for better visibility.
        )
        plt.title(f"Boxen Plot of {metric} Results", color=cmapColors[i])
        plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
        plt.ylabel("Performance Metric", color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"BoxenPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Boxen Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Boxen Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Lollipop Plots
  # ==============================================================================================================
  try:
    if ("LollipopPlots" in whichToPlot):
      print("Generating lollipop plots...")
      plt.figure(figsize=(factor * noCols, factor * noRows))
      for i, metric in enumerate(metrics):
        plt.subplot(noRows, noCols, i + 1)
        means = [np.mean(dataset[metric]["Trials"]) for dataset in data]
        plt.stem(range(len(names)), means, basefmt=" ", linefmt="-", markerfmt="o")
        plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
        plt.title(f"Lollipop Plot of {metric} Means", color=cmapColors[i])
        plt.ylabel("Mean Performance Metric", color=GetTickColor(i))
        plt.yticks(color=GetTickColor(i))
      plt.tight_layout()
      keywordRep = keyword.replace("\n", "_")
      plt.savefig(f"LollipopPlot_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()  # Close the figure to free memory.
      plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Lollipop Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Lollipop Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Slope Charts
  # ==============================================================================================================
  try:
    if ("SlopeCharts" in whichToPlot and len(metrics) >= 2):
      print("Generating slope charts...")
      for i, dataset in enumerate(data):
        plt.figure(figsize=(factor, factor))
        y1 = np.mean(dataset[metrics[0]]["Trials"])
        y2 = np.mean(dataset[metrics[1]]["Trials"])
        plt.plot([0, 1], [y1, y2], marker="o", color=cmapColors[i])
        plt.xticks([0, 1], [metrics[0], metrics[1]])
        plt.title(f"Slope Chart for {names[i]}: {metrics[0]} vs {metrics[1]}", color=cmapColors[i])
        plt.ylabel("Mean Value")
        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"SlopeChart_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Slope Charts: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Slope Charts.")
    print("=" * 80)

  # ==============================================================================================================
  # Dumbbell Plots
  # ==============================================================================================================
  try:
    if ("DumbbellPlots" in whichToPlot and len(metrics) >= 2):
      print("Generating dumbbell plots...")
      for i, dataset in enumerate(data):
        plt.figure(figsize=(factor, factor))
        y1 = np.mean(dataset[metrics[0]]["Trials"])
        y2 = np.mean(dataset[metrics[1]]["Trials"])
        plt.plot([0, 1], [y1, y2], "o-", color=cmapColors[i], linewidth=2)
        plt.hlines(y=[y1, y2], xmin=0, xmax=1, colors=cmapColors[i], linestyles="dotted")
        plt.xticks([0, 1], [metrics[0], metrics[1]])
        plt.title(f"Dumbbell Plot for {names[i]}: {metrics[0]} vs {metrics[1]}", color=cmapColors[i])
        plt.ylabel("Mean Value")
        plt.tight_layout()
        keywordRep = keyword.replace("\n", "_")
        plt.savefig(f"DumbbellPlot_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()  # Close the figure to free memory.
        plt.clf()  # Clear the current figure.
  except Exception as e:
    print(f"Error generating Dumbbell Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Dumbbell Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Treemap Plots
  # ==============================================================================================================
  try:
    if ("TreemapPlots" in whichToPlot):
      print("Generating treemap plots...")
      try:
        import squarify
        for i, dataset in enumerate(data):
          sizes = [np.mean(dataset[metric]["Trials"]) for metric in metrics]
          plt.figure(figsize=(factor * 2, factor * 2))
          squarify.plot(sizes=sizes, label=metrics, color=cmapColors[:len(metrics)], alpha=0.7)
          plt.title(f"Treemap Plot for {names[i]}", color=cmapColors[i])
          plt.axis("off")
          plt.tight_layout()
          keywordRep = keyword.replace("\n", "_")
          plt.savefig(f"TreemapPlot_{names[i]}_{keywordRep}{extension}", dpi=dpi, bbox_inches="tight")
          if (showFigures):
            plt.show()
          plt.close()  # Close the figure to free memory.
          plt.clf()  # Clear the current figure.
      except ImportError:
        print("squarify is required for TreemapPlots. Install with 'pip install squarify'. Skipping.")
      except Exception as e:
        print(f"TreemapPlots: {e}")
  except Exception as e:
    print(f"Error generating Treemap Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Treemap Plots.")
    print("=" * 80)

  # ==============================================================================================================
  # Sunburst Plots
  # ==============================================================================================================
  try:
    if ("SunburstPlots" in whichToPlot):
      print("Generating sunburst plots...")
      try:
        import plotly.graph_objects as go
        for i, dataset in enumerate(data):
          sizes = [np.mean(dataset[metric]["Trials"]) for metric in metrics]
          # Compose labels with both metric name and value
          labels = [f"{metric}: {size:.2f}" for metric, size in zip(metrics, sizes)]
          fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=["" for _ in metrics],
            values=sizes,
          ))
          fig.update_layout(title=f"Sunburst Plot for {names[i]}")
          keywordRep = keyword.replace("\n", "_")
          fig.write_image(f"SunburstPlot_{names[i]}_{keywordRep}.png")
      except ImportError:
        print("plotly is required for SunburstPlots. Install with 'pip install plotly'. Skipping.")
      except Exception as e:
        print(f"SunburstPlots: {e}")
  except Exception as e:
    print(f"Error generating Sunburst Plots: {str(e)}")
    print(traceback.format_exc())
    print("Skipping Sunburst Plots.")
    print("=" * 80)

  # Change back to the original directory.
  if (storeInsideNewFolder and newFolderName):
    os.chdir(originalDir)
