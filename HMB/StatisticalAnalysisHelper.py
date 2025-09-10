import os, matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import zconfint
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.power import TTestPower
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, bootstrap, wilcoxon
from HMB.Initializations import IgnoreWarnings

IgnoreWarnings()  # Suppress all warnings globally.


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
    '''
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

  def _bootstrapCIAdvanced(data, statFunc=np.mean, nBootstraps=nBootstraps):
    '''
    Calculate advanced bootstrap confidence interval using scipy's bootstrap.

    Parameters:
      data (array-like): Data to resample.
      statFunc (callable, optional): Statistic function to apply (default: np.mean).
      nBootstraps (int, optional): Number of bootstrap resamples.

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
    '''
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


def GetCmapColors(cmap, noColors, darkColorsOnly=True, darknessThreshold=0.7):
  r'''
  Utility to get a list of unique colors from a matplotlib colormap.

  Parameters:
    cmap (str or Colormap): Name of the colormap or a Colormap object.
    noColors (int): Number of distinct colors to generate.
    darkColorsOnly (bool, optional): Whether to filter out light colors. Default is True.
    darknessThreshold (float, optional): Brightness threshold to classify colors as dark. Default is 0.7.

  Returns:
    list: List of unique RGBA color tuples.
  '''

  # Filter out light colors based on perceived brightness (YIQ formula).
  def IsDark(color):
    r, g, b = color[:3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness < darknessThreshold

  # Get the colormap object.
  if (isinstance(cmap, str)):
    cmapObj = plt.get_cmap(cmap)
  else:
    cmapObj = cmap

  # To maximize uniqueness, sample evenly spaced points in the colormap.
  allColors = [cmapObj(i / max(noColors, 1)) for i in range(noColors)]

  if (darkColorsOnly):
    # Filter to keep only dark colors.
    darkColors = [c for c in allColors if IsDark(c)]

    # If not enough dark colors, try to fill with more from the colormap.
    if (len(darkColors) < noColors):
      # Try to get more dark colors by oversampling.
      extraColors = [cmapObj(i / 1000.0) for i in range(1000)]
      extraDark = [c for c in extraColors if (IsDark(c) and c not in darkColors)]
      darkColors.extend(extraDark)

      # Remove duplicates while preserving order.
      seen = set()
      uniqueDark = []
      for c in darkColors:
        if (c not in seen):
          uniqueDark.append(c)
          seen.add(c)
      darkColors = uniqueDark

    # If still not enough, repeat or fallback.
    if (len(darkColors) >= noColors):
      return darkColors[:noColors]
    elif (len(darkColors) > 0):
      repeats = (noColors // len(darkColors)) + 1
      extended = (darkColors * repeats)[:noColors]
      return extended
    else:
      return allColors
  else:
    # Remove duplicates if any (shouldn't be, but for safety).
    seen = set()
    uniqueColors = []
    for c in allColors:
      if (c not in seen):
        uniqueColors.append(c)
        seen.add(c)
    if (len(uniqueColors) >= noColors):
      return uniqueColors[:noColors]
    else:
      repeats = (noColors // len(uniqueColors)) + 1
      extended = (uniqueColors * repeats)[:noColors]
      return extended


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
  fixedTicksColor="black",  # Color to use for fixed ticks if fixedTicksColors is True.
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
      fixedTicksColor="black"
    )
  '''

  # Set the default Seaborn style for the plots.
  sns.set(style="whitegrid")

  # Set the default font size for Matplotlib plots.
  plt.rcParams.update({"font.size": fontSize})
  plt.rcParams["axes.titlesize"] = fontSize
  plt.rcParams["axes.labelsize"] = fontSize
  plt.rcParams["xtick.labelsize"] = fontSize
  plt.rcParams["ytick.labelsize"] = fontSize
  plt.rcParams["legend.fontsize"] = fontSize
  plt.rcParams["figure.titlesize"] = fontSize
  plt.rcParams["lines.linewidth"] = 2  # Set the default line width for plots.
  plt.rcParams["lines.markersize"] = 6  # Set the default marker size for plots.
  plt.rcParams["legend.loc"] = "best"  # Set the default legend location.
  plt.rcParams["figure.figsize"] = (factor * 5, factor * 5)  # Default figure size.
  plt.rcParams["figure.dpi"] = dpi  # Set the resolution of the figures.
  plt.rcParams["savefig.dpi"] = dpi  # Set the resolution for saved figures.
  plt.rcParams["savefig.bbox"] = "tight"  # Save figures with tight bounding box.

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
      # Note: AreaPlots (Stacked Line Plots) could be added if trial data represents parts of a whole over time.
    ]

  if (differentColors):
    # Get colors from the specified colormap.
    cmapColors = GetCmapColors(
      cmap,
      (len(names) * len(metrics)),
      darkColorsOnly=True,
      darknessThreshold=0.6
    )
    print(f"Using colormap '{cmap}' with {len(cmapColors)} colors.")
  else:
    cmapColors = ["blue"] * (len(names) * len(metrics))
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"ResidualPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
    else:
      print("ResidualPlots: No data available to plot.")

  # ===============================================================================================================
  # Q-Q Residual Plots (vs Normal Distribution)
  # Q-Q (Quantile-Quantile) plots compare the distribution of residuals
  # (calculated from fitting metric value vs trial index) to a theoretical normal distribution.
  # If the points fall approximately along the reference line, it suggests that the residuals are
  # normally distributed. Deviations indicate departures from normality.
  # This checks the normality assumption often made in statistical models.
  # ===============================================================================================================
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"QQResidualPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
    else:
      print("QQResidualPlots: No data available to plot.")

  # ===============================================================================================================
  # Bland-Altman Plots
  # Bland-Altman plots are used to visualize the agreement between two different measurement methods.
  # They plot the difference between the two methods against their average, helping to identify any systematic bias.
  # ===============================================================================================================
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
        plt.savefig(f"BlandAltmanPlot_{keyword}_{names[k]}.pdf", dpi=dpi, bbox_inches="tight")
        if (showFigures):
          plt.show()
        plt.close()
        plt.clf()  # Clear the current figure.
  elif ("BlandAltmanPlots" in whichToPlot and len(metrics) < 2):
    print("BlandAltmanPlots: Not enough metrics to generate the plots.")

  # ==============================================================================================================
  # Histograms
  # Histograms are fundamental plots for visualizing the frequency distribution of data.
  # They help understand the shape, central tendency, and spread of the data.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"Histogram_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Boxplots
  # Boxplots are useful for visualizing the distribution of data based on a five-number summary:
  # minimum, first quartile (Q1), median, third quartile (Q3), and maximum. They are particularly
  # effective for identifying outliers and comparing distributions across different groups.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"BoxPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()  # Close the figure to free memory.
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Violin Plots
  # Violin plots combine the benefits of boxplots and density plots. They show the distribution of data
  # across different groups, including the probability density of the data at different values. This makes
  # them ideal for comparing the shape and spread of distributions.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"ViolinPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Q-Q Plots
  # Q-Q (Quantile-Quantile) plots are used to assess whether a dataset follows a particular distribution,
  # often the normal distribution. They compare the quantiles of the dataset to the quantiles of a theoretical
  # distribution, helping to identify deviations from normality.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"QQPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Density Plots (KDE)
  # Density plots are a powerful visualization tool that provides insights into the distribution of data,
  # particularly in understanding the shape, central tendency, and spread of the data. They are useful for
  # identifying peaks, valleys, and overlaps in distributions.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"DensityPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Scatter Plots
  # Scatter plots are used to visualize the relationship between two variables. They are ideal for identifying
  # correlations, trends, and outliers in paired data.
  # ==============================================================================================================
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"ScatterPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
    else:
      print("ScatterPlots: Not enough metrics to generate pairs for plotting.")

  # ==============================================================================================================
  # Line Plots (Trend Analysis)
  # Line plots are used to visualize trends over time or across ordered categories. They are particularly
  # useful for showing changes in metrics over trials or iterations.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"LinePlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Bar Plots
  # Bar plots are used to compare the mean or median of metrics across different groups. They are effective
  # for summarizing and comparing aggregated data.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"BarPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Correlation Heatmaps
  # Correlation heatmaps visualize the correlation matrix of multiple metrics. They are useful for identifying
  # relationships and dependencies between different metrics.
  # ==============================================================================================================
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"CorrelationHeatmap_{names[i]}_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
  elif ("CorrelationHeatmaps" in whichToPlot and len(metrics) <= 1):
    print("CorrelationHeatmaps: Not enough metrics to generate the plots.")

  # ==============================================================================================================
  # Pair Plots (Pairwise Relationships)
  # Pair plots are used to visualize pairwise relationships between multiple variables. They are useful for
  # identifying correlations and patterns across multiple metrics.
  # ==============================================================================================================
  if ("PairPlots" in whichToPlot):
    print("Generating pair plots...")
    for i, dataset in enumerate(data):
      df = pd.DataFrame({metric: dataset[metric]["Trials"] for metric in metrics})
      sns.pairplot(df)
      plt.suptitle(f"Pair Plot for {names[i]}", y=1.02)
      plt.xticks(color=GetTickColor(i))
      plt.yticks(color=GetTickColor(i))
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"PairPlot_{names[i]}_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # CDF Plots
  # CDF (Cumulative Distribution Function) plots show the cumulative probability of a variable. They are useful
  # for understanding the distribution and comparing the spread of data across different groups.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"CDF_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # ECDF Plots
  # ECDF (Empirical Cumulative Distribution Function) plots are similar to CDF plots but focus on the empirical
  # distribution of the data. They are useful for visualizing the distribution of individual datasets.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"ECDF_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Swarm Plots
  # Swarm plots are used to visualize individual data points and their distribution. They are useful for
  # showing the density of data points and identifying patterns or clusters.
  # ==============================================================================================================
  if ("SwarmPlots" in whichToPlot):
    print("Generating swarm plots...")
    plt.figure(figsize=(factor * noCols, factor * noRows))
    for i, metric in enumerate(metrics):
      plt.subplot(noRows, noCols, i + 1)
      sns.swarmplot(
        data=[dataset[metric]["Trials"] for dataset in data],
        palette=cmapColors
      )
      color = cmapColors[i]
      plt.title(
        f"Swarm Plot of {metric} Results",
        color=color
      )
      plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
      plt.yticks(color=GetTickColor(i))
      plt.ylabel("Performance Metric", color=GetTickColor(i))
    plt.tight_layout()
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"SwarmPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Pie Charts
  # Pie charts are used to visualize the proportion of different categories within a dataset.
  # They are effective for showing the relative sizes of different groups or categories.
  # In this context, it shows the relative average values of different metrics for each dataset/system.
  # ==============================================================================================================
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

      # Improve layout to prevent label clipping (though pie charts can be tricky)
      plt.tight_layout()
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"PieChart_{names[i]}_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Area Plots
  # Area plots are used to visualize the cumulative total of a metric over time or across categories
  # They are effective for showing trends and the relative contribution of different groups.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"AreaPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Hexbin Plots
  # Hexbin plots are used to visualize the density of data points in a two-dimensional space
  # They are effective for showing the distribution of data points and identifying clusters.
  # ==============================================================================================================
  if ("HexbinPlots" in whichToPlot):
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

      # Adjust figure size based on the number of pairs.
      plt.figure(figsize=(factor * hbCols, factor * hbRows))

      for plotIdx, (i, j) in enumerate(uniquePairs):
        metric1 = metrics[i]
        metric2 = metrics[j]
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
            gridsize=30,
            cmap=cmap,
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"HexbinPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
    else:
      print("HexbinPlots: Not enough metrics to generate pairs for plotting.")

  # ==============================================================================================================
  # Contour Plots
  # Contour plots are used to visualize three-dimensional data in two dimensions by plotting contour lines.
  # They are effective for showing the relationship between two variables and a third variable represented
  # by contour lines.
  # ==============================================================================================================
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"ContourPlot_{names[i]}_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
  elif ("ContourPlots" in whichToPlot and noOfMetrics < 2):
    print("ContourPlots: Not enough metrics to generate the plots.")

  # ==============================================================================================================
  # Strip Plots
  # Strip plots show individual data points for each group, allowing overlap.
  # ==============================================================================================================
  if ("StripPlots" in whichToPlot):
    print("Generating strip plots...")
    plt.figure(figsize=(factor * noCols, factor * noRows))
    for i, metric in enumerate(metrics):
      plt.subplot(noRows, noCols, i + 1)
      sns.stripplot(
        data=[dataset[metric]["Trials"] for dataset in data],
        palette=cmapColors,
        alpha=0.7
      )
      color = cmapColors[i]
      plt.title(f"Strip Plot of {metric} Results", color=color)
      plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
      plt.yticks(color=GetTickColor(i))
      plt.ylabel("Performance Metric", color=GetTickColor(i))
    plt.tight_layout()
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"StripPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()

  # ==============================================================================================================
  # Dot Plots
  # Dot plots show dots for each data point, grouped by category.
  # ==============================================================================================================
  if ("DotPlots" in whichToPlot):
    print("Generating dot plots...")
    plt.figure(figsize=(factor * noCols, factor * noRows))
    for i, metric in enumerate(metrics):
      plt.subplot(noRows, noCols, i + 1)
      for j, dataset in enumerate(data):
        y = dataset[metric]["Trials"]
        x = np.full_like(y, j, dtype=float) + np.random.uniform(-0.1, 0.1, size=len(y))
        plt.plot(x, y, 'o', color=cmapColors[j], alpha=0.7, label=names[j] if i == 0 else None)
      color = cmapColors[i]
      plt.title(f"Dot Plot of {metric} Results", color=color)
      plt.xticks(range(len(names)), names, rotation=xTicksRotation, color=GetTickColor(i))
      plt.yticks(color=GetTickColor(i))
      plt.ylabel("Performance Metric", color=GetTickColor(i))
    plt.tight_layout()
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"DotPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()

  # ==============================================================================================================
  # Stacked Bar Plots
  # Stacked bar plots show the mean values of each metric for each dataset, stacked.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"StackedBarPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()

  # ==============================================================================================================
  # Stacked Area Plots
  # Stacked area plots show cumulative trends for each group/metric over trials.
  # ==============================================================================================================
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"StackedAreaPlot_{names[i]}_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()

  # ==============================================================================================================
  # Histogram2DPlots
  # 2D histograms for all unique pairs of metrics.
  # ==============================================================================================================
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
      keywordRep = keyword.replace('\n', '_')
      plt.savefig(f"Histogram2DPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
      # Show the figure if requested.
      if (showFigures):
        plt.show()
      plt.close()  # Close the current figure.
      plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # StepPlots
  # 2D histograms for all unique pairs of metrics.
  # ==============================================================================================================
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
    keywordRep = keyword.replace('\n', '_')
    plt.savefig(f"StepPlot_{keywordRep}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()  # Close the current figure.
    plt.clf()  # Clear the current figure.


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

    history, names, metrics = sah.ExtractDataFromSummaryFile("path/to/your/summary_file.csv")
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
      if numeric else f"{keyword}_EDA_Distribution_NonNumeric_Plots.pdf"
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
