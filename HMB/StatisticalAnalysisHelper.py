# Import necessary libraries.
import os
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
  '''
  Perform comprehensive statistical analysis on a list of results.

  Parameters:
    results (list): List of performance metrics (e.g., accuracy, scores).
    hypothesizedMean (float, optional): Mean value for one-sample t-test. Default is 0.
    secondMetricList (list, optional): Second list of metrics for correlation/regression analysis. Default is None.
    confidenceLevel (float, optional): Confidence level for confidence intervals. Default is 0.95.
    nBootstraps (int, optional): Number of bootstrap resamples for confidence intervals. Default is 1000.

  Returns:
    dict: Dictionary containing all statistical analysis results.

  Raises:
    ValueError: If the input results list is empty or contains non-numeric values.
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
):
  '''
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

  Raises:
    ValueError: If the input data is not in the expected format.
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
              plt.title(f"Residuals vs Trial Index\n{metric} - {names[j]}")
              plt.grid(True, alpha=0.5)
              plotIdx += 1
            except Exception as e:
              # Handle potential fitting errors (e.g., singular matrix if all X values are the same).
              plt.subplot(resRows, resCols, plotIdx)
              plt.text(
                0.5, 0.5, f"Fit Error:\n{str(e)[:30]}...", ha="center", va="center",
                transform=plt.gca().transAxes, fontsize=10
              )
              plt.title(f"Residuals vs Trial Index\n{metric} - {names[j]}\n(Fit Failed)")
              plt.xlabel("Trial Index")
              plt.ylabel("Residuals")
              plotIdx += 1
          else:
            # Not enough data points.
            plt.subplot(resRows, resCols, plotIdx)
            plt.text(0.5, 0.5, "N/A\n(< 2 trials)", ha="center", va="center", transform=plt.gca().transAxes)
            plt.title(f"Residuals vs Trial Index\n{metric} - {names[j]}\n(Insufficient Data)")
            plt.xlabel("Trial Index")
            plt.ylabel("Residuals")
            plotIdx += 1

      plt.tight_layout(pad=1.0)  # Add padding to prevent title overlap
      plt.savefig(f"ResidualPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
              plt.title(f"Q-Q Plot of Residuals\n{metric} - {names[j]}")
              # Labels are usually set by sm.qqplot, but ensure they are present.
              if (not plt.gca().get_ylabel()):
                plt.xlabel("Theoretical Quantiles (Normal)")
              if (not plt.gca().get_ylabel()):
                plt.ylabel("Sample Quantiles (Residuals)")

              plotIdx += 1
            except Exception as e:
              # Handle potential fitting errors.
              plt.subplot(qqRows, qqCols, plotIdx)
              plt.text(0.5, 0.5, f"Fit/Q-Q Error:\n{str(e)[:30]}...", ha="center", va="center",
                       transform=plt.gca().transAxes, fontsize=10)
              plt.title(f"Q-Q Plot of Residuals\n{metric} - {names[j]}\n(Fit/Q-Q Failed)")
              plotIdx += 1
              plt.xlabel("Theoretical Quantiles")  # Fallback labels.
              plt.ylabel("Sample Quantiles")
          else:
            # Not enough data points.
            plt.subplot(qqRows, qqCols, plotIdx)
            plt.text(0.5, 0.5, "N/A\n(< 2 trials)", ha="center", va="center",
                     transform=plt.gca().transAxes)
            plt.title(f"Q-Q Plot of Residuals\n{metric} - {names[j]}\n(Insufficient Data)")
            plotIdx += 1
            plt.xlabel("Theoretical Quantiles")  # Fallback labels.
            plt.ylabel("Sample Quantiles")

      plt.tight_layout(pad=1.5)  # Increase padding due to longer titles.
      plt.savefig(f"QQResidualPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
                plt.title(f"Bland-Altman: {metric1} vs {metric2}\n({names[k]})")
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
        plt.hist(dataset[metric]["Trials"], bins="auto", alpha=0.5, label=names[j], edgecolor="black")
      plt.title(f"Histogram of {metric} Results")
      plt.xlabel("Performance Metric")
      plt.ylabel("Frequency")
      plt.legend()
    plt.tight_layout()
    plt.savefig(f"Histogram_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.boxplot(
        [el[metric]["Trials"] for el in data],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=5, markerfacecolor="red", markeredgecolor="red")
      )
      plt.title(f"Boxplot of {metric} Results")
      plt.xticks(list(range(1, noOfDatasets + 1)), names, rotation=xTicksRotation)
      plt.ylabel("Performance Metric")
    plt.tight_layout()
    plt.savefig(f"BoxPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
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
      plt.violinplot(
        [el[metric]["Trials"] for el in data],
        showmeans=True,
        showmedians=True
      )
      plt.title(f"Violin Plot of {metric} Results")
      plt.xticks(list(range(1, noOfDatasets + 1)), names, rotation=xTicksRotation)
      plt.ylabel("Performance Metric")
    plt.tight_layout()
    plt.savefig(f"ViolinPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.title(f"Q-Q Plot of {metric} Results")
    plt.tight_layout()
    plt.savefig(f"QQPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
        sns.kdeplot(dataset[metric]["Trials"], label=names[j], fill=True)
        # Add rug plot for individual data points.
        sns.rugplot(dataset[metric]["Trials"], height=0.05, alpha=0.5)
      plt.title(f"Density Plot of {metric} Results")
      plt.xlabel("Performance Metric")
      plt.ylabel("Density")
      plt.legend()
    plt.tight_layout()
    plt.savefig(f"DensityPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
    if (showFigures):
      plt.show()
    plt.close()
    plt.clf()  # Clear the current figure.

  # ==============================================================================================================
  # Scatter Plots
  # Scatter plots are used to visualize the relationship between two variables. They are ideal for identifying
  # correlations, trends, and outliers in paired data.
  # ==============================================================================================================
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
          # Ensure data lengths match for scatter plot
          if (len(xVals) == len(yVals)):
            plt.scatter(xVals, yVals, label=names[k])
          else:
            # Handle mismatched lengths if necessary, or skip.
            # For now, we assume data integrity from ExtractDataFromSummaryFile.
            pass

        plt.title(f"Scatter Plot: {metric1} vs {metric2}")
        plt.xlabel(metric1)
        plt.ylabel(metric2)
        # Only add legend if there are multiple datasets to differentiate
        if (noOfDatasets > 1):
          plt.legend()

      plt.tight_layout()
      plt.savefig(f"ScatterPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
        plt.plot(dataset[metric]["Trials"], label=names[j])
      plt.title(f"Line Plot of {metric} Results")
      plt.xlabel("Trial")
      plt.ylabel("Performance Metric")
      plt.legend()
    plt.tight_layout()
    plt.savefig(f"LinePlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.bar(names, means, color="lightblue")
      plt.title(f"Bar Plot of {metric} Results")
      plt.xlabel("Dataset")
      plt.ylabel("Mean Performance Metric")
    plt.tight_layout()
    plt.savefig(f"BarPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
      plt.title(f"Correlation Heatmap for {names[i]}")
      plt.savefig(f"CorrelationHeatmap_{names[i]}_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.savefig(f"PairPlot_{names[i]}_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.title(f"CDF of {metric} Results")
      plt.xlabel("Performance Metric")
      plt.ylabel("Cumulative Probability")
      plt.legend()
    plt.tight_layout()
    plt.savefig(f"CDF_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.title(f"ECDF of {metric} Results")
      plt.xlabel("Performance Metric")
      plt.ylabel("Empirical Cumulative Probability")
      plt.legend()
    plt.tight_layout()
    plt.savefig(f"ECDF_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      sns.swarmplot(data=[dataset[metric]["Trials"] for dataset in data], palette="Set2")
      plt.title(f"Swarm Plot of {metric} Results")
      plt.xticks(range(len(names)), names, rotation=xTicksRotation)
      plt.ylabel("Performance Metric")
    plt.tight_layout()
    plt.savefig(f"SwarmPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.figure(figsize=(factor * 5, factor * 5))

      # Create the pie chart.
      # autopct displays percentages, startangle rotates the start, textprops adjusts label size.
      wedges, texts, autotexts = plt.pie(
        sizes,
        labels=metrics,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": max(8, fontSize - 2)}  # Adjust label font size slightly smaller.
      )

      # Set the title for the current dataset.
      plt.title(f"Pie Chart of Metrics for {names[i]}", fontsize=fontSize)
      # Equal aspect ratio ensures that pie is drawn as a circle.
      plt.axis("equal")

      # Improve layout to prevent label clipping (though pie charts can be tricky)
      plt.tight_layout()
      plt.savefig(f"PieChart_{names[i]}_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
    plt.figure(figsize=(factor * noCols, factor * noRows))
    for i, metric in enumerate(metrics):
      plt.subplot(noRows, noCols, i + 1)
      for j, dataset in enumerate(data):
        plt.fill_between(
          range(len(dataset[metric]["Trials"])),
          dataset[metric]["Trials"],
          label=names[j],
          alpha=0.5
        )
      plt.title(f"Area Plot of {metric} Results")
      plt.xlabel("Trial")
      plt.ylabel("Performance Metric")
      plt.legend()
    plt.tight_layout()
    plt.savefig(f"AreaPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
            cmap="Blues",
            mincnt=1  # Only show hexagons with at least one count.
          )
          plt.title(f"Hexbin Plot: {metric1} vs {metric2}")
          plt.xlabel(metric1)
          plt.ylabel(metric2)
          plt.colorbar(hb, ax=plt.gca(), label="Count")
        else:
          plt.title(f"Hexbin Plot: {metric1} vs {metric2}\n(No Data)")
          plt.xlabel(metric1)
          plt.ylabel(metric2)

      plt.tight_layout()
      plt.savefig(f"HexbinPlot_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
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
      plt.tricontourf(x, y, z, levels=14, cmap="RdYlBu")
      plt.colorbar(label="Metric Value")
      plt.title(f"Contour Plot for {names[i]}")
      plt.xlabel(metrics[0])
      plt.ylabel(metrics[1])
      plt.savefig(f"ContourPlot_{names[i]}_{keyword}.pdf", dpi=dpi, bbox_inches="tight")
      if (showFigures):
        plt.show()
      plt.close()
      plt.clf()  # Clear the current figure.
  elif ("ContourPlots" in whichToPlot and noOfMetrics < 2):
    print("ContourPlots: Not enough metrics to generate the plots.")


def ExtractDataFromSummaryFile(file):
  '''
  Extract and organize data from a summary CSV file containing names, metrics, and trial data.
  The file is expected to be structured as follows:
    - First line: Comma-separated names (headers for individuals or categories).
    - Second line: Comma-separated metrics (headers for performance or evaluation criteria).
    - Subsequent lines: Numerical values corresponding to metrics for each trial.

  Example of the file structure:
    System A, , , , , , System B, , , , ,
    Precision, Recall, F1, Accuracy, Specificity, Average, Precision, Recall, F1, Accuracy, Specificity, Average
    0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133, 0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133
    0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282, 0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282
    0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406, 0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406
    0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339, 0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339
    0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813, 0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813
    file (str): Path to the summary CSV file.

    history (list of dict): A list where each entry corresponds to a name and contains a dictionary mapping each metric to its trials and mean value.
    history (list of dict): List where each entry corresponds to a name and contains a dictionary mapping each metric to its trials and mean value.
    names (list of str): Cleaned list of names extracted from the first line of the file.
    metrics (list of str): Cleaned list of metrics extracted from the second line of the file.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file format is invalid or data cannot be parsed.
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
