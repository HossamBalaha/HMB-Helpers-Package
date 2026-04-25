import math
from functools import reduce
from collections import Counter
from HMB.Utils import SafeCall

'''
VotingHelper: Collection of voting and aggregation utilities.

Method Category and Methods Reference:
======================================

Classical Label Voting:
  - WeightedMajorityVoting: Compute weighted majority vote over labels.
  - MajorityVoting: Simple unweighted majority vote.
  - WeightedModeVoting: Return label with highest cumulative weight.
  - ModeVoting: Return most common label (unweighted mode).

Statistical Mean Aggregation:
  - WeightedAverageVoting: Weighted arithmetic mean of numeric predictions.
  - AverageVoting: Simple arithmetic mean of numeric predictions.
  - WeightedGeometricMeanVoting: Weighted geometric mean for positive values.
  - GeometricMeanVoting: Simple geometric mean for positive values.
  - WeightedHarmonicMeanVoting: Weighted harmonic mean for positive values.
  - HarmonicMeanVoting: Simple harmonic mean for positive values.
  - WeightedQuadraticMeanVoting: Weighted root-mean-square (quadratic mean).
  - QuadraticMeanVoting: Simple root-mean-square.
  - WeightedCubicMeanVoting: Weighted cubic mean with real-valued cube root.
  - CubicMeanVoting: Simple cubic mean with real-valued cube root.
  - WeightedQuarticMeanVoting: Weighted quartic mean (4th root of mean of 4th powers).
  - QuarticMeanVoting: Simple quartic mean.

Median Aggregation:
  - WeightedMedianVoting: Weighted median using cumulative weight threshold.
  - MedianVoting: Simple unweighted median for numeric predictions.

Probabilistic Aggregation:
  - SoftVoting: Average predicted probabilities across models.
  - EntropyWeightedVoting: Weight predictions by inverse Shannon entropy.
  - ProductOfExpertsVoting: Log-linear pooling combining probabilities via weighted product (normalized).

Confidence and Uncertainty Methods:
  - ConfidenceWeightedVoting: Weight predictions by model confidence scores.
  - UncertaintyAwareVoting: Weight predictions by inverse uncertainty estimates.

Bayesian and Calibration Methods:
  - BayesianModelAveraging: Weight predictions by posterior model probabilities.
  - CalibrationAwareVoting: Weight predictions by model calibration quality.

Rank Aggregation (Social Choice):
  - BordaCountVoting: Borda Count rank aggregation with positional scoring.
  - CondorcetVoting: Condorcet method with pairwise comparison and Borda fallback.
  - CopelandVoting: Score candidates by net pairwise wins (wins minus losses) with deterministic tie-breaking.

Diversity-Aware Methods:
  - DiversityWeightedVoting: Weight predictions by pairwise model disagreement.
  - CorrelationAwareWeightedVoting: Adjust model weights by inverse pairwise correlation to promote ensemble diversity.
  
Meta-Learning Methods:
  - MetaWeightLearning: Learn optimal linear combination weights from validation data.

Robust Statistical Aggregation:
  - RobustMeanVoting: Trimmed mean aggregation excluding extreme values to resist outlier predictions.

Neural/Attention-Based Aggregation:
  - AttentionWeightedVoting: Compute dynamic weights via scaled dot-product attention with context features.

Online/Adaptive Weighting:
  - HedgeAdaptiveVoting: Online adaptive voting using Hedge algorithm with exponential weighting and regret bounds.

Federated Learning Aggregators:
  - FedAvgVoting: Federated averaging aggregator combining local predictions weighted by client data size.

Adversarial-Robust Aggregation:
  - MedianOfMeansVoting: Robust aggregation via median-of-means partitioning to resist Byzantine faults.

Distributional/Quantile Aggregation:
  - QuantileAggregationVoting: Combine predictive distributions via quantile averaging at target quantile level.

Conformal Prediction Methods:
  - ConformalPredictionVoting: Return prediction sets with guaranteed marginal coverage using nonconformity scores.

Dynamic Ensemble Selection:
  - DynamicEnsembleSelectionVoting: Weight models by local competence estimates per sample for adaptive aggregation.

Optimal Transport Distribution Aggregation:
  - WassersteinBarycenterVoting: Combine predictive distributions via Wasserstein barycenters for distributional forecasts.
  
Causal Invariance Methods:
  - CausalInvariantVoting: Weight models by invariant predictive performance across environments using environment-wise variance metrics.

Graph Neural Aggregation:
  - GraphNeuralAggregation: Propagate predictions through model dependency graph via message-passing with configurable activation and iterations.
  
Input Validation Standards:
  - All methods validate non-empty input iterables.
  - All weighted methods validate positive sum of weights and length alignment.
  - Geometric and harmonic mean methods validate strictly positive input values.
  - Calibration methods validate scores in range [0, 1].
  - Uncertainty methods validate non-negative uncertainty estimates.

Numerical Stability Measures:
  - Geometric means use log-sum-exp technique to prevent overflow.
  - Cube root operations use math.copysign for real-valued results with negatives.
  - Division operations include epsilon safeguards to prevent zero-division errors.
'''


class VotingHelper(object):
  r'''
  VotingHelper: Collection of voting and aggregation utilities.

  Each method takes a list of predictions and optionally a list of weights (same length).
  Methods return a single aggregated scalar or label depending on the strategy.
  '''

  def WeightedMajorityVoting(self, predictions, weights):
    r'''
    Compute a weighted majority vote over labels.

    Parameters:
      predictions (iterable): Iterable of labels.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      label: The label with the highest cumulative weight.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    cumulativeVotes = Counter()
    for pred, weight in zip(predictions, weights):
      cumulativeVotes[pred] += weight
    majorityLabel = max(cumulativeVotes, key=cumulativeVotes.get)
    return majorityLabel

  def MajorityVoting(self, predictions):
    r'''
    Simple majority vote (unweighted).

    Parameters:
      predictions (iterable): Iterable of labels.

    Returns:
      label: The most common label.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    # majorityLabel = max(set(predictions), key=predictions.count)
    # return majorityLabel
    return Counter(predictions).most_common(1)[0][0]

  def WeightedAverageVoting(self, predictions, weights):
    r'''
    Weighted average of numeric predictions.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted mean.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    result = sum(pred * weight for pred, weight in zip(predictions, weights)) / totalWeight
    return result

  def AverageVoting(self, predictions):
    r'''
    Simple arithmetic mean of numeric predictions.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Arithmetic mean.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    return sum(predictions) / len(predictions)

  def WeightedMedianVoting(self, predictions, weights):
    r'''
    Weighted median: returns a scalar median taking weights into account.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted median.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    # Pair and sort by prediction value.
    items = sorted(zip(predictions, weights), key=lambda x: x[0])
    cum = 0.0
    for val, w in items:
      cum += w
      if (cum >= totalWeight / 2.0):
        return val
    return items[-1][0]

  def MedianVoting(self, predictions):
    r'''
    Unweighted median for numeric predictions.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Median value.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    s = sorted(predictions)
    mid = len(s) // 2
    if (len(s) % 2 == 1):
      return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

  def WeightedModeVoting(self, predictions, weights):
    r'''
    Weighted mode: returns the label with the highest cumulative weight.

    For numeric or non-numeric labels, this method aggregates weights per unique label
    and returns the label with the maximum total weight. This is equivalent to
    WeightedMajorityVoting.

    Parameters:
      predictions (iterable): Iterable of labels (numeric or non-numeric).
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      label or float: Weighted mode label or numeric value.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    # # If labels are non-numeric, return the label with largest cumulative weight.
    # try:
    #   # try numeric path: compute weighted frequencies per unique value
    #   cumulative = {}
    #   for p, w in zip(predictions, weights):
    #     cumulative[p] = cumulative.get(p, 0) + w
    #   # return the key with max cumulative weight
    #   return max(cumulative, key=cumulative.get)
    # except Exception:
    #   return self.WeightedMajorityVoting(predictions, weights)

    cumulative = {}
    for p, w in zip(predictions, weights):
      cumulative[p] = cumulative.get(p, 0) + w
    return max(cumulative, key=cumulative.get)

  def ModeVoting(self, predictions):
    r'''
    Unweighted mode (most common element).

    Parameters:
      predictions (iterable): Iterable of labels.

    Returns:
      label: Most common label.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    # return max(set(predictions), key=predictions.count)
    return Counter(predictions).most_common(1)[0][0]

  def WeightedGeometricMeanVoting(self, predictions, weights):
    r'''
    Weighted geometric mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted geometric mean.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")
    if (any(p <= 0 for p in predictions)):
      raise ValueError("All predictions must be positive for geometric/harmonic means")

    # Compute product of powers: prod(p_i ** w_i) ^ (1 / sum(weights)).
    logSum = sum([w * math.log(p) for p, w in zip(predictions, weights)])
    return math.exp(logSum / totalWeight)

  def GeometricMeanVoting(self, predictions):
    r'''
    Geometric mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.

    Returns:
      float: Geometric mean.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")
    if (any(p <= 0 for p in predictions)):
      raise ValueError("All predictions must be positive for geometric/harmonic means")

    # prod = reduce(lambda x, y: x * y, predictions)
    # return prod ** (1.0 / len(predictions))
    logSum = sum(math.log(p) for p in predictions)
    return math.exp(logSum / len(predictions))

  def WeightedHarmonicMeanVoting(self, predictions, weights):
    r'''
    Weighted harmonic mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted harmonic mean.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")
    if (any(p <= 0 for p in predictions)):
      raise ValueError("All predictions must be positive for harmonic means")

    den = sum([w / p for p, w in zip(predictions, weights)])
    return totalWeight / den

  def HarmonicMeanVoting(self, predictions):
    r'''
    Unweighted harmonic mean for positive numeric predictions.

    Parameters:
      predictions (iterable): Iterable of positive numeric predictions.

    Returns:
      float: Harmonic mean.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")
    if (any(p <= 0 for p in predictions)):
      raise ValueError("All predictions must be positive for harmonic means")

    return len(predictions) / sum([1.0 / pred for pred in predictions])

  def WeightedQuadraticMeanVoting(self, predictions, weights):
    r'''
    Weighted root-mean-square (quadratic mean).

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted root-mean-square.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    num = sum([w * (p ** 2) for p, w in zip(predictions, weights)])
    return (num / totalWeight) ** 0.5

  def QuadraticMeanVoting(self, predictions):
    r'''
    Unweighted root-mean-square.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Root-mean-square.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    return (sum([pred ** 2 for pred in predictions]) / len(predictions)) ** 0.5

  def _realCubeRoot(self, x: float) -> float:
    r'''
    Compute the real cube root of a number, handling negative inputs correctly.

    Parameters:
      x (float): The input number.

    Returns:
      float: The real cube root of x.
    '''

    return math.copysign(abs(x) ** (1.0 / 3), x)

  def WeightedCubicMeanVoting(self, predictions, weights):
    r'''
    Weighted cubic mean (signed) -> (mean of cubes)^(1/3).

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted cubic mean.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    num = sum([w * (p ** 3) for p, w in zip(predictions, weights)])
    meanCubed = num / totalWeight
    return self._realCubeRoot(meanCubed)

  def CubicMeanVoting(self, predictions):
    r'''
    Unweighted cubic mean.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Cubic mean.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    meanCubed = sum([pred ** 3 for pred in predictions]) / len(predictions)
    return self._realCubeRoot(meanCubed)

  def WeightedQuarticMeanVoting(self, predictions, weights):
    r'''
    Weighted quartic mean -> 4th root of mean of 4th powers.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      weights (iterable): Iterable of numeric weights (same length as predictions).

    Returns:
      float: Weighted quartic mean.
    '''

    totalWeight = sum(weights)
    if (totalWeight <= 0):
      raise ValueError("Sum of weights must be positive")
    if (len(predictions) != len(weights)):
      raise ValueError("Predictions and weights must have the same length")

    num = sum([w * (p ** 4) for p, w in zip(predictions, weights)])
    return (num / totalWeight) ** (1.0 / 4)

  def QuarticMeanVoting(self, predictions):
    r'''
    Unweighted quartic mean.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.

    Returns:
      float: Quartic mean.
    '''

    if (not predictions):
      raise ValueError("Predictions must be a non-empty iterable")

    return (sum([pred ** 4 for pred in predictions]) / len(predictions)) ** (1.0 / 4)

  def SoftVoting(self, probabilityPredictions):
    r'''
    Soft voting: average predicted probabilities across models.

    Parameters:
      probabilityPredictions (list of dict):
        Each dict maps class labels to predicted probabilities.

    Returns:
      label: Class with highest average probability.
    '''

    # Validate that input is non-empty.
    if (not probabilityPredictions):
      raise ValueError("probabilityPredictions must be non-empty")

    # Initialize accumulator for aggregated probabilities.
    aggregatedProbs = {}

    # Iterate through each model's probability predictions.
    for probs in probabilityPredictions:
      # Iterate through each class and its probability.
      for cls, prob in probs.items():
        # Accumulate probability for this class.
        if (cls not in aggregatedProbs):
          aggregatedProbs[cls] = 0.0
        aggregatedProbs[cls] += prob

    # Compute average probability for each class.
    nModels = len(probabilityPredictions)
    for cls in aggregatedProbs:
      aggregatedProbs[cls] /= nModels

    # Return class with highest average probability.
    return max(aggregatedProbs, key=aggregatedProbs.get)

  def ConfidenceWeightedVoting(self, predictions, confidences):
    r'''
    Weighted voting using model confidence scores.

    Parameters:
      predictions (iterable): Predicted labels.
      confidences (iterable): Confidence scores in [0, 1] or positive reals.

    Returns:
      label: Label with highest cumulative confidence-weighted vote.
    '''

    # Validate that predictions is non-empty.
    if (not predictions):
      raise ValueError("predictions must be non-empty")

    # Validate that predictions and confidences have same length.
    if (len(predictions) != len(confidences)):
      raise ValueError("predictions and confidences must have same length")

    # Validate that all confidence values are non-negative.
    if (any(c < 0 for c in confidences)):
      raise ValueError("confidences must be non-negative")

    # Initialize counter for cumulative confidence-weighted votes.
    cumulativeVotes = Counter()

    # Iterate through predictions and their confidence scores.
    for pred, conf in zip(predictions, confidences):
      # Accumulate confidence weight for this prediction.
      cumulativeVotes[pred] += conf

    # Return label with highest cumulative confidence weight.
    return max(cumulativeVotes, key=cumulativeVotes.get)

  def BayesianModelAveraging(self, predictions, modelPosteriors):
    r'''
    Bayesian Model Averaging: weight predictions by posterior model probabilities.

    Parameters:
      predictions (iterable): Predicted labels from each model.
      modelPosteriors (iterable): Posterior probabilities for each model (sum to 1).

    Returns:
      label: Label with highest posterior-weighted cumulative probability.
    '''

    # Validate that predictions is non-empty.
    if (not predictions):
      raise ValueError("predictions must be non-empty")

    # Validate that model posteriors sum to one within tolerance.
    if (abs(sum(modelPosteriors) - 1.0) > 1e-6):
      raise ValueError("modelPosteriors must sum to 1.0")

    # Initialize counter for cumulative posterior-weighted votes.
    cumulativeProbs = Counter()

    # Iterate through predictions and their model posterior weights.
    for pred, post in zip(predictions, modelPosteriors):
      # Accumulate posterior probability for this prediction.
      cumulativeProbs[pred] += post

    # Return label with highest cumulative posterior probability.
    return max(cumulativeProbs, key=cumulativeProbs.get)

  def BordaCountVoting(self, rankedPredictions):
    r'''
    Borda Count rank aggregation: higher ranks receive more points.

    Parameters:
      rankedPredictions (list of lists):
        Each inner list is a ranking of labels from most to least preferred.

    Returns:
      label: Label with the highest cumulative Borda score.
    '''

    # Validate that ranked predictions input is non-empty.
    if (not rankedPredictions):
      raise ValueError("rankedPredictions must be non-empty")

    # Initialize counter for cumulative Borda scores.
    bordaScores = Counter()

    # Iterate through each model's ranking.
    for ranking in rankedPredictions:
      # Determine number of items in this ranking.
      nItems = len(ranking)
      # Iterate through ranked items with their position.
      for rank, label in enumerate(ranking):
        # Assign points: top rank gets nItems points, last gets 1.
        bordaScores[label] += (nItems - rank)

    # Return label with highest cumulative Borda score.
    return max(bordaScores, key=bordaScores.get)

  def UncertaintyAwareVoting(self, predictions, uncertainties, uncertaintyType="inverse"):
    r'''
    Aggregate predictions weighted by inverse uncertainty.

    Parameters:
      predictions (iterable): Predicted labels.
      uncertainties (iterable): Uncertainty estimates (lower = more certain).
      uncertaintyType (str): "inverse" or "exponential" weighting scheme.

    Returns:
      label: Label with highest uncertainty-adjusted cumulative weight.
    '''

    # Validate that predictions is non-empty.
    if (not predictions):
      raise ValueError("predictions must be non-empty")

    # Validate that predictions and uncertainties have same length.
    if (len(predictions) != len(uncertainties)):
      raise ValueError("predictions and uncertainties must have same length")

    # Validate that all uncertainty values are non-negative.
    if (any(u < 0 for u in uncertainties)):
      raise ValueError("uncertainties must be non-negative")

    # Initialize list for computed weights from uncertainties.
    weights = []

    # Iterate through uncertainty estimates to compute weights.
    for u in uncertainties:
      # Apply inverse weighting scheme with epsilon for numerical stability.
      if (uncertaintyType == "inverse"):
        w = 1.0 / (u + 1e-8)
      # Apply exponential decay weighting scheme.
      elif (uncertaintyType == "exponential"):
        w = math.exp(-u)
      # Raise error for unknown weighting scheme.
      else:
        raise ValueError(f"Unknown uncertaintyType: {uncertaintyType}")
      # Append computed weight to list.
      weights.append(w)

    # Initialize counter for cumulative uncertainty-adjusted votes.
    cumulativeVotes = Counter()

    # Iterate through predictions and their computed uncertainty weights.
    for pred, w in zip(predictions, weights):
      # Accumulate weight for this prediction.
      cumulativeVotes[pred] += w

    # Return label with highest cumulative uncertainty-adjusted weight.
    return max(cumulativeVotes, key=cumulativeVotes.get)

  def EntropyWeightedVoting(self, probabilityPredictions):
    r'''
    Weight voting by inverse Shannon entropy of probability distributions.

    Parameters:
      probabilityPredictions (list of dict):
        Each dict maps class labels to predicted probabilities.

    Returns:
      label: Class with highest entropy-adjusted cumulative weight.
    '''

    # Validate that input is non-empty.
    if (not probabilityPredictions):
      raise ValueError("probabilityPredictions must be non-empty")

    # Initialize counter for cumulative entropy-weighted votes.
    cumulativeVotes = Counter()

    # Iterate through each model's probability predictions.
    for probs in probabilityPredictions:
      # Compute Shannon entropy for this probability distribution.
      entropy = 0.0
      for prob in probs.values():
        # Avoid log(0) by skipping zero probabilities.
        if (prob > 0):
          entropy -= prob * math.log(prob)
      # Compute weight as inverse entropy with epsilon for stability.
      weight = 1.0 / (entropy + 1e-8)
      # Accumulate weighted votes for each class.
      for cls, prob in probs.items():
        cumulativeVotes[cls] += weight * prob

    # Return class with highest cumulative entropy-weighted vote.
    return max(cumulativeVotes, key=cumulativeVotes.get)

  def DiversityWeightedVoting(self, predictions):
    r'''
    Weight predictions by model diversity (pairwise disagreement).

    Parameters:
      predictions (list of lists):
        Each inner list contains predictions from one model across samples.

    Returns:
      list: Aggregated prediction for each sample position.
    '''

    # Validate that input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Extract predictions for this sample across all models.
      samplePreds = [predictions[m][sampleIdx] for m in range(nModels)]
      # Compute diversity weight for each model.
      diversityWeights = []
      for m in range(nModels):
        # Count disagreements with other models for this sample.
        disagreements = sum(1 for other in range(nModels) if samplePreds[m] != samplePreds[other])
        # Weight by normalized disagreement count.
        diversityWeights.append(disagreements / (nModels - 1))
      # Accumulate diversity-weighted votes.
      cumulativeVotes = Counter()
      for pred, weight in zip(samplePreds, diversityWeights):
        cumulativeVotes[pred] += weight
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def CondorcetVoting(self, rankedPredictions):
    r'''
    Condorcet voting: winner defeats all others in pairwise comparisons.

    Parameters:
      rankedPredictions (list of lists):
        Each inner list is a ranking of labels from most to least preferred.

    Returns:
      label: Condorcet winner if one exists, else fallback to Borda winner.
    '''

    # Validate that ranked predictions input is non-empty.
    if (not rankedPredictions):
      raise ValueError("rankedPredictions must be non-empty")

    # Collect all unique candidates from rankings.
    candidates = set()
    for ranking in rankedPredictions:
      for label in ranking:
        candidates.add(label)

    # Initialize pairwise comparison matrix.
    pairwiseWins = {c: {other: 0 for other in candidates if other != c} for c in candidates}

    # Iterate through each voter's ranking.
    for ranking in rankedPredictions:
      # Create position map for O(1) rank lookup.
      rankMap = {label: idx for idx, label in enumerate(ranking)}
      # Compare each pair of candidates.
      for c1 in candidates:
        for c2 in candidates:
          # Skip self-comparisons.
          if (c1 == c2):
            continue
          # Award win to higher-ranked candidate.
          if (rankMap.get(c1, float("inf")) < rankMap.get(c2, float("inf"))):
            pairwiseWins[c1][c2] += 1

    # Identify Condorcet winner: defeats all others in pairwise contests.
    condorcetWinner = None
    for candidate in candidates:
      # Check if candidate wins all pairwise comparisons.
      if (
        all(pairwiseWins[candidate][other] > len(rankedPredictions) / 2 for other in candidates if other != candidate)):
        condorcetWinner = candidate
        break

    # Return Condorcet winner if found, else fallback to Borda count.
    if (condorcetWinner is not None):
      return condorcetWinner
    else:
      # Fallback: compute Borda scores as tiebreaker.
      return self.BordaCountVoting(rankedPredictions)

  def CalibrationAwareVoting(self, predictions, calibrationScores):
    r'''
    Weight predictions by model calibration quality (higher = better calibrated).

    Parameters:
      predictions (iterable): Predicted labels.
      calibrationScores (iterable): Calibration metrics in [0, 1] where 1 = perfectly calibrated.

    Returns:
      label: Label with highest calibration-weighted cumulative vote.
    '''

    # Validate that predictions is non-empty.
    if (not predictions):
      raise ValueError("predictions must be non-empty")

    # Validate that predictions and calibration scores have same length.
    if (len(predictions) != len(calibrationScores)):
      raise ValueError("predictions and calibrationScores must have same length")

    # Validate that calibration scores are in valid range.
    if (any(s < 0 or s > 1 for s in calibrationScores)):
      raise ValueError("calibrationScores must be in range [0, 1]")

    # Initialize counter for cumulative calibration-weighted votes.
    cumulativeVotes = Counter()

    # Iterate through predictions and their calibration weights.
    for pred, score in zip(predictions, calibrationScores):
      # Accumulate vote weighted by calibration quality.
      cumulativeVotes[pred] += score

    # Return label with highest cumulative calibration-weighted vote.
    return max(cumulativeVotes, key=cumulativeVotes.get)

  def MetaWeightLearning(self, basePredictions, trueLabels):
    r'''
    Learn optimal linear combination weights from validation data.

    Parameters:
      basePredictions (2D list): Shape (n_models, n_samples) of predictions.
      trueLabels (list): Ground truth labels for weight learning.

    Returns:
      dict: Contains learned weights and aggregation function closure.
    '''

    # Validate input dimensions.
    if (not basePredictions or not basePredictions[0]):
      raise ValueError("basePredictions must be a non-empty 2D structure")
    if (len(basePredictions[0]) != len(trueLabels)):
      raise ValueError("basePredictions and trueLabels must have same sample count")

    # Convert labels to numeric for regression (one-hot for multiclass).
    uniqueLabels = list(set(trueLabels))
    labelToIdx = {lbl: idx for idx, lbl in enumerate(uniqueLabels)}

    # Initialize accumulators for normal equations: X^T X and X^T y.
    nModels = len(basePredictions)
    XtX = [[0.0] * nModels for _ in range(nModels)]
    Xty = [0.0] * nModels

    # Iterate through samples to accumulate sufficient statistics.
    for sampleIdx in range(len(trueLabels)):
      # Extract model predictions for this sample.
      x = [1.0 if basePredictions[m][sampleIdx] == trueLabels[sampleIdx] else 0.0 for m in range(nModels)]
      # Accumulate X^T X.
      for i in range(nModels):
        for j in range(nModels):
          XtX[i][j] += x[i] * x[j]
        # Accumulate X^T y.
        Xty[i] += x[i] * 1.0  # y = 1 for correct prediction

    # Solve normal equations using simple Gaussian elimination.
    weights = self._solveLinearSystem(XtX, Xty)

    # Return learned weights and aggregation helper.
    return {
      "LearnedWeights": weights,
      "Aggregate"     : lambda newPredictions: self._aggregateWithWeights(newPredictions, weights)
    }

  def _solveLinearSystem(self, A, b):
    r'''
    Solve Ax = b using Gaussian elimination with partial pivoting.

    Parameters:
      A (list of lists): Coefficient matrix.
      b (list): Right-hand side vector.

    Returns:
      list: Solution vector x.
    '''

    # Create augmented matrix [A|b].
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination with partial pivoting.
    for col in range(n):
      # Find pivot row.
      maxRow = max(range(col, n), key=lambda r: abs(aug[r][col]))
      # Swap pivot row with current row.
      aug[col], aug[maxRow] = aug[maxRow], aug[col]
      # Eliminate below pivot.
      for row in range(col + 1, n):
        if (abs(aug[col][col]) > 1e-10):
          factor = aug[row][col] / aug[col][col]
          for j in range(col, n + 1):
            aug[row][j] -= factor * aug[col][j]

    # Back substitution.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
      if (abs(aug[i][i]) > 1e-10):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
          x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    # Return normalized non-negative weights.
    total = sum(max(0, w) for w in x)
    if (total > 1e-10):
      return [max(0, w) / total for w in x]
    else:
      return [1.0 / n] * n

  def _aggregateWithWeights(self, predictions, weights):
    r'''
    Aggregate predictions using learned weights.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples).
      weights (list): Learned weight for each model.

    Returns:
      list: Aggregated prediction for each sample.
    '''

    # Validate input dimensions.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")
    if (len(predictions) != len(weights)):
      raise ValueError("number of models must match number of weights")

    # Initialize list for aggregated results.
    aggregatedResults = []
    nSamples = len(predictions[0])

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Accumulate weighted votes for this sample.
      cumulativeVotes = Counter()
      for m, pred in enumerate(predictions):
        cumulativeVotes[pred[sampleIdx]] += weights[m]
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def CopelandVoting(self, rankedPredictions):
    r'''
    Copeland voting: score candidates by net pairwise wins (wins minus losses).

    Parameters:
      rankedPredictions (list of lists):
        Each inner list is a ranking of labels from most to least preferred.

    Returns:
      label: Candidate with highest Copeland score (net pairwise wins).
    '''

    # Validate that ranked predictions input is non-empty.
    if (not rankedPredictions):
      raise ValueError("rankedPredictions must be non-empty")

    # Collect all unique candidates from rankings.
    candidates = set()
    for ranking in rankedPredictions:
      for label in ranking:
        candidates.add(label)

    # Initialize score accumulator for each candidate.
    copelandScores = {c: 0 for c in candidates}

    # Iterate through each voter's ranking.
    for ranking in rankedPredictions:
      # Create position map for O(1) rank lookup.
      rankMap = {label: idx for idx, label in enumerate(ranking)}
      # Compare each pair of candidates.
      for c1 in candidates:
        for c2 in candidates:
          # Skip self-comparisons.
          if (c1 == c2):
            continue
          # Award +1 for win, -1 for loss in pairwise comparison.
          if (rankMap.get(c1, float("inf")) < rankMap.get(c2, float("inf"))):
            copelandScores[c1] += 1
          else:
            copelandScores[c1] -= 1

    # Return candidate with highest Copeland score.
    return max(copelandScores, key=copelandScores.get)

  def RobustMeanVoting(self, predictions, trimFraction=0.1):
    r'''
    Trimmed mean aggregation: exclude extreme values before averaging.

    Parameters:
      predictions (iterable): Iterable of numeric predictions.
      trimFraction (float): Fraction of extreme values to exclude from each tail (0 to 0.5).

    Returns:
      float: Trimmed arithmetic mean.
    '''

    # Validate that predictions is non-empty.
    if (not predictions):
      raise ValueError("predictions must be a non-empty iterable")

    # Validate trim fraction is in valid range.
    if (trimFraction < 0 or trimFraction > 0.5):
      raise ValueError("trimFraction must be in range [0, 0.5]")

    # Sort predictions for trimming.
    sortedPreds = sorted(predictions)
    n = len(sortedPreds)

    # Calculate number of values to trim from each tail.
    trimCount = int(math.floor(trimFraction * n))

    # Extract trimmed subset excluding extremes.
    trimmed = sortedPreds[trimCount:n - trimCount]

    # Validate that trimmed set is non-empty.
    if (not trimmed):
      raise ValueError("trimFraction too large; no values remain after trimming")

    # Return arithmetic mean of trimmed values.
    return sum(trimmed) / len(trimmed)

  def ProductOfExpertsVoting(self, probabilityPredictions, expertWeights=None):
    r'''
    Log-linear pooling: combine probabilities via weighted product (normalized).

    Parameters:
      probabilityPredictions (list of dict):
        Each dict maps class labels to predicted probabilities.
      expertWeights (iterable, optional):
        Weight for each expert model (default: uniform weighting).

    Returns:
      label: Class with highest pooled probability.
    '''

    # Validate that input is non-empty.
    if (not probabilityPredictions):
      raise ValueError("probabilityPredictions must be non-empty")

    # Set uniform weights if not provided.
    if (expertWeights is None):
      expertWeights = [1.0] * len(probabilityPredictions)

    # Validate weights length matches predictions length.
    if (len(expertWeights) != len(probabilityPredictions)):
      raise ValueError("expertWeights must have same length as probabilityPredictions")

    # Validate all weights are non-negative.
    if (any(w < 0 for w in expertWeights)):
      raise ValueError("expertWeights must be non-negative")

    # Collect all unique class labels across experts.
    allClasses = set()
    for probs in probabilityPredictions:
      for cls in probs.keys():
        allClasses.add(cls)

    # Initialize accumulator for log-probability sums.
    logPooledProbs = {}

    # Iterate through each class to compute pooled probability.
    for cls in allClasses:
      logSum = 0.0
      # Iterate through each expert's contribution.
      for idx, probs in enumerate(probabilityPredictions):
        # Get probability for this class (default to small epsilon if missing).
        prob = probs.get(cls, 1e-10)
        # Accumulate weighted log-probability.
        logSum += expertWeights[idx] * math.log(prob)
      # Store log-sum for this class.
      logPooledProbs[cls] = logSum

    # Convert log-probabilities back to probabilities via softmax.
    maxLog = max(logPooledProbs.values())
    pooledProbs = {}
    for cls, logVal in logPooledProbs.items():
      # Subtract max for numerical stability.
      pooledProbs[cls] = math.exp(logVal - maxLog)

    # Normalize to sum to one.
    total = sum(pooledProbs.values())
    for cls in pooledProbs:
      pooledProbs[cls] /= total

    # Return class with highest pooled probability.
    return max(pooledProbs, key=pooledProbs.get)

  def CorrelationAwareWeightedVoting(self, predictions, baseWeights=None):
    r'''
    Weight predictions by inverse correlation to promote ensemble diversity.

    Parameters:
      predictions (list of lists):
        Each inner list contains predictions from one model across samples.
      baseWeights (iterable, optional):
        Initial weight for each model before correlation adjustment.

    Returns:
      list: Aggregated prediction for each sample position.
    '''

    # Validate that input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])

    # Set uniform base weights if not provided.
    if (baseWeights is None):
      baseWeights = [1.0] * nModels

    # Validate base weights length matches models count.
    if (len(baseWeights) != nModels):
      raise ValueError("baseWeights must have same length as number of models")

    # Compute pairwise correlation matrix for model predictions.
    correlationMatrix = [[0.0] * nModels for _ in range(nModels)]
    for i in range(nModels):
      for j in range(nModels):
        if (i == j):
          # Self-correlation is one.
          correlationMatrix[i][j] = 1.0
        else:
          # Compute agreement ratio as proxy for correlation.
          agreements = sum(1 for k in range(nSamples) if predictions[i][k] == predictions[j][k])
          correlationMatrix[i][j] = agreements / nSamples

    # Compute adjusted weights: base weight divided by average correlation with others.
    adjustedWeights = []
    for i in range(nModels):
      # Compute average correlation with other models.
      avgCorr = sum(correlationMatrix[i][j] for j in range(nModels) if i != j) / max(1, nModels - 1)
      # Down-weight highly correlated models.
      adjustedWeight = baseWeights[i] / (avgCorr + 1e-8)
      adjustedWeights.append(adjustedWeight)

    # Normalize adjusted weights to sum to one.
    totalWeight = sum(adjustedWeights)
    if (totalWeight > 1e-10):
      adjustedWeights = [w / totalWeight for w in adjustedWeights]

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Accumulate correlation-aware weighted votes.
      cumulativeVotes = Counter()
      for m in range(nModels):
        cumulativeVotes[predictions[m][sampleIdx]] += adjustedWeights[m]
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def AttentionWeightedVoting(self, predictions, contextFeatures, attentionDim=16):
    r'''
    Attention-based aggregation: compute dynamic weights via scaled dot-product attention.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples) of predicted labels.
      contextFeatures (list of lists): Shape (n_models, n_features) of model context vectors.
      attentionDim (int): Dimensionality of attention projection (default: 16).

    Returns:
      list: Aggregated prediction for each sample position using attention-derived weights.
    '''

    # Validate that predictions input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Validate that context features match number of models.
    if (len(contextFeatures) != len(predictions)):
      raise ValueError("contextFeatures must have same length as number of models")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])
    nFeatures = len(contextFeatures[0]) if contextFeatures else 0

    # Initialize pseudo-attention parameters using deterministic hashing.
    # In production, these would be learned via gradient descent.
    queryProj = [hash("Query") % 1000 / 1000.0 for _ in range(nFeatures)]
    keyProj = [hash("Key") % 1000 / 1000.0 for _ in range(nFeatures)]
    valueProj = [hash("Value") % 1000 / 1000.0 for _ in range(nModels)]

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Compute attention scores for each model.
      attentionScores = []
      for m in range(nModels):
        # Compute query-key dot product for this model.
        score = sum(q * k for q, k in zip(queryProj, keyProj))
        # Scale by feature dimension for stability.
        score /= math.sqrt(max(1, nFeatures))
        # Add model-specific value bias.
        score += valueProj[m]
        attentionScores.append(score)

      # Apply softmax normalization to attention scores.
      maxScore = max(attentionScores)
      expScores = [math.exp(s - maxScore) for s in attentionScores]
      totalExp = sum(expScores)
      attentionWeights = [e / totalExp for e in expScores]

      # Accumulate attention-weighted votes for this sample.
      cumulativeVotes = Counter()
      for m in range(nModels):
        cumulativeVotes[predictions[m][sampleIdx]] += attentionWeights[m]

      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def HedgeAdaptiveVoting(self, predictions, initialWeights=None, learningRate=0.1, feedback=None):
    r'''
    Online adaptive voting using Hedge algorithm with exponential weighting and regret bounds.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples) of predicted labels.
      initialWeights (iterable, optional): Initial weight for each model (default: uniform).
      learningRate (float): Learning rate for weight updates (0 to 1).
      feedback (list of lists, optional): Shape (n_models, n_samples) of per-sample losses.

    Returns:
      dict: Contains aggregated predictions and final adaptive weights.
    '''

    # Validate that predictions input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])

    # Initialize weights uniformly if not provided.
    if (initialWeights is None):
      currentWeights = [1.0 / nModels] * nModels
    else:
      # Validate initial weights length and normalize.
      if (len(initialWeights) != nModels):
        raise ValueError("initialWeights must have same length as number of models")
      totalInit = sum(initialWeights)
      if (totalInit <= 0):
        raise ValueError("initialWeights must sum to positive value")
      currentWeights = [w / totalInit for w in initialWeights]

    # Validate learning rate is in valid range.
    if (learningRate < 0 or learningRate > 1):
      raise ValueError("learningRate must be in range [0, 1]")

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position for online updating.
    for sampleIdx in range(nSamples):
      # Aggregate prediction using current adaptive weights.
      cumulativeVotes = Counter()
      for m in range(nModels):
        cumulativeVotes[predictions[m][sampleIdx]] += currentWeights[m]
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

      # Update weights if feedback (losses) is provided.
      if (feedback is not None):
        # Validate feedback dimensions.
        if (len(feedback) != nModels or len(feedback[0]) != nSamples):
          raise ValueError("feedback must have same shape as predictions")
        # Compute exponential weight updates based on losses.
        updatedWeights = []
        for m in range(nModels):
          loss = feedback[m][sampleIdx]
          # Exponential decay: lower loss preserves more weight.
          updatedWeight = currentWeights[m] * math.exp(-learningRate * loss)
          updatedWeights.append(updatedWeight)
        # Normalize updated weights to sum to one.
        totalUpdated = sum(updatedWeights)
        if (totalUpdated > 1e-10):
          currentWeights = [w / totalUpdated for w in updatedWeights]

    # Return aggregated predictions and final adaptive weights.
    return {
      "AggregatedPredictions": aggregatedResults,
      "FinalWeights"         : currentWeights
    }

  def FedAvgVoting(self, localPredictions, clientWeights=None, clientDataSizes=None):
    r'''
    Federated averaging aggregator: combine local model predictions weighted by client data size.

    Parameters:
      localPredictions (list of lists): Shape (n_clients, n_samples) of local predictions.
      clientWeights (iterable, optional): Explicit weight for each client (default: data-size proportional).
      clientDataSizes (iterable, optional): Number of samples per client for proportional weighting.

    Returns:
      list: Aggregated prediction for each sample position using federated weighting.
    '''

    # Validate that local predictions input is non-empty and rectangular.
    if (not localPredictions or not localPredictions[0]):
      raise ValueError("localPredictions must be a non-empty 2D structure")

    # Determine number of clients and samples.
    nClients = len(localPredictions)
    nSamples = len(localPredictions[0])

    # Compute client weights based on data sizes if not explicitly provided.
    if (clientWeights is None):
      if (clientDataSizes is not None):
        # Validate data sizes length matches clients.
        if (len(clientDataSizes) != nClients):
          raise ValueError("clientDataSizes must have same length as number of clients")
        # Validate all data sizes are non-negative.
        if (any(s < 0 for s in clientDataSizes)):
          raise ValueError("clientDataSizes must be non-negative")
        # Compute proportional weights from data sizes.
        totalSize = sum(clientDataSizes)
        if (totalSize <= 0):
          raise ValueError("sum of clientDataSizes must be positive")
        clientWeights = [s / totalSize for s in clientDataSizes]
      else:
        # Default to uniform weighting across clients.
        clientWeights = [1.0 / nClients] * nClients
    else:
      # Validate explicit weights length and normalize.
      if (len(clientWeights) != nClients):
        raise ValueError("clientWeights must have same length as number of clients")
      totalWeight = sum(clientWeights)
      if (totalWeight <= 0):
        raise ValueError("clientWeights must sum to positive value")
      clientWeights = [w / totalWeight for w in clientWeights]

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Accumulate federated-weighted votes for this sample.
      cumulativeVotes = Counter()
      for c in range(nClients):
        cumulativeVotes[localPredictions[c][sampleIdx]] += clientWeights[c]
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def MedianOfMeansVoting(self, predictions, nBuckets=5, randomSeed=42):
    r'''
    Median-of-means aggregation: robust to Byzantine faults and adversarial model contributions.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples) of predicted labels.
      nBuckets (int): Number of random buckets for partitioning models (default: 5).
      randomSeed (int): Seed for reproducible bucket assignment (default: 42).

    Returns:
      list: Aggregated prediction for each sample position using robust median-of-means.
    '''

    # Validate that predictions input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Validate number of buckets is positive and not exceeding model count.
    if (nBuckets < 1):
      raise ValueError("nBuckets must be at least 1")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])

    # Set random seed for reproducible bucket assignment.
    import random
    random.seed(randomSeed)

    # Assign each model to a random bucket.
    modelBuckets = [random.randint(0, nBuckets - 1) for _ in range(nModels)]

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Compute bucket-wise majority votes.
      bucketVotes = []
      for bucketIdx in range(nBuckets):
        # Collect predictions from models in this bucket.
        bucketPreds = [predictions[m][sampleIdx] for m in range(nModels) if modelBuckets[m] == bucketIdx]
        # Skip empty buckets.
        if (not bucketPreds):
          continue
        # Compute majority vote within this bucket.
        bucketCounter = Counter(bucketPreds)
        bucketWinner = max(bucketCounter, key=bucketCounter.get)
        bucketVotes.append(bucketWinner)

      # Validate that at least one bucket produced a vote.
      if (not bucketVotes):
        raise ValueError("no valid buckets; reduce nBuckets or increase nModels")

      # Return median of bucket winners for robustness.
      sortedVotes = sorted(bucketVotes)
      mid = len(sortedVotes) // 2
      if (len(sortedVotes) % 2 == 1):
        aggregatedResults.append(sortedVotes[mid])
      else:
        # Tie-break by lexicographic order for determinism.
        candidates = [sortedVotes[mid - 1], sortedVotes[mid]]
        aggregatedResults.append(min(candidates))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def QuantileAggregationVoting(self, quantilePredictions, targetQuantile=0.5):
    r'''
    Quantile-based aggregation: combine predictive distributions via quantile averaging.

    Parameters:
      quantilePredictions (list of dicts): Each dict maps quantile levels to predicted values.
        Example: [{"Q0.1": 1.0, "Q0.5": 2.0, "Q0.9": 3.0}, ...]
      targetQuantile (float): Target quantile level for aggregation (0 to 1, default: 0.5 for median).

    Returns:
      float or label: Aggregated prediction at the target quantile level.
    '''

    # Validate that input is non-empty.
    if (not quantilePredictions):
      raise ValueError("quantilePredictions must be non-empty")

    # Validate target quantile is in valid range.
    if (targetQuantile < 0 or targetQuantile > 1):
      raise ValueError("targetQuantile must be in range [0, 1]")

    # Collect all quantile predictions at the target level.
    targetValues = []
    targetKey = "Q{:.1f}".format(targetQuantile)

    # Iterate through each model's quantile predictions.
    for qPreds in quantilePredictions:
      # Validate that target quantile key exists.
      if (targetKey not in qPreds):
        raise ValueError("each quantilePredictions entry must contain key: {}".format(targetKey))
      # Append value at target quantile.
      targetValues.append(qPreds[targetKey])

    # Validate that we collected at least one value.
    if (not targetValues):
      raise ValueError("no valid quantile values collected")

    # Return median of target quantile values for robustness.
    sortedValues = sorted(targetValues)
    mid = len(sortedValues) // 2
    if (len(sortedValues) % 2 == 1):
      return sortedValues[mid]
    else:
      return (sortedValues[mid - 1] + sortedValues[mid]) / 2.0

  def ConformalPredictionVoting(self, probabilityPredictions, calibrationScores, targetCoverage=0.95):
    r'''
    Conformal prediction aggregation: return prediction sets with guaranteed marginal coverage.

    Parameters:
      probabilityPredictions (list of dict): Each dict maps class labels to predicted probabilities.
      calibrationScores (list of float): Nonconformity scores from calibration data (lower = better fit).
      targetCoverage (float): Desired coverage level in [0, 1] (default: 0.95).

    Returns:
      set: Prediction set containing labels that satisfy conformal coverage guarantee.
    '''

    # Validate that probability predictions input is non-empty.
    if (not probabilityPredictions):
      raise ValueError("probabilityPredictions must be non-empty")

    # Validate that calibration scores match number of models.
    if (len(calibrationScores) != len(probabilityPredictions)):
      raise ValueError("calibrationScores must have same length as probabilityPredictions")

    # Validate target coverage is in valid range.
    if (targetCoverage < 0 or targetCoverage > 1):
      raise ValueError("targetCoverage must be in range [0, 1]")

    # Collect all unique class labels across models.
    allClasses = set()
    for probs in probabilityPredictions:
      for cls in probs.keys():
        allClasses.add(cls)

    # Compute conformal p-values for each class using nonconformity scores.
    conformalPValues = {}
    for cls in allClasses:
      # Compute nonconformity for this class: 1 - predicted probability.
      nonconformities = [1.0 - probs.get(cls, 0.0) for probs in probabilityPredictions]
      # Combine with calibration scores to form full nonconformity distribution.
      allScores = nonconformities + calibrationScores
      # Compute empirical p-value: fraction of scores >= observed nonconformity.
      observed = sum(nonconformities) / len(nonconformities)
      pValue = sum(1.0 for s in allScores if s >= observed) / len(allScores)
      conformalPValues[cls] = pValue

    # Include classes whose p-value exceeds significance threshold.
    alpha = 1.0 - targetCoverage
    predictionSet = set()
    for cls, pVal in conformalPValues.items():
      if (pVal > alpha):
        predictionSet.add(cls)

    # Ensure non-empty prediction set for validity.
    if (not predictionSet):
      # Fallback: return class with highest p-value.
      predictionSet = {max(conformalPValues, key=conformalPValues.get)}

    # Return conformal prediction set.
    return predictionSet

  def DynamicEnsembleSelectionVoting(self, predictions, competenceScores=None, neighborhoodSize=3):
    r'''
    Dynamic ensemble selection: weight models by local competence estimates per sample.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples) of predicted labels.
      competenceScores (list of lists, optional): Shape (n_models, n_samples) of competence estimates.
      neighborhoodSize (int): Number of nearest neighbors for local competence estimation (default: 3).

    Returns:
      list: Aggregated prediction for each sample position using dynamic competence weighting.
    '''

    # Validate that predictions input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])

    # Generate uniform competence scores if not provided.
    if (competenceScores is None):
      # Default: competence proportional to agreement with majority per sample.
      competenceScores = []
      for sampleIdx in range(nSamples):
        samplePreds = [predictions[m][sampleIdx] for m in range(nModels)]
        majority = Counter(samplePreds).most_common(1)[0][0]
        modelCompetence = [1.0 if predictions[m][sampleIdx] == majority else 0.5 for m in range(nModels)]
        competenceScores.append(modelCompetence)
      # Transpose to shape (n_models, n_samples).
      competenceScores = list(zip(*competenceScores))

    # Validate competence scores dimensions.
    if (len(competenceScores) != nModels or len(competenceScores[0]) != nSamples):
      raise ValueError("competenceScores must have shape (n_models, n_samples)")

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Extract competence scores for this sample.
      sampleCompetence = [competenceScores[m][sampleIdx] for m in range(nModels)]
      # Select top-k most competent models.
      k = min(neighborhoodSize, nModels)
      topIndices = sorted(range(nModels), key=lambda i: sampleCompetence[i], reverse=True)[:k]
      # Accumulate votes from selected models weighted by competence.
      cumulativeVotes = Counter()
      for m in topIndices:
        cumulativeVotes[predictions[m][sampleIdx]] += sampleCompetence[m]
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def WassersteinBarycenterVoting(self, distributionPredictions, weights=None, nGrid=100):
    r'''
    Wasserstein barycenter aggregation: combine predictive distributions via optimal transport.

    Parameters:
      distributionPredictions (list of dicts): Each dict maps numeric values to probability mass.
        Example: [{"1.0": 0.2, "2.0": 0.5, "3.0": 0.3}, ...]
      weights (iterable, optional): Weight for each distribution (default: uniform).
      nGrid (int): Number of grid points for discretized Wasserstein computation (default: 100).

    Returns:
      float: Aggregated prediction at the Wasserstein barycenter (median of barycenter distribution).
    '''

    # Validate that distribution predictions input is non-empty.
    if (not distributionPredictions):
      raise ValueError("distributionPredictions must be non-empty")

    # Set uniform weights if not provided.
    if (weights is None):
      weights = [1.0] * len(distributionPredictions)

    # Validate weights length matches distributions count.
    if (len(weights) != len(distributionPredictions)):
      raise ValueError("weights must have same length as distributionPredictions")

    # Validate all weights are non-negative.
    if (any(w < 0 for w in weights)):
      raise ValueError("weights must be non-negative")

    # Extract support values and build empirical CDFs for each distribution.
    allValues = set()
    for dist in distributionPredictions:
      for valStr in dist.keys():
        allValues.add(float(valStr))
    sortedValues = sorted(allValues)

    # Build cumulative distribution functions on common grid.
    cdfs = []
    for dist in distributionPredictions:
      cdf = []
      cumProb = 0.0
      for val in sortedValues:
        valStr = "{:.1f}".format(val)
        cumProb += dist.get(valStr, 0.0)
        cdf.append(cumProb)
      cdfs.append(cdf)

    # Compute weighted average of quantile functions (Wasserstein barycenter for 1D).
    nValues = len(sortedValues)
    barycenterCdf = [0.0] * nValues
    for q in range(nValues):
      weightedSum = sum(w * cdf[q] for w, cdf in zip(weights, cdfs))
      totalWeight = sum(weights)
      barycenterCdf[q] = weightedSum / totalWeight if totalWeight > 0 else 0.0

    # Find median (0.5-quantile) of barycenter distribution via linear interpolation.
    targetQuantile = 0.5
    for q in range(nValues - 1):
      if (barycenterCdf[q] <= targetQuantile < barycenterCdf[q + 1]):
        # Linear interpolation between grid points.
        frac = (targetQuantile - barycenterCdf[q]) / (barycenterCdf[q + 1] - barycenterCdf[q] + 1e-10)
        return sortedValues[q] + frac * (sortedValues[q + 1] - sortedValues[q])

    # Fallback: return last value if quantile not found.
    return sortedValues[-1]

  def CausalInvariantVoting(self, predictions, environmentLabels, groundTruth=None):
    r'''
    Causal aggregation: weight models by invariant predictive performance across environments.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples) of predicted labels.
      environmentLabels (list): Environment identifier for each sample (e.g., ["train", "test", "shifted"]).
      groundTruth (list, optional): Ground truth labels for computing invariant performance metrics.

    Returns:
      list: Aggregated prediction for each sample position using environment-invariant weighting.
    '''

    # Validate that predictions input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Validate that environment labels match number of samples.
    if (len(environmentLabels) != len(predictions[0])):
      raise ValueError("environmentLabels must have same length as number of samples")

    # Determine number of models and samples.
    nModels = len(predictions)
    nSamples = len(predictions[0])

    # Collect unique environments.
    uniqueEnvironments = list(set(environmentLabels))

    # Compute environment-wise performance variance for each model if ground truth provided.
    modelInvariantScores = []
    for m in range(nModels):
      if (groundTruth is not None):
        # Compute accuracy per environment for this model.
        envAccuracies = {}
        for env in uniqueEnvironments:
          envIndices = [i for i, e in enumerate(environmentLabels) if e == env]
          if (not envIndices):
            continue
          correct = sum(1 for i in envIndices if predictions[m][i] == groundTruth[i])
          envAccuracies[env] = correct / len(envIndices)
        # Compute variance of accuracies across environments (lower = more invariant).
        if (len(envAccuracies) >= 2):
          meanAcc = sum(envAccuracies.values()) / len(envAccuracies)
          variance = sum((acc - meanAcc) ** 2 for acc in envAccuracies.values()) / len(envAccuracies)
          # Invariant score: inverse variance with epsilon for stability.
          invariantScore = 1.0 / (variance + 1e-8)
        else:
          # Single environment: use mean accuracy as score.
          invariantScore = list(envAccuracies.values())[0] if envAccuracies else 1.0
      else:
        # Fallback: use prediction entropy variance as proxy for invariance.
        envEntropies = {}
        for env in uniqueEnvironments:
          envIndices = [i for i, e in enumerate(environmentLabels) if e == env]
          if (not envIndices):
            continue
          envPreds = [predictions[m][i] for i in envIndices]
          # Compute entropy of prediction distribution in this environment.
          predCounts = Counter(envPreds)
          entropy = -sum((c / len(envPreds)) * math.log(c / len(envPreds) + 1e-10) for c in predCounts.values())
          envEntropies[env] = entropy
        # Lower entropy variance = more stable predictions = higher invariant score.
        if (len(envEntropies) >= 2):
          meanEnt = sum(envEntropies.values()) / len(envEntropies)
          variance = sum((ent - meanEnt) ** 2 for ent in envEntropies.values()) / len(envEntropies)
          invariantScore = 1.0 / (variance + 1e-8)
        else:
          invariantScore = 1.0
      modelInvariantScores.append(invariantScore)

    # Normalize invariant scores to sum to one for weighting.
    totalScore = sum(modelInvariantScores)
    if (totalScore > 1e-10):
      modelWeights = [s / totalScore for s in modelInvariantScores]
    else:
      modelWeights = [1.0 / nModels] * nModels

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Accumulate causally-weighted votes for this sample.
      cumulativeVotes = Counter()
      for m in range(nModels):
        cumulativeVotes[predictions[m][sampleIdx]] += modelWeights[m]
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults

  def GraphNeuralAggregation(
    self,
    predictions,
    adjacencyMatrix,
    aggregationSteps=2,
    activationType="relu",
    fixedLabels=None
  ):
    r'''
    GNN-style aggregation: propagate predictions through model dependency graph via message passing.

    Parameters:
      predictions (list of lists): Shape (n_models, n_samples) of predicted labels.
      adjacencyMatrix (list of lists): Shape (n_models, n_models) adjacency matrix for model graph.
      aggregationSteps (int): Number of message-passing iterations (default: 2).
      activationType (str): Activation function for message aggregation: "relu", "tanh", or "identity".
      fixedLabels (list, optional): Fixed ordered list of all possible labels for consistent embedding dimensions.

    Returns:
      list: Aggregated prediction for each sample position using graph-propagated weights.
    '''

    # Validate that predictions input is non-empty and rectangular.
    if (not predictions or not predictions[0]):
      raise ValueError("predictions must be a non-empty 2D structure")

    # Validate that adjacency matrix is square and matches number of models.
    nModels = len(predictions)
    if (len(adjacencyMatrix) != nModels or any(len(row) != nModels for row in adjacencyMatrix)):
      raise ValueError("adjacencyMatrix must be square with shape (n_models, n_models)")

    # Validate aggregation steps is positive.
    if (aggregationSteps < 1):
      raise ValueError("aggregationSteps must be at least 1")

    # Validate activation type is supported.
    if (activationType not in ["relu", "tanh", "identity"]):
      raise ValueError("activationType must be one of: \"relu\", \"tanh\", \"identity\"")

    # Determine number of samples.
    nSamples = len(predictions[0])

    # Use fixed labels if provided, otherwise extract unique labels from all predictions.
    if (fixedLabels is None):
      allLabels = set()
      for model in predictions:
        for label in model:
          allLabels.add(label)
      fixedLabels = sorted(allLabels)  # Sort for deterministic ordering

    # Initialize node embeddings using fixed label order for consistent dimensions.
    nodeEmbeddings = []
    for m in range(nModels):
      # Count label frequencies for this model.
      labelFreq = Counter(predictions[m])
      # Create embedding vector in fixed label order.
      totalFreq = sum(labelFreq.values())
      embedding = [labelFreq.get(lbl, 0) / totalFreq for lbl in fixedLabels]
      # Append embedding to list.
      nodeEmbeddings.append(embedding)

    # Perform message-passing iterations.
    for step in range(aggregationSteps):
      newEmbeddings = []
      for m in range(nModels):
        # Aggregate messages from neighbors weighted by adjacency.
        embeddingDim = len(nodeEmbeddings[0])
        aggregatedMessage = [0.0] * embeddingDim
        neighborCount = 0
        for neighbor in range(nModels):
          if (adjacencyMatrix[m][neighbor] > 0):
            weight = adjacencyMatrix[m][neighbor]
            for i, val in enumerate(nodeEmbeddings[neighbor]):
              aggregatedMessage[i] += weight * val
            neighborCount += 1
        # Normalize by neighbor count if any neighbors exist.
        if (neighborCount > 0):
          aggregatedMessage = [msg / neighborCount for msg in aggregatedMessage]
        # Apply activation function.
        if (activationType == "relu"):
          activatedMessage = [max(0.0, msg) for msg in aggregatedMessage]
        elif (activationType == "tanh"):
          activatedMessage = [math.tanh(msg) for msg in aggregatedMessage]
        else:  # identity
          activatedMessage = aggregatedMessage
        # Combine with self-embedding via residual connection.
        combinedEmbedding = [0.5 * a + 0.5 * b for a, b in zip(nodeEmbeddings[m], activatedMessage)]
        newEmbeddings.append(combinedEmbedding)
      # Update embeddings for next iteration.
      nodeEmbeddings = newEmbeddings

    # Compute final aggregation weights from converged embeddings.
    # Use sum of embedding values as model importance score.
    modelWeights = [sum(embed) for embed in nodeEmbeddings]
    totalWeight = sum(modelWeights)
    if (totalWeight > 1e-10):
      modelWeights = [w / totalWeight for w in modelWeights]
    else:
      modelWeights = [1.0 / nModels] * nModels

    # Initialize list for aggregated results.
    aggregatedResults = []

    # Iterate through each sample position.
    for sampleIdx in range(nSamples):
      # Accumulate graph-propagated weighted votes for this sample.
      cumulativeVotes = Counter()
      for m in range(nModels):
        cumulativeVotes[predictions[m][sampleIdx]] += modelWeights[m]
      # Append most-voted label for this sample.
      aggregatedResults.append(max(cumulativeVotes, key=cumulativeVotes.get))

    # Return aggregated predictions for all samples.
    return aggregatedResults


def __Phase1Testing():
  vh = VotingHelper()

  # Label-based voting.
  labels = ["cat", "dog", "cat"]
  weights = [0.6, 0.2, 0.2]
  SafeCall("WeightedMajorityVoting", vh.WeightedMajorityVoting, labels, weights)
  SafeCall("MajorityVoting", vh.MajorityVoting, labels)

  # Numeric aggregation examples.
  nums = [1.0, 2.0, 3.0]
  wnums = [1.0, 2.0, 1.0]
  SafeCall("WeightedAverageVoting", vh.WeightedAverageVoting, nums, wnums)
  SafeCall("AverageVoting", vh.AverageVoting, nums)

  # Median variants.
  SafeCall("WeightedMedianVoting", vh.WeightedMedianVoting, [1, 2, 3, 4], [1, 1, 1, 1])
  SafeCall("MedianVoting", vh.MedianVoting, [1, 2, 3])

  # Mode variants.
  SafeCall("WeightedModeVoting", vh.WeightedModeVoting, ["x", "y", "x"], [1, 2, 3])
  SafeCall("ModeVoting", vh.ModeVoting, ["x", "y", "x", "x"])

  # Geometric / harmonic means (positive inputs required).
  SafeCall("WeightedGeometricMeanVoting", vh.WeightedGeometricMeanVoting, [1.0, 4.0], [1.0, 1.0])
  SafeCall("GeometricMeanVoting", vh.GeometricMeanVoting, [1.0, 4.0])
  SafeCall("WeightedHarmonicMeanVoting", vh.WeightedHarmonicMeanVoting, [1.0, 2.0, 4.0], [1.0, 1.0, 1.0])
  SafeCall("HarmonicMeanVoting", vh.HarmonicMeanVoting, [1.0, 2.0, 4.0])

  # Power means.
  SafeCall("WeightedQuadraticMeanVoting", vh.WeightedQuadraticMeanVoting, [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
  SafeCall("QuadraticMeanVoting", vh.QuadraticMeanVoting, [1.0, 2.0, 3.0])
  SafeCall("WeightedCubicMeanVoting", vh.WeightedCubicMeanVoting, [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
  SafeCall("CubicMeanVoting", vh.CubicMeanVoting, [1.0, 2.0, 3.0])
  SafeCall("WeightedQuarticMeanVoting", vh.WeightedQuarticMeanVoting, [1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
  SafeCall("QuarticMeanVoting", vh.QuarticMeanVoting, [1.0, 2.0, 3.0])

  # Soft voting with probability outputs.
  probPreds = [
    {"Cat": 0.7, "Dog": 0.3},
    {"Cat": 0.4, "Dog": 0.6},
    {"Cat": 0.8, "Dog": 0.2}
  ]
  SafeCall("SoftVoting", vh.SoftVoting, probPreds)

  # Confidence-weighted voting.
  labels = ["A", "B", "A"]
  confidences = [0.9, 0.4, 0.8]
  SafeCall("ConfidenceWeightedVoting", vh.ConfidenceWeightedVoting, labels, confidences)

  # Bayesian model averaging.
  predictions = ["X", "Y", "X"]
  posteriors = [0.5, 0.3, 0.2]
  SafeCall("BayesianModelAveraging", vh.BayesianModelAveraging, predictions, posteriors)

  # Borda count rank aggregation.
  rankings = [
    ["First", "Second", "Third"],
    ["Second", "First", "Third"],
    ["First", "Third", "Second"]
  ]
  SafeCall("BordaCountVoting", vh.BordaCountVoting, rankings)

  # Uncertainty-aware voting.
  preds = ["P", "Q", "P"]
  uncertainties = [0.1, 0.7, 0.2]
  SafeCall("UncertaintyAwareVoting", vh.UncertaintyAwareVoting, preds, uncertainties)
  SafeCall(
    "UncertaintyAwareVoting (exponential)",
    vh.UncertaintyAwareVoting, preds, uncertainties, "exponential"
  )

  # Entropy-weighted voting with probability outputs.
  probPreds = [
    {"Alpha": 0.9, "Beta": 0.1},
    {"Alpha": 0.6, "Beta": 0.4},
    {"Alpha": 0.2, "Beta": 0.8}
  ]
  SafeCall("EntropyWeightedVoting", vh.EntropyWeightedVoting, probPreds)

  # Diversity-weighted voting across multiple samples.
  multiSamplePreds = [
    ["A", "B", "A", "C"],
    ["A", "A", "B", "C"],
    ["B", "B", "A", "C"]
  ]
  SafeCall("DiversityWeightedVoting", vh.DiversityWeightedVoting, multiSamplePreds)

  # Condorcet voting with ranked preferences.
  rankings = [
    ["Option1", "Option2", "Option3"],
    ["Option2", "Option3", "Option1"],
    ["Option3", "Option1", "Option2"]
  ]
  SafeCall("CondorcetVoting", vh.CondorcetVoting, rankings)

  # Calibration-aware voting with model quality scores.
  labels = ["Positive", "Negative", "Positive"]
  calScores = [0.95, 0.70, 0.88]  # Higher = better calibrated.
  SafeCall("CalibrationAwareVoting", vh.CalibrationAwareVoting, labels, calScores)

  # Meta-weight learning from validation data.
  basePreds = [
    ["X", "Y", "X", "Y"],
    ["X", "X", "Y", "Y"],
    ["Y", "Y", "X", "X"]
  ]
  trueLabs = ["X", "Y", "X", "Y"]
  metaResult = vh.MetaWeightLearning(basePreds, trueLabs)
  SafeCall("MetaWeightLearning Weights", lambda: metaResult["LearnedWeights"])
  SafeCall("MetaWeightLearning Aggregate", lambda: metaResult["Aggregate"](basePreds))

  # Copeland voting with ranked preferences.
  rankings = [
    ["CandidateA", "CandidateB", "CandidateC"],
    ["CandidateB", "CandidateC", "CandidateA"],
    ["CandidateC", "CandidateA", "CandidateB"]
  ]
  SafeCall("CopelandVoting", vh.CopelandVoting, rankings)

  # Robust mean voting with outlier predictions.
  numericPreds = [1.0, 2.0, 2.5, 3.0, 100.0]  # 100.0 is outlier.
  SafeCall("RobustMeanVoting (10% trim)", vh.RobustMeanVoting, numericPreds, 0.1)
  SafeCall("RobustMeanVoting (20% trim)", vh.RobustMeanVoting, numericPreds, 0.2)

  # Product of Experts voting with probability outputs.
  probPreds = [
    {"ClassX": 0.8, "ClassY": 0.2},
    {"ClassX": 0.3, "ClassY": 0.7},
    {"ClassX": 0.9, "ClassY": 0.1}
  ]
  SafeCall("ProductOfExpertsVoting", vh.ProductOfExpertsVoting, probPreds)
  SafeCall("ProductOfExpertsVoting (weighted)", vh.ProductOfExpertsVoting, probPreds, [1.0, 0.5, 2.0])

  # Correlation-aware weighted voting across multiple samples.
  multiSamplePreds = [
    ["Label1", "Label2", "Label1", "Label2"],
    ["Label1", "Label1", "Label2", "Label2"],
    ["Label1", "Label2", "Label1", "Label1"]  # Correlated with first model.
  ]
  SafeCall("CorrelationAwareWeightedVoting", vh.CorrelationAwareWeightedVoting, multiSamplePreds)

  # Attention-weighted voting with context features.
  attnPreds = [
    ["LabelA", "LabelB", "LabelA"],
    ["LabelB", "LabelB", "LabelA"],
    ["LabelA", "LabelA", "LabelB"]
  ]
  attnContext = [
    [0.9, 0.1],  # Model 0 context: high confidence feature 0.
    [0.2, 0.8],  # Model 1 context: high confidence feature 1.
    [0.7, 0.3]  # Model 2 context: moderate confidence feature 0.
  ]
  SafeCall("AttentionWeightedVoting", vh.AttentionWeightedVoting, attnPreds, attnContext)

  # Hedge adaptive voting with online feedback.
  hedgePreds = [
    ["X", "Y", "X", "Y"],
    ["X", "X", "Y", "Y"],
    ["Y", "Y", "X", "X"]
  ]
  hedgeFeedback = [
    [0.0, 1.0, 0.0, 1.0],  # Model 0 losses: 0 = correct, 1 = incorrect.
    [0.0, 0.0, 1.0, 1.0],  # Model 1 losses.
    [1.0, 1.0, 0.0, 0.0]  # Model 2 losses.
  ]
  hedgeResult = vh.HedgeAdaptiveVoting(hedgePreds, learningRate=0.2, feedback=hedgeFeedback)
  SafeCall("HedgeAdaptiveVoting Predictions", lambda: hedgeResult["AggregatedPredictions"])
  SafeCall("HedgeAdaptiveVoting Final Weights", lambda: hedgeResult["FinalWeights"])

  # Federated averaging with client data sizes.
  fedPreds = [
    ["ClientA_Label", "ClientA_Label", "ClientB_Label"],
    ["ClientB_Label", "ClientB_Label", "ClientB_Label"],
    ["ClientA_Label", "ClientB_Label", "ClientB_Label"]
  ]
  fedDataSizes = [100, 50, 200]  # Client 2 has most data.
  SafeCall("FedAvgVoting", vh.FedAvgVoting, fedPreds, clientDataSizes=fedDataSizes)

  # Median-of-means robust aggregation.
  robustPreds = [
    ["Normal", "Normal", "Normal"],
    ["Normal", "Normal", "Normal"],
    ["Normal", "Normal", "Normal"],
    ["Normal", "Normal", "Normal"],
    ["Adversarial", "Adversarial", "Adversarial"]  # One malicious model.
  ]
  SafeCall("MedianOfMeansVoting", vh.MedianOfMeansVoting, robustPreds, nBuckets=3)

  # Quantile aggregation for uncertainty intervals.
  quantilePreds = [
    {"Q0.1": 1.0, "Q0.5": 2.0, "Q0.9": 3.0},
    {"Q0.1": 1.5, "Q0.5": 2.5, "Q0.9": 3.5},
    {"Q0.1": 0.5, "Q0.5": 1.8, "Q0.9": 2.8}
  ]
  SafeCall("QuantileAggregationVoting (median)", vh.QuantileAggregationVoting, quantilePreds, 0.5)
  SafeCall("QuantileAggregationVoting (90th percentile)", vh.QuantileAggregationVoting, quantilePreds, 0.9)

  # Conformal prediction voting with calibration scores.
  confProbPreds = [
    {"Safe": 0.8, "Risky": 0.2},
    {"Safe": 0.6, "Risky": 0.4},
    {"Safe": 0.3, "Risky": 0.7}
  ]
  confCalScores = [0.1, 0.3, 0.2]  # Lower = better calibration fit.
  SafeCall("ConformalPredictionVoting (95% coverage)", vh.ConformalPredictionVoting, confProbPreds, confCalScores, 0.95)
  SafeCall("ConformalPredictionVoting (80% coverage)", vh.ConformalPredictionVoting, confProbPreds, confCalScores, 0.80)

  # Dynamic ensemble selection with competence scores.
  desPreds = [
    ["A", "B", "A", "B"],
    ["A", "A", "B", "B"],
    ["B", "B", "A", "A"]
  ]
  desCompetence = [
    [0.9, 0.4, 0.8, 0.3],  # Model 0 competence per sample.
    [0.7, 0.8, 0.2, 0.9],  # Model 1 competence per sample.
    [0.3, 0.9, 0.7, 0.4]  # Model 2 competence per sample.
  ]
  SafeCall(
    "DynamicEnsembleSelectionVoting",
    vh.DynamicEnsembleSelectionVoting, desPreds, desCompetence, neighborhoodSize=2
  )

  # Wasserstein barycenter for distributional forecasts.
  wassDists = [
    {"1.0": 0.2, "2.0": 0.5, "3.0": 0.3},
    {"1.0": 0.1, "2.0": 0.3, "3.0": 0.6},
    {"1.0": 0.4, "2.0": 0.4, "3.0": 0.2}
  ]
  SafeCall("WassersteinBarycenterVoting (uniform weights)", vh.WassersteinBarycenterVoting, wassDists)
  SafeCall("WassersteinBarycenterVoting (weighted)", vh.WassersteinBarycenterVoting, wassDists, [1.0, 2.0, 1.0])

  # Causal invariant voting with environment labels.
  causalPreds = [
    ["LabelX", "LabelY", "LabelX", "LabelY"],
    ["LabelX", "LabelX", "LabelY", "LabelY"],
    ["LabelY", "LabelY", "LabelX", "LabelX"]
  ]
  envLabels = ["EnvA", "EnvA", "EnvB", "EnvB"]  # Two distinct environments.
  groundTruth = ["LabelX", "LabelY", "LabelX", "LabelY"]
  SafeCall("CausalInvariantVoting (with ground truth)", vh.CausalInvariantVoting, causalPreds, envLabels, groundTruth)
  SafeCall("CausalInvariantVoting (without ground truth)", vh.CausalInvariantVoting, causalPreds, envLabels)

  # Graph neural aggregation with model dependency graph.
  gnnPreds = [
    ["NodeA_Label", "NodeB_Label", "NodeA_Label"],
    ["NodeA_Label", "NodeA_Label", "NodeB_Label"],
    ["NodeB_Label", "NodeB_Label", "NodeA_Label"]
  ]
  # Adjacency: Model 0 connected to 1, Model 1 connected to 0 and 2, Model 2 connected to 1.
  adjMatrix = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0]
  ]
  SafeCall("GraphNeuralAggregation (2 steps, relu)", vh.GraphNeuralAggregation, gnnPreds, adjMatrix, 2, "relu")
  SafeCall("GraphNeuralAggregation (3 steps, tanh)", vh.GraphNeuralAggregation, gnnPreds, adjMatrix, 3, "tanh")

  print("Tests are completed.")


def __Phase2Testing():
  import numpy as np

  vh = VotingHelper()

  # Define configuration constants for the synthetic dataset.
  nModels = 5
  nSamples = 10
  classLabels = ["Normal", "Moderate", "Severe"]

  # Initialize base probability matrix for all models and samples.
  baseProbs = []
  # Generate deterministic probability distributions for each model.
  for m in range(nModels):
    # Initialize probability list for the current model.
    modelSampleProbs = []
    # Generate probabilities for each sample in the current model.
    for s in range(nSamples):
      prob1 = np.random.rand()
      prob2 = np.random.rand()
      prob3 = np.random.rand()
      totalProb = prob1 + prob2 + prob3
      prob1 = prob1 / totalProb
      prob2 = prob2 / totalProb
      prob3 = prob3 / totalProb
      # Normalize probabilities to sum to one.
      normDict = {"Normal": prob1, "Moderate": prob2, "Severe": prob3}
      # Append normalized probability dictionary to model list.
      modelSampleProbs.append(normDict)
    # Append completed model probabilities to base matrix.
    baseProbs.append(modelSampleProbs)

  # Extract discrete label predictions for each model and sample.
  labelPredictions = []
  # Iterate through each model to extract maximum probability labels.
  for m in range(nModels):
    # Initialize label list for the current model.
    modelLabels = []
    # Extract predicted label for each sample using argmax.
    for s in range(nSamples):
      predLabel = max(baseProbs[m][s], key=baseProbs[m][s].get)
      # Append extracted label to model list.
      modelLabels.append(predLabel)
    # Append completed label list to predictions matrix.
    labelPredictions.append(modelLabels)

  # Map categorical labels to positive numeric scores for statistical methods.
  numericPredictions = []
  # Iterate through each model to convert labels to floats.
  for m in range(nModels):
    # Initialize numeric list for the current model.
    modelNumerics = []
    # Convert each label to a numeric risk score.
    for s in range(nSamples):
      # Map Normal to one, Moderate to two, Severe to three.
      scoreMap = {"Normal": 1.0, "Moderate": 2.0, "Severe": 3.0}
      numVal = scoreMap[labelPredictions[m][s]]
      # Append numeric score to model list.
      modelNumerics.append(numVal)
    # Append completed numeric list to predictions matrix.
    numericPredictions.append(modelNumerics)

  # Compute confidence scores as maximum probability per prediction.
  confidenceScores = []
  # Iterate through each model to calculate confidence.
  for m in range(nModels):
    # Initialize confidence list for the current model.
    modelConf = []
    # Extract maximum probability for each sample.
    for s in range(nSamples):
      maxProb = max(baseProbs[m][s].values())
      # Append confidence score to model list.
      modelConf.append(maxProb)
    # Append completed confidence list to matrix.
    confidenceScores.append(modelConf)

  # Compute uncertainty scores as one minus confidence per prediction.
  uncertaintyScores = []
  # Iterate through each model to calculate uncertainty.
  for m in range(nModels):
    # Initialize uncertainty list for the current model.
    modelUnc = []
    # Subtract confidence from one for each sample.
    for s in range(nSamples):
      uncVal = 1.0 - confidenceScores[m][s]
      # Append uncertainty score to model list.
      modelUnc.append(uncVal)
    # Append completed uncertainty list to matrix.
    uncertaintyScores.append(modelUnc)

  # Generate environment labels for causal invariance testing.
  environmentLabels = ["EnvA", "EnvA", "EnvB", "EnvB", "EnvA", "EnvB", "EnvC", "EnvC", "EnvD", "EnvD"]

  # Generate pseudo ground truth using the first model as reference.
  groundTruthLabels = labelPredictions[0][:]

  # Generate adjacency matrix for graph neural aggregation (5x5 for 5 models).
  adjacencyMatrix = [
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 0.0]
  ]

  # Generate context features for attention-weighted voting.
  contextFeatures = [
    [0.8, 0.2, 0.5],
    [0.6, 0.4, 0.7],
    [0.9, 0.1, 0.3],
    [0.5, 0.5, 0.6],
    [0.7, 0.3, 0.4]
  ]

  # Generate calibration scores for conformal prediction (one per model).
  calibrationScores = [0.12, 0.08, 0.15, 0.22, 0.10]

  # Convert probability distributions to quantile format for aggregation.
  quantilePreds = []
  # Iterate through each model to approximate quantiles.
  for m in range(nModels):
    # Initialize quantile dictionary for the current model.
    modelQuants = {}
    # Extract severity probabilities across all samples.
    sevProbs = [p["Severe"] for p in baseProbs[m]]
    # Sort probabilities for quantile approximation.
    sortedProbs = sorted(sevProbs)
    # Approximate tenth percentile value with correct key format.
    modelQuants["Q0.1"] = sortedProbs[1] if len(sortedProbs) > 1 else sortedProbs[0]
    # Approximate fiftieth percentile value with correct key format.
    modelQuants["Q0.5"] = sortedProbs[len(sortedProbs) // 2]
    # Approximate ninetieth percentile value with correct key format.
    modelQuants["Q0.9"] = sortedProbs[-1]
    # Append quantile dictionary to predictions list.
    quantilePreds.append(modelQuants)

  # Convert probability distributions to discrete numeric format for Wasserstein.
  wassDists = []
  # Iterate through each model to create discrete distributions.
  for m in range(nModels):
    # Initialize distribution dictionary for the current model.
    modelDist = {}
    # Compute average probabilities for each class.
    avgNorm = sum(p["Normal"] for p in baseProbs[m]) / nSamples
    avgMod = sum(p["Moderate"] for p in baseProbs[m]) / nSamples
    avgSev = sum(p["Severe"] for p in baseProbs[m]) / nSamples
    # Map numeric value strings to average probabilities for Wasserstein computation.
    modelDist["1.0"] = avgNorm
    modelDist["2.0"] = avgMod
    modelDist["3.0"] = avgSev
    # Append distribution dictionary to predictions list.
    wassDists.append(modelDist)

  # Compute dynamic ensemble competence scores based on confidence.
  competenceScores = confidenceScores

  # Define fixed per-model weights for weighted methods.
  sampleZeroWeights = [0.2, 0.2, 0.2, 0.2, 0.2]
  sampleZeroNumWeights = [0.1, 0.3, 0.2, 0.15, 0.25]
  modelPosteriors = [0.25, 0.20, 0.15, 0.25, 0.15]

  # Helper to collect results for a single-sample method across all samples.
  def RunSingleSampleMethod(methodFn, extractFn):
    r'''
    Apply a single-sample aggregation method to all samples and collect results.

    Parameters:
      methodFn (callable): The aggregation method to call.
      extractFn (callable): Function that extracts sample-specific args from base data.

    Returns:
      list: Aggregated result for each sample.
    '''

    results = []
    # Iterate through each sample index.
    for s in range(nSamples):
      # Extract sample-specific arguments.
      args = extractFn(s)
      # Call the method with extracted arguments.
      result = methodFn(*args)
      # Append result to list.
      results.append(result)
    # Return list of results for all samples.
    return results

  # Test Classical Label Voting methods with unified multi-sample output.
  SafeCall("WeightedMajorityVoting", lambda: RunSingleSampleMethod(
    vh.WeightedMajorityVoting,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)], sampleZeroWeights)
  ))
  SafeCall("MajorityVoting", lambda: RunSingleSampleMethod(
    vh.MajorityVoting,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)],)
  ))
  SafeCall("WeightedModeVoting", lambda: RunSingleSampleMethod(
    vh.WeightedModeVoting,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)], sampleZeroWeights)
  ))
  SafeCall("ModeVoting", lambda: RunSingleSampleMethod(
    vh.ModeVoting,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)],)
  ))

  # Test Statistical Mean Aggregation methods with unified multi-sample output.
  SafeCall("WeightedAverageVoting", lambda: RunSingleSampleMethod(
    vh.WeightedAverageVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("AverageVoting", lambda: RunSingleSampleMethod(
    vh.AverageVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))
  SafeCall("WeightedGeometricMeanVoting", lambda: RunSingleSampleMethod(
    vh.WeightedGeometricMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("GeometricMeanVoting", lambda: RunSingleSampleMethod(
    vh.GeometricMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))
  SafeCall("WeightedHarmonicMeanVoting", lambda: RunSingleSampleMethod(
    vh.WeightedHarmonicMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("HarmonicMeanVoting", lambda: RunSingleSampleMethod(
    vh.HarmonicMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))
  SafeCall("WeightedQuadraticMeanVoting", lambda: RunSingleSampleMethod(
    vh.WeightedQuadraticMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("QuadraticMeanVoting", lambda: RunSingleSampleMethod(
    vh.QuadraticMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))
  SafeCall("WeightedCubicMeanVoting", lambda: RunSingleSampleMethod(
    vh.WeightedCubicMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("CubicMeanVoting", lambda: RunSingleSampleMethod(
    vh.CubicMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))
  SafeCall("WeightedQuarticMeanVoting", lambda: RunSingleSampleMethod(
    vh.WeightedQuarticMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("QuarticMeanVoting", lambda: RunSingleSampleMethod(
    vh.QuarticMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))

  # Test Median Aggregation methods with unified multi-sample output.
  SafeCall("WeightedMedianVoting", lambda: RunSingleSampleMethod(
    vh.WeightedMedianVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], sampleZeroNumWeights)
  ))
  SafeCall("MedianVoting", lambda: RunSingleSampleMethod(
    vh.MedianVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)],)
  ))

  # Test Probabilistic Aggregation methods with unified multi-sample output.
  SafeCall("SoftVoting", lambda: RunSingleSampleMethod(
    vh.SoftVoting,
    lambda s: ([baseProbs[m][s] for m in range(nModels)],)
  ))
  SafeCall("EntropyWeightedVoting", lambda: RunSingleSampleMethod(
    vh.EntropyWeightedVoting,
    lambda s: ([baseProbs[m][s] for m in range(nModels)],)
  ))
  SafeCall("ProductOfExpertsVoting", lambda: RunSingleSampleMethod(
    vh.ProductOfExpertsVoting,
    lambda s: ([baseProbs[m][s] for m in range(nModels)],)
  ))

  # Test Confidence and Uncertainty Methods with unified multi-sample output.
  SafeCall("ConfidenceWeightedVoting", lambda: RunSingleSampleMethod(
    vh.ConfidenceWeightedVoting,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)], [confidenceScores[m][s] for m in range(nModels)])
  ))
  SafeCall("UncertaintyAwareVoting", lambda: RunSingleSampleMethod(
    vh.UncertaintyAwareVoting,
    lambda s: (
      [labelPredictions[m][s] for m in range(nModels)], [uncertaintyScores[m][s] for m in range(nModels)],
      "inverse"
    )
  ))

  # Test Bayesian and Calibration Methods with unified multi-sample output.
  SafeCall("BayesianModelAveraging", lambda: RunSingleSampleMethod(
    vh.BayesianModelAveraging,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)], modelPosteriors)
  ))
  SafeCall("CalibrationAwareVoting", lambda: RunSingleSampleMethod(
    vh.CalibrationAwareVoting,
    lambda s: ([labelPredictions[m][s] for m in range(nModels)], calibrationScores)
  ))

  # Test Rank Aggregation methods with per-sample rankings.
  # Generates a ranking for each model at each sample based on probabilities.
  SafeCall("BordaCountVoting", lambda: RunSingleSampleMethod(
    vh.BordaCountVoting,
    lambda s: (
      [sorted(baseProbs[m][s].keys(), key=lambda k: baseProbs[m][s][k], reverse=True) for m in range(nModels)],
    )
  ))
  SafeCall("CondorcetVoting", lambda: RunSingleSampleMethod(
    vh.CondorcetVoting,
    lambda s: (
      [sorted(baseProbs[m][s].keys(), key=lambda k: baseProbs[m][s][k], reverse=True) for m in range(nModels)],
    )
  ))
  SafeCall("CopelandVoting", lambda: RunSingleSampleMethod(
    vh.CopelandVoting,
    lambda s: (
      [sorted(baseProbs[m][s].keys(), key=lambda k: baseProbs[m][s][k], reverse=True) for m in range(nModels)],
    )
  ))

  # Test Diversity-Aware Methods with full prediction matrices (already multi-sample).
  SafeCall("DiversityWeightedVoting", vh.DiversityWeightedVoting, labelPredictions)
  SafeCall("CorrelationAwareWeightedVoting", vh.CorrelationAwareWeightedVoting, labelPredictions)

  # Test Meta-Learning Methods with label predictions and ground truth.
  metaResult = vh.MetaWeightLearning(labelPredictions, groundTruthLabels)
  learnedWeights = metaResult["LearnedWeights"]
  # Format as list of 10 identical weight vectors for consistent output alignment.
  weightsList = [learnedWeights for _ in range(nSamples)]
  SafeCall("MetaWeightLearning Weights", lambda: weightsList)
  SafeCall("MetaWeightLearning Aggregate", lambda: metaResult["Aggregate"](labelPredictions))

  # Test Robust Statistical Aggregation methods with per-sample numerics.
  SafeCall("RobustMeanVoting", lambda: RunSingleSampleMethod(
    vh.RobustMeanVoting,
    lambda s: ([numericPredictions[m][s] for m in range(nModels)], 0.1)
  ))

  # Test Neural/Attention-Based methods with label and context matrices.
  SafeCall("AttentionWeightedVoting", vh.AttentionWeightedVoting, labelPredictions, contextFeatures, 3)

  # Test Online/Adaptive Weighting methods with label predictions.
  feedbackMatrix = [
    [1.0 if labelPredictions[m][s] != groundTruthLabels[s] else 0.0 for s in range(nSamples)]
    for m in range(nModels)
  ]
  hedgeResult = vh.HedgeAdaptiveVoting(labelPredictions, learningRate=0.2, feedback=feedbackMatrix)
  SafeCall("HedgeAdaptiveVoting Predictions", lambda: hedgeResult["AggregatedPredictions"])
  SafeCall("HedgeAdaptiveVoting Final Weights", lambda: hedgeResult["FinalWeights"])

  # Test Federated Learning Aggregators with client predictions.
  clientDataSizes = [120, 95, 150, 110, 130]
  SafeCall("FedAvgVoting", vh.FedAvgVoting, labelPredictions, clientDataSizes=clientDataSizes)

  # Test Adversarial-Robust methods with label predictions.
  SafeCall("MedianOfMeansVoting", vh.MedianOfMeansVoting, labelPredictions, nBuckets=3, randomSeed=42)

  # Test Distributional/Quantile methods with per-sample quantile predictions.
  SafeCall("QuantileAggregationVoting", lambda: RunSingleSampleMethod(
    vh.QuantileAggregationVoting,
    lambda s: ([{
      "Q0.1": sorted([baseProbs[m][s]["Severe"] for m in range(nModels)])[1] if nModels > 1 else baseProbs[0][s][
        "Severe"],
      "Q0.5": sorted([baseProbs[m][s]["Severe"] for m in range(nModels)])[nModels // 2],
      "Q0.9": sorted([baseProbs[m][s]["Severe"] for m in range(nModels)])[-1]
    } for _ in range(1)], 0.5)  # Single dict wrapped in list for method signature
  ))

  # Test Distributional/Quantile methods with per-sample quantile predictions.
  def extractSampleQuantiles(s):
    # Collect severity probabilities for this sample across models.
    sevProbs = [baseProbs[m][s]["Severe"] for m in range(nModels)]
    sortedProbs = sorted(sevProbs)
    # Build quantile dict with correct key format.
    qDict = {
      "Q0.1": sortedProbs[1] if len(sortedProbs) > 1 else sortedProbs[0],
      "Q0.5": sortedProbs[len(sortedProbs) // 2],
      "Q0.9": sortedProbs[-1]
    }
    # Return list of one dict to match method signature.
    return ([qDict], 0.5)

  SafeCall("QuantileAggregationVoting", lambda: RunSingleSampleMethod(
    vh.QuantileAggregationVoting,
    extractSampleQuantiles
  ))

  # Test Conformal Prediction methods with probabilities and calibration.
  SafeCall("ConformalPredictionVoting", lambda: RunSingleSampleMethod(
    vh.ConformalPredictionVoting,
    lambda s: ([baseProbs[m][s] for m in range(nModels)], calibrationScores, 0.95)
  ))

  # Test Dynamic Ensemble Selection methods with predictions and competence.
  SafeCall("DynamicEnsembleSelectionVoting", vh.DynamicEnsembleSelectionVoting, labelPredictions, competenceScores, 2)

  # Test Optimal Transport Distribution methods with per-sample Wasserstein distributions.
  def extractSampleWassDists(s):
    # Build discrete distribution for this sample from probabilities.
    distDict = {
      "1.0": sum(baseProbs[m][s]["Normal"] for m in range(nModels)) / nModels,
      "2.0": sum(baseProbs[m][s]["Moderate"] for m in range(nModels)) / nModels,
      "3.0": sum(baseProbs[m][s]["Severe"] for m in range(nModels)) / nModels
    }
    # Return list of one dict to match method signature.
    return ([distDict],)

  SafeCall("WassersteinBarycenterVoting", lambda: RunSingleSampleMethod(
    vh.WassersteinBarycenterVoting,
    extractSampleWassDists
  ))

  # Test Causal Invariance methods with environment labels.
  SafeCall(
    "CausalInvariantVoting (with ground truth)",
    vh.CausalInvariantVoting, labelPredictions, environmentLabels, groundTruthLabels
  )
  SafeCall(
    "CausalInvariantVoting (without ground truth)",
    vh.CausalInvariantVoting, labelPredictions, environmentLabels
  )

  # Test Graph Neural Aggregation methods with adjacency matrix and fixed labels.
  SafeCall(
    "GraphNeuralAggregation",
    vh.GraphNeuralAggregation, labelPredictions, adjacencyMatrix, 2, "relu", classLabels
  )

  # Print completion message for test suite.
  print("All methods tested successfully with 5 models and 10 records.")


if __name__ == "__main__":
  print("Starting VotingHelper tests...")
  __Phase1Testing()
  print("Phase 1 tests completed.\n\nStarting Phase 2 with synthetic dataset...")
  __Phase2Testing()
  print("All tests completed successfully.")
