import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWrapper(nn.Module):
  r'''
  Thin wrapper around torch.nn.CrossEntropyLoss to keep a consistent API.

  Parameters:
    weight (Tensor, optional): a manual rescaling weight given to each class.
    reduction (str): "mean" (default), "sum" or "none".
  '''

  def __init__(self, classWeight=None, reductionMode="mean"):
    super(CrossEntropyLossWrapper, self).__init__()
    # Store the reduction mode.
    self.reductionMode = reductionMode
    # Create the internal cross entropy loss function.
    self.lossFn = nn.CrossEntropyLoss(weight=classWeight, reduction=reductionMode)

  def forward(self, inputTensor, targetTensor):
    r'''
    Compute cross-entropy loss for multi-class classification.

    Parameters:
      inputTensor (Tensor): logits of shape (N, C).
      targetTensor (Tensor): long tensor of shape (N,) with class indices.

    Returns:
      torch.Tensor: computed loss.
    '''

    # Delegate to the internal loss function.
    return self.lossFn(inputTensor, targetTensor)


class LabelSmoothingCrossEntropy(nn.Module):
  r'''
  Cross entropy with label smoothing.

  The loss is computed on raw logits for numerical stability.

  Parameters:
    smoothing (float): label smoothing factor in [0, 1). Typical values 0.0 - 0.2.
    reduction (str): "mean", "sum" or "none".
  '''

  def __init__(self, labelSmoothing: float = 0.1, reductionMode: str = "mean"):
    super(LabelSmoothingCrossEntropy, self).__init__()
    # Validate smoothing value.
    assert (0.0 <= labelSmoothing < 1.0)
    self.labelSmoothing = labelSmoothing
    self.reductionMode = reductionMode

  def forward(self, inputTensor, targetTensor):
    r'''
    Compute label-smoothed cross-entropy loss.

    Parameters:
      inputTensor (Tensor): logits of shape (N, C).
      targetTensor (Tensor): long tensor of shape (N,) with class indices.

    Returns:
      torch.Tensor: computed loss.
    '''

    # Compute log probabilities for numerical stability.
    logProbs = F.log_softmax(inputTensor, dim=1)
    # Number of classes.
    nClasses = inputTensor.size(1)

    # Create smoothed target distribution.
    with torch.no_grad():
      trueDist = torch.zeros_like(logProbs)
      # Fill with the smoothing value for non-target classes.
      trueDist.fill_(self.labelSmoothing / (nClasses - 1))
      # Place the remaining mass on the true class.
      trueDist.scatter_(1, targetTensor.data.unsqueeze(1), 1.0 - self.labelSmoothing)

    # Compute per-sample loss as negative log-likelihood under smoothed targets.
    lossTensor = -torch.sum(trueDist * logProbs, dim=1)

    if (self.reductionMode == "mean"):
      return lossTensor.mean()
    elif (self.reductionMode == "sum"):
      return lossTensor.sum()
    else:
      return lossTensor


class BinaryFocalLoss(nn.Module):
  r'''
  Focal loss for binary classification (uses logits for numerical stability).

  .. math::

    \text{FL}(p_t) = -\alpha (1 - p_t)^{\gamma} \log(p_t)

  Parameters:
    alpha (float): balancing factor for the positive class (default 0.25).
    gamma (float): focusing parameter (default 2.0).
    reduction (str): "mean", "sum" or "none".
  '''

  def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reductionMode: str = "mean"):
    super(BinaryFocalLoss, self).__init__()
    # Store focal parameters.
    self.alpha = alpha
    self.gamma = gamma
    self.reductionMode = reductionMode

  def forward(self, inputTensor, targetTensor):
    r'''
    Compute binary focal loss.

    Parameters:
      inputTensor (Tensor): logits of shape (N,).
      targetTensor (Tensor): float tensor of shape (N,) with binary labels (0 or 1).

    Returns:
      torch.Tensor: computed loss.
    '''

    # Compute element-wise binary cross entropy with logits.
    bceLoss = F.binary_cross_entropy_with_logits(inputTensor, targetTensor, reduction="none")

    # Convert logits to probabilities.
    probTensor = torch.sigmoid(inputTensor)
    probTensor = probTensor.view(-1)
    targetTensor = targetTensor.view(-1)

    # Probability of the true class per example.
    probT = torch.where(targetTensor == 1, probTensor, 1 - probTensor)

    # Per-sample alpha factor depending on the target label.
    alphaFactor = torch.where(
      targetTensor == 1,
      self.alpha * torch.ones_like(targetTensor),
      (1.0 - self.alpha) * torch.ones_like(targetTensor)
    )

    # Focal modulation factor.
    focalFactor = alphaFactor * (1 - probT) ** self.gamma

    # Apply modulation to the base BCE loss.
    lossTensor = focalFactor * bceLoss.view(-1)

    if (self.reductionMode == "mean"):
      return lossTensor.mean()
    elif (self.reductionMode == "sum"):
      return lossTensor.sum()
    else:
      return lossTensor


class FocalLoss(nn.Module):
  r'''
  Multi-class focal loss (works with logits).

  Parameters:
    gamma (float): focusing parameter.
    alpha (None|float|list|Tensor): balancing factor. If None no class weighting is used.
      If float is provided it is assumed to be the weight for the class 1 in binary case.
      For multi-class you can pass a list/torch.Tensor of length C with class weights.
    reduction (str): "mean", "sum" or "none".
  '''

  def __init__(self, gamma: float = 2.0, alpha=None, reductionMode: str = "mean"):
    super(FocalLoss, self).__init__()
    # Store parameters.
    self.gamma = gamma
    self.reductionMode = reductionMode

    if (alpha is None):
      self.alpha = None
    else:
      if (isinstance(alpha, (float, int))):
        self.alpha = float(alpha)
      else:
        # Use as_tensor to avoid copying from existing tensors and suppress UserWarning
        self.alpha = torch.as_tensor(alpha, dtype=torch.float)

  def forward(self, inputTensor, targetTensor):
    r'''
    Compute multi-class focal loss.

    Parameters:
      inputTensor (Tensor): logits of shape (N, C).
      targetTensor (Tensor): long tensor of shape (N,) with class indices.

    Returns:
      torch.Tensor: computed loss.
    '''

    # Compute log-probabilities and probabilities.
    logProbs = F.log_softmax(inputTensor, dim=1)
    probTensor = torch.exp(logProbs)

    targetTensor = targetTensor.view(-1)

    # Gather log-probability of the true class per example.
    logPt = logProbs.gather(1, targetTensor.unsqueeze(1)).squeeze(1)
    # Gather probability of the true class per example.
    probT = probTensor.gather(1, targetTensor.unsqueeze(1)).squeeze(1)

    if (self.alpha is None):
      alphaFactor = torch.ones_like(probT)
    else:
      if (isinstance(self.alpha, float)):
        # Binary case: build [1-alpha, alpha] tensor if we have two classes.
        alphaTensor = torch.as_tensor(
          [1.0 - self.alpha, self.alpha],
          device=inputTensor.device,
          dtype=inputTensor.dtype,
        ) if (inputTensor.size(1) == 2) else None
        if (alphaTensor is not None):
          alphaFactor = alphaTensor[targetTensor]
        else:
          # Fallback to scalar alpha for non-binary cases.
          alphaFactor = torch.full_like(probT, fill_value=self.alpha)
      else:
        # Use per-class weights for alpha.
        alphaVec = self.alpha.to(device=inputTensor.device, dtype=inputTensor.dtype)
        alphaFactor = alphaVec[targetTensor]

    # Focal modulation factor.
    focalFactor = (1 - probT) ** self.gamma

    # Final per-sample focal loss.
    lossTensor = -alphaFactor * focalFactor * logPt

    if (self.reductionMode == "mean"):
      return lossTensor.mean()
    elif (self.reductionMode == "sum"):
      return lossTensor.sum()
    else:
      return lossTensor


class FocalLossAlt(nn.Module):
  r'''
  Focal loss for handling class imbalance in binary/multi-class classification.

  Down-weights easy examples and focuses training on hard negatives.
  Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

  Parameters:
    gamma (float): Focusing parameter that down-weights easy examples (typical: 2.0).
    weight (torch.Tensor or None): Optional per-class weights for imbalance handling.
    reduction (str): Reduction method: "mean", "sum", or "none".
  '''

  def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
    # Call superclass constructor.
    super(FocalLossAlt, self).__init__()
    # Store focal loss hyperparameters.
    self.gamma = gamma
    self.weight = weight
    self.reduction = reduction

  def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Expects inputs: Logits tensor of shape (batch_size, numClasses).
    # Expects targets: Class indices tensor of shape (batch_size,).
    # Returns: Loss tensor of shape () or (batch_size,) depending on reduction.
    # Compute log-probabilities with numerical stability.
    logProb = F.log_softmax(inputs, dim=1)
    # Gather log-probabilities for target classes.
    targetsLong = targets.long()
    logpt = logProb[torch.arange(targetsLong.size(0), device=targetsLong.device), targetsLong]
    # Convert to probability for focal weighting.
    pt = logpt.exp()
    # Compute focal loss per sample: -(1-pt)^gamma * log(pt).
    loss = -((1 - pt) ** self.gamma) * logpt
    # Apply class weights if provided.
    if (self.weight is not None):
      weight = self.weight.to(inputs.device) if (self.weight.device != inputs.device) else self.weight
      perSampleWeight = weight[targetsLong]
      loss = loss * perSampleWeight
    # Apply reduction method.
    if (self.reduction == "mean"):
      return loss.mean()
    if (self.reduction == "sum"):
      return loss.sum()
    return loss


if __name__ == "__main__":
  # Quick smoke tests for the implemented losses.
  # Multi-class example.
  logits = torch.randn(4, 3)
  targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

  ce = CrossEntropyLossWrapper()
  ls = LabelSmoothingCrossEntropy(labelSmoothing=0.1)
  focal = FocalLoss(gamma=2.0, alpha=None)

  # Call .forward() explicitly to satisfy static analyzers and be explicit.
  print(f"CrossEntropy: {ce.forward(logits, targets).item():.6f}")
  print(f"LabelSmoothed CE: {ls.forward(logits, targets).item():.6f}")
  print(f"Focal (multiclass): {focal.forward(logits, targets).item():.6f}")

  # Binary example.
  bLogits = torch.randn(6)
  bTargets = torch.randint(0, 2, (6,)).float()
  bf = BinaryFocalLoss()
  print(f"Binary Focal: {bf.forward(bLogits, bTargets).item():.6f}")
