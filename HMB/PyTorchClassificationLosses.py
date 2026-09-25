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


import torch
import torch.nn as nn


class FocalLossRobust(torch.nn.Module):
  r'''
  Focal Loss implementation with class-balanced alpha for histopathology.

  Parameters:
    gamma (float): focusing parameter.
    alpha (None|float|list|Tensor): balancing factor. If None, class-balanced alpha is calculated if numClasses and classCounts are provided.
    reduction (str): "mean", "sum" or "none".
    numClasses (int|None): number of classes for automatic alpha calculation.
    classCounts (list|None): list of class counts for automatic alpha calculation.
  '''

  # Initialize the loss function.
  def __init__(self, gamma=2.0, alpha=None, reduction="mean", numClasses=None, classCounts=None):
    # Call parent constructor.
    super(FocalLossRobust, self).__init__()
    # Store gamma.
    self.gamma = gamma
    # Store reduction method.
    self.reduction = reduction

    # Calculate class-balanced alpha if not provided.
    if (alpha is None and numClasses is not None and classCounts is not None):
      # Convert counts to tensor.
      counts = torch.tensor(classCounts, dtype=torch.float32)
      # Calculate inverse frequency.
      alpha = 1.0 / (counts + 1e-6)
      # Normalize alpha.
      alpha = alpha / alpha.sum()
      # Register alpha as a buffer so it moves to device automatically.
      self.register_buffer("alpha", alpha)
    # Check if alpha is provided explicitly.
    elif (alpha is not None):
      # Convert provided alpha to tensor if it is a list.
      if (isinstance(alpha, list)):
        # Create tensor from the list.
        alpha = torch.tensor(alpha, dtype=torch.float32)
      # Register alpha as a buffer.
      self.register_buffer("alpha", alpha)
    # Handle the case where no alpha is provided or calculated.
    else:
      # Set alpha to None.
      self.alpha = None

  # Define the forward pass.
  def forward(self, inputs, targets):
    r'''
    Compute focal loss with support for soft and hard labels.

    Parameters:
      inputs (Tensor): logits of shape (N, C).
      targets (Tensor): long tensor of shape (N,) with class indices or one-hot tensor of shape (N, C).

    Returns:
      torch.Tensor: computed loss.
    '''

    # Ensure alpha is on the same device as inputs.
    if (self.alpha is not None and self.alpha.device != inputs.device):
      # Move alpha to the correct device.
      self.alpha = self.alpha.to(inputs.device)

    # Check if targets are one-hot encoded from Mixup or Cutmix.
    if (targets.dim() > 1):
      # Compute log probabilities.
      logProbs = torch.nn.functional.log_softmax(inputs, dim=1)
      # Compute cross entropy manually for soft labels.
      ceLoss = -torch.sum(targets * logProbs, dim=1)
      # Compute probabilities.
      probs = torch.softmax(inputs, dim=1)
      # Gather the probabilities of the true classes using soft labels.
      pt = torch.sum(targets * probs, dim=1)
    # Handle standard hard labels.
    else:
      # Compute cross entropy loss for hard labels.
      ceLoss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
      # Compute probabilities.
      probs = torch.softmax(inputs, dim=1)
      # Gather the probabilities of the true classes.
      pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Compute focal weight.
    focalWeight = (1 - pt) ** self.gamma

    # Apply alpha balancing if available.
    if (self.alpha is not None):
      # Check if targets are one-hot or hard labels.
      if (targets.dim() > 1):
        # Use dot product for soft labels.
        alphaFactor = torch.sum(targets * self.alpha, dim=1)
      # Handle hard labels.
      else:
        # Index alpha for hard labels.
        alphaFactor = self.alpha[targets]
      # Multiply focal weight by alpha.
      focalWeight = focalWeight * alphaFactor

    # Compute focal loss.
    focalLoss = focalWeight * ceLoss

    # Apply reduction.
    if (self.reduction == "mean"):
      # Return mean.
      return focalLoss.mean()
    # Check for sum reduction.
    elif (self.reduction == "sum"):
      # Return sum.
      return focalLoss.sum()
    # Return unreduced loss.
    return focalLoss


class WassersteinTopologicalLoss(torch.nn.Module):
  r'''
  Composite loss combining Wasserstein contrastive loss and topological regularization.

  Parameters:
    numClasses (int): number of classes for prototype initialization.
    featureDim (int): feature dimension for prototype initialization.
    epsilon (float): entropic regularization parameter for Sinkhorn.
    lambdaTopo (float): weighting factor for the topological penalty.
    lambdaContrastive (float): weighting factor for the contrastive Wasserstein loss.
  '''

  # Initialize the Wasserstein and Topological loss module.
  def __init__(self, numClasses, featureDim, epsilon=0.1, lambdaTopo=0.1, lambdaContrastive=0.1):
    # Call the parent neural network module constructor.
    super(WassersteinTopologicalLoss, self).__init__()
    # Store the number of classes for prototype initialization.
    self.numClasses = numClasses
    # Store the feature dimension for prototype initialization.
    self.featureDim = featureDim
    # Store the entropic regularization parameter for Sinkhorn.
    self.epsilon = epsilon
    # Store the weighting factor for the topological penalty.
    self.lambdaTopo = lambdaTopo
    # Store the weighting factor for the contrastive Wasserstein loss.
    self.lambdaContrastive = lambdaContrastive
    # Initialize the class prototypes as a non-trainable buffer.
    self.register_buffer("classPrototypes", torch.randn(numClasses, featureDim))
    # Initialize the prototype momentum for exponential moving average updates.
    self.prototypeMomentum = 0.99

  # Define the forward pass for the composite loss.
  def forward(self, featureMaps, targetLabels, spatialPriors):
    r'''
    Compute the composite Wasserstein and topological loss.

    Parameters:
      featureMaps (Tensor): feature maps of shape (N, D).
      targetLabels (Tensor): long tensor of shape (N,) with class indices.
      spatialPriors (Tensor): spatial prior distribution of shape (N,).

    Returns:
      dict: dictionary containing "TotalLoss", "ContrastiveLoss", and "TopologicalLoss".
    '''

    # Check if the prototypes have been initialized with the correct feature dimension.
    if (self.classPrototypes.size(1) != featureMaps.size(1)):
      # Re-initialize the class prototypes with the correct feature dimension.
      self.classPrototypes = torch.randn(self.numClasses, featureMaps.size(1), device=featureMaps.device)
    # Determine the batch size from the feature maps.
    batchSize = featureMaps.size(0)
    # Verify that the batch size is strictly greater than zero.
    if (batchSize == 0):
      # Raise a value error if the batch is empty.
      raise ValueError("Batch size cannot be zero.")
    # Calculate the contrastive Wasserstein loss against class prototypes.
    contrastiveLoss = self.ComputeWassersteinContrastiveLoss(featureMaps, targetLabels)
    # Calculate the topological regularization penalty from spatial priors.
    topoLoss = self.ComputeTopologicalPenalty(featureMaps, spatialPriors)
    # Aggregate the contrastive loss and the topological penalty.
    totalLoss = (self.lambdaContrastive * contrastiveLoss) + (self.lambdaTopo * topoLoss)
    # Return a dictionary containing the decomposed loss components.
    return {"TotalLoss": totalLoss, "ContrastiveLoss": contrastiveLoss, "TopologicalLoss": topoLoss}

  # Define the method to compute the Wasserstein contrastive loss.
  def ComputeWassersteinContrastiveLoss(self, featureMaps, targetLabels):
    r'''
    Compute the Wasserstein contrastive loss for the given features and labels.

    Parameters:
      featureMaps (Tensor): feature maps of shape (N, D).
      targetLabels (Tensor): long tensor of shape (N,) with class indices.

    Returns:
      torch.Tensor: normalized contrastive loss value.
    '''

    # Initialize a list to collect losses for each class to avoid inplace accumulation issues.
    lossList = []

    # Iterate over each unique class present in the current batch.
    for currentLabel in torch.unique(targetLabels):
      # Create a boolean mask for the current class.
      classMask = (targetLabels == currentLabel)
      # Extract the feature maps belonging to the current class.
      classFeatures = featureMaps[classMask]

      # Retrieve a cloned prototype for the current class to decouple it from buffer version tracking.
      prototype = self.classPrototypes[currentLabel].clone().unsqueeze(0)

      # Compute the cost matrix between class features and the prototype.
      costMatrix = torch.cdist(classFeatures, prototype, p=2)

      # Compute the regularized optimal transport plan using Sinkhorn iterations.
      transportPlan = self.SinkhornIteration(costMatrix, self.epsilon)

      # Calculate the Wasserstein distance for the positive class.
      positiveCost = torch.sum(transportPlan * costMatrix)

      # Create a mask to exclude the positive class for negative sampling.
      negativeMask = torch.ones(self.numClasses, dtype=torch.bool, device=featureMaps.device)
      # Set the positive class index to false in the negative mask.
      negativeMask[currentLabel] = False

      # Initialize class loss with the positive cost.
      classLoss = positiveCost

      # Check if there are negative classes available.
      if (negativeMask.sum() > 0):
        # Extract cloned negative prototypes to prevent version tracking issues.
        negativePrototypes = self.classPrototypes[negativeMask].clone()
        # Compute the cost matrix between class features and negative prototypes.
        negativeCostMatrix = torch.cdist(classFeatures, negativePrototypes, p=2)
        # Compute the optimal transport plan for negative prototypes.
        negativeTransportPlan = self.SinkhornIteration(negativeCostMatrix, self.epsilon)
        # Calculate the minimum Wasserstein distance among negative classes.
        minNegativeCost = torch.min(torch.sum(negativeTransportPlan * negativeCostMatrix, dim=1))
        # Define the margin for the hinge loss formulation.
        margin = 1.0
        # Add the hinge loss for the current class.
        classLoss = classLoss + torch.relu(positiveCost - minNegativeCost + margin)

      # Append the computed loss for this class.
      lossList.append(classLoss)

      # Update the class prototype using exponential moving average.
      self.UpdatePrototype(currentLabel, classFeatures.detach())

    # Stack and average the losses to safely compute the final value.
    if (len(lossList) > 0):
      # Calculate the mean of the stacked losses.
      normalizedContrastiveLoss = torch.stack(lossList).mean()
    # Handle the case where the loss list is empty.
    else:
      # Create a zero tensor with gradient tracking enabled.
      normalizedContrastiveLoss = torch.tensor(0.0, device=featureMaps.device, requires_grad=True)

    # Return the normalized contrastive loss value.
    return normalizedContrastiveLoss

  # Define the method to update class prototypes.
  @torch.no_grad()
  def UpdatePrototype(self, classIndex, classFeatures):
    r'''
    Update the class prototype using exponential moving average.

    Parameters:
      classIndex (int): index of the class to update.
      classFeatures (Tensor): feature maps of the current class batch.
    '''

    # Compute the mean feature vector for the current class batch.
    batchMean = classFeatures.mean(dim=0)
    # Compute the new prototype value.
    newVal = (self.prototypeMomentum * self.classPrototypes.data[classIndex]) + (
        (1 - self.prototypeMomentum) * batchMean)
    # Use copy_ to safely update the buffer data without triggering autograd version errors.
    self.classPrototypes.data[classIndex].copy_(newVal)

  # Define the Sinkhorn iteration method for optimal transport.
  def SinkhornIteration(self, costMatrix, epsilon):
    r'''
    Compute the regularized optimal transport plan using Sinkhorn iterations.

    Parameters:
      costMatrix (Tensor): cost matrix of shape (N, M).
      epsilon (float): entropic regularization parameter.

    Returns:
      torch.Tensor: regularized optimal transport plan.
    '''

    # Compute the Gibbs kernel matrix from the cost matrix and regularization parameter.
    kernelMatrix = torch.exp(-costMatrix / epsilon)
    # Add a small epsilon to prevent division by zero.
    kernelMatrix = kernelMatrix + 1e-8
    # Initialize the dual variable u as a uniform distribution.
    u = torch.ones(kernelMatrix.size(0), 1, device=costMatrix.device) / kernelMatrix.size(0)
    # Initialize the dual variable v as a uniform distribution.
    v = torch.ones(kernelMatrix.size(1), 1, device=costMatrix.device) / kernelMatrix.size(1)
    # Iterate a fixed number of times to converge the Sinkhorn algorithm.
    for iteration in range(5):
      # Update the dual variable u based on the current v and kernel matrix.
      u = 1.0 / torch.matmul(kernelMatrix, v)
      # Update the dual variable v based on the current u and transposed kernel matrix.
      v = 1.0 / torch.matmul(kernelMatrix.T, u)
    # Construct the final transport plan using the converged dual variables.
    transportPlan = u * kernelMatrix * v.T
    # Return the regularized optimal transport plan.
    return transportPlan

  # Define the method to compute the topological penalty.
  def ComputeTopologicalPenalty(self, featureMaps, spatialPriors):
    r'''
    Compute the topological regularization penalty from spatial priors.

    Parameters:
      featureMaps (Tensor): feature maps of shape (N, D, H, W).
      spatialPriors (Tensor): spatial prior distribution of shape (N, H, W).

    Returns:
      torch.Tensor: computed topological penalty.
    '''

    # Compute the L2 norm of the feature maps to represent activation intensity.
    spatialActivations = torch.norm(featureMaps, p=2, dim=1)
    # Normalize the spatial activations to form a probability distribution.
    spatialDistribution = torch.nn.functional.softmax(spatialActivations, dim=0)
    # Compute the Wasserstein distance proxy between the spatial distribution and the prior.
    topoPenalty = torch.sum(torch.abs(spatialDistribution - spatialPriors))
    # Return the computed topological penalty.
    return topoPenalty


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
