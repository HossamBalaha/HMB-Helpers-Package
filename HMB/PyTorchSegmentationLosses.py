'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Aug 16th, 2025
# Last Modification Date: Aug 19th, 2025
# Permissions and Citation: Refer to the README file.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
  r'''
  Implements Dice Loss for binary segmentation tasks.
  Dice loss measures the overlap between predicted and ground truth masks.

  Parameters:
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.

  .. math::
    \text{Dice}=1-\frac{2 \times |X \cap Y| + \text{smooth}}{|X| + |Y| + \text{smooth}}
  '''

  def __init__(self, weight=None, size_average=True):
    # Call the parent class constructor to initialize the module.
    super(DiceLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    '''
    Computes the Dice loss between predictions and targets for binary segmentation.

    Parameters:
      inputs (torch.Tensor): Model outputs (logits or probabilities).
      targets (torch.Tensor): Ground truth binary mask.
      smooth (float, optional): Smoothing constant to avoid division by zero. Default is 1.

    Returns:
      torch.Tensor: Dice loss value.
    '''

    # Apply sigmoid activation to convert logits to probabilities.
    inputs = torch.sigmoid(inputs)
    # Flatten the input tensor to 1D for calculation.
    inputs = inputs.view(-1)
    # Flatten the target tensor to 1D for calculation.
    targets = targets.view(-1)
    # Calculate the intersection between inputs and targets.
    intersection = (inputs * targets).sum()
    # Compute Dice loss using the formula.
    diceLoss = 1.0 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    # Return the Dice loss value.
    return diceLoss


class DiceBCELoss(nn.Module):
  r'''
  Implements Dice + BCE Loss for binary segmentation tasks.
  Combines Dice loss and binary cross-entropy loss for improved performance on imbalanced data.

  Parameters:
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.

  Note:
    For best practice and autocasting safety, use raw logits as inputs.

  .. math::
    \text{Loss} = \text{BCE}(X, Y) + \left[1 - \frac{2 \times |X \cap Y| + \text{smooth}}{|X| + |Y| + \text{smooth}}\right]
  '''

  def __init__(self, weight=None, size_average=True):
    # Call the parent class constructor to initialize the module.
    super(DiceBCELoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    '''
    Computes the sum of Dice loss and BCE loss for binary segmentation.

    Parameters:
      inputs (torch.Tensor): Model outputs (raw logits).
      targets (torch.Tensor): Ground truth binary mask.
      smooth (float, optional): Smoothing constant to avoid division by zero. Default is 1.

    Returns:
      torch.Tensor: Combined Dice + BCE loss value.
    '''

    # Compute BCE loss directly from logits for autocasting safety.
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    # Apply sigmoid activation to convert logits to probabilities.
    inputs = torch.sigmoid(inputs)
    # Flatten the input tensor to 1D for calculation.
    inputs = inputs.view(-1)
    # Flatten the target tensor to 1D for calculation.
    targets = targets.view(-1)
    # Calculate the intersection between inputs and targets.
    intersection = (inputs * targets).sum()
    # Compute Dice loss using the formula.
    diceLoss = 1.0 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    # Return the sum of BCE and Dice loss values.
    return bce + diceLoss


class JaccardLoss(nn.Module):
  r'''
  Implements Jaccard Loss (IoU Loss) for binary segmentation tasks.
  Jaccard loss measures the intersection over union between predicted and ground truth masks.

  Parameters:
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.

  .. math::
    \text{Jaccard} = 1 - \frac{|X \cap Y| + \text{smooth}}{|X \cup Y| + \text{smooth}}
  '''

  def __init__(self, weight=None, size_average=True):
    # Call the parent class constructor to initialize the module.
    super(JaccardLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    '''
    Computes the Jaccard loss (1 - IoU) between predictions and targets for binary segmentation.

    Parameters:
      inputs (torch.Tensor): Model outputs (logits or probabilities).
      targets (torch.Tensor): Ground truth binary mask.
      smooth (float, optional): Smoothing constant to avoid division by zero. Default is 1.

    Returns:
      torch.Tensor: Jaccard loss value.
    '''

    # Apply sigmoid activation to convert logits to probabilities.
    inputs = torch.sigmoid(inputs)
    # Flatten the input tensor to 1D for calculation.
    inputs = inputs.view(-1)
    # Flatten the target tensor to 1D for calculation.
    targets = targets.view(-1)
    # Calculate the intersection between inputs and targets.
    intersection = (inputs * targets).sum()
    # Calculate the union between inputs and targets.
    total = (inputs + targets).sum()
    union = total - intersection
    # Compute Jaccard index and loss.
    jaccard = (intersection + smooth) / (union + smooth)
    # Return the Jaccard loss value.
    return 1.0 - jaccard


class TverskyLoss(nn.Module):
  r'''
  Implements Tversky Loss for binary segmentation tasks.
  Tversky loss generalizes Dice loss by allowing control over penalties for false positives and false negatives.

  Parameters:
    alpha (float): Weight for false positives.
    beta (float): Weight for false negatives.
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.

  .. math::
    \text{Tversky} = 1 - \frac{|X \cap Y| + \text{smooth}}{|X \cap Y| + \alpha \times |X \setminus Y| + \beta \times |Y \setminus X| + \text{smooth}}
  '''

  def __init__(self, alpha=0.5, beta=0.5, weight=None, size_average=True):
    # Store alpha and beta parameters for loss calculation.
    super(TverskyLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta

  def forward(self, inputs, targets, smooth=1):
    '''
    Computes the Tversky loss between predictions and targets for binary segmentation.

    Parameters:
      inputs (torch.Tensor): Model outputs (logits or probabilities).
      targets (torch.Tensor): Ground truth binary mask.
      smooth (float, optional): Smoothing constant to avoid division by zero. Default is 1.

    Returns:
      torch.Tensor: Tversky loss value.
    '''

    # Apply sigmoid activation to convert logits to probabilities.
    inputs = torch.sigmoid(inputs)
    # Flatten the input tensor to 1D for calculation.
    inputs = inputs.view(-1)
    # Flatten the target tensor to 1D for calculation.
    targets = targets.view(-1)
    # Calculate true positives.
    truePos = (inputs * targets).sum()
    # Calculate false negatives.
    falseNeg = ((1.0 - inputs) * targets).sum()
    # Calculate false positives.
    falsePos = (inputs * (1 - targets)).sum()
    # Compute Tversky index and loss.
    tversky = (truePos + smooth) / (truePos + self.alpha * falsePos + self.beta * falseNeg + smooth)
    # Return the Tversky loss value.
    return 1.0 - tversky


class FocalLoss(nn.Module):
  r'''
  Implements Focal Loss for binary segmentation tasks.
  Focal loss focuses training on hard examples and addresses class imbalance.

  Parameters:
    alpha (float): Weighting factor for the rare class. Default is 0.25.
    gamma (float): Focusing parameter for modulating factor (1 - p_t). Default is 2.0.

  .. math::
    \text{Focal}(p_t) = -\alpha \times (1 - p_t)^{\gamma} \times \log(p_t)
  '''

  def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
    # Store alpha, gamma, and reduction parameters for loss calculation.
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, inputs, targets):
    '''
    Computes the Focal loss between predictions and targets for binary segmentation.

    Parameters:
      inputs (torch.Tensor): Model outputs (logits).
      targets (torch.Tensor): Ground truth binary mask.

    Returns:
      torch.Tensor: Focal loss value.
    '''

    # Apply sigmoid activation to convert logits to probabilities.
    probs = torch.sigmoid(inputs)
    # Flatten tensors for calculation.
    probs = probs.view(-1)
    targets = targets.view(-1)
    # Compute the binary cross-entropy loss.
    bceLoss = F.binary_cross_entropy(probs, targets, reduction="none")
    # Compute the modulating factor.
    pt = torch.where(targets == 1, probs, 1 - probs)
    focalFactor = self.alpha * (1 - pt) ** self.gamma
    # Compute the focal loss.
    loss = focalFactor * bceLoss
    # Return the reduced loss value.
    if (self.reduction == "mean"):
      return loss.mean()
    elif (self.reduction == "sum"):
      return loss.sum()
    else:
      return loss


class GeneralizedDiceLoss(nn.Module):
  r'''
  Implements Generalized Dice Loss for multi-class segmentation tasks.
  Weights each class inversely to its frequency to address class imbalance.

  .. math::
    \text{Generalized\ Dice} = 1 - \frac{2 \times \sum_c w_c \sum_i p_{ci} \times g_{ci}}{\sum_c w_c \sum_i (p_{ci} + g_{ci})}
    \quad \text{where} \quad w_c = \frac{1}{(\sum_i g_{ci})^2}
  '''

  def __init__(self, epsilon=1e-6):
    # Store epsilon for numerical stability.
    super(GeneralizedDiceLoss, self).__init__()
    self.epsilon = epsilon

  def forward(self, inputs, targets):
    '''
    Computes the Generalized Dice loss for multi-class segmentation.

    Parameters:
      inputs (torch.Tensor): Model outputs (logits) of shape (N, C, ...).
      targets (torch.Tensor): Ground truth one-hot mask of shape (N, C, ...).

    Returns:
      torch.Tensor: Generalized Dice loss value.
    '''

    # Apply softmax activation to convert logits to probabilities.
    inputs = torch.softmax(inputs, dim=1)
    # Flatten spatial dimensions.
    inputs = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], -1)
    targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)
    # Compute class weights inversely proportional to ground truth volume.
    w = 1.0 / (torch.sum(targets, dim=2) ** 2 + self.epsilon)
    # Compute intersection and union for each class.
    intersection = torch.sum(inputs * targets, dim=2)
    union = torch.sum(inputs + targets, dim=2)
    # Compute generalized dice score.
    numerator = torch.sum(w * intersection, dim=1)
    denominator = torch.sum(w * union, dim=1)
    diceScore = 2.0 * numerator / (denominator + self.epsilon)
    # Return the mean dice loss over the batch.
    return 1.0 - diceScore.mean()


if __name__ == "__main__":
  # Example usage for all loss functions.
  # Simulate model output tensor with random values (logits).
  predictions = torch.randn(1, 1, 256, 256).float()
  # Simulate ground truth mask tensor with binary values.
  targets = torch.randint(0, 2, (1, 1, 256, 256)).float()
  # Instantiate each loss class.
  diceLoss = DiceLoss()
  diceBceLoss = DiceBCELoss()
  jaccardLoss = JaccardLoss()
  tverskyLoss = TverskyLoss(alpha=0.7, beta=0.3)
  focalLoss = FocalLoss()
  generalizedDiceLoss = GeneralizedDiceLoss()

  # Compute and print each loss value.
  print(f"Dice Loss: {diceLoss(predictions, targets).item()}")
  print(f"Dice + BCE Loss: {diceBceLoss(predictions, targets).item()}")
  print(f"Jaccard Loss: {jaccardLoss(predictions, targets).item()}")
  print(f"Tversky Loss: {tverskyLoss(predictions, targets).item()}")
  # Example for Focal Loss (binary).
  print(f"Focal Loss: {focalLoss(predictions, targets).item()}")

  # Example for Generalized Dice Loss (multi-class).
  # Simulate multi-class logits and one-hot targets.
  mcLogits = torch.randn(2, 3, 256, 256)
  mcTargets = torch.zeros(2, 3, 256, 256)
  mcTargets[:, 0] = (torch.randint(0, 2, (2, 256, 256)) == 0).float()
  mcTargets[:, 1] = (torch.randint(0, 2, (2, 256, 256)) == 1).float()
  mcTargets[:, 2] = (torch.randint(0, 2, (2, 256, 256)) == 2).float()
  print(f"Generalized Dice Loss: {generalizedDiceLoss(mcLogits, mcTargets).item()}")
