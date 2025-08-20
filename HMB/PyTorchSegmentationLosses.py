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
  '''
  Dice Loss for binary segmentation.
  Parameters:
    smooth (float): Smoothing constant to avoid division by zero.
  '''

  def __init__(self, weight=None, size_average=True):
    # Initialize the DiceLoss class by calling the parent class constructor.
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

    # Use sigmoid activation if inputs are logits
    inputs = torch.sigmoid(inputs)

    # Flatten the input and target tensors into 1D arrays for computation.
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Compute the intersection of the predicted and target tensors.
    intersection = (inputs * targets).sum()

    # Calculate the Dice loss using the formula.
    diceLoss = 1.0 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    # Return the Dice loss value.
    return diceLoss


class DiceBCELoss(nn.Module):
  '''
  Dice + BCE Loss for binary segmentation.
  Parameters:
    smooth (float): Smoothing constant to avoid division by zero.
  Note:
    For best practice and autocasting safety, use raw logits as inputs.
  '''

  def __init__(self, weight=None, size_average=True):
    # Initialize the DiceBCELoss class by calling the parent class constructor.
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

    # Use BCE with logits for safe autocasting.
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

    # Apply a sigmoid activation function to the inputs if the model does not include one.
    inputs = torch.sigmoid(inputs)

    # Flatten the input and target tensors into 1D arrays for computation.
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Compute the intersection of the predicted and target tensors.
    intersection = (inputs * targets).sum()

    # Calculate the Dice loss component using the formula.
    diceLoss = 1.0 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    # Combine the Dice loss and BCE loss to form the final loss value.
    DiceBCE = bce + diceLoss

    # Return the combined loss value.
    return DiceBCE
