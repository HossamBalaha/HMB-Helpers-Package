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

# Import the required libraries for tensor operations and neural network components.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    # Initialize the DiceLoss class by calling the parent class constructor.
    super(DiceLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    # Apply a sigmoid activation function to the inputs if the model does not include one.
    inputs = F.sigmoid(inputs)

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
  def __init__(self, weight=None, size_average=True):
    # Initialize the DiceBCELoss class by calling the parent class constructor.
    super(DiceBCELoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    # Apply a sigmoid activation function to the inputs if the model does not include one.
    inputs = F.sigmoid(inputs)

    # Flatten the input and target tensors into 1D arrays for computation.
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Compute the intersection of the predicted and target tensors.
    intersection = (inputs * targets).sum()

    # Calculate the Dice loss component using the formula.
    diceLoss = 1.0 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    # Calculate the binary cross-entropy (BCE) loss component.
    BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")

    # Combine the Dice loss and BCE loss to form the final loss value.
    DiceBCE = BCE + diceLoss

    # Return the combined loss value.
    return DiceBCE


if __name__ == "__main__":
  # Simulate a model output tensor with random values (e.g., logits).
  predictions = torch.randn(1, 1, 256, 256).float()  # Simulated model output.
  # Normalize the predictions to the range [-1, 1].
  # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 2 - 1
  # Normalize the predictions to the range [0, 1].
  predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

  # Simulate a ground truth mask tensor with binary values (0 or 1).
  targets = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Simulated ground truth mask.

  print(f"Maximum value in predictions: {predictions.max().item()}")
  print(f"Minimum value in predictions: {predictions.min().item()}")
  print(f"Maximum value in targets: {targets.max().item()}")
  print(f"Minimum value in targets: {targets.min().item()}")
  print(f"Shape of predictions: {predictions.shape}")
  print(f"Shape of targets: {targets.shape}")

  # Instantiate the classes.
  diceLoss = DiceLoss()
  diceBCELoss = DiceBCELoss()

  # Compute the losses between the simulated inputs and targets.
  print(f"Dice Loss: {diceLoss(predictions, targets).item()}")
  print(f"Dice + BCE Loss: {diceBCELoss(predictions, targets).item()}")
