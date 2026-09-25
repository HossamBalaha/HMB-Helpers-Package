import clip  # Import the clip library for CLIP models.
import timm  # Import the timm library for Swin Transformer.
import numpy  # Import the numpy module for numerical operations.
import time  # Import the time module for timing operations.
import torch  # Import the torch module for deep learning.
import torchvision  # Import the torchvision module for models and datasets.
import torch.nn  # Import the neural network module from torch.
import torch.optim  # Import the optim module from torch.
from tqdm import tqdm  # Import tqdm for progress bars.
from pathlib import Path  # Import the Path class from pathlib.
import torch.nn.functional as F  # Import the functional module from torch.
from sklearn.metrics import *  # Import metric functions from sklearn.
import pennylane as qml  # Import the PennyLane library for quantum machine learning.
from pennylane import numpy as qNumpy  # Import PennyLane's numpy for quantum operations.
from typing import Any, Dict, List, Optional, Tuple, Union


class KANLayer(torch.nn.Module):
  r'''
  Kolmogorov-Arnold Network (KAN) layer implementation.

  Parameters:
    inputDim (int): dimension of the input features.
    outputDim (int): dimension of the output features.
    gridPoints (int): number of grid points for the activation functions.
  '''

  # Define the initialization method for the KAN layer.
  def __init__(self, inputDim, outputDim, gridPoints=10):
    # Call the parent class initialization method.
    super(KANLayer, self).__init__()
    # Store the input dimension as an instance variable.
    self.inputDim = inputDim
    # Store the output dimension as an instance variable.
    self.outputDim = outputDim
    # Store the number of grid points for the activation functions.
    self.gridPoints = gridPoints
    # Initialize the base linear weights for the layer.
    self.baseWeight = torch.nn.Parameter(torch.randn(outputDim, inputDim))
    # Initialize the spline weights for the learnable activations.
    self.splineWeight = torch.nn.Parameter(torch.randn(outputDim, inputDim, gridPoints))
    # Initialize the grid buffer for the activation functions.
    self.register_buffer("Grid", torch.linspace(-1.0, 1.0, gridPoints))

  # Define the forward pass method for the KAN layer.
  def forward(self, x):
    r'''
    Compute the forward pass for the KAN layer.

    Parameters:
      x (Tensor): input tensor of shape (N, inputDim) or (B, N, inputDim).

    Returns:
      torch.Tensor: output tensor of shape (N, outputDim) or (B, N, outputDim).
    '''

    # Store the original shape to reshape back if necessary.
    originalShape = x.shape
    # Flatten the tensor to 2D if it is 3D from a Transformer block.
    if (x.dim() == 3):
      # Reshape the tensor to 2D.
      x = x.view(-1, x.size(-1))
    # Compute the base linear transformation output.
    baseOut = torch.nn.functional.linear(x, self.baseWeight)
    # Normalize the input to the grid range using tanh.
    xNorm = torch.tanh(x)
    # Initialize the spline output tensor with zeros.
    splineOut = torch.zeros(x.size(0), self.outputDim, device=x.device)
    # Iterate through each input dimension to compute activations.
    for i in range(self.inputDim):
      # Extract the current input feature vector.
      xI = xNorm[:, i]
      # Compute the absolute differences between input and grid points.
      diffs = torch.abs(xI.unsqueeze(1) - self.Grid.unsqueeze(0))
      # Compute the radial basis function values for the grid.
      rbfVals = torch.exp(-diffs ** 2)
      # Normalize the RBF values to sum to one.
      rbfVals = rbfVals / (rbfVals.sum(dim=1, keepdim=True) + 1e-8)
      # Compute the spline output for the current input dimension.
      splineOut = splineOut + torch.matmul(rbfVals, self.splineWeight[:, i, :].T)
    # Combine the base linear output and the spline output.
    output = baseOut + splineOut
    # Reshape the output back to the original 3D shape if necessary.
    if (len(originalShape) == 3):
      # Reshape the output to the original 3D shape.
      output = output.view(originalShape[0], originalShape[1], self.outputDim)
    # Return the final output tensor.
    return output


class VisionKANBlock(torch.nn.Module):
  r'''
  Vision KAN block combining multi-head self-attention and KAN feed-forward networks.

  Parameters:
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    mlpRatio (float): ratio for the hidden dimension in the KAN layers.
    gridPoints (int): number of grid points for the KAN activations.
  '''

  # Define the initialization method for the Vision KAN block.
  def __init__(self, embedDim, numHeads, mlpRatio=2.0, gridPoints=10):
    # Call the parent class initialization method.
    super(VisionKANBlock, self).__init__()
    # Define the first layer normalization module.
    self.norm1 = torch.nn.LayerNorm(embedDim)
    # Define the multi-head self-attention module.
    self.attn = torch.nn.MultiheadAttention(embed_dim=embedDim, num_heads=numHeads, batch_first=True)
    # Define the second layer normalization module.
    self.norm2 = torch.nn.LayerNorm(embedDim)
    # Calculate the hidden dimension for the KAN layer.
    hiddenDim = int(embedDim * mlpRatio)
    # Define the first KAN layer for the feed-forward network.
    self.kan1 = KANLayer(embedDim, hiddenDim, gridPoints)
    # Define the second KAN layer for the feed-forward network.
    self.kan2 = KANLayer(hiddenDim, embedDim, gridPoints)

  # Define the forward pass method for the Vision KAN block.
  def forward(self, x):
    r'''
    Compute the forward pass for the Vision KAN block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, embedDim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, embedDim).
    '''

    # Store the residual connection for the attention block.
    res = x
    # Apply the first layer normalization to the input.
    x = self.norm1(x)
    # Compute the multi-head self-attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection to the attention output.
    x = res + attnOut
    # Store the residual connection for the feed-forward block.
    res = x
    # Apply the second layer normalization to the input.
    x = self.norm2(x)
    # Pass the normalized input through the first KAN layer.
    x = self.kan1(x)
    # Pass the intermediate output through the second KAN layer.
    x = self.kan2(x)
    # Add the residual connection to the feed-forward output.
    x = res + x
    # Return the updated tensor.
    return x


class VisionKANModel(torch.nn.Module):
  r'''
  Vision KAN model for image classification using KAN-based transformer blocks.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    depth (int): number of Vision KAN blocks.
  '''

  # Define the initialization method for the Vision KAN model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(VisionKANModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of Vision KAN blocks.
    self.blocks = torch.nn.ModuleList([VisionKANBlock(embedDim, numHeads) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the Vision KAN model.
  def forward(self, x):
    r'''
    Compute the forward pass for the Vision KAN model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the Vision KAN blocks.
    for block in self.blocks:
      # Apply the current Vision KAN block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class NeuralODEBlock(torch.nn.Module):
  r'''
  Neural ODE block defining the continuous-time dynamics.

  Parameters:
    dim (int): dimension of the hidden state.
  '''

  # Define the initialization method for the Neural ODE block.
  def __init__(self, dim):
    # Call the parent class initialization method.
    super(NeuralODEBlock, self).__init__()
    # Define the underlying neural network for the ODE dynamics.
    self.net = torch.nn.Sequential(
      # Add the first layer normalization.
      torch.nn.LayerNorm(dim),
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the ODE dynamics.
  def forward(self, t, x):
    r'''
    Compute the derivative of the hidden state for the ODE solver.

    Parameters:
      t (Tensor): current time step.
      x (Tensor): current hidden state of shape (B, dim).

    Returns:
      torch.Tensor: derivative of the hidden state of shape (B, dim).
    '''

    # Compute and return the derivative of the hidden state.
    return self.net(x)


class NeuralODEViTModel(torch.nn.Module):
  r'''
  Neural ODE Vision Transformer model combining discrete and continuous depth.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    depth (int): number of standard transformer blocks before the ODE block.
  '''

  # Define the initialization method for the Neural ODE ViT model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(NeuralODEViTModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the standard transformer block for initial processing.
    self.preBlock = TokenTransformerBlock(dim=embedDim, numHeads=numHeads)
    # Define the Neural ODE block for continuous depth processing.
    self.odeBlock = NeuralODEBlock(dim=embedDim)
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)
    # Define the time tensor for the ODE solver.
    self.register_buffer("TimeTensor", torch.tensor([0.0, 1.0]))

  # Define the forward pass method for the Neural ODE ViT model.
  def forward(self, x):
    r'''
    Compute the forward pass for the Neural ODE ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Apply the pre-processing transformer block.
    x = self.preBlock(x)
    # Extract the class token for the ODE dynamics.
    h = x[:, 0]
    # Define the time step size for the Euler integration.
    dt = 0.1
    # Iterate through the time steps for the Euler integration.
    for step in range(10):
      # Compute the derivative at the current state.
      dh = self.odeBlock(self.TimeTensor[1], h)
      # Update the hidden state using the Euler step.
      h = h + dt * dh
    # Return the classification output using the integrated class token.
    return self.head(self.norm(h))


class LIFNeuron(torch.nn.Module):
  r'''
  Leaky Integrate-and-Fire (LIF) neuron model for spiking neural networks.

  Parameters:
    threshold (float): firing threshold for the membrane potential.
    decay (float): membrane potential decay factor.
  '''

  # Define the initialization method for the LIF neuron.
  def __init__(self, threshold=1.0, decay=0.5):
    # Call the parent class initialization method.
    super(LIFNeuron, self).__init__()
    # Store the firing threshold.
    self.threshold = threshold
    # Store the membrane potential decay factor.
    self.decay = decay
    # Initialize the membrane potential buffer.
    self.register_buffer("MembranePot", torch.tensor(0.0))

  # Define the forward pass method for the LIF neuron.
  def forward(self, x):
    r'''
    Compute the forward pass for the LIF neuron.

    Parameters:
      x (Tensor): input current tensor of arbitrary shape.

    Returns:
      torch.Tensor: generated spikes tensor of the same shape as input.
    '''

    # Initialize the membrane potential if it does not match the input shape.
    if (self.MembranePot.shape != x.shape):
      # Reset the membrane potential to match the input tensor shape.
      self.MembranePot = torch.zeros_like(x)
    # Update the membrane potential with decay and input current.
    self.MembranePot = self.MembranePot * self.decay + x
    # Generate spikes where the membrane potential exceeds the threshold.
    spikes = (self.MembranePot >= self.threshold).float()
    # Reset the membrane potential for neurons that fired.
    self.MembranePot = self.MembranePot * (1.0 - spikes)
    # Return the generated spikes.
    return spikes


class SpikingViTBlock(torch.nn.Module):
  r'''
  Spiking Vision Transformer block combining attention and spiking feed-forward networks.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the Spiking ViT block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(SpikingViTBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the LIF neuron for attention output.
    self.lif1 = LIFNeuron()
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the LIF neuron for activation.
      LIFNeuron(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )
    # Define the LIF neuron for MLP output.
    self.lif2 = LIFNeuron()

  # Define the forward pass method for the Spiking ViT block.
  def forward(self, x):
    r'''
    Compute the forward pass for the Spiking ViT block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Apply the LIF neuron to the attention output.
    attnOut = self.lif1(attnOut)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Apply the LIF neuron to the MLP output.
    mlpOut = self.lif2(mlpOut)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


class SpikingViTModel(torch.nn.Module):
  r'''
  Spiking Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    depth (int): number of Spiking ViT blocks.
  '''

  # Define the initialization method for the Spiking ViT model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(SpikingViTModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of Spiking ViT blocks.
    self.blocks = torch.nn.ModuleList([SpikingViTBlock(embedDim, numHeads) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the Spiking ViT model.
  def forward(self, x):
    r'''
    Compute the forward pass for the Spiking ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the Spiking ViT blocks.
    for block in self.blocks:
      # Apply the current Spiking ViT block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class HypernetworkGenerator(torch.nn.Module):
  r'''
  Hypernetwork generator for producing dynamic weights.

  Parameters:
    inputDim (int): dimension of the input context vector.
    weightDim (int): dimension of the generated weight vector.
  '''

  # Define the initialization method for the Hypernetwork generator.
  def __init__(self, inputDim, weightDim):
    # Call the parent class initialization method.
    super(HypernetworkGenerator, self).__init__()
    # Define the sequential network for weight generation.
    self.net = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(inputDim, inputDim * 2),
      # Add the ReLU activation function.
      torch.nn.ReLU(),
      # Add the second linear layer to output the target weight dimension.
      torch.nn.Linear(inputDim * 2, weightDim),
    )

  # Define the forward pass method for the Hypernetwork generator.
  def forward(self, z):
    r'''
    Compute the forward pass for the Hypernetwork generator.

    Parameters:
      z (Tensor): input context tensor of shape (B, inputDim).

    Returns:
      torch.Tensor: generated weights tensor of shape (B, weightDim).
    '''

    # Generate and return the dynamic weights.
    return self.net(z)


class HypernetworkViTBlock(torch.nn.Module):
  r'''
  Hypernetwork Vision Transformer block with dynamically generated MLP weights.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the Hypernetwork ViT block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(HypernetworkViTBlock, self).__init__()
    # Store the embedding dimension.
    self.dim = dim
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Calculate the dimension of the MLP weights.
    mlpWeightDim = dim * (dim * 2) + (dim * 2) + (dim * 2) * dim + dim
    # Define the Hypernetwork generator for MLP weights.
    self.hyperNet = HypernetworkGenerator(dim, mlpWeightDim)

  # Define the forward pass method for the Hypernetwork ViT block.
  def forward(self, x, z):
    r'''
    Compute the forward pass for the Hypernetwork ViT block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).
      z (Tensor): context tensor of shape (B, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Generate the dynamic MLP weights using the hypernetwork.
    dynWeights = self.hyperNet(z)
    # Reshape the generated weights for the first linear layer.
    w1 = dynWeights[:, :self.dim * (self.dim * 2)].view(-1, self.dim * 2, self.dim)
    # Extract the bias for the first linear layer.
    b1 = dynWeights[:, self.dim * (self.dim * 2):self.dim * (self.dim * 2) + (self.dim * 2)]
    # Apply the dynamic first linear layer and ReLU.
    x = torch.nn.functional.relu(torch.bmm(x, w1.transpose(1, 2)) + b1.unsqueeze(1))
    # Reshape the generated weights for the second linear layer.
    w2 = dynWeights[:, -((self.dim * 2) * self.dim + self.dim):-self.dim].view(-1, self.dim, self.dim * 2)
    # Extract the bias for the second linear layer.
    b2 = dynWeights[:, -self.dim:]
    # Apply the dynamic second linear layer.
    mlpOut = torch.bmm(x, w2.transpose(1, 2)) + b2.unsqueeze(1)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


class HypernetworkViTModel(torch.nn.Module):
  r'''
  Hypernetwork Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    depth (int): number of Hypernetwork ViT blocks.
  '''

  # Define the initialization method for the Hypernetwork ViT model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(HypernetworkViTModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of Hypernetwork ViT blocks.
    self.blocks = torch.nn.ModuleList([HypernetworkViTBlock(embedDim, numHeads) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)
    # Define the global context generator for the hypernetwork.
    self.contextGen = torch.nn.Linear(embedDim, embedDim)

  # Define the forward pass method for the Hypernetwork ViT model.
  def forward(self, x):
    r'''
    Compute the forward pass for the Hypernetwork ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Generate the global context vector from the class token.
    z = self.contextGen(x[:, 0])
    # Iterate through the Hypernetwork ViT blocks.
    for block in self.blocks:
      # Apply the current Hypernetwork ViT block with the context vector.
      x = block(x, z)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class LiquidSSMBlock(torch.nn.Module):
  r'''
  Liquid State Space Model block combining ODE dynamics and gating.

  Parameters:
    hiddenDim (int): hidden dimension of the block.
    stateDim (int): state dimension for the SSM.
  '''

  # Define the initialization method for the block.
  def __init__(self, hiddenDim, stateDim=16):
    # Call the parent class initialization method.
    super(LiquidSSMBlock, self).__init__()
    # Store the hidden dimension as an instance variable.
    self.hiddenDim = hiddenDim
    # Store the state dimension as an instance variable.
    self.stateDim = stateDim
    # Define the input projection layer.
    self.inputProj = torch.nn.Linear(hiddenDim, hiddenDim * 2)
    # Define the convolutional layer for the state space model.
    self.convLayer = torch.nn.Conv1d(hiddenDim, hiddenDim, kernel_size=3, padding=1, groups=hiddenDim)
    # Define the liquid ordinary differential equation weight matrix.
    self.liquidWeight = torch.nn.Parameter(torch.randn(hiddenDim, hiddenDim) * 0.01)
    # Define the liquid time constant parameter.
    self.timeConstant = torch.nn.Parameter(torch.ones(hiddenDim) * 0.1)
    # Define the output projection layer.
    self.outputProj = torch.nn.Linear(hiddenDim, hiddenDim)
    # Define the layer normalization module.
    self.normLayer = torch.nn.LayerNorm(hiddenDim)

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the Liquid SSM block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, hiddenDim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, hiddenDim).
    '''

    # Store the residual connection.
    res = x
    # Apply the layer normalization.
    x = self.normLayer(x)
    # Project the input to double dimension.
    xProj = self.inputProj(x)
    # Split the projected tensor into main and gate branches.
    xMain, xGate = xProj.chunk(2, dim=-1)
    # Transpose for convolutional processing.
    xConv = self.convLayer(xMain.transpose(1, 2)).transpose(1, 2)
    # Compute the liquid ordinary differential equation dynamics.
    dx = torch.matmul(xConv, self.liquidWeight) - (xConv * self.timeConstant)
    # Integrate the ordinary differential equation using a simple Euler step.
    xLiquid = xConv + dx * 0.1
    # Apply the gating mechanism.
    xGated = xLiquid * torch.sigmoid(xGate)
    # Project back to the hidden dimension.
    xOut = self.outputProj(xGated)
    # Add the residual connection.
    x = res + xOut
    # Return the updated tensor.
    return x


class LiquidSSMViTModel(torch.nn.Module):
  r'''
  Liquid State Space Model Vision Transformer for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of Liquid SSM blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(LiquidSSMViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of liquid state space blocks.
    self.blocks = torch.nn.ModuleList([LiquidSSMBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the Liquid SSM ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the liquid state space blocks.
    for block in self.blocks:
      # Apply the current liquid state space block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class TestTimeEvolvingBlock(torch.nn.Module):
  r'''
  Test-time evolving transformer block with auxiliary projection.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TestTimeEvolvingBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )
    # Define the self-supervised auxiliary projection for test-time adaptation.
    self.auxProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the test-time evolving block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Compute the auxiliary projection for internal test-time evolution tracking.
    _ = self.auxProj(x[:, 0])
    # Return the updated tensor.
    return x


class TestTimeEvolvingViTModel(torch.nn.Module):
  r'''
  Test-time evolving Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of test-time evolving blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(TestTimeEvolvingViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of test-time evolving blocks.
    self.blocks = torch.nn.ModuleList([TestTimeEvolvingBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the test-time evolving ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the test-time evolving blocks.
    for block in self.blocks:
      # Apply the current block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class TensorNetworkEntangledAttention(torch.nn.Module):
  r'''
  Tensor network entangled attention using Matrix Product State core.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
    bondDim (int): bond dimension for the MPS core tensor.
  '''

  # Define the initialization method for the attention.
  def __init__(self, dim, numHeads=4, bondDim=8):
    # Call the parent class initialization method.
    super(TensorNetworkEntangledAttention, self).__init__()
    # Store the dimension.
    self.dim = dim
    # Store the number of heads.
    self.numHeads = numHeads
    # Store the head dimension.
    self.headDim = dim // numHeads
    # Define the query projection layer.
    self.qProj = torch.nn.Linear(dim, dim)
    # Define the key projection layer.
    self.kProj = torch.nn.Linear(dim, dim)
    # Define the value projection layer.
    self.vProj = torch.nn.Linear(dim, dim)
    # Define the Matrix Product State core tensor for attention compression as a learned bilinear form.
    self.mpsCore = torch.nn.Parameter(torch.randn(numHeads, self.headDim, self.headDim) * 0.02)
    # Define the output projection layer.
    self.outProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the attention.
  def forward(self, x):
    r'''
    Compute the forward pass for the tensor network entangled attention.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Get the batch size and sequence length.
    b, n, _ = x.shape
    # Compute the query, key, and value projections.
    q = self.qProj(x).view(b, n, self.numHeads, self.headDim).transpose(1, 2)
    # Compute the key projection.
    k = self.kProj(x).view(b, n, self.numHeads, self.headDim).transpose(1, 2)
    # Compute the value projection.
    v = self.vProj(x).view(b, n, self.numHeads, self.headDim).transpose(1, 2)
    # Compute the entangled attention using the MPS core tensor as a learned bilinear form.
    attnWeights = torch.einsum("bhqd,bhke,hde->bhqk", q, k, self.mpsCore)
    # Apply softmax to the attention weights.
    attnWeights = torch.softmax(attnWeights, dim=-1)
    # Compute the attention output.
    out = torch.matmul(attnWeights, v)
    # Reshape the output to the original dimensions.
    out = out.transpose(1, 2).contiguous().view(b, n, self.dim)
    # Apply the output projection.
    return self.outProj(out)


class TensorNetworkEntangledBlock(torch.nn.Module):
  r'''
  Tensor network entangled transformer block.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TensorNetworkEntangledBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the tensor network entangled attention.
    self.attn = TensorNetworkEntangledAttention(dim, numHeads)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the tensor network entangled block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut = self.attn(x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


class TensorNetworkEntangledViTModel(torch.nn.Module):
  r'''
  Tensor network entangled Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of tensor network entangled blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(TensorNetworkEntangledViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of tensor network entangled blocks.
    self.blocks = torch.nn.ModuleList([TensorNetworkEntangledBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the tensor network entangled ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the tensor network entangled blocks.
    for block in self.blocks:
      # Apply the current block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class DiffusionPriorEnergyBlock(torch.nn.Module):
  r'''
  Diffusion prior energy transformer block with energy-based denoising score projection.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(DiffusionPriorEnergyBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )
    # Define the energy-based denoising score projection.
    self.energyProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the diffusion prior energy block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Compute the energy projection for internal out-of-distribution tracking.
    _ = self.energyProj(x[:, 0])
    # Return the updated tensor.
    return x


class DiffusionPriorEnergyViTModel(torch.nn.Module):
  r'''
  Diffusion prior energy Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of diffusion prior energy blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(DiffusionPriorEnergyViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of diffusion prior energy blocks.
    self.blocks = torch.nn.ModuleList([DiffusionPriorEnergyBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)
    # Define the energy scaling parameter for the EBM.
    self.energyScale = torch.nn.Parameter(torch.ones(1) * 0.1)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the diffusion prior energy ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the diffusion prior energy blocks.
    for block in self.blocks:
      # Apply the current block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Compute the classification logits.
    logits = self.head(x)
    # Compute the energy-based out-of-distribution score internally.
    _ = -self.energyScale * torch.logsumexp(logits / self.energyScale, dim=-1)
    # Return the classification logits.
    return logits


class FractalResonanceBlock(torch.nn.Module):
  r'''
  Fractal resonance transformer block with harmonic projection.

  Parameters:
    hiddenDim (int): hidden dimension of the block.
    fractalDepth (int): number of fractal iteration depths.
  '''

  # Define the initialization method for the block.
  def __init__(self, hiddenDim, fractalDepth=3):
    # Call the parent class initialization method.
    super(FractalResonanceBlock, self).__init__()
    # Store the hidden dimension as an instance variable.
    self.hiddenDim = hiddenDim
    # Define the harmonic projection layer.
    self.harmonicProj = torch.nn.Linear(hiddenDim, hiddenDim)
    # Store the fractal iteration depth.
    self.fractalDepth = fractalDepth
    # Define the resonance weight parameter.
    self.resonanceWeight = torch.nn.Parameter(torch.ones(1) * 0.5)

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the fractal resonance block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, hiddenDim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, hiddenDim).
    '''

    # Store the residual connection.
    res = x
    # Project the input to the harmonic space.
    xHarm = self.harmonicProj(x)
    # Initialize the fractal accumulation tensor.
    xFractal = torch.zeros_like(xHarm)
    # Iterate through the fractal depths.
    for i in range(self.fractalDepth):
      # Compute the resonance shift.
      shift = torch.sin(xHarm * (i + 1))
      # Accumulate the fractal resonance.
      xFractal = xFractal + shift
    # Apply the global resonance weight.
    xOut = xHarm + (self.resonanceWeight * xFractal)
    # Add the residual connection.
    x = res + xOut
    # Return the updated tensor.
    return x


class FractalResonanceViTModel(torch.nn.Module):
  r'''
  Fractal resonance Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of fractal resonance blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(FractalResonanceViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of fractal resonance blocks.
    self.blocks = torch.nn.ModuleList([FractalResonanceBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the fractal resonance ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the fractal resonance blocks.
    for block in self.blocks:
      # Apply the current fractal resonance block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class TopologicalQuantumAttention(torch.nn.Module):
  r'''
  Topological quantum attention using Betti number scaling.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the attention.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TopologicalQuantumAttention, self).__init__()
    # Store the dimension.
    self.dim = dim
    # Store the number of heads.
    self.numHeads = numHeads
    # Define the query projection layer.
    self.qProj = torch.nn.Linear(dim, dim)
    # Define the key projection layer.
    self.kProj = torch.nn.Linear(dim, dim)
    # Define the value projection layer.
    self.vProj = torch.nn.Linear(dim, dim)
    # Define the topological Betti number weight parameter.
    self.bettiWeight = torch.nn.Parameter(torch.ones(numHeads) * 0.1)
    # Define the output projection layer.
    self.outProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the attention.
  def forward(self, x):
    r'''
    Compute the forward pass for the topological quantum attention.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Get the batch size and sequence length.
    b, n, _ = x.shape
    # Compute the query projection.
    q = self.qProj(x).view(b, n, self.numHeads, -1).transpose(1, 2)
    # Compute the key projection.
    k = self.kProj(x).view(b, n, self.numHeads, -1).transpose(1, 2)
    # Compute the value projection.
    v = self.vProj(x).view(b, n, self.numHeads, -1).transpose(1, 2)
    # Compute the standard attention weights.
    attnWeights = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
    # Apply the topological Betti number scaling.
    attnWeights = attnWeights * self.bettiWeight.view(1, -1, 1, 1)
    # Apply softmax to the attention weights.
    attnWeights = torch.softmax(attnWeights, dim=-1)
    # Compute the attention output.
    out = torch.matmul(attnWeights, v)
    # Reshape the output to the original dimensions.
    out = out.transpose(1, 2).contiguous().view(b, n, self.dim)
    # Apply the output projection.
    return self.outProj(out)


class TopologicalQuantumBlock(torch.nn.Module):
  r'''
  Topological quantum transformer block.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TopologicalQuantumBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the topological quantum attention.
    self.attn = TopologicalQuantumAttention(dim, numHeads)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the topological quantum block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut = self.attn(x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


class TopologicalQuantumViTModel(torch.nn.Module):
  r'''
  Topological quantum Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of topological quantum blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(TopologicalQuantumViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of topological quantum attention blocks.
    self.blocks = torch.nn.ModuleList([TopologicalQuantumBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the topological quantum ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the topological quantum attention blocks.
    for block in self.blocks:
      # Apply the current topological quantum block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class HolographicInterferenceBlock(torch.nn.Module):
  r'''
  Holographic interference transformer block with phase shift encoding.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
  '''

  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(HolographicInterferenceBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the phase shift parameter for holographic encoding.
    self.phaseShift = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the holographic interference block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Compute the cosine component of the holographic interference.
    phaseCos = torch.cos(self.phaseShift)
    # Compute the sine component of the holographic interference.
    phaseSin = torch.sin(self.phaseShift)
    # Apply the holographic phase shift interference using sine and cosine.
    holoOut = (attnOut * phaseCos) + (torch.roll(attnOut, shifts=1, dims=-1) * phaseSin)
    # Add the residual connection.
    x = res + holoOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


class HolographicInterferenceViTModel(torch.nn.Module):
  r'''
  Holographic interference Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of holographic interference blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(HolographicInterferenceViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of holographic interference blocks.
    self.blocks = torch.nn.ModuleList([HolographicInterferenceBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the holographic interference ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the holographic interference blocks.
    for block in self.blocks:
      # Apply the current holographic interference block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


class NeuromorphicLiquidStateBlock(torch.nn.Module):
  r'''
  Neuromorphic liquid state block for temporal dynamics processing.

  Parameters:
    dim (int): embedding dimension.
    reservoirSize (int): size of the recurrent reservoir.
  '''

  # Define the initialization method for the block.
  def __init__(self, dim, reservoirSize=64):
    # Call the parent class initialization method.
    super(NeuromorphicLiquidStateBlock, self).__init__()
    # Define the input projection to reservoir.
    self.inputProj = torch.nn.Linear(dim, reservoirSize)
    # Define the recurrent reservoir weight matrix.
    self.reservoirWeight = torch.nn.Parameter(torch.randn(reservoirSize, reservoirSize) * 0.1)
    # Define the leaky integration decay factor.
    self.decay = torch.nn.Parameter(torch.ones(1) * 0.9)
    # Define the readout projection from reservoir.
    self.readoutProj = torch.nn.Linear(reservoirSize, dim)
    # Define the layer normalization.
    self.norm = torch.nn.LayerNorm(dim)

  # Define the forward pass method for the block.
  def forward(self, x):
    r'''
    Compute the forward pass for the neuromorphic liquid state block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the layer normalization.
    x = self.norm(x)
    # Project the input to the reservoir space.
    u = self.inputProj(x)
    # Initialize the reservoir state.
    state = torch.zeros(u.size(0), u.size(2), device=x.device)
    # Initialize the list to collect reservoir states.
    stateList = []
    # Iterate through the sequence length for temporal dynamics.
    for t in range(u.size(1)):
      # Compute the liquid state update with leaky integration.
      state = self.decay * state + torch.tanh(torch.matmul(u[:, t, :], self.reservoirWeight))
      # Append the current state to the list.
      stateList.append(state)
    # Stack the collected states into a single tensor.
    u = torch.stack(stateList, dim=1)
    # Project the reservoir state back to the embedding dimension.
    xOut = self.readoutProj(u)
    # Add the residual connection.
    x = res + xOut
    # Return the updated tensor.
    return x


class NeuromorphicLiquidStateViTModel(torch.nn.Module):
  r'''
  Neuromorphic liquid state Vision Transformer model for image classification.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    depth (int): number of neuromorphic liquid state blocks.
  '''

  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(NeuromorphicLiquidStateViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of neuromorphic liquid state blocks.
    self.blocks = torch.nn.ModuleList([NeuromorphicLiquidStateBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    r'''
    Compute the forward pass for the neuromorphic liquid state ViT model.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the neuromorphic liquid state blocks.
    for block in self.blocks:
      # Apply the current neuromorphic liquid state block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


def QuantumCircuit(qInputFeatures, qWeightsFlat, nQubits, qDepth):
  r'''
  Define the quantum circuit for the dressed quantum network.

  Parameters:
    qInputFeatures (Tensor): input features for the quantum circuit.
    qWeightsFlat (Tensor): flattened trainable quantum weights.
    nQubits (int): number of qubits in the circuit.
    qDepth (int): depth of the quantum circuit.

  Returns:
    tuple: tuple of expectation values for each qubit.
  '''

  # Reshape the flat weights into a depth by qubits matrix.
  qWeights = qWeightsFlat.reshape(qDepth, nQubits)
  # Apply a layer of Hadamard gates to all qubits.
  for idx in range(nQubits):
    # Apply the Hadamard gate to the current qubit.
    qml.Hadamard(wires=idx)
  # Apply a layer of parametrized Y rotations for feature embedding.
  for idx, element in enumerate(qInputFeatures):
    # Apply the RY gate with the embedded feature.
    qml.RY(element, wires=idx)
  # Apply the sequence of trainable variational layers.
  for k in range(qDepth):
    # Apply the entangling layer of CNOT gates.
    for i in range(0, nQubits - 1, 2):
      # Apply CNOT between even and odd qubits.
      qml.CNOT(wires=[i, i + 1])
    # Apply the shifted entangling layer of CNOT gates.
    for i in range(1, nQubits - 1, 2):
      # Apply CNOT between odd and even qubits.
      qml.CNOT(wires=[i, i + 1])
    # Apply the parametrized Y rotation layer with trainable weights.
    for idx, element in enumerate(qWeights[k]):
      # Apply the RY gate with the trainable weight.
      qml.RY(element, wires=idx)
  # Calculate the expectation values in the Z basis.
  expVals = [qml.expval(qml.PauliZ(position)) for position in range(nQubits)]
  # Return the tuple of expectation values.
  return tuple(expVals)


class DressedQuantumNet(torch.nn.Module):
  r'''
  Torch module implementing the dressed quantum net.

  Parameters:
    numClasses (int): number of output classes.
    nQubits (int): number of qubits in the quantum circuit.
    qDepth (int): depth of the quantum circuit.
    qDelta (float): initial spread of random quantum weights.
  '''

  # Initialize the dressed quantum network.
  def __init__(self, numClasses, nQubits=4, qDepth=10, qDelta=0.01):
    # Call the parent module constructor.
    super().__init__()
    # Store the number of output classes.
    self.numClasses = numClasses
    # Store the number of qubits.
    self.nQubits = nQubits
    # Store the quantum circuit depth.
    self.qDepth = qDepth
    # Store the initial spread of random quantum weights.
    self.qDelta = qDelta
    # Define the PennyLane quantum device.
    self.dev = qml.device("default.qubit", wires=self.nQubits)
    # Define the classical pre-processing linear layer.
    self.preNet = torch.nn.Linear(8, self.nQubits)
    # Define the quantum parameters as a trainable parameter.
    self.qParams = torch.nn.Parameter(self.qDelta * torch.randn(self.qDepth * self.nQubits))
    # Define the classical post-processing linear layer.
    self.postNet = torch.nn.Linear(self.nQubits, self.numClasses)
    # Create the PennyLane QNode for the quantum circuit.
    self.quantumNet = qml.QNode(QuantumCircuit, self.dev, interface="torch")

  # Define the forward pass of the dressed quantum network.
  def forward(self, inputFeatures):
    r'''
    Compute the forward pass of the dressed quantum network.

    Parameters:
      inputFeatures (Tensor): input tensor of shape (B, 8).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Obtain the input features for the quantum circuit via pre-processing.
    preOut = self.preNet(inputFeatures)
    # Scale the pre-processed features using tanh and pi.
    qIn = torch.tanh(preOut) * (qNumpy.pi / 2.0)
    # Initialize an empty tensor for the quantum output.
    qOut = torch.Tensor(0, self.nQubits)
    # Move the quantum output tensor to the correct device.
    qOut = qOut.to(inputFeatures.device)
    # Iterate over each element in the batch.
    for elem in qIn:
      # Apply the quantum circuit to the current element.
      qOutElem = torch.hstack(self.quantumNet(elem, self.qParams, self.nQubits, self.qDepth)).float().unsqueeze(0)
      # Concatenate the quantum output element to the batch output.
      qOut = torch.cat((qOut, qOutElem))
    # Return the final prediction from the post-processing layer.
    return self.postNet(qOut)


def BuildQuantumResNetModel(numClasses, device):
  r'''
  Build a hybrid quantum-classical ResNet model.

  Parameters:
    numClasses (int): number of output classes.
    device (torch.device): device to place the model on.

  Returns:
    torch.nn.Module: the constructed quantum hybrid model.
  '''

  # Load the pre-trained ResNet152 model from torchvision.
  baseModel = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
  # Freeze the parameters of the base ResNet model for transfer learning.
  for param in baseModel.parameters():
    # Set requires_grad to False to freeze the weights.
    param.requires_grad = False
  # Replace the final fully connected layer with a custom sequential block.
  baseModel.fc = torch.nn.Sequential(
    # Add the first linear layer to reduce dimensions.
    torch.nn.Linear(2048, 512),
    # Add a ReLU activation function.
    torch.nn.ReLU(inplace=True),
    # Add the second linear layer.
    torch.nn.Linear(512, 256),
    # Add another ReLU activation function.
    torch.nn.ReLU(inplace=True),
    # Add the third linear layer.
    torch.nn.Linear(256, 128),
    # Add the fourth linear layer.
    torch.nn.Linear(128, 64),
    # Add a ReLU activation function.
    torch.nn.ReLU(inplace=True),
    # Add the fifth linear layer.
    torch.nn.Linear(64, 16),
    # Add the sixth linear layer to match quantum input size.
    torch.nn.Linear(16, 8),
    # Add the dressed quantum network for quantum classification.
    DressedQuantumNet(numClasses=numClasses, nQubits=4, qDepth=10, qDelta=0.01)
  )
  # Move the model to the specified device.
  baseModel = baseModel.to(device)
  # Return the constructed quantum hybrid model.
  return baseModel


class SoftSplit(torch.nn.Module):
  r'''
  Soft split module for tokenizing images.

  Parameters:
    inChannels (int): number of input channels.
    patchSize (int): size of the patch.
    stride (int): stride of the convolution.
    projDim (int): projection dimension.
  '''

  # Initialize the soft split module.
  def __init__(self, inChannels=3, patchSize=7, stride=4, projDim=64):
    # Call the parent initialization.
    super().__init__()
    # Define the convolutional projection layer.
    self.proj = torch.nn.Conv2d(inChannels, projDim, kernel_size=patchSize, stride=stride, padding=patchSize // 2)

  # Define the forward pass method.
  def forward(self, x):
    r'''
    Compute the forward pass for the soft split module.

    Parameters:
      x (Tensor): input image tensor of shape (B, C, H, W).

    Returns:
      torch.Tensor: flattened and transposed tensor of shape (B, N, projDim).
    '''

    # Apply the projection layer.
    x = self.proj(x)
    # Get the batch size, channels, height, and width.
    batchSize, channels, height, width = x.shape
    # Flatten and transpose the tensor.
    return x.flatten(2).transpose(1, 2)


class TokenTransformerBlock(torch.nn.Module):
  r'''
  Transformer block for token processing.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
    mlpRatio (float): ratio for the MLP hidden dimension.
  '''

  # Initialize the transformer block.
  def __init__(self, dim, numHeads=4, mlpRatio=2.0):
    # Call the parent initialization.
    super().__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the multi-layer perceptron.
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(dim, int(dim * mlpRatio)),
      torch.nn.GELU(),
      torch.nn.Linear(int(dim * mlpRatio), dim)
    )

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the transformer block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Add the MLP output.
    x = x + self.mlp(self.norm2(x))
    # Return the updated tensor.
    return x


class T2TModule(torch.nn.Module):
  r'''
  Tokens-to-Token module for hierarchical tokenization.

  Parameters:
    inChannels (int): number of input channels.
    tokenDim (int): output token dimension.
  '''

  # Initialize the T2T module.
  def __init__(self, inChannels=3, tokenDim=64):
    # Call the parent initialization.
    super().__init__()
    # Define the first soft split.
    self.softSplit1 = SoftSplit(inChannels, patchSize=7, stride=4, projDim=32)
    # Define the first transformer block.
    self.trans1 = TokenTransformerBlock(dim=32)
    # Define the second soft split.
    self.softSplit2 = SoftSplit(32, patchSize=3, stride=2, projDim=tokenDim)

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the T2T module.

    Parameters:
      x (Tensor): input image tensor of shape (B, C, H, W).

    Returns:
      torch.Tensor: output tokens of shape (B, N, tokenDim).
    '''

    # Apply the first soft split.
    x = self.softSplit1(x)
    # Apply the first transformer block.
    x = self.trans1(x)
    # Get the tensor shape.
    batchSize, numTokens, channels = x.shape
    # Calculate the spatial dimensions.
    height = width = int(numpy.sqrt(numTokens))
    # Reshape the tensor back to spatial format.
    x = x.transpose(1, 2).reshape(batchSize, channels, height, width)
    # Apply the second soft split.
    x = self.softSplit2(x)
    # Return the tokens.
    return x


class T2TViT(torch.nn.Module):
  r'''
  Tokens-to-Token Vision Transformer model.

  Parameters:
    numClasses (int): number of output classes.
    tokenDim (int): token embedding dimension.
    depth (int): number of transformer blocks.
  '''

  # Initialize the T2T Vision Transformer.
  def __init__(self, numClasses=2, tokenDim=64, depth=4):
    # Call the parent initialization.
    super().__init__()
    # Define the T2T module.
    self.t2t = T2TModule(inChannels=3, tokenDim=tokenDim)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, tokenDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 1000, tokenDim))
    # Define the transformer blocks.
    self.blocks = torch.nn.ModuleList([TokenTransformerBlock(dim=tokenDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(tokenDim)
    # Define the classification head.
    self.head = torch.nn.Linear(tokenDim, numClasses)
    # Initialize the positional embedding.
    torch.nn.init.trunc_normal_(self.posEmbed, std=0.02)
    # Initialize the class token.
    torch.nn.init.trunc_normal_(self.clsToken, std=0.02)

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the T2T Vision Transformer.

    Parameters:
      x (Tensor): input image tensor of shape (B, C, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Get the batch size.
    batchSize = x.shape[0]
    # Apply the T2T module.
    x = self.t2t(x)
    # Expand the class token.
    clsTokens = self.clsToken.expand(batchSize, -1, -1)
    # Concatenate the class token with the patches.
    x = torch.cat((clsTokens, x), dim=1)
    # Check if positional embedding needs resizing.
    if (x.size(1) > self.posEmbed.size(1)):
      # Create a new positional embedding.
      newPos = torch.zeros(1, x.size(1), x.size(2), device=x.device)
      # Initialize the new positional embedding.
      torch.nn.init.trunc_normal_(newPos, std=0.02)
      # Update the positional embedding parameter.
      self.posEmbed = torch.nn.Parameter(newPos)
    # Add the positional embedding.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the transformer blocks.
    for block in self.blocks:
      # Apply the transformer block.
      x = block(x)
    # Apply the final normalization.
    x = self.norm(x)
    # Return the classification output.
    return self.head(x[:, 0])


class HierarchicalBlock(torch.nn.Module):
  r'''
  Hierarchical transformer block.

  Parameters:
    dim (int): embedding dimension.
    numHeads (int): number of attention heads.
    mlpRatio (float): ratio for the MLP hidden dimension.
  '''

  # Initialize the hierarchical block.
  def __init__(self, dim, numHeads, mlpRatio=4.0):
    # Call the parent initialization.
    super().__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the multi-layer perceptron.
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(dim, int(dim * mlpRatio)),
      torch.nn.GELU(),
      torch.nn.Linear(int(dim * mlpRatio), dim)
    )

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the hierarchical block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, dim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, dim).
    '''

    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Add the MLP output.
    x = x + self.mlp(self.norm2(x))
    # Return the updated tensor.
    return x


class PatchMerging(torch.nn.Module):
  r'''
  Patch merging module for hierarchical downsampling.

  Parameters:
    inDim (int): input dimension.
    outDim (int): output dimension.
  '''

  # Initialize the patch merging module.
  def __init__(self, inDim, outDim):
    # Call the parent initialization.
    super().__init__()
    # Define the reduction linear layer.
    self.reduction = torch.nn.Linear(inDim * 4, outDim)
    # Define the layer normalization.
    self.norm = torch.nn.LayerNorm(inDim * 4)

  # Define the forward pass.
  def forward(self, x, height, width):
    r'''
    Compute the forward pass for the patch merging module.

    Parameters:
      x (Tensor): input tensor of shape (B, N, inDim).
      height (int): current spatial height.
      width (int): current spatial width.

    Returns:
      tuple: merged tensor and new spatial dimensions (height, width).
    '''

    # Get the tensor shape.
    batchSize, numTokens, channels = x.shape
    # Reshape the tensor to spatial dimensions.
    x = x.view(batchSize, height, width, channels)
    # Check if padding is needed.
    if ((height % 2 == 1) or (width % 2 == 1)):
      # Pad the tensor.
      x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
    # Extract the four sub-patches.
    x0 = x[:, 0::2, 0::2, :]
    # Extract the second sub-patch.
    x1 = x[:, 1::2, 0::2, :]
    # Extract the third sub-patch.
    x2 = x[:, 0::2, 1::2, :]
    # Extract the fourth sub-patch.
    x3 = x[:, 1::2, 1::2, :]
    # Concatenate the sub-patches.
    x = torch.cat([x0, x1, x2, x3], dim=-1)
    # Reshape the tensor.
    x = x.view(batchSize, -1, 4 * channels)
    # Apply the layer normalization.
    x = self.norm(x)
    # Apply the reduction layer.
    x = self.reduction(x)
    # Return the merged tensor and new dimensions.
    return x, (height + 1) // 2, (width + 1) // 2


class HierarchicalViT(torch.nn.Module):
  r'''
  Hierarchical Vision Transformer model.

  Parameters:
    numClasses (int): number of output classes.
    embedDim (int): base embedding dimension.
  '''

  # Initialize the hierarchical Vision Transformer.
  def __init__(self, numClasses=2, embedDim=64):
    # Call the parent initialization.
    super().__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=4, stride=4)
    # Define the first stage blocks.
    self.stage1 = torch.nn.ModuleList([HierarchicalBlock(embedDim, 4) for _ in range(2)])
    # Define the first patch merging layer.
    self.merge1 = PatchMerging(embedDim, embedDim * 2)
    # Define the second stage blocks.
    self.stage2 = torch.nn.ModuleList([HierarchicalBlock(embedDim * 2, 8) for _ in range(2)])
    # Define the second patch merging layer.
    self.merge2 = PatchMerging(embedDim * 2, embedDim * 4)
    # Define the third stage blocks.
    self.stage3 = torch.nn.ModuleList([HierarchicalBlock(embedDim * 4, 16) for _ in range(2)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim * 4)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim * 4, numClasses)

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the hierarchical Vision Transformer.

    Parameters:
      x (Tensor): input image tensor of shape (B, 3, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Apply the patch embedding.
    x = self.patchEmbed(x)
    # Get the tensor shape.
    batchSize, channels, height, width = x.shape
    # Flatten and transpose the tensor.
    x = x.flatten(2).transpose(1, 2)
    # Iterate through the first stage blocks.
    for block in self.stage1:
      # Apply the block.
      x = block(x)
    # Apply the first patch merging.
    x, height, width = self.merge1(x, height, width)
    # Iterate through the second stage blocks.
    for block in self.stage2:
      # Apply the block.
      x = block(x)
    # Apply the second patch merging.
    x, height, width = self.merge2(x, height, width)
    # Iterate through the third stage blocks.
    for block in self.stage3:
      # Apply the block.
      x = block(x)
    # Apply the final normalization.
    x = self.norm(x)
    # Global average pooling.
    x = x.mean(dim=1)
    # Return the classification output.
    return self.head(x)


class PatchEmbedding(torch.nn.Module):
  r'''
  Patch embedding module for standard Vision Transformer.

  Parameters:
    imgSize (int): input image size.
    patchSize (int): size of each patch.
    inChannels (int): number of input channels.
    embedDim (int): embedding dimension.
  '''

  # Initialize the patch embedding.
  def __init__(self, imgSize=128, patchSize=16, inChannels=3, embedDim=128):
    # Call the parent initialization.
    super().__init__()
    # Define the convolutional projection.
    self.proj = torch.nn.Conv2d(inChannels, embedDim, kernel_size=patchSize, stride=patchSize)

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the patch embedding.

    Parameters:
      x (Tensor): input image tensor of shape (B, C, H, W).

    Returns:
      torch.Tensor: flattened and transposed tensor of shape (B, N, embedDim).
    '''

    # Apply the projection and flatten.
    return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(torch.nn.Module):
  r'''
  Transformer encoder block for standard Vision Transformer.

  Parameters:
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    mlpDim (int): MLP hidden dimension.
  '''

  # Initialize the encoder block.
  def __init__(self, embedDim, numHeads, mlpDim):
    # Call the parent initialization.
    super().__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(embedDim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(embedDim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(embedDim)
    # Define the multi-layer perceptron.
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(embedDim, mlpDim),
      torch.nn.ReLU(),
      torch.nn.Linear(mlpDim, embedDim)
    )

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the transformer encoder block.

    Parameters:
      x (Tensor): input tensor of shape (B, N, embedDim).

    Returns:
      torch.Tensor: output tensor of shape (B, N, embedDim).
    '''

    # Compute the attention output.
    attnOutput, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
    # Add the residual connection.
    x = x + attnOutput
    # Add the MLP output.
    x = x + self.mlp(self.norm2(x))
    # Return the updated tensor.
    return x


class StandardViT(torch.nn.Module):
  r'''
  Standard Vision Transformer model.

  Parameters:
    imgSize (int): input image size.
    patchSize (int): size of each patch.
    numClasses (int): number of output classes.
    embedDim (int): embedding dimension.
    numHeads (int): number of attention heads.
    depth (int): number of transformer blocks.
    mlpDim (int): MLP hidden dimension.
  '''

  # Initialize the standard Vision Transformer.
  def __init__(self, imgSize=128, patchSize=16, numClasses=2, embedDim=128, numHeads=4, depth=4, mlpDim=256):
    # Call the parent initialization.
    super().__init__()
    # Define the patch embedding.
    self.patchEmbedding = PatchEmbedding(imgSize, patchSize, 3, embedDim)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.randn(1, 1, embedDim))
    # Define the positional encoding parameter.
    self.posEncoding = torch.nn.Parameter(torch.randn(1, (imgSize // patchSize) ** 2 + 1, embedDim))
    # Define the transformer blocks.
    self.transformerBlocks = torch.nn.ModuleList(
      [TransformerEncoderBlock(embedDim, numHeads, mlpDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.mlpHead = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the standard Vision Transformer.

    Parameters:
      x (Tensor): input image tensor of shape (B, C, H, W).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Get the batch size.
    batchSize = x.size(0)
    # Apply the patch embedding.
    x = self.patchEmbedding(x)
    # Expand the class token.
    clsTokens = self.clsToken.expand(batchSize, -1, -1)
    # Concatenate the class token.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional encoding.
    x = x + self.posEncoding
    # Iterate through the transformer blocks.
    for block in self.transformerBlocks:
      # Apply the block.
      x = block(x)
    # Apply the final normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.mlpHead(x)


class CLIPClassifier(torch.nn.Module):
  r'''
  Classifier module for CLIP visual embeddings.

  Parameters:
    embedDim (int): embedding dimension from CLIP.
    numClasses (int): number of output classes.
  '''

  # Initialize the CLIP classifier.
  def __init__(self, embedDim, numClasses):
    # Call the parent initialization.
    super().__init__()
    # Define the fully connected layer.
    self.fc = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass.
  def forward(self, x):
    r'''
    Compute the forward pass for the CLIP classifier.

    Parameters:
      x (Tensor): input embedding tensor of shape (B, embedDim).

    Returns:
      torch.Tensor: classification logits of shape (B, numClasses).
    '''

    # Return the classification output.
    return self.fc(x)


def BuildCLIPModel(numClasses, device):
  r'''
  Build a CLIP-based classification model.

  Parameters:
    numClasses (int): number of output classes.
    device (torch.device): device to place the model on.

  Returns:
    tuple: a tuple containing the CLIP model and the classifier.
  '''

  # Load the CLIP model and preprocessing.
  modelClip, preprocess = clip.load("ViT-B/16", device=device, jit=False)
  # Get the visual dimension.
  visualDim = modelClip.visual.output_dim
  # Initialize the classifier.
  classifier = CLIPClassifier(visualDim, numClasses).to(device)
  # Return the CLIP model and classifier.
  return modelClip, classifier


def BuildViTModel(
  modelName: str,
  numClasses: int,
  device: str,
  imageSize: int = 224,
  usePretrainedCustomModels: bool = False,
  pretrainedModelName: str = "vit_base_patch16_224",
) -> Tuple[torch.nn.Module, Any]:
  r'''
  Build a Vision Transformer (ViT) model based on the specified model name.

  Parameters:
    modelName (str): Name of the ViT model to build.
    numClasses (int): Number of output classes for classification.
    device (str): Device to place the model on (e.g., "cpu" or "cuda").
    imageSize (int): Size of the input images (default is 224).
    usePretrainedCustomModels (bool): Whether to use custom pre-trained models (default is False).
    pretrainedModelName (str): Name of the pre-trained model to use if applicable (default is "vit_base_patch16_224").

  Returns:
    Tuple[torch.nn.Module, Any]: A tuple containing the constructed model and an optional secondary model (e.g., for CLIP). The secondary model is None for most cases except for CLIP, where it returns the CLIP model alongside the classifier.
  '''

  from HMB.PyTorchHelper import CreateTimmModel

  # Check if the model is T2TViT.
  if (modelName == "T2TViT"):
    # Initialize the T2T Vision Transformer.
    model = T2TViT(numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is HierarchicalViT.
  elif (modelName == "HierarchicalViT"):
    # Initialize the Hierarchical Vision Transformer.
    model = HierarchicalViT(numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is StandardViT.
  elif (modelName == "StandardViT"):
    # Initialize the Standard Vision Transformer with the correct image size.
    model = StandardViT(imgSize=imageSize, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is CLIPViT.
  elif (modelName == "CLIPViT"):
    # Build the CLIP model and classifier.
    modelClip, classifier = BuildCLIPModel(numClasses, device)
    # Return the classifier and the CLIP model.
    return classifier, modelClip

  # Check if the model is SwinTransformerV2.
  elif (modelName == "SwinTransformerV2"):
    # SwinV2 requires 256x256 input. Override imageSize for this model.
    model = CreateTimmModel("hf-hub:timm/swinv2_base_window12to16_192to256_22kft1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is SwinTransformer.
  elif (modelName == "SwinTransformer"):
    # Build the Swin Transformer V1 model.
    model = CreateTimmModel("hf-hub:timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is DeiT.
  elif (modelName == "DeiT"):
    # Build the Data-efficient Image Transformer model using timm.
    model = CreateTimmModel("hf-hub:timm/deit_base_patch16_224.fb_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is ConvNeXtV2.
  elif (modelName == "ConvNeXtV2"):
    # Build the ConvNeXt V2 model with improved stability and performance.
    model = CreateTimmModel("hf-hub:timm/convnextv2_base.fcmae_ft_in22k_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is ConvNeXt.
  elif (modelName == "ConvNeXt"):
    # Build the ConvNeXt V1 hybrid CNN-Transformer model using timm.
    model = CreateTimmModel("hf-hub:timm/convnext_base.fb_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is MaxViT.
  elif (modelName == "MaxViT"):
    # Build the Multi-axis Vision Transformer model using timm.
    model = CreateTimmModel("hf-hub:timm/maxvit_tiny_rw_224.sw_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is BEiT.
  elif (modelName == "BEiT"):
    # Build the Bidirectional Encoder representation from Image Transformers model using timm.
    model = CreateTimmModel("hf-hub:timm/beit_base_patch16_224.in22k_ft_in22k_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is FastViT.
  elif (modelName == "FastViT"):
    # Build the FastViT model using timm.
    model = CreateTimmModel("hf-hub:timm/fastvit_t8.apple_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is EVA02.
  elif (modelName == "EVA02"):
    # Build the EVA-02 model using timm.
    model = CreateTimmModel("hf-hub:timm/eva02_base_patch14_224.mim_in22k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is EVA02 Large (High Resolution)
  elif (modelName == "EVA02Large"):
    # Build the EVA-02 Large model pre-trained with MIM at 448x448 resolution.
    model = CreateTimmModel("hf-hub:timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", numClasses)
    # Note: You must set effectiveImageSize = 448 in CreateFitViTModel for this.
    return model, None

  # Check if the model is ConvNeXt Large CLIP
  elif (modelName == "ConvNeXtLargeCLIP"):
    # Build the ConvNeXt Large model with robust CLIP (LAION-2B) pre-training.
    model = CreateTimmModel("hf-hub:timm/convnext_large_mlp.clip_laion2b_augreg_ft_in1k", numClasses)
    return model, None

  # Check if the model is SwinTransformer Large 384
  elif (modelName == "SwinTransformerLarge384"):
    # Build the Swin Large model optimized for 384x384 high-resolution input.
    model = CreateTimmModel("hf-hub:timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k", numClasses)
    return model, None

  # Check if the model is EfficientNetV2 Large
  elif (modelName == "EfficientNetV2Large"):
    # Build the EfficientNetV2 Large model for highly efficient, robust feature extraction.
    model = CreateTimmModel("hf-hub:timm/efficientnetv2_rw_t.ra2_in1k", numClasses)
    return model, None

  # Check if the model is QuantumResNet.
  elif (modelName == "QuantumResNet"):
    # Build the Quantum ResNet hybrid model.
    model = BuildQuantumResNetModel(numClasses, device)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is VisionKANModel.
  elif (modelName == "VisionKANModel"):
    # Initialize the Vision KAN model.
    model = VisionKANModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is NeuralODEViTModel.
  elif (modelName == "NeuralODEViTModel"):
    # Initialize the Neural ODE ViT model.
    model = NeuralODEViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is SpikingViTModel.
  elif (modelName == "SpikingViTModel"):
    # Initialize the Spiking ViT model.
    model = SpikingViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None


  # Check if the model is HypernetworkViTModel.
  elif (modelName == "HypernetworkViTModel"):
    # Initialize the Hypernetwork ViT model.
    model = HypernetworkViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is LiquidSSMViTModel.
  elif (modelName == "LiquidSSMViTModel"):
    # Initialize the liquid state space vision transformer model.
    model = LiquidSSMViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is TestTimeEvolvingViTModel.
  elif (modelName == "TestTimeEvolvingViTModel"):
    # Initialize the test-time evolving vision transformer model.
    model = TestTimeEvolvingViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is TensorNetworkEntangledViTModel.
  elif (modelName == "TensorNetworkEntangledViTModel"):
    # Initialize the tensor network entangled vision transformer model.
    model = TensorNetworkEntangledViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is DiffusionPriorEnergyViTModel.
  elif (modelName == "DiffusionPriorEnergyViTModel"):
    # Initialize the diffusion prior energy vision transformer model.
    model = DiffusionPriorEnergyViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is FractalResonanceViTModel.
  elif (modelName == "FractalResonanceViTModel"):
    # Initialize the fractal resonance vision transformer model.
    model = FractalResonanceViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is TopologicalQuantumViTModel.
  elif (modelName == "TopologicalQuantumViTModel"):
    # Initialize the topological quantum vision transformer model.
    model = TopologicalQuantumViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is HolographicInterferenceViTModel.
  elif (modelName == "HolographicInterferenceViTModel"):
    # Initialize the holographic interference vision transformer model.
    model = HolographicInterferenceViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is NeuromorphicLiquidStateViTModel.
  elif (modelName == "NeuromorphicLiquidStateViTModel"):
    # Initialize the neuromorphic liquid state vision transformer model.
    model = NeuromorphicLiquidStateViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  else:
    # Raise an error for unsupported models.
    raise ValueError(f"Unsupported ViT model: {modelName}")
