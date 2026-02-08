import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

AVAILABLE_UNETS = [
  "original", "legacy",  # Original UNet architecture with standard convolutional blocks and skip connections.
  "dynamic",  # Dynamic convolutional blocks that adapt their weights based on the input features.
  "multiresunet",  # MultiResUNet architecture with multi-resolution blocks for improved feature extraction.
  "r2unet",  # Recurrent residual convolutional blocks for improved feature representation.
  "transunet",  # Transformer-based encoder with a UNet-style decoder for capturing long-range dependencies.
  "cbamunet",  # CBAM integrated into the UNet architecture for enhanced feature representation.
  "efficientunet",  # EfficientNet-based encoder with a lightweight decoder for efficient segmentation.
  "residual",  # Residual connections within the encoder and decoder blocks for improved gradient flow.
  "attention",  # Attention gates for skip connections to focus on relevant features.
  "mobile",  # Depthwise separable convolutions for lightweight segmentation.
  "se",  # Squeeze-and-Excitation blocks for channel-wise feature recalibration.
  "residual_attention",  # Combines residual connections with attention gates for enhanced feature learning.
  "boundary_aware",  # Two parallel branches for segmentation and boundary detection.
  "asppunet",  # Incorporates Atrous Spatial Pyramid Pooling (ASPP) in the bottleneck to capture multi-scale context.
  "denseunet",  # Dense connectivity pattern where each layer receives inputs from all previous layers.
]


# ---------------------------------------------------- #
# Basic utilities for handling model outputs           #
# ---------------------------------------------------- #

def PreparePredTensorToNumpy(
  predTensor: torch.Tensor,
  doScale2Image: bool = False,
) -> np.ndarray:
  r'''
  Utility to convert model output tensor after the sigmoid/softmax activation to a numpy array of class indices.
  It can be used also with the original mask tensor if it is already in the correct format,
  as it handles squeezing and type conversion.

  Short summary:
    Takes the raw output tensor from the model (after activation) and processes it to produce
    a 2D numpy array of class indices. This involves squeezing unnecessary dimensions,
    converting boolean masks to integers if needed, and ensuring the final output is in the
    correct format for evaluation or visualization.

  Parameters:
    predTensor (torch.Tensor): The raw output tensor from the model after activation, expected to be of shape [B, C, H, W] or [B, 1, H, W].
    doScale2Image (bool): If True, applies a threshold to convert probabilities to binary mask. Default False.

  Returns:
    numpy.ndarray: Numpy array of shape [B, H, W] containing class indices.
  '''

  # Convert the prediction tensor to a numpy array.
  predNp = predTensor.cpu().numpy()

  # If prediction has a leading channel dimension of size 1, squeeze it away.
  if (predNp.ndim == 3 and predNp.shape[0] == 1):
    # Squeeze away the channel dimension when it is singleton.
    predNp = np.squeeze(predNp, axis=0)

  # Ensure prediction mask is 2D (H,W). If it's boolean/0-1, keep as ints.
  if (predNp.dtype == np.bool_):
    # Convert boolean mask to uint8 for compatibility.
    predMask = predNp.astype(np.uint8)
  else:
    # Convert prediction mask to integer labels.
    predMask = predNp.astype(np.int64)

  if (doScale2Image):
    # If the prediction is a probability map, apply a threshold to convert to binary mask.
    predMask = (predMask >= 0.5).astype(np.uint8)
    predMask *= 255  # Scale binary mask to 0 and 255 for visualization.
    predMask = predMask.astype(np.uint8)

  return predMask


# ---------------------------------------------------- #
# Basic building blocks for UNet architectures.        #
# ---------------------------------------------------- #

class DoubleConv(nn.Module):
  r'''
  Double convolution block used throughout the U-Net encoder and decoder.

  Short summary:
    Two consecutive 3x3 convolution layers each followed by normalization
    and ReLU activation used as a basic encoder/decoder building block.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Attributes:
    net (torch.nn.Sequential): The internal sequential conv->norm->ReLU layers.

  Returns:
    torch.Tensor: Output feature map of shape [B, outChannels, H, W].
  '''

  # Initialize the block with input and output channels.
  def __init__(self, inChannels: int, outChannels: int):
    # Call super initializer.
    super(DoubleConv, self).__init__()
    # Build the sequential block consisting of two conv->BN->ReLU stages.
    self.net = nn.Sequential(
      nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
      nn.BatchNorm2d(outChannels),
      nn.ReLU(inplace=True),
      nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
      nn.BatchNorm2d(outChannels),
      nn.ReLU(inplace=True),
    )

  # Forward pass through the double conv block.
  def forward(self, x):
    # Return the processed tensor.
    return self.net(x)


class ConfigConv(nn.Module):
  r'''
  Configurable double-convolution block with selectable normalization, dropout and residual option.

  Short summary:
    A DoubleConv-like block with optional normalization type, dropout,
    and an optional residual connection when input and output channels match.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.
    norm (str): Normalization type: "batch" | "instance" | "none".
    dropout (float): Dropout probability applied after the block.
    residual (bool): Whether to add a residual skip when shapes permit.

  Attributes:
    net (torch.nn.Sequential): Core conv->norm->ReLU operations.
    dropout (torch.nn.Module|None): Dropout layer when requested.
    residual (bool): Flag indicating if residual connection will be used.

  Returns:
    torch.Tensor: Processed feature map with same spatial size as input.
  '''

  # Initialize configurable conv block.
  def __init__(
    self,
    inChannels: int,
    outChannels: int,
    norm: str = "batch",
    dropout: float = 0.0,
    residual: bool = False
  ):
    # Call super initializer.
    super(ConfigConv, self).__init__()
    # Store whether we will use a residual connection when channels match.
    self.residual = (residual and (inChannels == outChannels))
    # Create dropout layer if requested.
    self.dropout = (nn.Dropout2d(dropout) if (dropout and dropout > 0.0) else None)

    # Helper to choose normalization layer.
    def make_norm(ch):
      # Use BatchNorm when requested.
      if (norm == "batch"):
        return nn.BatchNorm2d(ch)
      # Use InstanceNorm when requested.
      elif (norm == "instance"):
        return nn.InstanceNorm2d(ch)
      # No normalization.
      else:
        return nn.Identity()

    # Build the sequential block with chosen normalization.
    self.net = nn.Sequential(
      nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
      make_norm(outChannels),
      nn.ReLU(inplace=True),
      nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
      make_norm(outChannels),
      nn.ReLU(inplace=True),
    )

  # Forward pass for the configurable conv block.
  def forward(self, x):
    # Compute the core block output.
    out = self.net(x)
    # Apply dropout when configured.
    if (self.dropout is not None):
      out = self.dropout(out)
    # Apply residual connection when configured and channels match.
    if (self.residual):
      return F.relu(out + x)
    # Return the output when no residual is used.
    return out


class ResidualBlock(nn.Module):
  r'''
  Residual convolutional block.

  Short summary:
    Two conv->BN->ReLU layers with an identity skip connection. A 1x1
    projection is applied to the identity when the number of channels differs.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Attributes:
    conv1, conv2 (torch.nn.Conv2d): Convolution layers.
    bn1, bn2 (torch.nn.BatchNorm2d): BatchNorm layers.
    proj (torch.nn.Conv2d|None): 1x1 conv used when channel projection is needed.

  Returns:
    torch.Tensor: Output tensor with applied residual addition.
  '''

  # Initialize residual block.
  def __init__(self, inChannels, outChannels):
    # Call super initializer.
    super(ResidualBlock, self).__init__()
    # First convolution.
    self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
    # First batch normalization.
    self.bn1 = nn.BatchNorm2d(outChannels)
    # Second convolution.
    self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)
    # Second batch normalization.
    self.bn2 = nn.BatchNorm2d(outChannels)
    # Projection flag when channels differ.
    self.needProj = (inChannels != outChannels)
    # Optional projection to match channels.
    if (self.needProj):
      self.proj = nn.Conv2d(inChannels, outChannels, kernel_size=1)

  # Forward pass for residual block.
  def forward(self, x):
    # Preserve identity for skip connection.
    identity = x
    # First conv->bn->relu.
    out = F.relu(self.bn1(self.conv1(x)))
    # Second conv->bn.
    out = self.bn2(self.conv2(out))
    # Project identity when necessary.
    if (self.needProj):
      identity = self.proj(identity)
    # Add skip connection.
    out += identity
    # Return activated output.
    return F.relu(out)


class AttentionGate(nn.Module):
  r'''
  Attention gating module for skip connections.

  Short summary:
    Lightweight gating unit that computes attention coefficients for encoder skip
    features using a gating signal from the decoder, improving focus on
    relevant spatial locations.

  Parameters:
    F_g (int): Channels of the gating signal from the decoder.
    F_l (int): Channels of the skip connection from the encoder.
    F_int (int): Intermediate channel width inside the gate.

  Attributes:
    W_g, W_x, psi (torch.nn.Sequential): Internal 1x1 conv + BN branches used to compute attention.

  Returns:
    torch.Tensor: Reweighted skip features of shape matching the input skip map.
  '''

  # Initialize attention gate.
  def __init__(self, F_g, F_l, F_int):
    # Call super initializer.
    super(AttentionGate, self).__init__()
    # Linear mapping for gating signal.
    self.W_g = nn.Sequential(
      nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
      nn.BatchNorm2d(F_int),
    )
    # Linear mapping for skip connection.
    self.W_x = nn.Sequential(
      nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
      nn.BatchNorm2d(F_int),
    )
    # Psi branch to compute attention coefficients.
    self.psi = nn.Sequential(
      nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
      nn.BatchNorm2d(1),
      nn.Sigmoid(),
    )
    # ReLU activation for gating.
    self.relu = nn.ReLU(inplace=True)

  # Forward pass for attention gate.
  def forward(self, g, x):
    # Map gating signal.
    g1 = self.W_g(g)
    # Map skip features.
    x1 = self.W_x(x)
    # Combine and activate.
    psi = self.relu(g1 + x1)
    # Compute attention map.
    psi = self.psi(psi)
    # Weight skip features and return.
    return x * psi


class DepthwiseSeparableConv(nn.Module):
  r'''
  Depthwise separable convolution block.

  Short summary:
    Implements a depthwise convolution followed by a pointwise convolution,
    used to reduce parameter counts and computation in lightweight models.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Attributes:
    dw (torch.nn.Conv2d): Depthwise convolution.
    pw (torch.nn.Conv2d): Pointwise convolution.
    bn (torch.nn.BatchNorm2d): Batch normalization on output channels.

  Returns:
    torch.Tensor: Activated output tensor with outChannels channels.
  '''

  # Initialize depthwise separable block.
  def __init__(self, inChannels, outChannels):
    # Call super initializer.
    super(DepthwiseSeparableConv, self).__init__()
    # Depthwise convolution.
    self.dw = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, groups=inChannels)
    # Pointwise convolution.
    self.pw = nn.Conv2d(inChannels, outChannels, kernel_size=1)
    # Batch normalization after pointwise.
    self.bn = nn.BatchNorm2d(outChannels)

  # Forward pass for depthwise separable conv.
  def forward(self, x):
    # Apply depthwise convolution.
    x = self.dw(x)
    # Apply pointwise convolution.
    x = self.pw(x)
    # Apply batch normalization.
    x = self.bn(x)
    # Return activated tensor.
    return F.relu(x)


class SEBlock(nn.Module):
  r'''
  Squeeze-and-Excitation (SE) block.

  Short summary:
    Performs global channel-wise pooling followed by a small bottleneck
    MLP that produces per-channel scaling weights applied to the input.

  Parameters:
    channels (int): Number of input/output channels.
    reduction (int): Reduction ratio for the bottleneck. Default 16.

  Attributes:
    pool (torch.nn.AdaptiveAvgPool2d): Global pooling to (B, C, 1, 1).
    fc (torch.nn.Sequential): Two-layer MLP with sigmoid producing channel weights.

  Returns:
    torch.Tensor: Recalibrated tensor of same shape as input.
  '''

  # Initialize the SE block with channel count and reduction factor.
  def __init__(self, channels, reduction=16):
    # Call the parent initializer.
    super(SEBlock, self).__init__()
    # Compute the reduced channel size for the bottleneck.
    reduced = max(1, channels // reduction)
    # Create an adaptive average pooling layer.
    self.pool = nn.AdaptiveAvgPool2d(1)
    # Build the small fully-connected excitation branch as a Sequential for clarity.
    self.fc = nn.Sequential(
      nn.Linear(channels, reduced),
      nn.ReLU(inplace=True),
      nn.Linear(reduced, channels),
      nn.Sigmoid(),
    )

  # Forward pass for SE block.
  def forward(self, x):
    # Read batch and channel dimensions from input tensor.
    b, c, _, _ = x.size()
    # Apply global average pooling and flatten to (B, C).
    s = self.pool(x).view(b, c)
    # Apply the FC excitation branch and reshape to (B, C, 1, 1).
    s = self.fc(s).view(b, c, 1, 1)
    # Scale the input by the learned channel weights and return.
    return x * s


class MultiResBlock(nn.Module):
  r'''
  Multi-resolution convolution block capturing features at multiple receptive fields.

  Short summary:
    Parallel convolutions with kernel sizes 3x3, 5x5, and 7x7 followed by channel weighting
    to capture multi-scale context within a single block while controlling parameter growth.

  Parameters:
    inChans (int): Input channel count.
    outChans (int): Output channel count (total across all paths).
    alpha (float): Scaling factor for path channel allocation. Default 1.67.

  Attributes:
    conv3x3 (nn.Sequential): 3x3 convolution path.
    conv5x5 (nn.Sequential): 5x5 convolution path (decomposed to two 3x3).
    conv7x7 (nn.Sequential): 7x7 convolution path (decomposed to three 3x3).
    convShortcut (nn.Conv2d): 1x1 shortcut for residual connection.
    batchNorm (nn.BatchNorm2d): Final batch normalization.

  Returns:
    torch.Tensor: Multi-resolution feature map with outChans channels.
  '''

  # Initialize MultiRes block.
  def __init__(self, inChans, outChans, alpha=1.67):
    # Call parent initializer.
    super(MultiResBlock, self).__init__()
    # Compute channel allocation per path using alpha scaling.
    u = int(outChans / alpha)
    # Compute channels for 3x3 path.
    c1 = u
    # Compute channels for 5x5 path.
    c2 = int(u / 2)
    # Compute channels for 7x7 path.
    c3 = outChans - (c1 + c2)
    # Create 3x3 convolution path with batch norm and ReLU.
    self.conv3x3 = nn.Sequential(
      nn.Conv2d(inChans, c1, kernel_size=3, padding=1),
      nn.BatchNorm2d(c1),
      nn.ReLU(inplace=True),
    )
    # Create 5x5 path using two cascaded 3x3 convolutions.
    self.conv5x5 = nn.Sequential(
      nn.Conv2d(inChans, c2, kernel_size=3, padding=1),
      nn.BatchNorm2d(c2),
      nn.ReLU(inplace=True),
      nn.Conv2d(c2, c2, kernel_size=3, padding=1),
      nn.BatchNorm2d(c2),
      nn.ReLU(inplace=True),
    )
    # Create 7x7 path using three cascaded 3x3 convolutions.
    self.conv7x7 = nn.Sequential(
      nn.Conv2d(inChans, c3, kernel_size=3, padding=1),
      nn.BatchNorm2d(c3),
      nn.ReLU(inplace=True),
      nn.Conv2d(c3, c3, kernel_size=3, padding=1),
      nn.BatchNorm2d(c3),
      nn.ReLU(inplace=True),
      nn.Conv2d(c3, c3, kernel_size=3, padding=1),
      nn.BatchNorm2d(c3),
      nn.ReLU(inplace=True),
    )
    # Create shortcut connection with 1x1 convolution when channels differ.
    if (inChans != outChans):
      self.convShortcut = nn.Conv2d(inChans, outChans, kernel_size=1)
    else:
      # Use identity when channels match.
      self.convShortcut = nn.Identity()
    # Create final batch normalization layer.
    self.batchNorm = nn.BatchNorm2d(outChans)

  # Forward pass through MultiRes block.
  def forward(self, x):
    # Process input through 3x3 path.
    path1 = self.conv3x3(x)
    # Process input through 5x5 path.
    path2 = self.conv5x5(x)
    # Process input through 7x7 path.
    path3 = self.conv7x7(x)
    # Concatenate all multi-resolution paths along channel dimension.
    multiRes = torch.cat(
      [
        path1,
        path2,
        path3,
      ],
      dim=1,
    )
    # Apply shortcut connection to input.
    shortcut = self.convShortcut(x)
    # Add residual connection and apply batch normalization.
    out = self.batchNorm(multiRes + shortcut)
    # Return activated output tensor.
    return F.relu(out)


class DenseBlock(nn.Module):
  r'''
  Dense connectivity block with bottleneck layers for parameter efficiency.

  Short summary:
    Stacks multiple bottleneck layers where each layer receives feature maps from
    all preceding layers as input, promoting feature reuse and gradient flow.

  Parameters:
    inChans (int): Input channel count.
    numLayers (int): Number of bottleneck layers in block. Default 4.
    growthRate (int): Channels added per layer. Default 32.
    bnSize (int): Bottleneck expansion factor. Default 4.

  Attributes:
    layers (nn.ModuleList): Sequential bottleneck layers with dense connectivity.

  Returns:
    torch.Tensor: Concatenated output of all layers with inChans + numLayers*growthRate channels.
  '''

  # Initialize Dense block.
  def __init__(self, inChans, numLayers=4, growthRate=32, bnSize=4):
    # Call parent initializer.
    super(DenseBlock, self).__init__()
    # Initialize module list for sequential layers.
    self.layers = nn.ModuleList()
    # Track current input channels for dense connectivity.
    currentChans = inChans
    # Build sequential bottleneck layers.
    for i in range(numLayers):
      # Create bottleneck layer with 1x1 reduction followed by 3x3 expansion.
      layer = nn.Sequential(
        nn.BatchNorm2d(currentChans),
        nn.ReLU(inplace=True),
        nn.Conv2d(currentChans, bnSize * growthRate, kernel_size=1),
        nn.BatchNorm2d(bnSize * growthRate),
        nn.ReLU(inplace=True),
        nn.Conv2d(bnSize * growthRate, growthRate, kernel_size=3, padding=1),
      )
      # Append layer to block sequence.
      self.layers.append(layer)
      # Update channel count after concatenation.
      currentChans += growthRate

  # Forward pass through Dense block.
  def forward(self, x):
    # Initialize list to accumulate feature maps.
    features = [x]
    # Process through each bottleneck layer sequentially.
    for layer in self.layers:
      # Concatenate all previous features as input.
      concated = torch.cat(features, dim=1)
      # Apply current bottleneck transformation.
      out = layer(concated)
      # Append output to feature list for next layer.
      features.append(out)
    # Concatenate all layer outputs including original input.
    return torch.cat(features, dim=1)


class RecurrentConvLayer(nn.Module):
  r'''
  Recurrent convolutional layer with internal state feedback.

  Short summary:
    Applies convolution repeatedly for T timesteps where each step receives feedback
    from its previous output, enabling iterative refinement of spatial features.

  Parameters:
    channels (int): Input/output channel count.
    t (int): Number of recurrent iterations. Default 2.

  Attributes:
    conv (nn.Conv2d): Shared convolution kernel applied at each timestep.

  Returns:
    torch.Tensor: Refined feature map after T recurrent steps.
  '''

  # Initialize recurrent convolution layer.
  def __init__(self, channels, t=2):
    # Call parent initializer.
    super(RecurrentConvLayer, self).__init__()
    # Store recurrence count parameter.
    self.t = t
    # Create shared convolution kernel with residual connectivity.
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    # Create batch normalization for stability.
    self.bn = nn.BatchNorm2d(channels)

  # Forward pass through recurrent layer.
  def forward(self, x):
    # Initialize hidden state with input features.
    hidden = x
    # Iterate for T timesteps with state feedback.
    for _ in range(self.t):
      # Apply convolution to combined input and hidden state.
      hidden = self.conv(x + hidden)
      # Apply batch normalization.
      hidden = self.bn(hidden)
      # Apply ReLU activation.
      hidden = F.relu(hidden)
    # Return final refined feature map.
    return hidden


class ASPP(nn.Module):
  r'''
  Atrous Spatial Pyramid Pooling for multi-scale context aggregation.

  Short summary:
    Parallel dilated convolutions at multiple rates plus image pooling to capture
    objects at different scales within a single feature map.

  Parameters:
    inChans (int): Input channel count.
    outChans (int): Output channel count after fusion.
    dilations (Tuple[int]): Dilation rates for parallel branches. Default (1, 6, 12, 18).

  Attributes:
    conv1x1 (nn.Sequential): 1x1 convolution branch (rate=1).
    conv3x3_1..3 (nn.Sequential): Dilated 3x3 convolutions at specified rates.
    imagePool (nn.Sequential): Global average pooling branch.
    project (nn.Sequential): Final projection and dropout.

  Returns:
   torch.Tensor: Context-enriched feature map with outChans channels.
  '''

  # Initialize ASPP module.
  def __init__(self, inChans, outChans, dilations=(1, 6, 12, 18)):
    # Call parent initializer.
    super(ASPP, self).__init__()
    # Create 1x1 convolution branch.
    self.conv1x1 = nn.Sequential(
      nn.Conv2d(inChans, outChans, kernel_size=1, bias=False),
      nn.BatchNorm2d(outChans),
      nn.ReLU(inplace=True),
    )
    # Create first dilated 3x3 convolution branch.
    self.conv3x3_1 = nn.Sequential(
      nn.Conv2d(inChans, outChans, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
      nn.BatchNorm2d(outChans),
      nn.ReLU(inplace=True),
    )
    # Create second dilated 3x3 convolution branch.
    self.conv3x3_2 = nn.Sequential(
      nn.Conv2d(inChans, outChans, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
      nn.BatchNorm2d(outChans),
      nn.ReLU(inplace=True),
    )
    # Create third dilated 3x3 convolution branch.
    self.conv3x3_3 = nn.Sequential(
      nn.Conv2d(inChans, outChans, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False),
      nn.BatchNorm2d(outChans),
      nn.ReLU(inplace=True),
    )
    # Create image pooling branch with adaptive pooling.
    self.imagePool = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(inChans, outChans, kernel_size=1, bias=False),
      # nn.BatchNorm2d(outChans),
      nn.ReLU(inplace=True),
    )
    # Create final projection layer after concatenation.
    self.project = nn.Sequential(
      nn.Conv2d(outChans * 5, outChans, kernel_size=1, bias=False),
      nn.BatchNorm2d(outChans),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
    )

  # Forward pass through ASPP.
  def forward(self, x):
    # Extract spatial dimensions for later upsampling.
    h, w = x.size(2), x.size(3)
    # Apply 1x1 convolution branch.
    feat1 = self.conv1x1(x)
    # Apply first dilated convolution branch.
    feat2 = self.conv3x3_1(x)
    # Apply second dilated convolution branch.
    feat3 = self.conv3x3_2(x)
    # Apply third dilated convolution branch.
    feat4 = self.conv3x3_3(x)
    # Apply image pooling branch.
    feat5 = self.imagePool(x)
    # Upsample pooled features to original spatial dimensions.
    feat5 = F.interpolate(
      feat5,
      size=(h, w),
      mode="bilinear",
      align_corners=True,
    )
    # Concatenate all five branches along channel dimension.
    concat = torch.cat(
      [
        feat1,
        feat2,
        feat3,
        feat4,
        feat5,
      ],
      dim=1,
    )
    # Project concatenated features to output dimension.
    return self.project(concat)


class PatchEmbedding(nn.Module):
  r'''
  2D image to patch embedding with optional positional encoding.

  Short summary:
    Converts input image into non-overlapping patches via convolutional projection
    and flattens them into a sequence for transformer processing.

  Parameters:
    inChans (int): Input channel count.
    embedDim (int): Embedding dimension per patch. Default 256.
    patchSize (int): Patch size (height=width). Default 4.

  Attributes:
    proj (nn.Conv2d): Convolutional patch projection layer.

  Returns:
    Tuple[torch.Tensor, Tuple[int, int]]: Patch sequence [B, N, embedDim] and (patchH, patchW).
  '''

  # Initialize patch embedding module.
  def __init__(self, inChans, embedDim=256, patchSize=4):
    # Call parent initializer.
    super(PatchEmbedding, self).__init__()
    # Store patch size parameter.
    self.patchSize = patchSize
    # Create convolutional projection layer.
    self.proj = nn.Conv2d(inChans, embedDim, kernel_size=patchSize, stride=patchSize)

  # Forward pass for patch embedding.
  def forward(self, x):
    # Apply convolutional patch projection.
    x = self.proj(x)
    # Extract spatial dimensions of patch grid.
    patchH = x.shape[2]
    # Extract patch width dimension.
    patchW = x.shape[3]
    # Permute to [B, C, H, W] -> [B, H, W, C] format.
    x = x.permute(0, 2, 3, 1)
    # Flatten spatial dimensions to sequence format.
    x = x.view(x.shape[0], patchH * patchW, x.shape[-1])
    # Return patch sequence and grid dimensions.
    return (
      x,
      (
        patchH,
        patchW,
      ),
    )


class TransformerBlock(nn.Module):
  r'''
  Standard transformer encoder block with pre-normalization.

  Short summary:
    Multi-head self-attention followed by position-wise feed-forward network
    with layer normalization and residual connections before each sub-layer.

  Parameters:
    embedDim (int): Embedding dimension.
    numHeads (int): Number of attention heads. Default 8.
    mlpRatio (float): Hidden dimension ratio for MLP. Default 4.0.
    dropout (float): Dropout probability. Default 0.1.

  Attributes:
    norm1, norm2 (nn.LayerNorm): Pre-normalization layers.
    attn (nn.MultiheadAttention): Multi-head self-attention module.
    mlp (nn.Sequential): Two-layer MLP with GELU activation.
    drop (nn.Dropout): Dropout layer after MLP.

  Returns:
    torch.Tensor: Transformed sequence of same shape as input.
  '''

  # Initialize transformer block.
  def __init__(self, embedDim, numHeads=8, mlpRatio=4.0, dropout=0.1):
    # Call parent initializer.
    super(TransformerBlock, self).__init__()
    # Create first layer normalization.
    self.norm1 = nn.LayerNorm(embedDim)
    # Create multi-head self-attention module.
    self.attn = nn.MultiheadAttention(embedDim, numHeads, dropout=dropout, batch_first=True)
    # Create second layer normalization.
    self.norm2 = nn.LayerNorm(embedDim)
    # Compute hidden dimension for MLP.
    hiddenDim = int(embedDim * mlpRatio)
    # Build MLP sequential block.
    self.mlp = nn.Sequential(
      nn.Linear(embedDim, hiddenDim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hiddenDim, embedDim),
      nn.Dropout(dropout),
    )

  # Forward pass through transformer block.
  def forward(self, x):
    # Apply pre-normalization before attention.
    xNorm = self.norm1(x)
    # Compute self-attention with residual connection.
    attnOut, _ = self.attn(xNorm, xNorm, xNorm)
    # Add residual connection.
    x = x + attnOut
    # Apply pre-normalization before MLP.
    xNorm = self.norm2(x)
    # Apply MLP with residual connection.
    x = x + self.mlp(xNorm)
    # Return transformed sequence.
    return x


class CBAM(nn.Module):
  r'''
  Convolutional Block Attention Module (channel + spatial attention).

  Short summary:
    Sequential channel attention followed by spatial attention to adaptively
    refine feature maps along both channel and spatial dimensions.

  Parameters:
    channels (int): Input/output channel count.
    reduction (int): Reduction ratio for channel attention. Default 16.

  Attributes:
    channelAttn (ChannelAttn): Channel attention submodule.
    spatialAttn (SpatialAttn): Spatial attention submodule.

  Returns:
    torch.Tensor: Attention-refined feature map with same shape as input.
  '''

  # Initialize CBAM module.
  def __init__(self, channels, reduction=16):
    # Call parent initializer.
    super(CBAM, self).__init__()
    # Create channel attention submodule.
    self.channelAttn = ChannelAttn(channels, reduction)
    # Create spatial attention submodule.
    self.spatialAttn = SpatialAttn()

  # Forward pass through CBAM.
  def forward(self, x):
    # Apply channel attention refinement.
    x = self.channelAttn(x)
    # Apply spatial attention refinement.
    x = self.spatialAttn(x)
    # Return fully refined feature map.
    return x


class ChannelAttn(nn.Module):
  r'''
  Channel attention branch of CBAM using squeeze-and-excitation.

  Short summary:
    Global average and max pooling followed by shared MLP to compute
    channel-wise attention weights that rescale feature channels.

  Parameters:
    channels (int): Input channel count.
    reduction (int): Reduction ratio for bottleneck MLP. Default 16.

  Attributes:
   mlp (nn.Sequential): Shared MLP for both pooling streams.

  Returns:
   torch.Tensor: Channel-refined feature map.
  '''

  # Initialize channel attention module.
  def __init__(self, channels, reduction=16):
    # Call parent initializer.
    super(ChannelAttn, self).__init__()
    # Compute reduced dimension for bottleneck.
    reduced = max(1, channels // reduction)
    # Create shared MLP for both pooling streams.
    self.mlp = nn.Sequential(
      nn.Linear(channels, reduced),
      nn.ReLU(inplace=True),
      nn.Linear(reduced, channels),
    )

  # Forward pass through channel attention.
  def forward(self, x):
    # Extract batch and channel dimensions.
    b, c, _, _ = x.size()
    # Apply global average pooling and flatten.
    avgPooled = F.adaptive_avg_pool2d(x, 1).view(b, c)
    # Apply global max pooling and flatten.
    maxPooled = F.adaptive_max_pool2d(x, 1).view(b, c)
    # Process average pooled features through MLP.
    avgOut = self.mlp(avgPooled)
    # Process max pooled features through same MLP.
    maxOut = self.mlp(maxPooled)
    # Combine both streams with element-wise addition.
    attn = avgOut + maxOut
    # Apply sigmoid activation for attention weights.
    attn = torch.sigmoid(attn).view(b, c, 1, 1)
    # Scale input features by attention weights.
    return x * attn


class SpatialAttn(nn.Module):
  r'''
  Spatial attention branch of CBAM using channel aggregation.

  Short summary:
    Channel-wise average and max pooling followed by convolution to compute
    spatial attention map that highlights important regions in feature maps.

  Parameters:
    kernelSize (int): Convolution kernel size for spatial attention. Default 7.

  Attributes:
    conv (nn.Conv2d): Convolution to generate spatial attention map.

  Returns:
    torch.Tensor: Spatially-refined feature map.
  '''

  # Initialize spatial attention module.
  def __init__(self, kernelSize=7):
    # Call parent initializer.
    super(SpatialAttn, self).__init__()
    # Create convolution layer for spatial attention map generation.
    self.conv = nn.Conv2d(
      2,
      1,
      kernel_size=kernelSize,
      padding=kernelSize // 2,
      bias=False,
    )

  # Forward pass through spatial attention.
  def forward(self, x):
    # Apply channel-wise average pooling.
    avgPooled = torch.mean(x, dim=1, keepdim=True)
    # Apply channel-wise max pooling.
    maxPooled = torch.max(x, dim=1, keepdim=True)[0]
    # Concatenate both pooling streams along channel dimension.
    concat = torch.cat(
      [
        avgPooled,
        maxPooled,
      ],
      dim=1,
    )
    # Generate spatial attention map via convolution.
    attn = self.conv(concat)
    # Apply sigmoid activation for attention weights.
    attn = torch.sigmoid(attn)
    # Scale input features by spatial attention map.
    return x * attn


# ---------------------------------------------------- #
# UNets implementations.                               #
# ---------------------------------------------------- #


class UNet(nn.Module):
  r'''
  Standard 2D U-Net architecture.

  Short summary:
    A typical encoder-decoder U-Net with four downsampling stages, a bottleneck,
    and four symmetric upsampling stages. Supports learned transpose convolution
    upsampling or bilinear upsampling followed by 1x1 projection.

  Parameters:
    inputChannels (int): Number of input image channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Filters in the first stage. Default 64.
    useConvTranspose2d (bool): Use ConvTranspose2d when True; otherwise use bilinear upsample.

  Attributes:
    enc1..enc4, center (torch.nn.Module): Encoder and bottleneck blocks.
    up1..up4, dec1..dec4 (torch.nn.Module): Upsampling and decoder blocks.
    finalConv (torch.nn.Conv2d): 1x1 conv to map to logits.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize UNet with input channels and number of classes.
  def __init__(
    self,
    inputChannels: int = 3,
    numClasses: int = 2,
    baseChannels: int = 64,
    useConvTranspose2d=True,
  ):
    # Call super initializer.
    super(UNet, self).__init__()
    # Save constructor parameters.
    self.inputChannels = inputChannels
    self.numClasses = numClasses
    self.baseChannels = baseChannels
    # Create encoder stage 1 block.
    self.enc1 = DoubleConv(self.inputChannels, self.baseChannels)
    # Create pool after encoder stage 1.
    self.pool1 = nn.MaxPool2d(2)
    # Create encoder stage 2 block.
    self.enc2 = DoubleConv(self.baseChannels, (self.baseChannels * 2))
    # Create pool after encoder stage 2.
    self.pool2 = nn.MaxPool2d(2)
    # Create encoder stage 3 block.
    self.enc3 = DoubleConv((self.baseChannels * 2), (self.baseChannels * 4))
    # Create pool after encoder stage 3.
    self.pool3 = nn.MaxPool2d(2)
    # Create encoder stage 4 block.
    self.enc4 = DoubleConv((self.baseChannels * 4), (self.baseChannels * 8))
    # Create pool after encoder stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Create center block.
    self.center = DoubleConv((self.baseChannels * 8), (self.baseChannels * 16))

    # Choose upsampling strategy based on the flag.
    if (useConvTranspose2d):
      # Learned upsampling using transposed convolutions.
      self.up4 = nn.ConvTranspose2d((self.baseChannels * 16), (self.baseChannels * 8), kernel_size=2, stride=2)
      # Decoder conv for stage 4.
      self.dec4 = DoubleConv((self.baseChannels * 16), (self.baseChannels * 8))
      # Learned upsampling for stage 3.
      self.up3 = nn.ConvTranspose2d((self.baseChannels * 8), (self.baseChannels * 4), kernel_size=2, stride=2)
      # Decoder conv for stage 3.
      self.dec3 = DoubleConv((self.baseChannels * 8), (self.baseChannels * 4))
      # Learned upsampling for stage 2.
      self.up2 = nn.ConvTranspose2d((self.baseChannels * 4), (self.baseChannels * 2), kernel_size=2, stride=2)
      # Decoder conv for stage 2.
      self.dec2 = DoubleConv((self.baseChannels * 4), (self.baseChannels * 2))
      # Learned upsampling for stage 1.
      self.up1 = nn.ConvTranspose2d((self.baseChannels * 2), self.baseChannels, kernel_size=2, stride=2)
      # Decoder conv for stage 1.
      self.dec1 = DoubleConv((self.baseChannels * 2), self.baseChannels)
    else:
      # Bilinear upsampling followed by 1x1 conv for stable resizing.
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((self.baseChannels * 16), (self.baseChannels * 8), kernel_size=1),
      )
      # Decoder conv for stage 4 using DoubleConv.
      self.dec4 = DoubleConv((self.baseChannels * 16), (self.baseChannels * 8))
      # Bilinear upsampling for stage 3.
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((self.baseChannels * 8), (self.baseChannels * 4), kernel_size=1),
      )
      # Decoder conv for stage 3.
      self.dec3 = DoubleConv((self.baseChannels * 8), (self.baseChannels * 4))
      # Bilinear upsampling for stage 2.
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((self.baseChannels * 4), (self.baseChannels * 2), kernel_size=1),
      )
      # Decoder conv for stage 2.
      self.dec2 = DoubleConv((self.baseChannels * 4), (self.baseChannels * 2))
      # Bilinear upsampling for stage 1.
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((self.baseChannels * 2), self.baseChannels, kernel_size=1),
      )
      # Decoder conv for stage 1.
      self.dec1 = DoubleConv((self.baseChannels * 2), self.baseChannels)

    # Final 1x1 convolution to produce logits for each class.
    self.finalConv = nn.Conv2d(self.baseChannels, self.numClasses, kernel_size=1)

  # Forward pass for UNet returns logits with shape [B, numClasses, H, W].
  def forward(self, x):
    # Run encoder stage 1 and store features.
    e1 = self.enc1.forward(x)
    # Downsample after encoder stage 1.
    p1 = self.pool1(e1)
    # Run encoder stage 2 and store features.
    e2 = self.enc2.forward(p1)
    # Downsample after encoder stage 2.
    p2 = self.pool2(e2)
    # Run encoder stage 3 and store features.
    e3 = self.enc3.forward(p2)
    # Downsample after encoder stage 3.
    p3 = self.pool3(e3)
    # Run encoder stage 4 and store features.
    e4 = self.enc4.forward(p3)
    # Downsample after encoder stage 4.
    p4 = self.pool4(e4)
    # Run center block.
    c = self.center.forward(p4)

    # Upsample from center for decoder stage 4.
    u4 = self.up4(c)
    # Concatenate encoder features for skip connection using a multi-line call.
    u4 = torch.cat(
      [
        u4,
        e4,
      ],
      dim=1,
    )
    # Decode stage 4.
    d4 = self.dec4.forward(u4)

    # Upsample for decoder stage 3.
    u3 = self.up3(d4)
    # Concatenate encoder features for skip connection using a multi-line call.
    u3 = torch.cat(
      [
        u3,
        e3,
      ],
      dim=1,
    )
    # Decode stage 3.
    d3 = self.dec3.forward(u3)

    # Upsample for decoder stage 2.
    u2 = self.up2(d3)
    # Concatenate encoder features for skip connection using a multi-line call.
    u2 = torch.cat(
      [
        u2,
        e2,
      ],
      dim=1,
    )
    # Decode stage 2.
    d2 = self.dec2.forward(u2)

    # Upsample for decoder stage 1.
    u1 = self.up1(d2)
    # Concatenate encoder features for skip connection using a multi-line call.
    u1 = torch.cat(
      [
        u1,
        e1,
      ],
      dim=1,
    )
    # Decode stage 1.
    d1 = self.dec1.forward(u1)
    # Compute final logits.
    logits = self.finalConv(d1)
    # Return logits tensor.
    return logits


class DynamicUNet(nn.Module):
  r'''
  Configurable U-Net implementation with variable depth.

  Short summary:
    Generalized U-Net that allows customizing depth, normalization type,
    dropout, residual usage and upsampling mode. Useful for experimenting
    with architectures without duplicating code.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    depth (int): Number of downsampling stages. Default 4.
    upMode (str): "transpose" or "bilinear" upsampling. Default "transpose".
    norm (str): Normalization type: "batch"|"instance"|"none". Default "batch".
    dropout (float): Dropout probability applied inside config blocks. Default 0.0.
    residual (bool): Whether to use residual connections inside blocks. Default False.

  Attributes:
    encs (nn.ModuleList): Encoder blocks.
    pools (nn.ModuleList): Pooling layers.
    ups (nn.ModuleList): Upsampling layers.
    decs (nn.ModuleList): Decoder blocks.
    finalConv (torch.nn.Conv2d): 1x1 conv to logits.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize dynamic UNet.
  def __init__(self, inputChannels: int = 3, numClasses: int = 2, baseChannels: int = 64, depth: int = 4,
               upMode: str = "transpose", norm: str = "batch", dropout: float = 0.0, residual: bool = False):
    # Call super initializer.
    super(DynamicUNet, self).__init__()
    # Validate input values.
    assert (depth >= 1), "depth must be >= 1"
    assert (upMode in ("transpose", "bilinear")), "upMode must be 'transpose' or 'bilinear'"

    # Save initialization parameters.
    self.inputChannels = inputChannels
    self.numClasses = numClasses
    self.baseChannels = baseChannels
    self.depth = depth
    self.upMode = upMode

    # Prepare encoder lists.
    self.encs = nn.ModuleList()
    self.pools = nn.ModuleList()
    # Set initial channel counters.
    inCh = inputChannels
    channels = baseChannels

    # Build encoder stages.
    for i in range(depth):
      # Append a configurable conv block for this encoder stage.
      self.encs.append(ConfigConv(inCh, channels, norm=norm, dropout=dropout, residual=residual))
      # Append a pooling layer after the encoder block.
      self.pools.append(nn.MaxPool2d(2))
      # Update channel counters for next stage.
      inCh = channels
      channels = (channels * 2)

    # Build center block.
    self.center = ConfigConv(inCh, channels, norm=norm, dropout=dropout, residual=residual)

    # Build decoder upsampling and conv lists.
    self.ups = nn.ModuleList()
    self.decs = nn.ModuleList()
    for i in range(depth):
      # Choose upsampling block based on selected mode.
      if (upMode == "transpose"):
        # Learned upsample block.
        self.ups.append(nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2))
      else:
        # Bilinear upsample followed by 1x1 conv.
        self.ups.append(
          nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
          )
        )
      # Append a configurable conv decoder block.
      self.decs.append(ConfigConv(channels, channels // 2, norm=norm, dropout=dropout, residual=residual))
      # Halve channels for the next decoder iteration.
      channels = (channels // 2)

    # Final 1x1 conv to map to numClasses.
    self.finalConv = nn.Conv2d(channels, numClasses, kernel_size=1)

  # Forward pass for DynamicUNet.
  def forward(self, x):
    # Collect encoder features for skip connections.
    features = []
    for enc, pool in zip(self.encs, self.pools):
      # Forward through encoder block.
      x = enc.forward(x)
      # Store the encoder feature map.
      features.append(x)
      # Pool to downsample.
      x = pool(x)

    # Forward through center block.
    x = self.center.forward(x)

    # Decoder: iterate in reverse pairing with stored encoder features.
    for up, dec, feat in zip(self.ups, self.decs, reversed(features)):
      # Upsample current feature.
      x = up(x)
      # Align spatial shapes when needed.
      if (x.size(2) != feat.size(2)) or (x.size(3) != feat.size(3)):
        x = F.interpolate(x, size=(feat.size(2), feat.size(3)), mode="bilinear", align_corners=True)
      # Concatenate skip features.
      x = torch.cat(
        [
          x,
          feat,
        ],
        dim=1,
      )
      # Decode concatenated features.
      x = dec.forward(x)

    # Map to logits with final conv.
    logits = self.finalConv(x)
    # Return logits.
    return logits


class AttentionUNet(nn.Module):
  r'''
  Attention U-Net variant with gating on skip connections.

  Short summary:
    Applies attention gating to encoder skip features before merging them into the decoder,
    improving localization and suppressing irrelevant responses in the skip maps.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    useConvTranspose2d (bool): Use ConvTranspose2d for upsampling when True.

  Attributes:
    enc1..enc4, center: Encoder/bottleneck blocks.
    att1..att4 (AttentionGate): Attention gating modules for skips.
    up1..up4, dec1..dec4: Decoder modules.
    finalConv (torch.nn.Conv2d): 1x1 conv mapping to logits.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize AttentionUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call super initializer.
    super(AttentionUNet, self).__init__()
    # Encoder stage 1.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    # Pool after stage 1.
    self.pool1 = nn.MaxPool2d(2)
    # Encoder stage 2.
    self.enc2 = DoubleConv(baseChannels, (baseChannels * 2))
    # Pool after stage 2.
    self.pool2 = nn.MaxPool2d(2)
    # Encoder stage 3.
    self.enc3 = DoubleConv((baseChannels * 2), (baseChannels * 4))
    # Pool after stage 3.
    self.pool3 = nn.MaxPool2d(2)
    # Encoder stage 4.
    self.enc4 = DoubleConv((baseChannels * 4), (baseChannels * 8))
    # Pool after stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Center block.
    self.center = DoubleConv((baseChannels * 8), (baseChannels * 16))

    # Choose upsampling method.
    if (useConvTranspose2d):
      # Learned upsampling layers.
      self.up4 = nn.ConvTranspose2d((baseChannels * 16), (baseChannels * 8), kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d((baseChannels * 8), (baseChannels * 4), kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d((baseChannels * 4), (baseChannels * 2), kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d((baseChannels * 2), baseChannels, kernel_size=2, stride=2)
    else:
      # Bilinear upsampling blocks as sequential containers.
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 16), (baseChannels * 8), kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 8), (baseChannels * 4), kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 4), (baseChannels * 2), kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 2), baseChannels, kernel_size=1),
      )

    # Instantiate attention gates for each skip connection.
    self.att4 = AttentionGate(F_g=(baseChannels * 8), F_l=(baseChannels * 8), F_int=(baseChannels * 4))
    self.att3 = AttentionGate(F_g=(baseChannels * 4), F_l=(baseChannels * 4), F_int=(baseChannels * 2))
    self.att2 = AttentionGate(F_g=(baseChannels * 2), F_l=(baseChannels * 2), F_int=baseChannels)
    self.att1 = AttentionGate(F_g=baseChannels, F_l=baseChannels, F_int=(baseChannels // 2 if baseChannels >= 2 else 1))

    # Decoder convs as DoubleConv blocks.
    self.dec4 = DoubleConv((baseChannels * 16), (baseChannels * 8))
    self.dec3 = DoubleConv((baseChannels * 8), (baseChannels * 4))
    self.dec2 = DoubleConv((baseChannels * 4), (baseChannels * 2))
    self.dec1 = DoubleConv((baseChannels * 2), baseChannels)
    # Final 1x1 conv for logits.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for AttentionUNet.
  def forward(self, x):
    # Encode with pooling at each stage and collect features.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3.forward(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4.forward(p3)
    p4 = self.pool4(e4)
    # Process center block.
    c = self.center.forward(p4)

    # Upsample and apply attention gating for stage 4.
    u4 = self.up4(c)
    e4_att = self.att4.forward(u4, e4)
    # Decode stage 4 using multi-line concatenation.
    d4 = self.dec4.forward(
      torch.cat(
        [
          u4,
          e4_att,
        ],
        dim=1,
      )
    )

    # Upsample and apply attention gating for stage 3.
    u3 = self.up3(d4)
    e3_att = self.att3.forward(u3, e3)
    # Decode stage 3 using multi-line concatenation.
    d3 = self.dec3.forward(
      torch.cat(
        [
          u3,
          e3_att,
        ],
        dim=1,
      )
    )

    # Upsample and apply attention gating for stage 2.
    u2 = self.up2(d3)
    e2_att = self.att2.forward(u2, e2)
    # Decode stage 2 using multi-line concatenation.
    d2 = self.dec2.forward(
      torch.cat(
        [
          u2,
          e2_att,
        ],
        dim=1,
      )
    )

    # Upsample and apply attention gating for stage 1.
    u1 = self.up1(d2)
    e1_att = self.att1.forward(u1, e1)
    # Decode stage 1 using multi-line concatenation.
    d1 = self.dec1.forward(
      torch.cat(
        [
          u1,
          e1_att,
        ],
        dim=1,
      )
    )

    # Compute final logits and return.
    logits = self.finalConv(d1)
    return logits


class MobileUNet(nn.Module):
  r'''
  Lightweight U-Net using depthwise separable convolutions.

  Short summary:
    Efficient UNet variant that replaces standard convolutions with
    depthwise separable convolutions to reduce parameters and compute.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 32.
    useConvTranspose2d (bool): Use ConvTranspose2d for upsampling when True.

  Attributes:
    enc1..enc4, center: Encoder and bottleneck sequences.
    up1..up4, dec1..dec4: Decoder modules.
    finalConv (torch.nn.Conv2d): 1x1 conv for logits.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize MobileUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=32, useConvTranspose2d=True):
    # Call super initializer.
    super(MobileUNet, self).__init__()
    # Encoder stage 1 using depthwise separable convs.
    self.enc1 = nn.Sequential(
      DepthwiseSeparableConv(inputChannels, baseChannels),
      DepthwiseSeparableConv(baseChannels, baseChannels),
    )
    # Pool after stage 1.
    self.pool1 = nn.MaxPool2d(2)
    # Encoder stage 2.
    self.enc2 = nn.Sequential(
      DepthwiseSeparableConv(baseChannels, (baseChannels * 2)),
      DepthwiseSeparableConv((baseChannels * 2), (baseChannels * 2)),
    )
    # Pool after stage 2.
    self.pool2 = nn.MaxPool2d(2)
    # Encoder stage 3.
    self.enc3 = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 2), (baseChannels * 4)),
      DepthwiseSeparableConv((baseChannels * 4), (baseChannels * 4)),
    )
    # Pool after stage 3.
    self.pool3 = nn.MaxPool2d(2)
    # Encoder stage 4.
    self.enc4 = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 4), (baseChannels * 8)),
      DepthwiseSeparableConv((baseChannels * 8), (baseChannels * 8)),
    )
    # Pool after encoder stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Create center block.
    self.center = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 8), (baseChannels * 16)),
      DepthwiseSeparableConv((baseChannels * 16), (baseChannels * 16)),
    )

    # Choose upsampling method.
    if (useConvTranspose2d):
      # Learned upsampling via transpose convs.
      self.up4 = nn.ConvTranspose2d((baseChannels * 16), (baseChannels * 8), kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d((baseChannels * 8), (baseChannels * 4), kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d((baseChannels * 4), (baseChannels * 2), kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d((baseChannels * 2), baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 16), (baseChannels * 8), kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 8), (baseChannels * 4), kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 4), (baseChannels * 2), kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 2), baseChannels, kernel_size=1),
      )

    # Decoder convolutional sequences.
    self.dec4 = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 16), (baseChannels * 8)),
      DepthwiseSeparableConv((baseChannels * 8), (baseChannels * 8)),
    )
    # Decoder stage 3.
    self.dec3 = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 8), (baseChannels * 4)),
      DepthwiseSeparableConv((baseChannels * 4), (baseChannels * 4)),
    )
    # Decoder stage 2.
    self.dec2 = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 4), (baseChannels * 2)),
      DepthwiseSeparableConv((baseChannels * 2), (baseChannels * 2)),
    )
    # Decoder stage 1.
    self.dec1 = nn.Sequential(
      DepthwiseSeparableConv((baseChannels * 2), baseChannels),
      DepthwiseSeparableConv(baseChannels, baseChannels),
    )
    # Final 1x1 conv to produce logits.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for MobileUNet.
  def forward(self, x):
    # Encoder forward and pooling.
    e1 = self.enc1(x)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3)
    p4 = self.pool4(e4)
    # Center forward.
    c = self.center(p4)
    # Upsample and align.
    u4 = self.up4(c)
    # Align spatial dims when necessary using a multi-line interpolate call.
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(
        u4,
        size=(e4.size(2), e4.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Decode stage 4 using multi-line concatenation.
    d4 = self.dec4(
      torch.cat(
        [
          u4,
          e4,
        ],
        dim=1,
      )
    )

    # Upsample and align for stage 3.
    u3 = self.up3(d4)
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(
        u3,
        size=(e3.size(2), e3.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Decode stage 3 using multi-line concatenation.
    d3 = self.dec3(
      torch.cat(
        [
          u3,
          e3,
        ],
        dim=1,
      )
    )

    # Upsample and align for stage 2.
    u2 = self.up2(d3)
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(
        u2,
        size=(e2.size(2), e2.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Decode stage 2 using multi-line concatenation.
    d2 = self.dec2(
      torch.cat(
        [
          u2,
          e2,
        ],
        dim=1,
      )
    )

    # Upsample and align for stage 1.
    u1 = self.up1(d2)
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(
        u1,
        size=(e1.size(2), e1.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Decode stage 1 using multi-line concatenation.
    d1 = self.dec1(
      torch.cat(
        [
          u1,
          e1,
        ],
        dim=1,
      )
    )
    # Final logits and return.
    logits = self.finalConv(d1)
    return logits


class ResidualUNet(nn.Module):
  r'''
  Residual U-Net variant built from ResidualBlock components.

  Short summary:
    A U-Net where encoder and decoder stages use residual blocks to
    ease optimization and improve gradient flow for deeper models.

  Parameters:
    inputChannels (int): Number of input channels.
    numClasses (int): Number of output classes.
    baseChannels (int): Base number of filters.
    useConvTranspose2d (bool): Use ConvTranspose2d when True.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize ResidualUNet.
  def __init__(self, inputChannels, numClasses, baseChannels, useConvTranspose2d=True):
    # Call super initializer.
    super(ResidualUNet, self).__init__()
    # Encoder stage 1 as residual block.
    self.enc1 = ResidualBlock(inputChannels, baseChannels)
    # Pool after stage 1.
    self.pool1 = nn.MaxPool2d(2)
    # Encoder stage 2.
    self.enc2 = ResidualBlock(baseChannels, (baseChannels * 2))
    # Pool after stage 2.
    self.pool2 = nn.MaxPool2d(2)
    # Encoder stage 3.
    self.enc3 = ResidualBlock((baseChannels * 2), (baseChannels * 4))
    # Pool after stage 3.
    self.pool3 = nn.MaxPool2d(2)
    # Encoder stage 4.
    self.enc4 = ResidualBlock((baseChannels * 4), (baseChannels * 8))
    # Pool after encoder stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Create center block.
    self.center = ResidualBlock((baseChannels * 8), (baseChannels * 16))

    # Choose upsampling method.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d((baseChannels * 16), (baseChannels * 8), kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d((baseChannels * 8), (baseChannels * 4), kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d((baseChannels * 4), (baseChannels * 2), kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d((baseChannels * 2), baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 16), (baseChannels * 8), kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 8), (baseChannels * 4), kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 4), (baseChannels * 2), kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d((baseChannels * 2), baseChannels, kernel_size=1),
      )

    # Decoder residual blocks.
    self.dec4 = ResidualBlock((baseChannels * 16), (baseChannels * 8))
    self.dec3 = ResidualBlock((baseChannels * 8), (baseChannels * 4))
    self.dec2 = ResidualBlock((baseChannels * 4), (baseChannels * 2))
    self.dec1 = ResidualBlock((baseChannels * 2), baseChannels)
    # Final logits conv.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for the nested ResidualUNet.
  def forward(self, x):
    # Encoder forward passes.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3.forward(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4.forward(p3)
    p4 = self.pool4(e4)
    # Center forward.
    c = self.center.forward(p4)

    # Upsample and align for stage 4.
    u4 = self.up4(c)
    if (u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3)):
      u4 = F.interpolate(
        u4,
        size=(e4.size(2), e4.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Concatenate and decode stage 4 using multi-line call.
    u4 = torch.cat(
      [
        u4,
        e4,
      ],
      dim=1,
    )
    d4 = self.dec4.forward(u4)

    # Upsample and align stage 3.
    u3 = self.up3(d4)
    if (u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3)):
      u3 = F.interpolate(
        u3,
        size=(e3.size(2), e3.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    u3 = torch.cat(
      [
        u3,
        e3,
      ],
      dim=1,
    )
    d3 = self.dec3.forward(u3)

    # Upsample and align stage 2.
    u2 = self.up2(d3)
    if (u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3)):
      u2 = F.interpolate(
        u2,
        size=(e2.size(2), e2.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    u2 = torch.cat(
      [
        u2,
        e2,
      ],
      dim=1,
    )
    d2 = self.dec2.forward(u2)

    # Upsample and align stage 1.
    u1 = self.up1(d2)
    if (u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3)):
      u1 = F.interpolate(
        u1,
        size=(e1.size(2), e1.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    u1 = torch.cat(
      [
        u1,
        e1,
      ],
      dim=1,
    )
    d1 = self.dec1.forward(u1)
    # Return final logits.
    return self.finalConv(d1)


class SEUNet(nn.Module):
  r'''
  U-Net with Squeeze-and-Excitation modules applied to encoder and center.

  Short summary:
    Inserts SE blocks after encoder and center blocks to recalibrate channel-wise
    responses, improving representational expressiveness.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): If True use learned transpose upsampling.
    seReduction (int): Reduction ratio for SE bottleneck. Default 16.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize the SE-UNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True, seReduction=16):
    # Call the parent initializer.
    super(SEUNet, self).__init__()
    # Create the first encoder block as a DoubleConv.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    # Create the SE block for the first encoder level.
    self.se1 = SEBlock(baseChannels, reduction=seReduction)
    # Create pooling after the first encoder level.
    self.pool1 = nn.MaxPool2d(2)

    # Create the second encoder block as a DoubleConv.
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    # Create the SE block for the second encoder level.
    self.se2 = SEBlock(baseChannels * 2, reduction=seReduction)
    # Create pooling after the second encoder level.
    self.pool2 = nn.MaxPool2d(2)

    # Create the third encoder block as a DoubleConv.
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    # Create the SE block for the third encoder level.
    self.se3 = SEBlock(baseChannels * 4, reduction=seReduction)
    # Create pooling after the third encoder level.
    self.pool3 = nn.MaxPool2d(2)

    # Create the fourth encoder block as a DoubleConv.
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    # Create the SE block for the fourth encoder level.
    self.se4 = SEBlock(baseChannels * 8, reduction=seReduction)
    # Create pooling after the fourth encoder level.
    self.pool4 = nn.MaxPool2d(2)

    # Create the center DoubleConv block.
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)
    # Create the SE block for the center.
    self.seCenter = SEBlock(baseChannels * 16, reduction=seReduction)

    # Choose upsampling method based on the flag.
    if (useConvTranspose2d):
      # Use ConvTranspose2d for learned upsampling.
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      # Create the second learned upsampling layer.
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      # Create the third learned upsampling layer.
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      # Create the fourth learned upsampling layer.
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      # Use bilinear upsampling followed by 1x1 conv for stable interpolation.
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      # Create the second bilinear upsampling block.
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      # Create the third bilinear upsampling block.
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      # Create the fourth bilinear upsampling block.
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )

    # Create decoder DoubleConv blocks.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    # Create the third decoder DoubleConv block.
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    # Create the second decoder DoubleConv block.
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    # Create the first decoder DoubleConv block.
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)

    # Create the final 1x1 conv to map features to classes.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for SEUNet.
  def forward(self, x):
    # Encode level 1 and apply SE.
    e1 = self.enc1.forward(x)
    # Recalibrate channels at level 1.
    e1 = self.se1.forward(e1)
    # Pool to downsample after level 1.
    p1 = self.pool1(e1)

    # Encode level 2 and apply SE.
    e2 = self.enc2.forward(p1)
    # Recalibrate channels at level 2.
    e2 = self.se2.forward(e2)
    # Pool to downsample after level 2.
    p2 = self.pool2(e2)

    # Encode level 3 and apply SE.
    e3 = self.enc3.forward(p2)
    # Recalibrate channels at level 3.
    e3 = self.se3.forward(e3)
    # Pool to downsample after level 3.
    p3 = self.pool3(e3)

    # Encode level 4 and apply SE.
    e4 = self.enc4.forward(p3)
    # Recalibrate channels at level 4.
    e4 = self.se4.forward(e4)
    # Pool to downsample after level 4.
    p4 = self.pool4(e4)

    # Process the center and apply SE.
    c = self.center.forward(p4)
    # Recalibrate channels at the center.
    c = self.seCenter.forward(c)

    # Upsample from center to level 4 decoder input.
    u4 = self.up4(c)
    # Align spatial dimensions if required.
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(
        u4,
        size=(e4.size(2), e4.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Concatenate decoder and skip features and decode level 4 using multi-line concatenation.
    d4 = self.dec4.forward(
      torch.cat(
        [
          u4,
          e4,
        ],
        dim=1,
      )
    )

    # Upsample to level 3 decoder input.
    u3 = self.up3(d4)
    # Align spatial dimensions if required.
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(
        u3,
        size=(e3.size(2), e3.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Concatenate and decode level 3 using multi-line concatenation.
    d3 = self.dec3.forward(
      torch.cat(
        [
          u3,
          e3,
        ],
        dim=1,
      )
    )

    # Upsample to level 2 decoder input.
    u2 = self.up2(d3)
    # Align spatial dimensions if required.
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(
        u2,
        size=(e2.size(2), e2.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Concatenate and decode level 2 using multi-line concatenation.
    d2 = self.dec2.forward(
      torch.cat(
        [
          u2,
          e2,
        ],
        dim=1,
      )
    )

    # Upsample to level 1 decoder input.
    u1 = self.up1(d2)
    # Align spatial dimensions if required.
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(
        u1,
        size=(e1.size(2), e1.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Concatenate and decode level 1 using multi-line concatenation.
    d1 = self.dec1.forward(
      torch.cat(
        [
          u1,
          e1,
        ],
        dim=1,
      )
    )

    # Map features to logits and return.
    return self.finalConv(d1)


class ResidualAttentionUNet(nn.Module):
  r'''
  Residual U-Net combined with Attention Gates.

  Short summary:
    Uses residual blocks in the encoder/decoder with attention gating applied
    to skip connections to guide feature fusion.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize ResidualAttentionUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call the parent initializer.
    super(ResidualAttentionUNet, self).__init__()
    # Create encoder residual blocks for each level.
    self.enc1 = ResidualBlock(inputChannels, baseChannels)
    # Create pooling after encoder level 1.
    self.pool1 = nn.MaxPool2d(2)
    # Create encoder level 2 residual block.
    self.enc2 = ResidualBlock(baseChannels, baseChannels * 2)
    # Create pooling after encoder level 2.
    self.pool2 = nn.MaxPool2d(2)
    # Create encoder level 3 residual block.
    self.enc3 = ResidualBlock(baseChannels * 2, baseChannels * 4)
    # Create pooling after encoder level 3.
    self.pool3 = nn.MaxPool2d(2)
    # Create encoder level 4 residual block.
    self.enc4 = ResidualBlock(baseChannels * 4, baseChannels * 8)
    # Create pooling after encoder level 4.
    self.pool4 = nn.MaxPool2d(2)

    # Create the center residual block.
    self.center = ResidualBlock(baseChannels * 8, baseChannels * 16)

    # Choose upsampling method based on the flag.
    if (useConvTranspose2d):
      # Learned transposed upsampling for each level.
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      # Bilinear upsampling followed by 1x1 conv blocks for each level.
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )

    # Create attention gates for each skip connection.
    self.att4 = AttentionGate(F_g=baseChannels * 8, F_l=baseChannels * 8, F_int=baseChannels * 4)
    # Create attention gate for level 3.
    self.att3 = AttentionGate(F_g=baseChannels * 4, F_l=baseChannels * 4, F_int=baseChannels * 2)
    # Create attention gate for level 2.
    self.att2 = AttentionGate(F_g=baseChannels * 2, F_l=baseChannels * 2, F_int=baseChannels)
    # Create attention gate for level 1.
    self.att1 = AttentionGate(F_g=baseChannels, F_l=baseChannels, F_int=max(1, baseChannels // 2))

    # Create decoder residual blocks for each level.
    self.dec4 = ResidualBlock(baseChannels * 16, baseChannels * 8)
    # Create decoder level 3 residual block.
    self.dec3 = ResidualBlock(baseChannels * 8, baseChannels * 4)
    # Create decoder level 2 residual block.
    self.dec2 = ResidualBlock(baseChannels * 4, baseChannels * 2)
    # Create decoder level 1 residual block.
    self.dec1 = ResidualBlock(baseChannels * 2, baseChannels)

    # Create final 1x1 conv mapping to classes.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for ResidualAttentionUNet.
  def forward(self, x):
    # Encode level 1 using residual block and pool.
    e1 = self.enc1.forward(x)
    # Pool after level 1.
    p1 = self.pool1(e1)
    # Encode level 2 using residual block and pool.
    e2 = self.enc2.forward(p1)
    # Pool after level 2.
    p2 = self.pool2(e2)
    # Encode level 3 using residual block and pool.
    e3 = self.enc3.forward(p2)
    # Pool after level 3.
    p3 = self.pool3(e3)
    # Encode level 4 using residual block and pool.
    e4 = self.enc4.forward(p3)
    # Pool after level 4.
    p4 = self.pool4(e4)

    # Process the center residual block.
    c = self.center.forward(p4)

    # Upsample to decoder level 4 input.
    u4 = self.up4(c)
    # Align spatial dimensions if required.
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(
        u4,
        size=(e4.size(2), e4.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Apply attention gate on encoder features and decode level 4.
    e4_att = self.att4.forward(u4, e4)
    # Concatenate and pass through decoder block.
    d4 = self.dec4.forward(
      torch.cat(
        [
          u4,
          e4_att,
        ],
        dim=1,
      )
    )

    # Upsample to decoder level 3 input.
    u3 = self.up3(d4)
    # Align spatial dimensions if required.
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(
        u3,
        size=(e3.size(2), e3.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Apply attention gate on encoder features and decode level 3.
    e3_att = self.att3.forward(u3, e3)
    # Concatenate and pass through decoder block.
    d3 = self.dec3.forward(
      torch.cat(
        [
          u3,
          e3_att,
        ],
        dim=1,
      )
    )

    # Upsample to decoder level 2 input.
    u2 = self.up2(d3)
    # Align spatial dimensions if required.
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(
        u2,
        size=(e2.size(2), e2.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Apply attention gate on encoder features and decode level 2.
    e2_att = self.att2.forward(u2, e2)
    # Concatenate and pass through decoder block.
    d2 = self.dec2.forward(
      torch.cat(
        [
          u2,
          e2_att,
        ],
        dim=1,
      )
    )

    # Upsample to decoder level 1 input.
    u1 = self.up1(d2)
    # Align spatial dimensions if required.
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(
        u1,
        size=(e1.size(2), e1.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    # Apply attention gate on encoder features and decode level 1.
    e1_att = self.att1.forward(u1, e1)
    # Concatenate and pass through decoder block.
    d1 = self.dec1.forward(
      torch.cat(
        [
          u1,
          e1_att,
        ],
        dim=1,
      )
    )

    # Map features to logits and return.
    return self.finalConv(d1)


class MultiResUNet(nn.Module):
  r'''
  MultiResUNet with MultiRes blocks per stage.

  Short summary:
    Employs multi-resolution convolution blocks that capture features at
    multiple receptive sizes within each stage for richer multi-scale context.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize MultiResUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call super initializer.
    super(MultiResUNet, self).__init__()
    # Encoder MultiRes blocks.
    self.enc1 = MultiResBlock(inputChannels, baseChannels)
    self.pool1 = nn.MaxPool2d(2)
    self.enc2 = MultiResBlock(baseChannels, baseChannels * 2)
    self.pool2 = nn.MaxPool2d(2)
    self.enc3 = MultiResBlock(baseChannels * 2, baseChannels * 4)
    self.pool3 = nn.MaxPool2d(2)
    self.enc4 = MultiResBlock(baseChannels * 4, baseChannels * 8)
    self.pool4 = nn.MaxPool2d(2)
    # Center block.
    self.center = MultiResBlock(baseChannels * 8, baseChannels * 16)

    # Upsampling strategy.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )

    # Decoder MultiRes blocks and final conv.
    self.dec4 = MultiResBlock(baseChannels * 16, baseChannels * 8)
    self.dec3 = MultiResBlock(baseChannels * 8, baseChannels * 4)
    self.dec2 = MultiResBlock(baseChannels * 4, baseChannels * 2)
    self.dec1 = MultiResBlock(baseChannels * 2, baseChannels)
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass.
  def forward(self, x):
    # Encoder path.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3.forward(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4.forward(p3)
    p4 = self.pool4(e4)
    # Center.
    c = self.center.forward(p4)
    # Decoder stage 4.
    u4 = self.up4(c)
    # Align shapes if required.
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(
        u4,
        size=(e4.size(2), e4.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    d4 = self.dec4.forward(
      torch.cat(
        [
          u4,
          e4,
        ],
        dim=1,
      )
    )
    # Decoder stage 3.
    u3 = self.up3(d4)
    # Align shapes if required.
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(
        u3,
        size=(e3.size(2), e3.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    d3 = self.dec3.forward(torch.cat([u3, e3], dim=1))
    # Decoder stage 2.
    u2 = self.up2(d3)
    # Align shapes if required.
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(
        u2,
        size=(e2.size(2), e2.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    d2 = self.dec2.forward(torch.cat([u2, e2], dim=1))
    # Decoder stage 1.
    u1 = self.up1(d2)
    # Align shapes if required.
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(
        u1,
        size=(e1.size(2), e1.size(3)),
        mode="bilinear",
        align_corners=True,
      )
    d1 = self.dec1.forward(torch.cat([u1, e1], dim=1))
    # Final logits.
    return self.finalConv(d1)


class DenseUNet(nn.Module):
  r'''
  DenseUNet integrating compact DenseBlocks and transition convolutions.

  Short summary:
    Uses DenseBlocks with 1x1 transition convolutions to manage channel growth,
    combining strengths of DenseNet connectivity with U-Net decoding.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base channel width. Default 32.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize DenseUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=32, useConvTranspose2d=True):
    # Call super initializer.
    super(DenseUNet, self).__init__()

    growth = baseChannels // 2

    # Encoder dense blocks and transition convs.
    self.enc1 = DenseBlock(inputChannels, numLayers=3, growthRate=growth)
    out1 = inputChannels + 3 * growth
    self.trans1 = nn.Conv2d(out1, baseChannels, kernel_size=1)
    self.pool1 = nn.MaxPool2d(2)

    self.enc2 = DenseBlock(baseChannels, numLayers=3, growthRate=growth)
    out2 = baseChannels + 3 * growth
    self.trans2 = nn.Conv2d(out2, baseChannels * 2, kernel_size=1)
    self.pool2 = nn.MaxPool2d(2)

    self.enc3 = DenseBlock(baseChannels * 2, numLayers=3, growthRate=growth)
    out3 = baseChannels * 2 + 3 * growth
    self.trans3 = nn.Conv2d(out3, baseChannels * 4, kernel_size=1)
    self.pool3 = nn.MaxPool2d(2)

    self.enc4 = DenseBlock(baseChannels * 4, numLayers=3, growthRate=growth)
    out4 = baseChannels * 4 + 3 * growth
    self.trans4 = nn.Conv2d(out4, baseChannels * 8, kernel_size=1)
    self.pool4 = nn.MaxPool2d(2)

    # Center dense block.
    CenterIn = baseChannels * 8
    self.center = DenseBlock(CenterIn, numLayers=3, growthRate=growth)
    centerOut = CenterIn + 3 * growth

    # Upsampling modules.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(centerOut, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(centerOut, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )

    # Decoder convolution blocks (to process concatenated features).
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)  # up4 + t4.
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)  # up3 + t3.
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)  # up2 + t2.
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)  # up1 + t1.

    # Final conv.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass.
  def forward(self, x):
    # Encoder 1.
    e1 = self.enc1(x)
    t1 = self.trans1(e1)
    p1 = self.pool1(t1)
    # Encoder 2.
    e2 = self.enc2(p1)
    t2 = self.trans2(e2)
    p2 = self.pool2(t2)
    # Encoder 3.
    e3 = self.enc3(p2)
    t3 = self.trans3(e3)
    p3 = self.pool3(t3)
    # Encoder 4.
    e4 = self.enc4(p3)
    t4 = self.trans4(e4)
    p4 = self.pool4(t4)
    # Center.
    c = self.center(p4)

    # Decoder stage 4.
    u4 = self.up4(c)
    # Optional: align size (in case of odd dimensions).
    if (u4.shape[2:] != t4.shape[2:]):
      u4 = F.interpolate(u4, size=t4.shape[2:], mode="bilinear", align_corners=True)
    d4 = self.dec4(torch.cat([u4, t4], dim=1))

    # Decoder stage 3.
    u3 = self.up3(d4)
    if (u3.shape[2:] != t3.shape[2:]):
      u3 = F.interpolate(u3, size=t3.shape[2:], mode="bilinear", align_corners=True)
    d3 = self.dec3(torch.cat([u3, t3], dim=1))

    # Decoder stage 2.
    u2 = self.up2(d3)
    if (u2.shape[2:] != t2.shape[2:]):
      u2 = F.interpolate(u2, size=t2.shape[2:], mode="bilinear", align_corners=True)
    d2 = self.dec2(torch.cat([u2, t2], dim=1))

    # Decoder stage 1.
    u1 = self.up1(d2)
    if (u1.shape[2:] != t1.shape[2:]):
      u1 = F.interpolate(u1, size=t1.shape[2:], mode="bilinear", align_corners=True)
    d1 = self.dec1(torch.cat([u1, t1], dim=1))

    # Final logits.
    return self.finalConv(d1)


class R2UNet(nn.Module):
  r'''
  Recurrent Residual U-Net using recurrent convolution layers.

  Short summary:
    Integrates RecurrentConvLayer modules inside encoder stages to capture
    iterative spatial context while preserving residual optimization benefits.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of segmentation classes. Default 2.
    baseChannels (int): Base channel count. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.
    t (int): Number of recurrent iterations in RCL. Default 2.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize R2UNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True, t=2):
    # Call super initializer.
    super(R2UNet, self).__init__()
    # Initial projection convs and RCLs.
    self.in1 = nn.Conv2d(inputChannels, baseChannels, kernel_size=3, padding=1)
    self.enc1 = RecurrentConvLayer(baseChannels, t=t)
    self.pool1 = nn.MaxPool2d(2)
    self.in2 = nn.Conv2d(baseChannels, baseChannels * 2, kernel_size=3, padding=1)
    self.enc2 = RecurrentConvLayer(baseChannels * 2, t=t)
    self.pool2 = nn.MaxPool2d(2)
    self.in3 = nn.Conv2d(baseChannels * 2, baseChannels * 4, kernel_size=3, padding=1)
    self.enc3 = RecurrentConvLayer(baseChannels * 4, t=t)
    self.pool3 = nn.MaxPool2d(2)
    self.in4 = nn.Conv2d(baseChannels * 4, baseChannels * 8, kernel_size=3, padding=1)
    self.enc4 = RecurrentConvLayer(baseChannels * 8, t=t)
    self.pool4 = nn.MaxPool2d(2)
    # Center with projection to match expected channels.
    self.centerProj = nn.Conv2d(baseChannels * 8, baseChannels * 16, kernel_size=1)
    self.center = RecurrentConvLayer(baseChannels * 16, t=t)

    # Upsampling.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )

    # Simple decoder convs and final conv.
    self.dec4 = nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=3, padding=1)
    self.dec3 = nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=3, padding=1)
    self.dec2 = nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=3, padding=1)
    self.dec1 = nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=3, padding=1)
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for R2UNet.
  def forward(self, x):
    # Project and recurrent encode.
    i1 = self.in1.forward(x)
    e1 = self.enc1.forward(i1)
    p1 = self.pool1(e1)
    i2 = self.in2.forward(p1)
    e2 = self.enc2.forward(i2)
    p2 = self.pool2(e2)
    i3 = self.in3.forward(p2)
    e3 = self.enc3.forward(i3)
    p3 = self.pool3(e3)
    i4 = self.in4.forward(p3)
    e4 = self.enc4.forward(i4)
    p4 = self.pool4(e4)
    # Center projection and recurrent processing.
    cProj = self.centerProj(p4)
    c = self.center.forward(cProj)
    # Decoder.
    u4 = self.up4(c)
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(u4, size=(e4.size(2), e4.size(3)), mode="bilinear", align_corners=True)
    d4 = F.relu(self.dec4.forward(torch.cat([u4, e4], dim=1)))
    u3 = self.up3(d4)
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(u3, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
    d3 = F.relu(self.dec3.forward(torch.cat([u3, e3], dim=1)))
    u2 = self.up2(d3)
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(u2, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
    d2 = F.relu(self.dec2.forward(torch.cat([u2, e2], dim=1)))
    u1 = self.up1(d2)
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(u1, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
    d1 = F.relu(self.dec1.forward(torch.cat([u1, e1], dim=1)))
    # Final logits.
    return self.finalConv(d1)


class ASPPUNet(nn.Module):
  r'''
  U-Net with ASPP at the bottleneck for expanded receptive field.

  Short summary:
    Places an Atrous Spatial Pyramid Pooling module at the bottleneck to
    aggregate multi-scale context using parallel dilated convolutions.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize ASPPUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call super initializer.
    super(ASPPUNet, self).__init__()
    # Encoder path.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = nn.MaxPool2d(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = nn.MaxPool2d(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = nn.MaxPool2d(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = nn.MaxPool2d(2)
    # ASPP at center.
    self.centerAspp = ASPP(baseChannels * 8, baseChannels * 16)
    # Upsampling modules.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )
    # Decoder convs and final conv.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for ASPPUNet.
  def forward(self, x):
    # Encoder.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3.forward(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4.forward(p3)
    p4 = self.pool4(e4)
    # ASPP center.
    c = self.centerAspp.forward(p4)
    # Decoder up and concat.
    u4 = self.up4(c)
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(u4, size=(e4.size(2), e4.size(3)), mode="bilinear", align_corners=True)
    d4 = self.dec4.forward(torch.cat([u4, e4], dim=1))
    u3 = self.up3(d4)
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(u3, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
    d3 = self.dec3.forward(torch.cat([u3, e3], dim=1))
    u2 = self.up2(d3)
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(u2, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
    d2 = self.dec2.forward(torch.cat([u2, e2], dim=1))
    u1 = self.up1(d2)
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(u1, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
    d1 = self.dec1.forward(torch.cat([u1, e1], dim=1))
    # Final logits.
    return self.finalConv(d1)


class TransUNet(nn.Module):
  r'''
  Transformer-enhanced U-Net with a small transformer stack at bottleneck.

  Short summary:
    Converts bottleneck conv maps into token sequences using PatchEmbedding,
    processes them via a transformer encoder stack, and projects back to
    convolutional feature maps for decoding.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    embedDim (int): Transformer embedding dimension. Default 256.
    numHeads (int): Number of attention heads. Default 8.
    numEncoders (int): Number of transformer encoder layers. Default 2.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize TransUNet.
  def __init__(
    self, inputChannels=3, numClasses=2, baseChannels=64, embedDim=256, numHeads=8, numEncoders=2,
    useConvTranspose2d=True
  ):
    # Call super initializer.
    super(TransUNet, self).__init__()
    # Encoder path.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = nn.MaxPool2d(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = nn.MaxPool2d(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = nn.MaxPool2d(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = nn.MaxPool2d(2)
    # Patch embedding and transformer stack.
    self.patchEmbed = PatchEmbedding(baseChannels * 8, embedDim=embedDim, patchSize=2)
    self.transformer = nn.Sequential(
      *[TransformerBlock(embedDim=embedDim, numHeads=numHeads) for _ in range(numEncoders)])
    # Project transformer output to conv channels.
    self.transformProj = nn.Conv2d(embedDim, baseChannels * 16, kernel_size=1)
    # Upsamplers and decoder convs.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for TransUNet.
  def forward(self, x):
    # Encoder.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3.forward(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4.forward(p3)
    p4 = self.pool4(e4)
    # Patch embed and transformer.
    patches, (ph, pw) = self.patchEmbed.forward(p4)
    t = self.transformer.forward(patches)
    b, n, d = t.size()
    t = t.permute(0, 2, 1).contiguous().view(b, d, ph, pw)
    c = self.transformProj.forward(t)
    # Decoder.
    u4 = self.up4(c)
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(u4, size=(e4.size(2), e4.size(3)), mode="bilinear", align_corners=True)
    d4 = self.dec4.forward(torch.cat([u4, e4], dim=1))
    u3 = self.up3(d4)
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(u3, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
    d3 = self.dec3.forward(torch.cat([u3, e3], dim=1))
    u2 = self.up2(d3)
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(u2, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
    d2 = self.dec2.forward(torch.cat([u2, e2], dim=1))
    u1 = self.up1(d2)
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(u1, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
    d1 = self.dec1.forward(torch.cat([u1, e1], dim=1))
    return self.finalConv(d1)


class CBAMUNet(nn.Module):
  r'''
  U-Net variant applying CBAM attention to encoder skip features.

  Short summary:
    Applies CBAM (channel + spatial attention) to encoder skip features
    before merging them into decoder stages to improve salient feature focus.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize CBAMUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call parent initializer.
    super(CBAMUNet, self).__init__()
    # Encoder.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = nn.MaxPool2d(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = nn.MaxPool2d(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = nn.MaxPool2d(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = nn.MaxPool2d(2)
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)
    # Upsamplers.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
      )
    # Decoder convs.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    # CBAM modules for each encoder skip.
    self.cbam4 = CBAM(baseChannels * 8)
    self.cbam3 = CBAM(baseChannels * 4)
    self.cbam2 = CBAM(baseChannels * 2)
    self.cbam1 = CBAM(baseChannels)
    # Final conv.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for CBAMUNet.
  def forward(self, x):
    # Encode.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    e3 = self.enc3.forward(p2)
    p3 = self.pool3(e3)
    e4 = self.enc4.forward(p3)
    p4 = self.pool4(e4)
    # Center.
    c = self.center.forward(p4)

    # Decoder stage 4 with CBAM on skip.
    u4 = self.up4(c)
    if ((u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3))):
      u4 = F.interpolate(u4, size=(e4.size(2), e4.size(3)), mode="bilinear", align_corners=True)
    e4_att = self.cbam4.forward(e4)
    # Decode stage 4 using multi-line concatenation.
    d4 = self.dec4.forward(
      torch.cat(
        [
          u4,
          e4_att,
        ],
        dim=1,
      )
    )

    # Decoder stage 3.
    u3 = self.up3(d4)
    if ((u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3))):
      u3 = F.interpolate(u3, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
    e3_att = self.cbam3.forward(e3)
    # Decode stage 3 using multi-line concatenation.
    d3 = self.dec3.forward(
      torch.cat(
        [
          u3,
          e3_att,
        ],
        dim=1,
      )
    )

    # Decoder stage 2.
    u2 = self.up2(d3)
    if ((u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3))):
      u2 = F.interpolate(u2, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
    e2_att = self.cbam2.forward(e2)
    # Decode stage 2 using multi-line concatenation.
    d2 = self.dec2.forward(
      torch.cat(
        [
          u2,
          e2_att,
        ],
        dim=1,
      )
    )

    # Decoder stage 1.
    u1 = self.up1(d2)
    if ((u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3))):
      u1 = F.interpolate(u1, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
    e1_att = self.cbam1.forward(e1)
    # Decode stage 1 using multi-line concatenation.
    d1 = self.dec1.forward(
      torch.cat(
        [
          u1,
          e1_att,
        ],
        dim=1,
      )
    )

    # Final logits.
    return self.finalConv(d1)


class EfficientUNet(nn.Module):
  r'''
  Efficient wrapper around MobileUNet.

  Short summary:
    Thin wrapper that exposes the same API but uses MobileUNet internals to
    provide a lightweight segmentation model.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of classes. Default 2.
    baseChannels (int): Base channels for underlying MobileUNet. Default 32.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize EfficientUNet.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=32, useConvTranspose2d=True):
    # Call parent initializer.
    super(EfficientUNet, self).__init__()
    # Reuse MobileUNet implementation.
    self.model = MobileUNet(inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
                            useConvTranspose2d=useConvTranspose2d)

  # Forward pass for EfficientUNet.
  def forward(self, x):
    # Forward through underlying MobileUNet.
    return self.model.forward(x)


class BoundaryAwareUNet(nn.Module):
  r'''
  Boundary-aware U-Net with explicit boundary detection branch.

  Short summary:
    Integrates parallel boundary detection pathway with auxiliary loss supervision
    using Sobel filters to enhance segmentation boundary precision and reduce
    boundary blurring common in standard encoder-decoder architectures.

  Parameters:
    inputChannels (int): Number of input image channels. Default 3.
    numClasses (int): Number of segmentation classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    useConvTranspose2d (bool): Use ConvTranspose2d for upsampling when True.
    boundaryWeight (float): Weighting factor for boundary supervision loss. Default 0.5.

  Attributes:
    encoder (nn.ModuleList): Standard U-Net encoder blocks.
    pools (nn.ModuleList): Max pooling layers.
    center (DoubleConv): Bottleneck block.
    upsamples (nn.ModuleList): Upsampling modules.
    decoders (nn.ModuleList): Decoder convolutional blocks.
    boundaryEncoder (nn.ModuleList): Parallel encoder for boundary features.
    boundaryCenter (DoubleConv): Boundary bottleneck block.
    boundaryUpsamples (nn.ModuleList): Boundary upsampling modules.
    boundaryDecoders (nn.ModuleList): Boundary decoder blocks.
    boundaryHead (nn.Conv2d): Boundary prediction head (binary edge map).
    segmentationHead (nn.Conv2d): Final segmentation logits head.

  Returns:
    Tuple[torch.Tensor, torch.Tensor]: Segmentation logits and boundary map.
  '''

  # Initialize boundary-aware U-Net.
  def __init__(
    self,
    inputChannels=3,
    numClasses=2,
    baseChannels=64,
    useConvTranspose2d=True,
    boundaryWeight=0.5
  ):
    # Call parent initializer.
    super(BoundaryAwareUNet, self).__init__()
    # Store boundary supervision weight.
    self.boundaryWeight = boundaryWeight

    # Initialize standard encoder blocks.
    self.encoder = nn.ModuleList()
    # Initialize pooling layers.
    self.pools = nn.ModuleList()
    # Initialize boundary encoder blocks.
    self.boundaryEncoder = nn.ModuleList()

    # Set initial channel counters for both pathways.
    inChansMain = inputChannels
    inChansBoundary = inputChannels
    boundaryBase = max(16, baseChannels // 4)
    boundaryEncChans = []

    # Build four-level encoder hierarchy.
    for i in range(4):
      # Create standard encoder block.
      outChansMain = baseChannels * (2 ** i)
      self.encoder.append(DoubleConv(inChansMain, outChansMain))
      inChansMain = outChansMain

      # Create boundary encoder block with reduced capacity.
      outChansBoundary = boundaryBase * (2 ** i)
      self.boundaryEncoder.append(DoubleConv(inChansBoundary, outChansBoundary))
      boundaryEncChans.append(outChansBoundary)
      inChansBoundary = outChansBoundary

      # Create shared pooling layer.
      self.pools.append(nn.MaxPool2d(2))

    # Create standard bottleneck block.
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)
    # Create boundary bottleneck block.
    self.boundaryCenter = DoubleConv(boundaryEncChans[-1], boundaryBase * 8)

    # Initialize upsampling and decoder modules.
    self.upsamples = nn.ModuleList()
    self.decoders = nn.ModuleList()
    self.boundaryUpsamples = nn.ModuleList()
    self.boundaryDecoders = nn.ModuleList()

    # Build symmetric decoder hierarchy for main segmentation path.
    for i in range(4):
      inChansDec = baseChannels * (2 ** (4 - i))
      outChansDec = baseChannels * (2 ** (3 - i))
      if useConvTranspose2d:
        upModule = nn.ConvTranspose2d(
          inChansDec,
          outChansDec,
          kernel_size=2,
          stride=2
        )
      else:
        upModule = nn.Sequential(
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
          nn.Conv2d(inChansDec, outChansDec, kernel_size=1)
        )
      self.upsamples.append(upModule)
      self.decoders.append(DoubleConv(inChansDec, outChansDec))

    # Build symmetric decoder hierarchy for boundary path.
    currentBoundaryIn = boundaryBase * 8
    for i in range(4):
      outChansBoundary = boundaryEncChans[3 - i]
      if useConvTranspose2d:
        boundaryUp = nn.ConvTranspose2d(
          currentBoundaryIn,
          outChansBoundary,
          kernel_size=2,
          stride=2
        )
      else:
        boundaryUp = nn.Sequential(
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
          nn.Conv2d(currentBoundaryIn, outChansBoundary, kernel_size=1)
        )
      self.boundaryUpsamples.append(boundaryUp)
      self.boundaryDecoders.append(DoubleConv(2 * outChansBoundary, outChansBoundary))
      currentBoundaryIn = outChansBoundary

    # Create heads.
    self.boundaryHead = nn.Conv2d(boundaryEncChans[0], 1, kernel_size=1)
    self.segmentationHead = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass through boundary-aware U-Net.
  def forward(self, x):
    # Extract original spatial dimensions.
    hOrig = x.shape[2]
    # Extract original width dimension.
    wOrig = x.shape[3]
    # Initialize lists for skip connections.
    encoderFeatures = []
    # Initialize boundary skip connection list.
    boundaryFeatures = []
    # Process through standard encoder path.
    xEnc = x
    # Process through boundary encoder path.
    xBoundary = x
    # Iterate through encoder stages.
    for i in range(4):
      # Apply standard encoder block.
      xEnc = self.encoder[i](xEnc)
      # Store standard encoder feature.
      encoderFeatures.append(xEnc)
      # Apply boundary encoder block.
      xBoundary = self.boundaryEncoder[i](xBoundary)
      # Store boundary encoder feature.
      boundaryFeatures.append(xBoundary)
      # Apply pooling to both pathways.
      xEnc = self.pools[i](xEnc)
      # Apply pooling to boundary pathway.
      xBoundary = self.pools[i](xBoundary)
    # Process standard bottleneck.
    xEnc = self.center(xEnc)
    # Process boundary bottleneck.
    xBoundary = self.boundaryCenter(xBoundary)
    # Process through decoder stages.
    for i in range(4):
      # Upsample standard features.
      xEnc = self.upsamples[i](xEnc)
      # Upsample boundary features.
      xBoundary = self.boundaryUpsamples[i](xBoundary)
      # Align spatial dimensions when necessary.
      if (xEnc.size(2) != encoderFeatures[-(i + 1)].size(2)) or (
        xEnc.size(3) != encoderFeatures[-(i + 1)].size(3)
      ):
        # Interpolate standard features to match skip connection.
        xEnc = F.interpolate(
          xEnc,
          size=(
            encoderFeatures[-(i + 1)].size(2),
            encoderFeatures[-(i + 1)].size(3)
          ),
          mode="bilinear",
          align_corners=True
        )
      # Align boundary features similarly.
      if (xBoundary.size(2) != boundaryFeatures[-(i + 1)].size(2)) or (
        xBoundary.size(3) != boundaryFeatures[-(i + 1)].size(3)
      ):
        # Interpolate boundary features to match skip connection.
        xBoundary = F.interpolate(
          xBoundary,
          size=(
            boundaryFeatures[-(i + 1)].size(2),
            boundaryFeatures[-(i + 1)].size(3)
          ),
          mode="bilinear",
          align_corners=True
        )
      # Concatenate standard skip connection.
      xEnc = torch.cat(
        [
          xEnc,
          encoderFeatures[-(i + 1)]
        ],
        dim=1
      )
      # Apply standard decoder block.
      xEnc = self.decoders[i](xEnc)
      # Concatenate boundary skip connection.
      xBoundary = torch.cat(
        [
          xBoundary,
          boundaryFeatures[-(i + 1)]
        ],
        dim=1
      )
      # Apply boundary decoder block.
      xBoundary = self.boundaryDecoders[i](xBoundary)
    # Generate boundary prediction map.
    boundaryMap = self.boundaryHead(xBoundary)
    # Upsample boundary map to original resolution.
    boundaryMap = F.interpolate(
      boundaryMap,
      size=(hOrig, wOrig),
      mode="bilinear",
      align_corners=True
    )
    # Generate segmentation logits.
    segmentationLogits = self.segmentationHead(xEnc)
    # Upsample segmentation to original resolution.
    segmentationLogits = F.interpolate(
      segmentationLogits,
      size=(hOrig, wOrig),
      mode="bilinear",
      align_corners=True
    )
    # Return both segmentation logits and boundary map.
    return (
      segmentationLogits,
      boundaryMap
    )


# Extended factory override to include the newly appended UNet variants.
def CreateUNet(
  inputChannels: int = 3,
  numClasses: int = 2,
  baseChannels: int = 64,
  depth: int = 4,
  upMode: str = "transpose",
  norm: str = "batch",
  dropout: float = 0.0,
  residual: bool = False,
  modelType: str = "dynamic",
):
  r'''
  Extended factory that supports a large set of UNet variants by modelType.

  Short summary:
    Returns an instantiated UNet variant selected by `modelType`. The factory
    exposes common constructor arguments used across variants for convenience.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base channel count for architectures. Default 64.
    depth (int): Depth parameter for dynamic variants. Default 4.
    upMode (str): "transpose" or "bilinear" upsampling. Default "transpose".
    norm (str): Normalization mode for configurable blocks. Default "batch".
    dropout (float): Dropout probability for configurable blocks. Default 0.0.
    residual (bool): Whether to use residual blocks where supported. Default False.
    modelType (str): Case-insensitive model selection string, e.g. "dynamic".

  Returns:
    nn.Module: Instantiated UNet variant ready for training or inference.
  '''

  # Keep the incoming modelType flexible and prepare lower-case matcher.
  mt = (modelType or "dynamic")
  mtLower = mt.lower()

  if (mtLower not in tuple(AVAILABLE_UNETS)):
    print(
      f"Warning: Unrecognized modelType '{modelType}'. Defaulting to 'dynamic' UNet.\n"
      f"Available types: {AVAILABLE_UNETS}"
    )

  if (mtLower in ("original", "legacy")):
    return UNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "multiresunet"):
    return MultiResUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "denseunet"):
    return DenseUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=max(16, baseChannels // 2),
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "r2unet"):
    return R2UNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "asppunet"):
    return ASPPUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "transunet"):
    return TransUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "cbamunet"):
    return CBAMUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "efficientunet"):
    return EfficientUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=max(16, baseChannels // 2),
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "dynamic"):
    return DynamicUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      depth=depth,
      upMode=upMode,
      norm=norm,
      dropout=dropout,
      residual=residual,
    )

  if (mtLower == "residual"):
    return ResidualUNet(
      inputChannels,
      numClasses,
      baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "attention"):
    return AttentionUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "mobile"):
    return MobileUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels if (baseChannels >= 32) else 32,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "se"):
    return SEUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "residual_attention"):
    return ResidualAttentionUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  if (mtLower == "boundary_aware"):
    return BoundaryAwareUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  # Default to DynamicUNet as a safe fallback.
  return DynamicUNet(
    inputChannels=inputChannels,
    numClasses=numClasses,
    baseChannels=baseChannels,
    depth=depth,
    upMode=upMode,
    norm=norm,
    dropout=dropout,
    residual=residual,
  )


if __name__ == "__main__":
  # Example to test all UNet variants.
  for unetType in AVAILABLE_UNETS:
    for imgSize in [128]:  # , 224, 256, 512
      print(f"Testing UNet variant: {unetType} with input size {imgSize}x{imgSize}")
      # Create model instance.
      model = CreateUNet(
        inputChannels=3,
        numClasses=5,
        baseChannels=64,
        depth=4,
        upMode="transpose",
        norm="batch",
        dropout=0.25,
        residual=True,
        modelType=unetType
      )
      model.train()
      X = torch.randn(1, 3, imgSize, imgSize)
      with torch.no_grad():
        Y = model(X)
      if (isinstance(Y, tuple)):
        print(f"Output shapes: {[y.shape for y in Y]}")
      else:
        print(f"Output shape: {Y.shape}")

      # Create dummy input tensor.
      inputTensor = torch.randn(1, 3, imgSize, imgSize)
      model.eval()
      # Forward pass.
      output = model(inputTensor)
      # Print output shape.
      if (isinstance(output, tuple)):
        print(f"Output shapes: {[o.shape for o in output]}")
      else:
        print(f"Output shape: {output.shape}")
      print("-" * 50)
