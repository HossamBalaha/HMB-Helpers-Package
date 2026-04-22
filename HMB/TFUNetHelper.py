import numpy as np
from typing import Tuple, List
import tensorflow as tf

AVAILABLE_UNETS = [
  "Original", "Legacy",  # Original UNet architecture with standard convolutional blocks and skip connections.
  "Dynamic",  # Dynamic convolutional blocks that adapt their weights based on the input features.
  "MultiResUNet",  # MultiResUNet architecture with multi-resolution blocks for improved feature extraction.
  "R2UNet",  # Recurrent residual convolutional blocks for improved feature representation.
  "TransUNet",  # Transformer-based encoder with a UNet-style decoder for capturing long-range dependencies.
  "CBAMUNet",  # CBAM integrated into the UNet architecture for enhanced feature representation.
  "EfficientUNet",  # EfficientNet-based encoder with a lightweight decoder for efficient segmentation.
  "Residual",  # Residual connections within the encoder and decoder blocks for improved gradient flow.
  "Attention",  # Attention gates for skip connections to focus on relevant features.
  "Mobile",  # Depthwise separable convolutions for lightweight segmentation.
  "SE",  # Squeeze-and-Excitation blocks for channel-wise feature recalibration.
  "ResidualAttention",  # Combines residual connections with attention gates for enhanced feature learning.
  "BoundaryAware",  # Two parallel branches for segmentation and boundary detection.
  "ASPPUNet",  # Incorporates Atrous Spatial Pyramid Pooling (ASPP) in the bottleneck to capture multi-scale context.
  "DenseUNet",  # Dense connectivity pattern where each layer receives inputs from all previous layers.
  # V-Net style residual encoder-decoder commonly used for volumetric/medical segmentation (adapted here to 2D),
  "VNet",
  "SegNet",  # SegNet architecture with encoder-decoder and max-pooling indices for upsampling.
]


# ---------------------------------------------------- #
# Basic utilities for handling model outputs           #
# ---------------------------------------------------- #


def PreparePredTensorToNumpy(predTensor: tf.Tensor, doScale2Image: bool = False) -> np.ndarray:
  r'''
  Utility to convert model output tensor after the sigmoid/softmax activation to a numpy array of class indices.
  It can be used also with the original mask tensor if it is already in the correct format,
  as it handles squeezing and type conversion.

  Short summary:
    Takes the raw output tensor from a TensorFlow model (after activation) and processes it to produce
    a 2D or 3D numpy array of class indices or binary masks. This involves squeezing unnecessary dimensions,
    converting boolean masks to integers if needed, and ensuring the final output is in the
    correct format for evaluation or visualization.

  Parameters:
    predTensor (tensorflow.Tensor): The raw output tensor from the model after activation, expected to be of
      shape [B, H, W, C] or [B, H, W, 1].
    doScale2Image (bool): If True, applies a threshold to convert probabilities to binary mask and scales to
      0..255. Default False.

  Returns:
    numpy.ndarray: Numpy array of shape [B, H, W] or [B, H, W, C] containing class indices or the scaled image.
  '''

  # Ensure tensor is converted to a numpy array.
  if (isinstance(predTensor, tf.Tensor)):
    predNp = predTensor.numpy()
  else:
    predNp = np.array(predTensor)

  # Handle common channel singleton shapes and squeeze them.
  if (predNp.ndim == 4 and predNp.shape[-1] == 1):
    # Squeeze the trailing singleton channel dimension.
    predNp = np.squeeze(predNp, axis=-1)
  elif (predNp.ndim == 4 and predNp.shape[1] == 1):
    # Squeeze the second dimension when channels are leading.
    predNp = np.squeeze(predNp, axis=1)

  # Convert boolean arrays to uint8, otherwise to integer labels.
  if (predNp.dtype == np.bool_):
    predMask = predNp.astype(np.uint8)
  else:
    predMask = predNp.astype(np.int64)

  # Optionally scale probability maps to 0..255 images using a threshold.
  if (doScale2Image):
    predMask = (predMask >= 0.5).astype(np.uint8)
    predMask *= 255
    predMask = predMask.astype(np.uint8)

  # Return the prepared numpy mask.
  return predMask


# ---------------------------------------------------- #
# Basic building blocks for UNet architectures.        #
# Implemented using tf.keras.layers and tf.keras.Model  #
# ---------------------------------------------------- #


class DoubleConv(tf.keras.layers.Layer):
  r'''
  Double convolution block used throughout the U-Net encoder and decoder.

  Short summary:
    Two consecutive 3x3 Conv2D layers each followed by normalization
    and ReLU activation used as a basic encoder/decoder building block.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Returns:
    tensorflow.Tensor: Output feature map of shape [B, H, W, outChannels].
  '''

  # Initialize the double convolution block.
  def __init__(self, inChannels: int, outChannels: int):
    super(DoubleConv, self).__init__()
    # Create the first convolution, normalization and activation.
    self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=3, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.act1 = tf.keras.layers.ReLU()
    # Create the second convolution, normalization and activation.
    self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=3, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.act2 = tf.keras.layers.ReLU()

  # Forward pass for the double conv block.
  def call(self, x, training=False):
    # Apply first conv branch.
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.act1(x)
    # Apply second conv branch.
    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.act2(x)
    # Return processed tensor.
    return x


class ConfigConv(tf.keras.layers.Layer):
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

  Returns:
    tensorflow.Tensor: Processed feature map with same spatial size as input.
  '''

  # Initialize configurable conv block.
  def __init__(self, inChannels: int, outChannels: int, norm: str = "batch", dropout: float = 0.0,
               residual: bool = False):
    super(ConfigConv, self).__init__()
    # Store residual and dropout configuration.
    self.residual = (residual and (inChannels == outChannels))
    self.dropoutRate = (dropout if (dropout and dropout > 0.0) else 0.0)

    # Helper to create the requested normalization layer.
    def make_norm():
      if (norm == "batch"):
        return tf.keras.layers.BatchNormalization()
      elif (norm == "instance"):
        # InstanceNorm substitute: LayerNormalization over channels.
        return tf.keras.layers.LayerNormalization(axis=[1, 2, 3])
      else:
        return None

    # Create convolutional and activation sub-layers.
    self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=3, padding="same")
    self.norm1 = make_norm()
    self.act1 = tf.keras.layers.ReLU()
    self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=3, padding="same")
    self.norm2 = make_norm()
    self.act2 = tf.keras.layers.ReLU()
    # Create optional dropout layer when configured.
    self.dropout = (tf.keras.layers.Dropout(self.dropoutRate) if (self.dropoutRate) else None)

  # Forward pass for the configurable block.
  def call(self, x, training=False):
    # Apply first conv and optional normalization.
    out = self.conv1(x)
    if (self.norm1 is not None):
      out = self.norm1(out)
    out = self.act1(out)
    # Apply second conv and optional normalization.
    out = self.conv2(out)
    if (self.norm2 is not None):
      out = self.norm2(out)
    out = self.act2(out)
    # Apply dropout when configured.
    if (self.dropout is not None):
      out = self.dropout(out, training=training)
    # Return with residual connection when configured.
    if (self.residual):
      return tf.nn.relu(out + x)
    return out


class ResidualBlock(tf.keras.layers.Layer):
  r'''
  Residual convolutional block.

  Short summary:
    Two conv->BatchNorm->ReLU layers with an identity skip connection. A 1x1
    projection is applied to the identity when the number of channels differs.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Returns:
    tensorflow.Tensor: Output tensor with applied residual addition.
  '''

  # Initialize residual block.
  def __init__(self, inChannels: int, outChannels: int):
    super(ResidualBlock, self).__init__()
    # Create convolutional and normalization layers.
    self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=3, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=3, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()
    # Determine whether a 1x1 projection is needed for identity.
    self.needProj = (inChannels != outChannels)
    if (self.needProj):
      self.proj = tf.keras.layers.Conv2D(outChannels, kernel_size=1)

  # Forward pass for residual block.
  def call(self, x, training=False):
    # Preserve identity for skip connection.
    identity = x
    # First conv->bn->relu.
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = tf.nn.relu(out)
    # Second conv->bn.
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    # Project identity when necessary.
    if (self.needProj):
      identity = self.proj(identity)
    # Add skip connection and activate.
    out = out + identity
    return tf.nn.relu(out)


class AttentionGate(tf.keras.layers.Layer):
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

  Returns:
    tensorflow.Tensor: Reweighted skip features of shape matching the input skip map.
  '''

  # Initialize attention gate sub-layers.
  def __init__(self, F_g: int, F_l: int, F_int: int):
    super(AttentionGate, self).__init__()
    # Create gating and skip mapping branches.
    self.W_g = tf.keras.Sequential([
      tf.keras.layers.Conv2D(F_int, kernel_size=1, padding="same"),
      tf.keras.layers.BatchNormalization(),
    ])
    self.W_x = tf.keras.Sequential([
      tf.keras.layers.Conv2D(F_int, kernel_size=1, padding="same"),
      tf.keras.layers.BatchNormalization(),
    ])
    # Psi branch computes attention coefficients and applies sigmoid.
    self.psi = tf.keras.Sequential([
      tf.keras.layers.Conv2D(1, kernel_size=1, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("sigmoid"),
    ])
    # ReLU activation used before psi.
    self.relu = tf.keras.layers.ReLU()

  # Forward pass for attention gate.
  def call(self, g, x, training=False):
    # Map the gating signal.
    g1 = self.W_g(g, training=training)
    # Map the encoder skip features.
    x1 = self.W_x(x, training=training)
    # Combine and activate.
    psi = self.relu(g1 + x1)
    # Compute attention map and weight encoder features.
    psi = self.psi(psi, training=training)
    return x * psi


class DepthwiseSeparableConv(tf.keras.layers.Layer):
  r'''
  Depthwise separable convolution block.

  Short summary:
    Implements a depthwise convolution followed by a pointwise convolution,
    used to reduce parameter counts and computation in lightweight models.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Returns:
    tensorflow.Tensor: Activated output tensor with outChannels channels.
  '''

  # Initialize depthwise separable conv block.
  def __init__(self, inChannels: int, outChannels: int):
    super(DepthwiseSeparableConv, self).__init__()
    self.sep = tf.keras.layers.SeparableConv2D(outChannels, kernel_size=3, padding="same")
    self.bn = tf.keras.layers.BatchNormalization()
    self.act = tf.keras.layers.ReLU()

  # Forward pass for separable conv.
  def call(self, x, training=False):
    # Apply separable convolution.
    x = self.sep(x)
    # Apply batch normalization.
    x = self.bn(x, training=training)
    # Apply activation and return.
    return self.act(x)


class SEBlock(tf.keras.layers.Layer):
  r'''
  Squeeze-and-Excitation (SE) block.

  Short summary:
    Performs global channel-wise pooling followed by a small bottleneck
    MLP that produces per-channel scaling weights applied to the input.

  Parameters:
    channels (int): Number of input/output channels.
    reduction (int): Reduction ratio for the bottleneck. Default 16.

  Returns:
    tensorflow.Tensor: Recalibrated tensor of same shape as input.
  '''

  # Initialize SE block components.
  def __init__(self, channels: int, reduction: int = 16):
    super(SEBlock, self).__init__()
    reduced = max(1, channels // reduction)
    self.pool = tf.keras.layers.GlobalAveragePooling2D()
    self.fc1 = tf.keras.layers.Dense(reduced, activation="relu")
    self.fc2 = tf.keras.layers.Dense(channels, activation="sigmoid")

  # Forward pass for SE block.
  def call(self, x):
    # Squeeze global context and excite channels.
    s = self.pool(x)
    s = self.fc1(s)
    s = self.fc2(s)
    s = tf.reshape(s, (-1, 1, 1, tf.shape(s)[-1]))
    return x * s


class MultiResBlock(tf.keras.layers.Layer):
  r'''
  Multi-resolution convolution block capturing features at multiple receptive fields.

  Short summary:
    Parallel convolutions with different receptive fields followed by channel weighting
    to capture multi-scale context within a single block while controlling parameter growth.

  Parameters:
    inChans (int): Input channel count.
    outChans (int): Output channel count (total across all paths).
    alpha (float): Scaling factor for path channel allocation. Default 1.67.

  Returns:
    tensorflow.Tensor: Multi-resolution feature map with outChans channels.
  '''

  # Initialize MultiRes block and its parallel paths.
  def __init__(self, inChans: int, outChans: int, alpha: float = 1.67):
    super(MultiResBlock, self).__init__()
    # Compute allocation for multi-resolution paths.
    u = int(outChans / alpha)
    c1 = u
    c2 = int(u / 2)
    c3 = outChans - (c1 + c2)
    # Build parallel convolutional paths.
    self.conv3x3 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(c1, 3, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    self.conv5x5 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(c2, 3, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(c2, 3, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    self.conv7x7 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(c3, 3, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(c3, 3, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(c3, 3, padding="same"),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    if (inChans != outChans):
      self.convShortcut = tf.keras.layers.Conv2D(outChans, 1)
    else:
      self.convShortcut = (lambda x: x)
    self.batchNorm = tf.keras.layers.BatchNormalization()

  # Forward pass for MultiRes block.
  def call(self, x, training=False):
    # Execute each parallel path.
    path1 = (self.conv3x3(x, training=training) if hasattr(self.conv3x3, "call") else self.conv3x3(x))
    path2 = (self.conv5x5(x, training=training) if hasattr(self.conv5x5, "call") else self.conv5x5(x))
    path3 = (self.conv7x7(x, training=training) if hasattr(self.conv7x7, "call") else self.conv7x7(x))
    # Concatenate multi-resolution features.
    multiRes = tf.concat([path1, path2, path3], axis=-1)
    # Apply shortcut mapping when needed.
    shortcut = (self.convShortcut(x) if not callable(self.convShortcut) else self.convShortcut(x))
    out = self.batchNorm(multiRes + shortcut)
    return tf.nn.relu(out)


class DenseBlock(tf.keras.layers.Layer):
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

  Returns:
    tensorflow.Tensor: Concatenated output of all layers including original input.
  '''

  # Initialize Dense block with bottleneck layers.
  def __init__(self, inChans: int, numLayers: int = 4, growthRate: int = 32, bnSize: int = 4):
    super(DenseBlock, self).__init__()
    self.layers_list = []
    current = inChans
    for i in range(numLayers):
      seq = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(bnSize * growthRate, 1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(growthRate, 3, padding="same"),
      ])
      self.layers_list.append(seq)
      current += growthRate

  # Forward pass for Dense block.
  def call(self, x, training=False):
    # Accumulate dense features progressively.
    features = [x]
    for layer in self.layers_list:
      concated = tf.concat(features, axis=-1)
      out = (layer(concated, training=training) if hasattr(layer, "call") else layer(concated))
      features.append(out)
    return tf.concat(features, axis=-1)


class RecurrentConvLayer(tf.keras.layers.Layer):
  r'''
  Recurrent convolutional layer with internal state feedback.

  Short summary:
    Applies convolution repeatedly for T timesteps where each step receives feedback
    from its previous output, enabling iterative refinement of spatial features.

  Parameters:
    channels (int): Input/output channel count.
    t (int): Number of recurrent iterations. Default 2.

  Returns:
    tensorflow.Tensor: Refined feature map after T recurrent steps.
  '''

  # Initialize recurrent conv layer.
  def __init__(self, channels: int, t: int = 2):
    super(RecurrentConvLayer, self).__init__()
    self.t = t
    self.conv = tf.keras.layers.Conv2D(channels, 3, padding="same")
    self.bn = tf.keras.layers.BatchNormalization()

  # Forward pass applying recurrence.
  def call(self, x, training=False):
    # Initialize hidden state with the input.
    hidden = x
    for _ in range(self.t):
      # Apply shared convolution on the sum of input and hidden state.
      hidden = self.conv(x + hidden)
      hidden = self.bn(hidden, training=training)
      hidden = tf.nn.relu(hidden)
    return hidden


class ASPP(tf.keras.layers.Layer):
  r'''
  Atrous Spatial Pyramid Pooling for multi-scale context aggregation.

  Short summary:
    Parallel dilated convolutions at multiple rates plus image pooling to capture
    objects at different scales within a single feature map.

  Parameters:
    inChans (int): Input channel count.
    outChans (int): Output channel count after fusion.
    dilations (Tuple[int]): Dilation rates for parallel branches. Default (1,6,12,18).

  Returns:
    tensorflow.Tensor: Context-enriched feature map with outChans channels.
  '''

  # Initialize ASPP branches.
  def __init__(self, inChans: int, outChans: int, dilations: Tuple[int] = (1, 6, 12, 18)):
    super(ASPP, self).__init__()
    # Create ASPP branches.
    self.conv1x1 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(outChans, 1, use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    self.conv3x3_1 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(outChans, 3, padding="same", dilation_rate=dilations[1], use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    self.conv3x3_2 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(outChans, 3, padding="same", dilation_rate=dilations[2], use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    self.conv3x3_3 = tf.keras.Sequential([
      tf.keras.layers.Conv2D(outChans, 3, padding="same", dilation_rate=dilations[3], use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
    ])
    self.imagePoolConv = tf.keras.layers.Conv2D(outChans, 1, use_bias=False)
    self.project = tf.keras.Sequential([
      tf.keras.layers.Conv2D(outChans, 1, use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dropout(0.5),
    ])

  # Forward pass for ASPP.
  def call(self, x, training=False):
    # Record spatial dimensions for upsampling pooled features later.
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    # Apply parallel atrous and 1x1 branches.
    feat1 = self.conv1x1(x, training=training)
    feat2 = self.conv3x3_1(x, training=training)
    feat3 = self.conv3x3_2(x, training=training)
    feat4 = self.conv3x3_3(x, training=training)
    # Global image pooling branch.
    feat5 = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    feat5 = self.imagePoolConv(feat5)
    feat5 = tf.image.resize(feat5, (h, w), method="bilinear")
    concat = tf.concat([feat1, feat2, feat3, feat4, feat5], axis=-1)
    return self.project(concat, training=training)


class PatchEmbedding(tf.keras.layers.Layer):
  r'''
  2D image to patch embedding with optional positional encoding.

  Short summary:
    Converts input image into non-overlapping patches via convolutional projection
    and flattens them into a sequence for transformer processing.

  Parameters:
    inChans (int): Input channel count.
    embedDim (int): Embedding dimension per patch. Default 256.
    patchSize (int): Patch size (height=width). Default 4.

  Returns:
    Tuple[tensorflow.Tensor, Tuple[int,int]]: Patch sequence [B, N, embedDim] and (patchH, patchW).
  '''

  # Initialize patch embedding projection.
  def __init__(self, inChans: int, embedDim: int = 256, patchSize: int = 4):
    super(PatchEmbedding, self).__init__()
    self.patchSize = patchSize
    self.proj = tf.keras.layers.Conv2D(embedDim, kernel_size=patchSize, strides=patchSize)

  # Forward pass producing patch sequence and grid size.
  def call(self, x):
    # Project image into patch embeddings.
    x = self.proj(x)
    patchH = tf.shape(x)[1]
    patchW = tf.shape(x)[2]
    x = tf.reshape(x, (tf.shape(x)[0], patchH * patchW, tf.shape(x)[-1]))
    return x, (patchH, patchW)


class TransformerBlock(tf.keras.layers.Layer):
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

  Returns:
    tensorflow.Tensor: Transformed sequence of same shape as input.
  '''

  # Initialize transformer encoder block.
  def __init__(self, embedDim: int, numHeads: int = 8, mlpRatio: float = 4.0, dropout: float = 0.1):
    super(TransformerBlock, self).__init__()
    # Layer normalization and multi-head attention.
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.attn = tf.keras.layers.MultiHeadAttention(num_heads=numHeads, key_dim=embedDim // numHeads)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    hiddenDim = int(embedDim * mlpRatio)
    self.mlp = tf.keras.Sequential([
      tf.keras.layers.Dense(hiddenDim, activation=tf.nn.gelu),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(embedDim),
      tf.keras.layers.Dropout(dropout),
    ])

  # Forward pass for transformer block.
  def call(self, x, training=False):
    # Pre-normalize and apply multi-head self-attention.
    x_norm = self.norm1(x)
    attn_out = self.attn(x_norm, x_norm)
    x = x + attn_out
    # Feed-forward MLP with residual connection.
    x_norm = self.norm2(x)
    x = x + self.mlp(x_norm, training=training)
    return x


class ChannelAttn(tf.keras.layers.Layer):
  r'''
  Channel attention branch of CBAM using squeeze-and-excitation.

  Short summary:
    Global average and max pooling followed by shared MLP to compute
    channel-wise attention weights that rescale feature channels.

  Parameters:
    channels (int): Input channel count.
    reduction (int): Reduction ratio for bottleneck MLP. Default 16.

  Returns:
    tensorflow.Tensor: Channel-refined feature map.
  '''

  def __init__(self, channels: int, reduction: int = 16):
    super(ChannelAttn, self).__init__()
    reduced = max(1, channels // reduction)
    self.fc1 = tf.keras.layers.Dense(reduced, activation="relu")
    self.fc2 = tf.keras.layers.Dense(channels)

  def call(self, x):
    # Extract batch and channel dimensions and apply pooled streams.
    avgPooled = tf.reduce_mean(x, axis=[1, 2])
    maxPooled = tf.reduce_max(x, axis=[1, 2])
    avgOut = self.fc2(self.fc1(avgPooled))
    maxOut = self.fc2(self.fc1(maxPooled))
    attn = tf.nn.sigmoid(avgOut + maxOut)
    attn = tf.reshape(attn, (-1, 1, 1, tf.shape(attn)[-1]))
    return x * attn


class SpatialAttn(tf.keras.layers.Layer):
  r'''
  Spatial attention branch of CBAM using channel aggregation.

  Short summary:
    Channel-wise average and max pooling followed by convolution to compute
    spatial attention map that highlights important regions in feature maps.

  Parameters:
    kernelSize (int): Convolution kernel size for spatial attention. Default 7.

  Returns:
    tensorflow.Tensor: Spatially-refined feature map.
  '''

  def __init__(self, kernelSize: int = 7):
    super(SpatialAttn, self).__init__()
    self.conv = tf.keras.layers.Conv2D(1, kernel_size=kernelSize, padding="same", use_bias=False)

  def call(self, x):
    # Aggregate along channel axis using average and max pooling.
    avgPooled = tf.reduce_mean(x, axis=-1, keepdims=True)
    maxPooled = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avgPooled, maxPooled], axis=-1)
    attn = self.conv(concat)
    attn = tf.nn.sigmoid(attn)
    return x * attn


class CBAM(tf.keras.layers.Layer):
  r'''
  Convolutional Block Attention Module (channel + spatial attention).

  Short summary:
    Sequential channel attention followed by spatial attention to adaptively
    refine feature maps along both channel and spatial dimensions.

  Parameters:
    channels (int): Input/output channel count.
    reduction (int): Reduction ratio for channel attention. Default 16.

  Returns:
    tensorflow.Tensor: Attention-refined feature map with same shape as input.
  '''

  def __init__(self, channels: int, reduction: int = 16):
    super(CBAM, self).__init__()
    self.channelAttn = ChannelAttn(channels, reduction)
    self.spatialAttn = SpatialAttn()

  def call(self, x):
    x = self.channelAttn(x)
    x = self.spatialAttn(x)
    return x


class UNet(tf.keras.Model):
  r'''
  Standard 2D U-Net implemented with tf.keras.

  Short summary:
    A typical encoder-decoder U-Net with four downsampling stages, a bottleneck,
    and four symmetric upsampling stages. Supports learned transpose convolution
    upsampling or bilinear upsampling followed by 1x1 projection.

  Parameters:
    inputChannels (int): Number of input image channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Filters in the first stage. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose when True; otherwise use bilinear upsample.

  Attributes:
    enc1..enc4, center (tf.keras.Model/Layer): Encoder and bottleneck blocks.
    up1..up4, dec1..dec4 (tf.keras.Layer): Upsampling and decoder blocks.
    finalConv (tf.keras.layers.Layer): 1x1 conv to map to logits.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  # Initialize UNet architecture.
  def __init__(self, inputChannels: int = 3, numClasses: int = 2, baseChannels: int = 64,
               useConvTranspose2d: bool = True):
    super(UNet, self).__init__()
    # Encoder blocks and pooling layers.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)

    # Choose learned transpose upsampling or bilinear upsampling.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
      self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])
      self.dec1 = DoubleConv(baseChannels * 2, baseChannels)

    # Final projection to logits.
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  # Forward pass through UNet.
  def call(self, x, training=False):
    # Encoder forward.
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    # Decoder forward with upsampling and skip connections.
    u4 = self.up4(c)
    if (u4.shape[1] != e4.shape[1] or u4.shape[2] != e4.shape[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    u4 = tf.concat([u4, e4], axis=-1)
    d4 = self.dec4(u4, training=training)

    u3 = self.up3(d4)
    if (u3.shape[1] != e3.shape[1] or u3.shape[2] != e3.shape[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    u3 = tf.concat([u3, e3], axis=-1)
    d3 = self.dec3(u3, training=training)

    u2 = self.up2(d3)
    if (u2.shape[1] != e2.shape[1] or u2.shape[2] != e2.shape[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    u2 = tf.concat([u2, e2], axis=-1)
    d2 = self.dec2(u2, training=training)

    u1 = self.up1(d2)
    if (u1.shape[1] != e1.shape[1] or u1.shape[2] != e1.shape[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    u1 = tf.concat([u1, e1], axis=-1)
    d1 = self.dec1(u1, training=training)

    # Project to logits and return.
    logits = self.finalConv(d1)
    return logits


class DynamicUNet(tf.keras.Model):
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
    encs (list): Encoder blocks.
    pools (list): Pooling layers.
    ups (list): Upsampling layers.
    decs (list): Decoder blocks.
    finalConv (tf.keras.layers.Layer): 1x1 conv to logits.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  # Initialize dynamic UNet components.
  def __init__(
    self, inputChannels: int = 3, numClasses: int = 2, baseChannels: int = 64, depth: int = 4,
    upMode: str = "transpose", norm: str = "batch", dropout: float = 0.0, residual: bool = False
  ):
    super(DynamicUNet, self).__init__()
    # Validate arguments.
    assert (depth >= 1)
    assert (upMode in ("transpose", "bilinear"))
    # Build encoder lists and pooling.
    self.encs = []
    self.pools = []
    inCh = inputChannels
    channels = baseChannels
    for i in range(depth):
      self.encs.append(ConfigConv(inCh, channels, norm=norm, dropout=dropout, residual=residual))
      self.pools.append(tf.keras.layers.MaxPool2D(2))
      inCh = channels
      channels = (channels * 2)

    # Create center block.
    self.center = ConfigConv(inCh, channels, norm=norm, dropout=dropout, residual=residual)
    self.ups = []
    self.decs = []
    for i in range(depth):
      if (upMode == "transpose"):
        self.ups.append(tf.keras.layers.Conv2DTranspose(channels // 2, 2, strides=2))
      else:
        self.ups.append(tf.keras.Sequential(
          [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(channels // 2, 1)]))
      self.decs.append(ConfigConv(channels, channels // 2, norm=norm, dropout=dropout, residual=residual))
      channels = (channels // 2)

    # Final layer mapping to classes.
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  # Forward pass for DynamicUNet.
  def call(self, x, training=False):
    # Collect encoder features for skip connections.
    features = []
    for enc, pool in zip(self.encs, self.pools):
      x = enc(x, training=training)
      features.append(x)
      x = pool(x)

    # Process center block.
    x = self.center(x, training=training)

    # Decoder: iterate reversed features with upsample and concat.
    for up, dec, feat in zip(self.ups, self.decs, reversed(features)):
      x = up(x)
      if (tf.shape(x)[1] != tf.shape(feat)[1] or tf.shape(x)[2] != tf.shape(feat)[2]):
        x = tf.image.resize(x, (tf.shape(feat)[1], tf.shape(feat)[2]))
      x = tf.concat([x, feat], axis=-1)
      x = dec(x, training=training)

    logits = self.finalConv(x)
    return logits


class AttentionUNet(tf.keras.Model):
  r'''
  Attention U-Net variant with gating on skip connections.

  Short summary:
    Applies attention gating to encoder skip features before merging them into the decoder,
    improving localization and suppressing irrelevant responses in the skip maps.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose for upsampling when True.

  Attributes:
    enc1..enc4, center: Encoder/bottleneck blocks.
    att1..att4 (AttentionGate): Attention gating modules for skips.
    up1..up4, dec1..dec4: Decoder modules.
    finalConv (tf.keras.layers.Layer): 1x1 conv mapping to logits.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  # Initialize Attention U-Net.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    super(AttentionUNet, self).__init__()
    # Encoder blocks.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)

    # Choose upsampling method.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Attention gates for skip connections.
    self.att4 = AttentionGate(baseChannels * 8, baseChannels * 8, baseChannels * 4)
    self.att3 = AttentionGate(baseChannels * 4, baseChannels * 4, baseChannels * 2)
    self.att2 = AttentionGate(baseChannels * 2, baseChannels * 2, baseChannels)
    self.att1 = AttentionGate(baseChannels, baseChannels, max(baseChannels // 2, 1))

    # Decoder convs and final projection.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  # Forward pass for AttentionUNet.
  def call(self, x, training=False):
    # Encoder forward and pooling.
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    # Decoder with attention gating for each skip.
    u4 = self.up4(c)
    e4_att = self.att4(u4, e4, training=training)
    d4 = self.dec4(tf.concat([u4, e4_att], axis=-1), training=training)

    u3 = self.up3(d4)
    e3_att = self.att3(u3, e3, training=training)
    d3 = self.dec3(tf.concat([u3, e3_att], axis=-1), training=training)

    u2 = self.up2(d3)
    e2_att = self.att2(u2, e2, training=training)
    d2 = self.dec2(tf.concat([u2, e2_att], axis=-1), training=training)

    u1 = self.up1(d2)
    e1_att = self.att1(u1, e1, training=training)
    d1 = self.dec1(tf.concat([u1, e1_att], axis=-1), training=training)

    # Final logits.
    logits = self.finalConv(d1)
    return logits


class MobileUNet(tf.keras.Model):
  r'''
  Lightweight U-Net using depthwise separable convolutions.

  Short summary:
    Efficient U-Net variant that replaces standard convolutions with
    depthwise separable convolutions to reduce parameters and compute.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 32.
    useConvTranspose2d (bool): Use Conv2DTranspose for upsampling when True.

  Attributes:
    enc1..enc4, center: Encoder and bottleneck sequences.
    up1..up4, dec1..dec4: Decoder modules.
    finalConv (tensorflow.keras.layers.Layer): 1x1 conv for logits.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  # Initialize MobileUNet using depthwise separable conv blocks.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=32, useConvTranspose2d=True):
    super(MobileUNet, self).__init__()
    self.enc1 = tf.keras.Sequential(
      [DepthwiseSeparableConv(inputChannels, baseChannels), DepthwiseSeparableConv(baseChannels, baseChannels)])
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels, baseChannels * 2),
                                     DepthwiseSeparableConv(baseChannels * 2, baseChannels * 2)])
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels * 2, baseChannels * 4),
                                     DepthwiseSeparableConv(baseChannels * 4, baseChannels * 4)])
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels * 4, baseChannels * 8),
                                     DepthwiseSeparableConv(baseChannels * 8, baseChannels * 8)])
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    self.center = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels * 8, baseChannels * 16),
                                       DepthwiseSeparableConv(baseChannels * 16, baseChannels * 16)])

    # Choose upsampling method.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Decoder sequences using separable convs.
    self.dec4 = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels * 16, baseChannels * 8),
                                     DepthwiseSeparableConv(baseChannels * 8, baseChannels * 8)])
    self.dec3 = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels * 8, baseChannels * 4),
                                     DepthwiseSeparableConv(baseChannels * 4, baseChannels * 4)])
    self.dec2 = tf.keras.Sequential([DepthwiseSeparableConv(baseChannels * 4, baseChannels * 2),
                                     DepthwiseSeparableConv(baseChannels * 2, baseChannels * 2)])
    self.dec1 = tf.keras.Sequential(
      [DepthwiseSeparableConv(baseChannels * 2, baseChannels), DepthwiseSeparableConv(baseChannels, baseChannels)])
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  # Forward pass for MobileUNet.
  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = self.dec4(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = self.dec3(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = self.dec2(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = self.dec1(tf.concat([u1, e1], axis=-1), training=training)
    logits = self.finalConv(d1)
    return logits


class ResidualUNet(tf.keras.Model):
  r'''
  Residual U-Net variant built from ResidualBlock components.

  Short summary:
    A U-Net where encoder and decoder stages use residual blocks to
    ease optimization and improve gradient flow for deeper models.

  Parameters:
    inputChannels (int): Number of input channels.
    numClasses (int): Number of output classes.
    baseChannels (int): Base number of filters.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  # Initialize ResidualUNet architecture.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    super(ResidualUNet, self).__init__()
    self.enc1 = ResidualBlock(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = ResidualBlock(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = ResidualBlock(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = ResidualBlock(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    self.center = ResidualBlock(baseChannels * 8, baseChannels * 16)

    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    self.dec4 = ResidualBlock(baseChannels * 16, baseChannels * 8)
    self.dec3 = ResidualBlock(baseChannels * 8, baseChannels * 4)
    self.dec2 = ResidualBlock(baseChannels * 4, baseChannels * 2)
    self.dec1 = ResidualBlock(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  # Forward pass for ResidualUNet.
  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = self.dec4(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = self.dec3(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = self.dec2(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = self.dec1(tf.concat([u1, e1], axis=-1), training=training)

    return self.finalConv(d1)


class SEUNet(tf.keras.Model):
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
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  # Initialize SE-UNet architecture.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True, seReduction=16):
    super(SEUNet, self).__init__()
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.se1 = SEBlock(baseChannels, reduction=seReduction)
    self.pool1 = tf.keras.layers.MaxPool2D(2)

    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.se2 = SEBlock(baseChannels * 2, reduction=seReduction)
    self.pool2 = tf.keras.layers.MaxPool2D(2)

    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.se3 = SEBlock(baseChannels * 4, reduction=seReduction)
    self.pool3 = tf.keras.layers.MaxPool2D(2)

    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.se4 = SEBlock(baseChannels * 8, reduction=seReduction)
    self.pool4 = tf.keras.layers.MaxPool2D(2)

    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)
    self.seCenter = SEBlock(baseChannels * 16, reduction=seReduction)

    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)]
      )
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)]
      )
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)]
      )
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)]
      )

    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  # Forward pass for SEUNet.
  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    e1 = self.se1(e1)
    p1 = self.pool1(e1)

    e2 = self.enc2(p1, training=training)
    e2 = self.se2(e2)
    p2 = self.pool2(e2)

    e3 = self.enc3(p2, training=training)
    e3 = self.se3(e3)
    p3 = self.pool3(e3)

    e4 = self.enc4(p3, training=training)
    e4 = self.se4(e4)
    p4 = self.pool4(e4)

    c = self.center(p4, training=training)
    c = self.seCenter(c)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = self.dec4(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = self.dec3(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = self.dec2(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = self.dec1(tf.concat([u1, e1], axis=-1), training=training)

    return self.finalConv(d1)


class ResidualAttentionUNet(tf.keras.Model):
  r'''
  Residual U-Net combined with Attention Gates.

  Short summary:
    A U-Net variant that composes residual blocks in the encoder/decoder and
    applies attention gating on the skip connections to focus decoder features.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose for learned upsampling when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    super(ResidualAttentionUNet, self).__init__()
    # Encoder residual blocks and pooling.
    self.enc1 = ResidualBlock(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = ResidualBlock(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = ResidualBlock(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = ResidualBlock(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)

    # Center block.
    self.center = ResidualBlock(baseChannels * 8, baseChannels * 16)

    # Upsamplers.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Attention gates.
    self.att4 = AttentionGate(baseChannels * 8, baseChannels * 8, baseChannels * 4)
    self.att3 = AttentionGate(baseChannels * 4, baseChannels * 4, baseChannels * 2)
    self.att2 = AttentionGate(baseChannels * 2, baseChannels * 2, baseChannels)
    self.att1 = AttentionGate(baseChannels, baseChannels, max(baseChannels // 2, 1))

    # Decoder residual blocks.
    self.dec4 = ResidualBlock(baseChannels * 16, baseChannels * 8)
    self.dec3 = ResidualBlock(baseChannels * 8, baseChannels * 4)
    self.dec2 = ResidualBlock(baseChannels * 4, baseChannels * 2)
    self.dec1 = ResidualBlock(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    # Encoder.
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    # Decoder with attention gating.
    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    e4_att = self.att4(u4, e4, training=training)
    d4 = self.dec4(tf.concat([u4, e4_att], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    e3_att = self.att3(u3, e3, training=training)
    d3 = self.dec3(tf.concat([u3, e3_att], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    e2_att = self.att2(u2, e2, training=training)
    d2 = self.dec2(tf.concat([u2, e2_att], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    e1_att = self.att1(u1, e1, training=training)
    d1 = self.dec1(tf.concat([u1, e1_att], axis=-1), training=training)

    return self.finalConv(d1)


class MultiResUNet(tf.keras.Model):
  r'''
  MultiResUNet using MultiResBlock at each stage.

  Short summary:
    Uses MultiResBlock modules to capture multi-scale features within each encoder/decoder
    stage while keeping parameter growth under control.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    super(MultiResUNet, self).__init__()
    # Encoder MultiRes blocks.
    self.enc1 = MultiResBlock(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = MultiResBlock(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = MultiResBlock(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = MultiResBlock(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    self.center = MultiResBlock(baseChannels * 8, baseChannels * 16)

    # Upsamplers and decoder.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    self.dec4 = MultiResBlock(baseChannels * 16, baseChannels * 8)
    self.dec3 = MultiResBlock(baseChannels * 8, baseChannels * 4)
    self.dec2 = MultiResBlock(baseChannels * 4, baseChannels * 2)
    self.dec1 = MultiResBlock(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = self.dec4(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = self.dec3(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = self.dec2(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = self.dec1(tf.concat([u1, e1], axis=-1), training=training)

    return self.finalConv(d1)


class DenseUNet(tf.keras.Model):
  r'''
  DenseUNet integrating DenseBlocks and transition convolutions.

  Short summary:
    Encoder built from DenseBlocks with 1x1 transition convolutions to control
    channel dimensionality, producing feature reuse and improved gradient flow.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count. Default 32.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=32, useConvTranspose2d=True):
    super(DenseUNet, self).__init__()
    # Growth controls for Dense blocks.
    growth = max(4, baseChannels // 2)
    # Encoder dense blocks and transition convs.
    self.enc1 = DenseBlock(inputChannels, numLayers=3, growthRate=growth)
    out1 = inputChannels + 3 * growth
    self.trans1 = tf.keras.layers.Conv2D(baseChannels, 1)
    self.pool1 = tf.keras.layers.MaxPool2D(2)

    self.enc2 = DenseBlock(baseChannels, numLayers=3, growthRate=growth)
    out2 = baseChannels + 3 * growth
    self.trans2 = tf.keras.layers.Conv2D(baseChannels * 2, 1)
    self.pool2 = tf.keras.layers.MaxPool2D(2)

    self.enc3 = DenseBlock(baseChannels * 2, numLayers=3, growthRate=growth)
    out3 = baseChannels * 2 + 3 * growth
    self.trans3 = tf.keras.layers.Conv2D(baseChannels * 4, 1)
    self.pool3 = tf.keras.layers.MaxPool2D(2)

    self.enc4 = DenseBlock(baseChannels * 4, numLayers=3, growthRate=growth)
    out4 = baseChannels * 4 + 3 * growth
    self.trans4 = tf.keras.layers.Conv2D(baseChannels * 8, 1)
    self.pool4 = tf.keras.layers.MaxPool2D(2)

    # Center dense block.
    centerIn = baseChannels * 8
    self.center = DenseBlock(centerIn, numLayers=3, growthRate=growth)

    # Upsampling modules.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Decoder convs.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    # Encoder 1.
    e1 = self.enc1(x, training=training)
    t1 = self.trans1(e1)
    p1 = self.pool1(t1)
    # Encoder 2.
    e2 = self.enc2(p1, training=training)
    t2 = self.trans2(e2)
    p2 = self.pool2(t2)
    # Encoder 3.
    e3 = self.enc3(p2, training=training)
    t3 = self.trans3(e3)
    p3 = self.pool3(t3)
    # Encoder 4.
    e4 = self.enc4(p3, training=training)
    t4 = self.trans4(e4)
    p4 = self.pool4(t4)
    # Center.
    c = self.center(p4, training=training)

    # Decoder.
    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(t4)[1]) or (tf.shape(u4)[2] != tf.shape(t4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(t4)[1], tf.shape(t4)[2]))
    d4 = self.dec4(tf.concat([u4, t4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(t3)[1]) or (tf.shape(u3)[2] != tf.shape(t3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(t3)[1], tf.shape(t3)[2]))
    d3 = self.dec3(tf.concat([u3, t3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(t2)[1]) or (tf.shape(u2)[2] != tf.shape(t2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(t2)[1], tf.shape(t2)[2]))
    d2 = self.dec2(tf.concat([u2, t2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(t1)[1]) or (tf.shape(u1)[2] != tf.shape(t1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(t1)[1], tf.shape(t1)[2]))
    d1 = self.dec1(tf.concat([u1, t1], axis=-1), training=training)

    return self.finalConv(d1)


class R2UNet(tf.keras.Model):
  r'''
  Recurrent Residual U-Net using recurrent convolution layers.

  Short summary:
    Introduces recurrent convolutional layers (RCL) that iteratively refine
    features inside each encoder stage, useful for capturing contextual
    information while keeping model size modest.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.
    t (int): Number of recurrent iterations inside RCL. Default 2.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True, t=2):
    super(R2UNet, self).__init__()
    # Initial conv projections and recurrent conv layers.
    self.in1 = tf.keras.layers.Conv2D(baseChannels, 3, padding="same")
    self.enc1 = RecurrentConvLayer(baseChannels, t=t)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.in2 = tf.keras.layers.Conv2D(baseChannels * 2, 3, padding="same")
    self.enc2 = RecurrentConvLayer(baseChannels * 2, t=t)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.in3 = tf.keras.layers.Conv2D(baseChannels * 4, 3, padding="same")
    self.enc3 = RecurrentConvLayer(baseChannels * 4, t=t)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.in4 = tf.keras.layers.Conv2D(baseChannels * 8, 3, padding="same")
    self.enc4 = RecurrentConvLayer(baseChannels * 8, t=t)
    self.pool4 = tf.keras.layers.MaxPool2D(2)

    # Center projection and recurrent block.
    self.centerProj = tf.keras.layers.Conv2D(baseChannels * 16, 1)
    self.center = RecurrentConvLayer(baseChannels * 16, t=t)

    # Upsamplers.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Decoder convs.
    self.dec4 = tf.keras.layers.Conv2D(baseChannels * 8, 3, padding="same")
    self.dec3 = tf.keras.layers.Conv2D(baseChannels * 4, 3, padding="same")
    self.dec2 = tf.keras.layers.Conv2D(baseChannels * 2, 3, padding="same")
    self.dec1 = tf.keras.layers.Conv2D(baseChannels, 3, padding="same")
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    i1 = self.in1(x)
    e1 = self.enc1(i1, training=training)
    p1 = self.pool1(e1)
    i2 = self.in2(p1)
    e2 = self.enc2(i2, training=training)
    p2 = self.pool2(e2)
    i3 = self.in3(p2)
    e3 = self.enc3(i3, training=training)
    p3 = self.pool3(e3)
    i4 = self.in4(p3)
    e4 = self.enc4(i4, training=training)
    p4 = self.pool4(e4)

    cProj = self.centerProj(p4)
    c = self.center(cProj, training=training)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = tf.nn.relu(self.dec4(tf.concat([u4, e4], axis=-1)))

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = tf.nn.relu(self.dec3(tf.concat([u3, e3], axis=-1)))

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = tf.nn.relu(self.dec2(tf.concat([u2, e2], axis=-1)))

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = tf.nn.relu(self.dec1(tf.concat([u1, e1], axis=-1)))

    return self.finalConv(d1)


class ASPPUNet(tf.keras.Model):
  r'''
  U-Net with ASPP at the bottleneck.

  Short summary:
    Incorporates Atrous Spatial Pyramid Pooling (ASPP) at the bottleneck to capture
    multi-scale context before upsampling in the decoder.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    super(ASPPUNet, self).__init__()
    # Encoder path.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    # ASPP center.
    self.centerAspp = ASPP(baseChannels * 8, baseChannels * 16)

    # Upsamplers.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Decoder convs.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.centerAspp(p4, training=training)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = self.dec4(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = self.dec3(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = self.dec2(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = self.dec1(tf.concat([u1, e1], axis=-1), training=training)

    return self.finalConv(d1)


class TransUNet(tf.keras.Model):
  r'''
  Transformer-enhanced U-Net with a small transformer stack at the bottleneck.

  Short summary:
    Applies a patch embedding on the bottleneck features and processes them with
    a small transformer encoder stack, then projects back to spatial features
    for decoding. Useful to capture long-range dependencies.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    embedDim (int): Embedding dimension for transformer. Default 256.
    numHeads (int): Number of attention heads. Default 8.
    numEncoders (int): Number of transformer encoder blocks. Default 2.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, embedDim=256, numHeads=8, numEncoders=2,
               useConvTranspose2d=True):
    super(TransUNet, self).__init__()
    # Encoder.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)

    # Patch embedding and transformer stack.
    self.patchEmbed = PatchEmbedding(baseChannels * 8, embedDim=embedDim, patchSize=2)
    self.transformer = tf.keras.Sequential(
      [TransformerBlock(embedDim=embedDim, numHeads=numHeads) for _ in range(numEncoders)])
    self.transformProj = tf.keras.layers.Conv2D(baseChannels * 16, 1)

    # Upsamplers and decoder convs.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)

    patches, (ph, pw) = self.patchEmbed(p4)
    t = self.transformer(patches, training=training)
    b = tf.shape(t)[0]
    n = tf.shape(t)[1]
    d = tf.shape(t)[2]
    # Reshape transformer output back to spatial grid and project.
    t = tf.reshape(tf.transpose(t, perm=[0, 2, 1]), (b, d, ph, pw))
    # Convert to NHWC for conv projection.
    t = tf.transpose(t, perm=[0, 2, 3, 1])
    c = self.transformProj(t)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    d4 = self.dec4(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    d3 = self.dec3(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    d2 = self.dec2(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    d1 = self.dec1(tf.concat([u1, e1], axis=-1), training=training)

    return self.finalConv(d1)


class CBAMUNet(tf.keras.Model):
  r'''
  U-Net variant applying CBAM attention to encoder skip features.

  Short summary:
    Uses the Convolutional Block Attention Module (CBAM) to refine encoder skip
    features via channel and spatial attention before merging them into the decoder.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    super(CBAMUNet, self).__init__()
    # Encoder.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool4 = tf.keras.layers.MaxPool2D(2)
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)

    # Upsamplers.
    if (useConvTranspose2d):
      self.up4 = tf.keras.layers.Conv2DTranspose(baseChannels * 8, 2, strides=2)
      self.up3 = tf.keras.layers.Conv2DTranspose(baseChannels * 4, 2, strides=2)
      self.up2 = tf.keras.layers.Conv2DTranspose(baseChannels * 2, 2, strides=2)
      self.up1 = tf.keras.layers.Conv2DTranspose(baseChannels, 2, strides=2)
    else:
      self.up4 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
      self.up3 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
      self.up2 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
      self.up1 = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])

    # Decoder convs and CBAM modules per skip.
    self.dec4 = DoubleConv(baseChannels * 16, baseChannels * 8)
    self.dec3 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.dec2 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.dec1 = DoubleConv(baseChannels * 2, baseChannels)
    self.cbam4 = CBAM(baseChannels * 8)
    self.cbam3 = CBAM(baseChannels * 4)
    self.cbam2 = CBAM(baseChannels * 2)
    self.cbam1 = CBAM(baseChannels)
    self.finalConv = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    e1 = self.enc1(x, training=training)
    p1 = self.pool1(e1)
    e2 = self.enc2(p1, training=training)
    p2 = self.pool2(e2)
    e3 = self.enc3(p2, training=training)
    p3 = self.pool3(e3)
    e4 = self.enc4(p3, training=training)
    p4 = self.pool4(e4)
    c = self.center(p4, training=training)

    u4 = self.up4(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    e4_att = self.cbam4(e4)
    d4 = self.dec4(tf.concat([u4, e4_att], axis=-1), training=training)

    u3 = self.up3(d4)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    e3_att = self.cbam3(e3)
    d3 = self.dec3(tf.concat([u3, e3_att], axis=-1), training=training)

    u2 = self.up2(d3)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    e2_att = self.cbam2(e2)
    d2 = self.dec2(tf.concat([u2, e2_att], axis=-1), training=training)

    u1 = self.up1(d2)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    e1_att = self.cbam1(e1)
    d1 = self.dec1(tf.concat([u1, e1_att], axis=-1), training=training)

    return self.finalConv(d1)


class EfficientUNet(tf.keras.Model):
  r'''
  Efficient wrapper around MobileUNet.

  Short summary:
    A thin wrapper that exposes the MobileUNet as an "efficient" option in the
    factory while keeping the same API.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=32, useConvTranspose2d=True):
    super(EfficientUNet, self).__init__()
    self.model = MobileUNet(inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
                            useConvTranspose2d=useConvTranspose2d)

  def call(self, x, training=False):
    return self.model(x, training=training)


class BoundaryAwareUNet(tf.keras.Model):
  r'''
  Boundary-aware U-Net with explicit boundary detection branch.

  Short summary:
    A dual-branch architecture with a main segmentation decoder and a lighter
    boundary decoder producing an auxiliary boundary map. Useful for losses that
    combine segmentation and boundary supervision.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of segmentation classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): Use Conv2DTranspose when True.
    boundaryWeight (float): Weighting factor for combining boundary predictions. Default 0.5.

  Returns:
    Tuple[tensorflow.Tensor, tensorflow.Tensor]: (segmentation_logits, boundary_map)
  '''

  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True, boundaryWeight=0.5):
    super(BoundaryAwareUNet, self).__init__()
    self.boundaryWeight = boundaryWeight
    # Build encoder and boundary encoder pathways.
    self.encoder = [DoubleConv(inputChannels if i == 0 else baseChannels * (2 ** i), baseChannels * (2 ** i)) for i in
                    range(4)]
    self.encoder = [tf.keras.models.clone_model(layer) if hasattr(layer, "call") else layer for layer in self.encoder]
    # Use explicit layers instead of ModuleList: create as attributes.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    self.enc2 = DoubleConv(baseChannels, baseChannels * 2)
    self.enc3 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.enc4 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.pool1 = tf.keras.layers.MaxPool2D(2)
    self.pool2 = tf.keras.layers.MaxPool2D(2)
    self.pool3 = tf.keras.layers.MaxPool2D(2)
    self.pool4 = tf.keras.layers.MaxPool2D(2)

    # Boundary encoder with reduced capacity.
    boundaryBase = max(16, baseChannels // 4)
    self.benc1 = DoubleConv(inputChannels, boundaryBase)
    self.benc2 = DoubleConv(boundaryBase, boundaryBase * 2)
    self.benc3 = DoubleConv(boundaryBase * 2, boundaryBase * 4)
    self.benc4 = DoubleConv(boundaryBase * 4, boundaryBase * 8)

    # Bottlenecks.
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)
    self.boundaryCenter = DoubleConv(boundaryBase * 8, boundaryBase * 8)

    # Upsamplers and decoder blocks for main and boundary paths.
    if (useConvTranspose2d):
      self.up1_main = tf.keras.layers.Conv2DTranspose(baseChannels * 16, baseChannels * 8, 2, strides=2)
    else:
      self.up1_main = tf.keras.Sequential(
        [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 8, 1)])
    self.dec1_main = DoubleConv(baseChannels * 16, baseChannels * 8)
    # For brevity create symmetric decoder levels using the same pattern.
    self.up2_main = tf.keras.layers.Conv2DTranspose(baseChannels * 8, baseChannels * 4, 2,
                                                    strides=2) if useConvTranspose2d else tf.keras.Sequential(
      [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 4, 1)])
    self.dec2_main = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.up3_main = tf.keras.layers.Conv2DTranspose(baseChannels * 4, baseChannels * 2, 2,
                                                    strides=2) if useConvTranspose2d else tf.keras.Sequential(
      [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels * 2, 1)])
    self.dec3_main = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.up4_main = tf.keras.layers.Conv2DTranspose(baseChannels * 2, baseChannels, 2,
                                                    strides=2) if useConvTranspose2d else tf.keras.Sequential(
      [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(baseChannels, 1)])
    self.dec4_main = DoubleConv(baseChannels * 2, baseChannels)

    # Boundary decoder path (simpler).
    self.bup1 = tf.keras.layers.Conv2DTranspose(boundaryBase * 8, boundaryBase * 4, 2,
                                                strides=2) if useConvTranspose2d else tf.keras.Sequential(
      [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(boundaryBase * 4, 1)])
    self.bdec1 = DoubleConv(boundaryBase * 8, boundaryBase * 4)
    self.bup2 = tf.keras.layers.Conv2DTranspose(boundaryBase * 4, boundaryBase * 2, 2,
                                                strides=2) if useConvTranspose2d else tf.keras.Sequential(
      [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(boundaryBase * 2, 1)])
    self.bdec2 = DoubleConv(boundaryBase * 4, boundaryBase * 2)
    self.bup3 = tf.keras.layers.Conv2DTranspose(boundaryBase * 2, boundaryBase, 2,
                                                strides=2) if useConvTranspose2d else tf.keras.Sequential(
      [tf.keras.layers.UpSampling2D(2, interpolation="bilinear"), tf.keras.layers.Conv2D(boundaryBase, 1)])
    self.bdec3 = DoubleConv(boundaryBase * 2, boundaryBase)

    self.boundaryHead = tf.keras.layers.Conv2D(1, 1)
    self.segmentationHead = tf.keras.layers.Conv2D(numClasses, 1)

  def call(self, x, training=False):
    # Encode main path.
    e1 = self.enc1(x, training=training)
    b1 = self.benc1(x, training=training)
    p1 = self.pool1(e1)
    pb1 = self.pool1(b1)

    e2 = self.enc2(p1, training=training)
    b2 = self.benc2(pb1, training=training)
    p2 = self.pool2(e2)
    pb2 = self.pool2(b2)

    e3 = self.enc3(p2, training=training)
    b3 = self.benc3(pb2, training=training)
    p3 = self.pool3(e3)
    pb3 = self.pool3(b3)

    e4 = self.enc4(p3, training=training)
    b4 = self.benc4(pb3, training=training)
    p4 = self.pool4(e4)
    pb4 = self.pool4(b4)

    # Bottleneck.
    c = self.center(p4, training=training)
    bc = self.boundaryCenter(pb4, training=training)

    # Main decoder.
    u4 = self.up1_main(c)
    if (tf.shape(u4)[1] != tf.shape(e4)[1] or tf.shape(u4)[2] != tf.shape(e4)[2]):
      u4 = tf.image.resize(u4, (tf.shape(e4)[1], tf.shape(e4)[2]))
    xEnc = self.dec1_main(tf.concat([u4, e4], axis=-1), training=training)

    u3 = self.up2_main(xEnc)
    if (tf.shape(u3)[1] != tf.shape(e3)[1] or tf.shape(u3)[2] != tf.shape(e3)[2]):
      u3 = tf.image.resize(u3, (tf.shape(e3)[1], tf.shape(e3)[2]))
    xEnc = self.dec2_main(tf.concat([u3, e3], axis=-1), training=training)

    u2 = self.up3_main(xEnc)
    if (tf.shape(u2)[1] != tf.shape(e2)[1] or tf.shape(u2)[2] != tf.shape(e2)[2]):
      u2 = tf.image.resize(u2, (tf.shape(e2)[1], tf.shape(e2)[2]))
    xEnc = self.dec3_main(tf.concat([u2, e2], axis=-1), training=training)

    u1 = self.up4_main(xEnc)
    if (tf.shape(u1)[1] != tf.shape(e1)[1] or tf.shape(u1)[2] != tf.shape(e1)[2]):
      u1 = tf.image.resize(u1, (tf.shape(e1)[1], tf.shape(e1)[2]))
    xEnc = self.dec4_main(tf.concat([u1, e1], axis=-1), training=training)

    # Boundary decoder.
    bu = self.bup1(bc)
    if (tf.shape(bu)[1] != tf.shape(b4)[1] or tf.shape(bu)[2] != tf.shape(b4)[2]):
      bu = tf.image.resize(bu, (tf.shape(b4)[1], tf.shape(b4)[2]))
    bDec = self.bdec1(tf.concat([bu, b4], axis=-1), training=training)

    bu = self.bup2(bDec)
    if (tf.shape(bu)[1] != tf.shape(b3)[1] or tf.shape(bu)[2] != tf.shape(b3)[2]):
      bu = tf.image.resize(bu, (tf.shape(b3)[1], tf.shape(b3)[2]))
    bDec = self.bdec2(tf.concat([bu, b3], axis=-1), training=training)

    bu = self.bup3(bDec)
    if (tf.shape(bu)[1] != tf.shape(b2)[1] or tf.shape(bu)[2] != tf.shape(b2)[2]):
      bu = tf.image.resize(bu, (tf.shape(b2)[1], tf.shape(b2)[2]))
    bDec = self.bdec3(tf.concat([bu, b2], axis=-1), training=training)

    # Heads.
    boundaryMap = self.boundaryHead(bDec)
    segLogits = self.segmentationHead(xEnc)

    # Resize outputs to input spatial size if required.
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    if (tf.shape(segLogits)[1] != h or tf.shape(segLogits)[2] != w):
      segLogits = tf.image.resize(segLogits, (h, w))
    if (tf.shape(boundaryMap)[1] != h or tf.shape(boundaryMap)[2] != w):
      boundaryMap = tf.image.resize(boundaryMap, (h, w))

    return segLogits, boundaryMap


class VNet(tf.keras.Model):
  r'''
  V-Net style residual encoder-decoder implemented as a lazily-built Model subclass.

  This class preserves the original functional builder behaviour but exposes it
  as a tf.keras.Model subclass so it matches the other model classes in this
  module. The internal functional model is constructed on first call using the
  same building logic as the previous functional `VNet` builder.

  Parameters:
    inputSize (tuple): Input size (H, W, C). Supports None for dynamic spatial dimensions. Default (256, 256, 1).
    kernelInitializer (str): Kernel initializer for convolutional layers. Default "he_normal".
    dropoutRatio (float): Dropout ratio for residual blocks. Default 0.0 (no dropout).
    dropoutType (str): Type of dropout to apply ("spatial" or "standard"). Default "spatial".
    activation (str): Activation function to use. Default "relu".
    applyBatchNorm (bool): Whether to apply batch normalization after convolutions. Default False.
    concatenateType (str): Method to merge skip connections ("concatenate" or "add"). Default "concatenate".
    noOfLevels (int): Number of encoder/decoder levels. Default 5.
    numClasses (int): Number of output classes for the final segmentation head. Default 1.

  Returns:
    tensorflow.Tensor: Logits tensor of shape [B, H, W, numClasses].
  '''

  def __init__(
    self,
    inputSize=(256, 256, 1),  # Input size (H,W,C) - supports None for dynamic spatial dims.
    kernelInitializer="he_normal",
    dropoutRatio=0.0,
    dropoutType="spatial",
    activation="relu",
    applyBatchNorm=False,
    concatenateType="concatenate",
    noOfLevels=5,
    numClasses: int = 1,
  ):
    super(VNet, self).__init__()
    # Store configuration; the internal functional model will be created lazily
    # on the first call so that construction does not require input tensors.
    self.inputSize = inputSize
    self.kernelInitializer = kernelInitializer
    self.dropoutRatio = dropoutRatio
    self.dropoutType = dropoutType
    self.activation = activation
    self.applyBatchNorm = applyBatchNorm
    self.concatenateType = concatenateType
    self.noOfLevels = noOfLevels
    self.numClasses = numClasses
    self._model = None

  def _build_internal_model(self):
    # Reuse the original functional builder's code to construct an inner Model.
    inputs = tf.keras.Input(shape=self.inputSize)

    # Small convolutional helper block used by the builder.
    def ConvBlock(x, filters, kernel_size=3):
      # Apply a Conv2D layer with specified filters and kernel size.
      x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=self.kernelInitializer)(x)
      # Optionally apply batch normalization.
      if (self.applyBatchNorm):
        x = tf.keras.layers.BatchNormalization()(x)
      # Optionally apply activation.
      if (self.activation):
        x = tf.keras.layers.Activation(self.activation)(x)
      # Return the processed tensor.
      return x

    # Residual block composed of two ConvBlock calls plus a projected shortcut.
    def ResidualBlock(x, filters, kernel_size=3, dropout=0.0):
      # Preserve the shortcut for residual addition.
      shortcut = x
      # First conv sub-block.
      out = ConvBlock(x, filters, kernel_size=kernel_size)
      # Second conv sub-block.
      out = ConvBlock(out, filters, kernel_size=kernel_size)
      # Project shortcut to requested width using 1x1 conv.
      shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same")(shortcut)
      # Add the residual connection.
      out = tf.keras.layers.Add()([out, shortcut])
      # Optionally apply dropout.
      if (dropout and dropout > 0.0):
        out = tf.keras.layers.Dropout(dropout)(out)
      # Apply activation after the residual addition.
      out = tf.keras.layers.Activation(self.activation)(out)
      # Return the residual output.
      return out

    # Small helper layer that resizes a tensor to match the spatial size of a
    # reference tensor. Wrapping tf.image.resize in a Keras Layer ensures the
    # op can accept KerasTensors when building the functional model.
    class ResizeToMatch(tf.keras.layers.Layer):
      def __init__(self, **kwargs):
        super(ResizeToMatch, self).__init__(**kwargs)

      def call(self, inputs):
        # inputs is a list or tuple (x, ref) where `x` is resized to match `ref`.
        x, ref = inputs
        return tf.image.resize(x, (tf.shape(ref)[1], tf.shape(ref)[2]))

    # Initial convolutional projection.
    x = ConvBlock(inputs, 16, kernel_size=5)
    stages = []
    channels = []
    filters = 16
    # Build encoder stages with residual blocks and pooling.
    for _ in range(self.noOfLevels):
      r = ResidualBlock(x, filters, kernel_size=5, dropout=self.dropoutRatio)
      stages.append(r)
      channels.append(filters)
      x = tf.keras.layers.MaxPool2D(pool_size=2)(r)
      filters = filters * 2

    # Center residual block.
    x = ResidualBlock(x, filters, kernel_size=3, dropout=self.dropoutRatio)

    # First upsample.
    filters = filters // 2
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    if (self.applyBatchNorm):
      x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(self.activation)(x)

    # Decoder.
    for skip, skipCh in zip(list(reversed(stages[:-1])), list(reversed(channels[:-1]))):
      # Resize decoder feature-map to match the current skip connection.
      x = ResizeToMatch()([x, skip])
      if (self.concatenateType == "concatenate"):
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip])
      else:
        # Project decoder features to match skip channels for elementwise addition.
        x = tf.keras.layers.Conv2D(int(skipCh), 1, padding="same")(x)
        x = tf.keras.layers.Add()([x, skip])
      # Process merged features with a residual block.
      x = ResidualBlock(x, int(skipCh), kernel_size=3, dropout=self.dropoutRatio)
      upFilters = max(1, int(skipCh) // 2)
      x = tf.keras.layers.Conv2DTranspose(upFilters, 2, strides=2, padding="same")(x)
      if (self.applyBatchNorm):
        x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation(self.activation)(x)

    # Last skip concat with the earliest encoder stage.
    lastSkip = stages[0]
    # Resize final decoder map to match the first encoder stage before merging.
    x = ResizeToMatch()([x, lastSkip])
    if (self.concatenateType == "concatenate"):
      x = tf.keras.layers.Concatenate(axis=-1)([x, lastSkip])
    else:
      # Project decoder features to match the earliest skip channels for elementwise addition.
      x = tf.keras.layers.Conv2D(int(channels[0]), 1, padding="same")(x)
      x = tf.keras.layers.Add()([x, lastSkip])
    # Final residual processing after the last merge.
    x = ResidualBlock(x, int(channels[0]), kernel_size=3, dropout=self.dropoutRatio)

    outputs = tf.keras.layers.Conv2D(self.numClasses, (1, 1), use_bias=True)(x)
    outputs = tf.keras.layers.Activation("sigmoid")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

  def build(self, input_shape):
    # Construct the internal functional model during the Keras build phase.
    # Creating sub-layers here ensures no new state is added after the layer
    # is marked as built which would raise an error.
    if (self._model is None):
      self._model = self._build_internal_model()
    # Call parent build to mark this layer/model as built.
    super(VNet, self).build(input_shape)

  def call(self, x, training=False):
    # Forward the input through the internal functional model.
    return self._model(x, training=training)


class SegNet(tf.keras.Model):
  r'''
  SegNet architecture for multi-class semantic segmentation (Class-based Implementation).

  Short summary:
    Memory-efficient encoder-decoder network supporting VGG16, ResNet50, 
    MobileNetV2, or Vanilla backbones. Uses Functional API internally for 
    optimal graph construction and memory usage.

  Parameters:
    inputChannels (int): Number of input image channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    level (int): Decoder starting stage (1-4). Higher levels use deeper features.
    encoder (str): Backbone type: "VGG16", "ResNet50", "MobileNet", or "Vanilla".
    inputSize (tuple, optional): Fixed input shape (H, W, C). None = dynamic.
    useBias (bool): Whether Conv layers use bias. Default False (BN compatible).

  Returns:
    tensorflow.Tensor: Segmentation probabilities [B, H, W, numClasses].
  '''

  def __init__(
    self, inputChannels=3, numClasses=2, level=3, encoder="VGG16",
    inputSize=None, useBias=False
  ):
    super(SegNet, self).__init__(name=f"SegNet_{encoder}_L{level}")

    # Validate the decoder starting level is within acceptable range.
    if (level not in [1, 2, 3, 4]):
      raise ValueError(f"level must be 1-4, got {level}")
    # Validate the requested encoder type is supported.
    if (encoder not in ["VGG16", "ResNet50", "MobileNet", "Vanilla"]):
      raise ValueError(f"Unsupported encoder: {encoder}")
    # Validate the number of classes is at least two.
    if (numClasses < 2):
      raise ValueError(f"numClasses must be >= 2, got {numClasses}")

    # Store the decoder start level configuration.
    self.level = level
    # Store the encoder type string.
    self.encoderType = encoder
    # Store the number of output classes.
    self.numClasses = numClasses
    # Store the number of input channels.
    self.inputChannels = inputChannels
    # Store whether convolution layers should include a bias term.
    self.useBias = useBias

    # Handle input size specification to decide dynamic vs fixed shapes.
    if (inputSize is None):
      # Default to dynamic spatial dimensions while preserving channel count.
      self.inputSize = (None, None, inputChannels)
      self._is_fixed_shape = False
    else:
      # If only H,W provided, append channels to form (H,W,C).
      if (len(inputSize) == 2):
        self.inputSize = tuple(inputSize) + (inputChannels,)
      # If full shape provided use it directly.
      elif (len(inputSize) == 3):
        self.inputSize = tuple(inputSize)
      else:
        # Reject invalid inputSize shapes.
        raise ValueError(f"inputSize must be (H, W) or (H, W, C), got {inputSize}")
      # Mark that a fixed shape functional model should be built.
      self._is_fixed_shape = True

    # === Build Model Components: encoder, decoder and head. ===
    # Initialize encoder layers based on selected backbone.
    self._build_encoder_layers()
    # Initialize decoder layers for the upsampling pathway.
    self._build_decoder_layers()
    # Initialize the final classification head.
    self._build_output_head()

    # Prepare a functional model instance when a fixed input size is requested.
    self._functional_model = None
    if (self._is_fixed_shape):
      self._build_functional_model()

  def _build_encoder_layers(self):
    '''Initialize encoder backbone layers based on selected type.'''

    # Route to the encoder initializer for the selected backbone.
    if (self.encoderType == "VGG16"):
      self._init_vgg_encoder()
    elif (self.encoderType == "ResNet50"):
      self._init_resnet50_encoder()
    elif (self.encoderType == "MobileNet"):
      self._init_mobilenet_encoder()
    elif (self.encoderType == "Vanilla"):
      self._init_vanilla_encoder()

  def _init_vgg_encoder(self):
    '''VGG16-style encoder: 4 blocks returning feature levels at strides 2,4,8,16.'''

    def _vgg_block(filters, block_num):
      layers_list = []
      reps = 2 if block_num <= 2 else 3
      for i in range(reps):
        layers_list.append(
          tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same",
                                 use_bias=self.useBias, name=f"vgg_b{block_num}_c{i + 1}"
                                 )
        )
      layers_list.append(
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=f"vgg_b{block_num}_pool")
      )
      return layers_list

    self.vgg_layers = []
    # Populate VGG block layers for each block configuration.
    for block_num, filters in [(1, 64), (2, 128), (3, 256), (4, 512), (5, 512)]:
      self.vgg_layers.extend(_vgg_block(filters, block_num))

  def _init_resnet50_encoder(self):
    '''ResNet50 encoder using keras.applications for memory efficiency.'''

    # We'll build this dynamically in _run_encoder to avoid graph duplication
    self.resnet_base = None

  def _init_mobilenet_encoder(self):
    '''MobileNetV2 encoder using keras.applications for memory efficiency.'''

    self.mb_base = None

  def _init_vanilla_encoder(self):
    '''Vanilla encoder with correct channel progression: in→64→128→256→512.'''

    # Kernel, padding and pooling configuration for vanilla blocks.
    kernel, pad, pool = 3, 1, 2
    # Container for the vanilla encoder layers.
    self.vanilla_layers = []

    # Block 1: in_ch -> 64
    # Block 1 layers from input channels to 64 filters.
    self.vanilla_layers.extend([
      tf.keras.layers.ZeroPadding2D((pad, pad), name="van_pad1"),
      tf.keras.layers.Conv2D(64, (kernel, kernel), padding="valid", use_bias=self.useBias, name="van_conv1"),
      tf.keras.layers.BatchNormalization(name="van_bn1"),
      tf.keras.layers.Activation("relu", name="van_relu1"),
      tf.keras.layers.MaxPooling2D((pool, pool), name="van_pool1"),
    ])
    # Block 2: 64 -> 128
    self.vanilla_layers.extend([
      tf.keras.layers.ZeroPadding2D((pad, pad), name="van_pad2"),
      tf.keras.layers.Conv2D(128, (kernel, kernel), padding="valid", use_bias=self.useBias, name="van_conv2"),
      tf.keras.layers.BatchNormalization(name="van_bn2"),
      tf.keras.layers.Activation("relu", name="van_relu2"),
      tf.keras.layers.MaxPooling2D((pool, pool), name="van_pool2"),
    ])
    # Block 3: 128 -> 256
    self.vanilla_layers.extend([
      tf.keras.layers.ZeroPadding2D((pad, pad), name="van_pad3"),
      tf.keras.layers.Conv2D(256, (kernel, kernel), padding="valid", use_bias=self.useBias, name="van_conv3"),
      tf.keras.layers.BatchNormalization(name="van_bn3"),
      tf.keras.layers.Activation("relu", name="van_relu3"),
      tf.keras.layers.MaxPooling2D((pool, pool), name="van_pool3"),
    ])
    # Block 4: 256 -> 512
    self.vanilla_layers.extend([
      tf.keras.layers.ZeroPadding2D((pad, pad), name="van_pad4"),
      tf.keras.layers.Conv2D(512, (kernel, kernel), padding="valid", use_bias=self.useBias, name="van_conv4"),
      tf.keras.layers.BatchNormalization(name="van_bn4"),
      tf.keras.layers.Activation("relu", name="van_relu4"),
      tf.keras.layers.MaxPooling2D((pool, pool), name="van_pool4"),
    ])

  def _build_decoder_layers(self):
    '''Build decoder layers with unique names to prevent duplication errors.'''

    # Decoder config: (filters, do_upsample) for 5 stages
    # Only first `level` stages perform upsampling
    # Decoder configuration tuples: (filters, do_upsample) for 5 stages.
    decoder_stages = [
      (512, True), (512, True), (256, True), (128, True), (64, False)
    ]

    # Container mapping for decoder layer components.
    self.dec_layers = {}
    # Iterate over decoder stage configurations and create layers with unique names.
    for i, (filters, do_upsample) in enumerate(decoder_stages):
      suffix = f"_d{i}"
      # Zero padding before the conv for this stage.
      self.dec_layers[f"pad{suffix}"] = tf.keras.layers.ZeroPadding2D((1, 1), name=f"dec_pad{suffix}")
      # Convolution for this decoder stage.
      self.dec_layers[f"conv{suffix}"] = tf.keras.layers.Conv2D(
        filters, (3, 3), padding="valid", use_bias=self.useBias, name=f"dec_conv{suffix}"
      )
      # Batch normalization for this decoder stage.
      self.dec_layers[f"bn{suffix}"] = tf.keras.layers.BatchNormalization(name=f"dec_bn{suffix}")
      # ReLU activation for this decoder stage.
      self.dec_layers[f"relu{suffix}"] = tf.keras.layers.Activation("relu", name=f"dec_relu{suffix}")
      # Optional upsampling operator when this stage is configured to upsample.
      if (do_upsample):
        self.dec_layers[f"up{suffix}"] = tf.keras.layers.UpSampling2D(
          size=(2, 2), interpolation="bilinear", name=f"dec_up{suffix}"
        )

  def _build_output_head(self):
    '''Build final classification head.'''

    # Configure output convolution and activation based on number of classes.
    if (self.numClasses > 2):
      # Multi-class classification head with softmax activation.
      self.out_conv = tf.keras.layers.Conv2D(
        self.numClasses, (3, 3), padding="same", use_bias=self.useBias, name="out_conv"
      )
      self.out_act = tf.keras.layers.Activation("softmax", name="out_softmax")
    else:
      # Binary segmentation head with sigmoid activation.
      self.out_conv = tf.keras.layers.Conv2D(
        1, (3, 3), padding="same", use_bias=self.useBias, name="out_conv"
      )
      self.out_act = tf.keras.layers.Activation("sigmoid", name="out_sigmoid")

  def _run_vgg_encoder(self, x, training=False):
    '''Execute VGG encoder and return 4 feature levels.'''

    # Container for collected feature maps at pooling milestones.
    levels = []
    # Counter to track encountered pooling layers.
    pool_count = 0
    # Execute each layer in the VGG sequence and collect the first four pooled outputs.
    for layer in self.vgg_layers:
      x = layer(x, training=training) if isinstance(layer, tf.keras.layers.BatchNormalization) else layer(x)
      if ("pool" in layer.name):
        pool_count += 1
        if (pool_count <= 4):  # Collect first 4 pooling outputs.
          levels.append(x)
    # Return the collected encoder levels.
    return levels

  def _run_vanilla_encoder(self, x, training=False):
    '''Execute Vanilla encoder and return 4 feature levels.'''

    # Container for vanilla encoder feature levels.
    levels = []
    # Execute each vanilla layer and collect outputs at pooling layers.
    for i, layer in enumerate(self.vanilla_layers):
      x = layer(x, training=training) if isinstance(layer, tf.keras.layers.BatchNormalization) else layer(x)
      if ("pool" in layer.name):
        levels.append(x)
    # Return the collected four levels.
    return levels

  def _run_resnet_encoder(self, x, training=False):
    '''Execute ResNet50 encoder and return 4 feature levels.'''

    # Lazily build a ResNet50 base model on first call to avoid duplicated graphs.
    if (self.resnet_base is None):
      # Create a Keras Input that reuses the tensor `x` for the base model.
      baseInput = tf.keras.layers.Input(tensor=x)
      baseModel = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_tensor=baseInput
      )
      try:
        level_outputs = [
          baseModel.get_layer("conv1_relu").output,
          baseModel.get_layer("conv2_block3_out").output,
          baseModel.get_layer("conv3_block4_out").output,
          baseModel.get_layer("conv4_block6_out").output,
        ]
      except ValueError:
        # Fallback for different Keras versions where layer names vary.
        level_outputs = [
          baseModel.get_layer("pool1").output,
          baseModel.get_layer("conv2_block3_out").output,
          baseModel.get_layer("conv3_block4_out").output,
          baseModel.get_layer("conv4_block6_out").output,
        ]
      # Construct a model that outputs the selected intermediate feature maps.
      self.resnet_base = tf.keras.models.Model(inputs=baseInput, outputs=level_outputs)

    # Execute the ResNet base to obtain encoder levels.
    return self.resnet_base(x, training=training)

  def _run_mobilenet_encoder(self, x, training=False):
    '''Execute MobileNetV2 encoder and return 4 feature levels.'''

    # Lazily build MobileNetV2 base the first time this path is executed.
    if (self.mb_base is None):
      baseInput = tf.keras.layers.Input(tensor=x)
      baseModel = tf.keras.applications.MobileNetV2(
        include_top=False, weights=None, input_tensor=baseInput, alpha=1.0
      )
      layer_names = [
        "block_1_expand_relu", "block_3_expand_relu",
        "block_6_expand_relu", "block_13_expand_relu"
      ]
      try:
        level_outputs = [baseModel.get_layer(name).output for name in layer_names]
      except ValueError:
        # Fallback for alternative layer naming conventions.
        level_outputs = []
        for layer in baseModel.layers:
          if ("expand_relu" in layer.name and len(level_outputs) < 4):
            level_outputs.append(layer.output)
      # Construct a model that returns the chosen intermediate layers.
      self.mb_base = tf.keras.models.Model(inputs=baseInput, outputs=level_outputs)

    # Execute MobileNet base to obtain encoder levels.
    return self.mb_base(x, training=training)

  def _run_encoder(self, x, training=False):
    '''Route to appropriate encoder implementation.'''

    # Route to the appropriate encoder runtime method based on selected encoder.
    if (self.encoderType == "VGG16"):
      return self._run_vgg_encoder(x, training)
    elif (self.encoderType == "Vanilla"):
      return self._run_vanilla_encoder(x, training)
    elif (self.encoderType == "ResNet50"):
      return self._run_resnet_encoder(x, training)
    elif (self.encoderType == "MobileNet"):
      return self._run_mobilenet_encoder(x, training)
    # Return empty list if encoder type is not recognized (should not happen).
    return []

  def _run_decoder(self, x, training=False):
    '''Execute decoder pathway from selected level.'''

    # Execute the decoder stages sequentially.
    for i in range(5):  # 5 decoder stages.
      suffix = f"_d{i}"
      # Apply padding, convolution, batchnorm and activation for this stage.
      x = self.dec_layers[f"pad{suffix}"](x)
      x = self.dec_layers[f"conv{suffix}"](x)
      x = self.dec_layers[f"bn{suffix}"](x, training=training)
      x = self.dec_layers[f"relu{suffix}"](x)

      # Only upsample for the configured initial `level` stages.
      if (i < self.level and f"up{suffix}" in self.dec_layers):
        x = self.dec_layers[f"up{suffix}"](x)
    # Return the decoded feature map.
    return x

  def _build_functional_model(self):
    '''Build static functional model for fixed input shapes (optimization).'''

    # Create a Keras Input for the configured fixed input size.
    inputs = tf.keras.layers.Input(shape=self.inputSize, name="segnet_input")
    # Run the encoder to obtain feature levels.
    levels = self._run_encoder(inputs, training=False)
    # Select the level where the decoder starts (1-indexed -> 0-indexed conversion).
    x = levels[self.level - 1]
    # Run the decoder from the selected level.
    x = self._run_decoder(x, training=False)
    # Apply output convolution and activation to obtain final logits/probabilities.
    x = self.out_conv(x)
    outputs = self.out_act(x)
    # Construct and store the internal functional Keras model for fixed-shape execution.
    self._functional_model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=self.name)

  def call(self, x, training=False):
    '''Forward pass: route to functional model if fixed shape, else dynamic.'''

    # If a fixed-shape functional model exists, route inputs there for execution.
    if (self._is_fixed_shape and self._functional_model is not None):
      return self._functional_model(x, training=training)

    # Dynamic shape execution path: run encoder, decoder and output head directly.
    levels = self._run_encoder(x, training=training)
    x = levels[self.level - 1]
    x = self._run_decoder(x, training=training)
    x = self.out_conv(x)
    # Return the activated output tensor.
    return self.out_act(x)

  def build(self, input_shape=None):
    '''Explicitly build model weights (useful for dynamic shapes).'''

    # If no input shape was provided use a dynamic 4D shape with known channels.
    if (input_shape is None):
      input_shape = (None, None, None, self.inputChannels)
    # Call the base class build implementation.
    super().build(input_shape)
    # Trigger one forward pass with zeros to ensure weights are created.
    _ = self.call(tf.zeros((1,) + input_shape[1:]), training=False)

  def get_config(self):
    '''Enable model serialization.'''

    # Serialize configuration for model reproduction.
    config = {
      "inputChannels": self.inputChannels,
      "numClasses"   : self.numClasses,
      "level"        : self.level,
      "encoder"      : self.encoderType,
      "inputSize"    : self.inputSize if (self._is_fixed_shape) else None,
      "useBias"      : self.useBias,
    }
    base_config = super().get_config()
    return {**base_config, **config}

  @classmethod
  def from_config(cls, config):
    '''Enable model deserialization.'''

    return cls(**config)


def CreateUNet(
  inputChannels: int = 3, numClasses: int = 2, baseChannels: int = 64, depth: int = 4,
  upMode: str = "transpose", norm: str = "batch", dropout: float = 0.0, residual: bool = False,
  modelType: str = "dynamic"
):
  r'''
  Extended factory that supports the UNet family and related variants.

  Short summary:
    Returns an instantiated tf.keras.Model implementing the requested UNet variant
    selected by the case-insensitive `modelType` string. This factory exposes the
    common constructor arguments used by the various implementations and forwards
    them to the selected model. Models in this file follow TensorFlow's NHWC
    convention (batch, height, width, channels).

  Parameters:
    inputChannels (int): Number of input image channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base filter count used by most architectures. Default 64.
    depth (int): Depth / number of downsampling stages for dynamic variants. Default 4.
    upMode (str): "transpose" or "bilinear" upsampling selection. Default "transpose".
    norm (str): Normalization mode for configurable blocks: "batch" | "instance" | "none". Default "batch".
    dropout (float): Dropout probability passed to configurable blocks. Default 0.0.
    residual (bool): Whether to enable residual connections where supported. Default False.
    modelType (str): Case-insensitive model selection string (e.g. "dynamic", "UNet", "ASPPUNet").

  Returns:
    tensorflow.keras.Model: Instantiated model ready for training or inference. The returned
      model accepts NHWC inputs and produces logits in NHWC format (shape [B, H, W, C]).

  Notes:
    - If an unrecognized `modelType` is provided the function falls back to the
      `DynamicUNet` configuration. Use the names exposed in AVAILABLE_UNETS or
      the CamelCase class names when calling `GetUNetModel`.
  '''
  mt = (modelType or "dynamic")
  mtLower = mt.lower()

  if (mtLower in ("original", "legacy")):
    return UNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "multiresunet"):
    return MultiResUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "denseunet"):
    return DenseUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=max(16, baseChannels // 2),
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "r2unet"):
    return R2UNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "asppunet"):
    return ASPPUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "transunet"):
    return TransUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "cbamunet"):
    return CBAMUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "efficientunet"):
    return EfficientUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=max(16, baseChannels // 2),
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "dynamic"):
    return DynamicUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels, depth=depth,
      upMode=upMode, norm=norm, dropout=dropout, residual=residual
    )

  if (mtLower == "residual"):
    return ResidualUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "attention"):
    return AttentionUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "mobile"):
    return MobileUNet(
      inputChannels=inputChannels, numClasses=numClasses,
      baseChannels=(baseChannels if baseChannels >= 32 else 32),
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "se"):
    return SEUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "residual_attention"):
    return ResidualAttentionUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "boundary_aware"):
    return BoundaryAwareUNet(
      inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  if (mtLower == "vnet"):
    # VNet is implemented as a functional builder that returns a tf.keras.Model using NHWC input ordering.
    return VNet(
      inputSize=(None, None, inputChannels),
      numClasses=numClasses,
      kernelInitializer="he_normal",
      dropoutRatio=dropout,
      dropoutType="spatial",
      activation="relu",
      applyBatchNorm=(norm == "batch"),
      concatenateType="concatenate",
      noOfLevels=(depth if depth >= 2 else 5),
    )

  if (mtLower == "segnet"):
    segLevel = max(1, min(depth, 4))
    return SegNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      level=segLevel,
      encoder="VGG16",
      inputSize=(None, None, inputChannels)
    )

  # Default fallback.
  return DynamicUNet(
    inputChannels=inputChannels, numClasses=numClasses, baseChannels=baseChannels, depth=depth,
    upMode=upMode, norm=norm, dropout=dropout, residual=residual
  )


def GetUNetModel(modelName: str, inputChannels: int, numClasses: int, baseChannels: int = 64):
  r'''
  Factory helper to instantiate a specific UNet variant by its CamelCase class name.

  Short summary:
    Convenience function that maps a model class name (exact CamelCase string)
    to the corresponding tf.keras.Model constructor in this module and returns
    an instantiated model. This is useful when model names are provided from
    configuration files or command-line arguments.

  Parameters:
    modelName (str): Exact CamelCase name of the desired model class (e.g. "UNet", "TransUNet").
    inputChannels (int): Number of input channels to construct the model with.
    numClasses (int): Number of output classes for the model.
    baseChannels (int): Base channel count used by many model constructors. Default 64.

  Returns:
    tf.keras.Model: Instantiated model corresponding to `modelName`.

  Raises:
    ValueError: If `modelName` is not recognized among the known UNet class names in this module.
  '''

  if (modelName == "ResidualAttentionUNet"):
    return ResidualAttentionUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "TransUNet"):
    return TransUNet(inputChannels, numClasses)
  elif (modelName == "UNet"):
    return UNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "CBAMUNet"):
    return CBAMUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "EfficientUNet"):
    return EfficientUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "BoundaryAwareUNet"):
    return BoundaryAwareUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "DynamicUNet"):
    return DynamicUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "MultiResUNet"):
    return MultiResUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "DenseUNet"):
    return DenseUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "R2UNet"):
    return R2UNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "ASPPUNet"):
    return ASPPUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "MobileUNet"):
    return MobileUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "SEUNet"):
    return SEUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "ResidualUNet"):
    return ResidualUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "AttentionUNet"):
    return AttentionUNet(inputChannels, numClasses, baseChannels)
  elif (modelName == "VNet"):
    # Build a VNet with dynamic spatial dimensions using inputChannels
    return VNet(inputSize=(None, None, inputChannels), numClasses=numClasses)
  elif (modelName == "SegNet"):
    return SegNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      level=3,
      encoder="VGG16",
      inputSize=(None, None, inputChannels)
    )

  raise ValueError(f"Unknown model name: {modelName}")


if __name__ == "__main__":
  # Example to test all UNet variants.
  for unetType in AVAILABLE_UNETS[::-1]:
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
      # Create dummy input and test forward pass.
      dummyInput = tf.random.normal((2, imgSize, imgSize, 3))
      if (unetType == "BoundaryAwareUNet"):
        segLogits, boundaryMap = model(dummyInput, training=False)
        print(f"Output shapes for {unetType}: segmentation logits {segLogits.shape}, boundary map {boundaryMap.shape}")
      else:
        output = model(dummyInput, training=False)
        print(f"Output shape for {unetType}: {output.shape}")
