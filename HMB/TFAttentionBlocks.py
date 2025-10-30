import math
import tensorflow as tf
from tensorflow.keras.layers import (
  Layer, Dense, Conv2D, Conv1D, GlobalAveragePooling2D,
  GlobalMaxPooling2D, Activation, Add, Multiply, Softmax,
  Reshape, Permute, Concatenate, DepthwiseConv2D, MultiHeadAttention
)


# Define CBAM attention block class.
class CBAMBlock(Layer):
  r'''
  Convolutional Block Attention Module (CBAM) implementation as a Keras Layer.
  This layer applies channel and spatial attention mechanisms to refine feature maps.
  
  Parameters:
    ratio (int): Reduction ratio for channel attention MLP. Default is 8.
    kernelSize (int): Kernel size for spatial attention convolution. Default is 7.
    
  Examples
  --------
  .. code-block:: python
  
  from HMB.TFAttentionBlocks import CBAMBlock
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, Input
  
  model = Sequential([
    Input(shape=(64, 64, 128)),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    CBAMBlock(ratio=8, kernelSize=7),
    Conv2D(256, (3, 3), padding="same", activation="relu")
  ])
  model.summary()
  '''

  # Store initialization args for later serialization.
  def __init__(self, ratio=8, kernelSize=7, **kwargs):
    r'''
    Initialize CBAM block with given parameters.

    Parameters:
      ratio (int): Reduction ratio for channel attention MLP. Default is 8.
      kernelSize (int): Kernel size for spatial attention convolution. Default is 7.
    '''

    # Call parent initializer.
    super(CBAMBlock, self).__init__(**kwargs)
    # Set channel reduction ratio.
    self.ratio = ratio
    # Set kernel size for spatial attention.
    self.kernelSize = kernelSize

    # Build layer components when input shape is known.

  def build(self, inputShape):
    r'''
    Build the CBAM block components based on input shape.
    This method initializes the shared MLP layers for channel attention
    and the convolutional layer for spatial attention.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # inputShape is (batch, H, W, C).
    # Determine number of channels.
    channel = int(inputShape[-1])
    # Compute bottleneck units.
    hiddenUnits = max(1, channel // self.ratio)
    self.sharedLayerOne = Dense(
      hiddenUnits,  # Reduced dimension.
      activation="relu",  # ReLU activation.
      kernel_initializer="he_normal",  # He normal initialization.
      use_bias=True,  # Use bias in dense layer.
    )

    self.sharedLayerTwo = Dense(
      channel,  # Restore original channel dimension.
      kernel_initializer="he_normal",  # He normal initialization.
      use_bias=True,  # Use bias in dense layer.
    )

    self.conv = Conv2D(
      1, (self.kernelSize, self.kernelSize),
      padding="same",  # Ensure output size matches input size.
      activation="sigmoid",  # Sigmoid for attention map.
      kernel_initializer="he_normal",  # He normal initialization.
    )

    # Create pooling layers.
    self.gap = GlobalAveragePooling2D()
    self.gmp = GlobalMaxPooling2D()

    super(CBAMBlock, self).build(inputShape)  # Finish build.

  # Forward pass applying channel and spatial attention.
  def call(self, inputs):
    # Channel attention.
    # Global average pooling across spatial dims.
    avgPool = self.gap(inputs)
    # Pass through shared MLP layer 1.
    avgPool = self.sharedLayerOne(avgPool)
    # Pass through shared MLP layer 2.
    avgPool = self.sharedLayerTwo(avgPool)

    # Global max pooling across spatial dims.
    maxPool = self.gmp(inputs)
    # Shared MLP layer 1 on max-pooled features.
    maxPool = self.sharedLayerOne(maxPool)
    # Shared MLP layer 2 on max-pooled features.
    maxPool = self.sharedLayerTwo(maxPool)

    # Sum channel attention contributions.
    channelAtt = Add()([avgPool, maxPool])
    # Sigmoid activation for channel weights.
    channelAtt = Activation("sigmoid")(channelAtt)
    # Reshape to (b,1,1,C).
    channelAtt = tf.expand_dims(tf.expand_dims(channelAtt, axis=1), axis=1)
    # Apply channel attention.
    x = Multiply()([inputs, channelAtt])

    # Spatial attention.
    # Average across channels for spatial attention.
    avgPoolSpatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    # Max across channels for spatial attention.
    maxPoolSpatial = tf.reduce_max(x, axis=-1, keepdims=True)
    # Concatenate spatial descriptors.
    concat = tf.concat([avgPoolSpatial, maxPoolSpatial], axis=-1)
    # Convolution to produce spatial attention map.
    spatialAtt = self.conv(concat)
    # Apply spatial attention.
    out = Multiply()([x, spatialAtt])

    return out  # Return refined features.

  # Make layer serializable by returning initialization args.
  def get_config(self):
    # Get base config from parent.
    config = super(CBAMBlock, self).get_config()
    config.update({
      "ratio"     : self.ratio,  # Include ratio in config.
      "kernelSize": self.kernelSize  # Include kernelSize in config.
    })
    return config  # Return the updated config.

  # Return the output shape unchanged.
  def compute_output_shape(self, inputShape):
    return inputShape  # Output shape equals input shape.


class SEBlock(Layer):
  r'''
  Squeeze-and-Excitation (SE) block implementation as a Keras Layer.
  This layer adaptively recalibrates channel-wise feature responses by
  explicitly modeling interdependencies between channels.

  Parameters:
    ratio (int): Reduction ratio for the bottleneck in the SE block. Default is 16.

  Examples
  --------
  .. code-block:: python

  from HMB.TFAttentionBlocks import SEBlock
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, Input

  model = Sequential([
    Input(shape=(64, 64, 128)),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    SEBlock(ratio=16),
    Conv2D(256, (3, 3), padding="same", activation="relu")
  ])
  model.summary()
  '''

  # Store initialization args for later serialization.
  def __init__(self, ratio=16, **kwargs):
    r'''
    Initialize SE block with given parameters.

    Parameters:
      ratio (int): Reduction ratio for the bottleneck in the SE block. Default is 16.
    '''

    # Call parent initializer.
    super(SEBlock, self).__init__(**kwargs)
    # Set reduction ratio.
    self.ratio = ratio

  # Build layer components when input shape is known.
  def build(self, inputShape):
    r'''
    Build the SE block components based on input shape.
    This method initializes the dense layers for the squeeze
    and excitation operations.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # inputShape is (batch, H, W, C).
    # Determine number of channels.
    channel = int(inputShape[-1])
    # Compute bottleneck units.
    hiddenUnits = max(1, channel // self.ratio)
    self.denseOne = Dense(
      hiddenUnits,  # Reduced dimension.
      activation="relu",  # ReLU activation.
      kernel_initializer="he_normal",  # He normal initialization.
      use_bias=True,  # Use bias in dense layer.
    )
    self.denseTwo = Dense(
      channel,  # Restore original channel dimension.
      activation="sigmoid",  # Sigmoid for attention weights.
      kernel_initializer="he_normal",  # He normal initialization.
      use_bias=True,  # Use bias in dense layer.
    )
    # Global average pooling layer.
    self.gap = GlobalAveragePooling2D()
    # Finish build.
    super(SEBlock, self).build(inputShape)

  # Forward pass applying squeeze and excitation.
  def call(self, inputs):
    # Squeeze: Global average pooling across spatial dims.
    squeeze = self.gap(inputs)
    # Excitation: First dense layer.
    excitation = self.denseOne(squeeze)
    # Excitation: Second dense layer.
    excitation = self.denseTwo(excitation)
    # Reshape to (b,1,1,C).
    excitation = tf.expand_dims(tf.expand_dims(excitation, axis=1), axis=1)
    # Scale input features by channel-wise weights.
    out = Multiply()([inputs, excitation])

    # Return recalibrated features.
    return out

  # Make layer serializable by returning initialization args.
  def get_config(self):
    # Get base config from parent.
    config = super(SEBlock, self).get_config()
    config.update({
      "ratio": self.ratio  # Include ratio in config.
    })
    # Return the updated config.
    return config

  # Return the output shape unchanged.
  def compute_output_shape(self, inputShape):
    # Output shape equals input shape.
    return inputShape


class ECABlock(Layer):
  r'''
  Efficient Channel Attention (ECA) block implementation as a Keras Layer.
  This layer captures local cross-channel interactions without dimensionality reduction.

  Parameters:
    gamma (int): Parameter to compute kernel size. Default is 2.
    b (int): Parameter to compute kernel size. Default is 1.
  '''

  # Store initialization args for later serialization.
  def __init__(self, gamma=2, b=1, **kwargs):
    r'''
    Initialize ECA block with given parameters.

    Parameters:
      gamma (int): Parameter to compute kernel size. Default is 2.
      b (int): Parameter to compute kernel size. Default is 1.
    '''

    # Call parent initializer.
    super(ECABlock, self).__init__(**kwargs)
    # Set gamma parameter.
    self.gamma = gamma
    # Set b parameter.
    self.b = b

  # Build layer components when input shape is known.
  def build(self, inputShape):
    r'''
    Build the ECA block components based on input shape.
    This method initializes the convolutional layer for channel attention.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # inputShape is (batch, H, W, C).
    # Determine number of channels.
    channel = int(inputShape[-1])
    # Compute kernel size using formula.
    t = int(abs((math.log(channel, 2) + self.b) / self.gamma))
    kSize = t if ((t % 2) == 1) else t + 1  # Ensure kernel size is odd.
    # Use Conv1D acting over channel dimension: after global-pool we get
    # (b, C) -> expanded to (b, C, 1) where C is the temporal dimension
    # for Conv1D. This implements the ECA 1D conv over channels cleanly.
    self.conv = Conv1D(
      1, kernel_size=kSize,
      padding="same",
      activation="sigmoid",
      kernel_initializer="he_normal",
      use_bias=False,
    )
    # Global average pooling layer.
    self.gap = GlobalAveragePooling2D()
    super(ECABlock, self).build(inputShape)  # Finish build.

  # Forward pass applying efficient channel attention.
  def call(self, inputs):
    # Global average pooling across spatial dims.
    squeeze = self.gap(inputs)
    # Reshape to (b, C, 1) so Conv1D can operate over the channel dimension
    # (treated as time/steps). Conv1D returns (b, C, 1).
    squeeze = tf.expand_dims(squeeze, axis=-1)  # (b, C, 1).
    excitation = self.conv(squeeze)  # (b, C, 1)
    # transpose to (b, 1, C) then expand to (b,1,1,C) to match inputs for
    # broadcasting when scaling.
    excitation = tf.transpose(excitation, perm=[0, 2, 1])  # (b,1,C)
    excitation = tf.expand_dims(excitation, axis=1)  # (b,1,1,C)
    # Scale input features by channel-wise weights.
    out = Multiply()([inputs, excitation])

    # Return recalibrated features.
    return out

  # Make layer serializable by returning initialization args.
  def get_config(self):
    # Get base config from parent.
    config = super(ECABlock, self).get_config()
    config.update({
      # Include gamma in config.
      "gamma": self.gamma,
      # Include b in config.
      "b"    : self.b
    })
    # Return the updated config.
    return config

  # Return the output shape unchanged.
  def compute_output_shape(self, inputShape):
    # Output shape equals input shape.
    return inputShape


class MultiHeadSelfAttention(Layer):
  r'''
  Multi-head self-attention adapted for 2D feature maps.

  This layer flattens spatial dimensions (H*W) and applies a Transformer-style
  multi-head self-attention on the flattened sequence, then projects the
  attended features back to the original spatial shape.

  Parameters:
    numHeads (int): Number of attention heads. Default is 8.
    keyDim (int): Dimensionality of each head's key/query vectors. Default is 64.
  '''

  def __init__(self, numHeads=8, keyDim=64, **kwargs):
    r'''
    Initialize MultiHeadSelfAttention.

    Parameters:
      numHeads (int): Number of attention heads.
      keyDim (int): Dimension of each head.
    '''

    super(MultiHeadSelfAttention, self).__init__(**kwargs)
    # Use camelCase for internal variable names as requested.
    self.numHeads = numHeads
    self.keyDim = keyDim
    # create underlying keras MultiHeadAttention here so it's available
    # before build/call (avoids NoneType callable issue in some call paths)
    self.mha = MultiHeadAttention(num_heads=self.numHeads, key_dim=self.keyDim)

  def build(self, inputShape):
    r'''
    Build an internal tf.keras.layers.MultiHeadAttention instance. Input is
    expected to be (batch, H, W, C).

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # Nothing else to build (self.mha created in __init__), keep for API.
    super(MultiHeadSelfAttention, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: reshape (b,H,W,C) -> (b,seq_len,C), call the underlying
    MultiHeadAttention with query=value=key and reshape the output back.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    b = tf.shape(inputs)[0]
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    c = tf.shape(inputs)[3]
    seqLen = h * w
    # Flatten spatial dims to sequence.
    x = tf.reshape(inputs, (b, seqLen, c))
    # Use the underlying multi-head attention; returns (b, seq_len, c)
    out = self.mha(query=x, value=x, key=x)
    out = tf.reshape(out, (b, h, w, c))
    return out

  def get_config(self):
    cfg = super(MultiHeadSelfAttention, self).get_config()
    cfg.update({"numHeads": self.numHeads, "keyDim": self.keyDim})
    return cfg


class NonLocalBlock(Layer):
  r'''
  Non-local block (self-attention over feature map positions).

  Computes pairwise affinities across spatial positions and aggregates global
  context. Useful to capture long-range dependencies in images.

  Parameters:
    interChannels (int|None): Number of channels for the internal projections; if None it defaults to C//2 where C is input channels.
  '''

  def __init__(self, interChannels=None, **kwargs):
    r'''
    Initialize NonLocalBlock.

    Parameters:
      interChannels (int|None): Internal projection channels.
    '''

    super(NonLocalBlock, self).__init__(**kwargs)
    self.interChannels = interChannels

  def build(self, inputShape):
    r'''
    Build 1x1 conv projections for theta, phi and g, and an output 1x1 conv.
    inputShape is expected as (batch, H, W, C).
    
    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    _, h, w, c = inputShape
    if self.interChannels is None:
      self.interChannels = max(1, c // 2)
    # Linear projections implemented as 1x1 convs.
    self.theta = Conv2D(self.interChannels, 1, padding="same", use_bias=False)
    self.phi = Conv2D(self.interChannels, 1, padding="same", use_bias=False)
    self.g = Conv2D(self.interChannels, 1, padding="same", use_bias=False)
    self.outConv = Conv2D(c, 1, padding="same", use_bias=False)
    super(NonLocalBlock, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: compute affinity matrix between positions and aggregate
    global context which is then projected and added to the input (residual).
    
    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    b = tf.shape(inputs)[0]
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    c = tf.shape(inputs)[3]

    thetaX = self.theta(inputs)
    phiX = self.phi(inputs)
    gX = self.g(inputs)

    # flatten spatial dims: (b, N, ic).
    thetaFlat = tf.reshape(thetaX, (b, -1, self.interChannels))
    phiFlat = tf.reshape(phiX, (b, -1, self.interChannels))
    gFlat = tf.reshape(gX, (b, -1, self.interChannels))

    logits = tf.matmul(thetaFlat, phiFlat, transpose_b=True)
    attn = tf.nn.softmax(logits, axis=-1)
    out = tf.matmul(attn, gFlat)
    out = tf.reshape(out, (b, h, w, self.interChannels))
    out = self.outConv(out)
    # residual connection.
    return inputs + out

  def get_config(self):
    cfg = super(NonLocalBlock, self).get_config()
    cfg.update({"interChannels": self.interChannels})
    return cfg


class BAM(Layer):
  r'''
  Bottleneck Attention Module (BAM) - lightweight channel + spatial attention.

  This implementation builds a compact channel attention branch (MLP on pooled
  features) and a spatial branch (dilated conv stack) and multiplies them with
  the input feature map.

  Parameters:
    reduction (int): Channel reduction ratio for the channel branch. Default 16.
    dilationRates (tuple): Dilation rates for the spatial conv stack. Default (1,2,4).
  '''

  def __init__(self, reduction=16, dilationRates=(1, 2, 4), **kwargs):
    r'''
    Initialize BAM.

    Parameters:
      reduction (int): Reduction ratio for channel MLP.
      dilationRates (tuple): Dilation rates for spatial convolutions.
    '''

    super(BAM, self).__init__(**kwargs)
    self.reduction = reduction
    self.dilationRates = dilationRates

  def build(self, inputShape):
    r'''
    Build channel MLP and spatial dilated conv stack based on input channels.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    c = int(inputShape[-1])
    hidden = max(1, c // self.reduction)
    # channel branch: small MLP.
    self.channelFc1 = Dense(hidden, activation="relu")
    self.channelFc2 = Dense(c, activation=None)
    # spatial branch: stack of dilated convs.
    self.spatialConvs = [
      Conv2D(1, 3, padding="same", dilation_rate=d, activation=None)
      for d in self.dilationRates
    ]
    self.sigmoid = Activation("sigmoid")
    self.globalAvgPool = GlobalAveragePooling2D()
    super(BAM, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: compute channel attention via pooled MLP and spatial
    attention via dilated convs, then apply both to input.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    # channel attention: global pooling -> MLP -> sigmoid -> reshape.
    ch = self.globalAvgPool(inputs)
    ch = self.channelFc1(ch)
    ch = self.channelFc2(ch)
    ch = self.sigmoid(ch)
    ch = tf.expand_dims(tf.expand_dims(ch, axis=1), axis=1)

    # spatial attention: pass input through dilated conv stack.
    x = inputs
    for conv in self.spatialConvs:
      x = conv(x)
    spat = self.sigmoid(x)

    attn = Multiply()([inputs, ch])
    attn = Multiply()([attn, spat])
    return attn

  def get_config(self):
    cfg = super(BAM, self).get_config()
    cfg.update({"reduction": self.reduction, "dilationRates": self.dilationRates})
    return cfg


class GCBlock(Layer):
  r'''
  Global Context (GC) block - simplified global attention for feature maps.

  Computes a soft-attention mask over spatial positions, aggregates a global
  context vector and applies a small transform to produce a context tensor that
  is added back to the input (residual).

  Parameters:
    reduction (int): Channel reduction for the transform MLP. Default 16.
  '''

  def __init__(self, reduction=16, **kwargs):
    r'''
    Initialize GCBlock.

    Parameters:
      reduction (int): Reduction factor for the internal transform.
    '''

    super(GCBlock, self).__init__(**kwargs)
    self.reduction = reduction

  def build(self, inputShape):
    r'''
    Build conv_mask and transform dense layers based on input channels.
    '''
    c = int(inputShape[-1])
    self.convMask = Conv2D(1, 1, padding="same")
    self.softmax = Softmax(axis=1)  # later applied over spatial positions.
    self.transform = Dense(max(1, c // self.reduction), activation="relu")
    self.transformOut = Dense(c, activation=None)
    super(GCBlock, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: compute attention mask, aggregate weighted context and apply
    small transform then broadcast-add to the input.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    b = tf.shape(inputs)[0]
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    c = tf.shape(inputs)[3]

    inputFlat = tf.reshape(inputs, (b, -1, c))  # (b, N, c).
    mask = self.convMask(inputs)  # (b,h,w,1).
    maskFlat = tf.reshape(mask, (b, -1, 1))  # (b,N,1).
    maskFlat = self.softmax(maskFlat)  # softmax over N.

    # context: weighted sum over positions -> (b,1,c) -> squeeze to (b,c).
    context = tf.matmul(maskFlat, inputFlat, transpose_a=True)
    context = tf.squeeze(context, axis=1)
    contextTrans = self.transform(context)
    contextTrans = self.transformOut(contextTrans)
    contextTrans = tf.expand_dims(tf.expand_dims(contextTrans, axis=1), axis=1)
    return inputs + contextTrans

  def get_config(self):
    cfg = super(GCBlock, self).get_config()
    cfg.update({"reduction": self.reduction})
    return cfg


class AxialAttention(Layer):
  r'''
  Axial attention: factorized attention applied along height then width axes.
  Applies a lightweight multi-head self-attention along one axis at a time to
  reduce computational cost compared to full 2D self-attention.

  Parameters:
    numHeads (int): Number of attention heads. Default 4.
    keyDim (int): Per-head dimensionality. Default 32.
  '''

  def __init__(self, numHeads=4, keyDim=32, **kwargs):
    r'''
    Initialize AxialAttention.

    Parameters:
      numHeads (int): Number of heads.
      keyDim (int): Dimension per head.
    '''

    super(AxialAttention, self).__init__(**kwargs)
    self.numHeads = numHeads
    self.keyDim = keyDim

  def build(self, inputShape):
    r'''
    Instantiate two MultiHeadSelfAttention modules (one for each axis).

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # use the correct camelCase keyword expected by MultiHeadSelfAttention
    self.mhsaH = MultiHeadSelfAttention(numHeads=self.numHeads, keyDim=self.keyDim)
    self.mhsaW = MultiHeadSelfAttention(numHeads=self.numHeads, keyDim=self.keyDim)
    super(AxialAttention, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: apply attention along height and width axes and sum results.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    # height-wise attention: transpose to bring height into the "sequence" axis.
    xH = tf.transpose(inputs, perm=[0, 2, 1, 3])  # (b, W, H, C).
    xH = self.mhsaH.call(xH)
    xH = tf.transpose(xH, perm=[0, 2, 1, 3])
    # width-wise attention (operate on original layout).
    xW = self.mhsaW.call(inputs)
    return xH + xW

  def get_config(self):
    cfg = super(AxialAttention, self).get_config()
    cfg.update({"numHeads": self.numHeads, "keyDim": self.keyDim})
    return cfg


class AttentionAugmentedConv(Layer):
  r'''
  Attention-augmented convolutional layer.

  Combines a standard convolutional branch with a self-attention branch and
  concatenates their outputs along the channel axis.

  Parameters:
    filters (int): Total output filters (conv + attention channels combined).
    kernelSize (int): Convolution kernel size for the conv branch.
    numHeads (int): Number of attention heads for the attention branch.
    keyDim (int): Per-head dimension for attention branch.
  '''

  def __init__(self, filters, kernelSize=3, numHeads=4, keyDim=32, **kwargs):
    r'''
    Initialize AttentionAugmentedConv.

    Parameters:
      filters (int): Total filters after concatenation.
      kernelSize (int): Kernel size for conv branch.
      numHeads (int): Number of attention heads.
      keyDim (int): Per-head dimension.
    '''

    super(AttentionAugmentedConv, self).__init__(**kwargs)
    self.filters = filters
    self.kernelSize = kernelSize
    self.numHeads = numHeads
    self.keyDim = keyDim

  def build(self, inputShape):
    r'''
    Build conv branch and attention branch. Note: user should ensure
    filters > numHeads * keyDim so the conv branch has a positive number of channels.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # conv branch output channels is total filters minus attention channels.
    self.conv = Conv2D(self.filters - self.keyDim * self.numHeads, self.kernelSize, padding="same")
    # instantiate with correct kwarg name
    self.mhsa = MultiHeadSelfAttention(numHeads=self.numHeads, keyDim=self.keyDim)
    self.concat = Concatenate(axis=-1)
    super(AttentionAugmentedConv, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: run conv branch and attention branch then concatenate.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    convOut = self.conv(inputs)
    attnOut = self.mhsa.call(inputs)
    out = self.concat([convOut, attnOut])
    return out

  def get_config(self):
    cfg = super(AttentionAugmentedConv, self).get_config()
    cfg.update(
      {
        "filters"   : self.filters,
        "kernelSize": self.kernelSize,
        "numHeads"  : self.numHeads,
        "keyDim"    : self.keyDim,
      }
    )
    return cfg


# python
class SKBlock(Layer):
  r'''
  Selective Kernel (SK) block: adaptively fuse multiple convolutional branches.

  Parameters:
    filters (int): Number of output channels for each branch (usually equals input channels).
    M (int): Number of branches (kernel choices). Default 2.
    G (int): Grouping parameter (kept for API compatibility).
    r (int): Reduction ratio for the selection MLP. Default 16.
  '''

  def __init__(self, filters, M=2, G=1, r=16, **kwargs):
    r'''
    Initialize SKBlock.

    Parameters:
      filters (int): Number of output channels per branch.
      M (int): Number of branches.
      G (int): Grouping parameter.
      r (int): Reduction ratio for selection MLP.
    '''
    super(SKBlock, self).__init__(**kwargs)
    self.filters = filters
    self.M = M
    self.G = G
    self.r = r

  def build(self, inputShape):
    r'''
    Build convolutional branches and selection MLP.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    # Build convolutional branches.
    self.branches = []
    for m in range(self.M):
      kernelSize = 3 + 2 * m
      conv = Conv2D(self.filters, kernelSize, padding="same", activation="relu")
      self.branches.append(conv)

    # Selection MLP.
    self.globalPool = GlobalAveragePooling2D()
    hiddenUnits = max(1, self.filters // self.r)
    self.fc1 = Dense(hiddenUnits, activation="relu")
    self.fcs = [Dense(self.filters, activation=None) for _ in range(self.M)]
    # Softmax across branches (axis=1 corresponds to M after stacking).
    self.softmax = Softmax(axis=1)

    super(SKBlock, self).build(inputShape)

  def call(self, inputs):
    r'''
    Forward pass: compute branch outputs, aggregate and apply attention-based fusion.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    # Compute each branch output -> list of tensors (b,H,W,C).
    branchOuts = [conv(inputs) for conv in self.branches]

    # Stack branch outputs -> (b, H, W, C, M).
    stacked = tf.stack(branchOuts, axis=-1)

    # Aggregate across branches: U shape (b,H,W,C).
    U = tf.reduce_sum(stacked, axis=-1)

    # Channel descriptor: (b, C).
    s = self.globalPool(U)

    # Shared FC -> (b, hidden).
    z = self.fc1(s)

    # Per-branch logits -> list of (b, C).
    logitsList = [fc(z) for fc in self.fcs]

    # Stack logits -> (b, M, C).
    logits = tf.stack(logitsList, axis=1)

    # Softmax over branches -> (b, M, C).
    attn = self.softmax(logits)

    # Reorder to (b, 1, 1, C, M) to broadcast with stacked (b,H,W,C,M).
    attn = tf.transpose(attn, perm=[0, 2, 1])  # (b, C, M).
    attn = tf.expand_dims(tf.expand_dims(attn, axis=1), axis=1)  # (b,1,1,C,M).

    # Weighted sum over branches -> (b,H,W,C).
    V = tf.reduce_sum(stacked * attn, axis=-1)

    return V

  def get_config(self):
    cfg = super(SKBlock, self).get_config()
    cfg.update({"filters": self.filters, "M": self.M, "G": self.G, "r": self.r})
    return cfg

  def compute_output_shape(self, inputShape):
    return inputShape


class TripletAttention(Layer):
  r'''
  Triplet attention: cross-dimension spatial attention using rotated inputs.
  Applies spatial attention on three views (original, HW-transposed, and original
  again — the current implementation applies the original view twice) and averages
  the three attended outputs.

  Parameters:
    kernelSize (int): Kernel size for the spatial conv used to compute attention maps.
  '''

  def __init__(self, kernelSize=7, **kwargs):
    r'''
    Initialize TripletAttention.

    Parameters:
      kernelSize (int): Kernel size for attention conv.
    '''

    super(TripletAttention, self).__init__(**kwargs)
    self.kernelSize = kernelSize

  def build(self, inputShape):
    r'''
    Build the small conv used to compute spatial attention maps.

    Parameters:
      inputShape (tuple): Shape of the input tensor.
    '''

    self.conv = Conv2D(1, self.kernelSize, padding="same", activation="sigmoid")
    super(TripletAttention, self).build(inputShape)

  def _spatial_att(self, x):
    r'''
    Compute a spatial attention map from per-pixel descriptors
    (avg and max across channels) and apply it to the input tensor.

    Parameters:
      x (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx = tf.reduce_max(x, axis=-1, keepdims=True)
    cat = tf.concat([avg, mx], axis=-1)
    att = self.conv(cat)
    return x * att

  def call(self, inputs):
    r'''
    Forward pass: apply spatial attention to original and transposed views,
    then average the results to obtain final output.

    Parameters:
      inputs (tensorflow.Tensor): Input feature map of shape (b, H, W, C).
    '''

    a = self._spatial_att(inputs)
    # HW-transposed view
    p = tf.transpose(inputs, perm=[0, 2, 1, 3])
    b = self._spatial_att(p)
    b = tf.transpose(b, perm=[0, 2, 1, 3])
    # Channel-transposed view: bring channels into the H position and apply
    # spatial attention over that view, then transpose back. This yields a
    # distinct third branch compared to the original implementation.
    q = tf.transpose(inputs, perm=[0, 3, 2, 1])
    c = self._spatial_att(q)
    c = tf.transpose(c, perm=[0, 3, 2, 1])
    out = (a + b + c) / 3.0
    return out

  def get_config(self):
    cfg = super(TripletAttention, self).get_config()
    cfg.update({"kernelSize": self.kernelSize})
    return cfg


if __name__ == "__main__":
  # Simple test case to verify layer instantiation.
  import numpy as np
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, Input

  model = Sequential([
    Input(shape=(64, 64, 128)),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    CBAMBlock(ratio=8, kernelSize=7),
    SEBlock(ratio=16),
    ECABlock(gamma=2, b=1),
    MultiHeadSelfAttention(numHeads=8, keyDim=64),
    NonLocalBlock(),
    BAM(reduction=16, dilationRates=(1, 2, 4)),
    GCBlock(reduction=16),
    AxialAttention(numHeads=4, keyDim=32),
    AttentionAugmentedConv(filters=256, kernelSize=3, numHeads=4, keyDim=32),
    SKBlock(filters=128, M=2, G=1, r=16),
    TripletAttention(kernelSize=7),
    Conv2D(256, (3, 3), padding="same", activation="relu")
  ])
  model.summary()

  # Test with random input.
  x = np.random.rand(1, 64, 64, 128).astype(np.float32)
  y = model.predict(x)
  print("Output shape:", y.shape)
