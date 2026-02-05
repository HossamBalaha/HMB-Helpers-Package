import torch
import torch.nn as nn
import torch.nn.functional as F


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
    self.need_proj = (inChannels != outChannels)
    # Optional projection to match channels.
    if (self.need_proj):
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
    if (self.need_proj):
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


class NestedUNet(nn.Module):
  r'''
  Compact U-Net++ (Nested U-Net) implementation.

  Short summary:
    Implements a practical and lightweight U-Net++ fusion pattern limited to
    four nested levels. Useful when multi-level dense skip fusion is desired.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize the nested U-Net.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call the parent initializer.
    super(NestedUNet, self).__init__()
    # Create the level-0 block.
    self.x00 = DoubleConv(inputChannels, baseChannels)
    # Create a shared pooling layer.
    self.pool = nn.MaxPool2d(2)

    # Create level-1 blocks.
    self.x10 = DoubleConv(baseChannels, baseChannels * 2)
    self.x01 = DoubleConv(baseChannels * 2, baseChannels)

    # Create level-2 blocks.
    self.x20 = DoubleConv(baseChannels * 2, baseChannels * 4)
    self.x11 = DoubleConv(baseChannels * 4, baseChannels * 2)
    self.x02 = DoubleConv(baseChannels * 4, baseChannels)

    # Create level-3 blocks.
    self.x30 = DoubleConv(baseChannels * 4, baseChannels * 8)
    self.x21 = DoubleConv(baseChannels * 8, baseChannels * 4)
    self.x12 = DoubleConv(baseChannels * 8, baseChannels * 2)
    self.x03 = DoubleConv(baseChannels * 8, baseChannels)

    # Create the center block.
    self.center = DoubleConv(baseChannels * 8, baseChannels * 16)

    # Build upsampling modules using a ModuleDict with CamelCase keys.
    if (useConvTranspose2d):
      # Use learned transposed convolutions for upsampling.
      self.up = nn.ModuleDict(
        {
          "Up1": nn.ConvTranspose2d(baseChannels * 16, baseChannels * 8, kernel_size=2, stride=2),
          "Up2": nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2),
          "Up3": nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2),
          "Up4": nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2),
        }
      )
    else:
      # Use bilinear upsampling blocks for each level.
      self.up = nn.ModuleDict(
        {
          "Up1": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(baseChannels * 16, baseChannels * 8, kernel_size=1),
          ),
          "Up2": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(baseChannels * 8, baseChannels * 4, kernel_size=1),
          ),
          "Up3": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(baseChannels * 4, baseChannels * 2, kernel_size=1),
          ),
          "Up4": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(baseChannels * 2, baseChannels, kernel_size=1),
          ),
        }
      )

    # Final convolution that maps the most nested representation to classes.
    self.finalConv = nn.Conv2d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for NestedUNet.
  def forward(self, x):
    # Compute x00 representation.
    x00 = self.x00.forward(x)
    # Compute x10 using pooled x00.
    x10 = self.x10.forward(self.pool(x00))
    # Compute x01 by concatenating x00 and upsampled x10.
    x01 = self.x01.forward(
      torch.cat(
        [x00, F.interpolate(x10, size=(x00.size(2), x00.size(3)), mode="bilinear", align_corners=True)],
        dim=1,
      )
    )

    # Compute x20 from pooled x10.
    x20 = self.x20.forward(self.pool(x10))
    # Compute x11 by concatenating x10 and upsampled x20.
    x11 = self.x11.forward(
      torch.cat(
        [x10, F.interpolate(x20, size=(x10.size(2), x10.size(3)), mode="bilinear", align_corners=True)],
        dim=1,
      )
    )
    # Compute x02 by concatenating x00, x01 and upsampled x11.
    x02 = self.x02.forward(
      torch.cat(
        [
          x00,
          x01,
          F.interpolate(x11, size=(x00.size(2), x00.size(3)), mode="bilinear", align_corners=True),
        ],
        dim=1,
      )
    )

    # Compute x30 from pooled x20.
    x30 = self.x30.forward(self.pool(x20))
    # Compute x21 by concatenating x20 and upsampled x30.
    x21 = self.x21.forward(
      torch.cat(
        [x20, F.interpolate(x30, size=(x20.size(2), x20.size(3)), mode="bilinear", align_corners=True)],
        dim=1,
      )
    )
    # Compute x12 by concatenating x10, x11 and upsampled x21.
    x12 = self.x12.forward(
      torch.cat(
        [
          x10,
          x11,
          F.interpolate(x21, size=(x10.size(2), x10.size(3)), mode="bilinear", align_corners=True),
        ],
        dim=1,
      )
    )
    # Compute x03 by concatenating x00, x01, x02 and upsampled x12.
    x03 = self.x03.forward(
      torch.cat(
        [
          x00,
          x01,
          x02,
          F.interpolate(x12, size=(x00.size(2), x00.size(3)), mode="bilinear", align_corners=True),
        ],
        dim=1,
      )
    )

    # Map the most nested representation to logits and return.
    return self.finalConv(x03)


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


# Additional U-Net variants appended at file end to keep all architectures together.

class UNet3Plus(nn.Module):
  r'''
  UNet3+ compact implementation for multi-scale full-resolution fusion.

  Short summary:
    Implements UNet3+ style full-scale skip aggregation across encoder and
    decoder levels to improve multi-scale feature fusion at each decoder stage.

  Parameters:
    inputChannels (int): Number of input channels. Default 3.
    numClasses (int): Number of output classes. Default 2.
    baseChannels (int): Base number of filters. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling.

  Attributes:
    enc1..enc4, center: Encoder/bottleneck blocks.
    up (nn.ModuleDict): Upsampling modules keyed by CamelCase names.
    fuse1..fuse3: Fusion conv blocks that aggregate multi-scale maps.
    finalConv (torch.nn.Conv2d): 1x1 conv to logits.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, H, W].
  '''

  # Initialize UNet3Plus.
  def __init__(self, inputChannels=3, numClasses=2, baseChannels=64, useConvTranspose2d=True):
    # Call super initializer.
    super(UNet3Plus, self).__init__()
    # Create encoder stages.
    self.enc1 = DoubleConv(inputChannels, baseChannels)
    # Create pooling after encoder stage 1.
    self.pool1 = nn.MaxPool2d(2)
    # Create encoder stage 2.
    self.enc2 = DoubleConv(baseChannels, (baseChannels * 2))
    # Create pooling after encoder stage 2.
    self.pool2 = nn.MaxPool2d(2)
    # Create encoder stage 3.
    self.enc3 = DoubleConv((baseChannels * 2), (baseChannels * 4))
    # Create pooling after encoder stage 3.
    self.pool3 = nn.MaxPool2d(2)
    # Create encoder stage 4.
    self.enc4 = DoubleConv((baseChannels * 4), (baseChannels * 8))
    # Create pooling after encoder stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Create center block.
    self.center = DoubleConv((baseChannels * 8), (baseChannels * 16))

    # Prepare upsampling modules as ModuleDict with CamelCase keys.
    if (useConvTranspose2d):
      # Create learned upsampling layers.
      self.up = nn.ModuleDict(
        {
          "UpC3": nn.ConvTranspose2d((baseChannels * 16), (baseChannels * 8), kernel_size=2, stride=2),
          "UpC2": nn.ConvTranspose2d((baseChannels * 8), (baseChannels * 4), kernel_size=2, stride=2),
          "UpC1": nn.ConvTranspose2d((baseChannels * 4), (baseChannels * 2), kernel_size=2, stride=2),
        }
      )
    else:
      # Create bilinear upsampling blocks for each level.
      self.up = nn.ModuleDict(
        {
          "UpC3": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d((baseChannels * 16), (baseChannels * 8), kernel_size=1),
          ),
          "UpC2": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d((baseChannels * 8), (baseChannels * 4), kernel_size=1),
          ),
          "UpC1": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d((baseChannels * 4), (baseChannels * 2), kernel_size=1),
          ),
        }
      )

    # Create fusion layers for full-scale aggregation.
    self.fuse3 = DoubleConv(
      (baseChannels * 8) + (baseChannels * 4) + (baseChannels * 2) + baseChannels,
      baseChannels * 8,
    )

    # Create second-level fusion layer.
    self.fuse2 = DoubleConv(
      (baseChannels * 4) + (baseChannels * 2) + baseChannels + baseChannels,
      baseChannels * 4,
    )

    # Create first-level fusion layer.
    self.fuse1 = DoubleConv(
      (baseChannels * 2) + baseChannels + baseChannels + baseChannels,
      baseChannels * 2,
    )

    # Create final logits conv.
    self.finalConv = nn.Conv2d(baseChannels * 2, numClasses, kernel_size=1)

  # Forward pass for UNet3Plus.
  def forward(self, x):
    # Compute encoder level 1.
    e1 = self.enc1.forward(x)
    # Compute pool after level 1.
    p1 = self.pool1(e1)
    # Compute encoder level 2.
    e2 = self.enc2.forward(p1)
    # Compute pool after level 2.
    p2 = self.pool2(e2)
    # Compute encoder level 3.
    e3 = self.enc3.forward(p2)
    # Compute pool after level 3.
    p3 = self.pool3(e3)
    # Compute encoder level 4.
    e4 = self.enc4.forward(p3)
    # Compute pool after level 4.
    p4 = self.pool4(e4)
    # Compute center features.
    c = self.center.forward(p4)

    # Upsample center to level 4 decoder input.
    upC3 = self.up["UpC3"](c)
    # Align shapes if required.
    if ((upC3.size(2) != e4.size(2)) or (upC3.size(3) != e4.size(3))):
      # Interpolate to match encoder spatial size.
      upC3 = F.interpolate(
        upC3,
        size=(e4.size(2), e4.size(3)),
        mode="bilinear",
        align_corners=True,
      )

    # Concatenate full-scale features for decoder level 3 and fuse.
    f3 = self.fuse3.forward(
      torch.cat(
        [
          upC3,
          e4,
          F.interpolate(
            e3,
            size=(upC3.size(2), upC3.size(3)),
            mode="bilinear",
            align_corners=True,
          ),
          F.interpolate(
            e2,
            size=(upC3.size(2), upC3.size(3)),
            mode="bilinear",
            align_corners=True,
          ),
        ],
        dim=1,
      )
    )

    # Upsample fused level 3 to level 2.
    upC2 = self.up["UpC2"](f3)
    # Align shapes if required.
    if ((upC2.size(2) != e3.size(2)) or (upC2.size(3) != e3.size(3))):
      # Interpolate to match encoder spatial size.
      upC2 = F.interpolate(
        upC2,
        size=(e3.size(2), e3.size(3)),
        mode="bilinear",
        align_corners=True,
      )

    # Concatenate and fuse for decoder level 2.
    f2 = self.fuse2.forward(
      torch.cat(
        [
          upC2,
          e3,
          F.interpolate(
            e2,
            size=(upC2.size(2), upC2.size(3)),
            mode="bilinear",
            align_corners=True,
          ),
          F.interpolate(
            e1,
            size=(upC2.size(2), upC2.size(3)),
            mode="bilinear",
            align_corners=True,
          ),
        ],
        dim=1,
      )
    )

    # Upsample fused level 2 to level 1.
    upC1 = self.up["UpC1"](f2)
    # Align shapes if required.
    if ((upC1.size(2) != e2.size(2)) or (upC1.size(3) != e2.size(3))):
      # Interpolate to match encoder spatial size.
      upC1 = F.interpolate(
        upC1,
        size=(e2.size(2), e2.size(3)),
        mode="bilinear",
        align_corners=True,
      )

    # Concatenate and fuse for decoder level 1.
    f1 = self.fuse1.forward(
      torch.cat(
        [
          upC1,
          e2,
          F.interpolate(
            e1,
            size=(upC1.size(2), upC1.size(3)),
            mode="bilinear",
            align_corners=True,
          ),
          F.interpolate(
            c,
            size=(upC1.size(2), upC1.size(3)),
            mode="bilinear",
            align_corners=True,
          ),
        ],
        dim=1,
      )
    )

    # Compute final logits and return.
    return self.finalConv(f1)


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
    d4 = self.dec4.forward(torch.cat([u4, e4], dim=1))
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
    # Encoder dense blocks and transition convs.
    self.enc1 = DenseBlock(inputChannels, numLayers=3, growthRate=baseChannels // 2)
    self.trans1 = nn.Conv2d(inputChannels + 3 * (baseChannels // 2), baseChannels, kernel_size=1)
    self.pool1 = nn.MaxPool2d(2)
    self.enc2 = DenseBlock(baseChannels, numLayers=3, growthRate=baseChannels // 2)
    self.trans2 = nn.Conv2d(baseChannels + 3 * (baseChannels // 2), baseChannels * 2, kernel_size=1)
    self.pool2 = nn.MaxPool2d(2)
    self.enc3 = DenseBlock(baseChannels * 2, numLayers=3, growthRate=baseChannels // 2)
    self.trans3 = nn.Conv2d(baseChannels * 2 + 3 * (baseChannels // 2), baseChannels * 4, kernel_size=1)
    self.pool3 = nn.MaxPool2d(2)
    self.enc4 = DenseBlock(baseChannels * 4, numLayers=3, growthRate=baseChannels // 2)
    self.trans4 = nn.Conv2d(baseChannels * 4 + 3 * (baseChannels // 2), baseChannels * 8, kernel_size=1)
    self.pool4 = nn.MaxPool2d(2)
    # Center.
    self.center = DenseBlock(baseChannels * 8, numLayers=3, growthRate=baseChannels // 2)

    # Upsampling.
    if (useConvTranspose2d):
      self.up4 = nn.ConvTranspose2d(baseChannels * 8 + 3 * (baseChannels // 2), baseChannels * 4, kernel_size=2,
                                    stride=2)
      self.up3 = nn.ConvTranspose2d(baseChannels * 4 + 3 * (baseChannels // 2), baseChannels * 2, kernel_size=2,
                                    stride=2)
      self.up2 = nn.ConvTranspose2d(baseChannels * 2 + 3 * (baseChannels // 2), baseChannels, kernel_size=2, stride=2)
      self.up1 = nn.ConvTranspose2d(baseChannels + 3 * (baseChannels // 2), baseChannels // 2, kernel_size=2, stride=2)
    else:
      self.up4 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 8 + 3 * (baseChannels // 2), baseChannels * 4, kernel_size=1),
      )
      self.up3 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 4 + 3 * (baseChannels // 2), baseChannels * 2, kernel_size=1),
      )
      self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels * 2 + 3 * (baseChannels // 2), baseChannels, kernel_size=1),
      )
      self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(baseChannels + 3 * (baseChannels // 2), baseChannels // 2, kernel_size=1),
      )

    # Final conv.
    self.finalConv = nn.Conv2d(baseChannels // 2, numClasses, kernel_size=1)

  # Forward pass.
  def forward(self, x):
    # Encoder 1.
    e1 = self.enc1.forward(x)
    t1 = self.trans1.forward(e1)
    p1 = self.pool1(t1)
    # Encoder 2.
    e2 = self.enc2.forward(p1)
    t2 = self.trans2.forward(e2)
    p2 = self.pool2(t2)
    # Encoder 3.
    e3 = self.enc3.forward(p2)
    t3 = self.trans3.forward(e3)
    p3 = self.pool3(t3)
    # Encoder 4.
    e4 = self.enc4.forward(p3)
    t4 = self.trans4.forward(e4)
    p4 = self.pool4(t4)
    # Center.
    c = self.center.forward(p4)
    # Decoder stage 4.
    u4 = self.up4.forward(torch.cat([c, t4], dim=1))
    # Decoder stage 3.
    u3 = self.up3.forward(torch.cat([d4, t3], dim=1))
    # Decoder stage 2.
    u2 = self.up2.forward(torch.cat([d3, t2], dim=1))
    # Decoder stage 1.
    u1 = self.up1.forward(torch.cat([d2, t1], dim=1))
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
    # Center.
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
    # Center.
    c = self.center.forward(p4)
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


class UNet3D(nn.Module):
  r'''
  3D U-Net for volumetric segmentation.

  Short summary:
    A compact 3D U-Net using Conv3d and Pool3d operations for volumetric data.

  Parameters:
    inputChannels (int): Number of input channels for 3D volumes. Default 1.
    numClasses (int): Number of segmentation classes. Default 2.
    baseChannels (int): Base channel width. Default 16.
    useConvTranspose3d (bool): If True use ConvTranspose3d for upsampling.

  Returns:
    torch.Tensor: Logits tensor of shape [B, numClasses, D, H, W].
  '''

  # Initialize UNet3D.
  def __init__(self, inputChannels=1, numClasses=2, baseChannels=16, useConvTranspose3d=True):
    # Call parent initializer.
    super(UNet3D, self).__init__()
    # Encoder 3D conv blocks.
    self.enc1 = nn.Sequential(
      nn.Conv3d(inputChannels, baseChannels, kernel_size=3, padding=1),
      nn.BatchNorm3d(baseChannels),
      nn.ReLU(inplace=True),
      nn.Conv3d(baseChannels, baseChannels, kernel_size=3, padding=1),
      nn.BatchNorm3d(baseChannels),
      nn.ReLU(inplace=True),
    )
    # Pooling and second encoder.
    self.pool1 = nn.MaxPool3d(2)
    self.enc2 = nn.Sequential(
      nn.Conv3d(baseChannels, baseChannels * 2, kernel_size=3, padding=1),
      nn.BatchNorm3d(baseChannels * 2),
      nn.ReLU(inplace=True),
    )
    self.pool2 = nn.MaxPool3d(2)
    # Center block.
    self.center = nn.Sequential(
      nn.Conv3d(baseChannels * 2, baseChannels * 4, kernel_size=3, padding=1),
      nn.BatchNorm3d(baseChannels * 4),
      nn.ReLU(inplace=True),
    )

    # Upsampling strategy.
    if (useConvTranspose3d):
      self.up1 = nn.ConvTranspose3d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose3d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
    else:
      self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
      self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    # Decoder 3D convs and final conv.
    self.dec1 = nn.Sequential(
      nn.Conv3d(baseChannels * 4, baseChannels * 2, kernel_size=3, padding=1),
      nn.BatchNorm3d(baseChannels * 2),
      nn.ReLU(inplace=True),
    )
    self.dec2 = nn.Sequential(
      nn.Conv3d(baseChannels * 2, baseChannels, kernel_size=3, padding=1),
      nn.BatchNorm3d(baseChannels),
      nn.ReLU(inplace=True),
    )
    self.finalConv = nn.Conv3d(baseChannels, numClasses, kernel_size=1)

  # Forward pass for UNet3D.
  def forward(self, x):
    # Encoder 3D forward.
    e1 = self.enc1.forward(x)
    p1 = self.pool1(e1)
    e2 = self.enc2.forward(p1)
    p2 = self.pool2(e2)
    # Center.
    c = self.center.forward(p2)
    # Decoder.
    u1 = self.up1(c)
    # Align spatial dims when necessary using a multi-line interpolate call.
    if ((u1.size(2) != e2.size(2)) or (u1.size(3) != e2.size(3))):
      u1 = F.interpolate(
        u1,
        size=(e2.size(2), e2.size(3)),
        mode="trilinear",
        align_corners=True,
      )
    d1 = self.dec1.forward(torch.cat([u1, e2], dim=1))
    u2 = self.up2(d1)
    # Align spatial dims when necessary using a multi-line interpolate call.
    if ((u2.size(2) != e1.size(2)) or (u2.size(3) != e1.size(3))):
      u2 = F.interpolate(
        u2,
        size=(e1.size(2), e1.size(3)),
        mode="trilinear",
        align_corners=True,
      )
    d2 = self.dec2.forward(torch.cat([u2, e1], dim=1))
    # Final logits.
    return self.finalConv(d2)


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
    modelType (str): Case-insensitive model selection string, e.g. "dynamic", "unet3plus".

  Returns:
    nn.Module: Instantiated UNet variant ready for training or inference.
  '''

  # Keep the incoming modelType flexible and prepare lower-case matcher.
  mt = (modelType or "dynamic")
  mtLower = mt.lower()

  # New variants mapping.
  if (mtLower == "unet3plus"):
    return UNet3Plus(
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

  if (mtLower == "unet3d"):
    return UNet3D(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=max(8, baseChannels // 2),
      useConvTranspose3d=(upMode == "transpose"),
    )

  if (mtLower in ("original", "legacy")):
    return UNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
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

  if (mtLower == "nested"):
    return NestedUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
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
