import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
  r'''
  Double convolution block used throughout the U-Net encoder and decoder.

  This block applies two sequential 3x3 convolution layers, each followed by
  Batch Normalization and ReLU activation.

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.

  Returns:
    torch.Tensor: The activated feature map after two conv->BN->ReLU layers.
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

  This block mirrors `DoubleConv` behavior but exposes runtime configuration
  for normalization type ("batch"|"instance"|"none"), dropout probability and
  an optional residual connection (adds input to output when in/out channels match).

  Parameters:
    inChannels (int): Number of input channels.
    outChannels (int): Number of output channels.
    norm (str): Normalization type to use. One of "batch", "instance", or "none". Default "batch".
    dropout (float): Dropout probability applied after the block (0.0 disables). Default 0.0.
    residual (bool): Whether to add a residual skip connection when shapes permit. Default False.

  Returns:
    torch.Tensor: Activated (and optionally regularized) feature map.
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
  Residual block: two conv->BN->ReLU layers with an identity skip when channels match.

  Parameters:
    inChannels (int): Input channels.
    outChannels (int): Output channels.
  Returns:
    torch.Tensor: Activated tensor (with residual addition if applicable).
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
  Attention gating for skip connections as used in Attention U-Net.

  Parameters:
    F_g (int): Channels of gating signal (from decoder).
    F_l (int): Channels of skip connection (from encoder).
    F_int (int): Intermediate channel count for the gating computations.
  Returns:
    torch.Tensor: Re-weighted skip features.
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
  Depthwise separable convolution block (depthwise conv + pointwise conv) followed by BN and ReLU.

  Parameters:
    inChannels (int): Input channels.
    outChannels (int): Output channels.
  Returns:
    torch.Tensor: Processed tensor.
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


class UNet(nn.Module):
  r'''
  U-Net architecture for image segmentation (logits output).

  Implements a standard encoder-decoder U-Net with four downsampling
  stages, a center block, and four upsampling stages. Upsampling can be
  performed with ConvTranspose2d (learned) or with bilinear upsampling
  followed by 1x1 convolutions.

  Parameters:
    inputChannels (int): Number of channels in the input image. Default 3.
    numClasses (int): Number of segmentation classes (output channels). Default 2.
    baseChannels (int): Number of filters in the first encoder stage. Default 64.
    useConvTranspose2d (bool): If True use ConvTranspose2d for upsampling, otherwise use bilinear Upsample + 1x1 conv. Default True.

  Attributes:
    enc1..enc4 (torch.nn.Module): Encoder DoubleConv blocks.
    center (torch.nn.Module): Center DoubleConv block.
    up1..up4, dec1..dec4 (torch.nn.Module): Decoder upsampling and DoubleConv blocks.
    finalConv (torch.nn.Conv2d): 1x1 conv mapping to numClasses logits.

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
    # Encoder stage 1 forward.
    e1 = self.enc1.forward(x)
    # Pool after stage 1.
    p1 = self.pool1(e1)
    # Encoder stage 2 forward.
    e2 = self.enc2.forward(p1)
    # Pool after stage 2.
    p2 = self.pool2(e2)
    # Encoder stage 3 forward.
    e3 = self.enc3.forward(p2)
    # Pool after stage 3.
    p3 = self.pool3(e3)
    # Encoder stage 4 forward.
    e4 = self.enc4.forward(p3)
    # Pool after stage 4.
    p4 = self.pool4(e4)
    # Center block forward.
    c = self.center.forward(p4)
    # Upsample from center for decoder stage 4.
    u4 = self.up4(c)
    # Concatenate encoder features for skip connection.
    u4 = torch.cat([u4, e4], dim=1)
    # Decode stage 4.
    d4 = self.dec4.forward(u4)
    # Upsample for decoder stage 3.
    u3 = self.up3(d4)
    # Concatenate encoder features for skip connection.
    u3 = torch.cat([u3, e3], dim=1)
    # Decode stage 3.
    d3 = self.dec3.forward(u3)
    # Upsample for decoder stage 2.
    u2 = self.up2(d3)
    # Concatenate encoder features for skip connection.
    u2 = torch.cat([u2, e2], dim=1)
    # Decode stage 2.
    d2 = self.dec2.forward(u2)
    # Upsample for decoder stage 1.
    u1 = self.up1(d2)
    # Concatenate encoder features for skip connection.
    u1 = torch.cat([u1, e1], dim=1)
    # Decode stage 1.
    d1 = self.dec1.forward(u1)
    # Compute final logits.
    logits = self.finalConv(d1)
    # Return logits tensor.
    return logits


class DynamicUNet(nn.Module):
  r'''
  Flexible U-Net implementation with configurable depth and block options.

  This implementation generalizes the fixed 4-stage `UNet` by allowing the
  user to specify encoder depth, base channel count, normalization, dropout,
  residual blocks, and the upsampling strategy (ConvTranspose2d or bilinear).

  Parameters:
    inputChannels (int): Number of channels in the input image. Default 3.
    numClasses (int): Number of segmentation classes (output channels). Default 2.
    baseChannels (int): Number of filters in the first encoder stage. Default 64.
    depth (int): Number of downsampling stages (encoder depth). Default 4.
    upMode (str): "transpose" for ConvTranspose2d, "bilinear" for interpolation + conv. Default "transpose".
    norm (str): Normalization type for inner blocks: "batch" | "instance" | "none". Default "batch".
    dropout (float): Dropout probability applied after each block. Default 0.0.
    residual (bool): Whether to use residual-style block outputs when possible. Default False.

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
      x = torch.cat([x, feat], dim=1)
      # Decode concatenated features.
      x = dec.forward(x)

    # Map to logits with final conv.
    logits = self.finalConv(x)
    # Return logits.
    return logits


class AttentionUNet(nn.Module):
  r'''
  U-Net with attention gates applied to skip connections.

  Parameters:
    inputChannels, numClasses, baseChannels, useConvTranspose2d (bool)
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
    # Encoder forward passes with pooling.
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
    # Upsample and apply attention gating for stage 4.
    u4 = self.up4(c)
    e4_att = self.att4.forward(u4, e4)
    d4 = self.dec4.forward(torch.cat([u4, e4_att], dim=1))
    # Upsample and apply attention gating for stage 3.
    u3 = self.up3(d4)
    e3_att = self.att3.forward(u3, e3)
    d3 = self.dec3.forward(torch.cat([u3, e3_att], dim=1))
    # Upsample and apply attention gating for stage 2.
    u2 = self.up2(d3)
    e2_att = self.att2.forward(u2, e2)
    d2 = self.dec2.forward(torch.cat([u2, e2_att], dim=1))
    # Upsample and apply attention gating for stage 1.
    u1 = self.up1(d2)
    e1_att = self.att1.forward(u1, e1)
    d1 = self.dec1.forward(torch.cat([u1, e1_att], dim=1))
    # Final logits computation.
    logits = self.finalConv(d1)
    # Return logits.
    return logits


class MobileUNet(nn.Module):
  r'''
  Lightweight U-Net variant using depthwise separable convolutions for efficiency.

  Parameters:
    inputChannels (int), numClasses (int), baseChannels (int), useConvTranspose2d (bool)
  Returns:
    torch.Tensor: Logits tensor.
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
    # Pool after stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Center block.
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
      # Bilinear upsampling alternatives.
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
    if (u4.size(2) != e4.size(2)) or (u4.size(3) != e4.size(3)):
      u4 = F.interpolate(u4, size=(e4.size(2), e4.size(3)), mode="bilinear", align_corners=True)
    # Decode stage 4.
    d4 = self.dec4(torch.cat([u4, e4], dim=1))
    # Upsample and align for stage 3.
    u3 = self.up3(d4)
    if (u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3)):
      u3 = F.interpolate(u3, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
    # Decode stage 3.
    d3 = self.dec3(torch.cat([u3, e3], dim=1))
    # Upsample and align for stage 2.
    u2 = self.up2(d3)
    if (u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3)):
      u2 = F.interpolate(u2, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
    # Decode stage 2.
    d2 = self.dec2(torch.cat([u2, e2], dim=1))
    # Upsample and align for stage 1.
    u1 = self.up1(d2)
    if (u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3)):
      u1 = F.interpolate(u1, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
    # Decode stage 1.
    d1 = self.dec1(torch.cat([u1, e1], dim=1))
    # Final logits and return.
    logits = self.finalConv(d1)
    return logits


class ResidualUNet(nn.Module):
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
    # Pool after stage 4.
    self.pool4 = nn.MaxPool2d(2)
    # Center block.
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
      u4 = F.interpolate(u4, size=(e4.size(2), e4.size(3)), mode="bilinear", align_corners=True)
    # Concatenate and decode stage 4.
    u4 = torch.cat([u4, e4], dim=1)
    d4 = self.dec4.forward(u4)
    # Upsample and align stage 3.
    u3 = self.up3(d4)
    if (u3.size(2) != e3.size(2)) or (u3.size(3) != e3.size(3)):
      u3 = F.interpolate(u3, size=(e3.size(2), e3.size(3)), mode="bilinear", align_corners=True)
    u3 = torch.cat([u3, e3], dim=1)
    d3 = self.dec3.forward(u3)
    # Upsample and align stage 2.
    u2 = self.up2(d3)
    if (u2.size(2) != e2.size(2)) or (u2.size(3) != e2.size(3)):
      u2 = F.interpolate(u2, size=(e2.size(2), e2.size(3)), mode="bilinear", align_corners=True)
    u2 = torch.cat([u2, e2], dim=1)
    d2 = self.dec2.forward(u2)
    # Upsample and align stage 1.
    u1 = self.up1(d2)
    if (u1.size(2) != e1.size(2)) or (u1.size(3) != e1.size(3)):
      u1 = F.interpolate(u1, size=(e1.size(2), e1.size(3)), mode="bilinear", align_corners=True)
    u1 = torch.cat([u1, e1], dim=1)
    # Decode final stage and return logits.
    d1 = self.dec1.forward(u1)
    return self.finalConv(d1)


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
  Factory to create a U-Net instance. Supported model types:
    - "dynamic" (default): flexible `DynamicUNet` supporting depth, norm, dropout.
    - "original" or "legacy": original fixed 4-stage `UNet`.
    - "residual": `ResidualUNet` using residual blocks.
    - "attention": `AttentionUNet` with attention gates on skip connections.
    - "mobile": `MobileUNet` using depthwise separable convs for efficiency.

  Parameters are passed to the corresponding constructor. Returns an nn.Module.
  '''

  # Normalize the model type string.
  modelType = (modelType or "dynamic").lower()

  # Return the original legacy UNet when requested.
  if (modelType in ("original", "legacy")):
    return UNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose"),
    )

  # Return the dynamic UNet variant.
  if (modelType == "dynamic"):
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

  # Residual UNet nested implementation for factory use.
  if (modelType == "residual"):
    # Return an instance of the nested ResidualUNet.
    return ResidualUNet(
      inputChannels,
      numClasses,
      baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  # Attention UNet variant.
  if (modelType == "attention"):
    return AttentionUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  # Mobile UNet variant.
  if (modelType == "mobile"):
    return MobileUNet(
      inputChannels=inputChannels,
      numClasses=numClasses,
      baseChannels=baseChannels,
      useConvTranspose2d=(upMode == "transpose")
    )

  # Unknown type: fallback to dynamic UNet.
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
