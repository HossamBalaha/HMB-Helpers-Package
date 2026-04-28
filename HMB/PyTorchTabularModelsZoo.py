import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorchTabularModelsZoo: Models for Tabular Data Classification.
#
# This module provides a collection of neural network architectures specifically designed
# for tabular data classification tasks, such as phishing detection, fraud analysis,
# and structured data prediction. All models accept 2D feature tensors and produce
# classification logits with a consistent interface.
#
# ================================================================================
# TABULAR DATA COMPATIBILITY
# ================================================================================
#
# All models in this zoo are designed for tabular data with the following contract:
#
#   Input:  torch.Tensor of shape (batch_size, inputSize)
#           - Each row represents one sample (e.g., one URL, one transaction).
#           - Each column represents one numeric feature (e.g., URL length, domain age).
#           - Features should be preprocessed: normalized, encoded, and missing values handled.
#
#   Output: torch.Tensor of shape (batch_size, numClasses)
#           - Raw logits (unnormalized scores) for each class.
#           - Apply torch.softmax(logits, dim=1) to obtain class probabilities.
#           - Apply torch.argmax(logits, dim=1) to obtain predicted class indices.
#
# ================================================================================
# MODEL CATEGORIES
# ================================================================================
#
# 1. NATIVE TABULAR MODELS (Recommended for most tasks)
#    These architectures were explicitly designed for feature-vector inputs:
#
#    - MLPModel: Baseline multi-layer perceptron with BatchNorm and Dropout.
#    - TabTransformerModel: Embeds each feature as a token; applies Transformer attention.
#    - FTTransformer: Prepends CLS token; classifies from CLS embedding after attention.
#    - GANDALFModel: Per-feature gating networks for adaptive feature selection.
#    - AutoIntModel: Multi-head self-attention for explicit feature crossing.
#    - SAINTModel: Combines intra-sample and inter-sample attention for few-shot learning.
#    - NeuralAdditiveModel: Interpretable architecture with independent feature subnetworks.
#    - DeepCrossNetwork: Polynomial feature crossing via cross layers + deep MLP.
#    - ResNetTabularModel: Residual blocks enable deeper tabular networks without vanishing gradients.
#
# 2. ADAPTED MODELS (Tabular via sequence/graph interpretation)
#    These models were originally designed for other domains but work with tabular data:
#
#    - TCNModel: Treats feature vector as 1D sequence; uses causal convolutions.
#      Note: Feature ordering may affect performance; consider domain-informed ordering.
#
#    - HybridCNNTransformerModel: CNN extracts local feature patterns; Transformer adds global context.
#      Note: Assumes features have local spatial relationships (e.g., grouped by category).
#
#    - GNNModel: Message passing over graph-structured tabular data.
#      Note: Requires adjacency matrix (adj) or edge index (edgeIndex) to define relationships.
#      For pure tabular data without inherent graph structure:
#        - Option A: Pass adj=None to degrade to MLP behavior.
#        - Option B: Construct feature-graph using correlation matrix: corr = np.corrcoef(X.T).
#        - Option C: Construct sample-graph using k-NN in feature space.
#
# 3. UTILITY MODELS (Specialized tabular workflows)
#    These support advanced tabular learning paradigms:
#
#    - VAEAnomalyDetector: Unsupervised anomaly detection via reconstruction error.
#      Output: (reconstruction, mu, logvar) tuple; use reconstruction_error() for scoring.
#
#    - VAEClassifier: Hybrid VAE + classifier; learns robust features then classifies.
#      Output: (batch_size, numClasses) logits for supervised training.
#
#    - ContrastiveEncoder: Self-supervised pretraining via SimCLR-style contrastive loss.
#      Output: (normalized_embedding, normalized_projection) tuple for contrastive objectives.
#
#    - ContrastiveClassifier: Two-stage workflow: pretrain encoder, then attach classification head.
#      Output: (batch_size, numClasses) logits for fine-tuning.
#
# ================================================================================
# USAGE EXAMPLES FOR PHISHING DETECTION
# ================================================================================
#
# # Basic classification workflow:
# import torch
# import torch.nn.functional as F
#
# # Load your tabular features (e.g., from CIC-Trap4Phish-2025 dataset)
# # X: numpy array or pandas DataFrame of shape (num_samples, 100)
# # y: numpy array of shape (num_samples,) with class labels {0, 1}
#
# # Convert to PyTorch tensors
# x = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, 100)
# y = torch.tensor(y, dtype=torch.long)     # Shape: (num_samples,)
#
# # Instantiate a model via factory
# model = GetModel("TabTransformerModel", inputSize=100, numClasses=2)
#
# # Forward pass: get logits
# logits = model(x)  # Shape: (num_samples, 2)
#
# # Convert to probabilities and predictions
# probs = F.softmax(logits, dim=1)
# predictions = torch.argmax(probs, dim=1)
#
# # Compute loss for training
# loss = F.cross_entropy(logits, y)
#
# # For GNN models, provide adjacency matrix (optional for tabular fallback)
# # Example: correlation-based feature graph
# import numpy as np
# corr = np.corrcoef(X.T)  # Feature correlation matrix
# adj = torch.tensor(np.abs(corr) > 0.3, dtype=torch.float32)  # Thresholded adjacency
# logits = model(x, adj=adj)  # GNN forward pass with graph structure
#
# ================================================================================
# TRAINING RECOMMENDATIONS
# ================================================================================
#
# 1. Data Preprocessing:
#    - Normalize numeric features to zero mean, unit variance.
#    - Encode categorical features via one-hot or learned embeddings.
#    - Handle missing values via imputation or masking.
#
# 2. Optimization:
#    - Use AdamW optimizer with weight decay (1e-4 to 1e-2).
#    - Apply learning rate scheduling (e.g., cosine annealing).
#    - Enable mixed precision training via enableMixedPrecision() for GPU acceleration.
#
# 3. Regularization:
#    - Apply dropout (0.1 to 0.3) in all models except lightweight baselines.
#    - Use early stopping on validation loss to prevent overfitting.
#    - Consider label smoothing for noisy phishing datasets.
#
# 4. Evaluation:
#    - Report accuracy, precision, recall, F1, and AUC-ROC for binary classification.
#    - Use stratified k-fold cross-validation for robust performance estimation.
#    - Monitor calibration error if probability thresholds matter for deployment.
#
# ================================================================================
# EXTENDING THE MODEL ZOO
# ================================================================================
#
# To add a new tabular model:
#
# 1. Create a class inheriting from nn.Module.
# 2. Implement __init__ with inputSize, numClasses parameters and validation.
# 3. Implement forward(self, x: torch.Tensor) -> torch.Tensor with shape documentation.
# 4. Add a route to GetModel() factory: if (name == "NewModel"): return NewModel(...).
# 5. Add test configuration to __main__ validation suite.
#
# class NewTabularModel(nn.Module):
#   r'''
#   Brief description of the new architecture.
#
#   Parameters:
#     inputSize (int): Number of input features.
#     numClasses (int): Number of output classes.
#     customParam (type): Description of custom hyperparameter.
#   '''
#
#   def __init__(self, inputSize: int, numClasses: int, customParam: float = 0.1):
#     if (inputSize <= 0):
#       raise ValueError(f"inputSize must be positive, got {inputSize}")
#     super(NewTabularModel, self).__init__()
#     # ... model definition ...
#
#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     # Expects x: Tensor of shape (batch_size, inputSize).
#     # Returns logits: Tensor of shape (batch_size, numClasses).
#     # ... forward implementation ...
#     return logits
#
# # Then add to GetModel:
# if (name == "NewTabular"):
#   return NewTabularModel(inputSize, numClasses)

AVAILABLE_MODELS = [
  "MLPModel",  # Baseline multi-layer perceptron with BatchNorm and Dropout.
  "TCNModel",  # Treats feature vector as 1D sequence; uses causal convolutions.
  "TabTransformerModel",  # Embeds each feature as a token; applies Transformer attention.
  "FTTransformerModel",  # Prepends CLS token; classifies from CLS embedding after attention.
  "GANDALFModel",  # Per-feature gating networks for adaptive feature selection.
  "HybridCNNTransformerModel",  # CNN extracts local feature patterns; Transformer adds global context.
  "VAEAnomalyDetector",  # Unsupervised anomaly detection via reconstruction error.
  "VAEClassifier",  # Hybrid VAE + classifier; learns robust features then classifies.
  "ContrastiveEncoder",  # Self-supervised pretraining via SimCLR-style contrastive loss.
  "ContrastiveClassifier",  # Two-stage workflow: pretrain encoder, then attach classification head.
  "GNNModel",  # Graph Neural Network for tabular data with graph structure.
]


class MLPModel(nn.Module):
  r'''
  Multi-layer perceptron for tabular classification.

  Builds a stack of Linear -> BatchNorm -> ReLU -> Dropout blocks followed by
  a final linear classification layer.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    hiddenSizes (list[int] or None): Hidden layer sizes. If None defaults are used.
    dropout (float): Dropout probability applied after activations.
  '''

  def __init__(self, inputSize: int, numClasses: int, hiddenSizes=None, dropout: float = 0.2):
    # Validate input parameters to prevent silent failures.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    if (not (0.0 <= dropout < 1.0)):
      raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    # Call the superclass constructor.
    super(MLPModel, self).__init__()
    # Set default hidden sizes when None.
    if (hiddenSizes is None):
      # Default hidden sizes list.
      hiddenSizes = [256, 128]
    # Build a list of layers for the sequential model.
    layers = []
    # Track the current size starting from input size.
    currSize = inputSize
    # Iterate over hidden sizes to create Linear -> ReLU -> Dropout blocks.
    for h in hiddenSizes:
      # Add a linear layer.
      layers.append(nn.Linear(currSize, h))
      # Add batch normalization.
      layers.append(nn.BatchNorm1d(h))
      # Add activation.
      layers.append(nn.ReLU())
      # Add dropout.
      layers.append(nn.Dropout(dropout))
      # Update current size.
      currSize = h
    # Add final classification layer mapping to number of classes.
    layers.append(nn.Linear(currSize, numClasses))
    # Create the sequential model from the layers.
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Ensure input is float tensor.
    x = x.float()
    # Pass input through the sequential network and return logits.
    return self.net(x)


class TCNModel(nn.Module):
  r'''
  Temporal convolutional network style model for tabular data treated as a
  1D sequence of features.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    channels (list[int] or None): Channel sizes for convolutional blocks.
    kernelSize (int): Kernel size for Conv1d layers.
    dropout (float): Dropout probability.
  '''

  # Initialize the TCN with input size and number of classes.
  def __init__(self, inputSize: int, numClasses: int, channels=None, kernelSize: int = 3, dropout: float = 0.2):
    # Call superclass constructor.
    super(TCNModel, self).__init__()
    # Set default channels when None.
    if (channels is None):
      # Default channels for convolutional blocks.
      channels = [64, 128]
    # Define a list to hold convolutional blocks.
    layers = []
    # Start with a projection from input features to channels[0].
    layers.append(nn.Conv1d(in_channels=1, out_channels=channels[0], kernel_size=1))
    # Add activation.
    layers.append(nn.ReLU())
    # Add subsequent conv blocks.
    inCh = channels[0]
    for ch in channels[1:]:
      # Add a convolutional layer with padding to preserve length.
      layers.append(
        nn.Conv1d(in_channels=inCh, out_channels=ch, kernel_size=kernelSize, padding=kernelSize // 2)
      )
      # Add batch normalization.
      layers.append(nn.BatchNorm1d(ch))
      # Add activation.
      layers.append(nn.ReLU())
      # Add dropout.
      layers.append(nn.Dropout(dropout))
      # Update input channels for next block.
      inCh = ch
    # Global pooling and final linear layer.
    self.conv = nn.Sequential(*layers)
    # Compute final linear after pooling.
    self.fc = nn.Linear(inCh, numClasses)

  # Define the forward pass for TCNModel.
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Returns logits: Tensor of shape (batch_size, numClasses).
    # Convert to (batch, 1, features) for Conv1d input.
    x = x.float().unsqueeze(1)
    # Pass through conv blocks.
    h = self.conv(x)
    # Global average pool over the sequence dimension.
    h = h.mean(dim=2)
    # Linear projection to num classes.
    return self.fc(h)


class TabTransformerModel(nn.Module):
  r'''
  Transformer-based model that embeds each numeric feature as a token and
  applies a Transformer encoder across features for classification.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    embedDim (int): Embedding dimension for each feature token.
    numHeads (int): Number of attention heads.
    numLayers (int): Number of Transformer encoder layers.
  '''

  # Initialize the TabTransformer with input size and number of classes.
  # Each input feature is embedded separately and treated as a token.
  def __init__(self, inputSize: int, numClasses: int, embedDim: int = 64, numHeads: int = 4, numLayers: int = 2):
    # Call superclass constructor.
    super(TabTransformerModel, self).__init__()
    # Embed each feature dimension independently into embedDim space.
    self.featureEmbed = nn.Linear(1, embedDim)
    # Learnable positional encoding per feature position.
    self.posEmbed = nn.Parameter(torch.randn(1, inputSize, embedDim) * 0.02)
    # Transformer encoder layers with batch_first for convenience.
    encoderLayer = nn.TransformerEncoderLayer(d_model=embedDim, nhead=numHeads, batch_first=True, dropout=0.1)
    # Create transformer encoder stack.
    self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
    # Classification head with layer normalization for stability.
    self.classifier = nn.Sequential(nn.LayerNorm(embedDim), nn.Linear(embedDim, numClasses))

  # Define the forward pass for the TabTransformerModel.
  # Expects x: Tensor of shape (batch_size, inputSize).
  # Returns logits: Tensor of shape (batch_size, numClasses).
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Ensure input is float tensor.
    x = x.float()
    # Reshape to (batch, inputSize, 1) to embed each feature separately.
    x = x.unsqueeze(-1)
    # Embed each feature: Linear(1, embedDim) outputs (batch, inputSize, embedDim) directly.
    h = self.featureEmbed(x)
    # Add learnable positional encoding per feature position.
    h = h + self.posEmbed
    # Apply transformer encoder over feature sequence.
    h = self.transformer(h)
    # Global average pool over feature positions.
    h = h.mean(dim=1)
    # Return classification logits.
    return self.classifier(h)


class FTTransformerModel(nn.Module):
  r'''
  Lightweight FT-Transformer style model embedding each feature as a token.

  A CLS token is prepended and the classification is performed from the CLS
  token embedding after Transformer encoding.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    embedDim (int): Token embedding dimensionality.
    numHeads (int): Number of attention heads.
    numLayers (int): Number of Transformer layers.
    dropout (float): Dropout probability inside Transformer layers.
  '''

  def __init__(
    self, inputSize: int, numClasses: int, embedDim: int = 64,
    numHeads: int = 4, numLayers: int = 2, dropout: float = 0.1
  ):
    super(FTTransformerModel, self).__init__()
    # Per-feature projection to token embeddings (treat each scalar as a token).
    self.feature_embed = nn.Linear(1, embedDim)
    # Learnable CLS token for pooling.
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embedDim))
    # Positional embedding for tokens (CLS + features).
    self.pos = nn.Parameter(torch.randn(1, inputSize + 1, embedDim) * 0.02)
    # Small Transformer encoder.
    encoderLayer = nn.TransformerEncoderLayer(
      d_model=embedDim, nhead=numHeads, batch_first=True, dropout=dropout
    )
    self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
    # Classification head from CLS token.
    self.classifier = nn.Sequential(nn.LayerNorm(embedDim), nn.Linear(embedDim, numClasses))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # X: (batch, features).
    x = x.float()
    # Embed each feature independently: (batch, features, 1) -> (batch, features, embedDim).
    x = x.unsqueeze(-1)
    h = self.feature_embed(x)
    # Prepend cls token.
    cls = self.cls_token.expand(h.size(0), -1, -1)
    h = torch.cat([cls, h], dim=1)
    # Add positional embedding with shape validation.
    if (h.shape[1] != self.pos.shape[1]):
      raise ValueError(f"Token count {h.shape[1]} must match positional embedding size {self.pos.shape[1]}")
    h = h + self.pos
    h = self.transformer(h)
    clsH = h[:, 0, :]
    return self.classifier(clsH)


class GANDALFModel(nn.Module):
  r'''
  GANDALF-inspired gated feature learning model.

  The model applies a per-feature gating network followed by a shared encoder
  and a classification head.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    hidden (int): Hidden size of the shared encoder.
    gate_hidden (int): Hidden size of the gate network.
  '''

  def __init__(self, inputSize: int, numClasses: int, hidden: int = 128, gate_hidden: int = 64):
    super(GANDALFModel, self).__init__()
    # Per-feature gating network (applied to each scalar feature).
    self.gate = nn.Sequential(nn.Linear(1, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1), nn.Sigmoid())
    # Shared feature encoder after gating.
    self.encoder = nn.Sequential(nn.Linear(inputSize, hidden), nn.ReLU(), nn.Dropout(0.1))
    # Classifier head.
    self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, numClasses))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # X: (batch, features).
    x = x.float()
    # Compute gates per feature using vectorized operation.
    # Reshape to (batch * features, 1) for batched gate computation.
    batchSize, numFeatures = x.shape
    xFlat = x.reshape(-1, 1)
    gatesFlat = self.gate(xFlat)
    gates = gatesFlat.reshape(batchSize, numFeatures)
    # Apply gating and encode.
    xg = x * gates
    h = self.encoder(xg)
    return self.classifier(h)


class HybridCNNTransformerModel(nn.Module):
  r'''
  Hybrid CNN + Transformer model for tabular inputs treated as feature
  sequences.

  The convolutional front-end extracts local feature patterns and the
  Transformer provides global context before classification.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
  '''

  def __init__(
    self, inputSize: int, numClasses: int, cnnChannels=None,
    kernelSize=3, embedDim=128, numHeads=4,
    numLayers=2, dropout=0.2
  ):
    super(HybridCNNTransformerModel, self).__init__()
    # Convolutional front-end.
    if (cnnChannels is None):
      cnnChannels = [64, 128]
    layers = []
    layers.append(nn.Conv1d(1, cnnChannels[0], kernel_size=1))
    layers.append(nn.ReLU())
    inCh = cnnChannels[0]
    for ch in cnnChannels[1:]:
      layers.append(nn.Conv1d(inCh, ch, kernel_size=kernelSize, padding=kernelSize // 2))
      layers.append(nn.BatchNorm1d(ch))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout))
      inCh = ch
    self.cnn = nn.Sequential(*layers)
    # Project CNN outputs to embedding dim for transformer.
    self.proj = nn.Linear(inCh, embedDim)
    # Positional embedding per feature position (must match inputSize).
    self.posEmbed = nn.Parameter(torch.randn(1, inputSize, embedDim) * 0.02)
    # Transformer encoder with dropout for regularization.
    encoderLayer = nn.TransformerEncoderLayer(d_model=embedDim, nhead=numHeads, batch_first=True, dropout=dropout)
    self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
    # Classifier.
    self.classifier = nn.Sequential(nn.LayerNorm(embedDim), nn.Linear(embedDim, numClasses))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Convert to (batch, 1, features) for Conv1d input.
    x = x.float().unsqueeze(1)
    # Apply convolutional front-end: (batch, channels, features).
    h = self.cnn(x)
    # Rearrange to (batch, features, channels) for transformer input.
    h = h.permute(0, 2, 1)
    # Project channel dimension to embedding dimension.
    h = self.proj(h)
    # Add positional embedding; assert shape compatibility for debugging.
    if (h.shape[1] != self.posEmbed.shape[1]):
      raise ValueError(f"Feature dimension {h.shape[1]} must match positional embedding size {self.posEmbed.shape[1]}")
    h = h + self.posEmbed
    # Apply transformer encoder over feature sequence.
    h = self.transformer(h)
    # Global average pool over feature positions.
    h = h.mean(dim=1)
    # Return classification logits.
    return self.classifier(h)


class VAEAnomalyDetector(nn.Module):
  r'''
  Variational autoencoder for anomaly detection on tabular features.

  Trains as an autoencoder and uses reconstruction error as an anomaly score.

  Parameters:
    inputSize (int): Number of input features.
    latentDim (int): Dimensionality of latent representation.
    hiddenSizes (list[int] or None): Hidden sizes for encoder/decoder.
  '''

  def __init__(self, inputSize: int, latentDim: int = 32, hiddenSizes=None):
    super(VAEAnomalyDetector, self).__init__()
    # Set default hidden sizes when not provided.
    if (hiddenSizes is None):
      hiddenSizes = [128, 64]
    # Build the encoder layers as a sequential list.
    encLayers = []
    # Track the current size from input size.
    curr = inputSize
    for h in hiddenSizes:
      # Append a linear layer to the encoder stack.
      encLayers.append(nn.Linear(curr, h))
      # Append a ReLU activation to the encoder stack.
      encLayers.append(nn.ReLU())
      # Update the current size for the next layer.
      curr = h
    # Create the encoder sequential module from the built layers.
    self.encoder = nn.Sequential(*encLayers)
    self.mu = nn.Linear(curr, latentDim)
    self.logvar = nn.Linear(curr, latentDim)
    # Build the decoder layers as a sequential list.
    decLayers = []
    # Track the current size starting from latent dimension.
    curr = latentDim
    for h in reversed(hiddenSizes):
      # Append a linear layer to the decoder stack.
      decLayers.append(nn.Linear(curr, h))
      # Append a ReLU activation to the decoder stack.
      decLayers.append(nn.ReLU())
      # Update the current size for the next decoder layer.
      curr = h
    # Append final linear layer to reconstruct the input features.
    decLayers.append(nn.Linear(curr, inputSize))
    # Create the decoder sequential module from the built layers.
    self.decoder = nn.Sequential(*decLayers)

  # Reparameterization trick to sample from latent distribution.
  def reparameterize(self, mu, logvar):
    r'''
    Reparameterization trick to sample z ~ N(mu, sigma^2) from parameters.

    Parameters:
      mu (torch.Tensor): Latent mean tensor.
      logvar (torch.Tensor): Latent log-variance tensor.

    Returns:
      torch.Tensor: Sampled latent tensor.
    '''
    # Compute standard deviation from log-variance.
    std = (0.5 * logvar).exp()
    # Sample noise with same shape as std.
    eps = torch.randn_like(std)
    # Return reparameterized latent vector.
    return mu + eps * std

  # Forward pass producing reconstruction, mean, and log-variance.
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Ensure input is float tensor.
    x = x.float()
    # Encode input into hidden representation.
    h = self.encoder(x)
    # Project hidden representation to latent mean.
    mu = self.mu(h)
    # Project hidden representation to latent log-variance.
    logvar = self.logvar(h)
    # Sample latent vector using reparameterization trick.
    z = self.reparameterize(mu, logvar)
    # Decode latent vector into reconstruction.
    recon = self.decoder(z)
    # Return reconstruction and latent statistics.
    return recon, mu, logvar

  # Compute per-sample reconstruction error for anomaly scoring.
  def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
    r'''
    Compute per-sample reconstruction error used for anomaly scoring.

    Parameters:
      x (torch.Tensor): Input batch of shape (batch, inputSize).

    Returns:
      torch.Tensor: Per-sample MSE reconstruction error of shape (batch,).
    '''
    # Execute forward pass to obtain reconstruction.
    recon, mu, logvar = self.forward(x)
    # Compute mean squared error per sample and return.
    return F.mse_loss(recon, x, reduction="none").mean(dim=1)


class VAEClassifier(nn.Module):
  r'''
  Hybrid model combining a VAE for feature learning with a classifier head.
  The VAE learns a latent representation of the input features, and the classifier operates on
  the VAE's reconstruction to perform classification.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    latentDim (int): Dimensionality of the VAE latent space.
  '''

  def __init__(self, inputSize: int, numClasses: int, latentDim: int = 32):
    super(VAEClassifier, self).__init__()
    self.vae = VAEAnomalyDetector(inputSize, latentDim)
    self.classifier = nn.Linear(inputSize, numClasses)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    recon, mu, logvar = self.vae(x)
    return self.classifier(recon)


class ContrastiveEncoder(nn.Module):
  """Lightweight encoder + projection head for contrastive pretraining (SimCLR style).
  Outputs normalized embeddings and a projection vector suitable for contrastive loss."""

  def __init__(self, inputSize: int, embDim: int = 128, projDim: int = 64, hiddenSizes=None):
    super(ContrastiveEncoder, self).__init__()
    # Set default hidden sizes when not provided.
    if (hiddenSizes is None):
      hiddenSizes = [256, 128]
    # Build encoder layers sequentially.
    layers = []
    # Track current feature size starting from input size.
    curr = inputSize
    for h in hiddenSizes:
      # Append a linear layer to the encoder stack.
      layers.append(nn.Linear(curr, h))
      # Append a ReLU activation to the encoder stack.
      layers.append(nn.ReLU())
      # Update current size for next layer.
      curr = h
    # Append final linear projection to embedding dimension.
    layers.append(nn.Linear(curr, embDim))
    # Create encoder sequential module from built layers.
    self.encoder = nn.Sequential(*layers)
    # Projection head.
    # Build projection head for contrastive loss.
    self.projector = nn.Sequential(nn.Linear(embDim, projDim), nn.ReLU(), nn.Linear(projDim, projDim))

  # Forward pass that returns normalized encoder embedding and projection.
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure input is float tensor.
    x = x.float()
    # Encode input to embedding space.
    h = self.encoder(x)
    # Project encoder embeddings to projection space.
    z = self.projector(h)
    # Normalize projection vector for contrastive loss stability.
    z = F.normalize(z, dim=1)
    # Normalize encoder embedding for optional usage.
    h = F.normalize(h, dim=1)
    # Return encoder embedding and projection.
    return h, z


class ContrastiveClassifier(nn.Module):
  r'''
  Simple classifier that uses the ContrastiveEncoder for feature extraction.
  The encoder can be pretrained with a contrastive loss and then fine-tuned for classification.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    embDim (int): Embedding dimension from the ContrastiveEncoder.
  '''

  def __init__(self, inputSize: int, numClasses: int, embDim: int = 128):
    # Validate dimensions to prevent silent mismatches.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    # Call superclass constructor.
    super(ContrastiveClassifier, self).__init__()
    # Reuse the pre-trained contrastive encoder.
    self.encoder = ContrastiveEncoder(inputSize, embDim=embDim)
    # Linear classification head attached to embedding dimension.
    self.classifier = nn.Linear(embDim, numClasses)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Ensure input is float tensor.
    x = x.float()
    # Validate that input is 2D to prevent batch dimension loss.
    if (x.dim() != 2):
      raise ValueError(f"Expected 2D input (batchSize, inputSize), got {x.shape}")
    # Extract encoder outputs as a tuple.
    encoderOutputs = self.encoder(x)
    # Unpack the normalized embedding explicitly.
    embedding = encoderOutputs[0]
    # Apply classification head to the embedding.
    logits = self.classifier(embedding)
    # Return single logits tensor for supervised training.
    return logits


class GNNModel(nn.Module):
  """Graph Neural Network for tabular data with graph structure.
  Supports both adjacency matrix and edge index formats.
  Implements basic message passing with optional residual connections.
  For production use with complex graphs, consider PyTorch Geometric or DGL."""

  def __init__(
    self, inputSize: int, numClasses: int, hiddenDim: int = 128, numLayers: int = 2,
    dropout: float = 0.2, useResidual: bool = True
  ):
    # Validate input parameters to prevent silent failures.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    if (hiddenDim <= 0):
      raise ValueError(f"hiddenDim must be positive, got {hiddenDim}")
    if (numLayers < 1):
      raise ValueError(f"numLayers must be at least 1, got {numLayers}")
    # Call superclass constructor.
    super(GNNModel, self).__init__()
    # Store configuration for graph format detection.
    self.inputSize = inputSize
    self.hiddenDim = hiddenDim
    self.useResidual = useResidual
    # Initial feature projection layer.
    self.inputProj = nn.Linear(inputSize, hiddenDim)
    # Build graph convolution layers.
    # Build the graph convolutional layers as a ModuleList.
    self.gnnLayers = nn.ModuleList()
    for _ in range(numLayers):
      # Append a graph convolution layer to the list.
      self.gnnLayers.append(GraphConvLayer(hiddenDim, hiddenDim, dropout=dropout))
    # Classification head with layer normalization.
    self.classifier = nn.Sequential(
      nn.LayerNorm(hiddenDim),
      nn.Linear(hiddenDim, numClasses)
    )

  # Forward pass for the GNN accepting multiple graph formats.
  def forward(
    self, x: torch.Tensor, adj: torch.Tensor = None, edgeIndex: torch.Tensor = None,
    edgeWeight: torch.Tensor = None, batch: torch.Tensor = None
  ) -> torch.Tensor:
    # Ensure input is float tensor.
    x = x.float()
    # Handle 3D input by reshaping to 2D for GNN layers.
    # Preserve original input shape for later pooling logic.
    originalShape = x.shape
    # If input is batched and adjacency provided, construct block-diagonal adjacency.
    if (x.dim() == 3 and adj is not None):
      # Unpack batch size and node count from input shape.
      batchSize, numNodes, _ = x.shape
      # Reshape batched node features to a flat node list.
      x = x.reshape(-1, x.size(-1))
      # Build a list of identical adjacency matrices for each graph in batch.
      adjBlocks = [adj for _ in range(batchSize)]
      # Create a block-diagonal adjacency matrix from list of adjacencies.
      adj = torch.block_diag(*adjBlocks)
    elif (x.dim() == 3):
      # Reshape batched input into flat node list when adjacency not provided.
      x = x.reshape(-1, x.size(-1))
    # Project input features to hidden dimension.
    h = self.inputProj(x)
    # Determine graph format and normalize adjacency if needed.
    # Normalize adjacency matrix when provided for stable message passing.
    if (adj is not None):
      adj = self._normalizeAdjacency(adj)
    # Apply graph convolution layers with optional residual connections.
    for layer in self.gnnLayers:
      hPrev = h
      if (adj is not None):
        h = layer(h, adj=adj, edgeWeight=edgeWeight)
      elif (edgeIndex is not None):
        h = layer(h, edgeIndex=edgeIndex, edgeWeight=edgeWeight)
      else:
        h = layer(h, adj=None)
      if (self.useResidual and hPrev.shape == h.shape):
        h = h + hPrev
    # Pool node embeddings to graph-level if batch information provided.
    # Pool node embeddings to graph-level when batch vector provided.
    if (batch is not None):
      h = self._globalMeanPool(h, batch)
    elif (len(originalShape) == 3):
      # Reshape back to (batch, nodes, features) and average over nodes.
      h = h.reshape(originalShape[0], originalShape[1], -1).mean(dim=1)
    # Apply classification head to pooled graph embedding and return logits.
    return self.classifier(h)

  # Symmetric normalization of adjacency matrix with self-loops.
  def _normalizeAdjacency(self, adj: torch.Tensor) -> torch.Tensor:
    r'''
    Symmetric normalization of an adjacency matrix with added self-loops.

    This computes D^{-1/2} A D^{-1/2} after adding self-loops to A for
    numerical stability during graph convolution operations.

    Parameters:
      adj (torch.Tensor): Dense adjacency matrix of shape (N, N).

    Returns:
      torch.Tensor: Symmetrically normalized adjacency matrix.
    '''

    # Add self-loops to adjacency matrix for stability.
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    # Compute degree vector for normalization.
    degree = adj.sum(dim=1, keepdim=True)
    # Compute D^{-1/2} while avoiding infinities.
    degreeInvSqrt = degree.pow(-0.5)
    degreeInvSqrt = torch.where(torch.isinf(degreeInvSqrt), torch.zeros_like(degreeInvSqrt), degreeInvSqrt)
    # Perform symmetric normalization: D^{-1/2} A D^{-1/2}.
    normAdj = degreeInvSqrt * adj * degreeInvSqrt.transpose(0, 1)
    # Return normalized adjacency matrix.
    return normAdj

  # Global mean pooling of node embeddings to produce graph-level vectors.
  def _globalMeanPool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    r'''
    Global mean pooling of node embeddings into graph-level vectors.

    Parameters:
      x (torch.Tensor): Node embeddings of shape (numNodes, hiddenDim).
      batch (torch.Tensor): Vector of length numNodes indicating graph id per node.

    Returns:
      torch.Tensor: Pooled graph embeddings of shape (batchSize, hiddenDim).
    '''

    # Determine number of graphs in batch from batch vector.
    batchSize = batch.max().item() + 1
    # Initialize output tensor for aggregated embeddings.
    out = torch.zeros(batchSize, x.size(1), device=x.device)
    # Initialize count tensor for nodes per graph.
    count = torch.zeros(batchSize, device=x.device)
    # Aggregate node embeddings into their destination graph index.
    out.index_add_(0, batch, x)
    # Aggregate counts per graph with float32 dtype.
    count.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float32))
    # Ensure count has at least 1 to avoid division by zero.
    count = count.clamp(min=1.0)
    # Divide aggregated sum by counts to compute mean per graph.
    return out / count.unsqueeze(1)


class GraphConvLayer(nn.Module):
  r'''
  Single graph convolution layer with optional edge weights.

  Implements: h_i' = W_1 * h_i + W_2 * sum_{j in N(i)} (edge_weight_ij * h_j).

  Parameters:
    inChannels (int): Input feature dimensionality per node.
    outChannels (int): Output feature dimensionality per node.
    dropout (float): Dropout probability applied after normalization.
  '''

  def __init__(self, inChannels: int, outChannels: int, dropout: float = 0.2):
    # Validate channel dimensions.
    if (inChannels <= 0 or outChannels <= 0):
      raise ValueError(f"Channel dimensions must be positive, got in={inChannels}, out={outChannels}")
    # Call superclass constructor.
    super(GraphConvLayer, self).__init__()
    # Linear transformations for self and neighbor messages.
    self.linSelf = nn.Linear(inChannels, outChannels)
    self.linNeighbor = nn.Linear(inChannels, outChannels)
    # Activation and normalization.
    self.activation = nn.ReLU()
    self.norm = nn.LayerNorm(outChannels)
    self.dropout = nn.Dropout(dropout)

  # Forward pass performing message passing with optional edge formats.
  def forward(
    self, x: torch.Tensor, adj: torch.Tensor = None, edgeIndex: torch.Tensor = None,
    edgeWeight: torch.Tensor = None
  ) -> torch.Tensor:
    r'''
    Forward pass performing message passing with optional adjacency formats.

    Parameters:
      x (torch.Tensor): Node feature matrix of shape (numNodes, inChannels).
      adj (torch.Tensor or None): Dense adjacency matrix of shape (numNodes, numNodes).
      edgeIndex (torch.Tensor or None): Edge index tensor of shape (2, numEdges).
      edgeWeight (torch.Tensor or None): Optional edge weights of shape (numEdges,) or compatible.

    Returns:
      torch.Tensor: Updated node embeddings of shape (numNodes, outChannels).
    '''

    # Compute transformed self features.
    hSelf = self.linSelf(x)
    # Aggregate neighbor messages depending on adjacency representation.
    if (adj is not None):
      # Matrix multiplication aggregates neighbor features for dense adjacency.
      hNeighbor = torch.matmul(adj, x)
      # Transform aggregated neighbor features.
      hNeighbor = self.linNeighbor(hNeighbor)
    elif (edgeIndex is not None):
      # Use scatter-based aggregation when edge list format provided.
      hNeighbor = self._scatterAggregate(x, edgeIndex, edgeWeight)
      # Transform aggregated neighbor features.
      hNeighbor = self.linNeighbor(hNeighbor)
    else:
      # Use zero neighbor contribution when no graph structure provided.
      hNeighbor = torch.zeros_like(hSelf)
    # Combine self and neighbor contributions.
    # Combine self and neighbor contributions.
    h = hSelf + hNeighbor
    # Apply activation function.
    h = self.activation(h)
    # Apply layer normalization for stability.
    h = self.norm(h)
    # Apply dropout for regularization.
    h = self.dropout(h)
    # Return updated node embeddings.
    return h

  # Aggregate messages from source to target nodes using scatter-add.
  def _scatterAggregate(
    self, x: torch.Tensor, edgeIndex: torch.Tensor,
    edgeWeight: torch.Tensor = None
  ) -> torch.Tensor:
    r'''
    Aggregate messages from sources to target nodes using scatter-add.

    Parameters:
      x (torch.Tensor): Node features of shape (numNodes, inChannels).
      edgeIndex (torch.Tensor): Edge index tensor with shape (2, numEdges) in [src, dst] format.
      edgeWeight (torch.Tensor or None): Optional edge weights of shape (numEdges,).

    Returns:
      torch.Tensor: Aggregated neighbor features per node of shape (numNodes, inChannels).
    '''

    # Unpack source and destination index arrays from edgeIndex.
    src, dst = edgeIndex[0], edgeIndex[1]
    # Multiply source features by edge weights when provided.
    if (edgeWeight is not None):
      messages = x[src] * edgeWeight.unsqueeze(-1)
    else:
      # Use raw source features as messages when no edge weights.
      messages = x[src]
    # Initialize output tensor for aggregated messages.
    out = torch.zeros(x.size(0), x.size(1), device=x.device)
    # Scatter-add messages into destination node positions.
    out.index_add_(0, dst, messages)
    # Return aggregated neighbor messages per node.
    return out


class ResidualBlock(nn.Module):
  r'''
  Residual block for tabular data with skip connection.

  Implements: output = activation(norm(linear(activation(norm(linear(input)))))) + input

  Parameters:
    inFeatures (int): Number of input features.
    outFeatures (int): Number of output features.
    dropout (float): Dropout probability applied after activation.
  '''

  def __init__(self, inFeatures: int, outFeatures: int, dropout: float = 0.2):
    # Validate feature dimensions.
    if (inFeatures <= 0 or outFeatures <= 0):
      raise ValueError(f"Feature dimensions must be positive, got in={inFeatures}, out={outFeatures}")
    # Call superclass constructor.
    super(ResidualBlock, self).__init__()
    # Build the main transformation path.
    self.mainPath = nn.Sequential(
      nn.Linear(inFeatures, outFeatures),
      nn.BatchNorm1d(outFeatures),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(outFeatures, outFeatures),
      nn.BatchNorm1d(outFeatures),
    )
    # Build skip connection projection if dimensions differ.
    if (inFeatures != outFeatures):
      self.skipProjection = nn.Linear(inFeatures, outFeatures)
    else:
      self.skipProjection = nn.Identity()
    # Activation after residual addition.
    self.activation = nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inFeatures).
    # Returns: Tensor of shape (batch_size, outFeatures).
    # Compute main transformation path.
    main = self.mainPath(x)
    # Compute skip connection with projection if needed.
    skip = self.skipProjection(x)
    # Add residual connection and apply activation.
    out = self.activation(main + skip)
    # Return residual block output.
    return out


class ResNetTabularModel(nn.Module):
  r'''
  Residual network for tabular data with skip connections.

  Implements residual blocks: y = F(x) + x to enable deeper networks
  without vanishing gradients. Suitable for high-dimensional tabular data.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    blockChannels (list[int]): Channel sizes for residual blocks.
    dropout (float): Dropout probability applied in residual blocks.
  '''

  def __init__(self, inputSize: int, numClasses: int, blockChannels=None, dropout: float = 0.2):
    # Validate input parameters.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    # Call superclass constructor.
    super(ResNetTabularModel, self).__init__()
    # Set default block channels when not provided.
    if (blockChannels is None):
      blockChannels = [128, 128, 128]
    # Build initial feature projection layer.
    self.inputProj = nn.Linear(inputSize, blockChannels[0])
    # Build residual block stack.
    self.resBlocks = nn.ModuleList()
    currChannels = blockChannels[0]
    for outCh in blockChannels:
      # Append residual block with current and output channels.
      self.resBlocks.append(ResidualBlock(currChannels, outCh, dropout=dropout))
      currChannels = outCh
    # Classification head with layer normalization.
    self.classifier = nn.Sequential(
      nn.LayerNorm(currChannels),
      nn.Linear(currChannels, numClasses)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Returns logits: Tensor of shape (batch_size, numClasses).
    # Ensure input is float tensor.
    x = x.float()
    # Project input features to first block channel dimension.
    h = self.inputProj(x)
    # Apply residual blocks sequentially.
    for block in self.resBlocks:
      h = block(h)
    # Apply classification head and return logits.
    return self.classifier(h)


class AutoIntAttentionLayer(nn.Module):
  r'''
  Multi-head self-attention layer for feature interaction modeling.

  Implements attention over feature tokens to capture explicit
  high-order feature interactions at different subspaces.

  Parameters:
    embedDim (int): Embedding dimension per feature token.
    numHeads (int): Number of attention heads.
    dropout (float): Dropout probability applied after attention.
  '''

  def __init__(self, embedDim: int, numHeads: int, dropout: float = 0.1):
    # Validate attention parameters.
    if (embedDim % numHeads != 0):
      raise ValueError(f"embedDim {embedDim} must be divisible by numHeads {numHeads}")
    # Call superclass constructor.
    super(AutoIntAttentionLayer, self).__init__()
    # Store configuration for head dimension computation.
    self.embedDim = embedDim
    self.numHeads = numHeads
    self.headDim = embedDim // numHeads
    # Linear projections for query, key, value.
    self.qProj = nn.Linear(embedDim, embedDim)
    self.kProj = nn.Linear(embedDim, embedDim)
    self.vProj = nn.Linear(embedDim, embedDim)
    # Output projection after multi-head attention.
    self.outProj = nn.Linear(embedDim, embedDim)
    # Layer normalization and dropout.
    self.norm = nn.LayerNorm(embedDim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, numFeatures, embedDim).
    # Returns: Tensor of shape (batch_size, numFeatures, embedDim).
    batchSize, numFeatures, _ = x.shape
    # Compute query, key, value projections.
    q = self.qProj(x)
    k = self.kProj(x)
    v = self.vProj(x)
    # Reshape for multi-head attention: (batch, heads, features, headDim).
    q = q.view(batchSize, numFeatures, self.numHeads, self.headDim).transpose(1, 2)
    k = k.view(batchSize, numFeatures, self.numHeads, self.headDim).transpose(1, 2)
    v = v.view(batchSize, numFeatures, self.numHeads, self.headDim).transpose(1, 2)
    # Compute scaled dot-product attention.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.headDim ** 0.5)
    attnWeights = F.softmax(scores, dim=-1)
    attnWeights = self.dropout(attnWeights)
    # Apply attention weights to values.
    context = torch.matmul(attnWeights, v)
    # Concatenate heads and project back to embedDim.
    context = context.transpose(1, 2).contiguous().view(batchSize, numFeatures, self.embedDim)
    out = self.outProj(context)
    # Apply residual connection and layer normalization.
    out = self.norm(x + out)
    # Return attention-enhanced feature representations.
    return out


class AutoIntModel(nn.Module):
  r'''
  Attentional Interaction Network for explicit feature crossing.

  Uses multi-head self-attention to model feature interactions
  at different representation subspaces, followed by classification.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    embedDim (int): Embedding dimension per feature token.
    numAttentionLayers (int): Number of attention interaction layers.
    numHeads (int): Number of attention heads per layer.
    dropout (float): Dropout probability in attention layers.
  '''

  def __init__(
    self, inputSize: int, numClasses: int, embedDim: int = 32,
    numAttentionLayers: int = 3, numHeads: int = 2, dropout: float = 0.1
  ):
    # Validate input parameters.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    # Call superclass constructor.
    super(AutoIntModel, self).__init__()
    # Embed each feature independently into token space.
    self.featureEmbed = nn.Linear(1, embedDim)
    # Learnable positional encoding per feature position.
    self.posEmbed = nn.Parameter(torch.randn(1, inputSize, embedDim) * 0.02)
    # Stack of attention interaction layers.
    self.attentionLayers = nn.ModuleList()
    for _ in range(numAttentionLayers):
      self.attentionLayers.append(AutoIntAttentionLayer(embedDim, numHeads, dropout=dropout))
    # Classification head with pooling and layer normalization.
    self.classifier = nn.Sequential(
      nn.LayerNorm(embedDim),
      nn.Linear(embedDim, numClasses)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Returns logits: Tensor of shape (batch_size, numClasses).
    # Ensure input is float tensor.
    x = x.float()
    # Embed each feature: (batch, inputSize, 1) -> (batch, inputSize, embedDim).
    x = x.unsqueeze(-1)
    h = self.featureEmbed(x)
    # Add positional encoding for feature order awareness.
    h = h + self.posEmbed
    # Apply attention interaction layers sequentially.
    for layer in self.attentionLayers:
      h = layer(h)
    # Global average pool over feature positions.
    h = h.mean(dim=1)
    # Apply classification head and return logits.
    return self.classifier(h)


class SAINTIntersampleAttention(nn.Module):
  r'''
  Intersample attention module for batch-level feature refinement.

  Computes attention across samples in a batch to enable
  few-shot generalization and context-aware representations.

  Parameters:
    embedDim (int): Embedding dimension per feature token.
    numHeads (int): Number of attention heads.
    dropout (float): Dropout probability after attention.
  '''

  def __init__(self, embedDim: int, numHeads: int, dropout: float = 0.1):
    # Validate attention parameters.
    if (embedDim % numHeads != 0):
      raise ValueError(f"embedDim {embedDim} must be divisible by numHeads {numHeads}")
    # Call superclass constructor.
    super(SAINTIntersampleAttention, self).__init__()
    self.embedDim = embedDim
    self.numHeads = numHeads
    self.headDim = embedDim // numHeads
    # Linear projections for intersample attention.
    self.qProj = nn.Linear(embedDim, embedDim)
    self.kProj = nn.Linear(embedDim, embedDim)
    self.vProj = nn.Linear(embedDim, embedDim)
    self.outProj = nn.Linear(embedDim, embedDim)
    self.norm = nn.LayerNorm(embedDim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, numFeatures, embedDim).
    # Returns: Tensor of shape (batch_size, numFeatures, embedDim).
    batchSize, numFeatures, _ = x.shape
    # Transpose to compute attention across batch dimension.
    # Reshape: (batch, features, embedDim) -> (features, batch, embedDim).
    xT = x.transpose(0, 1)
    # Compute projections for intersample attention.
    q = self.qProj(xT)
    k = self.kProj(xT)
    v = self.vProj(xT)
    # Reshape for multi-head: (features, heads, batch, headDim).
    q = q.view(numFeatures, batchSize, self.numHeads, self.headDim).transpose(1, 2)
    k = k.view(numFeatures, batchSize, self.numHeads, self.headDim).transpose(1, 2)
    v = v.view(numFeatures, batchSize, self.numHeads, self.headDim).transpose(1, 2)
    # Compute scaled dot-product attention across samples.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.headDim ** 0.5)
    attnWeights = F.softmax(scores, dim=-1)
    attnWeights = self.dropout(attnWeights)
    # Apply attention and reshape back.
    context = torch.matmul(attnWeights, v)
    context = context.transpose(1, 2).contiguous().view(numFeatures, batchSize, self.embedDim)
    out = self.outProj(context)
    # Transpose back to (batch, features, embedDim).
    out = out.transpose(0, 1)
    # Apply residual connection and normalization.
    out = self.norm(x + out)
    # Return intersample-refined representations.
    return out


class SAINTModel(nn.Module):
  r'''
  SAINT: Self-Attention and Intersample Attention Transformer.

  Combines intra-sample feature attention with inter-sample
  attention across the batch for few-shot generalization.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    embedDim (int): Token embedding dimension.
    numHeads (int): Number of attention heads.
    numFeatureLayers (int): Number of intra-sample attention layers.
    numIntersampleLayers (int): Number of inter-sample attention layers.
    useIntersampleAttention (bool): Enable batch-level attention.
    dropout (float): Dropout probability in attention layers.
  '''

  def __init__(
    self, inputSize: int, numClasses: int, embedDim: int = 32,
    numHeads: int = 2, numFeatureLayers: int = 2, numIntersampleLayers: int = 1,
    useIntersampleAttention: bool = True, dropout: float = 0.1
  ):
    # Validate input parameters.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    # Call superclass constructor.
    super(SAINTModel, self).__init__()
    # Embed each feature independently.
    self.featureEmbed = nn.Linear(1, embedDim)
    self.posEmbed = nn.Parameter(torch.randn(1, inputSize, embedDim) * 0.02)
    # Intra-sample feature attention layers.
    self.featureAttention = nn.ModuleList()
    for _ in range(numFeatureLayers):
      self.featureAttention.append(AutoIntAttentionLayer(embedDim, numHeads, dropout=dropout))
    # Intersample attention layers (optional).
    self.useIntersample = useIntersampleAttention
    if (useIntersampleAttention):
      self.intersampleAttention = nn.ModuleList()
      for _ in range(numIntersampleLayers):
        self.intersampleAttention.append(SAINTIntersampleAttention(embedDim, numHeads, dropout=dropout))
    # Classification head.
    self.classifier = nn.Sequential(
      nn.LayerNorm(embedDim),
      nn.Linear(embedDim, numClasses)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Returns logits: Tensor of shape (batch_size, numClasses).
    # Ensure input is float tensor.
    x = x.float()
    # Embed features: (batch, inputSize, 1) -> (batch, inputSize, embedDim).
    x = x.unsqueeze(-1)
    h = self.featureEmbed(x)
    h = h + self.posEmbed
    # Apply intra-sample feature attention.
    for layer in self.featureAttention:
      h = layer(h)
    # Apply intersample attention if enabled.
    if (self.useIntersample):
      for layer in self.intersampleAttention:
        h = layer(h)
    # Global average pool over features.
    h = h.mean(dim=1)
    # Apply classification head.
    return self.classifier(h)


class ShapeFunctionNet(nn.Module):
  r'''
  Small subnetwork that processes a single feature for additive modeling.

  Parameters:
    hiddenSize (int): Hidden dimension of the shape function network.
    outputDim (int): Output dimension (typically 1 for additive models).
    dropout (float): Dropout probability.
  '''

  def __init__(self, hiddenSize: int, outputDim: int = 1, dropout: float = 0.1):
    # Validate dimensions.
    if (hiddenSize <= 0 or outputDim <= 0):
      raise ValueError(f"Dimensions must be positive, got hidden={hiddenSize}, out={outputDim}")
    # Call superclass constructor.
    super(ShapeFunctionNet, self).__init__()
    # Build shape function subnetwork.
    self.net = nn.Sequential(
      nn.Linear(1, hiddenSize),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hiddenSize, hiddenSize),
      nn.ReLU(),
      nn.Linear(hiddenSize, outputDim)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, 1).
    # Returns: Tensor of shape (batch_size, outputDim).
    return self.net(x)


class NeuralAdditiveModel(nn.Module):
  r'''
  Interpretable neural additive model for tabular data.

  Each feature is processed by an independent subnetwork;
  outputs are summed for final prediction enabling interpretability.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    shapeFunctionHidden (int): Hidden size per feature subnetwork.
    dropout (float): Dropout probability in shape functions.
  '''

  def __init__(self, inputSize: int, numClasses: int, shapeFunctionHidden: int = 32, dropout: float = 0.1):
    # Validate input parameters.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    # Call superclass constructor.
    super(NeuralAdditiveModel, self).__init__()
    # Store input size for reference.
    self.inputSize = inputSize
    self.numClasses = numClasses
    # Create independent shape function for each feature.
    self.shapeFunctions = nn.ModuleList()
    for _ in range(inputSize):
      self.shapeFunctions.append(ShapeFunctionNet(shapeFunctionHidden, outputDim=numClasses, dropout=dropout))
    # Optional bias term for the additive model.
    self.bias = nn.Parameter(torch.zeros(numClasses))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Returns logits: Tensor of shape (batch_size, numClasses).
    # Ensure input is float tensor.
    x = x.float()
    # Initialize output logits accumulator.
    batchSize = x.size(0)
    logits = torch.zeros(batchSize, self.numClasses, device=x.device)
    # Process each feature through its shape function and accumulate.
    for i in range(self.inputSize):
      # Extract single feature column: (batch, 1).
      featureCol = x[:, i:i + 1]
      # Compute shape function output: (batch, numClasses).
      contribution = self.shapeFunctions[i](featureCol)
      # Accumulate additive contribution.
      logits = logits + contribution
    # Add bias term and return final logits.
    return logits + self.bias


class CrossLayer(nn.Module):
  r'''
  Cross network layer for explicit polynomial feature crossing.

  Implements: x_{l+1} = x_0 * (x_l^T * w_l + b_l) + x_l
  where x_0 is the original input and w_l, b_l are learned parameters.

  Parameters:
    inputSize (int): Number of input features (must match original input).
  '''

  def __init__(self, inputSize: int):
    # Validate input size.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    # Call superclass constructor.
    super(CrossLayer, self).__init__()
    # Store input size for cross product computation.
    self.inputSize = inputSize
    # Learnable weight and bias for crossing transformation.
    self.weight = nn.Parameter(torch.randn(inputSize, 1) * 0.01)
    self.bias = nn.Parameter(torch.zeros(inputSize, 1))

  def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
    # Expects x0: Original input (batch, inputSize).
    # Expects xl: Current layer output (batch, inputSize).
    # Returns: Crossed output (batch, inputSize).
    # Compute outer product term: xl^T * w -> (batch, 1).
    crossTerm = torch.matmul(xl, self.weight)
    # Apply bias and multiply by original input.
    crossOut = x0 * (crossTerm + self.bias.transpose(0, 1))
    # Add residual connection from current layer.
    return crossOut + xl


class DeepCrossNetwork(nn.Module):
  r'''
  Deep Cross Network for explicit low- and high-order feature crossing.

  Combines a cross network (polynomial feature interactions) with
  a deep network (non-linear transformations) in parallel.

  Parameters:
    inputSize (int): Number of input features.
    numClasses (int): Number of output classes.
    crossLayers (int): Number of cross network layers.
    deepHiddenSizes (list[int]): Hidden sizes for deep network branch.
    dropout (float): Dropout probability in deep network.
  '''

  def __init__(
    self, inputSize: int, numClasses: int, crossLayers: int = 3,
    deepHiddenSizes=None, dropout: float = 0.2
  ):
    # Validate input parameters.
    if (inputSize <= 0):
      raise ValueError(f"inputSize must be positive, got {inputSize}")
    if (numClasses < 1):
      raise ValueError(f"numClasses must be at least 1, got {numClasses}")
    # Call superclass constructor.
    super(DeepCrossNetwork, self).__init__()
    # Set default deep network hidden sizes.
    if (deepHiddenSizes is None):
      deepHiddenSizes = [256, 128]
    # Build cross network stack.
    self.crossLayers = nn.ModuleList()
    for _ in range(crossLayers):
      self.crossLayers.append(CrossLayer(inputSize))
    # Build deep network branch.
    deepLayers = []
    currSize = inputSize
    for h in deepHiddenSizes:
      deepLayers.append(nn.Linear(currSize, h))
      deepLayers.append(nn.ReLU())
      deepLayers.append(nn.Dropout(dropout))
      currSize = h
    self.deepNet = nn.Sequential(*deepLayers)
    # Final concatenation and classification layer.
    self.classifier = nn.Linear(inputSize + currSize, numClasses)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Expects x: Tensor of shape (batch_size, inputSize).
    # Returns logits: Tensor of shape (batch_size, numClasses).
    # Ensure input is float tensor.
    x = x.float()
    # Store original input for cross network.
    x0 = x
    # Apply cross network layers sequentially.
    xCross = x0
    for layer in self.crossLayers:
      xCross = layer(x0, xCross)
    # Apply deep network branch.
    xDeep = self.deepNet(x)
    # Concatenate cross and deep representations.
    combined = torch.cat([xCross, xDeep], dim=1)
    # Apply final classification layer.
    return self.classifier(combined)


def GetModel(name: str, inputSize: int, numClasses: int) -> nn.Module:
  r'''
  Factory that returns a model instance by its CamelCase name key.

  Parameters:
    name (str): CamelCase model identifier string.
    inputSize (int): Number of input features for the model.
    numClasses (int): Number of output classes for classification models.

  Returns:
    nn.Module: Instantiated PyTorch model corresponding to `name`.
  '''

  # Select and instantiate a model by its CamelCase name key.
  if (name == "MLPModel"):
    return MLPModel(inputSize, numClasses)
  if (name == "TCNModel"):
    return TCNModel(inputSize, numClasses)
  if (name == "TabTransformerModel"):
    return TabTransformerModel(inputSize, numClasses)
  if (name == "FTTransformer"):
    return FTTransformerModel(inputSize, numClasses)
  if (name == "GANDALF"):
    return GANDALFModel(inputSize, numClasses)
  if (name == "HybridCNNTransformer"):
    return HybridCNNTransformerModel(inputSize, numClasses)
  if (name == "VAEAnomalyDetector"):
    return VAEAnomalyDetector(inputSize)
  if (name == "ContrastiveEncoder"):
    return ContrastiveEncoder(inputSize)
  if (name == "VAEClassifier"):
    return VAEClassifier(inputSize, numClasses)
  if (name == "ContrastiveClassifier"):
    return ContrastiveClassifier(inputSize, numClasses)
  if (name == "GNNModel"):
    return GNNModel(inputSize, numClasses)
  if (name == "ResNetTabular"):
    return ResNetTabularModel(inputSize, numClasses)
  if (name == "AutoInt"):
    return AutoIntModel(inputSize, numClasses)
  if (name == "SAINT"):
    return SAINTModel(inputSize, numClasses)
  if (name == "NeuralAdditive"):
    return NeuralAdditiveModel(inputSize, numClasses)
  if (name == "DeepCross"):
    return DeepCrossNetwork(inputSize, numClasses)
  # Fallback to MLPModel when name not recognized.
  return MLPModel(inputSize, numClasses)


if (__name__ == "__main__"):
  # Run comprehensive validation of all models.
  print("\n" + "=" * 60)
  print("PyTorchTabularModelsZoo Validation Suite...")
  print("=" * 60)
  print("Configuration: inputSize=100, numClasses=10, batchSize=32")
  print("-" * 60)

  # Define test configurations for each model.
  # Format: (modelName, inputSize, numClasses, forwardFn, expectedOutputShape, extraArgs).
  # All models tested with 100 features and 10 classes for consistency.
  testConfigs = [
    # Standard classification models.
    ("MLPModel", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("TCNModel", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("TabTransformerModel", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("FTTransformer", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("GANDALF", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("HybridCNNTransformer", 100, 10, lambda m, x: m(x), (32, 10), {}),
    # New architecture additions.
    ("ResNetTabular", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("AutoInt", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("SAINT", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("NeuralAdditive", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("DeepCross", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("VAEClassifier", 100, 10, lambda m, x: m(x), (32, 10), {}),
    ("ContrastiveClassifier", 100, 10, lambda m, x: m(x), (32, 10), {}),
    # GNN model with adjacency matrix (node classification: output per node).
    ("GNNModel", 100, 10, lambda m, x: m(x, adj=torch.randint(0, 2, (100, 100)).float()), (100, 10), {}),
  ]

  # Track overall validation results.
  passedCount = 0
  failedCount = 0

  # Iterate over each test configuration.
  for config in testConfigs:
    # Unpack test configuration tuple.
    modelName, inputSize, numClasses, forwardFn, expectedShape, extraArgs = config
    try:
      # Instantiate the model via the GetModel factory.
      model = GetModel(modelName, inputSize=inputSize, numClasses=numClasses)
      # Choose batch size for the test input.
      batchSize = 32 if (modelName != "GNNModel") else 100
      # Create a random input tensor for the model.
      x = torch.randn(batchSize, inputSize)
      # Execute the forward function and capture output.
      output = forwardFn(model, x)
      # Compare output shape against expected shape and report.
      if (output.shape == expectedShape):
        print(f"✅ {modelName:25s}: PASSED (output: {list(output.shape)})")
        passedCount += 1
      else:
        print(f"❌ {modelName:25s}: FAILED (expected {expectedShape}, got {list(output.shape)})")
        failedCount += 1
    except Exception as e:
      # Report any exceptions that occurred during testing.
      print(f"❌ {modelName:25s}: ERROR - {type(e).__name__}: {e}")
      failedCount += 1

  # Print summary of validation results.
  print("\n" + "-" * 60)
  print(f"VALIDATION SUMMARY: {passedCount} passed, {failedCount} failed")
  print("-" * 60)

  # Report final status.
  if (failedCount == 0):
    print("🎉 All models validated successfully! PyTorchTabularModelsZoo is production-ready.")
  else:
    print(f"⚠️  {failedCount} model(s) failed validation. Please review errors above.")

  # Additional demonstration: show parameter counts for all models.
  print("\n" + "=" * 60)
  print("MODEL PARAMETER COUNTS (inputSize=100, numClasses=10)")
  print("=" * 60)
  # List all models in the zoo for comprehensive parameter reporting.
  allModels = [
    "MLPModel",
    "TCNModel",
    "TabTransformerModel",
    "FTTransformer",
    "GANDALF",
    "HybridCNNTransformer",
    "ResNetTabular",
    "AutoInt",
    "SAINT",
    "NeuralAdditive",
    "DeepCross",
    "VAEClassifier",
    "ContrastiveClassifier",
    "GNNModel",
  ]
  for name in allModels:
    try:
      # Instantiate model with consistent configuration.
      model = GetModel(name, inputSize=100, numClasses=10)
      # Count trainable parameters.
      paramCount = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"{name:25s}: {paramCount:>10,} parameters")
    except Exception as e:
      print(f"{name:25s}: ERROR - {e}")

  print("\n" + "=" * 60)
  print("PyTorchTabularModelsZoo validation complete.")
  print("=" * 60)
