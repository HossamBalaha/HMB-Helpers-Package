import torch, tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer


class EmbeddingsToTextModel(nn.Module):
  r'''
  A PyTorch model that generates text from input features using a pre-trained T5 model.
  The model includes a feature projection layer to transform input features into
  a suitable format for text generation. It also incorporates learnable prompt tokens
  to enhance the generation process.

  Attributes:
    t5 (T5ForConditionalGeneration): Pre-trained T5 model for text generation.
    tokenizer (T5Tokenizer): Tokenizer for the T5 model.
    featureProjection (nn.Sequential): Sequential layer for projecting input features.
    toT5Hidden (nn.Linear): Linear layer to project features to T5's hidden size.
    promptEmbeddings (nn.Parameter): Learnable prompt embeddings for the model.
    numPromptTokens (int): Number of learnable prompt tokens.
    generationMaxLength (int): Maximum length for generated text sequences.

  Examples
  --------
  .. code-block:: python

    import HMB.EmbeddingsToTextHelper as e2tt

    # Initialize the model with default parameters.
    model = e2tt.EmbeddingsToTextModel(
      tokenizeModelName="t5-small",
      inputFeatureDim=6144,
      hiddenDim=512,
      generationMaxLength=512,
      dropoutRatio=0.1,
      numPromptTokens=5
    )

    # Print the model architecture.
    print(model)
    # Example input features (batch size of 2, feature dimension of 6144).
    exampleFeatures = torch.randn(2, 6144)
    # Generate text from the example features.
    generatedIds = model.generate(exampleFeatures, max_length=50)
    # Decode the generated text to a human-readable format.
    generatedText = model.tokenizer.batch_decode(generatedIds, skip_special_tokens=True)
    print(generatedText)
  '''

  def __init__(
    self,
    tokenizeModelName="t5-small",  # Name of the pre-trained T5 model to use.
    inputFeatureDim=6144,  # Dimension of the input feature vector.
    hiddenDim=512,  # Hidden dimension for the feature projection layers.
    generationMaxLength=512,  # Maximum length for the generated text.
    dropoutRatio=0.1,  # Dropout ratio for regularization.
    numPromptTokens=5,  # Number of learnable prompt tokens.
  ):
    r'''
    Initialize the EmbeddingsToTextModel for generating text from features.
    This model uses a pre-trained T5 model and adds a feature projection layer
    to transform input features into a format suitable for text generation.

    Parameters:
      tokenizeModelName (str): Name of the pre-trained T5 model to use (default: "t5-small").
      inputFeatureDim (int): Dimension of the input feature vector (default: 6144).
      hiddenDim (int): Hidden dimension for the feature projection layers (default: 512).
      generationMaxLength (int): Maximum length for the generated text (default: 512).
      dropoutRatio (float): Dropout ratio for regularization (default: 0.1).
      numPromptTokens (int): Number of learnable prompt tokens (default: 5).
    '''

    # Calls the parent class constructor to initialize nn.Module.
    super(EmbeddingsToTextModel, self).__init__()

    # Loads the pre-trained T5 model for conditional text generation.
    self.t5 = T5ForConditionalGeneration.from_pretrained(tokenizeModelName)
    # Loads the tokenizer for the T5 model, using legacy mode for compatibility.
    self.tokenizer = T5Tokenizer.from_pretrained(tokenizeModelName, legacy=True)

    # Checks if the tokenizer does not have a padding token set.
    if (not self.tokenizer.pad_token):
      # Sets the padding token to the end-of-sequence token for proper handling.
      self.tokenizer.pad_token = self.tokenizer.eos_token
    # Checks if the padding token ID is not set.
    if (self.tokenizer.pad_token_id is None):
      # Sets the padding token ID to the end-of-sequence token ID if not set.
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # Defines a sequential feature projection layer to transform input features.
    self.featureProjection = nn.Sequential(
      nn.Linear(inputFeatureDim, hiddenDim * 2),  # Projects input features to a higher dimension.
      nn.ReLU(),  # Applies ReLU activation for non-linearity.
      nn.Dropout(dropoutRatio),  # Applies dropout for regularization.
      nn.Linear(hiddenDim * 2, hiddenDim),  # Projects features down to hiddenDim.
      nn.ReLU(),  # Applies ReLU activation again.
      nn.Dropout(dropoutRatio)  # Applies dropout for regularization.
    )

    # Projects the hidden features to match the T5 model's hidden size.
    self.toT5Hidden = nn.Linear(hiddenDim, self.t5.config.d_model)

    # Creates learnable prompt embeddings for the model.
    self.numPromptTokens = numPromptTokens
    self.promptEmbeddings = nn.Parameter(torch.randn(numPromptTokens, self.t5.config.d_model))

    # Sets the maximum length for generated text sequences.
    self.generationMaxLength = generationMaxLength

  # Defines the forward pass for the model.
  def forward(
    self,
    features,  # Input features to be projected and processed.
    input_ids=None,  # Input token IDs for the T5 model (optional).
    attention_mask=None,  # Attention mask to indicate which tokens are valid (optional).
    labels=None,  # Labels for the causal language modeling task (optional, default: None).
  ):
    r'''
    Forward pass through the EmbeddingsToTextModel.
    This method processes the input features, projects them to T5's hidden dimension,
    and generates text using the T5 model.

    Parameters:
      features (torch.Tensor): Input features to be projected and processed.
      input_ids (torch.Tensor, optional): Input token IDs for the T5 model (default: None).
      attentionMask (torch.Tensor, optional): Attention mask to indicate which tokens are valid (default: None).
      labels (torch.Tensor, optional): Labels for the causal language modeling task (default: None).

    Returns:
      transformers.modeling_outputs.Seq2SeqLMOutput: Output from the T5 model containing logits and loss. It includes the generated text and loss if labels are provided.
    '''

    # Gets the batch size from the input features.
    B = features.shape[0]
    # Projects the input features through the feature projection layer.
    projected = self.featureProjection(features)  # (B, hiddenDim).
    # Projects the features to match the T5 hidden dimension.
    encoderHidden = self.toT5Hidden(projected)  # (B, d_model).

    # Expands the feature to a sequence by repeating to match prompt length.
    encoderHidden = encoderHidden.unsqueeze(1).expand(-1, self.numPromptTokens, -1)  # (B, numPromptTokens, d_model).

    # Uses the expanded encoder hidden states as input embeddings.
    inputsEmbeds = encoderHidden  # Or: self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1) + encoder_hidden.

    # Creates an attention mask of ones for all prompt tokens.
    attentionMask = torch.ones(B, self.numPromptTokens, device=features.device)

    # Performs a forward pass through the T5 model.
    outputs = self.t5(
      inputs_embeds=inputsEmbeds,  # Uses the projected features as input embeddings.
      attention_mask=attentionMask,  # Uses the attention mask for the prompt tokens.
      decoder_input_ids=input_ids,  # Input IDs for the decoder (if provided).
      labels=labels,  # Labels for the causal language modeling task (if provided).
    )

    # Returns the outputs from the T5 model.
    return outputs

  # Defines the method to generate text from input features.
  def generate(
    self,
    features,  # Input features to be projected and processed.
    **kwargs  # Additional keyword arguments for the generation method.
  ):
    r'''
    Generate text from input features using the T5 model.
    This method projects the input features and generates text based on the provided parameters.

    Parameters:
      features (torch.Tensor): Input features to be projected and processed.
      **kwargs: Additional keyword arguments for the generation method.

    Returns:
      torch.Tensor: Generated text token IDs.
    '''

    # Disables gradient calculation for generation.
    with torch.no_grad():
      # Gets the batch size from the input features.
      B = features.shape[0]

      # Projects the input features through the feature projection layer.
      projected = self.featureProjection(features)
      # Projects the features to match the T5 hidden dimension.
      encoderHidden = self.toT5Hidden(projected)

      # Expands the encoder hidden states to match the prompt length.
      encoderHidden = encoderHidden.unsqueeze(1).expand(-1, self.numPromptTokens, -1)

      # Creates an attention mask of ones for all prompt tokens.
      attentionMask = torch.ones(B, self.numPromptTokens, device=features.device)

      # Generates text token IDs using the T5 model's generate method.
      generatedIds = self.t5.generate(
        inputs_embeds=encoderHidden,  # Uses the projected features as input embeddings.
        attention_mask=attentionMask,  # Uses the attention mask for the prompt tokens.
        max_length=self.generationMaxLength,  # Sets the maximum length for generated text.
        pad_token_id=self.tokenizer.pad_token_id,  # Ensures proper handling of padding tokens.
        eos_token_id=self.tokenizer.eos_token_id,  # Ensures proper handling of end-of-sequence tokens.
        **kwargs,  # Passes any additional generation parameters.
      )
    # Returns the generated token IDs.
    return generatedIds


# Defines the function to train the EmbeddingsToTextModel.
def TrainModel(
  model,  # Instance of the EmbeddingsToTextModel to be trained.
  trainLoader,  # DataLoader for training data.
  valLoader,  # DataLoader for validation data.
  numEpochs=10,  # Number of epochs to train the model (default: 10).
  learningRate=1e-4,  # Learning rate for the optimizer (default: 1e-4).
  optimizerType="adamw",  # Type of optimizer to use (default: "adamw" for AdamW).
  # Path to save the best model state (default: "BestModel.pth" in the current directory).
  modelStoragePath="BestModel.pth",  # Path to save the best model state.
  verbose=False  # Whether to print verbose output during training (default: False).
):
  r'''
  Train the EmbeddingsToTextModel using the provided training and validation data loaders.
  This function performs the training loop, including forward and backward passes,
  loss computation, and optimization steps. It also evaluates the model on the validation set
  after each epoch to monitor performance and saves the best model state based on validation loss.

  Parameters:
    model (EmbeddingsToTextModel): Instance of the EmbeddingsToTextModel to be trained.
    trainLoader (DataLoader): DataLoader for training data.
    valLoader (DataLoader): DataLoader for validation data.
    numEpochs (int): Number of epochs to train the model (default: 10).
    learningRate (float): Learning rate for the optimizer (default: 1e-4).
    optimizerType (str): Type of optimizer to use for training (default: "adamw").
    modelStoragePath (str): Path to save the best model state (default: "BestModel.pth").
    verbose (bool): Whether to print verbose output during training (default: False).

  Examples
  --------
  .. code-block:: python

    import HMB.EmbeddingsToTextHelper as e2tt

    # Initialize the model with default parameters.
    model = e2tt.EmbeddingsToTextModel(
      tokenizeModelName="t5-small",
      inputFeatureDim=6144,
      hiddenDim=512,
      generationMaxLength=512,
      dropoutRatio=0.1,
      numPromptTokens=5,
    )
    # Assume trainLoader and valLoader are predefined DataLoader instances.
    trainLoader = ...  # Your training DataLoader here.
    valLoader = ...    # Your validation DataLoader here.
    # Print the model architecture.
    print(model)

    # Train the model using the training and validation data loaders.
    e2tt.TrainModel(
      model=model,
      trainLoader=trainLoader,
      valLoader=valLoader,
      numEpochs=10,
      learningRate=1e-4,
      optimizerType="adamw",
      modelStoragePath="BestModel.pth",
      verbose=True
    )
  '''

  # Selects the device for training (GPU if available, otherwise CPU).
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Moves the model to the selected device.
  model.to(device)

  # Chooses the optimizer based on the specified type.
  if (optimizerType.lower() == "adamw"):
    # Uses AdamW optimizer for training.
    optimizer = optim.AdamW(model.parameters(), lr=learningRate)
  elif (optimizerType.lower() == "adam"):
    # Uses Adam optimizer for training.
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
  else:
    # Raises an error if the optimizer type is unsupported.
    raise ValueError(f"Unsupported optimizer type: {optimizerType}. Use 'adamw' or 'adam'.")

  # Initializes the best validation loss to infinity.
  bestValLoss = float("inf")

  # Main training loop for the specified number of epochs.
  for epoch in range(numEpochs):
    # Sets the model to training mode.
    model.train()

    # Initialize variables to track training loss and number of batches.
    totalTrainLoss = 0.0

    # Create a progress bar for the training loop.
    trainPbar = tqdm.tqdm(trainLoader, desc=f"Epoch {epoch + 1}/{numEpochs} [Train]")
    for batch in trainPbar:
      features = batch["features"].to(device)
      inputIds = batch["input_ids"].to(device)  # decoder input.
      attentionMask = batch["attention_mask"].to(device)
      labels = inputIds.clone()

      # Ignore padding in loss computation.
      labels[labels == model.tokenizer.pad_token_id] = -100

      # Apply zero grad to clear previous gradients.
      optimizer.zero_grad()

      # Forward pass through the model using PyTorch's __call__ mechanism.
      outputs = model.forward(
        features=features,
        input_ids=inputIds,
        attention_mask=attentionMask,
        labels=labels,
      )

      # Compute the loss from the model outputs.
      loss = outputs.loss
      # Apply backward pass to compute gradients.
      loss.backward()
      # Update the model parameters using the optimizer.
      optimizer.step()

      # Update the progress bar with the current loss.
      totalTrainLoss += loss.item()
      trainPbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate the average training loss for the epoch.
    avgTrainLoss = totalTrainLoss / len(trainLoader)
    if (verbose):
      print(f"Epoch {epoch + 1}/{numEpochs} - Average Training Loss: {avgTrainLoss:.4f}")

    # Validation phase.
    model.eval()

    # Initialize variables to track validation loss and number of batches.
    totalValLoss = 0.0

    # Create a progress bar for the validation loop.
    valPbar = tqdm.tqdm(valLoader, desc=f"Epoch {epoch + 1}/{numEpochs} [Val]")

    # Disable gradient computation for validation to save memory and computation.
    with torch.no_grad():
      for batch in valPbar:
        features = batch["features"].to(device)
        inputIds = batch["input_ids"].to(device)
        attentionMask = batch["attention_mask"].to(device)
        labels = inputIds.clone()

        # Ignore padding in loss computation.
        labels[labels == model.tokenizer.pad_token_id] = -100

        # Forward pass through the model using PyTorch's __call__ mechanism.
        outputs = model.forward(
          features=features,
          input_ids=inputIds,
          attention_mask=attentionMask,
          labels=labels,
        )

        # Compute the loss from the model outputs.
        loss = outputs.loss
        totalValLoss += loss.item()
        valPbar.set_postfix({"loss": loss.item()})

    # Calculate the average validation loss for the epoch.
    avgValLoss = totalValLoss / len(valLoader)
    if (verbose):
      print(f"Epoch {epoch + 1}/{numEpochs} - Average Validation Loss: {avgValLoss:.4f}")

    # Check if the current validation loss is the best so far.
    if (avgValLoss < bestValLoss):
      # If it is, save the model state as the best model.
      bestValLoss = avgValLoss
      if (verbose):
        print(f"New best validation loss: {bestValLoss:.4f}. Saving model state...")
      torch.save(model.state_dict(), modelStoragePath)

      # Generate sample text from the model using the first batch of features.
      sampleFeatures = features[:1]  # Take the first feature from the batch.
      generated = model.generate(sampleFeatures)
      # Decode the generated text to a human-readable format.
      decodedText = model.tokenizer.decode(generated[0], skip_special_tokens=True)
      if (verbose):
        print(f"Sample generated text: {decodedText}")
        print(model.tokenizer.batch_decode(generated, skip_special_tokens=True))

  if (verbose):
    print(f"Training completed. Best validation loss: {bestValLoss:.4f}.")
    print(f"Model saved to {modelStoragePath}.")
