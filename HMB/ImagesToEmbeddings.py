import tqdm
import os
import pickle
import torch
import numpy as np
from HMB.Initializations import IMAGE_SUFFIXES


class TransformersEmbeddingModel(object):
  r'''
  A class to extract embeddings from images using pre-trained models from the Hugging Face Transformers library.

  .. math::

    \mathrm{embedding} = \mathrm{model}(I)_{\mathrm{CLS}}

  where the ``CLS`` token (or first token) is used as the image-level embedding.

  .. note::
    The model and processor are loaded lazily when the first image is processed,
    to avoid unnecessary memory usage if the model is not needed immediately.

  Examples
  --------
  .. code-block:: python

    from HMB.ImagesToEmbeddings import TransformersEmbeddingModel
    import torch

    # Example 1: Phikon-v2 (Standard CLS token extraction).
    modelName = "owkin/phikon-v2"
    targetDevice = "cuda" if torch.cuda.is_available() else "cpu"
    embeddingModel = TransformersEmbeddingModel(modelName, targetDevice)
    embedding = embeddingModel.GetEmbedding("path/to/image.jpg", normalizeEmbedding=True)
    print(embedding.shape)

    # Example 2: Maira2 (CLS + Mean Patch Tokens).
    modelName = "microsoft/rad-dino-maira-2"
    embeddingModel = TransformersEmbeddingModel(modelName, targetDevice)
    embedding = embeddingModel.GetEmbedding("path/to/image.jpg", usePatchTokens=True)
    print(embedding.shape)

    # Example 3: Nomic Vision (Normalized CLS token).
    modelName = "nomic-ai/nomic-embed-vision-v1.5"
    embeddingModel = TransformersEmbeddingModel(modelName, targetDevice)
    embedding = embeddingModel.GetEmbedding("path/to/image.jpg", normalizeEmbedding=True)
    print(embedding.shape)
  '''

  def __init__(self, modelName, targetDevice):
    r'''
    Initialize the TransformersEmbeddingModel with a specified model name and device.

    Parameters:
      modelName (str): Name of the pre-trained model to load from Hugging Face.
      targetDevice (str or torch.device): Device to run the model on (e.g., "cuda", "cpu").
    '''

    # Assign the model name to the instance variable.
    self.modelName = modelName
    # Assign the target device to the instance variable.
    self.targetDevice = targetDevice
    # Initialize the model attribute to None.
    self.model = None
    # Initialize the processor attribute to None.
    self.processor = None

  def LoadModel(self):
    r'''
    Load the pre-trained model and processor from the specified model name.

    Returns:
      model (torch.nn.Module): The loaded pre-trained model.
      processor (transformers.AutoImageProcessor): The loaded image processor.
    '''

    # Import the AutoImageProcessor and AutoModel from transformers.
    from transformers import AutoImageProcessor, AutoModel
    # Load the processor and model from Hugging Face.
    self.processor = AutoImageProcessor.from_pretrained(self.modelName, use_fast=True)
    # Load the pre-trained model from Hugging Face.
    self.model = AutoModel.from_pretrained(self.modelName)
    # Set the model to evaluation mode.
    self.model.eval()
    # Move the model to the specified device with float32 precision.
    self.model.to(self.targetDevice, dtype=torch.float32)
    # Return the loaded model and processor.
    return self.model, self.processor

  def GetEmbedding(self, imagePath, normalizeEmbedding=False, usePatchTokens=False):
    r'''
    Extract embedding from an image using the loaded model and processor.

    Parameters:
      imagePath (str): Path to the input image.
      normalizeEmbedding (bool): Whether to apply L2 normalization to the embedding.
      usePatchTokens (bool): Whether to concatenate CLS token with mean of patch tokens.

    Returns:
      embedding (numpy.ndarray): The extracted embedding as a numpy array.
    '''

    # Import the Image module from the PIL library.
    from PIL import Image
    # Import the functional module from torch.nn.
    import torch.nn.functional as F
    # Assert that the provided image path exists on the filesystem.
    assert os.path.exists(imagePath), f"Image path {imagePath} does not exist."
    # Check if the model or processor has not been loaded yet.
    if (self.model is None or self.processor is None):
      # Load the model and processor lazily if they are not initialized.
      self.LoadModel()
    # Open the image file using a context manager to ensure it is closed properly.
    with Image.open(imagePath) as img:
      # Convert the image to RGB format to ensure consistent color channels.
      image = img.convert("RGB")
      # Process the image into tensor inputs using the loaded processor.
      inputs = self.processor(images=image, return_tensors="pt")
      # Move the input tensors to the specified device.
      inputs = {k: v.to(self.targetDevice) for k, v in inputs.items()}
      # Determine the device type string for autocast compatibility.
      deviceType = self.targetDevice if isinstance(self.targetDevice, str) else self.targetDevice.type
      # Perform inference without gradient computation and with automatic mixed precision.
      with torch.inference_mode(), torch.autocast(device_type=deviceType, dtype=torch.float32):
        # Get the model outputs by passing the inputs to the model.
        outputs = self.model(**inputs) if hasattr(self.model, "__call__") else self.model
      # Check if the output object has the last hidden state attribute.
      if (hasattr(outputs, "last_hidden_state")):
        # Extract the hidden states directly from the output object.
        hidden = outputs.last_hidden_state
      # Check if the output object has a return value attribute containing hidden states.
      elif (hasattr(outputs, "return_value")):
        # Extract the hidden states from the nested return value object.
        hidden = outputs.return_value.last_hidden_state
      else:
        # Assume the output object itself is the hidden states tensor.
        hidden = outputs
      # Check if the user requested the use of patch tokens in addition to the CLS token.
      if (usePatchTokens):
        # Extract the CLS token from the first position of the hidden states.
        classToken = hidden[:, 0]
        # Extract the patch tokens from the remaining positions of the hidden states.
        patchTokens = hidden[:, 1:]
        # Concatenate the CLS token with the mean of the patch tokens along the last dimension.
        embedding = torch.cat([classToken, patchTokens.mean(1)], dim=-1)
      else:
        # Extract only the CLS token from the first position as the embedding.
        embedding = hidden[:, 0, :]
      # Detach the embedding from the computation graph and convert it to float16 precision.
      embedding = embedding.detach().to(torch.float16).cpu()
      # Check if the user requested L2 normalization of the embedding.
      if (normalizeEmbedding):
        # Normalize the embedding vector using L2 normalization along the last dimension.
        embedding = F.normalize(embedding, p=2, dim=-1)
      # Convert the final embedding tensor to a numpy array.
      embedding = embedding.numpy()
      # Return the squeezed numpy array to remove any singleton dimensions.
      return embedding.squeeze()


def ExtractEmbeddingsTimm(
  datasetFolder,
  outputPicklePath,
  modelName="hf-hub:paige-ai/Virchow2",
  mlpLayer=None,
  actLayer=torch.nn.SiLU,
  device=None,
  normalizeEmbedding=False,
  patchTokenStartIndex=5,
  modelKwargs=None,
  customTransform=None,
  isPooledOutput=False,
  autocastDtype=torch.float16,
):
  r'''
  Extract embeddings from images in a dataset folder using a specified model from the timm library.

  Parameters:
    datasetFolder (str): Path to the root folder containing subfolders for each class.
    outputPicklePath (str): Path to save the output pickle file containing the embeddings lookup table.
    modelName (str): Name of the timm model to use.
    mlpLayer (nn.Module): MLP layer class to use in the model.
    actLayer (nn.Module): Activation layer class to use in the model.
    device (str or torch.device, optional): Device to run the model on.
    normalizeEmbedding (bool): Whether to apply L2 normalization to the embeddings.
    patchTokenStartIndex (int): The index from which to start extracting patch tokens.
    modelKwargs (dict, optional): Additional keyword arguments to pass to timm.create_model.
    customTransform (callable, optional): Custom image transformation pipeline.
    isPooledOutput (bool): Whether the model output is already pooled (no patch tokens).
    autocastDtype (torch.dtype): Data type for automatic mixed precision.

  Examples
  --------
  .. code-block:: python

    from HMB.ImagesToEmbeddings import ExtractEmbeddingsTimm
    from torchvision import transforms

    datasetFolder = "path/to/dataset"
    outputPickle = "Virchow2LUT.pkl"
    ExtractEmbeddingsTimm(
      datasetFolder,
      outputPickle,
      modelName="hf-hub:paige-ai/Virchow2",
      normalizeEmbedding=True,
      patchTokenStartIndex=5
    )

    # Example for H-optimus-0.
    outputPickle = "Hoptimus0LUT.pkl"
    customTransformPipeline = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
    ])
    ExtractEmbeddingsTimm(
      datasetFolder,
      outputPickle,
      modelName="hf-hub:bioptimus/H-optimus-0",
      modelKwargs={"init_values": 1e-5, "dynamic_img_size": False},
      customTransform=customTransformPipeline,
      isPooledOutput=True
    )

  Notes
  -----
  The function composes a per-image embedding by concatenating the class token and the mean of patch tokens::

    e = [class_token ; mean(patch_tokens)]
  '''

  # Import the timm library for image models.
  import timm
  # Import the garbage collection module.
  import gc
  # Import the Image module from the PIL library.
  from PIL import Image
  # Import the functional module from torch.nn.
  import torch.nn.functional as F
  # Import the data configuration resolver from timm.
  from timm.data import resolve_data_config
  # Import the transform factory from timm.
  from timm.data.transforms_factory import create_transform
  # Check if the MLP layer is not specified by the user.
  if (mlpLayer is None):
    # Import the SwiGLUPacked layer from timm.
    from timm.layers import SwiGLUPacked
    # Set the MLP layer to SwiGLUPacked as the default.
    mlpLayer = SwiGLUPacked
  # Set the target device to CUDA if available, otherwise use CPU.
  targetDevice = device or ("cuda" if torch.cuda.is_available() else "cpu")
  # Initialize the model keyword arguments if not provided.
  if (modelKwargs is None):
    # Set model keyword arguments to an empty dictionary.
    modelKwargs = {}
  # Create the embedding model using the timm library.
  embModel = timm.create_model(
    modelName,
    pretrained=True,
    mlp_layer=mlpLayer,
    act_layer=actLayer,
    **modelKwargs,
  )
  # Set the model to evaluation mode to disable training-specific layers.
  embModel.eval()
  # Move the model to the specified device with float32 precision.
  embModel.to(targetDevice, dtype=torch.float32)
  # Check if a custom transform pipeline is provided.
  if (customTransform is not None):
    # Use the provided custom transform pipeline.
    transformsPipeline = customTransform
  else:
    # Create the image transforms based on the model configuration.
    transformsPipeline = create_transform(
      **resolve_data_config(
        embModel.pretrained_cfg,
        model=embModel,
      )
    )
  # Initialize an empty dictionary for the lookup table.
  lookupTable = {}
  # Iterate over each class in the dataset folder with a progress bar.
  for cls in tqdm.tqdm(os.listdir(datasetFolder), desc="Classes"):
    # Get the full path to the class folder.
    clsPath = os.path.join(datasetFolder, cls)
    # Iterate over each image name in the class folder with a progress bar.
    for imgName in tqdm.tqdm(os.listdir(clsPath), desc=f"Images in {cls}", leave=False):
      # Check if the file does not have a valid image extension.
      if (not imgName.lower().endswith(tuple(IMAGE_SUFFIXES))):
        # Skip to the next file if it is not a valid image.
        continue
      # Get the full path to the image file.
      imgPath = os.path.join(clsPath, imgName)
      # Check if the file size is zero bytes.
      if (os.path.getsize(imgPath) == 0):
        # Skip to the next file if it is empty.
        continue
      # Move the model to the device to ensure correct placement.
      embModel.to(targetDevice, dtype=torch.float32)
      # Open the image using a context manager to prevent file handle leaks.
      with Image.open(imgPath) as temp:
        # Initialize the embedding variable to None before processing.
        embedding = None
        # Try to perform inference and extract embeddings from the image.
        try:
          # Perform inference without gradient computation and with automatic mixed precision.
          with torch.inference_mode(), torch.autocast(device_type=targetDevice, dtype=autocastDtype):
            # Apply transforms and add a batch dimension to the image.
            imgTrans = transformsPipeline(temp).unsqueeze(0)
            # Convert the transformed image to float32 and move to the device.
            imgTrans2Float = imgTrans.to(torch.float32).to(targetDevice)
            # Get the output from the embedding model.
            output = embModel(imgTrans2Float)
          # Check if the model output is already pooled.
          if (isPooledOutput):
            # Use the output directly as the embedding.
            embedding = output
          else:
            # Extract the class token from the first position of the output.
            classToken = output[:, 0]
            # Extract the patch tokens starting from the specified index.
            patchTokens = output[:, patchTokenStartIndex:]
            # Concatenate the class token with the mean of the patch tokens.
            embedding = torch.cat([classToken, patchTokens.mean(1)], dim=-1)
          # Detach the embedding from the computation graph and convert to float16.
          embedding = embedding.detach().to(torch.float16).cpu()
          # Check if the user requested L2 normalization of the embedding.
          if (normalizeEmbedding):
            # Normalize the embedding vector using L2 normalization.
            embedding = F.normalize(embedding, p=2, dim=-1)
          # Convert the final embedding tensor to a numpy array.
          embedding = embedding.numpy()
        # Catch any exceptions that occur during the inference process.
        except Exception:
          # Perform inference without gradient computation for the fallback mechanism.
          with torch.inference_mode():
            # Apply transforms and add a batch dimension for the fallback.
            imgTrans = transformsPipeline(temp).unsqueeze(0)
          # Calculate the mean value of the transformed image tensor.
          meanVal = float(imgTrans.mean().item())
          # Create a numpy array with the mean value as a fallback embedding.
          embedding = np.array([meanVal], dtype=np.float16)
        # Store the squeezed embedding in the lookup table with a dynamically generated key.
        lookupTable[f"{cls}_{imgName}"] = embedding.squeeze()
    # Open the output pickle file in write binary mode.
    with open(outputPicklePath, "wb") as fileObject:
      # Serialize and dump the lookup table into the pickle file.
      pickle.dump(lookupTable, fileObject)
  # Delete the model, transforms, and lookup table to free up memory.
  del embModel, transformsPipeline, lookupTable
  # Collect garbage to completely release the allocated memory.
  gc.collect()


def ExtractEmbeddingsTransformers(
  datasetFolder,
  outputPicklePath,
  modelName="owkin/phikon-v2",
  device=None,
  normalizeEmbedding=False,
  usePatchTokens=False,
):
  r'''
  Extract embeddings from images in a dataset folder using the TransformersEmbeddingModel.

  Parameters:
    datasetFolder (str): Path to the root folder containing subfolders for each class.
    outputPicklePath (str): Path to save the output pickle file containing the embeddings lookup table.
    modelName (str): Name of the Transformers model to use.
    device (str or torch.device, optional): Device to run the model on.
    normalizeEmbedding (bool): Whether to apply L2 normalization to the embeddings.
    usePatchTokens (bool): Whether to concatenate CLS token with mean of patch tokens.

  Examples
  --------
  .. code-block:: python

    from HMB.ImagesToEmbeddings import ExtractEmbeddingsTransformers

    datasetFolder = "path/to/dataset"
    outputPickle = "Maira2LUT.pkl"
    ExtractEmbeddingsTransformers(
      datasetFolder,
      outputPickle,
      modelName="microsoft/rad-dino-maira-2",
      normalizeEmbedding=False,
      usePatchTokens=True
    )
  '''

  # Import the garbage collection module.
  import gc
  # Import the Image module from the PIL library.
  from PIL import Image
  # Set the target device to CUDA if available, otherwise use CPU.
  targetDevice = device or ("cuda" if torch.cuda.is_available() else "cpu")
  # Initialize the Transformers embedding model with the specified name and device.
  embeddingModel = TransformersEmbeddingModel(modelName, targetDevice)
  # Initialize an empty dictionary for the lookup table.
  lookupTable = {}
  # Iterate over each class in the dataset folder with a progress bar.
  for cls in tqdm.tqdm(os.listdir(datasetFolder), desc="Classes"):
    # Get the full path to the class folder.
    clsPath = os.path.join(datasetFolder, cls)
    # Iterate over each image name in the class folder with a progress bar.
    for imgName in tqdm.tqdm(os.listdir(clsPath), desc=f"Images in {cls}", leave=False):
      # Check if the file does not have a valid image extension.
      if (not imgName.lower().endswith(tuple(IMAGE_SUFFIXES))):
        # Skip to the next file if it is not a valid image.
        continue
      # Get the full path to the image file.
      imgPath = os.path.join(clsPath, imgName)
      # Check if the file size is zero bytes.
      if (os.path.getsize(imgPath) == 0):
        # Skip to the next file if it is empty.
        continue
      # Try to perform inference and extract embeddings from the image.
      try:
        # Extract the embedding using the initialized model.
        embedding = embeddingModel.GetEmbedding(
          imgPath,
          normalizeEmbedding=normalizeEmbedding,
          usePatchTokens=usePatchTokens,
        )
      # Catch any exceptions that occur during the inference process.
      except Exception:
        # Create a numpy array with a zero value as a fallback embedding.
        embedding = np.array([0.0], dtype=np.float16)
      # Store the squeezed embedding in the lookup table with a dynamically generated key.
      lookupTable[f"{cls}_{imgName}"] = embedding.squeeze()
    # Open the output pickle file in write binary mode.
    with open(outputPicklePath, "wb") as fileObject:
      # Serialize and dump the lookup table into the pickle file.
      pickle.dump(lookupTable, fileObject)
  # Delete the embedding model and lookup table to free up memory.
  del embeddingModel, lookupTable
  # Collect garbage to completely release the allocated memory.
  gc.collect()


# Check if the script is being run as the main program.
if __name__ == "__main__":
  # Import the time module for timestamp generation.
  import time

  # Get the current timestamp formatted as a string.
  timeStamp = time.strftime("%Y%m%d-%H%M%S")
  # Set the dataset folder path variable.
  datasetFolder = "Data/Train"
  # Example 1: Virchow2 using Timm.
  # Set the output pickle file path variable with the timestamp.
  outputPicklePath = f"Data/Virchow2_LUT_{timeStamp}.p"
  # Run the embedding extraction function for Virchow2.
  ExtractEmbeddingsTimm(
    datasetFolder,
    outputPicklePath,
    modelName="hf-hub:paige-ai/Virchow2",
    device=None,
    normalizeEmbedding=True,
    patchTokenStartIndex=5
  )
  # Example 2: Maira2 using Transformers.
  # Set the output pickle file path variable with the timestamp.
  # outputPicklePath = f"Data/Maira2_LUT_{timeStamp}.p"
  # Run the embedding extraction function for Maira2.
  # ExtractEmbeddingsTransformers(
  #   datasetFolder,
  #   outputPicklePath,
  #   modelName="microsoft/rad-dino-maira-2",
  #   device=None,
  #   normalizeEmbedding=False,
  #   usePatchTokens=True
  # )
  # Example 3: Phikon-v2 using Transformers.
  # Set the output pickle file path variable with the timestamp.
  # outputPicklePath = f"Data/PhikonV2_LUT_{timeStamp}.p"
  # Run the embedding extraction function for Phikon-v2.
  # ExtractEmbeddingsTransformers(
  #   datasetFolder,
  #   outputPicklePath,
  #   modelName="owkin/phikon-v2",
  #   device=None,
  #   normalizeEmbedding=True,
  #   usePatchTokens=False
  # )
  # Example 4: Nomic using Transformers.
  # Set the output pickle file path variable with the timestamp.
  # outputPicklePath = f"Data/Nomic_LUT_{timeStamp}.p"
  # Run the embedding extraction function for Nomic.
  # ExtractEmbeddingsTransformers(
  #   datasetFolder,
  #   outputPicklePath,
  #   modelName="nomic-ai/nomic-embed-vision-v1.5",
  #   device=None,
  #   normalizeEmbedding=True,
  #   usePatchTokens=False
  # )
  # Example 5: H-optimus-0 using Timm.
  # Set the output pickle file path variable with the timestamp.
  # outputPicklePath = f"Data/Hoptimus0_LUT_{timeStamp}.p"
  # Import the transforms module from torchvision.
  # from torchvision import transforms
  # Define the custom transform pipeline for H-optimus-0.
  # customTransformPipeline = transforms.Compose([
  #   transforms.Resize((224, 224)),
  #   transforms.ToTensor(),
  #   transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
  # ])
  # Run the embedding extraction function for H-optimus-0.
  # ExtractEmbeddingsTimm(
  #   datasetFolder,
  #   outputPicklePath,
  #   modelName="hf-hub:bioptimus/H-optimus-0",
  #   device=None,
  #   modelKwargs={"init_values": 1e-5, "dynamic_img_size": False},
  #   customTransform=customTransformPipeline,
  #   isPooledOutput=True
  # )
