import timm, tqdm, os, pickle, torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import AutoImageProcessor, AutoModel


class TransformersEmbeddingModel(object):
  r'''
  A class to extract embeddings from images using pre-trained models from the Hugging Face Transformers library.
  '''

  def __init__(self, modelName, device):
    r'''
    Initialize the TransformersEmbeddingModel with a specified model name and device.

    Parameters:
      modelName (str): Name of the pre-trained model to load from Hugging Face.
      device (str or torch.device): Device to run the model on (e.g., "cuda", "cpu").
    '''

    self.modelName = modelName
    self.device = device
    self.model = None
    self.processor = None

  def LoadModel(self):
    r'''
    Load the pre-trained model and processor from the specified model name.

    Returns:
      model (nn.Module): The loaded pre-trained model.
      processor (AutoImageProcessor): The loaded image processor.
    '''

    # Load the processor and model from Hugging Face.
    self.processor = AutoImageProcessor.from_pretrained(self.modelName, use_fast=True)
    self.model = AutoModel.from_pretrained(self.modelName)
    # Set the model to evaluation mode and move to the specified device.
    self.model.eval()
    self.model.to(self.device, dtype=torch.float32)
    # Return the model and processor.
    return self.model, self.processor

  def GetEmbedding(self, imagePath):
    r'''
    Extract embedding from an image using the loaded model and processor.

    Parameters:
      imagePath (str): Path to the input image.

    Returns:
      embedding (numpy.ndarray): The extracted embedding as a numpy array.
    '''

    # Ensure the image path exists.
    assert os.path.exists(imagePath), f"Image path {imagePath} does not exist."
    # Load model and processor if not already loaded.
    if (not hasattr(self, "model")) or (not hasattr(self, "processor")):
      LoadModel()
    # Open and process the image.
    image = Image.open(imagePath).convert("RGB")
    # Prepare inputs for the model.
    inputs = self.processor(images=image, return_tensors="pt").to(self.device, dtype=torch.float32)
    # Extract features with inference mode and autocast.
    with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float32):
      outputs = self.model(**inputs)
    # Get the embedding from the [CLS] token.
    embedding = outputs.last_hidden_state[:, 0, :]
    # Convert embedding to float16 and numpy array.
    embedding = embedding.to(torch.float16)
    # Move embedding to CPU and convert to numpy.
    embedding = embedding.cpu().numpy()
    # Return the squeezed embedding.
    return embedding.squeeze()


def ExtractEmbeddingsTimm(
  datasetFolder,
  outputPicklePath,
  modelName="hf-hub:paige-ai/Virchow2",
  mlpLayer=SwiGLUPacked,
  actLayer=torch.nn.SiLU,
  device=None,
):
  r'''
  Extract embeddings from images in a dataset folder using a specified model from the timm library.

  Parameters:
    datasetFolder (str): Path to the root folder containing subfolders for each class, each with images.
    outputPicklePath (str): Path to save the output pickle file containing the embeddings lookup table.
    modelName (str): Name of the timm model to use. Default is "hf-hub:paige-ai/Virchow2".
    mlpLayer (nn.Module): MLP layer class to use in the model. Default is SwiGLUPacked.
    actLayer (nn.Module): Activation layer class to use in the model. Default is torch.nn.SiLU.
    device (str or torch.device, optional): Device to run the model on (e.g., "cuda", "cpu"). If None, uses CUDA if available.

  Returns:
    None. Saves a pickle file at outputPicklePath containing a dictionary mapping "class_imagename" to embedding numpy arrays.

  Examples
  --------
  Here is how to use this function in a script:

  .. code-block:: python

    from HMB.ImagesToEmbeddings import ExtractEmbeddingsTimm
    datasetFolder = "path/to/dataset"
    outputPickle = "embeddings.pkl"
    ExtractEmbeddingsTimm(datasetFolder, outputPickle)
  '''

  # Set device to CUDA if available, else CPU.
  DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

  # Create the embedding model.
  embModel = timm.create_model(
    modelName,
    pretrained=True,
    mlp_layer=mlpLayer,
    act_layer=actLayer,
  )
  # Set model to evaluation mode.
  embModel.eval()
  # Move model to device.
  embModel.to(DEVICE, dtype=torch.float32)
  # Create image transforms.
  transforms = create_transform(
    **resolve_data_config(
      embModel.pretrained_cfg,
      model=embModel,
    )
  )

  # Initialize lookup table.
  lookupTable = {}

  # Iterate over classes in dataset folder.
  for cls in tqdm.tqdm(os.listdir(datasetFolder), desc="Classes"):
    # Get class path.
    clsPath = os.path.join(datasetFolder, cls)
    # Iterate over images in class folder.
    for imgName in tqdm.tqdm(os.listdir(clsPath), desc=f"Images in {cls}", leave=False):
      # Skip non-image files.
      if (not imgName.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))):
        continue
      # Skip zero-byte files.
      if (os.path.getsize(os.path.join(clsPath, imgName)) == 0):
        continue
      # Move model to device (redundant, but kept as in original).
      embModel.to(DEVICE, dtype=torch.float32)
      # Get image path.
      imgPath = os.path.join(clsPath, imgName)
      # Open image.
      temp = Image.open(imgPath)
      # Extract embedding with inference mode and autocast.
      with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=torch.float32):
        # Apply transforms and add batch dimension.
        imgTrans = transforms(temp).unsqueeze(0)
        # Move image tensor to float32 and device.
        imgTrans2Float = imgTrans.to(torch.float32).to(DEVICE, dtype=torch.float32)
        # Get model output.
        output = embModel(imgTrans2Float)
      # Extract class token.
      classToken = output[:, 0]
      # Extract patch tokens.
      patchTokens = output[:, 5:]
      # Concatenate class token and mean patch tokens.
      embedding = torch.cat([classToken, patchTokens.mean(1)], dim=-1)
      # Convert embedding to float16.
      embedding = embedding.to(torch.float16)
      # Move embedding to CPU and convert to numpy.
      embedding = embedding.cpu().numpy()

      # Store embedding in lookup table.
      lookupTable[f"{cls}_{imgName}"] = embedding.squeeze()

    # Save lookup table to pickle file.
    with open(outputPicklePath, "wb") as f:
      pickle.dump(lookupTable, f)


if __name__ == "__main__":
  # Import time for timestamp.
  import time

  # Get current timestamp.
  timeStamp = time.strftime("%Y%m%d-%H%M%S")

  # Set dataset folder path.
  DATASET_FOLDER = "Data/Train"
  # Set output pickle file path.
  OUTPUT_PICKLE_PATH = f"Data/Virchow2_LUT_{timeStamp}.p"

  # Run embedding extraction.
  ExtractEmbeddingsTimm(
    DATASET_FOLDER,
    OUTPUT_PICKLE_PATH,
    modelName="hf-hub:paige-ai/Virchow2",
    device=None,
  )
