import timm, tqdm, os, pickle, torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


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
