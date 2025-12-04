import os, cv2, random, torch, timm
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from HMB.PyTorchHelper import LoadPyTorchDict


# Wrapper to return logits only.
class LogitsModelWrapper(torch.nn.Module):
  r'''
  Wrapper for a model to return only logits from the forward pass.

  Parameters:
    model (torch.nn.Module): The model to wrap.

  Returns:
    torch.Tensor: Logits output from the model.
  '''

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x):
    return self.model(x).logits


def HuggingFaceModel(
  modelName,
  numClasses,
  modelCheckpointPath,
  size,
  device,
  meanValues=[0.485, 0.456, 0.406],
  stdValues=[0.229, 0.224, 0.225],
):
  r'''
  Load a Hugging Face Vision Transformer model and its checkpoint, and prepare preprocessing transforms.

  Parameters:
    modelName (str): Name of the Hugging Face model to use.
    numClasses (int): Number of output classes.
    modelCheckpointPath (str): Path to the trained model checkpoint.
    size (int): Image size for resizing and visualization.
    device (str or torch.device): Device to run the model on ("cuda" or "cpu").
    meanValues (list): Mean values for normalization.
    stdValues (list): Standard deviation values for normalization.

  Returns:
    tuple: A tuple containing:
      - model (torch.nn.Module): Loaded model.
      - vitTargetLayer (torch.nn.Module): Target layer for CAM extraction.
      - transform (torchvision.transforms.Compose): Image preprocessing pipeline.

  Notes:
    - Loads the model weights from the checkpoint.
    - Sets the model to evaluation mode.
    - Prepares the image transform for inference.

  Examples
  --------
  .. code-block:: python

    import HMB.AttentionMapsHelper as amh

    model, vitTargetLayer, transform = amh.HuggingFaceModel(
      modelName="google/vit-base-patch16-224",
      numClasses=3,
      modelCheckpointPath="/path/to/checkpoint.pth",
      size=224,
      device="cuda",
      meanValues=[0.485, 0.456, 0.406],
      stdValues=[0.229, 0.224, 0.225]
    )
  '''

  # Create the model using Hugging Face transformers.
  model = ViTForImageClassification.from_pretrained(
    modelName,  # Model name.
    num_labels=numClasses,  # Set number of output classes.
    ignore_mismatched_sizes=True,  # Ignore size mismatches when loading weights.
  )
  # Load model checkpoint.
  stateDict = LoadPyTorchDict(modelCheckpointPath, device=device)
  if (stateDict and isinstance(stateDict, dict) and "model_state_dict" in stateDict):
    model.load_state_dict(stateDict["model_state_dict"])
  else:
    model.load_state_dict(stateDict)
  # Wrap the model to return logits only.
  model = LogitsModelWrapper(model)
  # Move model to device.
  model = model.to(device)
  # Set model to evaluation mode.
  model.eval()
  # Set target layer for CAM.
  vitTargetLayer = model.vit.encoder.layer[-1].output
  # Define image transform.
  transform = transforms.Compose([
    transforms.Resize((size, size)),  # Resize image to specified size.
    transforms.ToTensor(),  # Convert image to tensor.
    transforms.Normalize(mean=meanValues, std=stdValues),  # Normalize image.
  ])
  # Return model, target layer, and transform.
  return model, vitTargetLayer, transform


def GetDefaultVitTargetLayer(model):
  r'''
  Dynamically select a suitable target layer for CAM from a timm model.
  Tries common patterns for ViT, Swin, MaxViT, ConvNeXtV2, etc.
  Skips Identity layers as they are not suitable for CAM.

  Parameters:
    model (torch.nn.Module): The model to inspect.

  Returns:
    torch.nn.Module: The selected target layer for CAM.
  '''

  def IsContainer(x):
    return isinstance(x, (list, torch.nn.ModuleList, torch.nn.Sequential))

  def IsCAMLayer(module):
    # return isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.Identity))
    return isinstance(module, (torch.nn.Identity))

  def IsIdentity(module):
    return isinstance(module, torch.nn.Identity)

  # Helper: recursively search for the last normalization layer, skipping Identity.
  def FindLastCAMLayer(module):
    for name, submodule in reversed(list(module.named_modules())):
      if (IsCAMLayer(submodule)):
        return submodule
    return None

  # Some models have a "norm" attribute at the top level.
  if (hasattr(model, "norm")):
    norm = model.norm
    if (IsCAMLayer(norm)):
      return norm
  elif (hasattr(model, "norm_pre")):
    norm = model.norm_pre
    if (IsCAMLayer(norm)):
      return norm

  if (hasattr(model, "stages") and IsContainer(model.stages)):
    # print("Model has stages.")
    lastStage = list(model.stages)[-1]
    if (hasattr(lastStage, "norm")):
      norm = lastStage.norm
      if (IsCAMLayer(norm)):
        return norm
    if (hasattr(lastStage, "blocks") and IsContainer(lastStage.blocks)):
      # print("Last stage has blocks.")
      lastBlock = list(lastStage.blocks)[-1]
      # print("Last stage last block:", lastBlock)
      if (hasattr(lastBlock, "norm")):
        norm = lastBlock.norm
        if (IsCAMLayer(norm)):
          return norm
      if (hasattr(lastBlock, "norm1")):
        norm1 = lastBlock.norm1
        if (IsCAMLayer(norm1)):
          return norm1
      found = FindLastCAMLayer(lastBlock)
      # print("Found in last block:", found)
      if (found is not None):
        return found
      # return lastBlock
    found = FindLastCAMLayer(lastStage)
    if ((found is not None)):
      return found
    # return lastStage

  if (hasattr(model, "blocks") and IsContainer(model.blocks)):
    lastBlock = list(model.blocks)[-1]
    if (hasattr(lastBlock, "norm")):
      norm = lastBlock.norm
      if (IsCAMLayer(norm)):
        return norm
    if (hasattr(lastBlock, "norm1")):
      norm1 = lastBlock.norm1
      if (IsCAMLayer(norm1)):
        return norm1
    found = FindLastCAMLayer(lastBlock)
    if (found is not None):
      return found
    # return lastBlock

  if (hasattr(model, "layers") and IsContainer(model.layers)):
    lastLayer = list(model.layers)[-1]
    if (hasattr(lastLayer, "blocks") and IsContainer(lastLayer.blocks)):
      lastBlock = list(lastLayer.blocks)[-1]
      if (hasattr(lastBlock, "norm1")):
        norm1 = lastBlock.norm1
        if (IsCAMLayer(norm1)):
          return norm1
      if (hasattr(lastBlock, "norm")):
        norm = lastBlock.norm
        if (IsCAMLayer(norm)):
          return norm
      found = FindLastCAMLayer(lastBlock)
      if (found is not None):
        return found
      # return lastBlock
    found = FindLastCAMLayer(lastLayer)
    if (found is not None):
      return found
    # return lastLayer

  # Top-level norm.
  if (hasattr(model, "norm")):
    norm = model.norm
    if (IsCAMLayer(norm)):
      return norm

  # Fallback: recursively search for the last normalization layer in the whole model, skipping Identity.
  found = FindLastCAMLayer(model)
  if (found is not None):
    return found

  raise ValueError("Could not automatically determine a suitable target layer for CAM. Please specify manually.")


def TimmModel(
  modelName,
  numClasses,
  modelCheckpointPath,
  device,
  targetLayer=None,  # Optional: allow user to specify target layer
):
  r'''
  Load a timm Vision Transformer model and its checkpoint, and prepare preprocessing transforms.

  Parameters:
    modelName (str): Name of the timm model to use.
    numClasses (int): Number of output classes.
    modelCheckpointPath (str): Path to the trained model checkpoint.
    device (str or torch.device): Device to run the model on ("cuda" or "cpu").
    targetLayer (str or None): Optional string specifying the target layer for CAM.

  Returns:
    tuple: A tuple containing:
      - model (torch.nn.Module): Loaded model.
      - vitTargetLayer (torch.nn.Module): Target layer for CAM extraction.
      - transform (torchvision.transforms.Compose): Image preprocessing pipeline.

  Notes:
    - Loads the model weights from the checkpoint.
    - Sets the model to evaluation mode.
    - Prepares the image transform for inference.
    - Tested on:
       -- timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k from https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k

  Examples
  --------
  .. code-block:: python

    import HMB.AttentionMapsHelper as amh

    model, vitTargetLayer, transform = amh.TimmModel(
      modelName="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
      numClasses=3,
      modelCheckpointPath="/path/to/checkpoint.pth",
      device="cuda"
    )
  '''

  # Create the model using timm.
  model = timm.create_model(modelName, pretrained=False, num_classes=numClasses)
  # Load model checkpoint.
  stateDict = LoadPyTorchDict(modelCheckpointPath, device=device)
  if (stateDict and isinstance(stateDict, dict) and "model_state_dict" in stateDict):
    model.load_state_dict(stateDict["model_state_dict"])
  else:
    model.load_state_dict(stateDict)
  # Get data configuration for preprocessing.
  dataConfig = timm.data.resolve_model_data_config(model)
  # Create image transform for inference.
  transform = timm.data.create_transform(**dataConfig, is_training=False)
  # Move model to device.
  model = model.to(device)
  # Set model to evaluation mode.
  model.eval()
  # Set target layer for CAM.
  # vitTargetLayer = model.blocks[-1].norm1
  if (targetLayer is not None):
    vitTargetLayer = targetLayer if not isinstance(targetLayer, str) else eval(f"model.{targetLayer}")
  else:
    vitTargetLayer = GetDefaultVitTargetLayer(model)
  # Return model, target layer, and transform.
  return model, vitTargetLayer, transform


class AttentionMapsVisualizer(object):
  r'''
  Visualize attention maps for images using various CAM methods on a Vision Transformer (ViT) model.

  Parameters:
    baseFolder (str): Base directory containing model and dataset.
    dataFolder (str): Directory containing image data organized by class.
    modelName (str): Name of the timm model to use.
    modelCheckpointPath (str): Path to the trained model checkpoint.
    modelType (str): Type of model ("Timm" and "HuggingFace"). Default is "Timm".
    size (int): Image size for resizing and visualization. Default is 448.
    doReshape (bool): Whether to reshape transformer outputs for CAM. Default is False.
    device (str or torch.device): Device to run the model on ("cuda" or "cpu").

  Attributes:
    model (torch.nn.Module): Loaded vision transformer model.
    vitTargetLayer (torch.nn.Module): Target layer for CAM extraction.
    transform (torchvision.transforms.Compose): Image preprocessing pipeline.
    classes (list): List of class names from dataset.
    numClasses (int): Number of classes.

  Notes:
    - Supports GradCAM, ScoreCAM, EigenCAM, and other CAM methods.
    - Can select specific images or random images per class for visualization.
    - Figure size, CAM methods, and output options are customizable.
    - Saves the resulting attention map grid as a PNG file.

  Examples
  --------
  .. code-block:: python

    import HMB.AttentionMapsHelper as amh

    visualizer = amh.AttentionMapsVisualizer(
      baseFolder="/path/to/base/folder",
      dataFolder="/path/to/data/folder",
      modelName="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
      modelCheckpointPath="/path/to/checkpoint.pth",
      modelType="Timm",
      size=448,
      device="cuda"
    )
    visualizer.VisualizeAttentionMaps(
      cams=["GradCAM", "ScoreCAM"],
      figSize=(14, 8),
      imagesPerClass=2,
      save=True,
      display=True,
      outPrefix="AttentionMaps",
      dpi=300,
      fontSize=10,
      alpha=0.4,
      selectImages=None,
      allowedExtensions=(".jpg", ".jpeg", ".png", ".bmp")
    )
  '''

  def __init__(
    self,
    baseFolder,
    dataFolder,
    modelName,
    modelCheckpointPath,
    modelType="Timm",
    size=448,
    doReshape=False,
    device=None,
  ):
    # Set device.
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Set image size.
    self.size = size
    # Whether to reshape transformer outputs for CAM.
    self.doReshape = doReshape
    # Set base folder.
    self.baseFolder = baseFolder

    # Set data folder.
    self.dataFolder = dataFolder
    assert os.path.isdir(self.dataFolder), "Data folder does not exist."
    # Get class names from data folder.
    self.classes = sorted(os.listdir(self.dataFolder))
    assert self.classes, "No class folders found in the data folder."
    # Get number of classes.
    self.numClasses = len(self.classes)

    # Set model type.
    self.modelType = modelType
    # Set model name.
    self.modelName = modelName
    # Set model path.
    self.modelCheckpointPath = modelCheckpointPath
    assert os.path.isfile(self.modelCheckpointPath), "Model checkpoint file does not exist."

    # Load model and preprocessing transform.
    if (self.modelType == "Timm"):
      self.model, self.vitTargetLayer, self.transform = TimmModel(
        modelName=self.modelName,
        numClasses=self.numClasses,
        modelCheckpointPath=self.modelCheckpointPath,
        device=self.device,
      )
    elif (self.modelType == "HuggingFace"):
      self.model, self.vitTargetLayer, self.transform = HuggingFaceModel(
        modelName=self.modelName,
        numClasses=self.numClasses,
        modelCheckpointPath=self.modelCheckpointPath,
        size=self.size,
        device=self.device,
      )
    else:
      raise ValueError("Unsupported model type. Currently only 'Timm' and 'HuggingFace' are supported.")

    print(f"Model and transform loaded. Number of classes: {self.numClasses}")
    print(f"Classes: {self.classes}")
    print(f"Using device: {self.device}")
    print(f"Model type: {self.modelType}, Model name: {self.modelName}")
    print(f"Model checkpoint path: {self.modelCheckpointPath}")
    print(f"Image size: {self.size}, Do reshape: {self.doReshape}")
    print(f"Target layer for CAM: {self.vitTargetLayer}")
    print("Initialization complete.")

  @staticmethod
  def ReshapeTransform(outputs, height, width):
    r'''
    Reshape the transformer outputs to a 4D tensor suitable for CAM extraction.

    Parameters:
      outputs (torch.Tensor): Outputs from the transformer model.
      height (int): Height of the feature map.
      width (int): Width of the feature map.

    Returns:
      torch.Tensor: Reshaped tensor of shape (batch_size, channels, height, width).
    '''

    # Reshape transformer outputs for CAM.
    if (isinstance(outputs, tuple)):
      tensor = outputs[0]
    else:
      tensor = outputs
    # tensor.size(0): batch size.
    # tensor.size(1): number of tokens (including class token).
    # tensor.size(2): feature dimension.

    # Remove class token and reshape.
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    tensor = tensor.transpose(2, 3).transpose(1, 2)
    return tensor

  def VisualizeAttentionMaps(
    self,
    cams=None,  # List of CAM classes to use.
    alpha=0.35,  # Transparency for overlaying heatmap.
    save=True,  # Whether to save the resulting figure.
    display=False,  # Whether to display the figure.
    dpi=300,  # DPI for saving the figure.
    outPrefix="AttentionMapResults",  # Prefix for output file name.
    figSize=(12, 10),  # Figure size.
    fontSize=12,  # Font size for titles and labels.
    imagesPerClass=1,  # Number of images per class to visualize.
    selectImages=None,  # Optional dict: {class_name: [image filenames]}.
    allowedExtensions=(".jpg", ".jpeg", ".png", ".bmp"),
    doAverage=True,
  ):
    r'''
    Visualize and save attention maps for images using specified CAM methods.

    Parameters:
      cams (list): List of CAM classes to use (default: [GradCAM, ScoreCAM, EigenCAM]).
      alpha (float): Transparency for overlaying heatmap.
      save (bool): Whether to save the resulting figure.
      display (bool): Whether to display the figure.
      dpi (int): DPI for saving the figure.
      outPrefix (str): Prefix for output file name.
      figSize (tuple): Figure size in inches.
      imagesPerClass (int): Number of images per class to visualize.
      selectImages (dict or None): Optional dict mapping class names to list of image filenames.
      allowedExtensions (tuple): Allowed image file extensions.
      doAverage (bool): Whether to compute and display the average overlay per class.

    Notes:
      - If selectImages is provided, it should be a dictionary where keys are class names and
        values are lists of image filenames to visualize for that class. If not provided, random images will be selected.
      - The resulting figure will have rows corresponding to classes and columns corresponding to CAM methods and images.
      - The output file will be saved in the base folder with a timestamp.

    Raises:
      AssertionError: If no image files are found in the data folder.
      ValueError: If an unsupported CAM method is specified.
    '''

    # Set default CAMs if not provided.
    if (cams is None):
      cams = ["GradCAM", "ScoreCAM", "EigenCAM"]

    # Calculate total subplots.
    totalRows = len(self.classes)
    totalCols = len(cams) * imagesPerClass + (imagesPerClass if doAverage else 0)

    # Create figure with specified size.
    plt.figure(figsize=figSize)
    plotIdx = 1

    # Loop through each class.
    for c, cls in enumerate(self.classes):
      print(f"Processing class: {cls} ({c + 1}/{self.numClasses})")
      clsPath = os.path.join(self.dataFolder, cls)

      # Get image files for the class.
      imageFiles = [
        f for f in os.listdir(clsPath)
        if (f.lower().endswith(allowedExtensions))
      ]
      assert imageFiles, "No image files found in the data folder."

      # Select images for visualization.
      if (selectImages and cls in selectImages):
        chosenImages = selectImages[cls]
        if (imagesPerClass and imagesPerClass > 0):
          # Limit to imagesPerClass if more provided.
          chosenImages = chosenImages[:imagesPerClass]
      else:
        chosenImages = random.sample(imageFiles, min(imagesPerClass, len(imageFiles)))

      # Loop through selected images.
      for imgIdx, imageName in enumerate(chosenImages):
        imagePath = os.path.join(clsPath, imageName)

        # Open and preprocess image.
        image = Image.open(imagePath).convert("RGB")
        imgTensor = self.transform(image).unsqueeze(0).to(self.device)

        history = []

        # Loop through CAM methods.
        for i, cam in enumerate(cams):
          # Select CAM class.
          if (cam == "GradCAM"):
            camCls = GradCAM
          elif (cam == "HiResCAM"):
            camCls = HiResCAM
          elif (cam == "ScoreCAM"):
            camCls = ScoreCAM
          elif (cam == "GradCAMPlusPlus"):
            camCls = GradCAMPlusPlus
          elif (cam == "AblationCAM"):
            camCls = AblationCAM
          elif (cam == "XGradCAM"):
            camCls = XGradCAM
          elif (cam == "EigenCAM"):
            camCls = EigenCAM
          elif (cam == "FullGrad"):
            camCls = FullGrad
          else:
            raise ValueError(f"Unsupported CAM method: {cam}")

          print(f"Processing Class: {cls}, Image: {imageName}, CAM: {camCls.__name__}")

          self.reshapeTransformSize = None
          # Determine reshape size based on the input size.
          if (self.size == 224):
            self.reshapeTransformSize = 14
          elif (self.size == 384):
            self.reshapeTransformSize = 24
          elif (self.size == 448):
            self.reshapeTransformSize = 28
          elif (self.size == 512):
            self.reshapeTransformSize = 32
          elif (self.size == 576):
            self.reshapeTransformSize = 36

          if (self.doReshape and self.reshapeTransformSize is None):
            # Estimate reshape size for non-standard sizes.
            self.reshapeTransformSize = self.size // 16
            cam = camCls(
              model=self.model,
              target_layers=[self.vitTargetLayer],
              reshape_transform=lambda x: self.ReshapeTransform(
                x,
                self.reshapeTransformSize,
                self.reshapeTransformSize
              ),
            )
          else:
            # Use default (no reshape).
            self.reshapeTransformSize = None
            cam = camCls(
              model=self.model,
              target_layers=[self.vitTargetLayer],
              reshape_transform=None,
            )

          # Compute CAM.
          grayscaleCam = cam(
            input_tensor=imgTensor,
            targets=None,
            eigen_smooth=True,
            aug_smooth=True,
          )
          camMap = grayscaleCam[0]

          # Generate the heatmap.
          heatmap = cv2.applyColorMap(np.uint8(255 * camMap), cv2.COLORMAP_JET)
          heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

          # Resize and normalize image.
          imageNp = np.array(image.resize((self.size, self.size))) / 255.0

          # Overlay heatmap on image.
          overlay = heatmap * alpha + imageNp * (1.0 - alpha)
          overlay = np.uint8(255 * overlay)

          # Save overlay to history.
          history.append(overlay)

          # Add subplot.
          ax = plt.subplot(totalRows, totalCols, plotIdx)
          ax.imshow(overlay, vmin=0, vmax=255)

          # Set subplot title.
          ax.set_title(f"{camCls.__name__}", fontsize=fontSize)

          # Set class label on leftmost column.
          if (i == 0 and imgIdx == 0):
            ax.set_ylabel(cls, fontsize=fontSize, rotation=90, labelpad=40, va="center")

          # Remove axis ticks.
          ax.set_xticks([])
          ax.set_yticks([])
          plotIdx += 1

        # Combine the overlays for the current image and find the average.
        if (len(history) and doAverage):
          combinedOverlay = np.mean(np.array(history), axis=0).astype(np.uint8)
          combinedAx = plt.subplot(totalRows, totalCols, plotIdx)
          combinedAx.imshow(combinedOverlay, vmin=0, vmax=255)
          combinedAx.set_title("Combined", fontsize=fontSize)
          combinedAx.set_xticks([])
          combinedAx.set_yticks([])
          plotIdx += 1

    # Tight layout to minimize wasted space.
    plt.tight_layout()

    print("Visualization complete and ready for saving or displaying.")

    # Generate output file path.
    rndTimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outPath = os.path.join(self.baseFolder, f"{outPrefix}_{rndTimestamp}.png")

    # Save the plot if requested.
    if (save):
      plt.savefig(outPath, dpi=dpi, bbox_inches="tight")

    # Display the plot if requested.
    if (display):
      plt.show()

    # Close the plot to free memory.
    plt.close()


if __name__ == "__main__":
  import timm

  modelName = "maxvit_xlarge_tf_512.in21k_ft_in1k"
  model = timm.create_model(modelName, pretrained=True)
  # print(model)
  print("Getting default target layer for CAM...")
  print("Model:", modelName)
  targetLayer = GetDefaultVitTargetLayer(model)
  print("Target Layer:", targetLayer)

  modelName = "convnextv2_huge.fcmae_ft_in22k_in1k_512"
  model = timm.create_model(modelName, pretrained=True)
  # print(model)
  print("Getting default target layer for CAM...")
  print("Model:", modelName)
  targetLayer = GetDefaultVitTargetLayer(model)
  print("Target Layer:", targetLayer)
