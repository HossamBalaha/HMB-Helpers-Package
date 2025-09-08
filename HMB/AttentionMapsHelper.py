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


# Wrapper to return logits only.
class LogitsModelWrapper(torch.nn.Module):
  '''
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
):
  '''
  Load a Hugging Face Vision Transformer model and its checkpoint, and prepare preprocessing transforms.

  Parameters:
    modelName (str): Name of the Hugging Face model to use.
    numClasses (int): Number of output classes.
    modelCheckpointPath (str): Path to the trained model checkpoint.
    size (int): Image size for resizing and visualization.
    device (str or torch.device): Device to run the model on ("cuda" or "cpu").

  Returns:
    tuple: (model, vitTargetLayer, transform)
      - model (torch.nn.Module): Loaded model.
      - vitTargetLayer (torch.nn.Module): Target layer for CAM extraction.
      - transform (torchvision.transforms.Compose): Image preprocessing pipeline.

  Notes:
    - Loads the model weights from the checkpoint.
    - Sets the model to evaluation mode.
    - Prepares the image transform for inference.
  '''

  # Create the model using Hugging Face transformers.
  model = ViTForImageClassification.from_pretrained(
    modelName,
    num_labels=numClasses,
    ignore_mismatched_sizes=True,
  )
  # Load model checkpoint.
  checkpoint = torch.load(modelCheckpointPath, map_location=device)
  # Load state dict into model.
  model.load_state_dict(checkpoint["model_state_dict"])
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
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  # Return model, target layer, and transform.
  return model, vitTargetLayer, transform


def TimmModel(
  modelName,
  numClasses,
  modelCheckpointPath,
  device,
):
  '''
  Load a timm Vision Transformer model and its checkpoint, and prepare preprocessing transforms.

  Parameters:
    modelName (str): Name of the timm model to use.
    numClasses (int): Number of output classes.
    modelCheckpointPath (str): Path to the trained model checkpoint.
    device (str or torch.device): Device to run the model on ("cuda" or "cpu").

  Returns:
    tuple: (model, vitTargetLayer, transform)
      - model (torch.nn.Module): Loaded model.
      - vitTargetLayer (torch.nn.Module): Target layer for CAM extraction.
      - transform (torchvision.transforms.Compose): Image preprocessing pipeline.

  Notes:
    - Loads the model weights from the checkpoint.
    - Sets the model to evaluation mode.
    - Prepares the image transform for inference.
    - Tested on:
      - timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k from https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
  '''

  # Create the model using timm.
  model = timm.create_model(modelName, pretrained=False, num_classes=numClasses)
  # Load model checkpoint.
  checkpoint = torch.load(modelCheckpointPath, map_location=device)
  # Load state dict into model.
  model.load_state_dict(checkpoint["model_state_dict"])
  # Get data configuration for preprocessing.
  dataConfig = timm.data.resolve_model_data_config(model)
  # Create image transform for inference.
  transform = timm.data.create_transform(**dataConfig, is_training=False)
  # Move model to device.
  model = model.to(device)
  # Set model to evaluation mode.
  model.eval()
  # Set target layer for CAM.
  vitTargetLayer = model.blocks[-1].norm1
  # Return model, target layer, and transform.
  return model, vitTargetLayer, transform


class AttentionMapsVisualizer(object):
  '''
  Visualize attention maps for images using various CAM methods on a Vision Transformer (ViT) model.

  Parameters:
    baseFolder (str): Base directory containing model and dataset.
    folder (str): Subfolder for results and model checkpoint.
    dataFolder (str): Directory containing image data organized by class.
    modelName (str): Name of the timm model to use.
    modelCheckpointPath (str): Path to the trained model checkpoint.
    modelType (str): Type of model ("Timm" and "HuggingFace"). Default is "Timm".
    size (int): Image size for resizing and visualization. Default is 448.
    reshapeTransformSize (int): Size for reshape transform in CAM. Default is 32.
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

    from HMB.AttentionMapsHelper import AttentionMapsVisualizer

    visualizer = AttentionMapsVisualizer(
      baseFolder="/home/hmbala01/[B] BC Conf",
      folder="Results_E125_BS32_T5",
      dataFolder="/home/hmbala01/[B] BC Conf/Dataset_BUSI All",
      modelName="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
      modelCheckpointPath="/home/hmbala01/[B] BC Conf/Results_E125_BS32_T5/best_model.pth",
      size=448,
      reshapeTransformSize=32,
      device="cuda"
    )
    visualizer.VisualizeAttentionMaps(
      cams=[GradCAM, ScoreCAM],
      figSize=(14, 8),
      imagesPerClass=2,
      save=True,
      display=True,
      outPrefix="BUSI_AttentionMaps",
      dpi=300,
      fontSize=10,
      alpha=0.4,
      selectImages=None,
      allowedExtensions=(".jpg", ".jpeg", ".png", ".bmp"),
    )
  '''

  def __init__(
    self,
    baseFolder,
    folder,
    dataFolder,
    modelName,
    modelCheckpointPath,
    modelType="Timm",
    size=448,
    reshapeTransformSize=32,
    device=None,
  ):
    # Set device.
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Set image size.
    self.size = size
    # Set reshape transform size.
    self.reshapeTransformSize = reshapeTransformSize
    # Set base folder.
    self.baseFolder = baseFolder
    # Set results folder.
    self.folder = folder

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

  @staticmethod
  def ReshapeTransform(outputs, height, width):
    # Reshape transformer outputs for CAM.
    if isinstance(outputs, tuple):
      tensor = outputs[0]
    else:
      tensor = outputs
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
  ):
    '''
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
    '''

    # Set default CAMs if not provided.
    if (cams is None):
      cams = [GradCAM, ScoreCAM, EigenCAM]

    # Calculate total subplots.
    totalRows = len(self.classes)
    totalCols = len(cams) * imagesPerClass

    # Create figure with specified size.
    plt.figure(figsize=figSize)
    plotIdx = 1

    # Loop through each class.
    for c, cls in enumerate(self.classes):
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
      else:
        chosenImages = random.sample(imageFiles, min(imagesPerClass, len(imageFiles)))

      # Loop through selected images.
      for imgIdx, imageName in enumerate(chosenImages):
        imagePath = os.path.join(clsPath, imageName)

        # Open and preprocess image.
        image = Image.open(imagePath).convert("RGB")
        imgTensor = self.transform(image).unsqueeze(0).to(self.device)

        # Loop through CAM methods.
        for i, camCls in enumerate(cams):
          # Initialize CAM method.
          cam = camCls(
            model=self.model,
            target_layers=[self.vit_target_layer],
            ReshapeTransform=lambda x: self.ReshapeTransform(
              x,
              self.reshapeTransformSize,
              self.reshapeTransformSize
            ),
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

          # Add subplot.
          ax = plt.subplot(totalRows, totalCols, plotIdx)
          ax.imshow(overlay, vmin=0, vmax=255)

          # Set subplot title.
          ax.set_title(f"{cam_cls.__name__}", fontsize=fontSize)

          # Set class label on leftmost column.
          if (i == 0 and imgIdx == 0):
            ax.set_ylabel(cls, fontsize=fontSize, rotation=90, labelpad=40, va="center")

          ax.set_xticks([])
          ax.set_yticks([])
          plotIdx += 1

    # Tight layout to minimize wasted space.
    plt.tight_layout()

    # Generate output file path.
    rndTimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outPath = os.path.join(self.baseFolder, f"{outPrefix}_{rndTimestamp}.png")

    # Save the plot if requested.
    if (save):
      plt.savefig(outPath, dpi=dpi, bbox_inches="tight")

    # Display the plot if requested.
    if (display):
      plt.show()

    plt.close()
