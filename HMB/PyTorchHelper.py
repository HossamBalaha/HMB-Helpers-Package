import os
import torch
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
  '''
  PyTorch dataset for image classification tasks, loading images from a directory
  structure where each class has its own subfolder.

  Parameters:
    dataDir (str): Path to the root directory containing class subfolders with images.
    transform (callable, optional): Optional transform to be applied on a sample.
    allowedExtensions (tuple, optional): Tuple of allowed image file extensions. Defaults to (".png", ".jpg", ".jpeg").
  '''

  def __init__(
    self,
    dataDir,
    transform=None,
    allowedExtensions=(".png", ".jpg", ".jpeg")
  ):
    '''
    Initialize the custom dataset for image classification tasks.

    Parameters:
      dataDir (str): Path to the root directory containing class subfolders with images.
      transform (callable, optional): Optional transform to be applied on a sample.
      allowedExtensions (tuple, optional): Tuple of allowed image file extensions. Defaults to (".png", ".jpg", ".jpeg").
    '''

    # Setting data directory.
    self.dataDir = dataDir
    # Setting transform.
    self.transform = transform
    # Getting classes.
    self.classes = sorted(os.listdir(dataDir))
    # Creating class to index mapping.
    self.classToIdx = {}
    # Initializing samples list.
    self.samples = []
    for idx, cls in enumerate(self.classes):
      self.classToIdx[cls] = idx
      clsDir = os.path.join(dataDir, cls)
      if (not os.path.isdir(classDir)):
        continue
      for fname in os.listdir(clsDir):
        if (fname.lower().endswith(allowedExtensions)):
          # Getting image path.
          path = os.path.join(clsDir, fname)
          self.samples.append((path, idx))

  def __len__(self):
    '''
    Get the total number of samples in the dataset.

    Returns:
      int: Number of samples in the dataset.
    '''

    return len(self.samples)

  def __getitem__(self, idx):
    '''
    Retrieve an image and its label by index.

    Parameters:
      idx (int): Index of the sample to retrieve.

    Returns:
      tuple: (image, label) where image is a PIL Image or transformed tensor, and label is an int class index.
    '''

    path, label = self.samples[idx]
    img = Image.open(path).convert("RGB")
    if (self.transform):
      img = self.transform(img)
    return img, label
