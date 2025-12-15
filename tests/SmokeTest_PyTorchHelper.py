import torch, random, os, shutil, cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler

from HMB.PyTorchHelper import TrainEvaluateModel, GetOptimizer, InferenceWithPlots
from HMB.Initializations import SeedEverything

SeedEverything(42)

# Small synthetic dataset.
numSamples = 60
numClasses = 3
batchSize = 8
numEpochs = 3
imageSize = (8, 8)

# Create synthetic dataset for inference and store it in the specified directory.
x = torch.randn(numSamples, 3, imageSize[0], imageSize[1])  # Random images.
y = torch.randint(0, numClasses, (numSamples,))  # Random labels.

dataDir = "tests/PyTorchHelper/SyntheticDataset"

if (os.path.exists(dataDir)):
  shutil.rmtree(dataDir)
os.makedirs(dataDir, exist_ok=True)
for i in range(numSamples):
  img = x[i].permute(1, 2, 0).numpy()  # Convert to HWC format.
  label = y[i].item()
  classDir = os.path.join(dataDir, str(label))
  os.makedirs(classDir, exist_ok=True)
  imgPath = os.path.join(classDir, f"img_{i}.png")
  cv2.imwrite(imgPath, np.clip((img * 255), 0, 255).astype(np.uint8))

# Create DataLoader.
dataset = TensorDataset(x, y)
trainSize = int(0.8 * numSamples)
valSize = numSamples - trainSize
trainDataset, valDataset = torch.utils.data.random_split(dataset, [trainSize, valSize])
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

# Simple model.
model = nn.Sequential(
  nn.Flatten(),
  nn.Linear(3 * 8 * 8, 16),
  nn.ReLU(),
  nn.Linear(16, numClasses)
)

# Optimizer, criterion, scaler, scheduler.
optimizer = GetOptimizer(
  model,
  optimizerType="adamw",
  learningRate=0.001,
  weightDecay=1e-4
)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler(enabled=False)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)

# Run training on CPU for portability.
device = torch.device("cpu")
model.to(device)

history = TrainEvaluateModel(
  model=model,
  criterion=criterion,
  device=device,
  bestModelStoragePath="tests/PyTorchHelper/best_model.pth",
  noOfClasses=numClasses,
  numEpochs=numEpochs,
  optimizer=optimizer,
  scaler=scaler,
  scheduler=scheduler,
  trainLoader=trainLoader,
  valLoader=valLoader,
  resumeFromCheckpoint=False,
  finalModelStoragePath=None,
  judgeBy="val_loss",
  earlyStoppingPatience=None,
  verbose=True,
  gradAccumSteps=1,
  maxGradNorm=5.0,
  useAmp=False,
  useMixupFn=False,
  useEma=False,
  saveEvery=None,
)

InferenceWithPlots(
  dataDir=dataDir,  # Directory containing dataset.
  model=model,  # Model architecture.
  modelCheckpointName="best_model.pth",  # Path to model checkpoint.
  transform=None,  # Image transform to apply.
  device="cpu",  # Device to run inference on.
  batchSize=2,  # Batch size for inference.
  imageSize=imageSize[0],  # Image size for transforms.
  expDirs=["tests/PyTorchHelper"],  # List of experiment directories.
  overallResultsPath="tests/PyTorchHelper/Overall_Results.csv",  # Output CSV path for overall results.
  plotFontSize=8,  # Font size for plots.
  plotFigSize=(4, 4),  # Figure size for confusion matrix.
  rocFigSize=(3, 3),  # Figure size for ROC/PRC curves.
  dpi=180,  # DPI for saving plots.
  verbose=False,  # Whether to print progress.
)

# Clean up synthetic dataset and model checkpoint.
shutil.rmtree("tests/PyTorchHelper")
print("Smoke test for PyTorchHelper (CPU-only) passed successfully.")
