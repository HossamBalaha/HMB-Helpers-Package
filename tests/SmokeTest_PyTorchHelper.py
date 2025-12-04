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
numSamples = 100
numClasses = 3
batchSize = 16
numEpochs = 100
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
  cv2.imwrite(imgPath, (img * 255).astype(np.uint8))

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
  nn.Linear(3 * 8 * 8, 32),
  nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(32, 64),
  nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(64, 32),
  nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(32, numClasses)
)

# Optimizer, criterion, scaler, scheduler.
optimizer = GetOptimizer(
  model,
  optimizerType="adamw",
  learningRate=0.001,
  weightDecay=1e-4
)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)

# Run training for N epochs on `cuda`.
device = torch.device("cuda")

# Make sure model is on the correct device.
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
  useAmp=True,
  useMixupFn=True,
  mixUpAlpha=0.4,
  useEma=True,
  saveEvery=None,
)

# print("History:", history)


InferenceWithPlots(
  dataDir=dataDir,  # Directory containing dataset.
  model=model,  # Model architecture.
  modelCheckpointName="best_model.pth",  # Path to model checkpoint.
  transform=None,  # Image transform to apply.
  device="cuda" if (torch.cuda.is_available()) else "cpu",  # Device to run inference on.
  batchSize=1,  # Batch size for inference.
  imageSize=imageSize[0],  # Image size for transforms.
  expDirs=["tests/PyTorchHelper"],  # List of experiment directories.
  overallResultsPath="tests/PyTorchHelper/Overall_Results.csv",  # Output CSV path for overall results.
  plotFontSize=16,  # Font size for plots.
  plotFigSize=(8, 8),  # Figure size for confusion matrix.
  rocFigSize=(5, 5),  # Figure size for ROC/PRC curves.
  dpi=720,  # DPI for saving plots.
  verbose=True,  # Whether to print progress.
)

# Clean up synthetic dataset and model checkpoint.
shutil.rmtree("tests/PyTorchHelper")
print("Smoke test for PyTorchHelper passed successfully.")
