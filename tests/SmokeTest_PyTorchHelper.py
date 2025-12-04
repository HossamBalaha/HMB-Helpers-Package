import torch, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler

from HMB.PyTorchHelper import TrainEvaluateModel, GetOptimizer
from HMB.Initializations import SeedEverything

SeedEverything(42)

# Small synthetic dataset.
numSamples = 256
numClasses = 10
batchSize = 8
numEpochs = 1000

# Inputs: 3x8x8 images flattened.
x = torch.randn(numSamples, 3, 8, 8)
# Targets: integers 0..num_classes-1.
y = torch.randint(0, numClasses, (numSamples,))

dataset = TensorDataset(x, y)
trainLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

# Simple model.
model = nn.Sequential(
  nn.Flatten(),
  nn.Linear(3 * 8 * 8, 32),
  nn.ReLU(),
  nn.Linear(32, numClasses)
)

# Optimizer, criterion, scaler, scheduler.
optimizer = GetOptimizer(
  model,
  optimizerType="Adam",
  learningRate=0.001,
  weightDecay=1e-4
)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Run training for 2 epochs on `cuda`.
device = torch.device("cuda")

# Make sure model is on the correct device.
model.to(device)

history = TrainEvaluateModel(
  model=model,
  criterion=criterion,
  device=device,
  bestModelStoragePath="tests/best_model.pth",
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
  earlyStoppingPatience=3,
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
