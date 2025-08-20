# Import the required libraries.
import torch, os


# Function to save a PyTorch model's state dictionary to a file.
def SaveModel(model, filename="model.pth"):
  '''
  Save the model state to a file.
  Parameters:
    model (torch.nn.Module): The model to save.
    filename (str): The name of the file to save the model to.
  '''

  # Save the model's state dictionary to the specified file.
  torch.save(model.state_dict(), filename)

  # Print confirmation message with filename.
  print(f"Model saved to {filename}. You can load it later using LoadModel().")


# Function to load a PyTorch model's state dictionary from a file and move it to a device.
def LoadModel(model, filename="model.pth", device="gpu"):
  '''
  Load the model state from a file.
  Parameters:
    model (torch.nn.Module): The model to load the state into.
    filename (str): The name of the file to load the model from.
    device (str): The device to load the model onto (e.g., "cpu" or "cuda").
  '''

  # Check if the model file exists before loading.
  if (not os.path.exists(filename)):
    print(f"Model file not found: {filename}")
    return

  # Load the state dictionary from file and map to the specified device.
  model.load_state_dict(torch.load(filename, map_location=device))
  # Move the model to the specified device.
  model.to(device)

  # Print confirmation message with filename and device.
  print(f"Model loaded from {filename} and moved to {device}.")


def SaveCheckpoint(model, optimizer, filename="chk.pth.tar"):
  '''
  Save model and optimizer state to a checkpoint file.
  Useful for resuming training or inference later.
  Parameters:
    model (torch.nn.Module): The model to save.
    optimizer (torch.optim.Optimizer): The optimizer to save.
    filename (str): The name of the file to save the checkpoint to.
  '''

  # Create a dictionary containing model and optimizer state.
  checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer" : optimizer.state_dict(),
  }
  # Save the checkpoint dictionary to the specified file.
  torch.save(checkpoint, filename)

  # Print confirmation message with filename.
  print(f"Checkpoint saved to {filename}. You can load it later using LoadCheckpoint().")


def LoadCheckpoint(checkpointFile, model, optimizer, lr, device):
  '''
  Load model and optimizer state from a checkpoint file.
  Updates the learning rate of the optimizer if provided.
  Parameters:
    checkpointFile (str): The path to the checkpoint file.
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    lr (float): The learning rate to set for the optimizer.
    device (torch.device): The device to load the model onto (e.g., "cpu" or "cuda").
  '''

  # Check if the checkpoint file exists before loading.
  if (not os.path.exists(checkpointFile)):
    print(f"Checkpoint file not found: {checkpointFile}")
    return

  # Load the checkpoint dictionary from file and map to the specified device.
  checkpoint = torch.load(checkpointFile, map_location=device)
  # Load the model state from the checkpoint.
  model.load_state_dict(checkpoint["state_dict"])

  # If optimizer is provided, load its state and update learning rate.
  if (optimizer is not None):
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update learning rate for all parameter groups in the optimizer.
    for paramGroup in optimizer.param_groups:
      paramGroup["lr"] = lr

  # Print confirmation message with checkpoint file and device.
  print(f"Checkpoint loaded from {checkpointFile} and model moved to {device}.")
