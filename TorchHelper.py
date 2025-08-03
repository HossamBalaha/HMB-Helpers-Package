'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Aug 1st, 2025
# Last Modification Date: Aug 2nd, 2025
# Permissions and Citation: Refer to the README file.
'''

import torch, os


def SaveModel(model, filename="model.pth"):
  """
  Save the model state to a file.
  Parameters:
    model (torch.nn.Module): The model to save.
    filename (str): The name of the file to save the model to.
  """

  torch.save(model.state_dict(), filename)

  print(f"Model saved to {filename}. You can load it later using LoadModel().")


def LoadModel(model, filename="model.pth", device="gpu"):
  """
  Load the model state from a file.
  Parameters:
    model (torch.nn.Module): The model to load the state into.
    filename (str): The name of the file to load the model from.
    device (str): The device to load the model onto (e.g., "cpu" or "cuda").
  """

  if (not os.path.exists(filename)):
    print(f"Model file not found: {filename}")
    return

  model.load_state_dict(torch.load(filename, map_location=device))
  model.to(device)

  print(f"Model loaded from {filename} and moved to {device}.")


def SaveCheckpoint(model, optimizer, filename="chk.pth.tar"):
  """
  Save model and optimizer state to a checkpoint file.
  Useful for resuming training or inference later.
  Parameters:
    model (torch.nn.Module): The model to save.
    optimizer (torch.optim.Optimizer): The optimizer to save.
    filename (str): The name of the file to save the checkpoint to.
  """
  checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer" : optimizer.state_dict(),
  }
  torch.save(checkpoint, filename)

  print(f"Checkpoint saved to {filename}. You can load it later using LoadCheckpoint().")


def LoadCheckpoint(checkpointFile, model, optimizer, lr, device):
  """
  Load model and optimizer state from a checkpoint file.
  Updates the learning rate of the optimizer if provided.
  Parameters:
    checkpointFile (str): The path to the checkpoint file.
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    lr (float): The learning rate to set for the optimizer.
    device (torch.device): The device to load the model onto (e.g., "cpu" or "cuda").
  """

  if (not os.path.exists(checkpointFile)):
    print(f"Checkpoint file not found: {checkpointFile}")
    return

  checkpoint = torch.load(checkpointFile, map_location=device)
  model.load_state_dict(checkpoint["state_dict"])

  if (optimizer is not None):
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update learning rate for all parameter groups in the optimizer.
    for paramGroup in optimizer.param_groups:
      paramGroup["lr"] = lr

  print(f"Checkpoint loaded from {checkpointFile} and model moved to {device}.")
