'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 31th, 2025
# Last Modification Date: Jul 31th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import cv2, yaml, pickle
import numpy as np


def ReadProjectConfig(configFilePath):
  """
  Read the project configuration from a YAML file.

  Parameters:
    configFilePath (str): Path to the YAML configuration file.

  Returns:
    dict: Parsed configuration dictionary.
  """
  with open(configFilePath, "r") as file:
    config = yaml.safe_load(file)
  return config


def IsPointInsideContour(point, contour):
  """
  Check if a point is inside a contour.

  Parameters:
      point (tuple): Coordinates of the point (x, y).
      contour (numpy.ndarray): Contour to check against.

  Returns:
      bool: True if the point is inside the contour, otherwise False.
  """
  x, y = point  # Extract the x and y coordinates of the point.
  x = int(x)  # Convert x to an integer.
  y = int(y)  # Convert y to an integer.
  # Check if the point is inside the contour.
  flag = cv2.pointPolygonTest(contour, (x, y), False) >= 0
  return flag  # Return True if the point is inside the contour, otherwise False.


def IsIntersectingWithOtherContours(point, anListCoords):
  """
  Check if a point intersects with any other contours.

  Parameters:
    point (tuple): The point to check.
    anListCoords (list): List of coordinates of annotations.

  Returns:
    bool: True if the point intersects with any contour, False otherwise.
  """
  for coords in anListCoords:
    polygon = np.array(coords)
    if (IsPointInsideContour(point, polygon)):
      return True
  return False


def WritePickleFile(filePath, data):
  """
  Write data to a pickle file.

  Parameters:
    filePath (str): Path to the pickle file.
    data: Data to be written to the file.
  """
  with open(filePath, "wb") as f:
    pickle.dump(data, f)


def ReadPickleFile(filePath):
  """
  Read data from a pickle file.

  Parameters:
    filePath (str): Path to the pickle file.

  Returns:
    The data read from the pickle file.
  """
  with open(filePath, "rb") as f:
    data = pickle.load(f)
  return data
