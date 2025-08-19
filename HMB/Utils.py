'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 31th, 2025
# Last Modification Date: Aug 19th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import cv2, yaml, pickle
import numpy as np


def ReadProjectConfig(configFilePath):
  '''
  Read the project configuration from a YAML file.

  Parameters:
    configFilePath (str): Path to the YAML configuration file.

  Returns:
    dict: Parsed configuration dictionary.
  '''

  # Open the YAML configuration file in read mode.
  with open(configFilePath, "r") as file:
    # Parse the YAML file and load its contents into a dictionary.
    config = yaml.safe_load(file)
  # Return the parsed configuration dictionary.
  return config


def IsPointInsideContour(point, contour):
  '''
  Check if a point is inside a contour.

  Parameters:
      point (tuple): Coordinates of the point (x, y).
      contour (numpy.ndarray): Contour to check against.

  Returns:
      bool: True if the point is inside the contour, otherwise False.
  '''

  # Extract the x and y coordinates of the point.
  x, y = point
  # Convert x to an integer.
  x = int(x)
  # Convert y to an integer.
  y = int(y)
  # Check if the point is inside the contour using OpenCV's pointPolygonTest.
  flag = cv2.pointPolygonTest(contour, (x, y), False) >= 0
  # Return True if the point is inside the contour, otherwise False.
  return flag


def IsIntersectingWithOtherContours(point, anListCoords):
  '''
  Check if a point intersects with any other contours.

  Parameters:
    point (tuple): The point to check.
    anListCoords (list): List of coordinates of annotations.

  Returns:
    bool: True if the point intersects with any contour, False otherwise.
  '''

  # Iterate through each set of coordinates in the list.
  for coords in anListCoords:
    # Convert the coordinates to a NumPy array representing a polygon.
    polygon = np.array(coords)
    # Check if the point is inside the current polygon.
    if (IsPointInsideContour(point, polygon)):
      # Return True if the point is inside any polygon.
      return True
  # Return False if the point does not intersect with any polygon.
  return False


def WritePickleFile(filePath, data):
  '''
  Write data to a pickle file.

  Parameters:
    filePath (str): Path to the pickle file.
    data: Data to be written to the file.
  '''

  # Open the file in write-binary mode.
  with open(filePath, "wb") as f:
    # Serialize and write the data to the file using pickle.
    pickle.dump(data, f)


def ReadPickleFile(filePath):
  '''
  Read data from a pickle file.

  Parameters:
    filePath (str): Path to the pickle file.

  Returns:
    The data read from the pickle file.
  '''

  # Open the file in read-binary mode.
  with open(filePath, "rb") as f:
    # Deserialize and load the data from the file using pickle.
    data = pickle.load(f)
  # Return the loaded data.
  return data


def WriteTextFile(filePath, text):
  '''
  Write text to a file.

  Parameters:
    filePath (str): Path to the text file.
    text (str): Text to be written to the file.
  '''

  # Open the file in write mode.
  with open(filePath, "w") as f:
    # Write the text to the file.
    f.write(text)


def ReadTextFile(filePath):
  '''
  Read text from a file.

  Parameters:
    filePath (str): Path to the text file.

  Returns:
    str: The text read from the file.
  '''

  # Open the file in read mode.
  with open(filePath, "r") as f:
    # Read the text from the file.
    text = f.read()
  # Return the read text.
  return text
