# Import the required libraries.
import os  # Standard library for file and directory operations.
import cv2  # OpenCV library for image processing.
import yaml  # PyYAML library for YAML file parsing.
import pickle  # Pickle library for object serialization.
import json  # JSON library for JSON file parsing.
import numpy as np  # NumPy library for numerical operations.


def ReadProjectConfig(configFilePath):
  '''
  Read the project configuration from the file. This function loads a configuration
  file in either YAML or JSON format and returns its contents as a dictionary.

  Parameters:
    configFilePath (str): Path to the configuration file.

  Returns:
    dict: Parsed configuration dictionary.

  Raises:
    ValueError: If the file format is not supported (not YAML or JSON).
  Raises:
    AssertionError: If the configuration file does not exist.
  '''

  # Check if the configuration file exists.
  assert os.path.exists(configFilePath), f"Configuration file not found: {configFilePath}"

  # Extract the file extension from the configuration file path.
  extension = configFilePath.split(".")[-1]
  # Check if the file extension is YAML or JSON.
  if (extension not in ["yaml", "yml", "json"]):
    # Raise an error if the file is not a YAML or JSON file.
    raise ValueError(f"Unsupported configuration file format: {extension}. Supported formats are YAML and JSON.")

  # If the file is a YAML file, parse it using PyYAML.
  if (extension in ["yaml", "yml"]):
    # Open the YAML configuration file in read mode.
    with open(configFilePath, "r") as file:
      # Parse the YAML file and load its contents into a dictionary.
      config = yaml.safe_load(file)
  # If the file is a JSON file, parse it using the json library.
  elif (extension == "json"):
    # Open the JSON configuration file in read mode.
    with open(configFilePath, "r") as file:
      # Parse the JSON file and load its contents into a dictionary.
      config = json.load(file)
  # Return the parsed configuration dictionary.
  return config


def IsPointInsideContour(point, contour):
  r'''
  Check if a point is inside a contour.

  .. math::
      \text{IsPointInsideContour}(point, contour) =
      \begin{cases}
      \text{True} & \text{if point is inside contour} \\
      \text{False} & \text{otherwise}
      \end{cases}

  This function uses OpenCV's pointPolygonTest to determine if a point is inside a given contour.
  It returns True if the point is inside the contour, otherwise it returns False.

  Parameters:
    point (tuple): Coordinates of the point (x, y).
    contour (numpy.ndarray): Contour to check against the point. It should be a NumPy array of shape (n, 2),
      where n is the number of points in the contour and each point is represented by its (x, y) coordinates.

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
  r'''
  Check if a point intersects with any other contours.

  .. math::
      \text{IsIntersectingWithOtherContours}(point, anListCoords) =
      \begin{cases}
      \text{True} & \text{if point intersects with any contour} \\
      \text{False} & \text{otherwise}
      \end{cases}

  This function checks if a given point intersects with any contours defined by a list of coordinates.
  It iterates through each set of coordinates in the list and uses the IsPointInsideContour function to check for intersection.
  If the point is found to be inside any contour, it returns True; otherwise, it returns False.

  Parameters:
    point (tuple): The point to check.
    anListCoords (list): List of coordinates of annotations. Each set of coordinates represents a contour where
      the first element is the x-coordinate and the second element is the y-coordinate.

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
    data (any): Data to be written to the file.
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
    object: The data read from the pickle file. Data type can be any Python object that was previously serialized and saved using pickle.

  Raises:
    AssertionError: If the specified file does not exist.
 '''

  # Check if the file exists.
  assert os.path.exists(filePath), f"File not found: {filePath}"

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

  Raises:
    AssertionError: If the specified file does not exist.
  '''

  # Check if the file exists.
  assert os.path.exists(filePath), f"File not found: {filePath}"

  # Open the file in read mode.
  with open(filePath, "r") as f:
    # Read the text from the file.
    text = f.read()
  # Return the read text.
  return text
