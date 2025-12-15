# Import the required libraries.
import os  # Standard library for file and directory operations.
import yaml  # PyYAML library for YAML file parsing.
import pickle  # Pickle library for object serialization.
import json  # JSON library for JSON file parsing.
import csv  # CSV library for reading and writing CSV files.


def ReadProjectConfig(configFilePath):
  r'''
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

  Examples
  --------
  .. code-block:: python

    import HMB.Utils as utils

    config = utils.ReadProjectConfig("config.yaml")
    print(config["project_name"])
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
      text = file.read().strip()
      if (text == ""):
        raise ValueError("Configuration file is empty.")
      config = json.loads(text)
  # Validate empty content
  if (config is None):
    raise ValueError("Configuration file is empty or invalid.")
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

  import cv2  # OpenCV library for image processing.
  import numpy as np  # NumPy library for numerical operations.

  # Normalize contour to numpy int32 shape (N,1,2)
  cnt = np.array(contour, dtype=np.int32)
  if cnt.ndim == 2 and cnt.shape[1] == 2:
    cnt = cnt.reshape((-1, 1, 2))
  # If 'point' is actually a polygon, compute convex intersection area>0 with contour
  if isinstance(point, (list, tuple, np.ndarray)) and not (
    len(point) == 2 and not isinstance(point[0], (list, tuple, np.ndarray))):
    poly = np.array(point, dtype=np.float32)
    if poly.ndim == 2 and poly.shape[1] == 2:
      poly = poly.reshape((-1, 1, 2)).astype(np.float32)
      cntf = cnt.astype(np.float32)
      try:
        area, _ = cv2.intersectConvexConvex(poly, cntf)
        return area > 0
      except Exception:
        return False
  x, y = point
  flag = cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0
  return bool(flag)


def IsIntersectingWithOtherContours(pointOrPolygon, contours):
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

  # Handle None and degenerate
  for polygon in contours:
    if polygon is None:
      continue
    if IsPointInsideContour(pointOrPolygon, polygon):
      return True
  return False


def WritePickleFile(filePath, data):
  r'''
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
  r'''
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
  r'''
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
  r'''
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


def LoadYaml(yamlPath):
  r'''
  Load data from a YAML file.

  Parameters:
    yamlPath (str): Path to the YAML file.

  Returns:
    object: The data loaded from the YAML file as a Python object.

  Raises:
    AssertionError: If the specified file does not exist.
  '''

  # Check if the file exists.
  assert os.path.exists(yamlPath), f"File not found: {yamlPath}"

  # Open the YAML file in read mode and load its contents.
  with open(yamlPath, "r") as yamlFile:
    yamlData = yaml.load(yamlFile, Loader=yaml.FullLoader)
  return yamlData


def SaveYaml(yamlPath, yamlData):
  r'''
  Save data to a YAML file.

  Parameters:
    yamlPath (str): Path to the YAML file.
    yamlData (object): Data to be saved to the YAML file.
  '''

  # Open the YAML file in write mode and dump the data.
  with open(yamlPath, "w") as yamlFile:
    try:
      yaml.safe_dump(yamlData, yamlFile)
    except Exception as e:
      # Re-raise as ValueError to satisfy tests expecting error on anchors/non-serializable
      raise ValueError(f"Failed to serialize YAML data: {e}")


def Hex2RGB(hexColor, isRGBA=False):
  r'''
  Convert a hexadecimal color string to an RGB or RGBA tuple.

  Parameters:
    hexColor (str): Hexadecimal color string (e.g., "#RRGGBB" or "RRGGBB").
    isRGBA (bool): If True, return an RGBA tuple; otherwise, return an RGB tuple. Default is False.

  Returns:
    tuple: A tuple representing the RGB or RGBA color.

  .. code-block:: python

    import HMB.Utils as utils

    rgbColor = utils.Hex2RGB("#FF5733")  # Returns (255, 87, 51).
    rgbaColor = utils.Hex2RGB("#FF5733", isRGBA=True)  # Returns (255, 87, 51, 255).
    print(f"RGB Color for #FF5733: {rgbColor}")
    print(f"RGBA Color for #FF5733: {rgbaColor}")
  '''

  hexColor = hexColor.lstrip("#")
  if (len(hexColor) in (3, 4)):
    hexColor = ''.join([c * 2 for c in hexColor])
  if (len(hexColor) == 8):
    r = int(hexColor[0:2], 16)
    g = int(hexColor[2:4], 16)
    b = int(hexColor[4:6], 16)
    a = int(hexColor[6:8], 16)
    return (r, g, b, a) if isRGBA else (r, g, b)
  elif (len(hexColor) == 6):
    r = int(hexColor[0:2], 16)
    g = int(hexColor[2:4], 16)
    b = int(hexColor[4:6], 16)
    return (r, g, b, 255) if isRGBA else (r, g, b)
  else:
    raise ValueError("Invalid hex color length. Expected 3, 4, 6, or 8 characters.")


def GetCmapColors(cmap, noColors, darkColorsOnly=True, darknessThreshold=0.7):
  r'''
  Utility to get a list of unique colors from a matplotlib colormap.

  Parameters:
    cmap (str or Colormap): Name of the colormap or a Colormap object.
    noColors (int): Number of distinct colors to generate.
    darkColorsOnly (bool, optional): Whether to filter out light colors. Default is True.
    darknessThreshold (float, optional): Brightness threshold to classify colors as dark. Default is 0.7.

  Returns:
    list: List of unique RGBA color tuples.
  '''

  import matplotlib.pyplot as plt  # Matplotlib library for plotting and colormaps.

  # Filter out light colors based on perceived brightness (YIQ formula).
  def IsDark(color):
    r, g, b = color[:3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness < darknessThreshold

  # Get the colormap object.
  if (isinstance(cmap, str)):
    cmapObj = plt.get_cmap(cmap)
  else:
    cmapObj = cmap

  # To maximize uniqueness, sample evenly spaced points in the colormap.
  allColors = [cmapObj(i / max(noColors, 1)) for i in range(noColors)]

  if (darkColorsOnly):
    # Filter to keep only dark colors.
    darkColors = [c for c in allColors if IsDark(c)]

    # If not enough dark colors, try to fill with more from the colormap.
    if (len(darkColors) < noColors):
      # Try to get more dark colors by oversampling.
      extraColors = [cmapObj(i / 1000.0) for i in range(1000)]
      extraDark = [c for c in extraColors if (IsDark(c) and c not in darkColors)]
      darkColors.extend(extraDark)

      # Remove duplicates while preserving order.
      seen = set()
      uniqueDark = []
      for c in darkColors:
        if (c not in seen):
          uniqueDark.append(c)
          seen.add(c)
      darkColors = uniqueDark

    # If still not enough, repeat or fallback.
    if (len(darkColors) >= noColors):
      return darkColors[:noColors]
    elif (len(darkColors) > 0):
      repeats = (noColors // len(darkColors)) + 1
      extended = (darkColors * repeats)[:noColors]
      return extended
    else:
      return allColors
  else:
    # Remove duplicates if any (shouldn't be, but for safety).
    seen = set()
    uniqueColors = []
    for c in allColors:
      if (c not in seen):
        uniqueColors.append(c)
        seen.add(c)
    if (len(uniqueColors) >= noColors):
      return uniqueColors[:noColors]
    else:
      repeats = (noColors // len(uniqueColors)) + 1
      extended = (uniqueColors * repeats)[:noColors]
      return extended


def AppendOrCreateNewCSV(
  fileName,  # Path to the CSV file.
  data,  # Data to append or create.
  header=None,  # Header for the CSV file.
  mode="a",  # Mode to open the file (default is append).
):
  '''
  Append data to a CSV file or create a new one if it doesn't exist.

  Parameters:
    fileName (str): Path to the CSV file.
    data (list or dict): Data to append to the CSV file. Can be a list of rows (list of lists) or a dictionary.
    header (list, optional): Header for the CSV file. Required if creating a new file.
    mode (str, optional): Mode to open the file. Default is "a" (append).
  '''

  # Append data to a CSV file or create a new one if it doesn't exist.
  if (not os.path.exists(fileName)):
    # Create a new CSV file with the specified header.
    with open(fileName, "w", newline="") as f:
      writer = csv.writer(f)
      if (header is not None):
        writer.writerow(header)  # Write the header to the CSV file.

  # Append data to the CSV file.
  # newline="" is used to avoid extra blank lines in the CSV file.
  with open(fileName, mode, newline="") as f:
    writer = csv.writer(f)
    if (isinstance(data, list)):
      for row in data:
        # Validate row length against header if provided
        if (header is not None and isinstance(row, (list, tuple)) and len(row) != len(header)):
          raise ValueError("Row length does not match header length.")
        writer.writerow(row)  # Write each row of data to the CSV file.
    elif (isinstance(data, dict)):
      if (header is not None and len(list(data.values())) != len(header)):
        raise ValueError("Dict values length does not match header length.")
      writer.writerow(
        [data.get(h) for h in header]
        if (header is not None) else list(data.values())
      )  # Write the data to the CSV file.
    else:
      raise ValueError("Unsupported data type for CSV append.")


def AppendOrCreateNewDataFrameCSV(
  fileName,  # Path to the CSV file.
  data,  # Data to append or create.
  header=None,  # Header for the CSV file.
):
  '''
  Append a pandas DataFrame to a CSV file or create a new one if it doesn't exist.

  Parameters:
    fileName (str): Path to the CSV file.
    data (pandas.DataFrame): DataFrame to append to the CSV file.
    header (list, optional): Header for the CSV file. Required if creating a new file
  '''

  import pandas as pd

  # Append DataFrame to a CSV file or create a new one if it doesn't exist.
  if (not os.path.exists(fileName)):
    # Create new file; if header provided, ensure columns align
    cols = header if (header is not None) else list(data.columns)
    data.to_csv(fileName, index=False, header=cols)
  else:
    # Append without header
    data.to_csv(fileName, mode="a", index=False, header=False)
