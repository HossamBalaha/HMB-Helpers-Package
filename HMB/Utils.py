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
    print(config["projectName"])
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


def ConvertToJsonSerializable(obj):
  r'''
  Convert an object to a JSON-serializable format.
  This function attempts to convert various types of objects into formats that can be serialized to JSON.
  It handles common data types such as NumPy arrays, PyTorch tensors, TensorFlow tensors, bytes, and objects
  with __dict__ attributes. If an object cannot be converted to a JSON-serializable format, it falls back to
  using the string representation of the object. The function is designed to be robust and handle exceptions
  gracefully, ensuring that it can process a wide range of input types without crashing. It also includes
  structured representations for tensors to preserve data and metadata when possible, while providing fallback
  options when conversions fail. This makes it suitable for use in scenarios where you need to serialize complex
  objects to JSON, such as logging, configuration saving, or data interchange between systems that may include
  machine learning models and their parameters.

  Parameters:
    obj (any): The object to be converted to a JSON-serializable format.

  Returns:
    A JSON-serializable representation of the input object. The return type can vary depending on the input:
      - For NumPy arrays, it returns a dictionary containing the data as a list, shape, and data type.
      - For PyTorch tensors, it returns a dictionary with similar structure to preserve tensor information.
      - For TensorFlow tensors, it also returns a structured dictionary with data, shape, and data type.
      - For bytes and bytearrays, it attempts to decode them as UTF-8 strings, and if that fails, it returns their hexadecimal representation.
      - For objects with a __dict__ attribute, it returns a dictionary of their attributes, attempting to serialize each attribute directly or converting it to a string if it is not JSON-serializable.
      - For all other types of objects, it returns their string representation as a last resort. If the string conversion also fails, it returns None.
  '''

  # Initialize the numpy module variable.
  numpyModule = None
  # Initialize the torch module variable.
  torchModule = None
  # Initialize the tensorflow module variable.
  tensorflowModule = None
  # Attempt to import numpy.
  try:
    # Import numpy as numpyModule.
    import numpy as numpyModule
  except Exception:
    # Set numpyModule to None if import fails.
    numpyModule = None
  # Attempt to import torch.
  try:
    # Import torch as torchModule.
    import torch as torchModule
  except Exception:
    # Set torchModule to None if import fails.
    torchModule = None
  # Attempt to import tensorflow.
  try:
    # Import tensorflow as tensorflowModule.
    import tensorflow as tensorflowModule
  except Exception:
    # Set tensorflowModule to None if import fails.
    tensorflowModule = None
  # Handle torch device and dtype objects.
  try:
    # Check if torchModule is available.
    if (torchModule is not None):
      # Check if obj is a torch device or dtype.
      if (
        isinstance(obj, torchModule.device) or
        isinstance(obj, torchModule.dtype)
      ):
        # Return the string representation of the object.
        return str(obj)
  except Exception:
    # Pass silently if torch checks fail.
    pass
  # Handle torch Size and Python sets.
  try:
    # Check if torchModule is available.
    if (torchModule is not None):
      # Check if obj is a torch Size or a set.
      if (
        isinstance(obj, torchModule.Size) or
        isinstance(obj, set)
      ):
        # Return the list representation of the object.
        return list(obj)
  except Exception:
    # Pass silently if torch checks fail.
    pass
  # Handle torch Tensor objects.
  try:
    # Check if torchModule is available.
    if (torchModule is not None):
      # Check if obj is a torch Tensor.
      if (isinstance(obj, torchModule.Tensor)):
        # Move tensor to CPU and detach to avoid GPU issues.
        try:
          # Detach and move tensorObj to CPU.
          tensorObj = obj.detach().cpu()
        except Exception:
          # Assign obj to tensorObj if detach fails.
          tensorObj = obj
        # Attempt to convert tensor to list for data preservation.
        try:
          # Convert tensorObj to list format.
          dataList = tensorObj.tolist()
          # Return structured dict for reconstruction capability.
          return {"__Tensor__": True, "Data": dataList, "Shape": list(tensorObj.shape), "Dtype": str(tensorObj.dtype)}
        except Exception:
          # Return string representation if list conversion fails.
          return str(tensorObj)
  except Exception:
    # Pass silently if torch tensor checks fail.
    pass
  # Handle TensorFlow Tensor and Variable objects.
  try:
    # Check if tensorflowModule is available.
    if (tensorflowModule is not None):
      # Check if obj is a TensorFlow Tensor or Variable.
      if (
        isinstance(obj, tensorflowModule.Tensor) or
        isinstance(obj, tensorflowModule.Variable)
      ):
        # Attempt to convert to numpy array.
        try:
          # Convert tensorflow object to numpy array.
          numpyArray = obj.numpy()
          # Convert numpy array to list format.
          dataList = numpyArray.tolist()
          # Return structured dict for reconstruction capability.
          return {"__TensorFlowTensor__": True, "Data": dataList, "Shape": list(numpyArray.shape),
                  "Dtype"               : str(numpyArray.dtype)}
        except Exception:
          # Return a summary dict if numpy conversion fails.
          return {"__TensorFlowTensor__": True, "Shape": str(obj.shape), "Dtype": str(obj.dtype)}
  except Exception:
    # Pass silently if tensorflow checks fail.
    pass
  # Handle NumPy scalar types and arrays.
  if (numpyModule is not None):
    try:
      # Check if obj is a numpy integer or floating point scalar.
      if (isinstance(obj, (numpyModule.integer, numpyModule.floating))):
        # Return the Python scalar value.
        return obj.item()
      # Check if obj is a numpy ndarray.
      if (isinstance(obj, numpyModule.ndarray)):
        # Convert array to list format for data preservation.
        try:
          # Convert numpyArray to list format.
          dataList = obj.tolist()
          # Return structured dict for reconstruction capability.
          return {"__NdArray__": True, "Data": dataList, "Shape": list(obj.shape), "Dtype": str(obj.dtype)}
        except Exception:
          # Return string representation if conversion fails.
          return str(obj)
    except Exception:
      # Pass silently if numpy checks fail.
      pass
  # Handle bytes and bytearray objects.
  try:
    # Check if obj is bytes or bytearray.
    if (isinstance(obj, (bytes, bytearray))):
      # Attempt to decode as utf-8.
      try:
        # Return the decoded string.
        return obj.decode("utf-8")
      except Exception:
        # Return the hex representation if decode fails.
        return obj.hex()
  except Exception:
    # Pass silently if bytes checks fail.
    pass
  # Fallback to object dictionary representation.
  try:
    # Check if obj has a __dict__ attribute.
    if (hasattr(obj, "__dict__")):
      # Initialize the result dictionary.
      resultDict = {}
      # Iterate over items in the __dict__ attribute.
      for key, value in obj.__dict__.items():
        # Attempt to serialize the value directly.
        try:
          # Validate value with json.dumps.
          json.dumps(value)
          # Assign value to resultDict with key.
          resultDict[key] = value
        except TypeError:
          # Attempt to convert value to string.
          try:
            # Assign string representation to resultDict.
            resultDict[key] = str(value)
          except Exception:
            # Assign None if string conversion fails.
            resultDict[key] = None
      # Return the constructed result dictionary.
      return resultDict
  except Exception:
    # Pass silently if __dict__ checks fail.
    pass
  # Last-resort conversion to string.
  try:
    # Return the string representation of obj.
    return str(obj)
  except Exception:
    # Return None if string conversion fails.
    return None


def SimpleSerializeForJson(obj):
  r'''
  Serialize an object into JSON-serializable primitives.

  This function attempts to convert a variety of Python objects into structures
  that can be safely passed to json.dumps(). It performs a best-effort,
  recursive conversion for common types encountered in scientific and
  machine-learning workflows:
  - NumPy ndarrays -> nested lists
  - NumPy scalars -> native Python scalars
  - dict -> keys converted to strings and values serialized recursively
  - list/tuple -> serialized element-wise to a list
  - numeric types -> returned as-is
  - objects exposing a `tolist()` method -> converted via tolist() and serialized recursively

  If an object cannot be converted via the above rules, the function falls
  back to using str(obj) for objects that expose `tolist()` but fail, or
  returns the object unchanged as a last resort.

  Parameters:
    obj (any): The object to convert to JSON-serializable form.

  Returns:
    any: A JSON-serializable representation of `obj` (nested dicts/lists/primitives), or the original object when no conversion was applicable.
  '''

  # Note: If you need a more robust serializer that preserves type metadata
  # (dtype, shape) and includes explicit type tags for reconstruction of
  # tensors/arrays (NumPy, PyTorch, TensorFlow), consider using
  # `ConvertToJsonSerializable` which is designed as a best-effort converter
  # for many ML/data types. `ConvertToJsonSerializable` preserves metadata
  # and adds type tags so data can be reconstructed or identified later;
  # it also tries many conversion strategies and falls back to string or
  # None when conversion isn't possible.

  import numbers
  import numpy as np

  # Check whether the object is a NumPy ndarray.
  if (isinstance(obj, np.ndarray)):
    # Convert the ndarray to a nested list and serialize recursively.
    return SimpleSerializeForJson(obj.tolist())
  # Check whether the object is a NumPy scalar.
  if (isinstance(obj, np.generic)):
    # Convert the NumPy scalar to a native Python scalar and return.
    return obj.item()
  # Check whether the object is a dictionary.
  if (isinstance(obj, dict)):
    # Serialize dictionary keys and values recursively into JSON-serializable types.
    return {str(k): SimpleSerializeForJson(v) for k, v in obj.items()}
  # Check whether the object is a list or tuple.
  if (isinstance(obj, (list, tuple))):
    # Serialize each element of the list or tuple recursively.
    return [SimpleSerializeForJson(v) for v in obj]
  # Check whether the object is a numeric scalar.
  if (isinstance(obj, numbers.Number)):
    # Return numeric scalars directly as they are JSON-serializable.
    return obj
  # Check whether the object exposes a tolist method for conversion.
  if (hasattr(obj, "tolist")):
    try:
      # Attempt to convert the object via tolist and serialize recursively.
      return SimpleSerializeForJson(obj.tolist())
    except Exception:
      # Fallback to string representation when tolist conversion fails.
      return str(obj)
  # Return the object unchanged when no special handling applies.
  return obj


def DumpJsonFile(filePath, data, indent=2, ensureAscii=False):
  r'''
  Dump data to a JSON file.

  Parameters:
    filePath (str): Path to the JSON file.
    data (object): Data to be dumped to the JSON file.
    indent (int, optional): Number of spaces for indentation in the JSON file. Default is 2.
    ensureAscii (bool, optional): If True, all non-ASCII characters in the output are escaped. Default is False.
  '''

  # Open the JSON file in write mode and dump the data.
  with open(filePath, "w", encoding="utf-8") as jsonFile:
    json.dump(data, jsonFile, indent=indent, ensure_ascii=ensureAscii)


def ReadJsonFile(filePath):
  r'''
  Read data from a JSON file.

  Parameters:
    filePath (str): Path to the JSON file.

  Returns:
    object: The data read from the JSON file as a Python object.

  Raises:
    AssertionError: If the specified file does not exist.
  '''

  # Check if the file exists.
  assert os.path.exists(filePath), f"File not found: {filePath}"
  # Open the JSON file in read mode and load its contents.
  with open(filePath, "r", encoding="utf-8") as jsonFile:
    try:
      jsonData = json.load(jsonFile)
    except Exception:
      jsonFile.seek(0)
      jsonData = yaml.safe_load(jsonFile)
  return jsonData


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


def SaveYaml(yamlPath, yamlData, safe=True):
  r'''
  Save data to a YAML file.

  Parameters:
    yamlPath (str): Path to the YAML file.
    yamlData (object): Data to be saved to the YAML file.
  '''

  # Open the YAML file in write mode and dump the data.
  with open(yamlPath, "w") as yamlFile:
    if (not safe):
      with open(yamlPath, "w") as yamlFile:
        yaml.dump(yamlData, yamlFile)
    else:
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


def AppendOrCreateNewCSV(
  fileName,  # Path to the CSV file.
  data,  # Data to append or create.
  header=None,  # Header for the CSV file.
  mode="a",  # Mode to open the file (default is append).
):
  r'''
  Append data to a CSV file or create a new one if it doesn't exist.

  Parameters:
    fileName (str): Path to the CSV file.
    data (list or dict): Data to append to the CSV file. Can be a list of rows (list of lists) or a dictionary.
    header (list, optional): Header for the CSV file. Required if creating a new file.
    mode (str, optional): Mode to open the file. Default is "a" (append). It can be changed to "w" (write) if needed.
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
  if (header is not None):
    mode = "a"  # Ensure append mode when header is provided.
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


# Define the AppendOrCreateNewDataFrameCSV function to append or create a CSV from list or DataFrame.
def AppendOrCreateNewDataFrameCSV(
  fileName,  # Path to the CSV file.
  data,  # Data to append or create; can be a list (of lists/dicts) or a pandas DataFrame.
  header=None,  # Header for the CSV file; required if creating a new file and data is not a DataFrame with columns.
):
  r'''
  Append data to a CSV file or create a new one if it doesn't exist.
  Accepts data as either a pandas DataFrame or a list (of lists or dictionaries).

  Parameters:
    fileName (str): Path to the CSV file.
    data (list or pandas.DataFrame): Data to write. If list, must align with header.
    header (list, optional): Column names. Required if creating a new file and data lacks column info.
  '''

  # Import pandas locally to avoid global dependency.
  import pandas as pd

  # Convert input data to a pandas DataFrame if it is not already one.
  if (isinstance(data, pd.DataFrame)):
    # Use the DataFrame as-is.
    df = data
  elif (isinstance(data, list)):
    # Check if the list is non-empty and contains dictionaries (record-style).
    if (len(data) > 0 and isinstance(data[0], dict)):
      # Construct DataFrame from list of dictionaries.
      df = pd.DataFrame(data)
    else:
      # Assume list of lists/rows; require header for column names.
      if (header is None):
        # Raise an error if header is missing when needed.
        raise ValueError("Header must be provided when data is a list of values (not dicts).")
      # Construct DataFrame using the provided header as columns.
      df = pd.DataFrame(data, columns=header)
  else:
    # Raise an error for unsupported data types.
    raise TypeError("Data must be a pandas DataFrame or a list (of lists or dicts).")

  # Check whether the CSV file already exists.
  if (not os.path.exists(fileName)):
    # File does not exist; create a new one with header.
    finalHeader = header if (header is not None) else df.columns.tolist()
    # Write the DataFrame to a new CSV file with the specified header.
    df.to_csv(fileName, index=False, header=finalHeader)
  else:
    # File exists; append without writing the header.
    df.to_csv(fileName, mode="a", index=False, header=False)


def GroupImagesByClass(inputDir, imgExtensions=None):
  r'''

  Collect image paths grouped by class directory.

  This implementation:
  - walks the input directory recursively once
  - accepts a wider set of image extensions (case-insensitive)
  - groups images by the top-level directory under `inputDir` (so nested subfolders such as augmentation folders don't split a class into multiple keys)
  - sorts file lists deterministically and prints counts per class

  Parameters:
    inputDir (str): Path to the input directory containing images.
    imgExtensions (set, optional): Set of image file extensions to consider. Default includes common formats.

  Returns:
    dict: A dictionary where keys are class names (top-level folder names) and values are lists of image file paths.
  '''

  from pathlib import Path

  if (imgExtensions is None):
    imgExtensions = {
      ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif",
      ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".TIF", ".GIF",
    }

  imageGroups = {}
  inputPath = Path(inputDir)
  if (not inputPath.exists()):
    return imageGroups

  # Walk once and filter by extension (case-insensitive)
  for path in inputPath.rglob("*"):
    if (not path.is_file()):
      continue
    if (path.suffix.lower() not in imgExtensions):
      continue

    # Determine class name as the top-level folder under inputDir, if any.
    # Example: inputDir/className/aug1/image.jpg -> className
    try:
      rel = path.relative_to(inputPath)
      parts = rel.parts
      if (len(parts) >= 2):
        className = parts[0]
      elif (len(parts) == 1):
        # Image directly inside inputDir: group under the input directory name.
        className = inputPath.name
      else:
        className = path.parent.name
    except Exception:
      className = path.parent.name

    imageGroups.setdefault(className, []).append(path)

  # Sort keys and file lists for deterministic behavior and print counts
  for className in sorted(imageGroups.keys()):
    imageGroups[className] = sorted(imageGroups[className])
    print(f"  Class '{className}': {len(imageGroups[className])} images")

  return imageGroups


def SelectBalancedImages(imageGroups, maxImages, seed=42):
  r'''
  Select images evenly across classes up to maxImages. If there are not enough images,
  fill the remaining slots with random images from the available pool.
  The `imageGroups` parameter is a dictionary where keys are class names and values are lists of image file paths.
  You can obtain such a dictionary using the `GroupImagesByClass` function.

  Parameters:
    imageGroups (dict): Dictionary where keys are class names and values are lists of image file paths.
    maxImages (int): Maximum number of images to select.
    seed (int, optional): Random seed for reproducibility. Default is 42.

  Returns:
    list: List of selected image file paths.
  '''

  import numpy as np  # NumPy library for numerical operations.

  # Select a balanced subset of images across classes.
  randomGenerator = np.random.default_rng(seed)
  classNames = sorted(imageGroups.keys())
  print(f"Selecting balanced images across {len(classNames)} classes")
  if (len(classNames) == 0):
    return []
  perClass = max(1, maxImages // len(classNames))
  print(f"Selecting up to {perClass} images per class for {len(classNames)} classes")
  selected = []
  for className in classNames:
    classImages = imageGroups[className]
    if (len(classImages) > perClass):
      chosen = randomGenerator.choice(classImages, size=perClass, replace=False)
    else:
      chosen = classImages
    selected.extend(list(chosen))
    print(f"  Class '{className}': selected {len(chosen)}/{len(classImages)} images")
  remaining = maxImages - len(selected)
  if (remaining > 0):
    pool = [path for className in classNames for path in imageGroups[className] if (path not in selected)]
    if (len(pool) > 0):
      extra = min(remaining, len(pool))
      selected.extend(list(randomGenerator.choice(pool, size=extra, replace=False)))
  print(f"Total selected images: {len(selected)}")
  return selected[:maxImages]


def PrintHyperParamsList(hparamsFile, returnList=False):
  r'''
  Print the list of hyperparameter sets from a hyperparameters file.

  Parameters:
    hparamsFile (str): Path to the hyperparameters file.
    returnList (bool, optional): If True, return the list of hyperparameter sets instead of printing. Default is False.

  Returns:
    list or None: If `returnList` is True, returns the list of hyperparameter sets; otherwise, returns None.
  '''

  from pathlib import Path

  # Convert the input file path to a Path object for robust path handling.
  hparamsPath = Path(hparamsFile)

  # Attempt to read and parse the hyperparameters file as JSON.
  try:
    # Load the JSON content from the specified file.
    data = ReadJsonFile(hparamsPath)
  except Exception as e:
    # Report an error if the file cannot be read or parsed.
    print(f"ERROR: Could not read hyperparameters file for listing: {e}")
    # Return an empty list if returnList is True, otherwise return None.
    return [] if returnList else None

  # Verify that the top-level JSON structure is a list.
  if (not isinstance(data, list)):
    # Report an error if the file does not contain a JSON array.
    print(f"ERROR: Hyperparameters file does not contain a list: {hparamsPath}")
    # Return an empty list if returnList is True, otherwise return None.
    return [] if returnList else None

  # Initialize an empty list to store formatted hyperparameter summaries if needed.
  result = []

  # Print the header for the hyperparameter listing.
  print("Hyperparameter sets (name - active):")

  # Iterate over each hyperparameter set in the list with a 1-based index.
  for idx, hp in enumerate(data, start=1):
    # Check if the current item is a dictionary; if not, treat it as a raw value.
    if (isinstance(hp, dict)):
      # Extract the "name" field, defaulting to a generated name if missing.
      name = hp.get("name", f"unnamed_{idx}")
      # Extract the "active" field and convert it to a boolean; default to False if missing.
      active = bool(hp.get("active", False))
    else:
      # For non-dictionary entries, use string representation as the name.
      name = str(hp)
      # Mark as inactive since no 'active' key can exist.
      active = False

    # Format the display string for this hyperparameter set.
    line = f"  {idx}. {name} - active={active}"

    # Print the formatted line.
    print(line)

    # If returnList is True, append the original hyperparameter entry to the result.
    if (returnList):
      result.append(hp)

    # Print the contents of each hparams dict.
    print(f"     Contents ({type(hp)}): {hp}")

  # Return the list of hyperparameter sets if requested; otherwise, return None.
  return result if (returnList) else None


def FormatNumericWithDelta(value, baseValue=None, fmt="{:.2f}"):
  r'''
  Format a numeric value with its percentage delta relative to a baseline.

  Parameters:
    value (float or None): The current value to format. If None, "N/A" will be returned.
    baseValue (float or None): The baseline value for comparison. If None or zero, the delta will not be computed
      to avoid division by zero.
    fmt (str, optional): A format string for the value (default is "{:.2f}"). This should be a valid Python format
      string that can be used with the `format` method to format the value. For example, "{:.2f}" will format the
      value to two decimal places, while "{:.1f}" will format it to one decimal place. You can customize this
      format string based on your specific needs for displaying the value. The function will use this format
      string to format the value before appending the percentage delta in parentheses. If the value is missing
      (None), it will return "N/A" regardless of the format string. If the baseValue is missing or zero, it will
      return the formatted value without the delta to avoid division by zero errors.

  Returns:
    str: A formatted string that includes the value and its percentage delta relative to the baseline. The format of the returned string will be:
      - If value is None: "N/A"
      - If baseValue is None or zero: the formatted value without delta (e.g., "123.45")
      - Otherwise: the formatted value followed by the percentage delta in parentheses (e.g., "123.45 (+10.0%)" or "123.45 (-5.0%)").
  '''

  # Return placeholder when value is missing.
  if (value is None):
    return "N/A"
  # Avoid division-by-zero when baseValue is 0 or missing.
  if ((baseValue is None) or (baseValue == 0)):
    return fmt.format(value)
  # Compute percentage delta relative to baseline.
  delta = (value - baseValue) / baseValue * 100.0
  sign = "+" if (delta >= 0) else ""
  # Return formatted string including delta sign.
  return f"{fmt.format(value)} ({sign}{delta:.1f}%)"


def SafeCall(name, fn, *args, **kwargs):
  r'''
  Safely call a function and print its result or any exceptions. It also prints a separator line for clarity.

  Parameters:
    name (str): Name of the function being called (for reporting).
    fn (callable): The function to call.
    *args: Positional arguments to pass to the function.
    **kwargs: Keyword arguments to pass to the function.

  Returns:
    The result of the function call if successful, or None if an exception occurred.
  '''

  # Attempt to execute the target function with provided arguments.
  try:
    res = fn(*args, **kwargs)
    # Print the function name and its returned result.
    print(f"{name} ->", res)
    # Print a visual separator line.
    print("-" * 40)
    # Return the computed result to the caller.
    return res
  # Catch any runtime exceptions during function execution.
  except Exception as e:
    # Print the function name and exception details.
    print(f"{name} raised {type(e).__name__}:", e)
    # Print a visual separator line.
    print("-" * 40)
    # Return None to indicate failure.
    return None


def SafeTrapz(y, x=None):
  r'''
  Robust trapezoidal integration wrapper.

  Tries to use numpy.trapz when available. If NumPy is missing or the attribute
  is unavailable (some unusual environments), falls back to a pure-Python
  implementation that computes the trapezoidal rule over the provided samples.

  Parameters:
    y (sequence): y values (list/tuple/ndarray)
    x (sequence|None): x coordinates. If None, samples are assumed to be equally
      spaced at integer positions 0..len(y)-1.

  Returns:
    float: Approximated integral value (area).
  '''

  try:
    import numpy as np
    # Prefer numpy.trapz when available.
    if (hasattr(np, "trapz")):
      return float(np.trapz(y, x=x))
  except Exception:
    pass

  # Fallback pure-Python trapezoidal integration.
  try:
    ylist = list(y)
    if (x is None):
      xlist = list(range(len(ylist)))
    else:
      xlist = list(x)
    if (len(ylist) != len(xlist)):
      # Shapes mismatch - cannot integrate.
      raise ValueError("y and x must have the same length for trapezoidal integration.")
    area = 0.0
    for i in range(len(ylist) - 1):
      yi = float(ylist[i])
      yi1 = float(ylist[i + 1])
      xi = float(xlist[i])
      xi1 = float(xlist[i + 1])
      area += 0.5 * (yi + yi1) * (xi1 - xi)
    return area
  except Exception:
    # As a last resort, return 0.0.
    return 0.0


def SafeParseProbabilities(inputVar):
  r'''
  Parse a probabilities object into a Python list of floats robustly.

  Handling order (best-effort):
    - None -> [].
    - numeric scalar -> [float].
    - list/tuple/ndarray -> list of floats.
    - JSON string (e.g. "[0.1, 0.9]").
    - Python literal string (e.g. "[0.1, 0.9]").
    - strings containing NaN/nan -> handled (uses np.nan when numpy available).

  Returns an empty list on unrecoverable parse errors.

  Parameters:
    inputVar (any): The input variable to parse as probabilities. This can be of various types, including None,
      numeric scalars, lists, tuples, numpy arrays, or strings representing lists of probabilities.

  Returns:
    list: A list of floats representing the parsed probabilities. If the input cannot be parsed, an empty list is returned. The function is designed to be robust and handle a wide range of input formats gracefully, making it suitable for parsing user input or configuration values that may be provided in different forms.
  '''

  # Import the standard json module for string decoding.
  import json
  # Import the standard ast module for literal evaluation.
  import ast
  # Import the standard re module for pattern matching.
  import re
  # Import the numpy module under an alias for optional dependency.
  import numpy as np

  # Check if the provided input variable is exactly None.
  if (inputVar is None):
    # Provide an empty list as the result for null inputs.
    return []

  # Verify if the input variable is a boolean type before numeric checks.
  if (isinstance(inputVar, bool)):
    # Exclude boolean values from numeric processing and return an empty list.
    return []

  # Determine if the input variable is a numeric integer or floating point value.
  if (isinstance(inputVar, (int, float))):
    # Attempt to safely cast the numeric value to a standard float.
    try:
      # Wrap the converted float in a list and return it immediately.
      return [float(inputVar)]
    # Catch any unexpected errors during the type conversion process.
    except Exception:
      # Provide an empty list when the numeric conversion fails.
      return []

  # Check if numpy is available and if the input is a numpy array.
  if (isinstance(inputVar, np.ndarray)):
    # Attempt to transform the numpy array into a standard python list.
    try:
      # Convert each array element to a float and return the new list.
      return [float(arrayElement) for arrayElement in inputVar.tolist()]
    # Handle any failures that occur during the array conversion.
    except Exception:
      # Provide an empty list when the array processing fails.
      return []

  # Identify if the input variable is a native list or tuple sequence.
  if (isinstance(inputVar, (list, tuple))):
    # Create an empty container to hold the successfully parsed values.
    convertedList = []
    # Loop through every element contained within the input sequence.
    for listElement in inputVar:
      # Attempt to cast the current element to a floating point number.
      try:
        # Store the successfully converted float in a temporary variable.
        convertedValue = float(listElement)
        # Add the converted value to the accumulation container.
        convertedList.append(convertedValue)
      # Capture any errors that arise from invalid element types.
      except Exception:
        # Skip the problematic element and continue processing the sequence.
        continue
    # Output the fully populated list of converted probability values.
    return convertedList

  # Evaluate whether the input variable is a string type requiring parsing.
  if (isinstance(inputVar, str)):
    # Strip all leading and trailing whitespace characters from the string.
    trimmedString = inputVar.strip()
    # Begin the first parsing attempt using the json decoder.
    try:
      # Decode the json formatted text into a corresponding python object.
      parsedObject = json.loads(trimmedString)
      # Verify if the decoded object represents a sequence of values.
      if (isinstance(parsedObject, (list, tuple))):
        # Convert every item in the decoded sequence to a float.
        return [float(sequenceItem) for sequenceItem in parsedObject]
      # Wrap the single decoded value in a list after float conversion.
      return [float(parsedObject)]
    # Proceed past this block when json decoding raises an exception.
    except Exception:
      # Continue execution to the next parsing strategy.
      pass
    # Begin the second parsing attempt using the abstract syntax tree evaluator.
    try:
      # Safely evaluate the string as a python literal structure.
      parsedObject = ast.literal_eval(trimmedString)
      # Confirm whether the evaluated result forms a sequence.
      if (isinstance(parsedObject, (list, tuple))):
        # Transform each sequence component into a floating point number.
        return [float(sequenceItem) for sequenceItem in parsedObject]
      # Convert the singular evaluated result to a float and return it.
      return [float(parsedObject)]
    # Proceed past this block when literal evaluation raises an exception.
    except Exception:
      # Continue execution to the next parsing strategy.
      pass
    # Prepare the string for safe evaluation by replacing nan tokens.
    cleanedString = re.sub(r"\bNaN\b|\bnan\b", "float(\"nan\")", trimmedString, flags=re.IGNORECASE)
    # Initiate the final parsing attempt using restricted evaluation.
    try:
      # Execute the cleaned string within a constrained namespace.
      evaluatedResult = eval(cleanedString, {"__builtins__": {}, "NumpyModule": np}, {})
      # Check if the evaluation produced a sequence of values.
      if (isinstance(evaluatedResult, (list, tuple))):
        # Map each sequence element to float while preserving nan states.
        return [float(itemValue) if (itemValue is not None) else float("nan") for itemValue in evaluatedResult]
      # Convert the single evaluation result to a float and return it.
      return [float(evaluatedResult)]
    # Catch any remaining errors from the evaluation process.
    except Exception:
      # Provide an empty list when all parsing strategies have failed.
      return []

  # Return an empty list when the input type is entirely unsupported.
  return []


def CodeCarbonCodeEstimation(func):
  r'''
  Estimate the carbon emissions of a function call using CodeCarbon.
  Reference: https://docs.codecarbon.io/latest/

  You can also use:
    - codecarbon detect (to detect the hardware).
    - codecarbon monitor --no-api -- python XXXX.py (To track any script without changing the code).

  Parameters:
    func (callable): The function to estimate emissions for. This should be a callable that can be executed without arguments.

  Returns:
    float or None: The estimated carbon emissions in kg CO2 equivalent, or None if estimation fails.
  '''

  try:
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker()
    tracker.start()
    func()
    emissions = tracker.stop()
    print(f"Estimated carbon emissions: {emissions} kg CO2eq")
    return emissions
  except ImportError:
    print("CodeCarbon is not installed. Please install it to use this feature.")
    return None
  except Exception as e:
    print(f"An error occurred during carbon estimation: {e}")
    return None


def fprint(msg, *args, **kwargs):
  r'''
  Print a message with flush=True to ensure it appears immediately in the console.

  Parameters:
    msg (str): The message to print.
    *args: Additional positional arguments to pass to the print function.
    **kwargs: Additional keyword arguments to pass to the print function.
  '''

  print(msg, flush=True, *args, **kwargs)
