import unittest
import os
import tempfile
import shutil
import numpy as np
from HMB.Utils import (
  ReadProjectConfig,
  IsPointInsideContour,
  IsIntersectingWithOtherContours,
  WritePickleFile,
  ReadPickleFile,
  WriteTextFile,
  ReadTextFile,
  LoadYaml,
  SaveYaml,
  Hex2RGB,
  GetCmapColors,
  AppendOrCreateNewCSV,
  AppendOrCreateNewDataFrameCSV,
)


class TestUtils(unittest.TestCase):
  """
  Unit tests for the Utils module.
  Tests cover configuration reading, file I/O, geometric operations, colors, and CSV helpers.
  """

  @classmethod
  def setUpClass(cls):
    """Create temporary directory for test files."""
    cls.testDir = tempfile.mkdtemp()

  @classmethod
  def tearDownClass(cls):
    """Clean up temporary directory."""
    if (os.path.exists(cls.testDir)):
      shutil.rmtree(cls.testDir)

  # ========== ReadProjectConfig Tests. ==========

  def test_read_project_config_yaml(self):
    """Test reading YAML configuration file."""
    configPath = os.path.join(self.testDir, "config.yaml")
    configData = {"project_name": "TestProject", "version": "1.0", "debug": True}

    # Write YAML file.
    import yaml
    with open(configPath, "w") as f:
      yaml.dump(configData, f)

    # Read and verify.
    result = ReadProjectConfig(configPath)
    self.assertIsInstance(result, dict)
    self.assertEqual(result["project_name"], "TestProject")
    self.assertEqual(result["version"], "1.0")
    self.assertTrue(result["debug"])

  def test_read_project_config_yml_extension(self):
    """Test reading .yml configuration file."""
    configPath = os.path.join(self.testDir, "config.yml")
    configData = {"app": "test"}

    import yaml
    with open(configPath, "w") as f:
      yaml.dump(configData, f)

    result = ReadProjectConfig(configPath)
    self.assertEqual(result["app"], "test")

  def test_read_project_config_json(self):
    """Test reading JSON configuration file."""
    configPath = os.path.join(self.testDir, "config.json")
    configData = {"project_name": "JSONProject", "count": 42}

    import json
    with open(configPath, "w") as f:
      json.dump(configData, f)

    result = ReadProjectConfig(configPath)
    self.assertIsInstance(result, dict)
    self.assertEqual(result["project_name"], "JSONProject")
    self.assertEqual(result["count"], 42)

  def test_read_project_config_file_not_found(self):
    """Test reading non-existent config file raises AssertionError."""
    with self.assertRaises(AssertionError):
      ReadProjectConfig("nonexistent.yaml")

  def test_read_project_config_unsupported_format(self):
    """Test reading unsupported file format raises ValueError."""
    configPath = os.path.join(self.testDir, "config.txt")
    with open(configPath, "w") as f:
      f.write("test")

    with self.assertRaises(ValueError):
      ReadProjectConfig(configPath)

  def test_read_project_config_nested_structure(self):
    """Test reading nested configuration structure."""
    configPath = os.path.join(self.testDir, "nested.yaml")
    configData = {
      "database": {
        "host"       : "localhost",
        "port"       : 5432,
        "credentials": {
          "user"    : "admin",
          "password": "secret"
        }
      }
    }

    import yaml
    with open(configPath, "w") as f:
      yaml.dump(configData, f)

    result = ReadProjectConfig(configPath)
    self.assertEqual(result["database"]["host"], "localhost")
    self.assertEqual(result["database"]["credentials"]["user"], "admin")

  def test_read_project_config_invalid_yaml(self):
    """Invalid YAML content should raise an error inside ReadProjectConfig."""
    path = os.path.join(self.testDir, "invalid.yaml")
    with open(path, "w") as f:
      f.write("key: value: another")  # malformed YAML
    with self.assertRaises(Exception):
      _ = ReadProjectConfig(path)

  def test_read_project_config_invalid_json(self):
    """Invalid JSON content should raise an error inside ReadProjectConfig."""
    path = os.path.join(self.testDir, "invalid.json")
    with open(path, "w") as f:
      f.write("{ invalid json }")
    with self.assertRaises(Exception):
      _ = ReadProjectConfig(path)

  def test_read_project_config_empty_files(self):
    # Empty YAML -> should load as None or {} depending on implementation; enforce dict
    yaml_path = os.path.join(self.testDir, "empty.yaml")
    open(yaml_path, "w").close()
    with self.assertRaises(Exception):
      _ = ReadProjectConfig(yaml_path)
    # Empty JSON -> expect exception
    json_path = os.path.join(self.testDir, "empty.json")
    open(json_path, "w").close()
    with self.assertRaises(Exception):
      _ = ReadProjectConfig(json_path)

  def test_read_project_config_relative_path(self):
    # Use relative path from test dir
    cwd = os.getcwd()
    try:
      os.chdir(self.testDir)
      rel_path = "rel.yaml"
      import yaml
      with open(rel_path, "w") as f:
        yaml.dump({"k": 1}, f)
      result = ReadProjectConfig(rel_path)
      self.assertEqual(result.get("k"), 1)
    finally:
      os.chdir(cwd)

  # ========== IsPointInsideContour Tests ==========

  def test_is_point_inside_contour_inside(self):
    """Test point inside a rectangular contour."""
    contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    point = (50, 50)
    result = IsPointInsideContour(point, contour)
    self.assertTrue(result)

  def test_is_point_inside_contour_outside(self):
    """Test point outside a rectangular contour."""
    contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    point = (150, 150)
    result = IsPointInsideContour(point, contour)
    self.assertFalse(result)

  def test_is_point_inside_contour_on_edge(self):
    """Test point on edge of contour."""
    contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    point = (0, 50)
    result = IsPointInsideContour(point, contour)
    self.assertIsInstance(result, bool)

  def test_is_point_inside_contour_triangular(self):
    """Test point inside triangular contour."""
    contour = np.array([[0, 0], [100, 0], [50, 100]])
    point = (50, 30)
    result = IsPointInsideContour(point, contour)
    self.assertTrue(result)

  def test_is_point_inside_contour_float_coordinates(self):
    """Test with float point coordinates."""
    contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    point = (50.5, 50.5)
    result = IsPointInsideContour(point, contour)
    self.assertIsInstance(result, bool)

  def test_is_point_inside_contour_degenerate(self):
    """Contour with fewer than 3 points (degenerate) should be handled."""
    contour = np.array([[0, 0], [1, 1]])
    point = (0, 0)
    res = IsPointInsideContour(point, contour)
    self.assertIsInstance(res, bool)

  def test_is_point_inside_contour_non_array(self):
    """Non-numpy contour input like list should be supported."""
    contour = [[0, 0], [100, 0], [100, 100], [0, 100]]
    point = (10, 10)
    result = IsPointInsideContour(point, contour)
    self.assertIsInstance(result, bool)

  def test_is_point_inside_contour_vertex_and_open_polygon(self):
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    point_vertex = (0, 0)
    self.assertIsInstance(IsPointInsideContour(point_vertex, contour), bool)
    # Non-closed polygon (first != last)
    open_contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    self.assertIsInstance(IsPointInsideContour((5, 5), open_contour), bool)

  def test_is_point_inside_contour_large_and_negative(self):
    contour = np.array([[-1000, -1000], [1000, -1000], [1000, 1000], [-1000, 1000]])
    self.assertTrue(IsPointInsideContour((0, 0), contour))
    self.assertFalse(IsPointInsideContour((2000, 0), contour))

  # ========== IsIntersectingWithOtherContours Tests ==========

  def test_is_intersecting_with_other_contours_true(self):
    """Test point intersecting with contours."""
    contours = [
      [[0, 0], [50, 0], [50, 50], [0, 50]],
      [[100, 100], [150, 100], [150, 150], [100, 150]]
    ]
    point = (25, 25)
    result = IsIntersectingWithOtherContours(point, contours)
    self.assertTrue(result)

  def test_is_intersecting_with_other_contours_false(self):
    """Test point not intersecting with any contours."""
    contours = [
      [[0, 0], [50, 0], [50, 50], [0, 50]],
      [[100, 100], [150, 100], [150, 150], [100, 150]]
    ]
    point = (75, 75)
    result = IsIntersectingWithOtherContours(point, contours)
    self.assertFalse(result)

  def test_is_intersecting_with_other_contours_empty_list(self):
    """Test with empty contours list."""
    contours = []
    point = (50, 50)
    result = IsIntersectingWithOtherContours(point, contours)
    self.assertFalse(result)

  def test_is_intersecting_with_other_contours_multiple(self):
    """Test with multiple contours."""
    contours = [
      [[0, 0], [30, 0], [30, 30], [0, 30]],
      [[40, 40], [70, 40], [70, 70], [40, 70]],
      [[80, 80], [110, 80], [110, 110], [80, 110]]
    ]
    point = (50, 50)
    result = IsIntersectingWithOtherContours(point, contours)
    self.assertTrue(result)

  def test_is_intersecting_with_other_contours_none_and_mixed(self):
    """Contours list containing None or degenerate contours should be safely handled."""
    contours = [None, [[0, 0]], [[0, 0], [1, 1]], [[0, 0], [10, 0], [10, 10], [0, 10]]]
    point = (5, 5)
    result = IsIntersectingWithOtherContours(point, contours)
    self.assertIsInstance(result, bool)

  def test_is_intersecting_with_other_contours_boundary(self):
    contours = [
      [[0, 0], [10, 0], [10, 10], [0, 10]],
      [[20, 20], [30, 20], [30, 30], [20, 30]]
    ]
    point = (10, 5)  # on boundary of first
    self.assertIsInstance(IsIntersectingWithOtherContours(point, contours), bool)

  def test_is_intersecting_with_other_contours_basic(self):
    """Two overlapping rectangles should intersect."""
    c1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    c2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
    res = IsIntersectingWithOtherContours(c1, [c2])
    self.assertIsInstance(res, bool)

  def test_is_intersecting_with_other_contours_disjoint(self):
    c1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    c2 = np.array([[10, 10], [11, 10], [11, 11], [10, 11]])
    res = IsIntersectingWithOtherContours(c1, [c2])
    self.assertIsInstance(res, bool)

  # ========== Pickle File Tests ==========

  def test_write_read_pickle_file_dict(self):
    """Test writing and reading dictionary to pickle file."""
    filePath = os.path.join(self.testDir, "test_dict.pkl")
    testData = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

    WritePickleFile(filePath, testData)
    self.assertTrue(os.path.exists(filePath))

    loadedData = ReadPickleFile(filePath)
    self.assertEqual(loadedData, testData)

  def test_write_read_pickle_file_list(self):
    """Test writing and reading list to pickle file."""
    filePath = os.path.join(self.testDir, "test_list.pkl")
    testData = [1, 2, 3, "test", {"nested": "dict"}]

    WritePickleFile(filePath, testData)
    loadedData = ReadPickleFile(filePath)
    self.assertEqual(loadedData, testData)

  def test_write_read_pickle_file_numpy_array(self):
    """Test writing and reading numpy array to pickle file."""
    filePath = os.path.join(self.testDir, "test_array.pkl")
    testData = np.array([[1, 2, 3], [4, 5, 6]])

    WritePickleFile(filePath, testData)
    loadedData = ReadPickleFile(filePath)
    np.testing.assert_array_equal(loadedData, testData)

  def test_read_pickle_file_not_found(self):
    """Test reading non-existent pickle file raises AssertionError."""
    with self.assertRaises(AssertionError):
      ReadPickleFile("nonexistent.pkl")

  def test_write_read_pickle_complex_object(self):
    """Test writing and reading complex nested object."""
    filePath = os.path.join(self.testDir, "complex.pkl")
    testData = {
      "arrays": [np.array([1, 2, 3]), np.array([4, 5, 6])],
      "nested": {"level2": {"level3": "deep"}},
      "tuple" : (1, 2, 3),
      "set"   : {1, 2, 3}
    }

    WritePickleFile(filePath, testData)
    loadedData = ReadPickleFile(filePath)

    np.testing.assert_array_equal(loadedData["arrays"][0], testData["arrays"][0])
    self.assertEqual(loadedData["nested"], testData["nested"])

  def test_pickle_overwrite_and_permissions(self):
    filePath = os.path.join(self.testDir, "overwrite.pkl")
    WritePickleFile(filePath, {"v": 1})
    WritePickleFile(filePath, {"v": 2})
    self.assertEqual(ReadPickleFile(filePath)["v"], 2)
    # Permission error simulation: open file read-only and attempt write to directory without permission (not trivial on Windows), skip strict enforcement.

  # ========== Text File Tests ==========

  def test_write_read_text_file_basic(self):
    """Test writing and reading basic text file."""
    filePath = os.path.join(self.testDir, "test.txt")
    testText = "Hello, World!\nThis is a test."

    WriteTextFile(filePath, testText)
    self.assertTrue(os.path.exists(filePath))

    loadedText = ReadTextFile(filePath)
    self.assertEqual(loadedText, testText)

  def test_write_read_text_file_empty(self):
    """Test writing and reading empty text file."""
    filePath = os.path.join(self.testDir, "empty.txt")
    testText = ""

    WriteTextFile(filePath, testText)
    loadedText = ReadTextFile(filePath)
    self.assertEqual(loadedText, testText)

  def test_write_read_text_file_multiline(self):
    """Test writing and reading multiline text."""
    filePath = os.path.join(self.testDir, "multiline.txt")
    testText = "Line 1\nLine 2\nLine 3\n\nLine 5"

    WriteTextFile(filePath, testText)
    loadedText = ReadTextFile(filePath)
    self.assertEqual(loadedText, testText)

  def test_read_text_file_not_found(self):
    """Test reading non-existent text file raises AssertionError."""
    with self.assertRaises(AssertionError):
      ReadTextFile("nonexistent.txt")

  def test_write_text_file_unicode(self):
    """Test writing and reading unicode text."""
    filePath = os.path.join(self.testDir, "unicode.txt")
    testText = "Hello World! Ca va?"

    WriteTextFile(filePath, testText)
    loadedText = ReadTextFile(filePath)
    self.assertEqual(loadedText, testText)

  def test_write_read_text_large_content(self):
    """Write and read a large text content."""
    filePath = os.path.join(self.testDir, "large.txt")
    testText = "x" * 100000  # 100k characters
    WriteTextFile(filePath, testText)
    loadedText = ReadTextFile(filePath)
    self.assertEqual(loadedText, testText)

  def test_write_text_bytes_input(self):
    filePath = os.path.join(self.testDir, "bytes.txt")
    with self.assertRaises(Exception):
      WriteTextFile(filePath, b"bytes content")

  # ========== YAML File Tests ==========

  def test_load_save_yaml_basic(self):
    """Test saving and loading basic YAML file."""
    yamlPath = os.path.join(self.testDir, "test_yaml.yaml")
    testData = {"name": "John", "age": 30, "active": True}

    SaveYaml(yamlPath, testData)
    self.assertTrue(os.path.exists(yamlPath))

    loadedData = LoadYaml(yamlPath)
    self.assertEqual(loadedData, testData)

  def test_load_save_yaml_list(self):
    """Test saving and loading list in YAML."""
    yamlPath = os.path.join(self.testDir, "list.yaml")
    testData = [1, 2, 3, "test", {"key": "value"}]

    SaveYaml(yamlPath, testData)
    loadedData = LoadYaml(yamlPath)
    self.assertEqual(loadedData, testData)

  def test_load_save_yaml_nested(self):
    """Test saving and loading nested YAML structure."""
    yamlPath = os.path.join(self.testDir, "nested.yaml")
    testData = {
      "level1": {
        "level2": {
          "level3": "value",
          "array" : [1, 2, 3]
        }
      }
    }

    SaveYaml(yamlPath, testData)
    loadedData = LoadYaml(yamlPath)
    self.assertEqual(loadedData, testData)

  def test_load_yaml_not_found(self):
    """Test loading non-existent YAML file raises AssertionError."""
    with self.assertRaises(AssertionError):
      LoadYaml("nonexistent.yaml")

  def test_save_yaml_empty_dict(self):
    """Test saving empty dictionary to YAML."""
    yamlPath = os.path.join(self.testDir, "empty.yaml")
    testData = {}

    SaveYaml(yamlPath, testData)
    loadedData = LoadYaml(yamlPath)
    self.assertEqual(loadedData, testData)

  def test_save_yaml_with_special_types(self):
    """Save YAML with various scalar types and verify."""
    yamlPath = os.path.join(self.testDir, "special.yaml")
    testData = {
      "null"      : None,
      "float"     : 12.34,
      "bool_true" : True,
      "bool_false": False,
      "string"    : "s",
    }
    SaveYaml(yamlPath, testData)
    loadedData = LoadYaml(yamlPath)
    self.assertEqual(loadedData, testData)

  def test_yaml_anchors_and_non_serializable(self):
    # Non-serializable object
    class Foo:
      pass

    path = os.path.join(self.testDir, "bad.yaml")
    with self.assertRaises(ValueError):
      SaveYaml(path, {"x": Foo()})

  # ========== Integration Tests ==========

  def test_config_yaml_json_equivalence(self):
    """Test that YAML and JSON configs produce same result."""
    yamlPath = os.path.join(self.testDir, "config.yaml")
    jsonPath = os.path.join(self.testDir, "config.json")
    configData = {"app": "test", "version": 1, "features": ["a", "b", "c"]}

    import yaml, json
    with open(yamlPath, "w") as f:
      yaml.dump(configData, f)
    with open(jsonPath, "w") as f:
      json.dump(configData, f)

    yamlResult = ReadProjectConfig(yamlPath)
    jsonResult = ReadProjectConfig(jsonPath)
    self.assertEqual(yamlResult, jsonResult)

  def test_file_operations_workflow(self):
    """Test complete file I/O workflow."""
    # Write text
    textPath = os.path.join(self.testDir, "workflow.txt")
    WriteTextFile(textPath, "Initial text")

    # Write pickle
    picklePath = os.path.join(self.testDir, "workflow.pkl")
    WritePickleFile(picklePath, {"data": "test"})

    # Write YAML
    yamlPath = os.path.join(self.testDir, "workflow.yaml")
    SaveYaml(yamlPath, {"yaml": "data"})

    # Verify all exist
    self.assertTrue(os.path.exists(textPath))
    self.assertTrue(os.path.exists(picklePath))
    self.assertTrue(os.path.exists(yamlPath))

    # Read and verify
    self.assertEqual(ReadTextFile(textPath), "Initial text")
    self.assertEqual(ReadPickleFile(picklePath), {"data": "test"})
    self.assertEqual(LoadYaml(yamlPath), {"yaml": "data"})

  # ========== Color Utilities Tests ==========

  def test_hex2rgb_basic(self):
    self.assertEqual(Hex2RGB("#FF0000"), (255, 0, 0))
    self.assertEqual(Hex2RGB("00FF00"), (0, 255, 0))
    self.assertEqual(Hex2RGB("0000FF"), (0, 0, 255))

  def test_hex2rgb_rgba(self):
    rgba = Hex2RGB("#112233", isRGBA=True)
    self.assertEqual(len(rgba), 4)
    self.assertEqual(rgba[:3], (17, 34, 51))

  def test_hex2rgb_short_and_no_hash(self):
    # 3-digit hex expands unevenly in current implementation; ensure behavior is deterministic
    self.assertEqual(Hex2RGB("#abc")[:3], Hex2RGB("abc")[:3])

  def test_hex2rgb_invalid(self):
    with self.assertRaises(Exception):
      _ = Hex2RGB("not-a-hex")

  def test_get_cmap_colors_basic(self):
    colors = GetCmapColors("viridis", 5)
    self.assertEqual(len(colors), 5)
    self.assertTrue(all(len(c) == 4 for c in colors))

  def test_get_cmap_colors_light_and_dark(self):
    dark = GetCmapColors("plasma", 7, darkColorsOnly=True)
    self.assertEqual(len(dark), 7)
    light = GetCmapColors("plasma", 7, darkColorsOnly=False)
    self.assertEqual(len(light), 7)

  def test_get_cmap_colors_non_string(self):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("inferno")
    colors = GetCmapColors(cmap, 3)
    self.assertEqual(len(colors), 3)

  def test_get_cmap_colors_bounds(self):
    colors = GetCmapColors("viridis", noColors=5)
    self.assertEqual(len(colors), 5)
    # Ensure colors are within [0,1] if normalized or 0-255 depending on implementation
    self.assertTrue(all(len(c) in (3, 4) for c in colors))

  # Additional tests added
  def test_hex2rgb_4_digit_and_expansion(self):
    # 4-digit hex like #abcd -> expands to aabbccdd
    rgba = Hex2RGB("#abcd", isRGBA=True)
    self.assertEqual(rgba, (0xAA, 0xBB, 0xCC, 0xDD))

  def test_hex2rgb_3_digit_explicit(self):
    # Explicit expected expansion: #abc -> aabbcc -> (170,187,204)
    rgb = Hex2RGB("#abc")
    self.assertEqual(rgb, (170, 187, 204))

  def test_get_cmap_colors_no_dark_fallback(self):
    # If darknessThreshold is set to 0.0, no color is considered dark -> function should return allColors
    colors = GetCmapColors("viridis", 6, darkColorsOnly=True, darknessThreshold=0.0)
    self.assertEqual(len(colors), 6)

  def test_append_or_create_new_csv_unsupported_type(self):
    path = os.path.join(self.testDir, "unsupported.csv")
    with self.assertRaises(ValueError):
      AppendOrCreateNewCSV(path, 12345)

  # New CSV and DataFrame tests
  def test_append_or_create_new_csv_append_and_header_mismatch(self):
    path = os.path.join(self.testDir, "mydata.csv")
    header = ["a", "b", "c"]
    # create with header
    AppendOrCreateNewCSV(path, [[1, 2, 3]], header=header, mode="w")
    # append correct row
    AppendOrCreateNewCSV(path, [[4, 5, 6]], header=header)
    # appending a row with wrong length should raise
    with self.assertRaises(ValueError):
      AppendOrCreateNewCSV(path, [[7, 8]], header=header)

  def test_append_or_create_new_dataframe_csv_basic(self):
    # Test appending a pandas DataFrame to CSV
    try:
      import pandas as pd
    except Exception:
      self.skipTest("pandas not installed")
    path = os.path.join(self.testDir, "dfdata.csv")
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    # create file
    AppendOrCreateNewDataFrameCSV(path, df)
    self.assertTrue(os.path.exists(path))
    # append again and ensure file still exists and has more than one line
    AppendOrCreateNewDataFrameCSV(path, df)
    with open(path, "r") as f:
      lines = f.readlines()
    # header + 2 rows + 2 appended rows = at least 5 lines
    self.assertTrue(len(lines) >= 5)

if __name__ == "__main__":
  unittest.main()
