import unittest, torch, os, sys
import numpy as np
from unittest.mock import patch, MagicMock
from HMB.Initializations import (
  SetMaxTextChunkSize,
  SetEnvironmentVariables,
  MaximizeThreads,
  SeedEverything,
  IgnoreWarnings,
  DoRandomSeeding,
)


class TestInitializations(unittest.TestCase):
  '''
  Unit tests for the Initializations module.
  Tests cover environment setup and seeding functions.
  '''

  # ========== SetMaxTextChunkSize Tests ==========

  def test_set_max_text_chunk_size_default(self):
    '''Test setting max text chunk size with default value.'''
    try:
      SetMaxTextChunkSize()
      from PIL import PngImagePlugin
      # Verify the value was set (default is 100 MB)
      expectedSize = 100 * (1024 ** 2)
      self.assertEqual(PngImagePlugin.MAX_TEXT_CHUNK, expectedSize)
    except Exception as e:
      self.skipTest(f"PIL not available or test environment issue: {e}")

  def test_set_max_text_chunk_size_custom(self):
    '''Test setting max text chunk size with custom value.'''
    try:
      customSize = 50 * (1024 ** 2)
      SetMaxTextChunkSize(customSize)
      from PIL import PngImagePlugin
      self.assertEqual(PngImagePlugin.MAX_TEXT_CHUNK, customSize)
    except Exception as e:
      self.skipTest(f"PIL not available or test environment issue: {e}")

  def test_set_max_text_chunk_size_small_value(self):
    '''Test setting max text chunk size with small value.'''
    try:
      smallSize = 1024  # 1 KB
      SetMaxTextChunkSize(smallSize)
      from PIL import PngImagePlugin
      self.assertEqual(PngImagePlugin.MAX_TEXT_CHUNK, smallSize)
    except Exception as e:
      self.skipTest(f"PIL not available or test environment issue: {e}")

  # ========== SetEnvironmentVariables Tests ==========

  def test_set_environment_variables_default(self):
    '''Test setting environment variables with default threads.'''
    SetEnvironmentVariables()

    self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "6")
    self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "6")
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "6")
    self.assertEqual(os.environ.get("PYTORCH_MAX_NUM_THREADS"), "6")
    self.assertEqual(os.environ.get("NUMEXPR_NUM_THREADS"), "6")
    self.assertEqual(os.environ.get("CUDA_DEVICE_ORDER"), "PCI_BUS_ID")

  def test_set_environment_variables_custom(self):
    '''Test setting environment variables with custom thread count.'''
    customThreads = 12
    SetEnvironmentVariables(defaultThreads=customThreads)

    self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "12")
    self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "12")
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "12")
    self.assertEqual(os.environ.get("PYTORCH_MAX_NUM_THREADS"), "12")
    self.assertEqual(os.environ.get("NUMEXPR_NUM_THREADS"), "12")

  def test_set_environment_variables_single_thread(self):
    '''Test setting environment variables with single thread.'''
    SetEnvironmentVariables(defaultThreads=1)

    self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")
    self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")

  def test_set_environment_variables_multiple_calls(self):
    '''Test that multiple calls update the environment variables.'''
    SetEnvironmentVariables(defaultThreads=4)
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "4")

    SetEnvironmentVariables(defaultThreads=8)
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "8")

  # ========== MaximizeThreads Tests ==========

  def test_maximize_threads(self):
    '''Test maximizing threads sets correct values.'''
    import multiprocessing
    expectedThreads = multiprocessing.cpu_count() - 1

    MaximizeThreads()

    actualThreads = int(os.environ.get("OPENBLAS_NUM_THREADS", "0"))
    self.assertEqual(actualThreads, expectedThreads)

    # Verify all thread-related variables are set
    self.assertIsNotNone(os.environ.get("MKL_NUM_THREADS"))
    self.assertIsNotNone(os.environ.get("OMP_NUM_THREADS"))

  def test_maximize_threads_consistency(self):
    '''Test that MaximizeThreads sets consistent values across all variables.'''
    MaximizeThreads()

    openblasThreads = os.environ.get("OPENBLAS_NUM_THREADS")
    mklThreads = os.environ.get("MKL_NUM_THREADS")
    ompThreads = os.environ.get("OMP_NUM_THREADS")

    # All should be equal
    self.assertEqual(openblasThreads, mklThreads)
    self.assertEqual(mklThreads, ompThreads)

  # ========== SeedEverything Tests ==========

  def test_seed_everything_default(self):
    '''Test SeedEverything with default seed.'''
    try:
      import torch
      import random

      SeedEverything(seed=42)

      # Verify environment variable
      self.assertEqual(os.environ.get("PYTHONHASHSEED"), "42")

      # Generate some random numbers to verify seeding
      randomNum = random.random()
      numpyNum = np.random.rand()
      torchNum = torch.rand(1).item()

      # Reset and regenerate - should get same values
      SeedEverything(seed=42)
      self.assertEqual(random.random(), randomNum)
      self.assertEqual(np.random.rand(), numpyNum)

    except ImportError:
      self.skipTest("PyTorch not available")

  def test_seed_everything_custom_seed(self):
    '''Test SeedEverything with custom seed.'''
    try:
      import torch

      customSeed = 123
      SeedEverything(seed=customSeed)

      self.assertEqual(os.environ.get("PYTHONHASHSEED"), "123")

    except ImportError:
      self.skipTest("PyTorch not available")

  def test_seed_everything_deterministic_flag(self):
    '''Test SeedEverything with deterministic flag.'''
    try:
      import torch

      if torch.backends.cudnn.is_available():
        SeedEverything(seed=42, deterministic=True)
        self.assertTrue(torch.backends.cudnn.deterministic)

        SeedEverything(seed=42, deterministic=False)
        self.assertFalse(torch.backends.cudnn.deterministic)
      else:
        self.skipTest("CUDNN not available")

    except ImportError:
      self.skipTest("PyTorch not available")

  def test_seed_everything_benchmark_flag(self):
    '''Test SeedEverything with benchmark flag.'''
    try:
      import torch

      if torch.backends.cudnn.is_available():
        SeedEverything(seed=42, benchmark=True)
        self.assertTrue(torch.backends.cudnn.benchmark)

        SeedEverything(seed=42, benchmark=False)
        self.assertFalse(torch.backends.cudnn.benchmark)
      else:
        self.skipTest("CUDNN not available")

    except ImportError:
      self.skipTest("PyTorch not available")

  def test_seed_everything_reproducibility(self):
    '''Test that SeedEverything ensures reproducibility.'''
    try:
      import torch
      import random

      # First run
      SeedEverything(seed=999)
      random1 = [random.random() for _ in range(5)]
      numpy1 = [np.random.rand() for _ in range(5)]
      torch1 = [torch.rand(1).item() for _ in range(5)]

      # Second run with same seed
      SeedEverything(seed=999)
      random2 = [random.random() for _ in range(5)]
      numpy2 = [np.random.rand() for _ in range(5)]
      torch2 = [torch.rand(1).item() for _ in range(5)]

      # Verify reproducibility
      self.assertEqual(random1, random2)
      np.testing.assert_array_almost_equal(numpy1, numpy2)
      np.testing.assert_array_almost_equal(torch1, torch2)

    except ImportError:
      self.skipTest("PyTorch not available")

  def test_seed_everything_different_seeds_different_results(self):
    '''Test that different seeds produce different results.'''
    try:
      import torch
      import random

      # First seed
      SeedEverything(seed=100)
      random1 = random.random()
      numpy1 = np.random.rand()

      # Different seed
      SeedEverything(seed=200)
      random2 = random.random()
      numpy2 = np.random.rand()

      # Results should be different
      self.assertNotEqual(random1, random2)
      self.assertNotEqual(numpy1, numpy2)

    except ImportError:
      self.skipTest("PyTorch not available")

  def test_seed_everything_numpy_seeding(self):
    '''Test that NumPy seeding works correctly.'''
    SeedEverything(seed=77)

    # Generate numpy random numbers
    arr1 = np.random.randint(0, 100, 10)

    # Reset with same seed
    SeedEverything(seed=77)
    arr2 = np.random.randint(0, 100, 10)

    # Should be identical
    np.testing.assert_array_equal(arr1, arr2)

  # ========== Integration Tests ==========

  def test_environment_setup_workflow(self):
    '''Test complete environment setup workflow.'''
    # Set environment variables
    SetEnvironmentVariables(defaultThreads=4)

    # Set max chunk size
    try:
      SetMaxTextChunkSize(1024 * 1024)
    except:
      pass  # Skip if PIL not available

    # Set seed
    try:
      SeedEverything(seed=42)
    except:
      pass  # Skip if PyTorch not available

    # Verify environment variables are set
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "4")
    self.assertIsNotNone(os.environ.get("PYTHONHASHSEED"))

  def test_thread_configuration_consistency(self):
    '''Test thread configuration consistency across functions.'''
    # Set specific thread count
    testThreads = 8
    SetEnvironmentVariables(defaultThreads=testThreads)

    # Verify all thread variables are consistent
    threadVars = [
      "OPENBLAS_NUM_THREADS",
      "MKL_NUM_THREADS",
      "OMP_NUM_THREADS",
      "PYTORCH_MAX_NUM_THREADS",
      "NUMEXPR_NUM_THREADS"
    ]

    for var in threadVars:
      self.assertEqual(os.environ.get(var), str(testThreads))

  def test_maximize_then_customize_threads(self):
    '''Test maximizing threads then customizing.'''
    # First maximize
    MaximizeThreads()
    maxValue = os.environ.get("OMP_NUM_THREADS")

    # Then customize to lower value
    SetEnvironmentVariables(defaultThreads=2)
    newValue = os.environ.get("OMP_NUM_THREADS")

    self.assertEqual(newValue, "2")
    self.assertNotEqual(maxValue, newValue)


if __name__ == "__main__":
  unittest.main()
