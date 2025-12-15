import unittest
import os
from unittest.mock import patch, MagicMock
import numpy as np

from HMB.WSIHelper import ReadWSIViaOpenSlide, TileExtractionAlignmentHandler


class TestWSIHelper(unittest.TestCase):
  """
  Unit tests for WSIHelper with heavy dependencies mocked.
  Covers basic openslide path checks and tile extraction alignment pipeline with mocks.
  """

  def test_read_wsi_invalid_path_raises(self):
    with self.assertRaises(AssertionError):
      ReadWSIViaOpenSlide("does_not_exist.svs")

  @patch("HMB.WSIHelper.openslide.OpenSlide")
  @patch("HMB.WSIHelper.ExtractLargestContour")
  @patch("HMB.WSIHelper.MatchTwoImagesViaSIFT")
  @patch("HMB.WSIHelper.ExtractPatch")
  @patch("HMB.WSIHelper.plt")
  def test_tile_extraction_alignment_minimal(self, mockPlt, mockExtractPatch, mockMatchSIFT, mockExtractLargest,
                                             mockOpenSlide):
    # Create temporary slide files
    import tempfile
    tmpDir = tempfile.mkdtemp()
    hePath = os.path.join(tmpDir, "he.svs")
    mtPath = os.path.join(tmpDir, "mt.svs")
    open(hePath, "wb").close()
    open(mtPath, "wb").close()

    # Mock OpenSlide objects
    heSlide = MagicMock()
    mtSlide = MagicMock()
    heSlide.get_thumbnail.return_value = MagicMock()  # PIL-like
    mtSlide.get_thumbnail.return_value = MagicMock()
    heSlide.level_downsamples = [1, 2, 4]
    heSlide.level_count = 3
    mockOpenSlide.side_effect = [heSlide, mtSlide]

    # Mock ExtractLargestContour to return (img, contour, mask, draw)
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    mask = np.zeros((20, 20), dtype=np.uint8)
    draw = np.zeros((20, 20, 3), dtype=np.uint8)
    mockExtractLargest.return_value = (np.zeros((20, 20, 3), dtype=np.uint8), contour, mask, draw)

    # Mock MatchTwoImagesViaSIFT to return transformed thumbs, matched image, homography, shape
    homography = np.eye(3)
    shape = (20, 20)
    matched = np.zeros((20, 20, 3), dtype=np.uint8)
    mockMatchSIFT.return_value = (
      np.zeros((20, 20, 3), dtype=np.uint8),
      np.zeros((20, 20, 3), dtype=np.uint8),
      matched,
      homography,
      shape,
    )

    # Mock ExtractPatch to return tuple including flag True
    heRegion = np.zeros((256, 256, 3), dtype=np.uint8)
    mtRegion = np.zeros((256, 256, 3), dtype=np.uint8)
    heBW = np.zeros((256, 256), dtype=np.uint8)
    mtBW = np.zeros((256, 256), dtype=np.uint8)
    diff = np.zeros((256, 256), dtype=np.uint8)
    weighted = np.zeros((256, 256), dtype=np.uint8)
    mockExtractPatch.return_value = (heRegion, mtRegion, heBW, mtBW, diff, weighted, True)

    try:
      TileExtractionAlignmentHandler(hePath, mtPath, storageDir=tmpDir, patchesPerSlide=4, doPlotting=False,
                                     verbose=False)
      # Ensure that output directories were created
      self.assertTrue(os.path.isdir(os.path.join(tmpDir, "HE")))
      self.assertTrue(os.path.isdir(os.path.join(tmpDir, "MT")))
    finally:
      import shutil
      if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)


if __name__ == "__main__":
  unittest.main()
