import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from HMB.VideosHelper import VideosHelper


class TestVideosHelper(unittest.TestCase):
  """
  Unit tests for VideosHelper using mocks to avoid heavy I/O.
  """

  def setUp(self):
    self.helper = VideosHelper()

  @patch("HMB.VideosHelper.cv2.VideoCapture")
  def test_read_video_mocked(self, mockVideoCapture):
    mockCap = MagicMock()
    mockVideoCapture.return_value = mockCap
    video = self.helper.ReadVideo("dummy.mp4")
    self.assertEqual(video, mockCap)
    mockVideoCapture.assert_called_once_with("dummy.mp4")

  @patch("HMB.VideosHelper.cv2.VideoCapture")
  def test_get_video_frames_count(self, mockVideoCapture):
    mockCap = MagicMock()
    mockCap.get.return_value = 120.0
    mockVideoCapture.return_value = mockCap
    video = self.helper.ReadVideo("dummy.mp4")
    count = self.helper.GetVideoFramesCount(video)
    self.assertEqual(count, 120)

  @patch("HMB.VideosHelper.cv2.VideoCapture")
  def test_get_video_fps(self, mockVideoCapture):
    mockCap = MagicMock()
    mockCap.get.return_value = 30.0
    mockVideoCapture.return_value = mockCap
    video = self.helper.ReadVideo("dummy.mp4")
    fps = self.helper.GetVideoFPS(video)
    self.assertEqual(fps, 30.0)

  @patch("HMB.VideosHelper.cv2.VideoCapture")
  def test_get_video_frame_size(self, mockVideoCapture):
    mockCap = MagicMock()
    mockCap.get.side_effect = lambda prop: {3: 1920, 4: 1080}.get(prop, 0)
    mockVideoCapture.return_value = mockCap
    video = self.helper.ReadVideo("dummy.mp4")
    size = self.helper.GetVideoFrameSize(video)
    self.assertEqual(size, (1920, 1080))


if (__name__ == "__main__"):
  unittest.main()
