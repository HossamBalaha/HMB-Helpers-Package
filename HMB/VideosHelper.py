import cv2
import numpy as np


class VideosHelper(object):
  r'''
  VideosHelper: Helpers for common video operations using OpenCV.

  Methods provided:
    - ReadVideo(videoPath)
    - WriteVideo(videoPath, frames, fps, frameSize, fourccType="mp4v")
    - ShowVideo(video, frameIndex)
    - ShowVideoFrames(video, frameIndexRange)
    - GetVideoFrames(video)
    - GetVideoFramesCount(video)
    - GetVideoFPS(video)
    - GetVideoFrameSize(video)
    - GetVideoFrameSizeString(video)
    - GetVideoDuration(video)
    - GetVideoFrame(video, frameIndex)
    - GetVideoFrameTime(video, frameIndex)
    - GetVideoFrameIndex(video, frameTime)
    - GetVideoFrameIndexRange(video, frameTimeRange)
    - GetVideoFrameTimeRange(video, frameIndexRange)
    - GetVideoFrameRange(video, frameIndexRange)
  '''

  def ReadVideo(self, videoPath):
    r'''
    Open a video file and return a cv2.VideoCapture object.

    Parameters:
      videoPath (str): Path to the video file.

    Returns:
      cv2.VideoCapture: Opened video capture object.
    '''

    video = cv2.VideoCapture(videoPath)
    return video

  def WriteVideo(self, videoPath, frames, fps, frameSize, fourccType="mp4v"):
    r'''
    Write a sequence of frames to a video file.

    Parameters:
      videoPath (str): Output path for the video file.
      frames (iterable): Sequence of frames (NumPy arrays) to write.
      fps (float): Frames per second.
      frameSize (tuple): (width, height) of the frames.
      fourccType (str): FourCC codec string (default "mp4v").
    '''

    # fourccType can be: DIVX, XVID, MJPG, X264, WMV1, WMV2, I420.
    fourcc = cv2.VideoWriter_fourcc(*fourccType)
    video = cv2.VideoWriter(videoPath, fourcc, fps, frameSize)
    for frame in frames:
      video.write(frame)
    video.release()

  def ShowVideo(self, video, frameIndex):
    r'''
    Display a single frame from a video using OpenCV GUI (blocking until a key press).

    Parameters:
      video (cv2.VideoCapture): Video capture object.
      frameIndex (int): Index of the frame to display.
    '''

    frame = self.GetVideoFrame(video, frameIndex)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  def ShowVideoFrames(self, video, frameIndexRange):
    r'''
    Display multiple frames by index range.

    Parameters:
      video (cv2.VideoCapture): Video capture object.
      frameIndexRange (iterable): Iterable of frame indices to display.
    '''

    for frameIndex in frameIndexRange:
      self.ShowVideo(video, frameIndex)

  def GetVideoFrames(self, video):
    r'''
    Read all frames from an opened `cv2.VideoCapture` into a list.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.

    Returns:
      list: List of frames as NumPy arrays.
    '''

    frames = []
    while (True):
      success, frame = video.read()
      if (not success):
        break
      frames.append(frame)
    return frames

  def GetVideoFramesCount(self, video):
    r'''
    Get the total frame count of a video.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.

    Returns:
      int: Total number of frames (may be 0 for some live streams).
    '''

    framesCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return framesCount

  def GetVideoFPS(self, video):
    r'''
    Return the frames-per-second (FPS) rate for the video.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.

    Returns:
      float: FPS value.
    '''

    fps = video.get(cv2.CAP_PROP_FPS)
    return fps

  def GetVideoFrameSize(self, video):
    r'''
    Get the video frame size as (width, height).

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.

    Returns:
      tuple: (width, height) in pixels.
    '''

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (width, height)

  def GetVideoFrameSizeString(self, video):
    r'''
    Return a human-readable frame size string like "640x480".

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.

    Returns:
      str: Frame size string.
    '''

    (width, height) = self.GetVideoFrameSize(video)
    frameSizeString = str(width) + "x" + str(height)
    return frameSizeString

  def GetVideoDuration(self, video):
    r'''
    Estimate video duration in seconds using frame count and fps.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.

    Returns:
      float: Duration in seconds.
    '''

    framesCount = self.GetVideoFramesCount(video)
    fps = self.GetVideoFPS(video)
    duration = framesCount / fps
    return duration

  def GetVideoFrame(self, video, frameIndex):
    r'''
    Retrieve a specific frame by index from an opened video.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.
      frameIndex (int): Frame index.

    Returns:
      numpy.ndarray: Frame image array (BGR color by OpenCV convention).
    '''

    video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    success, frame = video.read()
    return frame

  def GetVideoFrameTime(self, video, frameIndex):
    r'''
    Convert a frame index to its timestamp (in seconds) using FPS.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.
      frameIndex (int): Frame index.

    Returns:
      float: Timestamp in seconds.
    '''

    fps = self.GetVideoFPS(video)
    frameTime = frameIndex / fps
    return frameTime

  def GetVideoFrameIndex(self, video, frameTime):
    r'''
    Convert a timestamp (seconds) to a fractional frame index.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.
      frameTime (float): Timestamp in seconds.

    Returns:
      float: Frame index (may be fractional).
    '''

    fps = self.GetVideoFPS(video)
    frameIndex = frameTime * fps
    return frameIndex

  def GetVideoFrameIndexRange(self, video, frameTimeRange):
    r'''
    Convert a list/iterable of timestamps to frame indices.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.
      frameTimeRange (iterable): Iterable of timestamps in seconds.

    Returns:
      list: List of frame indices (may be fractional).
    '''

    fps = self.GetVideoFPS(video)
    frameIndexRange = [frameTime * fps for frameTime in frameTimeRange]
    return frameIndexRange

  def GetVideoFrameTimeRange(self, video, frameIndexRange):
    r'''
    Convert a list/iterable of frame indices to timestamps.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.
      frameIndexRange (iterable): Iterable of frame indices.

    Returns:
      list: List of timestamps in seconds.
    '''

    fps = self.GetVideoFPS(video)
    frameTimeRange = [frameIndex / fps for frameIndex in frameIndexRange]
    return frameTimeRange

  def GetVideoFrameRange(self, video, frameIndexRange):
    r'''
    Retrieve multiple frames by their indices and return them as a list.

    Parameters:
      video (cv2.VideoCapture): Opened video capture object.
      frameIndexRange (iterable): Iterable of frame indices.

    Returns:
      list: List of frames as NumPy arrays.
    '''

    frames = []
    for frameIndex in frameIndexRange:
      frame = self.GetVideoFrame(video, frameIndex)
      frames.append(frame)
    return frames


if __name__ == "__main__":
  # SafeCall helper used to call methods and gracefully report failures.
  def SafeCall(name, fn, *args, **kwargs):
    try:
      res = fn(*args, **kwargs)
      print(f"{name} ->", res)
      print("-" * 40)
      return res
    except Exception as e:
      print(f"{name} raised {type(e).__name__}:", e)
      print("-" * 40)
      return None


  # Lightweight MockVideo to avoid depending on actual cv2.VideoCapture for demos.
  class MockVideo:
    def __init__(self, frames, fps=30.0):
      self._frames = list(frames)
      self._fps = float(fps)
      self._pos = 0

    def read(self):
      if self._pos < len(self._frames):
        f = self._frames[self._pos]
        self._pos += 1
        return True, f
      return False, None

    def set(self, prop, value):
      # Only support setting frame position
      if prop == 1 or getattr(prop, "name", None) == "CAP_PROP_POS_FRAMES":
        try:
          self._pos = int(value)
        except Exception:
          self._pos = 0

    def get(self, prop):
      # Map common CAP_PROP_* codes that OpenCV uses; use numeric constants if available.
      # We will accept integer codes for CAP_PROP_FRAME_COUNT=7, CAP_PROP_FPS=5, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4
      try:
        code = int(prop)
      except Exception:
        code = None
      # Common OpenCV property codes
      if (code == 7):  # CAP_PROP_FRAME_COUNT
        return len(self._frames)
      if (code == 5):  # CAP_PROP_FPS
        return self._fps
      if (code == 3):  # CAP_PROP_FRAME_WIDTH
        # Assume frames are numpy arrays HxWxC.
        if (len(self._frames) > 0):
          return self._frames[0].shape[1]
        return 0
      if (code == 4):  # CAP_PROP_FRAME_HEIGHT
        if (len(self._frames) > 0):
          return self._frames[0].shape[0]
        return 0
      # Fallback.
      return 0


  vh = VideosHelper()

  # Create a few dummy frames (small RGB images)
  frames = [np.zeros((10, 16, 3), dtype=np.uint8) + i for i in range(3)]
  mock = MockVideo(frames, fps=10.0)

  SafeCall("GetVideoFramesCount", vh.GetVideoFramesCount, mock)
  SafeCall("GetVideoFPS", vh.GetVideoFPS, mock)
  SafeCall("GetVideoFrameSize", vh.GetVideoFrameSize, mock)
  SafeCall("GetVideoFrameSizeString", vh.GetVideoFrameSizeString, mock)
  SafeCall("GetVideoDuration", vh.GetVideoDuration, mock)
  SafeCall("GetVideoFrame", vh.GetVideoFrame, mock, 1)
  SafeCall("GetVideoFrameTime", vh.GetVideoFrameTime, mock, 2)
  SafeCall("GetVideoFrameIndex", vh.GetVideoFrameIndex, mock, 0.5)
  SafeCall("GetVideoFrameIndexRange", vh.GetVideoFrameIndexRange, mock, [0.0, 0.5, 1.0])
  SafeCall("GetVideoFrameTimeRange", vh.GetVideoFrameTimeRange, mock, [0, 1, 2])
  SafeCall("GetVideoFrameRange", vh.GetVideoFrameRange, mock, [0, 1])

  # GetVideoFrames: will exhaust the mock read cursor; create a fresh mock to test.
  mock2 = MockVideo(frames, fps=10.0)
  SafeCall("GetVideoFrames", vh.GetVideoFrames, mock2)

  # Attempt to call WriteVideo (may require cv2.VideoWriter and filesystem access) - guarded.
  try:
    SafeCall("WriteVideo (to /tmp/test.mp4)", vh.WriteVideo, "hmb_test_video.mp4", frames, 10.0, (16, 10), "mp4v")
  except Exception as e:
    print("WriteVideo skipped:", type(e).__name__, e)

  print("VideosHelper demo completed.")
