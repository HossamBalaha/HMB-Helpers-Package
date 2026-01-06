import unittest
import numpy as np
import cv2
from HMB.ImagesNormalization import RGB2LAB, LAB2RGB, ReinhardColorNormalization, LabSplitMeanStd, CreateAverageHistogram, CreateLUTFromHistogram, ApplyLUT, LABSplit2RGB, RGB2OD


class TestImagesNormalization(unittest.TestCase):
  '''
  Unit tests for ImagesNormalization utilities.
  Focus on RGB/LAB conversions and ReinhardColorNormalization.
  '''

  def setUp(self):
    # Create small synthetic RGB images
    self.rgbSolid = np.ones((16, 16, 3), dtype=np.uint8) * 128  # mid-gray
    # Gradient image from black to white
    x = np.linspace(0, 255, 16, dtype=np.uint8)
    self.rgbGradient = np.stack([np.tile(x, (16, 1)), np.tile(x, (16, 1)), np.tile(x, (16, 1))], axis=2)

  # ========== RGB<->LAB ==========

  def test_rgb2lab_outputs_shapes(self):
    L, A, B = RGB2LAB(self.rgbSolid, isNorm=True)
    self.assertEqual(L.shape, (16, 16))
    self.assertEqual(A.shape, (16, 16))
    self.assertEqual(B.shape, (16, 16))

  def test_rgb2lab_normalization_ranges(self):
    L, A, B = RGB2LAB(self.rgbSolid, isNorm=True)
    # L in [0,100], A,B approx around 0
    self.assertTrue(np.all(L >= 0))
    self.assertTrue(np.all(L <= 100))
    self.assertTrue(np.abs(np.mean(A)) < 10)
    self.assertTrue(np.abs(np.mean(B)) < 10)

  def test_lab2rgb_roundtrip_mid_gray(self):
    L, A, B = RGB2LAB(self.rgbSolid, isNorm=True)
    rgb = LAB2RGB(L, A, B, isNorm=True)
    self.assertEqual(rgb.shape, self.rgbSolid.shape)
    # Roundtrip should be close
    diff = np.abs(rgb.astype(np.int32) - self.rgbSolid.astype(np.int32))
    self.assertTrue(np.mean(diff) < 5)

  def test_lab2rgb_roundtrip_gradient(self):
    L, A, B = RGB2LAB(self.rgbGradient, isNorm=True)
    rgb = LAB2RGB(L, A, B, isNorm=True)
    diff = np.abs(rgb.astype(np.int32) - self.rgbGradient.astype(np.int32))
    self.assertTrue(np.mean(diff) < 10)

  def test_rgb2lab_invalid_inputs(self):
    with self.assertRaises(Exception):
      _ = RGB2LAB(np.ones((16, 16), dtype=np.uint8), isNorm=True)
    with self.assertRaises(Exception):
      _ = RGB2LAB(np.ones((16, 16, 4), dtype=np.uint8), isNorm=True)

  def test_lab2rgb_invalid_inputs(self):
    L, A, B = RGB2LAB(self.rgbSolid, isNorm=True)
    with self.assertRaises(Exception):
      _ = LAB2RGB(L, A, np.ones_like(B)[..., None], isNorm=True)

  def test_rgb2lab_nan_handling(self):
    img = self.rgbSolid.astype(np.float32)
    img[0, 0, 0] = np.nan
    L, A, B = RGB2LAB(np.nan_to_num(img), isNorm=True)
    self.assertTrue(np.isfinite(L).all())

  def test_rgb2lab_uint16_and_float(self):
    img16 = (self.rgbSolid.astype(np.uint16) * 256)  # simulate 16-bit
    L, A, B = RGB2LAB(img16, isNorm=True)
    self.assertEqual(L.shape, (16, 16))
    imgf = self.rgbSolid.astype(np.float32)
    Lf, Af, Bf = RGB2LAB(imgf, isNorm=True)
    self.assertEqual(Lf.shape, (16, 16))

  def test_lab2rgb_non_normalized(self):
    L, A, B = RGB2LAB(self.rgbSolid, isNorm=False)
    rgb = LAB2RGB(L, A, B, isNorm=False)
    self.assertEqual(rgb.shape, self.rgbSolid.shape)
    self.assertTrue(np.all(rgb >= 0))

  def test_rgb2lab_invalid_channel_count(self):
    with self.assertRaises(Exception):
      _ = RGB2LAB(np.ones((16, 16, 2), dtype=np.uint8), isNorm=True)

  # ========== ReinhardColorNormalization ==========

  def test_reinhard_fit_sets_flags(self):
    rcn = ReinhardColorNormalization()
    rcn.Fit(self.rgbSolid)
    self.assertTrue(rcn.isFit)
    self.assertIsNotNone(rcn.targetMeans)
    self.assertIsNotNone(rcn.targetStds)

  def test_reinhard_normalize_requires_fit(self):
    rcn = ReinhardColorNormalization()
    with self.assertRaises(RuntimeError):
      rcn.Normalize(self.rgbSolid)

  def test_reinhard_normalize_changes_image(self):
    rcn = ReinhardColorNormalization()
    rcn.Fit(self.rgbSolid)
    out = rcn.Normalize(self.rgbGradient)
    self.assertEqual(out.shape, self.rgbGradient.shape)
    # Should be a valid RGB image
    self.assertTrue(np.all(out >= 0))
    self.assertTrue(np.all(out <= 255))

  def test_reinhard_normalize_invalid_shape(self):
    rcn = ReinhardColorNormalization()
    rcn.Fit(self.rgbSolid)
    with self.assertRaises(ValueError):
      rcn.Normalize(np.ones((16, 16), dtype=np.uint8))

  def test_reinhard_normalize_on_constant_and_noise(self):
    rcn = ReinhardColorNormalization()
    rcn.Fit(self.rgbSolid)
    const = np.ones_like(self.rgbSolid) * 64
    out_const = rcn.Normalize(const)
    self.assertEqual(out_const.shape, const.shape)
    noisy = np.clip(self.rgbGradient + (np.random.randn(*self.rgbGradient.shape) * 5).astype(np.int16), 0, 255).astype(
      np.uint8)
    out_noisy = rcn.Normalize(noisy)
    self.assertEqual(out_noisy.shape, noisy.shape)

  def test_labsplit_mean_std_basic(self):
    means, stds = LabSplitMeanStd(self.rgbSolid)
    # means and stds should be lists of length 3 with numeric entries
    self.assertEqual(len(means), 3)
    self.assertEqual(len(stds), 3)
    for m in means:
      # mean from cv2.meanStdDev returns array-like; ensure numeric
      self.assertTrue(np.isfinite(np.array(m)).all())
    for s in stds:
      self.assertTrue(np.isfinite(np.array(s)).all())

  def test_create_avg_hist_lut_and_apply(self):
    # Create a small set of reference images (use 3-channel)
    refs = [self.rgbSolid, self.rgbGradient]
    avgHist = CreateAverageHistogram(refs, noChannels=3)
    self.assertEqual(len(avgHist), 3)
    lut = CreateLUTFromHistogram(avgHist)
    self.assertEqual(len(lut), 3)
    for l in lut:
      self.assertEqual(len(l), 256)
    # Apply LUT to a small test image
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = np.arange(8).reshape(8, 1) * 32
    img_out = ApplyLUT(img.copy(), lut, noChannels=3)
    self.assertEqual(img_out.shape, img.shape)
    self.assertEqual(img_out.dtype, np.uint8)
    self.assertTrue(np.all(img_out >= 0) and np.all(img_out <= 255))

  def test_labsplit2rgb_consistency(self):
    L, A, B = RGB2LAB(self.rgbGradient, isNorm=True)
    rgb_back = LABSplit2RGB(L.copy(), A.copy(), B.copy())
    self.assertEqual(rgb_back.shape, self.rgbGradient.shape)
    # Values should be uint8 and within [0,255]
    self.assertEqual(rgb_back.dtype, np.uint8)
    self.assertTrue(np.all(rgb_back >= 0) and np.all(rgb_back <= 255))

  def test_rgb2od_range(self):
    od = RGB2OD(self.rgbSolid)
    self.assertEqual(od.shape, self.rgbSolid.shape)
    self.assertTrue(np.all(np.isfinite(od)))
    self.assertTrue(np.all(od >= 0))


if __name__ == "__main__":
  unittest.main()
