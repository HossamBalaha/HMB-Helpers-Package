import unittest
import numpy as np
import warnings

from HMB.AudioHelper import AudiosHelper


class TestAudioHelper(unittest.TestCase):
  """
  Unit tests for AudiosHelper using synthetic signals to avoid file I/O.
  Focus on feature shape, basic numerical properties, and edge-case handling.
  """

  def setUp(self):
    # Synthetic signals
    self.sr = 22050
    t = np.linspace(0, 1.0, self.sr, endpoint=False)
    # Simple sine wave and a percussive-like impulse train
    self.y_sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    self.y_impulse = np.zeros_like(self.y_sine)
    self.y_impulse[::1000] = 1.0
    # Shorter signal for edge cases
    self.y_short = np.sin(2 * np.pi * 440 * t[:512]).astype(np.float32)

    # Helper instance
    self.helper = AudiosHelper()

    # Silence noisy warnings from underlying libraries in tests
    warnings.filterwarnings("ignore")

  # ===== Duration and STFT =====

  def test_get_segment_duration_basic(self):
    dur = self.helper.GetSegmentDuration(self.y_sine, sr=self.sr, roundTo=3)
    self.assertAlmostEqual(dur, 1.000, places=3)

  def test_get_segment_duration_short(self):
    dur = self.helper.GetSegmentDuration(self.y_short, sr=self.sr, roundTo=6)
    self.assertGreater(dur, 0.0)
    self.assertLess(dur, 1.0)

  def test_get_stft_shapes(self):
    stft = self.helper.GetSTFT(self.y_sine)
    self.assertTrue(np.iscomplexobj(stft))
    self.assertGreater(stft.shape[0], 0)
    self.assertGreater(stft.shape[1], 0)

  def test_get_absolute_stft(self):
    S = self.helper.GetAbsoluteSTFT(self.y_sine)
    self.assertFalse(np.iscomplexobj(S))
    self.assertTrue(np.all(S >= 0))

  # ===== Harmonic / Percussive separation =====

  def test_harmonic_effect_on_sine(self):
    y_h = self.helper.GetHarmonicEffect(self.y_sine)
    self.assertEqual(y_h.shape, self.y_sine.shape)

  def test_percussive_effect_on_impulse(self):
    y_p = self.helper.GetPercussiveEffect(self.y_impulse)
    self.assertEqual(y_p.shape, self.y_impulse.shape)

  # ===== MFCCs (Slaney and HTK) =====

  def test_slaney_mfcc_default(self):
    mfcc = self.helper.GetSlaneyMFCC(self.y_sine, sr=self.sr)
    self.assertEqual(mfcc.ndim, 2)
    self.assertGreater(mfcc.shape[0], 0)

  def test_slaney_mfcc_ncoeff(self):
    n = 13
    mfcc = self.helper.GetSlaneyMFCC(self.y_sine, sr=self.sr, nMFCC=n)
    self.assertEqual(mfcc.shape[0], n)

  def test_mean_slaney_mfcc(self):
    m = self.helper.GetMeanSlaneyMFCC(self.y_sine, sr=self.sr, nMFCC=20)
    self.assertEqual(m.ndim, 1)
    self.assertEqual(m.shape[0], 20)

  def test_htk_mfcc_default(self):
    mfcc = self.helper.GetHtkMFCC(self.y_sine, sr=self.sr)
    self.assertEqual(mfcc.ndim, 2)
    self.assertGreater(mfcc.shape[0], 0)

  def test_htk_mfcc_ncoeff(self):
    n = 13
    mfcc = self.helper.GetHtkMFCC(self.y_sine, sr=self.sr, nMFCC=n)
    self.assertEqual(mfcc.shape[0], n)

  def test_mean_htk_mfcc(self):
    m = self.helper.GetMeanHtkMFCC(self.y_sine, sr=self.sr, nMFCC=10)
    self.assertEqual(m.ndim, 1)
    self.assertEqual(m.shape[0], 10)

  # ===== Chroma features =====

  def test_mean_chroma(self):
    c = self.helper.GetMeanChroma(self.y_sine, sr=self.sr)
    self.assertEqual(c.ndim, 1)
    self.assertEqual(c.shape[0], 12)

  def test_mean_chroma_stft(self):
    c = self.helper.GetMeanChromaSTFT(self.y_sine, sr=self.sr)
    self.assertEqual(c.ndim, 1)
    self.assertEqual(c.shape[0], 12)

  def test_mean_chroma_cqt(self):
    c = self.helper.GetMeanChromaCqt(self.y_sine, sr=self.sr)
    self.assertEqual(c.ndim, 1)
    self.assertEqual(c.shape[0], 12)

  def test_mean_chroma_cens(self):
    c = self.helper.GetMeanChromaCens(self.y_sine, sr=self.sr)
    self.assertEqual(c.ndim, 1)
    self.assertEqual(c.shape[0], 12)

  # ===== Spectrogram-derived features =====

  def test_mean_mel_spectrogram(self):
    m = self.helper.GetMeanMelSpectrogram(self.y_sine, sr=self.sr)
    self.assertGreater(m.size, 0)
    self.assertTrue(np.all(m >= 0))

  def test_mean_spectral_contrast(self):
    s = self.helper.GetMeanSpectralContrast(self.y_sine, sr=self.sr)
    self.assertGreater(s.size, 0)

  # ===== Tonnetz =====

  def test_mean_tonnetz(self):
    t = self.helper.GetMeanTonnetz(self.y_sine, sr=self.sr)
    self.assertEqual(t.ndim, 1)
    self.assertGreater(t.shape[0], 0)

  def test_mean_harmonic_tonnetz(self):
    t = self.helper.GetMeanHarmonicTonnetz(self.y_sine, sr=self.sr)
    self.assertEqual(t.ndim, 1)
    self.assertGreater(t.shape[0], 0)

  # ===== Edge cases =====

  def test_zero_input(self):
    y = np.zeros(1024, dtype=np.float32)
    # STFT and features should handle silence
    S = self.helper.GetAbsoluteSTFT(y)
    self.assertTrue(np.all(S >= 0))
    mfcc = self.helper.GetSlaneyMFCC(y, sr=self.sr, nMFCC=13)
    self.assertEqual(mfcc.shape[0], 13)

  def test_invalid_input_type(self):
    with self.assertRaises(Exception):
      _ = self.helper.GetSTFT(None)


if __name__ == "__main__":
  unittest.main()
