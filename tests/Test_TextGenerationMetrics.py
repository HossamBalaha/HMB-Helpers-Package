import unittest
from HMB.TextGenerationMetrics import TextGenerationMetrics


class TestTextGenerationMetrics(unittest.TestCase):
  '''
  Unit tests for TextGenerationMetrics focusing on lightweight metrics.
  Covers BLEU, ROUGE, METEOR (approximation), and EditDistance similarity.
  '''

  def setUp(self):
    self.tgm = TextGenerationMetrics()
    self.ref = "the quick brown fox jumps over the lazy dog"
    self.genIdentical = "the quick brown fox jumps over the lazy dog"
    self.genPartial = "the quick brown cat jumps over a lazy dog"
    self.genDifferent = "completely unrelated sentence here"

  # ========== BLEU ==========

  def test_bleu_identical_high(self):
    score = self.tgm.CalculateBLEU(self.genIdentical, self.ref)
    self.assertGreater(score, 0.9)

  def test_bleu_partial_mid(self):
    score = self.tgm.CalculateBLEU(self.genPartial, self.ref)
    self.assertGreaterEqual(score, 0.0)
    self.assertLessEqual(score, 1.0)

  def test_bleu_different_low(self):
    score = self.tgm.CalculateBLEU(self.genDifferent, self.ref)
    self.assertLess(score, 0.3)

  def test_bleu_empty_candidate(self):
    score = self.tgm.CalculateBLEU("", self.ref)
    self.assertGreaterEqual(score, 0.0)

  # ========== ROUGE ==========

  def test_rouge_identical_high(self):
    scores = self.tgm.CalculateROUGE(self.genIdentical, self.ref)
    self.assertTrue(all(k in scores for k in ["rouge-1", "rouge-2", "rouge-l"]))
    self.assertGreater(scores["rouge-1"], 0.9)
    self.assertGreater(scores["rouge-l"], 0.9)

  def test_rouge_different_low(self):
    scores = self.tgm.CalculateROUGE(self.genDifferent, self.ref)
    self.assertGreaterEqual(scores["rouge-1"], 0.0)
    self.assertLess(scores["rouge-1"], 0.6)

  def test_rouge_empty_reference(self):
    scores = self.tgm.CalculateROUGE(self.genIdentical, "")
    self.assertTrue(isinstance(scores, dict))

  # ========== METEOR (approx) ==========

  def test_meteor_identical_one(self):
    mete = self.tgm.CalculateMETEOR(self.genIdentical, self.ref)
    self.assertAlmostEqual(mete, 1.0, places=6)

  def test_meteor_partial_mid(self):
    mete = self.tgm.CalculateMETEOR(self.genPartial, self.ref)
    self.assertGreaterEqual(mete, 0.0)
    self.assertLessEqual(mete, 1.0)

  def test_meteor_different_zero(self):
    mete = self.tgm.CalculateMETEOR(self.genDifferent, self.ref)
    self.assertEqual(mete, 0.0)

  def test_meteor_unicode(self):
    mete = self.tgm.CalculateMETEOR("Café 😊", "Cafe")
    self.assertGreaterEqual(mete, 0.0)

  # ========== EditDistance similarity ==========

  def test_edit_distance_identical_one(self):
    sim = self.tgm.CalculateEditDistance(self.genIdentical, self.ref)
    self.assertAlmostEqual(sim, 1.0, places=6)

  def test_edit_distance_partial_mid(self):
    sim = self.tgm.CalculateEditDistance(self.genPartial, self.ref)
    self.assertGreaterEqual(sim, 0.0)
    self.assertLessEqual(sim, 1.0)

  def test_edit_distance_different_low(self):
    sim = self.tgm.CalculateEditDistance(self.genDifferent, self.ref)
    self.assertLess(sim, 0.5)

  def test_edit_distance_long_inputs(self):
    a = ("a " * 1000).strip()
    b = ("a " * 900 + "b " * 100).strip()
    sim = self.tgm.CalculateEditDistance(a, b)
    self.assertGreaterEqual(sim, 0.0)
    self.assertLessEqual(sim, 1.0)

  def test_invalid_inputs_raise(self):
    with self.assertRaises(Exception):
      _ = self.tgm.CalculateBLEU(None, self.ref)
    with self.assertRaises(Exception):
      _ = self.tgm.CalculateROUGE(self.genIdentical, None)


if __name__ == "__main__":
  unittest.main()
