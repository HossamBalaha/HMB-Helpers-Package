import unittest
from HMB.ArabicTextHelper import ArabicTextHelper


class TestArabicTextHelper(unittest.TestCase):
  '''
  Unit tests for ArabicTextHelper basic normalization/tokenization behavior.
  '''

  def test_basic_init_and_clean(self):
    ath = ArabicTextHelper()
    text = "السلام عليكم ورحمة الله"
    cleaned = ath.CleanAndNormalize(text)
    self.assertIsInstance(cleaned, str)
    tokens = ath.TokenizeArabic(cleaned)
    self.assertTrue(isinstance(tokens, list))
    self.assertGreater(len(tokens), 0)

  def test_diacritics_removal(self):
    ath = ArabicTextHelper()
    # Text with harakat (diacritics)
    text = "السَّلَامُ عَلَيْكُمْ"
    cleaned = ath.CleanAndNormalize(text, removeDiacritics=True)
    # Expect diacritics removed to basic letters
    self.assertTrue("َ" not in cleaned)
    self.assertTrue("ِ" not in cleaned)
    self.assertTrue("ُ" not in cleaned)

  def test_letter_forms_normalization(self):
    ath = ArabicTextHelper()
    # Different forms of alef and ya
    text = "إأآى ي"
    cleaned = ath.CleanAndNormalize(text, normalizeHamza=True)
    # Expect normalized forms (implementation-dependent but should be deterministic)
    self.assertIsInstance(cleaned, str)

  def test_tatweel_removal(self):
    ath = ArabicTextHelper()
    text = "ســــلام"
    cleaned = ath.CleanAndNormalize(text, removeElongation=True)
    self.assertTrue("ـ" not in cleaned)

  def test_punctuation_and_digits(self):
    ath = ArabicTextHelper()
    text = "مرحبا! ١٢٣, test?"
    cleaned = ath.CleanAndNormalize(text)
    self.assertIsInstance(cleaned, str)
    tokens = ath.TokenizeArabic(cleaned)
    self.assertTrue(all(isinstance(t, str) for t in tokens))

  def test_mixed_rtl_ltr_with_emojis(self):
    ath = ArabicTextHelper()
    text = "مرحبا 🌟 hello"
    cleaned = ath.CleanAndNormalize(text)
    tokens = ath.TokenizeArabic(cleaned)
    # We tokenize Arabic sequences; at least Arabic token should exist
    self.assertGreaterEqual(len(tokens), 1)

  def test_empty_and_none_inputs(self):
    ath = ArabicTextHelper()
    self.assertEqual(ath.CleanAndNormalize(""), "")
    # None handling: CleanAndNormalize returns empty string for None
    self.assertEqual(ath.CleanAndNormalize(None), "")

  def test_tokenizer_whitespace_and_punctuation(self):
    ath = ArabicTextHelper()
    text = "  مرحبا،   كيف الحال؟  "
    # TokenizeArabic extracts Arabic sequences only
    tokens = ath.TokenizeArabic(text)
    self.assertTrue(len(tokens) >= 2)
    self.assertTrue(tokens[0] != "")

  def test_arabic_numerals_conversion(self):
    ath = ArabicTextHelper()
    text = "١٢٣٤٥٦٧٨٩٠"
    converted = ath.ArabicToEnglishNumbers(text)
    self.assertEqual(converted, "1234567890")

  def test_punctuation_removal(self):
    ath = ArabicTextHelper()
    text = "مرحبا، كيف الحال؟!"
    cleaned = ath.RemovePunctuations(text)
    self.assertTrue("؟" not in cleaned)

  def test_mixed_scripts_and_whitespace(self):
    ath = ArabicTextHelper()
    text = "   مرحبا   hello   123   "
    cleaned = ath.CleanAndNormalize(text)
    tokens = ath.TokenizeArabic(cleaned)
    # Arabic tokens exist even if non-Arabic is removed
    self.assertGreaterEqual(len(tokens), 1)

  def test_invalid_input_types(self):
    ath = ArabicTextHelper()
    with self.assertRaises(Exception):
      _ = ath.TokenizeArabic(123)


if __name__ == "__main__":
  unittest.main()
