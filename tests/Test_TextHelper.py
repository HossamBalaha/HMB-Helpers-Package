import unittest
from HMB.TextHelper import CleanText


class TestTextHelper(unittest.TestCase):
  '''
  Unit tests for TextHelper functions that are deterministic and lightweight.
  '''

  def test_clean_text_basic(self):
    raw = "I can't believe it's not butter!   \nNew line."
    cleaned = CleanText(raw, removeNonAscii=True, lowercase=True, removeSpecialChars=True, normalizeWhitespace=True,
                        handleContractions=True, lemmatize=False, removeStopwords=False, removeCommonWords=False,
                        removeNonEnglishWords=False)
    self.assertIsInstance(cleaned, str)
    self.assertTrue("can't" not in cleaned)
    self.assertTrue("new line" in cleaned)

  def test_contractions_variants(self):
    raw = "I'm fine. It’s okay."
    c = CleanText(raw, handleContractions=True, lowercase=True)
    self.assertTrue("im" in c or "i am" in c)

  def test_non_ascii_preservation(self):
    raw = "Café 😊"
    c_preserve = CleanText(raw, removeNonAscii=False, lowercase=True)
    self.assertTrue("caf" in c_preserve)
    c_remove = CleanText(raw, removeNonAscii=True, lowercase=True)
    self.assertTrue("caf" in c_remove)
    self.assertTrue("😊" not in c_remove)

  def test_stopwords_and_lemmatize(self):
    raw = "This is a simple test for running tests"
    c = CleanText(raw, lowercase=True, handleContractions=False, removeStopwords=True, lemmatize=False)
    # Common stopwords removed
    self.assertTrue("this" not in c)
    # Lemmatize may reduce running->run if enabled
    c_lem = CleanText(raw, lowercase=True, lemmatize=True, removeStopwords=False)
    self.assertIsInstance(c_lem, str)

  def test_empty_and_whitespace_inputs(self):
    self.assertEqual(CleanText("", normalizeWhitespace=True), "")
    self.assertEqual(CleanText("   ", normalizeWhitespace=True), "")

  def test_idempotence(self):
    raw = "Text!! with   spaces"
    c1 = CleanText(raw, removeSpecialChars=True, normalizeWhitespace=True, lowercase=True)
    c2 = CleanText(c1, removeSpecialChars=True, normalizeWhitespace=True, lowercase=True)
    self.assertEqual(c1, c2)

  def test_large_input(self):
    raw = ("Hello!! ") * 10000
    c = CleanText(raw, removeSpecialChars=True, normalizeWhitespace=True)
    self.assertTrue(len(c) > 0)

  def test_clean_text_invalid_inputs(self):
    with self.assertRaises(Exception):
      _ = CleanText(None)
    with self.assertRaises(Exception):
      _ = CleanText(123)

  def test_emoji_and_symbols(self):
    raw = "Hello 😊 — © ™ ®"
    c = CleanText(raw, removeNonAscii=True, removeSpecialChars=True, lowercase=True)
    self.assertIsInstance(c, str)

  def test_mixed_whitespace_newlines_tabs(self):
    raw = "Line1\n\nLine2\t\tLine3  \n"
    c = CleanText(raw, normalizeWhitespace=True)
    self.assertTrue("\n" not in c)

  def test_aggressive_flags_combo(self):
    raw = "We're testing, with multiple FLAGS!!"
    c = CleanText(
      raw,
      removeNonAscii=True,
      lowercase=True,
      removeSpecialChars=True,
      normalizeWhitespace=True,
      handleContractions=True,
      lemmatize=True,
      removeStopwords=True,
      removeCommonWords=True,
      removeNonEnglishWords=True,
    )
    self.assertIsInstance(c, str)


if __name__ == "__main__":
  unittest.main()
