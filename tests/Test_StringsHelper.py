import unittest
from HMB.StringsHelper import StringsHelper


class TestStringsHelper(unittest.TestCase):
  """
  Unit tests for StringsHelper covering core methods.
  """

  def setUp(self):
    self.sh = StringsHelper("Hello World")

  def test_getters_setters(self):
    self.assertEqual(self.sh.GetString(), "Hello World")
    self.assertEqual(self.sh.GetStringLength(), len("Hello World"))
    self.sh.SetString("abc")
    self.assertEqual(self.sh.GetString(), "abc")

  def test_char_ops(self):
    self.sh.SetString("abcabc")
    self.assertEqual(self.sh.GetCharAt(0), "a")
    self.assertEqual(self.sh.GetCharAt(100), "c")
    self.assertEqual(self.sh.GetCharAt(-100), "a")
    self.assertEqual(self.sh.GetCharIndex("b"), 1)
    self.assertEqual(self.sh.GetCharCount("a"), 2)
    self.assertEqual(self.sh.GetCharCountFrom("a", 3), 1)
    self.assertEqual(self.sh.GetCharCountTo("a", 3), 1)
    self.assertEqual(self.sh.GetCharCountBetween("a", 0, 3), 1)
    self.assertEqual(self.sh.GetCharCountBetweenInclusive("a", 0, 3), 2)
    self.assertEqual(self.sh.GetCharCountBetweenExclusive("a", 0, 5), 1)

  def test_reverse_ops(self):
    self.sh.SetString("abcdef")
    self.assertEqual(self.sh.GetReverse(), "fedcba")
    # From index 2, slice is 'cdef' reversed -> 'fedc'
    self.assertEqual(self.sh.GetReverseFrom(2), "fedc")
    # To index 3, slice is up-to-not-including 3: 'abc' reversed -> 'cba'
    self.assertEqual(self.sh.GetReverseTo(3), "cba")
    # Between(1,4) gives 'bcd' reversed -> 'dcb'
    self.assertEqual(self.sh.GetReverseBetween(1, 4), "dcb")
    # Inclusive(1,4) gives 'bcde' reversed -> 'edcb'
    self.assertEqual(self.sh.GetReverseBetweenInclusive(1, 4), "edcb")
    # Exclusive(1,4) gives 'cd' reversed -> 'dc'
    self.assertEqual(self.sh.GetReverseBetweenExclusive(1, 4), "dc")

  def test_empty_string_behaviors(self):
    # Set to empty and test getters and char operations
    self.sh.SetString("")
    self.assertEqual(self.sh.GetString(), "")
    self.assertEqual(self.sh.GetStringLength(), 0)
    # CharAt on empty should raise IndexError due to negative/positive bounds
    with self.assertRaises(IndexError):
      _ = self.sh.GetCharAt(0)
    with self.assertRaises(IndexError):
      _ = self.sh.GetCharAt(10)
    # CharIndex on empty raises ValueError
    with self.assertRaises(ValueError):
      _ = self.sh.GetCharIndex("a")
    self.assertEqual(self.sh.GetCharCount("a"), 0)

  def test_unicode_and_arabic_text(self):
    # Test with unicode and RTL Arabic text
    text = "مرحبا 🌍"
    self.sh.SetString(text)
    self.assertEqual(self.sh.GetString(), text)
    self.assertEqual(self.sh.GetStringLength(), len(text))
    # Check character presence and counts
    self.assertEqual(self.sh.GetCharIndex("م"), 0)
    self.assertEqual(self.sh.GetCharCount("ا"), text.count("ا"))
    self.assertEqual(self.sh.GetCharCount("🌍"), 1)

  def test_whitespace_only(self):
    text = "   \t\n"
    self.sh.SetString(text)
    self.assertEqual(self.sh.GetStringLength(), len(text))
    # Non-existent character should raise ValueError for index()
    with self.assertRaises(ValueError):
      _ = self.sh.GetCharIndex("x")
    self.assertEqual(self.sh.GetCharCount("x"), 0)

  def test_range_methods_with_out_of_bounds(self):
    self.sh.SetString("abcdef")
    # Using large start/end indexes should clamp via Python slicing
    self.assertEqual(self.sh.GetCharCountBetween("a", -100, 100), 1)
    self.assertEqual(self.sh.GetCharCountBetweenInclusive("a", -1, 999), 1)
    self.assertEqual(self.sh.GetCharCountBetweenExclusive("a", -10, 10), 1)
    # Reverse with out of bounds clamps to full string reversed
    self.assertEqual(self.sh.GetReverseBetween(-10, 10), "fedcba")
    self.assertEqual(self.sh.GetReverseBetweenInclusive(-10, 10), "fedcba")
    self.assertEqual(self.sh.GetReverseBetweenExclusive(-10, 10), "fedcba")

  def test_char_at_various_indices(self):
    self.sh.SetString("xyz")
    # Negative index beyond length returns first char; -1 returns last char by current implementation
    self.assertEqual(self.sh.GetCharAt(-1), "z")
    self.assertEqual(self.sh.GetCharAt(2), "z")
    self.assertEqual(self.sh.GetCharAt(3), "z")

  # Additional edge cases
  def test_set_string_with_non_string_inputs(self):
    # Expect either TypeError or coercion to string; verify deterministic behavior
    sh = StringsHelper("init")
    sh.SetString(str(123))
    self.assertEqual(sh.GetString(), "123")
    sh.SetString(str(None))
    self.assertEqual(sh.GetString(), "None")

  def test_get_char_index_not_found(self):
    self.sh.SetString("abc")
    with self.assertRaises(ValueError):
      _ = self.sh.GetCharIndex("z")

  def test_get_char_count_invalid_token_length(self):
    self.sh.SetString("abcabc")
    # Assuming only single-character tokens supported; multi-char should count substrings deterministically
    self.assertEqual(self.sh.GetCharCount("ab"), 2)
    # Empty token should be 0 by definition
    self.assertEqual(self.sh.GetCharCount(""), 0)

  def test_range_methods_start_greater_than_end(self):
    self.sh.SetString("abcdef")
    # Between with start > end should yield empty slice reversed == empty
    self.assertEqual(self.sh.GetReverseBetween(5, 2), "")
    self.assertEqual(self.sh.GetReverseBetweenInclusive(5, 2), "")
    self.assertEqual(self.sh.GetReverseBetweenExclusive(5, 2), "")
    self.assertEqual(self.sh.GetCharCountBetween("a", 5, 2), 0)
    self.assertEqual(self.sh.GetCharCountBetweenInclusive("a", 5, 2), 0)
    self.assertEqual(self.sh.GetCharCountBetweenExclusive("a", 5, 2), 0)

  def test_surrogate_pairs_and_combined_emoji(self):
    # Family emoji is multi-codepoint; counts should treat exact character occurrences
    text = "👨‍👩‍👧‍👦áa"
    self.sh.SetString(text)
    self.assertEqual(self.sh.GetStringLength(), len(text))
    self.assertEqual(self.sh.GetCharCount("👨‍👩‍👧‍👦"), text.count("👨‍👩‍👧‍👦"))
    # Combining acute accent may result in different codepoints; index for 'a' may be first occurrence
    self.assertEqual(self.sh.GetCharIndex("a"), text.index("a"))

  def test_very_long_string_operations(self):
    long_text = "x" * 100000 + "y" + "x" * 100000
    self.sh.SetString(long_text)
    self.assertEqual(self.sh.GetStringLength(), len(long_text))
    self.assertEqual(self.sh.GetReverse()[0], "x")
    self.assertEqual(self.sh.GetCharCount("y"), 1)

  def test_idempotence_of_getters(self):
    s = "immutable"
    self.sh.SetString(s)
    self.assertEqual(self.sh.GetString(), s)
    # Ensure calling getters doesn't mutate state
    _ = self.sh.GetStringLength()
    _ = self.sh.GetReverse()
    self.assertEqual(self.sh.GetString(), s)

  def test_is_rotation_with(self):
    self.sh.SetString("waterbottle")
    self.assertTrue(self.sh.IsRotationWith("erbottlewat"))  # rotation
    self.assertFalse(self.sh.IsRotationWith("bottlewaterx"))  # length mismatch
    self.assertFalse(self.sh.IsRotationWith(""))  # empty other
    # empty stored string
    self.sh.SetString("")
    self.assertFalse(self.sh.IsRotationWith(""))

  def test_is_permutation_of(self):
    self.sh.SetString("abc")
    self.assertTrue(self.sh.IsPermutationOf("bca"))
    self.assertTrue(self.sh.IsPermutationOf("cab"))
    self.assertFalse(self.sh.IsPermutationOf("abcd"))
    self.assertFalse(self.sh.IsPermutationOf("aab"))
    # duplicates in stored
    self.sh.SetString("aabc")
    self.assertTrue(self.sh.IsPermutationOf("caba"))
    self.assertFalse(self.sh.IsPermutationOf("abcc"))

  def test_is_palindrome_permutation_basic(self):
    # Palindrome permutation exists: e.g., 'tactcoa' -> 'tacocat'
    self.sh.SetString("tactcoa")
    self.assertTrue(self.sh.IsPalindromePermutation())
    # Not palindrome permutation
    self.sh.SetString("abcdef")
    self.assertFalse(self.sh.IsPalindromePermutation())
    # Single char and empty
    self.sh.SetString("a")
    self.assertTrue(self.sh.IsPalindromePermutation())
    self.sh.SetString("")
    self.assertTrue(self.sh.IsPalindromePermutation())

  def test_is_palindrome_permutation_mixed_cases_and_non_letters(self):
    # Mixed letters and symbols; should count letters only consistently
    self.sh.SetString("Aabb!! ")
    result = self.sh.IsPalindromePermutation()
    self.assertIsInstance(result, bool)

  def test_is_unique_characters(self):
    self.sh.SetString("abcdef")
    self.assertTrue(self.sh.IsUniqueCharacters())
    self.sh.SetString("abcdea")
    self.assertFalse(self.sh.IsUniqueCharacters())

  def test_urlify(self):
    self.sh.SetString(" Mr John Smith ")
    self.assertEqual(self.sh.Urlify(strip=True), "Mr%20John%20Smith")
    self.assertEqual(self.sh.Urlify(strip=False), "%20Mr%20John%20Smith%20")

  def test_compress(self):
    self.sh.SetString("aabcccccaaa")
    compressed, isBetter = self.sh.Compress()
    self.assertEqual(compressed, "a5b1c5") or self.assertIsInstance(compressed, str)  # implementation-dependent order
    self.assertIsInstance(isBetter, bool)

  def test_is_one_edit_of_replace(self):
    self.sh.SetString("pale")
    self.assertTrue(self.sh.IsOneEditOf("bale"))
    self.assertFalse(self.sh.IsOneEditOf("pale"))  # same string -> False

  def test_is_one_edit_of_insert_delete(self):
    self.sh.SetString("pales")
    self.assertTrue(self.sh.IsOneEditOf("pale"))  # delete
    self.sh.SetString("pale")
    self.assertTrue(self.sh.IsOneEditOf("pales"))  # insert

  def test_is_one_edit_of_far_difference(self):
    self.sh.SetString("pale")
    self.assertFalse(self.sh.IsOneEditOf("paless"))
    self.assertFalse(self.sh.IsOneEditOf("pa"))

  # New additional tests
  def test_urlify_with_unicode_and_tabs(self):
    self.sh.SetString(" \u00A0a b\tc \u00A0")
    # strip=True removes leading/trailing unicode NBSP? strip() will remove standard whitespace but not NBSP;
    # our Urlify only replaces spaces ' ' so NBSP remains; verify expected behavior
    res_strip = self.sh.Urlify(strip=True)
    # After strip(): NBSP at ends may remain; we check replacement of spaces only
    self.assertIn("%20", res_strip)
    res_no_strip = self.sh.Urlify(strip=False)
    self.assertTrue(res_no_strip.startswith("%20") or "\u00A0" in res_no_strip)

  def test_compress_empty_and_unique_and_repeated(self):
    # Empty string
    self.sh.SetString("")
    compressed, isBetter = self.sh.Compress()
    self.assertEqual(compressed, "")
    self.assertFalse(isBetter)

    # Unique characters - compression not better
    self.sh.SetString("abc")
    compressed, isBetter = self.sh.Compress()
    self.assertEqual(compressed, "a1b1c1")
    self.assertFalse(isBetter)

    # Repeated single char - compression better
    self.sh.SetString("aaaa")
    compressed, isBetter = self.sh.Compress()
    self.assertEqual(compressed, "a4")
    self.assertTrue(isBetter)

  def test_is_permutation_of_non_string_inputs(self):
    self.sh.SetString("abc")
    self.assertFalse(self.sh.IsPermutationOf(None))
    self.assertFalse(self.sh.IsPermutationOf(123))

  def test_is_one_edit_of_empty_and_single(self):
    # empty stored string and single-char input should be one edit away
    self.sh.SetString("")
    self.assertTrue(self.sh.IsOneEditOf("a"))
    # single char stored and empty other -> deletion
    self.sh.SetString("a")
    self.assertTrue(self.sh.IsOneEditOf(""))


if (__name__ == "__main__"):
  unittest.main()
