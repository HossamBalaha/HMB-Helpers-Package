import re, emoji, nltk
from HMB.TextHelper import TextHelper


class ArabicTextHelper(TextHelper):
  r'''
  ArabicTextHelper: Utilities for Arabic text preprocessing.

  This helper provides common Arabic-specific text preprocessing routines
  such as regex-based normalization and cleaning, ISRI stemming, Qalsadi
  lemmatization, and stopword removal. The methods are thin wrappers
  around NLTK and qalsadi functionality where applicable and try to
  preserve the original behavior while providing clear documentation.

  Notes:
    - Some methods require external packages (nltk, qalsadi) to be installed.
    - Methods accept and return lists of strings for batch processing.
    - Can be installed via `pip install emoji nltk qalsadi`.
  '''

  def ArabicRegexPreprocessing(
    self, data, normalizeChars=True, removeHashes=True, removeUsernames=True,
    removeEmojis=True, removeLinks=True, removeDiacritics=True
  ):
    r'''
    Apply regex-based normalization and cleaning to Arabic text documents.

    Parameters:
      data (list-like): Iterable of strings to preprocess.
      normalizeChars (bool): Normalize common Arabic character variants.
      removeHashes (bool): Remove or normalize hashtags (#tag).
      removeUsernames (bool): Remove @user mentions.
      removeEmojis (bool): Remove emoji characters.
      removeLinks (bool): Remove URLs and web-like tokens.
      removeDiacritics (bool): Remove Arabic diacritics (tashkeel).

    Returns:
      list: Preprocessed documents as strings.

    Notes:
      - This function performs character normalization, optional diacritics
        stripping, removal of usernames/hashtags/links/emojis, and token
        cleaning (removing non-Arabic letters and digits).
    '''

    documents = []
    for i in range(0, len(data)):
      document = str(data[i])
      # Convert to lowercase.
      document = document.lower()
      if (normalizeChars):
        # Normalize common Arabic characters.
        document = self.NormalizeArabic(document, removeElongation=True, normalizeHamza=True)
      if (removeDiacritics):
        # Remove Arabic diacritics (tashkeel) using a verbose regex.
        document = self.RemoveDiacritics(document)
      # Optionally remove usernames (@user) from social media style text.
      if (removeUsernames):
        document = re.sub("@[^\s]+", " ", document)
      # Remove or normalize hashtags.
      if (removeHashes):
        document = re.sub(r"#[a-zA-Z0-9_\u0600-\u06FF]+", " ", document)
        document = re.sub(r"#([^\s]+)", r"\1", document)
      # Optionally strip emojis using the emoji package"s lookup.
      if (removeEmojis):
        document = "".join(c for c in document if c not in emoji.unicode_codes.EMOJI_DATA.keys())
      # Remove URLs and web-like tokens.
      if (removeLinks):
        document = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", " ", document)
        document = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", " ", document)
      # Replace non-word characters with space.
      document = re.sub(r"\W", " ", document)
      # Keep only Arabic letters and digits (then remove digits later).
      document = re.sub(r"[^0-9\u0600-\u06FF]+", " ", document)
      # Remove isolated single characters (Arabic letters or digits).
      document = re.sub(r"\s+[0-9\u0600-\u06FF]\s+", " ", document)
      # Remove single characters at the start of the string.
      document = re.sub(r"\^[0-9\u0600-\u06FF]\s+", " ", document)
      # Collapse multiple spaces into a single space.
      document = re.sub(r"\s+", " ", document, flags=re.I)
      # Remove a leading "b" prefix if present (from byte-string reprs).
      document = re.sub(r"^b\s+", "", document)
      # Remove digits entirely.
      document = re.sub(r"[0-9]+", " ", document)
      # Append the cleaned document to the output list.
      documents.append(document)
    return documents

  def ArabicISRIStemmerPreprocessing(self, data):
    r'''
    Apply ISRI stemming to each document using NLTK's ISRIStemmer.

    Parameters:
      data (list-like): Iterable of strings to stem. Each string is tokenized using the helper's SentenceTokeize method before stemming.

    Returns:
      list: Stemmed documents as joined strings.
    '''

    from nltk.stem.isri import ISRIStemmer

    documents = []
    # Initialize the ISRI stemmer.
    stemmer = ISRIStemmer()
    for i in range(0, len(data)):
      # Tokenize the sentence (using existing helper method) then stem tokens.
      document = self.SentenceTokeize(data[i])
      document = " ".join([stemmer.stem(token) for token in document])
      documents.append(document)
    return documents

  def ArabicQalsadiLemmatizerPreprocessing(self, data):
    r'''
    Lemmatize Arabic text using the Qalsadi lemmatizer.

    Parameters:
      data (list-like): Iterable of strings to lemmatize.

    Returns:
      list: Lemmatized documents as joined strings.

    Notes:
      - Requires the `qalsadi` package to be installed.
    '''

    from qalsadi.lemmatizer import Lemmatizer
    documents = []
    # Qalsadi Arabic Morphological Analyzer for Python.
    lemmer = Lemmatizer()
    for i in range(0, len(data)):
      document = data[i]
      # Lemmatize and join tokens back into a string.
      document = " ".join(lemmer.lemmatize_text(document))
      documents.append(document)
    return documents

  def ArabicStopwordsRemovalPreprocessing(self, data):
    r'''
    Remove Arabic stopwords from tokenized documents.

    Parameters:
      data (list-like): Iterable of strings to remove stopwords from. Each string is tokenized using the helper's SentenceTokeize method.

    Returns:
      list: Documents with stopwords removed and joined back into strings.
    '''

    from nltk.corpus import stopwords

    documents = []
    # Load Arabic stopwords from NLTK.
    stop = set(stopwords.words("arabic"))
    for i in range(0, len(data)):
      # Tokenize and filter stopwords, then join back to a string.
      document = self.SentenceTokeize(data[i])
      document = [word for word in document if (word not in stop)]
      document = " ".join(document)
      documents.append(document)
    return documents

  def RemovePunctuations(self, text, extra_punct=None):
    r'''
    Remove punctuation characters from a text string, including Arabic punctuation.

    Parameters:
      text (str): Input string.
      extra_punct (str or None): Additional characters to treat as punctuation.

    Returns:
      str: Text with punctuation replaced by spaces.
    '''
    import string
    # Common ASCII punctuation plus Arabic-specific punctuation marks.
    arabicPunct = "،؛؟«»…“”ـ"
    punct = string.punctuation + arabicPunct
    if (extra_punct):
      punct = punct + str(extra_punct)
    # Build a character class that escapes all chars and replace them with a space.
    pattern = f"[{re.escape(punct)}]"
    return re.sub(pattern, " ", text)

  def RemoveDiacritics(self, text):
    r'''
    Remove Arabic diacritics (tashkeel) from a string.

    Parameters:
      text (str): Input Arabic text.

    Returns:
      str: Text without Arabic diacritics.
    '''

    # Reuse the same verbose regex as other methods for consistency.
    arabicDiacritics = re.compile(
      """
                      ّ    | # Shadda
                      َ    | # Fatha
                      ً    | # Tanwin Fath
                      ُ    | # Damma
                      ٌ    | # Tanwin Damm
                      ِ    | # Kasra
                      ٍ    | # Tanwin Kasr
                      ْ    | # Sukun
                      ـ     # Tatwil/Kashida
                      """, re.VERBOSE
    )
    return re.sub(arabicDiacritics, '', text)

  def NormalizeArabic(self, text, removeElongation=True, normalizeHamza=True):
    r'''
    Normalize common Arabic characters and optionally remove elongation.

    Parameters:
      text (str): Input Arabic text.
      remove_elongation (bool): If True, collapse repeated characters (e.g., "جممميل" → "جميل").
      normalizeHamza (bool): If True, normalize hamza variants to bare alef/hamza forms.

    Returns:
      str: Normalized text.
    '''

    # Normalize common alef/hamza/y/ta/ka variants.
    if (normalizeHamza):
      text = re.sub("[إأآا]", "ا", text)
      text = re.sub("ى", "ي", text)
      text = re.sub("ة", "ه", text)
      text = re.sub("گ", "ك", text)
      text = re.sub("ؤ", "ء", text)
      text = re.sub("ئ", "ء", text)
    # Optionally remove elongation (character repetitions).
    if (removeElongation):
      # Replace 3 or more repetitions of the same character with a single char.
      text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text

  def ArabicToEnglishNumbers(self, text):
    r'''
    Convert Arabic-Indic and Persian digits to Western digits (0-9).

    Parameters:
      text (str): Input string containing digits.

    Returns:
      str: String where Arabic/Persian digits are mapped to ASCII digits.
    '''

    # Arabic-Indic digits and Eastern Arabic (Persian) digits.
    arabic_nums = "٠١٢٣٤٥٦٧٨٩"
    persian_nums = "۰۱۲۳۴۵۶۷۸۹"
    # Build translation table mapping codepoints to ASCII digits.
    trans = {}
    for i, ch in enumerate(arabic_nums):
      trans[ord(ch)] = str(i)
    for i, ch in enumerate(persian_nums):
      trans[ord(ch)] = str(i)
    return text.translate(trans)

  def IsArabic(self, text):
    r'''
    Heuristic check whether a string contains Arabic characters.

    Parameters:
      text (str): Input string.

    Returns:
      bool: True if any Arabic-range character is present, False otherwise.
    '''

    return bool(re.search(r"[\u0600-\u06FF]", text))

  def TokenizeArabic(self, text):
    r'''
    Simple Arabic tokenizer: extract contiguous Arabic letter sequences as tokens.

    Parameters:
      text (str): Input Arabic text.

    Returns:
      list: List of Arabic word tokens.

    Notes:
      - This is an intentionally lightweight tokenizer. For advanced tokenization
        consider using language-specific tokenizers (Farasa, Camel Tools, etc.).
    '''

    # Find sequences of Arabic letters and return them as tokens.
    tokens = re.findall(r"[\u0600-\u06FF]+", text)
    return tokens

  def CleanAndNormalize(
    self, text, removeDiacritics=True, removePunct=True,
    normalizeHamza=True, removeElongation=True, convertNumbers=True
  ):
    r'''
    Run a compact cleaning pipeline on a single text string.

    This helper composes commonly used operations: normalize characters,
    optionally remove diacritics, strip punctuation, convert Arabic numbers
    to Western digits, collapse whitespace, and trim the result.

    Parameters:
      text (str): Input text.
      removeDiacritics (bool): Remove Arabic diacritics if True.
      removePunct (bool): Remove punctuation if True.
      normalizeHamza (bool): Normalize hamza/alef variants if True.
      removeElongation (bool): Remove repeated character elongation if True.
      convertNumbers (bool): Convert Arabic/Persian digits to ASCII digits if True.

    Returns:
      str: Cleaned and normalized text.
    '''

    if (text is None):
      return ''
    s = str(text)
    # Normalize characters (hamza/alef/y/ta/ka) and optionally remove elongation.
    s = self.NormalizeArabic(s, removeElongation=removeElongation, normalizeHamza=normalizeHamza)
    # Remove diacritics if requested.
    if (removeDiacritics):
      s = self.RemoveDiacritics(s)
    # Remove punctuation if requested.
    if (removePunct):
      s = self.RemovePunctuations(s)
    # Convert Arabic/Persian digits to Western digits.
    if (convertNumbers):
      s = self.ArabicToEnglishNumbers(s)
    # Collapse whitespace and strip.
    s = re.sub(r"\s+", " ", s).strip()
    return s

  def StripNonArabic(self, text, keepSpaces=True):
    r'''
    Remove characters that are not in the Arabic Unicode block.

    Parameters:
      text (str): Input string.
      keepSpaces (bool): If True, preserve spaces between words; otherwise remove them.

    Returns:
      str: String containing only Arabic letters (and spaces if requested).
    '''

    if (text is None):
      return ''
    if (keepSpaces):
      return " ".join(re.findall(r"[\u0600-\u06FF]+", str(text)))
    # Return concatenated Arabic letters without spaces.
    return ''.join(re.findall(r"[\u0600-\u06FF]+", str(text)))

  def ArabicCharRatio(self, text):
    r'''
    Compute the ratio of Arabic characters to all characters in the string.

    Parameters:
      text (str): Input string.

    Returns:
      float: Fraction in [0,1] of characters that are Arabic. Returns 0.0 for empty input.
    '''

    if (not text):
      return 0.0
    s = str(text)
    total = len(s)
    if (total == 0):
      return 0.0
    arabicCount = len(re.findall(r"[\u0600-\u06FF]", s))
    return arabicCount / float(total)

  def ArabicWordCount(self, text):
    r'''
    Count Arabic words/tokens in a string using the lightweight tokenizer.

    Parameters:
      text (str): Input string.

    Returns:
      int: Number of Arabic word tokens found.
    '''

    if (not text):
      return 0
    tokens = self.TokenizeArabic(text)
    return len(tokens)

  def GetArabicCharNGrams(self, text, n=3):
    r'''
    Return character n-grams from the Arabic-only content of the text.

    Parameters:
      text (str): Input string.
      n (int): Size of the n-grams (default 3).

    Returns:
      list: List of character n-gram strings. Returns empty list if no Arabic text found.

    Notes:
      - This operates on the concatenated Arabic tokens (spaces removed) to produce contiguous character n-grams.
    '''

    if (not text or n <= 0):
      return []
    arabicOnly = self.StripNonArabic(text, keepSpaces=False)
    if (len(arabicOnly) < n):
      return []
    grams = [arabicOnly[i:i + n] for i in range(len(arabicOnly) - n + 1)]
    return grams

  def RemoveEmojis(self, text):
    r'''
    Remove emoji characters from a string.

    Parameters:
      text (str): Input string.

    Returns:
      str: String with emojis removed.
    '''

    return "".join(c for c in text if c not in emoji.unicode_codes.EMOJI_DATA.keys())


if __name__ == "__main__":
  # Quick interactive tests for the ArabicTextHelper methods.
  helper = ArabicTextHelper()

  sample = "مرحبا!!! هذا اختبار 😊 #هاشتاق @user ١٢٣٤٥٦"
  print("Original:", sample)


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


  SafeCall("CleanAndNormalize", helper.CleanAndNormalize, sample)
  SafeCall("ArabicRegexPreprocessing", helper.ArabicRegexPreprocessing, [sample])
  SafeCall("RemovePunctuations", helper.RemovePunctuations, sample)
  SafeCall("RemoveDiacritics", helper.RemoveDiacritics, "السَّلامُ")
  SafeCall("NormalizeArabic", helper.NormalizeArabic, "أهلاًاااا")
  SafeCall("ArabicToEnglishNumbers", helper.ArabicToEnglishNumbers, "١٢٣٤٥٦٧٨٩ ۰۱۲۳")
  SafeCall("IsArabic", helper.IsArabic, sample)
  SafeCall("TokenizeArabic", helper.TokenizeArabic, sample)
  SafeCall("CleanAndNormalize (no emojis)", helper.CleanAndNormalize, sample, True, True, True, True, True)
  SafeCall("StripNonArabic (keep spaces)", helper.StripNonArabic, sample, True)
  SafeCall("StripNonArabic (no spaces)", helper.StripNonArabic, sample, False)
  SafeCall("ArabicCharRatio", helper.ArabicCharRatio, sample)
  SafeCall("ArabicWordCount", helper.ArabicWordCount, sample)
  SafeCall("GetArabicCharNGrams", helper.GetArabicCharNGrams, sample, 3)
  SafeCall("RemoveEmojis", helper.RemoveEmojis, sample)

  # Methods that depend on external packages or NLTK corpora — run safely.
  try:
    SafeCall("ArabicISRIStemmerPreprocessing", helper.ArabicISRIStemmerPreprocessing, ["كتبت"])
  except Exception as e:
    print("ArabicISRIStemmerPreprocessing skipped:", type(e).__name__, e)

  try:
    SafeCall("ArabicQalsadiLemmatizerPreprocessing", helper.ArabicQalsadiLemmatizerPreprocessing, [sample])
  except Exception as e:
    print("ArabicQalsadiLemmatizerPreprocessing skipped:", type(e).__name__, e)

  try:
    SafeCall("ArabicStopwordsRemovalPreprocessing", helper.ArabicStopwordsRemovalPreprocessing, [sample])
  except Exception as e:
    print("ArabicStopwordsRemovalPreprocessing skipped:", type(e).__name__, e)

  print("Tests completed.")
