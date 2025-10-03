# Import the required libraries.
import re, contractions, nltk


# Define a function to clean and normalize text based on several options.
def CleanText(
  text,  # The raw input text to be cleaned.
  removeNonAscii=True,  # Whether to remove non-ASCII characters.
  lowercase=True,  # Whether to convert text to lowercase.
  removeSpecialChars=True,  # Whether to remove special characters and punctuation.
  normalizeWhitespace=True,  # Whether to replace multiple spaces with a single space.
  handleContractions=True,  # Whether to expand contractions (e.g., "don't" → "do not").
  lemmatize=False,  # Whether to lemmatize words (reduce to base form).
  removeStopwords=False,  # Whether to remove common stop words.
  removeCommonWords=False,  # Whether to remove common words.
  numOfCommonWords=10,  # Number of common words to remove if removeCommonWords is True.
  removeNonEnglishWords=False,  # Whether to remove non-English words.
):
  r'''
  Cleans the input text based on specified options. It applies multiple text normalization techniques including (1)
  removing non-ASCII characters, (2) converting to lowercase, (3) removing special characters and punctuation,
  (4) normalizing whitespace, (5) expanding contractions, (6) lemmatizing words, (7) removing stopwords,
  (8) removing common words, and (9) removing non-English words.

  Parameters:
    text (str): The raw input text to be cleaned.
    removeNonAscii (bool): Whether to remove non-ASCII characters. Default is True.
    lowercase (bool): Whether to convert text to lowercase. Default is True.
    removeSpecialChars (bool): Whether to remove special characters and punctuation. Default is True.
    normalizeWhitespace (bool): Whether to replace multiple spaces with a single space. Default is True.
    handleContractions (bool): Whether to expand contractions (e.g., "don't" → "do not"). Default is True.
    lemmatize (bool): Whether to lemmatize words (reduce to base form). Default is False.
    removeStopwords (bool): Whether to remove common stop words. Default is False.
    removeCommonWords (bool): Whether to remove common words. Default is False.
    numOfCommonWords (int): Number of common words to remove if removeCommonWords is True. Default is 10.
    removeNonEnglishWords (bool): Whether to remove non-English words. Default is False.

  Returns:
    str: The cleaned and normalized text.

  Examples
  --------
  .. code-block:: python

    import HMB.TextHelper as th

    raw = "I can't believe it's not butter!   "
    cleaned = th.CleanText(
      raw,
      removeNonAscii=True,
      lowercase=True,
      removeSpecialChars=True,
      normalizeWhitespace=True,
      handleContractions=True,
      lemmatize=True,
      removeStopwords=True,
      removeCommonWords=False,
      numOfCommonWords=10,
      removeNonEnglishWords=False,
    )
    print(cleaned)
  '''

  # Remove empty lines and replace newlines with spaces.
  cleanedText = " ".join([
    line.replace("\n", " ").strip()  # Replace newlines with spaces and strip whitespace.
    for line in text.splitlines()  # Split text into lines.
    if (line.strip() != "")  # Ignore empty lines.
  ])

  # Remove non-ASCII characters if specified.
  if (removeNonAscii):
    cleanedText = cleanedText.encode("ascii", "ignore").decode("ascii")  # Keep only ASCII characters.

  # Convert text to lowercase if specified.
  if (lowercase):
    cleanedText = cleanedText.lower()  # Lowercase all characters.

  # Expand contractions if specified.
  if (handleContractions):
    cleanedText = contractions.fix(cleanedText)  # Expand contractions (e.g., don't → do not).

  # Remove special characters and punctuation if specified.
  if (removeSpecialChars):
    # Remove non-alphanumeric characters.
    cleanedText = re.sub(r"[^a-zA-Z0-9\s]", "", cleanedText)

  # Normalize whitespace around punctuation.
  # Remove extra spaces before punctuation.
  cleanedText = re.sub(r"\s+([?.!,])", r"\1", cleanedText)

  # Lemmatize words if specified.
  if (lemmatize):
    from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer from NLTK.
    lemmatizer = WordNetLemmatizer()  # Create a lemmatizer instance.
    cleanedText = " ".join([lemmatizer.lemmatize(word) for word in cleanedText.split()])  # Lemmatize each word.

  # Remove stopwords if specified.
  # Stopwords are common words that may not add significant meaning to the text.
  if (removeStopwords):
    from nltk.corpus import stopwords  # Import stopwords from NLTK.
    stopWords = set(stopwords.words("english"))  # Get English stopwords.
    cleanedText = " ".join([word for word in cleanedText.split() if word.lower() not in stopWords])  # Remove stopwords.

  # Replace multiple spaces with a single space if specified.
  if (normalizeWhitespace):
    # Normalize whitespace.
    cleanedText = " ".join(cleanedText.split())

  # Remove common words if specified.
  if (removeCommonWords):
    # Get word frequency distribution.
    wordFreq = nltk.FreqDist(cleanedText.split())
    # Get the N most common words.
    commonWords = set([word for word, freq in wordFreq.most_common(numOfCommonWords)])
    # Remove common words.
    cleanedText = " ".join([word for word in cleanedText.split() if word not in commonWords])

  # Remove non-English words if specified.
  if (removeNonEnglishWords):
    from nltk.corpus import words  # Import words corpus from NLTK.
    # Get a set of valid English words.
    englishWords = set(words.words())
    # Keep only English words.
    cleanedText = " ".join([word for word in cleanedText.split() if word.lower() in englishWords])

    # Return the cleaned text.
  return cleanedText


class Summarizer(object):
  r'''
  Summarizer class using Hugging Face Transformers summarization pipeline.
  Loads the model once and provides a summarize method for repeated use.
  Uses chunking for long texts to handle model input size limitations.

  Parameters:
    modelName (str): The name of the Hugging Face model to use for summarization. Default is "facebook/bart-large-cnn".
    maxLength (int): The maximum length of the summary. Default is 130.
    minLength (int): The minimum length of the summary. Default is 30.
    maxInputLength (int): The maximum length of input text to process at once. Longer texts will be chunked. Default is 1024.

  Examples
  --------
  .. code-block:: python

    from HMB.TextHelper import Summarizer

    text = "Your long text to summarize goes here..."
    summarizer = Summarizer(
      modelName="facebook/bart-large-cnn",
      maxLength=130,
      minLength=30,
      maxInputLength=1024,
    )
    summary = summarizer.Summarize(text)
    print(summary)
  '''

  def __init__(
    self,
    modelName="facebook/bart-large-cnn",
    maxLength=130,
    minLength=30,
    maxInputLength=1024,
  ):
    r'''
    Initialize the Summarizer with the specified model and parameters.

    Parameters:
      modelName (str): The name of the Hugging Face model to use for summarization. Default is "facebook/bart-large-cnn".
      maxLength (int): The maximum length of the summary. Default is 130.
      minLength (int): The minimum length of the summary. Default is 30.
      maxInputLength (int): The maximum length of input text to process at once. Longer texts will be chunked. Default is 1024.
    '''

    import torch
    from transformers import pipeline

    self.modelName = modelName
    self.maxLength = maxLength
    self.minLength = minLength
    self.maxInputLength = maxInputLength
    self.summarizer = pipeline(
      "summarization",
      model=modelName,
      device=0 if (torch.cuda.is_available()) else -1,
    )

  def Summarize(self, text):
    r'''
    Summarize the input text.

    Parameters:
      text (str): The text to summarize.

    Returns:
      str: The summarized text.
    '''

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.modelName)

    if (len(text) > self.maxInputLength):
      chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
      summaries = []
      for chunk in chunks:
        chunkTokens = tokenizer(chunk, return_tensors="pt")
        inputLength = len(chunkTokens["input_ids"][0])

        if (inputLength == 0):
          continue
        elif (inputLength < self.minLength):
          summaries.append(chunk)
          continue

        maxLen = min(self.maxLength, inputLength)
        summary = self.summarizer(
          chunk,
          max_length=maxLen,
          min_length=min(self.minLength, maxLen),
          do_sample=False,
        )[0]["summary_text"]
        summaries.append(summary)
      return " ".join(summaries)
    else:
      chunkTokens = tokenizer(text, return_tensors="pt")
      inputLength = len(chunkTokens["input_ids"][0])
      maxLen = min(self.maxLength, inputLength)
      summary = self.summarizer(
        text,
        max_length=maxLen,
        min_length=min(self.minLength, maxLen),
        do_sample=False
      )[0]["summary_text"]
      return summary


if __name__ == "__main__":
  # Example usage of the CleanText function.
  sampleText = """
  This is an example text! It includes various elements:
  - Contractions like don't and it's.
  - Special characters: @#$%^&*()!
  - Multiple     spaces and newlines.

  Let's see how well the cleaning function works.
  """

  cleaned = CleanText(
    sampleText,
    removeNonAscii=True,
    lowercase=True,
    removeSpecialChars=True,
    normalizeWhitespace=True,
    handleContractions=True,
    lemmatize=True,
    removeStopwords=True,
    removeCommonWords=False,
    numOfCommonWords=10,
    removeNonEnglishWords=False,
  )

  print("Original Text:\n", sampleText)
  print("\nCleaned Text:\n", cleaned)

  # Example usage of the Summarizer class.
  sampleLongText = (
    "This is an example text! "
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains every letter of the English alphabet. "
    "It's often used to test fonts and keyboard layouts. "
    "In addition to its practical uses, it has a playful tone that makes it memorable. "
    "The fox is known for its cunning and agility, while the dog represents loyalty and patience. "
    "Together, they create a vivid image that captures the imagination."
  )
  summarizer = Summarizer()
  summary = summarizer.Summarize(sampleLongText)
  print("Length of Original Text:", len(sampleLongText))  # Output the length of the original text.
  print("Length of Summary:", len(summary))  # Output the length of the summary.
  print("\nOriginal Text:\n", sampleLongText)  # Output the original text.
  print("\nSummary:\n", summary)  # Output the summary.
