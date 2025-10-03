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
  Summarizer: Flexible text summarization using Hugging Face Transformers.

  This class provides a convenient interface for abstractive text summarization using pre-trained transformer models
  (e.g., BART, T5) via the Hugging Face pipeline. It automatically handles device selection (CPU/GPU), input chunking
  for long texts, and dynamic adjustment of summary length to avoid common warnings. The class is suitable for both
  short and long documents, and can be reused for multiple summarization tasks without reloading the model.

  Features:
    - Supports any Hugging Face summarization model (default: facebook/bart-large-cnn).
    - Automatically uses GPU if available, otherwise falls back to CPU.
    - Handles long texts by splitting into manageable chunks and summarizing each chunk.
    - Dynamically sets max_length and min_length based on tokenized input length to avoid warnings.
    - Returns a single concatenated summary for the entire input.
    - Easy to customize summary length and chunk size.

  Parameters:
    modelName (str): Name of the Hugging Face model to use for summarization. Default is "facebook/bart-large-cnn".
    maxLength (int): Maximum length of the summary (in tokens). Default is 130.
    minLength (int): Minimum length of the summary (in tokens). Default is 30.
    maxInputLength (int): Maximum input length (in characters) before chunking. Default is 1024.

  Notes:
    - For best results, choose a model appropriate for your language and domain.
    - If your input text is very long, the class will split it into chunks and summarize each chunk separately.
    - The final summary is a concatenation of all chunk summaries.
    - maxLength and minLength are automatically adjusted to avoid warnings about input length.
    - You can customize chunk size by changing maxInputLength.
    - The class is thread-safe for repeated use.

  Example Usage:
  --------------
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
      modelName (str): Hugging Face model name for summarization (e.g., "facebook/bart-large-cnn", "t5-base").
      maxLength (int): Maximum summary length in tokens (will be capped by input length).
      minLength (int): Minimum summary length in tokens (will be capped by input length).
      maxInputLength (int): Maximum input length in characters before chunking. Longer texts are split into chunks.

    Notes:
      - The model is loaded once and reused for all summarization calls.
      - Device selection is automatic: uses GPU if available, otherwise CPU.
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
    Summarize the input text using the loaded transformer model.

    Parameters:
      text (str): The text to summarize. Can be short or long (long texts are chunked automatically).

    Returns:
      str: The summarized text. For long inputs, returns a concatenation of chunk summaries.

    Behavior:
      - If the input text exceeds maxInputLength, it is split into chunks of 1000 characters.
      - Each chunk is summarized separately, with maxLength and min_length set based on tokenized input length.
      - Warnings about maxLength/inputLength mismatch are avoided by dynamic adjustment.
      - If a chunk is too short, it is returned as-is without summarization.
      - The final output is a single string containing all chunk summaries.

    Example:
    --------
    .. code-block:: python

        summarizer = Summarizer()
        summary = summarizer.Summarize("Very long text ...")
        print(summary)

    Notes:
      - For best results, clean your input text before summarization.
      - You can customize summary length by changing maxLength and minLength.
      - The method is robust to empty or very short inputs.
    '''

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.modelName)

    if (len(text) > self.maxInputLength):
      # Split text into chunks of maxInputLength characters.
      chunks = [text[i:i + self.maxInputLength] for i in range(0, len(text), self.maxInputLength)]
      # Collect summaries for each chunk.
      summaries = []
      # Summarize each chunk separately.
      for chunk in chunks:
        # Tokenize the chunk to get input length.
        chunkTokens = tokenizer(chunk, return_tensors="pt")
        # Get the length of the tokenized input.
        inputLength = len(chunkTokens["input_ids"][0])

        # If input is too short, skip summarization.
        if (inputLength == 0):
          continue
        elif (inputLength < self.minLength):
          summaries.append(chunk)
          continue

        # Dynamically set maxLength to avoid warnings.
        if (inputLength > 1):
          maxLen = min(self.maxLength, inputLength - 1)
        else:
          maxLen = 1

        # Generate summary for the chunk.
        summary = self.summarizer(
          chunk,
          max_length=maxLen,
          min_length=min(self.minLength, maxLen),
          do_sample=False,
        )[0]["summary_text"]
        # Append the chunk summary to the list.
        summaries.append(summary)

      # Return the concatenated summaries for all chunks.
      return " ".join(summaries)
    else:
      # Tokenize the entire text to get input length.
      textTokens = tokenizer(text, return_tensors="pt")
      inputLength = len(textTokens["input_ids"][0])

      # If input is too short, return as-is.
      if (inputLength == 0):
        return ""
      elif (inputLength < self.minLength):
        return text

      # Dynamically set maxLength to avoid warnings.
      if (inputLength > 1):
        maxLen = min(self.maxLength, inputLength - 1)
      else:
        maxLen = 1

      # Generate summary for the entire text.
      summary = self.summarizer(
        text,
        max_length=maxLen,
        min_length=min(self.minLength, maxLen),
        do_sample=False,
      )[0]["summary_text"]
      return summary


if __name__ == "__main__":
  # Example usage of the CleanText function.
  # Define a sample text with various elements for cleaning demonstration.
  sampleText = """
  This is an example text! It includes various elements:
  - Contractions like don't and it's.
  - Special characters: @#$%^&*()!
  - Multiple     spaces and newlines.

  Let's see how well the cleaning function works.
  """

  # Call the CleanText function with all options enabled except common word removal and non-English word removal.
  cleaned = CleanText(
    sampleText,  # The raw text to clean.
    removeNonAscii=True,  # Remove non-ASCII characters.
    lowercase=True,  # Convert text to lowercase.
    removeSpecialChars=True,  # Remove special characters and punctuation.
    normalizeWhitespace=True,  # Replace multiple spaces with a single space.
    handleContractions=True,  # Expand contractions (e.g., don't → do not).
    lemmatize=True,  # Lemmatize words to their base form.
    removeStopwords=True,  # Remove common stop words.
    removeCommonWords=False,  # Do not remove most common words.
    numOfCommonWords=10,  # Number of common words to remove if enabled.
    removeNonEnglishWords=False,  # Do not remove non-English words.
  )

  # Print the original text before cleaning.
  print("Original Text:\n", sampleText)
  # Print the cleaned text after processing.
  print("\nCleaned Text:\n", cleaned)

  # Example usage of the Summarizer class.
  # Define a sample long text for summarization demonstration.
  sampleLongText = (
    "This is an example text! "
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains every letter of the English alphabet. "
    "It's often used to test fonts and keyboard layouts. "
    "In addition to its practical uses, it has a playful tone that makes it memorable. "
    "The fox is known for its cunning and agility, while the dog represents loyalty and patience. "
    "Together, they create a vivid image that captures the imagination."
  )
  # Create a Summarizer instance with default parameters.
  summarizer = Summarizer(
    modelName="facebook/bart-large-cnn",  # Use the BART model for summarization.
    maxLength=150,  # Set maximum summary length to 130 tokens.
    minLength=25,  # Set minimum summary length to 30 tokens.
    maxInputLength=1024,  # Set maximum input length to 1024 characters before chunking.
  )
  # Summarize the sample long text using the Summarizer instance.
  summary = summarizer.Summarize(sampleLongText)
  # Print the length of the original text.
  print("Length of Original Text:", len(sampleLongText))  # Output the length of the original text.
  # Print the length of the generated summary.
  print("Length of Summary:", len(summary))  # Output the length of the summary.
  # Print the original text for reference.
  print("\nOriginal Text:\n", sampleLongText)  # Output the original text.
  # Print the generated summary.
  print("\nSummary:\n", summary)  # Output the summary.
