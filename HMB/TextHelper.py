# Import the required libraries.
import re, contractions


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
):
  r'''
  Cleans the input text based on specified options. It applies multiple text normalization techniques including (1)
  removing non-ASCII characters, (2) converting to lowercase, (3) removing special characters and punctuation,
  (4) normalizing whitespace, (5) expanding contractions, (6) lemmatizing words, and (7) removing stopwords.

  Parameters:
    text (str): The raw input text to be cleaned.
    removeNonAscii (bool): Whether to remove non-ASCII characters.
    lowercase (bool): Whether to convert text to lowercase.
    removeSpecialChars (bool): Whether to remove special characters and punctuation.
    normalizeWhitespace (bool): Whether to replace multiple spaces with a single space.
    handleContractions (bool): Whether to expand contractions (e.g., "don't" → "do not").
    lemmatize (bool): Whether to lemmatize words (reduce to base form).
    removeStopwords (bool): Whether to remove common stop words.

  Returns:
    str: The cleaned and normalized text.

  Examples
  --------
  .. code-block:: python

    import HMB.TextHelper as th
    raw = "I can't believe it's not butter!   "
    cleaned = th.CleanText(raw)
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
    cleanedText = re.sub(r"[^a-zA-Z0-9\s]", "", cleanedText)  # Remove non-alphanumeric characters.

  # Normalize whitespace around punctuation.
  cleanedText = re.sub(r"\s+([?.!,])", r"\1", cleanedText)  # Remove extra spaces before punctuation.

  # Lemmatize words if specified.
  if (lemmatize):
    lemmatizer = WordNetLemmatizer()  # Create a lemmatizer instance.
    cleanedText = " ".join([lemmatizer.lemmatize(word) for word in cleanedText.split()])  # Lemmatize each word.

  # Remove stopwords if specified.
  if (removeStopwords):
    stopWords = set(stopwords.words("english"))  # Get English stopwords.
    cleanedText = " ".join([word for word in cleanedText.split() if word.lower() not in stopWords])  # Remove stopwords.

  # Replace multiple spaces with a single space if specified.
  if (normalizeWhitespace):
    cleanedText = " ".join(cleanedText.split())  # Normalize whitespace.

  # Return the cleaned text.
  return cleanedText


if __name__ == "__main__":
  # Example usage of the CleanText function.
  sampleText = """
  This is an example text! It includes various elements:
  - Contractions like don't and it's.
  - Special characters: @#$%^&*()!
  - Multiple     spaces and newlines.

  Let's see how well the cleaning function works.
  """

  # Print the cleaned version of the sample text.
  print(CleanText(sampleText))  # Output cleaned text.
