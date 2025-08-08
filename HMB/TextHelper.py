'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Aug 7th, 2025
# Last Modification Date: Aug 7th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import re, contractions, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Automatically download NLTK resources if not already present.
nltk.download("stopwords")
nltk.download("wordnet")


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
  '''
  Cleans the input text based on specified options.
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
    str: The cleaned text.
  '''

  # Remove empty lines and replace newlines with spaces.
  cleanedText = " ".join([
    line.replace("\n", " ").strip()
    for line in text.splitlines()
    if (line.strip() != "")
  ])

  # Remove non-ASCII characters.
  if (removeNonAscii):
    cleanedText = cleanedText.encode("ascii", "ignore").decode("ascii")

  # Lowercase the text.
  if (lowercase):
    cleanedText = cleanedText.lower()

  # Handle contractions.
  if (handleContractions):
    cleanedText = contractions.fix(cleanedText)

  # Remove or replace special characters.
  if (removeSpecialChars):
    # Removes all non-alphanumeric characters.
    cleanedText = re.sub(r"[^a-zA-Z0-9\s]", "", cleanedText)

  # Normalize whitespace around punctuation.
  cleanedText = re.sub(r"\s+([?.!,])", r"\1", cleanedText)

  # Lemmatization (optional).
  if (lemmatize):
    lemmatizer = WordNetLemmatizer()
    cleanedText = " ".join([lemmatizer.lemmatize(word) for word in cleanedText.split()])

  # Remove stop words (optional).
  if (removeStopwords):
    stopWords = set(stopwords.words("english"))
    cleanedText = " ".join([word for word in cleanedText.split() if word.lower() not in stopWords])

  # Replace multiple spaces with a single space.
  if (normalizeWhitespace):
    cleanedText = " ".join(cleanedText.split())

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

  cleaned = CleanText(
    sampleText,
    removeNonAscii=True,  # Remove non-ASCII characters.
    lowercase=True,  # Convert text to lowercase.
    removeSpecialChars=True,  # Remove special characters and punctuation.
    normalizeWhitespace=True,  # Normalize whitespace.
    handleContractions=True,  # Expand contractions.
    lemmatize=False,  # Do not lemmatize words.
    removeStopwords=False,  # Do not remove stop words.
  )

  print("Original Text:\n", sampleText)
  print("\nCleaned Text:\n", cleaned)
