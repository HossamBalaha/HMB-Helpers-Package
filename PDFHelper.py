'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 31th, 2025
# Last Modification Date: Jul 31th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import the required libraries.
import fitz, re


def ReadFullPDF(filePath):
  """
  Reads the full content of a PDF file and returns it as a string.
  Parameters:
    filePath (str): The path to the PDF file to be read.
  Returns:
    str: The full text content of the PDF file.
  """

  # Open the PDF file using fitz (PyMuPDF).
  doc = fitz.open(filePath)

  # Initialize an empty string to hold the text.
  fullText = ""

  # Iterate through each page in the document.
  for page in doc:
    # Extract text from the current page and append it to fullText.
    fullText += page.get_text()

  # Close the document.
  doc.close()

  # Clean up the report text by removing newlines and extra spaces.
  fullText = fullText.replace("\n", " ").strip()
  fullText = re.sub(r"\s+", " ", fullText)

  return fullText
