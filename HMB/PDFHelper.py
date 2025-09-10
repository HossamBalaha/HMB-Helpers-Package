# Import the required libraries.
import fitz, re


# Define a function to read the full content of a PDF file.
def ReadFullPDF(filePath):
  r'''
  Reads the full content of a PDF file and returns it as a string.

  Parameters:
    filePath (str): The path to the PDF file to be read.

  Returns:
    str: The full text content of the PDF file.
  '''

  # Open the PDF file using fitz (PyMuPDF).
  doc = fitz.open(filePath)

  # Initialize an empty string to hold the text.
  fullText = ""

  # Iterate through each page in the document.
  for page in doc:
    # Extract text from the current page and append it to fullText.
    fullText += page.get_text()

  # Close the document to free resources.
  doc.close()

  # Clean up the report text by removing newlines and extra spaces.
  fullText = fullText.replace("\n", " ").strip()
  # Replace multiple spaces with a single space using regular expressions.
  fullText = re.sub(r"\s+", " ", fullText)

  # Return the cleaned full text content of the PDF file.
  return fullText


# Define a function to read the content of a specific page in a PDF file.
def ReadPDFPage(filePath, pageNum):
  r'''
  Reads the content of a specific page in a PDF file.

  Parameters:
    filePath (str): The path to the PDF file.
    pageNum (int): The page number to read (0-based).

  Returns:
    str: The text content of the specified page.
  '''

  doc = fitz.open(filePath)
  # Check if the page number is valid.
  if (pageNum < 0 or pageNum >= doc.page_count):
    doc.close()
    raise IndexError("Page number out of range.")
  # Extract text from the specified page.
  text = doc.load_page(pageNum).get_text()
  doc.close()
  # Return the cleaned text.
  return text.replace("\n", " ").strip()


# Define a function to get the total number of pages in a PDF file.
def GetPDFPageCount(filePath):
  r'''
  Returns the total number of pages in a PDF file.

  Parameters:
    filePath (str): The path to the PDF file.

  Returns:
    int: The number of pages in the PDF.
  '''

  doc = fitz.open(filePath)
  count = doc.page_count
  doc.close()
  return count


# Define a function to extract all text matching a regex pattern from a PDF file.
def ExtractPDFTextByRegex(filePath, pattern):
  r'''
  Extracts all text matching a regex pattern from a PDF file.

  Parameters:
    filePath (str): The path to the PDF file.
    pattern (str): The regex pattern to search for.

  Returns:
    list: A list of matched strings.
  '''

  text = ReadFullPDF(filePath)
  return re.findall(pattern, text)


# Define a function to extract metadata from a PDF file.
def ExtractPDFMetadata(filePath):
  r'''
  Extracts metadata from a PDF file.

  Parameters:
    filePath (str): The path to the PDF file.

  Returns:
    dict: Metadata dictionary.
  '''

  doc = fitz.open(filePath)
  metadata = doc.metadata
  doc.close()
  return metadata


# Define a function to extract images from all pages of a PDF file.
def ExtractPDFImages(filePath):
  r'''
  Extracts images from all pages of a PDF file.

  Parameters:
    filePath (str): The path to the PDF file.

  Returns:
    list: A list of image bytes objects.
  '''

  doc = fitz.open(filePath)
  images = []
  # Iterate through each page in the document.
  for pageIndex in range(doc.page_count):
    page = doc.load_page(pageIndex)
    # Extract images from the current page.
    for img in page.get_images(full=True):
      xref = img[0]
      baseImage = doc.extract_image(xref)
      images.append(baseImage["image"])
  doc.close()
  return images


# Define a function to extract tables from a PDF file using tabula-py.
def ExtractPDFTables(filePath, pages="all"):
  r'''
  Extracts tables from a PDF file using tabula-py.

  Installation:
    Use `pip install tabula-py` to install the library.
    Use `pip install jpype1` to install the JPype1 dependency.
    Also requires Java to be installed on the system. Install it from https://www.oracle.com/java/technologies/downloads/

  Parameters:
    filePath (str): The path to the PDF file.
    pages (str or int): Pages to extract tables from.

  Returns:
    list: List of pandas DataFrames for each table.
  '''

  try:
    import tabula
    tables = tabula.read_pdf(filePath, pages=pages, multiple_tables=True)
    return tables
  except Exception as e:
    print("Error extracting tables:", e)
    return []


# Define a function to save a range of pages from a PDF file as a new PDF.
def SplitPDF(filePath, startPage, endPage, outputPath):
  r'''
  Saves a range of pages from a PDF file as a new PDF.

  Parameters:
    filePath (str): The path to the PDF file.
    startPage (int): The starting page (0-based).
    endPage (int): The ending page (exclusive, 0-based).
    outputPath (str): The path to save the new PDF.
  '''

  doc = fitz.open(filePath)
  newDoc = fitz.open()
  # Insert the specified range of pages into the new PDF.
  for i in range(startPage, min(endPage, doc.page_count)):
    newDoc.insert_pdf(doc, from_page=i, to_page=i)
  newDoc.save(outputPath)
  newDoc.close()
  doc.close()


# Define a function to merge multiple PDF files into a single PDF.
def MergePDFs(pdfPaths, outputPath):
  r'''
  Merges multiple PDF files into a single PDF.

  Parameters:
    pdfPaths (list): List of PDF file paths to merge.
    outputPath (str): Path to save the merged PDF.
  '''

  merged = fitz.open()
  # Iterate through each PDF file and insert its pages.
  for path in pdfPaths:
    doc = fitz.open(path)
    merged.insert_pdf(doc)
    doc.close()
  merged.save(outputPath)
  merged.close()


# Define a function to delete specified pages from a PDF and save the result.
def DeletePDFPages(filePath, pagesToDelete, outputPath):
  r'''
  Deletes specified pages from a PDF and saves the result.

  Parameters:
    filePath (str): Path to the PDF file.
    pagesToDelete (list): List of 0-based page indices to delete.
    outputPath (str): Path to save the new PDF.
  '''

  doc = fitz.open(filePath)
  doc.delete_pages(pagesToDelete)
  doc.save(outputPath)
  doc.close()


# Define a function to rotate specified pages in a PDF by a given angle.
def RotatePDFPages(filePath, rotation, outputPath, pages=None):
  r'''
  Rotates specified pages in a PDF by a given angle.

  Parameters:
    filePath (str): Path to the PDF file.
    rotation (int): Angle to rotate (90, 180, 270).
    outputPath (str): Path to save the rotated PDF.
    pages (list or None): List of 0-based page indices to rotate. If None, rotate all.
  '''

  doc = fitz.open(filePath)
  # If no pages specified, rotate all pages.
  if (pages is None):
    pages = list(range(doc.page_count))
  for i in pages:
    page = doc.load_page(i)
    page.set_rotation(rotation)
  doc.save(outputPath)
  doc.close()


# Define a function to extract annotations/comments from all pages of a PDF file.
def ExtractPDFAnnotations(filePath):
  r'''
  Extracts annotations/comments from all pages of a PDF file.

  Parameters:
    filePath (str): Path to the PDF file.

  Returns:
    list: List of annotation texts.
  '''

  doc = fitz.open(filePath)
  annotations = []
  # Iterate through each page and extract annotations.
  for page in doc:
    for annot in page.annots() or []:
      if (annot.info and "content" in annot.info):
        annotations.append(annot.info["content"])
  doc.close()
  return annotations


# Define a function to check if the PDF contains the specified text.
def PDFContainsText(
  filePath,
  searchText,
  caseSensitive=False,
  useRegex=False,
  returnPositions=False
):
  r'''
  Checks if the PDF contains the specified text, with options for case sensitivity, regex, and returning positions.

  Parameters:
    filePath (str): Path to the PDF file.
    searchText (str): Text or regex pattern to search for.
    caseSensitive (bool): If True, search is case sensitive.
    useRegex (bool): If True, searchText is treated as a regex pattern.
    returnPositions (bool): If True, returns list of (pageNum, position) for all occurrences.

  Returns:
    bool or list: True/False if found (default), or list of (pageNum, position) if returnPositions is True.
  '''

  flags = 0 if (caseSensitive) else re.IGNORECASE
  doc = fitz.open(filePath)
  found = False
  positions = []
  # Iterate through each page in the document.
  for pageNum in range(doc.page_count):
    page = doc.load_page(pageNum)
    text = page.get_text()

    # Use regex if specified.
    if (useRegex):
      matches = list(re.finditer(searchText, text, flags))
      if (matches):
        found = True
        if (returnPositions):
          for m in matches:
            positions.append((pageNum, m.start()))
    else:
      search = searchText if (caseSensitive) else searchText.lower()
      pageText = text if (caseSensitive) else text.lower()
      idx = 0
      while (True):
        idx = pageText.find(search, idx)
        if (idx == -1):
          break
        found = True
        if (returnPositions):
          positions.append({
            "page" : pageNum,
            "index": idx,
          })
        idx += len(search)
  doc.close()
  if (returnPositions):
    return positions
  return found


# Define a function to add a bookmark to a specific page in the PDF.
def AddPDFBookmark(filePath, pageNum, title, outputPath):
  r'''
  Adds a bookmark to a specific page in the PDF.

  Parameters:
    filePath (str): Path to the PDF file.
    pageNum (int): Page number (0-based).
    title (str): Bookmark title.
    outputPath (str): Path to save the new PDF.
  '''

  doc = fitz.open(filePath)
  doc.set_toc(doc.get_toc() + [[1, title, pageNum]])
  doc.save(outputPath)
  doc.close()


# Define a function to extract all hyperlinks from the PDF.
def ExtractPDFLinks(filePath):
  r'''
  Extracts all hyperlinks from the PDF.

  Parameters:
    filePath (str): Path to the PDF file.

  Returns:
    list: List of URLs.
  '''

  doc = fitz.open(filePath)
  links = []
  # Iterate through each page and extract links.
  for page in doc:
    for lnk in page.get_links():
      if ("uri" in lnk):
        links.append(lnk["uri"])
  doc.close()
  return links


# Define a function to highlight all occurrences of a text string in the PDF.
def HighlightPDFText(filePath, searchText, outputPath, color=(1, 1, 0)):
  r'''
  Highlights all occurrences of a text string in the PDF.

  Parameters:
    filePath (str): Path to the PDF file.
    searchText (str): Text to highlight.
    outputPath (str): Path to save the highlighted PDF.
    color (tuple): RGB color tuple for highlight (default yellow).
  '''

  doc = fitz.open(filePath)
  # Iterate through each page and search for the text.
  for page in doc:
    areas = page.search_for(searchText)
    for area in areas:
      annot = page.add_highlight_annot(area)
      annot.set_colors(stroke=color)
      annot.update()
  doc.save(outputPath)
  doc.close()


# Define a function to encrypt the PDF with a password.
def EncryptPDF(filePath, password, outputPath):
  r'''
  Encrypts the PDF with a password.

  Parameters:
    filePath (str): Path to the PDF file.
    password (str): Password to set.
    outputPath (str): Path to save the encrypted PDF.
  '''

  doc = fitz.open(filePath)
  doc.save(outputPath, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password)
  doc.close()


# Define a function to remove password protection from a PDF.
def DecryptPDF(filePath, password, outputPath):
  r'''
  Removes password protection from a PDF.

  Parameters:
    filePath (str): Path to the encrypted PDF.
    password (str): Password for decryption.
    outputPath (str): Path to save the decrypted PDF.
  '''

  doc = fitz.open(filePath)
  # Authenticate using the provided password.
  if (not doc.authenticate(password)):
    doc.close()
    raise ValueError("Incorrect password for PDF decryption.")
  doc.save(outputPath)
  doc.close()


# Define a function to list all fonts used in the PDF.
def ExtractPDFFonts(filePath):
  r'''
  Lists all fonts used in the PDF.

  Parameters:
    filePath (str): Path to the PDF file.

  Returns:
    set: Set of font names.
  '''

  doc = fitz.open(filePath)
  fonts = set()
  # Iterate through each page and extract fonts.
  for page in doc:
    for font in page.get_fonts():
      fonts.add(font[3])
  doc.close()
  return fonts


# Main block for example usage.
if __name__ == "__main__":
  # Example PDF file path.
  pdfPath = r"Example.pdf"

  # Read the full content of the PDF.
  fullContent = ReadFullPDF(pdfPath)
  # Read the content of the first page.
  page0Content = ReadPDFPage(pdfPath, 0)
  # Get the total number of pages.
  pageCount = GetPDFPageCount(pdfPath)
  # Extract metadata from the PDF.
  metadata = ExtractPDFMetadata(pdfPath)
  # Extract email addresses using regex.
  emails = ExtractPDFTextByRegex(pdfPath, r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
  # Extract images from the PDF.
  images = ExtractPDFImages(pdfPath)
  # Extract tables from the PDF.
  tables = ExtractPDFTables(pdfPath)

  # Print the first 100 characters of the full text content.
  print("Full Text Content (first 100 chars):", fullContent[:100])
  # Print the length of the full text.
  print("Length of Full Text:", len(fullContent))
  # Print the first 100 characters of the first page content.
  print("Page 0 Content (first 100 chars):", page0Content[:100])
  # Print the length of the first page text.
  print("Length of Page 0 Text:", len(page0Content))
  # Print the total number of pages.
  print("Total Number of Pages:", pageCount)
  # Print the metadata.
  print("Metadata:", metadata)
  # Print the extracted email addresses.
  print("Extracted Emails:", emails)
  # Print the number of extracted images.
  print("Number of Extracted Images:", len(images))
  # Print the number of extracted tables.
  print("Number of Extracted Tables:", len(tables))
  # If tables are found, print a preview of the first table.
  if (len(tables) > 0):
    print("First Table Preview:\n", tables[0].head())

  # Split the first 2 pages into a new PDF.
  SplitPDF(pdfPath, 0, 2, r"ExtractedPages.pdf")
  # Merge multiple PDFs into one.
  MergePDFs(
    [pdfPath, r"ExtractedPages.pdf"],
    r"Merged.pdf"
  )
  # Delete the second page (index 1) from the PDF.
  DeletePDFPages(pdfPath, [1], r"DeletedPage.pdf")
  # Rotate the first page (index 0) by 90 degrees.
  RotatePDFPages(pdfPath, 90, r"Rotated.pdf", pages=[0])
  # Extract annotations from the PDF.
  annotations = ExtractPDFAnnotations(pdfPath)
  print("Extracted Annotations:", annotations)
  # Check if the PDF contains a specific text.
  containsText = PDFContainsText(pdfPath, "Lorem ipsum", caseSensitive=False)
  print("Contains 'Lorem ipsum':", containsText)
  # Get the positions of the word "Lorem" in the PDF.
  positions = PDFContainsText(pdfPath, "Lorem", returnPositions=True)
  print("First Found Positions of 'Lorem':", positions[:5])

  # Add a bookmark to the first page.
  AddPDFBookmark(pdfPath, 0, "Start of Document", r"Bookmarked.pdf")
  # Extract all hyperlinks from the PDF.
  links = ExtractPDFLinks(pdfPath)
  print("Extracted Links:", links)
  # Highlight the word "Lorem" in the PDF.
  HighlightPDFText(pdfPath, "Lorem", r"Highlighted.pdf")
  # Encrypt the PDF with a password.
  EncryptPDF(pdfPath, "securepassword", r"Encrypted.pdf")
  # Decrypt the previously encrypted PDF.
  DecryptPDF(
    r"Encrypted.pdf",
    "securepassword",
    r"Decrypted.pdf"
  )
  # Extract all fonts used in the PDF.
  fonts = ExtractPDFFonts(pdfPath)
  print("Extracted Fonts:", fonts)
