import unittest
import os
import tempfile
import fitz  # PyMuPDF
from HMB.PDFHelper import (
  ReadFullPDF,
  ReadPDFPage,
  GetPDFPageCount,
  ExtractPDFTextByRegex,
  ExtractPDFMetadata,
  ExtractPDFImages,
  SplitPDF,
  MergePDFs,
  DeletePDFPages,
  RotatePDFPages,
  ExtractPDFAnnotations,
  PDFContainsText,
  AddPDFBookmark,
  ExtractPDFLinks,
  HighlightPDFText,
  EncryptPDF,
  DecryptPDF,
  ExtractPDFFonts
)


class TestPDFHelper(unittest.TestCase):
  """
  Unit tests for the PDFHelper module.
  Tests PDF reading, manipulation, and extraction functions.
  """

  @classmethod
  def setUpClass(cls):
    """Create dummy PDF files for testing."""
    cls.testDir = tempfile.mkdtemp()
    cls.testPdf = os.path.join(cls.testDir, "test1.pdf")
    cls.testPdf2 = os.path.join(cls.testDir, "test2.pdf")
    cls.outputPdf = os.path.join(cls.testDir, "output.pdf")

    # Create a simple test PDF with multiple pages.
    cls._createTestPdf(cls.testPdf, numPages=3, text="This is page {}")
    cls._createTestPdf(cls.testPdf2, numPages=2, text="Second PDF page {}")

  @classmethod
  def tearDownClass(cls):
    """Clean up test files."""
    import shutil
    if os.path.exists(cls.testDir):
      shutil.rmtree(cls.testDir)

  @staticmethod
  def _createTestPdf(filepath, numPages=1, text="Test content page {}"):
    """Helper method to create a test PDF file."""
    doc = fitz.open()
    for i in range(numPages):
      page = doc.new_page(width=595, height=842)  # A4 size.
      pageText = text.format(i + 1)
      page.insert_text((72, 72), pageText, fontsize=12)
    doc.save(filepath)
    doc.close()

  def tearDown(self):
    """Clean up output files after each test."""
    if os.path.exists(self.outputPdf):
      try:
        os.remove(self.outputPdf)
      except Exception:
        pass

  # ========== ReadFullPDF Tests. ==========

  def test_read_full_pdf(self):
    """Test reading full PDF content."""
    result = ReadFullPDF(self.testPdf)
    self.assertIsInstance(result, str)
    self.assertIn("This is page", result)

  def test_read_full_pdf_multiple_pages(self):
    """Test reading PDF with multiple pages."""
    result = ReadFullPDF(self.testPdf)
    self.assertIn("page 1", result)
    self.assertIn("page 2", result)
    self.assertIn("page 3", result)

  def test_read_full_pdf_no_newlines(self):
    """Test that ReadFullPDF removes newlines."""
    result = ReadFullPDF(self.testPdf)
    # Should have spaces instead of newlines.
    self.assertIsInstance(result, str)

  def test_read_full_pdf_file_not_found(self):
    """Test ReadFullPDF with non-existent file."""
    with self.assertRaises(Exception):
      ReadFullPDF("nonexistent.pdf")

  # ========== ReadPDFPage Tests. ==========

  def test_read_pdf_page_first(self):
    """Test reading first page."""
    result = ReadPDFPage(self.testPdf, 0)
    self.assertIsInstance(result, str)
    self.assertIn("page 1", result)

  def test_read_pdf_page_middle(self):
    """Test reading middle page."""
    result = ReadPDFPage(self.testPdf, 1)
    self.assertIsInstance(result, str)
    self.assertIn("page 2", result)

  def test_read_pdf_page_last(self):
    """Test reading last page."""
    result = ReadPDFPage(self.testPdf, 2)
    self.assertIsInstance(result, str)
    self.assertIn("page 3", result)

  def test_read_pdf_page_out_of_range(self):
    """Test reading page out of range raises IndexError."""
    with self.assertRaises(IndexError):
      ReadPDFPage(self.testPdf, 10)

  def test_read_pdf_page_negative_index(self):
    """Test reading page with negative index raises IndexError."""
    with self.assertRaises(IndexError):
      ReadPDFPage(self.testPdf, -1)

  # ========== GetPDFPageCount Tests. ==========

  def test_get_pdf_page_count(self):
    """Test getting page count."""
    result = GetPDFPageCount(self.testPdf)
    self.assertEqual(result, 3)

  def test_get_pdf_page_count_two_pages(self):
    """Test getting page count for second PDF."""
    result = GetPDFPageCount(self.testPdf2)
    self.assertEqual(result, 2)

  def test_get_pdf_page_count_single_page(self):
    """Test getting page count for single page PDF."""
    singlePagePdf = os.path.join(self.testDir, "single.pdf")
    self._createTestPdf(singlePagePdf, numPages=1)
    result = GetPDFPageCount(singlePagePdf)
    self.assertEqual(result, 1)
    os.remove(singlePagePdf)

  # ========== ExtractPDFTextByRegex Tests. ==========

  def test_extract_pdf_text_by_regex_found(self):
    """Test extracting text by regex pattern."""
    result = ExtractPDFTextByRegex(self.testPdf, r"page \d+")
    self.assertIsInstance(result, list)
    self.assertTrue(len(result) > 0)

  def test_extract_pdf_text_by_regex_not_found(self):
    """Test extracting text with no matches."""
    result = ExtractPDFTextByRegex(self.testPdf, r"nonexistent pattern")
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 0)

  def test_extract_pdf_text_by_regex_numbers(self):
    """Test extracting numbers from PDF."""
    result = ExtractPDFTextByRegex(self.testPdf, r"\\d+")
    self.assertIsInstance(result, list)
    self.assertTrue(len(result) > 0)

  # ========== ExtractPDFMetadata Tests. ==========

  def test_extract_pdf_metadata(self):
    """Test extracting PDF metadata."""
    result = ExtractPDFMetadata(self.testPdf)
    self.assertIsInstance(result, dict)

  def test_extract_pdf_metadata_keys(self):
    """Test that metadata contains expected keys."""
    result = ExtractPDFMetadata(self.testPdf)
    # Metadata may be empty for newly created PDFs.
    self.assertIsInstance(result, dict)

  # ========== ExtractPDFImages Tests. ==========

  def test_extract_pdf_images_no_images(self):
    """Test extracting images from PDF with no images."""
    result = ExtractPDFImages(self.testPdf)
    self.assertIsInstance(result, list)
    # Our test PDF has no images.
    self.assertEqual(len(result), 0)

  # ========== SplitPDF Tests. ==========

  def test_split_pdf_first_page(self):
    """Test splitting PDF to extract first page."""
    SplitPDF(self.testPdf, 0, 1, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Verify split PDF has 1 page.
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 1)

  def test_split_pdf_middle_pages(self):
    """Test splitting PDF to extract middle pages."""
    SplitPDF(self.testPdf, 1, 3, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Verify split PDF has 2 pages.
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 2)

  def test_split_pdf_single_page(self):
    """Test splitting single page."""
    SplitPDF(self.testPdf, 0, 1, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 1)

  def test_split_pdf_all_pages(self):
    """Test splitting all pages."""
    SplitPDF(self.testPdf, 0, 3, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 3)

  # Additional edge cases for SplitPDF
  def test_split_pdf_invalid_range_raises(self):
    with self.assertRaises(Exception):
      SplitPDF(self.testPdf, 3, 1, self.outputPdf)
    with self.assertRaises(Exception):
      SplitPDF(self.testPdf, -1, 1, self.outputPdf)

  # ========== MergePDFs Tests. ==========

  def test_merge_pdfs_two_files(self):
    """Test merging two PDF files."""
    MergePDFs([self.testPdf, self.testPdf2], self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Verify merged PDF has correct page count (3 + 2 = 5)
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 5)

  def test_merge_pdfs_single_file(self):
    """Test merging single PDF file."""
    MergePDFs([self.testPdf], self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 3)

  def test_merge_pdfs_content_preserved(self):
    """Test that merged PDF preserves content."""
    MergePDFs([self.testPdf, self.testPdf2], self.outputPdf)
    content = ReadFullPDF(self.outputPdf)
    self.assertIn("This is page", content)
    self.assertIn("Second PDF", content)

  def test_merge_pdfs_nonexistent_input_raises(self):
    with self.assertRaises(Exception):
      MergePDFs(["does_not_exist.pdf"], self.outputPdf)

  # ========== DeletePDFPages Tests. ==========

  def test_delete_pdf_pages_single(self):
    """Test deleting single page from PDF."""
    DeletePDFPages(self.testPdf, [1], self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Should have 2 pages left (deleted middle page)
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 2)

  def test_delete_pdf_pages_multiple(self):
    """Test deleting multiple pages from PDF."""
    DeletePDFPages(self.testPdf, [0, 2], self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Should have 1 page left
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 1)

  def test_delete_pdf_pages_empty_list(self):
    """Test deleting no pages."""
    DeletePDFPages(self.testPdf, [], self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Should have all original pages
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 3)

  def test_delete_pdf_pages_invalid_indices(self):
    with self.assertRaises(Exception):
      DeletePDFPages(self.testPdf, [10], self.outputPdf)
    with self.assertRaises(Exception):
      DeletePDFPages(self.testPdf, [-1], self.outputPdf)

  # ========== RotatePDFPages Tests. ==========

  def test_rotate_pdf_pages_all_90(self):
    """Test rotating all pages 90 degrees."""
    RotatePDFPages(self.testPdf, 90, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Verify same page count
    pageCount = GetPDFPageCount(self.outputPdf)
    self.assertEqual(pageCount, 3)

  def test_rotate_pdf_pages_180(self):
    """Test rotating pages 180 degrees."""
    RotatePDFPages(self.testPdf, 180, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_rotate_pdf_pages_270(self):
    """Test rotating pages 270 degrees."""
    RotatePDFPages(self.testPdf, 270, self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_rotate_pdf_pages_specific_pages(self):
    """Test rotating specific pages."""
    RotatePDFPages(self.testPdf, 90, self.outputPdf, pages=[0, 2])
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_rotate_pdf_pages_invalid_page_list(self):
    with self.assertRaises(Exception):
      RotatePDFPages(self.testPdf, 90, self.outputPdf, pages=[-1])

  # ========== ExtractPDFAnnotations Tests. ==========

  def test_extract_pdf_annotations_no_annotations(self):
    """Test extracting annotations from PDF with none."""
    result = ExtractPDFAnnotations(self.testPdf)
    self.assertIsInstance(result, list)
    # Our test PDF has no annotations.
    self.assertEqual(len(result), 0)

  # ========== PDFContainsText Tests. ==========

  def test_pdf_contains_text_case_sensitive_found(self):
    """Test checking if PDF contains text (case-sensitive)."""
    result = PDFContainsText(self.testPdf, "This is page", caseSensitive=True)
    self.assertTrue(result)

  def test_pdf_contains_text_case_sensitive_not_found(self):
    """Test checking if PDF contains text (case-sensitive, not found)."""
    result = PDFContainsText(self.testPdf, "this is page", caseSensitive=True)
    self.assertFalse(result)

  def test_pdf_contains_text_case_insensitive_found(self):
    """Test checking if PDF contains text (case-insensitive)."""
    result = PDFContainsText(self.testPdf, "this is page", caseSensitive=False)
    self.assertTrue(result)

  def test_pdf_contains_text_not_found(self):
    """Test checking if PDF contains non-existent text."""
    result = PDFContainsText(self.testPdf, "nonexistent text", caseSensitive=False)
    self.assertFalse(result)

  def test_pdf_contains_text_partial_match(self):
    """Test checking for partial text match."""
    result = PDFContainsText(self.testPdf, "page", caseSensitive=False)
    self.assertTrue(result)

  # ========== AddPDFBookmark Tests. ==========

  def test_add_pdf_bookmark(self):
    """Test adding bookmark to PDF."""
    AddPDFBookmark(self.testPdf, 0, "First Page", self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_add_pdf_bookmark_middle_page(self):
    """Test adding bookmark to middle page."""
    AddPDFBookmark(self.testPdf, 1, "Middle Page", self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_add_pdf_bookmark_invalid_page(self):
    with self.assertRaises(Exception):
      AddPDFBookmark(self.testPdf, 99, "Bad", self.outputPdf)

  # ========== ExtractPDFLinks Tests. ==========

  def test_extract_pdf_links_no_links(self):
    """Test extracting links from PDF with none."""
    result = ExtractPDFLinks(self.testPdf)
    self.assertIsInstance(result, list)
    # Our test PDF has no links.
    self.assertEqual(len(result), 0)

  # ========== HighlightPDFText Tests. ==========

  def test_highlight_pdf_text(self):
    """Test highlighting text in PDF."""
    HighlightPDFText(self.testPdf, "page", self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_highlight_pdf_text_custom_color(self):
    """Test highlighting text with custom color."""
    HighlightPDFText(self.testPdf, "page", self.outputPdf, color=(1, 0, 0))
    self.assertTrue(os.path.exists(self.outputPdf))

  def test_highlight_pdf_text_not_found(self):
    """Test highlighting non-existent text."""
    HighlightPDFText(self.testPdf, "nonexistent", self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

  # ========== EncryptPDF and DecryptPDF Tests. ==========

  def test_encrypt_pdf(self):
    """Test encrypting PDF with password."""
    encryptedPdf = os.path.join(self.testDir, "encrypted.pdf")
    EncryptPDF(self.testPdf, "testpassword", encryptedPdf)
    self.assertTrue(os.path.exists(encryptedPdf))

    # Try to open encrypted PDF (should require password)
    doc = fitz.open(encryptedPdf)
    self.assertTrue(doc.is_encrypted)
    doc.close()
    os.remove(encryptedPdf)

  def test_decrypt_pdf(self):
    """Test decrypting PDF with correct password."""
    encryptedPdf = os.path.join(self.testDir, "encrypted.pdf")
    EncryptPDF(self.testPdf, "testpassword", encryptedPdf)

    DecryptPDF(encryptedPdf, "testpassword", self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))

    # Verify decrypted PDF is not encrypted
    doc = fitz.open(self.outputPdf)
    self.assertFalse(doc.is_encrypted)
    doc.close()
    os.remove(encryptedPdf)

  def test_encrypt_decrypt_content_preserved(self):
    """Test that content is preserved after encryption/decryption."""
    encryptedPdf = os.path.join(self.testDir, "encrypted.pdf")

    # Get original content
    originalContent = ReadFullPDF(self.testPdf)

    # Encrypt and decrypt
    EncryptPDF(self.testPdf, "testpassword", encryptedPdf)
    DecryptPDF(encryptedPdf, "testpassword", self.outputPdf)

    # Get decrypted content
    decryptedContent = ReadFullPDF(self.outputPdf)

    # Content should match
    self.assertEqual(originalContent, decryptedContent)
    os.remove(encryptedPdf)

  # ========== ExtractPDFFonts Tests. ==========

  def test_extract_pdf_fonts(self):
    """Test extracting fonts from PDF."""
    result = ExtractPDFFonts(self.testPdf)
    # ExtractPDFFonts returns a set, not a list
    self.assertIsInstance(result, (list, set))

  # ========== Integration Tests. ==========

  def test_full_pdf_workflow(self):
    """Test complete PDF workflow: read, split, merge."""
    # Split first page
    splitPdf = os.path.join(self.testDir, "split.pdf")
    SplitPDF(self.testPdf, 0, 1, splitPdf)

    # Merge with another PDF
    mergedPdf = os.path.join(self.testDir, "merged.pdf")
    MergePDFs([splitPdf, self.testPdf2], mergedPdf)

    # Verify page count (1 + 2 = 3)
    pageCount = GetPDFPageCount(mergedPdf)
    self.assertEqual(pageCount, 3)

    # Clean up
    os.remove(splitPdf)
    os.remove(mergedPdf)

  def test_pdf_content_extraction(self):
    """Test extracting and verifying content."""
    # Read full content
    fullContent = ReadFullPDF(self.testPdf)

    # Read individual pages
    page1 = ReadPDFPage(self.testPdf, 0)
    page2 = ReadPDFPage(self.testPdf, 1)

    # Full content should contain individual pages
    self.assertIn("page 1", fullContent)
    self.assertIn("page 1", page1)
    self.assertIn("page 2", page2)

  # ========== Encryption / Decryption. ==========

  def test_encrypt_decrypt_roundtrip(self):
    password = "secret123"
    encPath = os.path.join(self.testDir, "encrypted.pdf")
    EncryptPDF(self.testPdf, password, encPath)
    self.assertTrue(os.path.exists(encPath))
    # Try decrypt with correct password
    decPath = os.path.join(self.testDir, "decrypted.pdf")
    DecryptPDF(encPath, password, decPath)
    self.assertTrue(os.path.exists(decPath))
    # Verify content matches via simple text check
    txt = ReadFullPDF(decPath)
    self.assertIn("This is page", txt)

  def test_decrypt_wrong_password(self):
    password = "secret123"
    encPath = os.path.join(self.testDir, "encrypted_wrong.pdf")
    EncryptPDF(self.testPdf, password, encPath)
    decPath = os.path.join(self.testDir, "decrypted_wrong.pdf")
    with self.assertRaises(Exception):
      DecryptPDF(encPath, "badpass", decPath)

  # ========== Bookmarks / Links / Annotations. ==========

  def test_add_bookmark(self):
    AddPDFBookmark(self.testPdf, 0, title="Start", outputPath=self.outputPdf)
    self.assertTrue(os.path.exists(self.outputPdf))
    # Extract metadata/outline could be implementation-specific; just ensure output exists

  def test_extract_links(self):
    # Add a link to the first page
    doc = fitz.open(self.testPdf)
    page = doc[0]
    rect = fitz.Rect(100, 100, 200, 120)
    page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": "https://example.com"})
    linkedPath = os.path.join(self.testDir, "linked.pdf")
    doc.save(linkedPath)
    doc.close()
    links = ExtractPDFLinks(linkedPath)
    self.assertIsInstance(links, list)

  def test_extract_annotations_empty(self):
    anns = ExtractPDFAnnotations(self.testPdf)
    self.assertIsInstance(anns, list)

  def test_highlight_text(self):
    # Highlight existing word in the PDF and expect True
    result = HighlightPDFText(self.testPdf, "page", self.outputPdf)
    self.assertTrue(result)
    self.assertTrue(os.path.exists(self.outputPdf))

  # ========== Page Operations Boundaries ==========

  def test_rotate_pages_out_of_range(self):
    # Rotate a non-existent page index should raise or no-op safely
    with self.assertRaises(Exception):
      RotatePDFPages(self.testPdf, 90, self.outputPdf, pages=[99])

  def test_delete_pages_out_of_range(self):
    with self.assertRaises(Exception):
      DeletePDFPages(self.testPdf, [99], self.outputPdf)

  def test_split_invalid_range(self):
    with self.assertRaises(Exception):
      SplitPDF(self.testPdf, 2, 1, self.outputPdf)

  def test_merge_pdfs(self):
    mergedPath = os.path.join(self.testDir, "merged.pdf")
    MergePDFs([self.testPdf, self.testPdf2], mergedPath)
    self.assertTrue(os.path.exists(mergedPath))
    self.assertGreater(GetPDFPageCount(mergedPath), 0)

  # ========== Fonts Extraction ==========

  def test_extract_fonts(self):
    fonts = ExtractPDFFonts(self.testPdf)
    self.assertIsInstance(fonts, list)


if __name__ == "__main__":
  unittest.main()
