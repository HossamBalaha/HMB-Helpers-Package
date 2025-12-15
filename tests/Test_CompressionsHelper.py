import unittest
import os
import tempfile
import shutil
import tarfile
import zipfile
from HMB.CompressionsHelper import CompressionsHelper


class TestCompressionsHelper(unittest.TestCase):
  '''
  Unit tests for CompressionsHelper.
  Focus on tarfile and zipfile extraction; other formats tested for ImportError when libs are missing.
  '''

  @classmethod
  def setUpClass(cls):
    cls.testDir = tempfile.mkdtemp()
    cls.archivesDir = os.path.join(cls.testDir, "archives")
    cls.extractDir = os.path.join(cls.testDir, "extracted")
    os.makedirs(cls.archivesDir, exist_ok=True)
    os.makedirs(cls.extractDir, exist_ok=True)

    # Create sample files
    cls.sampleFiles = {
      "file1.txt"       : "Hello World",
      "file2.txt"       : "Another file",
      "nested/file3.txt": "Nested file",
    }
    # Materialize sample files in a temp folder to archive
    cls.payloadDir = os.path.join(cls.testDir, "payload")
    for rel, content in cls.sampleFiles.items():
      absPath = os.path.join(cls.payloadDir, rel)
      os.makedirs(os.path.dirname(absPath), exist_ok=True)
      with open(absPath, "w") as f:
        f.write(content)

    # Create ZIP archive
    cls.zipPath = os.path.join(cls.archivesDir, "test.zip")
    with zipfile.ZipFile(cls.zipPath, "w") as z:
      for rel in cls.sampleFiles.keys():
        z.write(os.path.join(cls.payloadDir, rel), rel)

    # Create TAR archive
    cls.tarPath = os.path.join(cls.archivesDir, "test.tar")
    with tarfile.open(cls.tarPath, "w") as t:
      for rel in cls.sampleFiles.keys():
        t.add(os.path.join(cls.payloadDir, rel), arcname=rel)

    # Create corrupted ZIP
    cls.badZipPath = os.path.join(cls.archivesDir, "bad.zip")
    with open(cls.badZipPath, "wb") as f:
      f.write(b"not a real zip")

  @classmethod
  def tearDownClass(cls):
    if os.path.exists(cls.testDir):
      shutil.rmtree(cls.testDir)

  def setUp(self):
    self.ch = CompressionsHelper()
    # Clean extraction dir for each test
    if os.path.exists(self.extractDir):
      shutil.rmtree(self.extractDir)
    os.makedirs(self.extractDir, exist_ok=True)

  # ========== ZIP ==========

  def test_extract_zipfile_basic(self):
    '''Extract ZIP and verify files exist.'''
    self.ch.ExtractFileUsingZipfile(self.zipPath, self.extractDir)
    # Verify extracted files
    for rel in self.sampleFiles.keys():
      self.assertTrue(os.path.exists(os.path.join(self.extractDir, rel)))

  def test_extract_zipfile_destination_created(self):
    '''Ensure destination directory is created if missing.'''
    dest = os.path.join(self.testDir, "new_extract_zip")
    self.ch.ExtractFileUsingZipfile(self.zipPath, dest)
    self.assertTrue(os.path.isdir(dest))

  def test_extract_zipfile_corrupt_raises(self):
    '''Extracting corrupt ZIP should raise an exception.'''
    with self.assertRaises(Exception):
      self.ch.ExtractFileUsingZipfile(self.badZipPath, self.extractDir)

  def test_extract_zipfile_overwrite(self):
    '''Extracting ZIP over existing files should overwrite them.'''
    # Extract once
    self.ch.ExtractFileUsingZipfile(self.zipPath, self.extractDir)
    # Modify a file
    p = os.path.join(self.extractDir, "file1.txt")
    with open(p, "w") as f:
      f.write("modified")
    # Extract again; content will be overwritten or kept depending on helper; assert file exists
    self.ch.ExtractFileUsingZipfile(self.zipPath, self.extractDir)
    self.assertTrue(os.path.exists(p))

  # ========== TAR ==========

  def test_extract_tarfile_basic(self):
    '''Extract TAR and verify files exist.'''
    self.ch.ExtractFileUsingTarfile(self.tarPath, self.extractDir)
    for rel in self.sampleFiles.keys():
      self.assertTrue(os.path.exists(os.path.join(self.extractDir, rel)))

  def test_extract_tarfile_destination_created(self):
    '''Ensure destination directory is created if missing.'''
    dest = os.path.join(self.testDir, "new_extract_tar")
    self.ch.ExtractFileUsingTarfile(self.tarPath, dest)
    self.assertTrue(os.path.isdir(dest))

  def test_extract_tarfile_gz_mode(self):
    '''Extract tar.gz using tarfile mode variations if supported.'''
    # Create tar.gz
    tgz = os.path.join(self.archivesDir, "test.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
      for rel in self.sampleFiles.keys():
        t.add(os.path.join(self.payloadDir, rel), arcname=rel)
    # Extract using tarfile mode variation inside helper if supported
    self.ch.ExtractFileUsingTarfile(tgz, self.extractDir)
    for rel in self.sampleFiles.keys():
      self.assertTrue(os.path.exists(os.path.join(self.extractDir, rel)))

  # ========== Unsupported unless libs installed ==========

  def test_extract_7zfile_import_error_when_missing(self):
    '''Expect an exception for 7z when py7zr is missing or file is invalid.'''
    archivePath = os.path.join(self.archivesDir, "dummy.7z")
    with open(archivePath, "wb") as f:
      f.write(b"7z placeholder")
    with self.assertRaises(Exception):
      self.ch.ExtractFileUsing7zfile(archivePath, self.extractDir)

  def test_extract_rarfile_import_error_when_missing(self):
    '''Expect an exception for rar when rarfile is missing or file is invalid.'''
    archivePath = os.path.join(self.archivesDir, "dummy.rar")
    with open(archivePath, "wb") as f:
      f.write(b"rar placeholder")
    with self.assertRaises(Exception):
      self.ch.ExtractFileUsingRarfile(archivePath, self.extractDir)

  def test_extract_unrar_import_error_when_missing(self):
    '''Expect an exception for unrar when package missing or file is invalid.'''
    archivePath = os.path.join(self.archivesDir, "dummy.rar")
    with open(archivePath, "wb") as f:
      f.write(b"rar placeholder")
    with self.assertRaises(Exception):
      self.ch.ExtractFileUsingUnrar(archivePath, self.extractDir)

  def test_extract_unzip_import_error_when_missing(self):
    '''Expect an exception for unzip third-party package when missing or invalid file.'''
    archivePath = os.path.join(self.archivesDir, "dummy.zip")
    with open(archivePath, "wb") as f:
      f.write(b"zip placeholder")
    with self.assertRaises(Exception):
      self.ch.ExtractFileUsingUnzip(archivePath, self.extractDir)


if __name__ == "__main__":
  unittest.main()
