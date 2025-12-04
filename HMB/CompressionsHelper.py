import os


class CompressionsHelper(object):
  r'''
  CompressionsHelper: Convenience wrappers to extract various archive formats.

  Methods:
    - ExtractFileUsingTarfile(path, destination)
    - ExtractFileUsingZipfile(path, destination)
    - ExtractFileUsing7zfile(path, destination)
    - ExtractFileUsingRarfile(path, destination)
    - ExtractFileUsingUnrar(path, destination)
    - ExtractFileUsingUnzip(path, destination)
  '''

  def ExtractFileUsingTarfile(self, path, destination):
    r'''
    Extract a tar/ tar.gz / tar.bz2 archive using the standard library `tarfile`.

    Parameters:
      path (str): Path to the archive file.
      destination (str): Directory to extract files into. Will be created if missing.
    '''

    import tarfile

    os.makedirs(destination, exist_ok=True)
    tar = tarfile.open(path)
    tar.extractall(destination)
    tar.close()

  def ExtractFileUsingZipfile(self, path, destination):
    r'''
    Extract a ZIP archive using the standard library `zipfile`.

    Parameters:
      path (str): Path to the .zip file.
      destination (str): Directory to extract files into. Will be created if missing.
    '''

    import zipfile

    os.makedirs(destination, exist_ok=True)
    zipRef = zipfile.ZipFile(path, "r")
    zipRef.extractall(destination)
    zipRef.close()

  def ExtractFileUsing7zfile(self, path, destination):
    r'''
    Extract a 7-zip (.7z) archive using the optional `py7zr` package.
    If `py7zr` is not installed an ImportError will be raised.

    Parameters:
      path (str): Path to the .7z file.
      destination (str): Directory to extract files into. Will be created if missing.
    '''

    try:
      import py7zr
    except Exception as e:
      raise ImportError("py7zr is required to extract .7z archives: " + str(e))

    os.makedirs(destination, exist_ok=True)
    with py7zr.SevenZipFile(path, "r") as z:
      z.extractall(destination)

  def ExtractFileUsingRarfile(self, path, destination):
    r'''
    Extract a RAR archive using the optional `rarfile` package.
    If `rarfile` is not installed an ImportError will be raised.

    Parameters:
      path (str): Path to the .rar file.
      destination (str): Directory to extract files into. Will be created if missing.
    '''

    try:
      import rarfile
    except Exception as e:
      raise ImportError("rarfile is required to extract .rar archives: " + str(e))

    os.makedirs(destination, exist_ok=True)
    with rarfile.RarFile(path) as rf:
      rf.extractall(destination)

  def ExtractFileUsingUnrar(self, path, destination):
    r'''
    Extract a RAR archive using the optional `unrar` package (alternate to rarfile).
    If `unrar` is not installed an ImportError will be raised.

    Parameters:
      path (str): Path to the .rar file.
      destination (str): Directory to extract files into. Will be created if missing.
    '''

    try:
      import unrar
    except Exception as e:
      raise ImportError("unrar is required to extract .rar archives with unrar: " + str(e))

    os.makedirs(destination, exist_ok=True)
    rar = unrar.RarFile(path)
    rar.extractall(destination)

  def ExtractFileUsingUnzip(self, path, destination):
    r'''
    Extract an archive with a third-party `unzip` package if available.
    This is provided for environments that prefer a non-stdlib unzip utility.
    If the `unzip` package is not installed an ImportError will be raised.

    Parameters:
      path (str): Path to the archive file.
      destination (str): Directory to extract files into. Will be created if missing.
    '''
    try:
      import unzip
    except Exception as e:
      raise ImportError("unzip package is required for ExtractFileUsingUnzip: " + str(e))

    os.makedirs(destination, exist_ok=True)
    zip = unzip.ZipFile(path)
    zip.extractall(destination)
