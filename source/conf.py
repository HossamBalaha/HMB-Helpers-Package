# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

project = "HMB"
copyright = "2025, Hossam Magdy Balaha"
author = "Hossam Magdy Balaha"

# Robust version reading: try to read from a version.py, fallback to hardcoded value.
version = "0.1"
release = "0.1.0"
version_file = os.path.join(os.path.dirname(__file__), "..", "HMB", "version.py")
if (os.path.exists(version_file)):
  with open(version_file) as vf:
    for line in vf:
      if (line.startswith("__version__")):
        release = line.split("=")[1].strip().replace('"', '').replace("'", "")
        version = release.split(".")[0]
        break

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  "sphinx.ext.autodoc",  # Extract documentation from docstrings.
  "sphinx.ext.napoleon",  # Support Google/NumPy style docstrings.
  "sphinx.ext.viewcode",  # Add links to source code.
  "sphinx.ext.intersphinx",  # Link to other projects' documentation.
  "sphinx.ext.todo",  # Enable todo items.
  "sphinx.ext.autosummary",  # Generate summary tables.
  "sphinx.ext.coverage",  # Check documentation coverage.
  "sphinx.ext.mathjax",  # Render math formulas.
  "sphinx.ext.imgmath",  # Alternative math rendering.
  "sphinx.ext.ifconfig",  # Include content based on configuration.
  "myst_parser",  # Support for Markdown files.
  "sphinx_copybutton",  # Add copy buttons to code blocks.
  "sphinx_design",  # Enhanced design elements.
  "sphinx_autodoc_typehints",  # Better handling of type hints.
  "rst2pdf.pdfbuilder",  # PDF output support.
  "sphinx_book_theme",  # Sphinx Book Theme.
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pdf_documents = [("index", "HMB-Helpers-Package", "HMB Helpers Package Documentation", "Hossam Magdy Balaha")]

# Autodoc configuration options.
autodoc_member_order = "bysource"  # Document members in source order.
autoclass_content = "both"  # Include class and __init__ docstrings.
autodoc_typehints = "description"  # Show type hints in the description.
autodoc_type_aliases = {}  # Add type aliases if needed.
autodoc_mock_imports = []  # Mock heavy dependencies for RTD. # "spams", "cv2"
autosummary_generate = True  # Always generate autosummary pages.

# Napoleon settings for Google / NumPy docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Todo extension settings.
todo_include_todos = True

# Intersphinx mapping for popular Python projects.
intersphinx_mapping = {
  "python"               : ("https://docs.python.org/3", None),
  "numpy"                : ("https://numpy.org/doc/stable/", None),
  "torch"                : ("https://pytorch.org/docs/stable/", None),
  "torchvision"          : ("https://pytorch.org/vision/stable/", None),
  "transformers"         : ("https://huggingface.co/docs/transformers/main/en/", None),
  "scikit-learn"         : ("https://scikit-learn.org/stable/", None),
  "matplotlib"           : ("https://matplotlib.org/stable/", None),
  "pandas"               : ("https://pandas.pydata.org/pandas-docs/stable/", None),
  "seaborn"              : ("https://seaborn.pydata.org/", None),
  "scipy"                : ("https://docs.scipy.org/doc/scipy/", None),
  "nltk"                 : ("https://www.nltk.org/", None),
  "datasets"             : ("https://huggingface.co/docs/datasets/main/en/", None),
  "pydantic"             : ("https://docs.pydantic.dev/latest/", None),
  "openslide"            : ("https://openslide.org/api/python/", None),
  "optuna"               : ("https://optuna.readthedocs.io/en/stable/", None),

  "pillow"               : ("https://pillow.readthedocs.io/en/stable/", None),
  "simpleitk"            : ("https://simpleitk.readthedocs.io/en/master/", None),
  "PyWavelets"           : ("https://pywavelets.readthedocs.io/en/latest/", None),
  "sympy"                : ("https://docs.sympy.org/latest/", None),
  "PyMuPDF"              : ("https://pymupdf.readthedocs.io/en/latest/", None),
  "sentence_transformers": ("https://sbert.net/", None),
  "textstat"             : ("https://textstat.readthedocs.io/en/latest/", None),
  "imbalanced-learn"     : ("https://imbalanced-learn.org/stable/", None),
  "nibabel"              : ("https://nipy.org/nibabel/", None),
  "pyglet"               : ("https://pyglet.readthedocs.io/en/latest/", None),
  "xgboost"              : ("https://xgboost.readthedocs.io/en/stable/", None),
  "shap"                 : ("https://shap.readthedocs.io/en/latest/", None),
  "jpype1"               : ("https://jpype.readthedocs.io/en/latest/", None),
  "praat-parselmouth"    : ("https://parselmouth.readthedocs.io/en/latest/", None),
  "gensim"               : ("https://radimrehurek.com/gensim/", None),
  "gtts"                 : ("https://gtts.readthedocs.io/en/latest/", None),
  "py7zr"                : ("https://py7zr.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Other options: "sphinx_rtd_theme", "furo", "pydata_sphinx_theme",
# "sphinx_book_theme", "alabaster", "classic", etc.
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/Logo.png"  # Add your logo file to _static.
html_favicon = "_static/Favicons/favicon.ico"  # Add your favicon file to _static.
html_theme_options = {
  # "description"    : "HMB Helpers Package Documentation",
  # "fixed_sidebar"  : False,
  # "github_user"    : "HossamBalaha",
  # "github_repo"    : "HMB-Helpers-Package",
  # "github_banner"  : True,
  # "show_powered_by": False,
  "repository_url"       : "https://github.com/HossamBalaha/HMB-Helpers-Package",
  "repository_branch"    : "main",
  # "path_to_docs": "source/",
  "use_repository_button": True,
  # "use_issues_button": True,
  "use_download_button"  : True,
  "use_fullscreen_button": True,
  # "use_edit_page_button": True,
  "home_page_in_toc"     : True,
  "show_navbar_depth"    : 2,
  # "logo_only": False,
  "show_toc_level"       : 2,
  # "announcement": "Welcome to the HMB Helpers Package documentation!",
  # "extra_navbar": "",
  # "extra_footer": "",
}

# -- Options for LaTeX/PDF output --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

# -- Miscellaneous -----------------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here, relative to this directory.
# Add custom CSS if needed:
html_css_files = ["custom.css"]

# -- Options for LaTeX/PDF output --------------------------------------------
# Use XeLaTeX for better font handling and Unicode support. This gives a nicer PDF output
# when combined with fontspec and microtype. Requires a TeX distribution (TeX Live / MiKTeX).
latex_engine = "xelatex"

# Ensure we have a known root doc name for documentation builds.
root_doc = "index"

# Configure documents produced for LaTeX builds (Sphinx -> LaTeX -> PDF).
latex_documents = [
  (root_doc, "HMB.tex", "HMB-Helpers-Package", "Hossam Magdy Balaha", "manual"),
]

# Improve LaTeX preamble for a more modern and polished PDF theme.
latex_elements = {
  "papersize": "a4paper",
  "pointsize": "11pt",
#   # Use fontspec for modern fonts and microtype for better justification.
#   "preamble" : r"""
# \usepackage{fontspec}
# \usepackage{microtype}
# \usepackage{geometry}
# \geometry{left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
# % Set a nice default serif font. Change if not installed on your system.
# \setmainfont{TeX Gyre Termes}
# \setsansfont{TeX Gyre Heros}
# \setmonofont{Latin Modern Mono}
# % Better line stretch and paragraph spacing for readability.
# \usepackage{setspace}
# \setstretch{1.08}
# \setlength{\parskip}{0.5\baselineskip}
# \setlength{\parindent}{0pt}
# % Hyperref options (colors for links)
# \usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}
# """,
}

# Print configuration summary for debugging.
print(f"Building docs for {project} v{release} by {author}")

# pip install -r requirements.txt
# pip install sphinx sphinx-book-theme myst-parser sphinx-copybutton sphinx-design sphinx-autodoc-typehints rst2pdf
# sphinx-build -b html source build\html
# sphinx-build -b pdf source build\pdf
