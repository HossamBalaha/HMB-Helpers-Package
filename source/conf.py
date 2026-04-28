# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Import the operating system interface module for path manipulation.
import os

# Define the project name string for documentation metadata.
project = "HMB"
# Define the copyright notice string for documentation metadata.
copyright = "2026, Hossam Magdy Balaha"
# Define the primary author string for documentation metadata.
author = "Hossam Magdy Balaha"

# Define the default version string for fallback purposes.
version = "0.1"
# Define the default release string for fallback purposes.
release = "0.1.0"
# Construct the absolute path to the external version source file.
versionFile = os.path.join(os.path.dirname(__file__), "..", "HMB", "version.py")
# Check if the external version source file exists in the filesystem.
if (os.path.exists(versionFile)):
  # Open the version source file in read-only mode for parsing.
  with open(versionFile) as vf:
    # Iterate through each line contained in the opened file object.
    for line in vf:
      # Check if the current line contains the version declaration marker.
      if (line.startswith("__version__")):
        # Extract the version value and strip whitespace or quote characters.
        release = line.split("=")[1].strip().replace('"', '').replace("'", "")
        # Extract the major version component for the short version variable.
        version = release.split(".")[0]
        # Terminate the loop immediately after successfully parsing the version.
        break

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Define the list of active Sphinx documentation extensions.
# Note: Themes are configured via html_theme, not listed in extensions.
extensions = [
  "sphinx.ext.autodoc",  # Extract documentation from docstrings.
  "sphinx.ext.napoleon",  # Support Google or NumPy style docstrings.
  "sphinx.ext.viewcode",  # Add links to source code.
  "sphinx.ext.intersphinx",  # Link to external project documentation.
  "sphinx.ext.todo",  # Enable todo items.
  "sphinx.ext.autosummary",  # Generate summary tables.
  "sphinx.ext.coverage",  # Check documentation coverage.
  "sphinx.ext.mathjax",  # Render math formulas for HTML output.
  "sphinx.ext.ifconfig",  # Include content based on configuration.
  "myst_parser",  # Support for Markdown files.
  "sphinx_copybutton",  # Add copy buttons to code blocks.
  "sphinx_design",  # Enhanced design elements.
  "sphinx_autodoc_typehints",  # Better handling of type hints.
]

# Specify the directory containing custom template files.
templates_path = ["_templates"]
# Define patterns to exclude from the documentation build process.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Configure autodoc member ordering to match source code sequence.
autodoc_member_order = "bysource"
# Configure class content to include both class and initialization docstrings.
autoclass_content = "both"
# Configure type hints to appear within the description section.
autodoc_typehints = "description"
# Initialize an empty dictionary for custom type aliases.
autodoc_type_aliases = {}
# Initialize an empty list for mocking heavy or unavailable imports.
autodoc_mock_imports = []
# Disable automatic summary table generation to prevent duplicate entries.
autosummary_generate = False

# Enable Google-style docstring parsing support.
napoleon_google_docstring = True
# Enable NumPy-style docstring parsing support.
napoleon_numpy_docstring = True
# Include initialization method documentation in the output.
napoleon_include_init_with_doc = True
# Exclude private members from the generated documentation.
napoleon_include_private_with_doc = False
# Include special method documentation in the output.
napoleon_include_special_with_doc = True
# Format example blocks as admonitions.
napoleon_use_admonition_for_examples = True
# Format note blocks as admonitions.
napoleon_use_admonition_for_notes = True
# Format reference blocks as admonitions.
napoleon_use_admonition_for_references = True
# Use instance variable format for attributes.
napoleon_use_ivar = True
# Use parameter format for function arguments.
napoleon_use_param = True
# Use return type format for function outputs.
napoleon_use_rtype = True

# Enable the display of todo items in the final documentation.
todo_include_todos = True

# Define cross project linking mappings for external documentation references.
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

# Assign the preferred HTML theme for web documentation.
html_theme = "furo"
# Configure Furo theme specific options using supported keys only.
html_theme_options = {
  # Brand color configuration for light and dark modes.
  "light_css_variables" : {
    "color-brand-primary" : "#2962FF",
    "color-brand-content" : "#2962FF",
    "color-api-background": "#F5F5F5",
    "color-api-name"      : "#2962FF",
  },
  "dark_css_variables"  : {
    "color-brand-primary" : "#82B1FF",
    "color-brand-content" : "#82B1FF",
    "color-api-background": "#1E1E1E",
    "color-api-name"      : "#82B1FF",
  },
  # Sidebar and navigation configuration.
  "sidebar_hide_name"   : False,
  "navigation_with_keys": True,
  # Top announcement bar for important notices.
  "announcement"        : "Welcome to the HMB Helpers Package documentation.",
  # Footer configuration with custom icons and links.
  "footer_icons"        : [
    {
      "name" : "GitHub",
      "url"  : "https://github.com/HossamBalaha/HMB-Helpers-Package",
      "html" : """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
      "class": "",
    },
  ],
  # Source repository integration for Furo.
  "source_repository"   : "https://github.com/HossamBalaha/HMB-Helpers-Package",
  "source_branch"       : "main",
  "source_directory"    : "source/",
}
# Define the directory containing static HTML assets.
html_static_path = ["_static"]
# Specify the path to the primary project logo image.
html_logo = "_static/Logo.png"
# Specify the path to the browser favicon file.
html_favicon = "_static/Favicons/favicon.ico"
# Add copyright to HTML footer
html_footer = """
<div style="text-align: center; font-size: 0.9em; color: #666; margin-top: 2em;">
  &copy; 2026 Hossam Magdy Balaha. Licensed under MIT. 
  <a href="https://github.com/HossamBalaha/HMB-Helpers-Package">GitHub</a>
</div>
"""
# -- Miscellaneous -----------------------------------------------------------

# Register custom CSS files for additional HTML styling.
html_css_files = ["custom.css"]

# -- Options for LaTeX/PDF output --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

# Define the root document filename for the documentation tree.
root_doc = "index"
# Select the XeLaTeX compiler engine for superior Unicode and font support.
latex_engine = "xelatex"
# Define LaTeX document generation parameters.
# Note: Sphinx framework strictly requires lowercase keys for the latex_elements dictionary.
latex_elements = {
  "papersize": "a4paper",
  "pointsize": "11pt",
  "preamble" : r"""\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{fontspec}
\usepackage{microtype}
\usepackage{geometry}
\geometry{left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\setlength{\headheight}{14.5pt}
\addtolength{\topmargin}{-2.5pt}
\usepackage{enumitem}
\setlistdepth{9}
\newlist{deepitemize}{itemize}{9}
\setlist[deepitemize]{label=\textbullet}
\newlist{deepenumerate}{enumerate}{9}
\setlist[deepenumerate]{label=\arabic*.}
\newlist{deepdescription}{description}{9}
\setlist[deepdescription]{font=\normalfont\bfseries}
\let\olditemize\itemize
\let\oldenditemize\enditemize
\let\oldenumerate\enumerate
\let\oldendenumerate\endenumerate
\let\olddescription\description
\let\oldenddescription\enddescription
\renewenvironment{itemize}{\deepitemize}{\enddeepitemize}
\renewenvironment{enumerate}{\deepenumerate}{\enddeepenumerate}
\renewenvironment{description}{\deepdescription}{\enddeepdescription}
\setlist[itemize]{leftmargin=*}
\setlist[enumerate]{leftmargin=*}
\setlist[description]{leftmargin=2em}
\setlength{\jot}{8pt}""",
  "fontpkg"  : r"\usepackage{fontspec}",
  "fncychap" : "",
}

# Define the metadata tuple for the primary LaTeX document generation.
latex_documents = [
  (root_doc, "HMB.tex", "HMB-Helpers-Package", "Hossam Magdy Balaha", "manual"),
]


# -- Validation and utility functions ----------------------------------------

# Create a validation function for the compilation settings.
def ValidateLatexConfiguration():
  """Verify that the LaTeX configuration is properly structured."""
  # Retrieve the currently selected compilation engine.
  selectedEngine = latex_engine
  # Determine if the Unicode capable engine is active.
  if (selectedEngine == "xelatex"):
    # Output a confirmation message for the selected engine.
    print("Selected xelatex for compilation.")
  # Determine if the alternative Lua based engine is active.
  elif (selectedEngine == "lualatex"):
    # Output a confirmation message for the alternative engine.
    print("Selected lualatex for compilation.")
  # Handle any non-standard or legacy engine configurations.
  else:
    # Output a warning regarding potential rendering deficiencies.
    print("Defaulting to pdflatex with limited Unicode support.")


# Create a summary output function for debugging purposes.
def PrintBuildSummary():
  """Output configuration summary for debugging purposes."""
  # Print the project name and version information.
  print(f"Building docs for {project} v{release} by {author}")
  # Print the active LaTeX compilation engine.
  print(f"LaTeX engine: {latex_engine}")
  # Print the final document generation workflow.
  print(f"Output format: LaTeX -> PDF via xelatex")


# -- Execution entry point ---------------------------------------------------

# Check if the configuration file is being executed as a standalone script.
if (__name__ == "__main__"):
  # Execute the LaTeX configuration validation routine.
  ValidateLatexConfiguration()
  # Execute the build summary output routine.
  PrintBuildSummary()

# Print configuration summary for debugging during Sphinx build.
PrintBuildSummary()

# -- Build instructions ------------------------------------------------------
# Install required dependencies including the Furo theme:
# pip install sphinx furo myst-parser sphinx-copybutton sphinx-design sphinx-autodoc-typehints

# Generate HTML documentation with Furo theme:
# python -m sphinx -b html source build/html

# Generate PDF documentation via LaTeX workflow:
# Step 1: Generate LaTeX source files from reStructuredText.
# python -m sphinx -b latex source build/latex
# Step 2: Compile LaTeX to PDF using XeLaTeX (execute twice for reference resolution).
# cd build/latex && xelatex HMB.tex && xelatex HMB.tex

# Note: Ensure a complete LaTeX distribution (TeX Live or MiKTeX) is installed
# and that xelatex is available in your system PATH for successful PDF compilation.
# Furo theme options are strictly validated; unsupported keys will generate warnings.
# Consult https://pradyunsg.me/furo/customisation/ for the complete option reference.
