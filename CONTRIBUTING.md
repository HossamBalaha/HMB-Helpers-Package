<!--
CONTRIBUTING.md for HMB-Helpers-Package
This file provides clear, practical instructions for contributors: how to report issues, set up a
development environment on Windows (cmd.exe) or POSIX systems, run tests, follow code style, and
submit pull requests.
-->

# Contributing to HMB-Helpers-Package

Thank you for your interest in improving HMB-Helpers-Package! This document explains how to report
issues, propose changes, run the project locally, and submit pull requests so maintainers can
review and merge your work quickly.

---

## Table of Contents

- [Getting started (quick setup)](#getting-started-quick-setup)
- [Development environment](#development-environment)
- [Running tests](#running-tests)
- [Linting and static checks](#linting-and-static-checks)
- [Documentation and examples](#documentation-and-examples)
- [How to contribute (issues & pull requests)](#how-to-contribute-issues--pull-requests)
- [Pull request checklist](#pull-request-checklist)
- [Code style, commit messages & branching](#code-style-commit-messages--branching)
- [Releasing and versioning](#releasing-and-versioning)
- [Reporting security issues](#reporting-security-issues)
- [Contact](#contact)

---

## Getting started (quick setup)

1. Fork the repository on GitHub and clone your fork locally.
2. Create and switch to a feature branch:

```bat
git checkout -b feature/my-awesome-change
```

3. Install a development environment (see next section).

## Development environment

This project supports development on Windows using `cmd.exe` (the repository includes `make.bat`) and
POSIX systems. The easiest way to get a development environment is with a virtual environment.

Recommended steps (Windows cmd.exe):

```bat
:: Clone your fork
git clone https://github.com/<your-user>/HMB-Helpers-Package.git
cd HMB-Helpers-Package

:: Create a venv and activate it (Windows)
python -m venv .venv
.venv\Scripts\activate

:: Install development extras
pip install -e ".[dev]"

:: (Optional) Install all extras if you need the full stack
pip install -e ".[all]"
```

POSIX (bash) alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Notes:

- The project `setup.py` defines extras groups like `pytorch`, `pdf`, `nlp`, `dev`, and `all`.
- Many optional dependencies are large (PyTorch, TensorFlow, etc.) — install only what you need.

## Running tests

Unit tests live under `tests/`. There are two supported ways to run tests:

- Using the bundled test runner (portable):

```bat
:: From repository root (Windows)
python tests/run_tests.py

:: Run a specific test file
python tests/run_tests.py Test_ImagesHelper.py
```

- Using pytest (recommended for contributors with pytest installed):

```bat
pytest -q
pytest -q tests/Test_ImagesHelper.py
```

If you added or modified functionality, include unit tests that exercise new behavior. Aim for
deterministic tests that run quickly and do not require GPU by default.

## Linting and static checks

This project includes recommended dev tools in the `dev` extra: `black`, `flake8`, and `mypy`.
Run them before submitting a PR.

Format with Black:

```bat
black .
```

Run flake8:

```bat
flake8 HMB tests
```

Run mypy (type checks):

```bat
mypy HMB
```

If you prefer, run these as part of a pre-commit hook (not provided by default) or integrate them
with your editor/IDE.

## Documentation and examples

- Documentation source is under `source/`. Build docs locally with Sphinx.

Windows (cmd.exe):

```bat
call make.bat html
:: Output will be in build/html/
```

POSIX:

```bash
cd source
make html
```

- Examples live under `HMB/Examples`. Use the platform wrapper scripts in
  `HMB/Examples/BAT Files` (Windows) and `HMB/Examples/SH Files` (POSIX) to run platform-specific
  scenarios. You can also run individual Python example files directly.

## How to contribute (issues & pull requests)

1. Open an issue to discuss large or design-changing work before implementing it. For small bugfixes
   or documentation edits you can open a PR directly.
2. Fork, create a branch, implement changes, add tests and documentation updates.
3. Run the test and linting suite locally and ensure all checks pass.
4. Push your branch and open a pull request against `main` (or the default branch).

When opening an issue or PR, include:

- A clear and descriptive title
- A short summary of the problem or feature
- Steps to reproduce (for bugs) and expected vs actual behavior
- The platform and Python version used
- Any screenshots, stack traces, or minimal reproducible examples

## Pull request checklist

Before marking your PR as ready for review, ensure the following:

- [ ] The change addresses a single logical set of functionality (avoid huge mixed PRs)
- [ ] All new and existing tests pass (run `pytest` or the test runner)
- [ ] New code includes unit tests and docstrings
- [ ] Documentation (`.rst` or README) updated when behavior or public API changes
- [ ] Code formatted with `black` and checked with `flake8` / `mypy` as appropriate
- [ ] Commit messages are clear and follow the project's commit style (see below)
- [ ] CI checks (if present) are green

Maintainers may request changes — please address review comments and push updates to the same branch.

## Code style, commit messages & branching

- Follow PEP 8 for Python style. Use `black` to auto-format code.
- Keep functions and classes small and focused; add docstrings for public APIs following NumPy or
  Google style (the codebase primarily uses docstrings in module files).
- Type hints are appreciated; add them when modifying public APIs.

Branch naming recommendations:

- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `test/<short-description>`

Commit message guidance (keep them informative):

```
<type>(<scope>): <short summary>

Longer description, if necessary. Explain why the change was made and
mention any backwards incompatible changes.

Examples of <type>: feat, fix, docs, style, refactor, test, chore
```

## Releasing and versioning

Releases are managed in `setup.py` (version string). If you are a maintainer preparing a release,
update the version in `setup.py`, update `CHANGELOG.md` (if you maintain one), and create a GitHub
release with the appropriate tag.

If you are a contributor, do not modify the package version; open a PR against the `main` branch and
the maintainer will manage releases.

## Reporting security issues

If you discover a security vulnerability, please do not open a public issue. Contact the maintainer
directly at h3ossam@gmail.com and include sufficient detail to reproduce and mitigate the issue. The
maintainer will coordinate a fix and disclosure timeline.

## Contact

For questions, bug reports, or contribution help:

- Open an issue on GitHub: https://github.com/HossamBalaha/HMB-Helpers-Package/issues
- Email: h3ossam@gmail.com

Thanks for contributing — we look forward to your improvements!
