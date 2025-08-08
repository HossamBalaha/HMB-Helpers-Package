from setuptools import setup, find_packages

setup(
  name="HMB",
  version="0.1.0",
  author="Hossam Magdy Balaha",
  author_email="hmbala01@louisville.edu",
  description="A Python package for various utilities.",
  packages=find_packages(
    exclude=[
      "tests",
      "*.tests",
      "*.tests.*",
      "tests.*",
    ]
  ),
)
