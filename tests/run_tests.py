import unittest, os, sys

# Output file path for test results in CamelCase filename.
outputFile = os.path.join(os.path.dirname(__file__), "TestResults.txt")

if (__name__ == "__main__"):
  loader = unittest.TestLoader()
  suite = loader.discover(start_dir="tests", pattern="*.py")
  with open(outputFile, "w", encoding="utf-8") as out:
    runner = unittest.TextTestRunner(stream=out, verbosity=2)
    result = runner.run(suite)
    out.write(f"\nRan {result.testsRun} tests\n")
    if (result.failures or result.errors):
      out.write("Failures:\n")
      for case, tb in result.failures:
        out.write(case.id() + "\n")
        out.write(tb + "\n")
      out.write("Errors:\n")
      for case, tb in result.errors:
        out.write(case.id() + "\n")
        out.write(tb + "\n")
  # Also print a brief summary to stdout to indicate where to find details.
  print(f"Test results written to {outputFile}.")
  if (result.failures or result.errors):
    sys.exit(1)
  sys.exit(0)
