"""
Pytest configuration for the Library test suite.

Adds the `Library/` folder itself to sys.path so tests can import the
calculation modules the same way a user would (e.g. `from covariance import
covariance_matrix`), and exposes the path to `test_files/` for loading
input/output CSVs.
"""

import os
import sys

LIBRARY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILES_DIR = os.path.join(LIBRARY_DIR, "test_files")

if LIBRARY_DIR not in sys.path:
    sys.path.insert(0, LIBRARY_DIR)
