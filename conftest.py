"""
Root conftest.py — adds src/ to sys.path so efv_nn is importable
when running pytest from the project root or the tests/ directory.
"""
import sys
import os

# Ensure src/ is on the path regardless of where pytest is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
