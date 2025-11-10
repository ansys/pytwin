# conftest.py
import pytest
import sys

def pytest_collection_modifyitems(config, items):
    if sys.platform == "linux":
        for item in items:
            if "TestTbRom" in item.nodeid:
                item.add_marker(pytest.mark.forked)
