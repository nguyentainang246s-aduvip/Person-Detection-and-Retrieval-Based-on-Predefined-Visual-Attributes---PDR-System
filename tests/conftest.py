"""
tests/conftest.py
=================
Shared fixtures cho pytest test suite của PDR-System.
"""
import sys
import os
import pytest
import numpy as np

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def dummy_frame():
    """Frame giả 640x480 BGR cho testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_crop():
    """Person crop giả 256x128 BGR."""
    return np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_query():
    """Query tìm kiếm mẫu."""
    return {
        "gender": "Male",
        "upper_color": "Black",
        "lower_color": "Blue",
        "hat": False,
        "glasses": False,
        "backpack": True,
    }
