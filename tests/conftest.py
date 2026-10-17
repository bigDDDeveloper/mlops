"""Shared pytest fixtures and path setup for the HeartCare test suite."""

import os
import sys

import pandas as pd
import pytest

# Ensure the project root is importable so `import src.*` works under pytest.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


@pytest.fixture
def raw_sample():
    """A small raw heart-disease-like dataframe with some issues to clean."""
    return pd.DataFrame({
        'age': [63, 37, 41, 63, 56],          # last row duplicates row 0
        'sex': [1, 1, 0, 1, 1],
        'cp': [1, 2, 1, 1, 2],
        'trestbps': [145, 130, 130, 145, 120],
        'chol': [233, 250, 204, 233, 236],
        'fbs': [1, 0, 0, 1, 0],
        'restecg': [2, 1, 0, 2, 1],
        'thalach': [150, 187, 172, 150, 178],
        'exang': [0, 0, 0, 0, 0],
        'oldpeak': [2.3, 3.5, 1.4, 2.3, 0.8],
        'slope': [3, 3, 1, 3, 1],
        'ca': [0, 0, None, 0, 0],             # missing value
        'thal': [6, 3, 3, 6, None],           # missing value
        'target': [0, 2, 1, 0, 3],            # multi-class -> binarize
    })


@pytest.fixture
def raw_patient():
    """A single raw patient record (as the API would receive)."""
    return {
        'age': 63, 'sex': 1, 'cp': 4, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 6,
    }
