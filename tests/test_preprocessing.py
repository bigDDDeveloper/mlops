"""Unit tests for the reproducible preprocessing transformer."""

import numpy as np
import pandas as pd

from src.packaging.preprocessing import (
    HeartDiseasePreprocessor, RAW_COLUMNS, CONTINUOUS_COLS,
)


def _raw_df(n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'age': rng.integers(40, 70, n),
        'sex': rng.integers(0, 2, n),
        'cp': rng.integers(1, 5, n),
        'trestbps': rng.integers(110, 160, n),
        'chol': rng.integers(180, 280, n),
        'fbs': rng.integers(0, 2, n),
        'restecg': rng.integers(0, 3, n),
        'thalach': rng.integers(120, 190, n),
        'exang': rng.integers(0, 2, n),
        'oldpeak': rng.random(n) * 3,
        'slope': rng.integers(1, 4, n),
        'ca': rng.integers(0, 4, n),
        'thal': rng.choice([3, 6, 7], n),
    })


def test_fit_learns_feature_columns():
    pre = HeartDiseasePreprocessor()
    pre.fit(_raw_df())
    assert hasattr(pre, 'feature_columns_')
    assert len(pre.feature_columns_) > len(RAW_COLUMNS) - 1  # dummies expand


def test_transform_output_shape():
    pre = HeartDiseasePreprocessor().fit(_raw_df(10))
    out = pre.transform(_raw_df(4))
    assert out.shape[0] == 4
    assert out.shape[1] == len(pre.feature_columns_)


def test_transform_is_numeric():
    pre = HeartDiseasePreprocessor().fit(_raw_df())
    out = pre.transform(_raw_df())
    assert np.issubdtype(out.dtype, np.number)
    assert not np.isnan(out).any()


def test_fixed_feature_columns_are_respected():
    cols = ['age', 'sex_1', 'cp_2', 'thal_7']
    pre = HeartDiseasePreprocessor(feature_columns=cols).fit(_raw_df())
    assert list(pre.feature_columns_) == cols
    out = pre.transform(_raw_df(3))
    assert out.shape[1] == len(cols)


def test_scaled_columns_are_standardized():
    pre = HeartDiseasePreprocessor().fit(_raw_df(50))
    # continuous columns in the fitted scaler should match CONTINUOUS_COLS
    assert set(pre.scaled_columns_) == set(
        c for c in CONTINUOUS_COLS if c in pre.feature_columns_
    )


def test_missing_input_columns_default_to_zero():
    pre = HeartDiseasePreprocessor().fit(_raw_df())
    # Provide only a couple of fields; transform should not raise.
    out = pre.transform(pd.DataFrame([{'age': 55, 'sex': 1}]))
    assert out.shape == (1, len(pre.feature_columns_))
