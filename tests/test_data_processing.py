"""Unit tests for data cleaning, encoding and class-imbalance analysis."""

import pandas as pd

from src.data_acquisition.acquire_data import (
    clean_data, encode_features, analyze_class_imbalance,
    CATEGORICAL_COLS, TARGET_COL,
)


def test_clean_data_removes_duplicates(raw_sample):
    cleaned = clean_data(raw_sample)
    # The sample has one duplicate row (row 4 == row 0).
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) < len(raw_sample)


def test_clean_data_fills_missing_values(raw_sample):
    cleaned = clean_data(raw_sample)
    # 'ca' and 'thal' had NaNs; cleaning must fill them all.
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_binarizes_target(raw_sample):
    cleaned = clean_data(raw_sample)
    # Original target had values 0..3; after cleaning only {0, 1} remain.
    assert set(cleaned[TARGET_COL].unique()).issubset({0, 1})


def test_clean_data_categorical_are_integers(raw_sample):
    cleaned = clean_data(raw_sample)
    for col in CATEGORICAL_COLS:
        if col in cleaned.columns:
            assert pd.api.types.is_integer_dtype(cleaned[col])


def test_encode_features_expands_columns(raw_sample):
    cleaned = clean_data(raw_sample)
    encoded = encode_features(cleaned)
    # One-hot encoding must keep the target column.
    assert TARGET_COL in encoded.columns
    # A categorical column with multiple values should be expanded into
    # prefixed dummy columns and the original raw column should be gone.
    assert 'cp' not in encoded.columns
    assert any(c.startswith('cp_') for c in encoded.columns)


def test_encode_features_no_object_columns(raw_sample):
    cleaned = clean_data(raw_sample)
    encoded = encode_features(cleaned)
    # All encoded columns should be numeric (no leftover object/bool dtype).
    assert encoded.select_dtypes(include='object').shape[1] == 0
    assert encoded.select_dtypes(include='bool').shape[1] == 0


def test_analyze_class_imbalance_structure(raw_sample):
    cleaned = clean_data(raw_sample)
    result = analyze_class_imbalance(cleaned)
    for key in ('counts', 'imbalance_ratio', 'severity', 'class_weights'):
        assert key in result
    assert result['imbalance_ratio'] >= 1.0
    assert result['severity'] in ('balanced', 'mild', 'moderate', 'severe')
