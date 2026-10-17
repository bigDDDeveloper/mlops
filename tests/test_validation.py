"""Unit tests for the data validation checks."""

import pandas as pd

from src.validation.validate_data import (
    check_missing_values, check_duplicates, check_schema,
    check_range, check_class_balance, SCHEMA,
)


def _valid_df():
    return pd.DataFrame({
        'age': [63, 37], 'sex': [1, 0], 'cp': [1, 2], 'trestbps': [145, 130],
        'chol': [233, 250], 'fbs': [1, 0], 'restecg': [2, 1],
        'thalach': [150, 187], 'exang': [0, 0], 'oldpeak': [2.3, 3.5],
        'slope': [3, 1], 'ca': [0, 0], 'thal': [6, 3], 'target': [0, 1],
    })


def test_check_missing_values_none():
    issues = check_missing_values(_valid_df(), 'ok')
    assert issues == []


def test_check_missing_values_detects():
    df = _valid_df()
    df.loc[0, 'chol'] = None
    issues = check_missing_values(df, 'bad')
    assert len(issues) == 1
    assert issues[0]['column'] == 'chol'


def test_check_duplicates():
    df = _valid_df()
    assert check_duplicates(df, 'ok') == 0
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert check_duplicates(dup, 'dup') == 1


def test_check_schema_pass():
    ok, missing, extra = check_schema(_valid_df(), 'ok', SCHEMA['columns'])
    assert ok is True
    assert missing == []


def test_check_schema_missing_column():
    df = _valid_df().drop(columns=['chol'])
    ok, missing, extra = check_schema(df, 'bad', SCHEMA['columns'])
    assert ok is False
    assert 'chol' in missing


def test_check_range_in_bounds():
    df = _valid_df()
    assert check_range(df, 'ok', 'age', 18, 100) == 0


def test_check_range_out_of_bounds():
    df = _valid_df()
    df.loc[0, 'age'] = 200
    assert check_range(df, 'bad', 'age', 18, 100) == 1


def test_check_class_balance():
    bal = check_class_balance(_valid_df(), 'ok')
    assert bal['counts'] == {0: 1, 1: 1}
    assert bal['positive_pct'] == 50.0
