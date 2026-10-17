"""
Feature Engineering - HeartCare MLOps Pipeline
Task 2 (part 1): Prepare final ML features (scaling, encoding) and produce a
reproducible train/test split.

Design notes:
- One-hot encoding of categorical columns already happened in data acquisition
  (heart_disease_encoded.csv). Here we focus on the train/test split and
  feature scaling.
- The StandardScaler is fit ONLY on the training split and then applied to the
  test split, to avoid data leakage. The fitted scaler and the feature column
  order are persisted so the serving API can reproduce the exact transform.

Reference structure adapted from Source_Code/feature_engineering/build_features.py.
"""

import os
import json
import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [FEATURES_DIR, MODELS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'features.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('features')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


TARGET_COL = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Continuous columns that benefit from standardization. One-hot dummy columns
# (0/1) are left untouched - scaling them adds no value and hurts readability.
CONTINUOUS_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


def load_encoded_data():
    """Load the one-hot encoded dataset produced by data acquisition."""
    path = os.path.join(PROCESSED_DIR, 'heart_disease_encoded.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Encoded data not found at {path}. Run data acquisition first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded encoded data: shape={df.shape}")
    return df


def split_features_target(df):
    """Separate feature matrix X and target vector y."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' missing from data")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    logger.info(f"Features: {X.shape[1]} columns, target positives={int(y.sum())}")
    return X, y


def make_split(X, y):
    """Create a stratified train/test split (preserves class ratio)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize continuous columns. Scaler is fit on train only, then applied
    to both splits. Returns scaled copies plus the fitted scaler.
    """
    cont = [c for c in CONTINUOUS_COLS if c in X_train.columns]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cont] = scaler.fit_transform(X_train[cont])
    X_test_scaled[cont] = scaler.transform(X_test[cont])

    logger.info(f"Scaled continuous columns: {cont}")
    return X_train_scaled, X_test_scaled, scaler, cont


def persist_artifacts(X_train, X_test, y_train, y_test, scaler, scaled_cols,
                      feature_columns):
    """Save split CSVs, the fitted scaler, and feature metadata."""
    # split CSVs (features + target recombined for convenience)
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    train_path = os.path.join(FEATURES_DIR, 'train.csv')
    test_path = os.path.join(FEATURES_DIR, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved train split: {train_path}")
    logger.info(f"Saved test split: {test_path}")

    # fitted scaler (needed by serving to reproduce the transform)
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved fitted scaler: {scaler_path}")

    # feature metadata (column order + which columns were scaled)
    meta = {
        'feature_columns': list(feature_columns),
        'scaled_columns': list(scaled_cols),
        'target_column': TARGET_COL,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'n_features': len(feature_columns),
    }
    meta_path = os.path.join(MODELS_DIR, 'feature_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved feature metadata: {meta_path}")


def run_feature_engineering():
    """Main feature engineering entry point."""
    logger.info("=" * 50)
    logger.info("STARTING FEATURE ENGINEERING")
    logger.info("=" * 50)

    df = load_encoded_data()
    X, y = split_features_target(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = make_split(X, y)
    X_train_s, X_test_s, scaler, scaled_cols = scale_features(X_train, X_test)

    persist_artifacts(
        X_train_s, X_test_s, y_train, y_test, scaler, scaled_cols, feature_columns
    )

    logger.info("FEATURE ENGINEERING COMPLETE")
    return X_train_s, X_test_s, y_train, y_test


if __name__ == '__main__':
    run_feature_engineering()
