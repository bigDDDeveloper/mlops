"""
Reproducible preprocessing transformer - HeartCare MLOps Pipeline
Task 4: a self-contained sklearn transformer that turns RAW patient input
(the 13 original clinical features) into the exact one-hot + scaled feature
matrix the model was trained on.

Bundling this transformer with the estimator in a single Pipeline means
inference is fully reproducible from raw input - the serving API only has to
pass through the raw JSON, with no manual feature wrangling.

This module must stay importable at load time (pickle references it by path),
so keep it dependency-light and side-effect free.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# Raw clinical input columns expected from an API client / raw CSV.
RAW_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Categorical columns that get one-hot encoded (drop_first=True, matching the
# encoding used in data acquisition).
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Continuous columns that get standardized.
CONTINUOUS_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms raw heart-disease features into the trained feature space.

    Steps (reproducing the training-time pipeline exactly):
      1. One-hot encode categorical columns (drop_first=True).
      2. Reindex to the fixed training feature column order, filling any
         category absent from the input batch with 0.
      3. Standardize the continuous columns with a scaler fit on training data.

    The transformer learns `feature_columns_` (final column order) and fits its
    own StandardScaler during `fit`, so the whole transform travels with the
    pickle - no external scaler file required.
    """

    def __init__(self, feature_columns=None):
        # Fixed output column order. If None, it is learned during fit.
        self.feature_columns = feature_columns

    def _encode(self, X):
        """One-hot encode a raw dataframe (no column alignment yet)."""
        df = pd.DataFrame(X).copy()

        # Ensure all raw columns exist; missing ones default to 0.
        for col in RAW_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df = df[RAW_COLUMNS]

        cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
        encoded = pd.get_dummies(
            df, columns=cat_present, prefix=cat_present, drop_first=True
        )
        bool_cols = encoded.select_dtypes(include='bool').columns
        encoded[bool_cols] = encoded[bool_cols].astype(int)
        return encoded

    def fit(self, X, y=None):
        encoded = self._encode(X)

        # Learn the final feature column order if not provided explicitly.
        if self.feature_columns is None:
            self.feature_columns_ = list(encoded.columns)
        else:
            self.feature_columns_ = list(self.feature_columns)

        encoded = encoded.reindex(columns=self.feature_columns_, fill_value=0)

        cont = [c for c in CONTINUOUS_COLS if c in self.feature_columns_]
        self.scaled_columns_ = cont
        self.scaler_ = StandardScaler()
        self.scaler_.fit(encoded[cont])
        return self

    def transform(self, X):
        encoded = self._encode(X)
        encoded = encoded.reindex(columns=self.feature_columns_, fill_value=0)
        encoded[self.scaled_columns_] = self.scaler_.transform(
            encoded[self.scaled_columns_]
        )
        return encoded.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_columns_)
