"""
Model Packaging & Reproducibility - HeartCare MLOps Pipeline
Task 4: Save the final model in a reusable format bundled with its
preprocessing pipeline, so inference is reproducible from raw patient input.

Produces a single artifact (models/heart_disease_pipeline.pkl) containing:
  raw features -> HeartDiseasePreprocessor -> trained classifier -> prediction

Also writes a model card (models/model_card.json) and exposes load/predict
helpers reused by the serving API and unit tests.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Allow running both as a module and as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.packaging.preprocessing import HeartDiseasePreprocessor, RAW_COLUMNS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [MODELS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'packaging.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('packaging')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

TARGET_COL = 'target'
RANDOM_STATE = 42
PIPELINE_PATH = os.path.join(MODELS_DIR, 'heart_disease_pipeline.pkl')
MODEL_CARD_PATH = os.path.join(MODELS_DIR, 'model_card.json')


def _rebuild_estimator(model_name, best_params):
    """Recreate the selected estimator with its tuned hyperparameters."""
    if model_name == 'logistic_regression':
        return LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
            class_weight='balanced', **best_params
        )
    if model_name == 'random_forest':
        return RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced', **best_params
        )
    raise ValueError(f"Unknown model '{model_name}'")


def build_pipeline():
    """
    Build and fit the end-to-end pipeline on raw cleaned data.

    Uses heart_disease_clean.csv (raw clinical values, pre-encoding) so the
    preprocessor inside the pipeline owns the full encode+scale transform.
    """
    clean_path = os.path.join(PROCESSED_DIR, 'heart_disease_clean.csv')
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"{clean_path} not found. Run data acquisition first."
        )
    df = pd.read_csv(clean_path)
    X = df[RAW_COLUMNS].copy()
    y = df[TARGET_COL].astype(int)

    # Load the selected model + tuned params chosen during training.
    info_path = os.path.join(MODELS_DIR, 'best_model_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"{info_path} not found. Run training first."
        )
    with open(info_path) as f:
        info = json.load(f)

    model_name = info['model_name']
    best_params = info.get('best_params', {})
    logger.info(f"Rebuilding '{model_name}' with params {best_params}")

    estimator = _rebuild_estimator(model_name, best_params)
    pipeline = Pipeline([
        ('preprocessor', HeartDiseasePreprocessor()),
        ('classifier', estimator),
    ])
    pipeline.fit(X, y)
    logger.info("Pipeline fit on raw cleaned data complete")
    return pipeline, info, X


def save_pipeline(pipeline, info, X_sample):
    """Persist the pipeline artifact and write a model card."""
    with open(PIPELINE_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved pipeline artifact: {PIPELINE_PATH}")

    card = {
        'name': 'heart_disease_risk_classifier',
        'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_type': info['model_name'],
        'hyperparameters': info.get('best_params', {}),
        'test_metrics': info.get('test_metrics', {}),
        'raw_input_columns': RAW_COLUMNS,
        'target': {'0': 'no_disease', '1': 'disease'},
        'framework': 'scikit-learn',
        'artifact': os.path.basename(PIPELINE_PATH),
        'created_at': datetime.now().isoformat(),
        'sample_input': X_sample.iloc[0].to_dict(),
    }
    with open(MODEL_CARD_PATH, 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=2, default=str)
    logger.info(f"Saved model card: {MODEL_CARD_PATH}")
    return card


def load_pipeline():
    """Load the packaged pipeline artifact (used by serving + tests)."""
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"{PIPELINE_PATH} not found. Run packaging first."
        )
    with open(PIPELINE_PATH, 'rb') as f:
        return pickle.load(f)


def predict(records, pipeline=None):
    """
    Predict heart disease risk for one or more raw patient records.

    `records` is a dict or list of dicts with the RAW_COLUMNS keys.
    Returns a list of {prediction, confidence, probability} dicts.
    """
    if pipeline is None:
        pipeline = load_pipeline()
    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame(records)
    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)

    results = []
    for pred, proba in zip(preds, probas):
        pred = int(pred)
        results.append({
            'prediction': pred,
            'label': 'disease' if pred == 1 else 'no_disease',
            'confidence': round(float(proba[pred]), 4),
            'probability_disease': round(float(proba[1]), 4),
        })
    return results


def run_packaging():
    """Main packaging entry point."""
    logger.info("=" * 50)
    logger.info("STARTING MODEL PACKAGING")
    logger.info("=" * 50)

    pipeline, info, X = build_pipeline()
    card = save_pipeline(pipeline, info, X)

    # Smoke test: round-trip load + predict on a sample row.
    loaded = load_pipeline()
    sample = X.iloc[0].to_dict()
    result = predict(sample, pipeline=loaded)
    logger.info(f"Smoke-test prediction on sample input: {result[0]}")

    logger.info("MODEL PACKAGING COMPLETE")
    return card, result


if __name__ == '__main__':
    run_packaging()
