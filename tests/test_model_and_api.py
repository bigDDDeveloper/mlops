"""
Unit tests for the packaged model pipeline and the serving API.

These tests depend on the trained artifact models/heart_disease_pipeline.pkl.
If it is not present (e.g. a fresh checkout before training), the model/API
tests are skipped rather than failing - CI builds the model before testing.
"""

import os

import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'heart_disease_pipeline.pkl')
pipeline_available = os.path.exists(PIPELINE_PATH)

skip_no_model = pytest.mark.skipif(
    not pipeline_available,
    reason="packaged pipeline not found - run training+packaging first",
)


@skip_no_model
def test_predict_single(raw_patient):
    from src.packaging.package_model import predict
    result = predict(raw_patient)[0]
    assert result['prediction'] in (0, 1)
    assert result['label'] in ('disease', 'no_disease')
    assert 0.0 <= result['confidence'] <= 1.0
    assert 0.0 <= result['probability_disease'] <= 1.0


@skip_no_model
def test_predict_batch(raw_patient):
    from src.packaging.package_model import predict
    results = predict([raw_patient, raw_patient])
    assert len(results) == 2
    # Same input -> same output (deterministic).
    assert results[0] == results[1]


@skip_no_model
def test_api_health():
    from fastapi.testclient import TestClient
    from src.serving.app import app
    with TestClient(app) as client:
        r = client.get('/health')
        assert r.status_code == 200
        assert r.json()['model_loaded'] is True


@skip_no_model
def test_api_predict_endpoint(raw_patient):
    from fastapi.testclient import TestClient
    from src.serving.app import app
    with TestClient(app) as client:
        r = client.post('/predict', json=raw_patient)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {
            'prediction', 'label', 'confidence', 'probability_disease'
        }


@skip_no_model
def test_api_rejects_invalid_input(raw_patient):
    from fastapi.testclient import TestClient
    from src.serving.app import app
    bad = dict(raw_patient)
    bad['sex'] = 9  # out of allowed range [0, 1]
    with TestClient(app) as client:
        r = client.post('/predict', json=bad)
        assert r.status_code == 422
