"""
Monitoring & Logging - HeartCare MLOps Pipeline
Task 8: centralizes Prometheus metrics and structured request logging so the
serving layer stays thin and the monitoring behaviour is reusable/testable.

If prometheus_client is not installed, the metric objects degrade to no-op
stubs so the API keeps working without a hard dependency.
"""

import os
import json
import logging
from datetime import datetime

LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
)
os.makedirs(LOG_DIR, exist_ok=True)

# Structured access log (one JSON object per request) for log-based monitoring.
access_logger = logging.getLogger('access')
access_logger.setLevel(logging.INFO)
if not access_logger.handlers:
    _h = logging.FileHandler(os.path.join(LOG_DIR, 'access.log'))
    _h.setFormatter(logging.Formatter('%(message)s'))
    access_logger.addHandler(_h)


try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    CONTENT_TYPE_LATEST = 'text/plain'

    # No-op stubs so callers don't need to branch everywhere.
    class _NoopMetric:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            pass

        def observe(self, *a, **k):
            pass

        def set(self, *a, **k):
            pass

    def Counter(*a, **k):      # noqa: N802 (match prometheus API)
        return _NoopMetric()

    def Histogram(*a, **k):    # noqa: N802
        return _NoopMetric()

    def Gauge(*a, **k):        # noqa: N802
        return _NoopMetric()

    def generate_latest():
        return b'# prometheus_client not installed\n'


# ---- Metric definitions ----
REQUEST_COUNT = Counter(
    'api_requests_total', 'Total API requests',
    ['endpoint', 'method', 'status']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'Request latency in seconds', ['endpoint']
)
PREDICTION_COUNT = Counter(
    'predictions_total', 'Total predictions by label', ['label']
)
PREDICTION_ERRORS = Counter(
    'prediction_errors_total', 'Total failed predictions'
)
MODEL_READY = Gauge(
    'model_ready', '1 if the model pipeline is loaded, else 0'
)


def record_request(endpoint, method, status, elapsed_seconds, client=None):
    """Record one request to Prometheus counters and the JSON access log."""
    REQUEST_COUNT.labels(endpoint, method, str(status)).inc()
    REQUEST_LATENCY.labels(endpoint).observe(elapsed_seconds)

    entry = {
        'ts': datetime.now().isoformat(),
        'method': method,
        'endpoint': endpoint,
        'status': status,
        'latency_ms': round(elapsed_seconds * 1000, 2),
        'client': client,
    }
    access_logger.info(json.dumps(entry))


def record_prediction(label):
    """Increment the prediction counter for a given label."""
    PREDICTION_COUNT.labels(label).inc()


def record_prediction_error():
    PREDICTION_ERRORS.inc()


def set_model_ready(ready):
    MODEL_READY.set(1 if ready else 0)


def metrics_payload():
    """Return (content, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
