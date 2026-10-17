"""
Pipeline Orchestration - HeartCare MLOps Pipeline
Runs the full data -> model path in order, with per-stage logging and a clear
failure point. Cross-platform (no shell assumptions) so it runs identically on
Linux, macOS and Windows.

Usage (from the MLOP project root):
    python -m src.orchestration.pipeline
or:
    python src/orchestration/pipeline.py
"""

import os
import sys
import time
import logging

# Make the project root importable whether run as a module or a script.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


def _run_stage(name, func):
    """Run one pipeline stage, timing it and surfacing failures clearly."""
    logger.info("=" * 60)
    logger.info(f"STAGE: {name}")
    logger.info("=" * 60)
    start = time.time()
    try:
        func()
    except Exception as e:
        logger.error(f"STAGE FAILED: {name} -> {e}")
        raise
    elapsed = time.time() - start
    logger.info(f"STAGE OK: {name} ({elapsed:.1f}s)")


def run_pipeline():
    """Execute all stages in dependency order."""
    logger.info("####### HEARTCARE PIPELINE START #######")
    overall_start = time.time()

    # Imported lazily so a failure in one module doesn't block importing others.
    from src.data_acquisition.acquire_data import run_acquisition
    from src.validation.validate_data import run_validation
    from src.feature_engineering.build_features import run_feature_engineering
    from src.training.train_model import run_training
    from src.packaging.package_model import run_packaging

    _run_stage("Data Acquisition & EDA", run_acquisition)
    # Validation is informational - it reports raw-vs-clean quality but must not
    # abort the pipeline, so swallow its return without failing on raw findings.
    _run_stage("Data Validation", run_validation)
    _run_stage("Feature Engineering", run_feature_engineering)
    _run_stage("Model Training & Tracking", run_training)
    _run_stage("Model Packaging", run_packaging)

    total = time.time() - overall_start
    logger.info("####### HEARTCARE PIPELINE COMPLETE #######")
    logger.info(f"Total runtime: {total:.1f}s")
    logger.info("Artifacts: models/heart_disease_pipeline.pkl, "
                "models/model_card.json, docs/*.md, plots/*.png")


if __name__ == '__main__':
    run_pipeline()
