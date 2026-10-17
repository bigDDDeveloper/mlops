"""
Model Development & Training - HeartCare MLOps Pipeline
Task 2 (part 2): Build and train at least two classifiers (Logistic Regression
and Random Forest), tune via cross-validated grid search, and evaluate with
accuracy, precision, recall, F1 and ROC-AUC.
Task 3: Experiment tracking via MLflow (optional - falls back to JSON tracking
if MLflow is not installed).
Task 4: Persist the best model + preprocessing artifacts for reproducibility.

Reference structure adapted from Source_Code/training/train_model.py.
"""

import os
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless-safe backend for artifact plots
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots', 'runs')
MLRUNS_DIR = os.path.join(BASE_DIR, 'mlruns')

for d in [MODELS_DIR, DOCS_DIR, LOG_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('training')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

TARGET_COL = 'target'
RANDOM_STATE = 42
CV_FOLDS = 5

# Optional MLflow experiment tracking.
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    # MLflow 3.x deprecated the plain file store. Use a local SQLite backend
    # for run metadata and keep artifacts under ./mlruns so the experiment is
    # self-contained and portable with the repo.
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    _mlflow_db = os.path.join(BASE_DIR, 'mlflow.db').replace('\\', '/')
    mlflow.set_tracking_uri('sqlite:///' + _mlflow_db)
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed - falling back to JSON experiment tracking")


# Candidate models and their hyperparameter grids. Documented tuning strategy:
# - Logistic Regression: regularization strength C and penalty type, with
#   balanced class weights as a safe default for a medical classifier.
# - Random Forest: tree count, depth and leaf constraints to control overfit.
MODEL_CONFIGS = {
    'logistic_regression': {
        'estimator': LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'param_grid': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
        },
    },
    'random_forest': {
        'estimator': RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
    },
}


def load_splits():
    """Load the train/test splits produced by feature engineering."""
    train_path = os.path.join(FEATURES_DIR, 'train.csv')
    test_path = os.path.join(FEATURES_DIR, 'test.csv')
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run feature engineering first."
            )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL].astype(int)
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].astype(int)
    logger.info(f"Loaded splits - train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def evaluate(model, X_test, y_test):
    """Compute evaluation metrics on the held-out test set."""
    y_pred = model.predict(X_test)
    # probability for the positive class (for ROC-AUC)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    return metrics, cm, y_pred, y_proba


def make_run_plots(name, y_test, y_pred, y_proba):
    """
    Create per-run confusion matrix and ROC curve plots.
    Returns the list of saved file paths (logged as MLflow artifacts).
    """
    paths = []

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center',
                color='white' if v > cm.max() / 2 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, f'{name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=100)
    plt.close()
    paths.append(cm_path)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_title(f'ROC Curve - {name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, f'{name}_roc_curve.png')
    plt.savefig(roc_path, dpi=100)
    plt.close()
    paths.append(roc_path)

    logger.info(f"[{name}] Saved run plots: {[os.path.basename(p) for p in paths]}")
    return paths


def train_one(name, config, X_train, y_train, X_test, y_test):
    """Train, tune (grid search CV) and evaluate a single model."""
    logger.info(f"\n--- Training: {name} ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Hyperparameter tuning via cross-validated grid search, optimizing ROC-AUC.
    search = GridSearchCV(
        config['estimator'], config['param_grid'],
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    logger.info(f"[{name}] Best params: {search.best_params_}")
    logger.info(f"[{name}] Best CV ROC-AUC: {round(search.best_score_, 4)}")

    # Cross-validated accuracy of the tuned model (for the report).
    cv_acc = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')

    # Held-out test metrics.
    metrics, cm, y_pred, y_proba = evaluate(best_model, X_test, y_test)
    logger.info(f"[{name}] Test metrics: {metrics}")

    # Per-run plots (logged as MLflow artifacts).
    plot_paths = make_run_plots(name, y_test, y_pred, y_proba)

    result = {
        'model_name': name,
        'best_params': search.best_params_,
        'cv_roc_auc_mean': round(float(search.best_score_), 4),
        'cv_accuracy_mean': round(float(cv_acc.mean()), 4),
        'cv_accuracy_std': round(float(cv_acc.std()), 4),
        'test_metrics': metrics,
        'confusion_matrix': cm,
        'plot_paths': plot_paths,
    }
    return best_model, result


def track_experiment(name, result, model=None):
    """
    Log an experiment run to MLflow (if available) and always to JSON.
    Logs parameters, metrics, plot artifacts and the trained model.
    """
    if MLFLOW_AVAILABLE:
        try:
            exp_name = 'heart_disease_classification'
            if mlflow.get_experiment_by_name(exp_name) is None:
                artifact_uri = 'file:///' + MLRUNS_DIR.replace('\\', '/')
                mlflow.create_experiment(exp_name, artifact_location=artifact_uri)
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=name):
                # Parameters
                mlflow.log_params(result['best_params'])
                mlflow.set_tag('model_type', name)

                # Metrics (CV + held-out test)
                mlflow.log_metric('cv_roc_auc', result['cv_roc_auc_mean'])
                mlflow.log_metric('cv_accuracy', result['cv_accuracy_mean'])
                for k, v in result['test_metrics'].items():
                    mlflow.log_metric(f'test_{k}', v)

                # Plot artifacts (confusion matrix + ROC curve)
                for p in result.get('plot_paths', []):
                    if os.path.exists(p):
                        mlflow.log_artifact(p, artifact_path='plots')

                # Model artifact. Use the cloudpickle serialization format so
                # tree-based models (RandomForest) are stored without tripping
                # skops' untrusted-types guard.
                if model is not None:
                    try:
                        mlflow.sklearn.log_model(
                            model, name='model',
                            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                        )
                    except Exception as me:
                        logger.warning(f"[{name}] Model logging failed: {me}")
            logger.info(f"[{name}] Logged run to MLflow (params, metrics, "
                        "plots, model)")
        except Exception as e:
            logger.warning(f"[{name}] MLflow logging failed: {e}")

    # JSON experiment tracking (always on, for reproducibility without MLflow)
    exp_path = os.path.join(MODELS_DIR, 'experiments.json')
    history = []
    if os.path.exists(exp_path):
        try:
            with open(exp_path) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append({'timestamp': datetime.now().isoformat(), **result})
    with open(exp_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def select_best(results):
    """Select the best model by test ROC-AUC (primary medical metric)."""
    best = max(results, key=lambda r: r['test_metrics']['roc_auc'])
    logger.info(f"Selected best model: {best['model_name']} "
                f"(ROC-AUC={best['test_metrics']['roc_auc']})")
    return best


def write_model_report(results, best, y_test):
    """Write a markdown model report documenting selection and metrics."""
    lines = []
    lines.append("# Model Development Report - HeartCare\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Cross-validation:** {CV_FOLDS}-fold stratified, "
                 "grid search optimizing ROC-AUC\n")

    lines.append("## Model Comparison\n")
    lines.append("| Model | CV ROC-AUC | Test Acc | Precision | Recall | F1 | Test ROC-AUC |")
    lines.append("|-------|-----------|----------|-----------|--------|----|--------------|")
    for r in results:
        m = r['test_metrics']
        lines.append(
            f"| {r['model_name']} | {r['cv_roc_auc_mean']} | {m['accuracy']} "
            f"| {m['precision']} | {m['recall']} | {m['f1']} | {m['roc_auc']} |"
        )
    lines.append("")

    lines.append("## Tuning Details\n")
    for r in results:
        lines.append(f"### {r['model_name']}\n")
        lines.append(f"- Best params: `{r['best_params']}`")
        lines.append(f"- CV accuracy: {r['cv_accuracy_mean']} "
                     f"(+/- {r['cv_accuracy_std']})")
        lines.append(f"- Confusion matrix: `{r['confusion_matrix']}`\n")

    lines.append("## Selected Model\n")
    lines.append(f"**{best['model_name']}** was selected based on the highest "
                 f"test ROC-AUC ({best['test_metrics']['roc_auc']}). ROC-AUC is "
                 "prioritized because, for a medical risk classifier, ranking "
                 "patients by risk and controlling false negatives matters more "
                 "than raw accuracy.\n")

    report_path = os.path.join(DOCS_DIR, 'model_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Model report saved to {report_path}")


def run_training():
    """Main training entry point."""
    logger.info("=" * 50)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 50)

    X_train, y_train, X_test, y_test = load_splits()

    trained = {}
    results = []
    for name, config in MODEL_CONFIGS.items():
        model, result = train_one(name, config, X_train, y_train, X_test, y_test)
        trained[name] = model
        results.append(result)
        track_experiment(name, result, model)

    best = select_best(results)
    best_model = trained[best['model_name']]

    # Persist the best model (Task 4: reusable format).
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Saved best model: {model_path}")

    # Save a pointer file describing the selected model.
    with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w') as f:
        json.dump({
            'model_name': best['model_name'],
            'best_params': best['best_params'],
            'test_metrics': best['test_metrics'],
            'saved_at': datetime.now().isoformat(),
        }, f, indent=2)

    write_model_report(results, best, y_test)

    logger.info("MODEL TRAINING COMPLETE")
    return results, best


if __name__ == '__main__':
    run_training()
