"""
Data Acquisition, Cleaning & EDA - HeartCare MLOps Pipeline
Task 1: Obtain the Heart Disease UCI dataset, clean/preprocess (handle missing
values, encode features) and perform EDA with professional visualizations
(histograms, correlation heatmap, class balance).

Reference structure adapted from Source_Code/preparation/clean_and_eda.py.
"""

import os
import json
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (headless-safe)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, DOCS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'data_acquisition.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_acquisition')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


# UCI Heart Disease dataset id (used by ucimlrepo.fetch_ucirepo)
UCI_DATASET_ID = 45

# Columns that are categorical in the Heart Disease dataset.
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# Continuous numeric columns.
CONTINUOUS_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

TARGET_COL = 'target'


def download_data():
    """
    Download the Heart Disease UCI dataset using ucimlrepo.
    Falls back to a cached raw CSV if already downloaded (reproducible offline).
    """
    raw_path = os.path.join(RAW_DIR, 'heart_disease_raw.csv')

    if os.path.exists(raw_path):
        logger.info(f"Using cached raw dataset: {raw_path}")
        return pd.read_csv(raw_path)

    logger.info(f"Downloading Heart Disease dataset (UCI id={UCI_DATASET_ID})...")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        logger.error("ucimlrepo not installed. Run: pip install ucimlrepo")
        raise

    heart_disease = fetch_ucirepo(id=UCI_DATASET_ID)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    try:
        meta_path = os.path.join(RAW_DIR, 'dataset_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'metadata': str(heart_disease.metadata)}, f, indent=2)
        heart_disease.variables.to_csv(
            os.path.join(RAW_DIR, 'variable_info.csv'), index=False
        )
        logger.info("Saved dataset metadata and variable info")
    except Exception as e:
        logger.warning(f"Could not persist metadata: {e}")

    df = X.copy()
    target_name = y.columns[0]
    df[TARGET_COL] = y[target_name].values

    df.to_csv(raw_path, index=False)
    logger.info(f"Raw dataset saved: {raw_path} (shape={df.shape})")
    return df


def clean_data(df):
    """
    Clean and preprocess the heart disease data.
    - Drop duplicate rows
    - Handle missing values (median for continuous, mode for categorical)
    - Binarize the multi-class target to presence (1) / absence (0)
    """
    logger.info("Cleaning data...")
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        logger.info(f"Dropped {before - len(df)} duplicate rows")

    if TARGET_COL in df.columns:
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    for col in df.columns:
        if col == TARGET_COL:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        if col in CATEGORICAL_COLS:
            fill_val = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(fill_val)
            logger.info(f"Filled {n_missing} missing in '{col}' with mode={fill_val}")
        else:
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.info(f"Filled {n_missing} missing in '{col}' with median={fill_val}")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    logger.info(f"Data cleaned. Shape: {df.shape}")
    return df


def encode_features(df):
    """One-hot encode categorical features. Continuous columns left for scaling later."""
    logger.info("Encoding categorical features (one-hot)...")
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]

    df_encoded = pd.get_dummies(
        df, columns=cat_present, prefix=cat_present, drop_first=True
    )
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    logger.info(f"Encoded dataframe shape: {df_encoded.shape}")
    return df_encoded


def analyze_class_imbalance(df):
    """
    Analyze target class imbalance.

    Computes per-class counts, the imbalance ratio (majority:minority), a
    severity label, and the class weights a model could use to compensate.
    This is EDA/reporting only - no resampling is applied to the saved data,
    because any resampling must be fit on the training split alone (handled
    later in feature engineering/training) to avoid leakage into EDA.
    """
    counts = df[TARGET_COL].value_counts().sort_index()
    counts_dict = {int(k): int(v) for k, v in counts.items()}
    total = int(counts.sum())

    majority = int(counts.max())
    minority = int(counts.min())
    imbalance_ratio = round(majority / minority, 2) if minority else float('inf')
    minority_pct = round(minority / total * 100, 2) if total else 0.0

    # Severity heuristic based on how skewed the classes are.
    if imbalance_ratio <= 1.5:
        severity = 'balanced'
    elif imbalance_ratio <= 3:
        severity = 'mild'
    elif imbalance_ratio <= 9:
        severity = 'moderate'
    else:
        severity = 'severe'

    # Balanced class weights (sklearn-style: n_samples / (n_classes * count)).
    n_classes = len(counts_dict)
    class_weights = {
        cls: round(total / (n_classes * cnt), 3)
        for cls, cnt in counts_dict.items()
    }

    result = {
        'counts': counts_dict,
        'total': total,
        'imbalance_ratio': imbalance_ratio,
        'minority_pct': minority_pct,
        'severity': severity,
        'class_weights': class_weights,
    }

    logger.info(
        f"Class imbalance - counts={counts_dict}, "
        f"ratio={imbalance_ratio}:1, severity={severity}"
    )

    print("\n--- Class Imbalance Analysis ---")
    print(f"Class counts: {counts_dict}")
    print(f"Imbalance ratio (majority:minority): {imbalance_ratio}:1")
    print(f"Minority class share: {minority_pct}%")
    print(f"Severity: {severity}")
    print(f"Suggested balanced class weights: {class_weights}")
    if severity in ('moderate', 'severe'):
        print("Recommendation: apply class_weight='balanced' or resampling "
              "(e.g. SMOTE) on the TRAIN split during model development.")
    else:
        print("Recommendation: classes are reasonably balanced; "
              "class_weight='balanced' is a safe default but not critical.")

    return result


def do_eda(df):
    """Run EDA and save professional visualizations."""
    logger.info("Starting EDA...")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_COL].value_counts().sort_index()
    counts.plot(kind='bar', ax=ax, color=['mediumseagreen', 'indianred'])
    ax.set_title('Target Class Balance (0 = No Disease, 1 = Disease)')
    ax.set_xlabel('Heart Disease')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Absence (0)', 'Presence (1)'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_balance.png'), dpi=100)
    plt.close()
    logger.info("Saved: class_balance.png")

    # 2. Histograms for continuous features
    cont = [c for c in CONTINUOUS_COLS if c in df.columns]
    if cont:
        ncols = 3
        nrows = int(np.ceil(len(cont) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(cont):
            axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        for j in range(len(cont), len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_histograms.png'), dpi=100)
        plt.close()
        logger.info("Saved: feature_histograms.png")

    # 3. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=100)
    plt.close()
    logger.info("Saved: correlation_heatmap.png")

    # 4. Boxplots by target
    if cont:
        ncols = 3
        nrows = int(np.ceil(len(cont) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(cont):
            sns.boxplot(x=df[TARGET_COL], y=df[col], ax=axes[i],
                        hue=df[TARGET_COL], palette='Set2', legend=False)
            axes[i].set_title(f'{col} by Target')
            axes[i].set_xlabel('Heart Disease')
        for j in range(len(cont), len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_by_target_boxplots.png'), dpi=100)
        plt.close()
        logger.info("Saved: feature_by_target_boxplots.png")

    # 5. Feature-target correlation
    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    target_corr.plot(kind='barh', ax=ax, color='slateblue')
    ax.set_title('Feature Correlation with Target')
    ax.set_xlabel('Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'target_correlation.png'), dpi=100)
    plt.close()
    logger.info("Saved: target_correlation.png")

    n_rows = len(df)
    n_pos = int(df[TARGET_COL].sum())
    n_neg = n_rows - n_pos
    pos_pct = round(n_pos / n_rows * 100, 2)

    print("\n--- EDA Summary ---")
    print(f"Total records: {n_rows}")
    print(f"Features (pre-encoding): {df.shape[1] - 1}")
    print(f"Positive (disease): {n_pos} ({pos_pct}%)")
    print(f"Negative (no disease): {n_neg} ({round(100 - pos_pct, 2)}%)")

    imbalance = analyze_class_imbalance(df)

    _write_eda_report(df, target_corr, n_rows, n_pos, n_neg, pos_pct, imbalance)
    logger.info("EDA complete")


def _write_eda_report(df, target_corr, n_rows, n_pos, n_neg, pos_pct, imbalance=None):
    """Generate a markdown EDA report."""
    lines = []
    lines.append("# EDA Report - HeartCare\n")
    lines.append(f"- **Total records:** {n_rows}")
    lines.append(f"- **Features (pre-encoding):** {df.shape[1] - 1}")
    lines.append(f"- **Positive (disease present):** {n_pos} ({pos_pct}%)")
    lines.append(f"- **Negative (no disease):** {n_neg} ({round(100 - pos_pct, 2)}%)\n")

    if imbalance is not None:
        lines.append("## Class Imbalance Analysis\n")
        lines.append(f"- **Class counts:** `{imbalance['counts']}`")
        lines.append(f"- **Imbalance ratio (majority:minority):** "
                     f"{imbalance['imbalance_ratio']}:1")
        lines.append(f"- **Minority class share:** {imbalance['minority_pct']}%")
        lines.append(f"- **Severity:** {imbalance['severity']}")
        lines.append(f"- **Suggested balanced class weights:** "
                     f"`{imbalance['class_weights']}`\n")
        if imbalance['severity'] in ('moderate', 'severe'):
            lines.append("> Recommendation: apply `class_weight='balanced'` or "
                         "resampling (e.g. SMOTE) on the **training split only** "
                         "during model development to avoid leakage.\n")
        else:
            lines.append("> Classes are reasonably balanced; "
                         "`class_weight='balanced'` is a safe default.\n")

    lines.append("## Top features correlated with target\n")
    lines.append("| Feature | Correlation |")
    lines.append("|---------|-------------|")
    top = target_corr.abs().sort_values(ascending=False).head(10)
    for feat in top.index:
        lines.append(f"| {feat} | {round(float(target_corr[feat]), 3)} |")
    lines.append("")

    lines.append("## Generated plots\n")
    for p in ['class_balance.png', 'feature_histograms.png',
              'correlation_heatmap.png', 'feature_by_target_boxplots.png',
              'target_correlation.png']:
        lines.append(f"- `plots/{p}`")
    lines.append("")

    report_path = os.path.join(DOCS_DIR, 'eda_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"EDA report saved to {report_path}")


def run_acquisition():
    """Main data acquisition + cleaning + EDA routine."""
    logger.info("=" * 50)
    logger.info("STARTING DATA ACQUISITION & EDA")
    logger.info("=" * 50)

    raw = download_data()
    cleaned = clean_data(raw)
    do_eda(cleaned)
    encoded = encode_features(cleaned)

    cleaned_path = os.path.join(PROCESSED_DIR, 'heart_disease_clean.csv')
    encoded_path = os.path.join(PROCESSED_DIR, 'heart_disease_encoded.csv')
    cleaned.to_csv(cleaned_path, index=False)
    encoded.to_csv(encoded_path, index=False)
    logger.info(f"Saved cleaned data: {cleaned_path}")
    logger.info(f"Saved encoded data: {encoded_path}")

    logger.info("DATA ACQUISITION & EDA COMPLETE")
    return cleaned, encoded


if __name__ == '__main__':
    run_acquisition()
