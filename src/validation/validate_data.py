"""
Data Validation Module - HeartCare MLOps Pipeline
Checks data quality, completeness and schema conformance for the
Heart Disease UCI dataset before it enters feature engineering/training.

Reference structure adapted from Source_Code/validation/validate_data.py.
"""

import os
import json
import logging
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
REPORT_DIR = os.path.join(BASE_DIR, 'docs')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'validation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validation')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


# Expected schema + plausible clinical ranges for the Heart Disease dataset.
# Range checks are based on documented valid encodings / realistic measurements.
SCHEMA = {
    'columns': [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ],
    'range_checks': {
        'age': (18, 100),       # patient age in years
        'sex': (0, 1),          # 0 = female, 1 = male
        'cp': (0, 4),           # chest pain type
        'trestbps': (80, 220),  # resting blood pressure (mm Hg)
        'chol': (100, 600),     # serum cholesterol (mg/dl)
        'fbs': (0, 1),          # fasting blood sugar > 120 mg/dl
        'restecg': (0, 2),      # resting ECG results
        'thalach': (60, 220),   # max heart rate achieved
        'exang': (0, 1),        # exercise induced angina
        'oldpeak': (0, 7),      # ST depression
        'slope': (0, 3),        # slope of peak exercise ST segment
        'ca': (0, 4),           # number of major vessels
        'thal': (0, 7),         # thalassemia
        'target': (0, 1),       # binarized disease presence
    },
}


def check_missing_values(df, name):
    """Check for missing/null values in each column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    issues = []
    for col in df.columns:
        if missing[col] > 0:
            issues.append({
                'column': col,
                'missing_count': int(missing[col]),
                'missing_pct': float(missing_pct[col])
            })
            logger.warning(
                f"[{name}] Column '{col}' has {missing[col]} missing "
                f"values ({missing_pct[col]}%)"
            )

    if not issues:
        logger.info(f"[{name}] No missing values found")
    return issues


def check_duplicates(df, name):
    """Check for fully duplicated patient rows."""
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        logger.warning(f"[{name}] Found {dupes} duplicate rows")
    else:
        logger.info(f"[{name}] No duplicate rows found")
    return dupes


def check_schema(df, name, expected_cols):
    """Verify the dataframe has the expected columns."""
    actual = set(df.columns)
    expected = set(expected_cols)

    missing_cols = expected - actual
    extra_cols = actual - expected

    if missing_cols:
        logger.error(f"[{name}] Missing columns: {missing_cols}")
    if extra_cols:
        logger.info(f"[{name}] Extra columns (ok): {extra_cols}")

    schema_ok = len(missing_cols) == 0
    logger.info(f"[{name}] Schema check: {'PASS' if schema_ok else 'FAIL'}")
    return schema_ok, list(missing_cols), list(extra_cols)


def check_range(df, name, col, min_val, max_val):
    """Check if values in a column fall within the expected range."""
    if col not in df.columns:
        return None
    series = df[col].dropna()
    out_of_range = series[(series < min_val) | (series > max_val)]
    count = int(len(out_of_range))

    if count > 0:
        logger.warning(
            f"[{name}] {count} records have {col} outside "
            f"range [{min_val}, {max_val}]"
        )
    else:
        logger.info(f"[{name}] Range check on '{col}': PASS")
    return count


def check_class_balance(df, name, target_col='target'):
    """Report target class balance (important for a medical classifier)."""
    if target_col not in df.columns:
        return None
    counts = df[target_col].value_counts().sort_index().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    total = sum(counts.values())
    pos = counts.get(1, 0)
    balance = {
        'counts': counts,
        'positive_pct': round(pos / total * 100, 2) if total else 0.0,
    }
    logger.info(f"[{name}] Class balance: {counts} "
                f"(positive={balance['positive_pct']}%)")
    return balance


def validate_dataset(file_path, name, schema):
    """Run all validation checks on a single dataset."""
    logger.info(f"\n--- Validating: {name} ---")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Could not read {file_path}: {e}")
        return None

    report = {
        'dataset': name,
        'file': file_path,
        'row_count': len(df),
        'col_count': len(df.columns),
        'checks': {}
    }

    # Schema check
    schema_ok, missing, extra = check_schema(df, name, schema['columns'])
    report['checks']['schema'] = {
        'passed': schema_ok,
        'missing_columns': missing,
        'extra_columns': extra
    }

    # Missing values
    missing_issues = check_missing_values(df, name)
    report['checks']['missing_values'] = {
        'passed': len(missing_issues) == 0,
        'details': missing_issues
    }

    # Duplicates
    dupe_count = check_duplicates(df, name)
    report['checks']['duplicates'] = {
        'passed': dupe_count == 0,
        'duplicate_count': dupe_count
    }

    # Range checks
    if 'range_checks' in schema:
        for col, (min_v, max_v) in schema['range_checks'].items():
            oor = check_range(df, name, col, min_v, max_v)
            if oor is not None:
                report['checks'][f'range_{col}'] = {
                    'passed': oor == 0,
                    'out_of_range_count': oor
                }

    # Class balance (informational, not a pass/fail)
    balance = check_class_balance(df, name)
    if balance is not None:
        report['class_balance'] = balance

    # Basic stats
    report['basic_stats'] = {}
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        report['basic_stats'][col] = {
            'mean': round(float(df[col].mean()), 2),
            'std': round(float(df[col].std()), 2),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }

    return report


def generate_quality_report(reports):
    """Generate a markdown quality report from validation results."""
    lines = []
    lines.append("# Data Quality Report - HeartCare\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    overall_pass = True

    for r in reports:
        if r is None:
            continue

        lines.append(f"## Dataset: {r['dataset']}\n")
        lines.append(f"- **Rows:** {r['row_count']}")
        lines.append(f"- **Columns:** {r['col_count']}")
        lines.append(f"- **File:** `{r['file']}`\n")

        lines.append("### Validation Checks\n")
        lines.append("| Check | Status |")
        lines.append("|-------|--------|")

        for check_name, check_data in r['checks'].items():
            status = "PASS" if check_data['passed'] else "FAIL"
            if not check_data['passed']:
                overall_pass = False
            lines.append(f"| {check_name} | {status} |")
        lines.append("")

        if r.get('class_balance'):
            cb = r['class_balance']
            lines.append("### Class Balance\n")
            lines.append(f"- Counts: `{cb['counts']}`")
            lines.append(f"- Positive (disease): {cb['positive_pct']}%\n")

        if r.get('basic_stats'):
            lines.append("### Basic Statistics\n")
            lines.append("| Column | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            for col, stats in r['basic_stats'].items():
                lines.append(
                    f"| {col} | {stats['mean']} | {stats['std']} | "
                    f"{stats['min']} | {stats['max']} |"
                )
            lines.append("")

        lines.append("---\n")

    status_text = ("ALL CHECKS PASSED" if overall_pass
                   else "SOME CHECKS FAILED - Review above")
    lines.append(f"## Overall Status: {status_text}\n")

    return '\n'.join(lines), overall_pass


def run_validation():
    """Main validation entry point."""
    logger.info("=" * 50)
    logger.info("STARTING DATA VALIDATION")
    logger.info("=" * 50)

    # Validate both the raw download and the cleaned output if present.
    candidates = {
        'heart_disease_raw': os.path.join(RAW_DIR, 'heart_disease_raw.csv'),
        'heart_disease_clean': os.path.join(PROCESSED_DIR, 'heart_disease_clean.csv'),
    }

    reports = []
    for name, path in candidates.items():
        if not os.path.exists(path):
            logger.warning(f"Skipping '{name}' - file not found: {path}")
            continue
        reports.append(validate_dataset(path, name, SCHEMA))

    if not reports:
        logger.error("No datasets available to validate. "
                     "Run data acquisition first.")
        return []

    # save json report
    json_path = os.path.join(REPORT_DIR, 'data_quality_report.json')
    with open(json_path, 'w') as f:
        json.dump([r for r in reports if r], f, indent=2)
    logger.info(f"JSON report saved to {json_path}")

    # save markdown report
    md_report, overall_pass = generate_quality_report(reports)
    md_path = os.path.join(REPORT_DIR, 'data_quality_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    logger.info(f"Markdown report saved to {md_path}")

    logger.info(f"VALIDATION COMPLETE - overall={'PASS' if overall_pass else 'FAIL'}")
    return reports


if __name__ == '__main__':
    run_validation()
