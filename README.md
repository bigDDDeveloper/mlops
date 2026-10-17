# HeartCare - MLOps Pipeline for Heart Disease Risk Prediction

**Course:** Machine Learning Operations (MLOps) AIMLCZG523 - Assignment 01
End-to-End ML Model Development, CI/CD, and Production Deployment.

Predicts heart disease risk from patient clinical data and serves it as a
cloud-ready, monitored API.

**Dataset:** Heart Disease UCI (UCI ML Repository, id=45) - 13 clinical features
+ a binary target (disease present / absent).

---

## Quick Start (Linux / macOS)

From a fresh machine with Python 3.10+ and git:

```bash
# 1. One-time setup: creates .venv and installs requirements
bash setup.sh

# 2. Activate the environment
source .venv/bin/activate

# 3. Run the full pipeline (acquire -> validate -> features -> train -> package)
bash run_pipeline.sh
```

That produces the trained model, EDA/model reports, and plots. To serve the API:

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

### Windows (PowerShell) equivalent

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.orchestration.pipeline
```

---

## Moving this project to another machine

Copy the project **source** only - the generated folders regenerate from the
pipeline and are excluded via `.gitignore`.

Do copy:
```
src/  tests/  docker/  k8s/  monitoring/  .github/
requirements.txt  setup.cfg  setup.sh  run_pipeline.sh
docker-compose.yml  .dockerignore  .gitignore  README.md
```

Do NOT copy (regenerated, machine-specific):
```
.venv/  __pycache__/  .pytest_cache/
data/  logs/  models/  plots/  mlruns/  mlflow.db
```

Easiest transfer options:
- **Git:** `git init && git add . && git commit -m "init"` then push; the
  `.gitignore` already keeps generated files out.
- **Archive:** create a tarball excluding generated dirs:
  ```bash
  tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
      --exclude='data' --exclude='logs' --exclude='models' --exclude='plots' \
      --exclude='mlruns' --exclude='mlflow.db' \
      -czf heartcare.tar.gz .
  ```

On the new machine, extract, then run `bash setup.sh && source .venv/bin/activate
&& bash run_pipeline.sh`. The dataset downloads automatically via `ucimlrepo` on
first run (needs internet), then caches to `data/raw/`.

---

## Project Structure

```
MLOP/
├── src/
│   ├── data_acquisition/   # download + clean + encode + EDA (Task 1)
│   ├── validation/         # schema / quality checks
│   ├── feature_engineering/# scaling + train/test split (Task 2)
│   ├── training/           # LogReg + RandomForest + MLflow (Tasks 2, 3)
│   ├── packaging/          # single reproducible pipeline artifact (Task 4)
│   ├── serving/            # FastAPI /predict API (Task 6)
│   ├── monitoring/         # Prometheus metrics + access logs (Task 8)
│   └── orchestration/      # runs all stages in order
├── tests/                  # pytest unit tests (Task 5)
├── docker/                 # Dockerfile + serving requirements (Task 6)
├── k8s/                    # manifests + Helm chart (Task 7)
├── monitoring/             # Prometheus + Grafana configs (Task 8)
├── .github/workflows/      # GitHub Actions CI (Task 5)
├── requirements.txt
├── setup.sh / run_pipeline.sh
└── docker-compose.yml      # API + Prometheus + Grafana
```

## Pipeline Flow

```
Acquire + EDA -> Validate -> Feature Engineering -> Train + Track -> Package -> Serve -> Monitor
```

---

## Containerized stack (API + monitoring)

```bash
docker compose up --build
# API        http://localhost:8000/docs
# Prometheus http://localhost:9090
# Grafana    http://localhost:3000  (admin / admin)
```

## Kubernetes

See `k8s/README.md` for Minikube / Helm / cloud deployment commands.

---

## Reproducibility notes

- All randomness is seeded (`random_state=42`).
- `requirements.txt` lists every dependency the code imports; a clean
  `pip install -r requirements.txt` is sufficient to run everything.
- `mlflow`, `fastapi`, and `prometheus-client` are only needed for the tracking,
  serving, and monitoring stages respectively; the core training pipeline
  degrades gracefully if they are absent.
```
