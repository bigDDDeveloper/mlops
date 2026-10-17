# HeartCare Risk Prediction API - serving container (Task 6)
# Build context is the MLOP project root:
#   docker build -f docker/Dockerfile -t heartcare-api:latest .
# Run:
#   docker run -p 8000:8000 heartcare-api:latest

FROM python:3.10-slim

# Keep Python lean and logs unbuffered for container-friendly logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install runtime dependencies first (better layer caching).
COPY docker/requirements-serving.txt ./requirements-serving.txt
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy the application source and the trained model artifacts.
# The packaged pipeline (models/heart_disease_pipeline.pkl) must exist - it is
# produced by running the training + packaging stages before the build.
COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user for safety.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container healthcheck hits the readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else 1)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
