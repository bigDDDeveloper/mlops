# Monitoring on Kubernetes - Prometheus + Grafana (Task 8)

Deploys Prometheus and Grafana into the same Minikube cluster as the API, with
Prometheus scraping `heartcare-api:80/metrics` and Grafana auto-provisioned with
the HeartCare dashboard.

## Deploy

Run from the project root (where `monitoring/grafana/.../heartcare_dashboard.json`
lives). The dashboard JSON is loaded into a ConfigMap from the existing file:

```bash
# 1. Prometheus (ConfigMap + Deployment + Service)
kubectl apply -f k8s/monitoring/prometheus.yaml

# 2. Create the dashboard ConfigMap from the existing dashboard JSON
kubectl create configmap grafana-dashboard-json \
  --from-file=heartcare_dashboard.json=monitoring/grafana/provisioning/dashboards/heartcare_dashboard.json

# 3. Grafana (datasource + dashboard provider + Deployment + Service)
kubectl apply -f k8s/monitoring/grafana.yaml

# 4. Check everything is running
kubectl get pods -l app=prometheus
kubectl get pods -l app=grafana
```

> The API must already be deployed (`k8s/deployment.yaml` + `k8s/service.yaml`)
> so Prometheus can resolve and scrape `heartcare-api`.

## View the dashboards

Both services are `NodePort`. On Minikube, get a browsable URL with:

```bash
minikube service prometheus --url   # Prometheus UI
minikube service grafana --url      # Grafana UI
```

- **Prometheus** (the URL from above): go to **Status -> Targets**; the
  `heartcare-api` target should be **UP**. Try a query like `predictions_total`
  or `rate(api_requests_total[1m])`.
- **Grafana** (the URL from above): log in with **admin / admin**. Open the
  dashboard list -> **"HeartCare API Monitoring"**. Panels: model status,
  request/prediction counts, request rate, p95 latency, predictions by label,
  status codes.

### Generate traffic so panels show data

```bash
URL=$(minikube service heartcare-api --url)
for i in $(seq 1 30); do
  curl -s -X POST "$URL/predict" -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' > /dev/null
done
```

Refresh Grafana (top-right time range -> last 15m) and the panels populate.

## Alternative: no Kubernetes, just docker-compose

If you only need the screenshots and not K8s, the simpler path is from the
project root:

```bash
docker compose up --build
# Prometheus -> http://localhost:9090
# Grafana    -> http://localhost:3000  (admin / admin)
```
