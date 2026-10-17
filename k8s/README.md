# HeartCare API - Kubernetes Deployment (Task 7)

Deploy the containerized prediction API to Kubernetes, exposed via a
LoadBalancer (or Ingress). Two options are provided: raw manifests and a Helm
chart. Capture screenshots of each verification step for the report (Task 9).

## Files

| File | Purpose |
|------|---------|
| `deployment.yaml` | Deployment: 2 replicas, `/health` liveness + readiness probes |
| `service.yaml` | LoadBalancer Service (port 80 -> container 8000) |
| `hpa.yaml` | HorizontalPodAutoscaler (2-6 pods @ 70% CPU) |
| `ingress.yaml` | Optional Ingress (alternative to LoadBalancer) |
| `helm/heartcare/` | Equivalent Helm chart |

---

## Prerequisites

Build the image first (from the `MLOP` project root):

```bash
docker build -f docker/Dockerfile -t heartcare-api:latest .
```

> The build requires `models/heart_disease_pipeline.pkl`. Generate it by running
> the pipeline: acquisition -> feature engineering -> training -> packaging.

---

## Option A: Minikube (local)

```bash
minikube start
# Make the locally-built image available inside the cluster:
minikube image load heartcare-api:latest

# Deploy with raw manifests:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify:
kubectl get pods
kubectl get svc heartcare-api
kubectl get hpa

# Expose the LoadBalancer service and get a URL:
minikube service heartcare-api --url
# (or: minikube tunnel  -- in a separate terminal, then use the EXTERNAL-IP)
```

Test the endpoint (replace URL with the one printed above):

```bash
curl <URL>/health
curl -X POST <URL>/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

## Option B: Helm

```bash
helm install heartcare k8s/helm/heartcare
# or override values:
helm install heartcare k8s/helm/heartcare \
  --set image.tag=1.0.0 --set service.type=LoadBalancer

helm status heartcare
kubectl get all -l app=heartcare-heartcare
```

Upgrade / uninstall:

```bash
helm upgrade heartcare k8s/helm/heartcare --set replicaCount=3
helm uninstall heartcare
```

## Option C: Cloud (GKE / EKS / AKS)

1. Push the image to a registry:
   ```bash
   docker tag heartcare-api:latest <registry>/<user>/heartcare-api:1.0.0
   docker push <registry>/<user>/heartcare-api:1.0.0
   ```
2. Update `image` in `deployment.yaml` (or `--set image.repository=...` for Helm).
3. `kubectl apply -f k8s/` then `kubectl get svc heartcare-api -w` and wait for
   the `EXTERNAL-IP` to be assigned by the cloud load balancer.

---

## Screenshots to capture for the report

- `kubectl get pods` showing pods Running
- `kubectl get svc` showing the EXTERNAL-IP / URL
- A successful `curl .../predict` response
- `kubectl get hpa` showing the autoscaler
