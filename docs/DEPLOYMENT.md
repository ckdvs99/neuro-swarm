# Deployment Guide: Distributed Evolutionary Computing

This guide covers deploying the neuro-swarm evolution system on Kubernetes (k3s).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                       │
│                                                                   │
│  ┌─────────────┐     ┌───────────────────────────────────────┐  │
│  │  Controller │     │            Worker Pool                 │  │
│  │             │     │  ┌────────┐ ┌────────┐ ┌────────┐     │  │
│  │  - ask()    │────▶│  │Worker 1│ │Worker 2│ │Worker N│     │  │
│  │  - tell()   │◀────│  │  Eval  │ │  Eval  │ │  Eval  │     │  │
│  │  - API      │     │  └────────┘ └────────┘ └────────┘     │  │
│  └─────────────┘     │           (Autoscaling)                │  │
│         │            └───────────────────────────────────────┘  │
│         │                           │                            │
│         └───────────┬───────────────┘                           │
│                     │                                            │
│              ┌──────▼──────┐                                    │
│              │    Redis    │                                    │
│              │  Task Queue │                                    │
│              │  Results    │                                    │
│              └─────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Kubernetes cluster (k3s recommended)
- kubectl configured
- Docker for building images
- Helm 3 (optional, for Helm deployment)

## Quick Start

### Option 1: Raw Kubernetes Manifests (Simplest)

```bash
# Build images
make build-images

# Deploy
kubectl apply -f deploy/k8s/all-in-one.yaml

# Check status
kubectl -n neuro-swarm get pods

# View logs
kubectl -n neuro-swarm logs -f deployment/evolution-controller

# Scale workers
kubectl -n neuro-swarm scale deployment/evolution-worker --replicas=10

# Access API
kubectl -n neuro-swarm port-forward svc/evolution-controller 8080:8080
# Then visit http://localhost:8080/status
```

### Option 2: Helm Chart (Recommended for Production)

```bash
# Add Redis dependency
helm repo add bitnami https://charts.bitnami.com/bitnami
helm dependency update deploy/helm

# Install with default values
helm install neuro-swarm deploy/helm -n neuro-swarm --create-namespace

# Install with custom experiment
helm install neuro-swarm deploy/helm \
  -n neuro-swarm --create-namespace \
  --set controller.algorithm=map_elites \
  --set controller.populationSize=100 \
  --set worker.replicaCount=10

# Upgrade
helm upgrade neuro-swarm deploy/helm -n neuro-swarm

# Uninstall
helm uninstall neuro-swarm -n neuro-swarm
```

## Configuration

### Environment Variables

**Controller:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ALGORITHM` | `map_elites` | `es`, `map_elites`, or `novelty` |
| `POPULATION_SIZE` | `50` | Genomes per generation |
| `MAX_GENERATIONS` | `1000` | When to stop |
| `SIMULATION_STEPS` | `300` | Steps per swarm simulation |

**Worker:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SIMULATION_STEPS` | `300` | Steps per evaluation |
| `WORKER_ID` | auto | Set by k8s pod name |

### Scaling

Workers scale based on CPU utilization. Tune for your hardware:

```yaml
# In values.yaml or via --set
worker:
  autoscaling:
    minReplicas: 3
    maxReplicas: 50  # Match your cluster capacity
    targetCPUUtilizationPercentage: 70
```

For your setup (32-core Threadripper + 2 Titans), you could run:
- 20-30 CPU workers for standard evolution
- GPU workers for neural network training (future)

## Monitoring

### API Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Current status
curl http://localhost:8080/status

# Evolution history
curl http://localhost:8080/history

# Best genome found
curl http://localhost:8080/best

# MAP-Elites archive (if using MAP-Elites)
curl http://localhost:8080/archive
```

### Logs

```bash
# Controller logs
kubectl -n neuro-swarm logs -f deployment/evolution-controller

# All worker logs
kubectl -n neuro-swarm logs -f -l app=evolution-worker --max-log-requests=20

# Specific worker
kubectl -n neuro-swarm logs -f evolution-worker-xxxxx
```

### Redis Inspection

```bash
# Connect to Redis
kubectl -n neuro-swarm exec -it deployment/redis -- redis-cli

# Check queue lengths
LLEN evolution:tasks
LLEN evolution:results

# View workers
HGETALL evolution:workers

# View checkpoint
GET evolution:checkpoint
```

## Experiments

Pre-defined experiments in `values.yaml`:

```bash
# Quick test (small population, few generations)
helm upgrade neuro-swarm deploy/helm \
  --set controller.algorithm=es \
  --set controller.populationSize=20 \
  --set controller.maxGenerations=100

# Full MAP-Elites exploration
helm upgrade neuro-swarm deploy/helm \
  --set controller.algorithm=map_elites \
  --set controller.populationSize=100 \
  --set controller.maxGenerations=2000 \
  --set worker.replicaCount=15

# Novelty search for behavioral diversity
helm upgrade neuro-swarm deploy/helm \
  --set controller.algorithm=novelty \
  --set controller.populationSize=50 \
  --set controller.maxGenerations=1000
```

## Troubleshooting

### Workers not starting

```bash
# Check pod status
kubectl -n neuro-swarm describe pod evolution-worker-xxxxx

# Check image pull
kubectl -n neuro-swarm get events --sort-by='.lastTimestamp'
```

### Tasks not being processed

```bash
# Check Redis connectivity
kubectl -n neuro-swarm exec -it deployment/evolution-worker-xxxxx -- \
  python -c "import redis; r=redis.Redis(host='redis'); print(r.ping())"

# Check queue
kubectl -n neuro-swarm exec -it deployment/redis -- redis-cli LLEN evolution:tasks
```

### Controller crashes

```bash
# Check logs for errors
kubectl -n neuro-swarm logs deployment/evolution-controller --previous

# Check resources
kubectl -n neuro-swarm top pods
```

## Development Workflow

1. Make code changes
2. Rebuild images: `make build-images`
3. Push to registry (if using remote cluster)
4. Restart deployments:
   ```bash
   kubectl -n neuro-swarm rollout restart deployment/evolution-controller
   kubectl -n neuro-swarm rollout restart deployment/evolution-worker
   ```

## GPU Support (Future)

For neural network evaluation, add GPU workers:

```yaml
# In values.yaml
gpuWorker:
  enabled: true
  replicaCount: 2
  resources:
    limits:
      nvidia.com/gpu: 1
```

Requires NVIDIA device plugin installed on your cluster.

---

## Your Setup

For your 256GB RAM / 32-core Threadripper / 2x Titan system running k3s:

**Recommended initial configuration:**
```bash
helm install neuro-swarm deploy/helm -n neuro-swarm --create-namespace \
  --set controller.algorithm=map_elites \
  --set controller.populationSize=50 \
  --set worker.replicaCount=20 \
  --set worker.autoscaling.maxReplicas=30 \
  --set redis.master.resources.limits.memory=2Gi
```

This will:
- Run MAP-Elites for quality-diversity
- 20 parallel evaluations (using ~20 cores)
- Scale up to 30 workers under load
- Leave resources for other k3s workloads

Monitor and adjust based on actual performance!
