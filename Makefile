# Neuro-Swarm Makefile
# Convenience commands for development and deployment

.PHONY: help install test lint format build-images deploy clean

REGISTRY ?= localhost:5000
VERSION ?= latest
NAMESPACE ?= neuro-swarm

help:
	@echo "Neuro-Swarm Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Studies:"
	@echo "  make study-01     - Run single agent study"
	@echo "  make study-02     - Run pair dynamics study"
	@echo ""
	@echo "Docker:"
	@echo "  make build-images - Build Docker images"
	@echo "  make push-images  - Push to registry"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make deploy       - Deploy to k8s (raw manifests)"
	@echo "  make deploy-helm  - Deploy via Helm"
	@echo "  make status       - Show deployment status"
	@echo "  make logs         - Tail controller logs"
	@echo "  make scale N=10   - Scale workers"
	@echo "  make clean        - Remove deployment"
	@echo ""
	@echo "Evolution:"
	@echo "  make evolution-status - Get evolution status"
	@echo "  make evolution-best   - Get best genome"

# ==================== Development ====================

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check neuro_swarm/
	mypy neuro_swarm/

format:
	black .
	ruff check --fix .

# ==================== Studies ====================

study-01:
	python -m neuro_swarm.studies.01_single_agent.observe

study-02:
	python -m neuro_swarm.studies.02_pair_dynamics.observe

study-02-pursuer:
	python -m neuro_swarm.studies.02_pair_dynamics.observe --scenario pursuer_evader

study-02-opposed:
	python -m neuro_swarm.studies.02_pair_dynamics.observe --scenario opposite_goals

study-03:
	python -m neuro_swarm.studies.03_triad_consensus.observe

study-04:
	python -m neuro_swarm.studies.04_small_swarm.observe --agents 7

# ==================== Docker ====================

build-images:
	docker build -t $(REGISTRY)/neuro-swarm/controller:$(VERSION) \
		-f services/controller/Dockerfile .
	docker build -t $(REGISTRY)/neuro-swarm/worker:$(VERSION) \
		-f services/worker/Dockerfile .

push-images:
	docker push $(REGISTRY)/neuro-swarm/controller:$(VERSION)
	docker push $(REGISTRY)/neuro-swarm/worker:$(VERSION)

# ==================== Kubernetes ====================

deploy:
	kubectl apply -f deploy/k8s/all-in-one.yaml

deploy-helm:
	helm repo add bitnami https://charts.bitnami.com/bitnami || true
	helm dependency update deploy/helm
	helm upgrade --install neuro-swarm deploy/helm \
		-n $(NAMESPACE) --create-namespace

status:
	@echo "=== Pods ==="
	kubectl -n $(NAMESPACE) get pods
	@echo ""
	@echo "=== HPA ==="
	kubectl -n $(NAMESPACE) get hpa
	@echo ""
	@echo "=== Services ==="
	kubectl -n $(NAMESPACE) get svc

logs:
	kubectl -n $(NAMESPACE) logs -f deployment/evolution-controller

logs-workers:
	kubectl -n $(NAMESPACE) logs -f -l app=evolution-worker --max-log-requests=10

scale:
	kubectl -n $(NAMESPACE) scale deployment/evolution-worker --replicas=$(N)

port-forward:
	kubectl -n $(NAMESPACE) port-forward svc/evolution-controller 8080:8080

clean:
	kubectl delete namespace $(NAMESPACE) --ignore-not-found

# ==================== Evolution API ====================

evolution-status:
	@curl -s http://localhost:8080/status | python -m json.tool

evolution-best:
	@curl -s http://localhost:8080/best | python -m json.tool

evolution-history:
	@curl -s http://localhost:8080/history | python -m json.tool

evolution-archive:
	@curl -s http://localhost:8080/archive | python -m json.tool

# ==================== Local Testing ====================

redis-local:
	docker run -d --name redis-local -p 6379:6379 redis:7-alpine

redis-stop:
	docker stop redis-local && docker rm redis-local

run-controller-local:
	REDIS_HOST=localhost python -m services.controller.main

run-worker-local:
	REDIS_HOST=localhost python -m services.worker.main

# ==================== Experiments ====================

experiment-quick:
	helm upgrade --install neuro-swarm deploy/helm \
		-n $(NAMESPACE) --create-namespace \
		--set controller.algorithm=es \
		--set controller.populationSize=20 \
		--set controller.maxGenerations=100 \
		--set worker.replicaCount=3

experiment-map-elites:
	helm upgrade --install neuro-swarm deploy/helm \
		-n $(NAMESPACE) --create-namespace \
		--set controller.algorithm=map_elites \
		--set controller.populationSize=100 \
		--set controller.maxGenerations=2000 \
		--set worker.replicaCount=15

experiment-novelty:
	helm upgrade --install neuro-swarm deploy/helm \
		-n $(NAMESPACE) --create-namespace \
		--set controller.algorithm=novelty \
		--set controller.populationSize=50 \
		--set controller.maxGenerations=1000 \
		--set worker.replicaCount=10
