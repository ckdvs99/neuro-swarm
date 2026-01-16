# Neuro-Swarm Base Image
# Build with: docker build -t neuro-swarm:latest .
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml ./
COPY neuro_swarm/ ./neuro_swarm/

# Install the package with redis support
RUN pip install --no-cache-dir -e . redis

# Default environment
ENV REDIS_HOST=redis \
    REDIS_PORT=6379 \
    PYTHONUNBUFFERED=1

# The CMD is specified in docker-compose or k8s manifests
CMD ["python", "-c", "print('Specify command: controller or worker')"]
