# Production Deployment Guide

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Database Setup](#database-setup)
- [Monitoring Setup](#monitoring-setup)
- [Security Hardening](#security-hardening)
- [Scaling Guidelines](#scaling-guidelines)
- [Backup & Recovery](#backup--recovery)
- [Maintenance](#maintenance)

---

## Prerequisites

### Required Services

- **PostgreSQL 16+**: Main data store
- **Redis 7+**: Cache, broker, circuit breaker state
- **Docker 24+**: Container runtime
- **Docker Compose 2.20+** OR **Kubernetes 1.28+**: Orchestration

### Required Secrets

Generate and store these in a secure vault (e.g., AWS Secrets Manager, HashiCorp Vault):

```bash
# Generate JWT secret (32+ characters)
openssl rand -hex 32

# Database credentials
DB_USER=content_automation_user
DB_PASSWORD=$(openssl rand -base64 32)
DB_NAME=content_automation_prod

# LLM API keys
OPENAI_API_KEY=sk-...  # From OpenAI dashboard
ANTHROPIC_API_KEY=sk-ant-...  # From Anthropic console
```

---

## Docker Compose Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/TensorScholar/content-automation-pipeline.git
cd content-automation-pipeline
```

### Step 2: Configure Environment

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql+asyncpg://content_user:$(openssl rand -base64 32)@postgres:5432/content_automation

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Monitoring (optional)
PROMETHEUS_PUSHGATEWAY_URL=http://pushgateway:9091
EOF

# Secure .env file
chmod 600 .env
```

### Step 3: Initialize Database

```bash
# Start PostgreSQL only
docker-compose -f docker-compose.prod.yml up -d postgres

# Wait for PostgreSQL to be ready
sleep 10

# Run migrations
docker-compose -f docker-compose.prod.yml run --rm api alembic upgrade head

# Seed initial data (optional)
docker-compose -f docker-compose.prod.yml run --rm api python scripts/seed_best_practices.py
```

### Step 4: Start All Services

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Verify services are running
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","database":"connected","redis":"connected","celery":"active_workers:2"}

# Metrics endpoint
curl http://localhost:8000/metrics

# API docs
open http://localhost:8000/docs
```

---

## Kubernetes Deployment

### Step 1: Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: content-automation
  labels:
    name: content-automation
    environment: production
```

```bash
kubectl apply -f namespace.yaml
```

### Step 2: Create Secrets

```bash
# Create secret for database credentials
kubectl create secret generic content-automation-secrets \
  --from-literal=database-url='postgresql+asyncpg://user:pass@postgres-service:5432/content_automation' \
  --from-literal=redis-url='redis://redis-service:6379/0' \
  --from-literal=celery-broker-url='redis://redis-service:6379/0' \
  --from-literal=jwt-secret='$(openssl rand -hex 32)' \
  --from-literal=openai-api-key='sk-...' \
  --from-literal=anthropic-api-key='sk-ant-...' \
  --namespace=content-automation
```

### Step 3: Deploy PostgreSQL

```yaml
# postgres-deployment.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: content-automation
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: content-automation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:16-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          value: content_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: database-password
        - name: POSTGRES_DB
          value: content_automation
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: content-automation
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

```bash
kubectl apply -f postgres-deployment.yaml
```

### Step 4: Deploy Redis

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: content-automation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--appendonly", "yes"]
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: content-automation
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

```bash
kubectl apply -f redis-deployment.yaml
```

### Step 5: Deploy API

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: content-automation-api
  namespace: content-automation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: content-automation-api
  template:
    metadata:
      labels:
        app: content-automation-api
    spec:
      containers:
      - name: api
        image: content-automation-pipeline:latest
        command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: redis-url
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: jwt-secret
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: content-automation-api-service
  namespace: content-automation
spec:
  selector:
    app: content-automation-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl apply -f api-deployment.yaml
```

### Step 6: Deploy Celery Workers

```yaml
# celery-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
  namespace: content-automation
spec:
  replicas: 4
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
      - name: celery
        image: content-automation-pipeline:latest
        command: ["celery", "-A", "orchestration.celery_app.app", "worker", "--loglevel=info", "-Q", "default,high,low,dead_letter"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: redis-url
        - name: CELERY_BROKER_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: celery-broker-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

```bash
kubectl apply -f celery-deployment.yaml
```

---

## Database Setup

### Initial Schema Creation

```bash
# Run Alembic migrations
docker-compose -f docker-compose.prod.yml run --rm api alembic upgrade head

# Or in Kubernetes
kubectl exec -it -n content-automation $(kubectl get pod -n content-automation -l app=content-automation-api -o jsonpath='{.items[0].metadata.name}') -- alembic upgrade head
```

### Seed Initial Data

```bash
# Seed best practices knowledge base
docker-compose -f docker-compose.prod.yml run --rm api python scripts/seed_best_practices.py

# Or in Kubernetes
kubectl exec -it -n content-automation $(kubectl get pod -n content-automation -l app=content-automation-api -o jsonpath='{.items[0].metadata.name}') -- python scripts/seed_best_practices.py
```

---

## Monitoring Setup

### Prometheus + Grafana

```yaml
# monitoring-stack.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: content-automation
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'content-automation-api'
        static_configs:
          - targets: ['content-automation-api-service:80']
        metrics_path: '/metrics'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: content-automation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus/prometheus.yml
          subPath: prometheus.yml
      volumes:
      - name: config
        configMap:
          name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: content-automation
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
```

```bash
kubectl apply -f monitoring-stack.yaml
```

### Grafana Dashboards

Import dashboard JSON from `docs/grafana-dashboard.json` (to be created) or manually create panels for:

1. **Request Rate**: `rate(http_requests_total[5m])`
2. **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
3. **P95 Latency**: `histogram_quantile(0.95, http_request_duration_seconds_bucket)`
4. **LLM Cost**: `sum(increase(llm_cost_usd[1h]))`
5. **Celery Queue Depth**: `celery_queue_length`

---

## Security Hardening

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: content-automation
spec:
  podSelector:
    matchLabels:
      app: content-automation-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Secret Rotation

```bash
# Rotate JWT secret every 90 days
NEW_JWT_SECRET=$(openssl rand -hex 32)
kubectl patch secret content-automation-secrets \
  --namespace content-automation \
  --patch="{\"data\":{\"jwt-secret\":\"$(echo -n $NEW_JWT_SECRET | base64)\"}}"

# Restart API pods to pick up new secret
kubectl rollout restart deployment/content-automation-api -n content-automation
```

### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: content-automation-sa
  namespace: content-automation
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: content-automation-role
  namespace: content-automation
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: content-automation-rolebinding
  namespace: content-automation
subjects:
- kind: ServiceAccount
  name: content-automation-sa
roleRef:
  kind: Role
  name: content-automation-role
  apiGroup: rbac.authorization.k8s.io
```

---

## Scaling Guidelines

### Horizontal Pod Autoscaling

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: content-automation-api-hpa
  namespace: content-automation
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: content-automation-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Celery Worker Scaling

```bash
# Scale workers based on queue depth
# Monitor: redis-cli LLEN celery

# Manual scaling
kubectl scale deployment celery-worker --replicas=10 -n content-automation

# Auto-scale based on queue depth (KEDA)
# Install KEDA: kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.11.0/keda-2.11.0.yaml
```

---

## Backup & Recovery

### Database Backups

```bash
# Automated daily backups
kubectl create cron job postgres-backup \
  --schedule="0 2 * * *" \
  --restart=Never \
  --namespace=content-automation \
  --image=postgres:16-alpine \
  -- /bin/sh -c "pg_dump -h postgres-service -U content_user -d content_automation | gzip > /backup/backup-$(date +%Y%m%d).sql.gz"

# Manual backup
kubectl exec -it -n content-automation $(kubectl get pod -n content-automation -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- pg_dump -U content_user content_automation > backup.sql
```

### Restore from Backup

```bash
# Restore database from backup
kubectl exec -i -n content-automation $(kubectl get pod -n content-automation -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- psql -U content_user content_automation < backup.sql
```

---

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/content-automation-api api=content-automation-pipeline:v2.0.0 -n content-automation

# Check rollout status
kubectl rollout status deployment/content-automation-api -n content-automation

# Rollback if needed
kubectl rollout undo deployment/content-automation-api -n content-automation
```

### Database Maintenance

```bash
# Vacuum and analyze
kubectl exec -it -n content-automation $(kubectl get pod -n content-automation -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- psql -U content_user -d content_automation -c "VACUUM ANALYZE;"

# Reindex
kubectl exec -it -n content-automation $(kubectl get pod -n content-automation -l app=postgres -o jsonpath='{.items[0].metadata.name}') -- psql -U content_user -d content_automation -c "REINDEX DATABASE content_automation;"
```

### Log Rotation

```bash
# Configure log rotation in docker-compose
# Add to docker-compose.prod.yml:
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
```

---

## Support & Troubleshooting

For detailed troubleshooting, see main [README.md#troubleshooting](./README.md#troubleshooting).

For production incidents, check:
1. **Logs**: `kubectl logs -f -n content-automation -l app=content-automation-api --tail=100`
2. **Metrics**: `curl http://content-automation-api-service/metrics`
3. **Database**: Query `task_results` table for failed tasks
4. **Redis**: Check idempotency keys and circuit breaker state
5. **Health Check**: `curl http://content-automation-api-service/health`
