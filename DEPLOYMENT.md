# Production Deployment Guide

Complete guide for deploying the Content Automation Pipeline to production.

## 📋 Pre-Deployment Checklist

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Production audit passing (`python3 scripts/run_production_audit.py`)
- [ ] Environment variables configured
- [ ] Database backup strategy in place
- [ ] Monitoring configured (Prometheus, Sentry)
- [ ] SSL/TLS certificates ready
- [ ] Load balancer configured

## 🔧 Environment Setup

### 1. Configure Environment Variables

Create `.env` file (never commit this):

```bash
# Copy template
cp .env.example .env

# Edit with production values
nano .env
```

**Critical Variables:**
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security (CHANGE THESE!)
SECRET_KEY=<generate-secure-random-string-64-chars>
JWT_SECRET_KEY=<generate-different-secure-random-string>

# Database (Production credentials)
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/content_automation

# Redis
REDIS_URL=redis://:password@redis:6379/0

# API Keys (from providers)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...

# CORS (Your frontend domains)
ALLOWED_ORIGINS=https://yourapp.com,https://www.yourapp.com

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Monitoring
SENTRY_DSN=https://...@sentry.io/...
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

### 2. Generate Secure Keys

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Generate JWT_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(64))"
```

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended for Single Server)

```bash
# 1. Build images
docker-compose build

# 2. Start services
docker-compose up -d

# 3. Run migrations
docker-compose exec api alembic upgrade head

# 4. Create initial admin user (if needed)
docker-compose exec api python -m scripts.create_admin

# 5. Verify health
curl http://localhost:8000/health
```

### Option 2: Kubernetes (Recommended for Multi-Server)

```bash
# 1. Create namespace
kubectl create namespace content-automation

# 2. Create secrets
kubectl create secret generic app-secrets \
  --from-env-file=.env \
  --namespace=content-automation

# 3. Deploy
kubectl apply -f k8s/ --namespace=content-automation

# 4. Check status
kubectl get pods -n content-automation

# 5. Run migrations
kubectl exec -it deployment/api -n content-automation -- alembic upgrade head
```

## 🗄️ Database Setup

### PostgreSQL Configuration

```bash
# 1. Create database
createdb content_automation

# 2. Create user
psql -c "CREATE USER content_user WITH PASSWORD 'secure_password';"

# 3. Grant privileges
psql -c "GRANT ALL PRIVILEGES ON DATABASE content_automation TO content_user;"

# 4. Run migrations
alembic upgrade head

# 5. Verify
psql -d content_automation -c "\dt"
```

### Database Optimization

```sql
-- Enable connection pooling (if not using PgBouncer)
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';

-- Reload config
SELECT pg_reload_conf();
```

## 🔴 Redis Setup

```bash
# 1. Configure Redis for production
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# 2. Enable persistence (optional)
redis-cli CONFIG SET save "900 1 300 10 60 10000"

# 3. Set password
redis-cli CONFIG SET requirepass "your-secure-password"

# 4. Restart Redis
systemctl restart redis
```

## 🌐 Reverse Proxy (Nginx)

### Nginx Configuration

Create `/etc/nginx/sites-available/content-automation`:

```nginx
upstream api_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;  # If scaling
}

server {
    listen 80;
    server_name api.yourcompany.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourcompany.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourcompany.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    # Max body size
    client_max_body_size 10M;

    # Proxy settings
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://api_backend;
        access_log off;
    }

    # Static files (if needed)
    location /static {
        alias /var/www/content-automation/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable and reload:
```bash
ln -s /etc/nginx/sites-available/content-automation /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

## 🔒 SSL/TLS Setup (Let's Encrypt)

```bash
# 1. Install certbot
apt-get install certbot python3-certbot-nginx

# 2. Get certificate
certbot --nginx -d api.yourcompany.com

# 3. Auto-renewal
certbot renew --dry-run

# 4. Add cron job
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -
```

## 📊 Monitoring Setup

### Prometheus

`prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'content-automation-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/system/metrics'
```

### Grafana Dashboards

Import these dashboards:
1. FastAPI Metrics (ID: 14694)
2. Redis Metrics (ID: 763)
3. PostgreSQL Metrics (ID: 9628)

### Sentry Error Tracking

```bash
# Set in .env
SENTRY_DSN=https://...@sentry.io/...
SENTRY_ENVIRONMENT=production
```

## 🚨 Logging

### Centralized Logging (ELK Stack)

```yaml
# docker-compose.override.yml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Log Aggregation (Filebeat)

```yaml
# filebeat.yml
filebeat.inputs:
  - type: container
    paths:
      - '/var/lib/docker/containers/*/*.log'

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

## 🔄 Scaling

### Horizontal Scaling

```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale Celery workers
docker-compose up -d --scale celery-worker=5
```

### Kubernetes Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 💾 Backup Strategy

### Database Backups

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
mkdir -p $BACKUP_DIR

pg_dump content_automation | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -type f -mtime +30 -delete
```

Add to cron:
```bash
0 2 * * * /usr/local/bin/backup_db.sh
```

### Redis Backups

```bash
# Enable RDB persistence
redis-cli CONFIG SET save "900 1 300 10 60 10000"

# Backup script
cp /var/lib/redis/dump.rdb /backups/redis/dump_$(date +%Y%m%d).rdb
```

## 🧪 Post-Deployment Verification

```bash
# 1. Health check
curl https://api.yourcompany.com/health

# 2. API docs
curl https://api.yourcompany.com/docs

# 3. Metrics endpoint
curl https://api.yourcompany.com/system/metrics

# 4. Test authentication
curl -X POST https://api.yourcompany.com/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=testpass"

# 5. Load test (optional)
ab -n 1000 -c 10 https://api.yourcompany.com/health
```

## 🔥 Rollback Procedure

```bash
# 1. Stop new version
docker-compose down

# 2. Restore previous image
docker-compose up -d --scale api=0
docker tag content-automation-api:previous content-automation-api:latest

# 3. Rollback database (if needed)
alembic downgrade -1

# 4. Start services
docker-compose up -d
```

## 📈 Performance Tuning

### Gunicorn Configuration

```python
# gunicorn_config.py
workers = 4  # (2 x CPU cores) + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 120
keepalive = 5
```

### Database Connection Pool

```python
# config/settings.py
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 10
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
```

## 🆘 Troubleshooting

### Common Issues

**1. Database Connection Errors**
```bash
# Check connectivity
docker-compose exec api psql $DATABASE_URL

# Check pool status
docker-compose logs api | grep "pool"
```

**2. Redis Connection Errors**
```bash
# Check Redis
docker-compose exec redis redis-cli ping

# Check connections
docker-compose exec api python -c "import redis; r=redis.from_url('$REDIS_URL'); print(r.ping())"
```

**3. High Memory Usage**
```bash
# Check container stats
docker stats

# Restart workers
docker-compose restart celery-worker
```

**4. Slow Response Times**
```bash
# Check database queries
docker-compose exec postgres psql -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check Redis latency
docker-compose exec redis redis-cli --latency
```

## 📞 Support

- Check logs: `docker-compose logs -f api`
- Health check: `/health` endpoint
- Metrics: `/system/metrics` endpoint
- Sentry: Error tracking dashboard

---

**Remember:** Always test in staging before deploying to production!
