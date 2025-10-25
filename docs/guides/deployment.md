# Deployment Guide

Instructions for deploying services to production.

## Overview

Echoes services are deployed as systemd services on the headquarters VM, served through nginx with SSL certificates from Let's Encrypt.

**Services**:
- MLflow UI (mlflow.lonel.ai)
- TensorBoard (tensorboard.lonel.ai)
- Custom Dashboard (dashboard.lonel.ai)
- Documentation (docs.lonel.ai)

## Automated Deployment

### Quick Deploy

Deploy all services at once:

```bash
conda activate echoes
./scripts/local_deploy.sh
```

This:
1. Validates system prerequisites
2. Creates/updates systemd service files
3. Rebuilds documentation
4. Deploys static files
5. Reloads nginx and systemd
6. Validates all services are running

### Deploy Specific Service

```bash
./scripts/local_deploy.sh --service mlflow
./scripts/local_deploy.sh --service tensorboard
./scripts/local_deploy.sh --service dashboard
./scripts/local_deploy.sh --service docs
```

### Validate Deployment

```bash
./scripts/local_deploy.sh --validate
```

Checks:
- All systemd services running
- nginx configuration valid
- SSL certificates valid
- Web endpoints responding

## Manual Deployment

If you need to manually configure a service:

### MLflow Service

**1. Create systemd unit file** (`/etc/systemd/system/mlflow.service`):

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=aclarke
WorkingDirectory=/home/aclarke/echoes
ExecStart=/home/aclarke/miniconda/envs/echoes/bin/mlflow ui \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri /mnt/echoes_data/mlruns \
  --default-artifact-root /mnt/echoes_data/mlruns/artifacts

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**2. Reload systemd**:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

**3. Check status**:

```bash
sudo systemctl status mlflow
sudo journalctl -u mlflow -f
```

### TensorBoard Service

**Create systemd unit file** (`/etc/systemd/system/tensorboard.service`):

```ini
[Unit]
Description=TensorBoard
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=aclarke
WorkingDirectory=/home/aclarke/echoes
ExecStart=/home/aclarke/miniconda/envs/echoes/bin/tensorboard \
  --logdir=/mnt/echoes_data/tfruns \
  --host=0.0.0.0 \
  --port=6006

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tensorboard
sudo systemctl start tensorboard
```

### Dashboard Service

**Create systemd unit file** (`/etc/systemd/system/dashboard.service`):

```ini
[Unit]
Description=Echoes Dashboard
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=aclarke
WorkingDirectory=/var/www/dashboard
ExecStart=/usr/bin/python3 -m http.server 8000

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Documentation Service

**Create systemd unit file** (`/etc/systemd/system/docs.service`):

```ini
[Unit]
Description=Echoes Documentation
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=aclarke
WorkingDirectory=/var/www/docs
ExecStart=/usr/bin/python3 -m http.server 8080

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Nginx Configuration

### MLflow Proxy

Create `/etc/nginx/sites-available/mlflow`:

```nginx
server {
    server_name mlflow.lonel.ai;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        auth_basic "MLflow";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }

    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/mlflow.lonel.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mlflow.lonel.ai/privkey.pem;
}

server {
    listen 80;
    listen [::]:80;
    server_name mlflow.lonel.ai;
    return 301 https://$server_name$request_uri;
}
```

Enable the site:

```bash
sudo ln -sf /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Apply to Other Services

Create similar configs for TensorBoard, Dashboard, and Docs, replacing:
- `mlflow.lonel.ai` with `tensorboard.lonel.ai`, etc.
- `localhost:5000` with `localhost:6006`, `localhost:8000`, `localhost:8080`
- Remove `auth_basic` lines for public services (if desired)

## SSL Certificates

### Get Certificates

```bash
sudo certbot --nginx -d mlflow.lonel.ai -d tensorboard.lonel.ai -d dashboard.lonel.ai -d docs.lonel.ai
```

Answer prompts for email and terms.

### Automatic Renewal

certbot installs a systemd timer that runs daily:

```bash
sudo systemctl status certbot.timer
```

Manually check renewal:

```bash
sudo certbot renew --dry-run
```

## Manage Services

### Start/Stop/Restart

```bash
# Start service
sudo systemctl start mlflow

# Stop service
sudo systemctl stop mlflow

# Restart service
sudo systemctl restart mlflow

# Enable on boot
sudo systemctl enable mlflow

# Disable on boot
sudo systemctl disable mlflow
```

### View Logs

```bash
# Last 50 lines
sudo journalctl -u mlflow -n 50

# Follow logs in real-time
sudo journalctl -u mlflow -f

# Since last boot
sudo journalctl -u mlflow -b

# Specific time range
sudo journalctl -u mlflow --since "2 hours ago"
```

### Check All Services

```bash
systemctl list-units --type=service | grep -E "mlflow|tensorboard|dashboard|docs"
```

## Troubleshooting

### Service Won't Start

Check logs first:

```bash
sudo journalctl -u mlflow -n 20
```

Common issues:
- **Port already in use**: `sudo lsof -i :5000` to find process
- **Permission denied**: Check file ownership and permissions
- **Conda environment not found**: Verify path in ExecStart

### Website Not Responding

```bash
# Check nginx is running
sudo systemctl status nginx

# Check nginx config is valid
sudo nginx -t

# Check service behind proxy is running
curl http://localhost:5000

# Check firewall
sudo ufw status

# Check DNS
nslookup mlflow.lonel.ai
```

### Restart Everything

```bash
# Reload nginx configuration
sudo systemctl reload nginx

# Reload all systemd units
sudo systemctl daemon-reload

# Restart all services
for service in mlflow tensorboard dashboard docs; do
    sudo systemctl restart $service
done
```

### Monitor Service Health

```bash
#!/bin/bash
# Check all services

for service in mlflow tensorboard dashboard docs; do
    status=$(sudo systemctl is-active $service)
    echo "$service: $status"
done
```

## Backup and Rollback

### Backup Deployment

```bash
# Backup configurations
tar czf ~/backups/deploy-$(date +%Y%m%d).tar.gz \
  /etc/nginx/sites-available/ \
  /etc/systemd/system/mlflow.service \
  /etc/systemd/system/tensorboard.service \
  /etc/systemd/system/dashboard.service \
  /etc/systemd/system/docs.service
```

### Rollback Service

```bash
# Revert to previous deployment
./scripts/local_deploy.sh --rollback

# Or manually restart service
sudo systemctl restart mlflow
```

## Scaling Considerations

### Multiple Instances

For high traffic, run multiple instances behind load balancer:

```bash
# Start MLflow on multiple ports
mlflow ui --port 5000
mlflow ui --port 5001
mlflow ui --port 5002

# nginx upstream configuration
upstream mlflow {
    server localhost:5000;
    server localhost:5001;
    server localhost:5002;
}
```

### Resource Limits

Add resource constraints to systemd services:

```ini
[Service]
MemoryLimit=4G
CPUQuota=80%
```

## Next Steps

- [Infrastructure](../architecture/infrastructure.md) - Understand the setup
- [Running Experiments](experiments.md) - Create experiments
