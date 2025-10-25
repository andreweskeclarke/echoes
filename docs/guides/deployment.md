# Deployment Guide

Instructions for deploying services to production.

## Overview

Echoes services are deployed as systemd services on the headquarters VM, served through nginx with SSL certificates from Let's Encrypt.

**Services**:
- MLflow UI (mlflow.lonel.ai)
- TensorBoard (tensorboard.lonel.ai)
- Custom Dashboard (dashboard.lonel.ai)
- Documentation (docs.lonel.ai)

## Prerequisites

### Azure VM Setup

Configure Azure VM and Network Security Group (NSG) rules:

**NSG Inbound Rules**:
- Port 80 (HTTP) - Allow from Any
- Port 443 (HTTPS) - Allow from Any
- Port 22 (SSH) - Allow from your IP

**VM Requirements**:
- Public IP address assigned
- Standard VM size (B2s or larger for responsive UI)

### DNS Configuration

Set A records for your domain to point to VM public IP:

```
lonel.ai          A    <public-ip>
mlflow.lonel.ai   A    <public-ip>
tensorboard.lonel.ai   A    <public-ip>
dashboard.lonel.ai     A    <public-ip>
docs.lonel.ai     A    <public-ip>
```

Or use CNAME records:

```
mlflow.lonel.ai   CNAME    lonel.ai
tensorboard.lonel.ai   CNAME    lonel.ai
```

### Authentication

Create htpasswd file for nginx basic auth:

```bash
sudo htpasswd -c /etc/nginx/.htpasswd mlflow
# Enter password when prompted
```

This creates credentials for MLflow/TensorBoard/Dashboard access.

## Deployment

Deploy all services with one command:

```bash
sudo ./scripts/local_deploy.sh
```

**What it does:**
- Auto-generates SSL certs via certbot (if missing)
- Detects and displays public IP for DNS troubleshooting
- Deploys 3 systemd services (MLflow, TensorBoard, Dashboard)
- Builds MkDocs docs and deploys to nginx
- Runs dashboard build_and_deploy.py
- Validates all services responding

**Config files** are version-controlled in `scripts/deploy/`:
- `systemd/` - Service unit files
- `nginx/` - Reverse proxy configs

## Service Management

View service status:

```bash
sudo systemctl status mlflow
sudo systemctl status tensorboard
sudo systemctl status dashboard
sudo journalctl -u mlflow -f  # Follow logs
```

Restart services:

```bash
sudo systemctl restart mlflow
sudo systemctl reload nginx
```

## Troubleshooting

Check service logs:

```bash
sudo journalctl -u mlflow -n 20
sudo journalctl -u tensorboard -f
```

Validate nginx config:

```bash
sudo nginx -t
curl http://localhost:5000  # Test MLflow backend
```

Check all services running:

```bash
systemctl list-units --type=service | grep -E "mlflow|tensorboard|dashboard"
```
