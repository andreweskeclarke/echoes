# Web Hosting Setup

This document describes the secure web hosting setup for MLflow and TensorBoard with HTTPS certificates and authentication.

## Architecture Overview

- **Domain**: lonel.ai
- **Subdomains**: mlflow.lonel.ai, tensorboard.lonel.ai
- **SSL/TLS**: Let's Encrypt certificates with auto-renewal
- **Authentication**: Basic HTTP auth with nginx
- **Reverse Proxy**: nginx forwards to local services

## Services

| Service | URL | Local Port | Config File |
|---------|-----|------------|-------------|
| MLflow | https://mlflow.lonel.ai | 5000 | `nginx-mlflow.conf` |
| TensorBoard | https://tensorboard.lonel.ai | 6006 | `nginx-tensorboard.conf` |

## Prerequisites

### Azure Configuration

1. **Network Security Group (NSG) Rules**:
   ```
   Port 80 (HTTP)  - Allow inbound from Any
   Port 443 (HTTPS) - Allow inbound from Any
   Port 22 (SSH)   - Allow inbound from your IP
   ```

2. **VM Requirements**:
   - Public IP address assigned
   - Standard VM size (performance affects UI responsiveness)

### DNS Configuration

Set up A records for your domain pointing to the Azure VM public IP:

```
lonel.ai               A    <vm-public-ip>
mlflow.lonel.ai        A    <vm-public-ip>  
tensorboard.lonel.ai   A    <vm-public-ip>
```

Alternatively, use CNAME records:
```
mlflow.lonel.ai        CNAME    lonel.ai
tensorboard.lonel.ai   CNAME    lonel.ai
```

## Installation Steps

### 1. Install nginx and certificate tools
```bash
sudo apt update
sudo apt install -y nginx apache2-utils certbot python3-certbot-nginx
```

### 2. Create authentication
```bash
sudo htpasswd -c /etc/nginx/.htpasswd mlflow
# Enter password when prompted
```

### 3. Set up nginx configurations
The nginx config files are stored in this repository and symlinked:

```bash
sudo ln -s /home/aclarke/echoes/nginx-mlflow.conf /etc/nginx/sites-available/mlflow
sudo ln -s /home/aclarke/echoes/nginx-tensorboard.conf /etc/nginx/sites-available/tensorboard
sudo ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/tensorboard /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
```

### 4. Get SSL certificates
```bash
sudo certbot --nginx -d mlflow.lonel.ai --email your-email@domain.com --agree-tos --non-interactive
sudo certbot --nginx -d tensorboard.lonel.ai --email your-email@domain.com --agree-tos --non-interactive
```

### 5. Test configuration
```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Usage

### Starting Services

1. **MLflow**:
   ```bash
   conda activate echoes
   mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **TensorBoard**:
   ```bash
   conda activate echoes
   tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
   ```

### Accessing Services
- MLflow: https://mlflow.lonel.ai
- TensorBoard: https://tensorboard.lonel.ai
- Authentication: Username "mlflow" + password set during setup

## Maintenance

### Certificate Renewal
Let's Encrypt certificates auto-renew via systemd timer:
```bash
sudo systemctl status certbot.timer
sudo certbot renew --dry-run  # Test renewal
```

### Updating Configurations
Since nginx configs are symlinked from this repository:
1. Edit the config files in this repo
2. Run `sudo nginx -t` to test
3. Run `sudo systemctl reload nginx` to apply changes
4. Commit changes to version control
