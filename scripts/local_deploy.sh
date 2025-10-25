#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$SCRIPT_DIR/deploy"
DATA_DIR="/mnt/echoes_data"
CONDA_ENV="echoes"
CONDA_PATH="/home/aclarke/miniconda/envs/${CONDA_ENV}"
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

[ "$(id -u)" = 0 ] || { log_err "Must run as root (needs systemd/nginx access)"; exit 1; }
[ -d "$DATA_DIR" ] || { log_err "Persistent disk not mounted at $DATA_DIR"; exit 1; }
[ -d "$CONDA_PATH" ] || { log_err "Conda environment not found"; exit 1; }

log_info "Validating prerequisites..."

PUBLIC_IP=$(curl -s http://ifconfig.me || echo "unknown")
log_info "Server public IP: $PUBLIC_IP"

for domain in mlflow tensorboard dashboard docs; do
  if [ ! -d "/etc/letsencrypt/live/${domain}.lonel.ai" ]; then
    log_info "Generating SSL cert for ${domain}.lonel.ai..."
    certbot certonly --nginx -d "${domain}.lonel.ai" --email admin@lonel.ai --agree-tos --non-interactive 2>&1 | grep -q "Successfully" && {
      log_ok "Cert generated for ${domain}.lonel.ai"
    } || {
      log_err "Failed to generate cert for ${domain}.lonel.ai"
      log_err "Ensure DNS records in Azure point to: $PUBLIC_IP"
      log_err "Update these A records:"
      log_err "  - lonel.ai → $PUBLIC_IP"
      log_err "  - mlflow.lonel.ai → $PUBLIC_IP"
      log_err "  - tensorboard.lonel.ai → $PUBLIC_IP"
      log_err "  - dashboard.lonel.ai → $PUBLIC_IP"
      log_err "  - docs.lonel.ai → $PUBLIC_IP"
      exit 1
    }
  fi
done
log_ok "Prerequisites OK"

# Deploy systemd services
log_info "Deploying systemd services..."
mkdir -p "$DATA_DIR/mlruns" "$DATA_DIR/tfruns"
cp "$DEPLOY_DIR"/systemd/{mlflow,tensorboard,dashboard}.service /etc/systemd/system/
log_ok "Systemd services deployed"

# Deploy nginx configs
log_info "Deploying nginx configs..."
mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled
rm -f /etc/nginx/sites-available/{mlflow,tensorboard,dashboard,docs}
rm -f /etc/nginx/sites-enabled/{mlflow,tensorboard,dashboard,docs}
cp "$DEPLOY_DIR"/nginx/* /etc/nginx/sites-available/
for conf in mlflow tensorboard dashboard docs; do
  ln -sf /etc/nginx/sites-available/$conf /etc/nginx/sites-enabled/$conf
done
rm -f /etc/nginx/sites-enabled/default
log_ok "Nginx configs deployed"

# Build and deploy docs (static files)
log_info "Building documentation..."
cd "$REPO_DIR"
"$CONDA_PATH/bin/mkdocs" build --quiet
mkdir -p /var/www/docs
rm -rf /var/www/docs/current
cp -r "$REPO_DIR/site" /var/www/docs/current
log_ok "Docs built and deployed"

# Deploy dashboard
log_info "Deploying dashboard..."
rm -rf /var/www/dashboard/current
mkdir -p /var/www/dashboard/current
cd "$REPO_DIR"
"$CONDA_PATH/bin/python" dashboard/build_and_deploy.py || log_info "Dashboard build skipped (no MLflow data)"
log_ok "Dashboard deployed"

# Reload services
log_info "Reloading services..."
nginx -t > /dev/null 2>&1 || { log_err "nginx config invalid"; exit 1; }
systemctl daemon-reload
systemctl reload nginx

for svc in mlflow tensorboard dashboard; do
  systemctl enable "$svc"
  systemctl restart "$svc"
done
log_ok "Services reloaded"

# Validate
log_info "Validating deployment..."
sleep 2

for svc in mlflow tensorboard dashboard; do
  systemctl is-active "$svc" > /dev/null || { log_err "$svc not running"; exit 1; }
done

curl -sf http://localhost:5000 > /dev/null || { log_err "MLflow not responding"; exit 1; }
curl -sf http://localhost:6006 > /dev/null || { log_err "TensorBoard not responding"; exit 1; }
curl -sf http://localhost:8000 > /dev/null || { log_err "Dashboard not responding"; exit 1; }
[ -f /var/www/docs/current/index.html ] || { log_err "Docs not deployed"; exit 1; }

log_ok "All services validated"
log_ok "Deployment complete!"
