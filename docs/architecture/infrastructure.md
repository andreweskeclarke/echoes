# Infrastructure

Overview of the hardware and cloud setup for Echoes.

## Headquarters VM (Persistent)

The main Azure VM serves as the central hub.

**Specifications**:
- **OS**: Ubuntu 20.04 LTS
- **Size**: Standard_D4s_v3 (4 vCPU, 16GB RAM)
- **OS Disk**: 29GB (ephemeral - code and environments)
- **Data Disk**: 118GB (persistent - `/mnt/echoes_data/`)

**Running Services**:
- nginx (reverse proxy)
- MLflow server
- TensorBoard server
- Custom dashboard server
- Code repository

**Storage Mounts**:

```
/home/aclarke/echoes/          # Code repository
├── experiments/               # Training scripts
├── models/                    # Model implementations
├── scripts/                   # Utilities (Azure runner, deploy, etc.)
├── logs → /mnt/echoes_data/logs/
├── mlruns → /mnt/echoes_data/mlruns/
└── tfruns → /mnt/echoes_data/tfruns/

/mnt/echoes_data/              # Persistent data disk
├── ucf101/                    # Dataset (13GB, 13,320 videos)
├── logs/                      # TensorBoard logs
├── mlruns/                    # MLflow experiment data
├── tfruns/                    # TensorFlow/TB runs
└── azure_results/             # Downloaded results from experiment VMs
```

## Experiment VMs (Ephemeral)

Temporary GPU VMs for running experiments.

**Provisioning**:
```bash
python scripts/azure_gpu_runner.py experiments/train_comprehensive.py --vm-size Standard_NC6s_v3
```

**VM Lifecycle**:

```
1. Create VM
   ├─ Configure networking
   ├─ Assign public IP
   └─ Create/attach data disk

2. Environment Setup
   ├─ Install dependencies (conda, etc.)
   ├─ Copy code from headquarters
   └─ Download dataset from persistent disk

3. Run Experiment
   ├─ Train model on GPU
   ├─ Log metrics to MLflow
   └─ Save artifacts to persistent disk

4. Result Collection
   ├─ Download artifacts locally
   └─ Update headquarters persistent storage

5. Cleanup
   ├─ Delete VM
   ├─ Release public IP
   └─ Delete OS disk (saves costs)
   └─ Keep data disk results
```

**Available Sizes**:

| Size | GPU | vCPU | RAM | Cost/hr |
|------|-----|------|-----|---------|
| Standard_NC6s_v3 | 1x K80 | 6 | 112GB | ~$0.94 |
| Standard_NC12s_v3 | 2x K80 | 12 | 224GB | ~$1.88 |
| Standard_NC24s_v3 | 4x K80 | 24 | 448GB | ~$3.76 |
| Standard_B1s | None | 1 | 1GB | ~$0.01 |

**Recommended**:
- `Standard_NC6s_v3` - Good balance of cost and performance
- `Standard_B1s` - For quick testing (no GPU)

## Networking

### Domain Names

```
lonel.ai                # Main domain
├── mlflow.lonel.ai     # MLflow UI (port 5000)
├── tensorboard.lonel.ai # TensorBoard (port 6006)
├── dashboard.lonel.ai   # Model comparison dashboard
└── docs.lonel.ai        # Documentation (this site)
```

### SSL/TLS Certificates

- **Provider**: Let's Encrypt
- **Renewal**: Automatic via certbot systemd timer
- **Duration**: 90 days

### Reverse Proxy

nginx forwards requests:

```
Client Request
    ↓
nginx (port 443 HTTPS)
    ├─ mlflow.lonel.ai → localhost:5000 (MLflow UI)
    ├─ tensorboard.lonel.ai → localhost:6006 (TensorBoard)
    ├─ dashboard.lonel.ai → localhost:8000 (Dashboard)
    └─ docs.lonel.ai → localhost:8080 (Documentation)
    ↓
Local Services (port <5000)
```

### Authentication

- **HTTP Basic Auth** for MLflow and TensorBoard
- **Credentials** stored in `/etc/nginx/.htpasswd`

## Deployment Architecture

### Services

#### MLflow Server

```
systemd service: mlflow.service
├─ Runs: mlflow ui --host 0.0.0.0 --port 5000
├─ Data: /mnt/echoes_data/mlruns/
├─ Logs: systemd journal
└─ nginx proxy: mlflow.lonel.ai
```

#### TensorBoard

```
systemd service: tensorboard.service
├─ Runs: tensorboard --logdir=/mnt/echoes_data/tfruns/
├─ Port: 6006
├─ Data: /mnt/echoes_data/tfruns/
├─ Logs: systemd journal
└─ nginx proxy: tensorboard.lonel.ai
```

#### Dashboard

```
systemd service: dashboard.service
├─ Serves: Static HTML/JS from /var/www/dashboard/
├─ Port: 8000
├─ Data Source: MLflow API
├─ Logs: systemd journal
└─ nginx proxy: dashboard.lonel.ai
```

#### Documentation

```
systemd service: docs.service
├─ Serves: Static HTML from /var/www/docs/site/
├─ Port: 8080
├─ Data: Built MkDocs output
├─ Logs: systemd journal
└─ nginx proxy: docs.lonel.ai
```

### Deployment Script

`scripts/local_deploy.sh` automates all deployment:

```bash
# Deploy all services
./scripts/local_deploy.sh

# Deploy specific service
./scripts/local_deploy.sh --service mlflow
./scripts/local_deploy.sh --service tensorboard
./scripts/local_deploy.sh --service dashboard
./scripts/local_deploy.sh --service docs

# Validate deployment
./scripts/local_deploy.sh --validate
```

## Data Flow

### Training Experiment

```
┌─────────────────────────────────────────┐
│     Create Experiment VM on Azure       │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Copy code + download dataset           │
│  (from headquarters persistent disk)    │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Run training on GPU                    │
│  Log to MLflow                          │
│  Save checkpoints to persistent disk    │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Download results to headquarters       │
│  (via persistent disk share)            │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Delete experiment VM (save costs!)     │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  View results in MLflow/Dashboard       │
│  at mlflow.lonel.ai                     │
└─────────────────────────────────────────┘
```

## Cost Optimization

### Headquarters
- Always running (~$150-200/month)
- Cost of persistent storage (~$50/month)

### Experiment VMs
- Create on-demand only
- Automatically destroy after experiments
- Standard_NC6s_v3 @ ~$0.94/hr = $11/12hr experiment

**Example**: 1 week of experiments (12hrs/day GPU)
- 7 days × 12 hours × $0.94/hr = ~$79
- Much cheaper than always-on GPU VM

## Troubleshooting

### Check Service Status

```bash
sudo systemctl status mlflow
sudo systemctl status tensorboard
sudo systemctl status dashboard
sudo systemctl status docs
```

### View Logs

```bash
sudo journalctl -u mlflow -f        # MLflow logs
sudo journalctl -u tensorboard -f   # TensorBoard logs
```

### Restart Services

```bash
sudo systemctl restart mlflow
sudo systemctl restart tensorboard
sudo systemctl restart dashboard
sudo systemctl restart docs

# Or use deployment script
./scripts/local_deploy.sh --restart-all
```

### Verify Web Services

```bash
curl -u mlflow:password https://mlflow.lonel.ai
curl https://docs.lonel.ai
```

## Next Steps

- [Deployment Guide](../guides/deployment.md) - How to deploy services
- [Running Experiments](../guides/experiments.md) - How to run on Azure VMs
