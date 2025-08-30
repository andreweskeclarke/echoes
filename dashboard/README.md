# MLflow Architecture Dashboard

Interactive web dashboard for visualizing and comparing neural network architectures from MLflow experiments.

## Features

- **Architecture Visualization**: Layer-by-layer diagrams with parameter counts
- **Parameter Analysis**: Visual comparison of trainable vs frozen parameters  
- **Model Comparison**: Side-by-side architecture comparison
- **Performance Metrics**: Accuracy, training time, and efficiency metrics
- **Interactive Filtering**: Filter by model type, sort by various metrics
- **Secure Deployment**: HTTPS with password protection

## Quick Start

### 1. Build and Deploy Dashboard

```bash
# Activate conda environment
conda activate echoes

# Extract MLflow data and deploy to nginx
python dashboard/build_and_deploy.py
```

### 2. Set up Nginx (First time only)

```bash
# Copy nginx configuration
sudo cp dashboard/nginx-dashboard.conf /etc/nginx/sites-available/dashboard

# Enable the site
sudo ln -sf /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 3. Set up DNS and SSL

```bash
# Add DNS A record for dashboard.lonel.ai pointing to your server IP

# Get SSL certificate with Let's Encrypt
sudo certbot --nginx -d dashboard.lonel.ai

# Verify certificate auto-renewal
sudo certbot renew --dry-run
```

### 4. Access Dashboard

Visit https://dashboard.lonel.ai and use the same credentials as MLflow/TensorBoard.

## Architecture

```
dashboard/
├── build_and_deploy.py     # Main build script
├── nginx-dashboard.conf    # Nginx configuration
├── src/                    # Source files
│   ├── index.html         # Main dashboard
│   ├── style.css          # Styling
│   ├── dashboard.js       # Visualization logic
│   └── d3.min.js         # D3.js library
└── README.md              # This file
```

## Development

### Local Testing

```bash
# Build dashboard locally (skips nginx deployment)
python dashboard/build_and_deploy.py --local

# Serve locally for testing
cd dashboard/src
python -m http.server 8000
# Visit http://localhost:8000
```

### Adding New Model Types

Edit `ModelArchitectureAnalyzer` in `build_and_deploy.py`:

1. Add new model type detection in `analyze_model()`
2. Implement architecture analysis method (e.g., `_analyze_new_model()`)
3. Define layer structure with parameter counts and trainable status

## Architecture Visualization

The dashboard creates interactive SVG diagrams showing:

- **Layer Height**: Proportional to parameter count
- **Color Coding**: 
  - 🔥 Red/Orange: Trainable parameters
  - ❄️ Blue/Gray: Frozen parameters
- **Hover Details**: Exact parameter counts and layer information
- **Model Comparison**: Side-by-side architecture comparison

### Model Types Supported

- **SimpleRNN**: Input → LSTM → FC
- **DeepRNN**: Input → Multi-LSTM → FC layers
- **SimpleESN**: Input(frozen) → Reservoir(frozen) → Readout(trainable)
- **DeepESN**: Input(frozen) → Multi-Reservoir(frozen) → Readout(trainable)

## Data Flow

1. **MLflow Experiments** → Extract parameters and metrics
2. **Architecture Analysis** → Reconstruct layer-wise structure
3. **Data Sanitization** → Remove sensitive information
4. **JSON Generation** → Create web-consumable data
5. **Static Deployment** → Copy to nginx directory
6. **Web Visualization** → Interactive dashboard

## Configuration

Key settings in `build_and_deploy.py`:

```python
NGINX_DASHBOARD_DIR = "/var/www/dashboard"  # Deployment directory
EXPERIMENT_NAME = "UCF101_Architecture_Comparison"  # MLflow experiment
```

## Updates

To update the dashboard with new experiment data:

```bash
python dashboard/build_and_deploy.py
```

No server restart required - static files are updated in place.
