#!/usr/bin/env python3
"""
MLflow Dashboard Builder and Deployment Script

Extracts MLflow experiment data, processes model architectures,
and deploys static dashboard to nginx directory with security.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import mlflow
    import pandas as pd
    import torch
except ImportError:
    print("Error: MLflow not available. Run: conda activate echoes")
    sys.exit(1)

# Configuration
NGINX_DASHBOARD_DIR = "/var/www/dashboard/current"
SOURCE_DIR = Path(__file__).parent / "src"
EXPERIMENT_NAME = "UCF101_Architecture_Comparison"

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelArchitectureAnalyzer:
    """Analyzes model configurations and reconstructs layer-wise architecture"""

    def __init__(self, input_size: int = 112 * 112 * 3, num_classes: int = 25):
        self.input_size = input_size
        self.num_classes = num_classes

    def analyze_model(self, run_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Reconstruct model architecture from MLflow parameters"""
        model_type = run_data.get("params.model_type", "Unknown")

        if model_type == "SimpleRNN":
            return self._analyze_simple_rnn(run_data)
        elif model_type == "DeepRNN":
            return self._analyze_deep_rnn(run_data)
        elif model_type == "SimpleESN":
            return self._analyze_simple_esn(run_data)
        elif model_type == "DeepESN":
            return self._analyze_deep_esn(run_data)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return []

    def _analyze_simple_rnn(self, run_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze SimpleRNN architecture"""
        hidden_size = int(run_data.get("params.hidden_size", 128))
        num_layers = int(run_data.get("params.num_layers", 1))

        # LSTM parameters: 4 * (input_size + hidden_size + 1) * hidden_size per layer
        lstm_params_per_layer = 4 * (self.input_size + hidden_size + 1) * hidden_size
        lstm_params = lstm_params_per_layer * num_layers

        # FC layer parameters
        fc_params = hidden_size * self.num_classes + self.num_classes

        layers = [
            {
                "name": "Input",
                "params": 0,
                "trainable": False,
                "type": "input",
                "description": f"Input features: {self.input_size:,}",
            },
            {
                "name": f"LSTM ({num_layers} layer{'s' if num_layers > 1 else ''})",
                "params": lstm_params,
                "trainable": True,
                "type": "rnn",
                "description": f"Hidden size: {hidden_size}, Layers: {num_layers}",
            },
            {
                "name": "Output Layer",
                "params": fc_params,
                "trainable": True,
                "type": "output",
                "description": f"Classes: {self.num_classes}",
            },
        ]

        return layers

    def _analyze_deep_rnn(self, run_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze DeepRNN architecture"""
        hidden_size = int(run_data.get("params.hidden_size", 128))
        num_layers = int(run_data.get("params.num_layers", 3))

        # LSTM layers
        lstm_params_per_layer = 4 * (self.input_size + hidden_size + 1) * hidden_size
        lstm_params = lstm_params_per_layer * num_layers

        # FC layers: hidden -> hidden//2 -> classes
        fc1_params = hidden_size * (hidden_size // 2) + (hidden_size // 2)
        fc2_params = (hidden_size // 2) * self.num_classes + self.num_classes

        layers = [
            {
                "name": "Input",
                "params": 0,
                "trainable": False,
                "type": "input",
                "description": f"Input features: {self.input_size:,}",
            },
            {
                "name": f"Multi-LSTM ({num_layers} layers)",
                "params": lstm_params,
                "trainable": True,
                "type": "rnn",
                "description": f"Hidden size: {hidden_size}, Dropout layers",
            },
            {
                "name": "FC Layer 1",
                "params": fc1_params,
                "trainable": True,
                "type": "fc",
                "description": f"{hidden_size} → {hidden_size // 2} + ReLU + Dropout",
            },
            {
                "name": "Output Layer",
                "params": fc2_params,
                "trainable": True,
                "type": "output",
                "description": f"Classes: {self.num_classes}",
            },
        ]

        return layers

    def _analyze_simple_esn(self, run_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze SimpleESN architecture"""
        reservoir_size = int(run_data.get("params.reservoir_size", 1000))

        # Input weights (frozen)
        input_params = self.input_size * reservoir_size

        # Reservoir weights (frozen)
        reservoir_params = reservoir_size * reservoir_size

        # Readout layer (trainable)
        readout_params = reservoir_size * self.num_classes + self.num_classes

        layers = [
            {
                "name": "Input Weights",
                "params": input_params,
                "trainable": False,
                "type": "input",
                "description": (
                    f"Input → Reservoir: {self.input_size:,} x {reservoir_size:,}"
                ),
            },
            {
                "name": "Reservoir",
                "params": reservoir_params,
                "trainable": False,
                "type": "reservoir",
                "description": (
                    f"Fixed random weights: {reservoir_size:,} x {reservoir_size:,}"
                ),
            },
            {
                "name": "Readout Layer",
                "params": readout_params,
                "trainable": True,
                "type": "output",
                "description": (
                    f"Only trainable part: {reservoir_size:,} → {self.num_classes}"
                ),
            },
        ]

        return layers

    def _analyze_deep_esn(self, run_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze DeepESN architecture"""
        reservoir_size = int(run_data.get("params.reservoir_size", 1000))
        num_layers = int(run_data.get("params.num_layers", 3))

        # Input weights (frozen)
        input_params = self.input_size * reservoir_size

        # Multiple reservoir layers (frozen)
        reservoir_params_per_layer = reservoir_size * reservoir_size
        total_reservoir_params = reservoir_params_per_layer * num_layers

        # Layer connections (frozen)
        connection_params = reservoir_size * reservoir_size * (num_layers - 1)

        # Readout from all layers (trainable)
        readout_params = (
            reservoir_size * num_layers
        ) * self.num_classes + self.num_classes

        layers = [
            {
                "name": "Input Weights",
                "params": input_params,
                "trainable": False,
                "type": "input",
                "description": (
                    f"Input → First reservoir: {self.input_size:,} x {reservoir_size:,}"
                ),
            },
            {
                "name": f"Multi-Reservoir ({num_layers} layers)",
                "params": total_reservoir_params,
                "trainable": False,
                "type": "reservoir",
                "description": (
                    f"Stacked reservoirs: {num_layers} x "
                    f"({reservoir_size:,} x {reservoir_size:,})"
                ),
            },
            {
                "name": "Layer Connections",
                "params": connection_params,
                "trainable": False,
                "type": "connection",
                "description": f"Inter-layer connections: {num_layers - 1} layers",
            },
            {
                "name": "Readout Layer",
                "params": readout_params,
                "trainable": True,
                "type": "output",
                "description": (
                    f"Concatenated readout: {reservoir_size * num_layers:,} → "
                    f"{self.num_classes}"
                ),
            },
        ]

        return layers


class PyTorchModelAnalyzer:
    """Analyzes saved PyTorch models to extract actual layer information"""

    def __init__(self):
        self.mlruns_dir = Path("/home/aclarke/echoes/mlruns")

    def find_model_file(self, run_id: str) -> Path | None:
        """Find the PyTorch model file for a given run ID"""
        try:
            # Look for model directories and check their metadata
            models_dir = self.mlruns_dir / "667638895546854907" / "models"
            if not models_dir.exists():
                return None

            for model_dir in models_dir.glob("m-*"):
                meta_path = model_dir / "meta.yaml"
                if meta_path.exists():
                    try:
                        # Read metadata to find source_run_id
                        with open(meta_path) as f:
                            content = f.read()
                            if f"source_run_id: {run_id}" in content:
                                # Found the right model, check for PyTorch file
                                model_path = (
                                    model_dir / "artifacts" / "data" / "model.pth"
                                )
                                if model_path.exists():
                                    return model_path
                    except Exception as e:
                        logger.debug(f"Could not read metadata for {model_dir}: {e}")
                        continue

            return None
        except Exception as e:
            logger.warning(f"Could not find model file for run {run_id}: {e}")
            return None

    def analyze_pytorch_model(self, model_path: Path) -> list[dict[str, Any]]:  # noqa: PLR0912, PLR0915
        """Extract layer information from a saved PyTorch model"""
        # TODO: Refactor this method - too many branches and statements
        try:
            # Load the model
            device = torch.device("cpu")
            model = torch.load(model_path, map_location=device, weights_only=False)

            # Try to get state_dict from model, otherwise assume it's already one
            if hasattr(model, "state_dict"):
                state_dict = model.state_dict()
            elif hasattr(model, "items"):
                state_dict = model
            else:
                # Try to get parameters from the model directly
                state_dict = {}
                if hasattr(model, "named_parameters"):
                    for name, param in model.named_parameters():
                        state_dict[name] = param
                else:
                    logger.warning(
                        f"Cannot extract parameters from model type: {type(model)}"
                    )
                    return []

            layers = []
            layer_groups = {}
            min_parts = 2

            # Group parameters by layer prefix
            for name, tensor in state_dict.items():
                parts = name.split(".")
                if len(parts) >= min_parts:
                    layer_name = ".".join(parts[:-1])  # Remove 'weight' or 'bias'
                    if layer_name not in layer_groups:
                        layer_groups[layer_name] = []
                    layer_groups[layer_name].append((name, tensor))

            # Analyze each layer group
            for layer_name, params in layer_groups.items():
                total_params = sum(tensor.numel() for _, tensor in params)

                # Determine layer type
                layer_type = "unknown"
                description = f"Parameters: {total_params:,}"

                if (
                    "w_in" in layer_name.lower()
                    or "input_weights" in layer_name.lower()
                ):
                    layer_type = "input"
                    description = (
                        f"Input → Reservoir weights: {total_params:,} parameters"
                    )
                elif "lstm" in layer_name.lower() or "rnn" in layer_name.lower():
                    layer_type = "rnn"
                    description = f"LSTM layer: {total_params:,} parameters"
                elif "fc" in layer_name.lower() or "linear" in layer_name.lower():
                    if "output" in layer_name.lower() or layer_name.endswith("fc2"):
                        layer_type = "output"
                        description = f"Output layer: {total_params:,} parameters"
                    else:
                        layer_type = "fc"
                        description = f"Fully connected: {total_params:,} parameters"
                elif "reservoir" in layer_name.lower():
                    layer_type = "reservoir"
                    description = f"Reservoir: {total_params:,} parameters"
                elif "readout" in layer_name.lower():
                    layer_type = "output"
                    description = f"Readout layer: {total_params:,} parameters"
                elif "layer_connection" in layer_name.lower():
                    layer_type = "connection"
                    description = (
                        f"Layer connection weights: {total_params:,} parameters"
                    )

                # Check if layer is trainable (simplified heuristic)
                trainable = not (
                    "reservoir" in layer_name.lower()
                    or "input_weights" in layer_name.lower()
                    or "w_in" in layer_name.lower()
                )

                layers.append(
                    {
                        "name": layer_name,
                        "params": total_params,
                        "trainable": trainable,
                        "type": layer_type,
                        "description": description,
                    }
                )

            # Post-process layers to add missing components for ESN models
            layers = self._add_missing_esn_components(layers)

            # Sort layers by parameter count (descending)
            layers.sort(key=lambda x: x["params"], reverse=True)

            return layers

        except Exception as e:
            logger.warning(f"Could not analyze PyTorch model {model_path}: {e}")
            return []

    def _add_missing_esn_components(
        self, layers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add missing reservoir components for ESN models"""
        if not layers:
            return layers

        # Check if this looks like an ESN model (has W_in but no explicit reservoir)
        has_w_in = any(layer["name"].lower() == "w_in" for layer in layers)
        has_explicit_reservoir = any(
            "reservoir" in layer["name"].lower() for layer in layers
        )

        if has_w_in and not has_explicit_reservoir:
            # This is likely a SimpleESN - add missing reservoir layer
            w_in_layer = next(
                (layer for layer in layers if layer["name"].lower() == "w_in"), None
            )
            if w_in_layer:
                # Estimate reservoir size from W_in parameters
                # W_in typically has input_size * reservoir_size parameters
                input_size = 112 * 112 * 3  # Video frame dimensions
                reservoir_size = w_in_layer["params"] // input_size
                reservoir_params = reservoir_size * reservoir_size

                # Create reservoir layer
                reservoir_layer = {
                    "name": "reservoir",
                    "params": reservoir_params,
                    "trainable": False,
                    "type": "reservoir",
                    "description": (
                        f"Hidden reservoir: {reservoir_size:,} x "
                        f"{reservoir_size:,} (frozen)"
                    ),
                }

                # Insert reservoir layer after W_in
                layers.append(reservoir_layer)

        return layers


class MLflowDashboardBuilder:
    """Main class for building and deploying the MLflow dashboard"""

    def __init__(self):
        self.analyzer = ModelArchitectureAnalyzer()
        self.pytorch_analyzer = PyTorchModelAnalyzer()
        self.data = {"experiments": [], "metadata": {}}

    def extract_mlflow_data(self) -> dict[str, Any]:
        """Extract experiment data from MLflow"""
        logger.info(f"Extracting MLflow data for experiment: {EXPERIMENT_NAME}")

        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if not experiment:
                logger.error(f"Experiment '{EXPERIMENT_NAME}' not found")
                return {
                    "experiments": [],
                    "metadata": {"error": "Experiment not found"},
                }

            # Search runs
            runs_df = mlflow.search_runs(experiment.experiment_id)
            logger.info(f"Found {len(runs_df)} runs")

            experiments = []

            for _, run in runs_df.iterrows():
                try:
                    experiment_data = self._process_run(run)
                    if experiment_data:
                        experiments.append(experiment_data)
                except Exception as e:
                    logger.warning(
                        f"Failed to process run {run.get('run_id', 'unknown')}: {e}"
                    )

            self.data = {
                "experiments": experiments,
                "metadata": {
                    "total_runs": len(experiments),
                    "experiment_name": EXPERIMENT_NAME,
                    "generated_at": pd.Timestamp.now().isoformat(),
                },
            }

            logger.info(f"Successfully processed {len(experiments)} experiments")
            return self.data

        except Exception as e:
            logger.error(f"Failed to extract MLflow data: {e}")
            return {"experiments": [], "metadata": {"error": str(e)}}

    def _process_run(self, run: pd.Series) -> dict[str, Any] | None:
        """Process a single MLflow run"""
        run_name = run.get("tags.mlflow.runName", "Unknown")
        model_type = run.get("params.model_type", "Unknown")

        # Extract parameters
        params = {}
        for col in run.index:
            if col.startswith("params."):
                param_name = col.replace("params.", "")
                value = run[col]
                if pd.notna(value):
                    try:
                        # Try to convert to number
                        if "." in str(value):
                            params[param_name] = float(value)
                        else:
                            params[param_name] = int(value)
                    except (ValueError, TypeError):
                        params[param_name] = str(value)

        # Extract metrics
        metrics = {}
        for col in run.index:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                value = run[col]
                if pd.notna(value):
                    metrics[metric_name] = float(value)

        # Try PyTorch model inspection first, fall back to manual analysis
        run_id = run.get("run_id", "")
        architecture = []

        # Attempt PyTorch inspection
        model_path = self.pytorch_analyzer.find_model_file(run_id)
        if model_path:
            logger.info(f"Found PyTorch model for run {run_id}: {model_path}")
            pytorch_architecture = self.pytorch_analyzer.analyze_pytorch_model(
                model_path
            )
            if pytorch_architecture:
                architecture = pytorch_architecture
                logger.info(f"Using PyTorch-extracted architecture for {run_name}")

        # Fall back to manual analysis if PyTorch inspection failed
        if not architecture:
            logger.info(f"Using manual architecture analysis for {run_name}")
            architecture = self.analyzer.analyze_model(run.to_dict())

        return {
            "name": run_name,
            "type": model_type,
            "run_id": run.get("run_id", ""),
            "params": params,
            "metrics": metrics,
            "architecture": architecture,
        }

    def build_dashboard_data(self) -> str:
        """Build the dashboard data JSON"""
        if not self.data["experiments"]:
            logger.info("No experiment data available, extracting from MLflow...")
            self.extract_mlflow_data()

        # Sanitize data for web consumption
        sanitized_data = self._sanitize_data(self.data)

        return json.dumps(sanitized_data, indent=2, default=str)

    def _sanitize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive information and ensure data is web-safe"""
        sanitized = {"experiments": [], "metadata": data.get("metadata", {})}

        for exp in data.get("experiments", []):
            sanitized_exp = {
                "name": exp.get("name", "Unknown"),
                "type": exp.get("type", "Unknown"),
                "params": {
                    k: v
                    for k, v in exp.get("params", {}).items()
                    if not any(
                        sensitive in k.lower()
                        for sensitive in ["password", "key", "secret", "token"]
                    )
                },
                "metrics": exp.get("metrics", {}),
                "architecture": exp.get("architecture", []),
            }
            sanitized["experiments"].append(sanitized_exp)

        return sanitized

    def deploy_to_nginx(self) -> bool:
        """Deploy dashboard to nginx directory"""
        try:
            logger.info(f"Deploying dashboard to {NGINX_DASHBOARD_DIR}")

            # Create nginx directory if it doesn't exist
            os.makedirs(NGINX_DASHBOARD_DIR, exist_ok=True)

            # Copy static files
            for file_path in SOURCE_DIR.glob("*"):
                if file_path.is_file():
                    dest_path = Path(NGINX_DASHBOARD_DIR) / file_path.name
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"Copied {file_path.name}")

            # Generate and write data JSON
            json_data = self.build_dashboard_data()
            json_path = Path(NGINX_DASHBOARD_DIR) / "models_data.json"
            with open(json_path, "w") as f:
                f.write(json_data)
            logger.info("Generated models_data.json")

            # Set appropriate permissions
            os.system(f"chmod -R 644 {NGINX_DASHBOARD_DIR}/*")
            os.system(f"chmod 755 {NGINX_DASHBOARD_DIR}")

            logger.info("✅ Dashboard deployed successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy dashboard: {e}")
            return False


def main():
    """Main entry point"""
    print("🏗️  MLflow Dashboard Builder")
    print("=" * 50)

    builder = MLflowDashboardBuilder()

    # Extract data
    print("📊 Extracting MLflow experiment data...")
    data = builder.extract_mlflow_data()

    if not data["experiments"]:
        print("❌ No experiment data found. Make sure experiments are running.")
        return

    print(f"✅ Found {len(data['experiments'])} experiments")

    # Deploy to nginx
    print("🚀 Deploying to nginx directory...")
    success = builder.deploy_to_nginx()

    if success:
        print("\n🎉 Dashboard deployment complete!")
        print(f"📁 Files deployed to: {NGINX_DASHBOARD_DIR}")
        print("🌐 Access via: https://dashboard.lonel.ai (after nginx setup)")
        print("\n💡 Next steps:")
        print("  1. Set up nginx configuration")
        print("  2. Configure DNS for dashboard.lonel.ai")
        print("  3. Set up SSL certificate")
    else:
        print("❌ Deployment failed. Check logs above.")


if __name__ == "__main__":
    main()
