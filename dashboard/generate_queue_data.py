#!/usr/bin/env python3
"""
Experiment Queue Data Generator

Reads all MLflow experiments and outputs experiment_queue.json for the
dashboard's experiment log panel. Run periodically to keep the panel fresh.

Usage:
    conda activate echoes
    python dashboard/generate_queue_data.py

Or watch-mode (re-run every 30s):
    watch -n 30 python dashboard/generate_queue_data.py
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLRUNS_DIR = Path("/mnt/echoes_data/mlruns")
OUTPUT_PATH = Path(__file__).parent / "src" / "experiment_queue.json"

# MLflow run status codes
MLFLOW_STATUS = {
    1: "running",
    2: "pending",
    3: "failed",
    4: "done",
    5: "killed",
}

# Model type short descriptions for the hypothesis blurb
MODEL_BLURBS = {
    "SimpleRNN": "Single-layer LSTM baseline for UCF101 action classification",
    "DeepRNN": "Multi-layer LSTM with dropout for deep sequential modeling",
    "SimpleESN": "Echo State Network — frozen reservoir, only readout trains",
    "DeepESN": "Stacked reservoirs with concatenated readout layer",
    "ESN": "Echo State Network reservoir computing approach",
}


def _read_file(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:
        return ""


MIN_METRIC_LINE_PARTS = 2


def _read_metric_last(metric_path: Path) -> float | None:
    try:
        last_line = metric_path.read_text().strip().split("\n")[-1]
        parts = last_line.split()
        if len(parts) >= MIN_METRIC_LINE_PARTS:
            return float(parts[1])
    except Exception:
        pass
    return None


def _parse_meta(run_dir: Path) -> dict | None:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return None

    meta = {}
    for line in meta_path.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def _build_hypothesis(run_name: str, params: dict, model_type: str) -> str:
    base = MODEL_BLURBS.get(model_type, f"{model_type} experiment")

    detail_parts = []
    if "hidden_size" in params:
        detail_parts.append(f"h={params['hidden_size']}")
    if "reservoir_size" in params:
        detail_parts.append(f"res={params['reservoir_size']}")
    if "num_layers" in params:
        detail_parts.append(f"layers={params['num_layers']}")
    if "learning_rate" in params:
        detail_parts.append(f"lr={params['learning_rate']}")
    if "num_epochs" in params:
        detail_parts.append(f"epochs={params['num_epochs']}")

    if detail_parts:
        return f"{base} [{', '.join(detail_parts)}]"
    return base


def process_run(run_dir: Path, experiment_name: str) -> dict | None:
    meta = _parse_meta(run_dir)
    if not meta:
        return None

    run_id = run_dir.name
    status_code = int(meta.get("status", 0))
    status = MLFLOW_STATUS.get(status_code, "unknown")

    run_name = (
        _read_file(run_dir / "tags" / "mlflow.runName")
        or meta.get("run_name", "")
        or run_id[:8]
    )

    params = {}
    params_dir = run_dir / "params"
    if params_dir.exists():
        for p in params_dir.iterdir():
            params[p.name] = _read_file(p)

    model_type = params.get("model", params.get("model_type", "Unknown"))

    metrics_dir = run_dir / "metrics"
    val_accuracy = None
    train_loss = None
    training_time = None

    if metrics_dir.exists():
        for candidate in ("final_val_accuracy", "val_accuracy"):
            p = metrics_dir / candidate
            if p.exists():
                val_accuracy = _read_metric_last(p)
                if val_accuracy is not None:
                    break

        for candidate in ("train_loss",):
            p = metrics_dir / candidate
            if p.exists():
                train_loss = _read_metric_last(p)

        p = metrics_dir / "training_time_seconds"
        if p.exists():
            training_time = _read_metric_last(p)

    def _parse_ms(v: str) -> int:
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    start_ms = _parse_ms(meta.get("start_time", "0"))
    end_ms = _parse_ms(meta.get("end_time", "0"))

    if training_time is None and start_ms and end_ms and status == "done":
        training_time = (end_ms - start_ms) / 1000.0

    hypothesis = _build_hypothesis(run_name, params, model_type)

    return {
        "run_id": run_id,
        "name": run_name,
        "model_type": model_type,
        "status": status,
        "hypothesis": hypothesis,
        "experiment_name": experiment_name,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "training_time_seconds": training_time,
        "val_accuracy": val_accuracy,
        "train_loss": train_loss,
        "params": {
            k: v
            for k, v in params.items()
            if k not in ("device",)
        },
    }


STATUS_ORDER = {
    "running": 0,
    "pending": 1,
    "done": 2,
    "failed": 3,
    "killed": 4,
    "unknown": 5,
}


def _sort_key(run: dict) -> tuple:
    return (STATUS_ORDER.get(run["status"], 5), -run["start_time_ms"])


def generate_queue_data() -> dict:
    runs = []

    if not MLRUNS_DIR.exists():
        logger.error(f"MLruns directory not found: {MLRUNS_DIR}")
        return {"runs": [], "generated_at": datetime.now(UTC).isoformat()}

    for exp_dir in MLRUNS_DIR.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        meta_path = exp_dir / "meta.yaml"
        if not meta_path.exists():
            continue

        exp_name = ""
        for line in meta_path.read_text().splitlines():
            if line.startswith("name:"):
                exp_name = line.split(":", 1)[1].strip()
                break

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name in ("models",):
                continue
            run = process_run(run_dir, exp_name)
            if run:
                runs.append(run)

    runs.sort(key=_sort_key)

    return {
        "runs": runs,
        "total": len(runs),
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "mlflow",
    }


def main():
    logger.info(f"Scanning MLflow runs in {MLRUNS_DIR}")
    data = generate_queue_data()
    logger.info(f"Found {data['total']} runs")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))
    logger.info(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
