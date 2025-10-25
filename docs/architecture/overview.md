# Architecture Overview

Echoes is designed to compare different neural network architectures for video classification through a modular, scalable system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Experiment Scripts                      │
│            (train_simple.py, train_comprehensive.py)     │
└────────────┬────────────────────────────────────────────┘
             │
             ├──────────────────────────┬──────────────────────────┐
             │                          │                          │
             ▼                          ▼                          ▼
        ┌─────────────┐          ┌──────────────┐         ┌─────────────┐
        │ Data Layer  │          │Model Layer   │         │Training Loop│
        ├─────────────┤          ├──────────────┤         ├─────────────┤
        │ UCF101      │          │ESN           │         │Metrics      │
        │ CIFAR-10    │          │RNN           │         │Checkpoints  │
        │ etc.        │          │LSTM/GRU      │         │Logging      │
        └─────────────┘          └──────────────┘         └─────────────┘
             │                          │                          │
             ▼                          ▼                          ▼
        ┌─────────────────────────────────────────────────────────┐
        │              MLflow Experiment Tracking                  │
        │   (metrics, params, models, artifacts)                  │
        └─────────────────────────────────────────────────────────┘
             │
             ├──────────────────┬──────────────────┬──────────────────┐
             │                  │                  │                  │
             ▼                  ▼                  ▼                  ▼
        TensorBoard          MLflow UI         Dashboard          Plots
        (local logs)      (experiment view)   (architecture viz)  (analysis)
```

## Component Overview

### Data Layer (`data/`)

Handles dataset loading, preprocessing, and batching:
- **Dataset classes**: `UCF101Dataset`, `VideoFrameDataset`
- **Dataloaders**: Batch management with custom transforms
- **Preprocessing**: Frame extraction, normalization, augmentation

### Model Layer (`models/`)

Implements different architectures:

| Model | Type | Trainable Params | Use Case |
|-------|------|------------------|----------|
| **SimpleRNN** | Traditional RNN | ~50K-500K | Baseline |
| **SimpleESN** | Echo State Network | ~10K-50K (readout only) | Research |
| **DeepRNN** | Multi-layer RNN | ~1M-10M | Complex patterns |
| **DeepESN** | Multi-layer ESN | ~50K-100K (readout only) | Scalability |
| **LSTM** | LSTM networks | Similar to RNN | Gradient stability |

### Training (`experiments/`)

Experiment orchestration:
- Data loading and preprocessing
- Model instantiation
- Loss computation and backpropagation
- Metrics logging (accuracy, loss, inference time)
- Model checkpointing and artifact storage

### Tracking (`mlruns/`)

MLflow stores:
- **Params**: model size, learning rate, batch size, epochs
- **Metrics**: loss, accuracy, inference time, memory usage
- **Artifacts**: trained models, configuration files
- **Metadata**: tags, notes, timestamps

## Storage Architecture

### Local Development

```
echoes/
├── models/           # Model implementations
├── data/             # Dataset loading code
├── experiments/      # Training scripts
├── logs/             # TensorBoard logs (local)
├── mlruns/           # MLflow tracking (local)
└── tfruns/           # TensorFlow/TB runs (local)
```

### Production (Azure)

```
Headquarters VM (persistent disk)
/mnt/echoes_data/
├── ucf101/           # Dataset (13GB)
├── logs/             # Training logs
├── mlruns/           # MLflow experiments
├── tfruns/           # TensorBoard logs
└── azure_results/    # Downloaded results
```

Ephemeral Experiment VMs automatically:
1. Copy code and dataset from headquarters
2. Run experiments
3. Stream results back to persistent storage
4. Self-destruct to save costs

## Data Flow

### Training Run

```
1. Load data from disk
   ↓
2. Create model (initialize weights)
   ↓
3. Forward pass → Compute loss
   ↓
4. Backward pass → Update weights
   ↓
5. Log metrics to MLflow
   ↓
6. Save checkpoint to persistent disk
   ↓
7. Repeat for N epochs
   ↓
8. Evaluate on test set
   ↓
9. Save final model and artifacts
```

### Experiment Comparison

```
1. MLflow stores all run data
   ↓
2. Dashboard queries MLflow API
   ↓
3. Extract model architectures
   ↓
4. Render comparison visualizations
   ↓
5. Display at https://dashboard.lonel.ai
```

## Key Design Principles

### Modularity
- Models are independent, can be swapped easily
- Datasets are abstracted, support multiple sources
- Training logic is separate from data/model

### Reproducibility
- All hyperparameters logged to MLflow
- Models saved with exact configuration
- Seeds set for deterministic results

### Scalability
- Local training for quick iteration
- Azure GPU VMs for serious experiments
- Persistent storage for long-term tracking

### Observability
- Comprehensive logging at each step
- Real-time monitoring via TensorBoard
- Experiment comparison via MLflow UI

## Next Steps

- [Models](models.md) - Detailed model implementations
- [Infrastructure](infrastructure.md) - Cloud setup and deployment
- [Running Experiments](../guides/experiments.md) - How to train models
