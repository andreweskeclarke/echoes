# Model Architectures

Overview of the neural network models implemented in Echoes.

## Traditional RNNs

### SimpleRNN

Basic recurrent neural network for sequence processing.

**Architecture**: Input → LSTM → Fully Connected → Output

**Parameters**:
- `input_size`: Feature dimension (512 for video frames)
- `hidden_size`: LSTM hidden state dimension (256)
- `num_classes`: Classification targets (101 for UCF101)

**Characteristics**:
- Fully trainable parameters
- Traditional gradient-based learning
- Baseline for comparison

**Use Case**: Baseline model to compare against ESNs

**Code Location**: `models/simple_models.py:SimpleRNN`

### DeepRNN

Multi-layer RNN for modeling complex temporal patterns.

**Architecture**: Input → LSTM(L1) → LSTM(L2) → LSTM(L3) → FC → Output

**Parameters**:
- `num_layers`: Number of LSTM layers (2-4)
- `hidden_size`: Hidden dimension per layer
- `dropout`: Regularization between layers

**Characteristics**:
- Stacked recurrent layers
- Captures hierarchical temporal patterns
- Higher capacity than SimpleRNN

**Use Case**: Modeling complex video dynamics

## Echo State Networks (ESNs)

### SimpleESN

Single-layer Echo State Network with frozen reservoir.

**Architecture**: Input(frozen) → Reservoir(frozen) → Readout(trainable)

**Parameters**:
- `input_size`: Feature dimension
- `reservoir_size`: Number of reservoir neurons (500-2000)
- `spectral_radius`: Largest eigenvalue of weight matrix (~0.9)
- `sparsity`: Proportion of zero weights in reservoir (~0.9)

**Characteristics**:
- Only readout layer is trainable (linear regression)
- Reservoir fixed after initialization
- Extremely fast training
- Lower memory usage

**Advantage**: 10-100x faster training than RNNs

**Code Location**: `models/esn.py:SimpleESN`

### DeepESN

Multi-layer Echo State Network.

**Architecture**: Input → Reservoir1(frozen) → Reservoir2(frozen) → Readout(trainable)

**Parameters**:
- `num_reservoirs`: Number of ESN layers (2-3)
- `reservoir_size`: Neurons per reservoir
- `coupling_strength`: How reservoirs connect

**Characteristics**:
- Hierarchical reservoir dynamics
- Still only trains readout layer
- Better feature extraction than single layer
- Maintains speed advantage over deep RNNs

**Use Case**: Research focus - exploring ESN scalability

## Comparison

| Aspect | SimpleRNN | DeepRNN | SimpleESN | DeepESN |
|--------|-----------|---------|-----------|---------|
| **Training Time** | 10 min/epoch | 30 min/epoch | <1 min/epoch | 1-2 min/epoch |
| **Trainable Params** | 500K | 2M+ | 50K | 100K |
| **Memory** | High | Very High | Low | Low |
| **Accuracy** | 65-70% | 70-75% | 55-65% | 60-70% |
| **Convergence** | Stable | Sometimes unstable | Stable | Stable |

## Creating Custom Models

Extend the base classes to create new models:

```python
import torch.nn as nn
from models.simple_models import BaseModel

class MyCustomRNN(BaseModel):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use last timestep
        out = self.fc(out)
        return out
```

## Performance Metrics

Each model tracks:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correct predictions / total predictions |
| **Training Time** | Wall-clock time per epoch |
| **Inference Time** | Time to process one batch |
| **Memory Peak** | Maximum GPU/CPU memory used |
| **Convergence** | Epoch when validation accuracy plateaus |

These are logged to MLflow for comparison.

## Next Steps

- [Infrastructure](infrastructure.md) - How models are trained and deployed
- [Running Experiments](../guides/experiments.md) - Create your own experiments
