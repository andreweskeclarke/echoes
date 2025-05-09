# Echoes: Exploring Echo State Networks

This repository contains experiments comparing Echo State Networks (ESNs) with traditional Recurrent Neural Networks (RNNs) and other architectures using PyTorch. The project focuses on video classification using the UCF101 dataset.

## Project Goals

- Implement and compare different neural network architectures:
  - Echo State Networks (ESNs)
  - Traditional RNNs
  - LSTM/GRU networks
  - Other relevant architectures
- Evaluate performance on video classification tasks
- Analyze computational efficiency and training dynamics
- Document best practices and findings

## Project Structure

```
echoes/
├── data/               # Data storage and preprocessing
├── models/            # Model implementations
│   ├── esn.py        # Echo State Network implementation
│   ├── rnn.py        # Traditional RNN implementation
│   └── utils.py      # Shared model utilities
├── experiments/       # Training and evaluation scripts
├── notebooks/        # Jupyter notebooks for analysis
├── environment.yml   # Conda environment specification
└── README.md        # This file
```

## Setup

1. Create and activate the Conda environment:
```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate echoes
```

2. Verify the installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Dataset

This project uses the UCF101 dataset, which consists of 13,320 video clips from 101 action categories. The dataset can be downloaded from the [UCF101 website](https://www.crcv.ucf.edu/data/UCF101.php).

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

MIT License
