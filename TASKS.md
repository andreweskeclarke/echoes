# Project Tasks

## Project Setup
- [ ] Add linters
- [ ] Add unit tests and first unit test

## Dataset Processing
- [ ] Create efficient data loading pipeline for feeding into PyTorch models
  - [ ] Implement PyTorch Dataset class
  - [ ] Add caching mechanism for processed frames
  - [ ] Implement parallel data loading
- [ ] Implement data augmentation
  - [ ] Random cropping
  - [ ] Horizontal flipping
  - [ ] Color jittering
  - [ ] Temporal augmentation

## Model Implementation
- [ ] Implement Echo State Network (ESN)
- [ ] Implement baseline models
  - [ ] Traditional RNN
  - [ ] LSTM
  - [ ] GRU
  - [ ] What else? Research

## Training Pipeline
- [ ] Create training scripts
  - [ ] Training loop
  - [ ] Validation loop
  - [ ] Early stopping
  - [ ] Learning rate scheduling
- [ ] Implement metrics
  - [ ] Accuracy
  - [ ] F1 score
  - [ ] Confusion matrix
  - [ ] Training time comparison
- [ ] Add logging and visualization
  - [ ] TensorBoard integration
  - [ ] Training curves
  - [ ] Model architecture visualization
  - [ ] Include test evaluation during training, I don't care too much about leakage.
- [ ] Train a simple and small network to test
- [ ] Save models somewhere, maybe a new gitignore'd models/ directory.
  - [ ] Save other training info like curves and metadata there too

## Evaluation
- [ ] Create evaluation pipeline
  - [ ] Test set evaluation
  - [ ] Cross-validation
  - [ ] Performance metrics
- [ ] Implement comparison framework
  - [ ] Model comparison
  - [ ] Model parameter count comparison
  - [ ] Training time analysis
  - [ ] Memory usage analysis
- [ ] Write analysis notebooks
  - [ ] Model comparison
  - [ ] Performance analysis
  - [ ] Visualization examples
  - [ ] Performance comparison
  - [ ] Computational efficiency
  - [ ] Training dynamics

## Documentation
- [ ] Add code documentation
  - [ ] Docstrings
  - [ ] Type hints
  - [ ] Usage examples
- [ ] Create usage guides
  - [ ] Installation guide
  - [ ] Dataset preparation
  - [ ] Training guide
  - [ ] Evaluation guide
- [ ] Write analysis notebooks
  - [ ] Model comparison
  - [ ] Performance analysis
  - [ ] Visualization examples

## Azure Cloud Experiment Pipeline
- [ ] Set up Azure infrastructure
  - [ ] Create blob storage containers (ucf101-dataset, processed-data, experiment-artifacts)
  - [ ] Configure service principal for authentication
  - [ ] Create GPU VM templates (NC/ND series) with auto-shutdown
- [ ] Data pipeline for cloud
  - [ ] Upload UCF101 dataset to blob storage
  - [ ] Implement blob storage data loading in PyTorch Dataset
  - [ ] Create preprocessing pipeline that caches to blob storage
- [ ] Experiment orchestration
  - [ ] Config-driven experiment runner (YAML configs)
  - [ ] VM provisioning and auto-cleanup scripts
  - [ ] TensorBoard real-time monitoring via SSH tunnel
  - [ ] Periodic sync of logs to blob storage during training
- [ ] MLflow integration
  - [ ] Set up MLflow tracking server (local or Azure)
  - [ ] Integrate MLflow logging in training scripts
  - [ ] Configure artifact storage to blob storage
  - [ ] Create experiment comparison dashboard
- [ ] Cost management
  - [ ] Auto-shutdown policies for VMs
  - [ ] Budget alerts and monitoring
  - [ ] Optimize blob storage access patterns

## Infrastructure
- [ ] Set up CI/CD
  - [ ] GitHub Actions
  - [ ] Automated testing
  - [ ] Code quality checks
- [X] Add development tools
  - [X] Pre-commit hooks
  - [X] Code formatting
  - [X] Linting

## Future Enhancements
- [ ] Add support for other video datasets
- [ ] Implement distributed training
- [ ] Add model quantization
- [ ] Create web demo
- [ ] Add API documentation
