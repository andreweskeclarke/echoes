# Simple Execution Plan

## Current Status
- ✅ UCF101 dataset working
- ✅ Two models: SimpleRNN (67% acc), SimpleESN (55% acc)
- ✅ Basic training script works

## Next Steps

### Task 1: Add MLflow for local model comparison
1. Install MLflow
2. Modify training script to log experiments
3. Run multiple experiments with different configs
4. Compare results in MLflow UI

### Task 2: Azure GPU automation
1. Create script to spin up Azure GPU VM
2. SSH, download data, run experiment
3. Get results back
4. Shutdown VM

## Start with Task 1