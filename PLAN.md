# Simple Execution Plan

## Current Status
- ✅ UCF101 dataset working
- ✅ Two models: SimpleRNN (67% acc), SimpleESN (55% acc)
- ✅ Basic training script works

## Next Steps

### Task: Azure GPU automation
1. Create script to spin up Azure GPU VM
2. SSH, download data, run experiment
3. Get results back
4. Shutdown VM

### Task: Experiment with ESN architectures
1. Simple ESNs have different values we want to search over:
  a. alpha, the sparsity parameter of W_res. right now we don't have any sparsity.
  b. spectral_radius, determines the echoiness of the reservoir over time.
  c. Continue experimenting with reservoir sizes. W_in, W_res, and W_readout
  d. Does it matter if we use a non-linearity for the final readout? That could help
  e. Train longer
2. Improve Deep ESNs.
  a. The learning curves are not completed, train longer
  b. I like the comparison of 2 layer reservoir and 3 layer reservoir. Compare for all the same reservoir sizes as in the Simple ESN.
  c. Later we want to compare over the same set of alpha and spectral_radius questions.
3. Train and validate over all classes now, no more downsampling by class.
4. Train and validate over multiple seeds.

### Task: Improve dashboard
1. Read the MLFlow objects and the pytorch objects and decide how to render the whole thing automatically
2. Are there any open source libraries to do this better?


### Task: Improve RNN architectures
1. These are not learning effectively, let's go back to basics and see how to make them work and scale as expected. Currently, their validation scores are all over the place, terrible, and not correlated to network size.


### Task: More datasets
1. Let's try datasets beyond UCF101, that are meant for time series tasks