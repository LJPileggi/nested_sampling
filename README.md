# nested_sampling
This project provides a dummy implementation of the Classical Nested Sampling (J. Skilling) and Diffusive Nested Sampling (B. J. Brewer, L. B. Pártay, G. Csányi) algorithms.

Papers:
>[1] J. Skilling, Nested Sampling for General Bayesian Computation,
Bayesian Analysis 4, pp. 833-860 (2006);

>[2] B. J. Brewer, L. B. Pártay, G. Csányi, Diffusive Nested Sampling, https://arxiv.org/abs/0912.2380.

### Run a simulation
See the list of different flags through:

$ python3 classical.py _or_ diffusive.py -h

output files are stored in the ./output/ folder.

### Analyse results of simulation
Analyse a previously generated dataset through:

$ python3 analysis.py --algorithm %%% --filename %%% --n_trials %%% --param %%%

graph files are stored in the ./graphs/ folder
