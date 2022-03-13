# nested_sampling
This project provides a dummy implementation of the Classical Nested Sampling (J. Skilling) and Diffusive Nested Sampling (B. J. Brewer, L. B. Pártay, G. Csányi) algorithms.

Papers:
>[1] J. Skilling, Nested Sampling for General Bayesian Computation,
Bayesian Analysis 4, pp. 833-860 (2006);
>[2] B. J. Brewer, L. B. Pártay, G. Csányi, Diffusive Nested Sampling, https://arxiv.org/abs/0912.2380.

### Run a simulation
See the list different flags through:
'''
$ python3 classical.py/diffusive.py --flag %%%
'''
output files are stored in the ./output/ folder.

### Output files
Output file for classical n. s.:\n
initial points	iterations	evidence	time		Mc step\n
100		10000		2.36548e-43	25.346		0.005\n
...   ...     ...         ...       ...

output file for diffusive n. s.:\n
max level	L per level	level finished	lam	beta	quantile	evidence	time taken	MC_step\n
100		100		110		10	10	0.36788		4.562e-50	5.214		0.0025\n
...		...		...		...	...	...		...		...		...

### Analyse results of simulation
Analyse a previously generated dataset through:
'''
$ python3 analysis.py --algorithm %%% --filename %%% --n_trials %%% --param %%%
'''
graph files are stored in the ./graphs/ folder
