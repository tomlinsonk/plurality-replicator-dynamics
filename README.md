# Code for Replicating Electoral Success

We have included all code for running simulations, generating plots, and all results files.

# Contents
- `results/`: our simulation results
- `replicator.py`: code for running replicator dynamics and variants
- `plot.py`: code for plotting simulation results

# Compute details
Simulations took 94 minutes to run with 100 threads on a server with Intel Xeon Gold 6254 CPUs and 1.5TB RAM (but would run fine on a more modest machine with more time).

Simulations run with:
- Python 3.8.10
    - numpy 1.22.4     
    - scipy 1.8.1 
    - tqdm 4.64.0  

Plotting code run with:
- Python 3.11.7
    - numpy 1.26.4
    - scipy 1.12.0
    - matplotlib 3.8.2

# Reproducibility
To generate plots from our results files, simply run 
```python3 plot.py```

To rerun all simulations:
```python3 replicator.py --threads [THREADS]```