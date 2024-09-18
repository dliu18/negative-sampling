Implementation for "Re-visiting Skip-Gram Negative Sampling: Dimension Regularization for More Efficient Dissimilarity Preservation in Graph Embeddings"

## Execution scripts

### Quick testing
The `code/scripts/run.sh` script can be used to quickly verify that the model functions properly. The script will train one single model and is useful for determining the runtime for a single epoch, for instance.

### Hyperparameter optimization
The script `code/scripts/hyperparameter-opt.sh` performs a grid search over hyperparameters such as `n_negative`, `lr`, `lambda`. The inputs to the script are of the format

```
./scripts/hyperparameter-opt.sh dataset epochs batch_size_n2v batch_size_line board_path
```

fter the script runs, the results can be exported using `python summary.py`. Note: this script requires tensorflow. 

### Final testing
Each network has its own execution script in `scripts/`. Configure the optimal hyperparameters before executing.

## Environment
All of the experiments were executed on a machine with a single NVIDIA V100 GPU. Unless otherwise noted, the conda environment specified in `requirements-sgns.txt` was used for all experiments.
