# Bypassing Skip-Gram Negative Sampling: Dimension Regularization as a More Efficient Alternative for Graph Embeddings

[David Liu](https://dliu18.github.io/), [Arjun Seshadri](https://arjunsesh.github.io/), [Tina Eliassi-Rad](https://eliassi.org/), [Johan Ugander](https://stanford.edu/~jugander/)

Published in KDD'25

Paper on [ArXiv](https://arxiv.org/abs/2405.00172)

## Code Environment
All of the experiments were executed on a machine with a single NVIDIA V100 GPU. Unless otherwise noted, the conda environment specified in `requirements-sgns.txt` was used for all experiments.

This code base is an adaptation of the [LightGCN PyTorch](https://github.com/gusye1234/LightGCN-PyTorch) codebase.

## Organization of Repository

### Data 
The dataloaders in `code/dataloader.py` handle the dataset pre-processing. The Cora, CiteSeer, PubMed, and SBM networks are all loaded with the PyG dataloaders in the `SmallBenchmark` class and the OGB networks are loaded with the the OGB dataloader in `OGBBenchmark`.

The dataloaders also provide positive training samples for LINE and node2vec. The LINE positive samples are accessed through `get_train_loader_edges` and the node2vec positive samples are accessed through `get_train_loader_rw`.

### Models
All three model variants are trained via the `SGModel` class in `code/models.py`. The model class implements three loss function compoents: 
* Positive loss (`sg_positive_loss`): used by all three variants.
* Negative loss (`sg_negative_loss`): used by the I variant.
* Dimension regularization (`dimension_reg`): used by the II variant.

Note that while the positive and negative losses are batch operations, the dimension regularization loss is over the global embedding matrix. 

### Execution Scripts
All training and hyperparameter tuning is executed through the shell scripts in `code/scripts`. See the following section for details on reproducing the results in the paper.

### Post-processing and Visualization
We also include several post-processing Python scripts for analyzing the tensorboard output from training and evaluation. These scripts are: `performance-vs-graph-feature.py`, `post-process.py`, `sbm-clustering.py`, `summary.py`, `gen_figs/metric-summary.py`.

## Reproducing the Paper
The below instructions are for reproducing the empirical evaluation results reported in Section 5 of our paper.

### Hyperparameter tuning

### Model training

### Table and figure generation 

* Tables 3 and 4:
* Figure 3:
* Figure 4:
* Table 5: 
* Table 6: 


## 