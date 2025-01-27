#!/bin/bash

./scripts/hyperparameter-lr/grid_search.sh ogbl-citation2 line 1024 1 lr_grid_search
./scripts/hyperparameter-lr/grid_search.sh ogbl-citation2 n2v 1024 1 lr_grid_search