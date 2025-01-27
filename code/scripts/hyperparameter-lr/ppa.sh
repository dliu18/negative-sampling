#!/bin/bash

./scripts/hyperparameter-lr/grid_search.sh ogbl-ppa line 1024 2 lr_grid_search
./scripts/hyperparameter-lr/grid_search.sh ogbl-ppa n2v 1024 2 lr_grid_search