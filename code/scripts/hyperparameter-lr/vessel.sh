#!/bin/bash

./scripts/hyperparameter-lr/grid_search.sh ogbl-vessel line 1024 1 lr_grid_search
./scripts/hyperparameter-lr/grid_search.sh ogbl-vessel n2v 1024 1 lr_grid_search