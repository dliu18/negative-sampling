#!/bin/bash

./scripts/hyperparameter-lr/grid_search.sh CiteSeer line 128 20 lr_grid_search
./scripts/hyperparameter-lr/grid_search.sh CiteSeer n2v 128 20 lr_grid_search