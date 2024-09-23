#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--n_negative=-1 \
--lr=0.1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=10 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--n_negative=-1 \
--lr=0.1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.1 \
--lam=1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50 \
--board_path="9-21-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--lam=1 \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50 \
--board_path="9-21-auc"