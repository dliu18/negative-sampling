#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=5 \
--board_path="9-25-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=5 \
--board_path="9-25-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=5 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=5 \
--board_path="9-25-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=5 \
--lr=0.01 \
--lam=0.001 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=5 \
--board_path="9-25-weighted"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.01 \
--lam=0.001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=1024 \
--epochs=5 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.01 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.0001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.01 \
--lam=0.0001 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="9-25-weighted"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.01 \
--lam=0.0001 \
--seed=2020 \
--dataset="ogbl-ppa" \
--recdim=128 \
--batch_size=256 \
--epochs=3 \
--board_path="9-25-auc"