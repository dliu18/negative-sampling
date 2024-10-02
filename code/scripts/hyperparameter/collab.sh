#!/bin/bash

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=1024 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=1024 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=100 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=1024 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=100 \
--lr=0.1 \
--lam=0.01 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=1024 \
--epochs=10 \
--board_path="9-25-weighted"

python main.py \
--base_model="n2v" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=1024 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=32 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg" \
--test_set="test" \
--lr=0.1 \
--n_negative=-1 \
--K=5 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=32 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=32 \
--epochs=10 \
--board_path="9-25-auc"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000 \
--lr=0.1 \
--lam=0.01 \
--alpha=0.75 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=32 \
--epochs=10 \
--board_path="9-25-weighted"

python main.py \
--base_model="line" \
--loss_func="sg_aug" \
--test_set="test" \
--n_negative=1000000000 \
--lr=0.1 \
--lam=0.01 \
--seed=2020 \
--dataset="ogbl-collab" \
--recdim=128 \
--batch_size=32 \
--epochs=10 \
--board_path="9-25-auc"