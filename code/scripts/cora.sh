#!/bin/bash

python ../main.py \
--base_model="n2v" \
--loss_func="$sg" \
--test_set="test" \
--lr="$lr" \
--lam="$lam" \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50

python ../main.py \
--base_model="n2v" \
--loss_func="$sg_aug" \
--test_set="test" \
--n_negative="$n_negative" \
--lr="$lr" \
--lam="$lam" \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=128 \
--epochs=50

python ../main.py \
--base_model="line" \
--loss_func="$sg" \
--test_set="test" \
--lr="$lr" \
--lam="$lam" \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=10

python ../main.py \
--base_model="line" \
--loss_func="$sg_aug" \
--test_set="test" \
--n_negative="$n_negative" \
--lr="$lr" \
--lam="$lam" \
--seed=2020 \
--dataset="Cora" \
--recdim=128 \
--batch_size=1 \
--epochs=50